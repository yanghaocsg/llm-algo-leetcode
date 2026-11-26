# 81. 分布式推理逻辑验证 — 小白讲解

> 对应 notebook：[`81_Distributed_Inference_Project.ipynb`](./81_Distributed_Inference_Project.ipynb)

## 一句话总结
“加 GPU”至少有三种含义：加完整副本分担请求、把一次模型执行切到多卡、把预填充和解码拆到不同工作池。本节用同一组请求和一套显式成本系数，让三个 TODO 分别估算单请求阶段成本、按最低负载路由请求、根据完成时间/负载偏斜/通信比例决定是否值得上真机复测。它是一个**筛选模型**，不是性能预测器；GPU 实验部分只搭了 G0／G1／G2 的空命令框架，当前没有真实服务测量。

## 为什么需要它：痛点是什么
初学者常把“副本数”“张量并行度”“PD 分离”混为一谈，看到 GPU 数变多就默认更快。实际上：副本路由改变的是“不同请求由谁处理”；张量并行改变的是“一次前向的层内计算怎么分”；PD 分离改变的是“输入阶段和生成阶段由谁处理”。三者的收益和新增代价不同，比较时必须先固定请求、模型、精度和总 GPU 预算。PD 分离尤其容易被误解为“分开就一定快”：它减少了阶段争用，但多了 KV 状态传输、两类队列和状态匹配，净收益要算总账。
生活类比：一家餐厅忙不过来。开分店（副本）能接待更多顾客，但每家店都要配齐厨房；把一道菜拆给几个厨师合作（模型并行）能做大菜，但厨师之间要不停传菜；把备菜和炒菜分到两个房间（PD 分离）能各司其职，但菜要从备菜间端过去。三种做法解决的不是同一个瓶颈。

## 核心问题
- 副本、模型内部并行、PD 分离分别改变什么？新增代价是什么？
- 固定请求与资源预算时，哪些量必须记录？怎样区分“等预算换分工”和“加设备扩容”？
- TTFT、TPOT、makespan、P95/P99 分别回答什么问题？
- 单请求成本里的 `comm_ms_per_token` 为什么不是“每词元通信时间”？
- least-loaded 路由怎样分配请求？makespan、吞吐、负载偏斜怎样算？
- `ready_for_gpu / tune / reject` 三条路径的判定条件是什么？
- 当前 notebook 为什么不能给出真实服务结论？

## 解决思路：一步步看懂

**第 1 步：三种服务形态。** 预填充一次处理输入词元并建立 KV 状态，解码再逐词元生成。完整副本是可独立处理请求的一套模型实例，增加副本主要分担并发；张量并行把层内计算分给多个进程，流水线并行把不同层分给多个进程，二者共同完成一次模型执行；PD 分离让预填充进程和解码进程分开，解码进程需要接收预填充产生的每层 Key/Value。notebook 的对照表把“分工对象、可能收益、新增代价”并列：副本是“不同请求 / 减少排队 / 每副本保存完整模型”；模型并行是“一次执行 / 分摊权重与计算 / 进程间通信与流水等待”；PD 是“输入与生成 / 分别调配资源 / KV 传输、两类队列与状态匹配”。

**第 2 步（TODO 1）：单请求成本。** `estimate_request_cost(request, config)` 把预填充、解码、通信、流水附加和 KV 交接分开：

```python
prefill_ms = prompt_tokens * config['prefill_ms_per_token']
decode_ms = generate_tokens * config['decode_ms_per_token']
comm_ms = max(config['tp_degree'] - 1, 0) * config['comm_ms_per_token'] * (
    prompt_tokens / 128.0 + generate_tokens / 64.0
)
pipeline_ms = max(config['pp_stages'] - 1, 0) * config['pipeline_penalty_ms']
handoff_bytes = prompt_tokens * config.get('kv_bytes_per_token', 0.0)
```

PD 模式再算 `handoff_ms = handoff_base_ms + handoff_bytes / handoff_bytes_per_ms`，并把它计入 `total_ms`；非 PD 模式交接时间为 0。交接的是预填充建立的 KV，所以 `handoff_bytes` 只按 `prompt_tokens` 计，不乘生成词元。注意 `comm_ms_per_token` 虽然名字叫“每词元”，实际是“输入每 128 词元、生成每 64 词元”的归一化成本项，`tp_degree - 1` 表示额外进程数；它不是实测的单词元通信时间。`pipeline_ms` 也只按 `pp_stages - 1` 加固定惩罚，没有模拟流水重叠。这个模型刻意保守：PD 只增加交接成本，不模拟两个工作池的流水收益，因此它只能用来筛掉“明显不划算”的配置，不能证明 PD 一定更快。

拿测试手算：`prompt_tokens=128`、`generate_tokens=64`、`prefill=0.01`、`decode=0.02`、`comm=0.03`、`tp_degree=2`、`pp_stages=1`。`prefill_ms=1.28`，`decode_ms=1.28`，`comm_ms=1×0.03×(128/128+64/64)=0.06`，`pipeline_ms=0`，`total_ms=2.62`。PD 配置加 `kv_bytes_per_token=16`、`handoff_base_ms=0.2`、`handoff_bytes_per_ms=1024`：`handoff_bytes=128×16=2048`，`handoff_ms=0.2+2048/1024=2.2`，`total_ms=1.28+1.28+0.06+0+2.2=4.82`。交接一项就占了总时间的近一半，这正是“分开不一定更快”的量化提醒。

**第 3 步（TODO 2）：least-loaded 路由。** `simulate_distributed_inference` 假设所有请求在起点同时可分配：

```python
for request in requests:
    cost = estimate_request_cost(request, config)
    single_replica_time_ms += cost['total_ms']
    chosen_replica = min(range(num_replicas), key=lambda idx: replica_loads[idx])
    assignments.append(chosen_replica)
    queue_wait_total_ms += replica_loads[chosen_replica]
    replica_loads[chosen_replica] += cost['total_ms']
```

每次选当前累计负载最小的副本（并列时 `min` 取索引更小的），该副本已有负载就是本请求的预计等待。最后 `makespan_ms=max(replica_loads)`，`throughput_req_per_ms=len(requests)/makespan`，`imbalance_ratio=(makespan-min_load)/avg_load`。测试两条请求：第一条总成本 2.62，分到副本 0；第二条 `prefill=0.64`、`decode=0.64`、`comm=0.03×(0.5+0.5)=0.03`，总成本 1.31，此时副本 1 负载 0，所以分到副本 1，`assignments=[0,1]`。单副本串行总时间 `2.62+1.31=3.93`，`makespan=2.62`，`throughput=2/2.62≈0.7634` 请求/毫秒，`imbalance=(2.62-1.31)/1.965≈0.6667`，累计通信成本 `0.06+0.03=0.09`。PD 版本还会累计 `handoff_total_bytes=2048+1024=3072`，`handoff_total_ms>0`。

**第 4 步（TODO 3）：三条决策路径。** `recommend_distributed_inference_run` 先算 `comm_ratio = comm_cost_ms / single_replica_time_ms`，然后：

```python
if candidate_makespan < baseline_time and imbalance <= max_imbalance_ratio and comm_ratio <= max_comm_ratio:
    return {'decision': 'ready_for_gpu', 'next_action': 'promote_to_real_cluster_check', ...}
if candidate_makespan < baseline_time:
    return {'decision': 'tune', 'next_action': 'refine_routing_or_parallel_degree', ...}
return {'decision': 'reject', 'next_action': 'fallback_to_single_replica_audit', ...}
```

三个条件必须同时满足才 `ready_for_gpu`：有完成时间收益、负载偏斜不超过阈值、通信比例不超过阈值。有时间收益但某一项超限时 `tune`，说明方向可能对但布局要调；连时间都没缩短就 `reject`。测试里主用例 `makespan=2.62<3.93`、`imbalance=0.6667≤0.8`、`comm_ratio=0.09/3.93≈0.0229≤0.05`，得到 `ready_for_gpu`；`weak_candidate` 的 `makespan=3.10` 虽小于 3.93，但 `imbalance=0.9>0.8`，得到 `tune`；`bad_candidate` 的 `makespan=4.30` 更大，直接 `reject`。

**第 5 步：为什么这只是筛选。** 模型假设请求同时到达、每个请求同形、副本能力相同、通信只按归一化长度计费；它没有排队论、没有网络拥塞、没有缓存复用、没有真实内核的并行效率。`comm_ratio` 只是“累计通信成本除以单副本总时间”的题设筛选量，不能替代通信等待测量。真正采用前，必须用固定模型、输入、到达过程、并发和 GPU 布局做端到端服务测量，并同时看 TTFT、TPOT、P95/P99、吞吐、成功率和成本。

**第 6 步：GPU 实验框架。** 5.1 设置 `RUN_GPU_EXPERIMENT=False`、`WORLD_SIZE=2`、`REPLICAS=2`、`TP_DEGREE=1`、`PP_STAGES=1`、`BACKEND_ADAPTER='vllm'`（可改 `sglang`）、`MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'`、`CONCURRENCY=1`、`PROMPT_TOKENS=512`、`MAX_TOKENS=64`、`KV_BYTES_PER_TOKEN=16384`，并把实验分成 `g0_single_replica`、`g1_multi_replica`、`g2_pd_disaggregated` 三组，每组的 `command` 默认是空列表。5.2 只有开启实验时才检查 `torch.cuda.device_count() >= WORLD_SIZE` 和 `torchrun`，并可选读取 79／87／80／70 的上游 JSON 引用；5.4 在 `command` 为空时直接抛 `ValueError`，不会伪造结果；5.5 要求每个结果文件含 `backend/workload/hardware/metrics/evidence_level/failure/decision` 以及 `ttft_ms/tpot_ms/e2e_ms/throughput_tokens_per_s/p95_ms/p99_ms/queue_wait_ms/kv_handoff_bytes/kv_handoff_ms/peak_memory_mb/success_rate` 十一个指标字段，才汇总成 `distributed-benchmark/v1` companion。即使汇总成功，决策也只是 `ready_for_review`（等待复核），不是“已经满足采用条件”。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 服务副本（replica） | 一套能独立处理请求的模型实例，副本之间分担请求 |
| 张量并行（TP） | 把一层内的矩阵计算切到多张卡，需要进程间通信 |
| 流水线并行（PP） | 把不同层放到不同进程，按阶段流水执行 |
| PD 分离 | 预填充进程与解码进程分开，需要传 KV 状态 |
| KV 交接 | 预填充产生的每层 Key/Value 传给解码端 |
| TTFT | 从请求发出到首个输出到达的时间 |
| TPOT | 首词元之后平均每个输出词元的间隔时间 |
| makespan | 一批请求从开始到最后一個请求结束的时间 |
| P95 / P99 | 延迟分布的 95%／99% 分位值，统计对象要先说清 |
| least-loaded | 每次把请求分给当前累计负载最小的副本 |
| 负载偏斜（imbalance） | `(最大负载-最小负载)/平均负载`，越大越不均 |
| 通信比例（comm ratio） | 累计通信成本除以单副本总时间，是本题的筛选量 |
| `ready_for_gpu` | 估算有时间收益且风险阈值都过，值得上真机复测 |

## 小白常见疑问
**Q1：为什么 `comm_ms_per_token` 不直接乘总词元数？**
因为它的单位不是“每词元毫秒”，而是题目约定的归一化成本：输入按每 128 词元、生成按每 64 词元计一个单位，再乘 `tp_degree-1`。测试里 128 输入 + 64 生成正好是 `1+1=2` 个单位，乘 0.03 得 0.06。把它当成“每词元通信时间”会多乘或少乘两个数量级。

**Q2：PD 分离在模型里为什么永远不占优？**
因为 `estimate_request_cost` 对 PD 只加 `handoff_ms`，没有减预填充或解码时间，也没有模拟两个工作池的流水重叠。notebook 明确说“PD仅增加KV交接成本，没有模拟两个工作池的流水收益”。所以这个模型只能回答“交接代价有多大、是否已经大到不值得测”，不能回答“PD 能否更快”；后者要真实服务测量。

**Q3：`assignments=[0,1]` 是怎么来的？第一条请求为什么不放副本 1？**
两条请求同时到达，初始两个副本负载都是 0。第一条按 `min` 选索引 0；分完后副本 0 负载 2.62、副本 1 仍是 0，第二条自然选副本 1。若三条请求，第三条会在 1.31 的副本 1 和 2.62 的副本 0 之间选副本 1，并产生 1.31 的队列等待。

**Q4：`imbalance_ratio` 为什么用平均负载做分母？**
它衡量“最忙和最闲差多少”，除以平均负载后变成相对量，便于跨不同请求规模比较。测试里最大 2.62、最小 1.31、平均 1.965，得到 `(2.62-1.31)/1.965≈0.6667`，小于阈值 0.8。注意这不是方差，也不反映排队时间分布。

**Q5：当前 notebook 能直接得出“副本比单卡快”吗？**
不能。5.1 的 `RUN_GPU_EXPERIMENT=False`，三组 `command` 都是空列表；5.2 只打印 `gpu_ready=false` 和空的 upstream；5.4 在关闭状态下只打印提示；5.5 找不到 JSON 就输出 `summary: pending`。所以当前只能验证成本估算、路由和决策逻辑，不能给出任何真实服务数字。

## 动手验证：怎么知道自己做对了
运行 `test_distributed_inference_project()`，通过后打印 `✅ 请求阶段成本、路由模拟与候选筛选通过基础校验。`。它检查：`estimate_request_cost` 的 `prefill_ms≈1.28`、`decode_ms≈1.28`、`comm_ms≈0.06`、`total_ms≈2.62`；PD 配置的 `handoff_bytes==2048`、`handoff_ms≈2.2`、`total_ms≈4.82`；`simulate_distributed_inference` 的 `routing_strategy=='least_loaded'`、`assignments==[0,1]`、`single_replica_time_ms≈3.93`、`makespan_ms≈2.62`、`throughput_req_per_ms≈0.7634`、`imbalance_ratio≈0.6667`、`comm_cost_ms≈0.09`；PD 汇总的 `serving_mode=='pd_disaggregated'`、`handoff_total_bytes==3072`、`handoff_total_ms>0`；`recommend_distributed_inference_run` 分别给出 `ready_for_gpu`（`next_action=='promote_to_real_cluster_check'`）、`tune`、`reject`。

这些断言都是四舍五入后的数值，改成本系数后要同步更新期望，不要只改代码不改测试。GPU 部分没有保存任何真实 benchmark 结果；要验证 5.4/5.5 的汇总路径，必须先为三组填上能启动、压测并写出同名 JSON 的命令，再核对 11 个指标字段和模型/输入/硬件布局是否一致。没有这些条件时，最诚实的结论就是“后端未配置，待复测”。
