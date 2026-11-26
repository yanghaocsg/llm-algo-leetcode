# INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界 — 小白讲解

> 对应 notebook：[`INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.ipynb`](./INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.ipynb)

## 一句话总结
Serving 的一轮迭代里，调度器要从队列里挑请求、执行引擎要跑一次前向，成本同时受权重读取、历史 KV 读取、新 KV 写入、计算量和未隐藏开销影响。本节用 `estimate_iteration_cost` 估算单轮下界，用 `compute_batch_bounds` 合并槽位、KV、词元、SLO、后端五类上界，再用 `select_operating_knee` 从批量扫描里选“延迟还没越界、吞吐已不再明显增长”的上一档。三个 TODO 都在 CPU 上可测；GPU 实验只用于观察真实形状的耗时与 KV 状态量，当前解码协议仍有待修正。

## 为什么需要它：痛点是什么
初学者容易把“batch 越大吞吐越高”当成结论。但一轮里多接纳几个请求，可能摊薄权重读取和提交开销，也可能让每个请求都要多读一截历史 KV，导致正在生成的请求更久拿不到下一步结果。更麻烦的是，“容量允许”不等于“延迟达标”：KV 缓存还放得下，不代表 P95 迭代时间还满足服务目标。另一个常见错误是把显存总量直接当成可用 KV 容量，忽略了模型权重、工作区、碎片和预留输出。
生活类比：小巴站排队。站台能站 64 人（槽位），车上的行李舱还能放多少包（KV 容量），每趟最多允许多少人上车（单轮词元预算），公司规定末班车不能晚点（SLO），车型本身最多坐多少人（后端限制）。真正能上多少人，取这几个数的最小值；而且“能上车”不代表“这趟车不会晚点”。

## 核心问题
- 一轮迭代的访存字节和计算量分别怎么算？谁是主导项？
- 为什么历史 KV 读取随“活跃请求的上下文总和”增长，而不是只随本轮新词元增长？
- 五类批次上界分别限制什么？并列瓶颈怎样保留？
- 批量扫描为什么“首次越界或收益饱和就停”，而不是找全局最大吞吐？
- GPU 实验里，为什么 12 个形状都跑完、KV 字节也记录到了，决策仍然是 `tune`？

## 解决思路：一步步看懂

**第 1 步：一轮里的三个层次。** 请求队列保存新到达和等待继续的请求；调度器结合优先级、KV 容量和本轮工作量预算选出批次；执行引擎完成一次前向、采样和状态更新。预填充处理尚未计算的输入词元，解码利用已有 KV 推进生成；完成的请求退出，未完成的携带新状态进入下一轮。这个分层很重要：队列决定“候选”，调度器决定“本轮实际执行谁”，执行引擎决定“这一轮花多久”。

**第 2 步（TODO 1）：单轮成本模型。** notebook 给出的总时间近似是

$$
T_{iter}\approx\max(T_{memory},T_{compute})+T_{overhead}.
$$

代码把字节和 FLOPs 分开累加：

```python
scheduled_tokens = scheduled_prefill_tokens + scheduled_decode_tokens
kv_read_bytes = kv_bytes_per_token * sum(active_context_tokens)
kv_write_bytes = kv_bytes_per_token * scheduled_tokens
memory_bytes = weight_bytes + kv_read_bytes + kv_write_bytes
total_flops = flops_per_token * scheduled_tokens
memory_ms = memory_bytes / (effective_bandwidth_gbps * 1e9) * 1e3
compute_ms = total_flops / (effective_compute_tflops * 1e12) * 1e3
dominant = 'memory' if memory_ms >= compute_ms else 'compute'
```

$T_{memory}$ 是字节数除以有效带宽，$T_{compute}$ 是浮点运算量除以有效算力，单位都换成毫秒；`1e9` 对应十进制 GB/s，`1e12` 对应 TFLOP/s。`weight_bytes` 每轮都算一次，是“权重至少要读一遍”的下界，不代表物理上只读一次；`sum(active_context_tokens)` 把每个活跃请求的历史长度加起来，所以上下文越长、请求越多，KV 读取越大；`kv_write_bytes` 只按本轮新处理的词元计。`dominant` 用 `>=`，并列时归为 `memory`。容易写错成把 `scheduled_tokens` 只算 decode、忘记加 prefill，或者把 GB/s 当成 GiB/s。

拿测试第一组手算：`weight_bytes=1_000_000_000`、`kv_bytes_per_token=1000`、上下文 `[100,200]`、prefill 16、decode 4、`flops_per_token=1e9`、带宽 100 GB/s、算力 10 TFLOP/s、开销 0.2 ms。`scheduled_tokens=20`；KV 读 `1000×300=300_000` 字节，KV 写 `1000×20=20_000`，总字节 `1_000_320_000`；`memory_ms=1_000_320_000/1e11×1e3=10.0032`；`total_flops=2e10`，`compute_ms=2e10/1e13×1e3=2.0`；所以 `iteration_ms=10.2032`，主导项 `memory`。第二组 `flops_per_token=1e12`、单 token、算力 1 TFLOP/s，`compute_ms=1000.0`，主导项变成 `compute`。

**第 3 步（TODO 2）：五类上界。** 每个请求按相同上下文和预留输出长度估算 KV：

```python
kv_per_request = kv_bytes_per_token * (context_tokens_per_request + reserved_output_tokens)
limits = {
    'slot': max_num_seqs,
    'kv': available_kv_bytes // kv_per_request,
    'token': max_num_batched_tokens // scheduled_tokens_per_request,
    'slo': slo_batch_limit,
    'backend': backend_batch_limit,
}
final_limit = min(limits.values())
binding_constraints = [name for name, limit in limits.items() if limit == final_limit]
```

槽位限制活跃请求数；KV 用整数除法看还能完整容纳几个请求；词元限制一轮总工作量；SLO 和 backend 是外部给出的批次上限。取最小值后，所有等于最小值的约束都保留，因为“并列瓶颈”对后续调优有意义：只报一个会误导你只改一个参数。测试 `test_batch_bounds` 里短上下文请求 `(64, 64_000_000, 1000, 1000, 100, 128, 1, 48, 64)` 的 KV 上界是 `64_000_000 // 1_100_000 = 58`，最终 `48` 来自 `slo`；把上下文换成 8000 后 KV 上界变成 `64_000_000 // 8_100_000 = 7`，明显小于 58。并列用例 `(8, 80_000, 1000, 9, 1, 8, 1, 8, 8)` 的 `kv_per_request=10_000`，五个上界全是 8，`binding_constraints` 必须同时含 `slot/kv/token/slo/backend`。零容量用例 `available_kv_bytes=0` 得到 `final_limit=0`、唯一边界 `['kv']`，同时 `kv_bytes_per_request == 9_000`；这说明“零上界”是合法的容量结论，不是异常。

**第 4 步（TODO 3）：运行拐点。** 记录按 `batch` 升序扫描，先看延迟，再看相邻吞吐增益：

```python
relative_gain = (record['throughput'] - previous['throughput']) / max(previous['throughput'], 1e-12)
throughput_saturated = relative_gain < min_relative_throughput_gain
```

`max(..., 1e-12)` 防止除零。延迟用 `>` 判断：恰好等于 `max_iteration_ms` 仍可接受；增益用 `<` 判断：恰好等于阈值不算饱和。命中任一条件就 `break`，并把 `selected` 保留在**上一档**；循环结束仍是默认原因 `largest_valid_batch`。测试 `test_operating_knee` 的记录是 `(4, 100, 8ms)`、`(8, 180, 11ms)`、`(16, 190, 18ms)`，阈值 25 ms、默认 0.10：4→8 的增益是 `(180-100)/100=0.8`，8→16 是 `(190-180)/180≈0.0556<0.10`，所以停在 8，原因 `throughput_saturation`。把 `max_iteration_ms` 改成 9.0，8 的 11 ms 越界，停在 4，原因 `slo_violation`。非单调用例 `(4,100,8ms)`、`(8,200,30ms)`、`(16,300,9ms)` 在 10 ms 门槛下会先遇到 8 的 30 ms 越界，直接返回 4，即使后面的 16 看起来更快——这条规则是保守的逐档扩容，不是全局最优搜索；要比较非单调曲线，必须保留全量测量另行分析。

**第 5 步：GPU 实验设计。** 固定 `Qwen/Qwen3-0.6B`、BF16、`GPU_SEED=2026`，扫描 `GPU_BATCH_SIZES=[1,2,4,8]` × `GPU_CONTEXT_LENGTHS=[128,512,1024]` 共 12 个形状；每个形状先预热 5 次、再正式 20 次，记录预填充和解码的中位数/P95、tokens/s、KV 字节、峰值显存和 OOM 恢复状态。自变量是 batch 与初始上下文；因变量是耗时和状态量。这里有一个关键实现细节：解码调用复用同一份 `past_key_values`，而 `_cuda_latency_ms` 会调用 `warmup + repeats = 25` 次，动态缓存因此每次追加 1 个位置，最后统计到的是“初始上下文 + 25”的 KV。`_cache_stats` 递归遍历 cache（兼容 legacy tuple 和 `DynamicCache`），统计张量字节数与张量个数。

**第 6 步：结果怎么读。** notebook 保存的 5.2 输出最后是 `RuntimeError: 模型已加载，但未获得 resolved commit；停止生成不可追溯结果。`，说明这次执行没有产出可追溯的 `GPU_REPORT`；5.4 解释的是历史保存的 `rtx5070ti_iteration_budget_v3`。那份记录里 12 个形状都完成、没有 OOM；KV 字节在 16.734–917.875 MiB 之间，全部满足“批量 ×（初始上下文 + 25）× 114688 字节”，正好对应 5 次预热加 20 次正式调用追加的 25 个位置；峰值约 1.259–6.064 GB（十进制）；记录了 56 个 KV 张量。决策是 `tune` 而不是接受：因为解码输入和注意力掩码仍沿用初始化长度，缓存却在增长，测到的不是固定上下文成本；5.4 要求下次复测前为每次解码恢复相同的 KV 状态、按历史与新增词元构造掩码并核对输出数值。也就是说，这份数据能证明“形状能跑、状态量能统计”，不能证明“固定上下文下的解码延迟曲线”。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 迭代（iteration） | 调度器选一批请求、执行一次前向并更新状态的一轮 |
| 预填充 / 解码 | 处理输入词元 / 逐词元继续生成 |
| KV 读取 | 本轮所有活跃请求的历史 Key/Value 都要读，越长越多 |
| KV 写入 | 本轮新处理词元产生的新状态 |
| Roofline 近似 | 时间至少是访存与计算中较慢的一侧，再叠加未隐藏开销 |
| 主导项（dominant） | `memory_ms` 与 `compute_ms` 中较大的那个 |
| 序列槽位 | 同时能管理的请求数量上限 |
| 批次上界 | 五类限制取最小值；并列时全部保留 |
| SLO | 服务等级目标，规定可接受的延迟或成功率 |
| 运行拐点（knee） | 延迟仍达标、但继续加 batch 收益已低于阈值的上一档 |
| 保守扫描 | 首次越界或饱和就停，不搜索全局最优 |
| makespan / P95 | 整批最后完成时间 / 延迟分布的 95% 分位 |

## 小白常见疑问
**Q1：为什么历史 KV 读取要乘 `sum(active_context_tokens)`，而不是只乘新词元？**
因为本轮前向里，每个活跃请求都要读它全部的 Key/Value 才能做注意力。`kv_read_bytes` 用上下文总和，`kv_write_bytes` 才用本轮 `scheduled_tokens`。测试第一组里上下文 `[100,200]` 产生 300,000 字节读取，而新词元 20 个只产生 20,000 字节写入，读取占了大头。

**Q2：`weight_bytes` 每轮都加，是不是重复计算？**
这是成本模型的下界假设：权重每轮至少要被读一遍，但真实内核可能通过缓存、融合或分块减少重复读取。notebook 的解析明确说“权重是否实际只读一次，需要另从内核访存与缓存行为核对”，所以 `memory_ms` 是对主导项的比较，不是精确的实测时间。

**Q3：为什么 `compute_batch_bounds` 用整除而不是四舍五入？**
因为要算“能完整容纳几个请求”。`64_000_000 // 1_100_000 = 58`，剩下 200,000 字节不够第 59 个请求的 1,100,000 字节；向上取整会给出一个实际放不下的 59。零容量时结果就是 0，`binding_constraints` 为 `['kv']`，这也是合法输出。

**Q4：批量扫描为什么不直接选吞吐最高的点？**
吞吐最高点可能已经超过延迟门槛，或者需要很长的队列等待。`select_operating_knee` 的规则是逐档扩容：先保证 `iteration_ms <= max_iteration_ms`，再看相邻吞吐增益是否还大于 `min_relative_throughput_gain`（默认 0.10）。它故意保守，返回的是“还能接受的上一档”，适合在线服务逐步加批；非单调曲线要保留全量测量另外比较。

**Q5：GPU 记录里 KV 字节为什么是“批量 ×（初始上下文+25）× 114688”？**
因为 `_cuda_latency_ms` 对每个形状先跑 5 次预热、再跑 20 次正式，共 25 次解码调用，每次复用同一份动态缓存并追加 1 个位置；`114688` 是该模型每个请求每词元的 KV 字节数（K 和 V 都算）。所以 5.4 的对照式是在验证“缓存确实按调用次数增长”，不是在验证固定上下文成本。也正因为这样，当前解码曲线要修正协议后才能用于容量决策。

## 动手验证：怎么知道自己做对了
`test_iteration_cost` 要求第一组 `scheduled_tokens == 20`、`memory_bytes == 1_000_320_000`、`memory_ms ≈ 10.0032`、`compute_ms == 2.0`、`iteration_ms ≈ 10.2032`、`dominant == 'memory'`；第二组 `dominant == 'compute'` 且 `iteration_ms == 1000.0`。非法带宽 0、负上下文、空迭代、`flops_per_token=nan` 都必须抛 `ValueError`。`test_batch_bounds` 要求短上下文 `final_limit == 48` 且 `binding_constraints == ['slo']`，长上下文 `limits['kv']` 更小，并列用例五个边界齐全，零容量 `final_limit == 0`、边界 `['kv']`、`kv_bytes_per_request == 9_000`；负可用字节和“上下文+预留=0”抛错。`test_operating_knee` 要求饱和时选 8，只有前两点时也选 8，单点时选 4，`max=9.0` 时原因为 `slo_violation`，非单调曲线选 4；`max_iteration_ms=1.0`、重复 batch、吞吐 0 都抛错。三个测试跑完打印 `PASS: 迭代成本、五类上界与保守批量扫描测试通过。`

GPU 部分没有可复用的当前结果：5.2 保存的输出以 `RuntimeError` 结束，5.3 的 `GPU_REPORT` 会是 `not_run`（除非你在同一内核里成功重跑）。5.4 的历史数字只能当“记录格式和读法”的示例；要得到自己的结论，先修好每次解码的缓存恢复与掩码长度，再按 `RUN_LABEL` 保存 JSON。
