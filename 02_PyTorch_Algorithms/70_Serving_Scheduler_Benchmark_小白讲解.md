# 70. 推理服务调度基准 — 小白讲解

> 对应 notebook：[`70_Serving_Scheduler_Benchmark.ipynb`](./70_Serving_Scheduler_Benchmark.ipynb)

## 一句话总结
本节先用一个单 worker 的 CPU 队列模型 `simulate_serving_scheduler`，把 FIFO、最短请求优先（shortest）、业务优先级（priority）三种排序策略的等待时间、完成顺序、吞吐和公平性算清楚；再用三个 TODO 学会"先核对对照契约、再核对请求轨迹、最后给采用结论"这条决策链；最后（可选）在单卡 vLLM 上用同一请求流配对比较 FCFS 与 priority，并登记了 `accept` 的实测结论。

## 为什么需要它：痛点是什么
初学者的直觉通常是"吞吐高就是好"。但调度会做一次**重新分配**：把长请求往后排，短请求就能更快结束；把高优先级请求提前，低优先级请求的等待就会变长。只看平均延迟或总吞吐，会把"谁的等待被转移了"藏起来；很多对比实验甚至连"两边跑的是不是同一个请求流"都没核对。

生活类比：超市收银台。只开一条队（FIFO）时，前面的人买了满满一车，后面只拿一瓶水的人也得等；开"10 件以内快速通道"（shortest）能让小单先走，但大单顾客等更久；会员优先（priority）同理，收益和代价必须一起看。

## 核心问题
- 请求在 pending / active / completed 之间怎样流转？一轮批次的服务时间由谁决定？
- FIFO、shortest、priority 三种策略分别改变什么，为什么 priority 约定"数值越小越优先"？
- `queue_wait_ms`、`makespan_ms`、`throughput_tps`、`utilization`、`fairness` 各自怎么算，哪些量不能混为一谈？
- 一个调度候选要经过哪些检查才能给出 `accept` / `tune` / `reject`？
- 真实 GPU 对照里，怎样用"首次内容返回顺序分数"证明 priority 确实生效，同时确认低优先级没有被牺牲到不可接受？

## 解决思路：一步步看懂

**第 1 步：三态队列与"最慢请求决定窗口"。** `simulate_serving_scheduler(requests, policy='fifo', batch_size=1, batch_speedup=1.0)` 先把请求按 `(arrival_ms, _index)` 排序，然后循环：把已到达的请求放进 `queue`，调用 `select_next_batch` 选出一批，批次持续时间是

$$\text{duration}=\frac{\max_{i\in batch}\text{service\_ms}_i}{\text{batch\_speedup}}$$

也就是**批内最慢的请求决定这一轮多久**，同批其他请求虽然先算完，也要等窗口结束才一起标记完成。这是 CPU 模型"整批结束才释放"的口径，和真实后端的连续批处理（完成一个补一个）不同。`batch_speedup` 是教学用的批处理收益系数，默认 1.0 表示没有加速。

`select_next_batch` 的三个分支：`fifo` 保持队列顺序；`shortest` 按 `(service_ms, _index)` 排序；`priority` 按 `(priority, arrival_ms, _index)` 排序。priority 的键里带到达时间和原始下标，是为了让同优先级请求仍"先到先得"，结果稳定可复现。非法 `policy`、`batch_size < 1`、`batch_speedup <= 0` 都抛 `ValueError`。

**第 2 步：用 3 个请求手算一遍。** 测试常量 `REQUESTS` 是 r1（到达 0，服务 10，20 token，priority 5）、r2（到达 0，服务 2，priority 0）、r3（到达 1，服务 3，priority 2）。

- FIFO、batch_size=1：r1 在 0–10 执行，r2 等到 10–12，r3 等到 12–15。`makespan_ms = 15`，`throughput_tps = (20+20+20)/(15/1000) = 4000`，`utilization = 15/15 = 1.0`，三次 `queue_wait_ms` 是 0、10、11，平均 7。
- `fairness = max(0, 1 - (max_wait - min_wait) / max(mean_wait, 1.0))`；这里极差 11、均值 7，算出负数后被截断为 0.0。它只服务于本题，不是通用公平性指标。
- shortest：t=0 先选服务 2ms 的 r2；t=2 时 r3 已到达，队列是 r1(10)、r3(3)，选 r3；最后才是 r1。priority：r2（0）→ r3（2）→ r1（5）。
- FIFO、batch_size=2：第一轮装 r1、r2，持续 `max(10,2)=10`；r3 已在 1ms 到达，第二轮 10–13 执行。`makespan_ms = 13 < 15`，说明"合并两轮"即使总服务量不变，窗口也可能缩短。

`makespan` 定义为"最后一个完成时刻 − 最小到达时刻"，所以全部 `arrival_ms` 加 100 后仍是 15，测试验证了这个平移不变性。

**第 3 步：TODO 1——先证明两组实验可比。** `validate_scheduler_contract` 的骨架已经列出 `fixed_fields = ('model_id', 'revision', 'backend', 'request_stream_id', 'measurement_key')`，并把空值也算作 mismatch。需要补的是：

```python
fixed_ok = not mismatches
changed_once = (bool(baseline.get('scheduler_policy')) and bool(candidate.get('scheduler_policy'))
                and baseline['scheduler_policy'] != candidate['scheduler_policy'])
issues = list(mismatches) + ([] if changed_once else ['scheduler_policy'])
return {'contract_ok': fixed_ok and changed_once, 'issues': issues}
```

容易写错的两点：`None == None` 是 `True`，所以不能只判断"两侧相等"，必须先排除空值——测试里两侧都删掉 `model_id` 仍要求 `contract_ok` 为假；策略必须"非空且不同"，两边都写 `fifo` 就没有解释变量。`issues` 要同时保留固定字段差异和策略问题。

**第 4 步：TODO 2——把"缺证"和"造假"分开。** `build_scheduler_trace_evidence` 要回答两件事：请求是不是都完成了、指标是不是都采到了。参考答案先取 `trace_complete = bool(request_ids) and bool(completed_ids)`，再逐个记录检查：

```python
if expected and len(expected) != len(set(expected)):
    hard_failures.append(f'{name}.duplicate_request_ids')      # 请求清单自己重复
if actual and (len(actual) != len(set(actual)) or (expected and set(actual) != set(expected))):
    hard_failures.append(f'{name}.invalid_completed_ids')      # 完成清单重复或对不上
if not expected or not actual:
    missing.append(f'{name}.request_trace')                    # 没采集，不是失败
```

另外 `baseline` 与 `candidate` 的 `request_ids` 必须完全相同，否则是两条不同流量，直接进硬失败；任一记录带 `failure` 也进硬失败。`required_metrics` 是 `ttft_p95_ms`、`tpot_p95_ms`、`throughput_tps`、`completion_rate`、`fairness`、`peak_memory_gb`。

骨架里的 `missing` 只查 candidate，参考答案改成两侧都查（`f'{name}.{field}'`）：baseline 缺 TTFT 时 `candidate - baseline` 根本算不出来，不能因为候选有值就当作证据齐全。这就是"缺证 `tune`、硬失败 `reject`"的分界——重复完成或丢请求意味着数据不可信，补采也救不了；没采集只是还没做。

**第 5 步：TODO 3——先失败、再缺证、最后比数值。** 顺序不能反：

```python
if evidence['hard_failures']:
    return {'decision': 'reject', 'next_action': 'repair_contract_or_backend'}
if evidence['missing_evidence']:
    return {'decision': 'tune', 'next_action': 'collect_request_trace'}
latency_ok = evidence['ttft_p95_ms'] <= max_ttft_p95_ms and evidence['tpot_p95_ms'] <= max_tpot_p95_ms
service_ok = (evidence['throughput_gain'] >= min_throughput_gain
              and evidence['completion_rate'] >= min_completion_rate
              and evidence['fairness'] >= min_fairness)
```

如果先比较数值，`None <= 200.0` 会直接抛 `TypeError`；测试 `missing = {... 'ttft_p95_ms': None, 'throughput_tps': None}` 就是专门抓这个错误的。还要注意 `throughput_gain` 是"候选减基线"的增量，不是比值；`completion_rate` 的门槛允许恰好等于（`>=`）。

**第 6 步：CPU 侧的汇总与决策函数。** `compare_scheduler_to_baseline` 输出五个差值：`ttft_delta_ms`、`tpot_delta_ms`、`throughput_gain_tps`、`fairness_delta`、`utilization_delta`。用测试常量算：180−220=−40，36−42=−6，205−180=+25，0.81−0.78=+0.03，0.79−0.70=+0.09。

`recommend_serving_scheduler_run` 是教学版三态：`latency_ok` 要求 TTFT **严格变小**、TPOT 不上升；`throughput_ok` 看增量是否达到 `min_throughput_gain`；`fairness_ok` 看候选绝对值是否达到 `min_fairness`。三者全过 `accept`；否则吞吐增量非负且利用率不下降 `tune`；再否则 `reject`。测试里 `weak_candidate`（TTFT 195、吞吐 202、fairness 0.76）是"延迟和吞吐略好、公平性差 0.04"→ `tune`；`bad_candidate`（TTFT 260、吞吐 170）→ `reject`。

**第 7 步（可选）：真实 GPU 上怎么设计对照。** 5.1 只允许改一个自变量——服务端 `scheduling-policy`：G0 是 `fcfs` 且不发送 priority，G1 是 `priority` 且发送（数值越小越早）。固定项有模型 `Qwen/Qwen2.5-0.5B-Instruct`、`NUM_PROMPTS=8`、`MAX_TOKENS=96`、`CONCURRENCY=8`、`MAX_NUM_SEQS=2`、`WARMUP=2`、`REPEATS=5`、`MAX_MODEL_LEN=2048`、`GPU_MEMORY_UTILIZATION=0.80`、`ENFORCE_EAGER=True`。

`MAX_NUM_SEQS=2` 必须小于请求数 8，否则 8 个请求一次全进模型，没有等待队列，priority 无从体现；`CONCURRENCY >= NUM_PROMPTS` 是为了同步发车，避免提交抖动主导到达顺序。四个门槛是：顺序分数至少提高 `MIN_PRIORITY_ORDER_GAIN=0.15`；吞吐回退不超过 `MAX_THROUGHPUT_REGRESSION=10%`；TTFT P95 回退不超过 `MAX_TTFT_P95_REGRESSION=25%`；低优先级 E2E P95 回退不超过 `MAX_LOW_PRIORITY_E2E_REGRESSION=50%`。5.5 用线性插值 `percentile`，把每轮 `priority == 5` 的请求单独取 E2E P95 再平均；5.6 用候选值除以基线值得到三个 ratio。登记结果是：顺序分数 0.4625 → 0.8625（提高 0.40），吞吐和 TTFT 通过，低优先级 E2E P95 增加约 6.9%，结论 `accept`，但限定"当前优先级混合 workload"，长期饥饿要另测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| pending / active / completed | 已到达待执行 / 本轮正在执行 / 已结束并移出队列 |
| FIFO / shortest / priority | 先到先服务 / 短请求先走 / 业务优先级（数值越小越优先） |
| batch_size / batch_speedup | 一轮最多几个请求 / 教学模型里的批处理加速系数 |
| queue_wait_ms | 从到达至获准执行的等待；`ttft_ms` 只是同值别名，不是真实首 token 延迟 |
| makespan_ms | 从最早到达至最后完成的观察窗口 |
| throughput_tps / utilization | 总 token 数 ÷ 窗口（秒） / 忙碌时间 ÷ 窗口 |
| fairness | `1 − 等待极差 / max(平均等待, 1ms)`，截断到非负，仅本题使用 |
| 首次内容顺序分数 | 不同优先级请求对中，高优先级先返回的比例；同优先级对不计 |
| 配对契约 | 模型、版本、后端、请求流、测量方式一致，只有调度策略变化 |

## 小白常见疑问
**Q1：为什么 priority 是"数值越小越优先"？**
`select_next_batch` 的排序键是 `item.get('priority', 0.0)`，升序排列，所以 0 排在 5 前面；测试里 r2 的 `priority=0` 第一个被选中。G1 组发送的 priority 也是"数值越小越早"。反过来改的话，排序键和 5.6 的顺序分数定义要一起改。

**Q2：CPU 模型里 `ttft_ms` 等于 `queue_wait_ms`，为什么不能当成真实 TTFT？**
代码注释写明"兼容旧示例的别名，不是首 token 实测值"。CPU 模型只模拟排队，没有首轮前向和网络传输；Step 3 把"到达、获准执行、首次收到内容、流结束"列为四个事件，就是提醒不要混用。

**Q3：既然 `accept` 了，是不是说明可以直接上线？**
不是。5.6 的 `accept` 只表示"固定模型、8 条请求、12GB 单卡、5 次重复"下四项门槛都过；notebook 还要求交给 72 节复核，并提示低优先级长期饥饿要另测。CPU 测试也明确不把教学耗时当作真实 GPU 性能。

**Q4：硬失败和缺证为什么要分开？**
重复完成、完成清单对不上、请求流不一致，说明这次运行本身不可信，补采也不能把错的变成对的，所以 `reject`；指标没采集只是还没做，补采后可重新判断，所以 `tune`。`test_scheduler_project_contract_and_trace` 用 `completed_ids=['r1','r1','r3']` 制造前者，用 `ttft_p95_ms=None` 制造后者，分别断言 `reject` 和 `tune`。

## 动手验证：怎么知道自己做对了
`test_serving_scheduler_benchmark_template()` 依次调用 6 个测试，全过才打印 `测试通过：调度对照契约、请求轨迹与采用状态均通过。`。注意这个调用写在题目区末尾，TODO 没填时会先抛 `NotImplementedError`。

- `test_serving_simulation_contract`：FIFO 下 `request_count=completed_count=3`、`makespan_ms=15.0`、`throughput_tps=4000.0`、`utilization=1.0`、平均等待 7.0、首行 `queue_wait_ms=0.0`、第二行 `ttft_ms=10.0`；每个请求满足 `arrival <= admitted <= finished`；平移 100ms 后 makespan 不变；重复 `request_id` 抛 `ValueError`；`batch_trace` 顺序是 `[['r1'],['r2'],['r3']]`。
- `test_policy_batching_and_input_guards`：shortest 首条 `r2`、末条 `r1`；priority 首轮 `['r2']`；batch_size=2 时 `makespan_ms=13.0 < 15`；空输入返回 0；非法 policy、`batch_size=0`、`batch_speedup=0.0` 抛 `ValueError`。
- `test_result_summary_contract`：`run_count=2`、最低 TTFT 是 `priority_scheduler`、平均吞吐 192.5；空列表返回 `{'run_count': 0, 'best_latency_run': None, 'avg_throughput_tps': 0.0}`。
- `test_baseline_comparison_contract`：五个差值分别为 −40、−6、+25、+0.03、+0.09。
- `test_decision_policy_contract`：完整候选 `accept` 且 next_action 是 `promote_to_serving_rollout`；weak 是 `tune`；bad 是 `reject`。
- `test_scheduler_project_contract_and_trace`：同后端、策略 fifo→priority 时 `contract_ok` 为真；后端换成 `other`、两侧都缺 `model_id` 时为假；正常两组 `hard_failures` 与 `missing_evidence` 为空；门槛 `(200.0, 40.0, 20.0, 1.0, 0.8)` 下 `accept`；完成清单重复 `reject`；缺 TTFT/吞吐 `tune`；空轨迹不等于 `accept`。

notebook 没有保存执行输出，以上阈值和结果都以断言与常量文本为准。Step 5 的真实结果来自 5.6 的登记表（`rtx5070ti_scheduler_priority_v4`），不是现场跑出的。
