# PAR-REPLICA. 服务副本与请求路由 — 小白讲解

> 对应 notebook：[`PAR-REPLICA_Serving_Replica_Routing.ipynb`](./PAR-REPLICA_Serving_Replica_Routing.ipynb)

## 一句话总结
同一个模型起多个服务副本后，请求该发给谁？本节用三个函数把问题拆开：`select_replica` 实现轮询、队列感知和带"额外等待容忍"的局部性感知三种策略；`simulate_replica_routing` 按到达顺序推进每个副本的可用时间、缓存命中与会话归属；`summarize_replica_service` 汇总排队 P95/P99、吞吐、缓存命中率和请求数偏斜。最后在同一请求流上比较三种策略，回答"为复用缓存多等一会儿值不值"。

## 为什么需要它：痛点是什么
初学者容易把负载均衡理解成"平均分配请求"：轮询最公平、每个副本请求数一样就万事大吉。但真实服务里，请求有长有短，某个副本可能正忙；更麻烦的是前缀缓存——如果某个副本已经缓存了相同开头的 K/V，把请求送过去能省掉重复预填充。于是"最空闲的副本"和"缓存命中的副本"不一定是同一个。

生活类比：超市结账。轮询是"按顺序排到下一个收银台"；队列感知是"看哪队排得短"；局部性感知是"你上次在这台机器办过业务，资料还在，但要多等 3 分钟——你愿不愿意等"。`locality_slack_ms` 就是这个"最多愿意多等多久"。

## 核心问题
- 训练的数据并行和服务副本有什么不同？
- 队列感知为什么按"预计可用时间"而不是"排了几个请求"选副本？
- 局部性感知如何在缓存/会话偏好和排队时间之间取舍？`locality_slack_ms` 怎么用？
- 时间线里开始时间、结束时间、缓存命中服务时间怎么算？为什么同一副本不能重叠？
- 排队 P95、吞吐、缓存命中率、负载偏斜分别怎么定义，容易误读在哪？

## 解决思路：一步步看懂

**第 1 步：先分清训练副本与服务副本（Step 1）。** 训练数据并行让多个进程处理不同数据再同步梯度；服务副本是能独立执行请求的一套模型实例，请求入口为它选队列。notebook 特别提醒：一个副本内部也可能用模型并行，所以"两个副本"不等于"两张 GPU"。缓存局部性表示所需 KV 状态已在某个实例附近；会话粘连让同一会话优先回原实例。两者都要和排队时间一起衡量。

**第 2 步：三种策略与等待公式（Step 2）。** 令 $a$ 为到达时间，$u_r$ 为副本 $r$ 的预计可用时间，则预计开始时间为 $\max(a,u_r)$。队列感知选择

$$
r_q=\arg\min_r \max(a,u_r),
$$

局部性候选还要满足

$$
\max(a,u_r)\le \max(a,u_{r_q})+\delta,
$$

$\delta$ 就是 `locality_slack_ms`。notebook 的例子：最早 11 ms 可用、缓存副本 14 ms 可用，缓存选择比基线多等 3 ms；容忍 3 ms 时可选，容忍 2 ms 时回退。注意代码里实际比较的是 `waits = max(0, available_at - arrival)`，与 $\max(a,u_r)-a$ 等价。

**第 3 步：TODO 1——选择副本并给出原因。** 参考实现按优先级检查：

```python
waits = [max(0.0, replica['available_at_ms'] - arrival_ms) for replica in replicas]
queue_choice = min(range(len(replicas)), key=lambda index: (waits[index], index))
...
if session_id in session_owner:
    owner = session_owner[session_id]
    if waits[owner] <= waits[queue_choice] + locality_slack_ms:
        return owner, 'sticky_session'
...
prefix_choice = min(prefix_candidates, key=lambda index: (waits[index], index))
if waits[prefix_choice] <= waits[queue_choice] + locality_slack_ms:
    return prefix_choice, 'prefix_locality'
return queue_choice, 'queue_fallback'
```

几个细节：轮询直接返回 `round_robin_index % len(replicas)`，不看队列；队列感知用 `(等待, 索引)` 做字典序比较，等待相同时取编号小的，保证结果确定；会话粘连优先于前缀匹配，但**两者都必须通过 slack 门槛**，否则回退到 `queue_fallback`。非法策略名或空副本列表会抛 `ValueError`。

`test_replica_selection_contract` 用两个副本验证：副本 0 可用时间 20 ms、缓存前缀 `'a'`，副本 1 可用时间 5 ms、无缓存，请求到达 0 ms：

- 轮询索引 3：3 % 2 = 1，选副本 1；
- 队列感知：副本 1 等待 5 ms 更短；
- 局部性 slack=20：副本 0 等待 20 ≤ 5+20，返回 `(0, 'prefix_locality')`；
- 局部性 slack=5：20 > 5+5，回退 `(1, 'queue_fallback')`。

**第 4 步：TODO 2——推进时间线。** 参考实现：

```python
start_ms = max(arrival_ms, replica['available_at_ms'])
cache_hit = request.get('prefix_key') in replica['cached_prefixes']
service_ms = float(request['service_ms']) * (1.0 - cache_hit_speedup if cache_hit else 1.0)
finish_ms = start_ms + service_ms
```

`start_ms` 取到达时间和副本可用时间的较大值，保证请求不会在到达前执行、也不与同副本上一个请求重叠；`cache_hit` 判断前缀键是否已在该副本的 `cached_prefixes` 集合里；命中后服务时间乘以 `1 - cache_hit_speedup`，默认 `cache_hit_speedup=0.30`，即缩短 30%。notebook 明确说这是**题目的成本假设**，不是测出来的加速比，也没有模拟真实 KV 大小或缓存驱逐。每处理完一个请求，更新 `available_at_ms=finish_ms`、`request_count += 1`，并把前缀键加入缓存集合、把 `session_id` 的归属更新为当前副本，供后续请求使用。

`test_routing_timeline_contract` 对 4 条请求断言：`start_ms >= arrival_ms`、`finish_ms >= start_ms`、`queue_wait_ms == start_ms - arrival_ms`，并且按副本记录"上一次结束时间"，要求下一次 `start_ms >= previous_finish`，最后至少有一条 `cache_hit`。

**第 5 步：TODO 3——指标定义。** 参考实现：

```python
p95_queue_wait_ms = nearest_rank(waits, 0.95)
throughput_requests_per_s = len(records) * 1000.0 / window_ms
load_skew_ratio = (max(counts) - min(counts)) / mean_count if mean_count else 0.0
```

`nearest_rank` 的实现是 `index = max(0, ceil(quantile * len(values)) - 1)`，即 P95 取排序后第 $\lceil 0.95n\rceil$ 个值（从 1 数），不是插值。`window_ms` 是最晚 `finish_ms` 减最早 `arrival_ms`，吞吐按"整个观察窗口内完成多少请求"算。`counts` 是每个副本处理的请求数，`mean_count = len(records)/num_replicas`，偏斜用 `(max - min)/mean`——**只看请求数，不看忙碌时间**，所以少量长请求占满一个副本时这个指标可能偏低。

`test_queue_metric_meaning` 用固定数据核对：等待 `[0,1,2,100,0]`，完成时间使窗口为 101 ms。P95 的索引是 $\lceil 0.95\times5\rceil-1=4$，排序后第 5 个值是 100；P99 同样取到 100；吞吐 = 5×1000/101 ≈ 49.50 req/s；命中 1/5=0.2；请求数 `[4,1]`，均值 2.5，偏斜 `(4-1)/2.5=1.2`。空输入返回全 0 的字典，`request_count=0`。

**第 6 步：策略比较与实验设计（Step 5）。** `compare_replica_strategies` 对同一请求流分别跑三种策略，排序键是 `(p95_queue_wait_ms, -throughput_requests_per_s)`，即先看排队 P95，再看向吞吐高者。`test_service_metrics_and_strategy_comparison` 还断言传入的原始列表没被改写（内部用 `deepcopy`），三种策略齐全，且 `cache_hit_rate` 在 0–1、吞吐大于 0。

可选 GPU 实验（5.x）把固定成本列表分给两个进程，比较轮询和"最小累计负载"分配，每个成本单位执行一次 768×768 的 BF16 方阵乘法，`WORLD_SIZE=2`、`WARMUP=2`、`REPEATS=5`。它读取的字段是 `request_cost_units_per_rank`、`rank_elapsed_ms`、`max_rank_makespan_ms`、`throughput_requests_per_s_proxy`。这不是在线路由，而是**已知成本下的静态分配**，用来观察任务不均如何影响整批完成时间；真实服务的到达、缓存和网络延迟要另测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 服务副本 | 能独立执行请求的一套模型实例，内部也可能用模型并行 |
| 轮询（round-robin） | 按请求序号轮流分配，不看队列 |
| 队列感知 | 选预计最早可用的副本，而不是排队长最短的 |
| 局部性感知 | 优先会话原副本或缓存命中副本，但受等待上限约束 |
| `locality_slack_ms` | 为复用缓存最多愿意比最短等待多等的毫秒数 |
| 会话粘连（sticky session） | 同一会话尽量回到原副本，方便复用状态 |
| 排队等待 | `start_ms - arrival_ms`，不含执行和传输 |
| nearest-rank 分位数 | 排序后取第 $\lceil qn\rceil$ 个值，不做插值 |
| `load_skew_ratio` | 各副本请求数极差除以平均值，只看数量不看时长 |
| `cache_hit_speedup` | 命中后服务时间缩短比例，是题目假设不是实测 |

## 小白常见疑问

**Q1：局部性感知为什么不总是选缓存命中的副本？**
因为可能等太久。`select_replica` 只有在候选的等待不超过 `waits[queue_choice] + locality_slack_ms` 时才选它，否则回退 `queue_fallback`。测试里 slack=5 时 20 ms 的缓存副本就被放弃了，slack=20 时才选中。

**Q2：队列感知为什么不用"队列里排了几个请求"？**
因为请求长度不同。排 2 个短请求可能比排 1 个长请求更早空出来，所以代码比较的是 `available_at_ms` 推算的等待时间，而不是 `request_count`。

**Q3：为什么 `start_ms` 要取 `max(arrival_ms, available_at_ms)`？**
两个约束同时成立：请求不能在到达前开始，同一副本也不能同时执行两个请求。`test_routing_timeline_contract` 按副本检查 `start_ms >= previous_finish`，正是防止重叠。

**Q4：缓存命中后服务时间乘 `1 - 0.30`，这个 30% 是测出来的吗？**
不是。notebook 把它定义为 `cache_hit_speedup` 的默认成本假设，并明确说没有模拟真实 KV 大小或缓存驱逐。它只用于让三种策略在时间线上产生可比较的差异。

**Q5：`load_skew_ratio=1.2` 说明什么？**
说明请求数最多的副本比最少的多 1.2 倍平均值。测试数据里副本 0 处理 4 条、副本 1 处理 1 条，均值 2.5，极差 3，所以是 1.2。它不反映忙碌时间，也不直接等于延迟差异。

## 动手验证：怎么知道自己做对了

测试区直接调用四个测试，最后打印 `✅ Serving 副本选择、时间线与策略对照验证通过。`

- `test_replica_selection_contract`：如上，轮询得 1、队列感知得 1、slack=20 得 `(0,'prefix_locality')`、slack=5 得 `(1,'queue_fallback')`。
- `test_queue_metric_meaning`：`p95_queue_wait_ms == 100.0`、`p99_queue_wait_ms == 100.0`、吞吐 `isclose(5000/101)`、`cache_hit_rate == 0.2`、`load_skew_ratio == 1.2`，空记录 `request_count == 0`。
- `test_routing_timeline_contract`：4 条记录满足到达/完成/等待关系，同副本不重叠，至少一次缓存命中。
- `test_service_metrics_and_strategy_comparison`：三种策略集合完整、排序与 `(p95, -throughput)` 一致、原始 workload 未被改写。

CPU 部分没有保存执行输出，以断言为准。GPU 部分默认 `RUN_MODE='dry_run'`，只打印配置；5.4 表格的"当前结果"写的是"运行后填写"，notebook 没有提供已保存的 GPU 数值，不能把任何耗时或加速比当成实测结论。
