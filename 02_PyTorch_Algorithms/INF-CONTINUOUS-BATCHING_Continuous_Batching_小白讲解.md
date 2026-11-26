# INF-CONTINUOUS-BATCHING. 连续批处理 — 小白讲解

> 对应 notebook：[`INF-CONTINUOUS-BATCHING_Continuous_Batching.ipynb`](./INF-CONTINUOUS-BATCHING_Continuous_Batching.ipynb)

## 一句话总结
本节用一次“迭代边界”的批次重组，把连续批处理讲成四个可测试的动作：算本轮 token 需求（`iteration_token_demand`）、释放终止请求并归还 KV block（`release_terminal_requests`）、按 slot → KV → token 三道门槛补入等待请求（`form_iteration_batch`）、执行后推进 Prefill/Decode 状态（`advance_iteration`）。核心结论是：补位不是“有空位就塞”，新请求还要有 KV 容量和本轮 token 预算；而且不能跳过队首，否则会破坏等待顺序。

## 为什么需要它：痛点是什么
四个请求一起开始生成，输出长度分别是 16、64、32、96 个 token。最短的那个最先结束，但如果调度器必须等整批四个都完成才提交下一组，空出来的并发槽位就只能干等。连续批处理（continuous batching）的做法是：在每个解码迭代的边界检查一次，谁结束了就移出并释放资源，等待队列里的新请求立刻补进来。

生活类比：四人一桌吃火锅，每人点不同数量的菜。固定批次像“必须等四个人都吃完才允许下一桌进来”，空椅子一直闲着；连续批处理像“谁吃完谁走，门口排队的人马上补位”。但补位不是无条件的：椅子（slot）、桌上的锅和菜架空间（KV cache）、这一轮还能下多少菜（token budget）都得够；而且如果队首的人还没吃完，不能先放后面的人进去。

初学者容易把“槽位空闲”当成“一定能加入”，也容易忽略请求结束时要归还它占的 KV 块。本节就是把这三类预算和状态机拆开。

## 核心问题
- 一个请求在当前迭代到底要处理多少 token？Prefill 和 Decode 有什么区别？
- 哪些请求应该在边界被释放？释放的 KV block 怎样回到可用容量？
- 新请求要同时满足哪三道门槛？为什么阻塞原因要按固定顺序报告？
- 为什么不能跳过队首去接纳后面的请求？
- 执行完一轮后，Prefill→Decode、Decode→completed 的状态怎样推进？
- 真实服务里，滚动补位相对整批屏障能改善哪些指标、代价是什么？

## 解决思路：一步步看懂

**第 1 步：先定状态与账本。** `TERMINAL_STATES = {'completed', 'cancelled', 'timed_out', 'failed'}` 定义了四种终止状态。每个请求是一个字典，含 `id`、`phase`（`prefill` 或 `decode`）、`remaining_prefill_tokens`、`remaining_decode_tokens`、`kv_blocks`、`status`（`active`/`waiting` 或终止状态）。`_validate_request_state` 负责最小校验：id 是非空字符串、阶段合法、状态合法、三个计数都是非负整数。所有函数都先校验再计算，非法输入抛 `ValueError`。

**第 2 步（TODO 1）：算本轮 token 需求。** 参考答案：

```python
prefill_demand = min(max(request['remaining_prefill_tokens'], 0), prefill_chunk_size)
decode_demand = 1 if request['remaining_decode_tokens'] > 0 else 0
```

Prefill 阶段一次可以处理一块输入，但有两个上限：不能超过剩余 prefill token，也不能超过本轮允许的 chunk 大小，所以取两者较小值。Decode 阶段每次只生成一个 token，只要还有剩余解码工作就排 1 个，否则排 0。写成式子（$R_p$ 是剩余 prefill，$R_d$ 是剩余 decode，$S$ 是 chunk 上限）：

$$
d_{\text{prefill}}=\min(R_p,S),\qquad
d_{\text{decode}}=\mathbf{1}[R_d>0].
$$

测试 `test_phase_token_demand`：`remaining_prefill=10, chunk=4` 得 4；`remaining_prefill=2, chunk=4` 得 2；`remaining_decode=3` 得 1。容易写错成 Prefill 直接返回 `remaining_prefill_tokens`，忽略 chunk 上限；或者 Decode 返回剩余量而不是 1。

**第 3 步（TODO 2）：释放终止请求。** `release_terminal_requests(active_requests)` 先把所有 id 收集起来查重，重复 id 直接报错（同一请求不能被释放两次）。然后：

```python
is_terminal = request['status'] in TERMINAL_STATES
target = terminal if is_terminal else active
target.append(deepcopy(request))  # 返回副本，不改调用方对象
```

只有 `completed/cancelled/timed_out/failed` 这四种状态会被移入 `terminal`；其余留在 `active`。最后

```python
released_blocks = sum(item['kv_blocks'] for item in terminal)
```

把终止请求占用的 KV 块加总归还。测试 `test_terminal_release`：一个 decode 请求 `blocks=3` 保持活跃，一个 `completed` 请求 `blocks=4` 被释放，结果是 `kept=['a']`、`terminal=['b']`、`released=4`。

**第 4 步（TODO 3）：三道门槛组成本轮批次。** `form_iteration_batch(active_requests, waiting_requests, max_batch_size, free_kv_blocks, max_num_batched_tokens, prefill_chunk_size)` 的顺序很重要：

1. 先 `release_terminal_requests`，把归还的块加到可用容量：`available_blocks = free_kv_blocks + released`。
2. 计算空闲槽位：`free_slots = max_batch_size - len(active)`。活跃请求一定进入本轮 batch，并各自算 `scheduled_tokens`。
3. 算剩余 token 预算：`free_tokens = max_num_batched_tokens - sum(active 的 scheduled_tokens)`。如果 `free_slots < 0` 或 `free_tokens < 0`，说明活跃请求已经超预算，直接报错。
4. 按等待队列的原始顺序逐个尝试，参考答案：

```python
block_reason = (
    'slot' if free_slots <= 0 else
    'kv_blocks' if request['kv_blocks'] > available_blocks else
    'token_budget' if demand > free_tokens else None
)
if block_reason is not None:
    blocked.append({'id': request['id'], 'reason': block_reason})
    break
```

三个条件按 `slot → kv_blocks → token_budget` 的固定顺序检查，报告第一个不满足的原因。满足后扣减三份预算：`free_slots -= 1`、`available_blocks -= item['kv_blocks']`、`free_tokens -= demand`，并把请求状态改成 `active`。
关键设计是 `break`：一旦队首被阻塞，后面的请求也不再尝试。这就是“不能跳过队首”。如果允许跳过，短请求可能一直插队，长请求会被饿死，等待顺序也不再可解释。
手算 `test_batch_gates_and_block_reason`：活跃 a 是 decode，`blocks=2`、需求 1；`max_batch=3` 所以 `free_slots=2`，`max_tokens=4` 所以 `free_tokens=3`。等待队列第一个 p 是 prefill，剩余 6、chunk 4，需求 `min(6,4)=4`；slot 够、p 的 `kv_blocks=2 ≤ available=4`，但 `4 > 3`，所以 `reason='token_budget'`。批次里只有 a，`waiting` 仍是 `['p','d']`（p 没被消费），`free_token_budget=3`。
`test_all_gates_and_input_preservation` 分别触发三种阻塞：`(slots=0, blocks=4, tokens=4)` 得 slot；`(1, 1, 4)` 时 p 需要 2 个块，得 kv_blocks；`(1, 4, 1)` 时需求 3 超过 1，得 token_budget。最后 `(1, 2, 3)` 成功准入，`free_kv_blocks=0`、`free_token_budget=0`、`scheduled_tokens=3`，并且原 `waiting` 列表未被修改（函数内部用 `deepcopy`）。

**第 5 步（TODO 4）：推进状态。** `advance_iteration(batch)` 先查重（同一请求不能在一轮里推进两次），再校验执行量：必须是非负整数，且 Decode 每轮最多推进 1 个 token。参考答案：

```python
key = 'remaining_prefill_tokens' if request['phase'] == 'prefill' else 'remaining_decode_tokens'
remaining_tokens = request[key] - amount
if remaining_tokens < 0:
    raise ValueError('scheduled_tokens 不能超过阶段剩余工作量')
```

扣完后：Prefill 的剩余量归零就切到 `decode`；Decode 的剩余量归零就把 `status` 改成 `completed`。测试 `test_iteration_state_advance`：prefill 剩 3、执行 3，变成 decode 且 `remaining_prefill_tokens=0`；decode 剩 1、执行 1，变成 completed 且 `remaining_decode_tokens=0`。
这里把“输入处理完”和“输出生成完”分成两个阶段迁移，没有把阶段切换当成一次完整模型调用的“首 token 事件”；真实系统还要处理采样、停止条件和取消，但状态账本的骨架就是这两步。

**第 6 步：跨轮容量守恒。** `test_multiple_boundaries_preserve_capacity` 用一个 decode 请求（blocks 2）和一个 prefill 请求（blocks 3）、初始 `free_blocks=1`，连续跑 5 轮，每轮断言：

```python
report['free_kv_blocks'] + sum(row['kv_blocks'] for row in report['batch']) == 3
```

也就是“空闲块 + 本轮占用块”恒等于总块数 3。终止请求释放的块进入下一轮的 `free_blocks`，但不能被重复归还；最终活跃和等待都为空、`free_blocks=3`。这个不变量是排查“块泄漏/重复回收”的第一道防线。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 连续批处理 | 每轮解码后重组批次，完成的走、等待的补 |
| 迭代边界 | 一次解码迭代结束、可以重新组批的时刻 |
| slot（槽位） | 一个并发请求名额 |
| KV cache / KV block | 缓存历史键值状态的内存块，请求结束要归还 |
| Prefill | 处理输入提示，一次可推进一块 token |
| Decode | 逐 token 生成，每轮最多推进 1 个 |
| 队首阻塞 | 队首不满足条件时，后面的请求也不放行 |
| token budget | 本轮所有请求合计最多处理多少 token |
| overflow / 溢出 | 请求需求超过剩余预算，只能留到下一轮 |
| 滚动补位 | 一有空位就提交下一条，而不是等整批结束 |
| 槽位利用率 | 请求占用并发名额的时间 ÷ 可用槽位时间 |
| 饥饿 | 某请求长期得不到执行机会 |

## 小白常见疑问
**Q1：为什么队首被阻塞时不试试后面的请求？后面可能刚好放得下。**
因为要保证等待顺序可解释、避免长请求饿死。函数在第一个 `block_reason` 处 `break`，`waiting` 返回的是从队首开始的剩余队列；`test_batch_gates_and_block_reason` 特意断言 `['p','d']` 原样保留。真实系统可以用“按剩余预算缩小输入块”或“重排优先级”等策略，但那是另一类选择，本节没有实现。

**Q2：`free_slots = max_batch_size - len(active)`，为什么活跃请求不检查 slot？**
活跃请求本来就在占用槽位，它们必须继续执行；slot 限制的是“还能不能再加新成员”。如果活跃数已经超过 `max_batch_size`，代码会通过 `free_slots < 0` 直接报错，而不是悄悄丢掉某个活跃请求。

**Q3：为什么终止请求归还的块要加到 `free_kv_blocks`，而不是直接覆盖？**
因为本轮开始时可能还有别的空闲块。`available_blocks = free_kv_blocks + released` 表示“原有的空闲 + 本轮释放的”，两者相加才是真正可用的容量。`test_multiple_boundaries_preserve_capacity` 用“空闲 + 占用 = 总量”检查没有漏加或重复加。

**Q4：Decode 为什么每轮最多 1 个 token，而 Prefill 可以一块？**
自回归生成每次只能基于已有上下文产出下一个 token，所以 Decode 的步长天然是 1；Prefill 是对整段输入做一次前向，可以按 chunk 并行处理多个位置。`advance_iteration` 对 Decode 的 `amount > 1` 直接抛 `ValueError`，防止状态账本被写成“一轮生成多个”。

**Q5：真实实验里 `ttft_ms` 只有几十毫秒，为什么 `workload_ttft_ms` 的 P99 到了 1 秒多？**
两个指标起点不同。逐请求的 `ttft_ms` 从请求实际提交到首 token 返回；`workload_ttft_ms = submitted_ms + ttft_ms`，从实验开始计时，包含了客户端等待提交的时间。整批屏障下，后几批请求要等前一批全部完成才能提交，所以工作负载级 TTFT 被拉长；滚动补位缩短的正是这段提交等待。

## 动手验证：怎么知道自己做对了
CPU 测试依次运行 7 个函数，最后打印 `✅ Continuous Batching 机制测试通过`。关键断言：
- `test_phase_token_demand`：prefill 10/chunk 4 → 4；prefill 2/chunk 4 → 2；decode 剩 3 → 1。
- `test_terminal_release`：completed 请求移出 active、`released == 4`。
- `test_batch_gates_and_block_reason`：队首 p 报 `token_budget`，批次只有 a，`waiting` 保留 `['p','d']`，`free_token_budget=3`。
- `test_iteration_state_advance`：prefill 归零转 decode；decode 归零转 completed。
- `test_all_gates_and_input_preservation`：三种门槛分别报 `slot`、`kv_blocks`、`token_budget`；成功准入后两份预算都归零；输入列表不被修改。
- `test_invalid_work_and_duplicate_ids`：负剩余量、重复 id、负预算、负执行量、decode 执行 2 个、未知阶段都必须抛 `ValueError`。
- `test_multiple_boundaries_preserve_capacity`：5 轮后活跃/等待清空、`free_blocks=3`，且每轮都满足“空闲 + 占用 = 3”。

Step 5 是可选真实后端实验，默认 `RUN_MODE="dry_run"`，只登记计划；改为 `real_backend` 需要先启动 vLLM 0.11 服务并确认 `/v1/models` 可访问。实验固定 `Qwen/Qwen3-0.6B`、12 条请求（客户端内容各 128 token，强制输出长度按 `[16,64,32,96]` 循环三遍）、并发 4。5.4 给出的两轮结果：
- 正向（屏障 → 滚动）：槽位利用率 `0.584 → 0.823`，空闲槽位时间比 `0.316`，客户端提交等待 P99 比 `0.558`，吞吐比 `1.346`，工作负载 E2E P99 比 `0.735`，接受。
- 反向（滚动 → 屏障）：`0.576 → 0.757`、`0.408`、`0.472`、`1.404`、`0.699`，仍接受。
文字总结是：滚动补位把槽位利用率提高 18.1–23.9 个百分点，空闲槽位时间减少 59.2%–68.4%，客户端提交等待 P99 减少 44.2%–52.8%，请求吞吐提高 34.6%–40.4%。notebook 保存的一次反向运行里，baseline 的 `request_throughput_per_s=7.3733`、`output_token_throughput_per_s=383.4122`、`slot_utilization=0.576`、`idle_slot_ms=2757.788`；候选的 `request_throughput_per_s=10.35`、`duration_s=1.1594`。决策规则要求吞吐比 `>1.03`、E2E P99 比 `≤1.05`、提交等待 P99 比 `<0.95`、槽位利用率增益 `≥0.05`、空闲时间比 `<0.90`，且 12 条请求全部成功、输出长度完全一致，才返回 `accept_rolling_admission`。P99 只由每组 12 条请求算出，扩大请求量后要重新检查尾部。
