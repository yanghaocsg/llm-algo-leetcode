# 38. Prefill/Decode 调度 — 小白讲解

> 对应 notebook：[`38_Prefill_Decode_Scheduling.ipynb`](./38_Prefill_Decode_Scheduling.ipynb)

## 一句话总结
本节先用 `build_chunked_prefill_plan` 把未命中的输入后缀切成有序块，再用 `plan_iteration_budget` 把一轮的 Token 预算分给"解码"和"分块预填充"，并比较解码优先、预填充优先、预留解码名额三种策略。最后用 vLLM 真实后端对比整段预填充与分块预填充，观察已有解码流的输出间隔和短探针的首 Token 延迟是否改善。

## 为什么需要它：痛点是什么
想象三个请求正在逐 Token 生成，这时来了一个 2048 Token 的长输入。即使 GPU 还有空闲的序列名额，把整段长输入一次性塞进去，也可能让原来三个请求的下一 Token 等很久——用户看到的就是输出"卡住"了。

初学者常有的困惑是："连续批处理不是已经能把新请求加进来了吗？为什么还会卡？"原因是连续批处理只决定"谁是一轮的成员"，不决定"这一轮每个成员能推进多少"。一个长预填充可能占掉整轮预算，解码成员虽然还在批次里，却排不到执行机会。

生活类比：一条单车道隧道，预填充是一辆超长卡车，解码是几辆小轿车。连续批处理决定哪些车能进隧道；分块预填充是把卡车切成几段，每放一段就重新决定下一段放卡车还是先放轿车。切块不会让卡车总长度变短，但给了调度器反复插队的机会。

## 核心问题
- 单轮的 Token 预算到底在数什么？为什么"位置数相同"不代表"耗时相同"？
- 分块预填充（Chunked Prefill）改变了什么，又不改变什么？
- 解码优先、预填充优先、预留解码名额三种策略，在同样预算下会得到怎样不同的成员？
- 分块有代价吗？什么时候才值得进一步考虑预填充/解码分离（PD disaggregation）？
- 真实后端实验中，为什么看"活跃解码流的输出间隔"和"短探针 TTFT"，而不是只看总吞吐？
- 配对实验的判据为什么要求 P95、最大值、吞吐和探针延迟同时满足？

## 解决思路：一步步看懂

**第 1 步：先分清预填充和解码的成本形态。** 预填充要处理尚未计算的输入 Token，一次可能送入很多位置，计算密集、持续时间长；解码利用历史 KV 缓存，每个请求每轮通常只推进一个 Token，但需要反复读写 KV。notebook 的 Step 1 强调：单轮词元预算统计的是"送入模型的处理位置数"，不是客户端已经收到的输出词元数。普通解码每个请求至多占一个位置，预填充可能占多个位置；两者相加不超过总预算，但位置数相同不代表耗时相同，历史长度和实际 kernel 仍会改变成本。

**第 2 步：用一个小例子理解三种策略。** 假设总预算 4，有两个解码请求 `d1`、`d2`，两条剩余输入分别是 `p1=5`、`p2=2`，块上限为 2：

- 解码优先：先安排 `d1`、`d2` 各一个位置，剩 2 个位置给输入，`p1` 处理 2 个 Token；
- 预填充优先：4 个位置全给输入，`p1` 处理 2 个、`p2` 处理 2 个（实际 `p2` 只有 2 个，正好用完），解码一个都不排；
- 预留 1 个解码名额再处理输入：先排 `d1`，剩 3 个位置，`p1` 处理 2 个、`p2` 处理 1 个，预算用完，第二次解码没有余量。

三者总工作量都接近或等于 4，但成员和等待完全不同。这也解释了为什么"预留"不是越多越好：预留太少保护不了解码，预留太多会压缩输入份额。

**第 3 步：TODO 1——有序分块。** 参考实现只有一行列表推导：

```python
return [suffix[start:start + chunk_size]
        for start in range(0, len(suffix), chunk_size)]
```

`suffix_tokens` 是"未命中前缀缓存、仍需计算"的后缀，不是完整 prompt。块必须保持原顺序，尾块不足 `chunk_size` 时保留实际长度，空输入返回空列表。`test_chunked_prefill_plan` 断言 `[9,10,11,12,13]` 按 2 切成 `[[9,10],[11,12],[13]]`，并检查 `chunk_size=0` 抛 `ValueError`。容易写错的是先对输入排序或用 `set` 去重——那会破坏位置语义，模型算出来的就不是原句了。

**第 4 步：TODO 2——阶段次序与首次解码额度。** 参考实现：

```python
phase_order = {
    'decode_first': ('decode', 'prefill'),
    'prefill_first': ('prefill', 'decode'),
    'balanced': ('decode', 'prefill', 'decode'),
}[policy]
first_decode_limit = decode_reserve_tokens if policy == 'balanced' else None
```

`phase_order` 描述一轮里阶段执行的先后。`balanced` 之所以有两次 `decode`，是为了"先预留少量解码 → 处理输入 → 用剩余额度补充解码"；第一次解码受 `decode_reserve_tokens` 限制，第二次把 `first_decode_limit` 置为 `None`，表示可以使用全部剩余额度。`schedule_decode` 每次都从"尚未安排的解码请求"里取，`scheduled_decode` 去重，所以同一请求不会在一轮里被执行两次；`schedule_prefill` 按字典顺序遍历后缀，每块取 `min(suffix_tokens, max_prefill_chunk_tokens, remaining)`，保证不超块上限也不超总预算。

**第 5 步：输入契约与预算守恒。** 函数开头检查：预算和块上限必须是正整数；`decode_reserve_tokens` 必须是非负整数；`policy` 只能是三种之一；后缀长度不能为负；请求 ID 必须是非空字符串；解码 ID 不能重复；同一请求不能同时出现在预填充和解码两个阶段。这些检查让 `test_iteration_budget_contract` 和 `test_empty_and_cross_phase_requests` 能分别用"预算为 0""未知策略""负后缀""重复 ID""跨阶段 ID"触发 `ValueError`。

返回字典里的守恒关系是：

$$
\text{used\_tokens}+\text{remaining\_tokens}=\text{max\_num\_batched\_tokens}
$$

其中 `used_tokens` 等于所有预填充块 Token 数之和加上本轮解码请求个数（每个解码请求占一个位置）。`test_all_policy_conservation` 在预算 `(1,2,4,8)`、预留 `(0,1,6)`、三种策略的笛卡尔积上逐项检查这条等式，并确认每个块满足 `0 < tokens <= min(2, suffix)`。

**第 6 步：什么时候才考虑 PD 分离。** 分块增加了重新调度的机会，也增加了提交和调度次数。Step 3 的判断顺序是：先看分块后解码等待是否仍过高；如果仍高，再检查块大小、并发和计算/带宽争用；持续达不到目标，才考虑给预填充和解码分别分配执行资源。PD 分离要把预填充侧算好的 KV 交给解码侧，新增搬运、同步、容量管理和失败回退成本；只有减少的争用超过这些代价才值得。单实例内的阶段配额和两个资源池之间的服务分离不是同一件事，后者由第 39 节继续验证。

**第 7 步：真实后端实验怎么设计。** Step 5 用 `INF-VLLM-011-CU128`、`Qwen/Qwen3-0.6B`，先建立 4 条活跃解码流（输入 128 Token、最多生成 256 Token），0.20 秒后注入一条 2048 Token 的长预填充（最多生成 16 Token），0.22 秒后再注入一条 128 Token 的短探针（最多生成 32 Token）。基线是整段预填充（`enable_chunked_prefill=False`），候选是分块预填充（`enable_chunked_prefill=True`、`max_num_batched_tokens=512`）。两次运行共享 `PAIR_LABEL`，只改 `SERVICE_VARIANT`，请求的模型 commit、Token 数和到达顺序必须一致，否则 5.3 会以 `PairMismatch` 拒绝比较。

因变量有四个：`short_probe_ttft_ms`（短探针首 Token 延迟）、`long_prefill_ttft_ms`（长输入首 Token 延迟）、`active_decode_stream_gap_p95_ms` 与 `active_decode_stream_gap_max_ms`（活跃解码流相邻 Token 的间隔 P95 和最大值）。间隔是从客户端流式读取观测到的，包含服务调度、网络传输和客户端读取时间，不等同于 GPU 单 Token 执行时间。

**第 8 步：结果怎么读。** 5.3 先算候选/基线的比值，再套四个判据：活跃流最大间隔比 `<= 0.75`、P95 间隔比 `<= 1.10`、请求吞吐比 `>= 0.90`、短探针 TTFT 比 `< 0.95`，且两个阶段失败请求数都为 0。只有这些都满足，才进入 `accept_chunked_prefill`；若探针没改善但解码尾部受保护，则记为 `accept_decode_protection_with_probe_tradeoff`；否则 `tune`。

notebook 5.4 记录的两轮配对结果：

| 运行顺序 | 间隔 P95 / 最大值（基线 → 候选） | 短探针 TTFT | 吞吐比 | TPOT P95 比 |
|---|---:|---:|---:|---:|
| v2：整段 → 分块 | 14.47 / 89.93 → 8.46 / 22.51 ms | 125.40 → 98.21 ms | 1.473 | 0.672 |
| v3：分块 → 整段 | 9.04 / 62.62 → 8.35 / 11.66 ms | 100.04 → 41.35 ms | 1.076 | 0.915 |

两轮方向一致：解码流的最大间隔明显缩短，短探针首 Token 更早，吞吐没有下降。notebook 特意说明"纯预填充负载曾得到候选无收益的负例"——所以结论不是"分块永远更快"，而是在"有活跃解码 + 长预填充 + 短探针"这种干扰负载下，分块给了短请求插队机会。吞吐和 TPOT 的改善幅度会随运行状态变化，只把稳定方向归因于分块，不把单轮加速比外推成服务容量结论。dump 里保存的 5.2 输出还显示本次分块阶段的服务环境是 torch 2.8.0+cu128、vLLM 0.11.0、模型 commit `c1899de…`，可作为复现口径。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 预填充（prefill） | 把尚未计算的输入 Token 送进模型，建立 KV 缓存 |
| 解码（decode） | 利用已有 KV，一次生成下一个 Token |
| 连续批处理 | 每轮重新决定哪些请求留在批次里、哪些补进来 |
| 分块预填充 | 把长输入切成有序小块，每块后重新调度 |
| 单轮 Token 预算 | 本轮允许送入模型的处理位置总数 |
| 阶段策略 | 一轮里先解码还是先预填充，以及是否预留解码名额 |
| `decode_reserve_tokens` | `balanced` 策略第一次解码最多安排几个请求 |
| KV 缓存 | 历史 Token 的 Key/Value，解码时复用，避免重算 |
| TTFT | 首 Token 延迟，从请求发出到收到第一个输出 Token |
| 流式间隔 | 相邻两个输出 Token 之间的时间差，反映卡顿 |
| PD 分离 | 预填充和解码用不同资源池，需要搬运 KV |
| P95 | 把样本排序后第 95 百分位，衡量尾部延迟 |

## 小白常见疑问
**Q1：分块预填充会让总计算量变少吗？**
不会。它只是把同一段输入切成多块、分多轮执行，总 Token 数不变，甚至因为调度次数增加而多一点开销。收益来自"每块之后可以重新选择先服务谁"，让解码请求不必等整段长输入算完。notebook Step 2 明确写了"分块不会改变前缀是否命中"。

**Q2：为什么 `balanced` 要安排两次解码？**
第一次解码受 `decode_reserve_tokens` 限制，只给少量解码请求"保底"；中间处理输入；第二次解码把限制解除，用剩余预算补充更多解码。这样既保证已有流不会完全挨饿，又不会让预留名额占死整轮预算。`test_iteration_budget_policies` 里预算 4、预留 1 的结果就是 `['d1']` 加 `p1:2`、`p2:1`，第二次解码没有余量。

**Q3：`used_tokens` 里为什么每个解码请求只算 1 个 Token？**
普通解码每个请求每轮至多推进一个位置。预填充后缀才可能一次占多个位置。`test_all_policy_conservation` 用 `sum(chunk tokens) + len(decode_request_ids) == used_tokens` 把这条口径固定下来；如果某个后端支持一次解码多个 Token，这个等式就需要相应修改。

**Q4：既然分块这么好，为什么还要 PD 分离？**
分块只在单实例内重新分配预算，预填充和解码仍共享同一套计算和带宽。如果分块后解码等待仍持续超标，说明争用不是靠调度顺序能解决的，才需要把两个阶段放到不同资源池。但 PD 分离要搬运 KV，反而可能引入新的延迟；notebook 要求先比较总成本，不能默认分离更好。

**Q5：实验中短探针 TTFT 下降，是不是说明所有请求都变快了？**
不是。短探针只有 128 Token 输入、32 Token 输出，受益于"长预填充被切块后产生的调度空隙"。长预填充自己的 TTFT、超大请求或不同到达顺序未必改善；notebook 也只把结论限定在固定到达序列和固定负载上。

## 动手验证：怎么知道自己做对了
`test_prefill_decode_scheduling()` 依次运行五个测试，全过打印 `PASS: 分块顺序、阶段策略、请求唯一性与多档预算守恒测试通过。`。
- `test_chunked_prefill_plan`：`[9,10,11,12,13]`、`chunk_size=2` 必须得到 `[[9,10],[11,12],[13]]`；`chunk_size=0` 抛 `ValueError`。
- `test_iteration_budget_policies`：预算 4、块上限 2、预留 1 时 `balanced` 的解码是 `['d1']`，块是 `p1:2`、`p2:1`；`decode_first` 预算 3 得到 `['d1','d2']` 和 `p1:1`；`prefill_first` 预算 3 得到空解码和 `p1:2`、`p2:1`；三者的 `used + remaining == budget`。
- `test_iteration_budget_contract`：预算 0、未知策略、负后缀都必须抛 `ValueError`。
- `test_empty_and_cross_phase_requests`：空负载 `used_tokens=0`、`remaining_tokens=4`；预留 0 时解码为空；同一 ID 跨阶段或重复解码 ID 必须报错。
- `test_all_policy_conservation`：预算 `(1,2,4,8)` × 预留 `(0,1,6)` × 三种策略，逐一验证块 Token 加解码数等于 `used_tokens`、预算守恒、解码不重复、每块不超过 `min(2, suffix)`；并断言输入字典没有被函数改写。
notebook 保存了 5.2 和 5.3 的部分输出；5.4 的两行配对数字来自 notebook 已记录结果。自己重跑时，以 `chunked_prefill_<PAIR_LABEL>_pair.json` 里的 `comparison`、`decision_checks` 和 `decision` 为准，不要只看单个 TTFT 数字。
