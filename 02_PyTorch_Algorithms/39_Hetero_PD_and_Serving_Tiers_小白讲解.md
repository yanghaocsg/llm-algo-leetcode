# 39. 异构 PD 与服务分层 — 小白讲解

> 对应 notebook：[`39_Hetero_PD_and_Serving_Tiers.ipynb`](./39_Hetero_PD_and_Serving_Tiers.ipynb)

## 一句话总结
本节把"Prefill 和 Decode 跑在不同算力、显存、带宽的资源上"收成一条可检查的决策链：用 `classify_resource_need` 判断请求更吃 Prefill 还是 Decode，用 `choose_handoff_mode` 比较 KV 传输与重算的耗时，用 `evaluate_pool_capability` 把"能否执行"和"能否接收 KV"分成两层门槛，最后用 `plan_route_with_fallback` 在主池、回退池和拒绝之间重规划。核心结论是：**KV 不兼容不等于资源池不能用，能执行但接不了 KV 时应改为重算，而不是直接拒绝。**

## 为什么需要它：痛点是什么
初学者容易把"PD 分离"理解成"把 Prefill 放一台机器、Decode 放另一台机器就完事了"。但真实系统里两个池的卡型、显存、互联带宽、后端版本可能都不同：Prefill 池算力强但显存小，Decode 池显存大但链路慢；KV 的 dtype 或内存布局还可能对不上。此时"要不要把 KV 搬过去""搬不动要不要重算""主池挂了找谁"才是真正的决策点。

生活类比：接力赛换人。交接棒（KV 状态）可以直接递过去（传输），也可以让下一棒自己从头跑一遍（重算），还可以干脆不换人（同池执行）。递棒要看跑道距离和交接区是否匹配；接棒的人跑不了这个项目（阶段不支持、模型版本不对、受伤了）才需要换人；只是"手型不合、接不了棒"时，让他自己起跑仍然可以完成比赛。

## 核心问题
- 长输入、长生成、两者都不长，分别该路由到哪种资源池？阈值边界怎么处理？
- KV 状态传输时间和重算时间怎么换算、怎么比较？为什么 `cache_transfer_mb * 8 / link_gbps` 的单位正好是毫秒？
- "资源池能执行请求"和"资源池能直接接收已有 KV"有什么区别？各自检查哪些字段？
- 主池 KV 不兼容、主池不健康、所有池都不行，三种情况分别怎么重规划？
- 显存压力和队列延迟同时出现时，先保护哪个目标？为什么？

## 解决思路：一步步看懂

**第 1 步：TODO 1——按"谁更长"分类请求压力。** `classify_resource_need(request, long_prompt_threshold, long_decode_threshold)` 读取 `prompt_tokens` 和 `decode_tokens`，参考答案是：

```python
if prompt_tokens > long_prompt_threshold and decode_tokens <= long_decode_threshold:
    return 'prefill'
if decode_tokens > long_decode_threshold and prompt_tokens <= long_prompt_threshold:
    return 'decode'
return 'shared'
```

两条判断都要求"另一侧不超过阈值"，所以三类互斥：长输入 + 长生成会落到 `shared`，短输入 + 短生成也是 `shared`。测试用阈值 `(2048, 256)`：`(3000, 64)` → `prefill`；`(128, 512)` → `decode`；`(800, 128)` → `shared`。注意用的是严格大于 `>`，所以 prompt 恰好 2048 不算长输入；如果改成 `>=`，边界样本的去向就变了。

为什么长输入优先给 Prefill 池？Prefill 是一次性处理整段 prompt，吃的是突发算力和输入带宽；Decode 是逐 token 生成，吃的是持续 KV 容量和稳定读带宽。两者都长时没有明显偏向，交给共享路径，避免把压力错误地全压到某一池。

**第 2 步：TODO 2——把 MB 和 Gbps 换成毫秒。** `choose_handoff_mode(link_gbps, cache_transfer_mb, recompute_ms)` 先校验 `link_gbps > 0`、`cache_transfer_mb >= 0`、`recompute_ms >= 0`，否则抛 `ValueError`。核心两行：

```python
transfer_ms = (cache_transfer_mb * 8) / link_gbps
mode = 'transfer' if transfer_ms <= recompute_ms else 'recompute'
```

单位推导：`cache_transfer_mb` 是 MB，乘 8 得 Mb（兆比特）；`link_gbps` 是 Gbps，即 $10^9$ bit/s。于是

$$
\frac{\text{Mb}}{\text{Gbps}}=\frac{10^6\ \text{bit}}{10^9\ \text{bit/s}}=10^{-3}\ \text{s}=1\ \text{ms}.
$$

所以数值上 `transfer_ms` 就等于"MB×8÷Gbps"的毫秒数，不需要再乘 1000。手算测试里的快链路：100 MB × 8 = 800 Mb，800 ÷ 200 = 4.0 ms，小于重算的 12 ms，选 `transfer`；慢链路 25 Gbps：800 ÷ 25 = 32 ms，大于 12 ms，选 `recompute`。相等时走 `transfer`（`<=`），这是"能传就传"的保守选择。

返回字典要求精确等于 `{'mode': 'transfer', 'transfer_ms': 4.0, 'recompute_ms': 12}`，测试用 `==` 整体比较，所以不能多塞字段，`transfer_ms` 也要 `round(..., 4)`。容易写错的是把 MB 当成 MB/s 直接除，或者忘了乘 8——前者会得到 0.5 ms，后者会得到 500 ms，都会改变选择。

**第 3 步：TODO 3——执行兼容与 KV 兼容是两层门槛。** `evaluate_pool_capability(request, pool)` 先检查执行侧四项：

- `healthy` 必须为真，否则 `unhealthy`；
- 请求的 `stage` 必须在池的 `supported_stages` 里，否则 `unsupported_stage`；
- `model_revision` 必须一致，否则 `model_revision_mismatch`；
- `available_memory_gb >= required_memory_gb`，否则 `insufficient_memory`。

四项都不报错时 `can_execute = True`。然后单独检查 KV 侧：`kv_dtype` 不一致记 `kv_dtype_mismatch`，`kv_layout` 不一致记 `kv_layout_mismatch`；`can_accept_kv = can_execute and not kv_reasons`。

这个设计的要点是：**`can_accept_kv` 为假不代表 `can_execute` 为假**。测试 `test_pool_capability_matrix` 把池的 `kv_layout` 改成 `contiguous`，断言 `can_execute is True`、`can_accept_kv is False`，并且 `reasons` 里同时有 `kv_layout_mismatch`；把 `healthy` 改成 `False` 时 `can_execute is False`。`reasons` 把执行原因和 KV 原因拼在一起返回，读的时候要结合两个布尔值判断，不能只看"有没有 reason"。

**第 4 步：TODO 4——三条路径：主池、重算、换池。** `plan_route_with_fallback(request, primary_pool, fallback_pools, preferred_handoff)` 先校验 `preferred_handoff` 只能是 `transfer` 或 `recompute`，然后把候选池排成 `[primary_pool, *fallback_pools]`，按顺序找第一个"能执行"的池：

```python
if not capability['can_execute']:
    rejected_reasons[pool_id] = list(capability['reasons'])
    continue
execution_mode = preferred_handoff
if preferred_handoff == 'transfer' and not capability['can_accept_kv']:
    execution_mode = 'recompute'
return {
    'selected_pool': pool_id,
    'execution_mode': execution_mode,
    'decision': 'primary' if index == 0 else 'reroute',
    ...
}
```

三条路径对应三种测试：

1. **主池能执行但 KV 不兼容**：`{**PRIMARY_POOL, 'kv_layout': 'contiguous'}`，`preferred_handoff='transfer'`，结果仍是 `decode-a`，但 `execution_mode='recompute'`、`decision='primary'`。这就是"不换池，改执行方式"。
2. **主池不能执行**：`healthy=False`，跳过主池并记录原因，回退池 `decode-b` 被选中，`decision='reroute'`。
3. **全部不能执行**：主池 `available_memory_gb=2 < 8`，没有回退池，返回 `selected_pool='none'`、`execution_mode='none'`、`decision='reject'`、`reasons=['no_executable_pool']`。

`rejected_pools` 会保留每个被跳过池的具体原因，方便事后排查。注意回退池不要求 KV 兼容：只要它能执行，就可以用重算兜底；但**不健康的池绝不会被选中**，这是硬不变量。

**第 5 步：服务动作的优先级。** `select_serving_action(memory_pressure, queue_delay_ms, slo_queue_budget_ms)` 是已经给出的策略：显存压力 > 0.9 时返回 `fallback_to_shared` / `protect='residency'`；否则队列等待超过预算时返回 `route_or_throttle` / `protect='latency'`；都正常时保持 `keep_heterogeneous_route` / `protect='throughput'`。测试用 `(0.92, 20, 80)` 验证显存优先，用 `(0.4, 120, 80)` 验证延迟次之。

为什么显存放第一位？显存打满会导致请求直接失败或触发抢占，属于"可运行性"问题；队列延迟超标是"体验"问题。先保证能跑，再谈快慢，这个顺序和 `decide_artifact_adoption` 里"先失败、再缺证、后比数值"是同一套工程思路。

**第 6 步：把决策写成可复查记录。** `build_route_record(request, route, action)` 输出六个字段：`resource_need`、`selected_pool`、`execution_mode`、`route_decision`、`service_action`、`protect`。测试要求它精确等于

```python
{'resource_need': 'decode', 'selected_pool': 'decode-a', 'execution_mode': 'transfer',
 'route_decision': 'primary', 'service_action': 'route_or_throttle', 'protect': 'latency'}
```

记录的作用是把"为什么选这个池、用什么方式、保护什么目标"固定下来，后续做 GPU 对照时才能区分"路由策略变了"还是"负载变了"。

**第 7 步：真实证据的边界。** notebook 最后明确说：合成 KV 搬运和矩阵重算只是成本下界，无法表达真实网络、worker 队列、状态序列化和 backend 同步，因此**不作为 GPU 证据**。真实对照要固定模型、请求清单和路由策略，比较同构 shared pool 与异构 PD 的 TTFT、TPOT、吞吐、queue wait、handoff/exposed time、失败和回退路径，这部分由 70 节 Serving 调度基准和 81 节分布式推理项目承接。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| PD 分离 | 把 Prefill（读 prompt）和 Decode（逐 token 生成）放到不同资源池 |
| 异构 PD | 两个池的算力、显存、带宽或后端能力不一样 |
| Prefill-heavy / Decode-heavy | 长输入短生成 / 短输入长生成，分别压不同的资源 |
| KV 状态 | Prefill 算出的 key/value 缓存，Decode 继续生成时要用 |
| KV handoff | 把 KV 从 Prefill 池交给 Decode 池的过程 |
| transfer / recompute / 同池 | 直接搬状态 / 目标池重新跑 Prefill / 留在原池继续 |
| `can_execute` | 池的健康、阶段、模型版本、显存都满足，能跑这个请求 |
| `can_accept_kv` | 在能执行的基础上，KV 的 dtype 和布局也兼容 |
| `reroute` / `reject` | 主池不行但回退池可用 / 没有任何可执行池 |
| SLO | 服务等级目标，如 P95/P99 延迟、超时率 |
| `memory_pressure` | 显存压力比例，超过 0.9 触发回退共享池 |

## 小白常见疑问
**Q1：为什么 `transfer_ms` 不用再乘 1000？MB 和 Gbps 不是差很多量级吗？**
因为 MB 到 Mb 乘 8 是 $10^6$ 量级，Gbps 是 $10^9$ 量级，两者相除正好是 $10^{-3}$ 秒，也就是毫秒。完整算是：100 MB = 800 Mb = $8\times10^8$ bit；200 Gbps = $2\times10^{11}$ bit/s；相除得 $4\times10^{-3}$ s = 4 ms。直接写 `cache_transfer_mb * 8 / link_gbps` 数值上就是毫秒，测试的 4.0 也是这么来的。

**Q2：KV 布局不兼容，为什么不让整个池"不可用"，而是改重算？**
因为布局只影响"能不能直接接收别人算好的 KV"，不影响这个池自己跑模型。`evaluate_pool_capability` 因此把 `can_execute` 和 `can_accept_kv` 分开；`plan_route_with_fallback` 只在 `can_accept_kv` 为假时把 `execution_mode` 从 `transfer` 改成 `recompute`，池子仍然是 `decode-a`。如果把它当成不可执行，就会白白丢掉一个能干活的池。

**Q3：长输入 + 长生成为什么归到 `shared`，而不是 `prefill` 或 `decode`？**
两类判断都带"另一侧不超过阈值"的条件，两个都超时两条 `if` 都不成立，落到最后的 `return 'shared'`。这是有意为之：两边都重，强行塞进任一专用池都会把该池的压力放大；共享路径让调度器按当前负载再决定。测试只覆盖了三类各一个代表点，但阈值边界的行为由代码结构决定。

**Q4：主池不健康、回退池健康，为什么结果是 `reroute` 而不是 `primary`？**
`decision` 取 `'primary' if index == 0 else 'reroute'`，`index` 是候选列表里的位置。主池被 `continue` 跳过后，回退池的下标是 1，所以是 `reroute`。这个字段是给日志和对照实验看的，表示"这次没有按首选路径走"，并不代表失败。

**Q5：显存压力 0.92、队列也超预算时，为什么只返回一个动作？**
`select_serving_action` 是互斥的三选一：先判显存，再判队列，最后才是保持。显存压力高时回退共享池可以同时缓解驻留压力，队列问题让共享池的调度去处理；返回两个动作会让调用方不知道先执行哪个。测试 `(0.92, 20, 80)` 和 `(0.4, 120, 80)` 分别钉住两个分支。

## 动手验证：怎么知道自己做对了
`test_heterogeneous_pd_policy()` 依次调用 6 个测试，全过才打印 `✅ 异构 PD 路由机制测试通过：资源分类、handoff、能力门槛、失败路由和服务动作均通过。`。它把 `NameError`、`AttributeError`、`TypeError`、`ValueError`、`AssertionError` 统一包成 `NotImplementedError('请先完成 TODO 代码或检查字段名！')`，所以看到这条消息时，先检查 TODO 是否为空、字段名是否拼错，不要只怀疑断言。

- `test_resource_need_classification`：`(3000, 64)` → `prefill`；`(128, 512)` → `decode`；`(800, 128)` → `shared`，阈值是 `(2048, 256)`。
- `test_handoff_mode_selection`：`link_gbps=200, cache_transfer_mb=100, recompute_ms=12` 必须精确返回 `{'mode': 'transfer', 'transfer_ms': 4.0, 'recompute_ms': 12}`；`link_gbps=25` 时 `mode == 'recompute'`。
- `test_pool_capability_matrix`：正常池 `can_execute` 和 `can_accept_kv` 都为真；`kv_layout='contiguous'` 时 `can_execute` 仍真、`can_accept_kv` 假、`reasons` 含 `kv_layout_mismatch`；`healthy=False` 时 `can_execute` 假。
- `test_fallback_replanning`：KV 不兼容 → 同池 `recompute`、`decision='primary'`；主池不健康 → 选 `decode-b`、`decision='reroute'`；显存 2 GB → `selected_pool='none'`、`decision='reject'`。
- `test_serving_action_policy`：`(0.92, 20, 80)` → `{'action': 'fallback_to_shared', 'protect': 'residency'}`；`(0.4, 120, 80)` → `{'action': 'route_or_throttle', 'protect': 'latency'}`。
- `test_route_record_contract`：`build_route_record` 的输出必须与前面那个六字段字典完全相同。

notebook 没有保存执行输出，也没有运行任何真实 GPU/backend 实验；以上结果都以测试断言和常量文本为准，不能把合成模型的 `transfer_ms` 当成真实网络的实测值。
