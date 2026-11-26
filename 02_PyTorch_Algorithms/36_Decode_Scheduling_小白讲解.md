# 36. 解码调度 — 小白讲解

> 对应 notebook：[`36_Decode_Scheduling.ipynb`](./36_Decode_Scheduling.ipynb)

## 一句话总结
多个请求同时生成时，调度器每步只能选一个推进，还要保证它的 KV 容量装得下。本节用 `RequestState` 和 `DecodeSchedulerSim` 搭一个纯 CPU 的离散事件模拟器，补全请求入队、排序键、容量准入/抢占、单步推进四个 TODO，比较固定优先级、等待老化（aging）和截止期限（deadline）三种规则，并用一组测试检查排序、公平性、容量守恒与恢复进度。

## 为什么需要它：痛点是什么
初学者容易把“调度”理解成“谁优先级高谁先跑”。但请求还有两个维度：它在 prefill 还是 decode 阶段，以及有没有拿到 KV 容量。高优先级请求如果需要的块数超过空闲容量，照样不能执行；低优先级请求可能因为缓存命中或阶段更靠前而先被选中。若只按优先级排序，就会写出“排序第一位却永远准入失败”的调度器。

生活类比：餐厅门口排队。优先级是 VIP 等级，阶段是“还在等位”还是“已经入座点菜”，KV 容量是桌位。VIP 来了但桌子不够，要么继续等，要么请一桌低优先级客人暂时让座（抢占）；让座后客人回来，不能把他已经点过的菜（生成进度）清空。等待老化则像“排队越久，临时提升一点优先级”，但再老也不能插到“已经入座”的客人前面去抢厨房。

## 核心问题
- 处理阶段（prefill/decode）和准入状态（waiting/running/preempted/finished）为什么是两个不同维度？
- 固定、aging、deadline 三种策略的排序键分别长什么样？为什么优先级要取负？
- 等待步数 `wait_steps` 什么时候只是观测值，什么时候会变成有效优先级？
- 容量不足时怎么选受害者？为什么只抢一个，为什么不可抢占请求受保护？
- 被抢占再恢复时，哪些进度必须保留，哪些资源要先归还再申请？
- 模拟器里的“一步”与真实服务的毫秒延迟是什么关系？

## 解决思路：一步步看懂

**第 1 步：TODO 1——登记请求状态。** `RequestState` 保存调度需要的最小状态：输入/生成长度、阶段、优先级、缓存命中、等待步数、容量需求、是否准入、截止步与可抢占标记，并提供 `total_len` 与 `done` 两个派生属性。参考答案：

```python
request = RequestState(
    request_id=request_id, prompt_len=prompt_len, max_new_tokens=max_new_tokens,
    priority=priority, cache_hit=cache_hit, required_blocks=required_blocks,
    deadline_step=deadline_step, preemptible=preemptible,
)
self.queue.append(request)
```

新请求默认 `phase='prefill'`、`generated_len=0`、`admitted=False`、`status='waiting'`。骨架先检查重复 `request_id`、非正的 `prompt_len/max_new_tokens/required_blocks`，以及 `deadline_step <= clock` 的非法期限；这些检查保证不会把负容量或过期期限塞进账本。

**第 2 步：TODO 2——构造排序键。** `_schedule_key` 返回一个六元组，比较顺序就是“从左到右”：

```python
deadline_slack = 1_000_000_000
if self.policy == 'deadline' and req.deadline_step is not None:
    deadline_slack = req.deadline_step - self.clock - self._remaining_steps(req)
phase_rank = 0 if req.phase == 'prefill' else 1
cache_rank = 0 if req.cache_hit else 1
return (deadline_slack, phase_rank, cache_rank,
        -self._effective_priority(req), req.total_len, req.request_id)
```

逐项解释：`deadline_slack` 是“截止步减去当前时钟和预计剩余步数”，越小越紧急；非 deadline 策略统一用 `1_000_000_000` 哨兵值，保证这一项不干扰排序。`phase_rank` 让 prefill 排在 decode 前面，`cache_rank` 让缓存命中优先，`-effective_priority` 让业务等级大的先执行（元组升序比较），`total_len` 和 `request_id` 是稳定平局字段。`_remaining_steps` 是 `(1 if phase=='prefill' else 0) + max(max_new_tokens - generated_len, 0)`。

`test_schedule_key_decision_order` 分三组验证：阶段比缓存和优先级更靠前；同阶段下缓存命中优先；同阶段同缓存下 `priority=3` 排在 `priority=1` 前，且 `step()` 返回 `request_id=2`。容易写错的是直接放优先级（大优先级反而排后面）或把 deadline 放在阶段之后。

**第 3 步：aging 与“等待只是观测”。** 固定策略下 `_effective_priority` 就是 `req.priority`；aging 策略下变成

$$
\text{effective\_priority}=\text{priority}+\left\lfloor\frac{\text{wait\_steps}}{\text{aging\_interval}}\right\rfloor.
$$

默认 `aging_interval=4`，即每等 4 步加 1 点有效优先级。`test_aging_changes_effective_priority` 用 `policy='aging', aging_interval=2`，让 `priority=1` 的请求等 6 步，有效优先级变成 $1+6//2=4$，从而压过 `priority=3` 的请求。但 aging 只改变“同一阶段、同一命中类别内”的次序：`test_aging_respects_phase_precedence` 让 decode 请求等 1000 步，prefill 请求仍然先被选中，因为 `phase_rank` 更靠前。`test_wait_steps_is_observation_not_priority` 则确认固定策略下把 `wait_steps` 改成 50 也不会改变排序键——等待步数是公平性观测，不是隐式优先级。

**第 4 步：TODO 3——容量准入与抢占。** 逻辑分三支：

```python
if challenger.admitted:                     # 已准入，直接继续
    challenger.status = 'running'; return None
if challenger.required_blocks <= self.free_kv_blocks:   # 容量够，预扣
    self.free_kv_blocks -= challenger.required_blocks
    challenger.admitted = True; challenger.status = 'running'; return None
victims = [req for req in self.queue
           if req.admitted and not req.done and req.preemptible
           and self._effective_priority(req) < self._effective_priority(challenger)]
victims.sort(key=lambda req: (self._effective_priority(req), -req.required_blocks, req.request_id))
victim = next((req for req in victims
               if self.free_kv_blocks + req.required_blocks >= challenger.required_blocks), None)
if victim is None:
    return None
```

已经准入的直接继续；容量够就预扣；不够时只考虑“已准入、未完成、可抢占、有效优先级更低”的请求，按“优先级最低、占用块数最多、编号最小”排序，挑第一个“释放后足够”的受害者。抢占时先把受害者的块还回空闲池，再扣挑战者的块，同时 `admitted=False`、`status='preempted'`、`preemptions += 1`；**不重置** `phase` 和 `generated_len`。这里只抢一个受害者，notebook 明确说“不合并多个受害者的容量”，所以释放的块仍不够时挑战者继续等。

测试 `test_preemption_and_resume_preserve_progress`：空闲 2 块，请求 1 用 2 块、优先级 1、可抢占；先让它完成 prefill 准入；再入队优先级 9、需要 2 块的请求 2。第二步必须抢占请求 1，事件里 `preempted_request_id == 1`，受害者 `phase` 仍是 `decode`、`generated_len` 仍是 0、`preemptions == 1`；继续跑到结束后请求 1 能完成。`test_non_preemptible_and_deadline_policy` 把请求 1 设为 `preemptible=False`，同样容量的高优先级请求 2 抢不动它，最终仍由请求 1 执行，`preempted_request_id is None`。

**第 5 步：TODO 4——排序、准入、推进一步。** 参考答案先按排序键逐个尝试准入，第一个成功准入的成为 `chosen`：

```python
chosen = None
for candidate in sorted(candidates, key=self._schedule_key):
    preempted_request_id = self._ensure_capacity(candidate)
    if candidate.admitted:
        chosen = candidate
        break
if chosen is None:
    return None
```

选中的请求前进一步：prefill 直接切到 decode（`prefill_to_decode`，不增加 `generated_len`）；decode 则 `generated_len += 1`，达到 `max_new_tokens` 时动作记为 `finish`，归还 `required_blocks`。未选中的请求 `wait_steps += 1`。事件记录动作、生成长度、等待步数、空闲块、被抢占 id 和 `deadline_missed`（时钟 + 1 + 剩余步数 > 截止步）。`run(max_steps=100)` 只是反复调用 `step()` 直到没有候选或到达防护上限。

**第 6 步：deadline 测试怎么读。** 在 `test_non_preemptible_and_deadline_policy` 里，请求 1 截止 20、剩 5 步，`slack=15`；请求 2 截止 1、剩 2 步，`slack=-1`，所以请求 2 先执行，且 `deadline_missed=True`。deadline 策略优先看“还来不来得及”，不是业务等级。另一个容量测试 `test_capacity_admission_and_release` 则展示：3 块的请求排序更靠前但被跳过，2 块的请求准入；完成后空闲从 0 回到 2。

**第 7 步：真实服务的边界。** 本节把时间抽象成“调度步”：输入处理占一步，之后每步生成一个 Token，不能直接换算成毫秒。notebook 明确说微基准不能反映队列等待、饥饿、抢占或尾延迟；真实结论要交给 70 节 Serving 调度基准，固定模型、到达轨迹、输入/输出长度和 KV 预算，只替换选择策略，并记录 queue wait、TTFT、TPOT、P95/P99、饥饿与拒绝请求。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| prefill | 先读整段输入、建立 KV 的阶段 |
| decode | 已经进入逐 Token 生成的阶段 |
| 准入（admission） | 为请求预留 KV 块，成功后才能执行 |
| `required_blocks` | 一个请求需要占用的 KV 块数 |
| 排序键 | 决定这一轮先试谁的六元组 |
| 固定优先级 | 只看业务等级，不随等待变化 |
| aging | 等待越久，有效优先级按间隔加一点 |
| deadline slack | 截止步减去当前时钟和预计剩余步数，越小越急 |
| 抢占（preemption） | 暂停一个低优先级请求，把容量让给更紧急的 |
| 受害者（victim） | 被选中让出容量的那个已准入请求 |

## 小白常见疑问
**Q1：为什么 prefill 一定排在 decode 前面？高优先级 decode 也不能先跑吗？**
因为排序键里 `phase_rank` 在 `cache_rank` 和优先级之前。`test_schedule_key_decision_order` 构造了“prefill 低优先级 vs decode 高优先级”，断言 prefill 的键更小。这是本节选定的策略，不代表所有真实后端都这么做。

**Q2：`wait_steps` 为什么在固定策略里完全不影响排序？**
因为 `_effective_priority` 只在 `policy == 'aging'` 时把 `wait_steps // aging_interval` 加进去。`test_wait_steps_is_observation_not_priority` 把等待改成 50 后排序键不变，保护的就是这个约定：等待是观测值，不能让固定策略偷偷变成 aging。

**Q3：抢占时为什么只抢一个受害者，不把几个请求的容量拼起来？**
notebook 明确说“只尝试暂停一个可抢占、有效优先级更低且能让出足够块数的请求，不合并多个受害者的容量”。这是刻意简化；真实系统可能有更复杂的重算/换出策略。

**Q4：被抢占的请求恢复时，为什么不重新从 prefill 开始？**
抢占只归还容量，不抹掉输入处理和生成进度。参考实现保留 `phase` 与 `generated_len`，只把 `admitted` 置 False、`status` 置 `preempted`；真实后端还要恢复被丢弃的 KV 或重算，这部分耗时不在容量账本里。

**Q5：`deadline_slack` 为什么要用 `1_000_000_000` 当哨兵？**
非 deadline 策略不该因为某个请求“恰好有 deadline”就改变排序；给所有请求同一个大值，等于把这一项从比较中关掉。只有 `policy == 'deadline'` 且 `deadline_step is not None` 时才计算真实余量。另外，`run(max_steps=2)` 只返回 2 个事件，是因为每个成功 `step()` 消耗一个调度步，`max_steps` 只是防死循环的计数。

## 动手验证：怎么知道自己做对了
`test_decode_scheduler()` 依次调用十个测试，全部通过后打印 `DecodeSchedulerSim 机制测试通过：排序、公平性、准入、抢占、恢复和 deadline 均通过。`。
- `test_enqueue_contract`：新请求 `phase='prefill'`、`admitted=False`。
- `test_schedule_key_decision_order`：阶段 > 缓存 > 优先级；`priority=3` 被选中。
- `test_wait_steps_is_observation_not_priority`：固定策略下改等待不改变排序键。
- `test_aging_changes_effective_priority`：`1 + 6//2 == 4`，压过 `priority=3`。
- `test_aging_respects_phase_precedence`：decode 等 1000 步也越不过 prefill。
- `test_capacity_admission_and_release`：3 块的被跳过，2 块的准入；完成后空闲回到 2。
- `test_preemption_and_resume_preserve_progress`：被抢占 id 为 1；阶段与生成进度不变；`preemptions=1`。
- `test_non_preemptible_and_deadline_policy`：不可抢占受保护；deadline 请求先执行且 `deadline_missed=True`。
- `test_state_progression_and_completion`：共 7 个事件，全部 `done`，存在 `wait_steps > 0`。
- `test_input_guards_and_step_limit`：零长度/零块数、负数 `max_steps` 抛 `ValueError`。

本节没有 GPU 测量单元，也没有保存运行输出；以测试断言和事件字段为准。真实服务的 TTFT/TPOT 指标需要到 70 节的 Serving 调度基准里看。
