# INF-CONTINUOUS-BATCHING. Continuous Batching Beginner | 连续批处理 · 小白入门

> 配套 notebook：[INF-CONTINUOUS-BATCHING. 连续批处理](./INF-CONTINUOUS-BATCHING_Continuous_Batching.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

一批请求一起生成，短的先结束就会留下空位。连续批处理在每轮结束时清走已结束的请求、补进新请求。这本 notebook 教你实现一次这样的补位。

## 为什么需要它

4 个请求一起开始生成，最短的先结束了。这时有两种做法：

- 等整批结束。空出来的名额一直闲着，要等最慢的那个请求也结束，才能提交下一组。
- 完成一个就补一个。新请求能更早开始，但它要和还没结束的请求抢同一轮的计算和缓存。

第二种就是连续批处理。它带来一个新问题：空位有了，新请求就一定能进来吗？

不一定。看一个具体情形：这一轮还剩 2 个名额、4 个 KV 块。等待队列的队首还有 6 个输入词元要处理，一次最多处理 4 个，而本轮的词元预算只剩 3 个。名额和缓存都够，词元预算不够，它只能继续等。

这本 notebook 只做一件事：把一轮结束时的那次补位写清楚。先清走结束的请求，再按顺序检查等待的请求能不能进来，最后推进每个请求的进度。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 迭代边界（iteration boundary） | 一轮计算结束、下一轮还没开始的时刻。连续批处理只在这个时刻调整批次成员 |
| 槽位（slot） | 一个并发名额。批次最多能有 `max_batch_size` 个请求 |
| 活跃 / 等待 / 终止 | `active` 是本轮在批次里的请求，`waiting` 是排队等进来的，`completed`、`cancelled`、`timed_out`、`failed` 是终止状态 |
| KV 块（block） | KV 容量的计数单位。每个请求声明要 `kv_blocks` 块，终止时归还 |
| 词元预算（token budget） | 一轮最多处理多少词元，即 `max_num_batched_tokens`。活跃请求先占，剩下的才给新请求 |
| 队首阻塞（head-of-line blocking） | 队首请求进不来时，后面的请求即使放得下也不插队 |
| 首词元延迟（TTFT）/ 尾延迟 | 首词元延迟是一个请求等到第一个输出词元的时间。P95、P99 表示 95%、99% 的请求都落在以内的延迟位置 |

## 核心思路

输入是三样东西：活跃请求、等待队列、三项预算（名额、空闲 KV 块、本轮词元数）。每个请求是一条字典，记着它在哪个阶段、还剩多少输入和输出没处理、占了几个 KV 块。

每到一个迭代边界，做四步：

```text
1. 释放：状态是 completed、cancelled、timed_out 或 failed 的请求离开批次，归还它的 kv_blocks
2. 占用：仍活跃的请求留在批次里，保留名额和原有的 KV 预留，并占用本轮的词元预算
3. 准入：按顺序检查等待队列，每个请求依次过三道门槛
       slot → kv_blocks → token_budget
       三道都够：加入批次，扣减三项预算，继续看下一个
       有一道不够：记下第一个不够的原因，立刻停下，后面的请求都不看
4. 推进：每个请求扣掉本轮处理的词元数，预填充归零就转解码，解码归零就标记 completed
```

每个请求这一轮要处理几个词元（`scheduled_tokens`）：

```text
预填充：min(剩余输入词元数, prefill_chunk_size)
解码：  剩余输出词元数大于 0 时为 1，否则为 0
```

预填充一次只处理一块输入，块的大小是 `prefill_chunk_size`。解码一轮只推进一个词元。

下面用 notebook 测试里的数字走一遍前三步：

```python
def demand(r, chunk):                       # 本轮要处理的词元数
    return min(r['prefill'], chunk) if r['phase'] == 'prefill' else (1 if r['decode'] > 0 else 0)

active = [{'id': 'a', 'phase': 'decode', 'prefill': 0, 'decode': 2, 'kv_blocks': 2}]
waiting = [{'id': 'p', 'phase': 'prefill', 'prefill': 6, 'decode': 3, 'kv_blocks': 2},
           {'id': 'd', 'phase': 'decode', 'prefill': 0, 'decode': 3, 'kv_blocks': 1}]
max_batch, free_blocks, max_tokens, chunk = 3, 4, 4, 4

batch = active[:]                                       # 活跃请求留在批次里，占掉名额和词元预算
free_slots, free_tokens = max_batch - len(batch), max_tokens - demand(batch[0], chunk)
blocked = []
for r in waiting:                                       # 按顺序检查等待队列
    need = demand(r, chunk)
    reason = ('slot' if free_slots <= 0 else
              'kv_blocks' if r['kv_blocks'] > free_blocks else
              'token_budget' if need > free_tokens else None)
    if reason:
        blocked.append((r['id'], reason))
        break
    batch.append(r)
    free_slots, free_blocks, free_tokens = free_slots - 1, free_blocks - r['kv_blocks'], free_tokens - need
print([r['id'] for r in batch], blocked, free_slots, free_blocks, free_tokens)
```

```text
['a'] [('p', 'token_budget')] 2 4 3
```

批次里只有活跃请求 `a`。名额还剩 2 个，KV 块还剩 4 个，队首 `p` 只要 2 块。但它这一轮要处理 4 个词元，词元预算只剩 3 个，所以被 `token_budget` 挡住，循环立刻停下。后面的 `d` 只要 1 个词元、1 个 KV 块，明明放得下，也没有被放进来。

补位能减少空位闲置的时间，但不增加设备算力。新请求进来后，这一轮要处理的词元变多，原有请求后面的词元可能更晚到。所以比较时要同时看每秒输出量、首词元延迟和尾部请求等待，不能只看平均值。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 批次为什么要在迭代边界重组，请求排序、批次重组、KV 容量各管什么 | 读 |
| Step 2 | 名额、KV 和单轮词元预算三道门槛，以及严格队列顺序带来的队首阻塞 | 读，对照上面的四步 |
| Step 3 | 补位能减少什么等待，又要观察什么代价 | 读 |
| Step 4 · TODO 1 | 计算本轮词元需求 | 在 `iteration_token_demand` 里补 2 行：`prefill_demand` 和 `decode_demand` |
| Step 4 · TODO 2 | 识别迭代边界需要释放的请求 | 在 `release_terminal_requests` 里补 1 行 `is_terminal`：状态在 `TERMINAL_STATES` 里才为真 |
| Step 4 · TODO 3 | 按资源门槛组成本轮批次 | 在 `form_iteration_batch` 的循环里补 `block_reason`：按名额、KV、词元的顺序返回第一个不够的原因，都够时为 `None` |
| Step 4 · TODO 4 | 推进请求剩余工作量 | 在 `advance_iteration` 里补 2 行左右：算出当前阶段执行后的剩余词元数 `remaining_tokens` |
| 测试 | 检查预填充与解码的本轮需求、终止请求释放 KV 块、三类阻塞原因、队首保留，以及两种阶段迁移 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实服务上比较“等整批结束再提交”和“空槽立即补位”两种提交方式 | 需要 `INF-VLLM-011-CU128` 环境和一个已启动的 vLLM 服务；默认是 `RUN_MODE = "dry_run"`，不执行真实后端实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 上的 vLLM 服务，可选，默认是 `RUN_MODE = "dry_run"`。

## 容易踩的坑

- **跳过了队首**：队首被挡住就立刻停下，不绕过它去选后面的短请求，也不把它的输入块切小。测试里 `d` 明明放得下，返回的 `waiting` 仍然是 `['p', 'd']`。
- **门槛顺序写反**：按名额、KV、词元的顺序检查，报第一个不够的原因。同一个请求可能同时缺两样，顺序不同，报出的原因就不同。
- **预填充需求没取上限**：预填充需求是剩余输入词元数和 `prefill_chunk_size` 中较小的那个。测试里剩余 10、块上限 4，需求是 4。
- **解码一轮推进多个词元**：解码请求每轮最多推进 1 个词元，`advance_iteration` 遇到更大的 `scheduled_tokens` 会报错。
- **重复释放**：终止请求的 KV 块归还一次就够了。下一轮要传入更新后的活跃集合，不能再带上已释放的成员，否则容量被重复归还。

## 读完后自查

- [ ] 能说出迭代边界上的四步，以及三道门槛的检查顺序。
- [ ] 一个预填充请求还剩 6 个输入词元、块上限是 4，能说出它本轮处理几个词元。
- [ ] 给定剩下的名额、KV 块和词元预算，能判断队首请求会被哪道门槛挡住。
- [ ] 能解释为什么有空槽位时，等待队列里的请求仍然可能进不来。

## 下一步

- 打开 notebook：[INF-CONTINUOUS-BATCHING. 连续批处理](./INF-CONTINUOUS-BATCHING_Continuous_Batching.md)
- 建议先读：[INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界 · 小白入门](./INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds_Beginner.md)、[36. 解码调度 · 小白入门](./36_Decode_Scheduling_Beginner.md)、[37. KV Cache 生命周期与容量控制 · 小白入门](./37_KV_Cache_Lifecycle_and_Capacity_Control_Beginner.md)
- 做完之后：[38. Prefill/Decode 调度 · 小白入门](./38_Prefill_Decode_Scheduling_Beginner.md)
