# 37. KV Cache 生命周期与容量控制 — 小白讲解

> 对应 notebook：[`37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb`](./37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb)

## 一句话总结
`KVCacheLifecycleSim` 用"离散 Block + 引用计数"模拟 KV Cache 的一生：预填充时分配、解码时向尾块追加、前缀命中时共享、请求结束时释放引用、容量不够时驱逐未引用缓存、必要时抢占或拒绝。四个 TODO 分别补上 token 到 Block 的向上取整、共享引用、跨尾块追加和释放回收；后半部分再用真实 vLLM 服务观察前缀复用与容量压力。

## 为什么需要它：痛点是什么
初学者对 KV Cache 的印象常停留在"会随序列变长而变大"。但真正写推理系统时，问题要细得多：一个请求结束后，它占的 Block 能立刻给别人吗？两个请求共享同一段前缀，为什么只占一份显存？解码一个 token，为什么有时不增加 Block、有时又突然多占一块？容量满了，应该踢掉谁、能不能踢正在用的？

生活类比：图书馆的共享书架。一本书（Block）可以被多张借阅卡（请求）同时借阅，借阅卡数量是引用计数；只要还有人借着，管理员就不能把书下架。一本书没借满一页（尾块没填满）时，继续写不需要新书；写满一页才要再拿一本。没人借的书可以留在架上等下次命中（reusable），也可以直接清掉腾地方（reclaim）。

## 核心问题
- 给定 token 数和 Block 大小，最少需要几个物理 Block？为什么必须向上取整？
- 两个请求命中同一前缀时，哪些状态会变、哪些绝对不能变？
- 追加一个 decode token，什么情况下新增 Block、什么情况下不新增？
- 引用归零后，什么时候保留成可复用缓存、什么时候立即回收？
- 容量不足时，驱逐顺序是什么？为什么活动请求不能被驱逐？真实服务里怎么观察这些行为？

## 解决思路：一步步看懂
### 第 1 步：先看清状态对象
`CacheRecord` 是一个 dataclass，字段就六个：

```python
prefix: str          # 前缀身份（调用方给的字符串）
used_tokens: int     # 这段缓存覆盖多少有效 token
block_count: int     # 占了多少物理 Block
ref_count: int = 1   # 还有多少请求在用
hits: int = 1        # 被命中过多少次
last_used: int = 0   # 最近访问的逻辑时间
```

`KVCacheLifecycleSim(total_blocks=16, block_size_tokens=4)` 维护 `records: Dict[str, CacheRecord]` 和 `log`。两个派生属性是容量账本的核心：

```python
allocated_blocks = sum(record.block_count for record in self.records.values())
free_blocks = self.total_blocks - self.allocated_blocks
```

注意 `allocated_blocks` 把 `ref_count == 0` 的 reusable 记录也算进去——它们还占着物理 Block，只是没人引用。所以"缓存条目有几个"和"占了多少 Block"是两回事。

### 第 2 步：TODO 1——token 到 Block 的向上取整
```python
block_count = (token_count + self.block_size_tokens - 1) // self.block_size_tokens
```
这是 `ceil(token_count / block_size_tokens)` 的整数写法。`block_size_tokens=4` 时：1 个 token 也要 1 块，`(1+3)//4=1`；正好 4 个 token 是 1 块；5 个 token 是 2 块。**为什么不能`//`？** 因为尾块哪怕只放 1 个 token，也真实占一个 Block 的显存，向下取整会低估容量、导致后面分配越界。`token_count <= 0` 时抛 `ValueError`。

### 第 3 步：TODO 2——共享已有前缀
`acquire` 先做输入校验：`prefix` 必须是非空字符串、`prompt_tokens > 0`；如果这个 prefix 已存在但 `used_tokens` 和这次传的 `prompt_tokens` 不一致，抛 `ValueError("同一前缀身份的有效长度必须一致")`——同一个身份不能指向两段不同长度的缓存。

命中时只改三样：
```python
record.ref_count += 1
record.hits += 1
record.last_used = self.time
```
**绝对不改** `used_tokens`、`block_count`。共享的物理含义是"多一个请求指向同一份 K/V"，不是"再复制一份"。所以测试里 `first is second`（同一个对象）、`ref_count == 2`、`hits == 2`，而 `allocated_blocks` 仍是 2。

没命中时走 `_blocks_for_tokens` 算块数、`_ensure_capacity` 确保有空位，再新建 `CacheRecord` 并记日志 `allocate:{prefix}:{block_count}`。

### 第 4 步：TODO 3——只在跨尾块时追加
`append` 有三道前置检查：记录必须存在且 `ref_count > 0`（不能给已释放的缓存追加）；`new_tokens > 0`；`ref_count == 1`（共享前缀禁止原地追加）。最后一条很关键：如果两个请求共享前缀，任何一个往后写都会污染另一个请求看到的历史，所以必须先建立私有后缀，本题直接用 `ValueError` 拒绝。

核心三行：
```python
target_tokens = record.used_tokens + new_tokens
target_blocks = self._blocks_for_tokens(target_tokens)
added_blocks = target_blocks - record.block_count
self._ensure_capacity(added_blocks)
record.used_tokens = target_tokens
record.block_count = target_blocks
```
先算"追加后的目标状态"，再和当前状态求差，只申请差值。尾块没满时 `added_blocks = 0`，`_ensure_capacity(0)` 直接通过；跨边界时只加必要的 1 块（如果一次加很多 token 也可能加多块）。顺序也很重要：先 `_ensure_capacity` 再改记录，容量不足抛 `RuntimeError` 时原状态保持不变。

测试 `test_append_allocates_only_at_tail_boundary` 的账：`acquire('decode', 3)` 占 1 块；`append(1)` 到 4 个 token，仍是 1 块，返回 0；再 `append(1)` 到 5 个 token，需要 2 块，返回 1；最终 `(used_tokens, block_count) == (5, 2)`。

### 第 5 步：TODO 4——释放引用与回收
```python
record.ref_count -= 1
if record.ref_count == 0 and not keep_reusable:
    self.records.pop(prefix)
    self.log.append(f'reclaim:{prefix}:{record.block_count}')
else:
    record.last_used = self.time
    self.log.append(f'release:{prefix}:{state}:{record.ref_count}')
```
`keep_reusable=True`（默认）时，引用归零的记录留在 `records` 里，`snapshot()` 把它标成 `reusable`，后续相同前缀还能命中，省一次 Prefill；`keep_reusable=False` 时立即 `pop`，Block 马上回到空闲池。`ref_count` 不能减到负数：入口已经检查 `ref_count <= 0` 就抛 `ValueError`。

### 第 6 步：驱逐顺序与容量保护
`_eviction_candidates` 只挑 `ref_count == 0` 的记录，排序键是 `(hits, last_used, -block_count)`：
1. `hits` 小的先踢——复用价值低；
2. 同样 hits 时 `last_used` 小的先踢——更久没访问；
3. 再同样时 `-block_count` 让大块先踢——一次能腾出更多空间。

`_ensure_capacity(needed_blocks)` 按这个顺序逐个 `pop`，直到 `free_blocks >= needed_blocks`；如果踢完所有可踢的还不够，抛 `RuntimeError('insufficient blocks: queue, preempt, or reject the request')`。**活动记录永远不在候选里**，所以不会为了接纳新请求而破坏正在生成的请求。真实系统还要等"在途读写"结束才能回收，本题只做引用计数层面的模拟。

把 `test_release_reclaim_and_active_protection` 走一遍（总 5 块、块大小 4）：
- `acquire('cold', 8)` → 2 块，已占 2，空闲 3；
- `release('cold')` → 引用 0，但 `keep_reusable` 默认 True，记录还在，仍占 2；
- `acquire('active', 8)` → 再占 2，已占 4，空闲 1；
- `acquire('new', 8)` 需要 2 块，空闲 1 不够，驱逐 `cold`（引用 0）腾出 2 块，再分配给 `new`；此时已占 4、空闲 1，`active` 的引用仍是 1；
- `release('new', keep_reusable=False)` → 直接回收，已占 2。

`test_capacity_failure_preserves_active_records` 则验证最坏情况：总 3 块，`a` 占 2、`b` 占 1，再来 `c` 需要 1 块，但没有任何引用 0 的记录可踢，于是抛 `RuntimeError`，并且 `records` 仍是 `{a, b}`、`allocated_blocks == 3`——失败不能把账本改坏。

### 第 7 步：真实服务实验怎么读
CPU 状态机只看逻辑账本；Step 5 用 `INF-VLLM-011-CU128` 环境启动 vLLM（模型 `Qwen/Qwen3-0.6B`），做两组实验：

**共享前缀配对**：12 条请求、并发 4、输出上限 32 token、只跑 1 轮（避免重复请求直接命中缓存）。baseline 每条前面加不同的 `独立前缀 {index*7919}`，candidate 共用同一段 `shared_context`，后缀模板相同。指标是 TTFT（首 token 延迟）、TPOT（每输出 token 时间）、吞吐、失败数和后端 Cache 计数器。保存的输出是 baseline（unique_prefix）的一次结果：12/12 成功，请求吞吐 14.6728 req/s，输出吞吐 469.53 tok/s，TTFT 均值 96.317 ms、P95 188.18 ms、P99 269.56 ms，TPOT 均值 5.643 ms，E2E 均值 271.264 ms，平均 prompt 603.75 token。

**容量压力矩阵**：固定服务配置和输出长度，只扫描上下文长度 `(512, 1024, 1536)` 与并发 `(4, 8, 16, 32)`，共 12 档；每档用请求级和 Block 级 nonce 尽量打散共享前缀，并检查 `actual_tokens + 32 <= 2048`。保存的首档（512 token、并发 4）TTFT 均值 3241.254 ms，明显是冷启动；notebook 5.4 的结论是 12/12 档成功、抢占计数为 0，因此**只能说明整体压力趋势，不能据此确定容量上限**——因为整轮根本没有触发驱逐、抢占或拒绝。最终 `decision` 为 `tune`，不是 `accept`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KV Cache | 缓存历史 token 的 Key/Value，避免解码时重算 |
| Block | 固定大小的物理缓存单位，本题每块 4 个 token 位置 |
| 尾块（tail block） | 最后一个没填满的 Block |
| 有效 token 数 `used_tokens` | 这段缓存实际覆盖多少 token |
| 引用计数 `ref_count` | 还有几个请求在用这份缓存 |
| 共享前缀 | 多个请求指向同一份物理 Block，只加引用不复制 |
| reusable | 引用为 0 但仍留在池里、可被后续命中的缓存 |
| 驱逐（evict） | 踢掉引用为 0 的缓存来腾空间 |
| 回收（reclaim） | 引用归零后立即删除记录、归还 Block |
| 抢占（preempt） | 暂停活跃请求，换出或丢弃其状态 |
| `free_blocks` | 总 Block 减已分配 Block |
| TTFT / TPOT | 首 token 延迟 / 每输出 token 时间 |

## 小白常见疑问
**Q1：六词元前缀、每块 4 个位置，为什么占 2 块而不是 1.5 块？两个请求共享后为什么还是 2 块？**
`(6 + 4 - 1) // 4 = 2`，尾块只有 2 个有效 token，但物理上仍是一整块。共享只让 `ref_count` 从 1 变 2、`hits` 从 1 变 2，`block_count` 保持 2，所以总占用不变。这正是 `test_shared_reference_does_not_duplicate_blocks` 断言 `allocated_blocks == blocks_before == 2` 的原因。

**Q2：`append` 为什么有时返回 0、有时返回 1？**
它返回的是"本次新增的物理 Block 数"，不是新增 token 数。3 个 token 占 1 块，追加 1 个到 4 个仍在同一块内，返回 0；再从 4 个追加到 5 个，必须开第二块，返回 1。这个返回值让调用方知道要不要向调度器申请新容量。

**Q3：`release` 的 `keep_reusable=True` 和 `False` 到底差在哪？**
差在"引用归零后记录是否立刻消失"。`True` 保留记录，`snapshot()` 状态变 `reusable`，下次相同前缀 `acquire` 能命中、省 Prefill，但 Block 仍被占着；`False` 直接 `pop`，Block 立刻回到空闲池，代价是下次要重算。notebook 的 Step 3 表把前者对应"回收/驱逐"的取舍，后者对应"立即释放"。

**Q4：为什么驱逐候选一定要 `ref_count == 0`？正在用的请求被踢掉不行吗？**
不行。活动请求的 K/V 是它继续生成的必要状态，踢掉会让它无法解码；真实系统若真要腾地方，只能走"抢占"：暂停请求、把状态换出到 CPU/磁盘，或者记录进度后丢弃并重算。本题的 `_ensure_capacity` 只做安全回收，回收不够就抛 `RuntimeError`，把"排队、抢占、拒绝"留给调度层。

**Q5：容量压力实验里 12 档都成功、抢占为 0，是不是说明容量很充足？**
不能这么读。抢占为 0 也可能是因为这组上下文和并发根本没把缓存压满；notebook 5.4 明确写"当前矩阵没有触发驱逐、抢占或拒绝，不能据此确定容量上限"。要测上限，得继续加大上下文或并发，直到出现失败/抢占，并同时看 P99 和吞吐拐点。

**Q6：为什么第一档 512 token、并发 4 的 TTFT 高达 3.2 秒？**
notebook 把它标为"明显冷启动异常"：服务刚启动、模型和 CUDA 图还没热，首轮请求承担了初始化开销。所以这一档不能拿来和后面的档位比延迟，只能作为"矩阵跑通"的证据；正式比较要预热或丢弃首档。

## 动手验证：怎么知道自己做对了
`test_kv_cache_lifecycle()` 汇总六个测试，全过打印 `✅ KV Cache 生命周期测试通过：分配、共享、追加、释放和容量保护均符合预期。`；TODO 未完成时 `NotImplementedError` 会直接抛出。
- `test_block_rounding_and_input_contract`：块大小 4 时，`_blocks_for_tokens(1) == 1`、`(4) == 1`、`(5) == 2`；`0` 抛 `ValueError`；`acquire('', 4)` 抛 `ValueError`。
- `test_shared_reference_does_not_duplicate_blocks`：两次 `acquire('shared', 6)` 返回同一个对象，`ref_count == 2`、`hits == 2`，`allocated_blocks == 2`。
- `test_append_allocates_only_at_tail_boundary`：`append` 两次分别返回 0 和 1，最终 `(used_tokens, block_count) == (5, 2)`。
- `test_release_reclaim_and_active_protection`：驱逐 `cold` 后 `'cold' not in records`，`active.ref_count == 1`，`allocated_blocks == 4 and free_blocks == 1`；`release('new', keep_reusable=False)` 后 `'new' not in records` 且 `allocated_blocks == 2`；再 `append('new', 1)` 抛 `ValueError`。
- `test_capacity_failure_preserves_active_records`：总 3 块时 `acquire('c', 4)` 抛 `RuntimeError`，`records` 仍是 `{a, b}`，`allocated_blocks == total_blocks`。
- `test_shared_state_is_immutable_and_identity_is_consistent`：共享记录上 `append` 抛 `ValueError`，同前缀不同长度 `acquire` 抛 `ValueError`，两次失败前后 `snapshot()` 完全相等。

dump 里 CPU 测试单元没有保存执行输出，所以以断言为准；Step 5 的两段 JSON 有历史输出（baseline 12/12、TTFT 均值 96.317 ms；压力首档 TTFT 均值 3241.254 ms），但那是特定机器和标签（`..._v3_01`、`..._v4_03`）的记录，复跑后以新 JSON 为准。
