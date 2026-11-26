# 22. vLLM 分页注意力 — 小白讲解

> 对应 notebook：[`22_vLLM_PagedAttention.ipynb`](./22_vLLM_PagedAttention.ipynb)

## 一句话总结
本节用一个 CPU 版的 `KVCacheManager` 讲清 PagedAttention 的核心：把 KV Cache 切成固定大小的物理 Block，用 `block_table` 记录逻辑位置到物理块的映射，按需分配、跨块扩容、释放后回收。四个 TODO 分别实现容量账本、Prefill 原子分配、Decode 边界判断和非连续物理块的逻辑恢复，测试覆盖 OOM 回滚、尾块浪费和块复用。最后强调教学池只是二维/三维小张量，不等同于 vLLM 的真实 KV Cache layout。

## 为什么需要它：痛点是什么
在线推理的请求不会同时开始、同时结束：有的提示很短，有的上下文很长，有的已经生成完，有的还在逐词元解码。如果每个请求都按 `max_len` 预留一整块连续显存，短请求会留下大量用不到的尾部，长请求之间还会出现难以复用的空洞。显存被"预留"占满，能同时服务的请求数就上不去。

生活类比：酒店房间。连续预留像给每位客人固定留一整层，哪怕他只住一晚；分页管理像把房间切成标准间，客人来了按需给房，房号可以不连续，退房后房间回到前台重新分配。前台登记表（block table）负责把"客人的第 3 晚"映射到实际房间号，客人不需要知道房间在走廊哪一端。

## 核心问题
- KV Cache 的理论容量由哪些维度决定？单 token、单 Block、整个池怎么逐级相乘？
- Prefill 时如何按 Block 大小向上取整分配？OOM 时为什么必须保持"什么都没发生"？
- Decode 每追加一个 token，什么时候才需要新 Block？
- 物理块编号不连续时，怎样按逻辑顺序恢复缓存并截掉尾块？
- 释放请求时应该检查哪些非法状态，释放后的块怎样复用？

## 解决思路：一步步看懂

**Step 1：先理解"块"是什么。** 这里的块是容纳若干词元 K/V 的存储单位，**不是** Transformer 模型块，也不是 CUDA 线程块。块内逻辑位置连续，物理块编号可以不连续。管理方式对比如下：连续预留按 `max_len` 一次分配，长度未知时产生尾部浪费；固定 Block 把分配粒度统一，便于回收；分页管理按需分配，物理块可以不连续；Block Table 记录逻辑 Block 到物理 Block 的映射，计算时仍按逻辑 token 顺序访问。

**Step 2：TODO 1——容量账本。** `estimate_kv_cache_bytes(num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes=2)` 的参考答案：

```python
kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
kv_bytes_per_block = block_size * kv_bytes_per_token
total_cache_bytes = num_blocks * kv_bytes_per_block
```

公式逐项解释：开头的 `2` 表示每个 token 同时存 Key 和 Value；`num_layers` 是层数；`num_kv_heads` 是 KV Head 数（GQA/MQA 下小于查询 Head 数）；`head_dim` 是每个 Head 的维度；`dtype_bytes` 是每个元素的字节数（默认 2，即 FP16/BF16）。用测试参数 `(4, 4, 2, 8, 16, 2)` 手算：

$$2\times2\times8\times16\times2=1024\ \text{字节/token}$$

$1024\times4=4096$ 字节/Block，$4096\times4=16384$ 字节总容量。测试断言 `kv_bytes_per_token == 1024`、`total_cache_bytes == 16384`。返回的 `layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]`，本例是 `[2, 2, 8, 4, 16]`，表示"层 × K/V × KV Head × Block 内 token × Head 维"。notebook 注明这只是理论容量，不代表 vLLM allocator 的实际峰值（真实实现还有对齐、元数据和碎片）。

**Step 3：TODO 2——Prefill 原子分配。** `allocate_for_prefill` 先拒绝重复分配：`if req.block_table: raise ValueError(...)`。然后：

```python
needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
if len(self.free_blocks) < needed_blocks:
    raise RuntimeError("OOM")
allocated_blocks = self.free_blocks[:needed_blocks]
del self.free_blocks[:needed_blocks]
req.block_table.extend(allocated_blocks)
```

第一行是"向上取整到块大小"的标准写法。`prompt_len=5`、`block_size=4` 时 `(5+3)//4=2`，需要两块。关键是 **先检查再修改**：容量不足时直接抛 `RuntimeError('OOM')`，此时 `req.block_table` 还是空的、`free_blocks` 也没动，不会留下"只分配了一半"的状态。`test_oom_is_atomic` 用 1 个块、`prompt_len=5` 验证这一点，断言异常信息含 `OOM`，且 `req.block_table == []`、`manager.free_blocks == free_before`。

**Step 4：TODO 3——Decode 跨块扩容。** 每解码一个 token，序列长度加一，但只有跨过当前尾块边界才需要新块：

```python
new_seq_len = req.seq_len + 1
is_new_block_needed = (new_seq_len + self.block_size - 1) // self.block_size > len(req.block_table)
if is_new_block_needed:
    if not self.free_blocks:
        raise RuntimeError("OOM")
    req.block_table.append(self.free_blocks.pop(0))
req.seq_len = new_seq_len
```

例：`block_size=4`、已有 6 个有效位置、占 2 块。第 7、8 个位置仍在尾块内，`ceil(8/4)=2` 不大于 `len(block_table)=2`，不分配；第 9 个位置 `ceil(9/4)=3 > 2`，才申请第三块。`pop(0)` 是 FIFO，具体取哪个物理块不重要，重要的是编号可以不连续。失败路径也要原子：无空闲块时在修改 `seq_len` **之前**抛 `RuntimeError`，所以长度和块表都不变。`test_single_token_blocks_and_decode_rollback` 用 `block_size=1` 让每个位置都扩容，第二次 Decode 时 OOM，断言 `(seq_len, block_table, free_blocks)` 与调用前完全一致。

**Step 5：TODO 4——按逻辑顺序恢复缓存。** `get_physical_cache` 先做两个检查：`block_table` 为空抛 `ValueError`；任一物理索引越界抛 `ValueError`。然后：

```python
blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
cat_blocks = torch.cat(blocks, dim=0)
return cat_blocks[:req.seq_len]
```

顺序由 `block_table` 决定，**不能按物理编号排序**。测试 `test_logical_cache_reconstruction` 用 `block_size=2`、`seq_len=3`，块表是 `[3, 1]`，把物理块 3 填成 3.0、物理块 1 填成 1.0；恢复后形状 `(3, 2)`，前两行是 3.0、最后一行是 1.0。如果把块表排序成 `[1, 3]`，顺序就反了。`torch.cat` 后还要 `[:req.seq_len]` 截断尾块，否则会多返回尾块里未使用的槽位。

**Step 6：报告、释放与复用。** `allocation_report` 用 `allocated_tokens = len(block_table) * block_size` 算已分配槽位，`unused_tail_tokens = allocated_tokens - seq_len`，`utilization = seq_len / allocated_tokens`。测试 `prompt_len=5`、2 个块、`block_size=4`，已分配 8 个槽位，尾块浪费 3。`release_request` 的校验顺序是：块表为空（可能已释放）→ 有重复物理块 → 有越界块 → 有的块已经在 `free_blocks` 里；全部通过后 `free_blocks.extend(...)`、排序、`block_table.clear()`。`test_release_and_reuse` 释放后 `free_blocks == [0, 1, 2]`，说明块回到池里可被下一个请求复用。注意这里描述的是**每请求独占块**，跨请求共享还要额外做引用计数，notebook 明确说这不在本节范围内。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KV Cache | 推理时缓存历史 token 的 Key/Value，避免每步重算 |
| PagedAttention | 把 KV Cache 切成固定大小的块，按需分配、用块表映射 |
| 物理 Block | 实际存 K/V 的固定大小存储单位，编号可不连续 |
| 逻辑 Block | 按 token 顺序切出来的第 i 段，和物理块通过块表对应 |
| Block Table | 记录 `block_table[i]` → 物理块编号的登记表 |
| Prefill | 处理提示词、建立初始 KV Cache 的阶段 |
| Decode | 逐词元生成、每次追加一个位置的阶段 |
| 尾块浪费 | 已分配槽位数减真实序列长度，最后一块里没用上的部分 |
| 向上取整 | `(seq_len + block_size - 1) // block_size`，保证容量不少于需求 |
| 原子分配 | 要么全部分到，要么状态完全不变 |
| 逻辑恢复 | 按块表顺序读取物理块，再截断到 `seq_len` |

## 小白常见疑问

**Q1：为什么容量公式里要乘 `2`，又乘 `num_layers`？**
`2` 是 Key 和 Value 两份；`num_layers` 是每一层都有自己的 K/V。以 `(4,4,2,8,16,2)` 为例，单 token 是 `2×2层×8头×16维×2字节 = 1024` 字节；只算一层会少一半，漏掉 Key 或 Value 也会少一半。测试 `test_request_and_ledger_contract` 直接断言 1024。

**Q2：`needed_blocks` 为什么要加 `block_size - 1` 再整除？**
这是向上取整。`prompt_len=5`、`block_size=4` 时 `5//4=1`，不够；`(5+3)//4=2` 才对。如果写成 `seq_len // block_size`，长度刚好整除时也会多算一块（例如 4//4=1 没问题，但 5 就会少算），边界很容易错。

**Q3：Decode 时为什么不直接每次分配一个新块？**
那样每个 token 占一个块，块内空间全浪费，块表也会变得很长。`block_size=4` 时，8 个位置只需要 2 块；只有第 9 个位置才需要第 3 块。测试 `test_prefill_and_decode_growth` 明确：`prompt_len=4` 分配 1 块，Decode 一次后 `seq_len=5`、`len(block_table)==2`，因为 5 跨过了 4 的边界。

**Q4：块表是 `[3, 1]`，恢复时能按 `[1, 3]` 读吗？**
不能。块表记录的是逻辑顺序：逻辑第 0 块对应物理 3，逻辑第 1 块对应物理 1。按物理编号排序会改变 token 顺序，注意力算出来的结果就错了。`test_logical_cache_reconstruction` 特意把物理块 3 填 3.0、物理块 1 填 1.0，断言恢复后前两行是 3.0、最后一行是 1.0。

**Q5：OOM 时为什么要求"长度和块表都不变"，而不是尽量分配已有的块？**
因为半分配会让请求处于不一致状态：块表指向了一部分块，但长度或后续逻辑以为分配完整。`test_oom_is_atomic` 和 `test_single_token_blocks_and_decode_rollback` 都检查失败前后 `block_table`、`free_blocks`、`seq_len` 完全一致。先检查容量、再修改状态，是避免这种问题的标准做法。

## 动手验证：怎么知道自己做对了
运行 `test_paged_attention_manager()`，它依次调用六个测试，全部通过打印 `✅ PagedAttention 机制测试通过：账本、扩容、映射、OOM 和回收均通过。`。
- `test_request_and_ledger_contract`：`Request(-1, 2)` 和 `Request(1, 0)` 必须抛 `ValueError`；`estimate_kv_cache_bytes(4, 4, 2, 8, 16, 2)` 的 `kv_bytes_per_token == 1024`、`total_cache_bytes == 16384`。
- `test_prefill_and_decode_growth`：`KVCacheManager(num_blocks=4, block_size=4, head_dim=8)`，`prompt_len=4` 的 Prefill 后 `len(block_table)==1`；Decode 一次后 `seq_len==5`、`len(block_table)==2`。
- `test_logical_cache_reconstruction`：`num_blocks=4, block_size=2, head_dim=2`，`prompt_len=3`、块表 `[3,1]`，物理块 3 填 3.0、物理块 1 填 1.0；恢复形状 `(3,2)`，`cache[:2]` 全是 3.0、`cache[2:]` 全是 1.0。
- `test_oom_is_atomic`：`num_blocks=1, block_size=4`、`prompt_len=5` 需要 2 块，必须抛含 `OOM` 的 `RuntimeError`；失败后 `req.block_table == []`、`free_blocks` 与调用前相同。
- `test_release_and_reuse`：`num_blocks=3, block_size=4`、`prompt_len=5`，`allocation_report` 的 `unused_tail_tokens == 3`；`release_request` 后块表清空、`free_blocks == [0, 1, 2]`。
- `test_single_token_blocks_and_decode_rollback`：`block_size=1` 时 Prefill 占 1 块、Decode 后 `seq_len==2` 且占 2 块；再 Decode 时 OOM，`(seq_len, block_table, free_blocks)` 与调用前一致。

notebook 的 CPU 单元没有保存输出，dump 里也没有运行结果，因此以这些断言为准。测试里的 `KVCacheManager.head_dim` 只影响教学张量的最后一维（如 `head_dim=8` 或 `2`），不参与 `estimate_kv_cache_bytes` 的容量公式——后者需要显式传入 `num_layers`、`num_kv_heads`、`head_dim` 和 `dtype_bytes`，两者不要混为一谈。
