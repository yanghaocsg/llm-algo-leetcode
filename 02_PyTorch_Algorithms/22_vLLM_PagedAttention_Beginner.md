# 22. vLLM PagedAttention Beginner | vLLM 分页注意力 · 小白入门

> 配套 notebook：[22. vLLM 分页注意力](./22_vLLM_PagedAttention.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

大模型生成时要为每个请求保存一份越来越长的缓存。这本 notebook 教你把缓存切成固定大小的小块，用到多少分多少，并用一张表记录每个请求的缓存放在哪些块里。

## 为什么需要它

大模型生成时，每处理一个位置，就把算出来的 K 和 V 存起来，后面的词元直接读取，不用重算。存下来的这些就是 KV Cache，序列越长它越大。

在线服务里，请求长短不一，开始和结束的时间也参差不齐：有的提示词很短，有的上下文很长，有的已经生成完，有的还在逐个词元生成。如果给每个请求都按最大长度预留一整块连续的 KV Cache，会出两种浪费：

- 尾部浪费：空间一次分满，请求实际只用到前面一部分，后面空着。
- 空洞难复用：每个请求占着一整块连续空间，别的请求用不上空着的部分。

空着的容量限制了同时能容纳的请求数。分页注意力（PagedAttention）借了操作系统虚拟内存的思路：把 KV Cache 切成固定大小的物理块，用到多少分多少；再用一张块表记下“这个请求的第几段放在哪个物理块”。物理块可以不连续，按块表的顺序取，逻辑上仍然连续。

这本 notebook 在 CPU 上模拟这套管理：算容量、按需分配块、跨块时扩容、按块表恢复缓存。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| KV Cache | 生成时把每个位置算出来的 K 和 V 存下来，后面的词元直接读取，不用重算；它随序列变长而增长 |
| 分页注意力（PagedAttention） | 把 KV Cache 放进固定大小的物理块，用一张块表记录每个请求用了哪些块 |
| 物理块（Block） | 固定大小的存储单位，每块放固定数量的词元的 K/V，例如 16 个。它不是 Transformer 的模型块，也不是 CUDA 线程块 |
| 块表（block table） | 每个请求一张表，第 `i` 项是第 `i` 个逻辑块放在哪个物理块。逻辑块按词元先后排列，物理块编号可以不连续 |
| 尾块 | 请求的最后一个块，通常只填了一部分 |
| 预填充（prefill）/ 解码（decode） | 预填充处理提示词并建立初始缓存；解码处理新位置并追加它的 K/V |
| OOM（out of memory） | 池里的空闲块不够，分配失败时抛出的错误 |

## 核心思路

notebook 用一个 `KVCacheManager` 类模拟块池的管理：物理块池是 `physical_kv_cache`，`free_blocks` 记录哪些块还空着，分配器负责取块和还块。每个请求是一个 `Request` 对象，`seq_len` 是已经处理的词元数，`block_table` 是逻辑块到物理块的对应关系。

四个 TODO 对应四个动作：

- 算容量：一个词元的 K/V 占多少字节，一个块多大，整个池多大。
- 预填充分配：把提示长度按块大小向上取整，算出块数，从空闲块里一次取够。
- 解码扩容：每处理一个新位置，`seq_len` 加 1；只有跨过块边界时才多要一个块。采样出一个新词元，不等于它的 K/V 已经写入缓存。
- 恢复缓存：按块表的顺序读出物理块，拼起来，再截到真实长度。

容量的算法：

```text
单词元字节数 = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
单块字节数   = block_size * 单词元字节数
整池字节数   = num_blocks * 单块字节数
```

开头的 `2` 是 K 和 V 各一份。`num_layers` 是层数，`num_kv_heads` 是 KV 头的个数，`head_dim` 是每个头的维度，`dtype_bytes` 是每个元素占几个字节，`block_size` 是每块的词元数，`num_blocks` 是池里的块数。

下面三组数字都来自 notebook 的 Step 2、Step 3 和测试：

```python
import torch

block_size = 4
for length in (6, 7, 8, 9):                                 # 有效长度
    print(length, (length + block_size - 1) // block_size)  # 需要的块数：向上取整

pool = torch.zeros(8, block_size, 1)                         # 8 个物理块，每块放 4 个词元
pool[7], pool[2] = 7.0, 2.0                                  # 物理块 7 的内容全是 7，物理块 2 的内容全是 2
block_table, seq_len = [7, 2], 6                             # 逻辑块 0 → 物理块 7，逻辑块 1 → 物理块 2
by_table = torch.cat([pool[b] for b in block_table], dim=0)[:seq_len]
by_index = torch.cat([pool[b] for b in sorted(block_table)], dim=0)[:seq_len]
print(by_table.flatten().tolist())                           # 按块表顺序
print(by_index.flatten().tolist())                           # 错误做法：按物理编号排序

per_token = 2 * 2 * 8 * 16 * 2                               # K 和 V 各一份、2 层、8 个 KV 头、每头 16 维、每元素 2 字节
print(per_token, 4 * per_token, 4 * 4 * per_token)           # 单词元、单块、整池
```

```text
6 2
7 2
8 2
9 3
[7.0, 7.0, 7.0, 7.0, 2.0, 2.0]
[2.0, 2.0, 2.0, 2.0, 7.0, 7.0]
1024 4096 16384
```

前四行是向上取整：块大小为 4 时，6、7、8 个有效位置都只占 2 块，到第 9 个才要第 3 块。中间两行是恢复缓存：块表是 `[7, 2]`，前 4 个位置来自物理块 7，后 2 个来自物理块 2；按物理编号排序就读反了。最后一行是容量：单词元 1024 字节，一块 4096 字节，整池 16384 字节。

这本 notebook 里每个请求独占自己的块，跨请求共享还要增加引用管理，这里没有涉及。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | PagedAttention 如何组织逻辑与物理 KV Cache：连续预留的问题，块和块表是什么 | 读 |
| Step 2 | 有效缓存长度怎样触发块分配：预填充、解码追加、释放、再分配四个阶段的边界 | 读，对照上面的向上取整 |
| Step 3 | Block Table 如何把逻辑访问映射到物理缓存：四个数据结构和它们要保持的规则 | 读，对照上面的块表例子 |
| Step 4 · TODO 1 | 计算单词元、单 Block 和总容量 | 写 `estimate_kv_cache_bytes`：补 3 行 `kv_bytes_per_token`、`kv_bytes_per_block`、`total_cache_bytes` |
| Step 4 · TODO 2 | 向上取整并选择待提交物理块 | 写 `allocate_for_prefill`：补 2 行 `needed_blocks` 和 `allocated_blocks` |
| Step 4 · TODO 3 | 识别追加词元是否跨越 Block 边界 | 写 `allocate_for_decode`：补 2 行 `new_seq_len` 和 `is_new_block_needed` |
| Step 4 · TODO 4 | 按逻辑映射表恢复缓存顺序 | 写 `get_physical_cache`：补 2 行 `blocks` 和 `cat_blocks`，按块表顺序拼接 |
| 测试 | 检查容量账本、预填充与解码扩容、非连续物理块的恢复、容量不足时是否留下半分配状态 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |

整本 notebook 只需要 CPU。

## 容易踩的坑

- **按物理编号排序块表**：恢复缓存要按块表的顺序读，也就是逻辑块的顺序。物理块编号可以不连续，排序后词元顺序就错了。
- **忘记截断尾块**：拼起来的缓存带着尾块里空着的位置，要截到 `seq_len`。
- **容量不足时留下半分配**：预填充要么一次取够所有块，要么报 OOM。报错之后，请求的块表和空闲块集合都要和原来一样。
- **解码每次都申请新块**：只有追加后跨过块边界才多要一个块。扩容失败时，长度和块表都不能变。
- **容量只算了一份**：单词元容量要同时计入 K 和 V，再乘层数、KV 头数、头维度和每个元素的字节数。

## 读完后自查

- [ ] 块大小是 4 时，能说出有效长度 6、8、9 各占几个块。
- [ ] 块表是 `[7, 2]`、有效长度是 6 时，能说出每个位置来自哪个物理块。
- [ ] 能写出单词元、单块、整池的容量算式。
- [ ] 能解释预填充分配失败时，为什么要保证请求和空闲池都不变。

## 下一步

- 打开 notebook：[22. vLLM 分页注意力](./22_vLLM_PagedAttention.ipynb)
- 做完之后可以对照：[24. Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.ipynb)、[34. SGLang 基数注意力](./34_SGLang_RadixAttention.ipynb)、[37. KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb)
- 做完之后：[37. KV Cache 生命周期与容量控制 · 小白入门](./37_KV_Cache_Lifecycle_and_Capacity_Control_Beginner.md)
