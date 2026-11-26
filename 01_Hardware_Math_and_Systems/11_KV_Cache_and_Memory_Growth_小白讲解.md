# 11. KV Cache 与显存增长 — 小白讲解

> 对应 notebook：[`11_KV_Cache_and_Memory_Growth.ipynb`](./11_KV_Cache_and_Memory_Growth.ipynb)

## 一句话总结
本节把推理时的 KV Cache 拆成一条可手算的账本：`kv_cache_bytes` 按 $2\times L\times B\times H_{kv}\times D\times S\times b$ 估算理论字节，32 层、32 个 KV 头、head dim 128、FP16 时，1024/2048/4096 个历史位置分别约 0.54/1.07/2.15 GB；`kv_cache_gb` 对比 MHA/GQA/MQA 的 32/8/1 个 KV 头，得到 2.15/0.54/0.07 GB；`paged_attention_pages`、`mla_cache_bytes` 和 `compare_cache_strategies` 再把"压缩表示"与"按页分配"分开，5 个位置的小例子从 320 B 压到 120 B，但按 4 位置一页分配后实际占 192 B，尾页浪费 3 个槽。

## 为什么需要它：痛点是什么
很多同学第一次遇到"上下文一长就 OOM"，会以为只是激活或权重太大，忽略了 KV Cache。它的麻烦在于三个乘数一起放大：序列长度变长、并发请求变多、层数又多，缓存按乘积增长；而且每生成一个新 token，每一层、每一个 KV 头都要追加一份 K 和 V，不是"偶尔多一点点"。

另一个常见混淆是头数。模型有 32 个 query head，不代表要存 32 组 K/V；MQA 只存 1 组，GQA 存若干组。搞不清这一点，就会把 KV Cache 估算大几倍。还有人听说 PagedAttention 能省显存，就以为分页一定减少字节数——分页主要解决"预留和碎片"，页粒度本身还可能带来尾页浪费。

生活类比：一栋 32 层的办公楼，每层有若干资料柜（KV 头）。每来一位新发言人（新 token），每层每个柜子都要新增一页纪要（K 和 V 各一页）。楼越高、柜子越多、会议越长、同时开的会越多，纸张就越多。MQA 相当于全楼共用一组柜子，GQA 是几层共用一组；PagedAttention 则是把纸张按固定页装订，最后可能剩半页空白。这个类比还能解释"历史状态"：已经开过的会不能重开，但纪要必须一直留着，供后面的发言人参考。

## 核心问题
- KV Cache 的字节数为什么正比于层数、batch、KV 头数、head dim、序列长度和每元素字节数？前面的 2 是什么？
- MHA、MQA、GQA 改变的是哪个维度？为什么 query head 多不代表缓存大？
- 单 token 成本过大、多请求碎片、前缀重复，分别该改表示、组织还是复用？
- 分页分配后，逻辑字节和实际占用的页容量为什么会不一样？
- 公式算出的 GB 能不能直接当成真实显存占用？

## 解决思路：一步步看懂

### 第 1 步：Q1 `kv_cache_bytes`——把公式落到代码
notebook 给的公式是：

$$\mathrm{KVBytes}=2\times L\times B\times H_{kv}\times D\times S\times b$$

- $L$：层数，每层都独立保存自己的 K/V；
- $B$：batch size，不同序列各存各的历史；
- $H_{kv}$：KV 头数，不是 query head 数；
- $D$：每个头的维度（head dim）；
- $S$：当前上下文长度，也就是要保留的历史位置数；
- $b$：每个元素的字节数，FP16/BF16 是 2，FP32 是 4；
- 最前面的 2：K 和 V 各存一份。

代码的参数顺序是 `seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2`，乘法可交换，结果一致：

```python
return 2 * seq_len * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes
```

函数只做一件事：任一参数小于等于 0 就抛 `ValueError`，否则返回理论字节数。注释明确说不包含 allocator、workspace 和碎片，所以它是下界，不是 `nvidia-smi` 上看到的全部占用。

先用小例子核对"元素数 × 每元素字节"：`seq_len=5, num_layers=2, num_kv_heads=3, head_dim=4`，K/V 共 $2\times5\times2\times3\times4=240$ 个元素。每元素 2 字节是 480 B，4 字节是 960 B；序列长度从 5 翻倍到 10，2 字节版本也从 480 变成 960。三个断言正是 `480`、`960`、`960`。

再看 notebook 的数量级示例（32 层、32 个 KV 头、head dim 128、batch 1、FP16）：

| 序列长度 | 计算 | 约 |
|---:|---|---:|
| 1024 | $2\times1024\times32\times32\times128\times2=536{,}870{,}912$ B | 0.54 GB |
| 2048 | 翻倍 | 1.07 GB |
| 4096 | 再翻倍 | 2.15 GB |

注意这里用十进制 GB（除以 $10^9$），和前面章节的显存账本口径一致。变化规律也很直观：`seq_len`、`batch_size`、`num_layers`、`num_kv_heads` 任一翻倍，缓存就翻倍；四项是相乘关系，不是相加。

一个容易混淆的点：注意力计算本身随 $S$ 平方增长（每个新 token 要和所有历史位置算分数），但 KV Cache 只随 $S$ 线性增长，因为保存的是每个位置的 K/V 向量，不是 $S\times S$ 的分数矩阵。分数矩阵是当前这一步的临时结果，算完就丢；K/V 要跨步保留，所以才叫"缓存"。这也是为什么长上下文下 KV Cache 会成为独立于计算量的显存瓶颈：计算可以分块、流式处理，历史状态却要一直占着。

把公式除以 $S$，可以得到"每个历史位置"的成本：

$$\frac{\mathrm{KVBytes}}{S}=2\times L\times B\times H_{kv}\times D\times b$$

再除以 $L$，就是"每个位置、每层"的成本 $2\times H_{kv}\times D\times b$。用 32 个 KV 头、head dim 128、FP16 算：$2\times32\times128\times2=16{,}384$ B，即 16 KiB。乘以 32 层，每个 token 的历史状态是 $16\times32=512$ KiB；4096 个 token 就是 $512\times4096=2{,}097{,}152$ KiB $=2$ GiB，也就是前面表里的 2.15 GB（十进制）。

这里出现了 GB 和 GiB 的差别：$2^{31}$ B 是 2 GiB，除以 $10^9$ 是 2.147 GB。notebook 用十进制 GB，是为了和"7B 参数 × 2 字节 = 14 GB"那套账本一致；如果你看的是 `nvidia-smi`，它有时按 GiB 显示，2.15 GB 会显示成约 2.0 GiB。两个数不是矛盾，是单位不同。

这个"每 token 512 KiB"的尺度很好用：batch 8、4096 上下文，KV Cache 就是 $8\times512$ KiB $\times4096=16$ GiB（约 17.2 GB）；batch 32 就是 64 GiB。长上下文服务里，batch 和长度一起乘，很快就会把显存吃满。

还要区分预填充（prefill）和解码（decode）。预填充把整段 prompt 并行送进模型，虽然一次算完，但每个 prompt token 的 K/V 都要写进缓存；解码每次只生成一个新 token，却要在每一层、每个 KV 头追加一份 K/V。两者都让 $S$ 增长，只是预填充一次跳一大段，解码一步一步加。公式里的 $S$ 是"当前要保留的历史位置总数"，包含 prompt 和已生成的 token；不是"这一秒新生成了多少"。所以即使解码速度不变，缓存也会随对话轮次线性上涨。

### 第 2 步：Q2 `kv_cache_gb`——MHA、MQA、GQA 只改 KV 头数
三种注意力的区别，从缓存角度看只有一件事：要存多少组 K/V。

- MHA（Multi-Head Attention）：每个 query head 都有自己的一组 K/V，$H_{kv}$ 等于 query head 数，缓存最大。
- MQA（Multi-Query Attention）：所有 query head 共享同一组 K/V，$H_{kv}=1$。
- GQA（Grouped-Query Attention）：把 query head 分组，每组共享一组 K/V，$H_{kv}$ 介于两者之间。

`kv_cache_gb` 只是把字节数除以 $10^9$：

```python
return kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim,
                      batch_size, dtype_bytes) / 1e9
```

固定 `seq_len=4096`、32 层、head dim 128，只改 KV 头数：

| 结构 | `kv_heads` | KV Cache |
|:---|---:|---:|
| MHA | 32 | 2.15 GB |
| GQA | 8 | 0.54 GB |
| MQA | 1 | 0.07 GB |

从 32 组降到 8 组是 4 倍，降到 1 组是 32 倍。query head 仍然是 32 个，计算量不会因为 KV 头数下降而等比例下降，但缓存会。这就是长上下文模型偏爱 MQA/GQA 的直接原因：它们改的是"历史状态要存几份"，不是"当前 token 要算几个头"。

换成每 token 的成本更直观：MHA 每 token 每层 $2\times32\times128\times2=16$ KiB，32 层共 512 KiB；GQA（8 组）是 128 KiB；MQA（1 组）是 16 KiB。4096 个 token 分别对应 2 GiB、512 MiB、64 MiB。也就是说，MQA 把"每个历史 token 512 KiB"压到了 16 KiB，长上下文下的差距是几十倍。

代价在表达能力：所有 query head 共享一组 K/V，能表达的头间差异变少。GQA 用 8 组做折中，就是想既保留部分分组差异，又不让缓存按 32 组膨胀。notebook 只算容量，不评估质量；实际选型要把模型精度、任务指标和缓存预算放在一起看，不能因为 MQA 省显存就直接替换。

### 第 3 步：Q3 表示、组织、复用——先判断压力来自哪

notebook 给了一张决策表，把三类问题分开：

| 观察到的现象 | 优先改什么 | 代表机制 | 主要观察量 |
|:---|:---|:---|:---|
| 单 token KV 成本过大 | 缓存表示 | MQA / GQA / MLA | 每 token、每层的缓存字节数 |
| 多请求预留或碎片明显 | 缓存组织 | PagedAttention | 页数、尾页浪费、复用情况 |
| 前缀重复导致重复计算 | 缓存复用 | Prefix Cache | 命中率、复用 token 数 |

这个分类很重要：如果每个 token 的状态本身就太大，先换表示（减少 KV 头或压缩成潜在向量）；如果表示已经能接受，但并发请求下预留和碎片浪费严重，改组织（分页）；如果多个请求共享相同前缀，改复用（前缀缓存）。三者的观察量不同，不能用一个"显存降了多少"混着比较。

分页的核心是两个小函数：

```python
pages = (seq_len + page_size - 1) // page_size          # 向上取整
waste = pages * page_size - seq_len                     # 尾页空槽
```

MLA 风格的压缩表示则用 `mla_cache_bytes`：

```python
return seq_len * num_layers * (latent_dim + rope_dim) * batch_size * dtype_bytes
```

它不再按"KV 头 × head dim"存，而是每层每个位置存一个共享潜在向量（`latent_dim`），再加位置键（`rope_dim`，即 RoPE 分支保存的位置表示）。注意这里没有前面的因子 2，因为 K/V 被压进同一个潜在向量，不再各存一份。

`compare_cache_strategies` 把三种口径放进一个返回值：

```python
'management': {'pages': pages, 'waste_tokens': ...},
'representation': {'base_kv_bytes': ..., 'compact_kv_bytes': ...,
                   'paged_compact_kv_bytes': ..., 'representation_saving_ratio': ...},
'reuse': {'prefix_reuse_tokens': prefix_len},
```

用 notebook 的缩小例子手算：2 层、1 个序列、2 个 KV 头、head dim 4、每元素 2 字节、5 个位置、页大小 4、潜在向量 4 维、位置键 2 维。

- MHA 表示：每位置每层 $2\times2\times4=16$ 个元素（K/V × 2 头 × 4 维）；5 个位置 2 层共 $2\times5\times2\times2\times4\times2=320$ B。
- 压缩表示：每位置每层 $4+2=6$ 个元素；$5\times2\times6\times1\times2=120$ B。
- 分页：$\lceil5/4\rceil=2$ 页，共 $2\times4=8$ 个槽，尾页浪费 $8-5=3$ 个；按 8 个位置算压缩表示，$8\times2\times6\times2=192$ B。
- 节省比例：$1-120/320=0.625$，即 62.5%。

这里出现了一个很值得注意的现象：逻辑压缩后只要 120 B，但按页分配实际占 192 B，反而比逻辑值多 60%。原因是页大小 4、序列只有 5，尾页空了一半多。若序列正好 8 个位置，$\lceil8/4\rceil=2$ 页、浪费 0，压缩逻辑值和分页值都是 192 B。所以"压缩表示"和"分页组织"要分别算：前者决定每个位置多少字节，后者决定分配粒度带来多少空槽。

把这两个口径拆成两个公式，就不容易混：

- 逻辑压缩字节 $=S\times L\times(\text{latent\_dim}+\text{rope\_dim})\times B\times b$；
- 分页字节 $=\lceil S/\text{page\_size}\rceil\times\text{page\_size}\times L\times(\text{latent\_dim}+\text{rope\_dim})\times B\times b$。

两者只差在"用 $S$ 还是用向上取整后的页容量"。页越大，尾页浪费的期望越大，但页表项和元数据越少；页越小，浪费越少，但页数和管理开销越多。这是一个组织层的权衡，不改每个 token 的表示。notebook 用 `page_size=4` 只是为了让 5 个位置的例子能心算；真实系统常取 16、32 或更大，要按序列长度分布选。

页大小对浪费的影响可以手算：序列 1000、页 16 时，$\lceil1000/16\rceil=63$ 页、1008 槽、浪费 8；页 32 时，$\lceil1000/32\rceil=32$ 页、1024 槽、浪费 24；页 4 时，250 页、1000 槽、浪费 0。页越小浪费越少，但页数从 32 涨到 250，元数据和查找成本上升。notebook 没有模拟页表开销，所以不能只看"浪费 tokens"就选最小页。

回到 MLA 的例子：每位置每层的压缩元素是 $4+2=6$，而 MHA 是 $2\times2\times4=16$，比例 $6/16=37.5\%$，省下 62.5%。这与 `representation_saving_ratio == 0.625` 一致。注意压缩后的 6 个元素里，4 个是共享潜在向量、2 个是位置键；位置键不是"额外 K/V"，而是 RoPE 分支需要保存的位置表示。

`prefix_len=2` 只是记录"这 2 个 token 可复用"，并不自动证明物理页被共享；要判断共享是否真的省页，还需要其他请求及它们的共享关系。函数也校验 `0 <= prefix_len <= seq_len`。

真正的复用收益取决于两件事：有多少个请求共享这段前缀，以及共享能持续多久。只有一个请求时，前缀缓存不省任何物理页；很多请求共享同一段系统提示时，才能把同一批页只存一份。实现上还要处理引用计数、写时复制和淘汰，notebook 的 `reuse` 子字典只记录可复用 token 数，正是为了不把这部分复杂度伪装成一个"省了多少 GB"的确定值。

容量规划可以拿这个尺度做除法：如果设备能留给 KV Cache 32 GB，每个 4096 长请求约 2.15 GB，理论上能放约 $32/2.15\approx14$ 个并发请求；再留出激活、workspace 和碎片，实际会更少。若 batch 8，每个请求 2.15 GB，就是 17.2 GB，只剩不到 15 GB 给其他状态。这也解释了为什么长上下文服务常把"最大并发"和"最大长度"放在一起限制，而不是单独承诺某一个。

公式的假设也要写清楚，否则会算偏：

- 各层使用相同的 KV 头数和 head dim；如果层间不同，要逐层算再求和。
- K 和 V 使用同一 dtype；如果 K 量化、V 不量化，要分开乘字节数。
- 保留完整上下文；如果使用滑动窗口或稀疏注意力，$S$ 应换成实际保留的位置数，而不是原始上下文长度。
- 不含 allocator 元数据、对齐填充、workspace 和碎片；这些要另加。
- 不自动包含跨请求共享；Prefix Cache 省的是物理页，不是公式里的逻辑字节。

把这些假设和数字一起写进报告，别人才知道你的 2.15 GB 是"张量下界"还是"包含运行时开销的实测值"。

实际排查时，先量三个数，再决定改哪一层：第一，用 `kv_cache_bytes` 除以 `seq_len` 和 `num_layers`，得到"每 token、每层字节数"，判断表示是否过大；第二，用 `paged_attention_pages` 和 `paged_cache_waste_tokens` 算出页数和尾页浪费，判断组织是否粗放；第三，统计多个请求的公共前缀长度和命中率，判断复用是否有空间。三个数分别对应决策表的三行，不要用一个"显存占用高"就同时改三层。

还有一个顺序问题：表示层的改动会改变组织层的页数。比如把 MHA 换成 GQA 后，每页能放更多 token，尾页浪费的绝对字节也变小；反过来，先调页大小再换表示，可能白调一轮。所以先定表示，再定页大小和复用策略。notebook 把 `compare_cache_strategies` 的 `representation`、`management`、`reuse` 分成三个子字典，就是让你按这个顺序看。

### 第 3 步的读法：三类问题不要混着改

按顺序排查最省事：先算"每个 token、每层多少字节"，如果这个数本身就大，换表示（MQA/GQA/MLA）；表示固定后，再看"每个请求分配了多少页、尾页浪费多少"，如果浪费大，改组织（分页、页大小）；最后看"多个请求有没有相同前缀"，如果有，改复用（Prefix Cache）。一次只改一层，才能把收益归因清楚。

反过来，如果单 token 成本已经很低，却去换更激进的压缩，可能牺牲质量却省不了多少；如果碎片不严重，却上分页，可能只增加元数据。notebook 的决策表把"观察量"写在第三列，就是提醒你先量再改：表示看每 token 字节，组织看页数和尾页浪费，复用看命中率和复用 token 数。

## 关键概念速查表
| 术语 | 大白话解释 |
|:---|:---|
| KV Cache | 解码时保存历史 token 的 Key 和 Value，避免每步重算 |
| $L$ / $B$ / $S$ | 层数 / batch 大小 / 当前上下文长度 |
| $H_{kv}$ | 要保存的 KV 头数，不是 query head 数 |
| head dim（$D$） | 每个注意力头的向量宽度 |
| MHA / MQA / GQA | 每个 query 头一组 K/V / 全部共享一组 / 分组共享 |
| MLA | 把 K/V 压成共享潜在向量，再加位置键的表示 |
| PagedAttention | 把缓存切成固定大小的页来分配，减少预留和碎片 |
| 尾页浪费 | 最后一页没用满的 token 槽位数 |
| 潜在向量 / 位置键 | 压缩缓存里的内容分支 / RoPE 分支保存的位置表示 |
| Prefix Cache | 复用多个请求相同前缀的缓存 |
| 理论字节数 | 只算张量本身的公式值，不含元数据、workspace、碎片 |

## 小白常见疑问

**Q1：公式前面的 2 是什么？为什么不是 3？**
2 表示 K 和 V 各存一份。Q、输出和中间激活不在 KV Cache 里，所以没有第 3 份。MLA 把 K/V 压成共享潜在向量后，这个 2 就不出现了，`mla_cache_bytes` 里只有 `latent_dim + rope_dim`。代码里乘法的顺序和公式不同（`kv_cache_bytes` 把 `seq_len` 放在 `num_layers` 前面），但乘法可交换，结果一样；读代码时不要因为顺序不同就以为公式变了。

**Q2：KV Cache 是随上下文长度增长，还是随生成 token 数增长？**
两者是同一件事的两面。公式里的 $S$ 是当前要保留的历史位置数；每生成一个新 token，$S$ 加 1，每一层、每个 KV 头都追加一份 K/V。所以解码过程是线性增长，长上下文只是把起点抬高。预填充一次写入整段 prompt，解码再逐个追加；如果实现会丢弃早期 token（滑动窗口），$S$ 就不是原始上下文长度，而是窗口实际保留的位置数。

**Q3：模型有 32 个 query head，为什么 GQA 只存 8 组 KV，缓存就小了 4 倍？**
因为缓存只保存 K/V，不保存 query。32 个 query head 仍然要计算，但它们可以分组共享 8 组 K/V；`kv_cache_gb` 的 $H_{kv}$ 从 32 改成 8，字节数就按比例降到四分之一。MQA 把这个数降到 1，所以是 32 分之一。换成每 token 成本，MHA 是 512 KiB、GQA 是 128 KiB、MQA 是 16 KiB；同样 4096 上下文，就是 2 GiB、512 MiB、64 MiB 的差距。

**Q4：压缩表示从 320 B 降到 120 B，为什么分页后变成 192 B，不是仍然 120 B？**
因为分页按 4 个位置一页分配，5 个位置要 2 页、8 个槽。实际分配按 8 个位置算，$8\times2\times6\times2=192$ B。逻辑压缩和物理分配是两个口径；如果序列正好 8 个位置，两者都是 192 B，没有尾页浪费。页越大，尾页浪费的期望越大；页越小，页数和元数据越多。notebook 的 `page_size=4` 是为了配合 5 个位置心算，不代表真实系统的最优页大小。

**Q5：公式算出的 2.15 GB，就是推理时真实占用的显存吗？**
不是。`kv_cache_bytes` 只算 K/V 张量本身，注释明确排除 allocator、workspace 和碎片。真实占用还可能包含元数据、对齐填充、临时缓冲，以及框架预留但未使用的页。公式给的是下界和增长趋势，不是 `nvidia-smi` 的读数。不过它足够用来做容量规划的除法：32 GB 预算除以 2.15 GB，约能放 14 个 4096 长请求；真实部署再按实测开销打折。如果换成 GQA 的 0.54 GB，同样 32 GB 理论上能放约 59 个，但这个数只比较 KV，不含权重和激活。

## 动手验证：怎么知道自己做对了
本节没有 TODO 空位，函数都是完整参考实现，验证靠单元里的 `assert`。notebook 没有保存执行输出，运行后每个单元打印尺寸和字典；以断言和返回值为准。

`kv_cache_bytes` 的断言：
- `kv_cache_bytes(5, 2, 3, 4, dtype_bytes=2) == 480`；
- 同样参数、`dtype_bytes=4` 时等于 960；
- `seq_len=10`、2 字节时等于 960（长度翻倍，字节翻倍）。

`kv_cache_gb` 没有单独断言，靠 Q2 循环打印 MHA/GQA/MQA 的 GB 值；数值应分别为 2.15、0.54、0.07。

`paged_attention_pages` / `paged_cache_waste_tokens` 通过 `compare_cache_strategies` 间接验证：5 个位置、页大小 4 时 `management == {'pages': 2, 'waste_tokens': 3}`。

`compare_cache_strategies` 的精确断言：
- `base_kv_bytes == 320`、`compact_kv_bytes == 120`、`paged_compact_kv_bytes == 192`；
- `representation_saving_ratio == 0.625`；
- `reuse['prefix_reuse_tokens'] == 2`；
- 序列长度改成 8 时，`waste_tokens == 0`，且 `compact_kv_bytes == paged_compact_kv_bytes == 192`。

函数里还有一条 `0 <= prefix_len <= seq_len` 的校验，越界会抛 `ValueError`；notebook 没有为它单独写断言，读代码时知道有这道保护即可。

运行后应看到两行字典：`5个位置 -> ...` 和 `8个位置 -> ...`，前者压缩逻辑 120 B、分页 192 B、2 页浪费 3 槽，后者浪费 0 槽。这些对上了，就说明你已经能把"表示压缩"和"分页粒度"这两笔账分开算了。

读输出时可以按字段核对，避免只看一个总数：

- `management.pages` 是向上取整的页数，5 个位置、页 4 得 2；如果得到 1，说明用了向下取整。
- `management.waste_tokens` 是 $2\times4-5=3$；如果得到 0，可能把 5 当成了 8。
- `representation.base_kv_bytes` 是 320，`compact_kv_bytes` 是 120，两者的比例给出 0.625；如果比例是 0.375，说明把"省下多少"和"还剩多少"弄反了。
- `representation.paged_compact_kv_bytes` 是 192，不是 120；这个字段专门用来提醒你"分页会按整页算"。
- `reuse.prefix_reuse_tokens` 只是 2，函数没有计算跨请求的物理页共享；不要把它当成"省了 2 个 token 的字节"。

另外注意 `compare_cache_strategies` 的 `num_layers=32`、`batch_size=1`、`dtype_bytes=2` 都有默认值，但例子显式传了 `num_layers=2`、`batch_size=1`、`dtype_bytes=2`；如果只传前几个位置参数，容易把 `page_size` 和 `num_layers` 的顺序搞混。最稳妥的写法是像 notebook 一样用关键字参数，并在报告里写清每个数的单位（B、KiB、GB）和口径（逻辑/分页）。
