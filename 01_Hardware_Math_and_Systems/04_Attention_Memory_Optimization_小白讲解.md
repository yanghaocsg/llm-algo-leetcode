# 04. 注意力机制变体与显存优化 — 小白讲解

> 对应 notebook：[`04_Attention_Memory_Optimization.ipynb`](./04_Attention_Memory_Optimization.ipynb)

## 一句话总结
这节把注意力的显存压力拆成三类——长期保存的 KV Cache、计算期间的临时工作集、服务运行时的缓存组织——再分清每种技术动的是哪一类：MQA / GQA 靠共享 K/V 头减少状态量，MLA 靠压缩表示，PagedAttention 靠按页分配改变缓存的物理放置。几个小计算器都是纯算术，CPU 上就能跑。

## 为什么需要它：痛点是什么
做大模型推理时显存不够、并发上不去，却不知道该换模型结构还是改推理框架；听到 MQA、GQA、MLA、PagedAttention 这些名词，也分不清它们各解决什么问题、能不能叠加。如果不知道"压力来自哪一类成本"，就很容易拿解决 A 问题的办法去对付 B 问题。
类比：健身房储物柜。MHA 是每位会员（每个 Query 头）配一个专属柜子（K/V）；MQA 是全员共用一个柜子，GQA 是每个小组共用一个；MLA 是柜子里只放一张"压缩清单"，要用时再还原；PagedAttention 改的是租法——不再要求每人一租就是按最大长度预留的一整排连续大柜，而是一格格按需租，再用一张登记表记住自己的东西在几号柜。

## 核心问题
- 标准多头注意力的 KV Cache 占多少显存，和哪些量成正比？
- MQA 和 GQA 各怎么共享 K/V 头，能把缓存缩到几分之几，代价是什么？
- MLA 存的是压缩后的潜在向量，不还原也能算注意力分数吗？
- PagedAttention 按页分配到底省了多少，什么时候反而更费？

## 解决思路：一步步看懂
1. **先算 KV Cache 的账（Q1）。** 自回归生成（一个 token 接一个 token 往后生成）时，标准多头注意力（MHA）要为每一层、每个历史 token 保存 K 和 V，这就是 KV Cache（键值缓存）。理论字节数是

   $$\text{KV Cache bytes} = 2 \times \text{batch} \times \text{layers} \times \text{seq\_len} \times \text{kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes}$$

   其中 2 对应 K 和 V 两份；batch 是同时处理的序列数；layers 是层数；seq_len 是已缓存的上下文长度；kv_heads 是 K/V 头数；head_dim 是单个头的维度；dtype_bytes 是每个元素的字节数（FP16 / BF16 通常为 2）。手算：`batch=1`、`layers=32`、`seq_len=2048`、`kv_heads=32`、`head_dim=128`、FP16，得 $2 \times 1 \times 32 \times 2048 \times 32 \times 128 \times 2 = 1{,}073{,}741{,}824$ 字节，约 1.07 GB；并发序列增加到 32，理论容量需求也近似扩大 32 倍。六个变量都是近似线性增长。`kv_cache_bytes(seq_len, layers, heads, head_dim, dtype_bytes=2, batch_size=1)` 只是理论值，不含 block metadata、padding、量化 scale、allocator reserve 和 runtime workspace；`heads` 要传 KV 头数，不是 query 头数。缓存会经历 Prefill（建立初始 K/V）、Decode（每生成一个 token 就追加并读取已有缓存）、回收（请求结束后释放）三个阶段：长度和并发增长带来容量压力，Decode 的反复读取还可能带来带宽压力，是否成为 Memory Bound 取决于模型、dtype、并发和实现。
2. **MQA / GQA：少存几份 K/V（Q2）。** 研究人员改的是注意力的投影结构，目的是减少要搬运的数据量。MQA（Multi-Query Attention，多查询注意力）：无论有多少个 Query 头，全部共享仅仅 1 个 Key 头和 1 个 Value 头，KV 头数从 $H$ 降到 1，缓存约为 MHA 的 $1/H$。GQA（Grouped-Query Attention，分组查询注意力）是折中：把 Query 头分组（例如 32 个分 8 组），每组共享 1 对 K/V 头，KV 头数从 $H$ 降到 $G$，缓存约为 $G/H$；$H=32, G=8$ 时约是 MHA 的 1/4。代价：MQA 共享 K/V 头会改变模型表达容量，可能影响部分任务质量；GQA 需要匹配的模型结构或 checkpoint。对照单元固定 `seq_len=4096`、`layers=32`、`head_dim=128`，只把 KV 头数换成 32、8、1，用 `kv_cache_gb` 除以 MHA 的结果，断言比例正好是 `{'MHA': 1.0, 'GQA': 0.25, 'MQA': 1/32}`。这些是理论容量比例，端到端速度是否提升仍取决于 kernel、backend 和 workload。
3. **MLA：存压缩表示，分数照样能算（Q3）。** MLA（Multi-head Latent Attention，多头潜在注意力）把 K 和 V 的内容信息联合压缩成低维向量，叫潜在表示（latent）。它和 GQA 的区别：GQA 减少 K/V 头数，MLA 改变保存的表示。以内容键为例，潜在向量为 $c$、上投影矩阵为 $U_K$，展开后的键是 $k = U_K c$（上投影就是把较短向量映射回计算所需的较宽表示）。算分数有两条路：
   - 路径一：先恢复键，再和查询 $q$ 点积；
   - 路径二：利用矩阵乘法结合律，先变换查询：$q^{\top}(U_K c) = (U_K^{\top} q)^{\top} c$。查询先被"压短"，直接和历史潜在向量点积，不必每轮展开全部历史 K/V；相应的投影还可以与查询侧、输出侧的计算合并，称为矩阵吸收。两条路径的算术操作和临时张量不同，成本要按实际实现比较。

   手算 notebook 的例子：查询 $[1, -1, 2, 0]$，潜在向量 $[2, 3]$，$U_K$ 是 4 行 2 列的 $[[1,0],[0,1],[1,1],[2,-1]]$。路径一恢复出键 $[2, 3, 5, 1]$，分数 $1 \times 2 + (-1) \times 3 + 2 \times 5 + 0 \times 1 = 9$；路径二把查询变换成 $[3, 1]$，分数 $3 \times 2 + 1 \times 3 = 9$，一致。旋转位置编码（RoPE）直接插入内容键会影响这种合并，所以 DeepSeek-V2 把内容与位置分开处理，缓存共享潜在向量和一份携带 RoPE 的位置键；小验证只核对内容分数，省略了缩放与位置分支。
4. **PagedAttention：按页租格子（Q4）。** 早期推理实现为每个请求预留连续空间，但生成长度不可预知，长短请求混合时容易预留浪费、空间难组织。机制有三点：固定大小的 block（把 KV Cache 按 token 切成固定大小的块）；非连续存储（通过 block table 找到逻辑 token 对应的物理位置）；按需增长（需要更多 token 时再申请）。
   - 五个请求，长度 64、96、128、320、512。连续方案每个请求预留 512 个位置：$5 \times 512 = 2560$ 个槽位。按 128 个位置一页分配，分别需要 1、1、1、3、4 页，共 $10 \times 128 = 1280$ 个槽位。有效位置 $64 + 96 + 128 + 320 + 512 = 1120$，利用率 $1120 / 1280 = 87.5\%$，其余 12.5% 是尾页空槽。
   - 尾页也有代价：单个长度 129 的请求，若连续方案恰好预留 129 个位置，只需 129 个槽位；分页却要两页、共 256 个槽位，多出 127 个。所以页大小和连续方案的预留规则要一起给出，不能从前一组例子推断所有输入都更省。
   - PagedAttention 改变的是缓存的分配和寻址方式，不是 KV 表示本身；实际并发收益还取决于 block size、调度、kernel、prefix sharing 和 workspace。要和 Prefix Cache / RadixAttention 分开：前者回答"缓存放在哪里"（请求内部的 block 分配、寻址与回收），后者回答"已有前缀能否重复使用"（请求之间的共享前缀复用）。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KV Cache | 生成时为每层、每个历史 token 存下的 K 和 V |
| Prefill / Decode / 回收 | 建立初始 K/V / 每生成一个 token 就追加并读取 / 请求结束后释放空间 |
| MHA（多头注意力） | 每个 Query 头配一套独立的 K/V 头 |
| MQA（多查询注意力） | 所有 Query 头共用 1 组 K/V 头，缓存约为 MHA 的 $1/H$ |
| GQA（分组查询注意力） | Query 头分组，每组共用 1 组 K/V 头，缓存约为 $G/H$ |
| $H$ / $G$ | Query 头数 / 分组后的 KV 头数 |
| MLA（多头潜在注意力） | 把 K、V 的内容压缩成低维潜在向量来保存 |
| 潜在向量 $c$ / 上投影矩阵 $U_K$ | 压缩后的表示 / 把它还原成完整键的矩阵 |
| 矩阵吸收 | 把还原矩阵挪到查询侧、输出侧合并计算，不必展开历史 K/V |
| RoPE（旋转位置编码） | 用旋转表示 token 位置，直接插进内容键会妨碍矩阵吸收 |
| PagedAttention | 把 KV Cache 切成固定大小的页，非连续存放，靠 block table 寻址 |
| Prefix Cache / RadixAttention | 让不同请求复用相同的开头部分，和"放在哪里"是两件事 |

## 小白常见疑问
**Q1：用了 GQA 或 MQA，推理就一定按比例变快吗？**
不能这么保证。`1/4`、`1/32` 是理论容量比例；端到端速度是否提升取决于 kernel、backend 和 workload，实际速度、质量和 backend 收益要在固定 workload 上验证。

**Q2：MQA 省得最多，为什么不都用 MQA？**
共享 K/V 头会改变模型表达容量，可能影响部分任务质量，影响程度取决于模型结构、训练方式和具体 checkpoint。GQA 就是这个折中。两者都需要匹配的模型结构，不是推理时随手能开的开关。

**Q3：MLA 存的是压缩表示，为什么不用每轮都把历史 K/V 还原出来？**
结合律允许换顺序：先恢复键再点积，和先用 $U_K^{\top}$ 把查询压短再点积，结果一样。后者只需读历史潜在向量。

**Q4：分页一定比连续预留省吗？**
不一定。长度 129 的请求，连续方案若恰好预留 129 个位置只要 129 槽，分页却要两页 256 槽。前一组例子能省，是因为连续方案按最大长度 512 给每个请求预留。

**Q5：PagedAttention 和 Prefix Cache 是一回事吗？**
不是。前者管请求内部的 block 分配、寻址与回收，看空间利用率、碎片、可容纳请求数；后者管请求之间的共享前缀复用，看命中率、复用 token 数、prefill 节省。

## 动手验证：怎么知道自己做对了
这一节没有独立的 `test_` 函数，断言直接写在单元里。Q1：`kv_cache_bytes(4096, 32, 32, 128)` 等于 `kv_cache_bytes(2048, 32, 32, 128)` 的 2 倍，并打印 seq_len 取 1024、2048、4096 时的 GB 数和翻倍比例。Q2：三种结构的缓存比例字典等于 `{'MHA': 1.0, 'GQA': 0.25, 'MQA': 1/32}`。Q3：`latent_content_scores([1, -1, 2, 0], [2, 3], [[1, 0], [0, 1], [1, 1], [2, -1]])` 返回的 `restored_key` 是 `[2, 3, 5, 1]`，`transformed_query` 是 `[3, 1]`，`restore_then_score` 与 `transform_then_score` 都等于 9；把查询换成 `[2, 0, -1, 1]`、潜在向量换成 `[-1, 4]` 后，两种算法都等于 -11。Q4：`sum(lengths) == 1120`，`contiguous_reserved_tokens` 得 2560、`paged_reserved_tokens` 得 1280，利用率 0.875、空槽比例 0.125；长度 128、页 128 时分页恰好 128 槽；长度 129 时连续方案 129 槽、分页 256 槽，相差 127。实验单元格打印 seq_len 翻倍的对比、三种结构的缓存 GB 与压缩比例、两种顺序的分数、五请求的槽位与利用率。notebook 没有 benchmark，dump 里没有保存输出；槽位模型只是计数，不执行真实分配，也不测服务吞吐或延迟。
