# 04. 多头注意力（MHA / GQA / MQA） — 小白讲解

> 对应 notebook：[`04_Attention_MHA_GQA.ipynb`](./04_Attention_MHA_GQA.ipynb)

## 一句话总结
本节用一个 `GroupedQueryAttention` 类同时实现 MHA、GQA、MQA：Q 始终投影出 `num_heads` 个头，K/V 只投影出 `num_kv_heads` 个头，计算注意力前用 `repeat_kv` 临时复制到查询头数，而 KV Cache 始终保存未扩展的 `H_kv` 个头。四个 TODO 分别完成多头重排、缓存拼接、缩放点积注意力与输出合并。

## 为什么需要它：痛点是什么
初学多头注意力，卡住的地方通常不是公式，而是形状和顺序：

- `[B, S, H×D]` 和 `[B, H, S, D]` 什么时候该转置、转哪一维？`reshape` 之后为什么还要 `transpose(1, 2)`？
- GQA 说"省 KV 显存"，可如果先把 K/V 复制成 `H` 个头再存进 cache，省下来的东西又原样还回去了；
- 掩码该加在 softmax 前还是后？为什么用 `-inf` 而不是 0？
- 单步解码只喂一个 token，为什么结果能和"把整段序列重新前向一遍"对上？

生活类比：把注意力想成图书馆。Q 是读者提出的问题，K 是每本书脊上的索引标签，V 是书里的内容。MHA 是每个读者都配一整套索引卡；GQA 是几排读者共用一套索引卡；MQA 是全馆只有一套。索引卡（K/V）越少，占的柜子（显存）越少，但不同读者之间的区分度也会下降——这正是 GQA 要在"省显存"和"表达能力"之间做的取舍。

## 核心问题
- Q/K/V 从 `[B, S, hidden]` 变成 `[B, H, S, D]` 的完整变换是什么？`head_dim` 由谁决定？
- GQA/MQA 只在哪些投影上减少了输出宽度？为什么要求 `num_heads % num_kv_heads == 0`？
- `repeat_kv` 为什么写成 `unsqueeze → expand → reshape`？`n_rep == 1` 时为什么直接返回？
- KV Cache 应该存扩展前还是扩展后的 K/V？预填充 + 单步解码怎样与完整前向对齐？
- 因果掩码为什么必须加在 softmax 之前，并且必须是负无穷？

## 解决思路：一步步看懂

**第 1 步：先统一符号。** 设 `B` 为批量、`S_q` 为当前查询词元数、`S_k` 为可读的键词元数（带缓存时包含历史）、`H` 为查询头数、`H_kv` 为 KV 头数、`D` 为每头宽度。notebook 的公式是

$$\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\!\left(\frac{QK^{T}}{\sqrt{D}}+M\right)V$$

`Q`、`K`、`V` 都是 `[B, H, S, D]`（K/V 复制后）；$QK^T$ 得到 `[B, H, S_q, S_k]` 的分数，除以 $\sqrt{D}$ 控制数值量级；$M$ 是加性掩码，可见位置为 0、不可见位置为 $-\infty$；Softmax 在最后一维（键方向）归一化；再乘 `V` 得到 `[B, H, S_q, D]`；最后合并头得到 `[B, S_q, H×D]`。

**第 2 步：`__init__` 定下契约。** 类里先做三层校验：`hidden_dim`、`num_heads` 为正；`hidden_dim % num_heads == 0`（否则每头宽度不是整数）；`num_kv_heads` 为正且 `num_heads % num_kv_heads == 0`（否则查询头没法平均分组）。接着：

```python
self.num_queries_per_kv = self.num_heads // self.num_kv_heads  # 每组几个查询头
self.head_dim = hidden_dim // num_heads                        # 每头宽度
self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
```

关键点：`q_proj` 输出宽度始终是 `H×D`（等于 `hidden_dim`），而 `k_proj`/`v_proj` 只有 `H_kv×D`。MQA 时 `H_kv=1`，K/V 投影只剩一个头；省参数、省 cache 的根源在这两行，而不是后面复制的时候。

**第 3 步：TODO 1——拆多头。** 线性层输出的是 `[B, S, H×D]`，特征维按 `(head, dim)` 排列。参考答案：

```python
xq = xq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
xk = xk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
xv = xv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
```

`reshape` 把最后一维拆成 `[H, D]`，`transpose(1, 2)` 把头维换到序列维前面，得到 `[B, H, S, D]` 和 `[B, H_kv, S, D]`。转置后张量在内存里不再连续，但 `matmul` 不要求连续，所以能直接算。容易写错的是把 `num_heads` 和 `num_kv_heads` 用混：K/V 如果用 `num_heads` 去 reshape，形状对不上会立刻报错；更隐蔽的是先扩到 `H` 再存 cache，测试里的 `cache[0].shape[1] == 2` 就会失败。

**第 4 步：TODO 2——沿序列维拼缓存。** `kv_cache` 是 `(k_cache, v_cache)`，形状 `[B, H_kv, cached_len, D]`。参考答案只有两行：

```python
xk = torch.cat([k_cache, xk], dim=2)   # 历史在前，当前步在后
xv = torch.cat([v_cache, xv], dim=2)
```

为什么是 `dim=2`？因为第 0 维是 batch、第 1 维是头、第 2 维才是序列。拼完总长度变成 `cached_len + S_q`，之后 `new_kv_cache = (xk, xv)` 保存的是**未扩展**的 K/V。notebook 的注释写得很直白："只拼接未扩展的 KV，避免把 GQA 的缓存放大到 MHA 大小。" 如果这里先把 `xk` 复制成 `H` 个头再拼，cache 就退化成 MHA 的大小，GQA 的显存优势消失。

**第 5 步：`repeat_kv`——只在计算时临时扩展。**

```python
if n_rep == 1:
    return hidden_states                      # MHA 不需要复制
hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

`unsqueeze(2)` 在头维后面插一个长度 1 的维，`expand` 把它扩成 `n_rep`，形状 `[B, H_kv, n_rep, S, D]`；`reshape` 再把 `H_kv × n_rep` 合并回头维。展开后的头顺序是"第 0 个 KV 头重复 `n_rep` 次，再第 1 个 KV 头重复 `n_rep` 次……"，正好对应查询头按顺序分组。`expand` 本身不复制数据，直到 `reshape` 需要连续内存时才真正复制；因此扩展只发生在当前这次注意力计算里，生命周期很短。

**第 6 步：TODO 3——缩放、掩码、Softmax、聚合。**

```python
scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
if attention_mask is not None:
    scores = scores + attention_mask
probs = F.softmax(scores, dim=-1)
output = torch.matmul(probs, xv)
```

`xk.transpose(2, 3)` 把 `[B, H, S_k, D]` 变成 `[B, H, D, S_k]`，矩阵乘得到 `[B, H, S_q, S_k]`。掩码是**加性**的：可见位置加 0 不变，不可见位置加 $-\infty$，经过 `exp` 后权重恰好为 0。如果换成乘 0 的掩码，必须在 softmax 之前处理，而且 0 会参与归一化分母的竞争，语义容易写错；notebook 的 `causal_mask` 用的是 `masked_fill(..., float('-inf'))`。`F.softmax(..., dim=-1)` 沿键方向归一化，每一行概率和为 1。最后 `probs @ xv` 把 `[B,H,S_q,S_k]` 和 `[B,H,S_k,D]` 合成 `[B,H,S_q,D]`。

**第 7 步：TODO 4——合并头。** 各头的结果先转回 `[B, S, H, D]`，再把后两维压平成 `H×D`：

```python
output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
return self.o_proj(output), new_kv_cache
```

输出 `[B, S_q, hidden_dim]` 与输入主状态对齐，才能接残差。头合并的顺序必须和 `repeat_kv` 的分组顺序一致，否则第 0 个查询头会读到别的 KV 组。

**第 8 步：KV Cache 到底省了多少。** 对单层、单 token，普通 K/V 状态元素数与 `2 × H_kv × D` 成正比（K 和 V 各一份）。举个手算例子：`H=32`、`H_kv=8`、`D=128`、FP16，每个 token 每层是 `2 × 8 × 128 × 2 字节 = 4096 字节 = 4 KiB`；如果退回 MHA（`H_kv=32`），就是 `2 × 32 × 128 × 2 = 16384 字节 = 16 KiB`，正好 4 倍。序列越长、层数越多，这个差额越可观。注意本节只算了普通 K/V，notebook 的 Step 2 表格也提醒：MLA 这类方法改变的是缓存内容本身，不只是减少 KV 头数。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Q / K / V | 查询（我要找什么）、键（每个位置能被怎样匹配）、值（匹配上之后取走的内容） |
| 注意力头（head） | 把隐藏维切成若干份，每份独立做一次注意力，读不同的表示子空间 |
| `head_dim` | 每个头的宽度，等于 `hidden_dim // num_heads` |
| MHA | 每个查询头都有自己的一组 K/V，`H_kv = H` |
| GQA | 多个查询头共享一组 K/V，`1 < H_kv < H` |
| MQA | 所有查询头共享唯一一组 K/V，`H_kv = 1` |
| `repeat_kv` | 计算前把 KV 头复制 `n_rep` 份，临时对齐查询头数 |
| KV Cache | 把历史 K/V 留在显存里，解码时只算新 token，不再重算整段 |
| 加性掩码 | 可见位置加 0、不可见位置加 `-inf`，softmax 后自然变成 0 权重 |
| 缩放因子 $\sqrt{D}$ | 防止点积随头宽变大而数值爆炸，让 softmax 不至于饱和 |
| 预填充（prefill） | 一次性处理整段输入，建立初始 cache |
| 单步解码（decode） | 每次只喂 1 个 token，复用 cache 算出下一个位置 |

## 小白常见疑问

**Q1：为什么一定要除以 $\sqrt{D}$，除以 `D` 不行吗？**
点积是 `D` 个数相乘再相加，若每个分量独立、均值 0、方差 1，点积的方差约为 `D`，标准差是 $\sqrt{D}$。除以 $\sqrt{D}$ 把标准差拉回 1 左右，softmax 的输入不会随头宽增大而越来越极端。notebook 的参考代码明确写的是 `/ math.sqrt(self.head_dim)`，测试里的 `head_dim = 128 // 4 = 32`，除的就是 $\sqrt{32}$。

**Q2：掩码为什么用 `-inf`，用 0 或一个很大的负数不行吗？**
`causal_mask` 用 `float('-inf')` 填未来位置，`exp(-inf) = 0`，这些位置的概率严格为 0。如果用 0，它表示"加 0"，等于没有屏蔽；如果用 `-1e9` 这类有限大负数，虽然 `exp` 通常也会下溢成 0，但它依赖具体 dtype 的表示范围，语义上也不如 `-inf` 明确。另一个关键是位置：必须在 `F.softmax` **之前**加，softmax 之后再加就改变不了归一化分母了。

**Q3：既然计算时反正要 `repeat_kv`，为什么 cache 不直接存复制后的？**
因为 cache 是长期驻留的，复制后的张量会一直占着 `H` 个头的内存；而 `repeat_kv` 的结果只在当前这次前向里用一下。测试 `test_gqa_output_contract` 断言 `cache[0].shape[1] == 2`（`num_kv_heads=2`），`test_mqa_output_contract` 断言等于 1，就是在防止"提前扩展"这个错误。`test_kv_cache_decode_equivalence` 里 MHA 的 cache 形状是 `(2, 4, 5, 32)`，也能反推 cache 存的是 `H_kv` 个头、`prefill_len` 个位置、`head_dim` 宽。

**Q4：`repeat_kv` 里为什么用 `reshape` 而不是 `view`？**
`expand` 返回的是广播视图，内存不连续，`view` 在非连续张量上会报错，`reshape` 会在需要时自动复制一份连续内存。这也是为什么 `n_rep == 1` 要提前返回：MHA 不需要复制，直接返回原张量既省一次拷贝，也保持语义最简。

**Q5：这个实现没有位置编码，单步解码凭什么和完整前向完全一致？**
因为本节是纯内容注意力：每个查询位置的输出只取决于它自己的 Q 和所有可见位置的 K/V，与"这段序列是一次性算的还是分两次算的"无关。`test_kv_cache_decode_equivalence` 先用 4 个 token 建 cache，再用第 5 个 token 解码，把结果和"5 个 token 带因果掩码一次前向"的最后一个位置比较，容差 `atol=1e-5, rtol=1e-4`。真实模型还要加 RoPE，位置信息由 Q/K 的旋转带入，本节明确不包含。

## 动手验证：怎么知道自己做对了

`test_attention_mechanisms` 依次调用六个测试，全过才打印 `✅ Attention 机制测试通过：MHA/GQA/MQA、因果 mask、KV Cache 与参数契约均符合预期。`。测试夹具 `_build_attention_fixture` 固定 `torch.manual_seed(41)`，`batch_size=2, seq_len=16, hidden_dim=128, num_heads=4`，所以 `head_dim=32`。

- `test_mha_output_contract` / `test_gqa_output_contract` / `test_mqa_output_contract`：分别用 `num_kv_heads=4/2/1`，断言输出形状都是 `(2, 16, 128)`、全部有限，且 `cache[0].shape[1]` 分别等于 4、2、1。
- `test_kv_cache_decode_equivalence`：`prefill_len=5`，先用前 4 个 token 建 cache，再用第 5 个 token 和全 0 的 `[1,1,1,5]` 掩码解码；断言新 cache 形状 `(2, 4, 5, 32)`，且 `full_out[:, -1:]` 与 `decode_out` 在 `atol=1e-5, rtol=1e-4` 下 `allclose`。
- `test_causal_mask_blocks_future`：把第 2 到第 16 个 token 全部加 1000，断言第 1 个 token 的输出不变（同样是 `atol=1e-5, rtol=1e-4`）。
- `test_attention_config_contract`：`(130, 4, 2)`、`(128, 3, 1)`、`(128, 4, 3)` 三组都必须抛 `ValueError`，否则 `raise AssertionError`。

notebook 的 dump 里没有保存执行输出，所以以断言为准：六个测试都不抛异常、最后打印那行 ✅，就说明四个 TODO 的形状、缓存语义和因果性都对上了。如果你只看到 `请先完成 TODO 部分的代码！`，说明还有 `raise NotImplementedError` 没删掉。
