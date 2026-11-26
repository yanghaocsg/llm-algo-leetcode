# 14. FlashAttention 显存模型 — 小白讲解

> 对应 notebook：[`14_FlashAttention_Memory_Model.ipynb`](./14_FlashAttention_Memory_Model.ipynb)

## 一句话总结
本节用三个计算器把 FlashAttention 的动机和代价算清楚：`attention_score_bytes` 显示单头、单样本、FP32 的 $N\times N$ 得分矩阵在 $N=2048$ 时占 16 MiB、$N=4096$ 时占 64 MiB（4 倍），批量 2、头数 4 时再乘 8；`online_attention_scalar` 证明块宽 1、2、3 逐块合并都能得到与完整 Softmax 相同的 3.5，分数整体加 1000 也不溢出；`flashattention_working_set_bytes` 算出块宽 128、头维 128 时片上工作集是 225 KiB，并用 `num_score_tiles(4096, 128) == 1024` 说明块组合数。

## 为什么需要它：痛点是什么
初学者知道「FlashAttention 省显存」，但经常说不清省的是哪一块：是省了 Q/K/V？还是省了输出？为什么序列长度翻倍，得分矩阵会变 4 倍？既然不保存完整矩阵，凭什么还能算出一样的 Softmax？另一个常见误解是把「完整矩阵与单块的大小比」当成「显存流量减少倍数」甚至「加速比」——notebook 专门用 `score_materialization_ratio` 澄清这一点。

生活类比：老师阅卷。一次性把所有 $N\times N$ 张答题卡摊在桌上，桌面很快不够，而且收起来再摊开很费事（物化 + 读写）。FlashAttention 是一摞一摞地看：桌上只放当前一叠，手里记三个数——目前为止的最高分 $m$、按最高分折算的指数和 $l$、以及加权总分 $u$。看到更高的分，就把之前记的 $l$ 和 $u$ 按新旧最高分的差缩小一次，再并入新一叠。最后 $u/l$ 就是加权平均。

## 核心问题
- 完整得分矩阵有多大，为什么序列翻倍是 4 倍而不是 2 倍？
- 不保存完整矩阵，怎样得到与普通 Softmax 相同的输出？
- 在线 Softmax 的 $m$、$l$、$u$ 各是什么，什么时候要 rescale？
- 块宽变大，片上工作集和块组合数各自怎么变？
- 225 KiB 的工作集、1024 倍的存储比，分别能说明什么、不能说明什么？

## 解决思路：一步步看懂

**第 1 步：先量「完全物化」的代价（Q1）。** 对长度为 $N$ 的自注意力，Q 与 K 相乘得到 $N\times N$ 个得分。`attention_score_bytes(seq_len, batch_size=1, num_heads=1, dtype_bytes=4)` 的公式就是：

```python
return batch_size * num_heads * seq_len * seq_len * dtype_bytes
```

四个参数都必须是正整数。代入 $N=2048$、四字节：$2048\times2048\times4 = 16{,}777{,}216$ B，正好 $16\times1024^2$，即 16 MiB。$N=4096$ 时是 $4096^2\times4 = 67{,}108{,}864$ B = 64 MiB，是前者的 4 倍：

```python
assert small == 16 * 1024**2
assert large == 64 * 1024**2 == 4 * small
assert attention_score_bytes(seq_len=4096, batch_size=2, num_heads=4) == 8 * large
```

最后一条说明批量 2、头数 4 会在这个基础上再乘 8，达到 512 MiB。为什么是 4 倍而不是 2 倍？因为矩阵的两个维度都随 $N$ 增长，元素数是 $N^2$。

如果完整写出一次、再读回一次，逻辑读写量是矩阵大小的两倍：16 MiB 对应 32 MiB 往返，64 MiB 对应 128 MiB。但 notebook 立刻划清边界：这里只统计一份得分张量，不含 Q/K/V、输出、Softmax 中间量；而且逻辑字节不等于实际 HBM 流量，缓存和实现方式会改变物理搬运。「矩阵越大越值得检查访存」不等于「一定受带宽限制」，还要结合运算量和执行测量。

**第 2 步：在线 Softmax 的递推（Q2）。** 普通 Softmax 要先看到整行所有得分才能算最大值和分母。分块之后，只能一块一块地看，所以每行维护三个状态：

- $m$：目前见过的最大得分；
- $l$：以 $m$ 为基准的指数和 $\sum e^{s_i-m}$；
- $u$：以 $m$ 为基准的指数加权值之和 $\sum e^{s_i-m}v_i$。

来了一块新得分，先算新的最大值 $m' = \max(m, \max(\text{block}))$。旧状态是按 $m$ 折算的，新状态要按 $m'$ 折算，缩放因子是 $e^{m-m'}$：

$$l_{\text{new}} = l\cdot e^{m-m'} + \sum_j e^{s_j-m'},\qquad u_{\text{new}} = u\cdot e^{m-m'} + \sum_j e^{s_j-m'}v_j$$

最后输出 $u/l$。`online_attention_scalar` 就是这段递推的标量版：

```python
updated_maximum = max(maximum, max(block_scores))
rescale = math.exp(maximum - updated_maximum)
weights = [math.exp(score - updated_maximum) for score in block_scores]
denominator = denominator * rescale + sum(weights)
numerator = numerator * rescale + sum(w * value for w, value in zip(weights, block_values))
maximum = updated_maximum
```

初始值是 $m=-\infty$、$l=0$、$u=0$，第一块到来时 $e^{-\infty-m'}=0$，旧状态自然清零。

拿 notebook 的例子手算：`scores = [0.0, 0.0, log(2)]`、`values = [1.0, 3.0, 5.0]`。以最大值 $\log 2$ 为基准，权重是 $[e^{-\log2}, e^{-\log2}, e^0] = [0.5, 0.5, 1]$，比例 $[1,1,2]$，输出 $(1\times1 + 1\times3 + 2\times5)/4 = 14/4 = 3.5$。

按块宽 2 走一遍：前两个得分 $[0,0]$ 得到 $m=0$、$l=2$、$u=1+3=4$。第三块得分 $\log2$ 更大，$m'=\log2$，`rescale` $=e^{0-\log2}=0.5$；旧状态缩小一半得 $l=1$、$u=2$，再加上新块 $l=1+1=2$、$u=2+5=7$，输出 $7/2=3.5$。块宽 1 和 3 的结果也一样，断言用 `math.isclose(actual, reference, abs_tol=1e-12)` 卡到 $10^{-12}$。

数值稳定性的验证是把三个得分都加 1000：`shifted = online_attention_scalar([s + 1000 for s in scores], values, block_size)`，结果仍是 3.5。因为 Softmax 对整体平移不变，而在线算法始终减去当前最大值，指数不会爆到 $e^{1000}$。

这个演示有明确边界：值是标量，不是向量；真实的注意力每行要维护一个向量累加器，还要处理因果掩码、dropout、浮点误差。notebook 的 markdown 把「块宽 1、2、3 结果一致」当作机制验证，不是完整实现的证明。

**第 3 步：块级工作集（Q3）。** `flashattention_working_set_bytes(tile_size, head_dim, dtype_bytes=2)` 把一块的片上状态拆成四项：

```python
qkv_bytes = 3 * tile_size * head_dim * dtype_bytes   # Q/K/V 各一份
score_bytes = tile_size * tile_size * 4              # 局部得分，FP32
output_bytes = tile_size * head_dim * 4              # 输出累加器，FP32
statistics_bytes = 2 * tile_size * 4                 # 每行的 m 和 l
```

代入 $t=128$、$d=128$、Q/K/V 每元素 2 字节：

- Q/K/V：$3\times128\times128\times2 = 98{,}304$ B = 96 KiB；
- 局部得分：$128\times128\times4 = 65{,}536$ B = 64 KiB；
- 输出累加器：$128\times128\times4 = 65{,}536$ B = 64 KiB；
- 行统计量：$2\times128\times4 = 1024$ B = 1 KiB。

合计 $96+64+64+1 = 225$ KiB，断言 `flashattention_working_set_bytes(tile_size=128, head_dim=128) == 225 * 1024`。块宽变化时各项的增长方式不同：

| 对象 | 载荷 | 块宽增大后 |
|---|---|---|
| Q/K/V 块 | $3td$ 个 2 字节元素 | 线性增长 |
| 局部得分 | $t^2$ 个 4 字节元素 | 平方增长 |
| 输出累加器 | $td$ 个 4 字节元素 | 线性增长 |
| 行统计量 | $2t$ 个 4 字节元素 | 线性增长 |

所以 $t=64$ 时总量约 96.5 KiB，$t=256$ 时是 578 KiB——块宽翻倍，工作集涨了约 2.6 倍，主要来自平方增长的局部得分（64 KiB → 256 KiB）。notebook 让读者自己跑 64/128/256 三档并打印，正是为了看到这个非线性。

块组合数用 `num_score_tiles(seq_len, tile_size) = num_1d_tiles(...) ** 2` 算，非因果情况下是 $\lceil N/t\rceil^2$。$N=4096$、$t=128$ 时是 $32^2 = 1024$；$t=64$ 时 4096；$t=256$ 时 256。块变少不等于内核启动次数变少：一个内核可以遍历多个块，`num_score_tiles` 的文档字符串特意写「不是内核启动次数」。尾块还要处理：`num_1d_tiles(130, 64) == 3`、`num_score_tiles(130, 64) == 9`，多出来的块要掩码，不能因为「按完整块预留空间」就跳过。

`score_materialization_ratio(4096, 128)` 返回 $4096^2/128^2 = 1024$，即完整矩阵是一个块的 1024 倍。这个数只比较存储规模，notebook 明确说它**不是**显存流量减少倍数，也不是加速比。真实实现还要面对 K/V 的重复读取、缓冲生命周期、寄存器用量和同步成本；而且长期 KV 缓存仍然要单独保存，FlashAttention 省掉的是临时的 $N\times N$ 矩阵，不是推理时那块随序列增长的 KV cache。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 得分矩阵 | $QK^\top$ 得到的 $N\times N$ 个相似度分数 |
| 物化（materialize） | 为中间结果分配存储并真正写进去 |
| 分块（tiling） | 把 Q/K/V 切成小块，每次只算一小块得分 |
| 在线 Softmax | 逐块更新最大值、指数和、加权和，不保存整行 |
| 行统计量 $m$ / $l$ | 当前最大得分 / 以它为基准的指数和 |
| 累加器 $u$ | 以当前最大值为基准的指数加权值之和 |
| rescale | 最大值变大时，把旧状态按 $e^{m-m'}$ 缩小 |
| 工作集（working set） | 一次分块计算需要在片上同时放下的数据量 |
| 块组合 | $\lceil N/t\rceil^2$ 个小块，不等于内核启动次数 |
| 尾块 | 序列长度不是块宽整数倍时多出来的、需要掩码的块 |
| 物化比 | 完整得分矩阵 ÷ 单块大小，只比较存储规模 |
| 头维（head_dim） | 每个注意力头的向量宽度，本例 128 |

## 小白常见疑问
**Q1：序列从 2048 到 4096 只翻了一倍，得分矩阵为什么是 4 倍？**
因为得分矩阵的大小是 $N\times N$，两个维度都翻倍，元素数就是 4 倍。`attention_score_bytes` 里 `seq_len * seq_len` 连续乘了两次；16 MiB → 64 MiB 正是 $2^2=4$。

**Q2：在线 Softmax 里的 `rescale` 到底在干什么？**
它把「按旧最大值折算」的 $l$ 和 $u$ 换算到「按新最大值折算」。旧状态每一项都含 $e^{-m}$，新状态要含 $e^{-m'}$，比值就是 $e^{m-m'}$。块宽 2 的例子中 $m=0$、$m'=\log2$，`rescale=0.5`，所以旧的 $l=2$ 变 1、$u=4$ 变 2，再并入新块。不做这一步，不同块的指数就不在同一尺度上，加起来是错的。

**Q3：块宽越大，块数越少，是不是一定越快？**
不一定。块宽从 128 到 256，块组合从 1024 降到 256，但工作集从 225 KiB 涨到 578 KiB，局部得分从 64 KiB 涨到 256 KiB。片上放不下就要降并行、增加溢出或同步；notebook 只算载荷，不给速度结论。块宽的选择要在「块数少」和「工作集小」之间权衡。

**Q4：225 KiB 是某个 GPU 内核真实的共享内存用量吗？**
不是。它是 Q/K/V、得分、输出累加器、行统计量四类载荷的总和，用来比较算法工作集。notebook 特意说明这个总和「不是某个内核的实际共享内存分配」；共享内存由线程块显式使用，寄存器保存线程临时值，两者不能当成一个可任意互换的 SRAM 预算。

**Q5：1024 倍的物化比，能不能直接说 FlashAttention 省了 1024 倍显存流量？**
不能。`score_materialization_ratio` 只比较「完整矩阵」和「一个块」的存储大小。真实流量还取决于 K/V 被重复读了多少次、块循环怎么排、缓存命中如何；notebook 的文档字符串明确写「不计算流量或加速比」。同理，也不能把它当加速比。

## 动手验证：怎么知道自己做对了
没有测试入口，也没有保存任何执行输出；检查是单元末尾的 `assert`，失败抛 `AssertionError`。

- Q1 断言 `small == 16 * 1024**2`、`large == 64 * 1024**2 == 4 * small`、`attention_score_bytes(4096, batch_size=2, num_heads=4) == 8 * large`；循环会打印 $N=2048$ 和 4096 的得分大小与「一次写出再读回」的两倍值。
- Q2 断言完整 Softmax 的参考值是 3.5；块宽 1、2、3 的 `online_attention_scalar` 结果都与参考值 `math.isclose(..., abs_tol=1e-12)`，整体加 1000 后也一样；循环打印三档块宽的加权输出。非法输入（空列表、长度不等、块宽非正、非有限值）抛 `ValueError`。
- Q3 断言 `flashattention_working_set_bytes(128, 128) == 225 * 1024`、`num_1d_tiles(130, 64) == 3`、`num_score_tiles(130, 64) == 9`、`score_materialization_ratio(4096, 128) == 1024`；循环打印块宽 64/128/256 的 KiB 数与块组合数。

想验证对增长速度的理解，可以把 `tile_size` 从 128 改成 256，手算 Q/K/V 192 KiB、得分 256 KiB、输出 128 KiB、统计 2 KiB，合计 578 KiB，再和函数输出对照。notebook 没有保存结果，所有数值以你自己跑出来的为准。
