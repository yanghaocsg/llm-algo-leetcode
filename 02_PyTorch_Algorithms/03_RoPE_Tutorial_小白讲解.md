# 03. RoPE 旋转位置编码 — 小白讲解

> 对应 notebook：[`03_RoPE_Tutorial.ipynb`](./03_RoPE_Tutorial.ipynb)

## 一句话总结
本节用三个 TODO 实现 RoPE：先把每个通道对的逆频率与位置做外积，生成 `[S, D/2]` 的单位复数旋转因子；再把 Q/K 最后一维的相邻两个数看成复平面上的一个点，乘以该因子完成旋转，最后还原成原 dtype 的实数张量。测试从频率表形状、位置 0 不旋转、旋转前后模长不变、FP16 输入输出 dtype 一致、奇数 `head_dim` 被拒绝五个角度确认实现正确。

## 为什么需要它：痛点是什么
注意力计算 $QK^\top$ 只比较向量内容，本身不知道谁在前、谁在后：把同一句话的 token 打乱，只要 Q/K 内容不变，点积结果就一样。最直觉的补法是"给每个位置加一个位置向量"，但这会改变 Q/K 的数值尺度，而且模型见到训练长度以外的位置编号时没有对应的向量。

初学者常见的三个卡点是：为什么"旋转"能表达位置，却不改变向量的长度？为什么频率表是 `[S, D/2]` 的复数，而不是 `[S, D]` 的实数？把 `theta` 从 10000 改大，是不是就能免费支持更长上下文？

生活类比：给每个 token 发一块手表，位置 $m$ 就是"走了 $m$ 秒"。第 $i$ 个通道对是一根转速固定的指针，转速由频率 $\omega_i$ 决定。比较两个 token 时，不看指针的绝对读数，而看两根指针的夹角——夹角只由它们出发的时间差 $m-n$ 决定。旋转不改变指针长度，所以 Q/K 的模长不变；同一位置的 Q 和 K 使用同一套旋转因子，相乘后内积里只剩下相对相位。

## 核心问题
- 位置信息怎样写进 Q/K，而不是额外加一个位置向量？
- 逆频率 $\theta^{-2i/d}$ 里的 $i$ 为什么取 $0,2,4,\dots$？高频和低频分别负责什么？
- 频率表为什么是 `[S, D/2]` 的复数，它和 `[S, D]` 的实数旋转是什么关系？
- 为什么旋转后每个通道对的模长不变，而 Q/K 内积只依赖相对位置？
- 调 `theta`、位置插值、频率缩放各自改变什么？为什么都不能自动保证训练长度之外的质量？

## 解决思路：一步步看懂

**第 1 步：把最后一维两两分组，看成二维旋转平面。** 设 `head_dim = d`，第 $i$ 个通道对由第 $2i$ 和第 $2i+1$ 个分量组成，$i=0,\dots,d/2-1$。位置 $m$ 在这对分量上转过的角度是

$$\phi_{m,i}=m\cdot\omega_i,\qquad \omega_i=\theta^{-2i/d}.$$

其中 $\theta$ 是基频（代码默认 `theta=10000.0`），$\omega_i$ 是第 $i$ 对的角速度（逆频率），$m$ 是 token 在序列里的位置。写成复数就是乘以单位模复数 $e^{j\phi_{m,i}}$：

$$(x_{2i},x_{2i+1})\mapsto \mathrm{Re}\big((x_{2i}+j x_{2i+1})e^{j m\omega_i}\big).$$

因为因子模长为 1，旋转只改方向不改长度，所以后面测试能用 `torch.linalg.vector_norm` 检查模长守恒。

**第 2 步：TODO 1，构造 `precompute_freqs_cis`。** 函数签名是 `precompute_freqs_cis(dim, end, theta=10000.0)`，骨架已做参数校验：`dim` 必须为正偶数、`end` 和 `theta` 必须为正，否则抛 `ValueError`。参考答案四行：

```python
pair_indices = torch.arange(0, dim, 2, dtype=torch.float32)  # 0,2,4,...,dim-2
inv_freq = 1.0 / (theta ** (pair_indices / dim))             # theta^(-2i/d)
positions = torch.arange(end, dtype=torch.float32)           # 0..end-1
angles = torch.outer(positions, inv_freq)                    # [end, dim/2]
return torch.polar(torch.ones_like(angles), angles)          # e^{j*angle}
```

`pair_indices` 只取偶数，是因为一个通道对共享一个频率；`pair_indices / dim` 得到 $2i/d$。`torch.outer` 把"第几个位置"和"第几对通道"组合成二维角度表，形状 `[end, dim/2]`；`torch.polar(r, phi)` 用极坐标构造复数，半径全为 1，得到 `cos(angle) + j·sin(angle)`。

拿测试里的 `dim=8, end=5` 手算：`pair_indices = [0,2,4,6]`，指数为 `[0, 0.25, 0.5, 0.75]`，所以逆频率是 `[1, 0.1, 0.01, 0.001]`（因为 $10000^{0.25}=10$）。对应波长 $2\pi/\omega$ 约为 6.3、62.8、628、6283 个 token：第 0 对转得最快、感知短距离；最后一对转得最慢、感知长距离。位置 0 的角度全为 0，`torch.polar(1, 0)` 就是 `1+0j`，这正是测试断言 `freqs[0] == ones` 的原因。

容易写错的地方：用 `torch.arange(dim)` 而不是步长 2，频率表会变成 `[end, dim]`，与 `reshape_for_broadcast` 要求的 `(S, D/2)` 对不上；把指数写成 `pair_indices / (dim / 2)`，频率范围会整体变一档；忘记 `torch.float32`，整数除法或 dtype 提升会带来意外结果。

**第 3 步：TODO 2，把成对实数维变成复数。** `apply_rotary_emb(xq, xk, freqs_cis)` 先检查 `xq` 与 `xk` 形状相同、都是四维、最后一维为偶数，否则 `ValueError`。Q/K 的形状约定是 `[B, S, H, D]`（batch、序列、注意力头、每头维度）。参考答案：

```python
xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
```

`reshape(..., -1, 2)` 把最后一维 `D` 拆成 `[D/2, 2]`，每个二元组是复数的实部和虚部；`view_as_complex` 再把最后一维"压掉"，得到 `[B, S, H, D/2]` 的复数张量。先 `.float()` 有两个作用：`view_as_complex` 只接受 FP32/FP64，直接传 FP16 会报错；旋转在 FP32 里算也更稳。这里没有动 `B/S/H` 三个轴，所以批次、序列、注意力头的语义不变。

**第 4 步：TODO 3，旋转并还原。** `reshape_for_broadcast(freqs_cis, xq_complex)` 把 `[S, D/2]` 变成 `[1, S, 1, D/2]`，让同一位置的因子同时作用到所有 batch 和所有头上，并且校验频率表长度与当前序列长度、通道对数一致。参考代码：

```python
freqs = reshape_for_broadcast(freqs_cis, xq_complex).to(xq_complex.dtype)
xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2).to(xq.dtype)
xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2).to(xk.dtype)
```

复数乘法就是二维旋转：`(a+bi)(cosφ+i sinφ)` 的实部是 $a\cos\phi-b\sin\phi$，虚部是 $a\sin\phi+b\cos\phi$。`view_as_real` 把 `[B,S,H,D/2]` 还原成 `[B,S,H,D/2,2]`，`flatten(-2)` 再把最后两维并回 `D`，输出与输入同形状；`.to(xq.dtype)` 把 FP32 的计算结果转回原来的 FP16/BF16。

**第 5 步：为什么内积只依赖相对位置。** 对某个通道对，Q 在位置 $m$、K 在位置 $n$，旋转后的内积是

$$\mathrm{Re}\big(q e^{jm\omega}\overline{k e^{jn\omega}}\big)=\mathrm{Re}\big(q\bar k\,e^{j(m-n)\omega}\big).$$

$m$ 和 $n$ 各自消失了，只剩差值 $m-n$。所以 RoPE 让注意力分数天然带相对距离信息，而且不需要额外的位置向量，也不改 V 和因果掩码决定的可见范围。这也是 notebook 表格里"RoPE 不决定哪些历史位置可见"的意思：它管的是相似度里的相位，不管访问权限。

**第 6 步：Step 3 的三种长度扩展，各自改什么。** 训练长度之外继续用原频率，会遇到训练时没充分覆盖的相位组合，所以"把 `max_position_embeddings` 调大"并不等于获得长上下文能力。notebook 列了三类做法：调基频 `theta` 改变所有通道对的转速，风险是短距离分辨率变化；位置插值把更长位置压回训练范围，风险是相邻位置被挤近；频率相关缩放对不同频率分别缩放，配置与训练方式强相关。三者都必须用训练长度内回归、目标长度任务质量、失败样例来验证，不能只看配置改没改。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| RoPE | 把位置写成 Q/K 成对维度上的旋转，让内积携带相对距离 |
| 通道对（pair） | 最后一维相邻的两个数，组成一个二维旋转平面 |
| 逆频率 $\omega_i$ | 第 $i$ 对通道每前进一个位置转过的角度，$\theta^{-2i/d}$ |
| 基频 `theta` | 逆频率的底数，默认 10000；越大低频转得越慢 |
| 频率表 `freqs_cis` | `[S, D/2]` 的单位复数，存每个位置每对通道的旋转因子 |
| `torch.polar` | 用"半径 + 角度"构造复数，这里半径恒为 1 |
| `view_as_complex` | 把最后一维长度为 2 的实数张量看成复数张量 |
| `view_as_real` | 复数的逆操作，拆回实部、虚部两个通道 |
| 广播 `[1,S,1,D/2]` | 让同一位置的旋转同时作用于所有 batch 和注意力头 |
| 模长守恒 | 乘单位模复数只转方向，向量长度不变 |
| 相对相位 | 旋转后内积里的 $e^{j(m-n)\omega}$，只与位置差有关 |
| 长度外推 | 让训练短、推理长的模型仍能处理位置，需要单独评测 |

## 小白常见疑问

**Q1：逆频率的指数为什么除以 `dim`，而不是 `dim/2`？**
`pair_indices` 取的是 `0,2,...,dim-2`，除以 `dim` 后指数落在 $[0, 1-2/d]$，逆频率从 1 平滑降到约 $1/\theta$，波长覆盖 $2\pi$ 到约 $2\pi\theta$。若除以 `dim/2`，指数范围会变成 $[0, 2-4/d]$，最高一对的频率变成约 $1/\theta^2$，等于把同样的频率段压缩到更窄的范围，和 LLaMA 等实现的约定不一致。代码、测试和 `theta` 的默认值都按前一种写法对齐。

**Q2：为什么先 `.float()` 再转复数，最后又 `.to(xq.dtype)`？**
`torch.view_as_complex` 只支持 FP32/FP64；而且旋转涉及三角函数和复数乘法，在 FP16 里做会放大舍入误差。所以中间用 FP32 计算，输出再转回输入 dtype。测试 `test_rope_position_and_dtype_contract` 专门用 `torch.float16` 输入，断言 `rotated.dtype == torch.float16`，就是防止实现忘记转回去。

**Q3：位置 0 的 token 真的完全不变吗？**
是。位置 0 的角度是 $0\cdot\omega_i=0$，因子是 $e^{j0}=1$，乘以 1 不改变任何分量。测试断言 `freqs[0]` 全为 1，又断言 `rotated[:, 0]` 与输入 `repeated[:, 0]` 全等（`allclose`），两层检查都指向这一点。位置 1 就必须变化，测试用 `not allclose` 反向确认旋转确实生效。

**Q4：既然旋转不改变模长，测试为什么用 `vector_norm` 而不是直接比数值？**
因为旋转后的每个分量都变了，逐元素比较一定失败；能保持不变的是每个通道对的长度，以及整个 `head_dim` 向量的 L2 范数。测试对最后一维求 `torch.linalg.vector_norm(..., dim=-1)`，容差 `atol=1e-5`，检查 Q/K 旋转前后范数一致。这同时验证了"因子是单位模复数"和"没有漏乘/多乘"。

**Q5：把 `theta` 改大就能支持更长上下文吗？**
不能直接这么推。`theta` 变大会让所有通道对转得更慢，相当于拉长波长，但短距离的分辨率也会变化；notebook 的 Step 3 把"调基频"列为一种需要单独验证的方案，要求补训练长度内回归和目标长度质量。同理，位置插值和频率缩放也不是改个配置就自动有效，必须固定模型和任务做评测。

## 动手验证：怎么知道自己做对了
完成三个 TODO 后运行 `run_rope_tests()`，它会依次调用三个测试，全部通过才打印 `✅ RoPE：频率表、旋转性质与输入契约测试通过`。notebook 的 dump 中没有保存这一格的输出，所以以断言为准，不要假设已经跑过。

- `test_rope_frequency_table`：`precompute_freqs_cis(dim=8, end=5)` 的形状必须是 `(5, 4)`；第 0 行必须等于 4 个 1；`(dim=7, end=5)` 和 `(dim=8, end=0)` 两组非法配置必须抛 `ValueError`，否则测试主动 `raise AssertionError`。
- `test_rope_rotation_contract`：输入 `xq = torch.randn(2, 4, 3, 8)`、`xk` 同形，频率表 `dim=8, end=4`；输出形状必须与输入一致；`q_out`、`k_out` 与输入的最后一维 L2 范数必须 `allclose(atol=1e-5)`。
- `test_rope_position_and_dtype_contract`：输入是 `torch.ones(1, 3, 1, 4, dtype=torch.float16)`，频率表 `dim=4, end=3`；输出 dtype 必须仍是 `float16`；位置 0 必须与输入全等，位置 1 必须不全等；最后传最后一维为 3 的奇数维张量，必须抛 `ValueError`。

自查时还可以手算小例子：`dim=4` 时逆频率是 `[1, 0.01]`，位置 1 的第一对角度为 1 弧度，输入 `(1,1)` 应旋转为约 `(-0.301, 1.382)`，长度仍是 $\sqrt2$；第二对角度 0.01，几乎不变但不应被 `allclose` 判为相等。
