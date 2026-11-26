# 01. RMSNorm 教程 — 小白讲解

> 对应 notebook：[`01_RMSNorm_Tutorial.ipynb`](./01_RMSNorm_Tutorial.ipynb)

## 一句话总结
RMSNorm（Root Mean Square Normalization，均方根归一化）是大模型里常用的一种归一化层：把每个 token（词元）的向量按"均方根"缩放到统一尺度，但不像 LayerNorm 那样先减均值。本节用三个 TODO 实现一个最小的 `RMSNorm` 模块，重点看三件事：沿哪一维做统计、`eps` 为什么不能省、FP16 等低精度输入下怎样保住数值。

## 为什么需要它：痛点是什么
Transformer 一深，向量在层与层之间被反复做矩阵乘法，幅度会越滚越大或越缩越小（尺度漂移）：后面的层拿到的输入一会儿特别大、一会儿特别小，训练就不稳定，所以要在子层入口做归一化。老办法 LayerNorm 每次都要算均值和方差两套统计，在大模型里是一笔额外开销。RMSNorm 的取舍是：不减均值，只用均方根把尺度管住，统计路径更短；许多 LLaMA 系模型都采用它。

生活类比：播放器的"响度均衡"。每首歌（每个 token 的向量）进混音台前，都按均方根电平把音量调到统一水平——这就是 RMSNorm。LayerNorm 相当于多一步：先把整首歌整体上下挪到以 0 为中心（减均值），再按起伏幅度（标准差）调音量。两种做法都让"音量"不再乱飘，但 RMSNorm 不去改动声音原本的基准线。

## 核心问题
- RMSNorm 和 LayerNorm 到底差在哪一步？少这一步意味着什么？
- 统计量沿哪一维算？为什么不能跨 token、跨 batch 混着算？
- 公式里的 $\epsilon$ 是干什么的？
- 输入是 FP16/BF16 时，为什么统计要搬到 FP32 里算，最后又转回去？

## 解决思路：一步步看懂

**第 1 步：先看归一化放在哪、做什么。** 一个 Transformer Block 里有注意力子层和 MLP 子层，外面套一条残差连接（把子层输入直接加到输出上），这条贯穿各层的主干叫残差流；归一化要解决的，就是残差流在多层变换后尺度漂移的问题。摆放方式有三种：Post-LayerNorm 放在"子层与残差合并之后"，深层训练更依赖稳定技巧；Pre-LayerNorm 和 Pre-RMSNorm 都放在子层输入之前，残差主干更直接——区别是前者仍要算均值与方差，后者只按均方根缩放、统计路径更短，但不保证输出零均值。现代解码器通常用 Pre 摆法。RMSNorm 只保留"尺度归一"，去掉了 LayerNorm 的"重新居中"。

**第 2 步：公式逐项拆开。** 对每个 token 的隐藏向量，RMSNorm 沿最后一维计算：

$$\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\qquad y=\frac{x}{\mathrm{RMS}(x)}\odot\gamma$$

- $x\in\mathbb{R}^d$：一个 token 的隐藏向量，$d$ 是隐藏维度（通道数），$x_i$ 是第 $i$ 个通道的值。
- $\frac{1}{d}\sum_{i=1}^{d}x_i^2$：均方——每个通道平方后再取平均；外面套根号就成了均方根（RMS）。
- $\epsilon$：一个极小的正数（本节默认 `1e-6`），防止零向量时除以 0。
- $x/\mathrm{RMS}(x)$：把向量整体缩放到"均方根等于 1"的尺度；$\odot$ 是逐元素相乘，$\gamma\in\mathbb{R}^d$ 是每个通道一个的可学习缩放系数（初值全 1），用来恢复逐通道的缩放能力。

批次维和序列维不参与统计：第几个样本、第几个 token，只看它自己的 $d$ 个通道。拿测试里的数字算一遍：$x=[3,4]$，均方 $=(3^2+4^2)/2=12.5$，$\mathrm{RMS}=\sqrt{12.5+10^{-6}}\approx3.536$，归一化后 $\approx[0.848,\,1.131]$（$\gamma$ 全 1，这就是输出）。换成 LayerNorm 会先减均值 3.5 得 $[-0.5,0.5]$，再除以标准差 0.5 得 $[-1,1]$——结果不同，所以 notebook 强调 RMSNorm 不是 LayerNorm 的数值近似，而是主动取消"重新居中"。再看 $x=[0,0]$：均方为 0，$\mathrm{rsqrt}(0+10^{-6})=1000$，$0\times1000=0$，输出有限；没有 $\epsilon$ 的话 $1/\sqrt{0}=\infty$，$0\times\infty$ 是 NaN。

**第 3 步：三个 TODO，三层契约。** 题目骨架把模块接口固定好，只留三处决定数值路径的空：

```python
self.weight = nn.Parameter(torch.ones(hidden_size))   # TODO 1（__init__）
x_fp32 = x.float()                                   # TODO 2（_norm）
mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
normalized = x_fp32 * torch.rsqrt(mean_square + self.eps)
weight = self.weight.to(dtype=x.dtype)               # TODO 3（forward）
normalized = self._norm(x)
output = (normalized * weight).to(dtype=x.dtype)
```

`nn.Parameter(torch.ones(hidden_size))` 把全 1 向量登记成可训练参数，形状 `[D]`：每个通道一个系数，全 1 表示一开始不改动归一化结果。`mean(dim=-1, keepdim=True)` 沿最后一维求平均，`keepdim=True` 把这根轴留成长度 1，结果形状 `[..., 1]`，才能和 `[..., D]` 的 `x_fp32` 广播（形状不同时自动对齐运算）；去掉它统计轴就被消掉，无法正确广播。`torch.rsqrt(a)` 是 $1/\sqrt{a}$，所以"x 乘以 rsqrt(均方 + eps)"就是 $x/\mathrm{RMS}(x)$，用乘倒数代替除法。`forward` 里先把缩放参数转成输入的数据类型，乘上去，再把输出转回输入的数据类型——形状和数据类型都与输入一致。骨架里另有两处输入检查（`hidden_size <= 0`、`x.ndim < 1` 时抛 `ValueError`），没有对应的测试。

**第 4 步：为什么统计要搬到 FP32。** FP16/BF16（16 位浮点数）省显存、矩阵乘更快，但能表示的范围窄、有效位数少：较大的激活一平方就可能溢出，较小的激活也会损失统计精度。所以 `_norm` 一进来就 `x.float()`，平方、求均值、`rsqrt` 全在 FP32 里做。notebook 给出的路径是：输入与模型主状态保持模型 dtype；平方、均值、`rsqrt` 用 FP32；归一化与通道缩放在 FP32 或实现规定的安全路径上；输出恢复输入数据类型。它只保证这一层的算子契约，深层网络稳不稳定，还要在固定模型、数据和训练预算下比较。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 归一化（normalization） | 把向量缩放到统一尺度，防止数值在层间越滚越大或越小 |
| 尺度漂移 | 多层变换后向量幅度不断变化的现象 |
| 残差流（residual stream） | 贯穿各层、把每个子层输出直接加上去的主干通路 |
| LayerNorm | 减均值再除标准差：重新居中 + 尺度归一 |
| RMSNorm | 只按均方根缩放、不减均值的归一化 |
| 均方 / 均方根（RMS） | 各分量平方的平均值 / 它的开方 |
| Pre / Post 归一化 | 归一化放在子层输入之前 / 子层与残差合并之后 |
| γ（`weight`） | 每通道一个的可学习缩放系数，形状 `[D]`，初值全 1 |
| ε（`eps`） | 加在均方里的极小正数，防零向量除零 |
| FP32 / FP16 / BF16 | 32 位与 16 位浮点数；16 位省显存但范围窄、精度低 |
| 广播（broadcast） | 形状不同的张量运算时自动对齐，如 `[..., D]` 乘 `[..., 1]` |

## 小白常见疑问

**Q1：不减均值，输出不"居中"没关系吗？**
这正是 RMSNorm 有意放弃的约束。几何上，减均值等于删掉向量在"全 1 方向"上的分量，RMSNorm 不做这个投影。行不行不能凭"算子更短"就推断质量等价，要由模型结构、初始化和训练结果共同决定。

**Q2：为什么不能沿 batch 维或序列维算均方？**
统计一跨 token 或跨样本，一个 token 的输出就取决于同批里的其他内容，换一组样本结果就变。契约是每个 token 只看自己的 $d$ 个通道，所以必须 `dim=-1`。

**Q3：`weight` 用一个标量不行吗？**
不行。`[D]` 形状让每个通道有独立的缩放系数；换成标量只剩一个整体系数，逐通道的缩放能力就丢了——notebook 把它列为常见错误。

**Q4：既然统计在 FP32 里算，为什么不干脆整层输出 FP32？**
归一化层不改模型主状态的数据类型：后面的注意力、MLP 按约定的 dtype 读输入。只把容易出问题的平方、均值、`rsqrt` 提到 FP32，算完再转回去，接口不变、精度也保住了。

## 动手验证：怎么知道自己做对了

参考结果统一由 `_rmsnorm_reference` 给出：显式用 FP32 算均方和 `rsqrt`，乘 `weight.float()` 后再转回输入 dtype。`run_rmsnorm_tests` 依次跑三个测试，全过才打印 `✅ RMSNorm：参数、数值路径与 dtype/梯度契约测试通过`。
- `test_rmsnorm_parameter_contract`：`RMSNorm(hidden_size=4)` 的 `weight` 必须是 `nn.Parameter`，形状 `(4,)`，与 `torch.ones(4)` 完全相等。
- `test_rmsnorm_numeric_contract`：输入 `[[[3.0, 4.0], [0.0, 0.0]]]`、`eps=1e-6`，输出与参考 `torch.allclose(..., atol=1e-6)`（rtol 未指定，用默认值），且输出全部有限。
- `test_rmsnorm_dtype_and_gradient_contract`：FP16 输入 `torch.randn(2, 3, 4)`，模块也 `.to(dtype=torch.float16)`；输出形状、dtype 与输入一致，与参考比较 `rtol=1e-3, atol=1e-4`；`output.float().sum().backward()` 之后 `norm.weight.grad` 和 `x.grad` 都不为 None。
