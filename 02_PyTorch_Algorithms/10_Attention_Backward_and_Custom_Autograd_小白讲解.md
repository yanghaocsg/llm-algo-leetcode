# 10. Attention 反向传播与自定义 Autograd — 小白讲解

> 对应 notebook：[`10_Attention_Backward_and_Custom_Autograd.ipynb`](./10_Attention_Backward_and_Custom_Autograd.ipynb)

## 一句话总结
本节把缩放点积 Attention 的反向拆成四段局部梯度：`dV = P^T dO`、`dP = dO V^T`、Softmax 反向后得到 `dS`、再由 `dS` 回传 `dQ` 和 `dK`。四个 TODO 分别实现这四段，最后装进一个 `torch.autograd.Function`，用自动求导逐段对照、用 `gradcheck` 校验组合后的完整 backward，并顺手算清保存 `P` 带来的 $O(N^2)$ 显存账本。

## 为什么需要它：痛点是什么
很多人能背出 Attention 前向 `softmax(QK^T/√d)V`，一到反向就只会调用 `loss.backward()`，说不出梯度为什么这样流。真正写自定义算子或读 FlashAttention 时，必须知道每个中间量的梯度从哪来、形状怎么对齐、哪些前向状态必须保存；否则很容易把转置写反、把 `scale` 漏在一条支路上，或者保存了本可以重算的张量。

生活类比：逆向物流。`dO` 是从损失端发回的「退货总单」，`P` 是前向时每个收货点分到了多少货。退给 Value 的货要按 `P` 的转置汇总回各个仓库；退给概率的货要经过 `V` 的转置；到了 Softmax 这个中转站，因为每一行概率之和必须为 1，退货单要先减去「行内加权平均」这个公共分量，剩下的才是真正该往前传的差异。最后到 Q/K 两个供应商，还要各自乘上缩放因子。

## 核心问题
- 输出梯度 `dO` 怎样分成 `dV` 和 `dP` 两条支路，再经过 Softmax 得到 `dS`，最后回到 `dQ` 和 `dK`？
- 为什么 `dV = P^T dO`、`dP = dO V^T`？各自的形状怎样对上？
- Softmax 反向为什么每行要减去一个加权和？为什么 `dS` 的每行和接近零？
- 缩放因子 $1/\sqrt{d}$ 为什么在 `dQ` 和 `dK` 两条支路上都要出现？
- `torch.autograd.Function` 的 `forward`/`backward` 契约是什么？`save_for_backward` 保存了什么？
- 保存完整概率矩阵 `P` 的显存代价怎么算？FlashAttention 省掉的到底是哪一项？

## 解决思路：一步步看懂

**第 1 步：前向依赖与形状。** 记 batch 为 $B$、序列长度为 $N$、每个头的特征维为 $d$。输入 $Q,K,V\in\mathbb R^{B\times N\times d}$，打分矩阵和概率矩阵 $S,P\in\mathbb R^{B\times N\times N}$，输出 $O\in\mathbb R^{B\times N\times d}$。前向链是：

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\mathrm{softmax}(S)\ \text{(按最后一维)},\qquad O=PV.
$$

`CustomAttention.forward` 先做五项契约检查：三者必须是三维张量；`q.shape[:2] == k.shape[:2] == v.shape[:2]`（batch 和序列长度一致）；三者 `head_dim` 相同；同一 device；同一 dtype。然后 `scale = 1.0 / math.sqrt(d_k)`，算 `scores = q @ k.transpose(-2, -1) * scale`、`p = F.softmax(scores, dim=-1)`、`out = p @ v`，最后 `ctx.save_for_backward(q, k, v, p)` 并把 `ctx.scale = scale` 存起来。

**第 2 步：反向总览。** 反向从 `dout`（即 $dO$）出发，依赖关系是 $O\leftarrow P\leftarrow S\leftarrow Q,K$，而 $O$ 还直接依赖 $V$。所以：

$$
dV=P^\top dO,\qquad dP=dO\,V^\top,
$$

$$
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_k P_{ik}dP_{ik}\right),
$$

$$
dQ=dS\,K\cdot\frac{1}{\sqrt d},\qquad dK=dS^\top Q\cdot\frac{1}{\sqrt d}.
$$

逐项看形状：`P^T` 是 $[B,N,N]$，乘 `dO` $[B,N,d]$ 得 `dV` $[B,N,d]$；`dO` 乘 `V^T` $[B,N,d]^T$ 得 `dP` $[B,N,N]$；`dS` 与 `K` 相乘得 $[B,N,d]$，正好是 `dQ`；`dS^T` 与 `Q` 相乘得 `dK`。四条公式里没有任何一个维度是「碰巧」对上的，写错转置通常会在形状检查或 `gradcheck` 里立刻暴露。

**第 3 步：Value 与概率两条支路（TODO 1、TODO 2）。** 这两个函数都来自同一个矩阵乘法 $O=PV$ 的求导：

```python
dv = torch.matmul(p.transpose(-2, -1), dout)   # [B,N,N]^T @ [B,N,d] -> [B,N,d]
dp = torch.matmul(dout, v.transpose(-2, -1))   # [B,N,d] @ [B,N,d]^T -> [B,N,N]
```

为什么 `dV` 用 `P` 的转置？把 $O_i=\sum_j P_{ij}V_j$ 展开看，某个 $V_j$ 收到的是所有查询位置 $i$ 按 $P_{ij}$ 加权后的 $dO_i$，所以要对 $i$ 求和，正好是 $P^\top dO$。同理，$P_{ij}$ 的梯度是 $dO_i$ 与 $V_j$ 的内积，写成矩阵就是 $dO V^\top$。`test_value_gradient` 和 `test_probability_gradient` 分别用 `(p @ v).backward(dout)` 的自动求导结果做对照，`atol=1e-8`，并且故意让 `p` 或 `v` 之一成为叶子变量，避免组合 backward 掩盖单段错误。

**第 4 步：穿过 Softmax（TODO 3）。** Softmax 是逐行的：$P_{ij}=e^{S_{ij}}/\sum_k e^{S_{ik}}$。它的雅可比矩阵不需要显式构造，对每一行有：

```python
weighted_sum = (dp * p).sum(dim=-1, keepdim=True)  # 每行一个标量，形状 [B,N,1]
ds = p * (dp - weighted_sum)                       # 广播回 [B,N,N]
```

为什么要减 `weighted_sum`？因为 Softmax 的输出每行加起来恒为 1，沿着「所有分量同时加同一个常数」的方向，输出不变，所以梯度在这个方向上的分量必须被扣掉。`(dp * p).sum(-1)` 正是 $dP$ 在概率分布 $P$ 下的期望，逐元素乘回 $P$ 就得到 $dS$。这个性质也给出一个免费的正确性检查：`dS` 的每一行和应该接近 0。`test_softmax_backward` 既和原生 `p.backward(dp)` 的结果比，又断言 `manual.sum(dim=-1)` 全零（`atol=1e-8`）。

**第 5 步：回到 Query 和 Key（TODO 4）。** 因为 $S=QK^\top/\sqrt d$，把 $dS$ 看成 $S$ 的梯度，对 $Q$ 求导得到 $dS\,K$，对 $K$ 求导得到 $dS^\top Q$，两者都要乘上同一个缩放因子：

```python
dq = torch.matmul(ds, k) * scale                  # [B,N,N] @ [B,N,d] -> [B,N,d]
dk = torch.matmul(ds.transpose(-2, -1), q) * scale
```

`scale` 不能只写在一支上：$S$ 的定义里 $Q$ 和 $K$ 是对称出现的一次乘积，对任一输入求导都会带出 $1/\sqrt d$。`test_query_key_gradients` 用 `ds` 形状 `[2,4,4]`、`q/k` 形状 `[2,4,6]`、`scale=6**-0.5`，把 `(q @ k^T * scale).backward(ds)` 的梯度与 `query_key_gradients` 对比，两个输入都要求 `atol=1e-8`。

**第 6 步：把它们装进自定义 Autograd。** `CustomAttention` 继承 `torch.autograd.Function`，`forward(ctx, q, k, v)` 返回一个输出，`backward(ctx, dout)` 必须按 `forward` 的输入顺序返回三个梯度 `(dq, dk, dv)`，不能多也不能少。backward 里先从 `ctx.saved_tensors` 取回 `q, k, v, p`，再检查 `dout.shape == q.shape`，然后按依赖顺序调用四个局部函数：

```python
dv = value_gradient(p, dout)
dp = probability_gradient(dout, v)
ds = softmax_backward(p, dp)
dq, dk = query_key_gradients(ds, q, k, scale)
return dq, dk, dv
```

注意 `dV` 必须先算、`dP` 也先算，`dS` 依赖 `dP`，`dQ/dK` 依赖 `dS`，顺序不能乱。`test_custom_attention_integration` 用 `q/k/v` 形状 `[1,3,4]` 先和显式 `softmax((q @ k^T)/sqrt(4)) @ v` 比输出（`atol=1e-8`），再用 `torch.autograd.gradcheck(CustomAttention.apply, (q,k,v), eps=1e-6, atol=1e-4)` 做数值梯度检查。`gradcheck` 会对每个输入元素做有限差分，逐个和解析梯度比，所以输入必须是 `float64`；它比单段测试更严格，因为任何一段写错都会让组合结果偏。

`test_attention_input_contract` 则验证前向契约：`q` 形状 `[2,3,4]`，`k=q[:, :2]` 序列长度只有 2，必须抛 `ValueError`。本节只实现无 mask、无 dropout、单头、Q/K/V 等长的教学版，不覆盖因果掩码和 GQA。

**第 7 步：保存状态的显存账本。** 反向需要前向的哪些量？`dV` 要 `P`，`dP` 要 `V`，`dS` 要 `P`，`dQ/dK` 要 `Q/K` 和 `scale`。所以 `save_for_backward(q, k, v, p)` 保存了四份张量。若每个元素占 $b$ 字节，教学显式路径的容量是：

$$
C_{saved}=b\left(3BNd+BN^2\right).
$$

其中 $3BNd$ 对应 $Q,K,V$，$BN^2$ 对应完整的概率矩阵 $P$。拿 $B=2$、$N=1024$、$d=64$、FP16（$b=2$）手算：$3BNd=3\times2\times1024\times64=393{,}216$，$BN^2=2\times1024^2=2{,}097{,}152$，合计 $2{,}490{,}368$ 个元素，乘 2 字节约 4.98 MB；其中 $P$ 占了约 84.2%。序列长度翻倍时 $BN^2$ 变 4 倍，这就是长上下文下激活显存吃紧的主因。FlashAttention 不去保存完整 $P$，而是分块计算、用 online softmax 保留必要统计量，改变的是「保存与搬运」的方式；真实收益仍要按 GPU、序列长度、dtype 和 workload 测。

**第 8 步：和 PyTorch SDPA 对账。** `check_attention_framework_conformance` 用 `float64`、形状 `[2,4,3]` 的同一组 `q/k/v` 克隆两份，一份走 `CustomAttention.apply`，一份走 `F.scaled_dot_product_attention(..., dropout_p=0.0)`；用同一个 `grad_out` 分别做 `torch.autograd.grad`，比较输出（`atol=1e-10`）和 `dq/dk/dv`（`atol=1e-9`）。它证明无 mask、无 dropout 时自定义反向与框架数学一致，不比较速度或显存。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| `dO` / `dout` | 损失对 Attention 输出的梯度，是反向链的起点 |
| `dV` | 输出梯度经 `P^T` 汇总回每个 Value 位置 |
| `dP` | 输出梯度经 `V^T` 投影回概率矩阵 |
| `dS` | 概率梯度穿过 Softmax 后得到的打分矩阵梯度 |
| Softmax 反向 | 每行减去概率加权均值，保证行和为零 |
| `scale` | $1/\sqrt d$，缩放点积的系数，Q/K 两支都要乘 |
| `autograd.Function` | 自定义前向/反向的接口，backward 按输入顺序返回梯度 |
| `save_for_backward` | 把反向需要的张量存进 `ctx`，反向时用 `ctx.saved_tensors` 取回 |
| `gradcheck` | 用有限差分逐元素核对解析梯度，需要双精度 |
| saved tensors | 前向为反向保留的中间状态，这里指 Q/K/V/P |
| $O(N^2)$ 项 | 完整概率矩阵带来的显存，随序列长度平方增长 |
| SDPA | PyTorch 的 `scaled_dot_product_attention`，融合实现 |

## 小白常见疑问
**Q1：为什么 `dV` 是 `P^T @ dO`，而不是 `P @ dO`？**
看形状和语义：`P` 是 `[N,N]`，`dO` 是 `[N,d]`，`P @ dO` 得到的是「每个查询位置把多少输出梯度分给 Value」，但 Value 的梯度应该按列汇总——第 $j$ 个 Value 收到所有查询位置 $i$ 的贡献，权重是 $P_{ij}$，所以要对 $i$ 求和，即 `P^T @ dO`。`test_value_gradient` 用自动求导直接验证。

**Q2：Softmax 反向为什么一定要减去行内加权和？不减会怎样？**
Softmax 每行输出之和恒为 1，输入整体加一个常数时输出不变，因此梯度在「全 1 方向」上的分量必须为零。`ds = p * (dp - weighted_sum)` 里的 `weighted_sum` 就是在扣掉这个公共分量；如果写成 `p * dp`，`dS` 的行和不再为零，`test_softmax_backward` 的第二条断言会失败，`gradcheck` 也过不了。

**Q3：为什么 `dQ` 和 `dK` 都要乘 `scale`，只乘一个不行吗？**
因为 $S=(QK^\top)\cdot scale$，`scale` 是对整个乘积的常数因子，对 $Q$ 或 $K$ 求导都会把它带出来。只给一边乘，等价于把 $S$ 当成 $(QK^\top)\cdot\sqrt{scale}$ 之类的错误定义，`test_query_key_gradients` 会分别对 `dq` 和 `dk` 报不一致。

**Q4：既然 PyTorch 能自动求导，为什么还要手写 backward？**
教学上是为了看清每一段梯度；工程上，自定义 backward 允许你控制保存什么、重算什么。比如 FlashAttention 不保存完整 `P`，而是在反向时用分块和 online softmax 恢复所需统计量，用额外计算换显存。本节仍保存了 `P`，所以账本是 $3BNd+BN^2$，不是 FlashAttention 的真实占用。

**Q5：`gradcheck` 为什么必须用 `float64`？**
它用有限差分近似导数，步长很小，`float32` 的舍入误差会把差分噪声放大到和真实梯度同量级，导致误报。`test_custom_attention_integration` 因此用 `dtype=torch.double`、`eps=1e-6`、`atol=1e-4`；框架核对也用 `float64`，输出容差 `1e-10`、梯度容差 `1e-9`。

## 动手验证：怎么知道自己做对了
`test_value_gradient`：`p=softmax(randn(2,4,4))`、`v=randn(2,4,6)`、`dout=randn(2,4,6)`，用 `(p @ v).backward(dout)` 得到 `v.grad`，要求 `value_gradient(p, dout)` 与它在 `atol=1e-8` 内一致。`test_probability_gradient`：`p` 为叶子、`v` 形状 `(2,4,6)`，同样 `atol=1e-8` 对照 `p.grad`。

`test_softmax_backward`：`scores` 形状 `(2,4,4)`、`dp=randn_like(scores)`，`p.backward(dp)` 后，手写结果既要与 `scores.grad` 在 `atol=1e-8` 内一致，又要满足 `manual.sum(dim=-1)` 全零。`test_query_key_gradients`：`q/k` 形状 `(2,4,6)`、`ds` 形状 `(2,4,4)`、`scale=6**-0.5`，`dq` 和 `dk` 分别与自动求导结果在 `atol=1e-8` 内一致。

`test_custom_attention_integration`：`q/k/v` 形状 `(1,3,4)`，输出与显式 `softmax((q @ k^T) / sqrt(4)) @ v` 在 `atol=1e-8` 内一致，且 `gradcheck(..., eps=1e-6, atol=1e-4)` 返回 `True`。`test_attention_input_contract`：`q` 形状 `(2,3,4)`、`k=q[:, :2]` 时 `CustomAttention.apply` 必须抛 `ValueError`。

六个测试全过打印 `✅ Attention 反向传播的局部梯度、组合 backward 与输入契约测试通过。`。notebook 没有保存任何单元格输出；可选框架核对通过时返回 `status: 'passed'` 和 `output_match`、`query_gradient_match`、`key_gradient_match`、`value_gradient_match` 四项布尔值。运行时间以你本机的 CPU 为准，`gradcheck` 在大形状上会明显变慢，教学示例故意用了很小的 `(1,3,4)`。
