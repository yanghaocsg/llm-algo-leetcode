# 10. Attention Backward and Custom Autograd Beginner | Attention 反向传播与自定义 Autograd · 小白入门

> 配套 notebook：[10. Attention 反向传播与自定义 Autograd](./10_Attention_Backward_and_Custom_Autograd.md) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

注意力算出输出之后，训练还需要把梯度沿原路传回它的三个输入。这本 notebook 把这条回传路径拆成四小段，让你逐段手写，再拼成一个完整的反向传播。

## 为什么需要它

注意力的前向计算只有三步：用 Q 和 K 算出打分矩阵，把分数变成概率，得到概率矩阵 P，再用 P 对 V 加权求和得到输出 O。把这三步交给 PyTorch，自动求导（Autograd）会替你算出 Q、K、V 的梯度。

想自己写注意力算子、控制它在反向时要保存哪些中间状态，就得手写反向传播。这条路径会分叉：输出梯度一路直接回到 V，另一路要穿过 Softmax（它负责把分数变成概率）再回到 Q 和 K。任何一段写错，最终的梯度都会错，拼在一起时局部的错误还容易被掩盖。

看一个很小的例子：只有第 1 个位置的第 1 个特征收到上游梯度 1。两个 Value 在这个位置的权重各占一半，所以各分到 0.5 的梯度。概率那一支还要减掉“每行的加权和”，否则传回打分矩阵的梯度每行之和不会接近 0。

这本 notebook 把路径拆成四个可以单独验证的小函数，每一段都和自动求导对照。最后把它们拼进一个自定义的 `torch.autograd.Function`，用数值梯度验证整条链。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| Q、K、V | 注意力的三路输入：查询、键、值。本题里形状都是 `[B, N, d]` |
| 打分矩阵 `S` / 概率矩阵 `P` | `S = QK^T / sqrt(d)`，对 `S` 的每行做 Softmax 得到 `P`，形状都是 `[B, N, N]` |
| 上游梯度 `dO` | 损失函数传给注意力输出 `O` 的梯度，是整条反向链的起点 |
| `dV`、`dP`、`dS`、`dQ`、`dK` | 损失对 V、P、S、Q、K 的梯度。量前面加个 `d`，就表示“对它的梯度” |
| Softmax | 把一行分数变成概率，每行加起来等于 1。本节对打分矩阵的每一行各做一次 |
| 自动求导（Autograd）与 `torch.autograd.Function` | 自动求导是 PyTorch 替你算梯度的机制。`Function` 让你自己写算子：`forward` 里用 `ctx.save_for_backward` 存下反向要用的张量，`backward` 里取回来用 |
| 数值梯度（`gradcheck`） | 用微小的扰动估计梯度，用来核对手写的反向传播对不对 |

## 核心思路

前向分三步，反向沿原路返回。记号约定：在某个量前面加 `d`，表示“损失对它的梯度”。

```text
前向：S = QK^T × scale  →  P = softmax(S)  →  O = P V         scale = 1 / sqrt(d)
反向：从上游梯度 dO 出发
      dV = P^T · dO                                 第 1 段：直接回到 V
      dP = dO · V^T                                 第 2 段：回到概率矩阵
      dS_ij = P_ij × (dP_ij - Σ_k P_ik dP_ik)       第 3 段：穿过 Softmax
      dQ = dS · K × scale，dK = dS^T · Q × scale     第 4 段：回到 Q 和 K
```

`B` 是 batch，`N` 是序列长度，`d` 是每个头的特征维数。`Q`、`K`、`V`、`O` 的形状是 `[B, N, d]`，`S`、`P` 是 `[B, N, N]`。`^T` 表示转置，也就是把矩阵的最后两维互换，代码里是 `transpose(-2, -1)`。`Σ_k P_ik dP_ik` 是第 `i` 行的加权和：`P` 和 `dP` 逐个相乘，再沿一行加起来。

用 2 个位置、特征维数 2 的单头例子走一遍：

```python
import math
import torch
q = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
k = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
v = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
dout = torch.tensor([[1.0, 0.0], [0.0, 0.0]])      # 上游梯度 dO
qd, kd, vd = q.detach(), k.detach(), v.detach()
scale = 1.0 / math.sqrt(2)                          # 特征维数 d = 2
p = torch.softmax(qd @ kd.transpose(-2, -1) * scale, dim=-1)   # 概率矩阵 P
dv = p.transpose(-2, -1) @ dout                     # TODO 1：P 转置后乘 dO
dp = dout @ vd.transpose(-2, -1)                    # TODO 2：dO 乘 V 的转置
ws = (dp * p).sum(dim=-1, keepdim=True)             # TODO 3：每行的加权和
ds = p * (dp - ws)                                  #         dS = P × (dP - 加权和)
dq = ds @ kd * scale                                # TODO 4：两条支路都乘 scale
dk = ds.transpose(-2, -1) @ qd * scale
print(p)
print(dp, dv, ws, sep='\n')                         # 依次是 dP、dV、每行的加权和
print(ds, dq, dk, sep='\n')                         # 依次是 dS、dQ、dK
(torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v).backward(dout)   # 交给自动求导
print(torch.allclose(dq, q.grad), torch.allclose(dk, k.grad), torch.allclose(dv, v.grad))
```

```text
tensor([[0.5000, 0.5000],
        [0.5000, 0.5000]])
tensor([[1., 0.],
        [0., 0.]])
tensor([[0.5000, 0.0000],
        [0.5000, 0.0000]])
tensor([[0.5000],
        [0.0000]])
tensor([[ 0.2500, -0.2500],
        [ 0.0000,  0.0000]])
tensor([[0.3536, 0.0000],
        [0.0000, 0.0000]])
tensor([[ 0.0000,  0.1768],
        [ 0.0000, -0.1768]])
True True True
```

Q 和 K 的点积全是 0，所以概率矩阵每个元素都是 0.5。`dout` 和 `dp` 只有左上角是 1。Value 那一支：两个 Value 在第 1 个位置的权重各是 0.5，所以 `dv` 的第 1 列是两个 0.5。概率那一支：第 1 行的加权和 `ws` 是 0.5 × 1 + 0.5 × 0 = 0.5；`ds` 的第 1 行是 0.5 × (1 − 0.5) = 0.25 和 0.5 × (0 − 0.5) = −0.25，加起来是 0。`dq` 和 `dk` 都乘了同一个 `scale`。末行的三个 `True` 说明手写的 `dq`、`dk`、`dv` 和自动求导的结果一致。

反向要用到前向留下的 Q、K、V、P。所以 `forward` 里用 `ctx.save_for_backward(q, k, v, p)` 存下这四个张量，`backward` 开头用 `ctx.saved_tensors` 取回来。P 的形状含 `N × N`，是保存量里随序列长度二次增长的那一项。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 把注意力看成一条依赖链：前向从 Q、K、V 到 S、P、O，反向从 `dO` 分解到 `dV`、`dP`、`dS`、`dQ`、`dK` | 读 |
| Step 2 | 四个梯度公式：矩阵乘法的两条支路、Softmax 的反向、缩放点积回到 Q 和 K | 读，对照上面的公式 |
| Step 3 | 反向要保存哪些前向状态，保存容量怎样估算，FlashAttention 怎样改变中间状态的保存方式 | 读 |
| Step 4 · TODO 1 | 回传 Value 分支 | 在 `value_gradient` 里写 1 行：`P` 转置后乘 `dO` |
| Step 4 · TODO 2 | 回传概率分支 | 在 `probability_gradient` 里写 1 行：`dO` 乘 `V` 的转置 |
| Step 4 · TODO 3 | 穿过 Softmax | 在 `softmax_backward` 里写 2 行：每行的加权和 `weighted_sum`，再算 `ds = p * (dp - weighted_sum)` |
| Step 4 · TODO 4 | 回传缩放点积的两条输入支路 | 在 `query_key_gradients` 里写 2 行：`dq` 和 `dk`，都乘上 `scale` |
| 测试 | 四段分别和自动求导对照（Softmax 那段还检查每行梯度和接近零），再用数值梯度检查拼起来的完整反向传播和输入形状 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把自定义的注意力输出和 Q/K/V 梯度，与 PyTorch 的 `scaled_dot_product_attention` 比较 | 需要 `TRAIN-CPU` 环境 |

Step 1–4 只需要 CPU，默认用 `BASE-CPU` 环境。框架一致性核对是可选项，需要 `TRAIN-CPU` 环境，可以先跳过。

## 容易踩的坑

- **转置的位置放错**：`dV` 是 `P` 转置后乘 `dO`，`dP` 是 `dO` 乘 `V` 的转置。两个公式都来自 `O = P V`，但转置的是不同的矩阵，`dP` 的形状必须和概率矩阵一致。
- **漏掉每行的加权和**：Softmax 的反向要先减掉 `Σ P × dP` 这个行内的公共分量。少了它，`dS` 每行的和就不会接近零。
- **只有一条支路乘了 `scale`**：`QK^T` 的两条输入支路都要带上同一个缩放因子，`dQ` 和 `dK` 分别保持 Q、K 的原始形状。
- **反向要用的张量没保存**：`forward` 里要用 `ctx.save_for_backward(q, k, v, p)` 保存，`backward` 才能取回。返回时按 `dq, dk, dv` 的顺序，和输入 `q, k, v` 一一对应。
- **把理论账本当显存峰值**：保存容量的估算只用来解释压力来源，不等同于 CUDA 峰值。实际显存和速度收益，要在 GPU 上结合序列长度、数据类型和工作负载（workload）测量。

## 读完后自查

- [ ] 能按顺序说出反向链：`dO` 先分成哪两路，再经过什么回到 Q 和 K。
- [ ] 能写出 `dV` 和 `dP` 的公式，并说出各自转置的是哪个矩阵。
- [ ] 能解释 Softmax 反向里“每行的加权和”起什么作用。
- [ ] 能说出 `forward` 里保存了哪四个张量，哪一个是随序列长度二次增长的。

## 下一步

- 打开 notebook：[10. Attention 反向传播与自定义 Autograd](./10_Attention_Backward_and_Custom_Autograd.md)
- 建议先读：[04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA_Beginner.md)
- 需要补基础：[PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- 做完之后：[11. 梯度累积 · 小白入门](./11_Gradient_Accumulation_Beginner.md)
