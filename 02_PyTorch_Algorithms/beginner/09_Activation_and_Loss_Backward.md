# 09. Activation and Loss Backward | 激活与损失反向传播 · 小白入门

> 配套 notebook：[09. 激活与损失反向传播](../09_Activation_and_Loss_Backward.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

训练时，误差从损失函数出发，一层一层往回传。这本 notebook 手写其中两段规则：激活函数 ReLU 怎么决定梯度能不能通过，交叉熵这个损失函数怎么把误差送回模型的输出分数。

## 为什么需要它

训练时先做一次前向计算得到损失，再从损失出发，把梯度沿原路传回去。回传的路上有两类环节：损失函数在终点产生梯度，激活函数在中间决定已有的梯度能不能继续通过。

两类环节都有容易做错的地方。ReLU 的输入是负数时，梯度应该被挡住。交叉熵里被标成“忽略”的位置，不该产生损失，也不该算进平均的分母。比如 3 个位置里 1 个被忽略，平均的分母应该是 2，不是 3。

PyTorch 的自动求导（Autograd）本来就能算这些梯度。自己手写一遍的意义在于，做完之后你能检查三件事：梯度的方向对不对，哪些位置受到了监督，平均的尺度对不对。这本 notebook 让你手写两段：ReLU 的反向门控，以及交叉熵对输出分数的梯度。写完用自动求导和原生的交叉熵来对照。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| logits | 模型输出的未归一化分数，形状是 `[batch, classes]`，每个位置一行 |
| Softmax 与交叉熵（cross-entropy） | Softmax 把一行分数变成概率，整行加起来等于 1。交叉熵是常用的损失（loss）函数，比较概率和正确答案，差得越远数值越大 |
| 目标分布（one-hot） | 正确答案的写法：正确类别那一位是 1，其余是 0 |
| 忽略位置 / `ignore_index` | 标签等于 `ignore_index`（默认 `-100`）的位置，比如填充位置。它不参与损失、梯度和平均分母 |
| 上游梯度 / 门控（gating） | 上游梯度是从后面传回来的梯度。门控是一个只含 0 和 1 的掩码，决定它在哪些位置能通过 |
| 归约（reduction） | 把每个位置的损失合成一个数：`mean` 是求平均，`sum` 是求和 |
| 自动求导（Autograd） | PyTorch 记录每一步运算，再倒着算出梯度的机制。本节手写的梯度要和它对照 |

## 核心思路

反向传播从损失开始。交叉熵负责把误差送回 logits，ReLU 负责决定梯度能不能通过：

```text
p_ic     = softmax(z_i)_c                # 把一行分数 z_i 变成概率
∂L/∂z_ic = (p_ic - y_ic) / N            # 有效位置：概率减去目标，再除以 N
∂L/∂z_ic = 0                            # 忽略位置：整行为 0
g(x)     = 1 如果 x > 0，否则 0          # ReLU 的门控，本节约定 x == 0 时为 0
∂L/∂x    = ∂L/∂y × g(x)                  # 上游梯度逐个位置乘上门控
```

`z_ic` 是位置 `i` 在类别 `c` 上的 logit，`p_ic` 是对应的概率，`y_ic` 是目标分布，`N` 是有效位置数（用 `sum` 归约时去掉 `/N`）。`x` 是 ReLU 前的输入，`y = ReLU(x)`，`∂L/∂y` 是上游梯度。Step 2 里说的“向量—雅可比乘积”，落到 ReLU 上就是最后一行。

下面分三段走一遍：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, 0.0, 3.0])            # 例 1：ReLU 门控
upstream = torch.tensor([1.0, 1.0, 2.0])
mask = (x > 0).to(upstream.dtype)             # TODO 1：正区间为 1，其余为 0
print(mask)
print(upstream * mask)

logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 1.0]])   # 例 2：交叉熵
labels = torch.tensor([0, 1, -100])           # 第 3 个位置是忽略位置
valid = labels != -100                        # TODO 2：有效位置掩码
safe = labels.masked_fill(~valid, 0)          # 忽略标签换成能索引的 0
log_probs = F.log_softmax(logits, dim=-1)     # 稳定的对数概率
one_hot = F.one_hot(safe, 2).to(logits.dtype) * valid[:, None]
probs = log_probs.exp()
diff = (probs - one_hot) * valid[:, None]     # p - y，忽略位置整行乘成 0
n = valid.sum()                               # TODO 3：分母只能是有效位置数
per = -(one_hot * log_probs).sum(1)           # 每个位置的损失
print(probs)
print(diff, n)
print(per, per.sum() / n, per.sum())          # mean 的损失，sum 的损失
print(diff / n)                               # mean 的梯度

big = torch.tensor([[1000.0, 0.0, -1000.0]])  # 例 3：分数很大时
print(torch.log(big.exp() / big.exp().sum()))  # 先 softmax 再取对数
print(F.log_softmax(big, dim=-1))
```

```text
tensor([0., 0., 1.])
tensor([0., 0., 2.])
tensor([[0.5000, 0.5000],
        [0.5000, 0.5000],
        [0.9820, 0.0180]])
tensor([[-0.5000,  0.5000],
        [ 0.5000, -0.5000],
        [ 0.0000,  0.0000]]) tensor(2)
tensor([0.6931, 0.6931, -0.0000]) tensor(0.6931) tensor(1.3863)
tensor([[-0.2500,  0.2500],
        [ 0.2500, -0.2500],
        [ 0.0000,  0.0000]])
tensor([[nan, -inf, -inf]])
tensor([[    0., -1000., -2000.]])
```

例 1 里，门控是 `[0, 0, 1]`，上游梯度 `[1, 1, 2]` 只有最后一个原样通过。例 2 里，前两个位置的分数相同，概率都是 `[0.5000, 0.5000]`。第 1 个位置的正确类别是 0，概率减目标是 `[-0.5, 0.5]`；第 2 个位置反过来。第 3 个位置被忽略：它的概率不是 0，所以 `p - y` 要再乘一次有效掩码，整行才是 0。有效位置 `n` 是 2，所以 `mean` 的损失是 1.3863 / 2 = 0.6931，`mean` 的梯度是 -0.5 / 2 = -0.25。每个位置的损失那一行里，忽略位置显示成 `-0.0000`，它就是 0。例 3 里，先算概率再取对数，分数是 1000 时算出了 `nan` 和 `-inf`；`log_softmax` 给出的三个数都是有限的。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 前向得到损失、从损失反向传播的整条路径，以及损失函数和局部算子各自负责什么 | 读 |
| Step 2 | ReLU 的输入为正、为负、等于 0 时，梯度怎样通过 | 读，对照例 1 |
| Step 3 | 交叉熵的公式：概率减去目标、忽略位置、`mean` 的分母 | 读，对照例 2 |
| Step 4 · TODO 1 | 构造 ReLU 的反向门控掩码 | 在 `relu_backward` 里写 1 行：`mask = (x > 0).to(grad_out.dtype)` |
| Step 4 · TODO 2 | 构造有效位置掩码、稳定概率和目标分布 | 在 `softmax_ce_loss_and_grad` 里补 `valid_mask`、`safe_labels`、`log_probs`、`one_hot`；全是忽略位置时要报错 |
| Step 4 · TODO 3 | 按有效位置数量归约损失和 logits 梯度 | 在同一个函数里补 `per_sample_loss`、`loss`、`grad`；`mean` 时除以有效位置数 |
| 测试 | ReLU 梯度和自动求导对照并检查零点；含忽略位置的交叉熵和原生算子对照；`mean` 与 `sum` 的比例、极值输入；越界标签、全忽略和形状错误的输入被拒绝 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 在同一组输入上，把手写的 ReLU 梯度、交叉熵损失和 logits 梯度与 PyTorch 原生算子比较 | 需要 `TRAIN-CPU` 环境 |

Step 1–4 只需要 CPU，默认用 `BASE-CPU` 环境。框架一致性核对是可选项，需要 `TRAIN-CPU` 环境，可以先跳过。

## 容易踩的坑

- **拿忽略标签去取目标分布**：`-100` 不是合法的类别编号，不能直接拿去做索引。要先把忽略位置换成占位的类别 0，再把这一行整体乘成 0，目标分布里不能留下概率。
- **`mean` 的分母用了全部位置数**：分母只能是有效位置数。忽略位置不算损失，也不算分母。整批都是忽略位置时，没有可以平均的对象，要直接报错。
- **忽略位置的梯度没清零**：`p - y` 在忽略位置不一定为 0，要整行乘成 0。`sum` 的损失和梯度，正好是 `mean` 的有效位置数倍。
- **ReLU 在零点的约定不一致**：本节约定 `x == 0` 时梯度为 0，门控要按 `x > 0` 来写。掩码还要转成和上游梯度相同的数据类型。
- **先算概率再取对数**：分数很大时会算出 `nan` 和 `-inf`。对数概率要用 `log_softmax` 直接算。

## 读完后自查

- [ ] ReLU 的输入是 `-2`、`0`、`3`，上游梯度是 `[1, 1, 2]` 时，能写出传回的梯度。
- [ ] 能写出交叉熵对 logits 的梯度公式，并说出忽略位置的梯度是多少。
- [ ] 3 个位置里 1 个被忽略时，能说出 `mean` 的分母是几。
- [ ] 能说出为什么对数概率要用 `log_softmax` 算。

## 下一步

- 打开 notebook：[09. 激活与损失反向传播](../09_Activation_and_Loss_Backward.ipynb)
- 需要补基础：[PyTorch 自动求导与反向传播](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)、[数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
- 做完之后：[10. Attention 反向传播与自定义 Autograd · 小白入门](./10_Attention_Backward_and_Custom_Autograd.md)
- 相关的 notebook：[19. 激活检查点](../19_Activation_Checkpointing.ipynb)
