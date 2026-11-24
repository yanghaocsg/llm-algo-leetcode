# 00. PyTorch Warmup Beginner | PyTorch 热身 · 小白入门

> 配套 notebook：[00. PyTorch 热身](./00_PyTorch_Warmup.ipynb) ｜ 难度：Easy ｜ 环境：CPU-first

## 一句话说清

这本 notebook 练三个 PyTorch 基本功：把一组数字整理成模型要的形状、按编号查表取向量、手算梯度并传回给参数。后面的模型组件都要用到它们。

## 为什么需要它

Part 02 后面的组件都走同一条路：文本先变成一串编号，编号再变成模型能计算的向量；训练时，梯度还要沿原路传回参数。这条路上有三处容易出错。

- **形状对了，含义错了**。把形状 `[1, 2, 2, 2]` 的特征整理成“4 个位置、每个位置 2 个通道”，直接 `reshape` 后元素个数没错，但第 1 个位置拿到的是 `[0, 1]`，它本该是 `[0, 4]`。
- **查的不是同一张表**。编号 2 的意思是“取词表里行号为 2 的那一行”，行号从 0 数起。两条路径如果各自用了一份权重，结果就对不上。
- **梯度传不回去**。自己写前向传播时，如果没有记下反向传播要用的输入、权重和“哪些位置能过梯度”，反向传播就无从算起。

这本 notebook 把这三处各做成一道小题，让你在进入正式组件之前先各练一遍。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 轴（axis） | 张量的一个维度。`[B, C, H, W]` 有 4 个轴，长度依次是批次数、通道数、图像高、图像宽 |
| 词表（vocabulary） | 模型认识的全部词元（token）的列表，`V` 是词元个数 |
| 词元编号（token id） | 一个词元在词表里的行号，从 0 开始，最大是 `V - 1` |
| 词嵌入（Embedding） | 一张 `[V, D]` 的表，每个词元占一行；查表就是按编号取出一行 |
| 计算图 / 自动求导（Autograd） | PyTorch 做运算时顺手记下每一步，之后倒着走一遍就能算出梯度，这套机制叫自动求导 |
| 上游梯度 | 反向传播时从后面传回来的梯度，是本层计算梯度的起点 |
| 门控状态 | 一个布尔掩码（mask），`True` 的位置允许梯度通过，`False` 的位置梯度记为 0 |

## 核心思路

这本 notebook 是三道互相独立的小题。第一题整理形状。输入是 `[B, C, H, W]` 的特征：先把通道轴挪到最后，变成 `[B, H, W, C]`，再把高、宽两轴拉平，输出 `[B, H×W, C]`。Step 1 表格的第二行是同一个道理：多头表示 `[B, H, S, D]` 合并成 `[B, S, H×D]`，词元顺序不变，只合并注意力头和头维度。注意这里的 `H` 是注意力头数，不是图像高度。

第二题查表。输入 `input_ids` 的形状是 `[B, S]`，里面是整数编号；再给一张 `[V, D]` 的词嵌入表。每个编号当作行号取出一行，输出 `[B, S, D]`。

第三题自己写一个“线性变换 + ReLU”的函数，分前向传播和反向传播两步：

```text
z    = x @ W^T + b      # 激活前值
y    = relu(z)          # 负数变 0
mask = z > 0            # 门控状态
grad_z      = grad_output * mask   # 上游梯度先过门控
grad_x      = grad_z @ W
grad_weight = grad_z 的转置 @ x
grad_bias   = grad_z 沿 batch 维求和
```

符号说明：`x` 是输入，`W` 是权重，形状是 `[out_features, in_features]`，所以要先转置再相乘；`b` 是偏置；`grad_output` 是上游梯度。前向传播用 `ctx.save_for_backward` 存下 `x`、`weight`、`mask` 这三样最小必要的状态，反向传播用 `ctx.saved_tensors` 取回来。下面用很小的数把三道题各走一遍：

```python
import torch

# 第 1 题：[B, C, H, W] 整理成 [B, H*W, C]
x = torch.arange(8.).reshape(1, 2, 2, 2)
print(x.reshape(1, 4, 2)[0, 0])                      # 直接 reshape
print(x.permute(0, 2, 3, 1).reshape(1, 4, 2)[0, 0])  # 先换轴再 reshape
print(x[0, :, 0, 0])                                 # 第 1 个空间位置的 2 个通道

# 第 2 题：编号是词表的行号
weight = torch.tensor([[0., 1.], [2., 3.], [4., 5.]])
print(weight[torch.tensor([[2, 0]])])

# 第 3 题：上游梯度全是 1
x = torch.tensor([[1., 2.], [2., 1.]])
w = torch.tensor([[1., -1.], [1., 1.]])
z = x @ w.T             # 线性变换，偏置取 0
grad_z = torch.ones_like(z) * (z > 0)
print(z)
print(grad_z)
print(grad_z @ w)       # 输入的梯度
print(grad_z.T @ x)     # 权重的梯度
print(grad_z.sum(0))    # 偏置的梯度
```

```text
tensor([0., 1.])
tensor([0., 4.])
tensor([0., 4.])
tensor([[[4., 5.],
         [0., 1.]]])
tensor([[-1.,  3.],
        [ 1.,  3.]])
tensor([[0., 1.],
        [1., 1.]])
tensor([[1., 1.],
        [2., 0.]])
tensor([[2., 1.],
        [3., 3.]])
tensor([1., 2.])
```

第 1 题：直接 `reshape` 后，第 1 个位置是 `[0, 1]`；真实的第 1 个空间位置的两个通道是 `[0, 4]`，先换轴再 `reshape` 才能得到它。第 2 题：编号 2 取到 `[4, 5]`，编号 0 取到 `[0, 1]`。第 3 题：`z` 里的 `-1` 被门控挡住，`grad_z` 对应位置是 0；最后三行依次是输入、权重、偏置的梯度。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 二维特征怎样整理成序列：每个轴承担什么含义，重排时要保持什么 | 读，对照上面例子的第 1 题 |
| Step 2 | 词元编号怎样通过词表变成连续表示，查表对输入有什么要求 | 读，对照上面例子的第 2 题 |
| Step 3 | 前向传播要保存哪些状态，反向传播怎样用它们算出梯度 | 读，对照上面的公式和第 3 题 |
| Step 4 · TODO 1 | 保持位置—通道对应关系的序列重排 | 在 `tensor_warmup` 里写 2 行：`x_native` 先换轴再 `reshape`，`x_einops` 用 `einops.rearrange` 写出同样的重排 |
| Step 4 · TODO 2 | 验证 Embedding 查表等价性 | 在 `embedding_warmup` 里写 3 行：建一个 `nn.Embedding`，用模块和按行索引各查一次表 |
| Step 4 · TODO 3 | 保存局部前向状态 | 在 `forward` 里写 3 行：激活前值 `z`、输出 `y` 和门控状态 `mask` |
| Step 4 · TODO 4 | 按链式法则回传梯度 | 在 `backward` 里写 4 行：上游梯度先过门控，再算输入、权重、偏置三类梯度 |
| 测试 | 检查编号校验、两种重排是否一致、两条查表结果是否一致、前向数值和三类梯度是否与 PyTorch 一致 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |

Step 1–4 只需要 CPU。

## 容易踩的坑

- **只 `reshape`，不先换轴**：元素个数对了，但位置和通道的对应关系错了。先用 `permute` 把通道轴挪到最后，再 `reshape`，并用第一个空间位置的通道值核对一遍。
- **两条查表路径用了两份权重**：编号只是词表的行号，`nn.Embedding` 模块和直接索引必须读同一份权重，否则数值对不上。
- **`input_ids` 不合规**：必须是二维的 `torch.long` 张量，取值在 `[0, V)` 之内，否则开头的 `validate_token_ids` 会报错。
- **门控状态和激活前值不一致**：`mask` 要按 `z > 0` 算，`z` 不大于 0 的位置都要挡住。反向传播时上游梯度先乘 `mask`，被挡住的位置梯度为 0。
- **梯度的形状或顺序不对**：输入、权重、偏置三类梯度要分别与 `x`、`weight`、`bias` 同形，偏置梯度要沿批次维求和，并按这个顺序返回。

## 读完后自查

- [ ] 能说出把 `[B, C, H, W]` 整理成 `[B, H×W, C]` 时，为什么不能直接 `reshape`。
- [ ] 给一张 `[V, D]` 的词表和一个编号，能说出取的是哪一行、结果形状是什么。
- [ ] 能说出 `forward` 保存了哪三样东西，各自在 `backward` 里做什么用。
- [ ] 能按顺序写出上游梯度变成输入、权重、偏置三类梯度的步骤。

## 下一步

- 打开 notebook：[00. PyTorch 热身](./00_PyTorch_Warmup.ipynb)
- 需要先补基础：[PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb)、[PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)、[PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb)
- 做完之后：[01. RMSNorm 教程 · 小白入门](./01_RMSNorm_Tutorial_Beginner.md)
