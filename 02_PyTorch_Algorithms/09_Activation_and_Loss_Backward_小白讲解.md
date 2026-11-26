# 09. 激活与损失反向传播 — 小白讲解

> 对应 notebook：[`09_Activation_and_Loss_Backward.ipynb`](./09_Activation_and_Loss_Backward.ipynb)

## 一句话总结

训练信号有两条腿：损失函数决定"误差从哪里产生"，激活函数决定"已有梯度能不能通过"。本节手写两个局部反向：`relu_backward` 用前向输入的正负构造门控 `mask=(x>0)`，`softmax_ce_loss_and_grad` 用 `log_softmax` 得到稳定对数概率、用 `ignore_index` 构造有效监督掩码，再按有效位置数归约出 `mean` 或 `sum` 的 loss 和 logits 梯度。三个 TODO 的测试分别对照 PyTorch 自动求导、检查忽略位置、检查 `mean/sum` 的缩放关系，最后还有可选框架核对，把数值误差压到 `1e-12` 的量级。

## 为什么需要它：痛点是什么

初学者对反向传播的困惑通常集中在几个"看起来会算、实际会错"的点：

- ReLU 在 $x=0$ 时导数到底取 0 还是 1？取错不会让程序报错，但会让梯度和框架对不上。
- 交叉熵的梯度为什么是 `p - y`？为什么还要除以有效位置数？
- `ignore_index=-100` 的位置，loss 为 0 可以理解，为什么它的梯度也必须整行为 0？
- 直接用 `log(softmax(z))` 为什么会在 logits 很大时变成 `-inf`？
- `mean` 的分母到底是 batch 大小、序列长度，还是有效 token 数？

生活类比：把反向传播想成供水系统。损失函数是水源，决定"哪里有水、水压多大"；激活函数是每段管子上的阀门，决定"上游的水能不能继续往下流"。水源再大，阀门关着（ReLU 负区间）也过不去；阀门开着，但水源本身是 0（忽略位置），流下去的还是 0。这一节就是把水源和阀门分别拆开看。

## 核心问题

- ReLU 的局部门控由什么决定？为什么 `x == 0` 要单独约定？
- 交叉熵对 logits 的梯度为什么是 $(p-y)/|\mathcal V|$？这个式子怎么推出来？
- `ignore_index` 的位置如何同时从 loss、梯度和 `mean` 分母中排除？
- 为什么用 `F.log_softmax` 而不是 `log(softmax(...))`？
- `mean` 和 `sum` 两种归约的 loss 与梯度差多少？测试怎么检查？
- 手写结果和 PyTorch 原生算子对齐时，为什么用 `float64` 和 `1e-12` 容差？

## 解决思路：一步步看懂

### Step 1：两个责任要分开

一条样本先前向得到输出和损失，再由损失产生输出端梯度，梯度沿计算依赖返回较早的表示，并在激活函数处受前向状态门控。Step 1 强调要区分两个责任：**损失决定训练信号从哪里产生，局部算子决定已有信号怎样继续传播**。这句话在代码里的体现是：`softmax_ce_loss_and_grad` 负责"产生" `(p-y)/|V|`，`relu_backward` 只负责"乘以 0 或 1"。两者都可以单独测试，也都可以单独替换。

放进真实网络看，顺序是：最后一层线性输出 logits → 交叉熵算出 `grad_logits` → 反向传到上一层的激活函数 → `relu_backward` 把上游梯度按门控过滤 → 再传给更早的线性层。本节只实现"交叉熵"和"ReLU"这两个局部节点，不实现整条链；但只要你确认每个节点的局部梯度是对的，整条链的正确性就可以由链式法则保证。这也是为什么测试都用"相同上游梯度 + 对照自动求导"的方式，而不是训练一个完整模型看 loss 降不降。

### Step 2：ReLU 的门控

对 $y=\mathrm{ReLU}(x)=\max(x,0)$，局部导数可以写成：

$$g(x)=\mathbf{1}[x>0],\qquad \frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\odot g(x)$$

$\mathbf{1}[\cdot]$ 是指示函数，条件成立取 1、否则取 0；$\odot$ 是逐元素相乘。按链式法则，$\partial L/\partial x=(\partial L/\partial y)(\partial y/\partial x)$，而 $\partial y/\partial x$ 在正区间是 1、负区间是 0，所以"乘 1 或乘 0"就是把局部导数代进去的结果。这里的 $\partial L/\partial y$ 就是函数参数 `grad_out`，也就是上游传回来的梯度；`relu_backward` 不关心上游梯度是怎么算出来的，只负责把门控乘上去。三个区间：

| 前向输入 `x` | ReLU 的局部导数 | 传回的输入梯度 |
|---|---|---|
| `x > 0` | `1` | 保留上游梯度 |
| `x < 0` | `0` | 梯度变为 `0` |
| `x == 0` | 本节约定为 `0` | 梯度变为 `0` |

为什么 $x=0$ 要单独约定？因为 ReLU 在 0 点不可导（左导 0、右导 1），数学上没有唯一答案。PyTorch 的 `F.relu` 反向在 0 点取 0，本节跟随这个约定。notebook 的测试同时检查"和自动求导一致"以及"`manual[2].item() == 0.0`"，后者就是防止有人写成 `x >= 0`。

### TODO 1：`relu_backward`

骨架先做两个校验：`grad_out.shape != x.shape` 抛 `ValueError`，`grad_out.device != x.device` 抛 `ValueError`。参考答案一行：

```python
mask = (x > 0).to(grad_out.dtype)
return grad_out * mask
```

拆开看：

- `(x > 0)` 得到布尔张量，True 表示正区间；
- `.to(grad_out.dtype)` 把布尔转成和上游梯度相同的浮点类型。PyTorch 里布尔乘浮点其实也能算，但显式转换能保证 dtype 一致，避免在混合精度或 `float64` 下出现意外的类型提升；
- `grad_out * mask` 逐元素相乘，负区间和 0 点变成 0，正区间原样保留。

容易写错的地方：用 `x >= 0` 会把 0 点的梯度放过去，和测试的 `manual[2] == 0` 冲突；用 `mask = x > 0` 后直接 `grad_out * mask` 在大多数情况下也能跑，但 dtype 不是显式控制的；忘记校验 shape 会在广播时悄悄算出一个形状不对的结果，而不是报错。

为什么门控用输入 `x`，而不是前向输出 `y`？因为 $y=\mathrm{ReLU}(x)$ 在 $y>0$ 时能反推出 $x>0$，但 $y=0$ 时既可能是 $x<0$ 也可能是 $x=0$，分不清 0 点该取哪个约定。直接用 `x > 0` 没有歧义，也省得再做一次比较。代价是反向时需要拿到前向输入（或它的符号），这也是为什么很多框架在前向时保存 mask，而不是保存完整输出。

`test_relu_backward` 的输入是 `x=[-2.0,-0.5,0.0,1.0,3.0]`、上游梯度 `[0.5,-1.0,2.0,0.25,-0.75]`。它先 `F.relu(x).backward(upstream)` 得到自动求导的 `x.grad`，再调用手写版本，断言 `torch.allclose(x.grad, manual)`。按公式，手写结果是 `[0,0,0,0.25,-0.75]`；0 点那一项单独断言为 0。

### Step 3：交叉熵的 loss 与 logits 梯度

设第 $i$ 个位置的 logits 为 $z_i$，Softmax 概率为 $p_i=\mathrm{softmax}(z_i)$，one-hot 目标为 $y_i$，有效监督位置集合为 $\mathcal V$。使用有效位置平均时：

$$L=-\frac{1}{|\mathcal V|}\sum_{i\in\mathcal V}\sum_c y_{ic}\log p_{ic},\qquad \frac{\partial L}{\partial z_{ic}}=\begin{cases}\dfrac{p_{ic}-y_{ic}}{|\mathcal V|}, & i\in\mathcal V,\\ 0, & i\notin\mathcal V.\end{cases}$$

符号解释：$c$ 是类别（词表里的某个 token），$|\mathcal V|$ 是有效位置数，$y_{ic}$ 是 one-hot（正确类别为 1，其余为 0）。如果 reduction 是 `sum`，去掉 $1/|\mathcal V|$ 即可。

梯度怎么来的？对单个位置，$p_c=\exp(z_c)/\sum_k\exp(z_k)$，Softmax 的雅可比是 $\partial p_c/\partial z_j=p_c(\delta_{cj}-p_j)$。代入 $L=-\log p_{y}$（one-hot 时只有正确类别那一项），得到：

$$\frac{\partial L}{\partial z_j}=p_j-y_j$$

直觉读法：如果模型给正确类别的概率是 0.6，那么正确类别那一维的梯度是 $0.6-1=-0.4$（要把它的 logit 往上推），其余类别是各自概率（要往下压）。所有维加起来是 0，因为 Softmax 的概率和恒为 1。

不跳步地推一遍单样本、one-hot 的情况。先对正确类别 $y$ 求导：

$$\frac{\partial L}{\partial p_y}=-\frac{1}{p_y}$$

Softmax 对正确类别自身 logit 的导数是：

$$\frac{\partial p_y}{\partial z_y}=p_y(1-p_y)$$

两者相乘：

$$\frac{\partial L}{\partial z_y}=-\frac{1}{p_y}\cdot p_y(1-p_y)=p_y-1$$

对错误类别 $c\neq y$，$L$ 只通过 $p_y$ 依赖 $z_c$，而 $\partial p_y/\partial z_c=-p_y p_c$，所以：

$$\frac{\partial L}{\partial z_c}=-\frac{1}{p_y}\cdot(-p_y p_c)=p_c$$

合起来就是 $p_j-y_j$：正确类别减 1，其余类别不减。平均归约再除以 $|\mathcal V|$，忽略位置则整行为 0。这个推导也解释了为什么梯度里会出现 `probs`：它是 Softmax 的雅可比带出来的，不是额外加的。

`mean` 和 `sum` 的区别只在缩放：`sum` 保留所有有效位置贡献之和，`mean` 除以有效位置数。忽略位置既不参与 loss，也不进分母——这一点和"除以 batch 大小"不同，后者会把忽略位置也算进平均。

### TODO 2：有效掩码、稳定概率与目标分布

骨架在进入 TODO 前已经做了完整的输入校验：`logits` 必须是二维、`labels` 必须是一维且 batch 对齐；batch 和 classes 不能为 0；两者同 device；`reduction` 只能是 `"mean"` 或 `"sum"`；`logits` 必须是浮点；`labels` 必须是 `int32` 或 `int64`；除 `ignore_index` 外的标签必须在 `[0, classes)` 内。最后一条的写法是：

```python
invalid_labels = (labels != ignore_index) & ((labels < 0) | (labels >= logits.size(1)))
```

它先把 `-100` 排除掉，再检查其余标签是否越界。如果不先排除，`-100 < 0` 会让所有带忽略位置的正常 batch 都报错。

参考答案四行：

```python
valid_mask = labels != ignore_index
if not valid_mask.any():
    raise ValueError('至少需要一个有效监督位置')
safe_labels = labels.masked_fill(~valid_mask, 0)
log_probs = F.log_softmax(logits, dim=-1)
one_hot = F.one_hot(safe_labels, num_classes=logits.size(1)).to(logits.dtype) * valid_mask.unsqueeze(1)
```

逐行解释：

1. `valid_mask` 是 `[batch]` 布尔张量，True 表示这个位置参与监督。
2. 如果整批都是 `ignore_index`，直接报错。为什么？因为 `mean` 要除以 `valid_count`，全忽略时分母为 0；与其返回 NaN，不如让调用方知道这批数据没有监督信号。
3. `safe_labels` 把忽略位置替换成类别 0。为什么要替换？因为 `F.one_hot` 不接受负数索引，`-100` 会直接越界。替换成 0 只是为了让 `one_hot` 能跑，后面会用 `valid_mask` 把整行乘成 0。
4. `log_probs` 用 `F.log_softmax`，它内部算 `z - logsumexp(z)`，而 `logsumexp` 会先减去最大值再取指数，所以 logits 是 1000 也不会溢出。直接 `torch.log(torch.softmax(z))` 在 $e^{1000}$ 时先溢出成 `inf`，再取 log 就废了。
5. `one_hot` 先转成 `logits.dtype`，再乘 `valid_mask.unsqueeze(1)`。`unsqueeze(1)` 把 `[batch]` 变成 `[batch,1]`，广播到 `[batch,classes]`；忽略位置整行乘 0，目标分布里不留任何概率质量。

骨架里接着写 `probs = log_probs.exp()`，而不是再调一次 `torch.softmax(logits, dim=-1)`。两者数值上等价，但 `exp(log_probs)` 复用了同一份 `log_probs`：既少一次归一化计算，也保证 `probs` 和 `log_probs` 严格对应同一分布，不会因为两次独立计算引入微小差异。在需要保证"loss 用的分布"和"梯度用的分布"一致的场景里，这一点很重要。

为什么要在 `one_hot` 上乘 mask，而不是只在最后算 loss 时乘？因为梯度也要用 `one_hot`：`grad = (probs - one_hot) * valid_mask.unsqueeze(1)`。如果 `one_hot` 在忽略位置还留着 1，即使 loss 被 mask 掉，梯度那一行也会是 `p - 1` 而不是 0。所以"整行清零"必须发生在目标分布这一层。

把每一步的张量形状列出来，可以避免广播错误：

| 变量 | 形状 | 含义 |
|---|---|---|
| `logits` | `[batch, classes]` | 未归一化分数 |
| `labels` | `[batch]` | 整数类别索引，含 `-100` |
| `valid_mask` | `[batch]` | True 表示参与监督 |
| `safe_labels` | `[batch]` | 忽略位置替换成 0 |
| `log_probs` | `[batch, classes]` | 稳定对数概率 |
| `one_hot` | `[batch, classes]` | 目标分布，忽略位置整行 0 |
| `probs` | `[batch, classes]` | `log_probs.exp()` |
| `per_sample_loss` | `[batch]` | 每个位置的损失 |
| `grad` | `[batch, classes]` | 与 logits 同形 |

一个常见的替代写法是用 `log_probs.gather(1, labels.unsqueeze(1))` 取正确类别的 logp。它更省内存，但 `-100` 不能直接做索引，必须先 `clamp` 或替换；而且 gather 出来的忽略位置仍然带着 `probs` 的梯度，还得再 mask 一次。notebook 选择 `one_hot`，是因为它把"目标分布"和"忽略位置"合成一个张量，loss 和梯度都能直接复用，教学上更不容易漏。真实框架里两种写法都有，关键不是选哪个，而是保证 loss、梯度、分母三处口径一致。

### TODO 3：归约 loss 与梯度

参考答案：

```python
per_sample_loss = -(one_hot * log_probs).sum(dim=1)
valid_count = valid_mask.sum()
loss_sum = per_sample_loss.sum()
loss = loss_sum / valid_count if reduction == "mean" else loss_sum
grad = (probs - one_hot) * valid_mask.unsqueeze(1)
if reduction == "mean":
    grad = grad / valid_count
```

- `per_sample_loss` 是 `[batch]`：`one_hot * log_probs` 只保留正确类别那一项（忽略位置整行为 0），沿类别维求和再取负。
- `valid_count` 是有效位置数（一个张量），不是 batch 大小。
- `loss` 在 `mean` 时除以 `valid_count`，`sum` 时保留总和。
- `grad` 先算 `probs - one_hot`，再乘 mask 把忽略位置整行清零；`mean` 时再除以 `valid_count`。

用测试里的 logits 手算一遍（不是保存输出，是按公式算的）：三行 logits 是 `[[1.0,0.5,-0.2],[0.3,0.1,-0.4],[0.2,-0.3,1.2]]`，labels 是 `[0,-100,2]`，所以有效位置是第 0、2 行，`valid_count=2`。

- 第 0 行 softmax ≈ `[0.5242,0.3179,0.1579]`，label 0，loss ≈ `0.6459`，梯度 `p - e_0 ≈ [-0.4758,0.3179,0.1579]`。
- 第 2 行 softmax ≈ `[0.2312,0.1402,0.6285]`，label 2，loss ≈ `0.4644`，梯度 `p - e_2 ≈ [0.2312,0.1402,-0.3715]`。
- 第 1 行忽略：loss 为 0，梯度整行为 0。

`mean` loss ≈ `(0.6459+0.4644)/2 ≈ 0.5551`；`sum` loss ≈ `1.1103 = mean × 2`。`mean` 梯度是上面的逐行梯度再除以 2；`sum` 梯度不除。测试 `test_reduction_and_stability` 断言 `sum_loss ≈ mean_loss * valid_count`、`sum_grad ≈ mean_grad * valid_count`，`atol=1e-6`。

真实训练里，`ignore_index` 通常来自数据整理阶段：padding 位置被标成 `-100`，只对回答做监督时 prompt 位置也标成 `-100`。所以"有效位置数"往往等于"本 batch 实际回答 token 数"，会随样本长度变化。`mean` 让不同 batch 的 loss 尺度大致可比，学习率不用随 batch 内 token 数频繁调整；`sum` 保留累积量，适合你想自己控制分母（比如梯度累积时按总 token 数归一）的场景。选哪个不是对错问题，而是要和优化器、学习率、累积策略一起定。

函数返回 `(loss, grad)` 而不是只返回 loss，也不是只返回标量 loss，是有意的：`grad` 的形状必须和 `logits` 一致，才能接回计算图；`loss` 是标量，方便打印和反向；保留逐位置的 `per_sample_loss` 中间量，则让"按样本加权"这种扩展有地方可插。`grad` 是手工推出来的解析梯度，不依赖 autograd，所以可以用来教学，也可以在和框架对不上时定位是哪一项符号错了。

### 输入契约与边界

`softmax_ce_loss_and_grad` 的报错路径：

- 形状不对（`logits` 不是二维、`labels` 不是一维、batch 不匹配）→ `ValueError`；
- batch 或 classes 为 0 → `ValueError`；
- device 不同 → `ValueError`；
- `reduction` 不是 `mean/sum` → `ValueError`；
- `logits` 不是浮点 → `TypeError`；
- `labels` 不是 `int32/int64` → `TypeError`；
- 标签越界 → `ValueError`；
- 全忽略 → `ValueError`（在 TODO 2 里）。

`relu_backward` 的形状不匹配也抛 `ValueError`。`test_activation_loss_input_contract` 用三个非法调用覆盖：标签 `3` 超出 3 类、`[-100,-100]` 全忽略、`relu_backward(torch.ones(2), torch.ones(3))` 形状不一致，要求都抛 `ValueError`。

为什么全忽略要显式报错，而不是返回 0？`mean` 的分母是 `valid_count`，全忽略时就是 0/0，结果依赖具体实现和版本；与其让一个可能为 `nan` 的值悄悄传进优化器、污染所有参数，不如在函数入口停下，让调用方知道"这批数据没有监督信号"。这也是测试把它列进非法输入的原因。

想做一次快速自检，可以把所有 logits 设为 0、类别数设为 3：此时每个类别的概率都是 $1/3$，loss 是 $\log 3\approx1.0986$，正确类别的梯度是 $1/3-1=-2/3$，其余两类各是 $1/3$。如果你的实现在这个输入上得不到这三个数，先别急着跑大模型，回到公式检查 one-hot 和分母。

### 常见错误清单

| 错误写法 | 会发生什么 | 哪个测试会发现 |
|---|---|---|
| `mask = (x >= 0)` | 0 点梯度变成上游值，不是 0 | `test_relu_backward` 的 `manual[2] == 0` |
| 忘记 `grad_out.shape` 校验 | 形状不同时广播出一个"能跑但错"的结果 | `test_activation_loss_input_contract` |
| `torch.log(torch.softmax(logits))` | 大 logits 时先溢出成 `inf` | `test_reduction_and_stability` 的 1000 输入 |
| `one_hot` 不乘 `valid_mask` | 忽略位置梯度不是 0 | `test_cross_entropy_gradient` 的 `count_nonzero == 0` |
| `mean` 除以 `labels.numel()` | `sum ≠ mean × 有效数` | `test_reduction_and_stability` |
| `mean` 只除 loss 不除梯度 | 梯度尺度大了一倍 | 同上，`sum_grad` 关系不成立 |
| 用 `labels` 直接做 `one_hot` | `-100` 越界，运行时直接报错 | `test_cross_entropy_gradient` 跑不到断言 |
| 全忽略时返回 0 而不是报错 | 训练悄悄没有信号 | `test_activation_loss_input_contract` |

调试时建议按依赖顺序来：先单独跑 `test_relu_backward`，确认门控和 0 点约定；再跑 `test_cross_entropy_gradient`，确认 loss 和梯度都与自动求导一致；然后跑 `test_reduction_and_stability`，确认分母和缩放；最后跑输入契约和框架核对。这样一旦失败，能立刻定位是"门控错""目标分布错"还是"归约错"，而不是面对一个综合报错猜。如果时间只够跑一个，选 `test_cross_entropy_gradient`：它同时覆盖 loss、梯度和忽略位置，信息量最大。

还有一点值得记住：这两个函数是**纯局部**的。`relu_backward` 不知道损失是什么，`softmax_ce_loss_and_grad` 不知道网络前面有多少层。这种拆分让它们可以独立测试，也让你在第 10 节分析 Attention 反向时，可以把"矩阵级链式法则"和"局部算子"分开看。

### 框架一致性核对

可选单元用 `float64` 和更严的容差 `atol=1e-12, rtol=1e-12` 对比手写与 PyTorch 原生：

- ReLU：`x=[[-1.5,0.0,2.0]]`、`grad_out=[[0.3,-0.7,1.2]]`，`torch.relu(x).backward(grad_out)` 的梯度应等于 `relu_backward`；
- 交叉熵：`logits=[[2.0,-1.0,0.5],[0.2,1.3,-0.4],[4.0,-3.0,0.0]]`、`labels=[0,-100,2]`，`F.cross_entropy(..., ignore_index=-100, reduction='mean')` 的 loss 和梯度分别对齐；
- 忽略位置：`local_grad[1]` 的非零个数必须为 0。

四个检查名是 `relu_gradient_match`、`cross_entropy_loss_match`、`cross_entropy_gradient_match`、`ignored_position_zero`；任一失败抛 `AssertionError`，全过返回 `status: passed`、`check_kind: framework_conformance` 和 torch 版本。用 `float64` 是为了让舍入误差远小于 `1e-12` 的容差，避免把"浮点误差"误判成"实现错误"；如果你用 `float32` 跑同样的对比，`1e-12` 是不可能达到的。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| 局部反向 | 一个算子只负责把上游梯度乘上自己的导数 |
| 门控 | 用前向输入的符号决定梯度是否通过 |
| ReLU | $\max(x,0)$，负区间梯度为 0 |
| 指示函数 $\mathbf{1}[x>0]$ | 条件成立取 1，否则取 0 |
| Softmax | 把 logits 变成和为 1 的概率 |
| logits | 模型输出的未归一化分数 |
| `log_softmax` | 数值稳定的对数概率，内部先减最大值 |
| one-hot | 正确类别为 1、其余为 0 的目标分布 |
| `ignore_index` | 不参与 loss、梯度和 `mean` 分母的标签值，默认 `-100` |
| 有效监督位置 $|\mathcal V|$ | 真正参与计算的回答 token 数 |
| `mean` / `sum` 归约 | 除以有效位置数 / 保留总和 |
| 梯度隔离 | 忽略位置的整行梯度必须为 0 |

## 小白常见疑问

**Q1：ReLU 在 0 点的导数，为什么不能取 1？**

数学上 0 点不可导，取 0 或 1 都是约定。PyTorch 的 `F.relu` 取 0，本节跟随它；测试 `test_relu_backward` 既用 `torch.allclose(x.grad, manual)` 对照自动求导，又单独断言 `manual[2].item() == 0.0`。如果你写成 `x >= 0`，第一项可能仍然过（因为自动求导也取 0），但第二项会失败。

**Q2：为什么 `mean` 的分母是有效位置数，不是 batch 大小？**

因为 `ignore_index` 的位置没有监督目标，不应该占平均的分母。测试里 batch 是 3，但只有 2 个有效位置，`mean_loss * 2 == sum_loss`；如果除以 3，这个关系就不成立。notebook 的 Step 3 公式里 $|\mathcal V|$ 就是这个意思，代码里对应 `valid_count = valid_mask.sum()`。

**Q3：为什么用 `F.log_softmax`，不先 `softmax` 再 `log`？**

因为 `softmax` 会先算指数。logits 里有 1000 时，$e^{1000}$ 超出浮点范围变成 `inf`，再 `log` 就得到 `inf` 或 `nan`。`log_softmax` 内部用 `logsumexp`，而 `logsumexp` 会先减去最大值，结果有限。测试 `test_reduction_and_stability` 用 `[[1000.0,0.0,-1000.0]]` 检查 loss 和梯度都有限。更一般地说，凡是"先指数再取对数"的写法，都应该换成对应的稳定组合（`log_softmax`、`logsigmoid`、`logsumexp`），这是数值实现的基本习惯。

**Q4：忽略位置的 `probs` 不是还有值吗，为什么梯度能整行为 0？**

`probs = log_probs.exp()` 在忽略位置确实有值，但 `grad = (probs - one_hot) * valid_mask.unsqueeze(1)` 最后乘了 mask，整行被清零；同时 `one_hot` 在忽略位置也已经被乘成 0。两道保险确保 loss 和梯度都不受忽略位置影响。`test_cross_entropy_gradient` 直接断言 `torch.count_nonzero(manual_grad[1]).item() == 0`。

**Q5：`sum` 和 `mean` 只是差一个常数，为什么还要分别测？**

因为差的不只是显示单位：`sum` 的梯度会随有效 token 数增大而增大，`mean` 则做了归一化。如果实现里 `mean` 忘了除分母，或者 `sum` 多除了一次，loss 可能看起来"差不多"，但梯度尺度已经错了。`test_reduction_and_stability` 用 `sum_loss == mean_loss * valid_count` 和 `sum_grad == mean_grad * valid_count` 同时锁住两者，`atol=1e-6`。

## 动手验证：怎么知道自己做对了

测试是直接调用，不是 pytest；TODO 未填会抛 `NotImplementedError`。四个测试跑完打印 `✅ 激活与损失的局部反向机制测试通过。`

- `test_relu_backward`：`x=[-2.0,-0.5,0.0,1.0,3.0]`、上游 `[0.5,-1.0,2.0,0.25,-0.75]`；`torch.allclose(x.grad, manual)` 必须为真；`manual[2].item() == 0.0`。
- `test_cross_entropy_gradient`：三行 logits 与 labels `[0,-100,2]`；手写 loss 与 `F.cross_entropy(..., ignore_index=-100)` 在 `atol=1e-6` 内一致；手写梯度与 `logits.grad` 一致；`manual_grad[1]` 非零元素个数为 0。
- `test_reduction_and_stability`：`sum_loss ≈ mean_loss * valid_count`、`sum_grad ≈ mean_grad * valid_count`（`atol=1e-6`）；`[[1000.0,0.0,-1000.0]]` + label 0 时 loss 和梯度都有限。
- `test_activation_loss_input_contract`：标签 3 越界、`[-100,-100]` 全忽略、`relu_backward` 形状不一致，三个非法调用都必须抛 `ValueError`。

可选的 `check_activation_loss_framework_conformance` 会返回一个字典，包含 `status: passed`、`check_kind: framework_conformance`、`versions: {'torch': ...}` 和四个布尔检查。它用 `float64` 和 `atol=1e-12, rtol=1e-12`，比主测试的 `1e-6` 严得多；如果这里失败，通常不是浮点误差，而是符号、mask 或分母写错了。排查时按检查名定位：`relu_gradient_match` 失败看门控和 0 点约定；`cross_entropy_loss_match` 失败看 one-hot、log_softmax 和分母；`cross_entropy_gradient_match` 失败看 `probs - one_hot` 的符号和 mask；`ignored_position_zero` 失败看 `valid_mask.unsqueeze(1)` 有没有漏乘。

notebook 没有保存任何执行输出（最后一个 `framework_check` 表达式也没有落盘结果），所以上面提到的 0.6459、0.4644、0.5551 都是按测试输入手算的演示值，不是你机器上的实测记录。这些测试全部是 CPU 小张量，不需要 GPU、不需要下载数据，跑一遍几秒钟；它们验证的是"局部梯度公式对不对"，不是"模型训练效果好不好"。如果四个测试和框架核对都通过，你会依次看到主测试打印的 ✅ 和核对字典里的 `status: passed`；只要有一项失败，先按上面的错误清单定位，不要靠调大容差把它"调过"。判断标准以测试断言和你运行时打印的 ✅ 为准；如果只跑一个测试，建议先跑 `test_cross_entropy_gradient`，它同时覆盖 loss、梯度和忽略位置三件事。等这些局部测试都稳定了，再去看完整训练 loss 曲线，才不会把"数据问题"误判成"反向公式问题"。
