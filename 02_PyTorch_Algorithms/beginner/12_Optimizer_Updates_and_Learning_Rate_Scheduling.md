# 12. Optimizer Updates and Learning Rate Scheduling | 优化器更新与学习率调度 · 小白入门

> 配套 notebook：[12. 优化器更新与学习率调度](../12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

学习率决定每次更新参数时走多大一步。这本 notebook 教你按训练进度调整它：先从小到大热身，中间保持不变，最后逐步减小，并且只在参数真正更新后才往前走一步。

## 为什么需要它

优化器（optimizer）负责沿着梯度修改参数，学习率（learning rate）决定这一次走多大一步。学习率从头到尾用同一个数并不合适，notebook 指出了三个阶段：

- 训练初期，模型内部的统计量和参数方向还不稳定。
- 主训练阶段，需要保持足够的更新幅度。
- 接近终点，要逐步降低步幅，才便于收敛（converge）。

WSD（Warmup–Stable–Decay）把这三段接成一条曲线：Warmup 从 0 升到基础学习率，Stable 保持不变，Decay 再降下来。

还有一个问题：这条曲线按什么往前走？按参数真正更新的次数走才对。有两种情况会数错：

- 梯度累积时，4 个微批次（micro-batch）才合成一次更新。如果每个微批次都让曲线走一步，参数才更新了 1 次，曲线已经走了 4 步。
- 混合精度（mixed precision）训练里梯度溢出，这一步被跳过，参数没动。如果曲线照走，学习率就领先于参数的真实进度。

这本 notebook 做两件事：实现这条曲线，并保证它只在成功更新之后推进。notebook 把这个计数叫调度时钟。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 优化器（optimizer） | 决定怎样沿着梯度修改参数的程序。`optimizer.step()` 执行一次参数更新 |
| 学习率调度器（scheduler） | 每次更新后给出下一个学习率的程序，`scheduler.step()` 让它往前走一步。学习率（learning rate）是每次更新走多大的一步，基础学习率是曲线的最高点 `eta_max` |
| 微批次（micro-batch） | 一次前向传播和反向传播实际处理的样本。梯度累积时，几个微批次才合成一次有效更新 |
| WSD（Warmup–Stable–Decay） | 把学习率分成三段的曲线：Warmup 升上去，Stable 保持，Decay 降下来 |
| 余弦衰减（cosine decay） | Decay 段的下降方式：学习率沿余弦曲线平滑下降，进度 `u` 从 0 到 1 |
| 收敛（converge） | 训练后期参数逐渐稳定下来的过程。notebook 说降低步幅是为了便于收敛 |
| 调度时钟 | 学习率曲线走到第几步的计数。只数成功的参数更新，也叫有效更新；参数没动的跳过更新不数 |

## 核心思路

输入是当前成功更新的编号 `t` 和三个阶段的配置。先判断 `t` 落在哪一段，再按那一段的规则算学习率。输出是第 `t` 次更新要用的学习率 `eta_t`：

```text
Warmup（0 <= t < T_w）：          eta_t = eta_max * t / (T_w - 1)
Stable（T_w <= t < T_w + T_s）：  eta_t = eta_max
Decay（之后）：                   eta_t = r * eta_max + (1 - r) * eta_max * (1 + cos(pi * u_t)) / 2
                                 u_t = (t - T_w - T_s) / (T_d - 1)
```

`t` 从 0 数起，`eta_max` 是基础学习率，`T_w`、`T_s`、`T_d` 是三段各有几次更新，`r` 是最小比例（代码里叫 `min_lr_ratio`，最后降到基础学习率的几成），`u_t` 是 Decay 段内的进度。`cos` 是余弦函数：`u_t` 为 0 时括号里是 2，系数是 1；`u_t` 为 1 时括号里是 0，系数是 0。学习率就这样从基础学习率降到 `r * eta_max`。

分母减 1 是为了让端点对齐：多步阶段的首步进度是 0，末步进度正好是 1。某一段只有 1 步时分母是 0，所以单独处理：单步 Warmup 直接用基础学习率，单步 Decay 直接到最小学习率。

下面用小数字把两个要点走一遍：曲线长什么样，跳过更新时时钟怎么走。基础学习率取 1.0，最小比例 0.1，三段各 3、2、3 步：

```python
import math

base_lr, r = 1.0, 0.1                   # 基础学习率、最小比例
T_w, T_s, T_d = 3, 2, 3                # 三个阶段的更新数

lrs = []
for t in range(T_w + T_s + T_d):       # t：第几次成功更新，从 0 数起
    if t < T_w:
        lr = base_lr * t / (T_w - 1)                       # Warmup：从 0 升到基础学习率
    elif t < T_w + T_s:
        lr = base_lr                                      # Stable：保持不变
    else:
        u = (t - T_w - T_s) / (T_d - 1)                   # Decay 进度：从 0 到 1
        lr = r * base_lr + (1 - r) * base_lr * (1 + math.cos(math.pi * u)) / 2
    lrs.append(lr)
print([round(lr, 2) for lr in lrs])

step = 0                                # 调度时钟：数成功更新的次数
for ok in [True, True, False, True]:    # 每次尝试更新：成功还是跳过
    if ok:
        print("成功，用第", step, "步的学习率", round(lrs[step], 2))
        step += 1
    else:
        print("跳过，时钟停在", step)
```

```text
[0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.55, 0.1]
成功，用第 0 步的学习率 0.0
成功，用第 1 步的学习率 0.5
跳过，时钟停在 2
成功，用第 2 步的学习率 1.0
```

列表里的 8 个数是 8 次成功更新各用的学习率：前 3 步从 0 升到 1，接着 2 步保持 1，最后 3 步降到 0.1。后半段模拟了 4 次尝试更新：第 3 次被跳过，时钟停在 2，第 4 次成功时用的还是第 2 步的学习率。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一次更新里的四个事件：哪个改变参数，哪个推进调度时钟 | 读 |
| Step 2 | Warmup、Stable、Decay 三段的公式和各自要验证的边界 | 读，对照上面的公式 |
| Step 3 | 为什么调度时钟只数成功更新，不数微批次和跳步 | 读，对照例子的后半段 |
| Step 4 · TODO 1 | 计算 Warmup 进度 | 在 `_lr_for_step` 的 Warmup 分支里补 1 行：`step` 除以 `num_warmup_steps - 1` |
| Step 4 · TODO 2 | 计算余弦衰减 | 在 `_lr_for_step` 的 Decay 分支里补 2 行：先算 `decay_progress`，再算 `cosine_factor` |
| 测试 | 检查非法配置是否报错、Warmup 的起点终点和 Stable、余弦衰减的边界和单调性，以及调度时钟是否只随有效更新推进 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 PyTorch `LambdaLR` 表达同一条 WSD 曲线，核对每步学习率和调度时钟 | 需要 `TRAIN-CPU` 环境，可以先跳过 |

整本 notebook 只需要 CPU（`BASE-CPU` 环境）。框架核对需要另外的 `TRAIN-CPU` 环境，可以先跳过。

## 容易踩的坑

- **每个微批次都推进调度器**：`scheduler.step()` 只在一次有效更新之后调用，紧跟 `optimizer.step()`。微批次不单独推进调度器，否则学习率曲线比参数更新更快走完。
- **跳步时也推进调度器**：梯度无效、参数没动时，调度时钟保持不变。否则学习率进度会领先于真实的参数状态。
- **进度没有对齐端点**：`t` 从 0 数起，多步阶段的首步进度要是 0，末步要是 1，所以分母要减 1。Warmup 的第一步学习率因此是 0。
- **单步阶段也去算进度**：某一段只有 1 步时，分母是 0。骨架已经把这种情况单独分出来，你补的公式只在多步阶段执行。
- **接受了非法配置**：阶段长度必须是非负整数，总步数不能为 0，`min_lr_ratio` 要在 0 到 1 之间，否则要抛出 `ValueError`。

## 读完后自查

- [ ] 能说出 WSD 三段各自的学习率怎么变。
- [ ] 基础学习率是 1.0、Warmup 有 3 步，能写出这 3 步的学习率。
- [ ] 能解释进度公式的分母为什么减 1，单步阶段怎么处理。
- [ ] 能说出调度时钟为什么只数成功更新，微批次和跳步算不算。

## 下一步

- 打开 notebook：[12. 优化器更新与学习率调度](../12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)
- 需要补基础：[优化器与损失](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)、[数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
- 做完之后：[55. 训练更新动力学 · 小白入门](./55_Training_Update_Dynamics.md)
- 还可以回看：[11. 梯度累积](../11_Gradient_Accumulation.ipynb)
