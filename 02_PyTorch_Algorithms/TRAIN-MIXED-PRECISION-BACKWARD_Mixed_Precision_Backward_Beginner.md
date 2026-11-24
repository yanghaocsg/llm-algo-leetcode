# TRAIN-MIXED-PRECISION-BACKWARD. Mixed Precision Backward Beginner | 混合精度反向传播与梯度安全 · 小白入门

> 配套 notebook：[TRAIN-MIXED-PRECISION-BACKWARD. 混合精度反向传播与梯度安全](./TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

用低精度数字训练时，太小的梯度会被当成 0。这本 notebook 教你先把损失放大再算梯度，更新前再除回来；遇到溢出，就跳过这次更新。

## 为什么需要它

训练里的数字可以用不同格式存。FP32 范围宽、精度高，但计算和状态的代价也高。改用 FP16 这种低精度格式，激活值占的字节更少，还能用上针对低精度计算的硬件单元 Tensor Core。代价是 FP16 能表示的数的范围窄。

于是有两种麻烦：

- 梯度很小，FP16 存不下，直接变成 0。这叫下溢（underflow）。
- 中间值太大，变成 Inf。这叫溢出（overflow），这样的梯度不能拿来更新参数。

对第一种麻烦，办法叫损失缩放（loss scaling）：反向传播之前先把损失乘一个缩放因子 `s`，梯度跟着放大 `s` 倍，就离开了下溢区。但缓冲区里的梯度因此是放大过的，更新前必须除以 `s` 还原。如果放大之后还是溢出了，就跳过这次更新，并把 `s` 调小。

这本 notebook 做的是更新之前的收尾：把梯度还原成真实尺度，算出所有梯度合起来有多大，再决定这一步是更新还是跳过（skip）。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 微批次（micro-batch） | 一次前向传播和反向传播实际处理的样本。几个微批次的梯度累加起来，才合成一次有效更新 |
| 混合精度（mixed precision） | 同一次训练里，一部分计算用 FP16 或 BF16 这类低精度格式，另一部分保留 FP32。`autocast` 是 PyTorch 里按算子自动选精度的开关 |
| 下溢（underflow）/ 溢出（overflow） | 数太小，格式存不下，变成 0，叫下溢；数太大，变成 Inf，叫溢出。FP16 的指数范围窄，两种情况都容易出现 |
| 损失缩放（loss scaling） | 反向传播前把损失乘上缩放因子 `s`（大于 0），梯度也跟着放大 `s` 倍，小梯度就不容易下溢 |
| 缩放梯度 `g'` / 真实梯度 `g` | `g'` 是参数缓冲区里放大后的梯度。除以 `s` 得到 `g`，这一步叫还原尺度（unscale）；只有 `g` 能用来检查、裁剪和更新 |
| 有限值（finite） | 不是 NaN 也不是 Inf 的数。任一梯度不是有限值，这一步就不能更新 |
| 全局范数（global norm）/ 梯度裁剪（gradient clipping） | 全局范数是把所有参数的梯度看成一个长向量算出的 L2 长度：各元素平方相加，再开方。它超过阈值 `max_norm` 时，所有梯度乘上同一个小于 1 的系数，就是裁剪 |

## 核心思路

输入是一个微批次（micro-batch）。前向传播放在 `autocast` 里做，由算子决定用低精度还是 FP32；算出损失后先乘 `s`，再做反向传播。这样参数缓冲区里存的是放大后的梯度 `g'`。到了更新边界再按固定顺序收尾，输出是一次更新，或者一次跳过。

notebook 用三个式子表示缩放和还原：

```text
L' = s * L          g' = s * g          g = g' / s
```

`L` 是原来的损失，`s` 是缩放因子，`L'` 是放大后的损失，`g` 是真实梯度，`g'` 是缩放梯度。只有 `g` 能用来做有限值检查（finite check）、裁剪和更新。

一次有效更新由 K 个微批次组成时，每个微批次都只做放大后的反向传播，`g'` 在参数上一直累加；到第 K 个微批次结束才统一收尾。哪个操作放在哪做，notebook 列了一张表：

```text
每个微批次：autocast 前向传播 → 损失乘 s → 反向传播
更新边界：  除以 s 还原 → 有限值检查 → 梯度裁剪 → optimizer.step() 或跳过 → 调整 s、清空梯度
```

BF16 的指数范围接近 FP32，常用路径通常不需要缩放。但它仍可能产生 NaN 或 Inf，所以有限值检查照样要做。

下面用小数字把收尾的几步走一遍：

```python
import torch

g = 1e-8                                          # 一个很小的真实梯度
fp16 = torch.tensor(g, dtype=torch.float16)
print("直接转 FP16：", f"{fp16.item():.3e}")
fp16 = torch.tensor(g * 1024, dtype=torch.float16)   # 先放大 1024 倍再转
print("放大后：", f"{fp16.item():.3e}", " 除回 1024：", f"{fp16.item() / 1024:.3e}")

scaled = torch.tensor([6.0, 8.0])                 # 缓冲区里的缩放梯度 g'
true = scaled / 2.0                               # 还原：除以缩放因子
norm = true.square().sum().sqrt()                 # 全局范数
coef = min(1.0, 2.5 / (float(norm) + 1e-12))      # 裁剪系数，上限是 1.0
print("真实梯度：", true, "范数：", norm, "裁剪系数：", round(coef, 3))
print("裁剪后：", true * coef)
print("梯度含 Inf 时是否有限：", bool(torch.isfinite(torch.tensor([float("inf")])).all()))
```

```text
直接转 FP16： 0.000e+00
放大后： 1.025e-05  除回 1024： 1.001e-08
真实梯度： tensor([3., 4.]) 范数： tensor(5.) 裁剪系数： 0.5
裁剪后： tensor([1.5000, 2.0000])
梯度含 Inf 时是否有限： False
```

前两行是放大再还原的效果：`1e-8` 直接转成 FP16 是 0，放大 1024 倍后能存下，除回去又接近原来的值。后三行是收尾：缩放梯度 `[6, 8]` 除以 2 得到真实梯度 `[3, 4]`，全局范数（global norm）是 5，超过阈值 2.5，于是乘 0.5 缩小到 `[1.5, 2]`。含 Inf 的梯度通不过有限值检查。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 低精度算出的梯度为什么要先放大、再还原，才能进入参数更新 | 读，对照上面的三个式子 |
| Step 2 | FP16、BF16、FP32 各自的收益、风险和常见保护 | 读 |
| Step 3 | 还原、检查、裁剪、更新这些操作，哪些每个微批次做，哪些只在更新边界做 | 读，对照上面的两行流程 |
| Step 4 · TODO 1 | 恢复真实梯度尺度 | 在 `unscale_gradients` 里补 1 行：列表推导，`None` 保持 `None`，其余除以 `scale` |
| Step 4 · TODO 2 | 累积真实梯度的平方和 | 在 `global_grad_norm` 里补 1 行：先 `.float()` 转成 FP32，再把平方和加进 `squared_sum` |
| Step 4 · TODO 3 | 检查所有真实梯度是否有限 | 在 `plan_mixed_precision_update` 里补 1 行：忽略 `None`，所有梯度都通过 `torch.isfinite` 才算有限 |
| 测试 | 检查还原后的数值和输入不被修改、跨参数的全局范数、裁剪系数，以及 Inf 触发跳步 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 PyTorch `GradScaler` 的 CPU 路径核对还原尺度和非有限梯度跳步 | 需要 `TRAIN-CPU` 环境，可以先跳过 |
| Step 5（可选） | 在 GPU 上核对 PyTorch AMP（自动混合精度）真实走的数据类型路径、溢出处理、时间和峰值显存 | 需要单张 CUDA GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `TRAIN-CPU` 环境。Step 5 需要 GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验。

## 容易踩的坑

- **直接拿缩放梯度去裁剪或更新**：`g'` 比真实梯度大 `s` 倍。更新前必须先除以 `s`，还原要在裁剪和更新之前完成。
- **先裁剪再检查有限值**：顺序要反过来。非有限梯度乘上裁剪系数，也变不回可靠的数；检查不通过就直接跳过更新。
- **每个微批次都还原或更新**：每个微批次只做放大后的反向传播。到第 K 个微批次结束才统一还原、检查、裁剪和更新，否则梯度的尺度和有效批次的意思都被破坏。
- **各张量分别算范数、分别裁剪**：裁剪作用于所有参数梯度合成的整体向量。要先把各张量的平方和加起来，再开方。
- **还原时改了输入张量**：要返回新张量，不能原地修改。`None` 要原样保留，它表示这个参数本轮没有梯度。

## 读完后自查

- [ ] 能按顺序说出更新边界的几件事：还原、有限值检查、裁剪、更新或跳过。
- [ ] 缩放因子是 2，缓冲区里的梯度是 `[6, 8]`，能写出真实梯度和它的范数。
- [ ] 能解释为什么损失要先乘 `s`，更新前又必须除以 `s`。
- [ ] 能说出梯度里出现 Inf 时这一步怎么做，缩放因子怎么变。

## 下一步

- 打开 notebook：[TRAIN-MIXED-PRECISION-BACKWARD. 混合精度反向传播与梯度安全](./TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb)
- 建议先读：[11. 梯度累积 · 小白入门](./11_Gradient_Accumulation_Beginner.md)、[09. 激活与损失反向传播 · 小白入门](./09_Activation_and_Loss_Backward_Beginner.md)
- 需要补基础：[Part 01 · 01：数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
- 做完之后：[12. 优化器更新与学习率调度 · 小白入门](./12_Optimizer_Updates_and_Learning_Rate_Scheduling_Beginner.md)
