# 11. Gradient Accumulation | 梯度累积 · 小白入门

> 配套 notebook：[11. 梯度累积](../11_Gradient_Accumulation.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

训练时显存装不下一整批数据，梯度累积就把这批数据分成几小批依次计算，把每小批算出的梯度攒起来，攒够再更新一次参数。这本 notebook 教你让分批算出的结果和整批一次算的一样。

## 为什么需要它

训练的每一步都一样：拿一批样本过一遍模型，算出损失和梯度，再用梯度更新参数。样本越多，前向传播留下的中间结果越多，显存占用也越大。

假设你想让一次参数更新看到 8 个样本，但显存一次只放得下 2 个。有三种做法：

- 直接改成每批 2 个样本。显存够了，但一次更新只看 2 个样本，和你想要的 8 个不是一回事。
- 分成 4 小批，每小批算完就更新一次参数。参数被更新了 4 次，而不是 1 次。
- 分成 4 小批，梯度攒起来，最后更新一次。但每小批的损失只对自己的 2 个样本求平均，梯度直接相加，大小就错了。

梯度累积是第三种做法的修正版。每小批叫一个微批次（micro-batch），4 个微批次合起来看到的 8 个样本叫有效批次（effective batch）。每个微批次单独做前向传播和反向传播，梯度不清空，留在显存里累加，4 个都算完才更新一次参数。同一时刻显存里只有 2 个样本的激活值（activation），也就是前向传播留下、反向传播还要用的中间结果。

这本 notebook 讲的就是其中的三个细节：微批次怎么切、损失怎么缩放、什么时候到了更新边界。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 微批次（micro-batch） | 一次前向传播和反向传播实际处理的样本，是计算单位 |
| 有效批次（effective batch） | 一次参数更新合起来看到的全部样本，等于微批次大小 × 累积次数 |
| 累积次数 | 一个逻辑批次被切成几个微批次，代码里叫 `accum_steps`，公式里记作 `K` |
| 激活值（activation） | 前向传播算出的中间结果，反向传播还要用，所以得先留在显存里 |
| 梯度缓冲区 | 每个参数旁边存放梯度的位置，代码里是 `w.grad` 这样的字段；反向传播往里累加，`zero_grad()` 把它清零 |
| 损失口径 | 损失是求和还是求平均、对哪些样本求平均。notebook 用 `nn.MSELoss(reduction="mean")`，对一个批次内的样本求平均 |
| 半开区间 | `[start:end]` 这种切片写法：包含 `start`，不包含 `end` |

## 核心思路

输入是一个逻辑批次的全部样本，处理分三步：切成 K 个微批次；逐个做前向传播和反向传播，反向传播之前把损失乘上这个微批次的权重，梯度不清空、直接累加；K 个都做完，调用一次 `optimizer.step()` 更新参数，再清空梯度。输出是一次参数更新，看到的样本数和整批一次算相同。

损失要乘的权重，notebook 给的公式是：

```text
grad L = sum_i (n_i / N) * grad L_i        N = sum_i n_i
```

`K` 是累积次数，`n_i` 是第 `i` 个微批次的样本数，`N` 是有效批次的样本总数，`L_i` 是这个微批次自己的样本平均损失，`grad L_i` 是它算出的梯度。微批次等长时 `n_i = N / K`，权重就是 `1 / K`。

下面用一个只有参数 `w` 的模型试一下。损失对批次内的样本求平均，`w` 从 0 开始，4 个样本先整批算，再切成 2 个微批次累加：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])          # 目标是 y = 2x

def accumulate(scale):                          # 4 个样本切成 2 个微批次
    w = torch.tensor(0.0, requires_grad=True)   # 两个微批次共用同一个 w
    for s in (0, 2):
        loss = ((w * x[s:s + 2] - y[s:s + 2]) ** 2).mean()   # 这个微批次的平均损失
        (loss * scale).backward()               # 梯度加进 w.grad，中间不清空
        print("  微批次", s // 2, "之后 w.grad =", w.grad.item())
    return w.grad.item()

w = torch.tensor(0.0, requires_grad=True)       # 对照：完整批次一次算完
((w * x - y) ** 2).mean().backward()
print("完整批次的梯度 =", w.grad.item())
for scale in (0.5, 1.0):
    print("scale =", scale)
    print("  最终梯度 =", accumulate(scale))
```

```text
完整批次的梯度 = -30.0
scale = 0.5
  微批次 0 之后 w.grad = -5.0
  微批次 1 之后 w.grad = -30.0
  最终梯度 = -30.0
scale = 1.0
  微批次 0 之后 w.grad = -10.0
  微批次 1 之后 w.grad = -60.0
  最终梯度 = -60.0
```

第 0 个微批次自己的梯度是 `-10`，乘 `1/2` 加进 `w.grad` 是 `-5`；第 1 个做完是 `-30`，和完整批次相同。不缩放时直接相加是 `-60`，是完整批次的 2 倍。

实际训练时逻辑批次一个接一个，所以要数已完成的微批次数：只有它是 `accum_steps` 的整数倍，才调用 `optimizer.step()` 并清空梯度。Step 3 的表格还列了别的对齐条件，不满足时即使样本相同也可能无法严格等价。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 微批次、累积次数、有效批次和参数更新之间的关系 | 读，对照上面的公式 |
| Step 2 | 梯度累积只降低激活值峰值，参数、优化器状态和梯度缓冲区仍然驻留 | 读 |
| Step 3 | 累积梯度要和完整批次对齐，需要满足哪些条件 | 读 |
| Step 4 · TODO 1 | 同步切分微批次 | 在 `slice_micro_batch` 里补 3 行：算出 `start` 和 `end`，所有字段用同一个 `[start:end]` |
| Step 4 · TODO 2 | 缩放平均损失 | 在 `train_step_with_accumulation` 里补 1 行：损失除以 `accum_steps`，得到 `scaled_loss` |
| Step 4 · TODO 3 | 识别有效更新边界 | 在 `is_update_boundary` 里补 1 行：已完成次数是 `accum_steps` 的整数倍才返回真 |
| 测试 | 检查字段对齐、更新等价、边界判断和非法输入 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 Accelerate 的梯度累积上下文核对更新边界和更新后的参数 | 需要 `TRAIN-CPU` 环境，可以先跳过 |
| Step 5（可选） | 在 GPU 上比较 1 / 2 / 4 次累积的更新一致性、峰值显存和 step 时间 | 需要单张 CUDA GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `TRAIN-CPU` 环境。Step 5 需要 GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **微批次的损失没有缩放**：梯度直接相加会变大。等长微批次要把损失乘 `1 / accum_steps`，不等长时权重是 `n_i / N`。
- **每个微批次做完就更新参数**：所有微批次都贡献梯度之后才调用一次 `optimizer.step()`，测试要求调用次数必须是 1。
- **中途清空了梯度**：梯度缓冲区要保留到这次有效更新完成，参考代码只在开头和更新之后各清一次。
- **各字段的切片范围不一致**：输入、掩码、监督目标必须用同一个 `[start:end]`，否则样本和答案错位。
- **以为它省掉了所有显存**：降低的只是激活值峰值，参数、梯度和优化器状态仍然驻留。notebook 的 GPU 实验里，序列长度 128 时峰值就没有下降。

## 读完后自查

- [ ] 能说出微批次、累积次数、有效批次三者的关系。
- [ ] 能解释例子里第 0 个微批次之后 `w.grad` 为什么是 `-5`。
- [ ] 能说出一次有效更新里 `zero_grad()`、`backward()`、`step()` 各在什么时候调用。
- [ ] 能说出梯度累积降低了哪部分显存，哪部分没有。

## 下一步

- 打开 notebook：[11. 梯度累积](../11_Gradient_Accumulation.ipynb)
- 需要补基础：[优化器与损失](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)、[数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
- 做完之后：[TRAIN-MIXED-PRECISION-BACKWARD. 混合精度反向传播与梯度安全 · 小白入门](./TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.md)
- 还可以接着读：[12. 优化器更新与学习率调度](../12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)、[15. SFT 有效更新与训练闭环](../15_End_to_End_Fine_Tuning_Experiment.ipynb)
