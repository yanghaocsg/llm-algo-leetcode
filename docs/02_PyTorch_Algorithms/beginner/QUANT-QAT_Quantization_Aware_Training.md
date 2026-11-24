# QUANT-QAT. Quantization Aware Training | 量化感知训练 · 小白入门

> 配套 notebook：[QUANT-QAT. 量化感知训练](../QUANT-QAT_Quantization_Aware_Training.md) ｜ 难度：Hard ｜ 环境：CPU-first，GPU 可选

## 一句话说清

量化感知训练是在训练时就让模型提前见到量化带来的误差，再用一个近似的梯度照常更新权重。这本 notebook 带你写出其中的范围统计、误差模拟和梯度传递。

## 为什么需要它

通常的做法是先训练、后量化（训练后量化）。模型从没见过量化误差，压得太狠时，质量容易掉下来。

想改成在训练里模拟量化，马上会遇到两个麻烦：

- 取整这一步没法求梯度。量化里有一步“四舍五入”，它的梯度几乎处处是 0。照这样反向传播，权重一点都更新不了。
- 范围一直在变。训练时每一批激活值的大小都不同，要有一个地方持续记着范围；到了部署，这个范围又必须固定下来。

量化感知训练的做法是：保留一份浮点主权重交给优化器更新；前向传播时用“伪量化”模拟裁剪、取整和恢复；反向传播时用直通估计器跳过取整，让梯度沿着裁剪后的浮点值传回去。这本 notebook 实现三个部件：记范围的观测器、带直通估计器的伪量化、以及对激活和权重分别做伪量化的线性层。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 训练后量化（post-training quantization，PTQ） | 模型训练完以后，再把它转成低比特 |
| 浮点主权重（master weight） | 优化器真正更新的那份浮点权重，部署时会被转换成量化权重 |
| 伪量化（fake quantization） | 在浮点计算里模仿“裁剪、取整、再恢复”，让损失看到量化误差，并没有真的变成整数 |
| 统计观测器（observer） | 训练时负责记录数值范围（绝对最大值）的对象，范围之后会固化成量化参数 |
| 指数移动平均（exponential moving average，EMA） | 旧值乘 `momentum`，新值乘剩下的比例，再相加，用来抹平一批一批的波动 |
| 直通估计器（Straight-Through Estimator，STE） | 前向用取整后的值，反向装作取整不存在，让梯度直接通过 |
| 可表示区间与饱和区 | 落在统计范围内的值叫可表示区间；超出范围、会被截到边界的值叫饱和 |

## 核心思路

输入是训练时流过某一层的张量 `x`，可能是激活，也可能是权重。三个部件各管一件事：

```text
观测器   首次      absmax = 当前张量的绝对最大值
         之后      absmax = momentum * absmax + (1 - momentum) * 当前张量的绝对最大值
伪量化   scale       = qmax / absmax
         clipped     = clamp(x, -absmax, absmax)              先裁剪到可表示区间
         dequantized = round(clipped * scale) / scale         再取整，并恢复成浮点数
STE      fake_quantized = clipped + (dequantized - clipped).detach()
```

符号说明：`absmax` 是观测器记下的范围；`qmax` 是最大的整数码，默认 127；`momentum` 是平滑系数，默认 0.9；`clamp` 把值限制在区间内；`detach()` 让括号里的差值不带梯度，所以反向传播时梯度只沿着 `clipped` 走。注意这里的 `scale` 是 `qmax / absmax`，和 25、52 两本 notebook 里的缩放因子（范围除以 `qmax`）互为倒数。

下面用 `momentum = 0.5` 走一遍。先是观测器的三次更新，再用固定的范围 1 对 `[-2.0, 0.5]` 做伪量化，比较“不用 STE”和“用 STE”的梯度：

```python
import torch
torch.set_printoptions(precision=4)

absmax, momentum = None, 0.5
for current in (2.0, 1.0, 4.0):                    # 训练时连续三次观察到的绝对最大值
    absmax = current if absmax is None else momentum * absmax + (1 - momentum) * current
    print(absmax)

def fake_quant(x):                                 # 范围固定为 1（评估时复用统计）
    clipped = torch.clamp(x, -1.0, 1.0)
    dequantized = torch.round(clipped * 127) / 127
    return dequantized, clipped + (dequantized - clipped).detach()

x = torch.tensor([-2.0, 0.5], requires_grad=True)
plain, ste = fake_quant(x)
plain.sum().backward()                             # 不用 STE：直接对取整结果反向传播
print(plain.detach(), x.grad)

x = torch.tensor([-2.0, 0.5], requires_grad=True)
plain, ste = fake_quant(x)
ste.sum().backward()                               # 用 STE
print(ste.detach(), x.grad)
```

```text
2.0
1.5
2.75
tensor([-1.0000,  0.5039]) tensor([0., 0.])
tensor([-1.0000,  0.5039]) tensor([0., 1.])
```

读三处：

- 前三行是观测器：第一次直接记下 2.0，之后是 0.5 × 2.0 + 0.5 × 1.0 = 1.5，再是 0.5 × 1.5 + 0.5 × 4.0 = 2.75。
- 第四行不用 STE：前向值是对的，但梯度全是 0，训练进行不下去。
- 第五行用 STE：前向值完全一样，梯度变成 0 和 1。输入 -2.0 超出范围，被截到 -1，所以梯度是 0；输入 0.5 在范围内，梯度原样通过。

线性层里激活和权重各有一个观测器。训练时更新范围，评估时只复用。优化器更新的始终是浮点主权重，不是取整后的值。训练结束后还要冻结量化参数、导出部署产物，伪量化节点在部署时会被真正的量化算子取代。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 浮点主权重、观测器统计、伪量化节点、任务损失四种状态在训练时做什么、部署时去哪里 | 读 |
| Step 2 | 舍入的梯度为什么是 0，直通估计器怎样让反向传播继续 | 读，对照上面的小例子 |
| Step 3 | 观测器冻结时机、粒度、目标数据类型等为什么要写进产物清单 | 读 |
| Step 4 · TODO 1 | 用指数移动平均维护训练期绝对最大值 | 在 `update_observer` 里写 1 行：首次直接赋值，之后用 `momentum` 平滑更新 |
| Step 4 · TODO 2 | 前向使用离散恢复值，反向沿裁剪值传播近似梯度 | 在 `fake_quant` 里写 1 行：组合 `clipped` 和停掉梯度的量化残差 |
| Step 4 · TODO 3 | 分别伪量化激活和权重后执行线性运算 | 在 `forward` 里写 3 行：激活和权重各伪量化一次，再做线性计算 |
| 测试 | 检查观测器更新后冻结、区间内和饱和区的梯度、训练与评估前向、浮点主权重是否被更新 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实的 Transformer Block 上比较直接做训练后量化和三档训练预算的量化感知训练 | 需要 GPU 和 `Q-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `Q-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **直接对取整结果反向传播**：`round` 的梯度几乎处处是 0，训练会停住。前向用恢复值，反向沿裁剪后的浮点值传梯度；超出范围的值梯度是 0。
- **评估时还在更新范围**：观测器只在训练时更新（`self.training` 为真），评估前向只复用已有范围。否则新来的异常值会改写已经定好的量化参数。
- **第一次观察也做了平滑**：首次观察要直接赋值，之后才用 `momentum` 更新。`momentum` 要满足 0 ≤ `momentum` < 1。
- **让优化器去更新取整后的权重**：伪量化只发生在前向计算里，被更新的始终是浮点主权重 `self.weight`。
- **把训练图当成量化产物**：训练结束后要冻结量化参数并导出。训练恢复的质量，可能在导出后丢失。

## 读完后自查

- [ ] 能说出训练后量化和量化感知训练的区别。
- [ ] 给连续三次观察到的绝对最大值和 `momentum`，能算出观测器里的范围。
- [ ] 能解释为什么不能直接对取整结果反向传播，以及直通估计器怎样绕过去。
- [ ] 能说出训练和评估时观测器的行为有什么不同。

## 下一步

- 打开 notebook：[QUANT-QAT. 量化感知训练](../QUANT-QAT_Quantization_Aware_Training.md)
- 建议先读：[52. 量化校准范围与裁剪 · 小白入门](./52_Quantization_Calibration_and_Error.md)、[25. W8A16 量化 · 小白入门](./25_Quantization_W8A16.md)、[11. 梯度累积 · 小白入门](./11_Gradient_Accumulation.md)
- 做完之后：[40. GPTQ 与 AWQ 权重量化 · 小白入门](./40_GPTQ_and_AWQ_Weight_Quantization.md)
