# 41. FP8 运行时低精度路径 — 小白讲解

> 对应 notebook：[`41_FP8_Runtime_Low_Precision_Paths.ipynb`](./41_FP8_Runtime_Low_Precision_Paths.ipynb)

## 一句话总结
本节用 `FP8RuntimeSim` 在 CPU 上模拟 FP8 的动态缩放与舍入，并补全一个"运行时准入"判断：只有设备、数据类型和原生计算内核同时成立，才能声称走了 FP8 计算路径。三个 TODO 分别实现缩放因子、按尾数位构造的量化网格、以及从证据到路径状态的映射；最后的 GPU 实验再用 torchao 区分"权重真的存成了 FP8"和"矩阵乘真的用了 FP8 内核"。

## 为什么需要它：痛点是什么
初学者看到"FP8 能省一半显存、还能加速"，很容易直接下单：把权重转成 FP8，就以为推理会变快。实际会踩三个坑：第一，FP8 只改变存储，矩阵乘可能仍回退到 BF16/FP16，甚至因为多了一次类型转换而更慢；第二，E4M3 和 E5M2 都叫 FP8，但最大有限值分别是 448 和 57344，动态范围差很多；第三，缩放因子取错方向（该乘还是该除）会让数值整体差几个数量级。

生活类比：把大件行李塞进不同尺寸的柜子。E4M3 是格子多但柜子小的柜子，适合放普通衣物、叠得精细；E5M2 是格子少但柜子大的柜子，能塞进羽绒被，但叠得粗。缩放因子就是你选择"把行李压缩多少倍再放进去"；如果压缩完不记得乘回来，拿出来的东西尺寸就全错了。

## 核心问题
- 一次低精度矩阵乘到底涉及哪些对象？只量化权重够不够？
- E4M3 和 E5M2 的位分布差异，如何变成"精度"和"动态范围"的取舍？
- 动态缩放因子 $s=\max(a/f_{\max},\varepsilon)$ 为什么用除法编码、乘法恢复？全零张量怎么办？
- 量化网格的步长为什么由指数和尾数位共同决定？次正规数为什么要单独处理？
- "能创建 FP8 数据类型"和"原生 FP8 计算内核已执行"之间差哪些检查？
- GPU 实验里存储下降多少算合理？为什么存储降了 22% 却没有加速？

## 解决思路：一步步看懂

**第 1 步：把运行时路径拆成四个对象。** Step 1 的表把输入激活、权重、累加、输出和执行路径分开：权重可以是 FP8，激活仍可能是 BF16；累加可能是 FP32；输出还要被下游算子接受。任何一格没对齐，"低精度"就只停留在存储层。判断时要记录的是"输入 dtype、缩放因子来源、累加 dtype、输出 dtype、实际内核"，而不是只看模型文件大小。

**第 2 步：格式决定范围，缩放决定映射。** `FORMAT_SPECS` 里写得很清楚：

```python
"e4m3": {"max_value": 448.0, "mantissa_bits": 3, "min_normal_exponent": -6}
"e5m2": {"max_value": 57344.0, "mantissa_bits": 2, "min_normal_exponent": -14}
```

E4M3 用 4 位指数换 3 位尾数，局部更精细；E5M2 用 5 位指数换 2 位尾数，能表示更大的数。设 $f_{\max}$ 为格式最大有限值，$a=\max_i|x_i|$ 为当前张量的绝对最大值：

$$
s=\max\left(\frac{a}{f_{\max}},\varepsilon\right),\qquad
x_{\mathrm{fp8}}=\mathrm{cast}_{\mathrm{fp8}}(x/s),\qquad
\hat x=s\cdot\mathrm{cast}_{\mathrm{high}}(x_{\mathrm{fp8}})
$$

$s$ 是恢复尺度：编码前除以它，恢复时乘以它。notebook 的 `scale_convention` 字段写成 `divide_before_cast_multiply_after_cast`，就是为了防止有人把"乘 $s$"和"乘 $1/s$"弄反。$\varepsilon>0$ 保证全零张量也得到正尺度，否则 $a=0$ 会得到 0，后面除法直接炸掉。静态缩放、动态缩放、延迟缩放的差别只在于 $a$ 来自固定校准集、当前张量还是历史窗口。

**第 3 步：TODO 1——`compute_dynamic_scale` 动态缩放因子。** 参考实现：

```python
absmax = x.detach().float().abs().max()
scale = (absmax / fp8_max).clamp_min(self.eps)
```

先 `detach().float()` 是为了不把计算图带进量化参数，也避免半精度比较的边界问题；`clamp_min(self.eps)` 对应公式里的 $\varepsilon$。测试 `test_dynamic_scale_contract` 用 `[-2.0, 0.0, 1.0]` 断言结果等于 `2.0 / 448`，并对全零输入断言仍是有限正值。容易写错的是取 `max(absmax/fp8_max, eps)` 时把 `eps` 加在分母或直接 `absmax.clamp_min(eps)`，那会改变公式含义。

**第 4 步：TODO 2——`simulate_roundtrip` 按尾数位构造量化网格。** 前置代码已经把输入裁到 $[-f_{\max},f_{\max}]$，并算了指数：

```python
scaled = torch.clamp(x / scale, -spec["max_value"], spec["max_value"])
magnitude = scaled.abs().clamp_min(2.0 ** spec["min_normal_exponent"])
exponent = torch.floor(torch.log2(magnitude))
step = torch.pow(2.0, exponent - spec["mantissa_bits"])
rounded_scaled = (torch.round(scaled / step) * step).clamp(-spec["max_value"], spec["max_value"])
restored = rounded_scaled * scale
```

`magnitude.clamp_min(2^min_normal_exponent)` 把小于最小正规数的值"封底"到最小正规数，这样算出的步长固定为最小步长 $2^{\text{min\_normal\_exponent}-\text{mantissa\_bits}}$，模拟次正规数的等间隔。若不做这一步，$x=0$ 会让 `log2(0)=-inf`，步长变成 0 或 NaN。`round(scaled/step)*step` 是按局部步长四舍五入；再 `clamp` 一次保证不越过最大有限值；最后乘回 $s$。

`test_subnormal_grid_and_format_limits` 把两种格式的 `maximum`、`smallest/4`、`smallest`、`1.5*smallest` 放在一起：$1/4$ 个最小步长舍入到 0，正好一个最小步长保持不变，$1.5$ 个舍入到 2 倍。该测试同时断言 `scale == 1.0`（因为输入里含 `maximum`，$a/f_{\max}=1$）和恢复值绝对值不超过 `maximum`。

**第 5 步：TODO 3——`select_runtime_path` 运行时准入。** 参考实现只有三层：

```python
if device_type == "cuda" and dtype_available and native_kernel:
    path = "native_fp8_kernel"
elif dtype_available:
    path = "dtype_preflight"
else:
    path = "cpu_simulation"
```

前面的校验拒绝自相矛盾的观测：`native_kernel=True` 但设备不是 CUDA，或 dtype 不可用，直接抛 `ValueError`。`test_runtime_path_contract` 断言 `("cuda", True, True) -> "native_fp8_kernel"`、`("cuda", True, False) -> "dtype_preflight"`、`("cpu", False, False) -> "cpu_simulation"`；`test_invalid_values_and_inconsistent_observations` 再检查 `('cuda', False, True)` 和 `('cpu', True, True)` 都会被拒绝。这里的核心思想是：数据类型能创建，只能拿到"预检"级别的证据；`cpu_simulation` 分支只是教学模拟，不构成 GPU 证据。

**第 6 步：六个测试分别在守什么。** `run_fp8_runtime_tests()` 汇总 `test_dynamic_scale_contract`、`test_roundtrip_contract`、`test_format_precision_tradeoff`、`test_runtime_path_contract`、`test_subnormal_grid_and_format_limits`、`test_invalid_values_and_inconsistent_observations`，全过打印 `✅ FP8 运行时机制测试通过：scale、格式舍入、误差和路径 gate 均已验证。`。`test_roundtrip_contract` 只要求形状不变、恢复值有限；`test_format_precision_tradeoff` 只要求两种格式的 `mse` 都非负且 `format` 不同，并不预设 E4M3 一定更准——误差取决于输入范围是否被裁剪。

**第 7 步：GPU 实验的自变量、因变量与判据。** Step 5 用 torchao 把同一个 BF16 模型转成三类候选：

- `weight_only`：`Float8WeightOnlyConfig()`，只要求权重物理存储变成 FP8；
- `dynamic_per_tensor`：`Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())`，要求观察到原生 FP8 计算事件；
- `dynamic_per_row`：同样的动态配置但 `granularity=PerRow()`，逐行缩放，元数据更多。

固定输入是两个提示 × Token 数 `(1, 32, 256)`，预热 5 次、计时 20 次，取 median 与 P95。数值门槛 `MIN_LOGIT_COSINE = 0.99`，加速判据是 `MEANINGFUL_MEDIAN_RATIO = 0.95`（中位延迟至多基线的 95%）且 `STABLE_P95_RATIO = 1.00`（P95 不回归）。`path_confirmed` 对仅权重候选看权重类型里是否含 `float8`，对动态候选看 profiler 事件里是否命中 `scaled_mm`、`float8`、`fp8`、`f8f8`。

**第 8 步：结果怎么读。** notebook 5.4 记录的 RTX 5070 Ti 复测中，BF16 模型物理存储约 942.29 MiB，三类 FP8 候选约 730.87～732.61 MiB，降低约 22%——不是 50%，因为部分参数仍保留 BF16，而且 FP8 还需要缩放因子；逐行缩放的 FP32 元数据约 1.74 MiB，逐张量缩放不足 0.001 MiB。三类候选都完成了 18 个矩阵点，动态两条路径观察到 `aten::_scaled_mm` 事件，但报告没有关联 CUDA 子内核与实际输入/累加类型，所以只能支持"调用了缩放矩阵乘接口"，不能单凭事件名宣布所有层都是原生 FP8。

性能上三类候选在所有 Token 负载都没有形成加速，中位延迟约为 BF16 的 2.32～5.33 倍。原因是小模型、batch=1、Eager 执行下，动态缩放、张量包装、类型转换和调度成本超过了 FP8 矩阵乘的收益。Token=256 时三类候选都通过局部数值一致性阈值，短输入更敏感。因此决策是 `tune_or_fallback`：接受存储与接口调用证据，不接受它作为当前工作负载的性能候选。这个结论只覆盖当前形状、当前后端，不能外推到编译模式、更大 batch 或生产 serving。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| FP8 | 8 位浮点数，只有符号、指数、尾数三部分，范围与精度都要取舍 |
| E4M3 / E5M2 | 4 位指数 3 位尾数 / 5 位指数 2 位尾数；前者精细，后者范围大 |
| $f_{\max}$ | 格式能表示的最大有限值，E4M3 为 448，E5M2 为 57344 |
| 恢复尺度 $s$ | 编码前除、恢复时乘的那个数，通常取 `absmax / f_max` |
| absmax | 当前张量绝对值最大值，动态缩放用它适应本轮分布 |
| 次正规数 | 小于最小正规数的区间，步长固定，不再随指数变小 |
| 饱和舍入 | 超出格式范围的值被夹到端点，误差可能很大 |
| 原生 FP8 内核 | 硬件真正用 FP8 执行矩阵乘，而不是先转高精度再算 |
| `dtype_preflight` | 只能证明 FP8 类型可创建、可转换，不能证明计算走 FP8 |
| 回退路径 | 输入转成 FP8 后又转回高精度计算，通常没有性能收益 |
| logit 余弦相似度 | 固定输入下候选输出与 BF16 基线的局部数值一致性 |

## 小白常见疑问
**Q1：`scale = absmax / 448` 之后，为什么编码是 `x / scale` 而不是 `x * scale`？**
因为要把张量压进 $[-448,448]$。若 `absmax=896`，$s=2$，`x/s` 的最大值才是 448；恢复时再乘 2 回到原量级。notebook 用 `scale_convention` 字段把这条约定显式写出来，框架里有些实现保存的是倒数，读字段前必须先核对。

**Q2：全零张量的 `absmax` 是 0，为什么不能直接让 `scale=0`？**
后面要做 `x / scale`，0 会直接产生 `inf`/`NaN`。`clamp_min(self.eps)` 把尺度托到 `1e-8`，全零输入仍然得到有限正值；`test_dynamic_scale_contract` 专门用 `torch.zeros(4)` 检查这一点。

**Q3：E4M3 尾数更多，是不是任何输入都比 E5M2 误差小？**
不是。E4M3 的最大有限值只有 448，输入稍大就会被裁到端点；E5M2 能表示到 57344，范围更大的数据反而误差更小。`test_format_precision_tradeoff` 只断言两种格式的 MSE 都可计算且格式名不同，刻意不写死"谁更准"，因为答案依赖输入分布和是否饱和。

**Q4：`aten::_scaled_mm` 出现了，就能说用了原生 FP8 吗？**
不能。profiler 事件名只能说明调用了缩放矩阵乘接口，不能证明底层 CUDA 子内核和实际输入/累加类型。notebook 5.4 明确要求"关联 CUDA 子内核与实际类型"，这正是把"接口事件"和"原生计算"分开的原因。

**Q5：存储降了 22%，为什么反而更慢？**
存储和计算是两条独立的账。权重确实变小了，但动态量化每轮都要算 `absmax`、做除法和类型转换，还可能插入额外的 kernel；在小模型、单请求、Eager 模式下这些固定开销盖过了矩阵乘的收益。`meaningful_speedup` 要求 median 和 P95 同时达标，当前结果没有满足。

## 动手验证：怎么知道自己做对了
`run_fp8_runtime_tests()` 跑完六组测试后打印 `✅ FP8 运行时机制测试通过：scale、格式舍入、误差和路径 gate 均已验证。`。
- `test_dynamic_scale_contract`：`[-2.0, 0.0, 1.0]` 的尺度必须 `assert_close` 到 `2.0/448`；全零输入尺度有限且大于 0。
- `test_roundtrip_contract`：输入 `[[-2.0,-0.25,0.0,0.75,3.0]]`，恢复值形状一致、全部有限，尺度也有限。
- `test_format_precision_tradeoff`：33 个点的 `linspace(-3,3)`，两种格式的 `mse >= 0`，`format` 字段不同。
- `test_runtime_path_contract`：三种组合分别得到 `native_fp8_kernel`、`dtype_preflight`、`cpu_simulation`。
- `test_subnormal_grid_and_format_limits`：两种格式的 `scale==1.0`；`smallest/4` 恢复为 0，`smallest` 保持不变，`1.5*smallest` 变成 `2*smallest`；最大恢复值不超过 `maximum`。
- `test_invalid_values_and_inconsistent_observations`：`eps=0`、`eps=nan`、空张量、`inf`、整型张量、`('cuda',False,True)`、`('cpu',True,True)` 都必须抛 `ValueError`。
notebook 的 GPU 单元在 dump 中只保存了配置和部分输出，5.4 的存储、延迟和决策数字来自 notebook 已记录的结果；若你自己重跑，以 JSON 里的 `mechanism_metrics`、`kernel_events`、`logit_cosine` 和 `decision` 为准。
