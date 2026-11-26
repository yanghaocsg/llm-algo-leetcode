# 52. 量化校准范围与裁剪 — 小白讲解

> 对应 notebook：[`52_Quantization_Calibration_and_Error.ipynb`](./52_Quantization_Calibration_and_Error.ipynb)

## 一句话总结
本节用 `calibrate_symmetric_scale`、`evaluate_clipping_candidate`、`select_clipping_candidate` 三个函数，演示"用校准数据的分位数生成候选范围 → 在独立验证数据上计算恢复误差和饱和率 → 先排除饱和率超限的候选，再选 MSE 最低的范围"。核心矛盾是：范围缩小能让常见数值的网格更细，但尾部会被裁到端点，两种误差必须一起看。

## 为什么需要它：痛点是什么
初学者做对称 INT8 量化时，通常直接取 `absmax / 127` 当步长。如果校准数据里有一个 100 的异常值，而绝大多数数值都在 `[-2, 2]`，那么步长会从约 `2/127 ≈ 0.0157` 变成 `100/127 ≈ 0.7874`——常见数值明明离异常值很远，却被迫用一张很粗的网格来表示，舍入误差大幅增加。

另一个常见误解是"原值超过范围才算饱和"。实际上量化先做舍入、再裁剪：`127.4` 舍入成 `127`，根本不需要裁；`127.6` 舍入成 `128`，才需要裁到 `127`。如果按"原值是否大于 127"统计，会把 `127.4` 也算成饱和，指标就虚高了。

生活类比：给一群人量身高，尺子只有 127 个刻度。如果按最高的人（比如 2 米 26）来定刻度，每个刻度约 1.78 cm，普通人 1.7 米附近的读数很粗；如果把尺子范围缩到 0～1.8 米，普通人读数精细了，但超过 1.8 米的人只能记成"1.8 米上限"。校准就是决定尺子量程的过程。

## 核心问题
- 校准范围 $c$ 和步长 $s$ 是什么关系？为什么异常值会"污染"普通数值的精度？
- 分位数校准怎样生成不同范围的候选？分位数越低，范围和网格怎么变？
- 恢复误差 MSE 和饱和率 $\rho_{\mathrm{sat}}$ 分别衡量什么？为什么饱和率要按"舍入后的整数码"统计？
- 为什么候选范围只能由校准数据估计，验证数据只用来评估？
- 选择规则为什么是"先卡饱和率，再选 MSE 最低"，而不是直接选 MSE 最低？
- 验证数据已经参与选择，为什么最终测试还要另留一份？

## 解决思路：一步步看懂

**第 1 步：固定量化口径。** 本节只用对称 INT8、逐张量粒度：整数码范围是 $[-127,127]$，所以 $q_{\max}=127$，没有零点偏移。候选范围 $c>0$ 决定步长：

$$
s(c)=\frac{c}{q_{\max}}
$$

如果 $c=2$，$s=2/127\approx0.0157$；如果异常值把 $c$ 拉到 100，$s=100/127\approx0.7874$。整数码数量还是 255 个，但每个码覆盖的数值区间大了 50 倍，普通值的舍入误差自然变大。

**第 2 步：量化、裁剪、恢复的完整链路。** 对验证集里的每个 $x_i$：

$$
u_i(c)=\mathrm{round}\left(\frac{x_i}{s(c)}\right),\qquad
q_i(c)=\mathrm{clip}\big(u_i(c),-q_{\max},q_{\max}\big),\qquad
\hat x_i(c)=s(c)\,q_i(c)
$$

$u_i$ 是舍入后的"未裁剪码"，$q_i$ 是被夹到端点后的整数码，$\hat x_i$ 是恢复值。MSE 汇总所有验证元素的恢复偏差，同时包含舍入误差和裁剪误差：

$$
\mathrm{MSE}(c)=\frac1N\sum_{i=1}^N\big(x_i-\hat x_i(c)\big)^2
$$

饱和率则只统计"舍入后仍然超界、需要裁剪"的比例：

$$
\rho_{\mathrm{sat}}(c)=\frac1N\sum_{i=1}^N\mathbf 1\big(|u_i(c)|>q_{\max}\big)
$$

注意是 $|u_i|>q_{\max}$，不是 $|x_i|>c$。notebook 的 `test_saturation_counts_rounded_codes` 用 `[-127.6, -127.4, 0, 127.4, 127.6]` 验证：舍入后是 `[-128, -127, 0, 127, 128]`，饱和率 $2/5$；而"原值绝对值超过 127"的比例是 $4/5$，两者必须区分。

**第 3 步：TODO 1——用分位数估计步长。** 参考实现：

```python
abs_values = values.detach().float().abs().flatten()
calibrated_range = torch.quantile(abs_values, clipping_quantile)
scale = calibrated_range.clamp_min(torch.finfo(torch.float32).eps) / qmax
```

`detach().float().abs().flatten()` 把校准值脱离计算图、转 FP32、取绝对值并拉平，`torch.quantile` 取指定分位数作为候选范围 $c$。`clipping_quantile=1.0` 就是 `absmax`；取 0.8 表示"忽略最大的 20% 尾部"。`clamp_min(eps)` 处理全零张量：范围 0 会让步长为 0，后面除法直接失效。测试 `test_clipping_scale_contract` 用 `[-100,-1,0,1,2]` 断言 0.8 分位数的步长小于 1.0 分位数，并对全零输入断言步长为正。

**第 4 步：TODO 2——只在验证数据上评估。** 参考实现：

```python
scale = calibrate_symmetric_scale(calibration_values, clipping_quantile=clipping_quantile)
quantized_validation = quantize_dequantize_symmetric(validation_values, scale)
_, reconstructed, saturation_rate = quantized_validation
delta = validation_values.float() - reconstructed.float()
```

步长只由 `calibration_values` 估计，然后原封不动地用在 `validation_values` 上；验证数据不能反过来重新估计步长，否则就是"用考题调参再在同一套考题上打分"。`quantize_dequantize_symmetric` 里 `raw_codes = torch.round(values / scale)` 先舍入，`saturation_rate` 用舍入后的 `raw_codes.abs() > qmax` 统计，`codes` 再 `clamp` 并转 `int8`，`reconstructed = codes.to(values.dtype) * scale`。

`test_candidate_uses_calibration_scale_on_validation` 用校准 `[-2,-1,0,1,100]`、验证 `[-2,-1,0,1,2]`，断言候选的 `scale` 恰好等于对校准集调用 `calibrate_symmetric_scale(..., 0.8)` 的结果，并检查返回字段包含 `mse`、`max_abs_error`、`saturation_rate`。

**第 5 步：TODO 3——先筛选再排序。** 参考实现：

```python
feasible = [candidate for candidate in candidates
            if candidate["saturation_rate"] <= max_saturation_rate]
selected = min(feasible, key=lambda item: (item["mse"], -item["quantile"]), default=None)
if not feasible:
    raise RuntimeError("no feasible clipping candidate")
```

选择规则对应：

$$
c^{\ast}=\arg\min_{c\in\mathcal C}\mathrm{MSE}(c)
\quad\text{s.t.}\quad \rho_{\mathrm{sat}}(c)\le\tau
$$

先把饱和率超过 $\tau$ 的候选全部淘汰，再在可行集合里选 MSE 最低的。`key=(mse, -quantile)` 处理平局：MSE 相同时，`-quantile` 更小意味着分位数更高，所以优先保留范围更大的候选。全部候选都超限时抛 `RuntimeError("no feasible clipping candidate")`，而不是硬塞一个不合格的结果。

**第 6 步：用图示算例把两种误差算出来。** `test_wide_narrow_range_illustration` 对 `[1.1, 100.0]` 做对照：宽范围（$s=100/127$）把 1.1 恢复成 $100/127\approx0.787$（误差 0.313），100 正好恢复；窄范围（$s=2/127$）把 1.1 恢复成 $140/127\approx1.102$（误差 0.002），但 100 被裁到 2.0（误差 98），饱和率 0.5。窄范围让普通值更准，却让尾部付出巨大代价；测试同时断言窄范围对 1.1 更准、对 100 更差。

**第 7 步：数据分工。** 生成候选用校准数值，验证候选用验证数值，应用约束看验证指标，选择范围在可行集合里挑 MSE 最低。验证数据虽然不参与"估计步长"，但参与了"选择分位数"，所以它已不是完全独立的测试集；最终部署前应另留一份未参与选择的测试负载。notebook 用 `calibration`、`validation` 两个张量演示，正是为了让这条边界可见。

**第 8 步：和成熟框架对齐。** 可选单元 `check_calibration_framework_conformance` 用 `MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-127, quant_max=127)` 计算量化参数，再用 `torch.fake_quantize_per_tensor_affine` 做伪量化。核对 `scale_match`（与本节最大范围步长在 `atol=1e-7` 内一致）、`zero_point_is_zero`、`reconstruction_match`、`saturation_match` 四项。该单元没有保存输出，以 `checks` 字典和 `AssertionError` 为准。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 对称量化 | 整数码范围关于 0 对称，零点固定为 0 |
| 逐张量粒度 | 整个张量共用一个缩放因子 |
| 校准数据 | 用来估计范围/步长的数据，不参与最终打分 |
| 验证数据 | 用来比较候选误差的数据，会参与范围选择 |
| 分位数校准 | 取绝对值分布的某个分位点当范围，忽略更高的尾部 |
| 步长 $s$ | 一个整数码代表多大的数值，$s=c/q_{\max}$ |
| 舍入误差 | 数值落在两个网格点之间，恢复后被拉到最近格点 |
| 裁剪 / 饱和 | 舍入后的码超出上下限，被夹到端点 |
| 饱和率 $\rho_{\mathrm{sat}}$ | 需要裁剪的整数码占验证元素的比例 |
| MSE | 恢复值与原值之差的平方均值，混合了舍入和裁剪误差 |
| $q_{\max}$ | 对称整数码上限，INT8 取 127 |
| 可行集合 | 饱和率不超过阈值的候选范围集合 |

## 小白常见疑问
**Q1：分位数取 0.8，为什么步长会变小？**
`torch.quantile(abs_values, 0.8)` 返回第 80 百分位，通常远小于 `absmax`；范围 $c$ 变小，$s=c/127$ 也变小，网格更细。`test_clipping_scale_contract` 用 `[-100,-1,0,1,2]` 断言 0.8 分位的步长严格小于 1.0 分位。

**Q2：饱和率为什么不直接用 `|x| > c` 统计？**
因为量化是先舍入再裁剪。`127.4` 在 $c=127$、$s=1$ 时舍入为 127，根本不会被裁；按原值统计却会把它算成饱和。notebook 的 `test_saturation_counts_rounded_codes` 给出饱和率 $2/5$、原值超范围比例 $4/5$，两个数必须分开。`quantize_dequantize_symmetric` 也是先算 `raw_codes = round(values/scale)`，再用 `raw_codes.abs() > qmax` 统计。

**Q3：为什么不能所有分位数都试一遍，直接选 MSE 最低的？**
MSE 最低可能是把范围压得很窄、尾部大量裁剪换来的；在某些负载上尾部误差很大，但平均值被大量普通值稀释了。饱和率约束就是一条硬门槛：先保证"需要裁剪的码不超过 $\tau$"，再在可行集合里比 MSE。`test_clipping_search_selects_lowest_feasible_mse` 先自己筛出 `saturation_rate <= 0.30` 的可行集，再断言选择结果等于 `min(feasible, key=(mse, -quantile))`。

**Q4：MSE 相同时为什么优先更高分位数？**
更高分位数对应更大范围、更少尾部裁剪，误差相同时更保守。实现用 `key=lambda item: (item["mse"], -item["quantile"])`，`-quantile` 让 1.0 排在 0.5 前面；`test_clipping_tie_prefers_higher_quantile` 用全零验证集和交换顺序断言结果始终是 1.0。

**Q5：验证数据已经用来选范围了，为什么还说它独立？**
它独立于"步长估计"——步长只看校准集；但它参与了"分位数选择"，所以不再完全 held-out。notebook 的 Step 3 明确写"最终负载的测试应另外保留"；实际工程通常分校准、验证、测试三份，避免选择偏差。

## 动手验证：怎么知道自己做对了
测试单元依次运行七个函数，全过打印 `✅ 7 项测试通过：步长、数据分工、码裁剪边界、图示算例与候选选择。`。
- `test_clipping_scale_contract`：`[-100,-1,0,1,2]` 的 0.8 分位步长小于 1.0 分位步长；全零输入步长大于 0。
- `test_candidate_uses_calibration_scale_on_validation`：候选 `scale` 等于只用校准集算出的步长，`quantile==0.8`，返回字段齐全，饱和率在 `[0,1]`。
- `test_clipping_search_selects_lowest_feasible_mse`：校准 `[-8,-2,-1,0,1,2,8]`、验证 `[-3,-2,-1,0,1,2,3]`、分位数 `[1.0,0.9,0.7]`、上限 0.30，选择结果必须等于 `min(feasible, key=(mse, -quantile))`。
- `test_no_feasible_clipping_candidate`：所有候选饱和率超限时抛 `RuntimeError`，消息含 `"clipping candidate"`。
- `test_saturation_counts_rounded_codes`：校准 `[-127,127]` 得 `scale=1.0`；验证码为 `[-127,-127,0,127,127]`，饱和率 `2/5`，原值超范围比例 `4/5`；上限 0.5 时该候选可通过。
- `test_clipping_tie_prefers_higher_quantile`：全零验证集下 MSE 都是 0，候选顺序 `[0.5,1.0]` 和 `[1.0,0.5]` 都必须选 1.0。
- `test_wide_narrow_range_illustration`：宽范围码 `[1,127]`、恢复 `[100/127,100]`；窄范围码 `[70,127]`、恢复 `[140/127,2.0]`；窄范围饱和率 0.5；窄范围对 1.1 更准、对 100 更差。
框架一致性单元 `check_calibration_framework_conformance` 若运行，会打印 `status: "passed"` 及四项 `checks`；它依赖 PyTorch 的 observer 实现，dump 中没有保存输出，以断言为准。
