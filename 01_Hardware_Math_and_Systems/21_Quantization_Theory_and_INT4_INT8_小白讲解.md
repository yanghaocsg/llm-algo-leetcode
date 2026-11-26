# 21. 量化理论与 INT4/INT8 — 小白讲解

> 对应 notebook：[`21_Quantization_Theory_and_INT4_INT8.ipynb`](./21_Quantization_Theory_and_INT4_INT8.ipynb)

## 一句话总结
这一节用五个可以在 CPU 上直接跑的“小计算器”，把量化从一句“把 FP16 换成 INT8/INT4”拆成四件事：权重和元数据各占多少字节、`scale`/`zero_point` 怎么把浮点数映射到整数格点、per-tensor 与 per-channel 的误差差在哪、以及 PTQ 质量不达标时该不该上 QAT。所有函数都只做数值推演，不依赖 GPU，也没有训练任何模型。

## 为什么需要它：痛点是什么
初学者对量化的第一印象往往是“位宽减半，显存减半”：7B 模型 FP16 是 14 GB，INT8 就是 7 GB，INT4 就是 3.5 GB。这个账没错，但它只是**纯权重载荷**。一旦动手就会遇到一串对不上的数：为什么实际 INT4 文件不是 3.5 GB？为什么同样 8 位，“整个张量一个 scale”和“每行一个 scale”误差差很多？为什么权重能压到 4 位，激活却很难？为什么 PTQ 掉点 0.6 个百分点，就不能硬说它“差不多合格”？

生活类比：把连续的海拔高度画成等高线地图。`scale` 是等高距——每隔多少米画一条线；`zero_point` 是海平面的位置，决定 0 米对应图上哪条线；位宽决定一共能画多少条线（INT8 的对称量化只有 127 个正格点，INT4 只有 7 个）。等高距越大，地图越省地方，但两座差得不多的小山包会被画成同一条线。量化误差就来自“真实高度落在两条等高线之间”。

## 核心问题
- 7B 模型在 FP16 / INT8 / INT4 下纯权重各占多少 GB？分组量化的 scale 元数据又占多少？
- `q = round(x / scale) + zero_point` 里每个符号是什么？对称量化为什么能省掉 `zero_point`？
- per-tensor 和 per-channel 的 scale 形状、误差和元数据代价分别怎样？
- 权重量化和激活量化，为什么难度不同？
- PTQ 掉点超过门槛时，判断“继续调参、估算 QAT、还是加预算”的依据是什么？
- 位宽账本之外，进入真实部署还要核对哪些执行条件？

## 解决思路：一步步看懂

### Q1：先算权重载荷，再算分组元数据
`calculate_weight_memory(num_params_b, dtype)` 只有一行核心逻辑：查表 `{'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}` 得到每参数字节数，再乘参数量（单位 B）。所以 7B 模型：

- FP16/BF16：$7 \times 2 = 14$ GB
- INT8：$7 \times 1 = 7$ GB
- INT4：$7 \times 0.5 = 3.5$ GB

注意这里的 “B” 既是“十亿参数”，GB 也按十进制 $10^9$ 理解，所以数字能直接相乘。测试 `test_calculate_weight_memory` 就断言了 14.0、7.0、3.5 三个值。

但真实 INT4 不是一个参数占半字节就完事。`quantized_weight_bytes(num_params, bitwidth=4, group_size=128, scale_bytes=2, zero_point_bytes=0)` 把账拆成两块：

```python
groups = (num_params + group_size - 1) // group_size          # 向上取整的组数
payload = (num_params * bitwidth + 7) // 8                    # 位打包后的权重字节
metadata = groups * (scale_bytes + zero_point_bytes)          # 每组一份 scale（和可选 zero_point）
```

代入 notebook 的例子：1024 个权重、INT4、每 128 个一组。权重载荷 $1024 \times 4 \div 8 = 512$ 字节；组数 $1024 \div 128 = 8$；元数据 $8 \times 2 = 16$ 字节；合计 **528 字节**。对照：1024 个数用 FP16 是 2048 字节。把 `group_size` 改成 32，组数变成 32，元数据 64 字节，合计 **576 字节**——分组越细，scale 越多，压缩率越差。这就是实际 INT4 文件大于理论 3.5 GB 的第一层原因；文件头、对齐和反量化等开销还没算。

### Q2：对称量化的映射、误差与粒度
最一般的仿射量化写成：

$$q = \mathrm{round}\!\left(\frac{x}{\mathrm{scale}}\right) + \mathrm{zero\_point},\qquad \hat{x} = (q - \mathrm{zero\_point}) \times \mathrm{scale}$$

其中 $x$ 是原始浮点值，$\mathrm{scale}$ 是“一个整数格代表多少真实值”，$\mathrm{zero\_point}$ 是真实 0 对应的整数编号，$q$ 是存下来的整数，$\hat{x}$ 是 `dequantize_per_tensor(q, scale)` 还原出的近似值。对称量化令 $\mathrm{zero\_point}=0$，正负格点关于 0 对称，实现最简单，权重量化常用它。

`quantize_per_tensor(x, num_bits=8)` 的实现是：

```python
qmax = 2 ** (num_bits - 1) - 1                       # 8 位是 127，4 位是 7
absmax = source.abs().max()
scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
q = torch.clamp(torch.round(source / scale), -qmax, qmax).to(torch.int8)
```

三个细节都值得停一下：

1. **为什么是 $2^{n-1}-1$ 而不是 $2^n-1$？** 因为对称量化要留出负数区间，INT8 的表示范围是 $-128 \sim 127$，正半轴最大 127；用 127 做分母能保证正数不溢出，代价是 $-128$ 这个格点空着。
2. **全零输入要特判。** `absmax=0` 时如果直接除，会得到 0/0 = NaN；代码用 `torch.where` 把 scale 换成 1，于是全零张量量化后全是 0，scale=1.0。测试里专门用 `torch.zeros(3, dtype=torch.float16)` 走 4 位路径验证了这一点。
3. **为什么在 FP32 里算 scale？** 全零 FP16 输入下，用 FP32 算中间量可避免最小 scale 下溢。

`test_quantize_per_tensor` 用 `[-1.0, -0.5, 0.0, 0.5, 1.0]` 验证：`absmax=1`，8 位 `scale=1/127≈0.007874`，量化结果恰好是 `[-127, -64, 0, 64, 127]`。其中 $-0.5/scale=-63.5$、$0.5/scale=63.5$，`torch.round` 采用“四舍六入五成双”，两个 .5 都落到偶数 64，所以不是 ±63 或 ±64 的随意组合。断言还要求反量化误差不超过 `scale / 2 + 1e-7`——这正是“最近格点”策略的理论上界：任何值离最近格点最远半个格子。

`per_channel_scales(x, axis=0, num_bits=8)` 则按行或按列各算一个 scale：`axis=0` 时沿列方向（`dim=1`）取 `amax` 并 `keepdim=True`，得到形状 `(行数, 1)`；`axis=1` 时得到 `(1, 列数)`。notebook 用矩阵 `[[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]]` 打印了两种形状：per-tensor 是 `()`（一个标量），per-channel（axis=0）是 `(2, 1)`。第二行整体小一个数量级，若与第一行共用一个 scale，0.1、0.2、0.3 会被压到接近 0 的格点；分行给 scale 后各自用满动态范围，这就是 per-channel 通常更准的原因，代价是每组多一份元数据。

Q2 扩展单元用固定种子生成 `torch.randn(1024) * 2`，分别跑 8 位和 4 位并打印 MSE 与最大误差。读法是：4 位只有 15 个格点（-7 到 7），误差应明显大于 8 位的 255 个格点。`test_quantization_practice` 把它写成断言：4 位的平均绝对误差不小于 8 位。

### Q3：量化对象、时机、交付形式是三件不同的事
`quantization_method_catalog()` 返回一张字典，把五种常见名词放回各自的层级：PTQ 是训练完成后得到量化权重，QAT 是训练或微调中得到量化感知模型，GPTQ/AWQ 是部署前校准得到的对应权重，GGUF 是模型交付时由 llama.cpp 等解释的文件格式。

`select_quantization_path(method)` 做一次查表并展开成 `{'method': ..., **catalog[method]}`；传入未知名字会抛 `ValueError`，错误信息里带 `sorted(catalog)`，方便看出可选值。最重要的认知是：**GGUF 不是一种量化算法，而是文件格式与交付封装**；GPTQ/AWQ 是权重校准路线；PTQ/QAT 描述的是“什么时候量化”。把三者混成一个排行榜，就会出现“GGUF 和 AWQ 哪个精度高”这种没有答案的问题。

### Q4：PTQ 掉点之后，先算账再决定
`qat_recommendation(ptq_drop, acceptable_drop=0.5, retrain_budget_hours=0, sensitivity='medium', estimated_qat_hours=None)` 是一棵很小的决策树：

```python
quality_met = ptq_drop <= acceptable_drop
if quality_met:            recommendation = 'PTQ_candidate'
elif estimated_qat_hours is None:  recommendation = 'estimate_QAT_cost'
elif retrain_budget_hours >= estimated_qat_hours:  recommendation = 'QAT_trial'
else:                      recommendation = 'adjust_quantization_or_budget'
```

几个边界要看清：

- 门槛是“小于等于”，所以 `ptq_drop=0.5` 时 `ptq_quality_met` 为真，0.6 才算超。**质量不合格不会因为预算多就改判合格**：掉 0.6、QAT 估算 10 小时、只有 8 小时预算时，结论只能是 `adjust_quantization_or_budget`。
- `estimated_qat_hours` 为 `None` 时先要求估算成本，不会凭空假设“10 小时”的通用门槛；预算正好等于估算（10 ≥ 10）时给 `QAT_trial`，是“可以试”而非“一定值得”。
- `inspect_sensitive_layers` 只在 `sensitivity == 'high'` 时为真，提示逐层看敏感层误差。函数还会校验三个非负量，并对 `estimated_qat_hours <= 0` 抛错。三个 `cases` 是：掉 0.2 → 候选；掉 0.6、预算 12、估算 10 → 试验；掉 0.6、预算 8、估算 10 → 调整或加预算。

### Q5：从“账本”走到“能跑”
`quantization_risk(bitwidth, hardware_support=True, calibration_quality=1.0, activation_sensitive=False)` 不计算风险分数，只列出必须核对的检查项：

- 基线三项：`validation_quality`、`complete_call_time`、`peak_memory`；
- 位宽是 4 时追加 `packing_and_metadata`（打包与元数据）；
- `hardware_support=False` 时追加 `fallback_execution_path`；
- `activation_sensitive=True` 时追加 `activation_error_by_layer`。

它只接受 4 或 8 位，`calibration_quality` 必须在 $[0,1]$，否则抛 `ValueError`。这段代码的价值观很明确：**不要把这些条件加权成一个“风险分”**。校准数据够不够、反量化调用多慢、后端能不能吃低比特权重，是三件独立的事，任何一项不满足都可能让“理论省一半显存”落空。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 量化（quantization） | 用有限个整数格点近似连续的浮点值 |
| `scale`（缩放因子） | 一个整数格代表多少真实值，类似等高线间距 |
| `zero_point`（零点） | 真实 0 对应的整数编号，对称量化时固定为 0 |
| 对称 / 非对称量化 | 格点关于 0 对称 / 允许零点偏移，后者更适合分布偏斜的数据 |
| per-tensor / per-channel | 整个张量共用一个 scale / 每行或每列各一个 scale |
| `qmax` | 对称量化的最大正整数格点，8 位是 127，4 位是 7 |
| 反量化 | 把整数乘回 scale，还原成近似的浮点值 |
| 分组量化 | 每 `group_size` 个权重共享一份 scale，组越多元数据越多 |
| PTQ / QAT | 训练完成后量化 / 训练中模拟量化误差 |
| GPTQ / AWQ | 两种常见的部署前权重量化校准路线 |
| GGUF | 一种模型文件格式与交付封装，不是量化算法本身 |
| 校准数据 | 用来估计激活范围、决定 scale 的代表性输入 |

## 小白常见疑问
**Q1：对称量化为什么用 127 做分母，不用 128？**
INT8 有符号范围是 $-128 \sim 127$。如果用 128，$x=\mathrm{absmax}$ 时 $x/\mathrm{scale}=128$，超出了正半轴最大值 127。用 127 保证正数不溢出，代价是负半轴的 $-128$ 格点闲置。4 位同理，`qmax=7`。

**Q2：per-channel 一定比 per-tensor 好吗？**
在精度上通常更好，因为每一行/列用自己的动态范围；但不是免费。`per_channel_scales` 输出的形状是 `(2, 1)` 这类按维度的 scale，元数据随通道数增长。notebook 的定位是“通常精度更好，但元数据更多”，是否值得要在固定模型上测。

**Q3：权重能 4 位，激活为什么不行？**
notebook 的 Q2 说明给了原因：权重分布相对稳定、离群值少；激活随 token、层、上下文变化，还可能在某些通道出现明显离群值。一个离群值会把整个张量的 `absmax` 拉高，其他值就被压到很少的几个格点上。这也是 SmoothQuant 这类“权重-激活协同处理”思路的动机。

**Q4：PTQ 掉 0.6 个百分点、门槛 0.5，能不能四舍五入说合格？**
不能。`qat_recommendation` 用的是 `ptq_drop <= acceptable_drop`，0.6 > 0.5 就是不合格；预算再多也不会改变 `ptq_quality_met`。预算只决定下一步是 `QAT_trial` 还是 `adjust_quantization_or_budget`。同样，`quantization_risk` 返回的 `required_checks` 不是风险分，只是把 4 位打包、回退路径、逐层激活误差等检查项列全，不能相加。

## 动手验证：怎么知道自己做对了
notebook 没有保存任何执行输出，下面以代码里的断言为准；你自己运行时，通过与否看有没有 `AssertionError`。

- `test_calculate_weight_memory`：`calculate_weight_memory(7, 'fp16') == 14.0`、`'int8'` 为 7.0、`'int4'` 为 3.5（容差 `1e-9`），通过后打印 `✅ calculate_weight_memory tests passed`。后面的模块级断言还检查 `quantized_weight_bytes(1024)` 等于 `{'weight_bytes': 512, 'metadata_bytes': 16, 'total_bytes': 528}`，以及 `group_size=32` 时 `total_bytes == 576`。
- `test_quantize_per_tensor`：8 位量化 `[-1.0, -0.5, 0.0, 0.5, 1.0]` 的整数结果必须精确等于 `[-127, -64, 0, 64, 127]`；反量化最大误差 `<= scale / 2 + 1e-7`；全零 FP16 输入走 4 位时 `zero_q` 全 0 且 `zero_scale == 1.0`。通过后打印量化整数和 scale（约 0.007874）。
- `test_quantization_practice`：检查 8 位和 4 位输出都是 `torch.int8`、形状与输入一致，且 4 位的平均绝对误差不小于 8 位；通过后打印 `✅ 21 Quantization tests passed`。
- Q3：`select_quantization_path('GGUF')['backend'] == 'llama.cpp 等'`；未知方法抛 `ValueError`。
- Q4：`qat_recommendation(0.5)['ptq_quality_met']` 为真；`(0.6, retrain_budget_hours=8, estimated_qat_hours=10)` 的完整返回是 `{'ptq_quality_met': False, 'recommendation': 'adjust_quantization_or_budget', 'inspect_sensitive_layers': False}`；预算 10、估算 10 时推荐 `QAT_trial`。
- Q5：4 位、`hardware_support=False` 时 `required_checks` 含 `fallback_execution_path`，`True` 时不含。
