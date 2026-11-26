# 25. W8A16 量化 — 小白讲解

> 对应 notebook：[`25_Quantization_W8A16.ipynb`](./25_Quantization_W8A16.ipynb)

## 一句话总结
本节用几个纯 CPU 的二维权重例子，把「权重存成 INT8」拆成四步：算缩放因子与零点、按粒度划组、反量化恢复权重、跑一次教学版 `W8A16Linear` 前向。随后用误差、余弦相似度和字节账本比较逐张量、逐通道、逐组三种粒度，并说明 W8A16 只保证「存得小」，不保证「算得快」。

## 为什么需要它：痛点是什么
初学者最常见的误解是：把 `weight.dtype` 从 `float16` 改成 `int8`，显存就变成一半，矩阵乘法也会自动变快。实际上一旦改成整数码，原来的浮点矩阵乘法就不能直接用了——你必须先知道每个整数码代表多大的实数，才能把它恢复成浮点权重；而「每个码代表多大」这件事，需要额外保存缩放因子（scale），非对称时还要保存零点（zero point）。

生活类比：把一条连续的高度刻度改刻成有限的格子。`scale` 是「一格代表多少厘米」，`zero_point` 是「实数 0 落在第几格」，粒度则是「整把尺子共用一套刻度，还是每一段、每一行各用一套」。格子越少越省地方，但读数越粗；刻度标签本身也要占地方，标签太细碎还会让读取变麻烦。

## 核心问题
- 对称量化和非对称量化的 `scale`、`zero_point` 分别怎么算？为什么要先把实数 0 纳入范围？
- 全零权重、常量权重这种退化输入，怎样避免 `scale=0` 和端点饱和？
- 逐张量、逐通道、逐组三种粒度，误差和元数据各自怎么变？尾组为什么不能丢？
- 反量化应该发生在哪里？W8A16 的名称到底承诺了什么、没承诺什么？
- 怎样把「权重更小」「结果仍可信」「当前负载更快」三个结论分开验证？

## 解决思路：一步步看懂

**第 1 步：W8A16 是表示约定，不是「纯整数计算」。** W8 指权重以 8 位整数保存，A16 指激活仍用 16 位浮点（FP16/BF16）。notebook 的 Step 3 把执行路径分成三档：eager dequantization 在通用矩阵乘之前显式恢复浮点权重；fused dequant + matmul 在计算内核读低比特权重时顺手恢复；integer / low-precision kernel 才让目标计算单元直接消费低精度格式。前两种能证明「权重存储和访存变小」，只有第三种才可能证明「低精度计算本身更快」。本节的教学实现走的是第一种。

**第 2 步：对称量化（TODO 1）。** 记 $x_i$ 为浮点权重，$q_i$ 为整数码，$s>0$ 为缩放因子，$[q_{\min},q_{\max}]$ 为码值范围。对称量化强制零点为 0，公式是：

$$
s=\frac{\max_i|x_i|}{q_{\max}},\qquad
q_i=\mathrm{clip}\left(\mathrm{round}\frac{x_i}{s},q_{\min},q_{\max}\right),\qquad
\hat{x}_i=sq_i.
$$

`calculate_qparams` 里对称分支取 `qmin, qmax = (-127, 127)`，参考答案是：

```python
absmax = values.detach().float().abs().max().clamp_min(eps)  # 绝对最大值，至少为 eps
scale = absmax / qmax                                        # 127 个正码覆盖 absmax
zero_point = torch.zeros_like(scale)                         # 对称：零点固定为 0
```

三个细节都对应测试：`detach()` 表示统计过程不保留计算图（量化参数不是可训练参数）；`clamp_min(eps)` 保证全零张量也得到正缩放因子，否则下一步会除以 0；`zeros_like` 让零点与 scale 同设备同形状。拿框架核对用的 `[-3.0, -0.7, 0.0, 0.9, 2.4]` 手算：`absmax=3.0`，`scale=3/127≈0.023622`，`-3.0` 被映射到 `-127`，`2.4` 映射到 `round(101.6)=102`。

**第 3 步：非对称量化（TODO 2）。** 非对称用完整的 255 个码（`qmin=-128, qmax=127`），并允许零点不为 0：

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad
z=\mathrm{clip}\left(\mathrm{round}\left(q_{\min}-\frac{x_{\min}}s\right),q_{\min},q_{\max}\right),
$$

$$
q_i=\mathrm{clip}\left(\mathrm{round}\frac{x_i}{s}+z,q_{\min},q_{\max}\right),\qquad
\hat{x}_i=s(q_i-z).
$$

参考答案先构造一个标量 0，再取 `minimum = min(x.min(), 0)`、`maximum = max(x.max(), 0)`。为什么一定要把 0 包进去？如果数据全正，例如 `[1.0, 3.0]`，只用数据端点会得到 `x_min=1`，算出的零点在裁剪后可能落到边界，于是最小的 1.0 反而被压到码值边界、恢复时饱和。把 0 纳入后：`x_min=0, x_max=3`，`scale=3/255≈0.011765`，`z=round(-128-0)=-128`，1.0 对应 `round(85)-128=-43`，3.0 对应 `127`，两端都能恢复。常量 `torch.ones(4)` 同理得到 `scale=1/255`；全零输入则 `maximum-minimum=0`，靠 `clamp_min(eps)` 保住正缩放因子。

**第 4 步：粒度与尾组（TODO 3）。** 粒度决定多少个权重共享一套 `scale/zero_point`：

| 粒度 | 共享范围 | `scale` 元素数 |
|---|---|---:|
| `per_tensor` | 整个权重张量 | 1 |
| `per_channel` | 每个输出通道（每一行） | `out_features` |
| `per_group` | 每行内固定长度 `group_size` | `out_features × ceil(in_features / group_size)` |

`quantize_weight` 已给出按行、按组切片的双层循环，只缺分组数。参考答案一行：

```python
n_groups = (in_features + effective_group_size - 1) // effective_group_size
```

这是「向上取整」的标准写法：先加 `group_size-1` 再整除，保证最后不足一组的尾巴也占一个组。写错成 `in_features // group_size` 会直接丢掉尾部权重，`quantize_weight` 里 `torch.empty_like` 留下的未初始化码值就会被当成有效数据。测试用两个形状验证：`2×5` 的权重、`group_size=2` 得到 `(2, 3)` 的 `scale`；`group_size=3` 时 5 个特征得到 `ceil(5/3)=2` 组，而不是 1 组。

**第 5 步：反量化与字节账本。** `dequantize_weight` 对逐张量直接算 `(q - zero_point) * scale`；对逐通道/逐组则按 `group_size` 循环切片恢复。`compare_granularity_tradeoffs` 在**同一权重、同一输入**上依次跑 per-tensor、per-channel 和若干 per-group，输出三组指标：
- `weight_mse`：恢复权重与原始权重的均方误差，衡量表示误差；
- `output_mse` / `output_cosine`：用恢复权重算出的输出与浮点参考输出的均方误差和余弦相似度，衡量误差经过线性层后的影响；
- `quantized_weight_bytes`、`qparam_bytes`、`minimum_state_bytes`、`stored_qparam_bytes`、`total_state_bytes`：字节账本。

这里有一个很容易漏的点：对称表示在数学上不需要零点，但教学状态仍然分配了 `zero_point` 张量，所以 `stored_qparam_bytes` 始终把零点算进去；而在对称模式下，`qparam_count`/`qparam_bytes` 只算「理论最少」的 scale。`test_actual_state_bytes_and_bias_reset` 因此断言 `total_state_bytes == quantized_weight_bytes + stored_qparam_bytes`、`stored_qparam_bytes == 2 * qparam_bytes`、并且 `total_state_bytes > minimum_state_bytes`——理论最少不等于实际占用。

拿测试里的 `2×4` 权重、`group_sizes=(2,)` 手算：INT8 权重共 8 字节；per-tensor 有 1 个 scale（4 字节），实际再存 1 个零点（4 字节），总计 16 字节；per-channel 有 2 个 scale，总计 24 字节；per-group 有 `2×2=4` 个 scale，总计 40 字节。这与测试断言的 `qparam_count == [1, 2, 4]` 以及字节数严格递增一致。粒度越细误差通常越小，但元数据从 1 份涨到 4 份。

**第 6 步：`W8A16Linear` 前向（TODO 4）。** 这个模块用 `register_buffer` 保存 `weight_int8`、`scale`、`zero_point`，用 `nn.Parameter` 保存 `bias`。`from_float` 把浮点 `nn.Linear` 转成对称逐通道状态；如果源层没有 bias，会把 `self.bias` 清零，避免重复转换时残留上一层偏置（测试专门验证这一点）。`forward` 的参考答案只有两行：

```python
weight = dequantize_weight(state, x.dtype)          # 恢复到输入的数据类型
output = F.linear(x, weight, self.bias.to(x.dtype)) # bias 也要同类型
```

为什么权重和 bias 都要转成 `x.dtype`？因为 `F.linear` 要求两个矩阵 dtype 一致，FP16 输入配 FP32 权重会直接报错。为什么不能原地修改 `weight_int8`？持久状态是 INT8，反量化只是临时的浮点视图，改掉它就等于每次前向都重新量化，既丢精度也失去「低比特存储」的意义。`test_w8a16_linear_contract` 用显式 `dequantize_weight` 的结果构造参考值，要求 `atol=1e-6` 一致；`test_fp16_activation_contract` 要求 FP16 输入得到 FP16 输出 `(2, 4)`。

**第 7 步：和 PyTorch 框架对账。** `check_w8a16_framework_conformance` 用 `MinMaxObserver(qscheme=torch.per_tensor_symmetric, quant_min=-127, quant_max=127)` 对齐对称路径，用 `torch.fake_quantize_per_tensor_affine` 对齐恢复结果；再对全正、全负、正常量、负常量、全零五组输入跑 `per_tensor_affine` 对照。检查项包括 `scale_match`、`zero_point_match`、`reconstruction_match`、`int8_state`、`code_range_valid`、`qparam_shape_valid`，任何一项失败就抛 `AssertionError`。它核对的是 CPU 上的数值映射，不测低比特内核。

**第 8 步：GPU 实验怎么设计、结果怎么读。** Step 5 用 `Qwen/Qwen2.5-0.5B-Instruct` 的三个真实层 `q_proj`、`gate_proj`、`down_proj`，Token 数取 `1, 32, 256`（覆盖 Decode 到较大 Prefill），把 FP16 Linear 与 torchao `Int8WeightOnlyConfig` 候选配对。自变量是「层角色 × Token 负载」，因变量是物理状态字节、输出误差、延迟和增量峰值显存。测量协议固定 `WARMUP_STEPS=10`、`REPEAT_STEPS=20`、`INNER_LOOPS=100`，用 CUDA Event 分块计时，取中位数和 P95。

判定门槛写在代码里：`state_storage_reduction_ratio > 0.35` 才算容量通过，`output_cosine_similarity > 0.99` 才算质量通过；加速候选必须同时满足中位延迟比 `≤ 0.90`、P95 延迟比 `≤ 0.95`，即中位数至少降 10%、P95 至少降 5%。notebook 正文登记的一次 RTX 5070 Ti 复测中，三类层都约有 50% 的物理状态缩减、余弦相似度高于 0.9999，但 9 组负载里只有 `q_proj × 32 Token` 同时过两个阈值，所以最终只是「容量可接受、存在加速候选」，还不能说 W8A16 已稳定加速。对应到代码里的 `decision`，这类结果会是 `accept_for_capacity_with_speedup_candidates`；如果只有容量和质量通过、没有任何加速候选，则是 `accept_for_capacity_only`。默认 `RUN_MODE="dry_run"` 只登记计划，不会下载模型或跑 GPU。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| W8A16 | 权重用 8 位整数存、激活用 16 位浮点算的表示约定 |
| scale（缩放因子） | 一个整数码代表多少实数，恢复时乘回去 |
| zero point（零点） | 实数 0 对应的整数码；对称量化固定为 0 |
| 对称量化 | 以 0 为中心、正负共用一套码，零点恒为 0 |
| 非对称量化 | 用最小/最大值覆盖实际范围，允许零点偏移 |
| 量化粒度 | 多少个权重共享一套 scale/zero point |
| 逐通道 / 逐组 | 每一行一套 / 每行内每 `group_size` 个元素一套 |
| 尾组（tail group） | 最后一组不足 `group_size` 时仍保留的那组 |
| 反量化 | 用 `(q - zero_point) × scale` 把整数码恢复成浮点 |
| 仅权重量化 | 只压缩权重存储，不保证矩阵乘走整数内核 |
| `absmax` | 权重绝对值最大值，对称量化的范围依据 |
| 字节账本 | 把整数码、scale、zero point 分开计数的容量统计 |

## 小白常见疑问
**Q1：对称量化零点明明是 0，为什么字节账本还要算它？**
因为数学上不需要，不代表代码里没分配。`W8A16Linear.__init__` 用 `register_buffer("zero_point", ...)` 实际创建了 `(out_features, 1)` 的 FP32 张量，`compare_granularity_tradeoffs` 的 `stored_qparam_bytes` 也始终把它计入。`test_actual_state_bytes_and_bias_reset` 用 `stored_qparam_bytes == 2 * qparam_bytes` 明确区分了「理论最少」和「实际保存」。

**Q2：全零张量为什么不能直接让 `scale=0`？**
后面要算 `x/scale` 和 `(q-z)*scale`，`scale=0` 会得到 `NaN` 或除零。参考答案用 `clamp_min(eps)`（默认 `1e-8`）把下界托住，`test_symmetric_qparams_contract` 断言 `torch.isfinite(calculate_qparams(torch.zeros(4), symmetric=True)[0])`，`test_asymmetric_qparams_contract` 要求常量输入的 scale 为正且有限。

**Q3：非对称量化为什么一定要把 0 放进 `[minimum, maximum]`？**
因为零点公式是 `z = round(qmin - minimum/scale)`，它假设 `minimum` 应该映射到 `qmin`。如果数据全正而 `minimum` 取数据最小值，零点裁剪后会偏，端部数值可能饱和。`test_asymmetric_range_and_constant_reconstruction` 对 `[1,3]`、`[-3,-1]`、正负常量和全零都要求恢复误差在 `scale/2 + 1e-6` 以内，就是在防这个边界问题。

**Q4：粒度越细误差越小，那是不是永远选逐组、组越小越好？**
不是。组越小，`scale/zero_point` 的数量按 `out_features × ceil(in_features/group_size)` 增长，字节账本里的 `qparam_bytes` 和 `total_state_bytes` 都会上升；计算内核还要按组索引参数，实现约束更多。notebook 用同一个 `compare_granularity_tradeoffs` 同时报告误差和字节，就是要求两边一起看，而不是只盯 `weight_mse`。

**Q5：为什么权重缩小了约一半，却不一定更快？**
因为教学路径和 eager 路径都要先反量化成浮点权重，再走通用矩阵乘；省下的是存储和读带宽，增加的是反量化开销。Step 3 的表格明确区分了「eager dequantization 能证明表示与误差」「fused 能减少中间权重」「integer kernel 才证明低精度执行」。GPU 实验里 9 组负载只有 1 组过加速阈值，正说明容量收益不能自动翻译成延迟收益。

## 动手验证：怎么知道自己做对了
`test_symmetric_qparams_contract`：`[-2.0, 1.0]` 的零点必须为 0，`qmin/qmax` 必须是 `(-127, 127)`，全零输入的 scale 必须有限。`test_asymmetric_qparams_contract`：`[1.0, 3.0]` 的零点不为 0，范围是 `(-128, 127)`，常量输入的 scale 为正且有限。

`test_granularity_state_shapes`：`2×5` 权重在 per-tensor / per-channel / per-group(`group_size=2`) 下，`scale` 形状分别是 `(1,1)`、`(2,1)`、`(2,3)`，`quantized` 保持 `weight.shape`。`test_tail_group_and_dequantization`：`group_size=3` 的 `2×5` 权重得到 `(2,2)` 的 scale，恢复形状不变且全部有限。

`test_granularity_tradeoff_matrix`：`2×4` 权重、`group_sizes=(2,)` 时 `qparam_count == [1, 2, 4]`，字节数严格递增，`weight_mse` 有限，`total_state_bytes >= quantized_weight_bytes`，`output_cosine` 落在 `[-1, 1]`。`test_w8a16_linear_contract`：教学前向与显式反量化参考在 `atol=1e-6` 下一致，`weight_int8.dtype` 必须是 `torch.int8`。`test_fp16_activation_contract`：FP16 输入得到 FP16 的 `(2,4)` 输出。

`test_actual_state_bytes_and_bias_reset`：`stored_qparam_bytes == 2 * qparam_bytes`，`total_state_bytes > minimum_state_bytes`；先转换一个带 `bias=3` 的层，再转换一个 `bias=False` 的层，后者 `bias` 必须全零。`test_invalid_quantization_inputs`：含 `NaN` 的输入、`eps=0`、空权重、`group_size=1.5` 都必须抛 `ValueError`。全部通过后打印 `✅ W8A16 机制测试通过：qparams、粒度、尾组与反量化前向均已验证。`

notebook 没有保存任何单元格输出，因此具体数值以你运行时打印的字典为准。可选框架核对 `check_w8a16_framework_conformance` 通过时返回 `status: "passed"`、`max_abs_error` 和各项布尔检查；GPU 实验则要读取 `benchmarks/results/25_torchao_w8a16_layer_microbenchmark_<run_label>.json` 里的 `summary`、`decision`、`state_storage_reduction_ratio`、`output_cosine_similarity` 与两种 `latency_ratio`。
