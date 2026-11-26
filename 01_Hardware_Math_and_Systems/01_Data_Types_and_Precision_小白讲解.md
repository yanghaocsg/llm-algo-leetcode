# 01. 大模型的数据格式与混合精度 — 小白讲解

> 对应 notebook：[`01_Data_Types_and_Precision.ipynb`](./01_Data_Types_and_Precision.ipynb)

## 一句话总结
用几个在 CPU 上就能跑的小计算器，把"数据类型（dtype，规定每个数字用多少个位、怎么存）"和三个实际问题连起来：模型权重占多少 GB、FP16 和 BF16 的范围与精度差在哪、混合精度训练为什么要多留一份 FP32 权重。最后再看 A100 上的 TF32 和 H100 上的 FP8（E4M3 / E5M2）是怎么在"怎么存"和"怎么算"之间做取舍的。

## 为什么需要它：痛点是什么
初学者容易把"7B 模型用 FP16 占 14 GB""BF16 比 FP16 稳""FP8 分 E4M3 和 E5M2"当口诀背，一遇到具体问题就卡住：我的显卡装不装得下？训练时梯度为什么变成 0？为什么权重只要 14 GB，训练账本却是 112 GB？这些答案都在同一条链上：数值怎么表示 → 每个参数占几字节 → 该选哪种精度。
类比：两把尺子总刻度都是 16 格，但分法不同。FP16 把格子多分给"细刻度"，量程短、读数精细，像卡尺；BF16 把格子多分给"量程"，读数粗，但能量很大的数，像卷尺。格子总数固定，量程和精度只能二选一。

## 核心问题
- 同一个模型，用 FP32、FP16、BF16、INT8、INT4 存，权重各占多少 GB？
- 同样是 16 位，FP16 和 BF16 的位分布差在哪？谁的范围大、谁更精细？
- 损失缩放能救 FP16 的哪种问题，救不了哪种？
- 混合精度训练为什么常保留一份 FP32 主权重？训练状态账本怎么逐项算？
- A100 的 BF16 / TF32、H100 的 FP8 E4M3 / E5M2，为什么要把"存储格式"和"计算路径"分开看？

## 解决思路：一步步看懂
1. **位宽换算成字节（Q1）。** 1 Byte（字节）= 8 bits（位）。FP32 是 32 位 = 4 字节；FP16 和 BF16 都是 16 位 = 2 字节；INT8 是 8 位 = 1 字节；INT4 是 4 位 = 0.5 字节（通常用于极度压缩的量化，如 AWQ/GPTQ）。`calculate_model_memory(num_params_b, dtype)` 的算法就是"参数量 × 每个参数的位数 ÷ 8 ÷ 10⁹"，参数量和 GB 都按十进制算（`PARAMS_PER_BILLION` 和 `BYTES_PER_DECIMAL_GB` 都是 1_000_000_000）。手算一遍：7B 模型用 FP16，7 × 10⁹ × 16 ÷ 8 ÷ 10⁹ = 14 GB。这只是**理论下界**：真实模型还可能带 scale、zero-point、packing、padding 和运行时 workspace，它也不是推理或训练的峰值显存。对照单元可以改 `MODEL_PARAMS_B`（默认 7）和 `DTYPE_CANDIDATES`（`MODEL_NAME` 只影响输出标签），看容量是否随参数量、位宽线性变化。
2. **指数管范围，尾数管精度（Q2）。** 一个浮点数由符号位（正负）+ 指数位（数能有多大、多小）+ 尾数位（同一量级内能分多细）组成。FP16 是 1 + 5 + 10：指数只有 5 位，最大只能到 65504，但尾数长、小数部分"精细"。BF16 是 1 + 8 + 7：指数和 FP32 一样是 8 位，最大有限值约 3.4 × 10³⁸，不容易因范围不足而溢出，代价是尾数从 10 位降到 7 位、舍入更粗——但它不等于不会出现 Inf/NaN。`cast_error_report(values, dtype_name)` 把同一组 FP32 数转成目标格式再转回 FP32，报告最大有限值 `max_finite`、1 附近的间隔 `eps_near_one`、有限值个数 `finite_count`、溢出个数 `overflow_count` 和 `max_abs_error`（误差只统计转换后仍有限的元素，全溢出时记 `None`）。探针 `PROBE_VALUES = [1.0, 1.001, 60_000.0, 70_000.0]` 里，1.001 用来压精度，70_000 超过 65504 用来压范围。
3. **损失缩放救的是下溢（Q3）。** FP16 范围窄，两头都危险：前向值太大会变 `Inf`（上溢），梯度太小会在转换时变成 0（下溢）。损失缩放（loss scaling）先把梯度乘一个缩放因子，在低精度里过一遍，再除回来：

   ```python
   low_precision = (source * loss_scale).to(target_dtype)   # 先放大，再转低精度
   restored = low_precision.to(torch.float32) / loss_scale  # 转回 FP32，再缩小
   ```

   `gradient_survival_report` 用 `TINY_GRADIENTS = [1e-4, 1e-6, 1e-8, 1e-10]` 对比三组：FP16 不缩放、FP16 缩放 1024.0、BF16 不缩放，统计"非零""有限""有限且非零"的个数。读法：缩放后"有限且非零"的个数不应少于不缩放，说明更多小梯度活了下来。它**不能修复前向阶段已经发生的上溢**；BF16 指数范围大，通常较少依赖它，但尾数更短，舍入误差可能高于 FP16。
4. **训练状态账本与 FP32 主权重（Q4）。** 参数更新是 $W_{new} = W_{old} - \mathrm{LR} \times \mathrm{Gradient}$：$W$ 是权重，$\mathrm{LR}$ 是学习率（每次走多大步），$\mathrm{Gradient}$ 是梯度。权重若只用 16 位存，很小的更新量会在舍入时消失，所以一些混合精度实现保留 FP32 主权重（Master Weights）：先更新这份高精度副本，再转成前向用的格式。`training_state_ledger` 默认配置是参数 2 字节、梯度 2 字节、Adam 优化器的一阶矩和二阶矩（优化器为每个参数额外记的两个统计量）各 4 字节，主权重可选 4 字节。7B 手算：14 + 14 + 28 + 28 + 28 = 112 GB；`keep_master_weights=False` 时是 84 GB，差额 28 GB 就是这份副本。账本只算静态训练状态（`scope` 为 `static_training_states_only`），激活和临时缓冲不在内；组件逐项列出，是为了避免把"每参数 12 字节"和 Master Weights 重复相加。
5. **A100 的两条路径（Q5）。** 关键是分开看"张量如何存储"和"矩阵乘法如何执行"。BF16 可以作为 16 位 Tensor Core 的输入，存储宽度和计算路径一起变；TF32 则**不是**需要单独保存的 19-bit 模型 dtype，输入输出仍按 FP32 存，只是矩阵乘法内部走精度较低的 Tensor Core 路径。V100 的 Tensor Core 乘加路径主要用 FP16 输入。`a100_precision_path(requested_dtype, allow_tf32=True)` 只返回理论候选：存储格式、计算路径、累加格式（四种都是 fp32）、是不是 Tensor Core 候选，外加 `requires_gpu_validation: True`——`torch.backends.cuda.matmul.allow_tf32` 这类开关的实际效果还取决于 PyTorch、CUDA、算子和硬件。
6. **H100 的 FP8：E4M3 还是 E5M2（Q6）。** 只有 8 位，指数和尾数更要取舍：E4M3（4 位指数 + 3 位尾数）侧重精度，常用于前向传播和激活值；E5M2（5 位指数 + 2 位尾数）侧重动态范围，常用于梯度。常见的 `HYBRID` recipe（前向、反向各用哪种格式的搭配方案）是前向 E4M3、反向 E5M2，但不是固定规则。用之前要缩放：先取张量的绝对最大值 `amax`，缩放因子 = 格式最大有限值 ÷ `amax`。`FP8_MAX_FINITE` 里 E4M3 是 448.0、E5M2 是 57_344.0，所以 `amax=2` 时 E4M3 的缩放因子是 448 ÷ 2 = 224。`fp8_variant_policy(recipe, forward_amax, backward_amax)` 对 `hybrid`、`e4m3`、`e5m2` 三种策略分别算前向、反向缩放因子；按公式手算，默认 `FORWARD_AMAX = 12.0`、`BACKWARD_AMAX = 800.0` 时 hybrid 的前向是 448 ÷ 12 ≈ 37.3，反向是 57344 ÷ 800 = 71.68，而 e4m3 策略的反向只有 448 ÷ 800 = 0.56——小于 1 说明张量太大，要先缩小才塞得进这个格式。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| dtype（数据类型） | 规定每个数字用多少位、这些位怎么解释 |
| FP32 / FP16 / BF16 | 分别用 32 / 16 / 16 位存的浮点数；后两者位数相同但分法不同 |
| 符号位 / 指数位 / 尾数位 | 正负号 / 数的范围 / 同一量级内的精细程度 |
| eps（机器精度） | 1 和比 1 大的下一个可表示数之间的间隔，越小越精细 |
| 上溢 / 下溢 | 数太大变成 Inf / 数太小变成 0 |
| 损失缩放（loss scaling） | 先把梯度放大、低精度过完再缩小，缓解小梯度下溢 |
| Master Weights（主权重） | 额外保留的 FP32 权重副本，专供参数更新 |
| Adam 一阶矩 / 二阶矩 | 优化器为每个参数额外维护的两个统计量 |
| TF32 | A100 上让 FP32 输入走 Tensor Core 的计算路径，不是存储格式 |
| FP8 E4M3 / E5M2 | 8 位浮点的两种分法：前者偏精度，后者偏动态范围 |
| amax / 缩放因子 | 张量的绝对最大值 / "格式最大有限值 ÷ amax" |
| recipe | 前向、反向各用哪种 FP8 格式的搭配方案 |

## 小白常见疑问
**Q1：BF16 范围大、更不容易溢出，为什么不干脆全用 BF16？**
范围不是唯一指标。BF16 的尾数只有 7 位，舍入误差可能高于 FP16。notebook 的对照表强调：两者都要用固定输入检查有限值、零值比例和误差，不能只根据格式名称判断稳定性。

**Q2：损失缩放能把前向的 Inf 也救回来吗？**
不能。它只对梯度动手，前向阶段已经发生的上溢修不了。BF16 的指数范围接近 FP32，通常不需要损失缩放，但仍要做有限值检查。

**Q3：Q1 算出 14 GB，Q4 算出 112 GB，到底哪个对？**
都对，统计口径不同。14 GB 只是权重这一份的理论下界；112 GB 是静态训练状态，还包含梯度、Adam 的两个矩和 FP32 主权重。两者都不含激活和临时缓冲。

**Q4：把模型"存成 TF32"能省显存吗？**
不能。TF32 不是需要单独保存的模型 dtype，`a100_precision_path('fp32')` 返回的 `storage_dtype` 仍然是 `'fp32'`，省不了存储，变的只是矩阵乘法的计算路径。

**Q5：打开 `allow_tf32` 或用了 BF16，就一定跑在 Tensor Core 上、一定更快吗？**
不一定。`a100_precision_path` 的返回里 `requires_gpu_validation` 恒为 `True`，它不估算 speedup，也不证明真实 Tensor Core 执行；真实情况要固定 workload 后用 profiler 验证。

## 动手验证：怎么知道自己做对了
`test_calculate_model_memory` 断言：7B 用 fp16 得 14、int8 得 7，13B 用 fp16 得 26，70B 用 int4 得 35，14B 是 7B 的 2 倍（线性），传未知 dtype 抛 `ValueError`。`test_cast_error_report` 用 `[1.0, 1.001, 60_000.0, 70_000.0]`：FP16 有限值 3 个、溢出 1 个；只给 `70_000.0` 时溢出 1 个且 `max_abs_error` 为 `None`；BF16 有限值 4 个；FP16 的 `eps_near_one` 小于 BF16 的；传 `'int8'` 抛 `ValueError`。`test_training_state_ledger` 断言 7B 保留主权重时参数 14 GB、一阶矩 28 GB、二阶矩 28 GB、主权重 28 GB，且保留与不保留的 `total_gb` 相差 28。
Q3、Q5、Q6 没有独立的测试函数，断言直接写在单元里：缩放后"有限且非零"的个数 ≥ 不缩放，BF16 不缩放时全部有限；`allow_tf32=False` 时 fp32 的 `compute_path` 为 `'fp32'`，bf16 的 `tensor_core_candidate` 为 `True`，传 `'int8'` 抛错并打印"✅ dtype 边界校验通过"；hybrid 前向 E4M3、反向 E5M2，`scaling_required` 为 `True`，`amax` 都是 2.0 时前向缩放因子为 224.0，未知 recipe 抛错并打印"✅ 非法 FP8 recipe 校验通过"。notebook 没有 benchmark 单元格，dump 里也没有保存任何输出，改参数后以你自己打印出的字典为准。
