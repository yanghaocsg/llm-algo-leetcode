# QUANT-QAT. 量化感知训练 — 小白讲解

> 对应 notebook：[`QUANT-QAT_Quantization_Aware_Training.ipynb`](./QUANT-QAT_Quantization_Aware_Training.ipynb)

## 一句话总结
量化感知训练（QAT）不直接把权重砍成低精度，而是保留一份可更新的浮点主权重，在前向里用“伪量化”模拟裁剪、舍入和恢复，反向用直通估计器（STE）绕过舍入的零梯度，让权重学会补偿离散误差。本节用三个 TODO 搭出 `FakeQuantizer` 与 `QATLinearSim`，再用可选的 GPU 实验比较 INT4 训练后量化（PTQ）与 5／20／50 步 QAT 的质量恢复率。

## 为什么需要它：痛点是什么
把训练好的 FP16/BF16 模型直接转成 INT8 或 INT4，权重和激活被舍入到很粗的格点，预测质量可能明显下降。最直觉的补救是“再训练几步”，但这里有一个数学障碍：舍入函数几乎处处导数为 0，梯度传不回去，优化器不知道该往哪走。另一个常见误解是“QAT 就是把权重存成整数继续训练”——真正的 QAT 训练时仍然更新浮点主权重，只是前向看到的是量化后的近似值。
生活类比：戴一副故意磨花的眼镜练投篮。训练时你看到的篮筐是模糊的（伪量化前向），但教练仍按你真实的投篮动作纠正你（浮点主权重的梯度）。比赛时换成同样度数的磨花眼镜（转换后的低精度表示），你已经学会在模糊视野下调整力度，而不是等比赛当天才第一次戴。

## 核心问题
- 伪量化怎样在浮点张量里模拟“缩放、裁剪、舍入、恢复”？
- 为什么 `round` 不能直接求导？STE 用什么公式把梯度放过去？
- 统计观测器的范围什么时候更新、什么时候冻结？
- 激活和权重为什么要各自维护一份范围？优化器更新的是谁？
- QAT 训练省下的质量，怎样在 `convert` 之后用留出 NLL 验证？
- 5／20／50 步的预算曲线怎样选？恢复率 103.4% 能不能说明 QAT 真的超过 BF16？

## 解决思路：一步步看懂

**第 1 步：伪量化的前向公式。** notebook 给出一般仿射形式：

$$
\widetilde{x}=s\left(\mathrm{clip}\left(\mathrm{round}(x/s)+z,q_{\min},q_{\max}\right)-z\right).
$$

$x$ 是浮点值，$s>0$ 是每个整数码对应的浮点步长，$z$ 是零点，$[q_{\min},q_{\max}]$ 是整数码范围，$\widetilde{x}$ 是恢复后的浮点近似。本节采用对称 INT8：$z=0$、$q_{\max}=127$、$q_{\min}=-127$。代码没有直接算 $s$，而是先取范围 `absmax`，令 `scale = qmax / absmax`（也就是 $1/s$），再写 `dequantized = round(clipped * scale) / scale`。两种写法等价：先乘 `127/absmax` 得到整数码附近的值，舍入后再乘回 `absmax/127`。

**第 2 步（TODO 1）：观测器范围与 EMA。** `FakeQuantizer.update_observer` 先用 `tensor.detach().float().abs().max().clamp_min(self.eps)` 取当前绝对最大值：`detach` 表示统计量不参与求导，`clamp_min(eps)` 防止全零张量带来除零。参考答案：

```python
self.observed_absmax = (
    current if self.observed_absmax is None
    else self.momentum * self.observed_absmax + (1.0 - self.momentum) * current
)
```

第一次观察直接赋值；之后按 `momentum=0.9` 做指数移动平均，把历史范围和当前批次混合，降低单个批次的波动。它解决的是“范围由谁决定”：训练时每个 batch 都可能出现新的极值，如果每步都取当前最大值，量化步长会剧烈抖动；EMA 让范围平滑。`test_observer_updates_then_freezes` 用 `momentum=0.5`，先对 `[-1.0, 2.0]` 观察得到 2.0，再用 `update_observer=False` 喂一个 100.0，断言范围仍是 2.0——评估路径冻结，不会被异常值改写。

**第 3 步（TODO 2）：STE 的“前向离散、反向连续”。** 舍入的真实导数几乎处处为 0，直接反传会让权重收不到信号。直通估计器把梯度近似成“只要还在可表示区间内就原样通过”：

$$
\frac{\partial\widetilde{x}}{\partial x}\approx
\mathbf{1}(x_{\min}\le x\le x_{\max}).
$$

notebook 的写法是：

```python
fake_quantized = clipped + (dequantized - clipped).detach()
```

前向时 `detach()` 不改变数值，所以结果等于 `dequantized`；反向时 `dequantized - clipped` 这一项没有梯度，真正参与求导的是 `clipped`。而 `clipped = clamp(tensor, -absmax, absmax)` 的导数在区间内是 1、饱和区是 0，正好就是上面的指示函数。容易写错成直接返回 `dequantized`（前向对、反向全断）或者 `tensor + (dequantized - tensor).detach()`（忘了饱和区截断，越界样本也会拿到梯度）。`test_ste_passes_in_range_gradient_and_clips_saturation` 先观察 `[-1.0, 1.0]`，再对 `[-2.0, 0.5]` 反传：`grad[0] == 0.0`（−2 被裁到 −1，饱和），`grad[1] == 1.0`（0.5 在区间内；尽管它前向被舍入到约 0.5039，梯度仍按 1 传）。

**第 4 步（TODO 3）：双路伪量化的线性层。** `QATLinearSim` 保留 `self.weight = nn.Parameter(...)` 作为浮点主权重，另外建 `activation_quantizer` 和 `weight_quantizer` 两个独立观测器。参考答案：

```python
quantized_inputs = self.activation_quantizer.fake_quant(
    inputs, update_observer=self.training
)
quantized_weight = self.weight_quantizer.fake_quant(
    self.weight, update_observer=self.training
)
outputs = F.linear(quantized_inputs, quantized_weight)
```

两个 `update_observer=self.training` 是训练/评估切换的开关：训练时更新范围，`layer.eval()` 后复用已有范围。激活和权重必须各有一份范围，因为它们的数值分布完全不同；共用一个会把权重的范围套到激活上，量化步长失真。`F.linear` 用的是伪量化后的值和权重，但计算图会通过 STE 把梯度送回 `self.weight`，所以优化器更新的是浮点主权重，不是某个整数副本。`test_qat_linear_training_and_eval_contract` 用 `QATLinearSim(4, 3)` 和 `(5, 4)` 输入检查输出形状 `(5, 3)`；`eval()` 后用 `inputs * 10` 再前向，断言 `activation_quantizer.observed_absmax` 完全没变。`test_qat_step_updates_float_master_weight` 用 SGD（`lr=0.1`）和 MSE 走一步，断言 `weight.grad is not None` 且更新后的权重不等于更新前。

**第 5 步：框架一致性核对。** 把 `observed_absmax` 固定为 1.0，本节伪量化在 `[-2.0, -0.5, 0.0, 0.5, 2.0]` 上的输出应与 `torch.fake_quantize_per_tensor_affine(x, scale=1/127, zero_point=0, -127, 127)` 一致（`atol=1e-7`），梯度逐元素相等；饱和点梯度为 0，内部三点梯度为 1。观测器侧用 `MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-127, quant_max=127)`，检查 `local_absmax / 127` 等于框架 scale、零点为 0。它说明教学实现和 PyTorch 的对称逐张量伪量化语义对齐，但不代表真实 INT4 内核已经验证。

**第 6 步：GPU 实验的变量与指标。** 固定 `Qwen/Qwen2.5-0.5B-Instruct` 的第 `TARGET_BLOCK_INDEX=12` 个 Block，数据是 WikiText-2 固定快照，训练 128 条、验证 64 条、测试 64 条，`MAX_LENGTH=128`，`GROUP_SIZE=64` 的 INT4 仅权重量化，学习率 `1e-5`。候选有四个：BF16 基线、直接 PTQ、以及 5／20／50 步 QAT（prepare → 训练 → convert）。自变量是训练预算；因变量是验证 NLL（用于选预算）、测试 NLL（只评一次）、质量恢复率、训练秒数、峰值显存。恢复率的定义是

$$
\text{recovery}=\frac{(\mathrm{NLL}_{PTQ}-\mathrm{NLL}_{BF16})-(\mathrm{NLL}_{QAT}-\mathrm{NLL}_{BF16})}{|\mathrm{NLL}_{PTQ}-\mathrm{NLL}_{BF16}|},
$$

也就是“QAT 把 PTQ 相对 BF16 的损失差距补回了多少”。分母用绝对值，避免差距为负时符号翻转。

**第 7 步：结果怎么读。** 历史 `default2` 的验证 NLL 随预算单调下降：5 步 3.569864、20 步 3.568311、50 步 3.566708，因此选中 50 步。重建后测试 NLL：BF16 3.379010、PTQ 3.387613、QAT 3.378714。PTQ 差距约 0.008603，QAT 差距约 −0.000296（比 BF16 还略低），恢复率约 `0.008899 / 0.008603 = 1.0343`，即 103.4%。决策 `accept_qat_candidate` 的条件是恢复率大于 0 且测试 NLL 相对 BF16 的比例不超过 `MAX_TEST_NLL_RATIO=1.10`。这里的差值只有千分之一量级，5.4 明确提醒“重复样本与随机种子复测才能判断是否稳定”，不能把 103.4% 读成“QAT 一定优于 BF16”。保存的 `block_12_50_steps.pt` 约 15.6 MB（15,624,533 字节），是参考 Block 状态，`storage_compression_evidence` 和 `kernel_evidence` 都是 `False`，不能当作可部署的压缩模型。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 伪量化（fake quant） | 在浮点张量里模拟“缩放→舍入→恢复”，前向看到低精度误差 |
| 浮点主权重 | 优化器真正更新的那份高精度权重副本 |
| 缩放因子 $s$ / `scale` | 一个整数码代表多少浮点值；代码里存的是它的倒数 |
| 零点 $z$ | 整数码 0 对应的浮点值；对称量化取 0 |
| 统计观测器 | 记录张量绝对最大值，用来算量化参数 |
| EMA（指数移动平均） | 用 `momentum` 把历史范围和当前范围混合，减少抖动 |
| STE（直通估计器） | 前向用离散值，反向假装舍入不存在，区间内梯度为 1 |
| 饱和区 | 超出 `[-absmax, absmax]` 的部分，梯度被 `clamp` 截成 0 |
| 逐张量 / 分组量化 | 整个权重一个 scale / 每 64 个权重一个 scale |
| PTQ / QAT | 训练后直接量化 / 训练中模拟量化误差再转换 |
| convert | 把 QAT 模块真正转成目标低精度表示 |
| 质量恢复率 | QAT 补回的 PTQ 相对 BF16 损失差距占比 |

## 小白常见疑问
**Q1：代码里的 `scale = qmax / absmax`，和公式里的 $s$ 是同一个东西吗？**
不是，它正好是 $s$ 的倒数。公式里 $s=\mathrm{absmax}/q_{\max}$，整数码是 `round(x/s)`；代码为了少做除法，先算 `scale=qmax/absmax`，于是 `round(x*scale)` 就是整数码，恢复时再 `/scale`。`framework_check` 里用 `local_absmax / 127.0` 去对 PyTorch 的 `scale`，就是在还原这个 $s$。

**Q2：STE 前向明明是 `dequantized`，为什么测试里 0.5 的梯度是 1？**
因为梯度只看计算图里的 `clipped`。`fake_quantized = clipped + (dequantized - clipped).detach()`，`dequantized` 被 `detach` 后不贡献梯度，剩下 `clipped` 对输入的导数在区间内为 1。前向数值仍然等于 `dequantized`，所以“前向离散、反向连续”可以同时成立。

**Q3：评估时把输入乘 10，为什么观测器范围不变？**
`forward` 里传的是 `update_observer=self.training`，`layer.eval()` 后 `self.training` 为 `False`，`fake_quant` 直接跳过 `update_observer`，用训练时存下的 `observed_absmax`。`test_qat_linear_training_and_eval_contract` 就是拿 `inputs * 10` 来验证这一点；它只断言形状和范围，不保证评估输出数值正确。

**Q4：QAT 训练后，保存的权重是整数吗？**
训练过程中 `self.weight` 始终是 `nn.Parameter` 浮点张量，优化器更新它；`convert` 之后才变成目标表示。GPU 实验里 `quantize_(target_block, QATConfig(..., step="convert"))` 之后才评估和保存，保存的 `state_dict` 是参考 Block 状态，不是完整可部署模型。

**Q5：为什么 50 步的验证 NLL 最低，却只比 5 步低 0.003？**
因为这里只训练一个 Block，其他参数冻结，且学习率只有 `1e-5`；QAT 的主要收益来自“让前向看到量化误差”，预算增加带来的边际改善本来就小。选预算是按验证 NLL 的 `min`，不是按“提升显著”的统计检验，所以 50 步只是当前候选里的最小点，不代表更多步一定更好。

## 动手验证：怎么知道自己做对了
`run_qat_tests()` 依次跑四个测试，全过打印 `✅ QAT 机制测试通过：observer、STE、双路径前向和主权重更新均已验证。`。`test_observer_updates_then_freezes` 要求首次观察后范围等于当前 absmax，评估分支喂 100.0 后范围逐元素相等不变。`test_ste_passes_in_range_gradient_and_clips_saturation` 要求 `inputs.grad[0] == 0.0`、`inputs.grad[1] == 1.0`。`test_qat_linear_training_and_eval_contract` 要求训练与评估输出都是 `(5, 3)`，且评估前后 `observed_absmax` 相等。`test_qat_step_updates_float_master_weight` 要求 `weight.grad is not None` 且权重在一步 SGD 后发生变化。

可选核对 `check_qat_framework_conformance` 的六项：`forward_match`（1e-7）、`gradient_match`（逐元素相等）、`saturation_gradient_zero`、`interior_gradient_one`、`observer_scale_match`、`observer_zero_point_is_zero`。没装 PyTorch 对应算子时该单元会报错，而不是静默跳过。GPU 结果以 JSON 的 `qat_candidates`、`selected_candidate`、`recovery_ratio_vs_ptq`、`decision` 为准；notebook 保存的 `default2` 输出显示选中 `qat_50_steps`、恢复率 1.0343、决策 `accept_qat_candidate`，但 `kernel_evidence` 明确是 `False`，不要把它当成 INT4 内核已经跑通的证据。
