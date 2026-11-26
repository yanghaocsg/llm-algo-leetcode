# 11. 梯度累积 — 小白讲解

> 对应 notebook：[`11_Gradient_Accumulation.ipynb`](./11_Gradient_Accumulation.ipynb)

## 一句话总结
梯度累积把一个大逻辑批次拆成多个微批次，每个微批次单独前向和反向，但梯度先不清零，等累计够 $K$ 个微批次后再做一次参数更新。本节用三个 TODO 实现“同步切片、按 $1/K$ 缩放损失、识别更新边界”，再用 `TinyRegressor` 对照完整批次，证明两条路径的损失、参数和输出在容差内一致；最后用 GPU 实验观察峰值显存与 step 时间如何随累积次数变化。

## 为什么需要它：痛点是什么
初学者常把“显存不够”直接等同于“把 batch 调小”，但 batch 一小，一次参数更新看到的样本就少了，优化器的更新次数和梯度噪声都会变。另一种背口诀式的理解是“梯度累积能省显存”，于是以为参数、梯度、优化器状态也会跟着变小，结果发现显存几乎没降，或者累积之后训练效果和 full batch 对不上。

生活类比：搬一车货，车太小装不下。梯度累积不是把货变少，而是分几趟搬：每趟只装一部分（微批次），搬完一趟先把货堆在仓库（梯度缓冲区），等全部搬完再一次性入库（参数更新）。这样每趟的“车上占用”（激活值峰值）降了，但仓库（参数、梯度、优化器状态）一直占着。

## 核心问题
- 微批次、累积次数、有效批次、参数更新这四个概念分别指什么，彼此怎么换算？
- 梯度累积到底降低了哪一部分显存，哪些状态仍然长期驻留？
- 等长微批次为什么要把每个损失除以 $K$？不等长时为什么要改成 $n_i/N$？
- 怎样保证所有输入字段用同一个切片范围，不出现样本与标签错位？
- 一次逻辑批次应该在哪一步调用 `optimizer.step()`，怎样用边界函数判断？
- GPU 上累积次数增加后，峰值显存和 step 时间一定一升一降吗？

## 解决思路：一步步看懂

**第 1 步：先把四个对象分清楚。** 微批次是单次前向和反向处理的样本，决定单次激活值峰值；累积次数是一个逻辑更新包含多少次微批次计算；有效批次是一次参数更新看到的样本总量，等于微批次大小 × 累积次数；参数更新是梯度写回参数的一次事件，每个有效批次恰好一次。notebook 的表格用这四行把“计算批次”和“更新批次”拆开，后面所有判断都围绕这个区分。

**第 2 步：梯度累积改变的是激活峰值，不是全部显存。** 把一个逻辑批次拆开后，每次只为当前微批次建立激活值，反向完成后这些激活可以释放，所以单次计算的峰值通常下降。但参数、梯度和优化器状态仍跨所有微批次驻留；梯度缓冲区还要在微批次之间持续累积，直到这次有效更新完成才能清空。输入和监督目标则要分批进入模型，所有字段必须同步切分。notebook 的结论很明确：梯度累积是“容量—时间交换”，不是无条件的显存优化。

**第 3 步：推导与完整批次对齐的梯度权重。** 设完整批次被均分为 $K$ 个微批次，每个微批次都计算样本平均损失。要让累计梯度等于完整批次的平均梯度，第 $i$ 个微批次的梯度权重应为 $1/K$：

$$
\nabla L = \sum_{i=1}^{K} \frac{n_i}{N}\nabla L_i,\qquad N=\sum_{i=1}^{K}n_i.
$$

其中 $n_i$ 是第 $i$ 个微批次的样本数，$N$ 是有效批次总样本数，$\nabla L_i$ 是该微批次的平均梯度。当所有微批次等长，即 $n_i=N/K$ 时，式子化为 $\nabla L=\frac{1}{K}\sum_{i=1}^{K}\nabla L_i$。本节固定形状的 MSE 每条样本含相同数量的目标元素，所以按样本数加权成立。如果换到 SFT 按有效词元平均，权重应改成各微批次有效目标数的占比：notebook 举例，两个微批次各含一条样本，却分别有 2 与 6 个目标，权重是 $1/4$ 与 $3/4$，而不是各 $1/2$。有效目标要在移位和标签屏蔽之后统计，实际做法在 15 节。

**第 4 步：TODO 1——同步切分微批次。** 骨架已经算好 `micro_size = batch_size // accum_steps`，并检查了 `accum_steps > 0`、`idx` 范围、所有张量第一维一致、batch 能被整除。参考答案：

```python
start = idx * micro_size
end = (idx + 1) * micro_size
micro_batch = {key: value[start:end] for key, value in batch.items()}
```

`start:end` 是半开区间，第 `idx` 个切片覆盖 `[idx*micro_size, (idx+1)*micro_size)`，连续、不重叠。必须对字典里所有字段用同一组 `start/end`：如果只切 `input_ids` 而忘记 `labels`，样本和监督目标就会错位。`test_micro_batch_slicing` 用 8 行数据、`idx=1, accum_steps=4`，断言每个字段都等于原张量的 `[2:4]`。

**第 5 步：TODO 2——缩放平均损失。** 参考答案：

```python
scaled_loss = micro_loss / accum_steps
scaled_loss.backward()
total_loss += micro_loss.detach().item() / accum_steps
```

`micro_loss` 是当前微批次的平均损失，除以 `accum_steps` 后反传，累计梯度才是完整批次平均梯度的 $1/K$ 加权和。注意记录日志时用的是**未缩放的** `micro_loss` 再除以 `accum_steps`，这样 `total_loss` 与 `train_step_full_batch` 返回的平均损失口径一致；如果把 `scaled_loss` 直接累加，数值会小 $K$ 倍。`test_accumulated_update_equivalence` 用 `abs(loss_full - loss_accum) < 1e-6` 检查这一点。

**第 6 步：TODO 3——识别有效更新边界。** 参考答案一行：

```python
update_now = completed_micro_steps % accum_steps == 0
```

只有已完成的微批次数达到累积次数的整数倍时，才构成一次有效更新。骨架在 `train_step_with_accumulation` 里每处理完一个微批次就调用 `is_update_boundary(idx + 1, accum_steps)`，为真时执行 `optimizer.step()` 并 `optimizer.zero_grad()`。常见错误是每个微批次都 step，或者循环结束后忘记 step；前者会把一次逻辑更新拆成 $K$ 次小更新，后者则根本没有更新。`test_update_boundary` 对 `range(1, 9)` 逐个检查，期望序列是 `[False, False, False, True, False, False, False, True]`。

**第 7 步：测试如何证明“等价”。** `test_accumulated_update_equivalence` 深拷贝同一个 `TinyRegressor`，两条路径都用 `SGD(lr=0.1)`。完整批次走 `train_step_full_batch`；累积路径走 `train_step_with_accumulation(..., accum_steps=4)`，并用 `counted_accum_step` 包住 `optimizer.step` 统计调用次数。断言三件事：两条路径记录的 loss 差小于 `1e-6`；`accum_calls[0] == 1`（一次逻辑批次只更新一次）；更新后模型输出和每个参数都在 `atol=1e-6` 内一致。`test_gradient_accumulation_input_contract` 再检查非法输入：`accum_steps` 为 0 或 3（8 不能被 3 整除）抛 `ValueError`，`x` 与 `y` 的 batch 维不一致抛 `ValueError`，`is_update_boundary(0, 4)` 和 `(1, 0)` 也抛 `ValueError`。

**第 8 步：可选的框架一致性核对。** `check_gradient_accumulation_framework_conformance` 用 Hugging Face Accelerate 的 `Accelerator(cpu=True, gradient_accumulation_steps=2)`，把 4 条样本分成 `[0:2]`、`[2:4]` 两个微批次。每次 `accelerator.accumulate(accum_model)` 内记录 `accelerator.sync_gradients`，期望 `sync_pattern == [False, True]`——第一个微批次不同步，第二个才同步；最后用 `torch.allclose(..., atol=1e-6, rtol=1e-6)` 比较完整批次与累积路径的参数。如果环境里没有安装 `accelerate`，这个单元会把 `framework_check` 置为 `{'status': 'skipped', ...}`，不算失败。

**第 9 步：GPU 实验怎么设计、结果怎么读。** `GPU_CONFIG` 固定 `effective_batch=4`、`micro_batch_counts=[1, 2, 4]`、`seq_len=128`、`warmup_steps=1`、`repeat_steps=20`、`learning_rate=1e-4`、`seed=42`，模型是 `Qwen/Qwen2.5-0.5B`。三条路径共用模型初值、样本顺序和更新次数，只改变每次更新用几个微批次；`run_effective_update` 按有效目标数加权（`loss_weight = micro_target_count / valid_target_count`），每条路径都做同样次数的有效更新。测量时先 `torch.cuda.reset_peak_memory_stats`，用 CUDA Event 计 step 时间，记录 `peak_allocated_mb` 和 `peak_incremental_mb`；`peak_allocated_mb` 包含训练及训练后的评测，不是“纯训练峰值”。正确性判据是：`update_delta_max > 0`（参数确实动了）、`parameter_max_abs_diff <= tolerance`、`loss_abs_diff <= tolerance`，FP32 的容差是 `2e-4`；容量改善判据是 `measured[-1]["peak_allocated_mb"] < baseline["peak_allocated_mb"]`。决策规则：正确性通过且容量改善为 `accept`，正确性通过但容量没改善为 `tune`，正确性不通过为 `reject`。

notebook 保存了一次 5.4 节标记为 `default12`、序列长度 128 的执行输出：`accum=1` 时 `median=243.24 ms`、`p95=288.53 ms`、`peak=4844.4 MB`、`incremental=2939.7 MB`；`accum=2` 时 `median=266.68 ms`、`peak=5361.9 MB`；`accum=4` 时 `median=345.79 ms`、`peak=5357.0 MB`。三条路径的 `update_delta=4.00e-06`、`parameter_diff=0.00e+00`，`correctness={'tolerance': 0.0002, 'passed': True}`，最终 `decision=tune`。这组数字说明：在这个序列长度下，梯度驻留和状态占用盖过了激活节省，峰值没有下降，反而略升；5.4 节另外记录了序列 256 的对照，2/4 次累积分别降低约 6.7%/17.3% 的峰值，但 median 增加约 21.4%/33.5%，那组结论是 `accept`。所以读结果时不能只看“累积次数越多越省显存”，要同时看序列长度、状态占比和 step 时间。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 微批次（micro-batch） | 单次前向和反向处理的样本，决定单次激活峰值 |
| 累积次数（accum_steps） | 一次参数更新包含几个微批次 |
| 有效批次（effective batch） | 一次更新看到的样本总量 = 微批次大小 × 累积次数 |
| 梯度缓冲区 | 跨微批次保留、等更新时才清空的梯度存储 |
| 损失缩放 | 把每个微批次损失除以 $K$，让累计梯度等于完整批次平均梯度 |
| 更新边界 | 已完成微批次数达到累积次数整数倍的那一刻 |
| 激活值峰值 | 单次前向/反向临时占用的显存，累积主要降低这一项 |
| `slice_micro_batch` | 用同一组 `[start:end]` 切分字典里所有字段 |
| `is_update_boundary` | 判断当前是否该执行 `optimizer.step()` |
| `peak_incremental_mb` | 相对本次测量前已分配显存的增量峰值 |

## 小白常见疑问

**Q1：为什么除以 `accum_steps` 就够了，不用再乘微批次大小？**
因为每个微批次的损失已经是 `MSELoss(reduction="mean")`，即该微批次内的样本平均。$K$ 个等长微批次的平均损失再各乘 $1/K$，加权和正好等于完整批次的平均损失。如果损失用的是 `sum` 而不是 `mean`，权重就要改成 $n_i/N$；notebook 在 SFT 例子里专门提醒了这一点。

**Q2：梯度累积和直接把 batch 调小有什么区别？**
直接调小 batch，一次 `optimizer.step()` 看到的样本变少，更新次数变多；梯度累积是把多个小 batch 的梯度攒起来，仍然只更新一次，有效批次不变。notebook 的等价测试要求两条路径使用同一组样本、同一顺序、同一损失口径和同一更新边界，缺一条就不能严格等价。

**Q3：为什么保存的输出里 `accum=2/4` 的峰值反而比 `accum=1` 高？**
因为这次跑的是序列 128。梯度缓冲区、参数和优化器状态一直驻留，激活节省不足以抵消累积带来的额外驻留和测量波动，所以峰值没有下降。5.4 节明确把结论限定为“受序列长度与状态占比约束的容量—时间交换”，并给出序列 256 时峰值下降、时间上升的对照。看到 `decision=tune` 不代表代码写错，而是这个负载下容量目标没达成。

**Q4：`update_delta_max` 和 `parameter_max_abs_diff` 都在看什么？**
`update_delta_max` 是比较更新前后某个探针参数（`self_attn.q_proj.weight` 的前 4096 个元素）的最大绝对变化，用来确认参数确实被更新了；`parameter_max_abs_diff` 是把各累积路径的探针参数与基线（`accum=1`）逐元素比较，确认不同累积次数得到同一次更新。前者要大于 0，后者要小于容差。

**Q5：`is_update_boundary` 为什么用取模，而不是用 `idx == accum_steps - 1`？**
取模对任意“已完成微批次数”都成立，既能判断一个窗口内的边界，也能在更长循环里复用；`idx == accum_steps - 1` 只适用于从 0 开始、刚好一个窗口的循环。`test_update_boundary` 直接把 1 到 8 都喂进去，期望每 4 个出现一次 `True`。

## 动手验证：怎么知道自己做对了
notebook 在题目区末尾直接调用四个测试，全部通过后打印 `✅ 微批次切分、梯度尺度和有效更新边界测试通过。`
- `test_micro_batch_slicing`：构造 `input_ids`、`attention_mask`、`labels` 三个 8 行张量，调用 `slice_micro_batch(batch, idx=1, accum_steps=4)`，断言每个字段都等于原张量的 `[2:4]`。
- `test_update_boundary`：对 `step=1..8` 调 `is_update_boundary(step, 4)`，断言结果严格等于 `[False, False, False, True, False, False, False, True]`。
- `test_accumulated_update_equivalence`：固定 `torch.manual_seed(42)`，输入 `torch.randn(8, 4)` 和 `torch.randn(8, 2)`，两个深拷贝的 `TinyRegressor` 都用 `SGD(lr=0.1)`；断言 `abs(loss_full - loss_accum) < 1e-6`、`accum_calls[0] == 1`、更新后输出与参数都在 `atol=1e-6` 内一致。
- `test_gradient_accumulation_input_contract`：`accum_steps` 取 0 和 3 必须抛 `ValueError`；`y[:-1]` 造成 batch 维不一致必须抛 `ValueError`；`is_update_boundary(0, 4)`、`(1, 0)` 必须抛 `ValueError`。
- 可选框架核对：若 `accelerate` 可用，`sync_pattern` 必须是 `[False, True]`，完整批次与累积路径参数在 `atol=1e-6, rtol=1e-6` 内一致；否则 `framework_check` 为 `skipped`。
- 可选 GPU 实验：默认 `RUN_MODE="dry_run"`，只打印提示、不训练；改成 `real_gpu` 后才加载模型并写 JSON。notebook 里保存的是序列 128、5.4 节标记为 `default12` 的结果，读法见上文第 9 步。
