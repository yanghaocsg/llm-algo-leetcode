# TRAIN-MIXED-PRECISION-BACKWARD. 混合精度反向传播与梯度安全 — 小白讲解

> 对应 notebook：[`TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb`](./TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb)

## 一句话总结
把前向换成 FP16 后，小梯度可能下溢成 0、大数值可能上溢成 Inf。本节用 `unscale_gradients`、`global_grad_norm`、`plan_mixed_precision_update` 三个函数，把"损失缩放 → 反向 → 还原尺度 → 有限值检查 → 全局裁剪 → 决定是否更新"这条链拆成可测试的数值步骤，再用 PyTorch `GradScaler` 的 CPU 路径核对语义，最后给出可选的 GPU 实验读取 step 时间与峰值显存。

## 为什么需要它：痛点是什么
初学者第一次用 `autocast` 时常犯三类错：以为"开了混合精度，参数就都变成 FP16 了"；以为"`GradScaler` 会自动帮你把梯度除回去"；看到 loss 变成 `NaN` 还继续 `optimizer.step()`。更隐蔽的问题是梯度累积：如果每个微批次都 unscale 一次、裁剪一次、更新一次，等效批次和缩放语义都会乱。真正需要记住的是一条顺序纪律：缩放发生在反向之前，还原和检查发生在更新边界，异常时跳过更新并回退缩放因子。

生活类比：用小杯子量很细的粉末。先把整批粉末放大 1024 倍再称（loss scaling），称完除以 1024 还原（unscale）；如果称的时候粉末撒了、读数变成"无穷大"（Inf/NaN），这一杯就只能倒掉，不能假装除以 2 就还能用（裁剪救不了 Inf），同时把下一杯的放大倍数调小（backoff）。

## 核心问题
- 损失乘了 $s$，梯度会变成多少？为什么更新前必须除以同一个 $s$？
- 为什么有限值检查要在裁剪之前？Inf 能不能靠裁剪救回来？
- 梯度裁剪的"全局范数"怎么算？为什么不是每个张量各自裁剪？
- 梯度累积时，unscale、裁剪、step 应该放在微批次里还是更新边界？
- FP16 和 BF16 的保护手段有什么不同？为什么 BF16 通常不用 `GradScaler` 但仍要检查有限值？

## 解决思路：一步步看懂

**Step 1：缩放与还原的数学。** 设原损失为 $L$、缩放因子为 $s>0$、参数为 $\theta$。前向得到 $L$ 后先算 $L'=sL$，反向传播的链式法则给出

$$g'=\nabla_\theta L'=s\nabla_\theta L,\qquad g=\frac{g'}{s}.$$

$g'$ 是梯度缓冲里实际存的值，被放大了 $s$ 倍，所以原本会下溢成 0 的小梯度有机会保留下来；更新前必须除以同一个 $s$ 得到真实梯度 $g$。`unscale_gradients` 做的就是第二式：`gradient / scale`。

**Step 2：TODO 1——保留 None，返回新张量。** 参考答案只有一行列表推导：

```python
return [None if gradient is None else gradient / scale for gradient in scaled_gradients]
```

三个细节都不能少：`scale <= 0` 要抛 `ValueError`；`None` 表示这个参数本轮没有梯度，必须原样保留，不能变成 0；除法会产生新张量，不能原地改输入。`test_unscale_without_mutation` 用 `[6.0, 8.0]` 和 `scale=2.0` 检查得到 `[3.0, 4.0]`、第二项仍是 `None`、原张量仍是 `[6.0, 8.0]`。

**Step 3：TODO 2——跨参数的全局 L2 范数。** 梯度裁剪把所有参数的梯度拼成一个大向量，再算它的二范数：

$$\|g\|_2=\sqrt{\sum_{p}\sum_{i} g_{p,i}^2}.$$

参考答案先转 FP32 再平方求和，最后开方：

```python
squared_sum = squared_sum + gradient.float().square().sum()
...
return squared_sum.sqrt()
```

`torch.zeros((), dtype=torch.float32)` 是标量累加器，保证低精度梯度不会在累加时丢精度。`test_global_norm_across_parameters` 用 `[3,4]` 和 `[12]` 两组梯度：$3^2+4^2+12^2=9+16+144=169$，开方恰好是 13。如果逐张量裁剪，会得到两个不同的系数，方向和相对大小都被破坏，所以必须全局算一次。

**Step 4：TODO 3——先判有限，再算裁剪系数。** 参考答案：

```python
finite = all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)
```

`None` 视为"没有梯度"，不算异常；只要有一个元素是 NaN 或 Inf，整体就 `finite=False`。异常分支返回 `should_step=False`、`grad_norm=inf`、`clip_coefficient=0.0`、`next_scale=scale*backoff_factor`，并把 `reason` 标成 `'non_finite_gradient'`。有限分支算

$$\text{coefficient}=\min\left(1,\frac{\text{max\_norm}}{\|g\|_2+10^{-12}}\right),$$

再让每个梯度乘这个系数。分母加 $10^{-12}$ 是防止范数为 0 时除零；`min(1, ...)` 保证范数已经小于阈值时不放大梯度，只裁剪不放大。`test_finite_update_and_clipping` 手算：缩放梯度 `[6,8]` 除以 2 得 `[3,4]`，范数 5；`max_norm=2.5` 时系数 $2.5/5=0.5$，裁剪后 `[1.5,2.0]`。`test_overflow_skips_update` 用 `[inf]`、`scale=1024`、`max_norm=1`，要求 `should_step=False`、`next_scale=512.0`、`reason='non_finite_gradient'`。

**Step 5：为什么顺序不能换。** 如果先裁剪再检查，`inf * 0.5` 还是 `inf`，`nan` 也还是 `nan`，看起来"处理过了"实际上更新仍会污染参数和优化器状态。所以顺序必须是 unscale → finite check → global norm → clip → step/skip。notebook 的 `plan_mixed_precision_update` 在 `max_norm <= 0` 时也直接抛 `ValueError`，把非法配置挡在计算之前。

**Step 6：梯度累积时，这些操作放在哪。** 设一次有效更新由 $K$ 个微批次组成，第 $i$ 个微批次权重为 $w_i$，则

$$g'_{acc}=s\sum_{i=1}^{K}w_i g_i,\qquad g_{acc}=\frac{g'_{acc}}{s}=\sum_{i=1}^{K}w_i g_i,\qquad \sum_{i=1}^{K}w_i=1.$$

每个微批次只做"缩放损失 + 反向 + 累积"，到第 $K$ 次反向结束才统一 unscale、检查、裁剪、step。权重不是无脑 $1/K$：notebook 举例，两批分别有 2 和 6 个有效目标，权重应是 $1/4$ 和 $3/4$，因为总目标数是 8；只有有效目标数相同时才各取 $1/K$。这样做是为了让累积后的损失口径与"把所有目标拼成一个大 batch 求平均"一致。

**Step 7：FP16、BF16、FP32 的分工。** `autocast` 按算子选择低精度或 FP32 计算，参数本身通常仍保留 FP32；这和"把整个模型 `.half()`"是两回事。FP16 指数范围窄，需要动态 loss scaling 和有限值检查；BF16 指数范围接近 FP32，通常不需要 `GradScaler`，但尾数只有 7 位，仍可能出 NaN/Inf，所以有限值检查和梯度范数监控不能省；FP32 是精度基线，用于敏感算子或主状态。

**Step 8：框架一致性核对。** `check_amp_framework_conformance` 在 CPU 上用 `torch.amp.GradScaler('cpu', init_scale=8.0, backoff_factor=0.5)` 对照两件事：`unscale_gradients` 的结果是否与 `scaler.unscale_(optimizer)` 一致；梯度填成 `inf` 后，`scaler.step` 是否真的不执行优化器、`scaler.update()` 是否把 scale 降到与本地计划相同。它还注册 `register_step_post_hook` 记录 SGD 是否实际执行，并用 `lr=0.0` 的零学习率用例确认"step 执行了但参数没变"也能被钩子观察到。五个检查键是 `unscaled_gradient_match`、`local_overflow_skips`、`framework_overflow_skips`、`scale_backoff_match`、`zero_lr_step_executed`，任一失败就抛 `AssertionError`。

**Step 9：可选 GPU 实验怎么读。** 固定模型 `Qwen/Qwen2.5-0.5B`、batch 1、序列 128、学习率 `1e-4`、初始 scale `65536.0`、warmup 1、repeats 20、seed 2026，优化器是普通 SGD。`run_amp_path` 分别跑 FP32、AMP 和"人工把 loss 乘 inf"的 overflow 路径；正常路径只统计成功且真正执行的 step，回退尝试不计入性能均值；`parameter_delta_max` 用 `q_proj.weight` 的探针差值独立确认参数确实变了。notebook 保存的读取输出显示：FP32 中位 96.50 ms、P95 152.78 ms、峰值 4841.4 MB；AMP 中位 94.21 ms、P95 157.10 ms、峰值 4321.7 MB、`backoff=6`；overflow 路径 `step=False`，scale 从 65536.0 降到 32768.0，最终 `decision=accept`。5.4 的文字记录还提到历史标签 `default14` 在 20 次成功测量中峰值约 4841 → 4322 MiB，降幅约 10.7%（$(4841.4-4321.7)/4841.4\approx10.7\%$），但中位时间只从 96.50 降到 94.21 ms，且 P95 反而略高，所以这组数字主要支持"显存有收益"，不能单独证明"更快"；notebook 也要求后续用交替配对复测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| autocast | 按算子自动选择低精度或 FP32 计算，参数通常仍是 FP32 |
| loss scaling | 把损失乘正数，放大反向梯度，减少 FP16 下溢 |
| unscale | 更新前把梯度除以同一个缩放因子，还原真实尺度 |
| 下溢 / 上溢 | 太小变成 0 / 太大变成 Inf |
| 有限值检查 | 检查梯度里是否有 NaN 或 Inf，有就跳过更新 |
| 全局梯度范数 | 把所有参数梯度拼成一个向量后的 L2 长度 |
| 裁剪系数 | $\min(1,\text{max\_norm}/\|\|g\|\|)$，只缩小不放大 |
| 更新边界 | 一次有效更新里最后一个微批次结束的位置 |
| GradScaler | PyTorch 的动态损失缩放器，负责 scale、unscale、跳过和回退 |
| backoff factor | 溢出时缩放因子的缩小比例，默认 0.5 |
| 主权重 | 保留高精度副本用于参数更新，避免小更新被舍入掉 |
| BF16 | 指数范围接近 FP32、尾数较短的 16 位格式 |

## 小白常见疑问

**Q1：既然裁剪能把大梯度压小，为什么 Inf 还要单独跳过？**
因为裁剪是乘法：`inf * 0.5` 仍是 `inf`，`nan * 0.5` 仍是 `nan`。`plan_mixed_precision_update` 先做 `finite` 判断，异常直接返回 `should_step=False`，根本不会进入裁剪分支。`test_overflow_skips_update` 就是固定这个行为的。

**Q2：`global_grad_norm` 为什么要先 `.float()` 再平方？**
如果梯度本身是 FP16，直接 `square().sum()` 可能在累加时溢出或丢精度。notebook 用 `torch.zeros((), dtype=torch.float32)` 作为累加器，并在每个张量上调用 `gradient.float().square().sum()`，把求和放在 FP32 里完成。

**Q3：为什么 `unscale_gradients` 不直接原地修改 `scaled_gradients`？**
因为测试要求输入不变，而且原地修改会让"缩放梯度"这个中间状态丢失，后续排查困难。参考答案用列表推导生成新张量，`test_unscale_without_mutation` 明确断言 `source` 仍是 `[6.0, 8.0]`。

**Q4：BF16 不用 `GradScaler`，是不是就不用管梯度异常了？**
不是。BF16 的指数范围和 FP32 接近，较少因范围不足而下溢，所以通常不缩放；但它仍可能产生 NaN/Inf，notebook 要求"有限值检查与梯度范数监控依然有意义"。GPU 实验里 BF16/FP32 在调用优化器前也会显式检查梯度和范数。

**Q5：为什么梯度累积的权重不是简单的 `1/K`？**
因为每个微批次的有效目标数可能不同。notebook 的例子是两批 2 和 6 个目标，权重应为 $2/8=1/4$ 和 $6/8=3/4$，这样累积损失才等价于把所有目标放在一起求平均；只有有效目标数相同才退化成 `1/K`。

## 动手验证：怎么知道自己做对了
测试单元按顺序调用四个函数，全过才打印 `All mixed-precision backward tests passed.`
- `test_unscale_without_mutation`：`[6,8]` 除以 2 得 `[3,4]`；`None` 保留；原输入不变。
- `test_global_norm_across_parameters`：`[3,4]` 与 `[12]` 的全局范数为 13。
- `test_finite_update_and_clipping`：`should_step=True`，范数 5，系数 0.5（误差 `<1e-6`），裁剪后 `[1.5,2.0]`。
- `test_overflow_skips_update`：`[inf]`、scale 1024 时 `should_step=False`、`next_scale=512.0`、`reason='non_finite_gradient'`。
- `check_amp_framework_conformance`：五个检查键全部为真才返回 `status='passed'`，否则抛 `AssertionError` 并列出失败项。

notebook 保存了 GPU 读取单元的输出（FP32/AMP/overflow 三行），但那是某个历史 `RUN_LABEL` 的记录，复现时必须以你自己新标签生成的 JSON 为准；没有 GPU 时保持 `RUN_MODE='dry_run'`，只核对配置，不会产生训练结果。
