# 55. 训练更新动力学 — 小白讲解

> 对应 notebook：[`55_Training_Update_Dynamics.ipynb`](./55_Training_Update_Dynamics.ipynb)

## 一句话总结
一次更新到底把参数移动了多远，不由学习率、梯度范数、裁剪或有效批次中的任何一项单独决定，而是"梯度信号 → 裁剪变换 → 学习率缩放 → 参数位移 → 诊断状态"这条链共同决定。本节用 `global_clip_scale`、`merge_token_weighted_gradients`、`summarize_update_state` 三个 TODO 把这条链写成可测的 CPU 函数，再用 `clip_grad_norm_` 和 SGD 核对数值，最后给出一个可选的 GPU 对照实验设计。核心结论是：梯度大、更新大、模型变好是三件事，必须分开看。

## 为什么需要它：痛点是什么
初学者遇到 loss 突然飙升，第一反应是"把学习率调小"；看到梯度范数很大，就以为参数一定移动了很多；发现梯度范数为 0，又去怪优化器。真实情况往往相反：梯度范数 20 但裁剪系数 0.02，参数实际只移动了原来的 2%；梯度很小但学习率很大，参数照样能走很远；更新过程完全有限、稳定，但验证集指标不涨，因为"更新稳定"和"能力提升"根本不是一回事。

生活类比：开车。梯度是路况给出的油门信号，裁剪是限速器，学习率是传动比，参数位移是这一秒实际走了多少米，任务评测是"有没有到目的地"。油门踩得深不代表车速高（限速器可能压住了），车速高不代表走对了路（可能方向错了）。诊断表要同时看这几块仪表，才能判断是"路况太猛但已限速"、"油门踩了但车没动"还是"仪表已经失灵"。

## 核心问题
- 全局范数裁剪系数 $c=\min(1, m/\lVert g\rVert_2)$ 是怎么来的？为什么零范数时取 1？
- 梯度累积时，为什么微批次梯度要按有效 Token 数加权，而不是简单平均？
- 相对更新比率 $\rho=\lVert\Delta\theta\rVert_2/(\lVert\theta\rVert_2+\varepsilon)$ 用来比较什么？不能说明什么？
- 诊断状态的优先级为什么是"非有限 > 无梯度 > 强裁剪 > 位移过小"？
- 裁剪系数 0.1、位移 $10^{-12}$ 这两个阈值为什么用严格不等号？
- GPU 对照实验的三个候选分别隔离了哪个变量？`finite_all` 为什么是第一道门？

## 解决思路：一步步看懂

**第 1 步：把更新写成四个量。** 设原始梯度向量为 $g$、全局裁剪阈值为 $m$，则

$$
c=\min\left(1,\frac{m}{\lVert g\rVert_2}\right),\qquad
\tilde g=cg,\qquad
\Delta\theta=-\eta\tilde g.
$$

逐项解释：$\lVert g\rVert_2$ 是所有参数梯度拼成一个大向量后的 L2 范数（`torch.linalg.vector_norm` 默认把所有元素展平求平方和再开方）；$c$ 被夹在 $[0,1]$，所以裁剪只会缩小、不会放大；$\tilde g$ 是裁剪后的有效梯度；$\eta$ 是当前成功更新步的学习率；$\Delta\theta$ 是最简 SGD 下的参数位移。把 $c$ 代进去可以看到：当 $\lVert g\rVert_2\le m$ 时 $c=1$，梯度原样通过；当 $\lVert g\rVert_2>m$ 时 $c=m/\lVert g\rVert_2$，裁剪后的范数恰好等于 $m$。

从几何上看，$c$ 是给整个梯度向量乘一个标量，方向不变、长度变成 $\min(\lVert g\rVert_2,m)$。这也是"全局范数裁剪"和"逐参数裁剪"的区别：逐参数裁剪每个张量各算各的范数、各自缩放到阈值，方向会被改变；全局裁剪只有一个标量，所有参数共享，方向保持。本节和 PyTorch 的 `clip_grad_norm_` 都是全局口径，所以才能用"一个系数乘原梯度"来核对。全局裁剪的代价是：如果某个参数组的梯度特别大，它会把整个向量的范数抬高，导致其他参数组被一起压小；逐参数裁剪不会这样，但会改变更新方向。两种口径没有绝对优劣，关键是不能混着比较。

手算一组（也是框架核对用的那组）：$g=[6,8]$，$\lVert g\rVert_2=\sqrt{36+64}=10$，$m=5$，于是 $c=0.5$，$\tilde g=[3,4]$；取 $\eta=0.1$，$\Delta\theta=[-0.3,-0.4]$，$\lVert\Delta\theta\rVert_2=0.5$。如果初始参数 $\theta=[3,4]$，$\lVert\theta\rVert_2=5$，相对更新比率 $\rho=0.5/(5+\varepsilon)\approx0.1$。

把 $\rho$ 写成公式是

$$
\rho=\frac{\lVert\Delta\theta\rVert_2}{\lVert\theta\rVert_2+\varepsilon},
$$

其中 $\lVert\Delta\theta\rVert_2$ 是本轮参数位移的长度，$\lVert\theta\rVert_2$ 是更新前参数向量的长度，$\varepsilon$ 是防止参数全零时除零的小正数。这个 $\rho$ 的意义是"这一步把参数整体挪动了原长度的约 10%"，可以跨参数组、跨 checkpoint 比较；但它不能说明泛化好坏，也不能说明某个任务上的能力变化。$\varepsilon$ 的取值不影响正常量级的结论。

**第 2 步：TODO 1 计算裁剪系数。** 参考答案：

```python
clip_scale = 1.0 if gradient_norm == 0.0 else min(1.0, max_norm / gradient_norm)
```

为什么零范数要单独处理？因为 $m/0$ 在浮点里会得到 `inf` 或 `nan`，而"梯度全零"的语义是"没有东西可裁"，按约定保持系数 1，让后续更新自然变成零位移。这一步不能写成 `min(1.0, max_norm / (gradient_norm + eps))` 来躲除零：那样零梯度会得到 $m/\varepsilon$，通常远大于 1，虽然也被夹成 1，但阈值很小时结果会依赖 $\varepsilon$ 的取值，语义不干净。

骨架已经把边界校验做完了：`max_norm` 必须有限且非负，`gradient_norm` 必须有限，否则抛 `ValueError`。范数用 `gradient.detach().float()` 计算，`detach` 保证不把范数计算接进反向图，`float()` 保证半精度输入也能稳定求和；整个过程不修改输入张量。测试专门验证了这一点：`test_clip_input_and_zero_threshold` 先 `before = gradient.clone()`，调用后 `torch.equal(gradient, before)` 必须为真。

**第 3 步：为什么微批次梯度要按 Token 数加权。** 设第 $i$ 个微批次有 $n_i$ 个有效监督 Token，它产生的是"在本微批次内按 Token 平均"的梯度 $g_i$。逻辑批次的总 Token 数是 $N=\sum_j n_j$，正确的合并梯度是

$$
g_{eff}=\sum_i\frac{n_i}{N}g_i.
$$

推导：$g_i=\frac{1}{n_i}\sum_{k\in i}\nabla\ell_k$，所以 $\frac{n_i}{N}g_i=\frac{1}{N}\sum_{k\in i}\nabla\ell_k$，对 $i$ 求和就得到"对整个逻辑批次的所有 Token 取平均"。如果偷懒写成 $\frac{1}{K}\sum_i g_i$（$K$ 是微批次个数），相当于让每个微批次权重相同，短微批次里的每个 Token 被放大了 $N/(K n_i)$ 倍，长微批次被缩小，合并结果就偏了。

TODO 2 的参考答案：

```python
total_targets = sum(valid_target_counts)
target_weights = [count / total_targets for count in valid_target_counts]
merged_gradient = sum(part * weight for part, weight in zip(gradient_parts, target_weights))
```

手算：`gradient_parts = [torch.tensor([2.0, 4.0]), torch.tensor([8.0, 10.0])]`，`valid_target_counts = [1, 3]`，总 Token 数 4，权重 `[0.25, 0.75]`，合并结果 $[0.25\times2+0.75\times8,\ 0.25\times4+0.75\times10]=[6.5, 8.5]$。`test_merge_matches_full_target_average` 把它和"四条目标直接平均"对比：四条梯度是 `[[2, 4], [8, 10], [8, 10], [8, 10]]`，均值也是 `[6.5, 8.5]`；同时断言"两个微批次简单平均"得到 `[5, 7]`，与正确答案不等——这条反向断言就是防止把权重写成 $1/K$。

"有效 Token 数"不是 `batch × seq_len`，而是真正参与 loss 的位置数。指令微调里 padding、被 mask 掉的 prompt、被截断的样本都不算；两个微批次形状可以完全一样，有效目标数却不同（比如一个批次里大部分是 padding）。如果按形状平均，短有效批次会被高估，长有效批次会被低估。notebook 用 `valid_target_counts` 显式传这个数，而不是从张量形状推，就是为了避免这个坑。反过来，如果所有微批次的有效 Token 数都相同，权重退化成 $1/K$，加权平均和简单平均等价——但这是特例，不能当默认。

还有一个容易忽略的不变量：权重必须归一化。测试把计数从 `[1, 3]` 同比放大到 `[2, 6]`，结果必须完全一样；如果忘了除以 `total_targets`，同比放大会让合并梯度整体放大 2 倍。骨架还检查了 `gradient_parts` 非空、与计数一一对应、计数为正、所有梯度同形，避免"形状不同硬 broadcast"这种静默错误。

**第 4 步：TODO 3 把三个布尔条件翻译成诊断状态。** 骨架给定两个阈值：`strong_clip_threshold = 0.1`、`minimum_parameter_motion = 1e-12`。参考答案：

```python
has_gradient = gradient_norm > 0.0
strong_clipping = clip_scale < strong_clip_threshold
has_parameter_motion = parameter_delta_norm > minimum_parameter_motion
```

三个条件分别回答三个不同问题：`has_gradient` 是"有没有训练信号"；`strong_clipping` 是"信号被压掉了多少"；`has_parameter_motion` 是"参数有没有真的写回变化"。它们不是同一件事的三种说法：梯度可以很大但被裁到很小；梯度可以很小但学习率很大；梯度有限、裁剪系数为 1，参数也可能因为学习率太小而几乎不动。

优先级由骨架的 `if` 顺序固定：
1. `not finite` 直接返回 `blocked_nonfinite`，连后续的有限性校验都不做。测试里 `finite=False, gradient_norm=float('nan')` 仍然返回 `blocked_nonfinite`，而不是抛 `ValueError`——因为"数值已经坏了"比"指标是否在合法范围"更优先，先止损再排查。
2. `not has_gradient` 返回 `no_gradient`。测试把 `gradient_norm=0.0` 和 `clip_scale=0.02` 放在一起，结果仍是 `no_gradient`：没有信号时，"裁剪很狠"没有意义，先查监督掩码、冻结参数和计算图。
3. `strong_clipping` 返回 `heavily_clipped`。`clip_scale=0.02` 表示梯度被压到原来的 2%，先查学习率、裁剪阈值和梯度来源。
4. `not has_parameter_motion` 返回 `low_parameter_motion`。位移小于等于 $10^{-12}$ 时，查学习率、冻结参数和优化器状态。
5. 都不触发才返回 `observe_with_evaluation`，提示去比较任务评测。

这五个状态和 Step 3 的排查表一一对应，可以放在一起看：

| 观察组合 | 先检查什么 | 局部结论 | 对应状态 |
|---|---|---|---|
| 非有限 loss、梯度或参数 | dtype、缩放、输入、最近一次更新 | 本次更新不可信，停止候选比较 | `blocked_nonfinite` |
| 梯度范数为零 | 监督掩码、冻结参数、计算图 | 当前参数组没有收到训练信号 | `no_gradient` |
| 梯度范数高、裁剪系数远小于 1 | 裁剪阈值、学习率 | 梯度被显著压缩 | `heavily_clipped` |
| 梯度有限、参数位移接近零 | 学习率、优化器状态、冻结参数 | 写回幅度可能不足 | `low_parameter_motion` |
| 更新记录稳定但评测未提升 | 独立评测、数据分布 | 更新稳定不等于能力提升 | `observe_with_evaluation` |

这张表的价值在于"先分诊、再深挖"。同样是"loss 不降"，`no_gradient` 要去查标签和 `requires_grad`，`heavily_clipped` 要去查阈值和梯度来源，`low_parameter_motion` 要去查学习率和优化器状态；如果一上来就统一调学习率，可能把真正的问题掩盖掉。

阈值的等号方向是测试重点：`clip_scale == 0.1` 时 `0.1 < 0.1` 为假，不算强裁剪，返回 `observe_with_evaluation`；`parameter_delta_norm == 1e-12` 时 `1e-12 > 1e-12` 为假，返回 `low_parameter_motion`，而 `1.1e-12` 就返回 `observe_with_evaluation`。这些边界不是随便挑的：0.1 表示"梯度至少被压了 10 倍"，$10^{-12}$ 是接近浮点噪声的量级，表示"基本没动"。

**第 5 步：框架一致性核对。** `check_update_dynamics_framework_conformance` 用同一组输入比较三件事：
- 原梯度范数：本节算出的 `10.0` 与 `clip_grad_norm_` 的返回值一致（它返回裁剪前的总范数）；
- 裁剪后梯度：`clip_grad_norm_([parameter], max_norm=5.0)` 写回的 `parameter.grad` 必须等于 `raw_gradient * local_scale`，即 `[3, 4]`；
- 参数位移：`torch.optim.SGD([parameter], lr=0.1)` 走一步后，`parameter - initial` 必须等于 `-0.1 * clipped_gradient`，即 `[-0.3, -0.4]`。

三项对应 `gradient_norm_match`、`clip_scale_match`、`parameter_delta_match`，全为 `True` 才返回 `status = 'passed'`。它只覆盖 SGD 无动量、单步、CPU 的最简路径，不覆盖 Adam、混合精度或梯度累积。

**第 6 步：GPU 实验怎么设计、怎么读。** Step 5 是可选的真实模型对照，工作负载是 `Qwen/Qwen2.5-0.5B`、batch 2、序列长度 128、FP32、预热 1 次、测量 8 次。三个候选只改学习率和裁剪阈值：
- `reference`：`learning_rate = 1e-4`、`max_grad_norm = 1.0`，作为基准；
- `high_lr_no_clip`：`learning_rate = 2e-3`、`max_grad_norm = None`，隔离"学习率放大 20 倍且不裁剪"；
- `high_lr_clipped`：`learning_rate = 2e-3`、`max_grad_norm = 1.0`，和上一个只差裁剪，用来隔离裁剪的影响。

每个候选都从同一不可变 checkpoint 重新加载；每次测量更新记录 loss、原始梯度范数、实际裁剪系数、参数探针增量、有限值标志、单步中位/P95 时间和峰值已分配显存。参数探针取第一个 `requires_grad` 的参数的前 4096 个元素，算更新前后的 L2 距离，所以 `parameter_delta_norm_median` 是"某个参数切片移动了多少"，不是全模型位移。

`one_update` 的顺序值得逐行看：先 `optimizer.zero_grad(set_to_none=True)` 清掉上一步的梯度；再前向算 loss、反向；在 `optimizer.step()` 之前用 `gradient_l2_norm(model.parameters())` 取原始范数，并手动算 `clip_scale`；如果配了阈值，再调用 `clip_grad_norm_` 真正写回梯度；最后 `optimizer.step()`。为什么手动算一遍系数？因为 `clip_grad_norm_` 只返回裁剪前的范数，不返回系数；要记录"这一步被压了多少"，只能自己用 `min(1, m/raw_norm)` 算。`finite` 同时检查 loss 和原始范数，任何一个非有限就标记；代码没有在这里提前返回，仍然执行 `step()`，所以非有限更新会写进参数，便于在参数上看到痕迹，但也意味着后续步骤可能被污染，读 `finite_all` 时要先止损。

`GPU_CONFIG['candidates']` 的顺序就是最终 `metrics` 列表的顺序，读 JSON 时不会错位。每个候选都重新 `load_causal_lm`，保证初始权重一致；`torch.cuda.empty_cache()`、`reset_peak_memory_stats()` 和 `synchronize()` 在每步测量前执行，避免上一步的缓存和异步提交污染峰值与计时。参数探针只取第一个参数的前 4096 个元素，是为了低成本地观察"有没有写回"，不是全模型位移；`peak_allocated_mb` 取 8 次里的最大值，`median_step_ms`/`p95_step_ms` 从 8 个样本算，`gradient_norm_median`、`clip_scale_median`、`parameter_delta_norm_median` 用中位数而不是均值，避免单步异常把结论拉偏。

读结果的顺序在 5.4 写得很明确：先确认三条路径使用同一模型 commit、输入、更新次数和 dtype；再看 `finite_all`，出现非有限值的候选只能进入故障定位，不能和其他路径比较；然后看 `clip_scale_median` 是否接近 1（裁剪是否频繁触发），比较 `parameter_delta_norm_median` 和 `median_step_ms`、`peak_allocated_mb`。`decision` 固定为 `observe_with_independent_evaluation`：局部更新稳定只说明"这组条件下可以继续观察"，不说明模型在独立任务上更好。默认 `RUN_MODE = 'dry_run'`，只核对配置并打印提示，不加载模型；notebook 没有保存 GPU 结果，所以本页不引用任何实测数字。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 全局范数裁剪 | 把所有梯度拼成一个大向量，按总长度决定缩放系数 |
| 裁剪系数 $c$ | 夹在 $[0,1]$ 的缩放比例，只在超阈值时小于 1 |
| 有效批次 | 一次参数更新实际用到的所有 Token，可能由多个微批次拼成 |
| 有效目标数 $n_i$ | 微批次里真正参与 loss 的 Token 数，padding 不算 |
| 合并梯度 $g_{eff}$ | 按有效 Token 数加权后的逻辑批次平均梯度 |
| 参数位移 $\Delta\theta$ | 一次更新把参数改了多少 |
| 相对更新比率 $\rho$ | 位移长度除以参数长度，用来跨规模比较 |
| 强裁剪 | `clip_scale < 0.1`，梯度被压到原来的不到一成 |
| 最小参数位移 | `parameter_delta_norm > 1e-12` 才算"动过" |
| 诊断优先级 | 非有限 > 无梯度 > 强裁剪 > 位移过小 > 继续观察 |
| `finite_all` | 所有测量步的 loss 和梯度范数是否都有限 |
| `observe_with_independent_evaluation` | 固定决策：局部稳定，能力提升另测 |

## 小白常见疑问

**Q1：`clip_grad_norm_` 的返回值是裁剪前还是裁剪后的范数？**
是裁剪前的总范数。框架核对里 `framework_norm` 与 `torch.linalg.vector_norm(raw_gradient)` 比较，两者都是 10.0；裁剪后的结果体现在 `parameter.grad` 被写成了 `[3, 4]`。如果误以为返回值是裁剪后的 5.0，`gradient_norm_match` 就会失败。

**Q2：梯度范数为 0 时，裁剪系数为什么不是 0？**
系数 0 的含义是"把梯度乘成 0"，但零梯度本来就是 0，不需要裁。约定取 1 是为了让公式在零范数时仍然有定义，并且不引入额外的除零分支。测试 `global_clip_scale(torch.tensor([0.0, 0.0]), 1.0) == 1.0` 固定了这个约定。另外，`gradient_norm == 0.0` 用的是严格相等；浮点下范数极小但不为 0 时，会走 `min(1, m/norm)`，只要 $m\ge$ 范数，结果仍然是 1，不会被误伤。零范数只表示"这一步所有梯度都是 0"，可能是标签全被 mask、参数全冻结，也可能是上游被 `detach` 了，要结合 `no_gradient` 的排查项一起看。

**Q3：为什么 `clip_scale == 0.1` 不算强裁剪，`parameter_delta_norm == 1e-12` 算位移过小？**
因为两个阈值的方向不同。强裁剪用 `clip_scale < 0.1`，等号表示"正好压到 10%"，仍算可接受；位移用 `parameter_delta_norm > 1e-12`，等号表示"刚好等于噪声地板"，仍算没动。`test_diagnostic_thresholds_and_priority` 把这两个等号都写成了断言，改比较符就会失败。

**Q4：有效批次变大，学习率要不要跟着调？**
本节只回答"梯度怎么合并"，不回答"学习率怎么缩放"。Step 2 的表格明确写了：固定 Token 预算改变有效批次，会改变更新次数和调度器步数；固定学习率改变有效批次，会改变梯度估计噪声和参数轨迹。这些量不能混在一起比较，学习率与批次的缩放规则属于优化器/调度那一节。一句话：有效批次改变的是"一次更新看多少数据"，不是"每一步走多远"。

**Q5：GPU 实验里 loss 从 `loss_start` 到 `loss_end` 下降了，能说明模型变好了吗？**
不能。8 次测量更新都在同一批数据上，`loss_start`/`loss_end` 只能说明"这批数据上的 loss 在动"，不能说明泛化。5.4 明确要求把候选带入独立评测项目，比较验证集、保留集与可恢复性；`decision` 也因此固定为 `observe_with_independent_evaluation`。

**Q6：框架核对为什么用 SGD，而不是 Adam？**
因为 SGD 的单步更新是闭式的 $\Delta\theta=-\eta\tilde g$，可以逐元素和 `parameter - initial` 比较；Adam 还要维护一阶矩、二阶矩、偏差校正和 $\varepsilon$，单步位移不是简单乘学习率，没法用三行断言核对。Step 1 也写明"以下用最简单的 SGD 说明这些量的关系"，目的就是把"裁剪 + 学习率"这条链单独拎出来，不被优化器状态干扰。

**Q7：`high_lr_no_clip` 的 `max_grad_norm` 为什么是 `None`，而不是 0 或很大的数？**
`None` 在 `one_update` 里走的是"跳过裁剪"分支，`clip_scale` 直接记 1.0；如果填 0，按 `min(1, 0/norm)` 的公式会把非零梯度整个乘成 0，参数根本不动，测的就不是"高学习率"而是"零更新"；如果填一个很大的数，虽然实际上不裁剪，但仍会调用 `clip_grad_norm_`，多了一次全参数范数计算和写回，和"不裁剪"路径不完全等价。`None` 是最干净的隔离。记录里 `max_grad_norm: None` 也会原样写进 JSON，读的时候一眼能看出这条路径没开裁剪。

## 动手验证：怎么知道自己做对了
notebook 先调三个测试，再调另外三个，六个全过才打印 `✅ 6 项更新动力学机制测试通过`；TODO 没填会抛 `NotImplementedError`。

- `test_global_clip_scale`：零梯度配 `max_norm=1.0` 得 `1.0`；`[3, 4]`（范数 5）配 `max_norm=5.0` 得 `1.0`；配 `max_norm=2.5` 得 `0.5`（`math.isclose`）；`max_norm=-1.0` 和梯度含 `inf` 都必须抛 `ValueError`。
- `test_token_weighted_gradient_merge`：`merge_token_weighted_gradients([tensor([2.,4.]), tensor([8.,10.])], [1, 3])` 必须 `allclose([6.5, 8.5])`；形状不同的两个梯度必须抛 `ValueError`。
- `test_update_state_priority`：`finite=False` 得 `blocked_nonfinite`；`gradient_norm=0.0` 得 `no_gradient`；`gradient_norm=20.0, clip_scale=0.02` 得 `heavily_clipped`；`parameter_delta_norm=0.0` 得 `low_parameter_motion`；全部正常得 `observe_with_evaluation`；`clip_scale=1.5` 抛 `ValueError`。
- `test_clip_input_and_zero_threshold`：`max_norm=0.0` 时 `[3, 4]` 的系数是 `0.0`，且输入张量在调用前后完全相等。
- `test_merge_matches_full_target_average`：`counts` 为 `[1, 3]` 和 `[2, 6]` 时，合并结果都等于四条目标的直接平均；简单平均 `torch.stack(parts).mean(0)` 必须不等于正确答案。
- `test_diagnostic_thresholds_and_priority`：零梯度加 `clip_scale=0.02` 仍是 `no_gradient`；`clip_scale=0.1` 是 `observe_with_evaluation`；位移 `1e-12` 是 `low_parameter_motion`，`1.1e-12` 是 `observe_with_evaluation`；`finite=False` 且范数为 `nan` 仍是 `blocked_nonfinite`。

测试还有两个容易忽略的覆盖面。`test_token_weighted_gradient_merge` 只显式检查了"形状不同"这一种非法输入，函数里还有"计数必须为正"和"列表非空/等长"两条校验，没有单独的测试用例；如果想让这两条也有证据，可以自己加 `merge_token_weighted_gradients([torch.tensor([1.0])], [0])` 和空列表调用，都会抛 `ValueError`。`test_global_clip_scale` 覆盖了零范数、阈值内、阈值外、负阈值和非有限梯度，但没有覆盖 `max_norm=0` 且梯度非零的情况——这一条在 `test_clip_input_and_zero_threshold` 里补上了，结果是 `0.0`，并且原张量不变。

框架核对 `check_update_dynamics_framework_conformance` 的 `checks` 字典要求 `gradient_norm_match`、`clip_scale_match`、`parameter_delta_match` 全为 `True`，返回 `status = 'passed'` 和 `torch` 版本。三项都用 `torch.allclose` 或 `math.isclose` 的默认容差，没有额外放宽；如果本地 PyTorch 版本的裁剪实现有差异，先看 `gradient_norm_match` 是否通过，再看 `clip_scale_match`，就能定位是范数口径还是写回口径的问题。Step 5 默认 `dry_run`，只打印"已核对固定工作负载"；改成 `real_gpu` 后需要 `TRAIN-CUDA-MECHANISM` 环境、CUDA GPU 和唯一 `RUN_LABEL`，结果写到 `benchmarks/results/train_update_dynamics/<semantic_id>_<run_label>.json`，5.3 只读该文件、不重新训练。
