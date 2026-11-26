# PERF-CANDIDATE. 从资源约束到候选改动与新增代价 — 小白讲解

> 对应 notebook：[`PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb`](./PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb)

## 一句话总结
本节用 `normalize_effects`、`primary_reduction`、`decide_candidate` 三个函数，把"观察到某个资源受限 → 提出候选改动 → 分别记账节省量与新增代价 → 按目标、质量和预算筛选"这套流程写成可测试的代码。核心结论是：一项改动可以在某个轴上省钱、同时在另一个轴上更贵，不能用"综合分"把不同单位的收益和代价相互抵消。

## 为什么需要它：痛点是什么
"权重从 FP32 换成 BF16，显存减半，所以一定更快、峰值一定更低"——这是性能优化里最常见的错觉。实际上：权重字节少了，反量化可能多出临时缓冲区；计算精度降了，可能触发额外的类型转换 kernel；某个阶段变快，可能只是把等待挪到了别处。

初学者还会把不同单位的数字直接相加：省了 32 MiB 显存、多花 3 ms、误差增加 0.001，然后拍一个"综合得分 8.5 分"决定用不用。这种加权分会掩盖"某一项已经严重超限"的事实。

生活类比：搬家时换更小的箱子。箱子本身便宜了 30 元（存储减少），但每件东西都要多包一层泡沫（新增临时开销），而且易碎品可能被压坏（质量损失）。正确的做法是分三栏记账：省了多少钱、多花多少时间、坏了多少东西，再分别和预算比较，而不是算一个"总幸福指数"。

## 核心问题
- 一个候选到底改变了哪条资源轴？计算、搬运、峰值、关键路径为什么要分开？
- 变化量 $\Delta x$ 的正负号怎样定义？为什么"减少量"要取目标轴变化量的相反数？
- 为什么新增开销只统计正的变化量，资源减少不能拿去抵扣别的轴超支？
- `reject`、`tune`、`measure` 三种状态的分界线在哪？为什么 `measure` 不等于"采用"？
- 为什么峰值必须测"同一时刻同时存活对象的总量"，不能用"模型字节 + 输入缓冲字节"代替？
- GPU 实验里 BF16 参数减半、速度 2.65×，为什么不能直接说"峰值也降了一半"？

## 解决思路：一步步看懂

**第 1 步：先把"受限对象"和"候选改动"对上。** Step 1 的表把资源分成四类：计算（运算次数、形状、精度、执行路径）、搬运（实际字节、复用、中间结果读写）、峰值容量（同一时刻参数、激活、临时缓冲区的总和）、关键路径（串行依赖、同步、未隐藏等待）。同一个候选可能只改善其中一类：例如换更高效的 kernel 可能 FLOPs 不变，但实际执行时间变短；这只属于"执行路径"证据，不能写成"计算量减少"。

**第 2 步：五个资源轴与变化量公式。** 代码里固定了：

```python
RESOURCE_AXES = ("compute_flops", "memory_bytes", "peak_bytes", "exposed_ms", "quality_loss")
CONSTRAINT_AXIS = {"compute": "compute_flops", "bandwidth": "memory_bytes",
                   "capacity": "peak_bytes", "critical_path": "exposed_ms"}
```

对某个可比较资源量 $x$，设基线为 $x_b$、候选为 $x_c$：

$$
\Delta x=x_c-x_b,\qquad R=-\Delta x_{\mathrm{target}},\qquad A_x=\max(0,\Delta x)
$$

$\Delta x$ 为负表示候选比基线少，为正表示多。$R$ 是目标轴的"减少量"，所以取相反数：目标轴减少 10 字节，$R=10$；目标轴反而增加 3 字节，$R=-3$。$A_x$ 只保留正的变化量，因为"这项资源减少了"不应该被记成"新增开销为负"，更不能拿去抵消另一项超支。

**第 3 步：TODO 1——检查变化轴。** 参考实现：

```python
unknown_axes = set(effects) - set(RESOURCE_AXES)
missing_axes = set(RESOURCE_AXES) - set(effects)
if unknown_axes or missing_axes:
    raise ValueError(...)
if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in effects.values()):
    raise TypeError("变化量必须为数值，不能使用布尔值")
normalized = {axis: float(effects[axis]) for axis in RESOURCE_AXES}
if any(not math.isfinite(value) for value in normalized.values()):
    raise ValueError("变化量必须有限")
```

五个轴必须完整：少一个轴，就等于默认"这项开销为 0"，而未知开销不能用 0 假装。多一个 `score` 这类综合字段也要拒绝，因为它没有单位、无法和预算比较。`True`/`False` 在 Python 里是 `int` 的子类，如果不显式排除，`True` 会被当成 1，测试 `test_effect_numeric_validation` 专门用 `True` 触发这个坑。`NaN` 和 `inf` 也必须拒绝，否则后续比较会静默失真。

**第 4 步：TODO 2——读取目标减少量。** 参考实现：

```python
target_axis = CONSTRAINT_AXIS[constraint]
reduction = -normalize_effects(effects)[target_axis]
```

目标轴由约束名显式映射，不能根据候选名称或调用者口头描述猜。`test_each_constraint_axis` 对四类约束逐一检查：把目标轴设成 `-12` 必须返回 `12`，设成 `0` 返回 `0`，设成 `3` 返回 `-3`；未知约束名抛 `ValueError`。

**第 5 步：TODO 3——逐项判断预算。** 参考实现：

```python
quality_passed = normalized["quality_loss"] <= budgets["max_quality_loss"]
added_costs_passed = (
    max(0.0, normalized["compute_flops"]) <= budgets["max_added_compute"]
    and max(0.0, normalized["memory_bytes"]) <= budgets["max_added_memory"]
    and max(0.0, normalized["peak_bytes"]) <= budgets["max_added_peak"]
    and max(0.0, normalized["exposed_ms"]) <= budgets["max_added_exposed_ms"]
)
primary_missed = reduction <= 0
```

`budgets` 必须恰好包含六个键：`min_primary_reduction`、`max_quality_loss`、`max_added_compute`、`max_added_memory`、`max_added_peak`、`max_added_exposed_ms`，缺项或多出未知项都拒绝，且每个值必须是非负有限数。质量轴单独比较，因为 `quality_loss` 也是"越大越差"；新增开销对四个轴分别检查，所有轴都满足才叫 `added_costs_passed`。

状态机只有三档：

- `reject`：`primary_missed`（目标没有减少）或质量超限。质量失败具有优先级，其他轴再漂亮也不能救。
- `tune`：目标方向对，但减少幅度小于 `min_primary_reduction`，或任一新增开销超预算。
- `measure`：目标减少足够、质量通过、四个新增轴都在预算内。它只表示"值得进入真实对照测量"，不是"已经采用"。

**第 6 步：用测试预算跑一遍。** `_budgets()` 给定：目标至少减少 10、质量损失最多 0.02、计算最多增加 20、搬运最多增加 5、峰值最多增加 5、暴露时间最多增加 1.0 ms。`_effects()` 给定：计算 +5、搬运 −30、峰值 −20、暴露时间 +0.2、质量损失 +0.01。

以 `bandwidth` 为目标时，减少量 $R=30$，质量 0.01 通过，新增计算 5、搬运 0（负数被 `max(0,·)` 截成 0）、峰值 0、暴露 0.2 都在预算内，所以是 `measure`。把搬运改成 −5，$R=5<10$，变成 `tune`；改成 0，$R=0$，`primary_missed` 成立，直接 `reject`；把质量损失改成 0.05，也直接 `reject`，即使其他轴都合格。`test_budget_equalities` 再检查"恰好等于门槛"可以通过：计算 20、峰值 5、暴露 1、质量 0.02 全部取等号，结果仍是 `measure`。

**第 7 步：为什么不能做综合加权分。** 省 32 MiB 和多花 3 ms 单位不同，权重怎么定都是主观的；更危险的是，大额节省会掩盖某一轴的严重超支。notebook 的做法是"每一项分别和自己的预算比"，所有约束同时满足才进入 `measure`。`test_each_added_cost_budget` 分别构造计算、搬运、峰值、暴露时间四个轴各自超预算 +1 的情况，并特意把目标换成另一条轴，确保失败确实来自新增代价而不是目标方向。

**第 8 步：GPU 实验在比较什么。** Step 5 用一个本机随机 SwiGLU MLP：隐藏维 2048、中间维 5504，8 个 batch，Token 数 32 和 256；12 轮轮换预热、24 轮逐轮轮换三条路径，减少"谁先跑谁占便宜"的先后偏差。三条路径是：

- `baseline_fp32_serial`：FP32 串行；
- `bf16_serial`：同源权重和输入转 BF16，改变存储与执行精度；
- `fp32_overlap`：FP32 双缓冲，改变流依赖，尝试重叠。

判据写在 `GPU_CONFIG`：`minimum_speedup = 1.03`（中位加速至少 3%，对应时间下降约 $1-1/1.03\approx2.91\%$，notebook 特意提醒不要写成"下降 3%"）、`minimum_cosine = 0.999`（BF16 输出与 FP32 的余弦相似度）、`maximum_p95_ratio = 1.05`（P95 最多回归 5%）。`direction_passed` 对 BF16 看 `model_bytes` 是否下降，对重叠路径看中位时间是否下降——方向不对直接 `reject`。

**第 9 步：结果怎么读。** 已保存的 `rtx5070ti_candidate_rotating_v2` 记录中，BF16 在 32 Token 的加速比约 2.651、P95 比 0.489、模型字节和"参数 + 输入缓冲"都减少 50%、余弦 0.999967，判 `accept_for_load`；256 Token 加速约 2.876、P95 比 0.330、余弦 1.0，也判 `accept_for_load`。FP32 重叠在 32 Token 加速约 1.023、P95 比 0.977，没到 1.03，判 `tune`；256 Token 约 1.03×，通过现有时间门槛。

这些数字的边界必须读清楚：`model_bytes_reduction = 0.5` 只说明参数和输入缓冲的字节少了，不表示独立峰值下降 50%；`active_bytes_reduction` 只统计参数加两个输入缓冲；`matrix_peak_allocated_bytes` 是整组候选共同驻留时的矩阵峰值，不能拿来比较单个候选的独立峰值。`cosine_vs_fp32` 来自独立前向输出，不是双缓冲路径的逐元素验证；`checksum` 单独记录，还没有被当作正确性门槛。因此 5.4 只把 BF16 称为"当前更快单项"，重叠收益依赖负载，组合前还要比较"组合相对最佳单项的边际收益"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 资源轴 | 计算、搬运、峰值、暴露时间、质量五条独立的账 |
| 变化量 $\Delta x$ | 候选值减基线值，负数是减少，正数是增加 |
| 主约束 / 目标轴 | 当前最想改善的那条资源，比如带宽或容量 |
| 减少量 $R$ | 目标轴变化量的相反数，正数才叫"省了" |
| 新增开销 $A_x$ | `max(0, Δx)`，只记增加的量，减少不能拿来抵扣 |
| 质量损失 | 越大越差的指标，例如误差增加量或准确率退化量 |
| `reject` | 目标没减少或质量超限，直接停用 |
| `tune` | 方向对但幅度不够或有新增开销超预算，继续调 |
| `measure` | 纸面条件都满足，进入真实对照测量，尚未采用 |
| 峰值容量 | 同一时刻参数、激活、临时缓冲同时存活的总字节 |
| 暴露时间 | 真正卡在关键路径上、没有被隐藏的等待 |
| 轮换测量 | 每轮改变三条路径的执行顺序，抵消先后偏差 |

## 小白常见疑问
**Q1：目标轴减少了 30，但计算增加了 100，为什么不能"净赚"？**
因为单位不同，而且预算约束是逐轴的。`decide_candidate` 对计算、搬运、峰值、暴露时间分别检查 `max(0, Δx) <= 预算`，任何一轴超限就 `tune`，不会用搬运的 −30 去补计算的 +100。`test_quality_priority_and_improvements` 还验证了：即使所有资源都减少，质量超限仍然 `reject`。

**Q2：为什么 `primary_missed` 用 `reduction <= 0`，而不是 `< 0`？**
减少量为 0 表示目标资源一点没省，候选对当前主约束没有贡献，继续调也没有方向，所以直接 `reject`。`test_measure_tune_and_reject` 把搬运从 −5 改成 0，就是为了锁定"零变化等于拒绝"。

**Q3：`measure` 是不是就可以上线了？**
不是。它只表示"目标、质量、新增开销三组纸面条件同时满足，值得做真实对照"。notebook 的 GPU 实验还要看实际速度、P95、独立峰值和完整路径正确性；`measure` 之后仍可能因为实测不达标而回退到 `tune` 或 `reject`。

**Q4：BF16 参数减半、速度 2.65×，为什么峰值不能也说减半？**
参数只是峰值的一部分。运行时还有激活、临时缓冲区、输入输出和框架 workspace；BF16 路径可能引入新的转换缓冲。`matrix_peak_allocated_bytes` 又是三条候选共同驻留时的数字，不是单候选独立峰值。notebook 因此把 `active_bytes_reduction` 限定为"参数 + 输入缓冲"，并明确写"不表示独立峰值下降 50%"。

**Q5：为什么 1.03× 不直接写成"快 3%"？**
加速比 $=t_b/t_c$，时间下降比例是 $1-t_c/t_b=1-1/1.03\approx2.91\%$，不是 3%。这两个数在接近 1 时差别不大，但在大加速比时会明显分叉；notebook 在 5.1 特意提醒，避免把比值和百分比混写。

## 动手验证：怎么知道自己做对了
测试单元依次运行八个函数，全过打印 `✅ 资源变化、预算边界与候选筛选测试通过`。
- `test_complete_effect_axes`：`normalize_effects(_effects())["memory_bytes"] == -30.0`；多出 `score` 字段或少 `quality_loss` 都要报错。
- `test_effect_numeric_validation`：`True`、`"0"`、`None`、`nan`、`inf`、`-inf` 放在 `peak_bytes` 上都必须被拒绝。
- `test_each_constraint_axis`：四类约束的目标轴设 `-12` 返回 12、设 `0` 返回 0、设 `3` 返回 −3；未知约束报错。
- `test_measure_tune_and_reject`：搬运 −30 得 `measure`、−5 得 `tune`、0 得 `reject`、质量 0.05 得 `reject`。
- `test_budget_equalities`：计算 20、峰值 5、暴露 1、质量 0.02 全部取等号时仍是 `measure`，且 `quality_passed` 与 `added_costs_passed` 都为真。
- `test_each_added_cost_budget`：四个新增轴分别超预算 +1，都必须 `tune` 且 `added_costs_passed=False`。
- `test_invalid_budgets`：负值、布尔值、`nan`、`inf`、缺字段都必须被拒绝。
- `test_quality_priority_and_improvements`：质量 0.03 时即使计算 +100 也 `reject`；五个轴全为 −30 时 `added_costs_passed=True`。
notebook 保存了 5.2 和 5.3 的输出，包含 BF16、FP32 重叠的加速比、P95 比、字节减少和决策；自己重跑时以 `benchmarks/results/perf_candidate/perf-candidate_<run_label>.json` 里的 `comparisons` 和 `matrix.protocol` 为准。
