# 75. AdamW 执行路径基准 — 小白讲解

> 对应 notebook：[`75_Optimizer_Execution_Path_Benchmark.ipynb`](./75_Optimizer_Execution_Path_Benchmark.ipynb)

## 一句话总结
AdamW 有三种执行路径：逐张量更新（`single_tensor`）、把同类张量运算批量提交（`foreach`）、把更新中的多个操作融合执行（`fused`）。本节先在 CPU 上补全 `validate_comparison_contract` 与 `select_optimizer_path` 的比较合同和采用门槛，再用单 GPU 固定负载测量“优化器调用”和“完整训练步”的耗时、峰值显存与参数误差，最后只按完整训练步收益做决策。

## 为什么需要它：痛点是什么
第一个坑：听说 `foreach=True` 更快，就把所有优化器都换掉，结果训练时间几乎没变。因为优化器只占整个训练步的一部分，把局部 2 倍加速代进整步，可能只得到 1.1 倍。第二个坑：只看“优化器那几十毫秒”就宣布采用，却没检查额外显存和参数误差——不同实现可能多分配缓冲区，也可能因为数值路径不同让参数偏离。第三个坑：比较时不小心改了学习率、权重衰减或调度器，测到的差异根本不是执行路径造成的。

生活类比：超市结账。逐个商品扫码（single-tensor）慢，把同类商品一起扫（foreach）快，自助通道还能把称重和扫码合并（fused）；但如果顾客大部分时间花在排队和装袋（前向、反向、数据加载），收银快一倍，总时间也只能快一点点。基准要回答的是“整趟购物快了多少”，不是“扫码快了多少”。

## 核心问题
- 三种 AdamW 执行路径在代码里到底由哪些开关定义？
- 怎么证明三个候选只改变了执行路径，而没有偷偷改学习率、权重衰减或调度器？
- 局部优化器加速比、完整训练步加速比、峰值显存增量和参数误差，分别该怎么用？
- 为什么“优化器调用最快”不等于“应该采用”？
- 一次基准的 `prepare / measure / close` 三阶段各做什么？结果 JSON 里哪些字段决定 `accept / tune / pending`？
- 参数误差和显存门槛为什么必须和速度一起看？

## 解决思路：一步步看懂

**第 1 步：认清三条路径的开关。** 骨架给出的常量是：

```python
OPTIMIZER_PATHS = {
    'single_tensor': {'foreach': False, 'fused': False},
    'foreach': {'foreach': True, 'fused': False},
    'fused': {'foreach': None, 'fused': True},
}
```

`foreach` 把一组同形状张量的同类运算交给批量实现；`fused` 则走融合 kernel，代码用 `None` 表示这条路径不由 `foreach` 开关描述。三者使用同一个 AdamW 更新公式，比较的是执行方式，不是优化算法。

**第 2 步：TODO 1——核对共同训练条件。** 要让差异只能归因于执行路径，所有候选必须满足：优化器是 `AdamW`、学习率等于基准值、权重衰减等于基准值、调度器不启用。参考答案：

```python
common_config_matches = (
    row.get('optimizer') == 'AdamW'
    and row.get('learning_rate') == learning_rate
    and row.get('weight_decay') == weight_decay
    and row.get('scheduler') is None
)
```

`test_contract_rejects_common_config_drift` 把某个候选的学习率改成 `2e-4`，或把优化器改成 `SGD`，都要求 `valid=False`。容易写错的是忘记检查 `scheduler`、漏掉优化器名称，或者只检查候选集合而不逐行检查。

**第 3 步：TODO 2——把候选名映射到开关。** 不能只看名字叫 `fused` 就相信它真的融合了，必须把 `foreach/fused` 两个字段与 `OPTIMIZER_PATHS[name]` 整体比较：

```python
execution_path_matches = {
    'foreach': row.get('foreach'),
    'fused': row.get('fused'),
} == OPTIMIZER_PATHS[name]
```

`test_contract_rejects_execution_path_mismatch` 把 `foreach` 候选的 `fused` 改成 `True`，要求合同判无效。为什么用字典整体比较而不是分别写两个 `and`？这样字段少一个、值多一个都会自然失配，不容易漏检查。

**第 4 步：TODO 3——三项门槛同时通过才可采用。** 对每个非基线候选，先算

$$
\text{speedup}=\frac{\text{baseline.training\_step\_ms}}{\text{row.training\_step\_ms}},\qquad
\text{memory\_increase}=\text{row.peak\_allocated\_mb}-\text{baseline.peak\_allocated\_mb}.
$$

参考答案：

```python
eligible = (
    speedup >= minimum_speedup
    and memory_increase <= max_memory_increase_mb
    and error <= max_parameter_error
)
```

默认门槛是 `minimum_speedup=1.05`、`max_memory_increase_mb=256.0`、`max_parameter_error=1e-4`。注意 `speedup` 用“基线耗时 ÷ 候选耗时”，所以大于 1 才是变快；显存用“增量”而不是候选总量，因为基线本来就要占内存。`test_selection_accepts_fastest_candidate_passing_all_gates` 的固定数据是：基线 10.0 ms、900 MB、误差 0；`foreach` 9.0 ms、910 MB、1e-6；`fused` 9.8 ms、910 MB、1e-6。于是 `foreach` 加速 $10/9\approx1.111$ 通过，`fused` 只有 $10/9.8\approx1.020$，低于 1.05，最终选中 `foreach`。`test_selection_rejects_memory_and_numeric_gate_failures` 则把候选显存改成 1200 MB（增量 300 > 256）或误差改成 `1e-2`，两者都必须落选。

**第 5 步：为什么整步加速会被稀释。** 设优化器更新占基线训练步时间的比例是 $p$，局部加速比是 $s$，其余阶段不变，则整步加速比

$$
S=\frac{1}{(1-p)+p/s}.
$$

例如 $p=0.2$、$s=2$，得到 $S=1/(0.8+0.1)\approx1.111$。notebook 特别说明这个 20% 是假设值，用来解释“局部收益为什么会被前向、反向、数据加载稀释”。基准采用门槛 1.05，等价于整步耗时至少下降约 $1-1/1.05\approx4.76\%$。

**第 6 步：待判断（pending）的几种情况。** `select_optimizer_path` 在证据不完整时不会“矮子里拔将军”，而是返回 `pending`。触发条件包括：阈值非法（`invalid_threshold`）、候选集合缺失或重复（`candidate_set_mismatch`）、基线状态不是 `ok`（`baseline_unusable`）、`workload_id` 或 `seed` 不一致（`incomparable_evidence`），以及测量值缺失、非有限或为负（骨架里写作 `missing_measurement`，参考实现里写作 `invalid_measurement`）。测试只断言 `decision == 'pending'`，不比较 `reason` 字符串，所以两个版本的措辞差异不影响通过；读 JSON 时可以把 `reason` 当诊断信息，不要把它当稳定 API。

**第 7 步：GPU 实验的自变量、因变量与阶段。** 自变量是执行路径（三选一）；固定的是同一个优化器配置、同一组初始权重、同一个小型 MLP、相同的 `seed`、`warmup`、`iterations`、`repeats`。因变量有三类：完整训练步的中位耗时 `training_step_ms_median`、峰值分配显存 `peak_allocated_mb_median`、相对单张量路径的最大参数误差 `max_parameter_error_vs_single_tensor`；另外还记录优化器调用耗时与两种张量布局（256 个小张量 vs 4 个大张量，总元素数都是 16,777,216）。

运行分三段：`RUN_STAGE='prepare'` 只做环境与路径预检；`'measure'` 才在 CUDA 可用且空闲显存至少 2 GiB 时真正测量，并拒绝覆盖已存在的同名原始 JSON（`FileExistsError`）；`'close'` 复用已保存测量、只写决策登记，不再跑 GPU。5.5 还会核对 schema 必须是 `optimizer-path-benchmark/v1`、候选集合、布局、重复次数和三类必需指标，任何一项不完整都直接 `RuntimeError`，不重新测量。

**第 8 步：结果怎么读。** notebook 登记的一次运行使用 RTX 5070 Ti Laptop GPU、PyTorch 2.11.0+cu128、固定 MLP、3 次重复，决策为 `accept`，选择 `fused`：完整训练步加速 1.691 倍，峰值分配显存减少 7.938 MiB，最大参数误差 1.79e-7。读法顺序是：先看 `schema`/`run_label`/`device` 是否对得上，再看 `training_step` 与 `paths` 的候选是否齐全，然后比较整步中位耗时之比，最后确认显存增量和误差没有越界。不要拿 `optimizer_step_ms_median` 单独下结论——它只用来解释“局部快了多少”，不参与 `eligible` 判断。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| `single_tensor` | 逐个张量调用 AdamW，最朴素但提交次数多 |
| `foreach` | 把一组张量的同类运算批量提交 |
| `fused` | 把更新中的多个操作融合成一个 kernel |
| 执行路径 | 更新公式不变，只改变算子怎么组织、怎么提交 |
| 完整训练步 | 前向 + 损失 + 反向 + 优化器更新，决定真实 wall-clock |
| 局部加速比 | 只测优化器调用得到的比值，容易高估收益 |
| 整步加速比 $S$ | $1/((1-p)+p/s)$，会被非优化器阶段稀释 |
| `minimum_speedup` | 采用门槛，默认 1.05，即整步至少快约 4.76% |
| `max_memory_increase_mb` | 候选相对基线允许多占的峰值显存，默认 256 MiB |
| `max_parameter_error` | 固定更新后相对单张量路径的最大参数误差，默认 1e-4 |
| `pending` | 证据缺失/不可比，暂不决策 |
| `RUN_STAGE` | `prepare` 预检、`measure` 测量、`close` 只读结果做决策 |

## 小白常见疑问
**Q1：为什么 `fused` 的 `foreach` 写 `None`，不是 `True`？**
因为 `fused` 是另一条独立实现路径，不靠 `foreach` 开关来描述。`OPTIMIZER_PATHS` 用 `{'foreach': None, 'fused': True}` 明确表示“这条路径只看 fused”；如果硬写 `True`，反而会在合同检查里与定义不符。

**Q2：优化器调用快了 2 倍，为什么整步可能只快 11%？**
代公式就清楚：优化器只占整步的 20%，$p=0.2$、$s=2$ 时 $S=1/(0.8+0.1)\approx1.111$。测试 `test_selection_uses_full_step_not_local_optimizer_time` 专门构造“局部更快但整步没到 1.05”的记录，要求决策为 `tune`，所有候选 `eligible=False`。

**Q3：为什么显存门槛看“增量”而不是“总量”？**
基线本身就要占 900 MB 左右，若看总量，任何候选都会被误判。门控关心的是“换执行路径额外多花多少”，所以用 `peak_allocated_mb - baseline.peak_allocated_mb`，默认不超过 256 MiB。

**Q4：为什么参数误差门槛是 1e-4，而不是要求完全相等？**
不同 kernel 的累加顺序、融合方式可能带来浮点舍入差异，不要求逐位相同；但误差必须小到不影响更新语义，所以给一个固定上限。notebook 登记的真实运行误差是 1.79e-7，远低于门槛。

**Q5：为什么最终速度用“中位数之比”，不是“逐对加速比的中位数”？**
notebook 的原话是“最终速度使用各路径重复测量中位数之比，不是逐对加速比的中位数”。前者先分别汇总每条路径的典型耗时再相除，避免某几轮配对噪声主导；后者会把这些噪声带进每个比值。这个区别在配套的配对基准一节还会展开。

**Q6：`reason='missing_measurement'` 和 `'invalid_measurement'` 到底哪个对？**
两个都出现在 notebook 里：题目骨架用前者，参考实现用后者。测试只断言 `decision == 'pending'`，没有锁定 `reason`；实际读结果时把它当诊断字段即可，不要写成依赖字符串的判断。

## 动手验证：怎么知道自己做对了
测试全部在 CPU 上运行，最后打印 `AdamW 执行路径契约、证据有效性与采用门槛测试通过。`。
- `test_contract_rejects_incomplete_and_duplicate_candidates`：三候选齐全时 `valid=True`；少一个候选或重复第一个候选都必须无效。
- `test_contract_rejects_common_config_drift`：学习率改成 `2e-4`、优化器改成 `SGD` 都必须无效。
- `test_contract_rejects_execution_path_mismatch`：把 `foreach` 候选的 `fused` 改成 `True`，必须无效。
- `test_selection_requires_complete_comparable_candidates`：缺 `foreach` 或 `seed` 不一致都返回 `pending`。
- `test_selection_uses_full_step_not_local_optimizer_time`：局部优化器更快但整步 10.0 ms，决策 `tune`，所有候选不合格。
- `test_selection_accepts_fastest_candidate_passing_all_gates`：三候选齐全时 `accept`，选中 `foreach`（1.111 > 1.020）。
- `test_selection_rejects_memory_and_numeric_gate_failures`：显存 1200 MB 或误差 `1e-2` 都只得到 `tune`。
- `test_selection_marks_invalid_evidence_pending`：缺字段、NaN 耗时、`minimum_speedup=0` 都返回 `pending`。

GPU 部分需要 `PERF-CUDA-MECHANISM` 环境和至少 8 GiB 单卡；默认 `RUN_STAGE='prepare'` 只打印预检 JSON，不会自动测量。只有把 `RUN_LABEL` 改成唯一值、`RUN_STAGE` 改成 `measure` 才会写原始 JSON，`close` 再生成决策登记。notebook 里已有一份 `accept` 登记可以作为对照，但不要把它当成你自己环境的结论——结论只适用于当次 PyTorch、GPU 与 workload。
