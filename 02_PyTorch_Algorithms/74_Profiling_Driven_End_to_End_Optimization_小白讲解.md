# 74. 性能剖析驱动的端到端优化 — 小白讲解

> 对应 notebook：[`74_Profiling_Driven_End_to_End_Optimization.ipynb`](./74_Profiling_Driven_End_to_End_Optimization.ipynb)

## 一句话总结
本节把“训练对照、激活显存实验、性能剖析”三类证据放在同一张桌子上：先用两个 TODO 保证不同实验条件的结果不被拼在一起，再用 `torch.profiler` 的配对 trace 和分阶段显存测量解释“为什么开了激活重算，前向省了显存，总峰值却几乎没降”。结论是：在 0.5B、FP32、AdamW、`seq_len=768` 的完整训练里 checkpoint 只省 16.43 MiB，低于 512 MiB 门槛，项目总决策仍是 `tune`；而在无动量 SGD、`seq_len=1280` 的激活隔离实验里它省 2820.34 MiB、吞吐为基线的 0.831477，只能在“激活显存”这个范围内记为 `accept_for_activation_memory_scope`，不能提升为完整训练结论。

## 为什么需要它：痛点是什么
初学者最容易犯的错，是把不同实验里最漂亮的数字挑出来拼成一句结论。比如看到“激活重算省了 2.8 GB”，就认为自己的 AdamW 训练也能省 2.8 GB；或者看到某个候选“前向峰值降了”，就宣布整步峰值降了。真实情况是：显存峰值由“同一时刻还活着的东西”决定，前向结束时的占用不等于优化器更新时的占用；而实验条件（优化器、序列长度）一换，结论就换了适用范围。

生活类比：体检报告。你不能拿“空腹血糖”的正常值去解释“餐后血糖”的异常，也不能把心内科的结论直接写进骨科的诊断书。本节要求每份报告都带着自己的“检查条件”（优化器、`seq_len`、是否含优化器状态），项目总结论只能由“完整训练”那份报告签字，专项报告只能在自己的科室里生效。

## 核心问题
- 两个实验组的配置字段到底要核对哪些？为什么缺一个组、错一个 `seq_len` 都要拒绝使用证据？
- 为什么项目总决策必须取 `full_training_adoption`，而不是报告顶层的 `decision`？
- 激活重算明明减少了前向保存的激活，为什么完整训练的全局峰值几乎不变？峰值通常落在哪个阶段？
- `peak_allocated_mb`、`peak_reserved_mb`、`allocator_slack_mb`、`throughput_ratio`、`loss_delta` 这些指标各自量的是什么？
- 为什么 16.43 MiB 的节省判 `tune`，而 2820.34 MiB 的节省也只算“范围内通过”？
- profiler 的计时为什么不拿来决定是否采用，而只用来解释瓶颈？

## 解决思路：一步步看懂

**第 1 步：先分清三类记录各自回答什么。** 完整训练对照固定模型、输入、优化器和更新流程，回答“这个策略在目标配置下能不能用”；激活显存对照固定模型、尽量去掉优化器状态干扰，回答“在已测长度下激活保存方式怎样影响容量”；算子与阶段剖析用同一套采集方法比较双方，回答“变化可能来自哪些计算、搬运或存活对象”。三者是“结论—机制—解释”的分工，不能互相替代。

**第 2 步：Step 4 的 TODO 1——逐字段核对证据范围。** 数据是模块级常量 `REPORT` 和 `EXPECTED_SCOPE_CONFIG`：

```python
EXPECTED_SCOPE_CONFIG = {
    'full_training_adoption': {'optimizer': 'AdamW', 'seq_len': 768},
    'activation_memory_mechanism': {'optimizer': 'SGD', 'momentum': 0, 'seq_len': 1280},
}
```

`validate_evidence_scopes(report, expected_config)` 先取 `scopes = report.get('decision_scopes', {})`，对每个 `scope_name` 取 `actual = scopes.get(scope_name)`；如果它根本不是 `dict`，就记一条 `f'{scope_name}:missing'` 并 `continue`。真正要补的只有一行：

```python
field_matches = actual.get(key) == expected_value
if not field_matches:
    issues.append(f'{scope_name}:{key}_mismatch')
```

这里用 `actual.get(key)` 而不是 `actual[key]`，是为了让“字段缺失”也走 `None != expected_value` 这条不匹配分支，而不是抛 `KeyError`。测试 `test_evidence_scope_config_gate` 覆盖四种情况：原样 `REPORT` 必须得到 `{'valid': True, 'issues': []}`；把激活组的 `optimizer` 改成 `'AdamW'` 后 `valid` 为假；删掉整个 `activation_memory_mechanism` 后 `issues` 里必须出现 `'activation_memory_mechanism:missing'`；把它的 `seq_len` 改成 `1536` 后 `valid` 为假。注意 `1279` 也不会通过，因为这里是精确相等，不是“差不多”。`REPORT` 里还有第三个范围 `profiler_attribution`（`'explain_only'`），但它不在 `EXPECTED_SCOPE_CONFIG` 里，不参与采用判定。

为什么只核这几个字段？因为它们是两组实验的“身份”：`full_training_adoption` 是 AdamW + 768，`activation_memory_mechanism` 是无动量 SGD（`momentum: 0`）+ 1280。只要优化器或长度变了，测到的就不是同一件事。配置通过只说明示例字段一致，原始性能测量仍要单独复核。

**第 3 步：Step 4 的 TODO 2——总决策只认完整训练组。** `build_project_decision(report, scope_summary)` 先从 `scope_summary` 里取 `full_training_adoption`，如果它不是字典或没有 `decision`，直接抛 `ValueError('完整训练范围缺失，不能生成项目总决策')`。要补的是：

```python
project_decision = full_training['decision']
```

不是 `report['decision']`。测试 `test_project_decision_uses_full_training_scope` 故意把顶层 `decision` 改成 `'accept'`，再断言结果仍是 `'tune'`；还构造了删掉完整训练组的 `missing_scope`，要求抛 `ValueError` 并被 `except` 捕获。这个设计的含义是：顶层字段可能是历史残留或人工汇总，只有完整训练组是“有资格签字”的那一份。

**第 4 步：已提供的 `summarize_scope_decisions` 在防什么。** 它把每个 scope 的 `decision`、`evidence_status` 和 `limitations` 复制出来，其中 `limitations` 用了 `list(...)`，是浅拷贝。测试 `test_scope_summary_preserves_status_and_copies_limits` 断言：激活组的 `decision` 是 `'accept_for_activation_memory_scope'`、`evidence_status` 是 `'blocked'`，第二条限制包含 `'未测得 OOM 边界'`；然后往摘要的 `limitations` 里追加 `'测试用本地修改'`，再断言原 `REPORT` 里没有这条。也就是说，摘要可以拿去做下游交接，但改摘要不能偷偷改原始证据。

**第 5 步：Step 5 的实验设计——自变量、因变量与门槛。** 这是本 notebook 的主体。自变量是“是否启用激活检查点”：`PROFILE_STRATEGIES = ['baseline', 'checkpoint']`，baseline 不启用，checkpoint 调用 `model.gradient_checkpointing_enable()`；其余（`Qwen/Qwen2.5-0.5B-Instruct`、batch 1、seq 768、FP32、seed 42、AdamW lr 1e-5）保持一致。因变量分三层：

- 执行 trace：`torch.profiler` 的 CPU/CUDA 活动，用 `record_function('phase.forward')`、`'phase.backward'`、`'phase.optimizer'` 给三个阶段打标签；保存为 `.pt.trace.json`，并用 `top_operators` 看热点算子。
- 阶段显存：`measure_phased_training_run` 在 forward/backward/optimizer 边界同步并记录每个阶段的 `peak_allocated_mb`，重复 `PHASE_REPEATS = 3` 次，最后给出 `global_peak_phase`（全局峰值落在哪个阶段）。
- 完整训练指标：不开启 profiler 的重复测量，包括 `step_time_ms`、`samples_per_s`、`loss`、`peak_memory_mb`、`peak_reserved_mb`、`allocator_slack_mb`。

门槛写在 `QUALIFICATION_GATES`：

```python
{'minimum_memory_saving_mb': 512.0,
 'minimum_throughput_ratio': 0.70,
 'maximum_loss_delta': 0.01}
```

它的读法是：checkpoint 相对 baseline 至少省 512 MiB、吞吐不能低于 baseline 的 0.70、loss 变化绝对值不超过 0.01，三条同时满足才可能在对应范围通过。注意 `0.70` 不是“加速 70%”，而是“保留至少 70% 的吞吐”，也就是允许重算带来最多约 30% 的减速。

**第 6 步：证据清单与阶段机。** `RUN_STAGE` 有四个值：`prepare` 只读检查、不碰 GPU；`profile` 重采配对 trace；`phase` 重测分阶段显存；`close` 读取已有证据并生成结论。`PROJECT_ID='74'`、`SEMANTIC_ID='PERF-PROFILING-CLOSURE'`，证据清单是 `benchmarks/results/74_evidence_manifest_rtx5070ti_profiling_closure_v5.json`。5.2 里每个上游文件都要过 `check_manifest_entry`：必须有 `path` 和 `sha256`，文件存在，且 `hashlib.sha256(...).hexdigest()` 与清单一致，否则直接报错，防止“换了上游结果却忘记换下游结论”。清单登记了 baseline（73 的 `..._training_baseline_v1.json`）、strategy（76 的 `..._activation_memory_v1.json`）、budget（75 的 `..._memory_budget_v1.json`）和激活专项。保存输出里清单哈希是 `0dea673f...`，`ready: True`，环境是 RTX 5070 Ti Laptop GPU、11.5 GiB、torch 2.8.0+cu128、CUDA 12.8。

**第 7 步：`profile` 阶段具体采什么。** `collect_training_trace(train_step, ..., warmup=2, iters=5)` 对每个策略用 `warmup=2`、`iters=5` 的 profiler 窗口采集，`train_step` 内部依次执行 `optimizer.zero_grad(set_to_none=True)`、带 `phase.forward` 标签的前向（`use_cache=False`）、`phase.backward` 的反向、`phase.optimizer` 的更新。checkpoint 策略在模型上启用 `gradient_checkpointing_enable()`，反向时会看到重算事件。`profile` 明确把 `measurement_role` 写成 `bottleneck_attribution_only`、`timing_for_adoption: False`——profiler 本身有开销，trace 里的 step time 只用来解释“多出来的时间花在哪”，不能拿去和 baseline 做采用判定。

**第 8 步：`phase` 阶段如何保证不 OOM。** 它先读 76 节 baseline 的 `peak_reserved_mb`，再取当前 GPU 的 `free_memory_mb`，要求：

```python
required_free_memory_mb = projected_peak_reserved_mb + PHASE_MIN_FREE_MARGIN_MB  # 512.0
phase_safety['admitted'] = free_memory_mb >= required_free_memory_mb
```

同时用 `torch.cuda.set_per_process_memory_fraction(PHASE_PROCESS_MEMORY_FRACTION, 0)` 把进程可用显存限制在 `0.92`，并在每次重复后 `gc.collect()`、`torch.cuda.empty_cache()`；捕获 `torch.OutOfMemoryError` 时把该次运行记为 `status: 'oom'`，而不是让整个 notebook 崩掉。只有两个策略的状态都是 `ok`，`phase_summary` 才会继续计算 `allocated_memory_saving_mb` 和 `global_peak_phase`。这也解释了 `tested_seq_lens` 会停在 `[768, 1024, 1280]`：再长就触到安全边界，notebook 选择记录“安全停止；未测得 OOM 边界”，而不是编一个 OOM 点。

**第 9 步：读结果——为什么总峰值不降。** 注册结果在 5.6：

| 比较 | 数字 | 判读 |
|---|---|---|
| 完整训练 `seq_len=768`、AdamW | checkpoint 省 16.43 MiB，低于 512 MiB；吞吐比 0.847209 | 总决策 `tune` |
| 激活隔离 `seq_len=1280`、无动量 SGD | 峰值 7663.69 → 4843.35 MiB，省 2820.34 MiB；吞吐比 0.831477；loss 差 0 | `accept_for_activation_memory_scope` |

2820.34 就是 `7663.69 - 4843.35`。完整训练只省 16.43 MiB，原因不是重算没用，而是“全局峰值不在前向”。0.5B 参数用 FP32 存约 2 GB，梯度再 2 GB，AdamW 的一阶矩和二阶矩各 2 GB（按 0.5B × 4 字节粗算），到 optimizer 阶段这些状态同时活着，前向保存的激活即使少了一大块，也未必是那一刻的最高点。

**第 10 步：为什么一个“通过”不能覆盖另一个“tune”。** 5.5 的 `assess_activation_memory_scope` 是独立重算，不从外部拿结论：它要求报告 `schema_version == 'training-performance-project/v1'`、`semantic_id == 'PERF-MEMORY-STRATEGY'`、`comparison_scope == 'activation_memory_isolation'`，并要求 `optimizer == 'SGD'` 且 `optimizer_momentum == 0.0`。然后从 `length_records` 里找 `comparison_seq_len` 对应的原始记录，要求 baseline 和 checkpoint 各自至少有 3 次成功运行，再用原始 `peak_memory_mb` 与 `samples_per_s` 重算节省和吞吐比。返回里固定写 `promotable_to_full_training: False`，三条 `limitations` 分别是“优化器状态被无动量 SGD 去掉”“不能证明 AdamW 完整训练可采用”“长度阶梯停在安全边界”。最终 JSON 同时存在两个 scope：

```json
"full_training_adoption": {"decision": "tune", ...},
"activation_memory_mechanism": {"decision": "accept_for_activation_memory_scope", ...}
```

顶层 `decision` 取前者，`reason` 是 `global_peak_memory_saving_below_threshold`，`next_action` 是 `retain_phase_evidence_and_do_not_adopt_at_fixed_workload`。这正是本节想验证的结论：策略在“激活占主导”的条件下有效，在“优化器状态占主导”的目标配置下不达标，两个结论可以同时成立。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 激活检查点 / 重算 | 前向不保存全部中间结果，反向需要时再算一遍，省显存但多花计算 |
| 激活卸载 | 把部分保存张量暂放到主机内存，反向时再搬回来，省显存但多花搬运 |
| `decision_scopes` | 按实验条件分栏存放结论，防止跨条件拼接 |
| `full_training_adoption` | 唯一有资格生成项目总决策的完整训练范围 |
| `evidence_status` | 证据状态，示例里激活专项是 `blocked`，表示还有未测边界 |
| `peak_allocated_mb` / `peak_reserved_mb` | 分配器实际给出的峰值 / 向 CUDA 预留的峰值 |
| `allocator_slack_mb` | 预留减分配，缓存池里没被用上的部分 |
| `global_peak_phase` | 全局峰值落在 forward、backward 还是 optimizer |
| `samples_per_s` | 每秒处理的样本数，吞吐指标 |
| `throughput_ratio` | checkpoint 吞吐 ÷ baseline 吞吐，0.70 是门槛 |
| `loss_delta` | 两者 loss 的绝对差，门槛 0.01 |
| SHA256 证据清单 | 用哈希锁定上游文件，防止结论和证据对不上 |

## 小白常见疑问

**Q1：既然 `profile` 里也有 step time，为什么不直接用它判断快慢？**
因为 profiler 会插桩、记录事件，本身改变执行时间。notebook 把 `profile` 的 `measurement_role` 写成 `bottleneck_attribution_only`、`timing_for_adoption: False`，采用判定用的是“关闭分析器后的重复测量”。trace 只回答“时间花在哪些算子和阶段”。

**Q2：checkpoint 在激活实验省 2820.34 MiB，为什么完整训练只省 16.43 MiB？是不是实验做错了？**
不是。两组条件不同：激活实验用无动量 SGD + `seq_len=1280`，优化器没有动量和二阶矩，峰值更容易被激活主导；完整训练用 AdamW + `seq_len=768`，更新阶段参数、梯度、两个矩和临时张量同时在场，峰值落在那里。notebook 明确要求“分别查看两组的峰值阶段，并回到目标训练配置核对收益”。

**Q3：`summarize_scope_decisions` 只是复制字段，为什么测试要专门检查“改摘要不影响原报告”？**
因为下游会拿着摘要继续加工；如果 `limitations` 直接复用原列表，任何 append 都会污染原始证据，导致“证据没变、结论变了”。`list(...)` 的浅拷贝对这个字符串列表足够。

**Q4：激活专项的 `evidence_status` 是 `blocked`，为什么还能给出 `accept`？**
`accept_for_activation_memory_scope` 只表示“在已测的 `seq_len=1280` 下，显存、吞吐、loss 三条门槛都过了”。`blocked` 指的是长度阶梯在安全边界停止、没有测到真正的 OOM 边界，所以它不能被提升为完整训练结论（`promotable_to_full_training: False`）。两个字段一起读才不会误读。

## 动手验证：怎么知道自己做对了

测试区在 Step 4 之后、Step 5 的 GPU 项目之前，三个函数直接调用，TODO 没填会抛 `NotImplementedError`，全过才打印 `证据范围配置、专项状态保留与项目总决策测试通过。`

- `test_evidence_scope_config_gate`：`validate_evidence_scopes(REPORT, EXPECTED_SCOPE_CONFIG)` 必须严格等于 `{'valid': True, 'issues': []}`；把激活组 `optimizer` 改成 `'AdamW'` 必须 `valid` 为假；删掉整组后 `issues` 含 `'activation_memory_mechanism:missing'`；`seq_len` 改成 `1536` 必须 `valid` 为假。
- `test_scope_summary_preserves_status_and_copies_limits`：摘要里激活组 `decision == 'accept_for_activation_memory_scope'`、`evidence_status == 'blocked'`、`'未测得 OOM 边界' in limitations[1]`；向摘要追加 `'测试用本地修改'` 后，原 `REPORT` 的 `limitations` 不能包含它。
- `test_project_decision_uses_full_training_scope`：`result['decision'] == 'tune'`，`result['scopes'] == scopes`；顶层 `decision` 改成 `'accept'` 后仍返回 `'tune'`；删掉 `full_training_adoption` 后必须抛 `ValueError`。

GPU 部分的验证不靠断言，而靠“保存的 JSON 与登记值是否一致”。notebook 保存的输出显示：5.2 预检 `ready: True`；5.3 的 `PROFILE_IDENTITY` 是 `Qwen/Qwen2.5-0.5B-Instruct`、batch 1、seq 768、`warmup: 2`、`iters: 5`、`phase_repeats: 3`、`run_stage: 'close'`；5.5 的 `evidence_review` 里 `profile.schema_ok: True`、`phase_memory.schema_ok: True`、`activation_memory_scope.decision` 为 `accept_for_activation_memory_scope`、`ready_for_decision: True`；5.6 最终输出 `status: 'completed'`、顶层 `decision: 'tune'`、激活组 `memory_saving_mb: 2820.34`、`throughput_ratio: 0.831477`、`loss_delta: 0.0`、`baseline_peak_memory_mb: 7663.69`、`checkpoint_peak_memory_mb: 4843.35`。如果你改参数重跑，先确认证据清单里的文件名和哈希都换成了新资产，再比对这三个数。

完整训练的 `16.43 MiB` 与 `0.847209` 写在 5.6 的“当前登记结果”表里，不是由两个 TODO 算出来的；两个 TODO 只负责“证据范围核对”和“总决策来源”。
