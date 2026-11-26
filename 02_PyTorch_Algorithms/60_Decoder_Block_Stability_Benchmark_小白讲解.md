# 60. Decoder Block 稳定性基准 — 小白讲解

> 对应 notebook：[`60_Decoder_Block_Stability_Benchmark.ipynb`](./60_Decoder_Block_Stability_Benchmark.ipynb)

## 一句话总结
本节把"Pre-Norm 还是 Post-Norm、RMSNorm 还是 LayerNorm、SwiGLU 还是 GELU"这类结构选择写成受控比较：先用 `BlockSpec` 和四个 TODO 检查单个规格合法、两份规格只差一个字段、配对结果工作负载一致、按门槛给出 `accept / tune / reject`；再用 `prepare → measure → close` 的 GPU 探针测量两个本地 checkpoint 的运行损失、步时间、峰值显存和梯度范数。CPU 部分给出可验证的比较契约，GPU 部分只形成"观察性"证据，不会因为跑通就升级为 `accept`。

## 为什么需要它：痛点是什么
初学者常把"结构名字"当成结论：听说 Pre-Norm 更稳，就把 Post-Norm 换掉；看到某个候选验证损失低 0.02，就认为它更好。但一次比较里如果同时改了 Norm 位置和 Norm 类型，或者两个模型的宽度、层数、数据、训练预算不同，结果根本无法归因。更隐蔽的是成本：质量略好但步时间或峰值显存超预算，候选同样不能采用。

生活类比：换药试验。你想知道"新药是否更好"，必须保证两组病人只在这一种药上不同，其他治疗方案、饮食、观察周期都一样；还要同时记录疗效和副作用。如果一组换了药又换了剂量，或者疗效数据缺失，就不能下结论。本节把"只改一个字段""同一工作负载""质量、梯度、时间、显存四类门槛"写成代码里的硬检查。

## 核心问题
- 一个 Decoder Block 规格要满足哪些结构契约？SwiGLU 的 `d_ff` 为什么必须是偶数？
- 怎样证明 baseline 和 candidate 只改变了一个声明字段，而不是"看起来差不多"？
- 工作负载不一致、运行失败、证据缺失，这三种情况分别应该 `reject` 还是 `tune`？
- 四类门槛的方向分别是什么？为什么质量、时间、显存查上限，梯度范数查下限？
- GPU 探针测的 `runtime_loss` 为什么不能当作验证集质量？`close` 为什么只给 `tune` 或 `reject`？

## 解决思路：一步步看懂

**第 1 步：TODO 1，校验单个 Block 规格。** `BlockSpec` 是 `@dataclass(frozen=True)`，字段为 `norm_position`、`norm_kind`、`activation`，默认 `d_model=16`、`d_ff=32`。`validate_block_spec(spec)` 要返回 `{'ready': bool, 'issues': list}`，参考答案逐项检查：

```python
if spec.norm_position not in {'pre_norm', 'post_norm'}: ...
if spec.norm_kind not in {'rmsnorm', 'layernorm'}: ...
if spec.activation not in {'swiglu', 'gelu'}: ...
if min(spec.d_model, spec.d_ff) <= 0: ...
if spec.activation == 'swiglu' and spec.d_ff % 2: ...
```

最后返回 `{'ready': not issues, 'issues': issues}`。为什么 SwiGLU 的 `d_ff` 必须为偶数？因为它的门分支和值分支要等宽，融合投影输出 `2 × d_ff` 后再对半切；`d_ff=31` 无法均分，结构契约就不成立。测试用 `BlockSpec('pre_norm','rmsnorm','swiglu')` 检查合法，用 `norm_position='unknown'` 和 `d_ff=31` 检查两种拒绝。

**第 2 步：TODO 2，只允许一个结构差分。** `validate_single_block_change(baseline, candidate, declared_field)` 先列出三个结构字段里真正变化的那些：

```python
fields = ('norm_position', 'norm_kind', 'activation')
changed_fields = [field for field in fields if getattr(baseline, field) != getattr(candidate, field)]
```

然后依次检查：两份规格都必须通过 `validate_block_spec`；`d_model` 和 `d_ff` 必须完全相等（宽度预算不能偷偷变）；`len(changed_fields)` 必须等于 1；声明的 `declared_field` 必须等于唯一变化字段。返回 `{'ready': ..., 'changed_fields': changed_fields, 'issues': issues}`。

测试里 `pre_norm → post_norm` 得到 `changed_fields == ['norm_position']` 且 `ready` 为真；同时把 Norm 位置改成 `post_norm`、类型改成 `layernorm`，即使声明字段是 `norm_position`，`ready` 也必须为假。原因很直接：两个变量一起动，质量差异无法归因到某一个。`issues` 记录具体原因，而不是只返回一个布尔值，方便定位。

**第 3 步：TODO 3，核对配对结果。** `validate_paired_block_results(baseline, candidate)` 处理的是两份运行记录。骨架把两侧的 `failures` 合并，再检查 `workload_id`：

```python
failures = list(baseline.get('failures', [])) + list(candidate.get('failures', []))
if baseline.get('workload_id') != candidate.get('workload_id'):
    failures.append('workload_mismatch')
required = ('validation_loss', 'grad_norm', 'step_time_ms', 'peak_memory_mb')
missing = [f'{role}.{field}' for role, record in (('baseline', baseline), ('candidate', candidate))
           for field in required if record.get(field) is None]
```

返回字段里，`comparable` 只表示"没有 `workload_mismatch`"，不表示运行成功；运行失败放在 `gate_failures`，证据缺失放在形如 `baseline.validation_loss` 的 `missing_evidence`。这个区分很重要：工作负载不同是"比较无效"，运行失败是"硬失败"，字段缺失是"证据不全"，三者的处理方式不同。

测试构造 baseline（loss 1.0、grad 0.8、step 10.0、peak 4000）和 candidate（loss 0.98、grad 0.9、step 10.5、peak 4200），两者 `workload_id` 都是 `block-v1`，要求 `comparable` 为真且 `missing_evidence == []`；把 candidate 的 `workload_id` 改成 `other` 后，`comparable` 为假，并且 `gate_failures` 里必须出现 `workload_mismatch`。

**第 4 步：TODO 4，按顺序形成决策。** `decide_block_candidate(pair, limits)` 的优先级是"硬失败 → 缺证据 → 门槛"：

```python
if pair['gate_failures']:
    return {'decision': 'reject', 'reason': 'paired execution or workload contract failed'}
if pair['missing_evidence']:
    return {'decision': 'tune', 'reason': 'paired evidence is incomplete'}
checks = (
    candidate['validation_loss'] <= limits['max_validation_loss'],
    candidate['grad_norm'] >= limits['min_grad_norm'],
    candidate['step_time_ms'] <= limits['max_step_time_ms'],
    candidate['peak_memory_mb'] <= limits['max_peak_memory_mb'],
)
```

四项全过才 `accept`，否则 `reject`。方向不能反：验证损失、步时间、峰值显存越大越糟，查上限；梯度范数太小意味着梯度消失或更新没有有效传回，查下限。测试的 `limits` 是 `max_validation_loss=1.05`、`min_grad_norm=0.1`、`max_step_time_ms=12.0`、`max_peak_memory_mb=5000`，正常 pair 应 `accept`；只把 `missing_evidence` 改成 `['validation_loss']` 应 `tune`；只把 `gate_failures` 改成 `['runtime_failure']` 应 `reject`。

**第 5 步：GPU 项目的实验设计。** 5.1 的表把条件写得很细：环境 `ARCH-CUDA-MECHANISM`、单卡、参考显存 8 GiB；模型资产是一对"训练流程一致、只改声明字段"的本地快照；工作负载 batch 1、长度 128、FP16、3 次重复、seed 7；配对变量 `changed_field` 默认检查 `norm_kind`；指标是随机词元运行损失、前向与反向时间、峰值显存、首个有梯度参数的范数；结果写到 `benchmarks/results/60_decoder_block_stability/60_decoder_block_<run_label>.json`。

自变量是"Block 的一个结构字段"，因变量分四类：`runtime_loss`（数值是否正常）、`step_time_ms_median`（成本）、`peak_memory_mb`（容量）、`grad_norm_first_parameter`（梯度是否可达）。模型宽度、层数、数据、训练预算、评测输入都必须固定，否则比较无效。

**第 6 步：`prepare → measure → close` 三个阶段各自做什么。** 5.2 读取 `benchmarks/project_inputs/60_block_stability.json`，取出 `baseline_model`、`candidate_model`、`changed_field`，并调用 `tools/environment_preflight.py --profile ARCH-CUDA-MECHANISM`。检查项包括：环境预检是否 `ready`、两个模型路径是否登记、`measure` 阶段 CUDA 是否可用、`transformers` 是否安装、同名结果是否已存在。`prepare` 只报告缺失项，不加载模型；`measure` 有 issue 直接 `RuntimeError`。

5.4 的 `_run_checkpoint_probe(model_id)` 对每个 checkpoint 做固定探针：`torch.manual_seed(BLOCK_SEED)`、按 FP16 加载、`local_files_only=True`、`.cuda().train()`；用 `torch.randint(0, vocab_size, (1, 128))` 生成随机输入，`labels` 就是输入本身；先做一次不计时的 `zero_grad + forward + backward` 预热，再重复 3 次，用 `time.perf_counter()` 包住 forward + backward + `torch.cuda.synchronize()`，取中位数 `sorted(elapsed_ms)[len(elapsed_ms)//2]`。梯度范数取"第一个有梯度的参数"的 `grad.norm()`，不是全部参数的全局范数；峰值显存取 `torch.cuda.max_memory_allocated()`。记录里的 `quality` 是 `{'validation_loss': None, 'status': 'runtime_only'}`。

**第 7 步：`close` 为什么只给 `tune` 或 `reject`。** 5.6 的 `review_checkpoint_pair_for_close(record, expected_label)` 先核对 `semantic_id == 'ARCH-BLOCK-STABILITY'` 和 `run_label`；然后检查两侧的 `runtime_loss`、`grad_norm_first_parameter`、`step_time_ms_median`、`peak_memory_mb` 都是有限、非负，且时间和显存严格大于 0；`quality_ready` 检查两侧 `validation_loss` 是否为有限非负数。最后：

```python
return {'runtime_ok': runtime_ok, 'quality_ready': quality_ready,
        'decision': 'tune' if runtime_ok else 'reject'}
```

注意 `quality_ready` 被算出来，但没有参与 `decision` 的升级。即使探针数值正常、质量字段也有值，`close` 最多给 `tune`；要 `accept` 还需要相同评测集、质量门槛和其他条件一致的对照。`make_producer_record` 生成的 companion 也把 `evidence_level` 写成 `observational_gpu_benchmark`，`decision.reason` 是"matched quality evidence is required before adoption"。

**第 8 步：结果怎么读。** 5.5 只读当前标签的原始 JSON，打印 `evidence_review`：`semantic_id`、`run_label`、`changed_field`、`workload`、`candidate.quality.status`，以及 `ready_for_decision`（只要求 `semantic_id` 匹配）。如果没有结果文件，打印 `{'status': 'pending', 'expected_result': ...}`。读一份真实结果时按顺序看：先确认 `changed_field` 和 `workload` 是不是计划中的比较；再看两侧四个运行时数值是否有限、时间和显存是否为正；然后看 `quality.status` 是不是仍是 `runtime_only`；最后看 `close` 的 `decision` 和 `next_action`。本 notebook 的 dump 中没有保存任何 GPU 运行输出，也没有登记结果，所以不要假设已经测过某对模型；仓库当前的 `benchmarks/project_inputs/60_block_stability.json` 里 `baseline_model` 和 `candidate_model` 还是空字符串，`status` 是 `awaiting_matched_checkpoints`，`prepare` 会如实报告缺模型，`measure` 会因 `issues` 非空抛 `RuntimeError`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Decoder Block | Norm、Attention、FFN 加两条 residual 的基本单元 |
| Pre-Norm / Post-Norm | 归一化放在子层之前还是 residual 相加之后 |
| RMSNorm / LayerNorm | 只按均方根缩放 / 还会减均值再缩放 |
| SwiGLU | 门控 FFN，门和值两条分支必须等宽 |
| `BlockSpec` | 冻结的数据类，记录一个 Block 的三个结构字段和宽度预算 |
| 单变量比较 | 两份规格只允许一个结构字段不同 |
| 工作负载 `workload_id` | 标识两次运行是否在同一输入、批量、长度、精度下 |
| `gate_failures` | 硬失败：运行失败或工作负载不一致 |
| `missing_evidence` | 证据缺失：质量、梯度、时间或显存字段没有值 |
| `runtime_loss` | 随机词元上的运行损失，只查数值是否正常 |
| 观察性证据 | 只说明"能跑、数值有限"，不足以支持采用 |
| `accept / tune / reject` | 全部过门槛 / 补证据 / 硬失败或门槛失败 |

## 小白常见疑问

**Q1：`validate_block_spec` 为什么要单独检查 `d_ff % 2`？宽度为正不就行了吗？**
因为 SwiGLU 的融合投影输出 `2 × d_ff`，前向时要对半切成 gate 和 up。如果 `d_ff` 是奇数，两半宽度不同，`torch.chunk` 无法均分，门控就没有定义。代码只在 `activation == 'swiglu'` 时检查，GELU 不受这个约束。测试 `BlockSpec(..., d_ff=31)` 正是卡这一点。

**Q2：`comparable` 为真，是不是说明两次运行都成功了？**
不是。`comparable` 只表示没有 `workload_mismatch`，也就是两次运行声称在同一工作负载上。运行失败放在 `gate_failures` 里，证据缺失放在 `missing_evidence` 里。测试里 candidate 的 `workload_id` 改成 `other` 后 `comparable` 才变假；如果把 `failures` 填成 `['runtime_failure']`，`comparable` 仍可能是真，但 `decide_block_candidate` 会先看到 `gate_failures` 并直接 `reject`。

**Q3：为什么运行失败是 `reject`，证据缺失只是 `tune`？**
两者性质不同。运行失败说明这个候选在当前条件下跑不起来，是硬失败，继续补数据也救不回来；证据缺失说明比较可能有效，只是还没测全，补测后可以重新判断。代码的顺序也体现了这一点：先查 `gate_failures`，再查 `missing_evidence`，最后才看四项门槛。如果反过来先看门槛，缺字段会直接抛 `KeyError`，而不是给出可解释的 `tune`。

**Q4：`runtime_loss` 和验证损失有什么区别？为什么不能拿它当质量？**
`runtime_loss` 是在固定形状的随机 token 上算出来的：输入是 `torch.randint`，标签就是输入本身，模型没有在这些数据上训练过。它只能说明前向、反向能跑、数值有限，不能说明模型在真实任务上的表现。记录里 `quality.validation_loss` 被显式写成 `None`、`status='runtime_only'`，`close` 也因此不会给 `accept`。要比较质量，必须用匹配的独立验证集。

**Q5：`step_time_ms_median` 为什么取 `sorted(...)[len//2]`，而不是平均值？**
探针只重复 `BLOCK_REPEATS = 3` 次，样本很少，平均值容易被一次抖动拉偏；中位数对单个异常值更稳健。代码先排序再取中间下标，3 次时取第 2 个（下标 1）。这也意味着结果只能作为观察性成本，不是严格的 benchmark；notebook 把它和"完整质量评测"分开记录，正是为了避免把小样本计时当成采用依据。

## 动手验证：怎么知道自己做对了
CPU 测试单元在文件底部依次执行四组测试，全部通过打印 `测试通过：Block 契约、单变量比较、配对证据与项目门禁一致。`。dump 中没有保存这一格的输出，以断言为准。

- `test_block_spec_contract`：`BlockSpec('pre_norm','rmsnorm','swiglu')` 的 `ready` 必须为真；`norm_position='unknown'` 必须为假；`d_ff=31` 的 SwiGLU 必须为假。
- `test_single_change_contract`：`pre_norm → post_norm` 必须 `ready` 且 `changed_fields == ['norm_position']`；同时改 `post_norm` 和 `layernorm` 必须 `ready` 为假。
- `test_paired_results`：同 `block-v1` 的正常记录必须 `comparable` 且 `missing_evidence == []`；candidate 改成 `other` 后 `comparable` 为假，`gate_failures` 含 `workload_mismatch`。
- `test_block_decision`：在 `max_validation_loss=1.05`、`min_grad_norm=0.1`、`max_step_time_ms=12.0`、`max_peak_memory_mb=5000` 下正常 pair 为 `accept`；加 `missing_evidence=['validation_loss']` 为 `tune`；加 `gate_failures=['runtime_failure']` 为 `reject`。

GPU 部分的自查分阶段：`prepare` 应打印 `preflight` 字典，`ready` 为真才继续；`measure` 应生成 `60_decoder_block_<RUN_LABEL>.json`，里面两侧各有 `runtime_loss`、`step_time_ms_median`、`peak_memory_mb`、`grad_norm_first_parameter`，且 `quality.status` 应为 `runtime_only`；`close` 应打印 `{'decision': 'tune', ...}` 并生成同名 `_architecture.json`。如果 `prepare` 报"输入清单尚未登记匹配的 baseline/candidate 模型快照"，说明 `benchmarks/project_inputs/60_block_stability.json` 还没填好，不要跳过预检直接跑 GPU。
