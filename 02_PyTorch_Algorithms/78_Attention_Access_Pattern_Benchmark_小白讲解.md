# 78. Attention 访问模式基准 — 小白讲解

> 对应 notebook：[`78_Attention_Access_Pattern_Benchmark.ipynb`](./78_Attention_Access_Pattern_Benchmark.ipynb)

## 一句话总结
这个项目不训练模型，而是教你**怎么审一份"稀疏注意力更好"的证据**：给定基线（`global`）和候选（如 `sliding_window`）两套访问方案，先用 `classify_access_comparison` 判断比较是否可控（同模型、同请求集、只改访问模式和后端），再用 `merge_access_evidence` 合并机制、质量、运行三层记录，最后用 `decide_access_candidate` 按"质量下限 + 连接密度、延迟、显存三项上限"给出 `accept` / `tune` / `reject`。CPU 部分用三个 TODO 实现这套规则；GPU 部分用 `prepare` / `probe` / `measure` / `close` 四步采集真实结果。

## 为什么需要它：痛点是什么
看到"滑动窗口注意力快 3 倍、显存降 80%"就想采用，是长上下文优化里最常见的坑：

- 候选换了模型或换了训练方案，性能差异未必来自访问模式；
- 请求集、分词器、生成长度变了，两组数字根本不可比；
- 连接数少了，不代表模型还能答对——"允许看哪些位置"和"真的答对"是两件事；
- 实现可能先在完整注意力矩阵上加掩码，或者稀疏内核回退到稠密计算，连接统计好看，实际计算量没降；
- 把别的方案的结果混进本次候选，或者缺一项指标就下结论。

生活类比：审核体检报告。先确认是同一个人（模型 commit）、同一套体检项目（请求集/分词器），再确认机制、质量、运行三张报告都来自这一次体检、没有串号，最后才看指标是否达标。任何一张缺失或来自别人，都不能签字。

## 核心问题
- 什么条件下，两组结果的差异才能归因于"访问模式 + 执行后端"这一组改变？
- 为什么候选自己声明的 `changed_fields` 不能代替逐字段核对？
- 机制、质量、运行三层证据各自回答什么？缺一层会怎样？
- 四个门槛为什么用合取？指标恰好等于阈值算通过吗？
- "回退到稠密计算"为什么要单独记录，且只能判 `tune` 而不是 `accept`？

## 解决思路：一步步看懂

**第 1 步：固定比较条件。** notebook 的 Step 1 表格要求同时记录四类信息：模型（ID、固定 revision、架构、训练方案）、访问规则（`global` / `sliding_window` / `block_sparse` / `selective_sparse` 及参数）、执行实现（框架、后端、内核、是否回退）、请求与运行设置（请求集、分词器、长度、生成上限、并发、重复次数）。模型版本必须用具体 commit 固定，请求集、分词器、长度、生成上限、并发、重复次数保持一致，只替换访问规则和真正执行它的后端。如果换了权重或训练方案，就要分别报告各模型表现，并列出同时改变的条件。

**第 2 步：把三类结果分开看。** Step 2 的表格把证据拆成三层：

| 层 | 记录什么 | 回答的问题 |
|---|---|---|
| 机制 | 连接密度、远处位置的可见范围、因果约束 | 模型被允许读取哪些历史位置 |
| 质量 | 固定任务集、得分、错误输出 | 限制访问范围后还能不能答对 |
| 运行 | 稀疏内核、是否回退、延迟、峰值显存 | 连接减少后是否真的更快、更省显存 |

"长程覆盖"指规则保留了多少远处连接，不等于模型已经找回远处答案。原生稀疏实现按选中的窗口或块执行、跳过未选连接；如果先算完整矩阵再加掩码，连接统计变少了，计算量仍接近稠密路径，所以必须记录内核名和 `fallback`。

**第 3 步：先判可比性，再判指标。** Step 3 的决策表是：记录不匹配、加载/请求失败、或条件匹配后未过质量/资源检查 → `reject`；同时改了其他条件、缺指标、稀疏路径未确认或回退 → `tune`；模型与运行条件匹配且质量、连接、延迟、显存都满足、原生稀疏已确认 → `accept`。这个顺序很重要：不可比的结果不能拿指标说事，低成本也不能掩盖答错。

**第 4 步：`validate_access_candidate`——身份门禁。** 它检查 9 个必填字段：`candidate_id`、`model_id`、`revision`、`family_id`、`training_recipe_id`、`pattern`、`backend`、`workload_id`、`tokenizer_revision`；`pattern` 还必须是四种合法值之一，否则把 `'pattern'` 加进 `issues`。返回 `{'ready': not issues, 'issues': issues}`。这一步只做字段级检查，不判断数值好坏。

**第 5 步：TODO 1——比较类型判断。** 参考答案三行：

```python
same_model = all(baseline.get(field) == candidate.get(field) for field in model_fields)
fixed_fields = (set(baseline) | set(candidate)) - allowed_changes
same_conditions = all(baseline.get(field) == candidate.get(field) for field in fixed_fields)
controlled = same_model and same_conditions and pattern_changed
```

其中 `model_fields = ('model_id', 'revision', 'family_id', 'training_recipe_id')`，`allowed_changes = {'candidate_id', 'pattern', 'backend', 'changed_fields'}`。`fixed_fields` 用两边键的并集减去允许变化的字段，所以**某一侧多出来的字段也会参与比较**：候选多写了一个 `max_tokens=128`，即使 `changed_fields` 只声明改了 `pattern`，`baseline.get('max_tokens')` 是 `None`、候选是 `128`，`same_conditions` 为假，结论降级为 `observational`。这就是"以实际配置为依据，不以声明代替核对"。

函数返回 `{'comparison_mode': mode, 'issues': issues}`。三种模式的含义：`invalid` 表示身份或请求集不匹配（`issues` 非空），连比较都不成立；`controlled` 表示同模型、其他条件一致、访问模式确实改变，可以把差异归因于"访问模式 + 后端"；`observational` 表示条件没控住，只能当作观察，不能证明因果收益。

**第 6 步：TODO 2——合并三层证据。** 参考实现：

```python
record_matches_candidate = record.get('candidate_id') == candidate_id
if not record_matches_candidate:
    failures.append(f'{role}.candidate_id')
```

三份记录（`mechanism`、`quality`、`runtime`）都要检查 `candidate_id`；`runtime` 自带的 `failures` 也会并入 `gate_failures`。然后按 `required` 字典收集缺失项：机制层要 `density`、`long_range_coverage`；质量层要 `quality_score`；运行层要 `latency_ms`、`peak_memory_mb`、`native_sparse_executed`、`fallback`。字段值为 `None` 就记成 `mechanism.density`、`quality.quality_score` 这样的路径，放进 `missing_evidence`。注意 `fallback=False` 是合法值、不算缺失；`None` 才算没测。返回值把三层的关键字段摊平成一个字典，并保留 `gate_failures`（身份串线、执行失败）和 `missing_evidence`（缺指标）两个问题清单。

**第 7 步：TODO 3——四项门槛联合判断。**

```python
checks = (
    evidence['quality_score'] >= limits['min_quality'],
    evidence['density'] <= limits['max_density'],
    evidence['latency_ms'] <= limits['max_latency_ms'],
    evidence['peak_memory_mb'] <= limits['max_peak_memory_mb'],
)
meets_limits = all(checks)
```

四个条件取合取：质量是"不低于下限"，其余三项是"不超过上限"，方向不能写反。`decide_access_candidate` 的判断顺序是：`comparison_mode == 'invalid'` 或 `gate_failures` 非空 → `reject`；不是 `controlled` → `tune`；`missing_evidence` 非空 → `tune`；`native_sparse_executed` 为假或 `fallback` 为真 → `tune`；再检查四项门槛，任一失守 → `reject`；全过 → `accept`，返回 `{'decision': ..., 'reason': ...}`。回退之所以只判 `tune`，是因为"这条实现路径没确认"，不代表方案本身差，补一次原生执行还能再测。

**第 8 步：用测试把规则钉死。** 测试固定 `BASELINE` 是 `global` + `dense-sdpa`，`CANDIDATE` 是 `sliding_window` + `native-window`，两者模型字段、`workload_id='long-context-v1'`、`tokenizer_revision='tok-a'` 相同，`changed_fields=['pattern']`。

- `test_comparison_contract` 断言这组是 `controlled`；把 `training_recipe_id` 改成 `'other'`、只换 `model_id`/`revision`/`family_id`、加 `max_tokens=128`、把 `pattern` 改回 `global`，六种情况都必须是 `observational`；只改 `changed_fields` 为 `'arbitrary_claim'` 仍是 `controlled`；`workload_id` 或 `tokenizer_revision` 不同则是 `invalid`。
- `test_evidence_merge` 用 `density=0.45`、`long_range_coverage=0.80`、`quality_score=0.91`、`latency_ms=8.0`、`peak_memory_mb=4200`、`native_sparse_executed=True`、`fallback=False`，断言缺失和失败都为空；混入 `candidate_id='other'` 的记录，`gate_failures` 必须包含 `mechanism.candidate_id`；三份记录都只有 `candidate_id` 时，`missing_evidence` 必须包含 `runtime.latency_ms`。
- `test_decision` 用 `min_quality=0.88`、`max_density=0.70`、`max_latency_ms=10.0`、`max_peak_memory_mb=5000`。上面的证据（0.91 / 0.45 / 8.0 / 4200）判 `accept`；把四项同时压到边界 `0.88 / 0.70 / 10.0 / 5000` 仍判 `accept`（等号算通过）；逐项改成 `0.87`、`0.71`、`10.01`、`5001` 都判 `reject`；`observational`、缺 `quality_score`、`fallback=True` 判 `tune`；`gate_failures=['request_failed']` 判 `reject`。

**第 9 步：Step 5 的真实采集管线。** 项目部分默认 `RUN_LABEL='replace_with_unique_label'`、`RUN_STAGE='prepare'`，四个阶段是 `prepare` / `probe` / `measure` / `close`，后三个阶段必须用唯一标签。它先读 `benchmarks/project_inputs/78_attention_access.json`，校验 `schema_version == 'architecture-project-input/v1'`、`semantic_id == 'ARCH-ATTENTION-ACCESS'`，取出 `model_id`、`model_revision`、`native_sparse_adapter`、`workload_path`；模型 revision 必须是 40 位十六进制 commit，请求集的 `request_id` 必须非空且唯一，每条请求要有 `prompt` 和 `expected_answer`，适配器文件必须存在并记录 SHA256。`preflight` 打印 `issues` 和 `ready_to_start`；`measure` 阶段如果仍有问题会直接 `RuntimeError`。

适配器协议是 `attention-access-native/v1`，接收 `--contract` 和 `--output`，结果 schema 是 `attention-access-native-run/v1`。`review_access_run` 会逐项核对：schema、`semantic_id`、`run_label`、contract 是否一致；`native_sparse_executed` 和 `fallback` 必须是真正的 `bool`；必须有 `failure` 字段；`backend` 和 `kernel` 非空；`baseline` 是字典、`candidates` 是非空列表、候选 ID 唯一；每条记录的模型 ID/revision 与 contract 一致；`workload.request_ids` 和 SHA256 与 contract 一致；`actual_input_tokens` 的键集合等于请求 ID 且都是正整数；`quality` 和 `metrics` 的键集合必须等于 baseline 加全部候选 ID 的集合，数值有限、非负，`quality_score` 不超过 1。

**第 10 步：结果怎么读、当前处于什么状态。** `close` 会重新校验所有来源文件的哈希，然后决定状态：如果只有启动前的能力检查（`capability`）文件，状态是 `pending`，理由"当前只有启动条件，尚无原生执行结果"；如果原始结果里有 `failure`，状态 `reject`；否则状态一律是 `tune`，理由按情况是"需要核对原生路径及独立质量集"或"原生执行记录完成，仍需独立质量集与采用门槛"。注意**即使稀疏执行成功、指标齐全，`close` 也不会直接给 `accept`**，因为采用还需要一组未参与参数选择的独立任务集和明确的采用门槛。notebook 5.6 的"当前登记结果"写明：模型与适配器尚未填写，未采集本版 GPU 结果，当前只准备了少量读取任务。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 访问模式（pattern） | 允许查询读取哪些历史：全局、滑动窗口、块稀疏、选择性稀疏 |
| 连接密度（density） | 保留的连接占完整注意力连接的比例，越低越稀疏 |
| 长程覆盖 | 规则保留了多少远处位置的可视连接，只是"允许看"，不代表"看懂了" |
| 可比性（controlled） | 同模型、同请求集、其他条件一致，只改访问模式和后端 |
| 观察性比较（observational） | 还改了别的条件，只能描述现象，不能归因 |
| 三层证据 | 机制（连接）、质量（答题得分）、运行（延迟/显存/内核） |
| 原生稀疏执行 | 内核真的跳过了未选连接，而不是先算完整矩阵再掩码 |
| 回退（fallback） | 稀疏路径不可用，实际退回了稠密计算 |
| 质量下限 / 成本上限 | 得分不能低于某值，密度、延迟、峰值显存不能高于某值 |
| 合取（all） | 四项必须同时满足，不能拿一项优秀抵消另一项失守 |
| 适配器（adapter） | 按约定参数启动后端、写出标准格式结果的调用脚本 |
| 固定输入 contract | 模型 commit、请求集哈希、适配器哈希等绑在一起的身份凭证 |

## 小白常见疑问

**Q1：候选自己在 `changed_fields` 里写"只改了 pattern"，为什么不算数？**
因为声明可能漏写。参考实现只把 `changed_fields` 放进 `allowed_changes`，不读它的内容，真正比较的是 `fixed_fields` 里每个字段的实际值。测试里 `{'changed_fields': ['arbitrary_claim']}` 仍判 `controlled`，而 `{'changed_fields': ['pattern'], 'max_tokens': 128}` 反而判 `observational`——多出来的 `max_tokens` 是实际差异，声明没提也照样被抓住。

**Q2：为什么 `{'pattern': 'global', 'backend': 'other-dense'}` 是 `observational` 而不是 `controlled`？**
因为基线本来就是 `global`，候选也改成 `global`，`pattern_changed` 为假。只换后端不构成"访问模式改变"，这次比较回答不了"窗口 vs 全局"的问题，所以只能当观察。`controlled` 要求 `same_model and same_conditions and pattern_changed` 三者同时成立。

**Q3：密度 0.45 是不是越低越好？**
不是单独看。密度低说明允许的连接少，可能更省计算，但也可能伤害质量。`test_decision` 里 0.45 能过 `max_density=0.70`，前提是 `quality_score=0.91` 过了 `min_quality=0.88`；如果把质量降到 0.87，即使密度仍是 0.45 也会 `reject`。四项是合取，低成本不能掩盖答错。

**Q4：为什么 `fallback=True` 只判 `tune`，不是 `reject`？**
`fallback` 表示这次实际跑的是稠密路径，不能用来证明稀疏实现的效果，但也不说明方案本身不可行。`decide_access_candidate` 把它和"未确认原生稀疏"一起归入 `tune`，下一步是保留实际路径、检查实现选择；只有身份/请求失败或指标失守才 `reject`。

**Q5：现在能在 notebook 里跑出真实的延迟和显存吗？**
不能。默认 `RUN_STAGE='prepare'`，只做只读的文件检查；而且 5.6 明确写着模型与适配器尚未填写、未采集 GPU 结果。CPU 测试里的 `8.0 ms`、`4200 MB` 只是规则测试样例，notebook 注释也写了"以下数值为 CPU 规则测试样例，不是后端实测结果"。要得到真实数字，必须补齐固定模型快照、请求集和原生稀疏适配器，再依次跑 `probe` / `measure` / `close`。

## 动手验证：怎么知道自己做对了

测试区依次调用 `test_candidate_contract()`、`test_comparison_contract()`、`test_evidence_merge()`、`test_decision()`，最后打印 `测试通过：两组条件、记录所属方案和指标筛选规则一致。`。

- `test_candidate_contract`：`CANDIDATE` 必须 `ready`；把 `revision` 置空、`pattern` 改成 `'unknown'` 后，`issues` 必须同时包含 `'revision'` 和 `'pattern'`。
- `test_comparison_contract`：`BASELINE` vs `CANDIDATE` 必须是 `controlled`；另外六种改动都必须降级为 `observational`——换 `training_recipe_id`、换 `model_id`、换 `revision`、换 `family_id`、加 `max_tokens=128`、把 `pattern` 改回 `global`；`changed_fields=['arbitrary_claim']` 仍是 `controlled`；`workload_id='other'` 和 `tokenizer_revision='tok-b'` 必须是 `invalid`。
- `test_evidence_merge`：完整记录的 `missing_evidence` 和 `gate_failures` 都为空；`{'candidate_id': 'other'}` 让 `gate_failures` 含 `mechanism.candidate_id`；三份只有 ID 的记录让 `missing_evidence` 含 `runtime.latency_ms`。
- `test_decision`：上述 `limits` 下，完整证据 `accept`；四项取边界 `0.88 / 0.70 / 10.0 / 5000` 仍 `accept`；`0.87`、`0.71`、`10.01`、`5001` 逐个 `reject`；`observational`、缺 `quality_score`、`fallback=True` 都是 `tune`；`gate_failures=['request_failed']` 是 `reject`。

notebook 没有保存这些 CPU 测试的执行输出，判断标准就是四个测试不抛异常并打印最后那行中文。GPU/项目部分需要真实输入文件，当前状态是"未采集"，不能根据 5.6 的 `pending` 反推性能数值。
