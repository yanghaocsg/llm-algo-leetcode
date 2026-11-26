# 82. 量化产物评估项目 — 小白讲解

> 对应 notebook：[`82_Quantization_Artifact_Evaluation_Project.ipynb`](./82_Quantization_Artifact_Evaluation_Project.ipynb)

## 一句话总结
本节把"拿到一个量化权重目录，能不能放心进入部署对比"拆成三层证据：静态契约（身份、量化配置、文件角色与 SHA256 清单）、数据与质量（登记划分、样本摘要、按指标方向重算的比值）、真实加载（目标后端加载成功并完成非空往返生成）。三个 TODO 分别把三层组合成 `ready` 布尔值，最后用 `decide_artifact_adoption` 区分 `reject` / `pending` / `tune` / `accept`，核心结论是**缺测不等于超预算，不同摘要也不等于样本无交集**。

## 为什么需要它：痛点是什么
初学者复制一份量化权重目录后，常见三种翻车：目标后端找不到分词器文件；`config.json` 里的量化配置加载器不认识；文件传了一半，权重 SHA256 对不上但服务"看起来能跑"。还有一种更隐蔽的：质量表上写着"精度 10.5，基线 10.0，通过"，但没人重算 10.5/10.0 是否真的低于声明的 1.10 倍门槛，也没人确认评测集和校准集是不是同一批样本。

生活类比：收快递。先看面单（产物身份、来源），再拆箱核对清单（文件角色、大小、哈希），然后通电试机（真实加载 + 往返生成），最后看质检报告（质量指标和方向）。面单对不上就不用拆了；试机没做只能说"待验证"，不能说"超重"；试机成功但超预算才是"需要调整"。

## 核心问题
- 一个量化候选包的身份由哪些字段确定？为什么 `revision` 不能写 `main` / `master` / `latest`？
- 文件清单为什么要同时记录"角色"（weights/config/tokenizer）和 SHA256，二者各防什么？
- 质量指标有 `lower_is_better` 和 `higher_is_better` 两个方向，比值该怎么算、门槛该怎么比？
- 为什么"校准集和评测集划分不同、样本哈希不同"仍不能证明两批样本没有交集？
- 未加载、加载失败、缺显存测量、超预算分别对应哪个状态，为什么不能合并？

## 解决思路：一步步看懂

**第 1 步：三层检查各回答什么。** notebook 的 Step 2 给出清晰分工：文件与配置层回答"这是不是同一份产物"，数据与质量层回答"在指定样本和门槛下表现如何"，真实加载层回答"目标后端能不能读它并完成一次往返生成"。三层不能互相替代：哈希对得上不代表质量达标，质量达标不代表 vLLM 能加载，能生成也不代表答对（往返只要求非空输出，正确性要另看评测规则）。

**第 2 步：TODO 1——静态契约是三个必要条件的合取。** `validate_candidate_contract` 调用 `candidate_contract_facts` 拿到三个布尔值，要补的只有一行：

```python
contract_ready = identity_ready and quantization_ready and files_ready
```

三个值的判定细节（来自 `tools/quantization_project_runtime.py`）值得逐条看：

- `identity_ready`：`schema_version` 必须精确等于 `"quantization-artifact-candidate/v1"`；`artifact_id`、`model.id`、`model.revision` 都要是非空字符串；`model.revision` 不能是 `main`、`master`、`latest`——这三个是浮动引用，今天和明天指向的可能不是同一份权重，无法复现。
- `quantization_ready`：`method`、`format` 是非空字符串；`bits`、`group_size` 必须是**正整数且类型就是 `int`**。测试用 `True`、`4.5`、`'4'` 三种值验证：`True` 虽然是 `int` 的子类，但 `type(value) is int` 会拒绝；`4.5` 是浮点；`'4'` 是字符串。三者都不能被"截断成 4"放行。
- `files_ready`：`inventory` 是非空列表，每项有相对路径和 64 位十六进制 `sha256`；`weights` 和 `tokenizer` 是非空的相对路径列表，`config` 是单个相对路径；清单内路径不能重复，且 weights/config/tokenizer 提到的每个文件都必须出现在 inventory 里。

为什么角色和哈希都要？角色防"文件在但类型缺"——比如 `tokenizer: []` 时哈希清单可能仍然完整，但后端就是没法分词；哈希防"文件名在但内容被换"。测试 `test_candidate_contract` 的三条断言正对应：完整包 `ready`；删掉 tokenizer 角色后 `issues` 里有 `files`；inventory 某一项缺 `sha256` 后 `ready` 为假。`test_invalid_numbers_roles_and_evidence_scope` 再把 inventory 截成只剩 `model.safetensors`，此时 config 和 tokenizer 角色找不到对应文件，同样失败。

注意这一层只核对**声明**，不读磁盘。真正的文件哈希比较在 Step 5 的 `prepare_artifact_gate` 里做，notebook 的 Step 4 说明写得很明确："磁盘文件内容校验由真实项目流程执行，CPU 样例检查的是声明与判断规则"。

**第 3 步：TODO 2——质量要按方向重算，不能只信 `passed`。** `evaluate_candidate_evidence` 先从 facts 取 `direction`、`observed_ratio`、`quality_limit`，再算两个布尔值：

```python
quality_ready = bool(facts['quality_inputs_ready'] and (
    (direction == 'lower_is_better' and ratio <= limit)
    or (direction == 'higher_is_better' and ratio >= limit)
))
evidence_ready = split_isolated and sample_hashes_distinct and quality_ready
```

`observed_ratio` 的定义是 `candidate_value / baseline_value`，其中 baseline 必须是有限正数，candidate 必须是有限非负数。两种方向的含义：

- `lower_is_better`（如困惑度、错误率）：`limit` 取 `max_degradation_ratio`，要求 $\text{ratio}\le\text{limit}$。测试 fixture 是 baseline 10.0、candidate 10.5、limit 1.10，即 $10.5/10.0=1.05\le1.10$ 通过；把 candidate 改成 12.0 得 1.2，失败。
- `higher_is_better`（如准确率、保留率）：`limit` 取 `min_retention_ratio`，要求 $\text{ratio}\ge\text{limit}$。测试用 candidate 9.6、limit 0.95，$9.6/10.0=0.96\ge0.95$ 通过。

`quality_inputs_ready` 还要求 `evaluation["passed"] is True`、数值有限、`quality_limit` 是有限正数，并且如果填了 `ratio_vs_baseline`，它必须与重算值在 `rel_tol=1e-6` 内一致。测试 `stale_ratio` 把 `ratio_vs_baseline` 写成 0.5（真实值 1.05），即使方向判断本身能过，也要求 `ready` 为假——这是在防"缓存了旧比值"。

数据侧的两个条件更微妙。`split_isolated` 要求校准和评测的 `dataset`、`revision`、`config` 完全相同，`split` 都非空且**不同**；`sample_hashes_distinct` 要求两侧都有 `sample_ids_hash` 且不相等。notebook 特意强调："两个摘要不同仅说明记录不同"，函数里也把 `sample_disjointness_proven` 恒设为 `False`，`test_invalid_numbers_roles_and_evidence_scope` 断言 `checked['sample_disjointness_proven'] is False`。要真正证明无交集，得比较实际样本集合或文本内容，不能拿两个不同的哈希当证据。

`issues` 的标签是粗粒度的：`split_overlap` 不只在"两边同名 split"时出现，dataset/revision 不一致（测试里的 `foreign` 把评测 revision 改成 `other`）也会让 `split_isolated` 为假并打上这个标签。读结果时要看 facts 里的具体字段，不能只看标签字面。

**第 4 步：TODO 3——加载与预算要分开成四个状态。** 函数骨架已经处理了前两个分支：契约或质量不过 → `reject`；`loader_result is None` → `pending`。TODO 只算两个布尔值：

```python
loader_ready = facts['load_complete'] and facts['roundtrip_complete']
memory_ready = facts['budget_valid'] and (max_memory_mb == 0 or facts['memory_within_budget'])
```

`load_complete` 要求 `load_status == "loaded"`、没有 `failure`、`status` 不在 `{"failed", "error"}`；`roundtrip_complete` 要求 `prompt_count` 是正整数且 `non_empty_generations == prompt_count`，两者都必须是真正的 `int`（布尔值不算）。`budget_valid` 要求预算是有限非负数；`memory_within_budget` 要求峰值有限且不大于预算。

之后的返回顺序决定了状态语义：

1. `loader_ready` 为假 → `reject`（真实加载或往返没完成）。
2. 预算非法，或测量值存在但非法 → `reject`（数字本身有问题）。
3. `max_memory_mb > 0` 且 `memory_missing`（峰值为 `None`）→ `pending`（还没测，不是超了）。
4. `memory_ready` 为假 → `tune`（加载成功，但有效记录超预算）。
5. 其余 → `accept`。

`test_adoption_decision` 用同一个 loader（2/2 非空、峰值 1200）走遍四种状态：`None` → `pending`；`load_status="failed"` → `reject`；预算 1000 → `tune`；预算 1500 → `accept`；预算 0 → `accept`（0 表示不启用预算）。`test_missing_memory_and_invalid_roundtrip` 再补三个边界：峰值 `None` 加预算 1000 → `pending`；预算 `float('nan')` → `reject`；`prompt_count`/`non_empty_generations` 都是 `True` → `reject`。

容易写错的地方：把 `peak_memory_mb is None` 当成 0 去比较，会得到"没超预算"的假通过；把 `max_memory_mb=0` 当成"预算为零、任何占用都超"，会把不启用预算的正常候选误判成 `tune`；用 `if peak > max` 而不先检查两者是否有限，NaN 比较恒为 `False`，会静默放行。

**第 5 步：真实项目的 prepare → measure → close。** 5.1 只有三个标签：`RUN_LABEL`、`RUN_STAGE`（默认 `prepare`）、`UPSTREAM_RUN_LABEL`。`prepare` 只读预检，检查 40 节获选候选的身份、数据与质量声明、候选包和磁盘哈希，不初始化 GPU；`measure` 才执行 `tools/validate_quantized_artifact.py` 调 vLLM 做最小往返生成，并把加载记录保存下来；`close` 重读证据，生成准入报告、交接引用和登记回执，`accept` 时才产出供 67 节消费的稳定引用。已有同标签且内容一致时只读复用，内容冲突则停止，不覆盖历史记录。

5.4 的注释说明了一个口径陷阱：兼容字段 `peak_memory_mb` 是"设备空闲显存差"，不是过程峰值，当前合同也不启用峰值预算。所以即使真实运行看到这个数，也不能直接当成 `maximum_peak_allocated_bytes` 那种峰值来读。

**第 6 步：当前登记结果怎么读。** 5.6 明确写着"尚无本项目真实加载结果"，状态表里 `pending` 对应"尚无完整测量"，要执行 `measure` 或复用已有加载记录；`accept` 才进入浮点/量化部署对照。也就是说本 notebook 的 CPU 部分只能验证判断规则，不能证明任何具体量化包已经通过。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 量化产物（artifact） | 权重、压缩配置、模型配置、分词器等文件的集合 |
| 产物身份 | 产物编号 + 基座模型 + 版本，用来确认"查的是哪份" |
| 文件角色 | weights / config / tokenizer 等分类，防止缺关键文件 |
| SHA256 清单 | 每个文件的 64 位十六进制摘要，防止内容被换或损坏 |
| 校准集 / 评测集 | 量化时用来定参数的样本 / 用来判断质量是否达标的样本 |
| `lower_is_better` | 指标越小越好，如困惑度；比值不得超过退化上限 |
| `higher_is_better` | 指标越大越好，如准确率；比值不得低于保留下限 |
| 往返生成（roundtrip） | 给固定提示、收到非空输出，只证明能跑通，不证明答对 |
| `pending` | 检查还没做（未加载或缺显存测量），不是失败 |
| `tune` | 能加载但资源超预算，需要调参数/产物/预算 |
| 稳定引用 | `accept` 后生成的产物引用，供下游部署对照消费 |

## 小白常见疑问
**Q1：质量表明明写了 `"passed": True`，为什么还要重算？**
因为 `passed` 是上游的结论，可能对应旧基线、旧阈值或旧数据；本节要验证"按当前登记的方向和门槛重算是否仍通过"。`quality_inputs_ready` 里 `evaluation.get("passed") is True` 只是必要条件之一，真正决定 `quality_ready` 的是 `ratio` 与 `limit` 的比较。测试 `stale_ratio` 就是"结论写着通过、但缓存的比值对不上"的反例。

**Q2：`bits=True` 为什么不行？Python 里 `True` 不就是 1 吗？**
`candidate_contract_facts` 用的是 `type(value) is int and value > 0`，`type(True)` 是 `bool` 不是 `int`，所以直接拒绝。这是刻意的严格类型检查：配置里的位数必须是实实在在的整数，不能从布尔或字符串"猜"出来。同理 `4.5` 和 `'4'` 也不接受。

**Q3：校准集 split 是 `train`、评测集是 `validation`，样本哈希也不同，为什么还说不能证明无交集？**
不同 split 名称只说明登记不同，不同哈希只说明两串摘要不同；两批样本完全可能重叠而哈希仍然不同（比如顺序不同、多了一条）。函数把 `sample_disjointness_proven` 固定为 `False`，notebook 也提示要"比较样本集合或文本内容"。本节只能确认"记录上分开了"，不能确认"实际没重叠"。

**Q4：加载成功但没测显存，为什么是 `pending` 而不是 `tune` 或 `accept`？**
`pending` 的语义是"还缺证据"。`max_memory_mb > 0` 表示启用了预算，此时没有有效峰值就无法判断是否超限；判 `tune` 等于凭空说它超了，判 `accept` 等于凭空说它没超。`test_missing_memory_and_invalid_roundtrip` 断言这种情况返回 `pending`，而预算本身是 NaN 时才返回 `reject`。

**Q5：`max_memory_mb=0` 时为什么直接 `accept`，不怕显存炸吗？**
0 在本项目里的约定是"不启用预算门槛"，不是"预算为零"。`memory_ready` 写成 `budget_valid and (max_memory_mb == 0 or memory_within_budget)`，测试也断言 `max_memory_mb=0` 时同一个 1200 MB 的 loader 返回 `accept`。真实项目要用预算时，必须显式传一个正数。

## 动手验证：怎么知道自己做对了
测试区依次调用 5 个测试，最后打印 `测试通过：候选契约、数据与质量证据、真实加载决策相互一致。`。TODO 没填时 `validate_candidate_contract` 会先抛 `NotImplementedError`。

- `test_candidate_contract`：完整 fixture `ready`；`tokenizer=[]` 时 `ready` 为假且 `issues` 含 `files`；inventory 某项缺 `sha256` 时 `ready` 为假。
- `test_candidate_evidence`：默认候选 `ready`；评测 split 改回 `train` 且哈希与校准相同 → 假；candidate 12.0 → 假；`higher_is_better` 且 candidate 9.6、`min_retention_ratio=0.95` → 真。
- `test_adoption_decision`：`None` → `pending`；加载失败 → `reject`；峰值 1200 对预算 1000 → `tune`；对 1500 → `accept`；预算 0 → `accept`。
- `test_invalid_numbers_roles_and_evidence_scope`：`bits` 为 `True`/`4.5`/`'4'` 都失败；inventory 截断失败；质量值 NaN/Inf/True 失败；candidate 0.0 通过（$0/10=0\le1.10$）；`ratio_vs_baseline=0.5` 失败；`sample_hashes_distinct` 为真但 `sample_disjointness_proven` 为假；评测 revision 改成 `other` 失败。
- `test_missing_memory_and_invalid_roundtrip`：峰值为 `None` 且预算 1000 → `pending`；预算 NaN → `reject`；布尔计数 → `reject`。

notebook 没有保存执行输出，也没有登记任何真实加载结果；所有结论以代码常量、断言和 5.6 的状态表为准，不要编造 `peak_memory_mb` 或 `decision` 的实测值。
