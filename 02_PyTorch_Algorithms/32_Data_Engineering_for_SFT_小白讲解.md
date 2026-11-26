# 32. SFT 数据契约与泄漏审计 — 小白讲解

> 对应 notebook：[`32_Data_Engineering_for_SFT.ipynb`](./32_Data_Engineering_for_SFT.ipynb)

## 一句话总结

本节回答一个比"怎么训练"更靠前的问题：一条示范数据到底能不能进 SFT 训练集。它先定义数据契约（身份、切分、任务输入、监督目标），再构造两种指纹——只看任务输入的 `task` 指纹用来查训练集与评测集的任务重合，包含回答的 `example` 指纹用来查训练集内部的内容重复；`audit_sft_dataset` 把契约、身份重复、内容重复、评测泄漏四类风险独立记录，最后只用审计时保存的 `admitted_records` 构造训练样本，而不是按 ID 回查候选池。

## 为什么需要它：痛点是什么

初学者准备 SFT 数据时最容易做三件事：把同一份示范复制两遍（换个 ID）、把评测题混进训练集、看到字段齐全就认为数据可用。三件事的后果不一样但都很隐蔽：

- 重复样本会改变任务比重。比如退款任务被复制了 10 份，模型就会过度偏向退款场景，哪怕你"感觉"数据量变大了。
- 评测任务进入训练就是泄漏。模型在评测集上表现好，可能只是背过题，不能说明它对未见任务的能力。
- 只看 ID 查重不够。同一内容换个 `sample_id` 就能绕过；反过来，同一个 ID 也可能对应两条不同内容，按 ID 回查会把被拒绝的那条又带回训练。

生活类比：图书馆查重。只看书脊上的条码（ID）发现不了"同一本书换了新条码"；只看书名（任务输入）则会把"同一道题换一版答案"仍认成同一道题，正好适合查泄漏，却不适合查重——同一道题的两种合法答法会被当成重复。所以 notebook 用了两把不同的尺子：查训练集内部重复时量"整本书"（任务+回答），查评测泄漏时只量"题目"（任务输入，不含回答）。

还有一个容易忽略的点：空回答的记录仍然可能命中评测任务。如果代码在字段检查失败后立刻 `return`，就丢掉了已经能识别的污染原因。notebook 要求"可比较的风险分别记录"，就是为了让审计报告能同时说清"这条缺回答"和"这条还泄漏了"。

## 核心问题

- 一条 SFT 记录要满足哪些字段才算"契约完整"？缺 `source_id` 和缺 `response` 有什么区别？
- 为什么需要两种指纹？任务输入指纹和完整样本指纹各自能发现什么问题？
- 一条记录同时命中多种风险时，拒绝原因应该怎么记录？顺序为什么重要？
- 为什么被拒绝的记录不能加入 `seen_sample_ids` / `seen_training_examples`？
- 为什么最终训练样本要从 `admitted_records` 构造，而不能拿 ID 回候选池查？
- 精确指纹能发现改写后的近重复吗？不能的话应该补什么？

## 解决思路：一步步看懂

### Step 1：数据契约的四个维度

Step 1 的表格把契约拆成四块：

| 契约维度 | 接受条件 | 不满足时说明什么 |
|---|---|---|
| 可追溯身份 | 样本标识和来源非空，生成数据还保留生成与验证记录 | 无法回查数据来源与失败样本 |
| 切分角色 | 明确属于训练、验证或测试 | 无法确定该记录的用途 |
| 任务输入 | 指令非空，可选上下文与指令共同定义任务 | 模型缺少任务条件 |
| 监督目标 | 回答非空，并与任务输入分离 | 缺少有效目标或无法限定监督范围 |

代码里对应两个骨架函数。`normalize_sft_record` 做两类处理：对 `sample_id`、`source_id`、`split`、`instruction`、`input`、`response` 六个字段检查"要么是字符串、要么是 None"，然后 `strip()` 去掉首尾空白（None 变成空串）；对 `provenance` 要求必须是字典，并用 `deepcopy` 复制。注意它只去掉首尾空白，不改内部空白，也不做大小写转换——大小写转换留给指纹函数。

`assess_sft_contract` 的输出是四项：

```python
return {"missing_identity_fields": missing_identity_fields,
        "supervision_complete": supervision_complete,
        "contract_issues": contract_issues, "admitted": admitted}
```

- `missing_identity_fields`：`sample_id`、`source_id`、`split` 里哪些为空；
- `supervision_complete`：`instruction` 和 `response` 是否都非空；
- `contract_issues`：把问题变成稳定字符串，如 `missing_sample_id`、`missing_source_id`、`missing_split`、`invalid_split`、`missing_instruction`、`missing_response`；
- `admitted`：`contract_issues` 为空才为 True。

有一个细节：`invalid_split` 只在 `split` 非空且不在 `{"train", "validation", "test"}` 时追加。所以 `split` 为空只会得到 `missing_split`，不会同时得到 `invalid_split`。测试 `test_contract_assessment` 精确比较了整个字典：合法记录是 `{"missing_identity_fields": [], "supervision_complete": True, "contract_issues": [], "admitted": True}`；空回答记录是 `supervision_complete: False`、`contract_issues: ["missing_response"]`、`admitted: False`；把 `source_id` 置空后 `missing_identity_fields == ["source_id"]`；把 `split` 改成 `"holdout"` 后 `contract_issues == ["invalid_split"]`。

### Step 2：两把尺子——任务输入指纹与完整样本指纹

设规范化后的指令、上下文和回答分别为 $i(x)$、$c(x)$、$r(x)$，两类比较键是：

$$f_{\mathrm{task}}(x)=\big(i(x),c(x)\big),\qquad f_{\mathrm{example}}(x)=\big(i(x),c(x),r(x)\big)$$

`f_task` 只描述"模型要回答什么"，`f_example` 还描述"希望模型怎样回答"。用它们比较的范围也不同：

| 比较键 | 组成 | 比较范围 | 发现的问题 |
|---|---|---|---|
| 样本身份 | 数据来源声明的稳定标识 | 训练集内部 | 同一身份被重复使用 |
| 任务输入指纹 | 规范化后的完整任务条件，不包含目标回答 | 训练集与独立评测集 | 训练任务泄漏到评测集 |
| 完整样本指纹 | 规范化后的任务条件与目标回答 | 训练集内部 | 不同身份指向相同监督样本 |

为什么泄漏检查不能把回答也算进去？因为"同一道题换一个回答"仍然是同一道题。如果指纹包含回答，出题人只要换一版参考答案，污染就能逃过检查。反过来，训练集去重必须包含回答：同一道题配两个不同答案，可能是两个合法样本（比如多风格示范），不能一律当重复。

指纹函数里已经写好了 `casefold()`：

```python
instruction = record["instruction"].casefold()
input_text = record["input"].casefold()
response = record["response"].casefold()
```

`casefold()` 比 `lower()` 更彻底，适合做大小写无关的比较。但 notebook 明确说这是"精确比较"：去掉首尾空白、统一大小写、保留正文内部空白。它能发现"换 ID、换回答"的重复，但发现不了改写、同义替换、换标点这类近重复；近重复需要相似度筛查和人工复核。对于大小写本身有含义的代码数据，notebook 的建议是调整规范化规则并重新审计，而不是硬套这套规则。

### Step 3：风险取并集，准入看是否为空

把命中的拒绝原因记为集合 $R(x)$：

$$R(x)=R_{\mathrm{contract}}(x)\cup R_{\mathrm{id}}(x)\cup R_{\mathrm{duplicate}}(x)\cup R_{\mathrm{leak}}(x),\qquad A(x)=\mathbf{1}[R(x)=\varnothing]$$

四个子集分别对应契约、身份重复、完整样本重复、评测泄漏；$\mathbf{1}$ 在条件成立时取 1，否则取 0。这个式子的工程含义是：四种检查都要跑，不能因为第一种失败就跳过后面三种。`audit_sft_dataset` 的代码结构正是这样：先算 `contract` 和 `fingerprints`，再分别算三个布尔量，最后把 `contract["contract_issues"]` 和三个风险标签依次 `append` 到 `reasons`。

### Step 4：TODO 1——构造两类指纹

参考答案只有两行：

```python
task_fingerprint = (instruction, input_text)
example_fingerprint = (instruction, input_text, response)
```

看起来简单，但容易写错成三种：

1. 两个都带 `response`：任务指纹失去"换回答仍能查泄漏"的能力。
2. 两个都不带 `response`：完整样本指纹退化成任务指纹，同一道题的两个合法答案会被误判重复。
3. 用列表而不是元组：列表不可哈希，放进 `set` 会直接 `TypeError`；而且列表比较是逐元素但不可作为字典键。

`test_fingerprint_semantics` 的断言很直接：两条记录 `instruction="解释 LoRA"`、`input=""`，回答分别是 `"答案 A"` 和 `"答案 B"`；`first_keys["task"] == second_keys["task"]` 必须为真，`first_keys["example"] != second_keys["example"]` 也必须为真。这一条就把"回答变化只改变完整样本指纹"锁死了。

### Step 5：TODO 2——独立判断重复与泄漏

参考答案：

```python
is_duplicate_example = (bool(record["instruction"] and record["response"])
                        and fingerprints["example"] in seen_training_examples)
is_evaluation_leak = (bool(record["instruction"])
                      and fingerprints["task"] in evaluation_task_fingerprints)
```

两个 `bool(...)` 前置条件不是装饰：

- `is_duplicate_example` 要求 `instruction` 和 `response` 都非空。空回答的记录没有完整监督内容，把它算成"重复样本"没有意义；它应该先因为 `missing_response` 被拒绝。
- `is_evaluation_leak` 只要求 `instruction` 非空，不要求 `response`。这正是"空回答不能掩盖泄漏"的实现：哪怕这条记录缺监督，只要任务输入命中冻结评测集，仍然要记 `evaluation_task_leak`。

`test_incomplete_supervision_still_reports_leak` 就是专门压这个点的：一条 `response=None` 的记录，任务输入在评测集合里，`reasons` 必须同时是 `["missing_response", "evaluation_task_leak"]`，而不是只有前者。

### Step 6：审计主循环的三个易错点

`audit_sft_dataset` 的循环里有三处值得逐行看：

**第一，跳过验证/测试切分。** `if record["split"] in {"validation", "test"}: continue`。这两类记录不进入训练审计，也不加入 `seen_*` 集合。注意 `test_reused_identity_cannot_reenter_handoff` 往列表末尾追加了一条 `split="validation"` 的副本，它会被直接跳过，不会污染训练侧的去重状态。

**第二，`seen_*` 只在契约通过时更新。**

```python
if contract["admitted"]:
    seen_sample_ids.add(record["sample_id"])
    seen_training_examples.add(fingerprints["example"])
```

为什么不是"只要处理过就加入"？假设第一条记录 `sample_id="same"` 但缺 `source_id`，第二条记录同 ID、字段完整、内容合法。如果第一条就把 `"same"` 加进 `seen_sample_ids`，第二条会被误判成 `duplicate_identity`，一条坏记录毒化了后续合法样本。`test_rejected_record_does_not_poison_seen_state` 断言第一条原因是 `["missing_source_id"]`、第二条 `decision == "admit"`，就是在防这个。

**第三，准入时保存的是 `deepcopy(record)`。**

```python
if admitted:
    admitted_sample_ids.append(record["sample_id"])
    admitted_records.append(deepcopy(record))
```

如果只存 ID，后面按 ID 回候选池查，可能把同 ID 的被拒行、甚至被外部修改过的行带回来。存副本则把"审计通过时看到的内容"固定下来。`test_reused_identity_cannot_reenter_handoff` 在审计后故意把 `records[0]["response"]` 改成 `"审计后的外部修改"`，再构造训练样本，断言输出仍然是原来的 `"申请、核验、确认。"` 和 `"核对订单、更新地址、确认结果。"`，并且没有 `source-d` 的行——这就是"快照"和"按 ID 回查"的区别。

### 用五条 fixture 走一遍完整审计

`_sample_sft_records` 的五条记录（都已规范化）是：

| 行 | sample_id | source_id | instruction | input | response | 说明 |
|---|---|---|---|---|---|---|
| 0 | train-1 | source-a | 按三步处理退款请求 | 订单已签收 | 申请、核验、确认。 | 干净但任务在评测集 |
| 1 | train-2 | source-b | 按三步处理退款请求 | 订单已签收 | 申请、核验、确认。 | 与行 0 内容完全相同 |
| 2 | train-3 | source-c | 说明退款进度 | 已经提交申请 | 空 | 缺监督 |
| 3 | train-1 | source-d | 解释退款时限 | 订单已签收 | 核验后告知处理时限。 | 复用行 0 的 ID，内容不同 |
| 4 | train-4 | source-e | 说明修改地址步骤 | 订单尚未发货 | 核对订单、更新地址、确认结果。 | 干净且任务不在评测集 |

评测集合设为 `{build_sft_fingerprints(records[0])["task"]}`，也就是"按三步处理退款请求 + 订单已签收"这一道题。审计结果：

- 行 0：契约通过，任务命中评测 → `["evaluation_task_leak"]`。虽然它被拒，但契约是 `admitted=True`，所以它的 ID 和完整样本指纹会进入 `seen_*`。
- 行 1：内容与行 0 完全相同 → `example` 命中已见集合；任务也命中评测 → `["duplicate_example", "evaluation_task_leak"]`。注意原因顺序是"先重复、后泄漏"，因为代码先 `append` 重复再 `append` 泄漏。
- 行 2：回答为空 → `["missing_response"]`。契约不通过，不进入 `seen_*`。
- 行 3：ID 与行 0 相同 → `["duplicate_identity"]`。它的任务不同，所以不泄漏；它的完整样本也不同，所以不算内容重复。
- 行 4：全部通过 → `admit`。

所以 `admitted_sample_ids == ["train-4"]`，`rows[0]["reasons"] == ["evaluation_task_leak"]`，`rows[1]["reasons"] == ["duplicate_example", "evaluation_task_leak"]`，`rows[2]["reasons"] == ["missing_response"]`，`rows[3]["reasons"] == ["duplicate_identity"]`。测试 `test_dataset_audit` 逐条断言这四个列表，任何顺序变化都会失败。

### 从准入清单到训练样本

`build_sft_training_examples` 只遍历 `audit_report["admitted_records"]`，拼接方式是：

```python
prompt = template_prefix + record["instruction"]
if record["input"]:
    prompt += "\n### Input\n" + record["input"]
```

输出的每条样本保留 `sample_id`、`source_id`、`split`、`prompt`、`response`、`provenance` 六个键。`test_training_interface_contract` 用 `set(examples[0]) == {...}` 精确检查字段集合，并断言 `source_id == "source-e"`、`split == "train"`，以及输出与 `admitted_records` 的 `(sample_id, source_id, response)` 逐项一致。这个接口是给 13 节 SFT 训练循环用的，字段多了少了都会破坏下游。

`test_provenance_survives_handoff` 还检查了血缘：原始 `provenance` 里嵌套 `{"verifier_evidence": {"format": {"passed": True, "version": "v1"}}}`，审计和模板化之后仍然相等；然后修改原始字典里的 `version`，断言输出仍是 `"v1"`，说明 `deepcopy` 真的隔离了可变对象，而不是共享引用。最后用 `response=123` 触发 `ValueError`，确认非字符串字段会被 `normalize_sft_record` 拒绝。

### 审计的范围与边界

`audit_sft_dataset` 只审训练侧：`validation` 和 `test` 行被 `continue` 跳过，函数不会去检查评测集内部是否重复，也不会修改评测集。它信任调用方传入的 `evaluation_task_fingerprints` 是"冻结的"——这些指纹应该在训练开始前就固定，而不是每轮从当前候选池重算。notebook 的 Step 2 解释了为什么：验证集可以用于候选选择，最终测试集只用于方案确定后的独立检验；如果评测任务本身在变，泄漏检查就失去锚点。

这也意味着审计通过只代表"相对于给定的评测指纹，没有精确任务重合"。如果评测集后来加了新题，或者指纹集合不完整，训练集里可能仍然存在未被发现的重合。所以 `evaluation_task_fingerprints` 的来源、版本和哈希应该和训练记录一起保存；notebook 的后续项目（62 指令微调和 64 SFT 数据质量）会在更大的数据清单上继续做这件事。

再补充一个报告粒度的细节：身份缺失会同时出现在两个地方——`missing_identity_fields`（结构化列表，方便下游按字段处理）和 `contract_issues`（字符串列表，方便直接展示原因）。监督缺失（`missing_instruction`、`missing_response`）只进 `contract_issues`，另有 `supervision_complete` 布尔量做汇总。为什么不全塞进一个列表？因为"身份"和"监督"是两类不同的契约：身份决定能不能回查来源，监督决定有没有训练目标。分开报告，修复时才知道是补元数据还是补内容。

### 框架一致性核对

可选单元把准入后的记录塞进 Hugging Face `Dataset.from_list`，`Features` 里显式声明 `sample_id`、`source_id`、`split`、`prompt`、`response` 五个字符串列，以及嵌套的 `provenance`（`generator_id` + `verifier_versions.format`）。六个检查是：

- `schema_preserved`：`dataset.features == features`；
- `values_and_order_preserved`：`dataset.to_list() == examples`，行内容和顺序都不能变；
- `admitted_rows_only`：Dataset 里的 ID 列表等于 `audit["admitted_sample_ids"]`；
- `rejected_duplicate_excluded`：没有 `source_id == "rejected-source"` 的行；
- `lineage_preserved`：每行 `source_id` 非空且 `split == "train"`；
- `supervision_fields_readable`：每行 `prompt` 和 `response` 都非空。

如果环境里没装 `datasets`，单元返回 `status: skipped`，不会伪装成通过。

这份核对里的 `rejected-source` 行也有讲究：它复用了 `raw_records[1]` 的 `sample_id="train-2"`，但把 `source_id` 换成 `"rejected-source"`、回答换成 `"同标识的被拒内容"`。它会被 `duplicate_identity` 拒掉，所以 `rejected_duplicate_excluded` 检查的是"同 ID 换个来源也不能重新进入训练"，对应 Step 2 表格里"同一身份被重复使用"那一行。注意它并不是内容重复的案例，因为回答已经被改掉了。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| 数据契约（data contract） | 对来源、切分、任务输入、监督目标的共同约定 |
| 规范化 | 入库时去首尾空白；指纹阶段再统一大小写，便于稳定比较 |
| 任务输入指纹 | `(instruction, input)`，描述"要回答什么" |
| 完整样本指纹 | `(instruction, input, response)`，描述"整条示范" |
| 身份重复 | 同一个 `sample_id` 被用了两次 |
| 内容重复 | 不同 ID 指向完全相同的任务+回答 |
| 评测泄漏（leakage） | 训练任务命中了冻结的验证/测试任务 |
| 冻结评测集 | 预先固定成员与用途、不进入训练的集合 |
| 准入（admit） | 所有可判定风险都没命中，记录可以进训练 |
| 记录快照 | 审计通过时保存的独立副本，防止后续被改 |
| `provenance` | 生成器、验证器版本等血缘证据 |
| 近重复 | 改写、同义替换后的相似样本；精确指纹查不到 |

## 小白常见疑问

**Q1：为什么泄漏检查不看回答，而去重检查要看回答？**

因为两者要回答的问题不同。泄漏问的是"这道题是不是已经用于评测"——同一道题换一版参考答案，还是同一道题，所以任务指纹只取 `(instruction, input)`。去重问的是"这条监督样本是不是已经出现过"——同一道题配两个不同答案，可能是两个合法示范，所以完整样本指纹要带 `response`。notebook 的 `test_fingerprint_semantics` 用"回答 A / 回答 B"直接演示了这个区别。

**Q2：为什么被拒绝的记录不能加入 `seen_sample_ids`？**

因为"见过"和"合法"是两件事。一条缺 `source_id` 的记录如果先占了 ID，后面同 ID 的合法记录会被误判成身份重复。`test_rejected_record_does_not_poison_seen_state` 的两条记录 ID 都是 `"same"`，断言第一条因 `missing_source_id` 被拒、第二条被 `admit`。所以 `seen_*` 的更新条件是 `contract["admitted"]`，不是"循环到了"。

**Q3：行 3 的 ID 和行 0 一样，为什么原因里没有 `duplicate_example`？**

因为它虽然复用了 ID，但任务和回答都不同：指令是"解释退款时限"、回答是"核验后告知处理时限。"，完整样本指纹和行 0 不一样。它命中的是身份重复，不是内容重复。这也说明两种检查不能互相替代：只查内容会漏掉"同 ID 不同内容"，只查 ID 会漏掉"不同 ID 同内容"。

**Q4：审计后我把原始列表改了，为什么训练样本没变？**

因为 `admitted_records.append(deepcopy(record))` 保存的是副本，不是引用。`test_reused_identity_cannot_reenter_handoff` 在审计后修改 `records[0]["response"]`，再调用 `build_sft_training_examples`，输出仍是旧内容。这也是为什么不能"只存 ID 再回查"：回查拿到的是被改过的当前状态，而不是审计时的快照。

**Q5：精确指纹能发现"把句子改写一下"的重复吗？**

不能。notebook 说得很明确：这套规则是"去掉首尾空白并统一大小写，保留正文内部空白"的精确比较，能发现换标识或换回答的重复任务；改写后的近重复需要相似度筛查与复核。所以审计通过不等于"绝对没有近重复"，只是"没有精确重复和精确任务重合"。如果你的数据里有大量同义改写，应该在这套精确审计之后再加一层相似度检查，而不是把 `casefold()` 当成语义去重。

## 动手验证：怎么知道自己做对了

测试是直接调用，不是 pytest；TODO 没填会抛 `NotImplementedError`。前六个测试跑完打印 `✅ SFT 契约、指纹、泄漏审计与准入清单测试通过。`，后面三个测试在打印之后继续执行。

- `test_contract_assessment`：合法记录返回 `{"missing_identity_fields": [], "supervision_complete": True, "contract_issues": [], "admitted": True}`；空回答返回 `["missing_response"]` 且 `admitted False`；缺 `source_id` 返回 `["source_id"]`；`split="holdout"` 返回 `["invalid_split"]`。
- `test_fingerprint_semantics`：两条只有回答不同的记录，`task` 指纹相等、`example` 指纹不等。
- `test_dataset_audit`：`admitted_sample_ids == ["train-4"]`；四行原因分别是 `["evaluation_task_leak"]`、`["duplicate_example", "evaluation_task_leak"]`、`["missing_response"]`、`["duplicate_identity"]`。
- `test_admission_handoff`：`build_sft_training_examples(report, "### Instruction\n")` 的 `sample_id` 列表是 `["train-4"]`。
- `test_training_interface_contract`：输出字段集合恰好是 `{"sample_id", "source_id", "split", "prompt", "response", "provenance"}`；`source_id == "source-e"`、`split == "train"`；三元组与 `admitted_records` 一致。
- `test_rejected_record_does_not_poison_seen_state`：第一条原因 `["missing_source_id"]`，第二条 `decision == "admit"`。
- `test_reused_identity_cannot_reenter_handoff`：在评测集合为空的前提下追加一条验证集副本，`admitted_sample_ids == ["train-1", "train-4"]`；修改原始内容后，输出回答仍是 `["申请、核验、确认。", "核对订单、更新地址、确认结果。"]`，且没有 `source-d`。
- `test_incomplete_supervision_still_reports_leak`：空回答 + 评测任务 → `["missing_response", "evaluation_task_leak"]`，`admitted_records == []`。
- `test_provenance_survives_handoff`：嵌套血缘值保持 `"v1"`，修改原字典不影响输出；`response=123` 抛 `ValueError`。

框架核对单元会返回 `status: passed`、`check_kind: data_interface`、`admitted_rows`、`rejected_rows` 和六项布尔检查；没装 `datasets` 时返回 `skipped`。notebook 没有保存任何执行输出，所以不存在"已跑过的数值"可以引用；判断标准就是上面这些断言和你运行时打印的 ✅。全部测试只在 CPU 上跑内存里的小列表，不加载模型、不下载数据集，因此几秒内就能得到结果。如果你要检查自己的实现，最快的方式是先单跑 `test_fingerprint_semantics` 和 `test_dataset_audit`——前者错通常是把回答混进了任务指纹，后者错通常是 `seen_*` 更新条件或原因顺序写错。
