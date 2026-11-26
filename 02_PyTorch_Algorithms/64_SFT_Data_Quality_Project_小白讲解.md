# 64. SFT 数据质量项目 — 小白讲解

> 对应 notebook：[`64_SFT_Data_Quality_Project.ipynb`](./64_SFT_Data_Quality_Project.ipynb)

## 一句话总结
本节把"这份 SFT 数据能不能拿来训练"做成一次可复查的审计：固定数据提交、分词器、对话模板和最大长度，先统计空指令、空回答、零监督词元、重复、截断和训练/验证任务重合，再清洗、切分、重新检查。三个 TODO 分别实现"准入三态判断""清洗后的版本能否替换旧版本""多个数据版本里选哪个交付"，最后在 CPU 上跑真实数据审计并保存数据清单。归档的 v1 结果是：512 条原始记录移除 1 条空回答后保留 511 条，448 条训练、63 条评测，重复率 0，截断率 1.17%，跨切分重合 0，决策 `accept`。

## 为什么需要它：痛点是什么
初学者检查 SFT 数据时通常只看两件事：文件能不能加载、每条样本有没有 `output` 字段。但 notebook 开头给了两个反例：一条空回答在追加 EOS 之后，仍然会得到**非零的监督词元数**；同一个问题换一个回答，就不再是"完整样本重复"，却仍然是同一个输入任务，可能同时落进训练集和验证集。只检查"能加载、有标签"，这些记录会一路进入训练，最后把评测分数抬高或把损失信号带偏。

生活类比：餐厅验收食材。只看"包装完好、标签齐全"不够，还要看有没有空箱（空回答）、有没有把同一批货换个箱子重复入库（换回答的重复任务）、有没有超过冷柜容量被切掉一半（截断），以及员工餐和顾客餐是不是从同一锅里盛出来的（训练/验证泄漏）。验收通过后要写一张入库单，后面的人按单子取货，而不是凭记忆。

## 核心问题
- 一条空回答为什么可能仍有监督词元？什么情况下连 EOS 都保不住？
- 样本级重复和任务级重合差在哪？为什么去重后仍要做切分泄漏检查？
- `reject`、`tune`、`accept` 三态的优先级是什么？两类问题同时出现时保留哪些原因？
- 清洗后的版本要满足哪些条件，才能替换掉原来未通过的版本？
- 多个候选都 `accept` 时，为什么先筛"清单完整且无失败项"，再按样本量排序？
- 真实数据审计的 `prepare → collect → close` 各做什么？归档的 512→511、448/63 应该怎么读？

## 解决思路：一步步看懂

**第 1 步：把数据从"原始记录"变成"固定快照"。** Step 1 的表把流程分成四段：记录来源（固定版本与原始行号）→ 原始检查（模板、分词器、长度；统计空内容、监督数量、重复、截断）→ 清洗与切分（去重后按固定种子打乱，分训练和验证）→ 重新检查（监督是否可用、任务是否重合、比例是否达标）。最终产物叫"数据快照"，清单里保存来源提交、原始行号、样本编号、分词配置和两侧成员。关键点是**原始检查和清洗后检查分开保存**：这样既知道删掉了什么，也知道最终还剩哪些问题。

**第 2 步：监督词元是按"实际序列"数的，不是按字段数的。** notebook 的做法是：提示和回答分别分词，给回答追加结束标记 EOS，拼接后从右侧截断到最大长度，再用 PAD 补齐短序列；提示位置和填充位置不参与损失，保留下来的回答与 EOS 参与监督。由此产生两个反直觉现象：

1. 空回答 → 回答部分没有词元，但 EOS 还在，标签里可能仍有 1 个监督词元。所以"监督词元数 > 0"不能证明回答非空，必须单独检查 `empty_response_count`。
2. 提示太长、占满整个窗口 → 回答和 EOS 都被截掉，监督词元变成 0。这时即使原始记录有回答，这条样本也无法按当前长度训练。

这两种情况说明：**数据质量检查必须和分词器、模板、最大长度绑定**。换了对话模板或分词方式，就要按新序列重新统计，不能复用旧比例。

**第 3 步：样本编号和任务编号是两把不同的尺子。** 样本编号由内容摘要生成：合并连续空白、保留大小写，再计算哈希；完整样本比较"指令 + 上下文 + 回答"三者。任务编号只比较"指令 + 上下文"，不看回答。于是"同一问题换一个回答"会产生两个不同的样本编号，却产生相同的任务编号。完整样本重复率的分母是总条数，第一份不计重复；任务重合数则是训练侧和验证侧任务哈希集合的交集大小。notebook 特别说明：这里只检查规范化文本完全相同的输入，改写和语义近似需要额外筛查。

**第 4 步：TODO 1——三态判断，阻断项优先。**

```python
decision = "reject" if hard_failure_names else "tune" if soft_failure_names else "accept"
```

这一行是嵌套条件表达式，读法是：只要有硬失败就 `reject`；否则只要有软失败就 `tune`；两者都没有才 `accept`。硬失败包括 `empty_instruction`、`empty_response`、`zero_supervision`、`split_leakage`、`missing_evaluation`；软失败只有 `duplicate_ratio` 和 `truncation_ratio`。为什么这样分？空内容、零监督、任务重合和验证样本不足会直接让训练信号或评测结论失效，必须先修数据；重复和截断可以通过去重或调整长度策略解决，所以给一个 `tune` 的中间态。

函数返回时同时保留 `hard_failures` 和 `soft_failures` 两个列表，而不是只返回第一个原因。测试 `test_all_failures_and_threshold_boundaries` 故意同时塞入 `empty_response_count=1` 和 `duplicate_ratio=0.03`，断言决策是 `reject`，且 `hard_failures == ["empty_response"]`、`soft_failures == ["duplicate_ratio"]`。如果只保留一个原因，修复者可能只处理空回答，漏掉重复问题。

阈值比较用的是严格大于：`audit["duplicate_ratio"] > limits["maximum_duplicate_ratio"]`。所以比例恰好等于上限算通过。同一个测试用 `evaluation_count=2`、`duplicate_ratio=0.02`、`truncation_ratio=0.10` 构造边界，断言 `accept`。这也解释了为什么 `_validate_admission_inputs` 要先把 `NaN`、`inf`、负数、大于 1 的比例挡掉：`NaN > 0.02` 是 `False`，如果不校验，一个非法的 `NaN` 会被当成"没超限"。

**第 5 步：TODO 2——清洗后的版本不是"看起来更好"就能替换。**

```python
identity_compatible = same_lineage and same_rules
risks_repaired = bool(resolved_failures) and not after_failures
eligible = (
    identity_compatible and risks_repaired and evaluation_preserved
    and retained_ratio >= minimum_retained_ratio and stage_transition
)
```

逐项解释：

- `same_lineage`：清洗前后的 `source_revision` 相同，不能拿另一批来源的数据冒充"修复版"。
- `same_rules`：`rule_version`、`tokenizer_revision`、`max_length` 都相同。换了分词器或长度，指标就不可比，属于新实验而不是修复。
- `resolved_failures = before_failures - after_failures`：旧版本有、新版本没有的问题；`introduced_failures = after_failures - before_failures`：新版本新出现的问题。
- `risks_repaired = bool(resolved_failures) and not after_failures`：必须**确实解决过**至少一个旧问题，而且新版本不能剩任何问题。`bool(resolved_failures)` 不能省：如果旧版本本来就是 `accept`，`resolved_failures` 为空，此时不该被当成"修复成功"。
- `evaluation_preserved`：`after["evaluation_count"] >= before["evaluation_count"]`，清洗不能把评测集越洗越小。
- `retained_ratio >= minimum_retained_ratio`：保留率要达标。
- `stage_transition`：`before["decision"] != "accept" and after["decision"] == "accept"`，只允许"未通过 → 通过"的方向。

测试里的手算：before 有 100 条、20 条评测、`decision="reject"`、硬失败 `split_leakage`；after 有 92 条、20 条评测、`decision="accept"`、无失败。`92 / 100 = 0.92 >= 0.90`，所以 `eligible` 为真。把 after 改成 89 条，`0.89 < 0.90`，不通过；把 `evaluation_count` 改成 19，`evaluation_preserved` 为假；把 `source_revision` 改成 `source-b`，`same_lineage` 为假；把硬失败改成 `zero_supervision`，`risks_repaired` 为假（新版本仍有失败）。`test_soft_only_cleaning_can_advance` 再补一个"旧版本只有软失败"的案例：before 是 `tune`，软失败 `duplicate_ratio`，after 是 `accept`、90 条、无失败，`0.90 >= 0.9`，也能晋级；但换 `tokenizer-b` 或把 `max_length` 从 256 改成 512 就不行。

**第 6 步：TODO 3——先筛资格，再排序，样本量不能抵消失败。**

```python
failures = candidate.get("hard_failures", []) + candidate.get("soft_failures", [])
handoff_allowed = (
    candidate["decision"] == "accept"
    and candidate["manifest_complete"]
    and not failures
)
```

`decision`、清单完整性、失败列表三项必须同时成立。注意 `not failures` 检查的是候选自己声明的失败列表，而不是重新跑一遍审计；这是"交接时的资格检查"，真正的审计在 Step 5。筛选完才进入骨架的排序：

```python
return sorted(eligible, key=lambda item: (-item["sample_count"], item["revision"]))[0]
```

先按样本量降序，样本量相同再按版本标识升序。为什么必须先筛后排？如果先按样本量取最大，再检查资格，一个 130 条但清单不完整的候选会把 90 条的合格版本挤掉。`test_training_snapshot_selection` 用 v1（90 条、accept、完整）、v2（90 条、accept、完整）、v3（120 条、tune、完整）验证：v3 虽然更大，但 `decision` 不是 `accept`，最终选 v1；v1 和 v2 同分时按版本标识升序选 v1。`test_handoff_excludes_inconsistent_and_incomplete_candidates` 再加两个更大的干扰项：v2 有 120 条但带 `soft_failures`，v3 有 130 条但 `manifest_complete=False`，无论输入顺序如何都选 v1；如果池子里只有这两个坏候选，函数抛 `ValueError`。

骨架还统一校验了候选的身份：`revision` 必须是非空字符串，重复的 `revision` 直接抛 `ValueError`（否则交接时无法唯一指认），`manifest_complete` 必须是布尔值，`sample_count` 必须是正整数。这些检查放在 TODO 之前，所以不要为了让测试通过而在 TODO 里重复写一遍。

**第 7 步：Step 5 的真实数据审计——三个阶段、三种职责。** 5.1 的 `RUN_STAGE` 只能取 `prepare`、`collect`、`close`：

- `prepare`：只读合同与依赖状态，不写正式资产；`RUN_LABEL` 允许是占位值 `replace_with_unique_label`。
- `collect`：检查 `datasets`、`transformers`、`huggingface_hub` 是否可导入，拒绝覆盖同名结果，然后读取固定数据与分词器、执行审计、保存原始证据。
- `close`：要求本次 `RESULT_PATH` 已存在，重新读取并校验，再生成数据清单与登记回执。

`RUN_LABEL` 的正则是 `[A-Za-z0-9][A-Za-z0-9_-]{0,95}`，只允许字母、数字、下划线和连字符，最多 96 个字符；`collect` 或 `close` 阶段如果还是占位标签就报错。实验从 `benchmarks/project_registry/part02.json` 里找 `semantic_id == "PT-SFT-DATA-QUALITY"` 的项目条目，再读它的 `default_experiment_contract`。合同里固定了 `DATASET_ID`、`DATASET_REVISION`、`MODEL_ID`（分词器）、`SAMPLE_COUNT`、`TRAIN_COUNT`、`MAX_LENGTH`、`SEED`、`LIMITS`，并有一段自检：

```python
if not (0 < TRAIN_COUNT < SAMPLE_COUNT and MAX_LENGTH >= 8
        and 0 < LIMITS["minimum_evaluation_count"] <= SAMPLE_COUNT - TRAIN_COUNT
        and all(0 <= LIMITS[key] <= 1 for key in (...))):
    raise ValueError("固定合同中的规模或门槛无效")
```

这段自检保证训练集至少留 1 条、评测下限能塞进剩余样本、长度至少为 8、比例阈值都在 $[0,1]$。`apply_data_admission_limits` 是题目函数的"真实数据版"：同样把审计事实分成硬/软两组，但返回 `hard_gate_passed`、`soft_gate_passed` 和两个失败列表，决策顺序仍是 `reject` 优先、其次 `tune`、最后 `accept`。

**第 8 步：`collect` 到底做了什么。** 它先用 `HfApi().dataset_info(...).sha` 和 `HfApi().model_info(...).sha` 把请求版本解析成不可变 commit，再 `load_dataset(DATASET_ID, split="train[:SAMPLE_COUNT]", revision=dataset_commit)`，只取固定条数。分词器也只读 tokenizer，不加载模型权重，所以整段可以在 CPU 上跑。

清洗循环逐条读取 `instruction`、`input`、`output`，分别做 `normalized_text`（合并空白等规范化），再用 `stable_text_hash((instruction, context, response))` 生成样本编号。跳过条件有三个：`not instruction`、`not response`、`sample_id in seen_sample_ids`。也就是说，清洗规则是"丢掉空指令、空回答和完整样本重复"，与记录里的 `rule` 字段 `drop_empty_instruction_response_and_exact_duplicates` 一致。

然后 `random.Random(SEED).shuffle(accepted_rows)` 固定打乱，计算

```python
cleaned_train_count = min(
    TRAIN_COUNT, len(cleaned_records) - int(LIMITS["minimum_evaluation_count"]),
)
```

即训练条数不能挤占最低评测数量；如果结果小于等于 0，就抛 `RuntimeError("清洗后样本不足，无法同时保留训练样本和最低数量的评测样本")`。接着用 `cleaned_train_count` 重新审计，保存 `raw_audit` 和 `audit` 两份事实，以及 `cleaning.removed_count`、`retained_ratio`、训练/评测的源索引、样本编号和哈希。最后 `validate_sft_admission_record` 复核合同与记录，`publish_json_record` 原子写入。

`close` 用 `read_sft_admission_record` 重新读取 JSON，按合同核对运行标签、来源提交、序列协议、计数、比例、切分成员和哈希，再从保存的事实复算决策；只有 `close` 写登记回执，只有 `accept` 才生成下游清单。回执绑定证据与清单的 SHA256，同名不同内容会停止，避免"改了文件还说是同一版"。

**第 9 步：归档结果怎么读。** 5.6 记录的是 v1 协议（手工指令模板 + 分段分词协议）的 1.5B 清单，`default` 标签为 `alpaca_qwen25_15b_v1`：

| 观察项 | 归档证据 | 读法 |
|---|---|---|
| 原始规模 | 512 条原始记录 | 固定 `SAMPLE_COUNT`，不是"越多越好" |
| 清洗 | 移除 1 条空回答后保留 511 条 | 511 = 512 − 1，说明只删了空回答 |
| 切分 | 448 条训练、63 条评测 | 448 + 63 = 511，成员没有丢 |
| 泄漏 | 跨切分任务重合 0 | 训练和评测没有同一输入任务 |
| 重复与长度 | 重复率 0、截断率 1.17% | 完整样本无重复，少量样本被长度截断 |
| 决策 | `accept` | 允许交给 65 的 1.5B QLoRA 容量分支 |

这里的 `accept` 只说明**数据准入通过**，不说明模型训练后一定更好。notebook 明确写了：模型训练后的质量在后续项目中评测；63 节继续读取已归档的 0.5B 清单，65 节容量分支读取 1.5B 清单，二者不能混为同一比较身份。v2 合同的真实采集结果尚未登记，所以不要拿 v1 的数字去填 v2 的表。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| SFT | 监督微调：用"提示 + 示范回答"训练模型模仿回答 |
| 监督词元 | 真正参与损失计算的回答词元和 EOS；提示与 PAD 不算 |
| 零监督 | 截断后一个回答词元都不剩，这条序列无法训练 |
| 样本编号 | 指令 + 上下文 + 回答的规范化哈希，用来抓完整重复 |
| 任务编号 | 只看指令 + 上下文，用来抓"换回答的同一问题" |
| 切分泄漏 | 同一任务同时出现在训练侧和验证侧，评测分数会虚高 |
| 重复率 | 重复条数 ÷ 总条数，第一份不计重复 |
| 截断率 | 完整序列超过长度上限的样本占比 |
| `reject` / `tune` / `accept` | 必须先修阻断项 / 清洗或调长度后重查 / 通过并冻结快照 |
| 数据快照 | 固定来源、成员、分词配置和哈希的一批训练/评测数据 |
| 保留率 | 清洗后条数 ÷ 清洗前条数，修复版本不能无限删数据 |
| `prepare` / `collect` / `close` | 只检查 / 采集并保存证据 / 复核并交接登记 |

## 小白常见疑问

**Q1：空回答为什么还能有监督词元？直接看 `output` 是否为空不就行了？**
因为监督是在"拼接后的序列"上数的：回答后面还追加了 EOS，即使回答本身是空串，EOS 仍可能落在窗口内并参与损失。所以 `zero_supervision_count` 和 `empty_response_count` 是两个独立检查，前者管"截断后还剩不剩监督"，后者管"示范回答是不是空的"。notebook 的表把两者分开列，并注明"EOS 不能补足缺失的示范回答"。

**Q2：完整样本重复率是 0，为什么还要检查任务重合？**
因为"同一问题换一个回答"会被样本编号判为两条不同样本，但它们的任务编号相同。`test_data_admission_priority` 里 `split_leakage_count=1` 直接触发 `reject`，不看重复率。若只做完整样本去重，训练集和验证集会共享同一输入任务，验证损失就不能代表泛化。

**Q3：重复率正好等于上限为什么算通过？清洗后 92 条能过、89 条为什么不过？**
两个阈值都允许取等号，但方向不同：重复率用 `>` 判断，所以 `0.02` 正好等于上限时不算超限，`test_all_failures_and_threshold_boundaries` 断言 `accept`；保留率用 `retained_ratio >= minimum_retained_ratio` 判断，`92/100 = 0.92 >= 0.90` 能过，`89/100 = 0.89 < 0.90` 不能过，`test_cleaned_snapshot_transition` 正是用这三个数验证边界。门槛不是"删得越少越好"，而是"修复问题时不能把数据删到不足以训练"。

**Q4：两个候选都是 `accept`、都是 90 条，为什么选 v1 不选 v2？**
排序键是 `(-item["sample_count"], item["revision"])`，样本量相同就按版本标识升序，`"v1" < "v2"`，所以选 v1。这是确定性选择：同样的输入无论顺序如何都得到同一个版本，方便交接和复现。如果两个候选的 `revision` 相同，函数会先抛 `ValueError`，因为无法唯一指认。

**Q5：`prepare` 阶段能在 BASE-CPU 上跑，为什么 `collect` 不行？**
`prepare` 只读项目注册表和合同 JSON、检查标签格式，不需要 `datasets`、`transformers`、`huggingface_hub`；代码只在 `RUN_STAGE == "collect"` 时检查这三个包是否可导入。`collect` 要下载数据、解析 commit、加载分词器，所以需要 `PT-SFT-CUDA` 那份数据与分词依赖（但仍然不加载模型权重、不调用 GPU）。`close` 只读已有 JSON，也可以在 CPU 上做。

## 动手验证：怎么知道自己做对了

notebook 的测试单元先直接调用前三个测试，打印 `所有测试通过！`；随后又定义并调用后三个测试。没有保存执行输出，所以下面以代码与断言为准。TODO 没填时，`decide_data_admission`、`check_cleaned_snapshot_transition`、`select_training_snapshot` 会分别抛 `NotImplementedError("TODO 1：判断数据准入状态")` 等异常。

- `test_data_admission_priority`：干净 fixture（`evaluation_count=16`、`duplicate_ratio=0.01`、`truncation_ratio=0.05`）必须返回 `("accept", "freeze_training_snapshot")`；`duplicate_ratio=0.03` 时 `next_action == "clean_and_reaudit"`；`zero_supervision_count=1` 或 `split_leakage_count=1` 时 `decision == "reject"`。
- `test_cleaned_snapshot_transition`：before 100 条 / 20 评测 / `reject` / `split_leakage`，after 92 条 / 20 评测 / `accept` / 无失败，`0.90` 门槛下 `eligible` 为真；换成 `source-b`、`evaluation_count=19`、`sample_count=89` 或新硬失败 `zero_supervision`，全部为假。
- `test_training_snapshot_selection`：v1/v2 都是 90 条 `accept` 且清单完整，v3 是 120 条 `tune`，最终 `selected["revision"] == "v1"`；只有 `reject` 候选时抛 `ValueError`。
- `test_soft_only_cleaning_can_advance`：before 是 `tune` + 软失败 `duplicate_ratio`、100 条 / 20 评测，after 是 `accept`、90 条、无失败，`0.9` 门槛下 `eligible` 为真；换 `tokenizer-b`、`max_length=512` 或新增 `truncation_ratio` 都为假；`sample_count=0` 和 `minimum_retained_ratio=NaN` 抛 `ValueError`。
- `test_all_failures_and_threshold_boundaries`：边界值 `evaluation_count=2`、`duplicate_ratio=0.02`、`truncation_ratio=0.10` 必须 `accept`；同时有 `empty_response_count=1` 和 `duplicate_ratio=0.03` 时必须 `reject`，且硬失败列只有 `["empty_response"]`、软失败列只有 `["duplicate_ratio"]`；比例取 `NaN`、`inf`、`-0.1`、`1.1`，以及 `evaluation_count=1.5`，都必须抛 `ValueError`。
- `test_handoff_excludes_inconsistent_and_incomplete_candidates`：v1（90 条、accept、完整）、v2（120 条、带软失败）、v3（130 条、清单不完整）混在一起，正序和逆序都必须选 v1；只剩坏候选时抛 `ValueError`；同一个 `revision` 出现两次也抛 `ValueError`。

真实数据部分没有单元测试，验证方式是按 `prepare → collect → close` 跑通并检查 JSON。`collect` 成功后打印 `{"result_path": ..., "decision": ...}`；`close` 成功后打印登记回执。重点核对四个字段：`decision` 是否为 `accept`、`raw_audit.sample_count` 与 `audit.sample_count` 的差值是否等于 `cleaning.removed_count`、`audit.evaluation_count` 是否达到合同下限、`train_sample_ids_hash` / `evaluation_sample_ids_hash` 是否与清单一致。归档 v1 的数字是 512 → 511、448 + 63、重复率 0、截断率 1.17%、泄漏 0；如果你复跑得到不同数字，先确认是不是换了数据提交、分词器版本或 `MAX_LENGTH`，而不是直接怀疑实现。
