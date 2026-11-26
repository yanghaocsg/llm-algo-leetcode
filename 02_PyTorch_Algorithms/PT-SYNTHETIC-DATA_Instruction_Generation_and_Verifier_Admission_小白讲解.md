# PT-SYNTHETIC-DATA. 合成指令生成与验证准入 — 小白讲解

> 对应 notebook：[`PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.ipynb`](./PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.ipynb)

## 一句话总结
本节用四个 TODO 搭出一条合成数据准入流水线：先用 `build_generation_identity` 记录"哪份蓝图、哪个生成模型、哪个模板、哪条候选"，再用字符 2-gram 的 Jaccard 相似度查模板复写与同批重复，然后用 `decide_synthetic_admission` 按"拒绝 → 复核 → 接纳"的优先级给出三态结论，最后用 `build_sft_candidate_records` 只把干净记录交给下游。核心结论是：**质量分数高不能抵消污染或验证失败，拒绝优先于复核**。

## 为什么需要它：痛点是什么
初学者常见的做法是"多生成一些、挑分数高的"：既然退款问答已经很多，就让模型再写几十条退款答案；既然回答读起来通顺，就打 0.9 分收下。结果训练集里全是"怎样退款"的轻微改写，真正缺的"未发货订单如何改地址"一条没补上；更糟的是，评测题被改写后混进训练集，离线分数虚高。

生活类比：招聘。蓝图是岗位 JD（要补什么能力），生成身份是简历上的学校、公司、项目（来源可查），相似度查重是"这份简历是不是把现有员工的简历改了个名字"，验证器是笔试，质量分是面试评分。HR 不会因为面试 95 分就忽略"笔试作弊"或"简历造假"——一票否决项优先于加权分。本节就是把这条招聘流程写成代码。

## 核心问题
- 一条合成样本的"生成身份"由哪四项组成，为什么换模板也必须换身份？
- 字符 $n$-gram 的 Jaccard 相似度怎么算？空文本、空参考集、完全相同分别返回什么？
- 相似度和质量分数如何组合成"接纳 / 复核 / 拒绝"，为什么拒绝优先？
- 什么记录才能交给 32 节继续做字段与切分审计？证据怎样随样本复制？
- 为什么字符重合只能当筛查信号，不能当"语义不重复"的证明？

## 解决思路：一步步看懂

**第 1 步：TODO 1——生成身份是四元组，缺一项就报错。** `build_generation_identity(candidate)` 按固定顺序读取 `("blueprint_id", "generator_id", "template_id", "candidate_id")`，每个字段先 `str(... or "").strip()`。骨架已经算好 `identity`，要补的是找出空分量：

```python
missing = [field for field, value in zip(fields, identity) if not value]
if missing:
    raise ValueError(f"生成身份字段缺失：{missing}")
return identity
```

为什么四项都要？蓝图说明"要补什么能力"，生成模型和模板说明"这条样本怎么来的"，候选标识说明"是哪一条"。只换模板（`self-instruct-v1` → `self-instruct-v2`）时，模型没变、蓝图没变，但生成分布可能变了，所以 `identity` 必须不同；测试 `test_generation_identity` 正是断言 `build_generation_identity(original) != build_generation_identity(changed)`，并把 `generator_id` 清空后期望 `ValueError`。容易写错的是只检查 `candidate_id`，或者用 `candidate.get(field)` 的真值判断却忘了 `strip()`——全是空格也算缺失。

**第 2 步：TODO 2——Jaccard 相似度与"两个空集算相同"。** `character_ngrams(text, n=2)` 先做 `normalize_text`（`casefold` 转小写 + `split` 合并空白），再滑窗取连续 $n$ 个字符。比如归一化后的"三步退款"在 $n=2$ 时得到 `{"三步", "步退", "退款"}`；文本长度小于 $n$ 时把整段作为一个片段；空文本返回空集。

`max_reference_similarity(instruction, references, n=2)` 对每条参考算

$$
S(q,d)=\frac{|N_n(q)\cap N_n(d)|}{|N_n(q)\cup N_n(d)|},
$$

其中 $N_n(\cdot)$ 是字符片段集合，$q$ 是候选指令，$d$ 是某条参考指令。要补的一行是：

```python
similarity = len(candidate_ngrams & reference_ngrams) / len(union) if union else 1.0
```

然后取所有参考中的最大值，`max(similarities, default=0.0)` 让空参考集自动返回 0。三个边界分别对应三种语义：**没有参考** → 最大相似度 0（无从判定重复）；**两份都空** → 相似度 1（完全相同）；**精确重复** → 交集等于并集 → 1。测试 `test_similarity_boundaries` 断言 `max_reference_similarity(instruction, []) == 0.0`、`[instruction] == 1.0`、与"计算两个矩阵的乘积"的相似度落在 $[0, 0.5)$，并检查 `n=0` 和 `n=True` 都抛 `ValueError`。`n=True` 这一条容易漏：`type(n) is not int` 是关键，因为 `bool` 是 `int` 的子类，写成 `isinstance(n, int)` 就会放行。

**第 3 步：TODO 3——先收集全部原因，再按优先级定性。** `decide_synthetic_admission` 的默认参数是 `similarity_review_threshold=0.72`、`quality_admit_threshold=0.80`、`quality_review_threshold=0.60`，函数开头先校验：至少一个具名验证器；相似度阈值在 $[0,1]$；质量阈值满足 $0\le\tau_r\le\tau_a\le1$。测试传 `quality_admit_threshold=0.5, quality_review_threshold=0.8` 时期望 `ValueError`，因为顺序反了。

函数会把所有问题追加到 `reasons`，包括：

| 原因字符串 | 触发条件 | 归类 |
|---|---|---|
| `missing_instruction` / `missing_response` | 归一化后为空 | 拒绝 |
| `evaluation_task_leak` | 指令归一化后与评测任务完全重合 | 拒绝 |
| `required_verifier_missing:<name>` | 必需验证器没有记录或没有 `version` | 拒绝 |
| `required_verifier_failed:<name>` | `passed is not True` | 拒绝 |
| `required_verifier_reason_missing:<name>` | 失败但没写 `failure_reason` | 拒绝 |
| `quality_below_minimum` | 质量 < 0.60 | 拒绝 |
| `high_training_similarity` / `high_batch_similarity` | 相似度 ≥ 0.72 | 复核 |
| `quality_needs_review` | 0.60 ≤ 质量 < 0.80 | 复核 |

骨架已经算出 `reject_reasons` 列表，TODO 只需一行三态：

```python
decision = "reject" if reject_reasons else "review" if reasons else "admit"
```

语义是：**只要有一个拒绝原因就拒绝；没有拒绝原因但有复核原因就复核；一个问题都没有才接纳**。这就是"拒绝优先"：`test_reject_precedes_review` 构造的候选同时有评测污染、format 验证失败、训练近重复、质量 0.70，最终 `decision == "reject"`，而 `reasons` 仍然是四项全保留——不能因为已经决定拒绝就把其他原因删掉，否则下游不知道该修几处。

质量边界要用 `test_quality_threshold_boundaries` 记牢：0.59 → `reject`，0.60 → `review`（恰好等于复核线进复核），0.799 → `review`，0.80 → `admit`（恰好等于接纳线进接纳）。NaN、−0.1、1.1 都抛 `ValueError`，因为 `0.0 <= quality_score <= 1.0` 对 NaN 为假。

另一个容易踩的坑是验证器"缺失"与"失败"的区分：测试 `test_empty_content_and_verifier_policy` 传入 `verifier_results={"format": {"passed": True}}`（没有 `version`），期望两条都是 `required_verifier_missing:*`，而不是 failed。没有版本号就无法回查是哪一版验证器给的结论，所以按缺失处理。

**第 4 步：TODO 4——交接要求"决定与原因同时干净"。** `build_sft_candidate_records` 只保留能继续审计的行：

```python
handoff_allowed = row["decision"] == "admit" and not row["reasons"]
if not handoff_allowed:
    continue
```

为什么要同时检查 `reasons`？因为可能出现"写着 admit 但原因列表非空"的矛盾记录（测试里用 `{**admitted, "decision": "review"}` 和 `{**row, "reasons": ["evaluation_task_leak"]}` 两种方式制造）。只看 `decision` 会把矛盾记录放过去；只看 `reasons` 又会把 review 但暂时没写原因的记录放过去。两条都满足才是真正干净。

通过后生成一条 SFT 记录，关键字段包括 `sample_id`、`source_id`、`split="train"`、`instruction`、`input=""`、`response`、`admission_decision`、`verifier_versions` 和嵌套的 `provenance`。`source_id` 用 `json.dumps(list(generation_identity), separators=(",", ":"))` 而不是字符串拼接，是为了保留分量边界：测试把 `blueprint_id` 设成 `"refund|v1"`，用 `|` 拼接再切分就会错位，而 JSON 解析后第一个分量仍是 `"refund|v1"`。`provenance` 里的证据用 `deepcopy`，测试在生成记录后把原始 `row["verifier_evidence"]["answer"]["version"]` 改成 `"changed"`，记录里仍是 `"answer-v2"`——证据必须独立，不能和上游共享可变对象。

同一批里 `sample_id` 重复会抛 `ValueError`，不是静默去重：重复标识意味着来源记录本身有问题，应该回查生成流程。

**第 5 步（可选）：框架一致性核对。** 如果环境里有 `datasets`，`check_synthetic_data_interface` 会把三态结果转成 `Dataset.from_list`，用 `filter(lambda row: row["decision"] == "admit")` 看接纳集合是否与本地列表一致，再用 `to_list()` 核对行顺序、嵌套字段和 SFT 交接字段；任一检查失败抛 `AssertionError`，全部通过返回 `status: "passed"` 并记录 `datasets` 版本。没有安装 `datasets` 时返回 `{"status": "skipped", "reason": "请使用 PT-CPU 环境运行本单元"}`，这是环境提示，不是实现错误。这一节只核对数据接口，准入决定仍由 Step 4 产生。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 任务蓝图（blueprint） | 生成前写好的"要补什么能力、输入范围、成功条件" |
| 生成身份 | 蓝图 + 生成模型 + 提示模板 + 候选标识四元组 |
| 字符 $n$-gram | 连续 $n$ 个字符组成的片段，如"三步退款"的 2-gram |
| Jaccard 相似度 | 交集大小 ÷ 并集大小，衡量两个片段集合的表面重合 |
| 新颖度（novelty） | $1-S(z,D)$，越大说明越不像已有指令 |
| 验证器（verifier） | 按任务规则检查候选的组件，如格式检查、答案核对 |
| 三态准入 | `admit` / `review` / `reject`，拒绝优先于复核 |
| 复核线 $\tau_r$ / 接纳线 $\tau_a$ | 质量低于前者拒绝、介于两者复核、达到后者才可能接纳 |
| 评测任务重合 | 候选指令与固定评测题规范化后完全相同，直接拒绝 |
| 来源（provenance） | 随样本保存的生成身份、验证结果、质量与相似度记录 |

## 小白常见疑问
**Q1：为什么用字符 2-gram，不用词向量或语义相似度？**
本节的目的是识别"模板复写"这种表面重复，字符片段足够便宜、无需模型、结果可解释；notebook 也明确说"字符重合是筛查信号，改写后的语义重合还需语义检索或人工检查"。用字符片段不会把"退款"和"退货"判成相似，这是它的局限，不是 bug。

**Q2：质量 0.70 为什么是复核而不是拒绝？0.60 又为什么不是拒绝？**
阈值定义是 $0\le\tau_r\le\tau_a\le1$，$\tau_r=0.60$ 是"最低可接受线"，低于它才拒绝；等于 0.60 落在 `quality_review_threshold <= score < quality_admit_threshold` 区间，进复核。0.70 同样在 $[0.60, 0.80)$ 内，所以是复核。测试把四个边界值逐个钉死，就是为了防止把 `<` 写成 `<=` 造成 0.60 被误拒。

**Q3：一条候选同时污染评测集、验证失败、又和训练集很像，为什么只返回一个 `decision`，却保留四条 `reasons`？**
`decision` 是给流水线用的动作，只能有一个；`reasons` 是给人看的修复清单，必须完整。`test_reject_precedes_review` 断言 `set(row["reasons"])` 恰好等于那四项，说明实现不能"发现第一个问题就提前 return"。

**Q4：`build_sft_candidate_records` 收到 `decision="admit"` 但 `reasons` 非空，为什么要丢弃？**
这是自相矛盾的记录，说明上游判定逻辑或数据被改过。允许它进入训练集，等于把"已知有问题"的样本洗白。测试专门用 `inconsistent = {**row, "reasons": ["evaluation_task_leak"]}` 断言返回空列表。

**Q5：生成身份里已经有 `candidate_id`，为什么还要保存蓝图和模板版本？**
`candidate_id` 只在当前集合里唯一，换一批生成可能复用；蓝图和模板决定"这条样本代表什么能力、用什么方式生成"。没有它们，回查时只能看到一条孤立文本，无法判断它是否真的补上了缺口，也无法在模板升级后区分新旧来源。

## 动手验证：怎么知道自己做对了
测试区先调用前五个测试，打印 `✅ 合成候选生成身份、相似度、验证器与三态准入测试通过。`，再调用后三个测试。注意打印语句在最后三个测试**之前**，所以看到这行字并不代表全部跑完；要以最后没有异常为准。TODO 没填时会在第一个测试里抛 `NotImplementedError`。

- `test_generation_identity`：换 `template_id` 后身份必须不同；`generator_id=""` 抛 `ValueError`。
- `test_similarity_boundaries`：空参考 0.0、完全重复 1.0、不同任务落在 $[0,0.5)$、`n=0` 抛 `ValueError`。
- `test_admission_decisions`：干净候选 `admit` 且 `reasons == []`；训练近重复和同批近重复都是 `review` 且带对应原因；评测污染 `reject`；`answer` 验证失败时 `reasons == ["required_verifier_failed:answer"]`；质量 0.70 → `review`；`quality_admit_threshold=0.5, quality_review_threshold=0.8` 抛 `ValueError`。
- `test_reject_precedes_review`：四项原因同时存在时 `decision == "reject"`，且 `reasons` 集合恰好是 `{"evaluation_task_leak", "required_verifier_failed:format", "high_training_similarity", "quality_needs_review"}`。
- `test_sft_candidate_handoff`：只交接 candidate-1；`source_id` 解析后等于 `["refund-v1", "teacher-v2", "self-instruct-v1", "candidate-1"]`；`split == "train"`；`verifier_versions == {"format": "format-v1", "answer": "answer-v2"}`；重复接纳同一条抛 `ValueError`。
- `test_empty_content_and_verifier_policy`：instruction/response 全空格 → `reject` 且带 `missing_*`；`template_id=None` 抛错；空验证器列表抛错；验证器无版本时两条 `required_verifier_missing:*`；`n=True` 抛错。
- `test_quality_threshold_boundaries`：0.59/0.60/0.799/0.80 分别对应 `reject`/`review`/`review`/`admit`；NaN、−0.1、1.1 抛 `ValueError`。
- `test_handoff_evidence_and_consistency`：`provenance` 保留四元组和原始验证结果；修改上游后记录不变；`admit` 但带 `evaluation_task_leak` 的记录被丢弃；`blueprint_id="refund|v1"` 经 JSON 往返后仍完整。

notebook 没有保存执行输出，所以不要引用"admitted_rows 等于几"之类的数字；可选框架单元的实际返回取决于环境是否装了 `datasets`，以 `status` 和 `checks` 字典为准。
