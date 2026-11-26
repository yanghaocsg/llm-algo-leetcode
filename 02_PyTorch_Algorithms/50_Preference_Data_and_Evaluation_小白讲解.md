# 50. 偏好数据与对齐评测 — 小白讲解

> 对应 notebook：[`50_Preference_Data_and_Evaluation.ipynb`](./50_Preference_Data_and_Evaluation.ipynb)

## 一句话总结
这一节不训练模型，而是回答"你凭什么相信一个偏好优化结果"。`audit_preference_dataset` 审计偏好对的可比性、身份与训练/评测污染（TODO 1），`compute_pairwise_outcomes` 把完成数、胜负平、覆盖率和明确判决率分开统计（TODO 2），`compute_judge_consistency` 用重复判分和交换顺序检查评审器稳定性（TODO 3），`classify_preference_evidence` 按"数据/安全先否决，其余不足复核"给出 `qualified / review / reject`（TODO 4）。

## 为什么需要它：痛点是什么
初学者常把"DPO loss 降了"或"候选胜率 0.7"当成模型变好的证据。但这里有三个坑：训练集和评测集可能只是换了个大小写、多打了几个空格，模型其实在背题；两个回答可能根本不在同一个任务下比较，"优选"反映的是题目难度而不是回答质量；自动评审器可能只是偏爱排在前面或更长的回答，重复问一次就翻脸。

生活类比：班级评优。如果参评同学的申请材料不是同一套题目、评委知道谁是"自己人"、投票顺序还会影响结果，那"得票率 70%"说明不了谁更优秀。偏好评测也是一场投票，必须先保证题目一致、评委独立、计票口径写清楚。

## 核心问题
- 一条合法的偏好样本要同时满足哪些关系？"训练—评测污染"为什么要先规范化再查？
- 覆盖率、明确判决率、非平局胜率分别回答什么问题？平局算不算"没评"？
- 评审一致性为什么取"重复判分"和"换序判分"的较小值？
- 数据/安全失败与样本不足，谁优先？返回原因能不能混在一起？
- 什么组合是 `qualified`，什么只能 `review`，什么直接 `reject`？

## 解决思路：一步步看懂

**第 1 步：偏好样本的最小契约。** `PreferenceSample` 有七个字段：`sample_id`、`split`、`prompt`、`chosen`、`rejected`、`source_id`、`preference_reason`。notebook 的 Step 1 强调四层关系必须同时成立：两个回答来自同一任务/输入/约束（可比性）；选择理由与目标能力一致（语义）；生成来源和判分依据可追溯（身份）；训练、验证、最终评测之间没有规范化后重复的任务输入（隔离）。工具任务还要求动作、观察、终态和验证结论属于同一次执行。任何一层失效，偏好信号就可能混进题目差异、标注噪声或记忆效应。

**第 2 步：TODO 1 —— 污染计数是集合交集。** `normalize_task_input` 只做两件事：`casefold()` 统一大小写，`split()` + `" ".join` 把连续空白压成一个空格。骨架已经分别收集 `train_prompts` 和 `eval_prompts`（`validation`、`eval`、`test` 都算评测侧），参考答案：

```python
contamination_count = len(train_prompts & eval_prompts)
```

用集合交集，而不是逐条 `==`，所以 `"Explain LoRA"` 和 `"  explain   lora  "` 会被认成同一条。`test_preference_data_audit` 的 fixture 正是这样构造的：`train-1` 的 prompt 是 `"Explain LoRA"`，`eval-1` 是 `"  explain   lora  "`，规范化后相交，`contamination_count == 1`。其余计数由骨架完成，但你要能读：`total=3`；`empty_count=1`（`holdout` 那条 `rejected=""` 且来源为空）；`invalid_pair_count=0`；`missing_source_count=1`；`missing_reason_count=1`；`invalid_split_count=1`（`"holdout"` 不在 `{"train","validation","eval","test"}` 里）；`duplicate_id_count=1`（两个 `train-1`）。

**第 3 步：TODO 2 —— 三个比例各管一段。** 设预期数 $n_{expected}$、实际完成 $n_{evaluated}$、候选胜 $n_{candidate}$、基线胜 $n_{baseline}$、平局 $n_{tie}$，则

$$
n_{evaluated}=n_{candidate}+n_{baseline}+n_{tie},\quad
C_{eval}=\frac{n_{evaluated}}{n_{expected}},\quad
R_{decision}=\frac{n_{candidate}+n_{baseline}}{n_{evaluated}},\quad
W_{non\text{-}tie}=\frac{n_{candidate}}{n_{candidate}+n_{baseline}}.
$$

对应四行答案：

```python
decided_pairs = wins + losses
evaluation_coverage = evaluated_pairs / expected_pairs
decision_rate = decided_pairs / evaluated_pairs
decided_win_rate = wins / decided_pairs if decided_pairs else 0.0
```

覆盖率回答"计划评的评完了吗"；明确判决率回答"评完的里面有多少真的分出了胜负"；非平局胜率只在已分胜负的样本里比。平局属于"已完成"，所以进覆盖率和明确判决率的分母，但不进胜率分母——否则评审器一律判平就能刷高覆盖率。没有胜负时胜率记 0，而不是除零或 1。

拿 `test_pairwise_outcome_protocol` 手算：p1 的 0.9 > 0.1 是胜，p2 的 0.2 == 0.2 是平，p3 的 0.4 < 0.5 是负；`expected=4`、`evaluated=3`，所以 `coverage=3/4=0.75`、`decided=2`、`decision_rate=2/3`、`decided_win_rate=1/2=0.5`，`ties=1`。`test_all_ties_keep_evaluation_but_no_decisions` 只有一条平局、预期 2：覆盖率 0.5，明确判决 0、判决率 0、胜率 0，验证"平局仍算完成，但不能制造胜负"。

**第 4 步：TODO 3 —— 逐对比较，不是比总分。** 评审器对每个偏好对给出 `-1 / 0 / 1`（基线更好 / 平 / 候选更好）。第一次判分和重复判分要按同一偏好对顺序对齐；换序那次要先由调用方把结果"还原到原方向"再传进来。参考答案：

```python
repeated_agreements = sum(first == repeated for first, repeated in zip(first_pass, repeated_pass))
swapped_agreements = sum(first == swapped for first, swapped in zip(first_pass, remapped_swapped_pass))
```

然后 `repeat_consistency = repeated_agreements / len(first_pass)`、`swap_consistency = swapped_agreements / len(first_pass)`、`judge_consistency = min(...)`。取最小值是保守做法：两种稳定性只要有一种差，就不能算可靠。`test_judge_consistency` 用 `[1,0,-1,1]`、`[1,0,1,1]`、`[1,0,-1,-1]`：重复判分在第 3 位不一致，换序判分在第 4 位不一致，各 3/4，所以三个值都是 0.75。只比较总体比例（比如"候选胜率都是 50%"）会漏掉逐对翻转。

**第 5 步：TODO 4 —— 状态优先级。** 骨架已经把 `rejection_reasons` 和 `review_reasons` 两个列表算好，参考答案是一行三元表达式：

```python
status = "reject" if rejection_reasons else "review" if review_reasons else "qualified"
reasons = rejection_reasons if status == "reject" else review_reasons
```

数据失败（`data_gate_failed`）或任一安全指标失败（`safety_gate_failed`）直接 `reject`，而且只返回拒绝原因，不把样本不足混进去；否则只要有复核原因就 `review`，返回**全部**复核原因；两者都空才 `qualified`。门槛默认是：`min_win_rate=0.6`、`min_evaluation_coverage=0.8`、`min_evaluated_pairs=3`、`min_decided_pairs=2`、`min_consistency=0.8`。`test_rejection_precedes_review_reasons` 专门用 `data_pass=False`、覆盖率 0.1、胜率 0 的组合，断言结果只有 `{"status": "reject", "reasons": ["data_gate_failed"]}`，不掺 `low_evaluation_coverage` 等复核原因。

**第 6 步：三个状态怎么读。** `test_preference_evidence_gate` 里，`qualified` 记录是 `evaluated=4`、`decided=3`、`coverage=1.0`、`win_rate=2/3`、`judge_consistency=0.9`：4≥3、3≥2、1.0≥0.8、0.667≥0.6、0.9≥0.8，全部过，返回 `{"status": "qualified", "reasons": []}`。同一条记录把 `safety_pass` 改成 `False`，就只有 `["safety_gate_failed"]`。`incomplete` 是 `evaluated=2`、`decided=1`、`coverage=0.25`、`win_rate=0.5`、`consistency=0.7`：五个复核门槛全部不达标，`review` 的原因集合正好是那五个字符串。这就是"评测不足"和"候选不行"的区别：前者补样本/复核，后者修数据/安全。

**第 7 步：可选的数据接口核对。** `check_preference_dataset_interface` 把同一批 `PreferenceSample` 用 `dataclasses.asdict` 转成 `Dataset.from_list`，检查字段 schema 不变、`sample_id` 顺序不变、`split` 过滤出的训练/评测 ID 正确，以及用同一个 `normalize_task_input` 在 Dataset 上算出的交集仍等于本地审计的 `contamination_count == 1`。它只复核数据接口，不对胜率或安全门禁负责。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 偏好对（preference pair） | 同一任务下的优选回答 `chosen` 与非优选回答 `rejected` |
| 规范化 | 统一大小写、压缩空白，防止换写法绕过重复检查 |
| 训练—评测污染 | 同一任务输入同时出现在训练集和评测集 |
| 评测覆盖率 | 实际完成数 ÷ 计划数 |
| 明确判决率 | 分出胜负的样本 ÷ 已完成样本 |
| 非平局胜率 | 候选胜 ÷（候选胜 + 基线胜），不含平局 |
| 判分一致性 | 重复判分与换序判分中，与首次判分一致的比例取较小值 |
| 位置偏差 | 评审器因为回答排在前面/后面而改变判断 |
| `qualified` | 数据、安全和所有质量门槛都通过 |
| `review` | 数据/安全没问题，但样本量、覆盖率、胜率或一致性不足 |
| `reject` | 数据契约失效或安全指标不达标 |

## 小白常见疑问
**Q1：为什么污染检查要规范化，直接比字符串不行吗？**
不行。`"Explain LoRA"` 和 `"  explain   lora  "` 在字节层面不同，但模型看到的是同一个任务。`normalize_task_input` 用 `casefold` 和空白压缩把这类变体折叠掉；`test_preference_data_audit` 就是拿这组变体证明 `contamination_count` 不会漏报。只比较原始字符串，换个大小写就能把泄漏藏起来。

**Q2：平局到底算不算"评过"？**
算完成，不算胜负。它进入 `evaluated_pairs`、覆盖率和明确判决率的分母，但不进入 `decided_win_rate` 的分子/分母。`test_all_ties_keep_evaluation_but_no_decisions` 用一条平局、预期两条，得到覆盖率 0.5，但 `decided_pairs=0`、`decision_rate=0.0`、`decided_win_rate=0.0`。

**Q3：一致性 0.75 没过 0.8，是数据坏了还是评审器坏了？**
只能说明评审器不稳定，不能直接判数据坏。`classify_preference_evidence` 把它归进 `review_reasons` 的 `low_judge_consistency`，和样本不足、覆盖率低并列；只有 `data_pass=False` 或 `safety_pass=False` 才进 `reject`。也就是说，先复核判分协议（换模型、换提示、人工抽检），而不是直接扔掉数据。

**Q4：`qualified` 里胜率 2/3 刚过 0.6，样本又只有 3 条明确判决，可信吗？**
按本节规则它合规，但规则本身也设了 `min_evaluated_pairs=3`、`min_decided_pairs=2` 这种"最低样本量"门槛，目的就是防止 1 胜 0 负也算 100%。真要上线还应扩大样本、看安全分项和任务类别分布；notebook 的 Step 3 表格明确说"通过门槛"只是允许准入，失败案例和评测身份要保留。

**Q5：交换回答顺序后，为什么还要"重映射"？**
因为交换后评审器说的"候选更好"可能指的是换了位置的那一方。调用方必须先把它还原成"原始候选 vs 原始基线"的方向，`compute_judge_consistency` 才能做逐位置比较。函数本身不读 `sample_id`，只要求三次列表非空、等长、取值在 `{-1,0,1}`；长度不匹配直接抛 `ValueError`。

## 动手验证：怎么知道自己做对了
`test_preference_data_audit` 断言 fixture 的八个计数完全等于 `{total:3, empty_count:1, invalid_pair_count:0, missing_source_count:1, missing_reason_count:1, invalid_split_count:1, duplicate_id_count:1, contamination_count:1}`。`test_pairwise_outcome_protocol` 断言 `coverage=0.75`、`decided_pairs=2`、`decision_rate=2/3`、`decided_win_rate=0.5`，并确认重复 `pair_id` 抛 `ValueError`。`test_judge_consistency` 断言三项一致性都是 0.75。`test_preference_evidence_gate` 断言合格样本返回 `{"status":"qualified","reasons":[]}`、安全失败只返回 `["safety_gate_failed"]`，以及不完整样本的五个 `review` 原因集合。`test_invalid_preference_evidence_domains` 检查空 `pair_id`、`NaN` 分数、`decided_pairs > evaluated_pairs` 都会被拒绝。`test_rejection_precedes_review_reasons` 检查拒绝不被复核原因污染。`test_all_ties_keep_evaluation_but_no_decisions` 检查全平局仍计完成但不产生胜负。全部通过后打印 `✅ 偏好数据隔离、成对统计与候选决策测试通过。` notebook 没有保存执行输出，`framework_check` 的 `check_kind: "data_interface"` 和 `checks` 要自己运行确认。
