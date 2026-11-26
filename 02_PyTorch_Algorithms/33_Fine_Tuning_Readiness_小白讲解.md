# 33. 后训练准入与阶段选择 — 小白讲解

> 对应 notebook：[`33_Fine_Tuning_Readiness.ipynb`](./33_Fine_Tuning_Readiness.ipynb)

## 一句话总结
本节回答“现在到底该不该微调、该进哪个训练阶段”。三个函数逐层把关：`assess_problem_evidence` 先筛出复测充分、失败率达到门槛的切片并检查评测隔离与基线覆盖；`assess_parameter_update_need` 扣掉已经被提示、RAG 或工具解决的切片；`select_next_training_stage` 再按“表示缺口还是行为缺口”和“反馈形态”选择持续预训练、SFT、离线偏好或在线强化学习，并逐项检查所需输入是否齐全。

## 为什么需要它：痛点是什么
初学者常见的反应是“模型效果不好就上 SFT/RLHF”，但很多失败根本不需要改参数：可能是提示没写清、知识过期需要 RAG、算术需要计算器、业务流程需要调 API。另一方面，“选 PPO 还是 GRPO”“用模型打分还是规则验证器”“基线是指令模型还是自己 SFT 的模型”经常被混成一团，导致数据还没准备好就开训。

生活类比：去医院看病。先确认症状是否稳定复现（复测次数、失败率），再排除“多喝水、换个姿势”就能解决的情况（提示/RAG/工具），最后才决定挂哪个科（持续预训练、SFT、离线偏好、在线 RL）。挂号时还要带齐材料（语料版本、示范版本、基线清单、预算），缺一项就只能先回去补。

## 核心问题
- 怎样判断一个失败切片是“可重复失败”，而不是偶发波动或样本不足？
- 提示、RAG、工具这些非参数方案通过后，怎样把对应切片从“需要改参数”的集合里扣掉？
- 领域语料、示范回答、偏好对、单条二元反馈分别对应哪个训练阶段？
- 为什么“奖励来源”和“PPO/GRPO/RLOO 更新方法”是两个独立问题？
- 选定阶段后，要检查哪些输入字段，缺字段时返回什么？
- 指令模型基线和本地 SFT 基线在准入上有什么区别？

## 解决思路：一步步看懂

**第 1 步：先定位“失败切片”，而不是看总体平均分。** 失败切片是按同类任务或场景划出的样本子集，例如“已签收订单的退款流程”。总体平均分可能被大量简单样本拉高，掩盖某个流程一直做错。notebook 要求先固定目标、验收门槛、工作负载和切片，并记录每个切片的复测次数与失败次数；达到重复次数和失败率门槛后，再判断缺口来自表达格式、外部知识、查询计算、外部动作，还是模型内部表示与行为。以流程顺序错误为例：模型已经拿到完整规则，却仍不能稳定按“申请、核验、确认”的顺序回答；格式提示和检索复测都没解决时，这个切片才进入参数更新候选。

**第 2 步：用两个指示函数把筛选条件写成公式。** 对失败切片 $s$，设复测次数 $n_s>0$、失败次数 $k_s$，最少复测次数 $n_{\min}$、最低失败率 $p_{\min}$：

$$
\hat p_s=\frac{k_s}{n_s},\qquad Q(s)=\mathbf{1}[n_s\ge n_{\min}]\,\mathbf{1}[\hat p_s\ge p_{\min}].
$$

$\hat p_s$ 是观测失败率，$\mathbf{1}[\cdot]$ 是条件成立为 1、否则为 0 的指示函数，$Q(s)=1$ 表示该切片进入“可重复失败集合” $F$。随后还要核对评测隔离（训练/候选调整不能接触独立评测答案）和基线覆盖（每个待处理切片都有已测结果），避免把样本不足或未测的切片误当成参数缺口。

**第 3 步：非参数方案解决后，用集合差得到 $F_{\mathrm{param}}$。** 设候选 $c$ 在切片 $s$ 上的指标为 $M(c,s)$，验收门槛为 $\tau$，新增失败集合为 $N(c,s)$：

$$
\mathrm{resolved}(c,s)=\mathbf{1}[M(c,s)\ge\tau]\,\mathbf{1}[N(c,s)=\varnothing].
$$

只有指标达到门槛**且**没有引入新的失败，才把切片记为已解决。若可重复失败集合为 $F$、被非参数方案解决的切片集合为 $S$，仍需考虑参数更新的集合是 $F_{\mathrm{param}}=F\setminus S$。这一步很重要：质量指标变好但引入了安全失败，不能算解决；`test_parameter_update_gate` 里 `behavior` 切片指标 0.92 已过 0.9 门槛，但 `introduced_failures=["safety"]`，所以仍留在残留集合。

**第 4 步：TODO 1——筛选可重复失败切片。** 参考答案：

```python
repeatable_failure_slices = sorted(
    name for name, run in failure_runs.items()
    if int(run["attempts"]) >= minimum_repeats
    and int(run["failures"]) / int(run["attempts"]) >= minimum_failure_rate
)
uncovered_failure_slices = sorted(set(repeatable_failure_slices) - baseline_slices)
```

第一行同时检查“复测次数足够”和“失败率达到门槛”，排序是为了让返回结果稳定可比。第二行用集合差找出“可重复失败但没有基线记录”的切片。函数最后返回 `ready = bool(repeatable_failure_slices) and not missing_fields and not failed_checks and not uncovered_failure_slices`：四个条件任一不满足都不能进入下一步。常见错误是只筛失败率、忘了 `attempts >= minimum_repeats`，或者把 `uncovered` 直接忽略。

**第 5 步：TODO 2——找出仍需考虑参数更新的问题。** 参考答案：

```python
resolved_slices = sorted({
    item["failure_slice"] for item in candidate_results
    if float(item["metric_value"]) >= target_threshold
    and not item.get("introduced_failures", [])
})
residual_parameter_slices = sorted(repeatable_failures - set(resolved_slices))
```

第一行只把“达标且无新增失败”的切片放进 `resolved_slices`；第二行用 `repeatable_failures - resolved_slices` 得到残留集合。骨架已经先拒绝了未登记切片、同一切片重复候选、非有限门槛和非有限指标，所以这里可以放心做集合运算。`test_parameter_update_gate` 的第一个用例 `format=0.94`、`fresh_knowledge=0.91` 都过 0.9，得到 `parameter_update_required=False`；第二个用例 `behavior=0.92` 因新增 `safety` 失败，得到 `residual_parameter_slices=["behavior"]`。

**第 6 步：TODO 3——按缺口与反馈形态选择阶段。** 参考答案的分支是：

```python
if parameter_gap == "representation":
    next_stage = "continued_pretraining"
elif feedback_mode == "demonstrations":
    next_stage = "sft"
elif feedback_mode in {"preference_pairs", "binary_feedback"}:
    next_stage = "offline_preference"
elif feedback_mode in {"learned_reward", "verifier"}:
    next_stage = "online_rl"
else:
    return {"decision": "stop_at_baseline", "reason": "no_new_feedback_signal", "missing": []}
```

`parameter_gap` 只允许 `representation` 或 `behavior`，默认 `behavior`；`feedback_mode` 允许 `demonstrations`、`preference_pairs`、`binary_feedback`、`learned_reward`、`verifier`、`none`，默认 `demonstrations`。领域语料支持持续预训练；示范回答支持 SFT；成对偏好进入 DPO 分支，单条二元反馈进入 KTO 分支；学习型奖励或验证器进入在线 RL，但 PPO/GRPO/RLOO 由 `update_method` 单独指定。notebook 反复强调：验证器（verifier）不会自动选择 GRPO，奖励来源和更新方法是两个问题。

**第 7 步：选完阶段还要查材料清单。** `requirements` 字典逐阶段列出必需字段：持续预训练要 `domain_corpus_version`、`evaluation_manifest`、`training_budget_gpu_hours`；SFT 要 `candidate_demonstration_version`、`evaluation_manifest`、`training_budget_gpu_hours`；离线偏好要 `baseline_ready`、`baseline_manifest`、`baseline_source`、`evaluation_manifest`，再按反馈形态追加 `preference_dataset_version` 或 `binary_feedback_dataset_version`；在线 RL 还要 `exploration_required`、`rollout_budget`、`rollback_checkpoint`、`update_method`，并按奖励来源追加 `reward_model_version` 或 `verifier_version`。`missing` 列表由“字段值属于 `(None, False, "", 0)`”筛出；`baseline_ready` 还额外要求 `is True`，所以传字符串 `"yes"` 也会被记为缺失。缺字段时返回 `{"decision": "blocked", "reason": "stage_requirements_not_ready", "missing": [...]}`。

**第 8 步：阶段选择的公式口径。** notebook 用

$$
G(z^{\ast})=\mathbf{1}[F_{\mathrm{param}}\neq\varnothing]\,\mathbf{1}[P_{z^{\ast}}\subseteq A]
$$

概括“能不能开始”：$z^{\ast}$ 是选中的阶段，$P_{z^{\ast}}$ 是该阶段所需的输入集合，$A$ 是现有训练输入集合。集合非空且所需输入齐全，才进入试跑准备。函数返回的阶段和 `ready` 状态只表示输入检查结果；训练后的任务收益仍要用独立评测判断。填了 `verifier_version` 只表示指定了评分程序，还要在试跑时核对它能否完成评分、分数是否符合任务含义、一次采样到更新能否在预算内完成。

**第 9 步：SFT 的交接物。** 当阶段是 `sft` 且材料齐全时，结果里会多一个 `data_request`：

```python
{
    "residual_failure_slices": list(parameter_report["residual_parameter_slices"]),
    "candidate_demonstration_version": context["candidate_demonstration_version"],
    "required_record_fields": ["sample_id", "source_id", "split", "instruction", "input", "response"],
    "next_section": "32_Data_Engineering_for_SFT",
}
```

它把“还有哪些失败没解决”“用哪版示范”“下游 32 节需要哪些字段”一起交给数据工程。`test_sft_data_request_contract` 逐字段检查这个字典，尤其 `next_section` 必须是 `32_Data_Engineering_for_SFT`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 失败切片（failure slice） | 按任务/场景切出的样本子集，用来定位被平均分掩盖的问题 |
| 可重复失败 | 复测次数够、失败率达到门槛的切片 |
| 评测隔离 | 训练/候选调整不能接触独立评测答案 |
| 基线覆盖 | 每个待处理切片都有已测的基线结果 |
| 非参数方案 | 提示、RAG、数据库、计算器、搜索、业务 API、验证器等不改参数的干预 |
| $F_{\mathrm{param}}$ | 非参数方案后仍未解决、需要考虑改参数的切片集合 |
| 持续预训练 | 用领域语料继续训练，补“表示/知识”缺口 |
| SFT | 用示范回答做监督微调，补“行为”缺口 |
| DPO / KTO | 离线偏好优化：成对偏好走 DPO，单条二元反馈走 KTO |
| 在线 RL | 用当前策略生成轨迹并依据奖励更新，如 PPO/GRPO/RLOO |
| 策略基线 | 固定版本、已复测的生成模型，作为更新和比较起点 |
| RLVR | 可验证奖励强化学习，描述奖励来源，不是独立训练阶段 |

## 小白常见疑问

**Q1：`rare` 切片复测 2 次失败 2 次，失败率 100%，为什么没进可重复失败集合？**
因为门槛还包括最少复测次数。`test_problem_evidence_gate` 里 `minimum_repeats=3`，`rare` 只有 `attempts=2`，所以被 `n_s >= n_min` 这一项挡掉。失败率再高，样本不足也不能作为稳定结论。

**Q2：指标已经超过门槛，为什么还不能算“已解决”？**
因为还要看有没有新增失败。`behavior` 切片指标 0.92 超过 0.9，但 `introduced_failures=["safety"]`，因此 `resolved_slices` 里没有它，它仍留在 `residual_parameter_slices`。这是“修好一个问题不能引入更严重的问题”的准入原则。

**Q3：为什么指令模型可以直接进 DPO/KTO，不用先 SFT 一遍？**
notebook 的 `baseline_source` 允许 `instruction_model` 或 `sft`，前提是基线已经通过复测（`baseline_ready is True`）并带 `baseline_manifest`。`test_instruction_and_sft_baselines` 对两种来源分别测试 `preference_pairs → dpo`、`binary_feedback → kto`，结果都进入 `offline_preference`。也就是说，阶段名称不是必须重训的理由，关键是基线可用。

**Q4：用了验证器，是不是就应该选 GRPO？**
不是。`test_online_feedback_and_method_are_independent` 把 `learned_reward` 和 `verifier` 两种奖励来源，分别与 `ppo`、`grpo`、`rloo` 三种更新方法交叉组合，六种都能进入 `online_rl`，并原样保留 `feedback_source` 和 `update_method`。奖励来源回答“分数从哪来”，更新方法回答“怎么用这些分数更新”。

**Q5：`baseline_ready` 传 `"yes"` 为什么仍然被判定缺失？**
因为骨架不只检查“非空”，还额外要求 `context.get("baseline_ready") is True`。字符串 `"yes"` 虽然不在 `(None, False, "", 0)` 里，但不是布尔 `True`，所以会被追加到 `missing`。`test_representation_and_stage_prerequisites` 最后一行专门验证了这一点。

## 动手验证：怎么知道自己做对了
notebook 在题目区末尾依次调用九个测试，全部通过后打印 `✅ 失败证据、策略基线、反馈分支与阶段准入测试通过。`
- `test_problem_evidence_gate`：第一组缺 `target_threshold`，期望 `repeatable_failure_slices=["json_format"]`、`missing_fields=["target_threshold"]`；第二组 `json_format` 3 次失败 2 次、`long_input` 4 次失败 3 次、`rare` 2 次失败 2 次，期望可重复集合为 `["json_format", "long_input"]`，同时 `failed_checks=["evaluation_ids_isolated"]`、`uncovered_failure_slices=["long_input"]`。
- `test_parameter_update_gate`：`format=0.94`、`fresh_knowledge=0.91` 过 0.9，期望 `parameter_update_required=False`；`behavior=0.92` 但新增 `safety` 失败，期望残留 `["behavior"]`；未登记切片必须抛 `ValueError`。
- `test_sft_data_request_contract`：断言 `data_request` 的四个字段完全等于预期，包括 `next_section="32_Data_Engineering_for_SFT"`。
- `test_training_stage_blockers`：只给 `candidate_demonstration_version` 时，期望缺 `["evaluation_manifest", "training_budget_gpu_hours"]`；表示缺口只给评测和预算时，期望缺 `["domain_corpus_version"]`。
- `test_invalid_evidence_and_candidate_domains`：`attempts=0`、`failures > attempts`、同一切片重复候选、`metric_value=NaN` 都必须抛 `ValueError`。
- `test_instruction_and_sft_baselines`：两种基线来源 × 两种离线反馈，检查阶段、方法（dpo/kto）、反馈来源和基线清单都不丢。
- `test_online_feedback_and_method_are_independent`：两种奖励 × 三种方法共六组，全部进入 `online_rl`，且 `feedback_source`、`update_method` 原样保留。
- `test_representation_and_stage_prerequisites`：表示缺口进入 `continued_pretraining`；离线分支逐项删除 `baseline_ready`、`baseline_manifest`、`baseline_source`、`evaluation_manifest`、`binary_feedback_dataset_version` 都会 `blocked`；在线分支逐项删除 `verifier_version`、`update_method`、`rollout_budget`、`rollback_checkpoint`、`evaluation_manifest` 也都会 `blocked`。
- `test_no_update_and_invalid_routing_inputs`：`feedback_mode="none"` 返回 `stop_at_baseline`；`parameter_update_required=False` 返回 `no_parameter_update`；证据未就绪返回 `missing_problem_evidence`；未知 `parameter_gap`、`feedback_mode`、`baseline_source`、`update_method` 都必须抛 `ValueError`。
notebook 没有保存测试输出，以断言为准；可选的 `check_stage_data_interface` 需要 `datasets`，会把阶段决策和 SFT 数据需求经 `Dataset.from_list` 往返一次，检查字段、列表顺序和分支身份是否保持，缺依赖时 `framework_check` 为 `skipped`。
