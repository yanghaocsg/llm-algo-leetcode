# 85. GRPO 组内奖励优化项目 — 小白讲解

> 对应 notebook：[`85_GRPO_Groupwise_Alignment_Project.ipynb`](./85_GRPO_Groupwise_Alignment_Project.ipynb)

## 一句话总结
本节把一道数学题生成多条回答、按规则打分、只在"同组奖励有差异"时形成组内优势、更新后再到留出题上复测这条链路，拆成 `check_rollout_contract`、`evaluate_grpo_candidate`、`decide_grpo_delivery` 三个 TODO。它同时定义了两个容易混淆的指标：有效组比例看"采样产生了多少相对信号"，留出评测的平均奖励与 `pass@group` 看"候选是否真的变好"。历史 v7 结果在这套门槛下是 `accept`。

## 为什么需要它：痛点是什么
GRPO 的直觉是"同一道题采样多条回答，比组内谁更好"。初学者常以为只要生成 8 条、跑完损失就会进步。实际会遇到两个坑：一是 8 条全对或全错，组内没有差异，优势全是 0，这次生成的计算白花；二是训练奖励涨了，但留出题上没变好，因为奖励只校验最终数值，可能只是学会了迎合验证器。
生活类比：班级模拟考。同一道题让 8 个学生各做一遍，只有有人对、有人错时，才能比较"哪种解法更好"；全班都满分或都零分，就得不到有效对比。而且平时小测分数提高，不等于期末换一套题也能提高——所以要另备一份没参与训练的留出卷。

## 核心问题
- 一条生成轨迹要满足哪些身份、版本、组大小和唯一性条件才算合格？
- 为什么"完整组"还要再筛"奖励有差异的组"？有效组比例和有效轨迹数怎么算？
- 有效轨迹吞吐的分母包含哪些阶段？为什么不能用全部生成回答数？
- 数据/执行/产物失败，与指标未达标，为什么要分成 `reject` 和 `tune`？
- 平均奖励和 `pass@group` 分别回答什么问题？
- 看历史结果时，为什么要先核对组规模、长度预算、目标形式和计时口径？

## 解决思路：一步步看懂

**第 1 步：生成组契约（TODO 1）。** `check_rollout_contract(groups, expected_policy_version, expected_verifier_version, expected_group_size)` 先展平所有 `trace_ids`，再逐项检查五个条件：`prompt_identity`（每题 `prompt_id` 非空）、`policy_version`（`source_policy_version` 等于期望值）、`verifier_version`、`group_size`（每组轨迹数正确）、`unique_traces`（全局轨迹编号不重复）。参考答案一行：

```python
admitted = bool(groups) and all(checks.values())
```

`bool(groups)` 不能省：空集合上的 `all([])` 返回 `True`，不特判就会把"一条轨迹都没有"当成合格。`failed_checks` 用列表推导把失败项名字返回，测试里重复轨迹那条要求恰好等于 `["unique_traces"]`。夹具里两组各 4 条轨迹、`p1`/`v1` 时通过；空组、空 `prompt_id`、组大小 1、策略 `p0`、验证器 `v0` 都必须失败。

**第 2 步：有效组与指标（TODO 2）。** `evaluate_grpo_candidate` 先做两道硬校验：`group_size` 必须是至少 2 的整数；所有奖励必须非 `None`、是 `int/float` 且 `math.isfinite`，否则抛 `ValueError`——评分没补齐就先算统计，NaN 会污染后续所有门槛。然后：

```python
complete_groups = [rewards for rewards in reward_groups if len(rewards) == group_size]
valid_groups = [rewards for rewards in complete_groups if max(rewards) > min(rewards)]
valid_group_ratio = len(valid_groups) / len(reward_groups) if reward_groups else 0.0
effective_trajectory_count = len(valid_groups) * group_size
effective_trajectories_per_second = effective_trajectory_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
```

设共 $N$ 组、组规模 $K$、有效组 $M$ 个，则有效组比例 $M/N$、有效轨迹数 $M\times K$、吞吐 $MK/t$，其中 $t$ = `rollout_seconds + update_seconds`。注意比例的分母是**全部** `reward_groups`，不是完整组数；不完整组会自动落到"无效"里，因此 `test_grpo_candidate_gates` 里两组 `[[0,1],[1,1]]` 得到 `valid_group_ratio == 0.0`。

夹具 `_grpo_candidate_fixture` 是理解口径的最好例子：4 组、`group_size=4`，奖励为 `[[0,1,0,1], [1,1,1,1], [0,0,0,0], [0,0,1,1]]`。第 1、4 组有差异，第 2、3 组全同，所以有效组 2 个、比例 `2/4 = 0.5`、有效轨迹 `2×4 = 8`；耗时 `4.0 + 2.0 = 6.0` 秒，吞吐 `8/6 ≈ 1.333` 条/秒，达到 `minimum_effective_trajectories_per_second=1.0`。八个 check 全过，`eligible` 为 `True`。

**第 3 步：三道交付决策（TODO 3）。** `decide_grpo_delivery` 只看三件事：`rollout["admitted"]`、`candidate["eligible"]`、`artifact` 同时有 `path` 和 `sha256`。参考答案：

```python
if not checks["rollout"] or not checks["artifact"] or not candidate.get("execution_ok", True):
    decision = "reject"
elif not checks["candidate"]:
    decision = "tune"
else:
    decision = "accept"
```

关键是 `execution_ok` 参与 reject：`evaluate_grpo_candidate` 的 `checks["execution"]` 是 `evidence["status"] == "ok"`。执行失败属于"要修"而不是"调参"，所以 `test_grpo_execution_failure_rejected` 用 `status="failed"` 构造候选，最终必须是 `reject`；缺产物 `{}` 也是 reject；只是指标不够（`{"eligible": False}`）才是 `tune`。

**第 4 步：指标门槛各管什么。** 在 `evaluate_grpo_candidate` 的 `checks` 里有八项：执行状态、有效组比例、有效吞吐、更新次数、平均奖励增益、`pass_at_group` 增益、长度触顶率、峰值显存。方向要分清：前六项是 `>=` 下限，后两项是 `<=` 上限。测试分别把长度触顶率改成 0.50（门槛 0.10）、峰值显存改成 20 000（门槛 10 000），都只能让候选"不合格"，不会抛异常。

**第 5 步：真实实验怎么设计。** Step 5 从 `62` 节 SFT manifest 出发，要求 `decision == "accept"` 并读取适配器哈希作为 `policy_version`；数据是 GSM8K 的 `train[:24]` 与 `test[:12]`，用 `stable_text_hash` 生成 prompt 身份并检查训练/评测不重合。配置写死 `GROUP_SIZE=8`、`MAX_PROMPT_LENGTH=192`、`MAX_NEW_TOKENS=384`、`CLIP_EPSILON=0.2`、`KL_COEFFICIENT=0.02`、`LEARNING_RATE=1e-5`、`DTYPE="bfloat16"`、`SEED=42`。注释说明 `GROUP_SIZE=8` 是因为 v1 的 group=4 只产生 1/6 个有效更新组；`MAX_NEW_TOKENS` 从 v6 的基线触顶率 51.04% 出发扩大预算。

规则验证器 `gsm8k-final-number-v1` 提取回答最后一个数值，与标准答案 `####` 后的数值比较，奖励 0/1。当前更新不是标准逐词元 GRPO：notebook 明确说"以回答平均词元对数概率构造新旧分数比，并加入与参考分数差的平方惩罚"，是回答级代理目标。所以阅读结果时不能把它当成完整的 token 级比率与 KL 估计。

**第 6 步：结果怎么读。** `decide_grpo_candidate_for_run` 把轨迹硬契约、执行、产物放一组（失败→reject），其余指标放另一组（不满足→tune）。历史 v7 的记录显示：24 组中 18 组奖励有差异、执行 18 次更新，有效组比例 `18/24 = 0.75`；留出 12 题、每题 8 条，平均奖励 `0.2708 → 0.3229`（增益约 0.0521），`pass@group` `0.5833 → 0.6667`（增益约 0.0833）；触顶率 11.46% / 18.75%，峰值分配约 7.00 GiB、保留约 9.50 GiB。按当时门槛全部通过，返回 `accept`。但要同时读口径：v7 的吞吐分母只算生成与更新局部耗时，v2 新协议才把生成、验证、旧/参考策略评分、反向和更新都计入完整训练循环；v7 的逐组记录只保存初始适配器版本，精确重放还需要生成词元、采样概率和检查点。这不是"结果无效"，而是"换协议要用新标签复测"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| RLVR | 用规则验证器给答案打 0/1 分的强化学习，不训练奖励模型 |
| GRPO | 同一问题采多条回答，用组内相对好坏形成优势 |
| 组规模 K | 每题生成几条回答，本节为 8 |
| 完整组 | 该组奖励条数等于 K，没有缺评 |
| 有效组 | 完整且奖励不全相同，能提供相对信号 |
| 有效组比例 | 有效组数 / 全部组数 |
| 有效轨迹数 | 有效组数 × 组规模，只统计真正进入优势计算的回答 |
| 有效轨迹吞吐 | 有效轨迹数 ÷ 完整训练循环秒数 |
| `pass@group` | 每题至少一条回答答对的题目比例 |
| 长度触顶率 | 达到生成长度上限的回答占比 |
| 轨迹契约 | 问题身份、策略/验证器版本、组大小、轨迹唯一性的检查 |
| 回答级代理目标 | 用回答平均对数概率构造的替代损失，不等于标准逐词元 GRPO |

## 小白常见疑问
**Q1：奖励全 1 的组也算"完整"，为什么不算"有效"？**
因为 GRPO 的优势来自组内比较。全 1 组里 `max == min`，每个回答的相对优势都是 0，更新信号为零。`valid_groups` 的条件就是 `max(rewards) > min(rewards)`；夹具里第 2、3 组因此被排除。

**Q2：有效组比例 0.5，为什么有效轨迹是 8 而不是 4？**
组规模是 4，两个有效组各贡献 4 条回答，`effective_trajectory_count = len(valid_groups) * group_size = 2 × 4 = 8`。剩下 8 条回答也花了生成和评分时间，但不计入"有效更新量"。

**Q3：吞吐为什么不用"总回答数 ÷ 时间"？**
总回答数包含全对/全错组的无效轨迹，会高估有效信号的生产率。notebook 的解析明确说"全部生成回答数会高估这项吞吐"；代码只用 `valid_groups × group_size` 做分子。

**Q4：`status="no_effective_update"` 是执行失败吗？**
不是。它说明生成完成了但没有有效组，属于"指标不够"，在 `evaluate_grpo_candidate` 里会让 `checks["execution"]` 为 False、`eligible` 为 False；`decide_grpo_delivery` 因为 `execution_ok` 为 False 把它判 `reject`。若只是普通指标（奖励、显存等）不达标，`execution_ok` 仍为 True，才是 `tune`。

**Q5：平均奖励涨了 0.052，就说明模型更强吗？**
只能说在这 12 道留出题、每题 8 条、固定种子下变好了。notebook 反复提醒：小题集、单次采样的增益还需更多问题和种子确认；奖励只校验最终数值，不评价推理过程，也可能被验证器漏洞利用。

## 动手验证：怎么知道自己做对了
notebook 的测试单元按 `test_unfinished_scores_block_candidate_statistics` → `test_rollout_contract` → `test_grpo_candidate_gates` → `test_grpo_delivery_decision` 调用，打印 `所有测试通过！`，最后再跑 `test_grpo_execution_failure_rejected`。TODO 未完成会抛 `NotImplementedError`。
- `test_rollout_contract`：合法两组 `admitted`；空组、空 `prompt_id`、组大小 1、策略 `p0`、验证器 `v0` 都 `not admitted`；重复轨迹的 `failed_checks == ["unique_traces"]`。
- `test_grpo_candidate_gates`：夹具 `eligible`；全零奖励 + `update_count=0`、长度 0.50、显存 20 000 都不合格；不完整组 `valid_group_ratio == 0.0`。
- `test_grpo_delivery_decision`：有 `path`+`sha256` 且候选合格→`accept`；空产物→`reject`；`eligible=False` 但产物齐全→`tune`。
- `test_grpo_execution_failure_rejected`：`status="failed"` 必须 `reject`，不能混成调参。
- `test_unfinished_scores_block_candidate_statistics`：`None`、`NaN`、`+inf`、`-inf` 评分都抛 `ValueError`；合法的 `0.0` 正常参与；四组 `[[0,0,1,1]]*4` 全有效时 `effective_trajectory_count == 16`。
- Step 5 保存了 v7 的真实运行证据（奖励增益 0.052083、`pass@group` 增益 0.083333），可对照 5.6 的表格；换 `RUN_LABEL` 重跑时以新证据文件和其中的 `gates` 为准，不要拿历史门槛套新协议。
