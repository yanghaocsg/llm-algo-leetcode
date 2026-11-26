# PT-REWARD-MODEL. 奖励模型训练、校准与过度优化 — 小白讲解

> 对应 notebook：[`PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb`](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb)

## 一句话总结
本节把“同一问题的两条回答，哪条更好”变成可训练的标量评分器：用 Bradley–Terry 成对损失拉大优选回答与非优选回答的分数差，再用留出偏好对选择温度校准概率，最后用三项门槛审计“奖励涨了、任务质量却掉了”的过度优化。三个 TODO 分别实现 `pairwise_reward_objective`、`select_reward_temperature` 和 `audit_reward_overoptimization`，CPU 测试即可验证；可选的 GPU 实验在冻结的 Qwen2.5-0.5B 表示上只训练标量奖励头，对比正确标签与 50% 错标标签。

## 为什么需要它：痛点是什么
初学者常见的误区是“奖励模型分数越高，回答就越好”。实际上奖励模型只是一个代理评分器：它可能只在训练偏好对上排序正确，换一批问题就排错；可能分数差很小却给出过度自信的概率；也可能被策略学会“刷分”，奖励上升而真实质量下降。只看训练损失或平均奖励，无法区分这三种情况。
生活类比：驾校教练按自己见过的路线打分。学员把教练熟悉的路线练到满分，不等于真的会开车；如果考试只考教练打分的项目，学员甚至可能学会讨好教练而不是安全驾驶。奖励模型就是这位教练，留出偏好对是没见过的路线，独立任务质量是真正的路考，温度校准则是把“教练有多自信”换算成更合理的概率。

## 核心问题
- 同一问题的两条回答怎样变成一个可反向传播的损失？
- 为什么两条分数同时加 100，分数差和损失都不变？
- 温度 $T$ 只改变概率的自信程度，为什么不能修复排错的样本？
- Brier 误差怎样在候选温度里选出“概率最不夸张”的那一个？
- 最高奖励候选在什么条件下必须被拒绝？“奖励高”和“质量合格”怎样分开检查？
- GPU 实验里，正确标签与错标标签两个奖励头的结果说明了什么？

## 解决思路：一步步看懂

**第 1 步：共享评分器与四类证据。** 对第 $i$ 个偏好对，问题记为 $x_i$，优选回答为 $y_i^+$，非优选回答为 $y_i^-$，共享奖励模型为 $r_\phi$。两条回答过同一组参数得到两个分数，训练只要求优选分数更高。notebook 把证据拆成五层：训练偏好对看监督是否明确；共享评分器看分数差与损失；留出偏好对看排序能否泛化；校准记录看概率是否过度自信；策略候选评测则把代理奖励与另外计算的任务质量对照。五层不能互相替代。

**第 2 步：Bradley–Terry 成对目标。** 分数差与优选概率为

$$
\Delta r_i=r_\phi(x_i,y_i^+)-r_\phi(x_i,y_i^-),\qquad p_i=\sigma(\Delta r_i),
$$

其中 $\sigma(a)=1/(1+e^{-a})$ 是逻辑函数。批次含 $N$ 个偏好对时，负对数似然可以写成 softplus：

$$
\mathcal L_{RM}=-\frac1N\sum_{i=1}^N\log\sigma(\Delta r_i)
=\frac1N\sum_{i=1}^N\mathrm{softplus}(-\Delta r_i).
$$

$\mathrm{softplus}(u)=\log(1+e^u)$。边际 $\Delta r_i=0$ 时 $p_i=1/2$，损失为 $\log 2\approx0.693$；边际越大，$p_i$ 越接近 1，损失越接近 0。关键不变量是“整体平移”：分数从 $(2,1)$ 同时变成 $(102,101)$，差值仍是 1，概率和损失完全不变。因此成对训练没有建立跨问题的绝对质量刻度，不同问题之间的原始分数不能直接比较。

**第 3 步（TODO 1）：`pairwise_reward_objective`。** 参考答案只有三行核心：

```python
score_margin = chosen_scores - rejected_scores          # [pairs]
pair_losses = F.softplus(-score_margin)                  # 每个偏好对的损失
accuracy = (score_margin > 0).float().mean()             # 严格正边际比例
```

`torch.as_tensor(..., dtype=torch.float32)` 先把输入统一成浮点；随后检查一维、同形、非空、有限。测试 `test_pairwise_reward_objective` 用 `[2.0, 1.2]` 对 `[0.5, -0.2]`，边际是 `[1.5, 1.4]`，准确率 1.0，损失约 0.211（两项分别约 0.201 和 0.220），落在 `0 < loss < 1` 内；把两组分数都加 10 变成 `[12.0, 11.2]` 对 `[10.5, 9.8]`，断言边际 `allclose`、损失差小于 `1e-6`。`test_reward_input_contract` 用 `[0.0, 2.0]` 对 `[1.0, 1.0]` 得到边际 `[-1, 1]`、准确率 0.5，并用 `[1.0]` 对 `[0.0, 1.0]` 验证形状不一致抛 `ValueError`。

**第 4 步（TODO 2）：温度校准。** 校准记录固定回答 A/B 的顺序，$\Delta_i=r(A)-r(B)$，$z_i=1$ 表示 A 被偏好，$z_i=0$ 表示 B 被偏好。预测概率和 Brier 误差为

$$
\hat p_i(T)=\sigma(\Delta_i/T),\qquad
\mathrm{Brier}(T)=\frac1M\sum_{i=1}^M(\hat p_i(T)-z_i)^2.
$$

$T>0$ 时只缩放分数差，不改变符号和排序：$T<1$ 让概率更极端，$T>1$ 让概率更保守。骨架已经把每个候选温度的 `brier` 算进 `records`，TODO 只需选最小：

```python
best_record = min(records, key=lambda item: (item["brier"], item["temperature"]))
```

元组键的第二项保证**并列时选更小温度**，而不是依赖候选列表顺序。`test_temperature_calibration` 给边际 `[0.2, 0.4, -0.3, -0.8]`、标签 `[1, 1, 0, 0]` 和温度 `[0.25, 0.5, 1.0, 2.0]`，选中 0.25：T 越小，正确对概率越靠近 1、错误对越靠近 0，Brier 越小。`test_temperature_tie_uses_smaller_temperature` 把边际全设为 0，任何温度都得到 `sigmoid(0)=0.5`，每条记录的 Brier 都是 `(0.5-0)^2=0.25`，于是无论候选顺序是 `[2.0, 0.5, 1.0]` 还是 `[1.0, 2.0, 0.5]`，都必须选 0.5。`test_invalid_temperature_candidates` 检查空列表、0、负数、`inf` 都抛 `ValueError`。

**第 5 步（TODO 3）：三项过度优化门槛。** 骨架先做了三件准备工作：用 `torch.argmax(rewards)` 找到奖励最高候选；用 `candidate_ids.index(baseline_id)` 找到基线；对每一对候选比较奖励方向与质量方向，只有两者都非零才计入 `comparable_pairs`，方向相反计入 `disagreements`，得到 `rank_disagreement = disagreements / comparable_pairs`（没有可比对时为 0）。TODO 只需填三个布尔值：

```python
absolute_quality_failed = quality_selected < min_quality
quality_drop_failed = quality_drop > max_quality_drop
rank_disagreement_failed = rank_disagreement > max_rank_disagreement
```

`quality_drop` 的定义是“基线质量减最高奖励候选质量”，所以正数表示候选更差。`test_overoptimization_gate` 里 `candidate_b` 奖励 1.4 最高，质量 0.61；基线质量 0.82，`min_quality=0.80`、`max_quality_drop=0.05`，于是绝对质量（0.61 < 0.80）和相对退化（0.82−0.61=0.21 > 0.05）同时失败，`decision` 为 `reject_overoptimization`。`test_rank_disagreement_gate` 的奖励递增 `[0.1, 0.6, 1.0]`、质量递减 `[0.90, 0.88, 0.86]`，三对全部方向相反，`rank_disagreement` 恰为 1.0，只有这一项失败。`test_aligned_reward_candidate` 的候选质量 0.86、基线 0.82，三项都过，`decision` 为 `accept`。注意边界是严格的 `<` 与 `>`：恰好等于门槛不算失败。

**第 6 步：GPU 实验设计。** 固定 `Qwen/Qwen2.5-0.5B-Instruct`，取训练 6 对、校准 4 对、隐藏质量 4 对；主干只做一次前向，取每条回答最后一个有效位置（`attention_mask.sum(dim=1)-1`）的表示后释放。两个线性奖励头从相同初始化出发，分别使用正确标签和 `CORRUPTED_PAIR_INDICES=[0,2,4]`（50% 错标）训练 60 步，`LEARNING_RATE=1e-2`，`MAX_LENGTH=128`，温度候选 `[0.5, 1.0, 2.0]`。自变量是标签是否正确；因变量包括首尾损失、训练/校准/留出排序准确率、选中温度与 Brier、更新时间与峰值显存。程序通过条件是四项：两候选齐全、正确标签损失下降且训练准确率至少 0.75、留出排序至少 0.5、温度扫描记录齐全——错标背离只记录，不作为通过门槛。

**第 7 步：结果怎么读。** 历史 `rtx5070ti_reward_model_v1` 的正确标签头：损失从 1.163 降到 `8.60e-14`，训练、校准、留出准确率都是 1.0，选中温度 0.5，Brier `7.18e-9`，奖励头更新中位 0.270 ms，全程峰值 `1,021,895,680` 字节（约 0.95 GiB），`decision=accept_mechanism`。错标头则出现分离：代理训练准确率仍是 1.0，但校准准确率 0.25、留出排序 0.5。这说明“拟合金标签”与“留出排序泛化”是两件事。还要注意 GPU 的 `calibrate(margins)` 把标签全设为 1（`labels = torch.ones_like(margins)`），所以那个极小的 Brier 只说明模型对 4 条校准对的“优选方向”非常自信，并不是混合正负标签下的完整校准，更不能替代任务质量或安全评测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 奖励模型（RM） | 给“问题+回答”打一个标量的评分器，是训练信号而非真理 |
| 偏好对 | 同一问题下一条优选、一条非优选的训练样本 |
| Bradley–Terry | 用分数差经 Sigmoid 得到优选概率的成对模型 |
| 分数差 / 边际（margin） | 优选分数减非优选分数，正数表示排序正确 |
| softplus | $\log(1+e^x)$，把负边际的损失写得数值更稳 |
| 排序准确率 | 批次中严格正边际所占比例，只看方向不看幅度 |
| 温度 $T$ | 缩放分数差；只改概率自信度，不改排序 |
| Brier 误差 | 预测概率与 0/1 标签的平均平方差，越小越校准 |
| 过度优化（reward hacking） | 代理奖励上升，独立任务质量却停滞或下降 |
| 排序背离比例 | 可比较候选对中，奖励排序与质量排序相反的比例 |
| 冻结主干 / 奖励头 | 不更新语言模型，只训练最后那个标量线性层 |

## 小白常见疑问
**Q1：为什么两条分数同时加 100，损失完全不变？**
因为损失只依赖 $\Delta r$。$\sigma$ 的输入是差值，`softplus(-margin)` 也是差值的函数；`pairwise_reward_objective([2.0,1.2],[0.5,-0.2])` 与 `([12.0,11.2],[10.5,9.8])` 的边际都是 `[1.5,1.4]`，测试因此断言损失差小于 `1e-6`。这也提醒你：不同问题的原始分数没有统一刻度。

**Q2：温度能把排错的一对“救回来”吗？**
不能。$T>0$ 时 $\sigma(\Delta/T)$ 与 $\Delta$ 同号，负边际永远小于 0.5。温度只改变概率离 0.5 有多远；排错要靠更好的数据或模型，不是校准。

**Q3：Brier 误差和排序准确率有什么区别？**
排序准确率只问“正边际的比例”，不看概率大小；Brier 问“给出的概率离真实标签有多远”。一个模型可以全排对但概率全是 0.51（Brier 偏大），也可以概率很自信但个别样本排错。notebook 先看留出排序，再用 Brier 选温度，最后用独立质量检查防止刷分。

**Q4：最高奖励候选为什么还要被拒绝？**
因为奖励是代理。`audit_reward_overoptimization` 先选 `argmax(rewards)`，再检查它是否低于绝对质量下限、相对基线退化是否超过容忍量、候选集合里奖励与质量的排序是否大量相反。`test_overoptimization_gate` 中奖励最高的 `candidate_b` 质量只有 0.61，低于 0.80 且比基线 0.82 掉了 0.21，所以即使奖励第一也必须拒绝。

**Q5：GPU 结果里 Brier 是 `7.18e-9`，是不是说明校准完美？**
不能这么读。那段 GPU 代码的 `calibrate` 用 `labels = torch.ones_like(margins)`，4 条校准对全被当作“优选方向”，Brier 只衡量概率离 1 多近；温度候选只有 `[0.5,1.0,2.0]`，样本也只有 4 条。它证明的是闭环能跑、能记录，不是校准质量已经达到任务要求。

## 动手验证：怎么知道自己做对了
先跑 `test_pairwise_reward_objective`：`[2.0,1.2]` 对 `[0.5,-0.2]` 必须 `accuracy == 1.0`、`margins > 0`、`0 < loss < 1`；整体平移后边际一致、损失差 `< 1e-6`。`test_reward_input_contract` 要求错序批次准确率 0.5、形状不一致抛 `ValueError`。`test_temperature_calibration` 要求选中 0.25，且等于按 `(brier, temperature)` 排序的最小项；`test_temperature_tie_uses_smaller_temperature` 要求零边际下所有 `brier == 0.25` 且始终选 0.5；`test_invalid_temperature_candidates` 要求空、0、负、`inf` 都抛错。`test_overoptimization_gate` 要求 `candidate_id == "candidate_b"`、失败项同时含 `"absolute_quality"` 与 `"quality_drop"`、决策为 `reject_overoptimization`；`test_rank_disagreement_gate` 要求 `rank_disagreement == 1.0` 且失败项只有 `["rank_disagreement"]`；`test_aligned_reward_candidate` 要求 `accept`；`test_nonfinite_reward_evidence` 要求 NaN、非法标签 0.5、`inf` 质量都抛 `ValueError`。九个测试全部通过会打印 `All reward model mechanism tests passed.`。

可选核对单元用受控模型把同一组分数喂给 TRL 的 `RewardTrainer.compute_loss`，检查 `margin_match`、`batch_loss_match`（`atol=1e-6`）、`accuracy_match`；没装 TRL 时状态为 `skipped`，不是通过。GPU 单元的 `evidence_level` 应为 `real_frozen_backbone_reward_head_benchmark`、`decision` 为 `accept_mechanism`，但 `mechanism_checks` 里 `negative_control_divergence_observed` 只记录、不参与 `passed` 判定；没有 GPU 时以 CPU 测试和断言为准，不要编造 Brier 或显存数字。
