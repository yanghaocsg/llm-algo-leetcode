# 18. GRPO 组相对策略优化与可验证奖励 — 小白讲解

> 对应 notebook：[`18_GRPO_Loss_Tutorial.ipynb`](./18_GRPO_Loss_Tutorial.ipynb)

## 一句话总结
本节用 `compute_group_advantages` 和 `compute_grpo_loss` 两个函数，把"同一提示采样多个回答 → 验证器打分 → 组内算相对优势 → 带裁剪与参考策略 KL 的策略更新"这条链路在 CPU 上复现出来。四个 TODO 分别对应组内归一化、策略比率、受约束的 Token 损失和候选等权聚合；最后用 Qwen2.5-0.5B + LoRA 的小实验检查真实奖励能否沿组内优势方向推动策略更新。

## 为什么需要它：痛点是什么
初学者最容易把三件事混成一件：GRPO、RLVR、PPO。常见的困惑是"GRPO 是不是一种新的奖励模型""有了可验证奖励是不是就不用策略优化了""既然 PPO 已经有优势函数，为什么还要组内比较"。

痛点的根源在于 PPO 需要一个价值模型（critic）来估计基线：它要为每个状态预测"预期能拿多少分"，这个网络本身也占显存、也要训练。对数学题、代码题这类同一提示可以采样多个答案的任务，其实有更便宜的基线——同一组候选的平均奖励。GRPO 就是把这个"组内平均分"当作基线，省掉价值模型。

生活类比：老师给同一道题的五个学生答案打分。PPO 像给每个学生配一位私人教练，预测他"本来应该考多少分"；GRPO 直接看这五个人的平均分，谁高于平均就表扬、谁低于平均就抑制。老师不需要预知绝对水平，只需要组内排名。

另一个必须分开的概念是 RLVR（可验证奖励强化学习）：它只规定奖励从哪里来（答案匹配、单元测试、环境终态），不规定参数怎么更新。GRPO 规定更新方式，RLVR 只是可以接进来的奖励来源。

## 核心问题
- 同一提示的多个候选，怎样变成一个"有正有负"的训练信号？
- 组内优势 $A_i$ 为什么要除以组内标准差，`eps` 起什么作用？单元素组怎么办？
- 策略比率 $\rho_{i,t}$ 和裁剪目标怎样限制单步更新？裁剪真的能保证比率不越界吗？
- 参考策略 KL 为什么用 $\exp(d)-d-1$，它和 $\beta=0$ 时的行为差在哪？
- 为什么每条回答先按自己的有效 Token 求均值，再在候选间等权，而不是把所有 Token 混在一起平均？
- GPU 实验里的"策略分离度"和"裁剪比例"分别说明什么，为什么分离度高不等于模型更好？

## 解决思路：一步步看懂

**第 1 步：先理清候选组、奖励与更新的分工。** notebook 的 Step 1 用一张表把 GRPO 和 RLVR 拆开：GRPO 负责"哪些回答进一组、用什么基线、怎么更新"；RLVR 负责"奖励由哪条规则产生、验证器版本是什么"。二者共同需要的证据是组规模、奖励离散度和零方差组比例。关键结论是：组内归一化只消除不同任务之间的奖励尺度差异，不能证明奖励规则真的覆盖了任务质量。

**第 2 步：把四个公式对应到代码。** 对组 $g$ 中的候选 $i$：

$$
A_i=\frac{r_i-\mu_g}{\max(\sigma_g,\varepsilon)}
$$

$r_i$ 是候选奖励，$\mu_g$ 是同一组奖励均值，$\sigma_g$ 是组内标准差，$\varepsilon$ 是防止除零的 `eps`（默认 `1e-6`）。负优势表示"比同组平均差"，正优势表示"比同组平均好"。

对候选中的有效 Token $t$，策略比率和裁剪目标是：

$$
\rho_{i,t}=\exp\big(\log\pi_\theta(y_{i,t})-\log\pi_{old}(y_{i,t})\big)
$$

$$
s_{i,t}=\min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\big)
$$

$\pi_\theta$ 是当前策略，$\pi_{old}$ 是采样时冻结的旧策略，$\epsilon$ 是 `clip_range`（默认 `0.2`）。当优势为正时，比率涨到 $1+\epsilon$ 以上就不再给额外奖励；优势为负时同理，防止一次更新把某个 Token 的概率推得太狠。

参考策略 KL 采用 TRL 标准 GRPO 路径的非负逐 Token 估计：

$$
d_{i,t}=\log\pi_{ref}(y_{i,t})-\log\pi_\theta(y_{i,t}),\qquad
\widehat D_{KL,i,t}=\exp(d_{i,t})-d_{i,t}-1
$$

因为 $e^d-d-1\ge 0$ 恒成立（等号只在 $d=0$），所以这一项只会抬高损失，不会抵消任务奖励。最终损失先在每条候选内部按有效 Token 求均值，再在 $G$ 条候选间等权：

$$
\mathcal L_{GRPO}=\frac1G\sum_{i=1}^{G}\frac{1}{\sum_t m_{i,t}}\sum_t m_{i,t}\big[-s_{i,t}+\beta\widehat D_{KL,i,t}\big]
$$

$m_{i,t}$ 是回答掩码（1 表示有效 Token），$\beta$ 是 KL 权重；`beta=0` 时关闭参考策略约束。奖励全相同或组里只有一个候选时，$A_i=0$，损失对策略没有方向性梯度。

**第 3 步：TODO 1——组内优势。** 参考实现：

```python
centered = group_rewards - group_rewards.mean()
scale = group_rewards.std(unbiased=False).clamp_min(eps)
advantages[group_mask] = centered / scale
```

`unbiased=False` 用总体标准差（除以 $n$ 而不是 $n-1$），组内 $n$ 个候选就是全部候选，不需要无偏修正；`clamp_min(eps)` 防止组内奖励完全相同时除零。单元素组的 `centered` 是 0，所以优势自然是 0，不会凭空制造更新方向。容易写错的是用整个批次的均值和标准差，而不是按 `group_id` 分组——那会把不同提示的奖励混在一起，正是 GRPO 要避免的。

**第 4 步：TODO 2——策略比率。** 参考实现：

```python
ratio = torch.exp((log_probs_new - log_probs_old.detach()).masked_fill(~response_mask, 0.0))
clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
```

`log_probs_old.detach()` 很关键：旧策略只是采样记录，不接收梯度。测试 `test_grpo_constraints_and_gradient` 最后断言 `old.grad is None`。`masked_fill(~response_mask, 0.0)` 只影响被屏蔽位置的比率数值，避免无效位置出现奇怪的指数；真正是否参与损失由后面的掩码求和决定。

**第 5 步：TODO 3——受约束的 Token 损失。** 参考实现：

```python
token_advantages = advantages.detach().unsqueeze(-1)   # [候选, 1]
surrogate = ratio * token_advantages
clipped_surrogate = clipped_ratio * token_advantages
conservative = torch.minimum(surrogate, clipped_surrogate)
per_token_loss = -conservative
```

`unsqueeze(-1)` 把候选级的 $A_i$ 广播到该候选的所有 Token；`torch.minimum` 取原始项和裁剪项中更保守的一个。优势也 `detach()`，因为优势来自奖励和组内统计，不是要更新的参数。若提供 `log_probs_ref` 且 `beta > 0`：

```python
ref_delta = (log_probs_ref.detach() - log_probs_new).masked_fill(~response_mask, 0.0)
per_token_kl = torch.exp(ref_delta) - ref_delta - 1.0
per_token_loss = per_token_loss + beta * per_token_kl
```

参考策略同样 `detach()`，测试断言 `ref.grad is None`。

**第 6 步：TODO 4——候选等权聚合。** 参考实现：

```python
token_counts = mask.sum(dim=-1)
per_candidate = (per_token_loss * mask).sum(dim=-1) / token_counts
loss = per_candidate.mean()
```

先乘掩码再求和，把 Padding 和提示词位置排除；除以每条回答自己的有效 Token 数，让长短回答在候选维上权重相同。如果直接把所有 Token 拉平求平均，长回答会仅因为 Token 多而主导损失，测试 `test_grpo_candidate_equal_weighting` 专门用"2 个有效 Token 的候选 + 1 个有效 Token 的候选"检查最终损失为 0。

**第 7 步：CPU 测试在验证什么。** `_grpo_fixture()` 构造 5 条候选、2 个 Token 位置：组 0 是候选 0/1，组 1 是候选 2/3，组 2 只有候选 4；掩码里候选 1 和候选 4 的第二个位置无效。奖励是 `[1.0, 2.0, 0.5, 1.5, 3.0]`，所以前两组各得到 `[-1, 1]` 的优势，单元素组得到 0。`test_grpo_group_advantage` 断言每组优势均值约为 0、最后一个优势恰为 0。

**第 8 步：框架一致性核对。** 可选单元用 `GRPOConfig(loss_type="grpo", scale_rewards="group", importance_sampling_level="token", epsilon=0.2, beta=0.1, use_bias_correction_kl=False)`，把受控的逐 Token 对数概率喂给 `GRPOTrainer._compute_loss`，再与本节 `compute_grpo_loss` 的结果比较。核对项包括 `config_contract_match`、`group_advantage_match`、`reward_columns_match`、`unscorable_preserved`、`weighted_rewards_match` 和 `objective_match`；奖励函数 `?` 这类无法解析的回答保持 `NaN` 而不是被当成 0 分。该单元需要 PT-CPU 环境和 TRL，dump 中没有保存它的运行输出，以代码和断言为准。

**第 9 步：GPU 实验怎么设计、结果怎么读。** Step 5 固定两个提示、每组最多 4 个候选，用 `verify_integer_response` 产生正确性与格式奖励：`CORRECTNESS_WEIGHT = 1.0`、`FORMAT_WEIGHT = 0.1`、`VERIFIER_ID = "integer-answer-v1"`。训练只更新 LoRA（`LORA_RANK = 8`、`LORA_ALPHA = 16`，目标层 `q_proj`/`v_proj`），4 个优化器步骤，学习率 `2e-4`，裁剪范围 `0.2`。自变量是候选覆盖条件 `minimal_pair`（每组 2 个）与 `expanded_group`（每组 4 个）；因变量包括验证器通过率、不可判定数、`reward_std_mean`、`nonzero_advantages`、`policy_separation`、`mean_clip_fraction`、更新时间与峰值显存。`policy_separation` 的定义是正优势回答的对数概率平均提升减去负优势回答的平均提升，大于 0 才说明策略朝组内优势方向分离。

已保存的 `default3` 结果中，最小组正确率 0.50、格式通过率 1.00、不可判定 0，`reward_std_mean = 0.500`、非零优势 4 个、分离度 1.476、裁剪比例 0.438、更新中位数 47.58 ms、峰值约 2.16 GB；扩展组正确率仍为 0.50，格式通过率 0.625，出现 1 个不可判定候选，`reward_std_mean = 0.489`、非零优势 8 个、分离度 1.731、裁剪比例 0.594、更新中位数 76.88 ms、峰值约 3.29 GB。决策 `accept_mechanism` 只表示"验证器奖励 → 组内优势 → 有限更新"这条机制成立；最高分离度字段带有 `diagnostic_only_not_configuration_selection`，不能解读成"扩展组更好"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| GRPO | 用同一提示下候选奖励的组内均值和标准差当基线，替代 PPO 的价值模型 |
| RLVR | 可验证奖励：奖励来自答案检查、测试或环境终态，不规定更新算法 |
| 组内优势 $A_i$ | 候选奖励减组均值、再除以组标准差后的相对分数 |
| 策略比率 $\rho$ | 当前策略与采样时旧策略对同一 Token 的概率之比 |
| 裁剪（clip） | 把比率限制在 $[1-\epsilon,1+\epsilon]$，防止单步更新过猛 |
| 参考策略 KL | 当前策略偏离参考策略的惩罚，权重为 $\beta$ |
| 回答掩码 | 标出哪些位置是有效回答 Token，排除提示词和 Padding |
| 候选等权 | 每条回答先按自己的 Token 数求均值，再对回答取平均 |
| 奖励塌缩 | 一组候选奖励完全相同，优势全为 0，没有更新方向 |
| 奖励投机 | 模型学会利用验证器漏洞拿分，而不是真正完成任务 |
| LoRA | 只训练低秩适配器参数的低成本微调方式 |

## 小白常见疑问
**Q1：有了组内均值，为什么还要除以组内标准差？只减均值不行吗？**
减均值只把基线对齐，不同组奖励的离散程度仍不同：一组奖励在 0.9～1.0 之间，另一组在 0～10 之间，同样的绝对差异含义完全不同。除以 $\sigma_g$ 把优势变成无量纲的相对分数，`test_grpo_group_advantage` 因此能断言每组优势均值约为 0。`clamp_min(eps)` 是奖励全相同时的除零保护。

**Q2：`clip_range=0.2` 是不是保证更新后比率一定在 0.8～1.2？**
不是。裁剪改变的是损失对"比率继续变大"的激励，不限制策略参数实际走多远。notebook 的 Step 2 明确写了"裁剪不保证实际比率始终处于区间内"；`mean_clip_fraction` 只统计这一步有多少比率落在裁剪外，是诊断量，不是安全保证。

**Q3：KL 项为什么写成 $\exp(d)-d-1$，而不是常见的 $\log(\pi_\theta/\pi_{ref})$？**
这是 TRL 标准 GRPO 路径采用的逐 Token 非负估计：令 $d=\log\pi_{ref}-\log\pi_\theta$，则 $\exp(d)-d-1$ 恒大于等于 0，且在当前策略等于参考策略时为 0。它比带符号的近似更稳定，避免 KL 项出现负值去抵消奖励。`test_grpo_constraints_and_gradient` 断言加 KL 后的损失不小于不加 KL 的损失。

**Q4：为什么不能把所有 Token 放在一起求平均？**
因为不同回答长度不同。长回答 Token 多，如果拉平平均，它对损失的权重就更大，模型会偏向"写长"而不是"写对"。TODO 4 先对每条候选求均值再等权，`test_grpo_candidate_equal_weighting` 用长短两条候选构造总损失为 0 的例子来锁定这个行为。

**Q5：GPU 实验里扩展组分离度更高，是不是说明 4 个候选比 2 个候选更好？**
不能。该实验同时改变了候选数量和边缘状态覆盖，`comparison_scope` 写明是 `candidate_coverage_not_group_size_causality`；扩展组还出现裁剪比例 0.594、最后一步梯度范数为 0 的饱和迹象。最高分离度只是诊断字段，用来暴露现象，不是自动选优结果。

## 动手验证：怎么知道自己做对了
`run_grpo_tests()` 依次跑四个具名测试，全过才打印 `✅ GRPO 组内优势、裁剪、Reference KL、候选等权与输入契约测试通过。`；测试区还会先直接调用 `test_masked_extreme_scores_and_required_reference()`。
- `test_grpo_group_advantage`：用 `_grpo_fixture()` 的 5 条候选，断言每个 `group_id` 内优势均值 `atol=1e-6` 为 0，且单元素组最后一条优势恰为 0。
- `test_grpo_constraints_and_gradient`：断言损失是 0 维有限值、加 KL 后损失不小于不加、`exp(new-old)>0`；反向后 `new.grad` 有限、`old.grad is None`、`ref.grad is None`。
- `test_grpo_candidate_equal_weighting`：两条候选奖励 `[0,2]`、掩码一条 2 个有效位置一条 1 个，断言损失为 0，证明先各自聚合再等权。
- `test_grpo_input_contract`：形状不一致、`eps=0`、参考策略形状不符、`beta=-0.1`、掩码不是 bool、奖励含 `NaN`、`clip_range=inf` 等都必须抛 `ValueError`。
- `test_masked_extreme_scores_and_required_reference`：被屏蔽位置的梯度必须为 0，`beta>0` 却没给参考分数必须报错。
notebook 已保存 GPU 单元和 5.3 读取单元的输出（最小组/扩展组的通过率、分离度、裁剪比例、耗时与显存如上）；框架一致性单元没有保存输出，跑通与否以它内部的 `checks` 字典和 `AssertionError` 为准，不要根据截图臆测数值。
