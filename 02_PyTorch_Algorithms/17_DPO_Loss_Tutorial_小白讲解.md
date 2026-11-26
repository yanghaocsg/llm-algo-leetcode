# 17. 直接偏好优化损失 — 小白讲解

> 对应 notebook：[`17_DPO_Loss_Tutorial.ipynb`](./17_DPO_Loss_Tutorial.ipynb)

## 一句话总结

DPO 把"同一个提示下，优选回答要比非优选回答更好"写成一个可以直接反向传播的损失：先把每条回答的 token 对数概率聚合成一个得分，再比较当前策略相对冻结参考策略的偏移，最后对"优选偏移减非优选偏移"做 `-logsigmoid`。本节用三个 TODO 实现 `aggregate_response_logps` 和 `dpo_loss`，并用测试锁定掩码、方向、数值稳定性和梯度隔离；可选的 GPU 实验在 Qwen2.5-0.5B 上扫描 `beta = 0.05 / 0.10 / 0.30`，三档都得到正偏好边际。

## 为什么需要它：痛点是什么

PPO 做 RLHF 时，训练现场要同时维护四个模型：策略模型、参考模型、奖励模型、价值模型。策略还要在线采样、等奖励打分、算优势、做裁剪，任何一环不稳定都会让训练崩掉。对初学者来说，这条链子的认知负担和工程负担都很重。

DPO 的观察是：偏好数据本身已经给了"同一提示下 A 比 B 好"的比较关系，不一定非要先训一个奖励模型再去做强化学习。于是它只保留两个模型：当前策略（要训练的）和冻结参考策略（SFT 基线，只用来当锚点），把偏好比较直接写进损失。生活类比：PPO 像请一位裁判现场给每个动作打分，再按分数调整；DPO 像手里已经有一叠"这两份作业里哪份更好"的批注，直接拿批注来改，不用再养一位裁判。

但 DPO 也有代价：它依赖离线偏好数据的覆盖，不会主动探索；参考策略必须固定，否则"偏移"没有基准；`beta`、求和还是平均、忽略位置怎么处理，每个细节都会改变目标。notebook 的测试就是围绕这些细节展开的。换句话说，DPO 简化的是"把偏好变成损失"这段链路，不是"偏好数据从哪来、质不质量"这件事；数据侧的契约仍然要单独检查。

## 核心问题

- 一条回答的"得分"为什么要对 token 对数概率求和？换成平均会改变什么？
- 为什么参考策略要 `detach`？不 detach 会发生什么？
- `beta` 到底控制什么？它和"学习率"是一回事吗？
- 为什么用 `-F.logsigmoid(z)` 而不是 `torch.log(torch.sigmoid(z))`？
- chosen 和 rejected 的顺序写反，损失会怎样变化？
- GPU 实验里"偏好准确率 1.0、边际 7.57–8.12"能说明什么，不能说明什么？

## 解决思路：一步步看懂

### Step 1：DPO 和 PPO 的分工差异

notebook 的对照表把差异列得很清楚：

| 维度 | PPO / RLHF | DPO |
|---|---|---|
| 主要反馈 | 采样后的奖励与优势 | 已标注的优选 / 非优选回答对 |
| 主要模型状态 | 策略、参考、奖励与价值模型 | 当前策略与冻结参考策略 |
| 是否主动探索 | 是，依赖采样与反馈 | 否，依赖离线数据覆盖 |
| 适合的条件 | 环境反馈、探索或可验证奖励 | 偏好对已可靠、需要稳定离线训练 |

关键不是"DPO 更先进"，而是"问题形态不同"。如果你有可靠的成对偏好、又想要稳定的离线训练，DPO 的链路短、复现容易；如果你需要探索、需要在线环境反馈，PPO 那一套仍然有位置。

### Step 2：一条 DPO 样本的五个契约

Step 2 的表格列了五个必须确认的事实：成对性（优选与非优选对应同一输入和约束）、偏好来源（差异来自正确性/帮助性/安全，而不是长度或模板）、参考策略（来自固定的 SFT 版本）、回答范围（两个策略评价完全相同的回答内容）、数据切分（训练与评测隔离）。

对代码最直接的是"回答范围"和"参考策略"。`dpo_loss` 的 docstring 原话是："四组 log probability 必须使用同一 tokenization 和回复范围；Reference 固定为 SFT 基线，用于度量 Policy 的偏好偏移。"如果 chosen 和 rejected 的 tokenization 不同，或者参考策略偷偷更新了，`policy - reference` 这个差值就不再是"偏好偏移"，而混进了长度和版本差异。

其余三条契约同样会在代码之外埋雷。"成对性"不成立时，你比较的是两个不同提示下的回答，偏好差异无从谈起；"偏好来源"不纯时，模型可能学到"更长就更优""带某个模板词就更优"，而不是正确性或帮助性；"数据切分"不隔离时，评测集里的偏好对进了训练，评测分数会变成记忆成绩。notebook 把这些写在 Step 2 而不是塞进某个断言，是因为它们大多无法只靠张量形状检查出来，必须在数据准备阶段人工确认。

### Step 3：从回答得分到 DPO 损失

先定义回答得分。设第 $t$ 个位置是否属于回答由 $m_t\in\{0,1\}$ 表示（代码里的 `response_mask`），则：

$$s_{\pi}(y\mid x)=\sum_{t=1}^{T}m_t\log \pi(y_t\mid x,y_{<t})$$

符号逐个解释：$x$ 是提示，$y$ 是回答，$T$ 是序列长度，$\pi(y_t\mid x,y_{<t})$ 是模型在给定提示和前文后对第 $t$ 个 token 给出的概率，$m_t$ 把 prompt 和 padding 位置排除掉。取对数是因为概率连乘容易下溢，取 log 后连乘变连加；乘 $m_t$ 是因为只有回答 token 参与得分。

notebook 特别提醒：长度归一化形式

$$\bar s_\pi=\frac{s_\pi}{\sum_t m_t}$$

**改变了目标**，不只是换单位。测试 `test_length_normalization_changes_objective` 给了一个能 flip 符号的例子：两条回复的 mask 分别是长度 2 和长度 1，policy 减 reference 的 sum 差值是 1.0 和 0.75（第一条更大），但 mean 差值是 0.5 和 0.75（第一条反而更小）。所以"用 sum 还是 mean"必须在比较前固定，当前与参考、优选与非优选四个得分要用同一规则。notebook 还注明：84 节用的是均值变体，不能和本页的标准求和结果混读。

接着定义相对参考策略的偏移：

$$d_y=s_{\pi_\theta}(y\mid x)-\mathrm{sg}\left(s_{\pi_{\mathrm{ref}}}(y\mid x)\right),\qquad y\in\{y^+,y^-\}$$

$\pi_\theta$ 是当前策略，$\pi_{\mathrm{ref}}$ 是冻结参考策略，$\mathrm{sg}$ 是 stop-gradient（前向照常算，反向不传梯度）。$y^+$ 是 chosen，$y^-$ 是 rejected。

然后构造偏好间隔并取损失：

$$z=\beta(d_{y^+}-d_{y^-}),\qquad \ell_{\mathrm{DPO}}=-\log\sigma(z)=\mathrm{softplus}(-z)$$

$\beta$ 是缩放系数，$\sigma$ 是 Sigmoid。直观读法：如果当前策略相对参考策略，把 chosen 提得比 rejected 更多，那么 $d_{y^+}-d_{y^-}$ 为正，$z$ 为正，$\sigma(z)$ 接近 1，损失接近 0；反过来，如果模型更偏向 rejected，$z$ 为负，损失变大。对 $z$ 求导：

$$\frac{\partial \ell}{\partial z}=\sigma(z)-1<0$$

所以"扩大优选回答的相对间隔"一定会降低损失——这正是训练想要的方向。batch 上再做平均：

$$\mathcal{L}_{\mathrm{batch}}=\frac{1}{B}\sum_{i=1}^{B}\ell_{\mathrm{DPO}}^{(i)}$$

为什么 `beta` 不是学习率？学习率决定参数每一步走多远；`beta` 决定"偏好间隔"在进入 sigmoid 前被放大还是缩小。`beta` 很小时 $z$ 接近 0，损失接近 $\log 2\approx0.693$，而且 $\partial\ell/\partial z$ 还要再乘 $\beta$ 才传到 log-prob，所以对参数的推力也小；`beta` 很大时 $z$ 容易饱和，损失对间隔的变化不敏感。它是目标函数的形状参数，不是优化器参数。

### Step 4：TODO 1——聚合回答得分

输入是 `token_logps`（形状 `[batch, seq]`）和 `response_mask`（同形状布尔张量）。骨架先做四项校验：形状相同且是二维、mask 是 `torch.bool`、每条回复至少一个有效 token（`counts == 0` 抛 `ValueError`）、有效位置的 logp 必须有限。

参考答案：

```python
response_logps = token_logps.masked_fill(~mask, 0.0).sum(dim=-1)
if normalize:
    response_logps = response_logps / counts
```

为什么先 `masked_fill(~mask, 0.0)` 再求和，而不是直接 `(token_logps * mask).sum(-1)`？两个原因。第一，被屏蔽位置可能是 `-inf`（比如某些实现用 `-inf` 表示不参与），`-inf * 0` 仍然是 `nan`，会污染整条得分；`masked_fill` 直接把它替换成 0，干净。第二，反向传播时被屏蔽位置的梯度严格为 0，不会出现 `0 * inf` 的 NaN。测试里专门放了 `[[-1.0, -2.0, -inf]]`、mask `[[True, True, False]]`，断言得分是 `-3.0`，就是在检查这一点。

`normalize=True` 时除以 `counts`（每条回复的有效 token 数），得到平均 logp。`test_response_logp_aggregation` 的手算：第一行有效位置是前两个，`-1 + -2 = -3`，平均 `-1.5`；第二行有效位置是后两个，`-0.5 + -2.5 = -4`，平均 `-2.0`。

### Step 5：TODO 2——计算相对参考的偏移

参考答案两行：

```python
chosen_log_ratio = policy_chosen_logps - reference_chosen_logps.detach()
rejected_log_ratio = policy_rejected_logps - reference_rejected_logps.detach()
```

`.detach()` 是这一问的核心。参考策略是"固定的比较锚点"，它的得分只提供常数偏移，不参与梯度。如果不 detach，反向传播会把梯度也送进参考模型的参数（如果它们是叶子张量），或者至少让计算图多背一份不必要的依赖。`test_dpo_gradient_path` 直接检查：`losses.mean().backward()` 之后，`values[0].grad` 和 `values[1].grad` 不为 None（Policy 两支有梯度），`values[2].grad` 和 `values[3].grad` 必须为 `None`（Reference 冻结）。

为什么要用"相对参考"而不是直接用 policy 得分？因为偏好数据只告诉你"chosen 比 rejected 好"，没告诉你"chosen 的绝对概率应该是多少"。减去参考得分，相当于问"当前策略相对 SFT 基线，把 chosen 抬了多少、把 rejected 压了多少"，这样模型不会为了增大 chosen 概率而整体漂移。

### Step 6：TODO 3——构造偏好间隔与稳定损失

参考答案：

```python
chosen_rewards = beta * chosen_log_ratio
rejected_rewards = beta * rejected_log_ratio
preference_logits = chosen_rewards - rejected_rewards
losses = -F.logsigmoid(preference_logits)
```

`chosen_rewards` / `rejected_rewards` 是"隐式奖励"：DPO 把 $\beta d_y$ 当作奖励的代理，返回值里也一并给出，方便观察。`preference_logits` 就是 $z$；`-F.logsigmoid(z)` 就是 $\mathrm{softplus}(-z)$。

为什么不用 `-torch.log(torch.sigmoid(z))`？因为当 $z$ 很负时，`sigmoid(z)` 会下溢到 0，`log(0)` 得到 `-inf`，损失爆炸。`logsigmoid` 内部用数值稳定写法，等价于 `z - log(1 + exp(z))` 的分段计算，$z=\pm10^4$ 也能给出有限值。测试 `test_dpo_preference_direction_and_stability` 用 `z = 1e4 - (-1e4) = 2e4`（`beta=1.0`）断言 `torch.isfinite(extreme)` 为真，就是在压这个边界。

从渐近行为看也更直观：$z$ 很负（模型更偏向 rejected）时，$\mathrm{softplus}(-z)\approx -z$，损失随间隔线性增大，梯度大小接近 1；$z$ 很正（已经偏向 chosen）时，$\mathrm{softplus}(-z)\approx e^{-z}$，损失趋近 0，梯度也趋近 0。这解释了为什么 DPO 不会对"已经推对的样本"持续施加很大压力，也解释了为什么极端间隔下训练信号会自然衰减。

用 fixture 手算第一条：`policy_chosen[0] = -1.0`，`reference_chosen[0] = -2.0`，所以 `chosen_log_ratio = 1.0`；`policy_rejected[0] = -3.0`，`reference_rejected[0] = -2.0`，所以 `rejected_log_ratio = -1.0`。于是 `chosen_rewards[0] = 0.1`、`rejected_rewards[0] = -0.1`、`z = 0.2`、`loss = -logsigmoid(0.2) = log(1+e^{-0.2}) ≈ 0.5981`。测试正是断言 `torch.allclose(chosen_rewards[0], tensor(0.1))`、`rejected_rewards[0] == -0.1`，以及 `losses[0] ≈ -F.logsigmoid(tensor(0.2))`。

把 fixture 的四条全算出来，可以更清楚地看到"损失只由相对间隔决定"：

| 样本 | policy chosen | policy rejected | chosen 偏移 | rejected 偏移 | $z$ | loss |
|---|---|---|---|---|---|---|
| 0 | −1.0 | −3.0 | 1.0 | −1.0 | 0.20 | 0.5981 |
| 1 | −1.2 | −2.5 | 0.8 | −0.5 | 0.13 | 0.6303 |
| 2 | −0.8 | −4.0 | 1.2 | −2.0 | 0.32 | 0.5459 |
| 3 | −1.5 | −2.8 | 0.5 | −0.8 | 0.13 | 0.6303 |

四条的平均损失约 `0.6011`。注意第 0 条和第 3 条的绝对得分差很多（chosen 分别是 −1.0 和 −1.5），但相对间隔分别是 2.0 和 1.3，所以损失只反映"相对参考抬了多少"，不反映"绝对概率高不高"。fixture 里 reference 全部取 −2.0，是为了把"参考"变成常数，单独观察 policy 的偏移；真实训练里参考得分会随回答长度和内容变化，但 `detach` 之后它同样只是常数。

反向传播时，`losses.mean().backward()` 对 $z_i$ 的梯度是 $(\sigma(z_i)-1)/B$。以 $z=0.2$、$B=4$ 为例，$\sigma(0.2)\approx0.5498$，梯度约 $-0.1125$；负号表示"增大 $z$ 会降低损失"。这也是为什么不需要在函数内部再做一次 `mean`：归约方式由调用方决定，函数只负责给出逐样本损失及其可微的计算图。

方向测试则把 chosen 和 rejected 对调：`dpo_loss(values[1], values[0], values[2], values[3])`，断言 `worse > losses`。对调后 $z$ 变负，损失必然变大。这个测试防止把 `(policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)` 写成相反的顺序。

### 输入契约与 batch 归约

`dpo_loss` 的骨架校验：四组得分形状必须完全一致、必须是一维非空张量、必须全部有限、`beta` 必须是正有限数。测试 `test_dpo_input_contract` 用 `values[1][:-1]` 制造形状不一致，并覆盖 `beta=0.0`、`beta=nan`、含 NaN 的得分；同一个测试还检查 `aggregate_response_logps` 对非布尔 mask 和被屏蔽位置 NaN 的处理。这些非法输入都要求抛 `ValueError`。

注意 `dpo_loss` 返回的是**逐样本**损失，不是标量。batch 均值由调用方做：`losses.mean()`。测试 `test_dpo_batch_reduction` 断言 `losses.ndim == 1`、长度等于 batch 大小、`batch_loss.ndim == 0` 且有限。为什么不让函数内部直接平均？因为调用方可能需要按样本权重、梯度累积或分布式归约做自己的聚合；把"逐样本"和"归约"分开，接口更可控。框架核对里 TRL 返回的是标量均值，所以对比时要显式取 `local_losses.mean()`。

### 框架一致性核对

notebook 用 TRL 的 `DPOTrainer._compute_loss` 做对照：构造一个 `_ControlledPolicy`，让"模型"直接返回指定的逐 token logp，绕开真实前向；配置 `f_divergence_type="reverse_kl"`、`loss_types=["sigmoid"]`、`beta=0.1`、`precompute_ref_logps=True`，把 `ref_chosen_logps` / `ref_rejected_logps` 直接喂进去。两个检查是 `sequence_sum_match`（TRL 汇总的序列得分等于给定得分）和 `batch_loss_match`（框架标量等于本地 `losses.mean()`，`atol=1e-6`、`rtol=1e-6`）。如果没装 TRL，单元返回 `status: skipped`，不会假装通过。

### Step 5：GPU 实验怎么设计的

CPU 题目只证明"张量级方向和梯度隔离"，GPU 实验要回答"真实模型上，chosen 的 log-prob 边际会不会真的变大"。设计如下：

| 配置项 | 值 | 作用 |
|---|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`，`main` 解析成不可变 commit | 固定快照 |
| 偏好数据 | 4 组固定 prompt/chosen/rejected | 隔离数据变化 |
| 候选 | `beta = 0.05 / 0.10 / 0.30` | 比较约束缩放 |
| 更新预算 | LoRA，`r=8`、`alpha=16`、4 个 optimizer step、`lr=2e-4` | 固定可训练参数与步数 |
| 长度 | `MAX_LENGTH=128` | 固定序列口径 |
| 种子 | 42 | 固定初始化与采样 |

每个 `beta` 都从同一模型快照重新开始：先 `model.disable_adapter()` 在 `torch.no_grad()` 下算参考分数 `ref_chosen`、`ref_rejected`；再启用 LoRA 做 4 步更新，每步的损失是：

```python
logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
loss = -F.logsigmoid(logits).mean()
```

更新完再算一次 `final_chosen`、`final_rejected`，得到：

- `chosen_shift = final_chosen - ref_chosen`：chosen 被抬了多少；
- `rejected_shift = final_rejected - ref_rejected`：rejected 被抬/压了多少；
- `preference_margin = chosen_shift - rejected_shift`：偏好边际；
- `preference_accuracy = (preference_margin > 0).mean()`：4 条里有多少条边际为正；
- `median_update_ms` / `p95_update_ms`：每步更新耗时；
- `peak_allocated_bytes`：峰值已分配显存。

选择规则是 `max(candidates, key=(preference_accuracy, mean_preference_margin))`，通过条件是 `preference_accuracy >= 0.75` 且 `mean_preference_margin > 0`。

这里用 `model.disable_adapter()` 而不是再加载一份参考模型，是有意的省显存设计：LoRA 适配器关闭后，同一个底座就是 SFT 参考，不需要额外的模型副本；代价是参考分数必须在更新前先算好并保存下来。`ref_chosen` / `ref_rejected` 只算一次，之后 4 步更新里都当常数用，这和 CPU 题目里的 `detach` 是同一个语义。

`RUN_MODE` 默认是 `dry_run`：不加载模型，只写一份带 `evidence_level: dry_run`、`decision: not_measured` 的元数据 JSON。只有改成 `real_gpu` 才会进入训练；如果中途报错，`except` 会把 `execution_stage`、错误类型和错误消息写进 `failure`，并把 `decision` 置为 `reject_until_fixed`，而不是让 notebook 静默失败。结果读取单元只读当前 `RUN_LABEL` 对应的文件，找不到就打印"尚未找到结果文件"，不会回退到别的运行——这防止你把旧标签的结果当成新实验的结论。

### 结果怎么读

保存的输出显示，选中的是 `beta=0.05`，模型 commit 为 `7ae557604adf67be50417f59c2c2f167def9a775`，`decision=accept_mechanism`、`failure=None`。三档结果：

| beta | 平均偏好边际 | 偏好准确率 | 中位更新 ms | p95 更新 ms | 峰值已分配 |
|---|---|---|---|---|---|
| 0.05 | 8.1155 | 1.0 | 117.72 | 162.33 | 3.426 GB |
| 0.10 | 8.0313 | 1.0 | 119.44 | 123.27 | 3.423 GB |
| 0.30 | 7.5735 | 1.0 | 122.65 | 128.85 | 3.423 GB |

读法要克制：三档都形成正边际，说明"Reference 冻结 + DPO 更新方向"这条机制链在真实模型上成立；4 条固定偏好对的准确率都是 1.0，只说明这 4 条被推对了，不说明模型整体偏好质量提升。`beta` 从 0.05 增到 0.30，平均边际从 8.12 降到 7.57，说明更大的缩放并没有带来更大的实测边际（这里只跑了 4 步，不能外推成"beta 越小越好"）。更新耗时 117–123 ms、峰值约 3.43 GB，也只描述这次有限更新，不是完整训练吞吐。notebook 的 5.4 表格自己就写了："仅证明固定偏好对上的更新方向""不外推整体偏好质量或完整训练吞吐"。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| DPO | 直接用成对偏好数据优化策略，不需要单独训奖励模型 |
| 参考策略（Reference） | 冻结的 SFT 模型，只当"偏移了多少"的锚点 |
| 回答得分 $s_\pi$ | 回答 token 对数概率之和（或平均） |
| 隐式奖励 | $\beta$ 乘上相对参考的偏移，DPO 用它替代显式奖励 |
| 偏好间隔 $z$ | chosen 隐式奖励减 rejected 隐式奖励 |
| `beta` | 缩放偏好间隔的形状参数，不是学习率 |
| `-logsigmoid` | 数值稳定的二分类式损失，等价于 `softplus(-z)` |
| response mask | 标记哪些位置属于回答、参与得分与梯度 |
| stop-gradient / `detach` | 前向照算，反向不传梯度，用来冻结参考 |
| 长度归一化 | 得分除以有效 token 数；会改变目标而不只是单位 |
| 偏好边际（margin） | 更新后 chosen 偏移减 rejected 偏移 |
| 偏好准确率 | 边际大于 0 的样本比例 |

## 小白常见疑问

**Q1：`beta` 越大，模型是不是越"偏好"chosen？**

不一定。`beta` 只是把 $d_{y^+}-d_{y^-}$ 放大后送进 sigmoid。`beta` 太小时 $z$ 接近 0，损失接近 0.693，梯度信号弱；`beta` 太大时 $z$ 容易饱和，损失对间隔变化不敏感。GPU 结果里 `beta=0.30` 的平均边际 7.57 反而低于 `beta=0.05` 的 8.12，但因为只跑了 4 步、只有 4 条样本，不能据此下"越小越好"的结论。正确说法是：`beta` 控制约束强度，具体取值要实验。

**Q2：为什么参考得分要 `detach`，不 detach 会怎样？**

不 detach 时，`reference_chosen_logps` 会和计算图连在一起，反向传播会把梯度也送给参考模型（如果它需要梯度），同时让"参考"不再是固定锚点。`test_dpo_gradient_path` 用 `values[2].grad is None` 和 `values[3].grad is None` 明确锁定了这一点。参考模型在真实训练里通常连优化器都没有，但 detach 仍然是表达"这是常数"的正确方式，也能省掉不必要的图。

**Q3：为什么用 sum 而不是 mean？mean 不是更公平吗？**

mean 看起来能消除长度影响，但它改变了优化目标。notebook 的 `test_length_normalization_changes_objective` 给了一个具体反例：两条回复的 sum 差值是 1.0 和 0.75，mean 差值是 0.5 和 0.75，排序直接翻转。所以不是"哪个更公平"，而是"必须在比较前固定同一个口径"。标准 sigmoid DPO 用 sum；如果你要用 mean，就要保证 policy/reference、chosen/rejected 四方都用 mean，并且不能把结果和 sum 版本的实验混读。

**Q4：chosen 和 rejected 的顺序写反了，测试会怎么发现？**

`test_dpo_preference_direction_and_stability` 把 policy 的 chosen/rejected 对调后重新算，断言 `torch.all(worse > losses)`。因为对调后 $z$ 变负，`-logsigmoid(z)` 变大。如果实现里把顺序写死成"总是第一组减第二组"而不检查语义，这个测试就会失败。

**Q5：GPU 实验里准确率都是 1.0，是不是说明 DPO 训练很成功？**

不能这么说。只有 4 条固定偏好对、每个 `beta` 只更新 4 步，`preference_accuracy=1.0` 只说明这 4 条在更新后边际为正。notebook 的结论字段是 `accept_mechanism`，不是"质量提升"；表格里也写明"仅证明固定偏好对上的更新方向""真实质量仍需独立评测"。如果要判断模型整体是否更符合偏好，需要独立评测集、更多样本和完整训练预算，这不是 4 条 fixture 能回答的。

## 动手验证：怎么知道自己做对了

CPU 测试是直接调用，不是 pytest；TODO 未填会抛 `NotImplementedError`。`run_dpo_tests` 依次跑七个测试，全过打印 `✅ DPO 回复聚合、隐式奖励、偏好方向、梯度隔离、稳定性与输入契约测试通过。`

- `test_response_logp_aggregation`：`[[-1,-2,-9],[-0.5,-1.5,-2.5]]` + mask `[[T,T,F],[F,T,T]]`，sum 得 `[-3.0,-4.0]`，normalize 得 `[-1.5,-2.0]`；被屏蔽位置是 `-inf` 时得分仍为 `-3.0`。
- `test_response_logp_input_contract`：形状不一致抛 `ValueError`；mask 全 False（counts 为 0）抛 `ValueError`。
- `test_dpo_implicit_rewards`：`losses.shape == chosen_rewards.shape == rejected_rewards.shape == (4,)`；`chosen_rewards[0] == 0.1`、`rejected_rewards[0] == -0.1`。
- `test_dpo_preference_direction_and_stability`：`losses` 全有限；`losses[0] ≈ -F.logsigmoid(tensor(0.2))`；对调后 `worse > losses`；`beta=1.0`、$z=2\times10^4$ 时仍有限。
- `test_dpo_batch_reduction`：`losses.ndim == 1`、长度 4；`losses.mean()` 是 0 维且有限。
- `test_dpo_gradient_path`：Policy 两支 `grad is not None`，Reference 两支 `grad is None`。
- `test_dpo_input_contract`：形状不一致、`beta=0.0`、`beta=nan`、NaN 得分、非布尔 mask、被屏蔽位置 NaN，全部抛 `ValueError`。
- `test_length_normalization_changes_objective`（单独调用）：`sums[0] > sums[1]` 但 `means[0] < means[1]`；对 `aggregate_response_logps(tokens, mask).sum()` 反向，`tokens.grad[1,1] == 0`，有效位置梯度全为 1。

可选的框架核对单元会打印 `status: passed`、`framework_path: DPOTrainer._compute_loss`、`pair_count: 4`、`local_loss`、`framework_loss` 和 `absolute_error`；如果环境里没有 TRL，则返回 `skipped`，不是通过。注意这里的 `local_loss` 是逐样本损失取均值后的标量，和 TRL 返回的标量对齐；两者误差应接近 0，若明显不为 0，先检查是不是把 sum 和 mean 口径混了。GPU 实验保存的 JSON 在 `benchmarks/results/pt_dpo/` 下，读取单元只认当前 `RUN_LABEL`，不会回退到别的运行；输出里能看到 `selected_candidate.beta=0.05`、三档候选的边际/准确率/耗时/显存，以及 `decision=accept_mechanism`。这些数字来自 notebook 已保存的输出；如果你重新跑，以你新的 JSON 为准。
