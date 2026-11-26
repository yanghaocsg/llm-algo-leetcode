# PT-KTO. 二元反馈对齐 — 小白讲解

> 对应 notebook：[`PT-KTO_Binary_Feedback_Alignment.ipynb`](./PT-KTO_Binary_Feedback_Alignment.ipynb)

## 一句话总结
DPO 需要同一条问题下“一条好回答 + 一条坏回答”的成对数据，但真实反馈常常只有单条回答的点赞/点踩。KTO 把这种回答级 0/1 标签用起来：先算当前策略与固定参考策略在同一回答上的对数概率差 $d_\theta$，再用另一组“错配样本”估计一个非负 KL 基线 $z_{KL}$，最后用两个方向相反的逻辑函数分别处理可接受与不可接受样本。本节用 `aggregate_response_logps`、`estimate_kl_baseline`、`kto_loss`、`summarize_kto_batch` 四个 TODO 复现这条链路，并检查梯度只回到策略、不回到参考或基线。

## 为什么需要它：痛点是什么
初学者容易把对齐训练等同于“必须有偏好对”。可实际业务里，标注员往往只给单条回答“通过/不通过”，或用户只点一个赞；拿不到同一问题的另一条对照。丢弃这些单边数据会浪费反馈，直接把点赞/点踩当二分类又不是策略优化的目标。

生活类比：老师批改作文。DPO 是“把同一道题的两篇作文放在一起比好坏”；KTO 是“只给每篇打合格/不合格”。老师心里有一条基准线——大部分作文能达到的水平。KTO 让合格作文努力超过它、不合格作文努力低于它；基准线不参与学习，只作比较尺。

## 核心问题
- 回答级 0/1 标签和成对偏好标签，在数据形式和优化目标上差在哪？
- 回答得分 $s_\theta(x,y)$ 怎么从逐词元对数概率累加得到？padding 位置怎么处理？
- KL 基线 $z_{KL}$ 为什么取“平均偏移 → 截断为非负 → 停止梯度”？
- 正样本和负样本的损失方向为什么相反？权重 $w_+$、$w_-$ 为什么不能简单归一化掉？
- 隐式奖励 $r_i=\beta d_i$ 是什么？为什么批次里只有正类时，奖励差要返回 `None`？

## 解决思路：一步步看懂

**Step 1：数据从“成对”变成“单条 + 标签”。** 每条记录是 $(x,y,c)$：问题 $x$、回答 $y$、标签 $c\in\{1,0\}$。$c=1$ 表示该回答被当前规则接受（desirable），$c=0$ 表示不可接受（undesirable）。同一问题可以只有一条带标签回答，不需要凑对。notebook 的表格强调：标签只描述“具体来源的判断标准”，把不同渠道、不同验收规则的数据混在一起时，要保留来源并检查含义；训练问题与评测问题也要分开。

**Step 2：回答得分与策略偏移。** 设回答有 $T$ 个词元，$m_t\in\{0,1\}$ 是回答掩码（提示和 padding 为 0），则

$$
s_\theta(x,y)=\sum_{t=1}^{T}m_t\log\pi_\theta(y_t\mid x,y_{<t}),\qquad
d_\theta(x,y)=s_\theta(x,y)-s_{ref}(x,y).
$$

$s_\theta$ 是“当前策略生成这条回答的对数概率”，越大表示越可能生成它；$s_{ref}$ 是固定参考策略（通常是 SFT 后的模型）的同一量；$d_\theta$ 是策略相对参考的偏移。参考策略不训练，所以它的分数在反向传播时是常数。

**Step 3：KL 比较基线。** KTO 还要一组“比较样本”来估计当前策略整体偏离参考多少。notebook 说明这些样本**不按好坏标签筛选**，一种实现是在训练批次内把提示与其他回答错配，而不是另建数据集。设比较对有 $B_{KL}$ 个：

$$
z_{KL}=\max\left(0,\frac{1}{B_{KL}}\sum_{j=1}^{B_{KL}}d_\theta(x_j^{KL},y_j^{KL})\right).
$$

先求平均偏移，再截断为非负，最后停止梯度。为什么截断？如果策略在比较样本上已经比参考低（平均偏移为负），不需要额外的“下限惩罚”，取 0 即可。为什么停止梯度？$z_{KL}$ 在本批里是一个固定的比较尺，不能让模型通过压低基线来“作弊”降低损失。

**Step 4：两个方向相反的目标。** 记 $\sigma(a)=1/(1+e^{-a})$，$\beta>0$：

$$
\ell^+(d)=1-\sigma\big(\beta(d-z_{KL})\big),\qquad
\ell^-(d)=1-\sigma\big(\beta(z_{KL}-d)\big).
$$

正样本 $d$ 越大，$\ell^+$ 越小——鼓励提高这条回答的概率；负样本 $d$ 越小，$\ell^-$ 越小——鼓励降低它。两者形式相同、方向相反，这就是“二元反馈”的核心。

**Step 5：批次损失与类别权重。** 正样本乘 $w_+$、负样本乘 $w_-$，再对 $B$ 条记录取**普通均值**：

$$
\mathcal L_{KTO}=\frac{1}{B}\sum_{i=1}^{B}\left[c_iw_+\ell^+(d_i)+(1-c_i)w_-\ell^-(d_i)\right].
$$

分母是 $B$，不是权重之和 $w_+\cdot n_+ + w_-\cdot n_-$；测试专门断言 `losses.mean()` 不等于 `losses.sum() / weight_sum`。归一化会消掉权重的整体尺度，只留相对比例；普通均值让权重同时影响相对比例和损失量级。

**Step 6：TODO 1——聚合回答得分。** 参考答案：

```python
return token_logps.masked_fill(~mask, 0.0).sum(dim=-1)
```

`token_logps` 和 `response_mask` 都是 `[batch, seq]`。`masked_fill(~mask, 0.0)` 先把非回答位置清零，再沿序列维求和。为什么不能直接 `(token_logps * mask).sum(-1)`？因为 padding 位置可能是 `-inf`，`-inf * 0` 仍是 `nan`；先把它们替换成 0 才安全。测试用的正是 `[[-1,-2,-inf],[-0.5,-1.5,-2]]` 和掩码 `[[T,T,F],[F,T,T]]`，期望 `[-3.0, -3.5]`；空回答（掩码全 False）必须抛 `ValueError`。

**Step 7：TODO 2——估计 KL 基线。** 参考答案：

```python
return (policy_kl_logps - reference_kl_logps).mean().clamp_min(0.0).detach()
```

三个动作对应公式的三个部分：`mean()` 是 $1/B_{KL}\sum$，`clamp_min(0.0)` 是非负截断，`.detach()` 是停止梯度。测试里 `policy=[-1,-2,-3]`、`reference=[-2,-2,-2]`，差是 `[1,0,-1]`，均值 0，截断后仍是 0；把 policy 整体加 2 后差是 `[3,2,1]`，均值 2。测试同时断言 `baseline.requires_grad is False`。常见错误是忘了 `detach`，让梯度从基线流回策略，等于允许模型通过改变基线来降低损失；或者忘了 `clamp_min`，负基线会让正负样本的目标方向整体偏移。

**Step 8：TODO 3——逐样本损失与隐式奖励。** 参考答案：

```python
log_ratios = policy_logps - reference_logps.detach()
baseline = torch.as_tensor(kl_baseline, dtype=log_ratios.dtype, device=log_ratios.device).detach()
implicit_rewards = beta * log_ratios
losses = torch.where(
    labels,
    desirable_weight * (1.0 - torch.sigmoid(beta * (log_ratios - baseline))),
    undesirable_weight * (1.0 - torch.sigmoid(beta * (baseline - log_ratios))),
)
```

`reference_logps.detach()` 保证参考策略不更新；`baseline` 先转成同设备同类型的零维标量再 `detach`。`torch.where` 按 0/1 标签选分支：正样本用 `log_ratios - baseline`，负样本用 `baseline - log_ratios`。函数返回 `(losses, implicit_rewards)`，其中 `implicit_rewards = beta * log_ratios` 是“策略相对参考的缩放偏移”，notebook 明确它不是另外训练的奖励模型分数。

用测试数据手算：`policy=[-1,-3,-0.8,-2.8]`、`reference` 全 `-2`，所以 $d=[1,-1,1.2,-0.8]$；`beta=0.1` 时奖励为 `[0.1,-0.1,0.12,-0.08]`。`kl_baseline=0.25`、正权重 2、负权重 3。第 0 条是正样本：$1-\sigma(0.1\times(1-0.25))=1-\sigma(0.075)\approx0.4813$，乘 2 得约 `0.9625`；第 1 条是负样本：$1-\sigma(0.1\times(0.25+1))=1-\sigma(0.125)\approx0.4688$，乘 3 得约 `1.4064`。测试用同样表达式逐项 `torch.allclose`。

梯度方向也由测试锁定：`losses.mean().backward()` 后，`policy.grad` 在正样本位置 `< 0`、负样本位置 `> 0`；`reference.grad is None`、`baseline.grad is None`。负梯度经过梯度下降会提高正样本的对数概率，正梯度会降低负样本的对数概率，方向正确。

**Step 9：TODO 4——批次诊断。** 参考答案：

```python
desirable_values = implicit_rewards[labels]
undesirable_values = implicit_rewards[~labels]
desirable_mean = desirable_values.mean().item() if desirable_values.numel() else None
undesirable_mean = undesirable_values.mean().item() if undesirable_values.numel() else None
margin = desirable_mean - undesirable_mean if desirable_mean is not None and undesirable_mean is not None else None
```

`reward_margin` 只有在两类都存在时才计算。测试里 `rewards=[0.4,-0.2,0.2,-0.6]`、`labels=[1,0,1,0]`：正类均值 `(0.4+0.2)/2=0.3`，负类均值 `(-0.2-0.6)/2=-0.4`，差 `0.7`。只有正类时 `undesirable_reward_mean` 和 `reward_margin` 都是 `None`；只有负类时 `desirable_reward_mean` 和 `reward_margin` 是 `None`。不能用 0 冒充缺失类别，否则“只有正样本”的批次会显示奖励差等于正类均值，误导读结果。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KTO | 用单条回答的“好/坏”标签做策略对齐，不要求成对偏好 |
| desirable / undesirable | 标签 1 / 0：这条回答被当前规则接受 / 不接受 |
| 回答掩码 `response_mask` | 标记哪些位置属于回答、需要计分；提示和 padding 为 False |
| 对数概率 `logps` | 模型给目标词元的对数概率，越大越“愿意”生成 |
| 策略偏移 $d_\theta$ | 当前策略得分减参考策略得分 |
| KL 基线 $z_{KL}$ | 比较样本上平均偏移的非负截断，停止梯度 |
| $\beta$ | 缩放偏移的温度/强度系数，必须为正 |
| 隐式奖励 $r_i=\beta d_i$ | 策略相对参考的缩放偏移，不是独立奖励模型 |
| 类别权重 $w_+,w_-$ | 调整正/负样本在普通均值中的贡献 |
| 梯度隔离 | 只有策略参数收到梯度，参考和基线都是常量 |
| 奖励差 margin | 正类平均奖励减负类平均奖励，两类都有才计算 |

## 小白常见疑问
**Q1：KL 基线为什么不能直接用当前 batch 的好坏样本算？**
好坏标签是训练信号，用它算基线会把“优化对象”和“比较尺”混在一起。notebook 的比较样本不按好坏标签筛选，一种实现是把提示与其他回答错配；这样基线描述“策略整体相对参考偏移多少”，与单条样本的标签无关。

**Q2：基线已经是从策略分数算出来的，为什么还要 `detach`？**
不 `detach` 的话，梯度会同时通过损失里的 $d$ 和基线 $z_{KL}$ 回到策略，模型可以压低基线来降损失，而不是真正提高正样本、降低负样本的概率。测试断言 `baseline.requires_grad is False` 和 `baseline.grad is None`。

**Q3：正负样本的损失为什么一个用 $d-z$，一个用 $z-d$？**
优化方向相反。正样本希望 $d>z$，所以 $d-z$ 越大损失越小；负样本希望 $d<z$，所以 $z-d$ 越大（即 $d$ 越小）损失越小。两个式子是同一逻辑函数换了自变量符号。

**Q4：类别权重为什么用普通均值而不是加权平均？**
加权平均会除以权重之和，让损失尺度与权重绝对大小无关；普通均值让权重既改相对比例，也改损失量级。测试用 `losses.mean()` 与 `losses.sum()/weight_sum` 对比，确保实现的是前者。

**Q5：只有正样本时，奖励差为什么返回 `None` 而不是 0？**
没有负类就没有“差值”这个量。返回 0 会让人误以为正负奖励相等，实际负类不存在。批次诊断应暴露组成，缺失就显式缺失。

## 动手验证：怎么知道自己做对了
入口是 `run_kto_tests()`，依次跑五个测试，全过打印 `✅ KTO 回答聚合、KL 基线、标签方向、批次诊断与梯度隔离通过。`
- `test_kto_aggregation_and_contract`：含 `-inf` padding 的两条变长回答返回 `[-3.0, -3.5]`；空回答抛 `ValueError`。
- `test_kto_kl_baseline`：负均值截断为 0；整体加 2 后为 2；`requires_grad is False`。
- `test_kto_direction_weights_and_gradients`：核对 `rewards == beta*(policy-reference)`、正负目标、双侧权重；`mean` 不等于 `sum/weight_sum`；非零基线改变损失；提高正样本或降低负样本分数让对应损失下降；反向后 `policy.grad` 正类为负、负类为正，`reference.grad` 与 `baseline.grad` 为 `None`。
- `test_kto_batch_summary`：双类别计数各 2、正类均值 0.3、margin 0.7；单类别时缺失均值与 margin 都是 `None`；非二元标签抛错。
- `test_kto_input_checks`：非二元标签、`beta=0`、负基线、基线非零维、NaN 策略分数、inf KL 分数、非 bool 掩码都必须抛 `ValueError`。

可选框架核对调用 TRL 的 `KTOTrainer._compute_loss`，用同一组策略/参考/KL 分数和标签比较标量损失，检查 `sequence_sum_match`、`kl_baseline_match`（`policy_kl=[0.4,0.2,0.3,0.1]` 均值 0.25）和 `batch_loss_match`，容差 `1e-6`。它依赖 PT-CPU 与固定 TRL 版本，ImportError 时返回 `{"status": "skipped"}`。notebook 没保存执行输出，不要引用实际损失值。
