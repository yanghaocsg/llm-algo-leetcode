# PT-RLOO. 留一基线策略优化 — 小白讲解

> 对应 notebook：[`PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb`](./PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb)

## 一句话总结
RLOO（留一基线策略优化）不为价值模型单独开一套参数，而是用"同组其他回答的平均奖励"给每条回答当基线：同一道题生成三条回答、奖励为 1、2、4 时，三条的基线分别是 3、2.5、1.5，优势分别是 −2、−0.5、2.5。本节四个 TODO 依次完成"按有效词元聚合序列对数概率""用旧策略与参考策略的差值塑形奖励""计算缺失值感知的留一优势""构造序列级非对称裁剪损失"，并用一组手算测试和可选的 TRL `RLOOTrainer._compute_loss` 对照检查实现。

## 为什么需要它：痛点是什么
策略梯度需要一个"比平均好多少"的信号。最直接的做法是训练一个价值模型（critic）来预测每个状态的期望回报，但价值模型本身要占显存、要训练、还会引入偏差。GRPO 改用"同组回答的组均值"当基线，省掉了 critic，但组均值里包含当前回答自己：回答越极端，越会把自己的基线拉向自己，优势被稀释。

生活类比：三个学生参加同一场考试，分别是 60、70、90 分。评价 90 分那位时，如果拿"全组平均 73.3"当基准，会低估他相对同伴的优势；更公平的是拿"另外两人的平均 65"当基准，得到 +25。RLOO 就是"评价谁，就把谁从基准里拿掉"，所以叫 leave-one-out（留一）。

第二个痛点是"缺失评分"。同一组里某条回答可能因为超时、格式错误或奖励服务失败而没有分数。如果把它当成 0 分，会污染整组基线；如果把它当成"和均值一样"，又会凭空制造一条中性样本。notebook 的做法是用 `NaN` 标记缺失，从基线的求和与计数里排除，并让这条回答的优势保持 0。

## 核心问题
- 留一基线 $b_{g,-i}$ 怎么算？为什么它等于 $\frac{n}{n-1}$ 倍的"奖励减组均值"？
- 奖励塑形里的 $k_i$ 是什么？为什么它可以为负，而全分布 KL 通常非负？
- 序列概率比 $\rho_i$ 为什么在整段回答上算，而不是逐词元算？
- 裁剪目标里正优势和负优势分别被哪一侧的边界限制？为什么取 `min`？
- 缺失评分、零分、单例组分别怎么处理？损失分母为什么有两种口径？
- `old_logps`、`reference_logps`、`advantages` 为什么要 `detach()`？

## 解决思路：一步步看懂

**第 1 步：留一基线——每条回答有自己的比较对象。** 设同一提示 $x_g$ 生成了若干回答，$S_g$ 是其中**可评分**回答的索引集合，$R_{g,i}$ 是回答 $i$ 塑形后的奖励，$|S_g|$ 是可评分数量。对 $i\in S_g$ 且 $|S_g|\ge2$：

$$
b_{g,-i}=\frac{1}{|S_g|-1}\sum_{j\in S_g,\ j\ne i}R_{g,j},
\qquad A_{g,i}=R_{g,i}-b_{g,-i}.
$$

$b_{g,-i}$ 就是"除我之外，同组其他人的平均奖励"，$A_{g,i}$ 是留一优势。手算 $[1,2,4]$：

- 评价第 1 条：基线 $(2+4)/2=3$，优势 $1-3=-2$；
- 评价第 2 条：基线 $(1+4)/2=2.5$，优势 $2-2.5=-0.5$；
- 评价第 3 条：基线 $(1+2)/2=1.5$，优势 $4-1.5=2.5$。

三条优势之和为 0。这不是巧合：把基线代入并通分，$A_{g,i}=\frac{nR_{g,i}-\sum_j R_{g,j}}{n-1}=\frac{n}{n-1}(R_{g,i}-\bar R_g)$，其中 $n=|S_g|$、$\bar R_g$ 是含自身的组均值。所以 RLOO 和 GRPO 的组均值中心化只差一个 $\frac{n}{n-1}$ 的缩放；$n$ 越大，两者越接近。notebook 的 Step 1 把这条关系写出来，是为了让你不要把"留一"误解成完全不同的算法。

缺失值怎么进这个式子？`[0, NaN, 3]` 的可评分集合是 $\{0,3\}$，$n=2$：0 的基线是 3，优势 −3；3 的基线是 0，优势 3；中间的 `NaN` 优势记 0，只作占位。如果错误地把 `NaN` 填成 0，变成 `[0,0,3]`，$n=3$，三条优势会是 `[-1.5,-1.5,3]`——原来那条 0 分回答的优势从 −3 变成了 −1.5，整组信号都被带偏。`test_missing_score_retry_and_denominators` 专门把这两种情况放在一起对照。

**第 2 步：奖励塑形——把任务奖励和参考约束合在一起。** 任务奖励可能来自规则、环境或奖励模型；多分量时先加权：

$$
r_i^{ext}=\sum_m w_m r_{i,m},
\qquad
k_i=\sum_t m_{i,t}\left[\log\pi_{old}(y_{i,t}\mid x_i,y_{i,<t})
-\log\pi_{ref}(y_{i,t}\mid x_i,y_{i,<t})\right],
$$

$$
R_i=r_i^{ext}-\beta k_i.
$$

$r_{i,m}$ 是第 $m$ 个奖励分量，$w_m$ 是权重；$m_{i,t}$ 是有效回答词元的掩码；$\pi_{old}$ 是生成时的旧策略，$\pi_{ref}$ 是固定参考策略；$k_i$ 是整段回答上"旧策略对数概率减参考策略对数概率"的和；$\beta\ge0$ 是约束系数。$k_i>0$ 表示旧策略比参考策略更偏好这段回答，要从奖励里扣掉 $\beta k_i$；$k_i<0$ 表示旧策略反而更不偏好它，扣负数等于加奖励。

notebook 特意提醒：$k_i$ 是**采样估计**，单条样本可以为负，不要拿"KL 散度非负"去要求它。全分布 KL 是对所有可能序列的期望，非负；这里只是对采样到的这一段回答求和，符号没有保证。

**第 3 步：TODO 1——聚合有效词元，别让 `-inf` 漏进来。**

```python
sequence_logps = token_logps.masked_fill(~mask, 0.0).sum(dim=-1)
```

为什么不是 `(token_logps * mask).sum(dim=-1)`？因为填充位置常常用 `-inf` 当占位值，`-inf * 0` 在 IEEE 浮点里是 `NaN`，不是 0；一旦出现 `NaN`，整条序列分数就废了。`masked_fill(~mask, 0.0)` 是先把无效位置**替换**成 0，再求和，根本不发生乘法。测试 `test_rloo_sequence_aggregation` 故意构造：

```python
token_logps = torch.tensor([[-1.0, -2.0, -float("inf")], [-0.5, -1.5, -2.0]])
mask = torch.tensor([[True, True, False], [False, True, True]])
```

第一条只加前两个，得 −3；第二条只加后两个，得 −3.5。如果实现用了乘法，第一条会得到 `NaN`，测试立刻失败。函数开头还会检查：两个输入必须是同形二维张量、`completion_mask` 必须是布尔、每行至少一个有效词元、有效位置的值必须有限。这些检查把"形状对但语义错"的调用挡在前面。

**第 4 步：TODO 2——塑形奖励，并且冻结旧策略与参考策略。**

```python
sequence_kl = (
    (old_token_logps.detach() - reference_token_logps.detach())
    .masked_fill(~mask, 0.0).sum(dim=-1)
)
shaped_rewards = external_rewards.detach() - beta * sequence_kl
```

两个 `.detach()` 是必须的：旧策略和参考策略是"生成时/固定"的评分，不参与当前参数更新；如果保留计算图，反向传播会试图更新它们（或至少让图无谓变大）。`external_rewards.detach()` 同理，奖励是外部给的常数。缺失奖励用 `NaN` 表示，`NaN - β·k` 仍然是 `NaN`，所以"缺失保持缺失"不需要额外分支。

测试 `test_rloo_reward_shaping` 的手算：`external=[2.0, NaN]`，`old=[[-0.8,-0.4],[-0.5,-0.5]]`，`reference=[[-1.0,-0.5],[-0.4,-0.7]]`，`mask=[[True,True],[True,False]]`，`beta=0.5`。

- 第 0 行：$(-0.8)-(-1.0)=0.2$，$(-0.4)-(-0.5)=0.1$，$k=0.3$；$R=2-0.5\times0.3=1.85$。
- 第 1 行：只算第一个有效位置，$(-0.5)-(-0.4)=-0.1$，$k=-0.1$；外部奖励是 `NaN`，所以塑形后仍是 `NaN`。第二个位置 $(-0.5)-(-0.7)=0.2$ 被掩码排除。

测试还断言 `not shaped.requires_grad`、`old.grad is None`、`reference.grad is None`，确保梯度隔离真的生效。函数入口会拒绝无穷大的外部奖励（缺失只能用 `NaN`）、负的 `beta`、非布尔的 mask、有效位置的非有限分数，避免"看起来跑通、其实信号是 `inf`"。

**第 5 步：TODO 3——按组遍历，排除自身和缺失值。**

```python
group_sum = group_rewards[scorable].sum()
baselines = (group_sum - group_rewards[scorable]) / (count - 1)
group_advantages = torch.zeros_like(group_rewards)
group_advantages[scorable] = group_rewards[scorable] - baselines
advantages[mask] = group_advantages
```

`scorable = ~torch.isnan(group_rewards)` 只排除 `NaN`，不排除 0；0 是合法分数。`count < 2` 时直接 `continue`，整组优势保持 0——组里只有一条可评分回答，没有"其他回答"可以当基线，强行算会除以 0，或者制造一个虚假的更新信号。`group_sum - group_rewards[scorable]` 是"总和减自己"，再除以 `count - 1`，正好是留一均值。布尔索引保持了组内原来的顺序，所以写回 `advantages[mask]` 时不会错位。

测试 `test_leave_one_out_advantages` 用两组：`rewards=[1,2,4,3,NaN,7]`、`groups=[0,0,0,1,1,1]`，期望 `[-2,-0.5,2.5,-4,0,4]`。第二组的可评分集合是 $\{3,7\}$：3 的基线是 7，优势 −4；7 的基线是 3，优势 4；`NaN` 位置是 0。测试还断言两组优势各自求和为 0，这是留一中心化的直接推论。`test_leave_one_out_contract` 用 `[1,2]` 配 `[0,1]`（两个单例组）断言全零，防止实现把"组内只有自己"算成"基线等于自己、优势 0"之外的错误值。

评分补回的场景也在这个测试里：原始 `[0, NaN, 3]` 得到 `[-3,0,3]`；把中间回答重试评分成 1，得到 `[0,1,3]`，整组优势重算为 `[-2,-0.5,2.5]`。注意第 1 条和第 3 条的优势也变了（−3 → −2，3 → 2.5），所以**不能只更新被补分的那一条**，必须整组重算。notebook 还区分"重试评分"和"重新生成"：重试沿用原采样事件和组标识；重新生成要另记采样事件。

**第 6 步：TODO 4——序列概率比与非对称裁剪。**

$$
\rho_i=\exp\left(\log\pi_\theta(y_i\mid x_i)-\log\pi_{old}(y_i\mid x_i)\right),
$$

$$
L_i=-\min\left(\rho_iA_i,\ \mathrm{clip}(\rho_i,1-\epsilon_l,1+\epsilon_h)A_i\right).
$$

$\log\pi_\theta(y_i\mid x_i)$ 是新策略对整段回答的对数概率（有效词元求和），$\log\pi_{old}$ 是生成时的旧策略；$\rho_i$ 是序列级概率比。$\epsilon_l\in(0,1)$ 是下侧范围，$\epsilon_h>0$ 是上侧范围，所以裁剪区间是 $[1-\epsilon_l, 1+\epsilon_h]$，可以不对称。

参考答案：

```python
ratios = torch.exp(new_logps - old_logps.detach())
if not torch.isfinite(ratios).all():
    raise ValueError("序列概率比溢出，请检查新旧策略偏移与数值精度")
fixed_advantages = advantages.detach()
clipped = ratios.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
losses = -torch.minimum(ratios * fixed_advantages, clipped * fixed_advantages)
```

为什么取 `min`？这是 PPO 式"保守目标"：原始目标和裁剪目标里选更小的那个，防止一次更新走太远。正优势 $A>0$ 时，$\rho$ 超过 $1+\epsilon_h$ 后，裁剪目标把收益封顶，`min` 选裁剪项；负优势 $A<0$ 时，$\rho$ 低于 $1-\epsilon_l$ 后，裁剪目标给出更负的值，`min` 选裁剪项。反方向（正优势但 $\rho$ 很小、负优势但 $\rho$ 很大）不会被裁剪"救回来"，仍然按原始目标计算——这正是 PPO 想要的不对称行为。

`test_rloo_asymmetric_clipping_boundaries` 用 `ratios=[1.5,0.6,0.6,1.5]`、`advantages=[1,-1,1,-1]`、`epsilon_low=0.1`、`epsilon_high=0.3`（裁剪区间 $[0.9,1.3]$）手算：

| 位置 | $\rho$ | $A$ | 原始 $\rho A$ | 裁剪后 | `min` | 损失 $-\min$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5 | 1 | 1.5 | 1.3 | 1.3 | −1.3 |
| 1 | 0.6 | −1 | −0.6 | −0.9 | −0.9 | 0.9 |
| 2 | 0.6 | 1 | 0.6 | 0.9 | 0.6 | −0.6 |
| 3 | 1.5 | −1 | −1.5 | −1.3 | −1.5 | 1.5 |

损失正好是 `[-1.3, 0.9, -0.6, 1.5]`。梯度检查也很有意思：位置 0、1 落在裁剪项上，`min` 选的是常数（相对于新策略），所以梯度为 0；位置 2、3 落在原始项上，梯度分别是 $-A\rho=-0.6$ 和 $-A\rho=1.5$。测试断言 `new.grad[:2]` 为零、`new.grad[2:]` 为 `[-0.6,1.5]`，并确认 `old.grad`、`advantages.grad` 都是 `None`。边界值 $\rho=0.9$ 和 $1.3$ 也测试了：此时原始项和裁剪项相等，损失仍是 $[-0.9,1.3]$，不会因为浮点误差突然跳变。

**第 7 步：溢出和分母——两个容易被忽略的工程点。** 即使 `new_logps` 和 `old_logps` 都是有限数，两者相差很大时 `exp` 仍可能溢出成 `inf`。参考答案在算完 `ratios` 后立刻检查 `torch.isfinite(ratios).all()`，不满足就抛 `ValueError`；`test_ratio_overflow_is_rejected` 用 `new=0.0`、`old=-10000.0` 构造 `exp(10000)`，要求显式失败，而不是返回"零优势乘无穷"的 `NaN` 继续更新。

另一个是损失分母。`rloo_policy_loss` 返回**逐回答**损失，不替你做归约；训练循环取均值时，零占位的回答仍然留在分母里。测试里的手算：三条优势 `[-3,0,3]`、旧分数全为 −1、比率 `[1,1,1.1]`，损失是 `[3,0,-3.3]`。三条全平均：$(3+0-3.3)/3=-0.1$；只对两条可比较回答平均：$(3-3.3)/2=-0.15$。两种口径都合法，但必须和训练配置一起记录。注意"可比较"是按"有评分且组内至少两条"筛的，不是按"优势是否为 0"筛：一条评分恰好等于组均值的回答，优势为 0，但它仍然可比较，应该进分母。

**第 8 步：可选的框架核对。** 如果环境里装了 TRL，最后一节会把固定的逐词元分数注入 `RLOOTrainer._compute_loss`，用 `SimpleNamespace` 造一个最小 trainer，`_ControlledPolicy.training=False`，`_LocalAccelerator` 的单进程 `reduce`/`gather` 直接返回原值。它构造四条回答、每条两个词元，把 `new / 2` 重复两次，使逐词元求和等于 `new`；然后比较框架返回的批次损失与本地 `rloo_policy_loss(...).mean()`，要求 `atol=1e-6, rtol=1e-6` 内一致。没装 TRL 时 `framework_check` 是 `{"status": "skipped"}`，这不是实现失败。这个核对只证明"在固定输入下，本节的序列裁剪目标与当前 TRL 版本的内部入口一致"，不证明所有超参、奖励配置或分布式归约都等价。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| RLOO | 用同组其他回答的平均奖励当基线的策略优化方法 |
| 留一基线 | 评价谁就把谁从均值里拿掉，避免自己拉自己的基线 |
| 优势（advantage） | 这条回答比基线好多少，正数鼓励、负数抑制 |
| 奖励塑形 | 任务奖励减去 $\beta$ 倍的旧策略/参考策略对数概率差 |
| $k_i$ | 整段回答上旧策略与参考策略的对数概率差之和，可为负 |
| 序列对数概率 | 回答有效词元对数概率的和，不是逐词元比率 |
| 序列概率比 $\rho$ | 新旧策略对整段回答的概率之比，等于对数差的指数 |
| 裁剪区间 | $[1-\epsilon_l,1+\epsilon_h]$，可上下不对称 |
| 保守目标 | 原始目标和裁剪目标取较小值，防止单次更新走太远 |
| 缺失评分 `NaN` | 没有分数，不参与基线求和与计数，优势记 0 |
| 单例组 | 组内可评分回答少于 2 条，没有可比对象，优势全 0 |
| 归约分母 | 全批次平均还是只对可比较回答平均，必须随训练配置记录 |

## 小白常见疑问

**Q1：留一基线和 GRPO 的组均值到底差多少？**
差一个 $\frac{n}{n-1}$ 的系数。设组均值 $\bar R_g$，则 $A_{g,i}=\frac{n}{n-1}(R_{g,i}-\bar R_g)$。三条回答时系数是 $1.5$，所以 `[1,2,4]` 的留一优势 `[-2,-0.5,2.5]` 正好是"减组均值 2.333"后 `[-1.333,-0.333,1.667]` 的 1.5 倍。组越大差异越小；组内只有一条时无定义，所以代码 `count < 2` 直接跳过。

**Q2：为什么缺失评分要用 `NaN`，不能直接填 0？**
0 是一个合法分数，填 0 会改变组内可评分数量和基线。测试里 `[0, NaN, 3]` 的留一优势是 `[-3,0,3]`；错误填成 `[0,0,3]` 后变成 `[-1.5,-1.5,3]`。前者把"未评分"排除在比较之外，后者把"未评分"当成"考了 0 分"，两条已评分回答的优势都被稀释。`leave_one_out_advantages` 只把 `torch.isinf` 当非法值拒绝，`NaN` 是允许的缺失标记。

**Q3：`aggregate_sequence_logps` 为什么不用 `token_logps * mask`？**
因为填充位置可能是 `-inf`，`-inf * 0` 得到 `NaN`，不是 0。`test_rloo_sequence_aggregation` 第一条序列的第三个位置就是 `-float("inf")`，用乘法会污染整条序列。`masked_fill(~mask, 0.0)` 先把无效位置替换成 0，再求和，才是安全的。这个细节在变长 batch 里非常常见，不能靠"padding 一般写 0"来赌。

**Q4：裁剪目标为什么用 `min`，而且正负优势的边界感觉不一样？**
`min` 是保守目标：原始目标和裁剪目标里选更小的收益（等价于更大的损失），防止策略一次更新偏离旧策略太远。正优势时，$\rho$ 超过 $1+\epsilon_h$ 的额外收益被封顶；负优势时，$\rho$ 低于 $1-\epsilon_l$ 的额外惩罚被封顶。反方向不裁剪，因为"正优势但概率比很小"或"负优势但概率比很大"本身就是需要保留的更新信号。测试里的四行表格就是按这个规则手算的。

**Q5：为什么 `old_logps`、`advantages`、`external_rewards` 都要 `detach()`？**
它们是采样时固定下来的量：旧策略是行为策略，优势由旧策略的奖励和基线算出，外部奖励是环境给的。当前要更新的是新策略 $\pi_\theta$，只有 `new_logps` 应该保留梯度。如果不 detach，反向传播会试图穿过这些常数，轻则浪费显存，重则把"行为策略"也改掉，PPO/RLOO 的信任域假设就不成立了。测试 `test_rloo_policy_loss` 断言 `old.grad is None and advantages.grad is None`，`test_rloo_reward_shaping` 断言 `old.grad is None and reference.grad is None`。

## 动手验证：怎么知道自己做对了

notebook 先调用 `test_ratio_overflow_is_rejected()`，再调用 `run_rloo_tests()`。后者依次跑八个测试，全部通过才打印 `✅ RLOO 序列聚合、奖励塑形、缺失奖励、留一优势与非对称裁剪通过。`。没有保存执行输出，以下以代码与断言为准；TODO 没填时，四个函数会分别抛 `NotImplementedError("TODO 1：聚合有效回答 Token")` 等异常。

- `test_rloo_sequence_aggregation`：变长输入 `[[-1,-2,-inf],[-0.5,-1.5,-2]]` 配 mask `[[T,T,F],[F,T,T]]`，序列分数必须是 `[-3.0,-3.5]`。
- `test_rloo_reward_shaping`：`beta=0.5` 时 `sequence_kl` 必须是 `[0.3,-0.1]`；`shaped[0]` 必须是 `1.85`，`shaped[1]` 必须是 `NaN`；`shaped.requires_grad` 为假，`old.grad`、`reference.grad` 都是 `None`。
- `test_leave_one_out_advantages`：`[1,2,4,3,NaN,7]` 配 `[0,0,0,1,1,1]`，优势必须是 `[-2,-0.5,2.5,-4,0,4]`，且两组各自求和为 0。
- `test_leave_one_out_contract`：`[1,2]` 配 `[0,1]`（两个单例组），优势必须是全零。
- `test_missing_score_retry_and_denominators`：`[0,NaN,3]` → `[-3,0,3]`；错误填零的 `[0,0,3]` → `[-1.5,-1.5,3]`；把中间补成 1 的 `[0,1,3]` → `[-2,-0.5,2.5]`，且第 1、3 条也一起变了。损失 `[3,0,-3.3]` 的全批次均值是 `-0.1`，两条可比较回答的均值是 `-0.15`。全 `NaN` 或只有一个可评分回答的组，优势全零。
- `test_rloo_policy_loss`：四条回答、`epsilon_low=0.1`、`epsilon_high=0.2`，`ratios` 必须等于 `exp(new-old)`，`losses.shape == (4,)` 且有限；反传后 `new.grad` 非空、`old.grad` 和 `advantages.grad` 都是 `None`。
- `test_rloo_asymmetric_clipping_boundaries`：上表的 `[-1.3,0.9,-0.6,1.5]` 必须精确匹配；`new.grad[:2]` 为零、`new.grad[2:]` 为 `[-0.6,1.5]`；边界比率 `[0.9,1.3]` 的损失是 `[-0.9,1.3]`。
- `test_rloo_input_checks`：新旧分数形状不一致、`epsilon_low=1.0`、`epsilon_high=inf`、奖励含 `inf`、mask 不是布尔，都必须抛 `ValueError`。
- `test_ratio_overflow_is_rejected`：`new=0.0`、`old=-10000.0` 时 `exp(10000)` 溢出，必须抛 `ValueError`，不能返回 `NaN`。

框架核对部分取决于环境：没有 TRL 时 `framework_check` 显示 `status: "skipped"`；有 TRL 时要求 `sequence_sum_match` 和 `batch_loss_match` 都为真，后者在 `1e-6` 容差内比较框架批次损失与本地逐回答损失的均值。跑完测试后，重点看三件事：padding 位置有没有 `NaN`、缺失奖励有没有被错误填零、梯度是不是只落在 `new_logps` 上。这三项对了，RLOO 的信号链就基本正确。
