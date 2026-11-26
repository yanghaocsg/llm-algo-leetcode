# PT-REWARD-SIGNAL. 奖励、价值与优势信号 — 小白讲解

> 对应 notebook：[`PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb`](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb)

## 一句话总结
一个回答往往只在最后拿到一个任务奖励，但策略更新需要知道每个生成位置该加分还是扣分。本节在 CPU 上补全 `build_token_rewards`、`discounted_returns`、`compute_advantages`、`group_relative_advantages`、`generalized_advantages` 五个函数，把终局奖励变成逐 Token 奖励、折扣回报、PPO 价值基线与 GAE 优势，以及 GRPO 的同题组内优势；最后用可选的 GPU 实验观察同一批固定回答上，约束系数 `beta_kl` 与折扣 `gamma` 如何改变信号。

## 为什么需要它：痛点是什么
初学者常见的困惑是：模型生成了一整段回答，验证器只给一个 1.0 分，那到底是第一个词写得好，还是最后一个词写得好？如果直接把 1.0 平均分给每个 Token，前面的词会被无关地奖励；如果只放在最后一个 Token，又需要一种机制把“未来会拿到 1.0”这件事向前传播。再往下一层，PPO 与 GRPO 用来比较的“基线”并不是同一个东西：PPO 减的是价值模型对当前状态的预测，GRPO 减的是同一道题其他回答的平均奖励。混用这两种基线，更新方向就会解释错。

生活类比：老师只给整张卷子一个总分。要让每道题都知道自己贡献了多少，可以（1）把总分记在最后一题，再按“离结束还有多远”向前折算；（2）找一份“这个学生平时能考多少”的预期分数作对照，超出预期的题才算优势；（3）或者把同一道题几个同学的分数放一起，比班级平均高多少。本节就是把这三件事写成可测试的代码。

## 核心问题
- 终局任务奖励应该加在哪个 Token 上，提示与 padding 位置为什么必须保持 0？
- 折扣回报为什么要从序列末尾向前递推？`gamma=0.5` 时 `[0,0,1]` 为什么变成 `[0.25,0.5,1]`？
- 回报减去价值基线后，优势的符号与大小说明什么？标准化又在做什么？
- GRPO 的组内优势为什么对“整组奖励同时加 7”不敏感？全组同分时为什么返回 0？
- GAE 的 `delta`、`gamma`、`lambda` 各自控制什么？自然结束与采集截断为什么一个取零末值、一个要 bootstrap？
- 这些量在真实训练里是“固定输入”还是“可求导参数”？

## 解决思路：一步步看懂

**第 1 步：TODO 1——把任务奖励与采样对数概率比组合成 Token 奖励。** 公式是

$$
r_t=m_t\left[\mathbf 1(t=T)r^{task}
-\beta_{KL}\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{ref}(a_t\mid s_t)\right)\right].
$$

其中 $m_t\in\{0,1\}$ 是动作掩码（提示和 padding 为 0），$T$ 是接收终局奖励的位置（默认 `terminal_index=-1`，即最后一个 Token），$\mathbf 1(t=T)$ 只在终止位置为 1，$r^{task}$ 是标量任务奖励，$\beta_{KL}\ge0$ 控制“当前策略比参考策略更自信”要扣多少分，$\pi_\theta$ 与 $\pi_{ref}$ 分别是当前策略和冻结参考策略。注意括号里是**已采样 Token** 的对数概率差，单个 Token 上可正可负，它的期望才是该位置的 KL；所以这一项既可能扣分，也可能加分。

参考答案的核心几行：

```python
sampled_log_ratio = actor_log_probs - reference_log_probs
token_rewards = torch.where(
    action_mask, -beta_kl * sampled_log_ratio, torch.zeros_like(sampled_log_ratio)
)
token_rewards = token_rewards.clone()
token_rewards[terminal_index] += task_reward_tensor
```

先按掩码写入约束项，再单独把任务奖励加到终止位置，两个来源不会混在一起。用测试数据核对符号：`actor=[-0.2,-0.3,-0.4]`、`reference=[-0.3,-0.3,-0.2]`、`beta_kl=0.5`、掩码 `[False,True,True]`。第 3 个位置的概率差是 $-0.4-(-0.2)=-0.2$，约束项是 $-0.5\times(-0.2)=+0.1$，再加终局奖励 1.0 得 1.10；第 2 个位置概率差为 0；第 1 个位置被掩码清零。所以期望是 `[0.0, 0.0, 1.10]`。容易写错的地方：忘记掩码、把任务奖励加到所有位置、把 `terminal_index` 当成 Token id 而不是位置。骨架已经检查了 1D 同形、有限值、标量奖励、非负有限 `beta_kl`、布尔掩码、终止位置范围与终止位置必须是有效动作。

**第 2 步：TODO 2——从后向前累计折扣回报。** 回报定义为

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k=r_t+\gamma G_{t+1},\qquad G_{T+1}=0.
$$

$\gamma\in[0,1]$ 是折扣因子，越接近 0 越只看眼前。递推式右边依赖 $G_{t+1}$，所以循环必须从最后一个位置往前走：

```python
returns = torch.empty_like(token_rewards)
running_return = torch.zeros((), dtype=token_rewards.dtype, device=token_rewards.device)
for index in range(token_rewards.numel() - 1, -1, -1):
    running_return = token_rewards[index] + gamma * running_return
    returns[index] = running_return
```

测试 `[0,0,1]`、`gamma=0.5`：最后一个位置是 1，中间是 $0+0.5\times1=0.5$，第一个是 $0+0.5\times0.5=0.25$。如果写成从前往后累计，会得到完全不同的数，测试正是用来区分递推方向的。`gamma` 超出 `[0,1]` 必须抛 `ValueError`。

**第 3 步：TODO 3——减去价值基线。** 逐位置优势是 $A_t=G_t-V_\phi(s_t)$。同样的回报 1.0，如果价值模型预测 0.7，优势是 +0.3；如果预测 1.2，优势是 −0.2——基线越高，同一个动作显得越“不稀奇”。参考答案：

```python
advantages = returns - values
if normalize:
    std = advantages.std(unbiased=False)
    if std > eps:
        advantages = (advantages - advantages.mean()) / (std + eps)
    else:
        advantages = advantages - advantages.mean()
```

标准化用总体标准差（`unbiased=False`），并且只在标准差大于 `eps` 时才缩放；接近零方差时只做中心化，避免把数值噪声放大。测试用 `returns=[1,2,4]`、`values=[0.5,2.5,2.0]`，原始优势应为 `[0.5,-0.5,2.0]`，标准化后均值接近 0。注意这一步的 `eps` 必须为正有限数。

**第 4 步：TODO 4——GRPO 的组内相对优势。** 对同一提示的 $n$ 条回答，奖励为 $R_i$，组均值 $\mu_g$、标准差 $\sigma_g$：

$$
A_i^{GRPO}=
\begin{cases}
(R_i-\mu_g)/(\sigma_g+\varepsilon), & \sigma_g>\varepsilon,\\
0, & \sigma_g\le\varepsilon.
\end{cases}
$$

参考答案先中心化，再按 `correction` 选总体或样本标准差；若缩放因子不大于 `eps` 就直接返回 0：

```python
centered = group_rewards - group_rewards.mean()
scale = group_rewards.std(correction=correction)
return centered / (scale + eps) if scale > eps else torch.zeros_like(centered)
```

`correction=0` 是总体标准差（除以 $n$），`correction=1` 是样本标准差（除以 $n-1$）。测试 `[0,1,2]` 整组加 7 后结果不变（平移不变性），均值接近 0；三条都是 1.0 时返回全 0；`correction=1` 的结果必须等于 `(rewards - mean) / (rewards.std(correction=1) + 1e-4)`。容易错的是把 `correction=1` 当成总体标准差，或者在零方差时仍做除法。

**第 5 步：TODO 5——GAE 残差累计与价值目标。** 先算时序差分残差，再向前累计：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
A_t^{GAE}=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$ 时只保留一步残差，$\lambda=1$ 时累计整段。参考答案：

```python
advantages = torch.empty_like(values)
running_advantage = torch.zeros_like(next_value)
for index in range(values.numel() - 1, -1, -1):
    delta = token_rewards[index] + gamma * next_value - values[index]
    running_advantage = delta + gamma * gae_lambda * running_advantage
    advantages[index] = running_advantage
    next_value = values[index]
value_targets = advantages + values
```

循环里的 `next_value` 初值由结尾类型决定：自然结束 `terminated=True` 时强制取 0；采集截断 `truncated=True` 时必须提供 `bootstrap_value`（同一轨迹末状态的价值），因为外部时间限制只是停止采集，任务还可以继续。函数带 `@torch.no_grad()`，返回值是固定的更新输入，不参与求导。测试用 `rewards=[0.2]`、`values=[0.4]`、`bootstrap=0.5`：自然结束的残差是 $0.2+0-0.4=-0.2$，目标为 $-0.2+0.4=0.2$；截断的残差是 $0.2+0.5-0.4=0.3$，目标为 $0.7$。两步测试 `rewards=[0,1]`、`values=[0.2,0.4]`、`bootstrap=0.6`、`gamma=0.5`、`lambda=0.5` 得到优势 `[0.225,0.9]`、目标 `[0.425,1.3]`；`lambda=0` 时第一项优势为 0，`lambda=1` 时目标为 `[0.65,1.3]`。还要注意 `terminated` 与 `truncated` 必须恰好一个为真。

**第 6 步：可选的 GPU 实验。** 固定三条提示/回答与任务奖励（1.0、0.6、0.9），模型是 `Qwen/Qwen2.5-0.5B-Instruct`，LoRA 秩 4、`alpha=8`、只训练 `q_proj/v_proj`、做 2 次监督更新，学习率 `1e-4`。自变量是 `BETA_KL_CANDIDATES=[0.0,0.05,0.20]` 与 `GAMMA_CANDIDATES=[0.90,1.00]` 的六组组合；因变量是每条轨迹的起始回报、终止回报和标准化优势标准差。notebook 保存的一次成功运行（`run_label` 为 `rtx5070ti_reward_signal_v3`）记录：两次准备损失约 6.5152 / 6.4829，平均绝对对数概率偏移约 0.0902，峰值分配约 2.16 GB（2.01 GiB），决策 `accept_mechanism`。读法上，`gamma=1.0`、`beta_kl=0` 时起始回报等于终止回报 0.8333，回报在各位置恒定，中心化后优势标准差接近 0（约 $6\times10^{-8}$）；`gamma=0.9` 时起始回报约 0.2829，说明折扣把终局反馈向前衰减。标准化后的标准差通常接近 1 是“标准化成功”的副产品，不是任务质量指标。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 任务奖励 $r^{task}$ | 验证器/环境/奖励模型给整条回答的一个分数 |
| 动作掩码 $m_t$ | 只有真正生成的 Token 为 1，提示和 padding 为 0 |
| 参考策略 $\pi_{ref}$ | 冻结的旧模型，用来衡量当前策略偏离了多少 |
| 对数概率比 | $\log\pi_\theta-\log\pi_{ref}$，单个 Token 可正可负 |
| 回报（Return）$G_t$ | 从当前位置起，未来折扣奖励的总和 |
| 价值（Value）$V_\phi(s_t)$ | 价值模型对“当前状态通常能拿多少回报”的预测 |
| 优势（Advantage） | 回报或回答奖励相对基线的差值，决定加/减概率 |
| 折扣因子 $\gamma$ | 远期奖励的权重，越小越短视 |
| GAE | 用多步残差加权累计优势，$\lambda$ 控制看多远 |
| 自然结束 / 采集截断 | 任务真的到终态 / 外部限制停止采集，任务还可继续 |
| bootstrap 值 | 截断时用末状态价值补足“后面还可能拿到的回报” |
| GRPO 组内基线 | 同提示多条回答的均值与标准差 |

## 小白常见疑问
**Q1：为什么 KL 项用“已采样 Token 的对数概率差”，而不是直接算完整 KL？**
因为我们手上只有实际采样出来的 Token。单个 Token 的 $\log\pi_\theta-\log\pi_{ref}$ 可正可负，notebook 的原话是“在给定上下文按当前策略采样并取期望，才得到该位置的 KL 散度”。所以这一项是逐 Token 的采样估计，不是闭式 KL。

**Q2：终局奖励为什么默认放在 `terminal_index=-1`？放在中间行不行？**
默认放在最后一个 Token，是因为它代表整条回答完成后的反馈。函数允许你显式指定 `terminal_index` 来支持过程奖励，但骨架会检查它必须落在有效动作上，否则抛 `ValueError`；越界则抛 `IndexError`。

**Q3：优势减去价值基线，不会改变策略梯度的期望吗？**
减去的基线不依赖当前动作，期望意义下不改变梯度方向，但会改变每个样本的数值大小和方差——这正是“同样的回报，基线越高优势越小”的含义。测试用同一个 `returns` 配不同 `values` 来展示这一点。

**Q4：GRPO 全组同分时为什么返回 0，而不是报错或随便给个方向？**
全组同分意味着组内没有“谁更好”的信息，标准化会除以接近 0 的标准差，所以函数在 `scale <= eps` 时返回全 0。测试 `group_relative_advantages(torch.ones(3)) == 0` 就是这条退化分支。

**Q5：既然 GAE 里有价值模型，为什么还加 `@torch.no_grad()`？**
notebook 把优势和价值目标当作“固定更新输入”：先算好，再交给策略损失使用，不对它们反传。测试还断言四个输出都 `requires_grad=False`，并保持输入 dtype。

**Q6：采集截断为什么一定要 `bootstrap_value`，自然结束却不用？**
自然结束表示任务已到终态，最后动作之后的价值就是 0；采集截断只是记录停止，任务仍可继续，所以要用同一轨迹末状态的价值估计补上后续回报。骨架对“截断但没给 bootstrap”直接抛 `ValueError`。

## 动手验证：怎么知道自己做对了
测试都在 CPU 上直接调用，TODO 没填会抛 `NotImplementedError`；全部通过后打印 `All reward-signal tests passed, including GAE and ending boundaries.`。
- `test_token_reward_placement`：`[0,0,1.10]` 的逐项断言，同时检查掩码与终止位置。
- `test_discounted_return_direction`：`[0,0,1]` 在 `gamma=0.5` 下必须得到 `[0.25,0.5,1]`。
- `test_advantage_baseline_and_normalization`：原始优势 `[0.5,-0.5,2.0]`，标准化后均值 `atol=1e-6` 接近 0。
- `test_group_relative_advantage`：加 7 不变、均值 0、全 1 返回 0、`correction=1` 与手算公式一致。
- `test_signal_input_contracts`：`gamma=1.2` 抛 `ValueError`，`terminal_index=3` 抛 `IndexError`，NaN 奖励、非布尔掩码、`eps=0`、无穷组奖励都要被拒绝。
- `test_scalar_reward_and_zero_tail_return`：单元素回报 `[0.2]`；形状 `[1.0]` 和 `[[1.0]]` 的任务奖励必须抛 `ValueError`。
- `test_gae_endings_and_targets`：自然结束优势 −0.2/目标 0.2，截断优势 0.3/目标 0.7；float32 与 float64 都保持 dtype 且不求导。
- `test_gae_recurrence_and_limits`：两步优势 `[0.225,0.9]`、目标 `[0.425,1.3]`；`lambda=0/1`、`gamma=0` 的边界。
- `test_gae_input_boundaries`：终止状态互斥、bootstrap 缺失/非标量/NaN、`gamma=-0.1`、`lambda=1.1`、空段、形状错配、整型 dtype、double 与 float 混用全部抛 `ValueError`。

notebook 的 GPU 单元保存了执行输出：cell 20 的输出还保留了一次失败尝试（`failure.stage='signal_candidate_matrix'`、`NameError: build_token_rewards is not defined`），而 5.3 读取的是成功的 v3 记录。读结果时先看 `failure` 是否为 `null`，再看 `mechanism_checks.candidate_matrix_complete` 与 `terminal_reward_placement_passed`，最后才看六组候选数值。
