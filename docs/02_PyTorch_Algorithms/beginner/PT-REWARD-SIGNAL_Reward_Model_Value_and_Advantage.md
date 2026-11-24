# PT-REWARD-SIGNAL. Reward Model Value and Advantage | 奖励、价值与优势信号 · 小白入门

> 配套 notebook：[PT-REWARD-SIGNAL. 奖励、价值与优势信号](../PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

模型答完一道题只得到一个总分。这本 notebook 教你把总分变成每个词元都能用的训练信号，再比较两种“比基线好多少”的算法：PPO 和 GRPO。

## 为什么需要它

用强化学习的方式训练模型时，模型先生成一段回答，答完后环境、规则或奖励模型给出一个分数。比如答对得 1 分，答错得 0 分。

麻烦在于，策略更新要落在每个生成的词元上，而分数只有一个。一段回答有几十个词元，哪些该加强、哪些该减弱？分数自己不会说。

除此之外还有两件事要定：

- 当前策略训练着训练着，可能离原来的模型越来越远。需要一个代价项，惩罚这种偏离。
- “这次得了 0.8 分”算好还是算坏，要看和谁比。和模型自己的预期比是一种办法，和同一道题的其他回答比是另一种办法。

这本 notebook 把这条链拆成四步，每步写一个小函数：拼出每个词元的奖励，从后往前累计成回报，再分别算 PPO 和 GRPO 的优势。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 词元（token） | 模型处理文本的最小单位，这里指回答里每个生成出来的位置 |
| 参考策略（reference policy） | 冻结不训练的原模型。正在训练的那个模型叫当前策略 |
| 对数概率（log probability） | 模型给某个词元打的概率取了对数，总是不大于 0 |
| 动作掩码（action mask） | 一排 True / False，标出哪些位置是模型生成的词元。提示词和填充位置是 False |
| 回报（return） | 从某个位置开始，把后面的奖励打折加起来的总和 |
| 价值（value） | 价值模型对“从这个位置往后大概能拿多少回报”的估计 |
| 优势（advantage） | 回报比基线高多少。正数表示比基线好，负数表示比基线差 |

## 核心思路

输入有三种：任务奖励（回答结束后才有的一个分数）、当前策略和参考策略给每个已采样词元的对数概率、动作掩码。处理分三步：拼出每个词元的奖励，从后往前累计成回报，再用回报减去一个基线得到优势。PPO 的基线是价值模型给出的估计，逐词元相减；GRPO 的基线是同一提示下几个回答的平均奖励，按回答相减。完整 PPO 常用广义优势估计（GAE），这本 notebook 的 Step 4 不实现它，只实现回报减价值。

```text
r_t = m_t * [ 1(t = T) * r_task - beta_KL * (log pi_theta(a_t) - log pi_ref(a_t))]
G_t = r_t + gamma * G_{t+1}                    # 从后往前算，G_{T+1} = 0
A_t(PPO) = G_t - V(s_t)
A_i(GRPO) = (R_i - mu_g) / (sigma_g + eps)     # 仅当 sigma_g > eps，否则为 0
```

符号说明：`t` 是词元位置，`T` 是最后一个有效词元的位置，`m_t` 是动作掩码（有效位置为 1），`1(t = T)` 表示只在终止位置取 1，`r_task` 是任务奖励，`beta_KL` 是约束系数，`pi_theta` 和 `pi_ref` 是当前策略和参考策略，`a_t` 是位置 `t` 实际生成的词元，`gamma` 是折扣因子，`G_t` 是从位置 `t` 开始的回报，`V(s_t)` 是价值模型对位置 `t` 的估计，`R_i` 是第 `i` 个回答的奖励，`mu_g` 和 `sigma_g` 是组内均值和标准差，`eps` 是防止除零的小正数。下面用 3 个位置走一遍：

```python
import torch

actor = torch.tensor([-0.2, -0.3, -0.4])       # 当前策略给已采样词元的对数概率
reference = torch.tensor([-0.3, -0.3, -0.2])   # 参考策略给同一批词元的对数概率
mask = torch.tensor([False, True, True])       # 只有生成出来的词元才算动作

penalty = -0.5 * (actor - reference)           # 偏离参考策略的代价，beta_kl = 0.5
rewards = torch.where(mask, penalty, torch.zeros(3))
rewards[-1] += 1.0                             # 任务奖励只加在最后一个有效位置
print(penalty, rewards)

returns, running = torch.zeros(3), torch.tensor(0.0)
for t in reversed(range(3)):                   # 从后往前累计
    running = rewards[t] + 0.5 * running       # gamma = 0.5
    returns[t] = running
print(returns)

print(torch.tensor([1.0, 2.0, 4.0]) - torch.tensor([0.5, 2.5, 2.0]))   # PPO：回报减价值

group = torch.tensor([0.0, 1.0, 2.0])          # GRPO：同一提示下 3 个回答的奖励
centered = group - group.mean()
print(centered / (group.std(correction=0) + 1e-8), centered / (group.std(correction=1) + 1e-8))
print((group + 7 - (group + 7).mean()) / (group.std(correction=0) + 1e-8))
```

```text
tensor([-0.0500, -0.0000,  0.1000]) tensor([0.0000, -0.0000, 1.1000])
tensor([0.2750, 0.5500, 1.1000])
tensor([ 0.5000, -0.5000,  2.0000])
tensor([-1.2247,  0.0000,  1.2247]) tensor([-1.,  0.,  1.])
tensor([-1.2247,  0.0000,  1.2247])
```

第一行是词元奖励。位置 0 不是动作，奖励是 0。位置 1 两个策略的对数概率相同，约束项是 0。位置 2 当前策略的对数概率比参考策略低 0.2，约束项是 +0.1（这种情况下它是加分，不是代价），再加上任务奖励 1.0，得到 1.1。

第二行是回报，`gamma` 取 0.5，从后往前算：位置 2 是 1.1，位置 1 是 0 加上 1.1 的一半，位置 0 是 0.275。终局的任务奖励就这样传到了前面的词元，越靠前衰减得越多。`gamma` 取 1 时不衰减。

第三行是 PPO 优势：回报减去价值估计，正数表示比预期好，负数表示比预期差。最后两行是 GRPO 优势：3 个回答的奖励是 0、1、2，减去组均值 1 得到 -1、0、1，再按组标准差缩放。`correction=0` 用总体标准差，`correction=1` 用样本标准差，缩放幅度不同。整组奖励一起加 7，优势不变。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 任务反馈和参考策略约束怎样拼成每个词元的奖励 | 读 |
| Step 2 | 只在回答结束时才有的奖励，怎样从后往前传到每个有效词元 | 读，对照上面的例子 |
| Step 3 | PPO 用价值模型做基线，GRPO 用同组回答的平均奖励做基线 | 读 |
| Step 4 · TODO 1 | 仅在有效动作上组合采样对数概率比约束与终局任务奖励 | 写 `build_token_rewards`：补 2 个变量 `sampled_log_ratio` 和 `token_rewards`，无效位置保持 0，再把任务奖励加到 `terminal_index` |
| Step 4 · TODO 2 | 按 `G_t = r_t + gamma * G_{t+1}` 反向递推 | 写 `discounted_returns`：补 2 个变量 `running_return` 和 `returns`，从最后一个位置往前循环 |
| Step 4 · TODO 3 | 计算回报相对价值基线的优势 | 写 `compute_advantages`：补 2 行，回报减价值；`normalize=True` 时再减均值、除以总体标准差 |
| Step 4 · TODO 4 | 以组均值为基线，并在方差有效时按组标准差缩放 | 写 `group_relative_advantages`：补 2 行，奖励减组均值，按 `correction` 选标准差，标准差太小时返回 0 |
| 测试 | 检查奖励的位置和约束符号、回报的递推方向、优势对价值基线的响应、GRPO 优势的平移不变性，以及非法输入 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 TRL 的公开张量工具核对已采样词元对数概率的读取和 GRPO 组内缩放 | 需要 `PT-CPU` 环境 |
| Step 5（可选） | 在真实模型上改变约束强度和折扣系数，看它们怎样改变奖励和回报 | 需要 GPU 和 `PT-RL-CUDA` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。框架核对需要另外的 `PT-CPU` 环境。Step 5 需要 GPU，默认不执行真实实验。

## 容易踩的坑

- **任务奖励加错了位置**：任务奖励只能加在 `terminal_index` 指向的有效动作词元上，这个位置在掩码里必须是 True。提示词和填充位置的奖励保持为 0。
- **约束项的符号写反**：对数概率差是“当前策略减参考策略”，再乘 `-beta_kl`。当前策略给这个词元的对数概率高于参考策略时，这一项是负数，作为偏离的代价；反过来，低于参考策略时它是正数。
- **回报从前往后算**：每个位置的回报依赖后面位置的回报，所以必须从最后一个位置往前递推。`gamma` 必须在 `[0, 1]` 之间。
- **混淆两种标准差**：`correction=0` 是总体标准差，`correction=1` 是样本标准差。两者不改变中心化的方向，但缩放幅度不同，要显式写明。
- **方差接近 0 时硬除**：组内奖励完全相同时，直接除以标准差没有意义。`group_relative_advantages` 在标准差不大于 `eps` 时返回全 0，`compute_advantages` 的标准化也有近零方差的分支。

## 读完后自查

- [ ] 能说出词元奖励由哪两部分组成，各自加在什么位置。
- [ ] 奖励是 `[0, 0, 1]`、`gamma` 是 0.5 时，能算出三个位置的回报。
- [ ] 能说出 PPO 和 GRPO 分别拿什么当基线。
- [ ] 能解释为什么整组奖励一起加 7，GRPO 优势不变。

## 下一步

- 打开 notebook：[PT-REWARD-SIGNAL. 奖励、价值与优势信号](../PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)
- 建议先读：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)、[50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation.md)、[PT-REWARD-MODEL. 奖励模型训练、校准与过度优化 · 小白入门](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)
- 做完之后：[17. 直接偏好优化损失 · 小白入门](./17_DPO_Loss_Tutorial.md)
