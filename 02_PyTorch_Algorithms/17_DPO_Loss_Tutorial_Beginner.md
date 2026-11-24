# 17. DPO Loss Tutorial Beginner | 直接偏好优化损失 · 小白入门

> 配套 notebook：[17. 直接偏好优化损失](./17_DPO_Loss_Tutorial.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

你已经准备好“同一个问题、两个回答、哪个更好”的数据，DPO 直接拿它训练模型，让模型更偏向好的回答。这本 notebook 带你把这个比较写成一个可以求梯度的损失。

## 为什么需要它

想让模型的回答更符合人的喜好，经典的办法是基于人类反馈的强化学习（RLHF）：先训练一个奖励模型给回答打分，再用近端策略优化（PPO）根据分数更新模型。notebook 的导读说这条链路比较重：要同时维护策略模型、参考模型、奖励模型和价值模型，还要处理采样、打分和策略更新的稳定性。

其实手里的偏好数据已经写明了“同一个提示词下，回答 A 比回答 B 好”。DPO 的想法是不再绕奖励模型，直接把这条比较关系写成一个损失：

- 当前策略相对参考策略，给优选回答加的分比给非优选回答加的多，损失就小。
- 反过来，非优选回答被加的分更多，损失就大。

这本 notebook 只做计算链路里的三件事：把一条回答的词元对数概率汇总成一个得分，算出当前策略相对参考策略的偏移，再拼成 DPO 损失。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 优选 / 非优选回答（`chosen` / `rejected`） | 同一个提示词下的两个回答，人觉得好的叫优选，差的叫非优选 |
| 当前策略 / 参考策略（Policy / Reference） | 当前策略是正在训练的模型；参考策略是冻结不训练的监督微调版本，只提供对照分数 |
| 回答得分 | 回答范围内每个词元的对数概率加起来的总和，越大表示模型越倾向于生成这条回答 |
| 隐式奖励（implicit reward） | 当前策略的回答得分减参考策略的回答得分，再乘 `beta` |
| 偏好间隔（preference logit） | 优选回答的隐式奖励减非优选回答的隐式奖励，越大越好 |
| 停止梯度（`detach`） | 把一个张量当成常数来用，反向传播时不更新它 |
| `logsigmoid` | `sigmoid` 函数取对数。`sigmoid` 把任意实数压到 0 和 1 之间 |

## 核心思路

输入是四组回答得分：当前策略和参考策略分别给优选、非优选回答打的分。分三步：先算当前策略相对参考策略的偏移，再乘 `beta` 得到两条隐式奖励，最后用两条隐式奖励的差算出一个损失。输出是每个偏好对一个损失值，整个批次怎么取平均由调用方决定。

```text
s(y)  = Σ_t m_t * log pi(y_t | x, y_<t)     # 回答得分：回答范围内词元对数概率之和
s_bar = s(y) / Σ_t m_t                      # 长度归一化口径
d(y)  = s_policy(y) - sg(s_reference(y))    # 当前策略相对参考策略的偏移
z     = beta * (d(chosen) - d(rejected))    # 偏好间隔
loss  = -log sigmoid(z) = softplus(-z)      # 每个偏好对的损失
L     = (1 / B) * Σ loss_i                  # 整个批次取平均
```

符号说明：`x` 是提示词，`y` 是回答，`y_t` 是回答的第 `t` 个词元，`m_t` 是掩码（回答词元为 1，提示词和填充为 0），`pi` 是模型给这个词元的概率，`sg` 表示停止梯度，`beta` 是缩放系数，`B` 是一批里的偏好对数量。`z` 越大，损失越小。下面用 notebook 测试里的数字算一遍：

```python
import torch
import torch.nn.functional as F

token_logps = torch.tensor([[-1.0, -2.0, -9.0], [-0.5, -1.5, -2.5]])
mask = torch.tensor([[True, True, False], [False, True, True]])   # True 的位置才是回答词元
masked = token_logps.masked_fill(~mask, 0.0)
print(masked.sum(dim=-1), masked.sum(dim=-1) / mask.sum(dim=-1))  # 求和口径、长度归一化口径

policy_chosen, policy_rejected = torch.tensor([-1.0, -1.2]), torch.tensor([-3.0, -2.5])
reference_chosen = reference_rejected = torch.tensor([-2.0, -2.0])
beta = 0.1
chosen_rewards = beta * (policy_chosen - reference_chosen)        # 优选回答的隐式奖励
rejected_rewards = beta * (policy_rejected - reference_rejected)  # 非优选回答的隐式奖励
logits = chosen_rewards - rejected_rewards                        # 偏好间隔
print(chosen_rewards, rejected_rewards, logits)
print(-F.logsigmoid(logits), -F.logsigmoid(logits).mean())        # 每个偏好对的损失、批次平均
print(-F.logsigmoid(-logits), -F.logsigmoid(torch.zeros(1)))      # 方向颠倒、当前策略等于参考策略
```

```text
tensor([-3., -4.]) tensor([-1.5000, -2.0000])
tensor([0.1000, 0.0800]) tensor([-0.1000, -0.0500]) tensor([0.2000, 0.1300])
tensor([0.5981, 0.6303]) tensor(0.6142)
tensor([0.7981, 0.7603]) tensor([0.6931])
```

第一行是回答得分。两条回答各有 2 个有效词元，求和得到 -3 和 -4；按长度归一化后是 -1.5 和 -2。`-9.0` 在掩码之外，不参与计算。

第二、三行是隐式奖励和损失。看第 1 个偏好对：优选回答的偏移是 -1.0 减 -2.0，等于 1.0，隐式奖励是 0.1；非优选回答的偏移是 -3.0 减 -2.0，等于 -1.0，隐式奖励是 -0.1。偏好间隔是 0.2，损失是 0.5981。

最后一行是两个对照。把方向颠倒，间隔变成 -0.2，损失升到 0.7981。当前策略和参考策略完全相同时，间隔是 0，损失是 0.6931。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | DPO 和 PPO / RLHF 的差别：用离线偏好对代替奖励模型、价值模型和采样 | 读 |
| Step 2 | 一条 DPO 样本要满足哪些条件：成对、偏好来源、参考策略、回答范围、数据切分 | 读 |
| Step 3 | 回答得分、相对偏移、隐式奖励、偏好间隔和 DPO 损失怎样连起来 | 读，对照上面的公式 |
| Step 4 · TODO 1 | 只聚合 response mask 内的词元；`normalize=True` 时按有效长度归一化 | 写 `aggregate_response_logps`：补 1 个变量 `response_logps`，掩码外置零后沿最后一维求和，`normalize=True` 时再除以有效词元数 |
| Step 4 · TODO 2 | 冻结 `Reference`，分别计算 `chosen` / `rejected` 的 `Policy`—`Reference` 偏移 | 写 `dpo_loss`：补 2 行 `chosen_log_ratio` 和 `rejected_log_ratio`，参考策略的得分要 `detach` |
| Step 4 · TODO 3 | 用两条隐式奖励形成 preference logit，并计算稳定的 DPO 损失 | 写 `dpo_loss`：补 2 行 `preference_logits` 和 `losses`，用 `-F.logsigmoid` 算出每个偏好对的损失 |
| 测试 | 检查回答得分的求和与归一化、隐式奖励的方向、偏好方向、极端数值下的稳定性，以及形状和空掩码等输入限制 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用受控的逐词元对数概率调用 TRL 的标准 DPO 损失，核对批次损失是否一致 | 需要 `PT-CPU` 环境 |
| Step 5（可选） | 在真实模型上换三档 `beta`，观察偏好间隔、更新时间和显存 | 需要 GPU 和 `PT-RL-CUDA` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。框架核对需要 `PT-CPU` 环境。Step 5 需要 GPU，默认不执行真实实验。

## 容易踩的坑

- **回答得分算错了**：只用掩码为 True 的位置求和，被屏蔽的位置先置零，否则那里的非有限值会污染得分。求和还是按长度归一化，四组得分必须用同一种口径。
- **参考策略接收了梯度**：参考策略是固定的对照，它的得分要 `detach()`，梯度只进入当前策略的两支。
- **`chosen` 和 `rejected` 写反了**：偏好间隔必须是“优选的隐式奖励减非优选的隐式奖励”。写反之后损失反而变大。
- **直接算 `log(sigmoid(z))`**：`z` 是很大的负数时，`sigmoid(z)` 接近 0，直接取对数会丢精度。用 `-F.logsigmoid(z)` 算同一个式子，保持数值稳定。
- **在函数里把批次取了平均**：`dpo_loss` 返回的是每个偏好对的损失，批次归约由调用方完成。

## 读完后自查

- [ ] 能说出 DPO 和 PPO 相比省掉了哪些模型。
- [ ] 能解释隐式奖励和偏好间隔分别怎么算。
- [ ] 给两条回答的词元对数概率和掩码，能算出求和口径的回答得分。
- [ ] 能说出当前策略等于参考策略时，一个偏好对的损失是多少。

## 下一步

- 打开 notebook：[17. 直接偏好优化损失](./17_DPO_Loss_Tutorial.ipynb)
- 建议先读：[15. SFT 有效更新与训练闭环 · 小白入门](./15_End_to_End_Fine_Tuning_Experiment_Beginner.md)、[16. RLHF / PPO 资源与状态 · 小白入门](./16_RLHF_PPO_Memory_Beginner.md)
- 做完之后：[PT-KTO. 二元反馈对齐 · 小白入门](./PT-KTO_Binary_Feedback_Alignment_Beginner.md)
