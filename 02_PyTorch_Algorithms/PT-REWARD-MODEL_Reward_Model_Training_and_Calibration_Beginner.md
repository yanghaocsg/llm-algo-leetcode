# PT-REWARD-MODEL. Reward Model Training and Calibration Beginner | 奖励模型训练、校准与过度优化 · 小白入门

> 配套 notebook：[PT-REWARD-MODEL. 奖励模型训练、校准与过度优化](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

奖励模型给每个回答打一个分数，后面的训练靠这个分数判断好坏。这本 notebook 教你用“两个回答哪个更好”的数据训练它，再检查排序对不对、打分有多自信、分数高是否质量真的好。

## 为什么需要它

用强化学习训练模型时，需要一个东西给模型的回答打分，这个打分器就是奖励模型。它把一条回答变成一个数字，分数越高代表越好。

训练它最省事的数据是偏好对：同一个提示词下，人已经标出哪个回答更好。奖励模型要学会的只是“给好的那个更高的分”。

但分数容易骗人，notebook 里举了三种情况：

- 训练集上排得很准，留出数据上却排不对。可能是模型记住了训练数据，或者训练和评测的数据有重叠。
- 排序是对的，但分数差距特别大，显得非常自信。这时要调整的是分数到概率的换算，不是重写标签。
- 代理奖励一路上升，独立的质量指标却在下降。模型学会了钻奖励的漏洞。

这本 notebook 做三件事：用偏好对算出训练损失，在留出数据上选一个校准温度，再用三项门槛审计“奖励最高的候选”是不是真的质量合格。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 奖励模型（Reward Model） | 给每条回答打一个分数的模型，分数越高表示它认为回答越好 |
| 偏好对 | 同一个提示词下的两个回答，一个被选中（`chosen`），一个被淘汰（`rejected`） |
| 分数差（margin） | 优选分数减非优选分数。正数表示排序方向正确 |
| Bradley–Terry 成对模型 | 把分数差换算成“优选回答胜出的概率”的公式，分数差越大概率越接近 1 |
| 校准温度（temperature） | 一个正数，除在分数差上，用来调整置信度的高低，不改变排序。它在留出集（没参与训练的数据）上选出来 |
| Brier 误差 | 预测概率和标签（1 或 0）之差的平方再取平均，越小说明概率越可信 |
| 代理奖励 / 隐藏质量 | 代理奖励是奖励模型给的分数；隐藏质量是另外冻结的质量指标，拿来和代理奖励对照 |

## 核心思路

整条链路有三个检查点。第一个在训练时：同一提示词下的优选和非优选回答经过同一个奖励模型，得到两个分数，用分数差算损失。第二个在训练后：拿没参与训练的留出偏好对试几个温度，挑 Brier 误差最小的那个。第三个在使用前：对策略产生的候选回答，检查奖励最高的那个质量是否合格。公式如下：

```text
Δr_i = r(x_i, y_i+) - r(x_i, y_i-)          # 分数差：优选分数减非优选分数
p_i  = sigmoid(Δr_i)                        # 优选回答胜出的概率
L    = -(1/N) * Σ log sigmoid(Δr_i) = (1/N) * Σ softplus(-Δr_i)   # 成对损失
p_hat_i(T) = sigmoid(Δr_i / T)              # 校准概率，T 是正的温度
Brier(T)   = (1/M) * Σ (p_hat_i(T) - z_i)^2 # 校准误差，z_i 是标签 0 或 1
```

符号说明：`x_i` 是第 `i` 个偏好对的提示词，`y_i+` 和 `y_i-` 是它的优选、非优选回答，`r` 是奖励模型，`N` 是批次里的偏好对数，`M` 是校准集的样本数，`T` 是温度，`z_i` 是第 `i` 个样本的标签。分数差为 0 时概率是 0.5；分数差越大，损失越小。下面用 notebook 测试里的数字算一遍：

```python
import torch
import torch.nn.functional as F

chosen, rejected = torch.tensor([2.0, 1.2]), torch.tensor([0.5, -0.2])
margin = chosen - rejected                      # 优选分数减非优选分数
pair_losses = F.softplus(-margin)               # 每个偏好对的损失
print(margin, pair_losses, pair_losses.mean(), (margin > 0).float().mean())
print((chosen + 10) - (rejected + 10), F.softplus(-(chosen - rejected)).mean())   # 两个分数一起加 10

labels = torch.tensor([1.0, 1.0, 0.0, 0.0])     # 留出偏好对的标签
for T in (0.25, 0.5, 1.0, 2.0):                 # 候选温度
    prob = torch.sigmoid(torch.tensor([0.2, 0.4, -0.3, -0.8]) / T)
    print(T, round(torch.mean((prob - labels) ** 2).item(), 4))   # Brier 误差

ids, rewards, quality = ["base", "a", "b"], [1.0, 3.0, 2.0], [0.8, 0.5, 0.7]
top = rewards.index(max(rewards))               # 奖励最高的候选
print(ids[top], quality[top] < 0.6, quality[0] - quality[top] > 0.1)
disagree = sum((rewards[i] - rewards[j] > 0) != (quality[i] - quality[j] > 0)
               for i in range(3) for j in range(i + 1, 3))
print(disagree, disagree / 3 > 0.2)
```

```text
tensor([1.5000, 1.4000]) tensor([0.2014, 0.2204]) tensor(0.2109) tensor(1.)
tensor([1.5000, 1.4000]) tensor(0.2109)
0.25 0.0449
0.5 0.1027
1.0 0.1602
2.0 0.2008
a True True
3 True
```

前两行是训练目标。两个偏好对的分数差是 1.5 和 1.4，都是正数，所以排序准确率是 1；逐对损失是 0.2014 和 0.2204，平均 0.2109。把四个分数一起加 10，分数差和损失完全不变。成对损失只认同一偏好对里的相对顺序，分数的绝对大小不能只靠训练损失来解释。

中间四行是校准。留出集上的 4 个分数差配上标签 1、1、0、0，四个候选温度里 0.25 的 Brier 误差最小，所以选 0.25。温度只改置信度，不改排序。

最后两行是过度优化审计。3 个候选里奖励最高的是 `a`，但它的质量 0.5 低于下限 0.6，比基线 `base` 的 0.8 掉了 0.3（超过 0.1），而且 3 对候选的奖励排序和质量排序全部相反。三项门槛都失败，这个候选要被拒绝。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 同一提示的优选和非优选回答怎样经过同一个奖励模型变成训练信号 | 读 |
| Step 2 | 分数差怎样换算成胜出概率和成对损失，为什么只认相对顺序 | 读，对照上面的公式 |
| Step 3 | 留出排序、校准温度、隐藏质量三道门槛各自查什么 | 读 |
| Step 4 · TODO 1 | 计算成对奖励目标 | 写 `pairwise_reward_objective`：补 3 个变量 `score_margin`、`pair_losses`、`accuracy` |
| Step 4 · TODO 2 | 选择留出校准温度 | 写 `select_reward_temperature`：补 1 行 `best_record`，按 Brier 误差取最小，并列时取温度更小的 |
| Step 4 · TODO 3 | 审计奖励过度优化 | 写 `audit_reward_overoptimization`：补 3 个判断 `absolute_quality_failed`、`quality_drop_failed`、`rank_disagreement_failed` |
| 测试 | 检查成对目标的方向和共同平移、温度选择的并列规则和非法候选，以及代理奖励与隐藏质量冲突时的拒绝条件 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 让受控评分器返回指定奖励，调用 TRL 的 `RewardTrainer.compute_loss`，核对成对损失和批次均值 | 需要 `PT-CPU` 环境 |
| Step 5（可选） | 冻结真实模型主干，只训练标量奖励头，用正确标签和固定错标的负对照做比较 | 需要 GPU 和 `PT-RL-CUDA` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。框架核对需要 `PT-CPU` 环境。Step 5 需要 GPU，默认不执行真实实验。

## 容易踩的坑

- **分别训练了两个打分器**：同一提示词下的两个回答必须经过同一个奖励模型，分数才可比。也不能拿不同提示词的原始分数来比较回答质量。
- **拿原始奖励值跨提示词比较**：成对损失只认偏好对内的相对顺序，两个分数一起平移损失不变。分数的尺度要靠留出校准来定，不能从训练集损失直接推出。
- **用训练数据选温度**：温度要在独立的留出集上选，校准数据不能参与奖励模型训练。温度必须是正数，Brier 误差并列时选更小的温度。
- **只看绝对质量门槛**：审计要同时查三项：奖励最高候选的绝对质量、相对基线的质量退化、奖励排序与隐藏质量排序的背离。每一项失败都要单独保留原因。
- **把三道门槛通过当成可以采用**：门槛通过只说明奖励信号可以继续试验，不等于候选模型已经可以采用。出现奖励上升但质量下降时，要停止扩大更新并回查偏好覆盖、评分器版本和失败样本。

## 读完后自查

- [ ] 能解释为什么四个分数一起加 10，成对损失不变。
- [ ] 能说出分数差为 0 时，优选回答的胜出概率是多少。
- [ ] 能说出校准温度改变什么、不改变什么。
- [ ] 能说出审计“奖励最高的候选”时要检查哪三项。

## 下一步

- 打开 notebook：[PT-REWARD-MODEL. 奖励模型训练、校准与过度优化](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb)
- 建议先读：[50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation_Beginner.md)
- 做完之后：[16. RLHF / PPO 资源与状态 · 小白入门](./16_RLHF_PPO_Memory_Beginner.md)
