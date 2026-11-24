# 18. GRPO Loss Tutorial | 组相对策略优化与可验证奖励 · 小白入门

> 配套 notebook：[18. 组相对策略优化与可验证奖励](../18_GRPO_Loss_Tutorial.md) ｜ 难度：Medium-Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

让模型对同一个问题写出几个回答并分别打分，组相对策略优化根据每个回答比同组平均分好多少，来决定提高还是降低它出现的概率，并且限制每次改动的幅度。

## 为什么需要它

组相对策略优化（GRPO）训练模型时不给标准答案，只给分数，这个分数叫奖励（reward）。比如一道题，让模型写出 4 个候选回答，检查程序判定 2 个对、2 个错。你希望模型以后多写对的那两种，少写错的那两种。这里的“策略”（policy）就是正在训练的模型。要做到这一点，得回答三个问题：

- **分数怎么变成方向。** 不同题的分数大小不一样：有的题几个候选都接近满分，有的题都接近 0 分。直接拿分数当更新力度，分数大的题会盖过分数小的题。GRPO 只在同一个问题的候选之间比较，这个“比同组平均分好多少”的数叫优势（advantage）：正数提高概率，负数降低概率。
- **一次能改多少。** 好回答的概率如果一步就拉高很多，就是单步更新过强，所以每次更新都限制在采样时的概率附近。
- **改久了会不会跑偏。** 更新很多步以后，模型可能离原来的样子越来越远，出现奖励投机或能力退化，所以要和一份固定不动的参考策略（Reference）比较。

另外，GRPO 和可验证奖励强化学习（RLVR）不是一回事。前者规定拿到分数以后怎么更新模型，后者规定分数从哪里来，比如答案检查。

分数本身也得可信：组内排名只说明哪个候选相对更好，说明不了模型真的学会了任务。这本 notebook 把“候选组、验证器（verifier）奖励、组内优势、受约束更新”这条链拆开讲，最后让你在 CPU 上把它写出来。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 策略（policy） | 正在训练的模型。它决定生成每个词元的概率，训练就是在调整这些概率 |
| 候选（candidate） | 模型对同一个提示词（prompt）写出的一个回答，几个候选组成一组 |
| 奖励（reward） | 给每个候选打的分，可以来自奖励模型、规则验证器或环境 |
| 优势（advantage） | 一个候选比同组平均水平好多少，正数提高它的概率，负数降低 |
| 对数概率（log prob） | 概率取对数后的数。两个概率相除等于对数概率相减，所以代码里都用它 |
| 策略比率（ratio） | 当前策略和采样时的旧策略给同一个词元的概率之比，等于 1 表示没变 |
| 参考策略（Reference） | 一份固定不动的模型，用来限制当前策略离它太远 |

## 核心思路

输入是每个候选的奖励 `rewards`、组编号 `group_ids`，以及三张 `[候选数, 序列长度]` 的表：当前策略、旧策略和参考策略给每个词元的对数概率。处理分四步：

```text
组内优势     A_i = (r_i - mu_g) / max(sigma_g, eps)
策略比率     rho_{i,t} = exp( log_new(y_{i,t}) - log_old(y_{i,t}) )
裁剪目标     s_{i,t} = min( rho_{i,t} * A_i,  clip(rho_{i,t}, 1 - epsilon, 1 + epsilon) * A_i )
参考策略项   KL_{i,t} = exp(d_{i,t}) - d_{i,t} - 1,  其中 d_{i,t} = log_ref(y_{i,t}) - log_new(y_{i,t})
聚合         损失 = 候选之间求平均( 候选内部求平均( -s_{i,t} + beta * KL_{i,t} ) )
```

`i` 是候选编号，`t` 是词元位置，`y_{i,t}` 是实际生成的词元。`r_i` 是奖励，`mu_g` 和 `sigma_g` 是第 `g` 组奖励的平均值和标准差。`eps` 是防止除以 0 的小正数，`epsilon` 在代码里叫 `clip_range`（默认 0.2），`beta` 是参考策略项的权重，默认 0 表示不启用。

下面把第一步和第二步放在一个小例子里：

```python
import torch

eps = 1e-6
rewards = torch.tensor([0.0, 1.0, 1.0, 0.0])   # 同一个提示词下 4 个候选的奖励
centered = rewards - rewards.mean()
scale = rewards.std(unbiased=False).clamp_min(eps)
advantages = centered / scale
print(centered, scale)
print(advantages)

same = torch.tensor([1.0, 1.0])                 # 两个候选奖励相同
print((same - same.mean()) / same.std(unbiased=False).clamp_min(eps))

clip_range = 0.2
log_new = torch.tensor([-0.5, -1.5])   # 当前策略给两个词元的对数概率
log_old = torch.tensor([-1.0, -1.0])   # 采样时旧策略的对数概率
ratio = torch.exp(log_new - log_old)
clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
advantage = torch.tensor([1.0, -1.0])  # 左边是好候选，右边是坏候选
print(ratio, clipped)
print(torch.minimum(ratio * advantage, clipped * advantage))
```

```text
tensor([-0.5000,  0.5000,  0.5000, -0.5000]) tensor(0.5000)
tensor([-1.,  1.,  1., -1.])
tensor([0., 0.])
tensor([1.6487, 0.6065]) tensor([1.2000, 0.8000])
tensor([ 1.2000, -0.8000])
```

前两行是第一步的组内优势：每个候选减去本组平均奖励，再除以本组标准差。好回答是正的，坏回答是负的，整组平均是 0。第三行是得分相同的两个候选，优势全是 0，不产生更新方向。

第四、五行是第二步的策略比率，也就是当前策略生成这个词元的概率是采样时的多少倍。好候选的概率涨到 1.65 倍，超过上限 1.2；坏候选的概率降到 0.61 倍，低于下限 0.8。第三步的裁剪（clip）目标在“没裁剪”和“裁剪后”两个结果里取较小的那个，所以这两个词元都按裁剪后的比率算。

参考策略项衡量当前策略和参考策略差多远，不会是负数，差得越远越大。把它乘上 `beta` 加进损失，用来限制长期漂移。

第四步是求平均。回答掩码（`response_mask`）标出哪些位置是真正的回答，提示词和填充（padding）不算。每个候选先在自己的有效词元上求平均，再让每个候选等权。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 同一输入的候选如何变成相对训练信号：GRPO 管怎么更新，RLVR 管奖励从哪来 | 读 |
| Step 2 | 组内优势、策略裁剪与参考策略如何共同限制更新 | 读，对照上面的四步公式 |
| Step 3 | 把组内排名与真实质量分开验证：每个奖励分量要配什么证据 | 读 |
| Step 4 · TODO 1 | 逐组计算中心化 reward、稳定尺度和相对 advantage | 在 `compute_group_advantages` 里补 3 行：减去组均值、求组内标准差并加下限、相除后写回 `advantages` |
| Step 4 · TODO 2 | 冻结旧策略记录，计算词元级策略比率及其裁剪值 | 在 `compute_grpo_loss` 里补 2 行：`ratio` 和 `clipped_ratio` |
| Step 4 · TODO 3 | 广播候选优势，构造裁剪目标与可选 Reference KL | 在 `compute_grpo_loss` 里补全 `token_advantages` 到 `per_token_loss` 这 5 个变量，参考策略项另加 3 行 |
| Step 4 · TODO 4 | 每个候选先按有效词元求均值，再在候选间等权聚合 | 在 `compute_grpo_loss` 里补 3 行：`token_counts`、`per_candidate` 和 `loss` |
| 测试 | 四个测试检查输入契约、组内优势、裁剪与参考策略约束、梯度隔离和候选等权 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用受控的逐词元对数概率调用 TRL 的标准 `grpo` 损失路径，核对奖励、组内优势、裁剪、参考策略项和归约 | 需要 `PT-CPU` 环境，可以先跳过 |
| Step 5（可选） | 让验证器给固定候选逐项打分，在 GPU 上比较最小候选组和扩展候选组的组内优势与有限更新 | 需要 GPU 和 `PT-RL-CUDA` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。框架核对需要 `PT-CPU` 环境。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **旧策略或参考策略接收了梯度**：`log_probs_old` 和 `log_probs_ref` 是固定的对照记录，计算时要 `.detach()`，测试会检查它们的梯度是 `None`。
- **奖励全同时凭空造出优势**：候选得分相同，或者组里只有一个候选，优势都应该是 0。标准差要用 `clamp_min(eps)` 加下限，避免除以接近 0 的数。
- **按词元数分配候选的权重**：长回答不能因为词元多就占更大权重。提示词和填充位置不算，先在每个候选内部按有效词元数求平均，再对候选求平均。
- **参考策略项的方向写反**：损失是“负的裁剪目标加 `beta` 倍参考策略项”，带这项的损失不应小于不带的。

## 读完后自查

- [ ] 能说出 GRPO 和 RLVR 各回答什么问题。
- [ ] 给一组奖励 `[0, 1, 1, 0]`，能说出每个候选的优势是多少。
- [ ] `clip_range` 是 0.2 时，比率 1.65 会被裁剪成多少，为什么。
- [ ] 能说明 `log_probs_old` 和 `log_probs_ref` 为什么要用 `.detach()`。

## 下一步

- 打开 notebook：[18. 组相对策略优化与可验证奖励](../18_GRPO_Loss_Tutorial.md)
- 建议先读：[15. SFT 有效更新与训练闭环 · 小白入门](./15_End_to_End_Fine_Tuning_Experiment.md)、[16. RLHF / PPO 资源与状态 · 小白入门](./16_RLHF_PPO_Memory.md)
- 做完之后：[PT-RLOO. 留一基线策略优化 · 小白入门](./PT-RLOO_Leave_One_Out_Policy_Optimization.md)
