# PT-RLOO. Leave One Out Policy Optimization Beginner | 留一基线策略优化 · 小白入门

> 配套 notebook：[PT-RLOO. 留一基线策略优化](./PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

给同一个问题让模型写出几个回答并分别打分，留一基线策略优化在评价其中一个回答时，把其他回答的平均分当作参照，再克制地更新模型。

## 为什么需要它

这类训练只给回答打分，不给标准答案。可是单个分数说明不了问题：有的题随便写都是高分，有的题写得再好分也不高。所以每条回答都要和一个参照比，这个参照叫基线（baseline）。回答比基线好，就提高它出现的概率；比基线差，就降低。导读里说的“低方差”，就是让用作更新信号的数波动更小。

同一个提示词（prompt）下，让模型写出 3 条回答，奖励分别是 1、2、4。参照从哪里来？常见的做法有三种：

- 另外训练一个价值模型（Critic）来估计参照。
- 用 3 条回答的平均分，包括当前这条。GRPO 常这样做，但同一条回答的分数会同时出现在被评价项和参照里，RLOO 要避免这一点。
- 用除当前回答外另外两条的平均分。这就是留一（leave-one-out），是 RLOO 的做法。

真实数据还有几种情况要处理：奖励函数可能给不出分；一组里可能只有一条回答拿到了分；不同提示、不同奖励定义或不同策略版本的回答不能混进同一组。这本 notebook 把这些规则和后面的裁剪更新一起写成四个函数，让你在 CPU 上实现。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 基线（baseline） | 拿来比较的参照分。回答的奖励减去基线，才知道它比一般水平好多少 |
| 留一（leave-one-out） | 算某条回答的基线时，把它自己从组里拿掉，只用其余回答 |
| 价值模型（Critic） | 单独训练的一个模型，用来估计参照。RLOO 不需要它 |
| 可评分回答（scorable） | 拿到了有效奖励的回答。奖励函数返回缺失值时，这条回答不可评分 |
| 序列得分（sequence logps） | 一条回答里所有有效词元的对数概率加起来，一条回答一个数 |
| 塑形奖励（shaped reward） | 外部奖励减去参考策略惩罚以后的奖励，留一优势在它上面计算 |
| 策略比率（ratio） | 当前策略和采样时的旧策略给同一条回答的概率之比 |

## 核心思路

RLOO 一次更新分四步。输入是三张词元对数概率表（当前策略、采样时的旧策略、固定的参考策略各一张）、回答掩码、外部奖励和组编号：

```text
序列得分   s_i = 回答 i 的有效词元对数概率之和
塑形奖励   R_i = 外部奖励_i - beta * k_i，  k_i = 回答 i 上 (log_old - log_ref) 的和
留一优势   b_i = (组内可评分奖励之和 - R_i) / (可评分回答数 - 1)，  A_i = R_i - b_i
裁剪目标   rho_i = exp(s_new_i - s_old_i)
           L_i = -min( rho_i * A_i, clip(rho_i, 1 - epsilon_low, 1 + epsilon_high) * A_i )
```

`i` 是回答编号。外部奖励是奖励函数从外面给出的分，可以由几个分量加权相加得到。`log_old` 和 `log_ref` 是旧策略和参考策略给每个词元的对数概率，`beta` 是惩罚的权重。`b_i` 是留一基线，`A_i` 是回答级优势。`epsilon_low` 和 `epsilon_high` 是下侧和上侧的裁剪范围，两者可以不相等，所以叫非对称裁剪。

下面用小数字看序列得分和留一优势：

```python
import torch

token_logps = torch.tensor([[-1.0, -2.0, -float("inf")]])   # 最后一个位置是填充占位
mask = torch.tensor([[True, True, False]])
print(token_logps.masked_fill(~mask, 0.0).sum(dim=-1))

def leave_one_out(rewards):
    ok = ~torch.isnan(rewards)                          # 可评分的回答
    count = ok.sum()
    if count < 2:                                       # 没有可比较的对象
        return torch.zeros_like(rewards)
    baselines = (rewards[ok].sum() - rewards[ok]) / (count - 1)
    out = torch.zeros_like(rewards)
    out[ok] = rewards[ok] - baselines
    return out

rewards = torch.tensor([1.0, 2.0, 4.0])                 # 同一提示词下 3 条回答的奖励
print(rewards - rewards.mean())                         # 对照：不排除自己的组均值
print(leave_one_out(rewards))
print(leave_one_out(torch.tensor([1.0, float("nan"), 4.0])))
print(leave_one_out(torch.tensor([1.0, float("nan")])))
```

```text
tensor([-3.])
tensor([-1.3333, -0.3333,  1.6667])
tensor([-2.0000, -0.5000,  2.5000])
tensor([-3.,  0.,  3.])
tensor([0., 0.])
```

第一行是序列得分：填充位置先清零再求和，负无穷的占位值不会传进结果。

第二行是对照，用包含自己的组均值当基线。第三行是留一：评价第一条时只用 2 和 4 的平均 3.0 当基线，1.0 − 3.0 = −2.0。两种算法得到的数不一样，所以 notebook 强调两者的基线不能互换。

第四行里中间那条回答的奖励缺失。它不进入另外两条的基线，自己的优势保持 0。第五行里只有一条可评分回答，没有可比较的对象，优势都是 0。

参考策略惩罚 `k_i` 可能是负数，notebook 提醒不要和 GRPO 里非负的参考策略项混为一谈：GRPO 把那一项加在损失里，RLOO 把它从奖励里扣掉。策略比率是序列级的，一条回答只有一个比率。裁剪时取“没裁剪”和“裁剪后”两个结果里较小的那个：优势为正时主要限制概率上升过快，优势为负时主要限制概率下降过快。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 同一提示的回答如何组成可比较的采样组：留一基线和回答级优势的公式 | 读 |
| Step 2 | 外部奖励和 Reference KL 如何形成塑形奖励，以及和 PPO、GRPO 的基线对照 | 读，对照上面的公式 |
| Step 3 | 留一优势如何进入受约束更新：序列级策略比率和非对称裁剪 | 读 |
| Step 4 · TODO 1 | 聚合有效回答词元 | 在 `aggregate_sequence_logps` 里补 1 行：无效位置清零后沿最后一维求和 |
| Step 4 · TODO 2 | 塑形回答级奖励 | 在 `shape_rloo_rewards` 里补 2 行：`sequence_kl` 和 `shaped_rewards` |
| Step 4 · TODO 3 | 计算缺失奖励感知的留一优势 | 在 `leave_one_out_advantages` 里补 2 行：`baselines` 和 `group_advantages` |
| Step 4 · TODO 4 | 构造非对称裁剪目标 | 在 `rloo_policy_loss` 里补 4 行：`ratios`、`fixed_advantages`、`clipped` 和 `losses` |
| 测试 | 检查序列得分、奖励塑形、留一优势、单一可评分回答和裁剪损失的梯度边界 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 向 TRL 的 `RLOOTrainer._compute_loss` 提供受控的对数概率和优势，核对序列求和、策略比率与裁剪目标 | 需要 `PT-CPU` 环境，可以先跳过 |

Step 1–4 只需要 CPU。框架核对需要另外的 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **当前回答进了自己的基线**：基线只能是同组其他可评分回答的均值，算法是“组和减去自己，再除以可评分数减 1”。不同提示的回答也不能混进同一组。
- **把缺失奖励当成 0 或普通分数**：缺失只能用 `NaN` 表示。缺失的回答不进别人的基线，自己的优势保持 0；塑形以后它也要保持缺失，不能被惩罚项凑成一个有效奖励。
- **填充（padding）位置进了序列得分**：序列比率只覆盖回答词元，提示词、填充和占位值都不能进分数。要先把无效位置清零再求和。
- **只有一条可评分回答还算优势**：少于两条可评分回答就没有留一参照，优势应该是 0。
- **旧策略、参考策略或优势带了梯度**：它们都是采样后固定的记录，只有当前策略的得分参与更新，其余都要 `.detach()`。

## 读完后自查

- [ ] 能说出 RLOO 的基线和 GRPO 的组均值基线差在哪里。
- [ ] 给一组奖励 `[1, 2, 4]`，能算出每条回答的留一优势。
- [ ] 一组奖励是 `[1, NaN, 4]` 时，能说出三条回答各自的优势。
- [ ] 能说明为什么旧策略得分、参考策略得分和优势都要 `.detach()`。

## 下一步

- 打开 notebook：[PT-RLOO. 留一基线策略优化](./PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb)
- 建议先读：[PT-REWARD-SIGNAL. 奖励、价值与优势信号 · 小白入门](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage_Beginner.md)、[18. 组相对策略优化与可验证奖励 · 小白入门](./18_GRPO_Loss_Tutorial_Beginner.md)、[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness_Beginner.md)
- 做完之后：[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness_Beginner.md)
