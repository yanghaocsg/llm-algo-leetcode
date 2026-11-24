# 16. RLHF PPO Memory | RLHF / PPO 资源与状态 · 小白入门

> 配套 notebook：[16. RLHF / PPO 资源与状态](../16_RLHF_PPO_Memory.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

PPO 是一种让模型根据奖励小幅调整生成倾向的训练方法。这本 notebook 先讲清一轮 PPO 要用到哪几个模型，再带你实现其中最核心的一步：把每次更新限制在安全范围内。

## 为什么需要它

监督微调之后的模型已经会模仿数据里的回答，但还不知道在几个都行得通的答案里更偏向哪一个。基于人类反馈的强化学习（RLHF）用近端策略优化（PPO）来做这件事：策略先生成回答，再依据奖励调整生成倾向。

一轮 PPO 里有四个模型角色同时在场：

- 策略模型（Actor）：生成回答，接收策略梯度。
- 参考模型（Reference）：冻结不动，提供对照，限制策略的长期偏移。
- 奖励模型：给回答打分，本身不更新。
- 价值模型（Critic）：估计从某个状态往后大概能拿多少回报，当作比较的基线，通常也要训练。

有了奖励，更新也不能一步迈得太大。PPO 的办法是先看“当前策略给这个词元的概率，是采样时的多少倍”，倍数超出范围，就不再放大这次的更新。

这本 notebook 只实现策略模型这一支的损失，也就是这个限制步幅的裁剪目标。Step 3 另外讲了一轮训练的每个阶段有哪些状态要占着显存。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| Actor / Reference / Reward Model / Critic | 一轮 PPO 的四个模型角色：生成回答的策略模型、冻结的参考模型、打分的奖励模型、估计回报基线的价值模型 |
| 行为策略记录 | 采样时保存下来的旧策略对数概率。它只是一份记录，不是第五个模型 |
| 优势（advantage） | 这次动作比基线好多少。正数表示该加强，负数表示该减弱 |
| 策略比率（ratio） | 当前策略给某个词元的概率，除以采样时旧策略给它的概率 |
| 裁剪（clip） | 把比率限制在 `1 ± clip_range` 里，默认是 0.8 到 1.2 |
| 近似 KL（`approx_kl`） | 当前策略和旧策略差了多远的一个估计，只用来观察 |
| 裁剪比例（`clip_fraction`） | 有效词元里被裁剪掉的比例，只用来观察 |

## 核心思路

输入是三个同形状的二维张量（行是回答，列是词元）：当前策略的对数概率、采样时旧策略的对数概率、优势，外加一个回答掩码。分四步：算比率，裁剪并取保守目标，按口径聚合成一个损失，再算几个诊断量。输出是一个标量损失和三个诊断值。

```text
rho_t     = exp(log pi_new - log pi_old)            # 策略比率：当前概率是旧概率的多少倍
surrogate = rho_t * A_t                             # 没有裁剪的目标
clipped   = clip(rho_t, 1 - eps, 1 + eps) * A_t     # 把比率限制在 1 ± eps 之内
L_clip    = min(surrogate, clipped)                 # 两者取小
loss      = -mean(L_clip)                           # 训练时最小化它的相反数
```

符号说明：`pi_new` 是当前策略给已采样词元的概率，`pi_old` 是采样时旧策略给同一批词元的概率，两者取对数后相减、再取指数，就是概率之比。`A_t` 是优势，`eps` 是裁剪范围，代码里叫 `clip_range`，默认 0.2。Step 2 还给了完整 PPO 的总损失，里面有价值损失、熵项和 KL 项，Step 4 不实现它们。下面用 notebook 测试里的数字算一遍：

```python
import torch

eps = 0.2
ratio = torch.exp(torch.log(torch.tensor([2.0, 0.5, 1.0])))   # 当前概率是旧概率的多少倍
advantage = torch.tensor([1.0, -1.0, 1.0])
surrogate = ratio * advantage                               # 没有裁剪的目标
clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage   # 把比率限制在 [0.8, 1.2]
conservative = torch.minimum(surrogate, clipped)             # 两者取小
print(ratio, surrogate, clipped)
print(conservative, -conservative.mean())

targets = torch.tensor([[1.0, 1.0], [3.0, 0.0]])      # 两行是两条回答的目标
mask = torch.tensor([[True, True], [True, False]])    # 第二条回答只有 1 个有效词元
per_sequence = (targets * mask).sum(-1) / mask.sum(-1)   # 先在每条回答内部平均
print(-targets[mask].mean(), -per_sequence.mean())
```

```text
tensor([2.0000, 0.5000, 1.0000]) tensor([ 2.0000, -0.5000,  1.0000]) tensor([ 1.2000, -0.8000,  1.0000])
tensor([ 1.2000, -0.8000,  1.0000]) tensor(-0.4667)
tensor(-1.6667) tensor(-2.)
```

第一行是比率、没有裁剪的目标和裁剪后的目标。第 1 个词元优势为正，比率 2.0 超过上界，裁剪后的目标是 1.2。第 2 个词元优势为负，比率 0.5 低于下界：没裁剪的目标是 -0.5，裁剪后是 -0.8。第 3 个词元比率是 1，两种目标相同。

第二行是逐点取小之后的结果和损失。第 2 个词元取了更小的 -0.8。正优势和负优势都一样，越界的更新不会被放大。三个词元取平均再取相反数，损失是 -0.4667。

第三行是两种聚合口径。两条回答，第一条有 2 个有效词元，第二条只有 1 个。按词元平均，3 个词元权重相同，损失是 -1.6667。按序列平均，先在每条回答内部平均，再让两条回答各占一半，损失是 -2。两种口径结果不同，要固定用一种。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一轮 PPO 需要哪四个模型角色，各自产出什么、是否更新、占哪些状态 | 读 |
| Step 2 | 奖励怎样变成优势，策略比率和裁剪目标怎样限制更新步幅 | 读，对照上面的公式 |
| Step 3 | 采样、评分、优势估计、策略更新四个阶段各有哪些状态要驻留 | 读 |
| Step 4 · TODO 1 | 冻结旧策略记录，从当前与旧策略的对数概率差构造词元级 `ratio` | 写 `compute_actor_loss`：补 1 行 `ratio`，对数概率相减后取指数，旧策略要 `detach` |
| Step 4 · TODO 2 | 冻结 `advantage`，构造原始、裁剪和保守目标 | 写 `compute_actor_loss`：补 3 行 `surrogate`、`clipped_surrogate`、`conservative` |
| Step 4 · TODO 3 | 按 `reduction` 口径聚合有效回答词元 | 写 `compute_actor_loss`：两个分支各补 1–3 行，`token_mean` 对掩码内取平均，`sequence_mean` 先按每条回答的有效词元数归一化 |
| Step 4 · TODO 4 | 在有效位置统计近似 KL、裁剪比例和词元数 | 写 `compute_actor_loss`：补 4 个变量 `log_ratio`、`approx_kl`、`clip_fraction`、`valid_tokens`，都从 `detach` 后的张量算 |
| 测试 | 检查比率和裁剪目标、正负优势的裁剪边界、两种聚合口径、诊断值、梯度只回到当前策略，以及非法输入 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把同一组对数概率和优势交给 TorchRL 的 `ClipPPOLoss`，核对裁剪目标和裁剪比例 | 需要 `PT-CPU` 环境（含 TorchRL） |

整本 notebook 只需要 CPU。框架核对需要带 TorchRL 的 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **旧策略或优势接收了梯度**：旧策略的对数概率和优势都是固定的记录，要 `detach()`。梯度只回到当前策略的对数概率。
- **裁剪后没有取小**：要保留优势的正负方向，原始目标和裁剪目标要逐点取较小值。只用裁剪后的值，负优势那一侧的边界就错了。
- **两种聚合口径写成了同一个结果**：`token_mean` 是掩码内直接取均值；`sequence_mean` 先按每条回答的有效词元数归一化，再对回答取平均。变长回答下两者不同。
- **提示词和填充位置算进了损失**：只聚合掩码为 True 的位置。每条回答至少要有一个有效位置，否则报错。
- **诊断量带进了梯度**：近似 KL 和裁剪比例只用来观察，要从 `detach` 后的张量计算，不进入损失。

## 读完后自查

- [ ] 能说出一轮 PPO 的四个模型角色，以及哪些会被更新。
- [ ] 比率 2.0、优势 1、裁剪范围 0.2 时，能说出裁剪后的目标是多少。
- [ ] 能解释为什么原始目标和裁剪目标要取较小值。
- [ ] 能说出 `token_mean` 和 `sequence_mean` 在变长回答下的区别。

## 下一步

- 打开 notebook：[16. RLHF / PPO 资源与状态](../16_RLHF_PPO_Memory.ipynb)
- 建议先读：[15. SFT 有效更新与训练闭环 · 小白入门](./15_End_to_End_Fine_Tuning_Experiment.md)
- 需要补基础：[PyTorch 优化器与损失函数](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)、[数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
- 做完之后：[18. 组相对策略优化与可验证奖励 · 小白入门](./18_GRPO_Loss_Tutorial.md)、[17. 直接偏好优化损失](../17_DPO_Loss_Tutorial.ipynb)
