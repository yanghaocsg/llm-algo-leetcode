# 56. Continual Pretraining and Retention | 持续预训练与能力保留 · 小白入门

> 配套 notebook：[56. 持续预训练与能力保留](../56_Continual_Pretraining_and_Retention.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

持续预训练是拿一个新领域的文本接着训练已有模型。这本 notebook 教你同时检查两件事：模型在新领域上有没有变好，原有能力有没有变差，然后决定继续、调整还是回退。

## 为什么需要它

你手上已经有一个训练好的语言模型，想让它更懂某个新领域。做法很直接：找来这个领域的文本，接着训练，这就是持续预训练。麻烦出在训练之后：这一轮值不值得？训练损失只能说明模型对参与更新的数据拟合得怎样，回答不了另外两个问题：

- 模型在新领域的文本上真的变好了，还是只是记住了训练时看过的那几句？
- 模型原来会的通用能力，有没有在这次训练中变差？

比如更新之后，领域评测损失从 2.30 降到 2.10，保留集上的损失却从 1.50 涨到 1.58。如果你事先只允许它涨 0.05，这一轮就不能照单全收。

这本 notebook 把判断拆成三步：损失怎么算才对，三份数据怎么确认没有混用，最后按固定顺序给出继续、调整或回退的建议。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 持续预训练（continual pretraining） | 拿一个已经训练好的模型，用某个新领域（domain）的文本接着训练 |
| 全序列监督 | 一条文本里每个真实词元都用来算损失；监督微调（SFT）只让回答部分算损失 |
| 词元（token） | 模型处理文本的最小单位，文本先被切成词元，再变成编号 |
| 填充（padding） | 为了把短样本补到统一长度加上的无效位置，在掩码里记为 0 |
| 保留集（retention set） | 一份不参与训练的通用文本，专门检查模型原来的能力有没有变差 |
| 检查点（checkpoint） | 训练途中存下来的一整套训练状态，用来中断后接着练，或者退回去 |

## 核心思路

题目区的输入有四样：词元编号和掩码、模型给出的预测分数（logits）、三份数据各自的样本编号、更新前后测到的损失。

先看损失。模型读完前 `t` 个词元就要预测第 `t + 1` 个，每个非填充位置都算一次。Step 1 的公式是 `L_CLM = -(1 / |V|) * Σ log p(x[t+1] | x[1..t])`，对集合 `V` 里的每个位置 `t` 求和。

符号说明：`V` 是有效预测位置的集合，也就是非填充的位置，`|V|` 是它的个数；`x[1..t]` 是当前位置及以前的上下文，`x[t+1]` 是要预测的下一个词元；`p` 是模型给这个下一个词元的概率；`L_CLM` 是各位置“对数概率取负”后的平均值，也就是损失。用一条长度为 4 的序列看标签怎么来（对应 TODO 1 和 TODO 2）：

```python
import torch

input_ids = torch.tensor([[5, 6, 7, 0]])
attention_mask = torch.tensor([[1, 1, 1, 0]])
labels = input_ids.clone()
labels[attention_mask == 0] = -100          # 填充位置不参与损失
print(labels.tolist())
shift_labels = labels[:, 1:]                # 位置 t 的预测对准位置 t+1 的标签
valid = shift_labels.ne(-100)
print(shift_labels[valid].tolist(), int(valid.sum()))
```

```text
[[5, 6, 7, -100]]
[6, 7] 2
```

标签和输入一样，只有填充位置变成了 `-100`。算损失时预测和标签要错开一位：位置 0 的预测对标签 6，位置 1 的预测对标签 7，位置 2 的预测对到 `-100`，被忽略。所以有效目标有 2 个，损失是这两个位置的损失之和除以 2。

再看判断（对应 TODO 3 和 TODO 4）。更新前后，在同一份领域评测集和同一份保留集上各测一次损失，再相减：`delta_domain = L_domain(after) - L_domain(before)`，`delta_retain = L_retain(after) - L_retain(before)`。前者小于 0 表示领域上变好，后者大于 0 表示原有能力退化。最低门槛是 `delta_domain < 0` 且 `delta_retain <= tau_retain`，其中 `tau_retain` 是事先定好的上限：保留集上的损失最多允许涨多少，不能是负数。

```python
before = {'domain': 2.30, 'retain': 1.50}
after = {'domain': 2.10, 'retain': 1.58}
delta_domain = round(after['domain'] - before['domain'], 4)
delta_retain = round(after['retain'] - before['retain'], 4)
tau_retain = 0.05
print(delta_domain, delta_retain)
print('domain_gain:', delta_domain < 0, '| retention_ok:', delta_retain <= tau_retain)
```

```text
-0.2 0.08
domain_gain: True | retention_ok: False
```

领域损失降了 0.2，保留集损失却涨了 0.08，超过了上限 0.05。notebook 的测试里就有这组数字，对应的决定是 `tune_or_rollback`。判断有先后顺序，前提不成立时后面的数字都不用看：

```text
检查点字段齐全吗     否 → blocked_unrecoverable
更新和损失是有限值   否 → blocked_nonfinite
保留退化没超上限     否 → tune_or_rollback
领域损失降低了吗     否 → tune_data_or_schedule
以上全部通过        → continue_with_evaluation
```

检查点要包含五样：模型、优化器、学习率调度器、随机数状态和全局步。“有限值”指不是 NaN 这样的数。全部通过也只表示可以进入更完整的任务评测，notebook 特意说明这不能证明真实领域能力已经提升。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 哪些位置参与持续预训练的损失：每个非填充词元都要预测下一个词元 | 读，对照上面的公式 |
| Step 2 | 领域训练集、领域评测集、通用保留集各自回答什么问题，更新前后的损失变化怎么算 | 读，对照相减的小例子 |
| Step 3 | 何时继续、调低或回退：先查检查点和数值，再查保留上限，最后看领域收益 | 读，对照最后的判断顺序 |
| Step 4 · TODO 1 | 检查形状和二值掩码，并把填充（padding）位置写为 -100 | 在 `build_continual_pretraining_labels` 里补：形状不一致、掩码不是 0 或 1 就报错，再复制输入并把掩码为 0 的位置改成 `-100` |
| Step 4 · TODO 2 | 对齐下一 Token 目标，忽略 -100，并按有效位置归约 | 在 `continual_pretraining_loss` 里补：检查维度，把预测和标签错开一位，用 `ignore_index=-100` 求损失总和，再除以有效目标数 |
| Step 4 · TODO 3 | 拒绝空集合与任意集合交叠，并返回每组大小 | 在 `audit_continual_splits` 里补：三个集合非空、两两没有交集，返回 `clean` 和 `counts` |
| Step 4 · TODO 4 | 验证阈值和指标，再按不可恢复、非有限、保留退化与领域收益判断 | 在 `decide_continual_pretraining` 里补：先检查阈值，再按四个分支依次返回 `decision` |
| 测试 | 四个测试依次检查标签和非法掩码、损失数值和空目标、切分是否隔离、判断优先级 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 Transformers 的微型随机 Causal LM 核对全序列标签和下一词元损失 | 需要 `TRAIN-CPU` 环境，可以先跳过 |
| Step 5（可选） | 在真实模型上测量更新前后，领域文本和保留文本上的损失变化 | 需要 GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `TRAIN-CPU` 环境。Step 5 需要 GPU，默认是 `RUN_MODE = 'dry_run'`。

## 容易踩的坑

- **照搬 SFT 的“只监督回答”**：持续预训练是全序列监督，每个非填充词元都要保留，只有填充位置的标签写成 `-100`。
- **掩码不合法却没报错**：输入和掩码的形状必须一致，掩码只能取 0 或 1。测试专门用形状不一致、掩码里含 2 的输入，检查会不会抛出 `ValueError`。
- **损失按全部位置平均**：分母只能是非 `-100` 的有效目标数。一个有效目标都没有时要报错，不能悄悄返回一个数。
- **数据有重叠还接着判断**：三份数据必须非空且两两不相交，否则损失变化分不清是训练记忆、领域收益还是通用退化。
- **先看领域收益，后查前提**：检查点缺字段或数值不是有限值时直接阻断，领域损失降得再多也不能盖过去。阈值必须是有限的非负数。

## 读完后自查

- [ ] 能说出持续预训练和 SFT 在“哪些位置算损失”上的区别。
- [ ] 给一条长度为 4、最后一位是填充的序列，能写出错开一位后剩下的有效目标。
- [ ] 能写出两种损失变化的算式，并说出最低的接受门槛。
- [ ] 能按顺序说出四个判断分支，以及各自返回的决定。

## 下一步

- 打开 notebook：[56. 持续预训练与能力保留](../56_Continual_Pretraining_and_Retention.md)
- 建议先读：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)、[55. 训练更新动力学 · 小白入门](./55_Training_Update_Dynamics.md)
- 想补专题背景：[后训练优化 · 01 SFT 数据与监督契约](../../topic_discussion/post_training_optimization/01_sft_data_and_loss.md)、[后训练优化 · 10 独立评测与采用决策](../../topic_discussion/post_training_optimization/10_evaluation_and_adoption_decision.md)
