# PT-KTO. Binary Feedback Alignment | 二元反馈对齐 · 小白入门

> 配套 notebook：[PT-KTO. 二元反馈对齐](../PT-KTO_Binary_Feedback_Alignment.md) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

DPO 要求同一条数据的两个回答配成对，KTO 不用：每条回答只要一个“好”或“不好”的标记就能训练。这本 notebook 教你把这个标记变成损失，并分点赞、点踩两类做好记录。

## 为什么需要它

DPO 需要同一个提示词下的一对回答：一个被选中，一个被淘汰。可实际收到的反馈常常是单条的：用户给一条回答点了赞或点了踩，或者只留下一个“验收通过”。配成对要花成本，配不上的反馈就用不上。

KTO 取消了配对要求。每条样本只有三样东西：提示词、回答、一个二元标签（1 表示可取，0 表示不可取）。比较的对手也从“另一个回答”换成一条基线：当前策略相对参考策略的偏移，要比这条基线高还是低，看标签是 1 还是 0。

配对要求取消了，数据检查没有取消。notebook 的 Step 1 特意说明：标签必须指向完整回答，提示词和回答不能在训练与评测之间重复，同一个来源的接受标准也要保持稳定。

这本 notebook 做四件事：汇总回答得分，估计这条基线，算出每条样本的加权损失，再按点赞、点踩两类分别汇总奖励。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 二元标签（label） | 每条回答只有一个标记：1 表示可取（desirable），0 表示不可取（undesirable） |
| 当前策略 / 参考策略（Policy / Reference） | 当前策略是正在训练的模型；参考策略是冻结不训练的版本，只提供对照分数 |
| 回答得分 | 回答范围内每个词元的对数概率之和 |
| 策略偏移 | 当前策略的回答得分减参考策略的回答得分，大于 0 表示当前策略更倾向于生成这条回答 |
| KL 基线 | 另一批独立样本的平均偏移，截断为不小于 0。它是比较偏移高低的参照线 |
| 隐式奖励（implicit reward） | 策略偏移乘 `beta`，按类别汇总诊断时用的就是它 |
| `sigmoid` | 把任意实数压到 0 和 1 之间的函数 |

## 核心思路

输入是每条样本的回答得分（当前策略和参考策略各一份）、二元标签、一批用来估计基线的独立样本得分、`beta` 和两个类别权重。分四步：汇总回答得分，估计基线，按标签给每条样本算损失并乘上类别权重，最后按类别汇总隐式奖励。`kto_loss` 返回每条样本的损失和隐式奖励，整个批次取平均由调用方完成。

```text
d     = s_policy(x, y) - s_reference(x, y)     # 策略偏移：当前策略得分减参考策略得分
z_KL  = max(0, 独立样本偏移的平均值)           # KL 基线，停止梯度
r     = beta * d                              # 隐式奖励
loss+ = 1 - sigmoid(beta * (d - z_KL))        # 标签为 1 的样本用这个
loss- = 1 - sigmoid(beta * (z_KL - d))        # 标签为 0 的样本用这个
L     = (1 / B) * Σ [ c_i * w_+ * loss+(d_i) + (1 - c_i) * w_- * loss-(d_i) ]
```

符号说明：`s` 是回答得分，`d` 是策略偏移，`z_KL` 是 KL 基线，`beta` 控制偏移的尺度，`r` 是隐式奖励，`B` 是批次里的样本数，`c_i` 是第 `i` 条样本的标签（1 或 0），`w_+` 和 `w_-` 是点赞、点踩两类的权重。下面用 notebook 测试里的数字算一遍：

```python
import torch

policy = torch.tensor([-1.0, -3.0, -0.8, -2.8])   # 当前策略给 4 条回答的得分
reference = torch.full((4,), -2.0)                 # 参考策略给同 4 条回答的得分
labels = torch.tensor([True, False, True, False])  # True 是点赞，False 是点踩
beta = 0.1

kl_shifts = torch.tensor([0.4, 0.2, 0.3, 0.1])     # 另一批独立样本的偏移
baseline = kl_shifts.mean().clamp_min(0.0)         # 取平均，再截断为非负
print(baseline, torch.tensor([1.0, 0.0, -1.0]).mean().clamp_min(0.0))

log_ratios = policy - reference                    # 每条回答相对参考策略的偏移
rewards = beta * log_ratios                        # 隐式奖励
loss_pos = 1 - torch.sigmoid(beta * (log_ratios - baseline))   # 点赞的回答：偏移高于基线才好
loss_neg = 1 - torch.sigmoid(beta * (baseline - log_ratios))   # 点踩的回答：偏移低于基线才好
print(log_ratios, rewards)
print(loss_pos, loss_neg)
losses = torch.where(labels, 2.0 * loss_pos, 3.0 * loss_neg)   # 两类权重分别是 2 和 3
print(losses, losses.mean())

pos, neg = rewards[labels], rewards[~labels]
print(round(pos.mean().item(), 4), round(neg.mean().item(), 4), round((pos.mean() - neg.mean()).item(), 4))
```

```text
tensor(0.2500) tensor(0.)
tensor([ 1.0000, -1.0000,  1.2000, -0.8000]) tensor([ 0.1000, -0.1000,  0.1200, -0.0800])
tensor([0.4813, 0.5312, 0.4763, 0.5262]) tensor([0.5187, 0.4688, 0.5237, 0.4738])
tensor([0.9625, 1.4064, 0.9525, 1.4213]) tensor(1.1857)
0.11 -0.09 0.2
```

第一行是基线。4 个独立样本的偏移平均是 0.25，基线就是 0.25；如果平均是负数，像后面那个例子，就截断成 0。第二行是策略偏移和隐式奖励。4 条回答的偏移分别是 1.0、-1.0、1.2、-0.8，乘 `beta` 0.1 得到隐式奖励。

第三行对每条样本同时算出两种损失，`loss_pos` 给点赞的样本用，`loss_neg` 给点踩的样本用。点赞的回答偏移比基线高得越多，`loss_pos` 越小；点踩的回答偏移比基线低得越多，`loss_neg` 越小。第四行是加权损失。标签交替是 1、0、1、0，所以第 1、3 条乘点赞权重 2，第 2、4 条乘点踩权重 3，整个批次取平均是 1.1857。

最后一行是按类别汇总：点赞的回答平均隐式奖励是 0.11，点踩的是 -0.09，两者相差 0.2。只看总损失看不出这个差别，所以要分开记录。如果一批里只有点赞的样本，点踩那一类的均值和奖励差都返回 `None`，不能拿 0 顶替。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 单条二元反馈长什么样，和成对偏好相比取消了什么、没取消什么 | 读 |
| Step 2 | 回答得分、策略偏移、KL 基线怎样变成点赞和点踩两种损失 | 读，对照上面的公式 |
| Step 3 | 类别权重怎样进损失，批次怎么归约，要记录哪些诊断量 | 读 |
| Step 4 · TODO 1 | 只聚合回答掩码内的词元 | 写 `aggregate_response_logps`：补 1 行，掩码外置零后沿最后一维求和 |
| Step 4 · TODO 2 | 计算平均策略偏移、截断为非负并停止梯度 | 写 `estimate_kl_baseline`：补 1 行，偏移求平均、`clamp_min(0.0)`、`detach()` |
| Step 4 · TODO 3 | 计算停止梯度的比较量、隐式奖励和逐样本加权损失 | 写 `kto_loss`：补 4 个变量 `log_ratios`、`baseline`、`implicit_rewards`、`losses`，按标签选方向并乘类别权重 |
| Step 4 · TODO 4 | 计算两类均值与奖励差；缺失类别返回 `None`，不伪造均值 | 写 `summarize_kto_batch`：补 3 个变量 `desirable_mean`、`undesirable_mean`、`margin`，缺失的类别返回 `None` |
| 测试 | 检查变长回答、非有限填充、负基线截断、单一类别和非法标签，以及损失方向、权重和梯度隔离 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把受控的得分和独立样本送进 TRL 的 `KTOTrainer._compute_loss`，核对 KL 基线和批次损失 | 需要 `PT-CPU` 环境 |

整本 notebook 只需要 CPU。框架核对需要 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **直接拿对数概率乘掩码**：被屏蔽的位置可能是 `-inf`，乘 0 仍是非有限值。要先用 `masked_fill` 把无效位置换成 0，再求和。
- **KL 基线来路不对**：基线要从独立的样本估计，不能由当前带标签的批次或测试集构造。算完要截断为非负，并且 `detach`，不能接收梯度。
- **标签和损失方向对不上**：点赞的样本用 `1 - sigmoid(beta * (d - z_KL))`，点踩的样本方向相反。写反之后损失的方向就反了。
- **批次均值除以了权重之和**：批次目标是加权后的逐样本损失直接取均值，不除以权重之和。函数返回逐样本损失，归约由调用方完成。
- **只看混合后的总损失**：要分别记录正负样本数量、两类隐式奖励的均值和差值，否则类别比例变化可能被误判为策略改善。

## 读完后自查

- [ ] 能说出 KTO 的一条样本有哪三个字段，和 DPO 相比少了什么要求。
- [ ] 能写出点赞样本和点踩样本各自的损失公式。
- [ ] 能解释 KL 基线从哪里估计，为什么要截断为非负并停止梯度。
- [ ] 一批里只有点赞的样本时，能说出 `summarize_kto_batch` 的奖励差返回什么。

## 下一步

- 打开 notebook：[PT-KTO. 二元反馈对齐](../PT-KTO_Binary_Feedback_Alignment.md)
- 建议先读：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)、[17. 直接偏好优化损失 · 小白入门](./17_DPO_Loss_Tutorial.md)、[50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation.md)
- 做完之后：[PT-REWARD-MODEL. 奖励模型训练、校准与过度优化 · 小白入门](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)
