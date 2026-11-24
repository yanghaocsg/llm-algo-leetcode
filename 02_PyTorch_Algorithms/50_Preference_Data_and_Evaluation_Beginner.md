# 50. Preference Data and Evaluation Beginner | 偏好数据与对齐评测 · 小白入门

> 配套 notebook：[50. 偏好数据与对齐评测](./50_Preference_Data_and_Evaluation.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

训练出一个新模型后，怎样判断它真的比原来好？这本 notebook 先检查“两个回答里选一个”的训练数据能不能用，再用覆盖率、胜率等数字决定新模型是通过、复核还是拒绝。

## 为什么需要它

偏好类训练用的数据叫偏好对：同一个提示词（prompt）给出两个回答，一个被选中，一个被淘汰。DPO、Online DPO 直接用这种数据，PPO、GRPO 等方法用的是轨迹和奖励。训练完得到一个候选模型，你怎么知道它比原来的基线模型更好？最省事的看法是“训练损失降了”“奖励涨了”。notebook 的导读说这不够：这些都不能直接说明候选模型更有帮助、更安全或更可靠，最终仍要和固定基线在隔离的评测集上比较。

比较的时候，三种情况容易把结论带偏：

- 评测的题目出现在训练数据里。模型靠记忆答对了，却被当成能力提升。
- 平局和没评完混在一起。计划评 4 对，只完成 3 对，其中 1 对打平。没评的 1 对和打平的 1 对性质不同，统计时要分开。
- 打分的评审器不稳定。同一批数据判两次结果不一样，或者回答换个顺序，判决就翻转。

这本 notebook 把这些检查写成四个函数：审计数据、汇总胜负平、检查判分一致性、判定候选模型能不能通过。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 偏好对（preference pair） | 同一个提示词下的两个回答，一个被选中（`chosen`），一个被淘汰（`rejected`） |
| 基线 / 候选（baseline / candidate） | 基线是拿来对照的原模型，候选是训练后要检验的新模型 |
| 污染（contamination） | 同一道题既在训练数据里，又在评测数据里。数据属于训练、验证还是评测，叫切分（split） |
| 规范化（normalize） | 统一大小写和空白，用来判断两条数据是不是同一道题 |
| 平局（tie） | 候选和基线打成平手 |
| 判分一致性（judge consistency） | 让打分器重复判、换顺序判，看结果有多稳定 |
| `qualified` / `review` / `reject` | 候选模型的三种结论：准入、复核、拒绝 |

## 核心思路

整条流程分两段。前一段看数据：审计每条偏好对有没有空字段、来源缺失、重复编号，以及训练和评测之间有没有重复的题目。后一段看候选模型：把评测记录汇总成几个比例，再和门槛比较，给出 `qualified`（准入）、`review`（复核）或 `reject`（拒绝）。汇总评测记录用到下面几个量：

```text
n_evaluated = n_candidate + n_baseline + n_tie
C_eval     = n_evaluated / n_expected                   # 评测覆盖率：完成了多少
R_decision = (n_candidate + n_baseline) / n_evaluated   # 明确判决率：完成的里面多少分出了胜负
W_non_tie  = n_candidate / (n_candidate + n_baseline)   # 非平局胜率：分出胜负的里面候选赢了多少
C_repeat   = 与首次判分一致的偏好对数 / N               # 重复判分的稳定性
C_swap     = 换序后仍与首次一致的偏好对数 / N           # 交换回答顺序后的稳定性
C_judge    = min(C_repeat, C_swap)                      # 判分一致性，取两项里较低的一项
```

符号说明：`n_expected` 是计划评测的对数，`n_evaluated` 是实际完成的对数，`n_candidate`、`n_baseline`、`n_tie` 分别是候选胜、基线胜、平局的对数，`N` 是参与一致性核对的偏好对数量。下面用 notebook 测试里的数字算一遍：

```python
def normalize(text):
    return " ".join(text.casefold().split())

train = {normalize("Explain LoRA")}
eval_set = {normalize("  explain   lora  "), normalize("Explain PPO")}
print(len(train & eval_set))                   # 训练与评测重复的任务输入数

scores = [(0.9, 0.1), (0.2, 0.2), (0.4, 0.5)]  # 每对：(候选得分, 基线得分)
wins = sum(c > b for c, b in scores)
losses = sum(c < b for c, b in scores)
evaluated, decided = len(scores), wins + losses
coverage = evaluated / 4                       # 计划评 4 对
win_rate = wins / decided
print(wins, losses, evaluated - wins - losses)  # 候选胜、基线胜、平局
print(coverage, decided / evaluated, win_rate, wins / evaluated)

first, repeated, swapped = [1, 0, -1, 1], [1, 0, 1, 1], [1, 0, -1, -1]
repeat = sum(a == b for a, b in zip(first, repeated)) / len(first)
swap = sum(a == b for a, b in zip(first, swapped)) / len(first)
print(repeat, swap, min(repeat, swap))

reasons = [name for ok, name in [
    (evaluated >= 3, "insufficient_evaluated_pairs"),
    (coverage >= 0.8, "low_evaluation_coverage"),
    (decided >= 2, "insufficient_decided_pairs"),
    (win_rate >= 0.6, "low_decided_win_rate"),
    (min(repeat, swap) >= 0.8, "low_judge_consistency"),
] if not ok]
print(reasons)
```

```text
1
1 1 1
0.75 0.6666666666666666 0.5 0.3333333333333333
0.75 0.75 0.75
['low_evaluation_coverage', 'low_decided_win_rate', 'low_judge_consistency']
```

第一个 `1` 是污染数：`"Explain LoRA"` 和 `"  explain   lora  "` 规范化之后是同一道题，所以训练集和评测集有 1 道重复。第二段是胜负统计。3 对里候选胜 1、基线胜 1、平局 1。计划评 4 对，只完成 3 对，覆盖率是 0.75。完成的 3 对里只有 2 对分出胜负，明确判决率是 2/3。非平局胜率只在分出胜负的 2 对里算，是 0.5；如果把平局也放进分母，就成了最后一列的 0.33。

第三、四段是判分一致性和候选门禁。4 个偏好对里，重复判分有 3 个和首次相同，换序后也有 3 个相同，两项都是 0.75。`classify_preference_evidence` 的默认门槛是：至少评 3 对、覆盖率 0.8、至少 2 对分出胜负、非平局胜率 0.6、判分一致性 0.8。上面的数字在覆盖率、胜率、一致性三处不达标，所以返回这三条原因。数据和安全检查都通过时，状态是 `review`；只要数据检查或任一安全指标失败，就直接 `reject`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一条偏好对要同时成立哪些关系才算可比：同一任务、偏好理由、来源可追溯、训练与评测隔离 | 读 |
| Step 2 | 评测覆盖率、明确判决率、非平局胜率、判分一致性怎么算，各说明什么 | 读，对照上面的公式 |
| Step 3 | 哪些证据不足会进复核，哪些情况直接拒绝 | 读 |
| Step 4 · TODO 1 | 审计偏好数据 | 写 `audit_preference_dataset`：补 1 行，算出规范化后训练集与评测集任务输入的交集大小 |
| Step 4 · TODO 2 | 汇总成对结果 | 写 `compute_pairwise_outcomes`：补 4 行，算出明确判决数、覆盖率、判决率和非平局胜率 |
| Step 4 · TODO 3 | 检查判分一致性 | 写 `compute_judge_consistency`：补 2 行，逐位置数出重复判分、换序判分与首次一致的个数 |
| Step 4 · TODO 4 | 判定候选状态 | 写 `classify_preference_evidence`：补 1 行，按“拒绝、复核、准入”的优先级给出 `status` |
| 测试 | 检查数据审计、成对统计、判分一致性和候选决策：空字段、重复与污染、平局与部分完成、顺序翻转，以及通过、复核、拒绝三类结果 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把同一批偏好记录交给 Hugging Face `Dataset`，核对字段结构、样本身份、切分筛选和污染计数 | 需要 `PT-CPU` 环境 |

整本 notebook 的主线只需要 CPU。框架核对需要另外的 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **把平局算进胜率的分母**：平局算“已完成评测”，进入覆盖率和明确判决率，但不进入非平局胜率。胜率只用候选胜和基线胜来算；没有胜负时记为 0。
- **把没评的当成平局**：评测缺失既不算平局，也不进入胜负统计。完成数、平局数要分开记录，覆盖率单独用“完成数 ÷ 计划数”。
- **不规范化就比较题目**：训练和评测的重复检查要先统一大小写和空白，否则换个大小写或空格就能绕过污染检查。
- **只比较总体比例**：`compute_judge_consistency` 只按位置逐个比较，不读样本 ID。三次判分要按同一偏好对顺序传入，长度一致，取值只能是 `-1`、`0`、`1`。
- **让数据或安全失败被降级**：数据检查失败或任一安全指标低于阈值时必须返回 `reject`，不能让后面的普通原因把它变成 `review`。返回时要带上全部失败原因。

## 读完后自查

- [ ] 能说出一条可信的偏好对要同时成立哪几种关系。
- [ ] 计划评 4 对、完成 3 对、其中 1 对平局，能算出评测覆盖率和明确判决率。
- [ ] 能解释为什么平局算“评完了”，却不算“分出胜负”。
- [ ] 能说出候选被 `reject`、`review`、`qualified` 的条件。

## 下一步

- 打开 notebook：[50. 偏好数据与对齐评测](./50_Preference_Data_and_Evaluation.ipynb)
- 建议先读：[33. 后训练准入与阶段选择 · 小白入门](./33_Fine_Tuning_Readiness_Beginner.md)
- 做完之后：[PT-REWARD-SIGNAL. 奖励、价值与优势信号 · 小白入门](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage_Beginner.md)
- 想看偏好数据怎么用于训练：[17. 直接偏好优化损失](./17_DPO_Loss_Tutorial.ipynb)、[18. 组相对策略优化与可验证奖励](./18_GRPO_Loss_Tutorial.ipynb)、[51. 在线 DPO](./51_Online_DPO.ipynb)
