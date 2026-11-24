# 33. Fine Tuning Readiness | 后训练准入与阶段选择 · 小白入门

> 配套 notebook：[33. 后训练准入与阶段选择](../33_Fine_Tuning_Readiness.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

动手训练模型之前，要先判断该不该训练、该用哪种训练。这本 notebook 给出一套检查顺序：先确认失败是稳定出现的，再试不改参数的办法，最后按手头的反馈选出下一训练阶段。

## 为什么需要它

你在做一个固定任务，比如处理退款流程。模型总是答错，第一反应往往是“微调一下”。但这一步可能走错方向：

- 失败只出现过一两次就开练。次数太少，分不清是稳定失败还是偶然波动。
- 没试不改参数的办法就去训练。也许换个固定的提示模板、接上检索或工具，问题就解决了。
- 直接跳到某个训练算法。手里只有示范回答，却选了需要奖励的在线训练，输入根本凑不齐。

notebook 的做法是一条链路：先用复测次数和失败率筛出稳定失败的那几组样本，再扣掉不改参数就能解决的，剩下的才考虑改参数。最后看“缺什么”和“手里有什么反馈”选出训练阶段，并检查输入齐不齐。这本 notebook 就带你把这三步判断写成代码。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 失败切片 | 按同类任务或场景划出的一小组样本，例如“已签收订单的退款流程” |
| 复测 | 把同一批样本再评测几遍，记下每条切片复测了几次、失败了几次 |
| 基线（baseline） | 拿来对照的起点。这里的策略基线指固定版本、已经复测过的生成模型 |
| 非参数方案 | 不改模型参数的办法，比如固定提示模板、检索增强生成（RAG）、调用工具 |
| 参数缺口（parameter gap） | 改参数要补什么：`representation` 是补充领域表示，`behavior` 是改变任务行为 |
| 反馈形态（feedback mode） | 训练信号长什么样，代码里是 `feedback_mode`：示范、偏好对、二元反馈、学习型奖励或验证器 |
| 轨迹（trajectory） | 一次生成的输入、回答和对应奖励等记录，在线训练需要它 |

## 核心思路

整条链路有三步，正好对应三个 TODO。

第一步，筛出稳定失败的切片（TODO 1）。Step 2 的公式写成纯文本是：

```text
p_s    = k_s / n_s                          失败率 = 失败次数 / 复测次数
Q(s)   = 1[n_s >= n_min] * 1[p_s >= p_min]  两项相乘，都是 1 才算可重复失败
```

符号说明：`s` 是一个失败切片；`n_s` 是它的复测次数，`k_s` 是失败次数；`n_min` 是最少复测次数，`p_min` 是最低失败率；`1[…]` 是指示函数，括号里的条件成立时取 1，否则取 0。筛完还要分别记下三类缺口：缺必需字段、评测没有隔离、切片没有基线结果。至少有一个切片入选，并且三类缺口都为空，才算 `ready`。

第二步，扣掉不改参数就能解决的切片（TODO 2）。候选方案要同时满足两件事才算“已解决”：指标达到验收门槛，并且没有带来新的失败切片。

```text
resolved(c, s) = 1[M(c, s) >= tau] * 1[N(c, s) 为空]
仍需考虑参数更新的切片 = 可重复失败切片 - 已解决的切片
```

`c` 是候选方案，`M(c, s)` 是它在切片上的指标值，`tau` 是验收门槛，`N(c, s)` 是它新增的失败切片集合。用 notebook 测试里的数字试一下：

```python
failure_runs = {'json_format': {'attempts': 3, 'failures': 2},
                'long_input': {'attempts': 4, 'failures': 3},
                'rare': {'attempts': 2, 'failures': 2}}
n_min, p_min = 3, 0.6
print({s: round(r['failures'] / r['attempts'], 2) for s, r in failure_runs.items()})
repeatable = sorted(s for s, r in failure_runs.items()
                    if r['attempts'] >= n_min and r['failures'] / r['attempts'] >= p_min)
print(repeatable)

candidates = [{'failure_slice': 'format', 'metric_value': 0.95, 'introduced_failures': []},
              {'failure_slice': 'behavior', 'metric_value': 0.92, 'introduced_failures': ['safety']}]
resolved = sorted(c['failure_slice'] for c in candidates
                  if c['metric_value'] >= 0.9 and not c['introduced_failures'])
print(resolved)
```

```text
{'json_format': 0.67, 'long_input': 0.75, 'rare': 1.0}
['json_format', 'long_input']
['format']
```

`rare` 的失败率是 1.0，但只复测了 2 次，不到下限 3 次，不算可重复失败。`behavior` 上的指标 0.92 过了门槛 0.9，可候选方案带来了新的失败切片 `safety`，不能记为已解决。

第三步，选训练阶段（TODO 3）。先看缺口，再看反馈：

```text
parameter_gap 是 representation                      → continued_pretraining
feedback_mode 是 demonstrations                     → sft
feedback_mode 是 preference_pairs 或 binary_feedback → offline_preference
feedback_mode 是 learned_reward 或 verifier          → online_rl
没有新的反馈                                         → stop_at_baseline
```

选定之后再查这个阶段需要的输入在不在，缺一项就返回 `blocked`，并列出缺的是哪几项。notebook 还有两点提醒：可验证奖励强化学习（RLVR）说的是奖励来源，不是一个单独的阶段，也不自动等于 GRPO；这条判断只检查启动条件，训练之后的收益仍要独立验证。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 怎样复测稳定失败并定位问题来源：失败切片、复测次数、失败率、评测隔离、基线覆盖 | 读，对照第一步的公式 |
| Step 2 | 怎样筛选重复失败与已经解决的切片：先筛稳定失败，再扣掉非参数方案解决的 | 读，对照小例子 |
| Step 3 | 基线与反馈怎样决定下一训练阶段：四个阶段各自需要哪些输入 | 读，对照最后的分支表 |
| Step 4 · TODO 1 | 筛选可重复失败切片 | 在 `assess_problem_evidence` 里补 2 行：按复测次数和失败率筛出切片，再找出其中没有基线记录的 |
| Step 4 · TODO 2 | 找出仍需考虑参数更新的问题 | 在 `assess_parameter_update_need` 里补 2 行：找出指标达标且没有新增失败的切片，再从失败集合里减掉 |
| Step 4 · TODO 3 | 选择训练阶段 | 在 `select_next_training_stage` 里补：按缺口和反馈形态给 `next_stage` 赋值，没有新反馈时返回 `stop_at_baseline` |
| 测试 | 九个测试检查证据门槛、参数更新判断、SFT 数据交接、各阶段的阻断原因、基线来源和反馈分支 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把阶段决策和 SFT 数据需求写入 Hugging Face Datasets 再读回，检查字段是否保持 | 需要 `PT-CPU` 环境，可以先跳过 |

整本 notebook 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `PT-CPU` 环境。

## 容易踩的坑

- **失败次数不够就开练**：复测次数不到 `n_min`，或失败率不到 `p_min` 的切片不算可重复失败。先继续复测，或排除运行异常。
- **指标达标就算解决**：候选方案还得没有新增失败切片。带来了新的失败，原来的切片不能记为已解决。候选结果里出现没登记的切片，也会报错。
- **把缺口合并成一个布尔值**：缺字段、检查未通过、切片未覆盖要分别记录在三个列表里，不能合成一个 False 就算完。
- **把验证器当成一个训练阶段**：RLVR 说的是奖励来源，不是和在线强化学习并列的阶段，也不自动等于 GRPO。反馈来源和更新方法要分别记录。
- **没有新反馈还硬切算法**：这时返回 `stop_at_baseline`，保留当前基线，去检查反馈缺在哪里。

## 读完后自查

- [ ] 能说出一个切片要同时满足哪两个条件，才算可重复失败。
- [ ] 能说出候选方案把切片记为“已解决”的两个条件。
- [ ] 给出缺口和反馈形态，能选出下一训练阶段，并说出没有新反馈时的结果。
- [ ] 能说出策略基线的两种来源，以及它为什么不限定为 SFT。

## 下一步

- 打开 notebook：[33. 后训练准入与阶段选择](../33_Fine_Tuning_Readiness.ipynb)
- 需要补基础：[性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb)
- 想补专题背景：[后训练优化学习路线](../../topic_discussion/post_training_optimization/intro.md)
- 做完之后：[32. SFT 数据契约与泄漏审计 · 小白入门](./32_Data_Engineering_for_SFT.md)
