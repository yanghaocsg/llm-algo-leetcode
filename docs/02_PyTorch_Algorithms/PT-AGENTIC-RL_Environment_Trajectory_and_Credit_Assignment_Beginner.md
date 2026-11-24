# PT-AGENTIC-RL. Environment Trajectory and Credit Assignment Beginner | 环境轨迹与奖励归因 · 小白入门

> 配套 notebook：[PT-AGENTIC-RL. 环境轨迹与奖励归因](./PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

模型调用工具、分几步完成任务时，成功是哪一步的功劳要算清楚。这本 notebook 用一个两步的小环境走通这条多步记录，再判断该继续用现成的成功案例训练，还是可以开始在线探索。

## 为什么需要它

前面的训练里，模型回答一次就结束。智能体（Agent）任务不一样：模型要自己决定下一步做什么，做完看到结果，再决定下一步。比如先调用工具查一个数，再把答案提交上去。

这种多步过程带来三个新问题：

- **功劳算给谁。** “提交成功”是第二步才拿到的反馈，但第一步的查询也有功劳。只看最后一步，前面的正确步骤得不到任何信号。
- **哪些内容是模型做的。** 工具返回的结果是环境给的，不是模型生成的。把它当成模型的动作去更新，就更新错了对象。
- **任务是怎么结束的。** 自己做对了或做错了，和“还没做完但步数用完了”，是两种不同的结束，不能混在一起记。

还有一个前置问题：是不是一上来就该用智能体强化学习（Agentic RL）？notebook 的回答是先用验证过的成功轨迹做监督微调，建立基线。如果错误主要出在工具格式、参数约束或观察的使用上，补充监督轨迹通常比在线探索更直接。

这本 notebook 把这些问题放进一个很小的确定性环境：先查询，再提交。你要写的是结束方式、轨迹推进、每一步的回报，以及选择训练阶段的规则。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 智能体（Agent） | 会自己决定下一步做什么的模型：先出一个动作，看到结果，再决定下一个动作 |
| 状态（state）/ 观察（observation） | 状态是轮到模型决策时它能看到的全部信息；观察是环境执行完动作后返回的结果 |
| 动作（action） | 模型自己生成的内容，比如一段文本、一次函数调用或一段代码 |
| 奖励归因（credit assignment） | 把拿到的反馈算到具体是哪一步动作造成的 |
| 终止（terminated）/ 截断（truncated） | 终止是任务自己结束了，比如成功、失败或非法动作；截断是任务没结束但步数用完了 |
| 回报（return）和折扣因子（gamma） | 回报是从某个动作开始，把后面的奖励加起来的总数；`gamma` 是 0 到 1 之间的数，越往后的奖励乘的次数越多 |
| 动作掩码（action mask） | 标出哪些位置是模型真正做的动作，只有这些位置能进入策略更新 |

## 核心思路

一次任务从固定的任务和环境开始。模型在状态 `s_t` 上给出动作 `a_t`，环境执行后返回观察、奖励分量和是否结束，观察再进入下一轮的状态。

```text
轨迹      tau = (s_0, a_0, o_1, r_0, s_1, a_1, o_2, r_1, ..., s_T)
动作奖励  r_t = r_task + r_process - r_safety - r_cost
回报      G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...   （加到轨迹结束为止）
```

`s_t` 是第 `t` 步决策时模型能看到的状态，`a_t` 是模型生成的动作，`o_{t+1}` 是环境返回的观察，`r_t` 是和动作 `a_t` 对应的奖励，`T` 是结束的位置。四类奖励分量里，任务奖励通常只在最后出现，过程奖励落在正确的调用上，安全和成本是惩罚，所以用减号。

题目里的环境每一步都记 0.05 的成本惩罚，动作效果是固定的：

```text
动作        观察                       奖励分量            结束方式
lookup      value=4                    process +0.2       不结束
submit:4    accepted（之前查过）       task +1.0          终止，原因 success
submit:4    rejected:lookup_required   safety 0.5         终止，原因 failure
其他        invalid_action             safety 1.0         终止，原因 invalid_action
```

如果这一步没有自然结束，而步数已经用完，就记为截断，原因写 `timeout`。下面用测试里的“先查询、再提交”算一遍奖励和回报：

```python
rewards = {'task': [0.0, 1.0], 'process': [0.2, 0.0], 'safety': [0.0, 0.0], 'cost': [0.05, 0.05]}
r = [round(rewards['task'][t] + rewards['process'][t] - rewards['safety'][t] - rewards['cost'][t], 10)
     for t in range(2)]
print(r)

gamma = 0.5
running, returns = 0.0, []
for t in (1, 0):                      # 从最后一个动作往前算
    running = round(r[t] + gamma * running, 10)
    returns.append(running)
print(returns)
```

```text
[0.15, 0.95]
[0.95, 0.625]
```

第一行是两步各自的奖励：查询得 0.15，提交得 0.95。第二行是回报，从后往前算：第二步的回报就是 0.95；第一步的回报是 0.15 加上打折后的 0.95，等于 0.625。最后的成功就这样沿模型动作向前归因，前面的查询也分到了信号。

有了轨迹，还要决定训练阶段。`recommend_agentic_stage` 按固定顺序做四个判断，命中第一个就返回：

```text
奖励不能复算                → blocked
安全回归超过上限            → blocked
工具契约通过率低于 0.8      → continue_sft
在线探索增益低于 0.05       → continue_sft
都不命中                    → agentic_rl_candidate
```

工具契约是调用工具时要遵守的格式和参数规定，探索增益是在线采样比监督基线多出来的独立成功。前两个是硬阻塞，后两个只说明现在更适合继续监督微调。notebook 也说明，这些检查只用来选择训练阶段，不能代替策略更新所需的动作概率和版本兼容性检查。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 模型动作和工具观察如何形成多步环境轨迹：轨迹的写法，以及每一类对象要记什么 | 读 |
| Step 2 | 任务、过程、安全与成本反馈如何归因到模型动作：四类奖励分量和折扣回报 | 读，对照上面的小例子 |
| Step 3 | 何时继续轨迹 SFT，何时进入在线探索：四种观察结果对应的阶段判断 | 读 |
| Step 4 · TODO 1 | 区分终止与截断 | 在 `environment_step` 里补 4 行：`terminated`、`truncated`、截断时的 `reason` 和 `done` |
| Step 4 · TODO 2 | 推进并封装多步轨迹 | 在 `build_agent_trajectory` 里补 1 行：`stop_sampling`，结束后的动作不再执行 |
| Step 4 · TODO 3 | 计算动作级折扣回报 | 在 `compute_action_returns` 里补 1 行：`running`，从最后一个动作往前递推 |
| Step 4 · TODO 4 | 选择训练阶段 | 在 `recommend_agentic_stage` 里补全 `gates`：按“奖励可复算、安全、监督就绪、探索增益”的顺序列出 4 个判断 |
| 测试 | 检查成功、失败和非法动作的终止，超时与动作耗尽，奖励归因，更新准入和阶段选择 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把本节的查询—提交环境包装成 Gymnasium 环境，核对重置、单步转移、终止与截断协议 | 需要 `PT-CPU` 环境，可以先跳过 |

Step 1–4 只需要 CPU。框架核对需要另外的 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **把截断当成终止**：成功、失败和非法动作是任务自然结束，用 `terminated`。只有任务没结束而步数用完才是截断，用 `truncated`，原因写 `timeout`。两种结束都要把 `done` 设为真。
- **结束以后还继续执行动作**：第一个终止或截断之后，剩下的动作不属于这条轨迹。测试里多出来的 `'unused'` 动作不能改变轨迹 ID。
- **给环境观察算回报**：只有模型动作才有动作掩码，才能得到回报。`action_mask` 不是 1 的轨迹要报错。
- **折扣的方向或符号写反**：回报从最后一个动作往前递推，每一步是“当前奖励加 `gamma` 乘以后一个动作的回报”。安全和成本是惩罚，用减号。
- **把判断顺序写错**：奖励不可复算和安全回归是硬阻塞，排在最前；通过率或探索增益不足只是继续监督微调。

## 读完后自查

- [ ] 能说出终止和截断的区别，并各举一个结束原因。
- [ ] 两步奖励是 0.15 和 0.95 时，能算出 `gamma` 为 0.5 时第一步的回报。
- [ ] 能说出哪两类奖励分量用减号合成。
- [ ] 能说出哪两种情况会阻塞在线探索，哪两种情况只是继续监督微调。

## 下一步

- 打开 notebook：[PT-AGENTIC-RL. 环境轨迹与奖励归因](./PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md)
- 建议先读：[PT-REWARD-SIGNAL. 奖励、价值与优势信号 · 小白入门](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage_Beginner.md)、[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness_Beginner.md)
- 需要补背景：[Agent 工作负载与执行链](../topic_discussion/agent/00_agent_workload_and_trace.md)、[后训练优化 · 轨迹生命周期与环境交互](../topic_discussion/post_training_optimization/08_trajectory_lifecycle_and_environment_interaction.md)
- 做完之后：[PT-RL-SYSTEM. 采样、训练与奖励系统 · 小白入门](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System_Beginner.md)
