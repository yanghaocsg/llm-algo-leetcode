# PT-AGENTIC-RL. 环境轨迹与奖励归因 — 小白讲解

> 对应 notebook：[`PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.ipynb`](./PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.ipynb)

## 一句话总结
这一节用一个确定性的"查询—提交"环境，把智能体的一次任务拆成带动作、观察、奖励和结束原因的多步轨迹：`environment_step` 执行单步并区分终止与截断（TODO 1），`build_agent_trajectory` 在首次结束后停止采样（TODO 2），`compute_action_returns` 从后往前算动作回报（TODO 3），`recommend_agentic_stage` 用安全、监督能力与试采增益决定下一步走 SFT 还是在线 RL（TODO 4）。核心结论是：奖励必须绑定到触发它的动作，观察只进入下一轮上下文。

## 为什么需要它：痛点是什么
只看最后一句答案，你会遇到两个麻烦。第一，模型做对了，但你不知道是"查对了数再提交"还是"蒙对了"；做错了，也不知道是没查、查错、还是提交格式不对。第二，任务有步数预算：模型可能是成功结束，也可能是预算用完被迫停下，这两种"结束"在训练目标里处理方式不同，混在一起就会把"超时"当成"失败"或反过来。

生活类比：老师批一道实验题，不只看最终结论，还要看实验记录——先取样、再读数、最后写结论。如果记录里"取样"这一步成功了，哪怕最后结论错，老师也知道该补哪一步；如果记录到一半下课铃响了，那是"时间到"，不是"做错"。轨迹就是把这份实验记录按动作顺序存下来。

## 核心问题
- 一条多步轨迹由哪些字段组成？动作、观察、奖励、结束原因分别记在哪？
- "终止（terminated）"和"截断（truncated）"为什么必须互斥？成功提交恰好踩到步数上限算哪种？
- 即时奖励 $r_t$ 怎么由任务、过程、安全、成本四项合成？动作回报 $G_t$ 为什么要从后往前算？
- 为什么观察不能拿动作回报、无效动作位置也不能拿？
- 一条轨迹要满足哪些版本、奖励与消费条件，才允许进入策略更新？
- 监督基线、试采增益和安全退化，谁有否决权？

## 解决思路：一步步看懂

**第 1 步：一条轨迹长什么样。** 环境从固定初态 $s_0$ 开始，策略根据可见历史 $s_t$ 选动作 $a_t$，环境返回观察 $o_{t+1}$ 和奖励 $r_t$，历史推进到 $s_{t+1}$。notebook 写成

$$
\tau=(s_0,a_0,o_1,r_0,s_1,a_1,o_2,r_1,\ldots,s_T),
$$

$T$ 是动作数，最后动作下标是 $T-1$。返回的字典里，`state_before` / `state_after` 是前后状态快照，`action` 是模型动作，`observation` 是工具返回文本，`reward_components` 是四项奖励，`action_mask` 标这是不是模型动作，`terminated` / `truncated` 是两个布尔结束标记，`termination_reason` 是人类可读原因。**观察是给下一轮决策看的，回报是给动作算的**，两者不能混。

**第 2 步：奖励怎么合成。** 四项分别是任务奖励 $r_t^{task}$、过程奖励 $r_t^{process}$、安全扣减 $r_t^{safety}$、成本扣减 $r_t^{cost}$：

$$
r_t=r_t^{task}+r_t^{process}-r_t^{safety}-r_t^{cost},
\qquad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

$\gamma\in[0,1]$ 是折扣因子：越靠后的奖励打折越多。notebook 的确定性规则是每次动作都有 `cost=0.05`；`lookup` 给 `process=0.2`；`submit:4` 在已查到 4 时给 `task=1.0` 并成功，否则给 `safety=0.5` 并失败；其他动作给 `safety=1.0` 并记 `invalid_action`。所以查询的即时奖励是 $0.2-0.05=0.15$，成功提交是 $1.0-0.05=0.95$。$\gamma=1$ 时查询动作回报 $0.15+0.95=1.10$；$\gamma=0.5$ 时是 $0.15+0.5\times0.95=0.625$。

**第 3 步：TODO 1 —— 终止优先，截断只是预算用完。** 骨架先用 `reason` 记录动作结果：成功是 `'success'`，提前提交是 `'failure'`，非法动作是 `'invalid_action'`，查询本身没有结束原因。参考答案：

```python
terminated = reason is not None
truncated = not terminated and next_state['step_index'] >= max_steps
reason = 'timeout' if truncated else reason
next_state['done'] = terminated or truncated
```

先看任务有没有给结论，再看预算。这样"成功提交恰好用掉最后一步"会走 `terminated`，不会同时被打上 `truncated`；而 `max_steps=1` 时只做一次 `lookup`，任务没结束但步数到顶，才是 `truncated=True`、原因 `timeout`。反过来写（先判断步数）会让成功和超时同时为真，`compute_action_returns` 会直接拒绝这种轨迹。

**第 4 步：TODO 2 —— 第一次结束就停。** 循环每执行一步就把 transition 收进列表，参考答案：

```python
stop_sampling = transition['terminated'] or transition['truncated']
if stop_sampling:
    break
```

如果不 break，模型会继续"消费"结束之后的动作，轨迹里出现 `transition_after_terminal`，评估函数也会判不合格。注意 `environment_step` 自己也会拒绝在 `done` 状态上继续执行。动作列表比预算短、且没有自然结束时，函数在循环后补一个 `truncated=True`、原因 `action_exhausted`，把"动作用完"和"时间用完"区分开。轨迹 ID 由 `task_id:environment_version:policy_version` 加动作串的 SHA-256 前 12 位组成（动作之间用 `chr(31)` 分隔），所以同一动作序列得到同一 ID；测试里给成功轨迹追加 `'unused'` 尾部，ID 不变，因为尾部根本没被执行。

**第 5 步：TODO 3 —— 反向递推。** 骨架已经把四项合成 `rewards[index]`，答案是一行：

```python
running = rewards[index] + gamma * running
```

从最后一个动作往前，`running` 始终是"当前动作奖励 + 折扣后的下一动作回报"。输出每行 `{'step_index', 'reward', 'return'}`，最后 `reversed(rows)` 恢复正序。容易写错的是写成 `running = gamma * (rewards[index] + running)`，那会把当前奖励也多打一次折；或者从前往后累加，算出来的是"到当前为止"而不是"从当前往后"。

**第 6 步：TODO 4 —— 阶段选择是"硬门槛优先"。** 参考答案按顺序定义四个条件：

```python
gates = [
    (not reward_replayable, 'blocked', 'reward_not_replayable'),
    (safety_regression > max_safety_regression, 'blocked', 'safety_regression'),
    (sft_tool_contract_pass_rate < min_sft_readiness, 'continue_sft', 'tool_contract_not_ready'),
    (novel_success_gain < min_exploration_gain, 'continue_sft', 'no_independent_exploration_gain'),
]
```

默认门槛是 `min_sft_readiness=0.8`、`min_exploration_gain=0.05`、`max_safety_regression=0.0`。顺手解释四个返回：奖励不可复算或安全退化超限 -> `blocked`，先修数据/安全，别谈训练；监督基线连工具调用规则都过不了（0.6 < 0.8）-> `continue_sft`；监督 OK 但试采只涨 0.02（< 0.05）-> 还是 `continue_sft`，说明探索没带来独立收益；全部通过（0.9、0.08、可复算、无退化）才给 `agentic_rl_candidate`。顺序很重要：如果不先判 `blocked`，一个高监督分可能掩盖安全问题。

**第 7 步：谁能进入更新：`assess_agentic_update`。** 这个函数不训练，只做准入检查。它依次返回具体的 `reason`：环境版本不一致 `environment_version_mismatch`；轨迹策略版本比当前新 `future_policy`；落后超过 `max_policy_lag` 是 `stale_policy`；`reward_status != 'completed'` 或 `lifecycle_state != 'rewarded'` 是 `reward_not_ready`；已被别的更新消费是 `already_consumed`；结束标记非布尔或最后一步同时终止+截断是 `invalid_end_flags`；最后一步没有任何结束证据是 `missing_terminal_evidence`；结束之后还有 transition 是 `transition_after_terminal`；动作掩码或奖励字段不全、奖励非有限是 `invalid_action_evidence`。全部通过才返回 `{'eligible': True, 'reason': 'ok', 'policy_lag': lag}`。

**第 8 步：可选 Gymnasium 核对。** `ControlledAgentGymEnv` 把同一个状态机包成 `gym.Env`：动作空间 `Discrete(3)`（0=`lookup`、1=`submit:4`、2=`invalid`），观察是 `Dict`（`step_index`、`known_value` 是否为 4、`done`）。`check_env` 检查协议，随后核对同种子重置一致、观察在空间内、本地 transition 一致、查询奖励 0.15、成功时 `terminated=True` 且奖励 0.95、`max_steps=1` 时只截断且原因 `timeout`。它验证"接口没写错"，不是另一套独立算法。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 轨迹（trajectory） | 一次任务里按顺序记录的状态、动作、观察和奖励 |
| 动作（action） | 模型选择的一步操作，如查询或提交 |
| 观察（observation） | 环境/工具返回、供下一步决策使用的信息 |
| 终止（terminated） | 任务按规则结束：成功、失败或非法动作 |
| 截断（truncated） | 任务没结束但步数预算用完 |
| 即时奖励 $r_t$ | 四项奖励按"任务+过程−安全−成本"合成 |
| 动作回报 $G_t$ | 从当前动作到结束的折扣奖励总和 |
| 折扣因子 $\gamma$ | 后面奖励的折扣力度，0 到 1 |
| 奖励归因 | 把反馈记到真正触发它的动作上 |
| 策略版本 | 生成这条轨迹时所用策略的编号，用于判断新鲜度 |
| 消费状态（`consumed_by`） | 这条轨迹是否已经被某次更新用过 |

## 小白常见疑问
**Q1：为什么成功提交恰好用掉最后一步，不算截断？**
因为规则是"任务结果优先"。`terminated = reason is not None` 先成立，`truncated = not terminated and ...` 就被短路成 `False`。`test_end_flag_boundaries` 用 `max_steps=2` 跑 `['lookup', 'submit:4']`，断言最后一步 `terminated` 为真、`truncated` 为假。这样训练时不会把"做完了"误判成"时间到"。

**Q2：为什么观察不能拿动作回报？**
观察是环境的输出，不是模型选的动作。`compute_action_returns` 要求每条 transition 的 `action_mask == 1`，并且 `test_reward_attribution` 把第一条的 `action_mask` 改成 0，期望抛 `ValueError`。如果给观察也算回报，模型会学到"多产生观察就能加分"，而不是"选对动作"。

**Q3：$\gamma=0.5$ 时查询回报 0.625，为什么不是 0.15×0.5？**
因为折扣只作用于**未来**奖励，当前动作的 0.15 不打折。式子从后往前：提交那步 `running=0.95`；查询那步 `0.15 + 0.5×0.95 = 0.625`。如果把当前奖励也乘 $\gamma$，会得到 0.075 + 0.475 = 0.55，测试的 `abs(...-0.625) < 1e-9` 就过不了。

**Q4：`action_exhausted` 和 `timeout` 有什么区别？**
两者都是截断，但原因不同：`timeout` 是步数预算 `max_steps` 用完（`test_timeout_and_action_exhaustion` 用 `max_steps=1`、动作 `['lookup']`）；`action_exhausted` 是动作列表先没了、任务还没结束（同一测试用 `max_steps=3`、动作 `['lookup']`）。分开记录，才能判断是环境预算太紧还是模型/采样器动作给少了。

**Q5：监督通过率 0.9、增益 0.08，但奖励不可复算，会怎样？**
返回 `{'decision': 'blocked', 'reason': 'reward_not_replayable'}`。`gates` 的第一条就命中，后面的能力和增益不再看。`test_stage_decision` 用 `recommend_agentic_stage(0.9, 0.08, False, 0.0)` 断言 `blocked`；安全退化 0.01 则返回 `reason == 'safety_regression'`。硬门槛是否决项，不是加权平均。

## 动手验证：怎么知道自己做对了
`test_success_and_failure_trajectories` 检查查询后提交的 `termination_reason == 'success'`、第一步观察 `'value=4'`、提前提交为 `'failure'`、非法动作为 `'invalid_action'`，并确认不同动作序列的 `trajectory_id` 不同。`test_timeout_and_action_exhaustion` 检查 `max_steps=1` 时 `truncated is True` 且原因 `'timeout'`，动作耗尽时原因 `'action_exhausted'`。`test_reward_attribution` 检查末步奖励 0.95、首步 0.15、$\gamma=1$ 回报 1.10、$\gamma=0.5$ 回报 0.625、追加尾部不改变 ID、`action_mask=0` 时抛错。`test_update_admission` 逐项命中 `environment_version_mismatch`、`stale_policy`（版本 6、`max_policy_lag=2`）、`future_policy`（版本 2）、`invalid_action_evidence`（`NaN` 奖励）和 `already_consumed`。`test_stage_decision` 覆盖两条 `continue_sft`、候选、`blocked`、`safety_regression`，并确认 `NaN` 指标、布尔值冒充整数、负策略版本都抛 `ValueError`。`test_end_flag_boundaries` 检查矛盾/非布尔结束标记。全部通过后打印 `✅ Agentic RL 环境轨迹、奖励归因与阶段准入测试通过。` notebook 没有保存执行输出；`framework_check` 的 `checks` 字典和 `✅` 行要自己跑出来才算数。
