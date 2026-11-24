# 86. DPO Online Benchmark | 在线后训练生命周期与恢复 · 小白入门

> 配套 notebook：[86. 在线后训练生命周期与恢复](../86_DPO_Online_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU + 真实产物

## 一句话说清

84 或 85 节训练出的候选，交给下一环评测前还要过一道检查：是否来自当前基线、证据是否还有效、出了问题能否恢复。这本 notebook 就是这道检查。

## 为什么需要它

84 节训练出一个 DPO 候选，85 节训练出一个 GRPO 候选。它们都从同一份 SFT 基线派生，是并列的两份候选。交给下一环独立评测之前，还有几件事要确认：

- 候选是不是从当前这份 SFT 基线派生的。基线已经换了，旧候选的证据就不再对应当前基线。
- 在线候选的轨迹是不是太旧。notebook 用窗口编号记录轨迹是第几批生成的，`window_age` 超过上限就算过期。
- 出了问题能不能退回去。更新失败时有没有更新前的检查点，产物损坏时能不能回滚到已验证的基线。

这本 notebook 解决的是“交出去之前把好关”：检查血缘和新鲜度，按失败阶段选择恢复点，最后给出交接状态。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 候选清单（manifest） | 84 或 85 节交付出来的记录，写明候选的策略版本、评测版本、适配器路径和哈希 |
| 血缘 | 一个候选是从哪个基线、经哪次训练得来的。这里主要看源策略版本是不是当前基线 |
| 新鲜度（freshness） | 候选的证据现在还有没有效：基线没换、评测版本还在、分支证据没过期 |
| 轨迹窗口 / `window_age` | 在线候选的轨迹是第几批生成的。`window_age` 是当前窗口减去证据窗口，越大越旧 |
| 检查点（checkpoint） | 更新前保存的状态，更新失败时可以从它恢复 |
| 回滚（rollback） | 出问题时退回到之前那份已验证的模型，这里是 SFT 基线 |
| 交接状态 | 本节的输出，三种：`ready_for_evaluation`、`tune`、`reject` |

## 核心思路

输入有四样：上游的候选清单、当前 SFT 基线的版本和允许的最大窗口年龄、失败发生在哪个阶段以及还有哪些可信状态、产物和回滚的核对结果。你补三处判断，输出是一个交接状态。三处判断写成纯文本是：

```text
新鲜      <=>  四项共同检查通过 且 分支证据有效
恢复状态  =  这个阶段由近到远，第一个还可信的状态；都没有就是无
交接状态  =  reject                 若 有任何硬失败
          =  tune                   若 上游是 tune
          =  ready_for_evaluation   其余情况
```

`<=>` 读作“当且仅当”。四项共同检查是：候选类型是 `dpo`、`grpo`、`rlvr` 之一，源策略版本等于当前基线，候选自己的策略版本和评测版本都不为空。分支证据分两种：DPO 候选要有偏好证据版本；在线候选要有验证器版本，并且 `window_age` 在 0 到上限之间。

看一个小例子，数字取自 notebook 的测试：

```python
# 新鲜度：两种候选检查的分支证据不同
maximum_window_age = 2
dpo = {"candidate_kind": "dpo", "preference_evidence_version": "pref-train-v1", "window_age": 99}
grpo = {"candidate_kind": "grpo", "verifier_version": "math-v1", "window_age": 2}

def branch_fresh(c):
    if c["candidate_kind"] == "dpo":
        return bool(c.get("preference_evidence_version"))
    return bool(c.get("verifier_version")) and 0 <= c.get("window_age", -1) <= maximum_window_age

print(branch_fresh(dpo), branch_fresh(grpo), branch_fresh({**grpo, "window_age": 3}))

# 恢复：按失败阶段，从近到远找第一个还可信的状态
by_stage = {"reward": ["trajectories", "prompt_queue"],
            "update": ["pre_update_checkpoint", "verified_baseline"]}
action_for = {("reward", "trajectories"): "resume_reward_from_saved_trajectories",
              ("reward", "prompt_queue"): "restart_sampling_from_prompt_queue",
              ("update", "pre_update_checkpoint"): "restore_pre_update_checkpoint",
              ("update", "verified_baseline"): "rollback_to_verified_baseline"}
fallback = {"reward": "restart_sampling", "update": "block_update"}

def choose(stage, available):
    state = next((s for s in by_stage[stage] if available.get(s)), None)
    return state, (action_for[(stage, state)] if state is not None else fallback[stage])

print(choose("reward", {"trajectories": True}))
print(choose("reward", {"trajectories": False, "prompt_queue": True}))
print(choose("update", {}))
```

```text
True True False
('trajectories', 'resume_reward_from_saved_trajectories')
('prompt_queue', 'restart_sampling_from_prompt_queue')
(None, 'block_update')
```

第一行：DPO 候选的 `window_age` 是 99 也算新鲜，因为它不看窗口，看的是偏好证据版本。GRPO 候选的窗口年龄 2 等于上限，算新鲜；改成 3 就过期。

后三行是恢复：奖励失败时，轨迹还在就从轨迹重新算奖励；轨迹没了但问题队列还在，就从队列重新采样；更新失败又没有任何可信状态，就阻止更新。

交接这一步，硬失败包括：证据不新鲜、回滚没准备好、产物没验证通过、上游状态不是三种之一、上游是 `reject`。有任何一个硬失败就是 `reject`。`ready_for_evaluation` 只表示候选获得进入独立评测的资格，质量、安全、鲁棒性和成本由 Part 05 再检查。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 把并列候选送入生命周期交接流程：分支状态、身份核对、产物检查、恢复与交接四个阶段 | 读 |
| Step 2 | 用证据身份判断候选是否仍然有效：基线、偏好数据、参考模型、评测版本、产物哈希哪一项变了，对两种候选各有什么影响 | 读 |
| Step 3 | 从最晚可信状态恢复并形成交接状态：四个失败阶段各能复用什么 | 读，对照上面的恢复例子 |
| Step 4 · TODO 1 | 检查候选血缘与分支证据有效期 | 在 `check_candidate_freshness` 里补 2 行：先按分支判断证据是否有效，再和共同检查一起取与 |
| Step 4 · TODO 2 | 按失败阶段选择恢复点 | 在 `choose_recovery_action` 里补 2 行：按顺序找第一个可用的可信状态，再选对应动作，没有就用回退动作 |
| Step 4 · TODO 3 | 传播上游状态并形成交接判断 | 在 `decide_lifecycle_handoff` 里补 1 行：有硬失败是 `reject`，上游是 `tune` 就是 `tune`，否则 `ready_for_evaluation` |
| 测试 | 检查基线推进、在线窗口过期、评测或分支身份缺失、四个阶段的恢复动作、上游三种状态的交接结果 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 读取 62、84、85 节的记录和适配器目录，重新计算哈希，演练四个阶段的故障恢复 | 只需要 CPU，但要有 62、84、85 节留下的真实清单和适配器目录；默认是 `RUN_MODE = "dry_run"`，只检查配置，不审核真实产物 |

Step 1–4 只需要 CPU。Step 5 也不需要 GPU，但要用到 62、84、85 节留下的真实产物。

## 容易踩的坑

- **基线换了还沿用旧候选**：所有候选都必须来自当前基线。测试里源策略版本从 `sft-v1` 变成 `sft-v0`，新鲜度就是假。基线、评测集或验证规则变化后，旧结果只能保留为历史记录。
- **DPO 和在线候选用同一套检查**：DPO 要有偏好证据版本；GRPO/RLVR 还要有验证器身份，并且窗口没过期。上游是 `accept` 还是 `tune` 不属于新鲜度，测试里上游为 `tune` 的候选也可以是新鲜的。
- **跳过较近的可信状态**：按失败阶段由近到远找第一个可用状态。奖励失败且轨迹可信时复用轨迹；更新失败时先选检查点，再选基线，都没有才回退。
- **把选好动作当成已经恢复**：解析提醒，动作选择不等于已恢复优化器和消费账本，不能单凭这个函数证明不会重复消费。恢复完成后仍要重新检查血缘、新鲜度、产物和回滚条件。
- **把 `ready_for_evaluation` 当成最终采用**：它只表示候选获得进入独立评测的资格。本节不比较 DPO 和 GRPO 的指标，也不形成最终采用结论。

## 读完后自查

- [ ] 能说出 DPO 候选和在线候选的分支证据分别要检查什么。
- [ ] 窗口年龄上限是 2，两个候选的 `window_age` 分别是 2 和 3，能判断哪个过期。
- [ ] 奖励失败时，轨迹已保存、轨迹没保存但问题队列还在、两者都没有，分别该做什么。
- [ ] 硬检查都通过时，上游是 `reject`、`tune`、`accept`，交接状态分别是什么。

## 下一步

- 打开 notebook：[86. 在线后训练生命周期与恢复](../86_DPO_Online_Benchmark.md)
- 建议先读：[51. 在线 DPO · 小白入门](./51_Online_DPO.md)、[PT-RL-SYSTEM. 采样、训练与奖励系统 · 小白入门](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)、[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)、[84. DPO 偏好优化项目 · 小白入门](./84_DPO_Preference_Project.md)
- 做完之后：[66. 单卡推理测量基线 · 小白入门](./66_Inference_Performance_Comparison.md)
