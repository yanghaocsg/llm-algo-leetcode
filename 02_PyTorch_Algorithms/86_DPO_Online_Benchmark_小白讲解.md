# 86. 在线后训练生命周期与恢复 — 小白讲解

> 对应 notebook：[`86_DPO_Online_Benchmark.ipynb`](./86_DPO_Online_Benchmark.ipynb)

## 一句话总结
候选适配器训练完之后，"文件还在"不等于"它还是那次训练的产物"。本节把候选清单、SFT 源基线和本地适配器目录放在一起核对：三个 TODO 分别检查来源与分支记录是否仍然新鲜、按失败阶段选择最小恢复动作、把上游决策与本地检查合成为 `reject` / `tune` / `ready_for_evaluation` 三态。Step 5 再重新计算候选与回退目录的哈希，输出可交接的候选清单；历史上 v6 的两条候选（DPO v5 与 GRPO/RLVR v7）都通过并标为 `ready`。

## 为什么需要它：痛点是什么
初学者容易把"训练完成"当成一个单点事件：脚本退出、目录里多了 `adapter_config.json` 和权重文件，就认为可以拿去评测。真实流程里至少有三类错位：

1. **基线换了**：候选是基于 `sft-v1` 训练的，但当前交接的是 `sft-v2`，源策略不匹配，评测结果无法归因。
2. **目录内容变了**：文件还在，但有人覆盖、补写或截断过，哈希与清单不一致。
3. **失败后乱重试**：更新到底提交没有？如果已经提交还重放一次，可能把同一批数据更新两遍；如果状态未知就回退，可能丢掉已经生效的更新。

生活类比：快递柜取件。取件码（候选标识）要对，柜门（目录哈希）要没被撬过，包裹上的寄件人（源策略）要和订单一致；如果取件失败，先看是"还没放进去"（未提交）、"已经放进去了"（已提交）还是"不知道放没放"（未知），再决定是重放、查记录还是隔离，而不是一律重试。

## 核心问题
- "候选仍然新鲜"要检查哪些字段？离线 DPO 和在线 GRPO/RLVR 的分支条件差在哪？
- 窗口年龄 `window_age` 是什么？为什么 DPO 候选的 `window_age=99` 也能通过？
- 更新失败时，为什么必须先区分 `not_committed`、`committed`、`unknown`？
- 恢复动作的优先级表怎么读？没有检查点也没有基线时应该做什么？
- 上游 `accept` / `tune` / `reject` 与本页硬检查怎样合成最终状态？
- `ready_for_evaluation` 到底证明了什么、没证明什么？

## 解决思路：一步步看懂

**第 1 步：三类核对不要混成一个判断。** Step 1 把工作分成三块：记录检查（源策略是否相符、候选与评测标识是否存在、分支特有字段是否齐全）、文件检查（重新计算候选和回退适配器的哈希，与清单值比较）、恢复动作（按失败阶段和"声明可用的状态"选择动作）。notebook 特别强调"记录齐全""文件相符""恢复已经执行"是三件不同的事。本页只做前两项核对和第三项的建议；真正的恢复还要加载检查点、核对优化器状态、随机数状态和消费记录。

**第 2 步：TODO 1——共同检查 + 分支检查。**

```python
common_checks = {
    "candidate_kind": candidate.get("candidate_kind") in {"dpo", "grpo", "rlvr"},
    "source_policy": candidate.get("source_policy_version") == active_policy_version,
    "candidate_identity": bool(candidate.get("policy_version")),
    "evaluation_identity": bool(candidate.get("evaluation_version")),
}
if candidate_kind == "dpo":
    branch_evidence_fresh = bool(candidate.get("preference_evidence_version"))
else:
    branch_evidence_fresh = (
        bool(candidate.get("verifier_version"))
        and 0 <= candidate.get("window_age", -1) <= maximum_window_age
    )
checks = {**common_checks, "branch_evidence": branch_evidence_fresh}
fresh = all(checks.values())
```

共同检查有四项：候选类型必须是 `dpo`、`grpo`、`rlvr` 之一；`source_policy_version` 必须等于当前激活的 SFT 基线；`policy_version` 和 `evaluation_version` 必须非空。为什么用 `bool(...)` 而不是 `is not None`？因为空字符串 `""` 也代表"标识没填"，`bool("")` 为 `False` 正好挡掉。

分支检查是两条完全不同的证据链：

- 离线 DPO 需要 `preference_evidence_version` 非空，证明这份候选有对应的偏好数据/训练记录。它**不看** `window_age`，所以测试里 DPO 候选即使 `window_age=99` 仍然 `fresh=True`。
- 在线 GRPO/RLVR 需要 `verifier_version` 非空，并且 `0 <= window_age <= maximum_window_age`。测试里在线候选窗口年龄 2、上限 2 时通过，改成 3 就失败；`verifier_version=""` 也失败。

`window_age` 是"当前窗口序号减生成窗口序号"，例如当前窗口 5、生成窗口 3，年龄是 2。它不是"过了两小时"，也不是"策略更新了两次"，只是窗口计数。`candidate.get("window_age", -1)` 的默认值 -1 是故意的：字段缺失时 `0 <= -1` 为假，直接判不新鲜，避免"没填就当成最新"。

返回值除了 `fresh`，还有 `failed_checks` 列表，按 `checks` 的插入顺序列出失败项。这个列表很重要：交接时不仅要知道"不新鲜"，还要知道"是源策略不符，还是验证器缺失，还是窗口过期"。

**第 3 步：TODO 2——先看提交状态，再谈恢复。**

更新失败（`stage == "update"`）时，函数先读 `commit_state = failure.get("commit_state", "unknown")`，只接受三个值：

```python
if commit_state != "not_committed":
    action = ("inspect_committed_update" if commit_state == "committed"
              else "quarantine_and_reconcile_commit")
    return {"stage": stage, "recovery_state": None, "reused_state": False, "action": action}
```

- `committed`：更新已经提交，不能再重放；动作是 `inspect_committed_update`，去核对已完成的记录。
- `unknown`：状态不明，先隔离本批次，动作是 `quarantine_and_reconcile_commit`，查提交日志和检查点。
- 省略 `commit_state`：按 `unknown` 处理，同样隔离核查。

只有确认 `not_committed`，才进入恢复点选择：

```python
recovery_state = next(
    (state for state in recovery_candidates_by_stage[stage] if available.get(state)), None
)
action = (action_by_stage_and_state[(stage, recovery_state)]
          if recovery_state is not None else fallback_action_by_stage[stage])
```

`recovery_candidates_by_stage` 的顺序就是优先级：

| 失败阶段 | 候选状态（从前到后） | 有状态时的动作 | 都没有时的回退 |
|---|---|---|---|
| `sampling` | `prompt_queue` | `retry_sampling_from_prompt_queue` | `rebuild_prompt_queue` |
| `reward` | `trajectories` → `prompt_queue` | `resume_reward_from_saved_trajectories` → `restart_sampling_from_prompt_queue` | `restart_sampling` |
| `update` | `pre_update_checkpoint` → `verified_baseline` | `restore_pre_update_checkpoint` → `rollback_to_verified_baseline` | `block_update` |
| `artifact_validation` | `verified_baseline` | `rollback_to_verified_baseline` | `block_release` |

这张表的逻辑是"复用最晚、最省工作的可信状态"：奖励阶段如果回答还在，就只重算奖励；回答丢了但问题队列还在，就重新生成；更新阶段优先回到更新前检查点，其次回退已验证基线；两者都没有就停止，不能凭空继续。`artifact_validation` 失败只信已验证基线，没有基线就阻止发布。`next(...)` 取第一个"可用"的状态，`available.get(state)` 用真值判断，所以清单里应该写 `True/False`，不要写字符串。

测试 `test_recovery_actions` 覆盖了每条分支：`reward` 只有 `trajectories` 时返回 `recovery_state == "trajectories"`、`reused_state=True`；`trajectories=False, prompt_queue=True` 时退到 `prompt_queue`；两个都没有时 `restart_sampling`；`update` 确认未提交且两个状态都没有时 `block_update`，只有基线时 `rollback_to_verified_baseline`。`test_uncertain_or_completed_update_is_not_replayed` 还断言：状态为 `None`、`"unknown"`、`"committed"` 时 `recovery_state` 都是 `None`、`reused_state` 都是 `False`，并且传入的 `available` 字典**没有被修改**——恢复选择是只读决策，不能顺手把状态清掉。

**第 4 步：TODO 3——硬失败优先，再传播上游。**

```python
hard_checks = {
    "freshness": freshness["fresh"],
    "rollback": evidence["rollback_ready"],
    "artifact": evidence["artifact_verified"],
}
hard_failures = [name for name, passed in hard_checks.items() if not passed]
if upstream_decision not in {"accept", "tune", "reject"}:
    hard_failures.append("upstream_state")
elif upstream_decision == "reject":
    hard_failures.append("upstream_reject")
handoff_status = ("reject" if hard_failures
                  else "tune" if upstream_decision == "tune"
                  else "ready_for_evaluation")
```

三个硬检查是"候选新鲜""回退基线可用""候选产物校验通过"。任一失败就 `reject`，即使上游已经 `accept`。上游决策非法（不在三态里）记 `upstream_state`；上游明确 `reject` 记 `upstream_reject`。只有硬检查全过，才传播上游状态：上游 `tune` 就保持 `tune`，上游 `accept` 才进入 `ready_for_evaluation`。

注意这里没有把"恢复动作"当成硬检查。恢复动作是失败之后的建议，不是候选能否评测的前置条件；一个候选可以既 `ready_for_evaluation`，又在另一份失败场景记录里给出恢复建议。这也是 Step 1 强调"三项都应留下具体结果"的原因。

**第 5 步：Step 5 怎么把规则用到真实清单上。** 5.1 的 `RUN_MODE` 默认 `dry_run`，只检查配置；改成 `real_artifact` 才读取真实清单，并且 `RUN_LABEL` 不能还是 `replace_with_unique_label`。固定输入是：

- SFT 基线：`benchmarks/results/62_sft_delivery/sft_delivery_manifest_sft_delivery_v2_1.json`
- 候选：84 节的 `dpo_manifest_dpo_offline_len1024_v5.json` 和 85 节的 `grpo_manifest_grpo_rlvr_len384_v7.json`
- `MAXIMUM_WINDOW_AGE = 2`，`ACTIVE_WINDOW_INDEX = 1`；85 v7 在窗口 1 生成，所以年龄 `1 - 1 = 0`

`real_artifact` 模式先检查结果文件是否已存在（存在就要求换 `RUN_LABEL`），再用 `require_input_file` 确认 SFT 清单存在，并至少找到一个候选清单；两个候选都没生成就抛 `FileNotFoundError("84/85 尚无候选 manifest")`。

5.3 的 `check_candidate_freshness_for_run` 是 CPU 版本的"真实清单适配器"：它用 `semantic_id` 而不是 `candidate_kind` 判断分支，`PT-DPO-OFFLINE` 看 `source_evidence`，`PT-GRPO-GROUPWISE` 看 `trajectory_identity.verifier_version`。`window_age` 由 `ACTIVE_WINDOW_INDEX - candidate.get("evidence_window_index", -1)` 算出。这里有一个容易忽略的细节：如果清单里没有 `evidence_window_index`，默认 -1 会得到年龄 `1 - (-1) = 2`，恰好落在上限 2 内；所以真实清单必须显式写这个字段，不能依赖默认值来"碰巧通过"。

5.4 逐候选做三件事：用 `active_policy_version = sft_manifest["artifacts"]["adapter"]["sha256"]` 比对源策略；用 `hash_directory` 重新计算候选和 SFT 回退目录的哈希；把 `freshness`、`rollback_ready`、`artifact_verified` 送进 `decide_lifecycle_handoff_for_run`。证据结构里同时保留 `artifact_check` 和 `rollback_check` 的 `expected` / `actual`，这样失败时能直接看出"是期望值缺失，还是实际目录算出来不一样"。

5.5 读取证据，只展示紧凑状态：哪些 `ready_for_evaluation`、每个 `semantic_id` 的 `handoff_status`。5.6 才写候选清单，`handoff_status` 是 `"ready"` 当且仅当至少有一个候选 ready，否则 `"no_ready_candidate"`。历史 `post_training_lifecycle_v6` 的结果是两条候选的源基线、分支记录、目录哈希与回退目录都通过，状态 `ready_for_evaluation`，汇总 `ready`。

**第 6 步：结果怎么读。** 这份清单回答的是"来源和文件是否对得上"，不回答"哪个模型更好"。`ready_for_evaluation` 只表示可以进入独立任务评测；DPO 的偏好边际和 GRPO/RLVR 的数学题得分不能直接放在一张表里排序。`tune` 表示本页检查通过但上游还在调整；`reject` 表示来源、产物、回退或上游至少一项不成立。恢复建议也只是"应该复用哪个状态、下一步做什么"，页面明确说明实际恢复要加载状态并核对优化器、随机状态和消费记录，不能把建议当成已经恢复。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 候选（candidate） | 某个分支训练出的策略版本，附带来源、评测和产物记录 |
| 源策略版本 | 候选训练所基于的 SFT 适配器标识，必须与当前基线一致 |
| `window_age` | 当前窗口序号减生成窗口序号，衡量在线候选有多"旧" |
| 分支证据 | DPO 看偏好记录，GRPO/RLVR 看验证器标识和窗口年龄 |
| 目录哈希 | 对适配器目录内容重新计算指纹，用来发现文件被改过 |
| `pre_update_checkpoint` | 更新前的检查点，更新失败且确认未提交时优先回到这里 |
| `verified_baseline` | 已验证的基线，检查点不可用时才回退到这里 |
| `commit_state` | 更新是否已提交：`not_committed` / `committed` / `unknown` |
| `unknown` 提交状态 | 不知道更新有没有生效，先隔离并核对日志，不能直接重放 |
| `ready_for_evaluation` | 记录与文件核对通过，可以进入独立任务评测 |
| `tune` / `reject` | 本页通过但上游仍要调整 / 至少一项硬检查失败 |
| `hash_directory` | 对目录内文件逐个计算哈希并汇总的工具函数 |

## 小白常见疑问

**Q1：候选是 DPO 的，`window_age=99` 为什么还能 `fresh=True`？**
因为 DPO 分支只要求 `preference_evidence_version` 非空，窗口年龄是**在线**分支的条件。`test_candidate_freshness` 里第一条候选就是 `candidate_kind="dpo"`、`window_age=99`、上限 2，断言 `fresh` 为真。它不是在说"99 的窗口没问题"，而是在说"离线 DPO 不使用这个窗口计数"。如果候选类型写成 `grpo`，同样的 99 立刻失败。

**Q2：`commit_state` 省略时为什么要按 `unknown` 处理，而不是按 `not_committed`？**
因为"没写"和"确认没提交"是两种证据强度。按 `not_committed` 处理会直接进入恢复点选择，可能在更新其实已经生效时重放；按 `unknown` 处理则走 `quarantine_and_reconcile_commit`，先隔离再查日志。`test_uncertain_or_completed_update_is_not_replayed` 专门把 `commit_state` 省略、设为 `"unknown"`、设为 `"committed"` 三种情况放在一起，断言都不会选择恢复点，且 `available` 字典不变。

**Q3：更新失败、确认未提交，但检查点和基线都没有，为什么是 `block_update` 而不是随便重跑？**
因为没有可信的起点。`fallback_action_by_stage["update"]` 定义为 `block_update`，意思是停止更新，先补齐检查点或基线。随便重跑会让"这批数据到底更新了几次"无法追踪，也会破坏消费记录。`test_recovery_actions` 用 `{"stage": "update", "commit_state": "not_committed", "available": {}}` 断言这个动作。

**Q4：上游已经 `accept`，本页 `freshness=False`，最终是什么？**
`reject`。`decide_lifecycle_handoff` 先算硬失败，`freshness` 是三个硬检查之一；只要它为假，`handoff_status` 就是 `reject`，上游的 `accept` 不会覆盖它。反过来，上游 `tune` 且硬检查全过，结果是 `tune`，不是 `ready_for_evaluation`。测试 `test_lifecycle_handoff` 分别验证了这两条。

**Q5：`ready_for_evaluation` 是不是说明这个候选质量更好？**
不是。它只说明源策略匹配、候选/评测标识非空、分支证据齐全、目录哈希与清单一致、回退基线可用。质量比较要在统一任务、统一生成预算下另做；notebook 5.6 明确说"任务质量和实际恢复另行验证"，Part05 MiniMind 项目也不读取这份历史清单。

## 动手验证：怎么知道自己做对了

notebook 的测试单元按 `test_uncertain_or_completed_update_is_not_replayed()`、`test_candidate_freshness()`、`test_recovery_actions()`、`test_lifecycle_handoff()` 的顺序直接调用，最后打印 `所有测试通过！`。没有保存执行输出，以下以代码和断言为准。TODO 没填时，三个函数会分别抛 `NotImplementedError("TODO 1：请检查来源与分支条件")`、`NotImplementedError("TODO 2：请搜索最晚可信恢复点并选择对应动作")`、`NotImplementedError("TODO 3：请形成待评测状态")`。

- `test_candidate_freshness`：DPO 候选（`source_policy_version="sft-v1"`、`policy_version="dpo-v1"`、`evaluation_version="pref-eval-v1"`、`preference_evidence_version="pref-train-v1"`、`window_age=99`）在 `active_policy_version="sft-v1"`、上限 2 时必须 `fresh`；把 `decision` 改成 `"tune"` 仍然 `fresh`（`decision` 不参与新鲜度）；源策略改 `sft-v0`、偏好证据置空、候选类型改 `unknown`、评测标识置空，都必须不新鲜。在线候选 `grpo` + `verifier_version="math-v1"` + `window_age=2` 通过；`window_age=3` 或验证器置空失败。
- `test_recovery_actions`：`sampling` + `prompt_queue` → `retry_sampling_from_prompt_queue`；`reward` + `trajectories` → `resume_reward_from_saved_trajectories`，且 `recovery_state=="trajectories"`、`reused_state` 为真；`update` + `not_committed` + `pre_update_checkpoint` → `restore_pre_update_checkpoint`；`artifact_validation` + `verified_baseline` → `rollback_to_verified_baseline`；`reward` 只有 `prompt_queue` → `restart_sampling_from_prompt_queue`；`reward` 什么都没有 → `restart_sampling`；`update` 确认未提交且什么都没有 → `block_update`；确认未提交但只有基线 → `rollback_to_verified_baseline`。
- `test_lifecycle_handoff`：`("accept", fresh, rollback_ready, artifact_verified)` → `ready_for_evaluation`；上游 `tune` → `tune`；上游 `reject` → `reject`，且 `hard_failures` 含 `upstream_reject`；`fresh=False` 或 `rollback_ready=False` → `reject`。
- `test_uncertain_or_completed_update_is_not_replayed`：`available={"pre_update_checkpoint": True, "verified_baseline": True}` 时，`commit_state` 省略/`unknown` → `quarantine_and_reconcile_commit`，`committed` → `inspect_committed_update`；三种情况 `recovery_state is None`、`reused_state is False`，且 `available` 与调用前相等；`commit_state="maybe"` 必须抛 `ValueError`。

Step 5 没有单元测试。`dry_run` 只打印配置和候选路径，不写文件；`real_artifact` 成功时打印证据路径和候选数量，随后 5.5 打印 `ready_for_evaluation` 列表与每个候选的 `handoff_status`，5.6 打印 `handoff_path`、`handoff_status` 和 `ready_count`。核对时至少看四项：`active_policy_version` 是否等于 SFT 清单里的 adapter SHA256、每个候选的 `artifact_check.expected` 是否非空且等于 `actual`、`rollback_check.expected` 是否非空且等于 `actual`、`missing_candidate_manifests` 是否为空。历史 v6 的结论是两条候选都 `ready_for_evaluation`、汇总 `ready`；如果你复跑得到 `no_ready_candidate`，先查目录哈希和源策略，而不是先改 `MAXIMUM_WINDOW_AGE`。
