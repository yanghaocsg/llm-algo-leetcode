# 51. 在线 DPO — 小白讲解

> 对应 notebook：[`51_Online_DPO.ipynb`](./51_Online_DPO.ipynb)

## 一句话总结
在线 DPO 的关键不是「收到偏好就更新」，而是先做准入：这条偏好对是否完整、是否来自训练切分、是否命中风险、生成它的策略版本和当前策略差多少、是否已经被消费过。通过检查的样本按 ID 稳定排序，取前 `max_pairs` 条组成一个身份可复现的窗口，只触发一轮更新；更新后再用非平局胜率、重复评测通过率、安全指标、一票否决测试和成本，决定 `adopt / defer / rollback / halt`。本节用三个 TODO 和一组边界值测试把这条状态机写清楚。

## 为什么需要它：痛点是什么
离线 DPO 的数据是提前收集好的，训练一次就结束；在线 DPO 则要持续吃新反馈。初学者容易把它理解成「来一条偏好对就 `train_step` 一次」，于是遇到几类问题：反馈来自已经被替换掉的旧策略，分布对不上；评测集里的样本混进训练，分数虚高；同一条反馈被重复消费，某一轮被它反复影响；高风险样本没经过人工复核就进了更新；以及更新后安全指标掉了，却没有可回滚的检查点。

生活类比：餐厅根据顾客反馈改菜单。不能把今天的每条意见立刻下锅：要先确认意见来自哪一版菜单、是不是同一位顾客重复提交、有没有涉及过敏原的风险、是不是来自竞争对手的试吃；改完一轮后，还要用没参与试吃的顾客做盲测。万一出现食品安全问题，必须能立刻退回上一版菜单，否则只能停业整顿。

## 核心问题
- 一条在线偏好对要通过哪些检查才能进入本轮更新？每个拒绝原因的顺序为什么重要？
- 策略版本差 $\Delta v$ 为什么必须落在 $[0, L_{max}]$？版本比当前还新为什么要拒绝？
- 固定窗口怎样做到「同样输入产生同样窗口 ID」？跨轮消费和流内重复有什么区别？
- 内容冲突的同一 `pair_id` 为什么整组丢弃，而不是保留先到的那条？
- 更新后哪些指标是硬门槛、哪些只是软门槛？为什么胜率提高仍可能回滚？
- 没有可用检查点时，为什么是 `halt` 而不是 `rollback`？

## 解决思路：一步步看懂

**第 1 步：在线偏好的生命周期。** Step 1 把流程分成五段：成对构造（同一任务下的优选/非优选回答）、反馈准入（身份、策略版本、来源、切分、风险）、固定窗口更新（一轮只消费排序和容量都固定的清单）、独立评测（候选与固定基线用同一评测集、指标和阈值）、采用与恢复（采用则登记为下一轮基线，退化则加载最近一个通过评测的检查点）。本节代码只覆盖准入、窗口和决策三段，训练本身不在 notebook 里执行。

**第 2 步：版本差与窗口的形式化。** 设当前目标策略版本为 $v_{current}$，偏好对的生成策略版本为 $v_{generated}$，允许的最大版本差为 $L_{max}$，则

$$
\Delta v=v_{current}-v_{generated},\qquad 0\leq \Delta v\leq L_{max}.
$$

$\Delta v<0$ 表示这条数据来自比当前目标还新的策略，在线场景里属于「未来数据」，不能拿来更新当前版本；$\Delta v>L_{max}$ 表示数据太旧，生成它的策略和当前策略差太远。通过全部准入的偏好对按 ID 稳定排序，取前 $M$ 条：

$$
I_t=\mathrm{Sort}(\lbrace id_i\mid pair_i\text{ 通过准入}\rbrace)[:M],\qquad
w_t=H(v_t,I_t).
$$

其中 $M$ 是 `max_pairs`，$v_t$ 是第 $t$ 轮目标策略版本，$w_t$ 是窗口 ID，$H$ 是稳定摘要函数。相同的策略版本和入选 ID 顺序必须产生相同 $w_t$；容量外的样本只是延后，不能悄悄挤进本轮窗口。

**第 3 步：单条偏好对的准入（TODO 1）。** `classify_preference_pair(pair, current_policy_version, max_policy_lag)` 返回一个字符串：`eligible` 或一个**首要**拒绝原因。参考答案的检查顺序是：

1. `current_policy_version` 和 `max_policy_lag` 必须是非负整数，且不能是 `bool`（`bool` 是 `int` 的子类，`True` 会被误当成 1），否则抛 `ValueError`；
2. `pair_id`、`prompt_id`、`chosen`、`rejected` 缺失，或没有 `policy_version` 键 → `missing_identity`；
3. `policy_version` 必须是整数、非布尔、非负，否则 `missing_identity`；
4. 把 `chosen`/`rejected` 做 `casefold()` 和空白归一化后比较，相等 → `invalid_pair`；
5. `split != 'train'` → `evaluation_leakage`；
6. `risk` 为真 → `risk`；
7. `policy_lag = current_policy_version - generated_version`，小于 0 → `future`，大于 `max_policy_lag` → `stale`；
8. 全过 → `eligible`。

顺序不是随便排的。一条既来自评测集、又命中风险的样本，只会返回 `evaluation_leakage`；一条既是旧版本、又有风险的样本，返回 `risk`。这样每个样本只贡献一个拒绝计数，不会在多个桶里重复累计。归一化比较也必要：`' Same '` 和 `'same'` 归一化后相等，应判 `invalid_pair`，否则模型会被要求「偏好一个和另一个其实相同的回答」。`test_pair_admission_reasons` 用默认 `version=10`、`current=10`、`max_lag=2` 覆盖了全部原因：`ok` → `eligible`；同答 → `invalid_pair`；`split='evaluation'` → `evaluation_leakage`；`risk=True` → `risk`；`version=7` → `stale`（差 3）；`version=11` → `future`（差 -1）；`10.9`、`True`、`None` → `missing_identity`。

**第 4 步：形成固定窗口（TODO 2）。** `build_preference_window` 先拒绝 `max_pairs <= 0`，然后遍历反馈流：

```python
if pair_id in consumed_ids:
    reason = 'consumed'
elif pair_id in seen_identities:
    reason = 'duplicate' if seen_identities[pair_id] == identity_payload else 'identity_conflict'
else:
    reason = classify_preference_pair(pair, current_policy_version, max_policy_lag)
```

历史窗口消费过的 ID 优先判 `consumed`；同一轮里再次出现、且身份字段完全相同的判 `duplicate`；同一 `pair_id` 但 `prompt_id`/`chosen`/`rejected`/`policy_version`/`split`/`source`/`risk` 有任何不同，判 `identity_conflict` 并记录到 `conflicted_ids`。最后：

```python
ordered_pairs = sorted(
    (pair for pair in accepted_pairs if str(pair['pair_id']).strip() not in conflicted_ids),
    key=lambda item: str(item['pair_id']),
)
selected_pairs, overflow_pairs = ordered_pairs[:max_pairs], ordered_pairs[max_pairs:]
identity_payload = json.dumps([current_policy_version, pair_ids], separators=(',', ':'))
window_id = hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:16]
```

窗口只由「策略版本 + 入选 ID 列表」决定，所以把整个反馈流反转，只要准入集合相同，算出的 `window_id` 也相同（测试里的 `replay`）。容量外的 `overflow_pairs` 单独列出，下一轮还能用；它们不计入 `rejection_counts`。`eligible_sources` 是入选样本来源的去重排序，用来知道这一轮被 `prod` 和 `shadow` 各贡献了多少样本。

两个容易写错的地方。第一，内容冲突必须**整组排除**，不能按到达顺序保留第一条：`build_preference_window` 最后用 `conflicted_ids` 过滤所有同 ID 样本，测试 `conflicting` 两条同 ID、不同 `chosen`，结果 `pair_ids == []`、`rejection_counts == {'identity_conflict': 1}`，而且反转输入顺序结果不变。第二，窗口 ID 的拼接不能有歧义：测试用 `pair_id='a|b'` 与 `pair_id='a'`、`'b|c'` 两组输入，确保 JSON 列表的边界不会因为分隔符相同而算出同一个摘要，`left['window_id'] != right['window_id']`。

`test_preference_window_identity_and_capacity` 的主场景是：流里依次有 `b(shadow)`、`a`、`c`、重复的 `b(shadow)`、已消费的 `used`、`stale(version=6)`、`unsafe(risk=True)`，参数 `current=10, max_lag=2, max_pairs=2, previously_consumed_ids={'used'}`。结果 `pair_ids == ['a', 'b']`、`overflow_pair_ids == ['c']`、`rejection_counts == {'consumed': 1, 'duplicate': 1, 'risk': 1, 'stale': 1}`、`eligible_sources == ['prod', 'shadow']`。注意 `stale` 的版本 6 与当前 10 差 4，超过 `max_lag=2`，所以虽然它也是旧数据，却只记为 `stale` 而不是 `consumed`。

**第 5 步：更新后的采用/回滚决策（TODO 3）。** `recommend_online_dpo` 先校验：`win_rate`、`repeat_pass_rate`、`safety_score`、`update_cost` 四个字段必须齐全；所有比率和成本必须有限；三个比率必须在 $[0,1]$；门槛非负、成本为正。然后计算比较量：

```python
comparison = {
    'win_rate_gain': candidate_run['win_rate'] - baseline_run['win_rate'],
    'repeat_pass_rate_delta': candidate_run['repeat_pass_rate'] - baseline_run['repeat_pass_rate'],
    'safety_delta': candidate_run['safety_score'] - baseline_run['safety_score'],
    'cost_increase_ratio': (candidate_run['update_cost'] - baseline_run['update_cost'])
                           / baseline_run['update_cost'],
    'blocking_failure_count': int(candidate_run.get('blocking_failures', 0)),
}
```

形式上，$\Delta W=W_{c}-W_{b}$、$\Delta R=R_{c}-R_{b}$、$\Delta S=S_{c}-S_{b}$，成本比 $\rho_C=(C_c-C_b)/C_b$。决策按固定顺序：

1. `safety_delta < -max_safety_drop` → `rollback`（有检查点）或 `halt`（无检查点），原因 `safety_regression`；
2. `blocking_failure_count > 0` → 同样回滚/停止，原因 `blocking_evaluation_failure`；
3. `repeat_pass_rate_delta < -max_repeat_pass_rate_drop` → 回滚/停止，原因 `repeat_pass_rate_regression`；
4. `cost_increase_ratio > max_cost_increase_ratio` → `defer`，原因 `cost_gate_not_met`；
5. `win_rate_gain >= min_win_rate_gain` → `adopt`，原因 `quality_and_cost_pass`；
6. 否则 `defer`，原因 `soft_gate_not_met`。

默认门槛是 `max_repeat_pass_rate_drop=0.05`、`max_safety_drop=0.02`、`max_cost_increase_ratio=0.5`。为什么安全和重复性排在成本、胜率前面？因为它们是一票否决：不能因为胜率涨了 3 个点，就接受安全分掉 5 个点。为什么 `halt`？没有可用检查点时，「回滚」这个动作根本执行不了，只能停止自动更新，等人修。

**第 6 步：用测试数字读懂边界。** `test_adopt_defer_rollback_and_halt` 的基线是 `win_rate=0.52`、`repeat_pass_rate=0.82`、`safety_score=0.95`、`update_cost=1.0`。候选 `adopt` 是 `0.59 / 0.79 / 0.94 / 1.2`：胜率 +0.07 过 0.05，重复率 -0.03 在 -0.05 内，安全 -0.01 在 -0.02 内，成本 +20% 在 50% 内 → `adopt`。把胜率改成 0.54（+0.02）→ `defer` 的 `soft_gate_not_met`。把成本改成 1.8（+80%）→ 即使胜率 +0.07，也先被成本门槛拦成 `defer` 的 `cost_gate_not_met`。把重复率改成 0.74（-0.08）→ `repeat_pass_rate_regression`。加 `blocking_failures=1` → `blocking_evaluation_failure`。把安全分改成 0.90（-0.05）→ `rollback`；同一个候选在 `rollback_ready=False` 时 → `halt`。

边界组是 `win=0.57`（+0.05，刚好达到 `>=`）、`repeat=0.77`（-0.05，不满足严格 `< -0.05`）、`safety=0.93`（-0.02，不满足严格 `< -0.02`）、`cost=1.5`（+50%，不满足严格 `> 0.5`），四项全部卡在等号上，结果仍是 `adopt`。这说明门槛的语义是「超过才拒绝，等于放行」。最后把胜率改成 `nan`，函数必须抛 `ValueError`。

**第 7 步：和 TRL 的数据契约对账。** `check_online_dpo_data_contract` 用 `build_preference_window` 处理三条反馈：`pair-b`（版本 4）、`pair-a`（版本 5）、`pair-old`（版本 1），当前版本 5、`max_policy_lag=1`、`max_pairs=2`。`pair-b` 差 1 通过，`pair-a` 差 0 通过，`pair-old` 差 4 被拒；排序后窗口是 `['pair-a', 'pair-b']`。然后用 `datasets.Dataset.from_list` 构造 `pair_id/prompt/chosen/rejected` 四列，实例化 CPU `DPOConfig(output_dir=..., beta=0.2, max_length=256, use_cpu=True, dataloader_pin_memory=False, report_to='none')`。检查项包括窗口身份保持、DPO 字段完整、`chosen != rejected`、配置保留。它只证明窗口能喂进 TRL 的训练入口，不证明训练效果，也不覆盖反馈时延、采用和回滚。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 在线 DPO | 持续收集新偏好对、分批更新策略的 DPO 流程 |
| 偏好对 | 同一任务下一条优选回答 `chosen` 和一条非优选 `rejected` |
| 策略版本差 $\Delta v$ | 当前目标策略版本减数据生成策略版本 |
| `max_policy_lag` | 允许的最大版本差，超过就算 `stale` |
| `missing_identity` | 偏好对缺少唯一身份或合法版本号 |
| `evaluation_leakage` | 样本来自评测切分，不能进训练 |
| `identity_conflict` | 同一 `pair_id` 出现内容不同的记录，整组丢弃 |
| 反馈窗口 | 本轮固定容量、固定顺序的偏好对清单 |
| `window_id` | 由策略版本和入选 ID 列表哈希出的窗口身份 |
| 非平局胜率 | 排除平局后候选被选中的比例 |
| 重复评测通过率 | 同一候选重复评测仍通过的比例，衡量稳定性 |
| `adopt / defer / rollback / halt` | 采用 / 保留基线补证据 / 有检查点回滚 / 无检查点停止 |

## 小白常见疑问
**Q1：为什么不能收到一条反馈就立刻更新？**
因为这条反馈可能来自旧策略、评测集、风险样本，或者已经被上一轮消费过。`build_preference_window` 先做 `consumed`、`duplicate`、`identity_conflict` 和 `classify_preference_pair` 四类检查，再把通过者按 ID 排序、按容量截断。没有这层准入，更新就不可复现，也无法解释指标变化来自哪批数据。

**Q2：为什么版本差不能为负？比当前更新的策略生成的数据不是更好吗？**
在线流程里，当前目标策略是 $v_{current}$，数据应该由它或它之前的策略产生。来自更新策略的数据会让「这轮更新到底在拟合谁」变得不清楚，所以 `policy_lag < 0` 直接判 `future`。测试用 `version=11`、当前 10 验证了这一点。

**Q3：窗口 ID 为什么要把策略版本也哈希进去？**
因为同样的偏好对 ID 在不同策略版本下代表不同的更新轮次。只哈希 ID 列表，两个不同版本的窗口可能得到同一个 ID，无法追溯「这轮到底对应哪个模型」。参考答案把 `[current_policy_version, pair_ids]` 一起 JSON 序列化后做 SHA-256，取前 16 个十六进制字符。

**Q4：胜率涨了 7 个点，为什么安全分掉 5 个点就要回滚？**
因为安全是一票否决门槛，不是可以线性交换的分数。`recommend_online_dpo` 先检查 `safety_delta < -max_safety_drop`，命中就 `rollback`/`halt`，根本不会走到胜率分支。测试里 `unsafe` 的胜率仍是 +0.07，但结果仍是回滚。

**Q5：`defer` 和 `rollback` 的区别是什么？**
`defer` 是「没有硬失败，但软门槛没达到」：成本超了，或胜率增幅不够，保留当前基线、补证据后再试。`rollback` 是「已经出现安全、重复性或一票否决测试的硬退化」，要退回上一个通过评测的检查点。没有检查点时连回滚都做不到，所以是 `halt`。

## 动手验证：怎么知道自己做对了
`test_pair_admission_reasons`：`classify_preference_pair(_pair('ok'), 10, 2) == 'eligible'`；同答、`split='evaluation'`、`risk=True`、版本 7、版本 11、带空白同答、版本 `10.9`/`True`/`None` 分别返回 `invalid_pair`、`evaluation_leakage`、`risk`、`stale`、`future`、`invalid_pair`、`missing_identity`。

`test_preference_window_identity_and_capacity`：主场景必须得到 `pair_ids == ['a', 'b']`、`overflow_pair_ids == ['c']`、`policy_version == 10`、反转输入后 `window_id` 相同、`rejection_counts == {'consumed': 1, 'duplicate': 1, 'risk': 1, 'stale': 1}`、`eligible_sources == ['prod', 'shadow']`。两条同 ID 不同内容的记录必须得到空 `pair_ids` 和 `{'identity_conflict': 1}`，反转顺序结果不变；`_pair('a|b')` 与 `_pair('a') + _pair('b|c')` 的 `window_id` 必须不同。

`test_adopt_defer_rollback_and_halt`：`adopt` 候选得到 `adopt`；胜率 0.54 得到 `defer`；成本 1.8 的 `reason == 'cost_gate_not_met'`；重复率 0.74 的 `reason == 'repeat_pass_rate_regression'`；`blocking_failures=1` 的 `reason == 'blocking_evaluation_failure'`；安全 0.90 且有检查点得到 `rollback`，无检查点得到 `halt`；边界组四项卡等号仍得到 `adopt`；胜率为 `nan` 抛 `ValueError`。

三个测试都通过会打印 `✅ 在线反馈窗口、更新对照与回滚准备测试通过。`；任一失败会被 `try/except` 包成 `NotImplementedError("请先完成 TODO 代码！")`。notebook 没有保存任何单元格输出。可选的 TRL 核对通过时返回 `status: 'passed'`、`window_id`、`pair_count` 和四项布尔检查；它只在 `PT-CPU` 环境里验证数据与 `DPOConfig` 契约，不执行训练。
