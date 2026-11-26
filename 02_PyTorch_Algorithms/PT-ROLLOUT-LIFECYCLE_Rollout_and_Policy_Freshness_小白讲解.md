# PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 — 小白讲解

> 对应 notebook：[`PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb`](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb)

## 一句话总结
本节把「一条生成轨迹还能不能用于当前策略更新」拆成两条独立证据：**版本号只说明新旧顺序，动作对数概率差才说明行为变了多少**。代码用 `LIFECYCLE_TRANSITIONS` 管轨迹从生成到消费的状态，用 `assess_policy_compatibility` 联合检查版本滞后与掩码内平均绝对对数概率差，用 `compare_and_reserve` 按修订号做竞争预留，用 `activate_policy_version` 在新权重提交、版本递增且所有必需工作进程确认后才切换活动指针。Step 5 还给出一次真实的单卡 verl 监督更新记录：8 条旧轨迹最终 7 条准入、1 条拒绝，说明"同一版本差"不等于"同一行为偏离"。

## 为什么需要它：痛点是什么
后训练系统里，生成（rollout）和训练（learner）往往异步：一批回答还在算奖励，学习器已经更新了一版策略。初学者容易犯三种错：

- **只看版本号**：两条轨迹都滞后 1 个版本，就认为它们一样旧。但策略在这一个版本里可能只微调了一点点，也可能发生了剧烈变化；版本差只记录顺序，不记录幅度。
- **把"生成过"当成"可以用"**：轨迹可能还在等奖励、已奖励、已预留、已消费，甚至提交结果不明。不区分状态，两个学习器可能把同一条轨迹重复绑定到不同更新。
- **权重还没同步完就切流量**：新 checkpoint 已保存，但某个生成进程还加载着旧版本；协调器提前切活动指针，用户就会拿到新旧混合的输出。

生活类比：餐厅换菜单。版本号是"第 3 版菜单"和"第 4 版菜单"的编号，但第 4 版可能只改了一个菜名，也可能整本重写。厨师（生成进程）必须确认自己拿到的是第 4 版、且内容和总部哈希一致，前台（协调器）才能把"当前菜单"的牌子换掉。点单记录（轨迹）也要标明"这桌是按第 3 版做的"，并且同一张单不能同时被两个服务员认领。

## 核心问题
- 一条轨迹从 `generated` 到 `consumed` 要经过哪些状态和事件？为什么 `abort` 只能把 `reserved` 退回 `rewarded`？
- 版本差 $\Delta v$ 和平均绝对对数概率差 $D_{beh}$ 分别回答什么问题？为什么两个都要检查？
- 动作掩码里的 `False` 位置为什么可以放任意大的数而不影响判断？
- 两个更新同时拿修订号 7 来预留，为什么只有一个成功？`commit_uncertain` 为什么进入 `quarantined` 而不能直接 `abort`？
- 策略激活需要同时满足哪四个条件？失败时为什么必须保留原指针？
- Step 5 的 8 条轨迹为什么是 7 准入、1 拒绝？这验证了什么、没验证什么？

## 解决思路：一步步看懂

### 第 1 步：生命周期状态表
```python
LIFECYCLE_TRANSITIONS = {
    ('generated', 'reward_ready'): 'rewarded',
    ('rewarded', 'reserve'): 'reserved',
    ('reserved', 'commit'): 'consumed',
    ('reserved', 'abort'): 'rewarded',
    ('reserved', 'commit_uncertain'): 'quarantined',
}
```

五条边对应五种合法动作：

| 当前状态 | 事件 | 下一状态 | 含义 |
|---|---|---|---|
| `generated` | `reward_ready` | `rewarded` | 评分完成，轨迹可用于更新 |
| `rewarded` | `reserve` | `reserved` | 某个更新把轨迹绑定到自己名下 |
| `reserved` | `commit` | `consumed` | 更新确认使用完毕 |
| `reserved` | `abort` | `rewarded` | 确认更新尚未执行，释放预留 |
| `reserved` | `commit_uncertain` | `quarantined` | 提交结果不明，暂停重新预留，等待核对 |

注意几个"没有"：

- 没有 `consumed → reserved`。已消费的轨迹不能重新申请，测试里 `('consumed', 'reserve')` 必须抛 `ValueError`。
- 没有 `generated → reserved`。评分没完成不能直接预留，必须先 `reward_ready`。
- 没有 `quarantined → rewarded`。提交状态不明时，`abort` 也不允许，测试 `test_uncertain_commit_is_quarantined` 专门断言这一点。

### 第 2 步：TODO 1 `transition_lifecycle`——查表，不猜
参考实现三行：

```python
transition = (state, event)
if transition not in LIFECYCLE_TRANSITIONS:
    raise ValueError(f'非法生命周期迁移：{state} + {event}')
return LIFECYCLE_TRANSITIONS[transition]
```

为什么用元组当字典键？因为 `(state, event)` 是天然的组合键，查表 $O(1)$，也避免写一长串 `if/elif`。为什么非法组合必须抛 `ValueError` 而不是返回原状态？因为状态机里"什么都没发生"和"发生了非法事件"是两种完全不同的情况：前者可能是幂等重试，后者说明调用方逻辑有 bug，静默吞掉会让账本停在错误状态。

测试 `test_lifecycle_transitions` 先验证四条正常边（`generated+reward_ready → rewarded`、`rewarded+reserve → reserved`、`reserved+commit → consumed`、`reserved+abort → rewarded`），再逐一要求 `('generated','reserve')`、`('rewarded','commit')`、`('consumed','reserve')` 抛 `ValueError`。

### 第 3 步：TODO 2 `assess_policy_compatibility`——版本与偏离双门槛
函数签名很长，先把各参数对上：

```python
assess_policy_compatibility(old_log_probs, current_log_probs, action_mask,
                            behavior_version, current_version,
                            max_lag, max_mean_abs_log_ratio)
```

- `old_log_probs`：行为策略在生成时对每个动作记录的对数概率。
- `current_log_probs`：当前策略在**同一批动作位置**重新算出的对数概率。
- `action_mask`：布尔列表，`True` 表示该位置是有效动作，参与比较。
- `behavior_version` / `current_version`：生成时版本与当前版本。
- `max_lag`、`max_mean_abs_log_ratio`：允许的最大版本滞后与最大平均偏离。

骨架已经做了输入校验：版本必须是非负整数且不能是 `bool`；阈值必须非负有限；三个序列必须等长；掩码必须全是布尔值；掩码全 `False` 时直接返回 `{'accepted': False, 'reason': 'empty_action_mask', 'freshness': None, 'mean_abs_log_ratio': None}`。TODO 补的是核心计算：

```python
lag = current_version - behavior_version
freshness = 'future' if lag < 0 else ('fresh' if lag == 0 else ('tolerated' if lag <= max_lag else 'stale'))
active_values = [float(values[i]) for values in (old_log_probs, current_log_probs) for i in active_indices]
if any(not math.isfinite(value) for value in active_values):
    raise ValueError('有效动作位置的对数概率必须为有限数')
mean_abs_log_ratio = sum(abs(float(current_log_probs[i]) - float(old_log_probs[i]))
                         for i in active_indices) / len(active_indices)
```

逐项解释：

1. **`lag = current - behavior`**：当前版本更大，`lag` 为正，说明轨迹旧了；相等为 0，最新；为负说明轨迹来自"未来版本"（版本登记错乱或回滚场景）。
2. **`freshness` 四分类**：`lag < 0 → future`；`lag == 0 → fresh`；`0 < lag <= max_lag → tolerated`；`lag > max_lag → stale`。这对应 Step 2 的表格：当前版本记 `fresh`、允许的旧版本记 `tolerated`、超过上限记 `stale`、未来版本记 `future`。
3. **`active_indices` 只取掩码为真的位置**：后面求和除以 `len(active_indices)`，不是总长度。这就是掩码的作用——padding、无效动作、工具调用占位符不参与偏离统计。
4. **有限性检查**：`old_log_probs` / `current_log_probs` 在有效位置必须是有限数。NaN 会让 `sum` 变成 NaN，后续 `NaN > threshold` 恒为 `False`，坏数据就被"静默接受"了；所以在计算均值前显式抛 `ValueError`。
5. **`mean_abs_log_ratio` 实际是"平均绝对对数概率差"**：公式是 $\frac{1}{|\text{active}|}\sum |\log \pi_{current} - \log \pi_{behavior}|$，不是"比值的对数"。变量名沿用题目接口，读的时候以公式为准。

对应 Step 2 的数学定义：

$$
\Delta v=v_{current}-v_{behavior},\qquad
D_{beh}=\frac{\sum_t m_t\left|\log\pi_{current}(a_t\mid s_t)-\log\pi_{behavior}(a_t\mid s_t)\right|}{\sum_t m_t}
$$

其中 $s_t$ 是位置 $t$ 的上下文，$a_t$ 是动作，$m_t\in\{0,1\}$ 是掩码，$\pi_{current}$、$\pi_{behavior}$ 分别是当前与行为策略。$m_t$ 为 0 的项分子分母都不计。

算完偏离后，参考实现按固定优先级返回：

```python
if freshness in {'future', 'stale'}:
    return {'accepted': False, 'reason': f'{freshness}_policy', ...}
if mean_abs_log_ratio > max_mean_abs_log_ratio:
    return {'accepted': False, 'reason': 'policy_drift', ...}
return {'accepted': True, 'reason': 'ok', ...}
```

也就是说，接受条件是 Step 2 的合取：

$$
0\le\Delta v\le L_{max}\quad\land\quad D_{beh}\le D_{max}
$$

**为什么不能只看版本？** 版本差是离散的顺序，两个都滞后 1 版的轨迹，行为可能天差地别。notebook 举的例子是：两条轨迹都滞后一个版本，动作分数平均变化可能分别是 0.01 和 0.6，阈值 0.05 时只接受前者。Step 5 的真实结果更直接：8 条旧轨迹版本条件相同，仍然 7 条准入、1 条拒绝，被拒那条平均偏离 3.60039。

### 第 4 步：测试里的数值怎么读
`test_policy_compatibility` 用固定数据把每条路径都走了一遍：

```python
old     = [-0.2, -0.2, -9.0]
current = [-0.21, -0.19, 9.0]
mask    = [True, True, False]
```

- `fresh`：`behavior_version=3, current_version=3`，`lag=0`。有效位置只有前两个，差值分别是 $|-0.21-(-0.2)|=0.01$、$|-0.19-(-0.2)|=0.01$，均值 `0.01`，小于阈值 `0.05`，`accepted=True`、`freshness='fresh'`。第三个位置从 −9 变成 9，差 18，但掩码是 `False`，完全不影响结果——这就是测试想证明的"掩码外占位值应忽略"。
- `tolerated`：`behavior=2, current=3, max_lag=1`，`lag=1 <= 1`，偏离仍 0.01，`accepted=True`。
- `stale`：`behavior=1, current=3`，`lag=2 > 1`，返回 `reason='stale_policy'`。
- `future`：`behavior=4, current=3`，`lag=-1`，返回 `reason='future_policy'`。
- `drift`：`current=[-0.8,-0.7,9.0]`，有效位置差值 $0.6$、$0.5$，均值 $0.55>0.05$，返回 `reason='policy_drift'`。
- `empty`：掩码全 `False`，返回 `reason='empty_action_mask'`，且 `freshness`、`mean_abs_log_ratio` 都是 `None`。
- `NaN`：有效位置放 `float('nan')`，函数必须抛 `ValueError`。

注意 `stale`/`future` 的返回里仍然带着 `mean_abs_log_ratio`，因为偏离已经算出来了；拒绝原因优先按版本分类，而不是先报偏离。

### 第 5 步：TODO 3 `compare_and_reserve`——修订号就是乐观锁
函数前面已经处理了三种情况：`update_id` 为空抛 `ValueError`；轨迹不存在返回 `missing_rollout`；**同一更新**已经预留过，返回 `idempotent_retry`（在检查修订号之前）。TODO 补的是首次预留：

```python
revision_matches = current.get('revision') == expected_revision
reservable = current.get('lifecycle_state') == 'rewarded'
reserved_record = current | {
    'lifecycle_state': transition_lifecycle(current.get('lifecycle_state'), 'reserve'),
    'reserved_by': update_id,
    'revision': current.get('revision', 0) + 1,
} if revision_matches and reservable else dict(current)
```

三个判断：

1. **`revision_matches`**：调用方读取记录时看到的修订号必须等于当前修订号。如果别人已经改过，修订号变了，你就不能基于旧快照写入——这就是乐观并发控制（compare-and-swap）。
2. **`reservable`**：只有 `rewarded` 状态能首次预留。还在 `generated`（评分未完成）、`reserved`（别人占着）、`consumed`、`quarantined` 都不行。
3. **`revision + 1`**：成功预留必须递增修订号，让其他持有旧快照的调用方失败。

随后按优先级返回：

```python
if not revision_matches:  return {'reserved': False, 'reason': 'revision_conflict', ...}
if not reservable:        return {'reserved': False, 'reason': 'not_rewarded', ...}
ledger[rollout_id] = reserved_record
return {'reserved': True, 'reason': 'reserved', 'record': dict(reserved_record)}
```

为什么"修订号冲突"排在"状态不对"前面？如果调用方拿的是过期快照，那么它看到的 `rewarded` 可能早就不是最新状态了；先报冲突能提示它重新读取，而不是误以为"状态本来就不能预留"。

用 `test_compare_and_reserve` 走一遍：

- 初始 `{'lifecycle_state': 'rewarded', 'revision': 7, 'reserved_by': None}`。
- `update-a` 带 `expected_revision=7` 预留成功，`revision` 变成 8，`reserved_by='update-a'`。
- `update-b` 也带 7，但当前已是 8，返回 `revision_conflict`，且账本里的 `reserved_by` 仍是 `update-a`（失败的写入不能污染记录）。
- `update-a` 再带 7 重试，命中函数开头的 `idempotent_retry` 分支，返回成功但修订号**不再增加**（仍是 8）。这保证重试不会把锁越推越远。
- 不存在的 `missing` 返回 `missing_rollout`；空 `update_id` 抛 `ValueError`；`consumed` 状态带正确修订号 2 仍返回 `not_rewarded`。

**一个必须说清的边界**：函数 docstring 写的是「在串行调用中按修订号判断预留；调用方须保证检查和写入互斥」。测试里的两个更新是顺序调用的，所以能稳定复现；真实多进程并发时，"读当前记录 → 比较 → 写回"之间仍可能被插队，需要锁、数据库事务或真正的 CAS 保证原子性。notebook 在 Step 1 和解析里都强调了这一点。

### 第 6 步：评分未完成与提交不明的两条特殊路径
`test_score_retry_before_reservation` 模拟一条 `reward_status='timeout'`、`score=None` 的轨迹：

1. 直接预留 → `not_rewarded`（还在 `generated`）。
2. 调用方确认评分成功后，把记录改成 `reward_status='completed'`、`score=0.0`、`score_attempt=2`，并执行 `reward_ready`，修订号从 0 加到 1。
3. 再用 `expected_revision=1` 预留成功，修订号变 2。

这个用例强调：**评分超时先保留，补评后仍可进入更新；已确认答错的 0 分是"评分完成"，不是"缺失值"**。测试还断言 `sampling_event_id` 仍是 `'sample-1'`、`behavior_version` 仍是 3，说明补评分不能换掉原始采样事件——否则就变成"另一次生成"了。

`test_uncertain_commit_is_quarantined` 走另一条路：`reserved + commit_uncertain → quarantined`。此后 `compare_and_reserve` 返回 `reserved=False`（状态不是 `rewarded`），而 `transition_lifecycle('quarantined', 'abort')` 必须抛 `ValueError`。为什么这么严？因为"提交结果不明"意味着更新可能已经执行了；如果允许 `abort` 释放，另一个更新可能把同一条轨迹再消费一次，造成重复训练。正确做法是等检查点与提交记录核对后再人工/外部流程处理。

### 第 7 步：TODO 4 `activate_policy_version`——四道门同时打开才切换
```python
missing_workers = sorted(worker for worker in required_workers if worker not in worker_states)
mismatched_workers = sorted(
    worker for worker in required_workers if worker in worker_states
    and (worker_states[worker].get('version') != artifact_version
         or worker_states[worker].get('artifact_hash') != artifact_hash)
)
monotonic_version = artifact_version > active_version
activated = (bool(artifact.get('committed', False))
             and monotonic_version
             and not missing_workers
             and not mismatched_workers)
```

四个条件：

| 条件 | 变量 | 不满足时的 `reason` |
|---|---|---|
| 产物已提交 | `artifact['committed']` 为真 | `artifact_not_committed` |
| 版本严格递增 | `artifact_version > active_version` | `non_monotonic_version` |
| 必需进程都已确认 | `missing_workers` 为空 | `missing_worker_ack` |
| 确认的版本与哈希都一致 | `mismatched_workers` 为空 | `worker_mismatch` |
| 全部满足 | `activated=True` | `activated` |

注意 `reason` 的判断顺序：先看提交，再看版本，再看缺失，最后看哈希不匹配。这个顺序只影响"同时犯多个错时报哪个"，四个条件本身都必须满足。

`missing_workers` 只检查 `required_workers` 里的进程；不在必需列表里的进程即使没确认也不影响。`mismatched_workers` 同时比较 `version` 和 `artifact_hash`，版本对但哈希错也算失败——这正是 Step 3 强调的"确认加载相同版本和哈希"。

返回的指针：

```python
next_pointer = {'version': artifact_version, 'artifact_hash': artifact_hash} if activated else dict(active_pointer)
```

失败时用 `dict(active_pointer)` 复制原指针，而不是直接返回同一个对象，避免调用方修改返回值时把全局活动指针也改掉。

`test_version_activation` 逐项拆门：

- 正常：active v3 / `policy-3`，artifact v4 / `policy-4` / committed，两个 worker 都报 v4 / `policy-4` → `activated=True`，指针版本 4。
- 只给一个 worker → `missing_worker_ack`，指针仍是 `{'version': 3, 'artifact_hash': 'policy-3'}`。
- 第二个 worker 哈希是 `'other'` → `worker_mismatch`，指针不变。
- `committed=False` → `artifact_not_committed`。
- artifact 版本 2 < active 3 → `non_monotonic_version`。
- 最后断言 `active` 字典本身没有被修改。

### 第 8 步：Step 5 的真实单卡实验怎么设计
这一部分不是纯 CPU 练习，而是一套可在 `PT-VERL-CUDA` 上运行的机制验证。设计如下。

**固定项（控制变量）**：

| 项目 | 固定内容 |
|---|---|
| 模型 | `Qwen2.5-0.5B-Instruct`，LoRA rank 16，只作用于 q/v 投影 |
| 更新 | verl 内置监督损失，只做一次优化器更新，不跑 GRPO/PPO |
| 轨迹 | 读取已有 `rtx5070ti_rollout_arithmetic_v1` 中每题第一条回复，共 8 条；选择不看奖励，保留 EOS 和全部动作 Token |
| 更新输入 | 另用 4 条固定算术监督样本；**不使用**这 8 条轨迹训练 |
| 初始参数 | 创建模型前固定种子 42；释放组与保留组训练前的初始 LoRA 哈希必须一致 |
| 运行纪律 | 不自动重试、不追加训练、不放宽阈值；已有文件不覆盖 |

**自变量（两组）**：释放组开启引擎释放/恢复，保留组仅关闭引擎释放/恢复。两组用相同固定轨迹和相同监督输入，比较的是"释放/恢复路径"是否影响更新与同步。

**因变量（保存的证据）**：初始化种子与参数指纹、参数差值范数、loss、梯度、更新前后同 Token 概率、同步后版本与新生成记录、逐轨迹准入结果、阶段资源。

**baseline / 对照**：
- `source_vs_actor_before`：更新前源模型与 actor 的逐 Token 对数概率对比，确认起点一致。
- `same_version_before`：同一版本下两次测量的误差，作为"测量噪声底"。
- `same_version_after`：更新并同步后，同版本概率误差，检查同步是否准确。
- `actor_change` 与 `rollout_change`：actor 侧和生成侧的参数变化幅度，二者应接近；`delta_alignment.rmse` 衡量两条变化路径的一致性。

**结果怎么读？** `RUN_MODE` 默认 `dry_run`（只预检），改成 `real_gpu` 才真实执行。notebook 保存的这次运行标签是 `rtx5070ti_verl_mechanism_closure_v2`，预检显示 `ready: true`：torch 2.11.0+cu130、vllm 0.24.0、ray 2.55.1、主机内存 30.525 GiB、可用 22.071 GiB、GPU 为 RTX 5070 Ti Laptop。5.3 读取的 `decision.checks` 六项全为 `true`：`parameters_changed`、`same_version_probabilities`、`frozen_scores_changed`、`synchronized_change_visible`、`sync_rpc_and_version`、`real_generation_records`。

关键数字（来自保存的输出和 5.4 的汇总表）：

| 观察项 | 数值 | 怎么解释 |
|---|---|---|
| 同版本概率误差（更新后） | 最大 0.06250，平均 0.002653 | 低于原定上限 0.15 和 0.03，同步后生成侧与 actor 基本一致 |
| 更新前同版本误差 | 最大 0.02539，平均 0.001104 | 作为测量噪声底，更新后误差略大但仍达标 |
| actor 变化 | 平均 0.44891，最大 10.7688 | 24 个 Token 上参数更新带来的对数概率变化 |
| rollout 变化 | 平均 0.44740，最大 10.7317 | 与 actor 变化接近，`delta_alignment.rmse=0.0075766` |
| 参数更新 | 梯度范数 0.03428，`delta_l2=0.61536`，48 个张量改变 | 4 条监督样本、12 个监督 Token、一次真实更新 |
| 参数哈希 | 初始均 `06380c94…`，更新后均 `369392e9…` | 两组起点和一次更新结果一致，可比较释放/保留路径 |
| 旧轨迹准入 | 8 条中 7 准入、1 拒绝；拒绝项平均偏离 3.60039 | 阈值 0.2，按实际概率偏离判断，不只比版本号 |
| 真实生成 | `81−47` 从 24 变成 34 | 新策略确实在生成侧生效 |

**结论边界**：notebook 明确说「是否提高奖励或触发拒绝不作为本节通过条件」「不据单题变化报告整体质量提升」。也就是说，这次实验验证的是**机制**：参数确实变了、冻结轨迹重评确实变了、同步后的生成侧确实用上了新版本、旧轨迹按偏离阈值被分别准入/拒绝。它没有验证多工作进程的原子激活，也没有验证奖励收益；v1 还保留着初始参数不一致和概率误差超限的记录，说明同一套流程也可能失败。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 轨迹（rollout） | 一次生成得到的回答或交互过程，含动作与概率记录 |
| 行为策略 | 生成这条轨迹时使用的策略 |
| 版本滞后 $\Delta v$ | 当前版本减生成时版本，只描述新旧顺序 |
| $D_{beh}$ | 有效动作位置上，当前与行为策略对数概率差的绝对值均值 |
| 动作掩码 | 布尔列表，只有 `True` 位置参与偏离统计 |
| 预留（reserve） | 把轨迹绑定到某个更新，防止被别人同时使用 |
| 修订号（revision） | 记录的版本号，成功写入后加一，用于乐观并发控制 |
| 幂等重试 | 同一更新对同一轨迹再次预留，直接返回成功但不重复写 |
| `quarantined` | 提交结果不明时的隔离态，暂停重新预留 |
| 活动指针 | "接下来应使用哪个策略"的记录 |
| 产物哈希 | 权重文件的内容指纹，用于确认各进程加载的是同一份 |
| 新鲜度（freshness） | `future` / `fresh` / `tolerated` / `stale` 四分类 |

## 小白常见疑问

**Q1：为什么版本差为负要单独标成 `future`，而不是直接当 `fresh`？**
`lag = current_version - behavior_version` 为负，说明轨迹来自比当前更高的版本，这在正常流程里不应该出现，通常意味着版本登记错乱、回滚或读到了别的实验。`future` 会被直接拒绝（`reason='future_policy'`），而 `fresh` 还要继续检查动作偏离。测试用 `behavior_version=4, current_version=3` 验证 `future_policy`，就是为了让这种异常显式暴露。

**Q2：`mean_abs_log_ratio` 名字里有 ratio，为什么代码算的是差值？**
接口名沿用题目，实际公式是 $D_{beh}=\frac{1}{|\text{active}|}\sum|\log\pi_{current}-\log\pi_{behavior}|$，即平均绝对对数概率差。测试里 `old=[-0.2,-0.2]`、`current=[-0.21,-0.19]`，两个差值都是 0.01，期望 `mean_abs_log_ratio=0.01`；如果真去算"比值的对数"，会得到 $\ln(0.21/0.2)\approx0.0488$ 和 $\ln(0.19/0.2)\approx-0.0513$，均值接近 0，对不上 0.01。读代码时以公式为准，不要被名字带偏。

**Q3：掩码第三个位置从 −9 变成 9，差 18，为什么 `fresh` 还接受？**
因为 `mask = [True, True, False]`，`active_indices` 只有 `[0,1]`，第三个位置不进入分子也不进入分母。这个位置可能是 padding、无效动作或占位符，它的对数概率没有意义，放多大都不该影响判断。测试正是用这个巨大差值来证明掩码真的生效；如果实现忘了过滤，`mean_abs_log_ratio` 会变成约 6.006，直接触发 `policy_drift`。

**Q4：为什么 `commit_uncertain` 不能走 `abort` 退回 `rewarded`？**
因为 `commit_uncertain` 的含义是"更新可能已经执行，但提交结果不明"。如果允许 `abort`，另一个更新就能重新预留并消费同一条轨迹，导致同一份数据被训练两次。`quarantined` 状态不在 `LIFECYCLE_TRANSITIONS` 的任何起点里，所以 `('quarantined','abort')` 会抛 `ValueError`；`compare_and_reserve` 也会因为状态不是 `rewarded` 而拒绝。正确做法是核对检查点与提交记录后再决定。

**Q5：Step 5 里 7 条准入、1 条拒绝，能说明策略更新成功了吗？**
只能说明"按当前偏离阈值，8 条旧轨迹里 7 条仍可用、1 条偏离过大"。它验证的是准入机制在工作，不能推出"新策略更好"或"旧轨迹一定该拒"。notebook 明确写「是否提高奖励或触发拒绝不作为本节通过条件」「不据单题变化报告整体质量提升」；奖励收益和多进程切换需要另外的实验。

## 动手验证：怎么知道自己做对了
测试区在 cell 10，先跑两个专项测试 `test_score_retry_before_reservation()`、`test_uncertain_commit_is_quarantined()`，再跑四个主测试，最后打印 `✅ 轨迹状态迁移、策略兼容性、竞争预留与版本激活测试通过`。TODO 未填时先抛 `NotImplementedError`。

- `test_lifecycle_transitions`：四条合法边必须返回 `rewarded` / `reserved` / `consumed` / `rewarded`；`('generated','reserve')`、`('rewarded','commit')`、`('consumed','reserve')` 必须抛 `ValueError`。
- `test_policy_compatibility`：`fresh` 的 `mean_abs_log_ratio` 与 0.01 的差小于 `1e-8`；`tolerated` 接受；`stale_policy`、`future_policy`、`policy_drift`、`empty_action_mask` 四种 reason 分别命中；有效位置含 `NaN` 必须抛 `ValueError`。
- `test_compare_and_reserve`：首次成功且 `revision` 从 7 变 8；竞争更新返回 `revision_conflict` 且 `reserved_by` 仍是 `update-a`；重试返回 `idempotent_retry`；缺失轨迹返回 `missing_rollout`；空 `update_id` 抛 `ValueError`；`consumed` 返回 `not_rewarded`。
- `test_version_activation`：正常激活后指针版本为 4；缺确认、哈希不匹配、未提交、版本倒退分别返回 `missing_worker_ack`、`worker_mismatch`、`artifact_not_committed`、`non_monotonic_version`，且每次 `active_pointer == active`，最后 `active` 仍是 `{'version': 3, 'artifact_hash': 'policy-3'}`。
- `test_score_retry_before_reservation`：超时轨迹先被拒（`not_rewarded`）；补评分并 `reward_ready` 后修订号从 0 到 1，再预留成功到 2；`sampling_event_id` 和 `behavior_version` 不变。
- `test_uncertain_commit_is_quarantined`：`reserved + commit_uncertain → quarantined`；隔离后再预留失败；`quarantined + abort` 抛 `ValueError`。

GPU 部分保存了真实输出，可以对照：5.1 打印的是 `RUN_MODE`、`RUN_LABEL` 和 `RESULT_PATH`；5.2 会读取已有结果或调用 `tools.verl_mechanism_closure`；5.3 打印 `read_summary(..., "lifecycle")`。本次保存的输出显示 `decision.passed=true`，六项 `checks` 全真，同版本更新后最大误差 0.06250、平均 0.002653，8 条轨迹 7 准入 1 拒绝。注意这些数字来自已有的 `real_gpu` 运行记录，不是每个读者重新跑出来的；如果你用 `dry_run`，只会得到预检结果，不会产生性能或质量结论。
