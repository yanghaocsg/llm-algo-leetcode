# 51. Online DPO Beginner | 在线 DPO · 小白入门

> 配套 notebook：[51. 在线 DPO](./51_Online_DPO.md) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

在线 DPO 一边收集“哪个回答更好”的反馈，一边一轮一轮地更新模型。这本 notebook 规定每轮用哪些反馈、怎样让每轮的样本清单可以复现，以及更新后何时采用、何时退回旧模型。

## 为什么需要它

直接偏好优化（DPO）用的是偏好对：同一个任务的两个回答，加上“哪个更好”的标注。离线做法是准备一份数据，训练一次。在线做法是反馈不断到来，模型一轮一轮地更新。

最直接的想法是“来一条反馈就更新一次”，但这样会出几种错：

- 同一条反馈被两轮更新各用一次，或者在同一轮里到达了两次。
- 反馈里的回答是很旧的模型版本生成的，和当前版本差得太远。
- 反馈来自评测集，训练数据和评测数据混在了一起。
- 更新出来的模型其实变差了，却没人比较过，下一轮接着在它上面更新。

这本 notebook 把一轮更新固定成三步：先判断每条反馈能不能用；把通过的反馈排好序、截取固定的数量，组成一个窗口；更新后和固定的基线比较，再决定采用、延迟、回滚还是停止。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 偏好对（preference pair） | 同一个任务的两个回答，加上“哪个更好”的标注。好的叫 `chosen`，差的叫 `rejected` |
| 基线 / 候选（baseline / candidate） | 基线是当前在用的模型版本，候选是这一轮更新出来的新模型 |
| 反馈窗口 | 一轮更新要用的那批偏好对，数量和顺序都是固定的 |
| 窗口 ID | 由策略版本和入选的偏好对 ID 算出的一串指纹，用来标识一个窗口 |
| 独立评测 | 用没参与训练的评测集给候选打分，防止训练反馈代替最终评测 |
| 非平局胜率（win rate） | 候选和基线比较时分出胜负的那些里，候选赢的比例 |
| 检查点（checkpoint） | 保存下来的一份模型权重，出问题时可以加载回去 |

## 核心思路

一轮在线 DPO 分三步：

```text
准入   每条偏好对按固定顺序检查，命中第一个原因就拒绝：
       身份不全 → 两个回答相同 → 不在训练切分 → 有风险 → 版本比当前新（future）→ 版本太旧（stale）
版本差 delta_v = v_current - v_generated，要求 0 <= delta_v <= L_max
窗口   通过准入的偏好对按 ID 排序，取前 M 条；多出来的只延后到下一轮
       窗口 ID = H(v_t, 入选的 ID 列表)：策略版本和 ID 顺序相同，窗口 ID 就相同
评测   dW = W_candidate - W_baseline（非平局胜率），dR、dS 同理
       rho_C = (C_candidate - C_baseline) / C_baseline（更新成本的增幅）
决策   安全指标降幅超限、一票否决测试失败、重复评测通过率降幅超限 → rollback 或 halt
       成本增幅超限 → defer；非平局胜率增幅达到门槛 → adopt；其余 → defer
```

`v_current` 是当前要更新的策略版本，`v_generated` 是生成这条反馈的策略版本，`L_max` 是允许的最大版本差。`M` 是窗口容量，`v_t` 是第 `t` 轮的策略版本，`H` 是稳定的摘要函数。`W`、`R`、`S`、`C` 分别是非平局胜率、重复评测通过率、安全指标和更新成本。一票否决测试是预先登记的安全或关键能力测试，失败一次就不能采用。

下面用测试里的反馈流走一遍窗口的形成。当前策略版本是 10，窗口容量是 2，`used` 在以前的窗口里用过：

```python
import hashlib, json

stream = ['b', 'a', 'c', 'b', 'used', 'stale', 'unsafe']   # 反馈到达的顺序
consumed, seen = {'used'}, set()                           # 以前的窗口用过的 ID
rejected, accepted = {}, []
for pair_id in stream:
    if pair_id in consumed:
        reason = 'consumed'                                # 以前的窗口用过
    elif pair_id in seen:
        reason = 'duplicate'                               # 这一轮里重复到达
    else:
        reason = {'stale': 'stale', 'unsafe': 'risk'}.get(pair_id, 'eligible')
    seen.add(pair_id)
    if reason == 'eligible':
        accepted.append(pair_id)
    else:
        rejected[reason] = rejected.get(reason, 0) + 1
accepted.sort()                                            # 按 ID 排序再截取容量
selected, overflow = accepted[:2], accepted[2:]
payload = json.dumps([10, selected], separators=(',', ':'))
window_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
print(selected, overflow, rejected)
print(window_id == hashlib.sha256(json.dumps([10, ['a', 'b']], separators=(',', ':')).encode()).hexdigest()[:16])
```

```text
['a', 'b'] ['c'] {'duplicate': 1, 'consumed': 1, 'stale': 1, 'risk': 1}
True
```

第一行：第二个 `b` 是这一轮里的重复，`used` 以前的窗口用过，`stale` 和 `unsafe` 没通过准入，各记一次拒绝。剩下的 a、b、c 按 ID 排序。容量是 2，所以 a 和 b 入选，c 延后。第二行：窗口 ID 只由策略版本 10 和入选的 ID 列表决定，同样的内容算出来的 ID 相同，所以反馈到达的顺序不影响窗口。

四个决策里，`rollback` 和 `halt` 都对应硬门槛被越过，区别是有没有可用的检查点：有就加载最近一个通过评测的检查点，没有就停止自动更新。`defer` 是保留当前基线并补充证据。`adopt` 要求非平局胜率增幅和成本增幅同时过线，候选才登记为下一轮的基线。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一条在线偏好如何进入本轮更新：成对构造、反馈准入、固定窗口更新、独立评测、采用与恢复 | 读 |
| Step 2 | 哪些偏好对能进入固定反馈窗口：六个判断维度，以及窗口 ID 怎样生成 | 读，对照上面的小例子 |
| Step 3 | 何时采用候选，何时延迟、回滚或停止：和固定基线比较的四个变化量 | 读 |
| Step 4 · TODO 1 | 判断单个偏好对能否进入在线更新 | 在 `classify_preference_pair` 里补 1 组判断：按“身份、成对语义、切分、风险、版本差”的顺序返回第一个命中的原因 |
| Step 4 · TODO 2 | 形成边界固定的在线偏好窗口 | 在 `build_preference_window` 里补 3 部分：`consumed_ids`、逐条的准入判断、`selected_pairs` 和 `overflow_pairs` |
| Step 4 · TODO 3 | 根据独立评测和检查点状态决定是否采用候选 | 在 `recommend_online_dpo` 里补 2 部分：`comparison` 字典和 `decision` 的判断链 |
| 测试 | 三个测试分别检查偏好对的准入原因、窗口身份与容量、采用或回滚决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 把准入后的窗口转成 TRL DPO 使用的公开数据字段，并实例化 CPU 上的 `DPOConfig` | 需要 `PT-CPU` 环境，可以先跳过 |

Step 1–4 只需要 CPU。框架核对需要另外的 `PT-CPU` 环境，可以先跳过。

## 容易踩的坑

- **一条样本返回了多个拒绝原因**：每条数据只保留一个首要拒绝原因，按固定顺序检查，拒绝计数才不会重复累计。生成策略版本只接受非负整数，小数和布尔值都按身份缺失处理。
- **同一偏好对被用了两次**：先排除历史窗口已经消费的 ID，再处理当前流里的重复。同一个 ID 的内容或风险状态冲突时整组拒绝，不能按到达顺序留第一条。
- **窗口随到达顺序变化**：通过的偏好对要先按 ID 排序再截取容量，容量外的只延后。窗口 ID 由策略版本和入选 ID 列表生成，两者相同，窗口 ID 就必须相同。
- **用胜率增幅抵消安全退化**：安全指标或重复评测通过率降幅超限、一票否决测试失败时，先回滚或停止，不再看胜率。没有可用检查点就停止自动更新。
- **成本或胜率没过线就采用**：成本增幅超限、胜率增幅不足都是延迟，保留当前基线并补充证据。胜率和成本都过线才能采用。

## 读完后自查

- [ ] 能说出偏好对准入的检查顺序，以及同一任务的两个回答完全相同会得到哪个原因。
- [ ] 通过准入的 ID 是 a、b、c、容量是 2 时，能说出谁入选、谁延后。
- [ ] 能说出 `rollback` 和 `halt` 的区别。
- [ ] 能说出候选要满足哪两个条件才会被采用。

## 下一步

- 打开 notebook：[51. 在线 DPO](./51_Online_DPO.md)
- 建议先读：[50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation_Beginner.md)、[17. 直接偏好优化损失 · 小白入门](./17_DPO_Loss_Tutorial_Beginner.md)、[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness_Beginner.md)
- 相关 notebook：[PT-RL-SYSTEM. 采样、训练与奖励系统](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)、[84. DPO 偏好优化项目](./84_DPO_Preference_Project.md)、[86. 在线后训练生命周期与恢复](./86_DPO_Online_Benchmark.md)
