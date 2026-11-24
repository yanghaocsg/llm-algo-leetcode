# PT-ROLLOUT-LIFECYCLE. Rollout and Policy Freshness | 轨迹与策略新鲜度 · 小白入门

> 配套 notebook：[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度](../PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

在线训练时模型一边生成数据、一边更新自己。这本 notebook 给每条生成记录记下“由哪个版本生成、处理到哪一步”，只放行还合用的数据，等所有进程确认后再切换到新模型。

## 为什么需要它

在线后训练里，模型生成一批回答，打分，拿去更新自己。更新完模型就变了，下一批数据由新模型生成。更新产出的新权重会保存成一个新的策略版本。负责更新参数的那一端，下面叫学习器。

麻烦在于，生成和更新不是同步的。一条轨迹由版本 1 生成，等它排到训练队列前面，学习器已经是版本 3。这条数据还能用吗？不加检查直接送进更新，会出现三种错误：

- 这条轨迹由哪个版本、用什么方式生成没记清楚，出了问题没法追溯，也没法去重。
- 两个学习器同时拿到同一条轨迹，它被更新了两次。
- 新权重只写到一部分进程，就宣布新版本已经激活，其实还有工作进程没加载新权重。

这本 notebook 用四道关卡挡住这些错误：轨迹带齐身份并沿固定顺序经历四个状态；版本号和行为偏离一起判断轨迹能不能用；一条轨迹只能绑定一次更新；新策略要等所有必需的进程确认后才激活。你要把后三道和状态迁移写成四个函数。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 轨迹（trajectory） | 模型一次采样（rollout）生成的完整记录，包括提示、写出的动作和当时给每个动作的概率 |
| 行为策略（behavior policy） | 生成这条轨迹的那个策略版本；“当前策略”则是学习器手里现在的版本 |
| 新鲜度（freshness） | 这条轨迹相对当前策略有多旧，分为 `fresh`、`tolerated`、`stale`、`future` |
| 行为偏离（behavior drift） | 当前策略和行为策略在这条轨迹的动作上，对数概率相差多少 |
| 动作掩码（action mask） | 标出哪些位置是模型真正做的动作，只有这些位置参与计算 |
| 预留（reserve） | 把一条轨迹独占地绑定到一次更新上，别的更新不能再拿它 |
| 修订号（revision） | 记录每被改一次就加 1 的计数器，用来发现记录已经被别人改过 |

## 核心思路

一条轨迹进入更新之前要过三道关，一个新策略版本生效之前还有第四道：

```text
状态   已生成 → 已奖励 → 已预留 → 已消费        （更新中止：已预留 → 已奖励）
版本   lag = current_version - behavior_version
       lag < 0 是 future，lag = 0 是 fresh，0 < lag <= max_lag 是 tolerated，lag > max_lag 是 stale
偏离   D_beh = 有效动作位置上 |log_now - log_old| 的平均值
准入   新鲜度是 fresh 或 tolerated，并且 D_beh <= max_mean_abs_log_ratio
预留   记录里的修订号 == 调用方看到的修订号，并且状态是 rewarded
       成功后状态改为 reserved，记下 reserved_by，修订号加 1
激活   产物已提交，版本严格更新，必需的工作进程都已确认，且版本和哈希一致
```

`current_version` 是学习器当前的策略版本，`behavior_version` 是生成轨迹的版本，`max_lag` 是允许的最大版本滞后。`max_mean_abs_log_ratio` 是允许的最大行为偏离，`log_now` 和 `log_old` 是当前策略和行为策略给同一个动作的对数概率。工作进程（worker）是各自加载策略去生成或训练的进程。哈希（hash）是由文件内容算出的一串指纹，内容变了它就变。

下面用测试里的数字试一下版本、偏离和预留：

```python
old = [-0.2, -0.2, -9.0]      # 生成轨迹时，行为策略给 3 个位置的对数概率
now = [-0.21, -0.19, 9.0]     # 当前策略给同样 3 个位置的对数概率
mask = [True, True, False]    # 只有前两个位置是有效动作
active = [i for i, m in enumerate(mask) if m]
print(round(sum(abs(now[i] - old[i]) for i in active) / len(active), 4))

def freshness(current, behavior, max_lag):
    lag = current - behavior
    if lag < 0:
        return 'future'
    if lag == 0:
        return 'fresh'
    return 'tolerated' if lag <= max_lag else 'stale'

print([freshness(3, v, 1) for v in (3, 2, 1, 4)])

ledger = {'r-1': {'lifecycle_state': 'rewarded', 'revision': 7, 'reserved_by': None}}
record = ledger['r-1']
print(record['revision'] == 7 and record['lifecycle_state'] == 'rewarded')
ledger['r-1'] = record | {'lifecycle_state': 'reserved', 'reserved_by': 'update-a', 'revision': 8}
record = ledger['r-1']
print(record['revision'] == 7 and record['lifecycle_state'] == 'rewarded', record['revision'])
```

```text
0.01
['fresh', 'tolerated', 'stale', 'future']
True
False 8
```

第一行是行为偏离。第三个位置的两个值相差很大，但它不在动作掩码里，不参与计算，所以平均值只有 0.01。第二行是新鲜度：学习器在版本 3，最多容忍落后 1 个版本。版本 3 的轨迹是 `fresh`，版本 2 的是 `tolerated`，版本 1 的是 `stale`。版本 4 的轨迹比学习器还新，属于版本登记错误，记为 `future`，要立即拒绝。

后两行是预留。`update-a` 读到修订号 7，状态是已奖励，预留成功，修订号变成 8。`update-b` 手里还是旧的 7，再核对就对不上，只能失败。

版本号只说明新旧关系，说明不了两个版本在这些动作上的实际差别，所以两道门槛要一起看。notebook 也提醒，单独哪一项都不能当成“轨迹一定符合算法假设”的证明。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一条轨迹需要记录哪些身份和处理状态：五类信息，以及四个状态之间的合法迁移 | 读 |
| Step 2 | 版本滞后与行为偏离如何决定轨迹准入：四种新鲜度和两道门槛 | 读，对照上面的小例子 |
| Step 3 | 哪些产物与工作进程确认完成后才能切换策略 | 读 |
| Step 4 · TODO 1 | 只接受 `LIFECYCLE_TRANSITIONS` 明确列出的状态—事件组合 | 在 `transition_lifecycle` 里补 2 行：查表得到下一状态，查不到就抛 `ValueError` |
| Step 4 · TODO 2 | 计算版本状态和动作位置平均绝对对数概率差 | 在 `assess_policy_compatibility` 里补 3 行：`lag`、`freshness` 和 `mean_abs_log_ratio` |
| Step 4 · TODO 3 | 核对修订号与可预留状态，并构造下一版记录 | 在 `compare_and_reserve` 里补 3 行：`revision_matches`、`reservable` 和 `reserved_record` |
| Step 4 · TODO 4 | 检查必需工作进程确认，并形成完整激活门槛 | 在 `activate_policy_version` 里补 4 行：`missing_workers`、`mismatched_workers`、`monotonic_version` 和 `activated` |
| 测试 | 检查合法与非法的状态迁移、版本与行为偏离两道门槛、竞争预留和幂等重试、激活失败时保留旧指针 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 verl 里做一次受控更新，重评同样的旧轨迹，观察行为偏离和权重同步 | 需要 GPU 和 `PT-VERL-CUDA` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `PT-VERL-CUDA` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **跳过阶段或重用终态**：只有 `LIFECYCLE_TRANSITIONS` 列出的状态和事件组合才合法。不能从已生成直接预留，已消费的轨迹不能再进更新；更新中止只能从已预留回到已奖励。
- **只比较版本号就放行**：版本差合格不代表行为仍兼容。`future` 和 `stale` 都要拒绝，行为偏离超过阈值也要拒绝，动作掩码全为假时同样拒绝。
- **在无效位置上算偏离**：只平均 `active_indices` 指向的位置。测试里被掩码排除的位置放了相差很大的值，算进去结果就错了。
- **预留时没核对修订号**：只有修订号一致且状态是已奖励才能首次预留。两个更新拿着同一个旧修订号申请，只能一个成功；同一个更新重试不能再把修订号加一。题目用内存字典按顺序调用模拟，测试不证明并发安全。
- **部分确认就宣布新版本激活**：产物未提交、有必需进程没确认、版本或哈希不一致、版本没有变大，这几种情况都要保持原来的活动策略。

## 读完后自查

- [ ] 能按顺序说出一条轨迹正常经过的四个状态，以及更新中止时回到哪个状态。
- [ ] 当前版本是 3、最大版本滞后是 1 时，能判断版本 1、2、3、4 生成的轨迹各属于哪种新鲜度。
- [ ] 给两组对数概率和一个动作掩码，能算出平均绝对对数概率差。
- [ ] 能说出新策略产物要满足哪些条件才算激活。

## 下一步

- 打开 notebook：[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度](../PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
- 建议先读：[PT-REWARD-SIGNAL. 奖励、价值与优势信号 · 小白入门](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)、[50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation.md)、[17. 直接偏好优化损失 · 小白入门](./17_DPO_Loss_Tutorial.md)
- 做完之后：[PT-AGENTIC-RL. 环境轨迹与奖励归因 · 小白入门](./PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md)
- 相关专题页：[PT-RL-SYSTEM. 采样、训练与奖励系统](../PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)、[51. 在线 DPO](../51_Online_DPO.md)
