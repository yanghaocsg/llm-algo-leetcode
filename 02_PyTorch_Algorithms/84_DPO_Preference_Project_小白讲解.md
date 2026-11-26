# 84. DPO 偏好优化项目 — 小白讲解

> 对应 notebook：[`84_DPO_Preference_Project.ipynb`](./84_DPO_Preference_Project.ipynb)

## 一句话总结
本节把"用偏好对继续更新 SFT 模型"做成一条可验收的项目链：先检查训练/评测的偏好对与问题是否隔离，再用同一批留出回答比较更新前后的偏好准确率、平均边际和最差边际，最后结合截断率、显存和适配器哈希给出 `accept` / `tune` / `reject`。核心工程结论是：**损失下降不等于偏好排序变好；验收要看留出集上的配对差值，并且平均边际上升不能掩盖最差边际下降**。

## 为什么需要它：痛点是什么
SFT 之后，模型可能"答得像"，但同一问题的两个回答谁更好，SFT 并没有直接教。偏好数据补上了这一点：每条样本给出共享问题和一优一劣两条回答。初学者常见两个误判：一是只看训练 loss 曲线下降就宣布成功；二是把训练用过的偏好对再拿来评测，分数当然好看，却说明不了泛化。

生活类比：教练看训练赛和正式比赛。训练 loss 是训练赛比分，留出偏好对是正式比赛；把训练赛对手再请来打一遍（数据泄漏），赢了也不算数。另一个类比是评分表：平均分提高 0.002 看着是进步，但如果有一个人从 −0.78 掉到 −0.79，这个"最差学生"的退步也要单独记录，不能被平均分抹平。

## 核心问题
- DPO 的两个模型分别是谁？为什么参考模型要冻结？
- 本项目的"回答分数"为什么是词元平均对数概率，而不是序列求和？这会改变什么解释？
- 偏好准确率、平均边际、最小边际变化分别怎么算，为什么最小边际可能来自不同样本？
- 数据隔离、执行状态、离线指标、截断、显存、产物记录，哪几项失败该 `reject`，哪几项该 `tune`？
- 真实项目里 `MAX_LENGTH`、`BETA`、`UPDATE_STEPS` 和门槛怎么固定，登记结果说明什么、不说明什么？

## 解决思路：一步步看懂

**第 1 步：DPO 在做什么。** 设问题为 $x$，优选回答为 $y_w$、非优选为 $y_l$，策略为 $\pi_\theta$，冻结参考为 $\pi_{\text{ref}}$。标准 DPO 损失是

$$
\mathcal{L}=-\log\sigma\!\left(\beta\left[\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}\right)-\left(\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]\right),
$$

其中 $\sigma$ 是 Sigmoid，$\beta$ 控制"偏离参考"的惩罚强度。括号里可以重排成"策略认为优选比非优选高多少"减去"参考认为高多少"：只有当策略相对参考更偏爱优选回答时，损失才会下降。参考模型不更新，否则两边一起漂移，比较就失去锚点。

本项目用的不是序列求和，而是**回答词元的平均对数概率**：

$$
s_\theta(x,y)=\frac{1}{N}\sum_{t=1}^{N}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

$N$ 是保留的回答词元数（含未被截掉的结束标记），$y_t$ 是第 $t$ 个目标词元。每个偏好对的边际是 $m=s_\theta(x,y_w)-s_\theta(x,y_l)$。用平均而不是求和，长短回答的分数尺度更接近，但解释时必须保留这个约定：notebook 明确说"与按回答序列求和的标准 DPO 目标不同"。

**第 2 步：TODO 1——两个层次的隔离。** `check_preference_partition` 检查四项：

```python
checks = {
    "pair_isolation": not pair_overlap,        # 完整偏好对不能同时出现在训练和评测
    "prompt_isolation": not prompt_overlap,    # 同一个问题不能两边都有
    "valid_ratio": audit["valid_ratio"] >= minimum_valid_ratio,
    "distinct_responses": audit["identical_response_count"] == 0,
}
admitted = all(checks.values())
```

只查偏好对 ID 不够：把同一问题的两条回答换掉一条，pair ID 变了，但问题仍然泄漏，所以还要查 prompt ID。`valid_ratio` 约束"能解析出共享问题 + 两条非空回答"的比例，门槛是 0.95；`identical_response_count` 统计优选和非优选文本完全相同的情况，这种样本没有偏好信息。测试 `test_preference_partition_contract` 分别制造 pair 重合、prompt 重合、`identical_response_count=1`，三次都要求 `admitted` 为假。

真实项目 5.3 用 `stable_text_hash` 给"问题+两条回答"和"仅问题"分别做哈希，再传给同一个检查逻辑；数据集是 `trl-lib/ultrafeedback_binarized`，固定 commit 后从 `train` 池 256 条里抽 24 对、从 `test` 池 128 条里抽 16 对，共 40 对。保存的输出显示 `valid_ratio=1.0`、`pair_overlap=[]`、`prompt_overlap=[]`、`duplicate_count=0`。

**第 3 步：TODO 2——先配对，再算三个指标。** `evaluate_dpo_candidate` 首先要求 `baseline_margins` 和 `candidate_margins` 都非空且长度相同，否则抛 `ValueError("更新前后必须使用同一组非空偏好对")`；列表顺序由调用方保证一一对应。`summarize` 对每组边际算：

- `preference_accuracy = sum(margin > 0) / len(margins)`：边际大于 0 的对数占比。注意 `0.0` 不算正，测试里的基线边际含一个 `0.00`，它贡献 0 分。
- `margin_mean = sum / len`：整体分数差。
- `margin_min = min`：最差的那一对。

然后三个变化量做差：`preference_accuracy_gain`、`margin_mean_gain`、`margin_floor_gain`。六项门槛全部合取：

```python
checks = {
    "execution": evidence["status"] == "ok",
    "accuracy_gain": comparison["preference_accuracy_gain"] >= gates["minimum_accuracy_gain"],
    "margin_gain": comparison["margin_mean_gain"] >= gates["minimum_margin_gain"],
    "margin_floor": comparison["margin_floor_gain"] >= gates["minimum_margin_floor_gain"],
    "truncation": evidence["evaluation"]["truncation_ratio"] <= gates["maximum_truncation_ratio"],
    "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
}
eligible = all(checks.values())
```

用 `_dpo_candidate_fixture` 手算：基线 `[-0.20, 0.00, 0.20, 0.40]`，准确率 2/4=0.5，均值 0.10，最小值 −0.20；候选 `[-0.19, 0.05, 0.25, 0.45]`，准确率 3/4=0.75，均值 0.14，最小值 −0.19。于是准确率 +0.25 ≥ 0.05、均值 +0.04 ≥ 0.02、最差 +0.01 ≥ 0.0，显存 8000 ≤ 10000、截断 0.01 ≤ 0.05，`eligible` 为真。

`test_dpo_candidate_gates` 的四个反例很值得读：显存 10 001 超限失败、恰好 10 000 通过（说明是 `<=`）；候选改成 `[-0.21, 0.05, 0.25, 0.47]` 后均值仍是 0.14、准确率仍是 0.75，但最差从 −0.20 掉到 −0.21，`margin_floor_gain=-0.01 < 0.0`，整体不可用——这就是"平均边际不能抵消最差边际"；截断 0.20 失败；候选边际只给 1 个而基线 4 个时抛 `ValueError`。另外，`test_dpo_execution_failure_rejected` 把 `status` 改成 `failed`，此时 `execution_ok` 为假，后面步骤必须拒绝而不是调参。

**第 4 步：TODO 3——三种状态的分界线。** `decide_dpo_delivery` 先看三件事：数据是否可比（`partition["admitted"]`）、执行是否成功（`candidate.get("execution_ok", True)`）、产物路径与哈希是否都有。任一项失败 → `reject`；否则若 `candidate["eligible"]` 为假 → `tune`；全过 → `accept`。

```python
if not checks["partition"] or not checks["artifact"] or not candidate.get("execution_ok", True):
    decision = "reject"
elif not checks["candidate"]:
    decision = "tune"
else:
    decision = "accept"
```

关键细节：判断执行失败用的是 `execution_ok`，**不是** `eligible`。如果只写 `elif not checks["candidate"]: tune`，训练直接崩掉的候选会被误判成"调调参就好"。测试 `test_dpo_execution_failure_rejected` 就是抓这个：`decide_dpo_delivery({"admitted": True}, candidate, {"path": "adapter", "sha256": "abc"})` 必须返回 `reject`。同理，缺 `path` 或 `sha256` 是 `reject`，因为无法证明交付的是哪份适配器。

**第 5 步：真实项目的固定条件与门槛。** 5.1 的常量决定实验口径：`SFT_MANIFEST` 指向 62 节的 `sft_delivery_v2_1`，`TRAIN_PAIRS=24`、`EVALUATION_PAIRS=16`、`MAX_LENGTH=1024`、`UPDATE_STEPS=12`、`BETA=0.1`、`LEARNING_RATE=1e-5`、`DTYPE="bfloat16"`、`SEED=42`。门槛是 `minimum_accuracy_gain=0.0`、`minimum_margin_gain=0.0`、`minimum_margin_floor_gain=-0.05`、`maximum_prompt_truncation_ratio=0.25`、`maximum_response_truncation_ratio=0.10`、`maximum_peak_allocated_bytes=10 * 1024**3`（10 GiB）。

`MAX_LENGTH` 从 v4 的 768 提到 1024，注释写明了原因："v4 在 768 Token 下输入截断率为 27.5%；本轮只扩大序列覆盖"。5.4 的 `run_dpo_delivery` 会从同一 SFT 适配器初始化策略（可训练）和参考（冻结），只对策略参数建 AdamW；训练时每步取 `train_pairs[step % len(train_pairs)]`，12 步正好轮完 24 对的一半。序列构造 `build_preference_pair_sequences` 对长 prompt 从左侧截断，chosen/rejected 共享同一段保留的 prompt，并各自至少保留一个回答词元。

**第 6 步：怎么读 5.5/5.6 的登记结果。** notebook 保存了上一次 `dpo_offline_len1024_v5` 的 `real_gpu` 输出（注意源单元格默认仍是 `RUN_MODE="dry_run"`、`RUN_LABEL="replace_with_unique_label"`，所以这些输出对应的是历史标签，不是当前默认配置）。`decide_dpo_candidate_for_run` 在 CPU 三态之外，把截断拆成 prompt/response 两项，并再查一次 `artifact`。读出的数据是：

| 指标 | 基线 | 候选 | 变化 |
|---|---|---|---|
| 偏好准确率 | 0.5625 | 0.5625 | 0.0 |
| 平均边际 | 0.2814 | 0.2836 | +0.0022 |
| 最差边际 | −0.7857 | −0.7893 | −0.0036 |

0.5625 恰好是 9/16，说明 16 组留出对里有 9 组的优选分数更高；变化门槛是 0.0，所以"没有退步"也算过。最差边际下降 0.0036，但门槛允许 −0.05，所以也过。峰值分配显存 6 970 438 656 字节 = 6.49 GiB，低于 10 GiB；峰值保留显存 10 877 927 424 字节 = 10.13 GiB，已经超过 10 GiB，但门槛字段是 `peak_allocated_bytes`，保留显存不参与这轮判断——这是一个值得记住的口径差异。截断方面，5.6 写的是"任务输入截断率从 v4 的 27.5% 降至 5%，回答截断率为 0"，5% 对应 40 对里 2 对输入被截。

四项全过后 `decision == "accept"`，`failed_checks == []`。但这个 `accept` 只表示"离线偏好候选检查通过"，notebook 明确说"不表示已完成最终算法采用"，实际生成质量、安全性和成本要由后续评测决定，候选清单交给 86 节核对生命周期。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 偏好对（preference pair） | 同一问题的优选回答与非优选回答，附共享 prompt |
| 策略 $\pi_\theta$ / 参考 $\pi_{\text{ref}}$ | 正在训练、会更新的模型 / 冻结不动、用来当锚点的旧模型 |
| $\beta$ | 控制策略能偏离参考多远的缩放系数 |
| 回答分数 $s_\theta(x,y)$ | 回答词元的平均对数概率，越大表示模型越"认可"这条回答 |
| 边际（margin） | 优选分数减非优选分数，大于 0 表示排序正确 |
| 偏好准确率 | 边际大于 0 的偏好对占比 |
| 平均边际 / 最差边际 | 所有边际的均值 / 最小值，分别看整体和短板 |
| 长度归一化 | 用平均而不是求和，避免长回答天然分数更低 |
| 截断率 | 超过 `MAX_LENGTH` 被截掉部分的样本比例 |
| 适配器哈希（sha256） | 证明交付的是哪一版权重，缺它无法追溯 |
| `reject` / `tune` / `accept` | 数据或执行/产物失败 / 指标未达标 / 全部通过 |

## 小白常见疑问
**Q1：为什么参考模型必须冻结？不冻结不是学得更快吗？**
DPO 的损失比较的是"策略相对参考的偏好增量"。如果参考也跟着更新，基准会一起漂移，`policy_chosen - policy_rejected` 和 `reference_chosen - reference_rejected` 同向变化，损失可能下降但偏好排序没有真正改善。代码里 `load_policy(False)` 后把 `requires_grad_(False)`，且只在 `evaluate` 和训练打分时用 `torch.no_grad()`。

**Q2：偏好准确率一个都没提高（0.0），为什么还能 `accept`？**
因为门槛 `minimum_accuracy_gain=0.0` 只要求"不退步"，而另外两项——平均边际 +0.0022、最差边际 −0.0036（下限 −0.05）——都过了。这是当前项目设定的验收口径，不代表模型变强了很多；notebook 也提醒"不据此自动采用"。16 组样本里准确率的分辨率是 1/16=0.0625，想提高一档需要至少多对一对。

**Q3：最差边际变化为什么不能直接看候选最小值减基线最小值？**
因为两次最小值可能来自不同的偏好对。notebook 在 Step 2 明确写了这一点：`margin_floor_gain` 只回答"最小值的差"，逐条退化还要看每对的 `candidate_margin - baseline_margin`。测试里候选 `[-0.21, 0.05, 0.25, 0.47]` 的均值仍是 0.14，就是靠最差项才发现问题。

**Q4：训练崩了（`status="failed"`）为什么是 `reject` 而不是 `tune`？**
`tune` 的语义是"训练成功，但指标或资源没达标，调参后再试"；执行失败连候选都没产生，没有可调的指标。`test_dpo_execution_failure_rejected` 专门验证这一点，实现里必须用 `candidate.get("execution_ok", True)` 而不是只看 `eligible`。

**Q5：5.6 说 `accept`，是不是可以直接把这个适配器上线？**
不是。这里的 `accept` 只覆盖"当前 SFT 基线、固定 40 组偏好对、12 次更新、1024 序列上限"下的离线检查。notebook 明确写"实际生成回答的质量、安全性与任务成本，再用相应评测检查"，并把候选清单交给 86 节做生命周期核对；84 与 85 是并列分支，84 的产物不是 85 的输入。

## 动手验证：怎么知道自己做对了
测试区先跑 `test_preference_partition_contract()`、`test_dpo_candidate_gates()`、`test_dpo_delivery_decision()`，打印 `所有测试通过！`，**然后**才跑 `test_dpo_execution_failure_rejected()`。也就是说这行打印出现在最后一个测试之前，不能把它当成"全部通过"的凭证；真正的通过标准是整格执行完没有异常。

- `test_preference_partition_contract`：`["a","b"]` 对 `["c"]`、`["p1","p2"]` 对 `["p3"]`、`valid_ratio=1.0` 时 `admitted`；pair 重合、prompt 重合、`identical_response_count=1` 三种情况都为假。
- `test_dpo_candidate_gates`：fixture 下 `eligible`；显存 10 001 为假、10 000 为真；`[-0.21, 0.05, 0.25, 0.47]` 为假；截断 0.20 为假；候选边际长度 1 时抛 `ValueError`。
- `test_dpo_delivery_decision`：产物 `{"path": "adapter", "sha256": "abc"}` 时 `accept`；空产物 `reject`；`{"eligible": False}` 加完整产物 `tune`。
- `test_dpo_execution_failure_rejected`：`status="failed"` 的候选与 `{"admitted": True}` 组合必须 `reject`。

notebook 保存的执行输出对应历史标签 `dpo_offline_len1024_v5`：`decision=accept`、`failed_checks=[]`、峰值分配 6.49 GiB、峰值保留 10.13 GiB、准确率 0.5625、平均边际 +0.0022、最差边际 −0.0036。复跑时如果改了 `RUN_LABEL`，这些数字不能直接搬用；以新证据 JSON 为准。
