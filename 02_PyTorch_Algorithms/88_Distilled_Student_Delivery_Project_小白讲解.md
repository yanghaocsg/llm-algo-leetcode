# 88. 蒸馏学生交付项目 — 小白讲解

> 对应 notebook：[`88_Distilled_Student_Delivery_Project.ipynb`](./88_Distilled_Student_Delivery_Project.ipynb)

## 一句话总结
这是一个「蒸馏学生能不能交付」的项目页：它把问题拆成两层——先用四个 CPU 函数检查实验合同、选择训练路线、计算质量/大小/延迟指标、按优先级给出 `accept/tune/measure/reject` 决策；再给出一套完整的 GSM8K 蒸馏实验（G0 训练前学生、G1 固定轨迹、G2 在线 GKD、混合路线），说明数据切分、门槛和结果怎么读。历史终态运行给出了一个关键结论：混合候选把正确数从 19/64 提到 22/64（质量增益 0.046875），但没达到 0.05 的交付门槛，最终决策是 `tune`，不交接部署。

## 为什么需要它：痛点是什么
初学者做完蒸馏，最常见的三个错觉是：

- **损失降了就等于学生变好了。** 训练 loss 只说明学生拟合了教师分布或参考答案，不说明它在验证题上多解出了几道。
- **学生比教师小就一定更划算。** 小模型可能质量掉太多，或者为了答对而输出变长、延迟反而超过教师；文件小不等于服务便宜。
- **同一份数据切两半就互不重叠。** 训练集和评测集的文件哈希不同，只能说明"两个文件集合不同"，不能证明没有单条样本交叉。

生活类比：这不是"把大厨的菜谱抄给小厨"就结束了，而是一次交付验收：小厨要先和原来的自己比（多会做几道菜），再和大厨比（保留了多成水准），还要看厨房占地（文件大小）、出菜速度（延迟）、以及菜谱有没有缺页（产物校验与加载）。任何一项不合格，都不能挂牌营业。

## 核心问题
- 怎样证明 G0/G1/G2 是"从同一个学生起点出发"的公平比较？哪些字段不一致会让实验直接失去可比性？
- 固定轨迹蒸馏（G1）和在线 GKD（G2）各自的准入条件是什么？为什么两条都通过才叫 `hybrid`？
- 质量增益、教师保持率、大小比、延迟比分别以谁为分母？候选超过教师时为什么保持率可以大于 1？
- 为什么先判 `reject`，再判 `tune`，再判 `measure`，最后才 `accept`？顺序换一下会怎样？
- Step 5 的 GSM8K 四个切分各干什么？开发集、验证集、最终留出集为什么不能混用？
- 历史结果 19/64 → 22/64、增益 0.046875，为什么不是 `accept` 而是 `tune`？

## 解决思路：一步步看懂

### 第 1 步：三条路线与两个参照
Step 1 先固定比较框架：

| 路线 | 学生起点 | 回答来源 | 教师的作用 |
|---|---|---|---|
| G0：训练前学生 | 固定初始版本 | 学生推理输出 | 单独测量教师作为参照 |
| G1：固定轨迹蒸馏 | 同一初始版本 | 数据集参考答案 | 对固定前缀给出下一词元分布 |
| G2：在线 GKD | 同一初始版本 | 当前学生采样 | 对学生访问的前缀给出分布 |
| 混合路线 | 固定轨迹阶段保存的模型 | 先参考答案，后学生采样 | 两阶段分别反馈 |

两个参照必须分清：

- **和 G0 比**：回答"蒸馏改善了多少、推理是否变慢"。
- **和教师比**：回答"当前题集保留了多少质量、文件与延迟缩小多少"。

这就是 Step 2 五个公式的由来。设教师、G0、候选的质量为 $Q_T,Q_0,Q_c$，文件大小为 $S_T,S_0,S_c$，延迟为 $L_T,L_0,L_c$：

$$
\Delta Q_c=Q_c-Q_0,\qquad
R_c=\frac{Q_c}{Q_T},\qquad
C_c=\frac{S_c}{S_T},\qquad
P_c=\frac{L_c}{L_T},\qquad
B_c=\frac{L_c}{L_0}.
$$

逐项解释：$\Delta Q_c$ 是绝对质量增益；$R_c$ 是教师质量保持率；$C_c$ 是候选文件相对教师的大小；$P_c$ 是候选延迟相对教师；$B_c$ 是候选延迟相对训练前学生。质量越高越好，大小和延迟越小越好。若候选在某个指标上超过教师，比如 $Q_c>Q_T$，$R_c$ 就会大于 1；代码没有把它截断到 1，这是有意的。

G1 与 G2 的差别不只在"谁生成回答"，还在训练信号落在哪些位置：

- **G1 固定轨迹**：回答是数据集参考答案，学生被强制在这条固定序列上前进；教师在这些固定前缀上给出下一词元分布。好处是可复现、可离线准备；坏处是训练分布固定，学生推理时一旦走偏，训练里没见过这种前缀。
- **G2 在线 GKD**：回答由当前学生自己采样，教师再在学生实际访问的前缀上给分布。好处是训练分布跟着学生走，能纠正学生自己会犯的错；坏处是要在线生成、要同时驻留教师和学生、要保证两边词表逐词元对齐。
- **混合路线**：先 G1 得到一个不太差的学生，再 G2 在这个学生自己的分布上继续纠正。notebook 的 `HYBRID_RECIPE` 把两阶段的学习率、步数和 `hard_label_weight` 分开写，就是为了让"哪一阶段贡献了什么"可追溯。

这也是为什么 `select_distillation_route` 把在线条件写成 `teacher_ready`、`full_logit_compatible`、`memory_admitted` 三项：教师没就绪、词表对不齐或显存放不下，G2 根本无法开始，和"固定轨迹质量好不好"是两组独立条件。

### 第 2 步：TODO 1 `validate_distillation_contract`——先保证可比
这个函数不训练任何东西，只回答"三组实验是不是同一条起跑线"。参考实现按顺序收集 `issues`：

```python
if spec.get("schema_version") not in {"distilled-student-project/v1", "distilled-student-project/v2"}:
    issues.append("schema")
if not all((teacher.get("model_id"), teacher.get("revision"),
            student.get("model_id"), student.get("revision"))):
    issues.append("model_identity")
if teacher.get("frozen") is not True:
    issues.append("teacher_not_frozen")
if not data.get("train_hash") or not data.get("evaluation_hash") \
        or data.get("train_hash") == data.get("evaluation_hash"):
    issues.append("evaluation_overlap")
```

四项分别对应：

| 问题标签 | 触发条件 | 为什么必须拦 |
|---|---|---|
| `schema` | 版本不是 `v1`/`v2` | 字段结构可能不同，后续读取不可信 |
| `model_identity` | 教师或学生缺 `model_id`/`revision` | 无法确认比的是哪两个快照 |
| `teacher_not_frozen` | `frozen` 不是严格的 `True` | 教师若在训练中更新，"教师分布"就不是固定参照 |
| `evaluation_overlap` | 哈希缺失或训练/评测哈希相同 | 训练集与评测集相同，分数虚高 |

然后检查组：

```python
if any(name not in groups for name in ("g0", "g1", "g2")):
    issues.append("missing_group")
else:
    if any(groups[name].get("student_revision") != student.get("revision") for name in required_groups):
        issues.append("student_start_mismatch")
    if groups["g1"].get("train_steps") != groups["g2"].get("train_steps"):
        issues.append("training_budget_mismatch")
    if groups["g1"].get("trajectory_mode") != "fixed" or groups["g2"].get("trajectory_mode") != "on_policy":
        issues.append("trajectory_mode")
```

- `student_start_mismatch`：G0/G1/G2 必须都等于同一个 `student.revision`。否则"训练前 vs 训练后"的差值混入了不同初始权重，不可比。
- `training_budget_mismatch`：G1 和 G2 的 `train_steps` 必须相同。测试 fixture 都是 20 步；步数不同时，质量差异可能只是"谁训练得更久"。
- `trajectory_mode`：G1 必须是 `fixed`，G2 必须是 `on_policy`。如果填反，路线标签和实际数据来源对不上。

返回值是 `{"ready": not issues, "issues": issues}`。注意 `ready` 不是"遇到第一个错就停"，而是所有问题都收集完再判断；这样一次运行就能看到全部合同缺陷。

测试 `test_distillation_contract` 用 fixture（教师 `Qwen/Qwen2.5-1.5B-Instruct`、学生 `Qwen/Qwen2.5-0.5B-Instruct`、`train-hash`/`eval-hash`、G1/G2 各 20 步）验证 `ready` 为真，再分别改三处：

- 把 `g2.student_revision` 改成 `other-rev` → `student_start_mismatch`；
- 把 `evaluation_hash` 改成和 `train_hash` 相同 → `evaluation_overlap`；
- 把 `teacher.frozen` 改成 `False` → `teacher_not_frozen`。

**一个容易忽略的边界**：两个整集哈希不同只能说明文件集合不同，不能证明没有单条样本交叉。notebook 解析明确要求真实数据隔离还要"逐条比较样本或提示"。

### 第 3 步：TODO 2 `select_distillation_route`——两条路线各自过门
固定轨迹和在线蒸馏的准入条件不同，参考实现先分别算布尔字典：

```python
offline_checks = {
    "trace_available": offline.get("trace_available") is True,
    "parse_ratio": offline.get("parse_ratio", 0.0) >= gates["min_trace_parse_ratio"],
    "quality": offline.get("quality_score", 0.0) >= offline.get("baseline_quality_score", 1.0),
    "saturation": offline.get("saturation_ratio", 1.0) <= gates["max_trace_saturation_ratio"],
}
online_checks = {
    "teacher_ready": online.get("teacher_ready") is True,
    "full_logit_compatible": online.get("full_logit_compatible") is True,
    "memory_admitted": online.get("memory_admitted") is True,
}
```

- 离线四项：轨迹可用；解析率不低于阈值；质量不低于基线；截断/饱和率不高于阈值。注意默认值的方向：`parse_ratio` 缺省 0.0 会失败，`saturation_ratio` 缺省 1.0 会失败，`baseline_quality_score` 缺省 1.0 让质量几乎不可能通过——缺字段不会被当成"合格"。
- 在线三项：教师就绪、全词表分布兼容、显存准入。在线蒸馏要拿教师的完整下一词元分布，词表对不齐就无法算 KL；两个模型同时驻留还要显存够。

路线组合只有一行嵌套条件：

```python
route = "hybrid" if offline_ready and online_ready else "offline" if offline_ready else "online" if online_ready else "blocked"
```

读法：两条都过 → `hybrid`；只过固定轨迹 → `offline`；只过在线 → `online`；都不过 → `blocked`。返回里还带 `offline_ready`、`online_ready` 和两个检查字典，方便定位是哪一项失败。

测试 `test_distillation_route_selection` 的门槛是 `min_trace_parse_ratio=0.80`、`max_trace_saturation_ratio=0.25`：

- `offline` 解析率 0.90、质量 0.75 ≥ 基线 0.70、饱和率 0.10，在线三项全真 → `hybrid`。
- 把 `full_logit_compatible` 改成 `False` → 只剩离线通过 → `offline`。
- 再把解析率降到 0.50、质量降到 0.60（低于基线 0.70）→ 两边都失败 → `blocked`。

解析里补充：在线单独通过的分支也应返回 `online`，只是现有测试没有构造这个用例。

### 第 4 步：TODO 3 `summarize_distilled_candidate`——五类指标与三重校验
函数先做输入校验，再做计算。校验分四层：

1. **类型**：三组记录的 `quality_score`、`artifact_size_mb`、`latency_ms` 必须是 `int` 或 `float`，否则抛"缺少可比较数值"。
2. **正数**：教师质量必须 $>0$；三组的大小和延迟都必须 $>0$。因为后面要做分母，教师延迟 0 会直接除零。
3. **有限**：所有数值必须 `math.isfinite`，NaN/Inf 拒绝。
4. **类别集合**：教师、G0、候选必须提供**同一个非空**类别集合，且每个类别质量有限；否则无法比较类别退化。

核心计算：

```python
regressed_categories = [name for name in common_categories
                        if candidate_categories[name]["quality_score"] < baseline_categories[name]["quality_score"]]
return {
    "candidate": str(candidate.get("name", "candidate")),
    "quality_gain": candidate["quality_score"] - student_baseline["quality_score"],
    "teacher_retention": candidate["quality_score"] / teacher["quality_score"],
    "teacher_size_ratio": candidate["artifact_size_mb"] / teacher["artifact_size_mb"],
    "teacher_latency_ratio": candidate["latency_ms"] / teacher["latency_ms"],
    "baseline_size_ratio": candidate["artifact_size_mb"] / student_baseline["artifact_size_mb"],
    "baseline_latency_ratio": candidate["latency_ms"] / student_baseline["latency_ms"],
    "regressed_categories": regressed_categories,
    "regressed_category_count": len(regressed_categories),
    "artifact_verified": candidate.get("artifact_verified") is True,
    "load_ready": candidate.get("load_status") == "loaded",
    "latency_measured": latency_protocol.get("repeats", 0) >= 2
                        and latency_protocol.get("sample_count", 0) >= 2,
}
```

代入 `_evidence_fixture()` 手算：

| 指标 | 计算 | 结果 |
|---|---|---|
| `quality_gain` | $0.72-0.55$ | 0.17 |
| `teacher_retention` | $0.72/0.80$ | 0.90 |
| `teacher_size_ratio` | $1000/3000$ | $1/3\approx0.3333$ |
| `teacher_latency_ratio` | $24/60$ | 0.40 |
| `baseline_latency_ratio` | $24/24$ | 1.0 |
| `baseline_size_ratio` | $1000/1000$ | 1.0 |
| `regressed_category_count` | core 0.70 ≥ 基线 0.60 | 0 |

`latency_measured` 要求 `repeats >= 2` 且 `sample_count >= 2`；fixture 是 3 和 24，所以为真。`artifact_verified` 必须严格 `is True`，`load_ready` 必须 `load_status == "loaded"`——一个字符串写错就会让后面直接 `reject`。

测试 `test_candidate_summary` 断言前五个值和退化数；再检查三类坏输入：质量 NaN、类别字典为空、教师延迟 0，都必须抛 `ValueError`。注意 `baseline_size_ratio` 虽然返回了，但测试没有断言它；它主要供报告阅读。

### 第 5 步：TODO 4 `decide_distilled_student`——决策有严格优先级
参考实现的判断顺序是固定的：

```python
if not contract.get("ready"):
    return {"decision": "reject", "reason": "; ".join(contract.get("issues", []))}
if not summary.get("artifact_verified") or not summary.get("load_ready"):
    return {"decision": "reject", "reason": "学生产物未通过完整性或加载检查"}
if summary["quality_gain"] < gates["min_quality_gain"] or summary["teacher_retention"] < gates["min_teacher_retention"]:
    return {"decision": "tune", "reason": "学生增益或教师能力保持率未达到门槛"}
if not summary.get("latency_measured"):
    return {"decision": "measure", "reason": "缺少同协议教师—学生推理测量"}
if (summary["teacher_size_ratio"] > gates["max_teacher_size_ratio"]
        or summary["teacher_latency_ratio"] > gates["max_teacher_latency_ratio"]
        or summary["baseline_latency_ratio"] > gates["max_baseline_latency_ratio"]
        or summary["regressed_category_count"] > gates["max_regressed_categories"]):
    return {"decision": "tune", "reason": "学生质量通过，但部署收益未达到门槛"}
return {"decision": "accept", "reason": "质量、产物、大小和延迟门槛全部通过"}
```

为什么顺序不能乱？

- **`reject` 最先**：合同不成立（起点、数据、训练设置不一致）或产物无法校验/加载，后面所有数字都不可信。先修这些，再谈调参。
- **质量不足 `tune`**：学生还没学到东西，补测延迟没有意义，先调训练。
- **`measure` 在质量之后**：质量已达标，但只测了 1 次延迟，不能拿单次数字做部署决定；先按同协议重复测量。
- **部署资源不达标 `tune`**：质量够了但文件不够小、延迟不够低或类别退化，需要调训练/结构，而不是"再测一次"。
- **最后才 `accept`**：所有条件同时满足，才允许交接。

测试 `test_delivery_decision` 的门槛是 `min_quality_gain=0.10`、`min_teacher_retention=0.85`、`max_teacher_size_ratio=0.50`、`max_teacher_latency_ratio=0.60`、`max_baseline_latency_ratio=1.05`、`max_regressed_categories=0`，四种输入分别命中：

| 输入 | 决策 | 原因 |
|---|---|---|
| `contract={'ready': False, 'issues': ['evaluation_overlap']}` | `reject` | 合同不成立 |
| `summary` 的 `quality_gain` 改成 0.02 | `tune` | 低于 0.10 |
| `summary` 的 `latency_measured=False` | `measure` | 质量 0.17/0.90 已过，但计时样本不足 |
| 原始 summary | `accept` | 0.17≥0.10、0.90≥0.85、$1/3$≤0.50、0.40≤0.60、1.0≤1.05、退化 0 |

注意第三个用例：`latency_measured=False` 时质量检查仍然先通过，才会落到 `measure`；如果质量也不达标，会先返回 `tune`。

### 第 6 步：Step 5 的真实实验怎么设计
Step 5 是可选 GPU 实验，核心复现链是：

```text
prepare → prompt_check → preflight → g0 → g1 → g2 → close
```

可选分支：`teacher_check → trace_check`（换教师/检查离线轨迹）、`diagnose`（G2 诊断）、`hybrid`（先固定轨迹 64 步，再在线 32 步）、`confirm`（多种子复测）。

**固定项（控制变量）**：

| 项目 | 固定内容 |
|---|---|
| 教师 / 学生 | `Qwen/Qwen2.5-1.5B-Instruct` / `Qwen/Qwen2.5-0.5B-Instruct` |
| 数据类型与批量 | `bfloat16`、`batch_size=1`、`use_vllm=False` |
| 长度与步数 | prompt/completion 各最多 256，`max_steps=128`，`dose_steps=[128]` |
| 优化 | `learning_rate=1e-5`、`optimizer=sgd`、`beta=0.0`、`hard_label_weight=0.0`、warmup 0、weight decay 0 |
| 随机性 | 训练种子 42，确认种子 42/43 |
| 计时 | 2 条预热提示，重复 2 次 |

这里 `beta=0.0`、`hard_label_weight=0.0`、SGD 的组合，对应 notebook 说的"forward-KL 基准配方"：G1 只学教师分布，不混硬标签。混合配方单独定义：anchor 阶段 64 步、lr $5\times10^{-5}$、`hard_label_weight=0.5`；online 阶段 32 步、lr $5\times10^{-5}$、`beta=0.1`；优化器换成 AdamW，LoRA rank 16、alpha 32、dropout 0.05，作用于 q/k/v/o 和 gate/up/down 全部投影。

**数据切分**：GSM8K 题目按"参考解答里的计算标记数与非空行数的较大值"排序，再以解答长度等打破并列，均分四档。四档描述的是参考解答的形式复杂度。

| 切分 | 样本数 | 用途 |
|---|---|---|
| 训练集 | 512 | 四档均衡抽样，提供训练提示与参考答案 |
| 开发集 | 32（四档各 8） | 选择提示词与训练剂量 |
| 验证集 | 64（四档各 16） | 比较教师、候选配方与输出成本 |
| 最终留出集 | 64（四档各 16） | 只登记身份与提示重合；**最终评分尚未执行** |

开发集只用来选提示词和剂量，验证集用来比较候选，最终留出集必须等配方冻结后再开。notebook 反复强调这一点，因为"拿验证集选完模型再报验证集分数"就是选择性偏差。

**自变量**：G0（不训练）、G1（固定参考答案 + 教师分布）、G2（学生自己采样 + 教师分布）、混合（先 G1 后 G2）。**因变量**：质量分数、教师保持率、文件大小、延迟、类别退化、截断率、格式合规率、显存与训练成本。**baseline**：G0 是"训练前学生"，教师是"质量上界"；两者都不能少。

**门槛**（`DELIVERY_GATES`）：质量增益 ≥ 0.05、教师保持率 ≥ 0.85、大小比 ≤ 0.50、教师延迟比 ≤ 0.70、G0 延迟比 ≤ 1.05、退化类别 ≤ 0。确认阶段还要求格式合规率 ≥ 0.80、延迟比 ≤ 1.10、两个种子都合格（`required_qualified_runs=2`）。

**结果怎么读**：`close` 只读取同标签的 G0/G1/G2 记录，先核对 `config_hash`、开发/验证集身份，再对 G1/G2 分别调用 `summarize_distilled_candidate` 和 `decide_distilled_student`。如果 G1 只有矩阵筛选、没有持久化产物，`close_promotable=False`，会被直接判 `tune` 而不是拿内存里的筛选分数去交接。最终按 `decision_rank`（`accept` 3、`tune` 2、`measure` 1、`reject` 0）选一个候选，只有 `accept` 才写 `handoff`。

### 第 7 步：历史结果为什么是 tune
notebook 末尾登记了一次已结束的混合实验 `rtx5070ti_distill_gsm8k_hybrid_terminal_v9`：RTX 5070 Ti Laptop、PyTorch 2.11.0+cu128、1.5B 教师 / 0.5B 学生、512 训练 / 32 开发 / 64 验证，先在固定参考答案上训练 64 步，再在线蒸馏 32 步。

| 观察项 | 数值 |
|---|---|
| 学生基线 | 19/64 |
| 混合候选 | 22/64 |
| 质量增益 | $22/64-19/64=3/64=0.046875$ |
| 四档类别 | 均未下降 |
| 截断率 | 0.2500 → 0.1406 |
| 格式合规率 | 0.5312 |
| 平均延迟 | 1010.2 ms → 1337.6 ms |

对照门槛就能读懂决策：

- 质量增益 $0.046875 < 0.05$（`min_quality_gain`），差 0.003125，所以先卡在 `tune`。
- 格式合规率 0.5312 < 0.80（确认门槛），也不够。
- 延迟从 1010.2 涨到 1337.6 ms，比值约 $1337.6/1010.2\approx1.324$，超过 `max_baseline_latency_ratio=1.05`，同样指向 `tune`。
- 最终 `selection_status=no_eligible_candidates`，`deployment_ready` 为假，`handoff` 为空，最终留出集没有评分。

这不是"实验失败"，而是**交付门槛在起作用**：候选确实比 G0 多对了 3 题、截断更少、类别没退化，但质量增益没到线、格式和延迟也不达标，所以不能交接。notebook 另外说明：DeepSeek 离线轨迹未通过质量与可用率门槛；当前教师的在线全词表分布通过了兼容性和显存预检；但最终训练候选仍未同时满足交付门槛。历史表只登记了这些指标与决策，不能据此报告"学生整体质量提升"或"可以部署"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 知识蒸馏 | 让小模型学习大模型的输出分布或回答 |
| 固定轨迹蒸馏（G1） | 用数据集参考答案当学生输入，教师在这些固定前缀上给分布 |
| 在线 GKD（G2） | 学生自己采样回答，教师在学生访问的前缀上给分布 |
| 混合路线 | 先固定轨迹训练一个中间模型，再从它继续在线蒸馏 |
| G0 | 训练前的学生，用来衡量"蒸馏到底改善了多少" |
| 教师保持率 $R_c$ | 候选质量 ÷ 教师质量；超过 1 表示在该题集上超过教师 |
| 大小比 $C_c$ | 候选文件大小 ÷ 教师文件大小 |
| 延迟比 $P_c$ / $B_c$ | 候选延迟相对教师 / 相对 G0 |
| 质量增益 $\Delta Q_c$ | 候选分数减 G0 分数，绝对差值 |
| 类别退化 | 候选在某个类别上的质量低于 G0 |
| 验证集 / 最终留出集 | 用来比较候选 / 配方冻结后才允许评分的独立数据 |
| `reject` / `tune` / `measure` / `accept` | 修合同或产物 / 调训练 / 补计时 / 可交接 |

## 小白常见疑问

**Q1：为什么 G1 和 G2 必须 `train_steps` 相同、学生版本也相同，才算公平？**
因为要比较的是"轨迹来源"这一个变量。学生版本不同，差异可能来自初始化；步数不同，差异可能只是训练更久。`validate_distillation_contract` 用 `student_start_mismatch` 和 `training_budget_mismatch` 两个标签分别拦住这两种情况。测试 fixture 让 G1/G2 都是 20 步、都等于 `student-rev`，改其中一项就会命中对应问题。

**Q2：两个整集哈希不同，为什么还不能保证训练集和评测集不重叠？**
哈希是对整个文件算的。两个文件只要有一条样本不同，整文件哈希就不同；但它们仍可能共享另外 99 条样本。notebook 的解析原话是「两个整集哈希不同只说明文件集合不同，部分样本仍可能相同」，真实隔离要"逐条比较样本或提示"。代码里的 `evaluation_overlap` 只能抓最粗暴的"哈希完全相同"。

**Q3：`teacher_retention=0.90` 和 `quality_gain=0.17` 不是都很好吗，为什么还要看大小和延迟？**
因为交付的是"更便宜的服务"，不是"更小的论文数字"。候选可能质量接近教师，但文件没小多少、延迟反而更高，那部署它没有收益。`decide_distilled_student` 在质量通过后还会检查 `teacher_size_ratio`、`teacher_latency_ratio`、`baseline_latency_ratio` 和类别退化；任何一项超门槛都返回 `tune`。测试 fixture 的大小比 $1/3$、教师延迟比 0.40、G0 延迟比 1.0 都在门槛内，才是 `accept`。这也说明交付门槛必须分质量、计时、资源三层，不能用一个总分替代。


**Q4：历史结果 22/64 比 19/64 多对 3 题，为什么质量增益只有 0.046875？**
因为质量分数是正确率：$22/64=0.34375$，$19/64=0.296875$，差值 $3/64=0.046875$。交付门槛是 0.05，所以差 0.003125。这个例子说明"多对 3 题"听起来不错，但换算成比例后可能刚好不到线；报告必须写清分母是 64，而不是只说"提升 3 题"。

**Q5：为什么最终留出集不能拿来选模型？**
因为一旦用最终留出集比较候选，它就不再是"没见过的数据"，报出来的分数会偏乐观。notebook 的分工是：开发集选提示词和剂量，验证集比较候选，最终留出集"登记身份与提示重合检查；最终评分尚未执行"。历史结果的 `final_holdout_opened` 也记录了是否打开留出集，`handoff` 为空时不应把验证集结论说成最终结论。

## 动手验证：怎么知道自己做对了
测试区在 cell 9，依次调用四个测试，最后打印 `✅ 蒸馏项目合同、候选证据和交付判断测试通过。`。TODO 未填时先抛 `NotImplementedError`。

- `test_distillation_contract`：fixture 的 `validate_distillation_contract(spec)["ready"]` 为真；改 `g2.student_revision` 后 `issues` 含 `student_start_mismatch`；让 `evaluation_hash == train_hash` 后含 `evaluation_overlap`；把 `teacher.frozen` 设为 `False` 后含 `teacher_not_frozen`。
- `test_candidate_summary`：`quality_gain` 与 0.17、`teacher_retention` 与 0.90、`teacher_size_ratio` 与 $1/3$、`teacher_latency_ratio` 与 0.40、`baseline_latency_ratio` 与 1.0 的误差都小于 `1e-9`；`regressed_category_count == 0`；质量 NaN、类别为空、教师延迟 0 三种坏输入都必须抛 `ValueError`。
- `test_distillation_route_selection`：两边都过返回 `hybrid`；在线 `full_logit_compatible=False` 返回 `offline`；离线解析率 0.50、质量 0.60 且在线不兼容返回 `blocked`。
- `test_delivery_decision`：合同未就绪 → `reject`；`quality_gain=0.02` → `tune`；`latency_measured=False` → `measure`；原始 summary → `accept`。

Step 5 保存了部分真实输出：5.3 的 `PROJECT_CONTRACT` 显示 `schema_version='distilled-student-delivery/v4'`、`run_label='rtx5070ti_distill_deepseek_trace_v2'`、`stage='trace_check'`，工作负载 `record_count=608`（train 512、development 32、evaluation 64；development 四档各 8，evaluation 四档各 16），`train_reference_count=512`；5.4 的 trace_check 记录显示 `trace_check_hash='b29aa16384d2c072cd53f211b1941fed65d9a19afacc1c6a0c6817e3d45e9566'`，教师轨迹配置为四类、每类 16 条、`max_completion_length=1024`、温度 0.6、top_p 0.95、种子 42、前缀 `<think>\n`；5.5 的证据清单里 preflight 和 G0 的 `config_hash` 都是 `8de01761…`，G0 状态 `ok`。5.6 的历史登记则给出 19/64 → 22/64、增益 0.046875、决策 `tune`、`selection_status=no_eligible_candidates`。这些数字来自保存的输出，不是要求你重跑；CPU 部分以四个测试的断言为准。
