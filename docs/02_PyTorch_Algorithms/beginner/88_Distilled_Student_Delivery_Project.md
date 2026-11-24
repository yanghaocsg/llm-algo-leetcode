# 88. Distilled Student Delivery Project | 蒸馏学生交付项目 · 小白入门

> 配套 notebook：[88. 蒸馏学生交付项目](../88_Distilled_Student_Delivery_Project.md) ｜ 难度：Hard ｜ 环境：CPU 完成合同与决策练习；`PT-DISTILL-CUDA` 执行真实蒸馏实验

## 一句话说清

蒸馏训练后损失下降，不等于学生模型可以交付。这本 notebook 教你先固定三组对比实验的条件，再用质量、模型文件、大小和延迟的证据，决定接受、调整、补测还是拒绝。

## 为什么需要它

你把一个小模型（学生）用大模型（教师）蒸馏了一遍，训练时的蒸馏损失一路下降。这只说明学生更接近教师的分布，不能证明学生真的变好了，更不能证明它可以交付。比如学生总分高了，但某一类题比蒸馏前还差；或者它比原来的小模型慢了不少。

只看损失会出三种错：

- 对比不公平：两组蒸馏实验的学生起点或训练步数不同，分数差没法归因到蒸馏方法上。
- 评测被污染：评测数据和训练数据重叠，分数再高也不可信。
- 证据不全就下结论：没检查模型文件能不能独立加载，或者没在相同的输入和解码参数下量延迟。

这本 notebook 把这件事当成一个项目：先用一份“合同”固定并核对三组实验的共同条件，再选训练路线、算候选相对学生基线和教师的指标，最后按固定顺序给出 `reject`、`tune`、`measure` 或 `accept`。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 项目合同（contract） | 让三组实验可以比较的固定条件：教师冻结、同一个学生起点、相同的训练预算、训练和评测数据不重叠 |
| G0 / G1 / G2 | 三组实验：G0 是不做蒸馏的学生基线，G1 用固定轨迹蒸馏，G2 用在策略蒸馏 |
| 固定轨迹 / 在策略轨迹（offline trace / on-policy） | 固定轨迹是数据集的参考答案或教师预先写好的文本；在策略轨迹是学生自己生成的回答 |
| 准入（admission） | 一条路线开始训练前必须通过的检查，比如轨迹能不能解析、教师和学生的词表能不能对齐、显存够不够 |
| 交付门禁（delivery gate） | 训练出来的候选交出去之前必须通过的检查：合同、质量、产物、大小和延迟 |
| 评测污染 | 评测数据和训练数据重叠，分数不可信；合同要求两者的哈希不同 |
| `reject` / `tune` / `measure` / `accept` | 交付判断的四种结果：拒绝、调整后复测、补做推理测量、接受并交接 |

## 核心思路

项目里有三组实验，共用同一个学生起点。G0 是学生基线，不做蒸馏训练，用来标出学生现在的水平；G1 用固定轨迹蒸馏；G2 用在策略蒸馏，学生自己生成回答，教师对它访问的前缀给出逐词元分布。程序做四件事：核对合同、选路线、算候选证据、给出交付判断。选路线是第一层决策：只有离线通过选 `offline`，只有在线通过选 `online`，两条分别通过才能选 `hybrid`，都没通过是 `blocked`。

候选训练完以后，用五个比值描述它。`Q` 是质量分数，`S` 是学生产物（artifact，模型文件、配置和分词器）的大小，`L` 是推理延迟；下标 `T` 是教师，`0` 是学生基线 G0，`c` 是候选：

```text
ΔQ_c = Q_c - Q_0     质量增益：候选比 G0 高多少
R_c  = Q_c / Q_T     教师保持率：候选保住了教师几成能力
C_c  = S_c / S_T     候选大小是教师的几成
P_c  = L_c / L_T     候选延迟是教师的几成
B_c  = L_c / L_0     候选延迟是 G0 的几倍，用来防止蒸馏后反而变慢
```

用测试里的数字算一遍，再看交付判断的顺序（判断里的门槛 0.10 和 0.80 是示意值）：

```python
teacher   = {"quality": 0.80, "size_mb": 3000.0, "latency_ms": 60.0}   # 教师
g0        = {"quality": 0.55, "size_mb": 1000.0, "latency_ms": 24.0}   # 学生基线
candidate = {"quality": 0.72, "size_mb": 1000.0, "latency_ms": 24.0}   # 蒸馏出的候选

q, s, l = candidate["quality"], candidate["size_mb"], candidate["latency_ms"]
gain = round(q - g0["quality"], 2)              # 质量增益 ΔQ_c
keep = round(q / teacher["quality"], 2)         # 教师保持率 R_c
size_vs_teacher = round(s / teacher["size_mb"], 3)         # C_c
latency_vs_teacher = round(l / teacher["latency_ms"], 2)   # P_c
latency_vs_g0 = round(l / g0["latency_ms"], 2)             # B_c
print(gain, keep)
print(size_vs_teacher, latency_vs_teacher, latency_vs_g0)


def decide(contract_ok, artifact_ok, gain, keep, measured):   # 门槛 0.10 和 0.80 是示意
    if not contract_ok or not artifact_ok:
        return "reject"        # 合同或产物有问题
    if gain < 0.10 or keep < 0.80:
        return "tune"          # 质量增益或教师保持率不够
    if not measured:
        return "measure"       # 质量过了，但缺少同协议的推理测量
    return "accept"


print(decide(False, True, gain, keep, True))
print(decide(True, False, gain, keep, True))
print(decide(True, True, 0.05, keep, True))
print(decide(True, True, gain, keep, False))
print(decide(True, True, gain, keep, True))
```

```text
0.17 0.9
0.333 0.4 1.0
reject
reject
tune
measure
accept
```

- 前两行是五个比值：候选比 G0 高 0.17，质量是教师的 0.9；大小约是教师的三分之一，延迟是教师的 0.4 倍，和 G0 一样快（`B_c` 是 1.0，没有延迟回归）。
- 前两个 `reject` 是合同有问题，或者学生产物没通过检查；`tune` 是质量增益 0.05，不到门槛。
- `measure` 是质量够了，但还没有同协议的推理测量；都满足才是 `accept`。

判断顺序是固定的：合同和产物是硬门禁，先判 `reject`；再看质量增益和教师保持率，不够是 `tune`；质量过了但缺少推理证据是 `measure`；最后看大小、延迟和类别退化的门槛，全过才是 `accept`，这一关不过是 `tune`。每个评测类别还要分别和 G0 比较，避免总分掩盖局部退化。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 怎样固定共同合同并区分蒸馏路线：G0、离线轨迹、在线 GKD、混合四条路线的学生起点、轨迹来源和教师作用 | 读 |
| Step 2 | 每条路线需要哪些准入与交付证据：五个比值的含义，以及从路线准入到部署收益每一阶段查什么、不过怎么办 | 读，对照上面的式子 |
| Step 3 | 怎样选择训练路线并形成交付决策：`offline`、`online`、`hybrid`、`blocked` 四种路线状态，以及 `reject`、`tune`、`measure`、`accept` 四种交付状态 | 读，对照上面的判断顺序 |
| Step 4 · TODO 1 | 检查模型身份、教师冻结、数据隔离与 G1/G2 配对条件 | 在 `validate_distillation_contract` 里写约 25 行：建一个 `issues` 列表，逐项检查并把发现的问题名加进去，最后返回 `ready` 和 `issues` |
| Step 4 · TODO 2 | 分别判断 offline 与 online 是否准入，再选择路线 | 在 `select_distillation_route` 里写约 17 行：分别算离线和在线两组检查结果，全为真才算通过，再组合成路线名 |
| Step 4 · TODO 3 | 计算质量、教师侧资源比、G0 延迟比和类别退化 | 在 `summarize_distilled_candidate` 里写约 31 行：检查数值、类别集合和分母，再算出质量与部署的比值、退化的类别，以及产物和延迟协议的状态 |
| Step 4 · TODO 4 | 按 reject > tune > measure > accept 的顺序判断 | 在 `decide_distilled_student` 里写约 11 行：按硬失败、质量、推理证据、部署门槛的顺序，返回决定和原因 |
| 测试 | 没有单独的标题，紧跟在 Step 4 的题目单元后面；检查路线阻塞、学生起点漂移、评测污染、质量不足、缺少部署证据和完整通过 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实环境里做完整实验：教师和轨迹检查、预检、G0 基线、G1 和 G2 训练、确认复测，最后形成决策和交接 | 需要 `PT-DISTILL-CUDA` 环境，默认是 `RUN_STAGE = "prepare"` |

Step 1–4 只需要 CPU。Step 5 需要 CUDA GPU，环境名是 `PT-DISTILL-CUDA`，默认是 `RUN_STAGE = "prepare"`：每次只切换一个阶段，默认先运行核心复现路径，这一步只登记计划、不运行训练；进入真实阶段前，要先把 `RUN_LABEL` 设成唯一标签。

## 容易踩的坑

- **三组实验的条件没对齐**：G1 和 G2 必须用同一个学生版本和相同的 `train_steps`，学生的版本变了，分数差就归因不到蒸馏方法上。训练和评测数据的哈希相同是评测污染，教师不是冻结状态，也都会被拒绝。
- **混合路线只看一边**：`hybrid` 要求离线和在线分别通过各自的检查。一边没通过，只能走通过的那一条；两边都没通过是 `blocked`。
- **分母为零，或者类别对不上**：教师质量和三组的大小、延迟都必须是正数，否则比值没有意义，要报错。教师、G0 和候选还必须给出同一组非空的评测类别，类别缺失不能被集合交集掩盖。
- **总分掩盖了类别退化**：除了总的质量分，每个评测类别都要分别和 G0 比，候选低于 G0 的类别要记下来，退化类别太多不能交付。
- **跳过顺序，或者没有推理证据就接受**：质量够了但没有同协议的推理测量，只能是 `measure`。候选必须带内容哈希、加载状态和重复延迟协议，产物没通过完整性或加载检查直接 `reject`。

## 读完后自查

- [ ] 能说出 G0、G1、G2 各是什么，以及它们必须共享哪些条件。
- [ ] 教师质量 0.80、G0 质量 0.55、候选质量 0.72，能算出质量增益和教师保持率。
- [ ] 能说出 `offline`、`online`、`hybrid`、`blocked` 各在什么情况下出现。
- [ ] 能按顺序说出 `reject`、`tune`、`measure`、`accept` 的触发条件。

## 下一步

- 打开 notebook：[88. 蒸馏学生交付项目](../88_Distilled_Student_Delivery_Project.md)
- 建议先读：[DISTILL-OPD. 在策略蒸馏与 GKD · 小白入门](./DISTILL-OPD_On_Policy_Distillation_and_GKD.md)
- 前置阅读对应：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)、[PT-SYNTHETIC-DATA. 合成指令生成与验证准入 · 小白入门](./PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.md)、[PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
