# 63. LoRA 变体选型项目 — 小白讲解

> 对应 notebook：[`63_LoRA_Variants_Benchmark.ipynb`](./63_LoRA_Variants_Benchmark.ipynb)

## 一句话总结
本节在固定模型、数据、更新预算和随机种子集合的前提下，只改变 LoRA 秩（rank 4/8/16），先检查每份运行的质量与资源是否过门槛，再用“共享种子下的配对损失差”判断质量是否等价，最后在等价候选里选显存和步时更低的配置。三个 TODO 分别补全候选可比性检查、逐项硬门槛、配对损失差判断；CPU 测试覆盖了质量更好但显存超限、摘要通过但重复运行失败、以及“逐种子差中位数”和“两个中位数相减”不同的反例。

## 为什么需要它：痛点是什么
初学者容易认为“秩越大，可训练参数越多，效果一定越好”。但秩变大时，标准 LoRA 的缩放系数 $s=\alpha/r$ 会跟着变小：固定 $\alpha=16$ 时，秩 4、8、16 对应的系数分别是 4、2、1。也就是说，这次比较同时改变了参数数量和派生缩放，不能把结果单独解释成“容量增加”。更麻烦的是，很小的评测损失差可能随随机种子翻转；如果不做配对比较，只看两个候选各自的中位数，甚至会得出相反结论。

生活类比：选车不能只看“马力最大”。马力（秩）上去了，油耗（显存）、保养成本（步时）也可能上去；而且不同司机（随机种子）开同一辆车，成绩会波动。正确做法是让同一批司机分别开每辆车，比较同一司机下的差值，再在“性能差不多”的车里选更省油、更好停的那辆。

## 核心问题
- 怎样保证候选之间只改变了一个声明的字段，没有偷偷改步数、数据或最大长度？
- LoRA 的秩和缩放是什么关系，为什么本次比较不能单独归因于“参数变多”？
- 评测损失按样本平均还是按词元平均，两种口径差多少？
- 质量、显存、吞吐三个门槛为什么必须同时通过，质量更好能不能抵消显存超限？
- 为什么要用同一种子配对比较损失差，而不是比较两个候选各自的损失中位数？
- 主指标持平时，怎样在等价带内选资源成本更低的配置？

## 解决思路：一步步看懂

**第 1 步：固定共同条件，只留一个自变量。** 项目固定模型版本、数据切分、训练步数、学习率、目标层、dropout 和随机种子集合，只改变 LoRA 秩。notebook 的表格把条件分成五类：秩与缩放（4/8/16，缩放超参数固定 16）、模型与数据（同一基座提交、训练/评测成员、分词、最大长度）、训练预算（相同学习率、样本顺序、更新次数）、适配范围（相同目标层和 dropout）、重复运行（每个候选使用同一组种子）。每份配置都从同一基座重新训练，得到留出验证损失、显存与步时；只有通过检查后才比较资源。这里的“相同预算”是相同样本顺序、每步一条样本、相同更新次数，不能据此推断更长训练后的排名。

**第 2 步：先算清楚缩放系数。** 标准 LoRA 的增量写成 $\Delta W = \frac{\alpha}{r}BA$，其中 $r$ 是秩，$\alpha$ 是缩放超参数，$B$、$A$ 是低秩矩阵。固定 $\alpha=16$ 时：

- 秩 4：$s=16/4=4$；
- 秩 8：$s=16/8=2$；
- 秩 16：$s=16/16=1$。

所以本次实验回答的是“哪份秩配置适合这次训练”，而不是孤立测量“秩本身的能力”。如果想让比较只归因于参数数量，需要另行设计相同缩放或相同参数上限的对照，notebook 把这一点放在前置阅读 31 节里区分。

**第 3 步：定义评测损失口径。** 本项目先对每条样本的有效监督词元取平均损失，再对全部评测样本取平均。设样本 $i$ 的损失总和为 $S_i$、有效目标数为 $N_i>0$，样本总数为 $M$：

$$
L_{\mathrm{example}}=\frac1M\sum_{i=1}^{M}\frac{S_i}{N_i}.
$$

它让每条样本权重相同，和 13/15 节的全词元平均 $\sum_i S_i/\sum_i N_i$ 不同。notebook 的例子：两条回答分别有 2、6 个目标，平均损失为 1、3，则本项目指标为 $(1+3)/2=2$，全词元平均为 $(2\times1+6\times3)/(2+6)=20/8=2.5$。比较候选必须使用同一种计算方法和同一批评测样本，否则损失差没有意义。

**第 4 步：资源指标与通过条件。** 损失和显存是上限，吞吐是下限，每一项都要通过。监督词元吞吐按训练更新计时窗口计算，包含前向、反向和优化器更新，不包含模型加载、评测和适配器保存；峰值分配显存包含训练及训练后的评测，因此不是“纯训练峰值”。fixture 里的门槛是：`maximum_eval_loss=1.30`、`maximum_peak_allocated_bytes=3_000`、`minimum_tokens_per_second=100.0`、`minimum_repeat_count=2`、`maximum_repeat_eval_loss_spread=0.02`、`quality_equivalence_band=0.03`。候选是 `rank4`（loss 1.24，peak 2000，tps 130，step 50）、`rank8`（1.18，2500，120，55）、`rank16`（1.10，3500，110，60）。`rank16` 的损失最好，但峰值 3500 超过 3000，仍要被排除。

**第 5 步：TODO 1——检查候选是否真正可比。** 骨架已经算出 `fixed_mismatches`（固定字段哪些变了）、`changed_variant_fields`（声明可变的字段哪些变了）、`untracked_changes`（两张表都没列、但两边值不同的字段，`name` 除外）。参考答案一行：

```python
comparable = not fixed_mismatches and len(changed_variant_fields) == 1 and not untracked_changes
```

三个条件分别是：固定条件一个都没变；恰好有一个声明的候选字段发生变化；没有未声明的字段变化。为什么必须“恰好一个”？如果同时改 `rank` 和 `alpha`，就无法判断结果由谁造成；如果什么都不改，也不是一个有效候选。`test_candidate_comparability` 用 `rank16 = {**baseline, "rank": 16}` 得到 `comparable=True`、`changed_variant_fields=["rank"]`；把 `steps` 同时改成 20，或同时改 `rank=16` 与 `alpha=32`，都返回不可比。

**第 6 步：TODO 2——逐项检查质量与资源。** 参考答案：

```python
eligible = all(checks.values())
```

`checks` 有四项：`execution`（`status == "ok"`）、`quality`（`final_eval_loss <= maximum_eval_loss`）、`memory`（`peak_allocated_bytes <= maximum_peak_allocated_bytes`）、`throughput`（`supervised_tokens_per_second >= minimum_tokens_per_second`）。四项必须同时成立，不能把“质量更好”当成显存超限的补偿。函数还返回 `failed_checks` 和 `margins`：`margins` 是各项距离门槛的差值，恰好等于门槛时三项余量都是 0。`test_candidate_hard_gates` 断言 `rank16` 的 `failed_checks == ["memory"]`；构造一个三项都正好等于门槛的 `boundary`，断言 `eligible=True` 且所有 `margins` 为 0；把 `status` 改成 `"failed"` 后 `eligible=False`。比较前还会先拒绝 NaN、Inf、负数和零步时。

**第 7 步：TODO 3——判断配对损失差是否落入允许范围。** 参考答案一行：

```python
quality_equivalent = abs(item["paired_delta_median"]) <= gates["quality_equivalence_band"]
```

这里的 `paired_delta_median` 不是“两个候选各自中位数相减”，而是先对同一个种子求差，再取中位数：

$$
d_s = L_c(s) - L_{\mathrm{best}}(s),\qquad \Delta_c = \operatorname{median}_s(d_s).
$$

其中 $L_c(s)$ 是候选 $c$ 在种子 $s$ 下的损失，$L_{\mathrm{best}}(s)$ 是参考候选（重复损失中位数最低者）在同一种子下的损失。只有 $|\Delta_c|$ 不超过允许偏差，候选才进入资源比较；等号计入。notebook 的反例：两组损失为 `[0,0,2]` 和 `[0,1,1]`，逐种子差是 `[0,1,-1]`，中位数是 0；但两个独立中位数分别是 0 和 1，差为 1。如果直接比较中位数，会错误地把两者判成相差 1。`test_paired_differences_and_inclusive_band` 专门断言 `paired_delta_median == 0.0`，并断言 `statistics.median([0,1,1]) - statistics.median([0,0,2]) == 1`。

**第 8 步：完整选型流水线。** `select_lora_candidate` 的顺序是：先对每个候选跑 `evaluate_lora_candidate`，全部不合格返回 `reject` 和 `no_eligible_candidate`；筛出合格候选后，只保留这些候选的重复记录，任一重复状态不是 `ok` 返回 `tune` 和 `repeat_execution_failure`；每个候选的种子不能重复，数量不能少于 `minimum_repeat_count`，否则返回 `duplicate_repeat_seed` 或 `paired_repeat_incomplete`；所有候选的种子集合必须相同，否则也是 `paired_repeat_incomplete`；每条重复记录还要再跑一遍 `evaluate_lora_candidate`，失败项加上 `repeat_` 前缀写回候选的 `failed_checks`；然后检查每个候选的 `repeat_spread = max(losses) - min(losses)` 是否超过 `maximum_repeat_eval_loss_spread`，超过返回 `repeat_instability`；接着以 `repeat_median_loss` 最低者为参考，算各自的配对差中位数，筛出等价候选；最后按 `(peak_allocated_bytes, median_step_time_ms, name)` 从小到大选，资源完全相同时用名称消除顺序差异。

**第 9 步：fixture 里为什么选中 `rank4`。** `rank8` 的重复损失中位数是 $(1.18+1.19)/2=1.185$，`rank4` 是 $(1.20+1.21)/2=1.205$，所以参考是 `rank8`。`rank4` 的逐种子差是 $1.20-1.18=0.02$ 和 $1.21-1.19=0.02$，中位数 0.02，绝对值不超过 `quality_equivalence_band=0.03`，因此质量等价；`rank8` 的差是 0，当然也等价。进入资源比较后，`rank4` 的峰值 2000 低于 `rank8` 的 2500，步时 50 也低于 55，所以选 `rank4`。`test_quality_equivalent_selection` 断言 `decision="accept"`、`selected["name"]=="rank4"`、`shared_seeds==[11,29]`。

**第 10 步：真实实验怎么跑、结果怎么读。** Step 5 使用 `PT-SFT-CUDA` 单卡环境，模型 `Qwen/Qwen2.5-0.5B`、长度 256、三个秩候选、相同更新预算；12 GB 单卡即可。阶段是 `prepare → measure → close`：`prepare` 只核对合同和本地数据清单；`measure` 检查环境、加载模型、按 `CANDIDATES × RUN_SEEDS` 训练并保存适配器和原始扫描 JSON；`close` 重新读取证据，生成报告、选型清单和 SHA256 回执。数据来自 64 节 `accept` 的准入清单，训练和评测样本的文本哈希必须与已审计的 `train_sample_ids`、`evaluation_sample_ids` 对得上，否则 `measure` 直接报错。每个候选的第一个种子保存适配器并绑定路径与哈希，其余重复只保存质量和资源指标；因此适配器对应的是某一个种子，而候选质量摘要来自全部重复。

notebook 保存了一次 `lora_variant_selection_v2` 的归档结果：9 次运行（3 个候选 × 种子 42、123、2024），决策 `accept`，选中 `rank4`。三个候选的重复损失中位数分别是 `rank4=1.753862...`、`rank8=1.748455...`、`rank16=1.751827...`；`rank8` 的损失最低，所以它是配对参考，`rank4` 的 `paired_delta_median=0.004195...`，`rank16` 是 `0.003783...`，都在 0.01 等价带内。资源上 `rank4` 的中位峰值 `1,891,813,376` 字节、中位步时 `35.05 ms`，都低于 `rank8` 的 `1,895,254,016` 字节 / `37.25 ms` 和 `rank16` 的 `1,902,135,296` 字节 / `37.20 ms`，所以选 `rank4`。适配器 SHA256 是 `d21da703...ae407a`。注意：保存输出里的字段名是 `quality_equivalent`，来自旧 v2 协议；当前 v3 的 `select_lora_candidate_for_run` 返回的是 `equivalent_candidates`，并且 5.6 节明确说“新 v3 合同尚无真实采集记录”，不要把旧归档字段当成新代码的返回键。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| LoRA 秩（rank） | 低秩分支的宽度，决定可训练参数数量 |
| 缩放系数（alpha/r） | LoRA 增量前的乘数，固定 alpha 时秩越大会越小 |
| 可比候选 | 固定字段不变、恰好一个声明字段变化、没有未声明变化 |
| 硬门槛（hard gates） | 执行、损失、显存、吞吐四项必须同时通过 |
| 峰值分配显存 | 训练及训练后评测期间分配过的显存峰值 |
| 监督词元吞吐 | 单位时间处理的监督 token 数，越高越好 |
| 配对（pairing） | 同一种子下的两份运行一一对应 |
| 配对损失差中位数 | 先逐种子相减，再取中位数，不是两个中位数相减 |
| 等价带 | 配对差绝对值不超过的允许偏差 |
| 重复波动（spread） | 同一候选多次运行的最大损失减最小损失 |
| 选型清单 / 适配器 | `accept` 后交给下游的配置记录与权重文件 |

## 小白常见疑问

**Q1：`rank16` 损失最低（1.10），为什么不能选它？**
因为显存硬门槛不过。fixture 里 `rank16` 的 `peak_allocated_bytes=3500`，超过 `maximum_peak_allocated_bytes=3000`，`evaluate_lora_candidate` 返回 `failed_checks=["memory"]`。notebook 的设计原则是“质量更好也不能抵消显存超限”，资源是独立的硬约束，不是可以加权的分数。

**Q2：为什么要用配对差，直接比两个候选的中位数不行吗？**
中位数会丢掉“同一种子”的对应关系。反例里 `rank4=[0,0,2]`、`rank8=[0,1,1]`，逐种子差 `[0,1,-1]` 的中位数是 0，说明同种子下互有胜负、整体等价；但两个独立中位数 1 和 0 相差 1，会造成“差距很大”的错觉。`test_paired_differences_and_inclusive_band` 用这个反例把两种算法区分开。

**Q3：等价带取 0.125 和 0.124 为什么结果不同？**
因为边界是“小于等于”。测试构造 `rank4` 损失 1.25、`rank8` 损失 1.125，逐种子差都是 0.125。`band=0.125` 时 `abs(0.125) <= 0.125` 成立，`rank4` 等价，再按资源选 `rank4`；`band=0.124` 时 `rank4` 不等价，只剩 `rank8`，所以选 `rank8`。把候选和重复记录整体反转后，`band=0.125` 仍选 `rank4`，说明结果不依赖输入顺序。

**Q4：候选摘要已经通过门槛，为什么还要检查每一条重复运行？**
因为摘要是汇总值，可能掩盖某一次失败。`test_repeated_measurements_cannot_bypass_gates` 把 `rank4` 的某次重复损失改成 2.0、峰值改成 4000、吞吐改成 1.0，分别触发 `repeat_quality`、`repeat_memory`、`repeat_throughput`，最终都改选 `rank8`。只有每条重复都过门槛，候选才真正合格。

**Q5：为什么 `rank4` 的损失比 `rank8` 略高，仍然可以选它？**
因为质量等价规则允许小差异。归档结果里 `rank4` 的配对差中位数约 `0.0042`，在 0.01 等价带内，说明这次短预算训练下两者质量差别很小；在这个前提下，`rank4` 的显存和步时更低，所以按资源规则选它。这不是说 `rank4` 在所有任务上都更好，结论只适用于当前模型、数据、预算和种子集合。

## 动手验证：怎么知道自己做对了
notebook 在题目区末尾依次调用六个测试，全部通过后打印 `LoRA配置比较、逐次门槛与配对选择测试通过。`
- `test_candidate_comparability`：`rank16` 单变量变化 `comparable=True` 且 `changed_variant_fields=["rank"]`；同时把 `steps` 改成 20，或同时改 `rank=16`、`alpha=32`，都必须不可比。
- `test_candidate_hard_gates`：`rank16` 的 `failed_checks == ["memory"]`；三项都等于门槛的 `boundary` 必须 `eligible=True` 且所有 `margins` 为 0；`status="failed"` 必须不合格。
- `test_quality_equivalent_selection`：fixture 选中 `rank4`、`shared_seeds==[11,29]`；只有失败候选时返回 `reject`；缺 `rank8` 的种子 29 返回 `paired_repeat_incomplete`；某次重复失败返回 `repeat_execution_failure`；重复损失极差 0.10 超过 0.02 返回 `repeat_instability`。
- `test_undeclared_changes_and_input_validation`：`max_length` 从 256 改到 512 必须记入 `untracked_changes` 并判不可比；基线与自身比较（零变化）也不可比；固定字段与候选字段重叠、重复候选名、未声明候选、`seed=True`、四项指标的 NaN/Inf/-1、`minimum_repeat_count=1.5`、`quality_equivalence_band=NaN`、重复损失 NaN，全部必须抛 `ValueError`。
- `test_repeated_measurements_cannot_bypass_gates`：`rank4` 单次损失 2.0 / 峰值 4000 / 吞吐 1.0 分别触发 `repeat_quality` / `repeat_memory` / `repeat_throughput`，最终都选 `rank8`；峰值改成 2800（高于 `rank8` 的 2500）也选 `rank8`；重复种子重复返回 `duplicate_repeat_seed`；种子集合不一致返回 `paired_repeat_incomplete`。
- `test_paired_differences_and_inclusive_band`：`[0,0,2]` 对 `[0,1,1]` 时选中 `rank8` 且 `paired_delta_median=0.0`；非对称例子 `[1,1,11]` 对 `[0,2,10]` 在放宽绝对门槛后选中 `rank4`；边界 0.125 选 `rank4`，0.124 选 `rank8`，反转输入顺序仍选 `rank4`。
notebook 的 CPU 部分没有保存测试输出，以代码和断言为准；Step 5 的归档输出来自旧 v2 协议，运行 `prepare` 只打印配置、不训练，`measure` 需要 `PT-SFT-CUDA` 环境和唯一 `RUN_LABEL`，`close` 需要已有扫描文件。
