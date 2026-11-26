# 31. LoRA 变体缩放与参数预算 — 小白讲解

> 对应 notebook：[`31_LoRA_Variants_Theory.ipynb`](./31_LoRA_Variants_Theory.ipynb)

## 一句话总结
同样把秩设为 8，LoRA、rsLoRA、DoRA 的可训练参数量、增量缩放和有效权重构造并不相同。本节用 `count_adapter_parameters`、`build_lora_variant_spec`、`max_rank_for_budget` 三个函数建立“参数账本”，先在相同秩下比较缩放，再在相同参数上限下反推各方案能用的最大秩；最后用可选的 PEFT/GPU 实验核对理论值与真实适配器。

## 为什么需要它：痛点是什么
初学者容易把“秩相同”当成“实验条件相同”：都设 `r=8`，就直接比较 LoRA 与 rsLoRA 的训练效果。但两者的增量系数一个是 $\alpha/r$，一个是 $\alpha/\sqrt r$，更新幅度根本不一样；DoRA 还会为每个输出通道多存一个幅值参数。若只看方案名字，看到的差异可能来自缩放、来自新增参数，也可能来自归一化带来的额外计算，无法归因。

生活类比：三支施工队都派 8 个人（秩相同），但包工头给的“放大系数”不同；第三支队伍还额外给每层楼配了一个音量旋钮（幅值参数）。比较工期时如果不把人数、系数、旋钮分开记账，结论就没有意义。

## 核心问题
- LoRA、rsLoRA、DoRA 分别改变了什么？哪些量相同，哪些量不同？
- 给定若干目标层的输入/输出维度，LoRA 与 DoRA 各需要多少可训练参数？
- 给定参数上限 $P$，怎么反推最大可用秩 $r_{\max}$？为什么 DoRA 有时只能取更小的秩？
- “相同秩”和“相同参数上限”两种比较分别能回答什么问题？
- 参数量接近，为什么峰值显存和更新时间仍可能差很多？

## 解决思路：一步步看懂

**第 1 步：TODO 1——按模块累计适配器参数。** 对一个输入维 $d_{\mathrm{in}}$、输出维 $d_{\mathrm{out}}$ 的线性层，秩为 $r$ 时低秩旁路有两组矩阵（降维 $A$ 与升维 $B$），参数量是 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$；DoRA 再为每个输出通道保存一个幅值参数，多出 $d_{\mathrm{out}}$ 个。对目标模块集合 $T$：

$$
N_{\mathrm{LoRA}}(T,r)=r\sum_{\ell\in T}(d_{\mathrm{in},\ell}+d_{\mathrm{out},\ell}),\qquad
N_{\mathrm{DoRA}}(T,r)=N_{\mathrm{LoRA}}(T,r)+\sum_{\ell\in T}d_{\mathrm{out},\ell}.
$$

参考答案就一行：

```python
module_params = rank * (in_features + out_features) + (out_features if use_dora else 0)
```

测试形状是 `{"q_proj": (16, 16), "v_proj": (16, 8)}`。每增加一阶需要的参数是 $(16+16)+(16+8)=56$；DoRA 的固定幅值参数是 $16+8=24$。所以 `r=4` 时 LoRA 是 $4\times56=224$，DoRA 是 $224+24=248$。容易写错的是把 `d_in` 与 `d_out` 相加写成相乘、忘记乘秩、或者把非目标模块也算进去；骨架会拒绝未知模块（`KeyError`）、重复模块、空目标、非正整数维度和非整数秩。

**第 2 步：TODO 2——生成统一规格并计算缩放。** LoRA 的增量系数是

$$
s_{\mathrm{LoRA}}=\frac{\alpha}{r},\qquad s_{\mathrm{rsLoRA}}=\frac{\alpha}{\sqrt r}.
$$

$\alpha>0$ 是缩放超参数，$r$ 是秩。`r=4, alpha=8` 时 LoRA 为 $8/4=2$，rsLoRA 为 $8/2=4$——即使低秩矩阵完全一样，乘到增量上的系数也不同。参考答案：

```python
scaling = alpha / math.sqrt(rank) if variant_type == "rslora" else alpha / rank
trainable_params = count_adapter_parameters(
    module_shapes, target_modules, rank, use_dora=variant_type == "dora"
)
```

注意 DoRA 使用标准 LoRA 缩放（代码走 `else` 分支），它的特殊之处在幅值参数和有效权重，而不是缩放公式。规格函数会拒绝未知变体名、空 `target_modules`、小数/布尔秩、非有限或非正 `alpha`；测试还专门传 `rank=1.5/True/"4"`、`alpha=nan/inf/0` 来检查不会静默截断。

**第 3 步：TODO 3——在参数上限内反推最大秩。** 先扣掉 DoRA 的固定幅值参数，再看剩余预算能装几阶：

$$
r_{\max}=\left\lfloor
\frac{P-\delta_{\mathrm{DoRA}}\sum_{\ell\in T}d_{\mathrm{out},\ell}}
{\sum_{\ell\in T}(d_{\mathrm{in},\ell}+d_{\mathrm{out},\ell})}
\right\rfloor.
$$

其中 $P$ 是参数上限，$\delta_{\mathrm{DoRA}}\in\{0,1\}$ 表示是否启用 DoRA，$\lfloor\cdot\rfloor$ 是向下取整。参考答案：

```python
rank_unit = sum(sum(module_shapes[name]) for name in target_modules)
fixed_params = sum(module_shapes[name][1] for name in target_modules) if use_dora else 0
max_rank = (max_trainable_params - fixed_params) // rank_unit
```

用同一组形状手算：`rank_unit=56`，DoRA 的 `fixed_params=24`。上限 248 时，LoRA 可到 $248//56=4$（用 224，余 24），DoRA 可到 $(248-24)//56=4$（正好用 248）；上限 224 时，LoRA 仍是 4，DoRA 只有 $(224-24)//56=3$。若结果小于 1，返回 `{"max_rank": 0, "trainable_params": 0, "decision": "reject"}`，表示连秩 1 都装不下。测试 `test_insufficient_budget` 用 `q_proj` 的 `(16,16)`（每阶 32 个参数）配预算 31，正好触发拒绝。

**第 4 步：DoRA 到底改了什么。** 记冻结基座为 $W$、低秩矩阵为 $A,B$、逐输出通道幅值为 $m$。DoRA 先把“方向”和“长度”拆开：把 $W+sBA$ 的每一行除以它自己的二范数得到单位方向，再用 $m$ 给每个输出通道重新定长度：

$$
W'=m\odot\frac{W+sBA}{\lVert W+sBA\rVert_{2,\mathrm{row}}}.
$$

例如某行权重是 $(3,4)$，长度 $\sqrt{3^2+4^2}=5$，归一化方向是 $(0.6,0.8)$；把幅值改成 10，权重变成 $(6,8)$，方向不变。$\odot$ 是逐输出通道缩放，分母按行算二范数。方向归一化和幅值学习都要额外计算，所以“参数上限相同”只控制参数预算，不能直接保证峰值显存或更新时间相同。

**第 5 步：两种比较条件怎么设计。** notebook 明确区分：相同秩控制“低秩空间宽度一致”，用来观察缩放或幅值参数化带来的差异；相同参数上限控制“总可训练参数不超过同一上限”，用来观察各方案在有限容量内实际能取多大秩、还剩多少预算。除此之外还要单独核对初始化与合并的数值一致性，以及任务质量——这几件事不能混在一次比较里。

**第 6 步：PEFT 框架核对。** `check_peft_lora_variant_conformance` 用一个微型 `LlamaConfig`（`hidden_size=16`、1 层、2 个注意力头）注入 `r=4, lora_alpha=8` 的 LoRA、rsLoRA、DoRA，读取 `q_proj.scaling["default"]` 与 `requires_grad=True` 的参数总数。检查项包括 `lora_scaling_match`、`rslora_scaling_match`、`lora_parameter_count_match`、`dora_parameter_count_match`、`budget_aligned_framework_count_match` 和 `dora_magnitude_present`（确认存在 `lora_magnitude_vector`）。这段核对依赖 PEFT/Transformers，导入失败时返回 `{"status": "skipped"}`，不会假装通过。

**第 7 步：可选的 GPU 实验与结果读法。** 固定 `Qwen/Qwen2.5-0.5B`、目标 `q_proj + v_proj`、`rank=8`、`alpha=16`、5 次更新、学习率 `1e-3`，先跑固定秩的三个候选，再用 LoRA rank 8 的参数量作上限反推 DoRA 秩。保存的输出显示：命中 48 个线性模块，`rank_unit_params=67584`，DoRA 固定参数 24576，`parameter_budget=540672`，反推 `budget_aligned_dora_rank=7`；`lora_r8` 的 `actual_scaling=2.0`、`trainable_params=540672`、`adapter_bytes=2162688`、中位更新约 34.26 ms、峰值分配约 2.11 GB、合并余弦 0.999679（这些是 v2 JSON 的候选值；5.4 的 v3 文字另给出 38.3–50.8 ms 的区间）。5.4 的文字还给出关键结论：`dora_r7` 的合并余弦只有 0.99839，低于 `MIN_OUTPUT_COSINE=0.999`，因此该候选是 `tune_candidate`，整体决策为 `tune`。读结果时先看每个候选的 `failed_checks`，不要只看整体 `decision`；同时注意 dump 里 JSON 的 `run_label` 是 `rtx5070ti_lora_variants_v2`，而 5.4 文字引用的是 v3，跨标签的数字不能直接拼接。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 秩 $r$ | 低秩旁路的中间宽度，决定 A/B 矩阵有多“窄” |
| 缩放 $s$ | 乘在低秩增量前的系数，LoRA 是 $\alpha/r$，rsLoRA 是 $\alpha/\sqrt r$ |
| $\alpha$ | 缩放超参数，和秩一起决定增量幅度 |
| rsLoRA | 换一种“随秩缩小”的速度，矩阵形状不变 |
| DoRA | 把权重拆成方向与幅值，用低秩改方向、单独学幅值 |
| 幅值参数 $m$ | 每个输出通道一个可训练长度，DoRA 额外占用预算 |
| 参数账本 | 按目标模块逐层累计可训练参数 |
| 参数上限 $P$ | 允许的最大可训练参数总量 |
| $r_{\max}$ | 在上限内能取到的最大整数秩 |
| 固定秩比较 | 控制低秩空间宽度，观察缩放/参数化差异 |
| 固定预算比较 | 控制总参数量，观察各方案实际能用的秩 |
| 合并（merge） | 把低秩增量并回基座权重，部署时不再保留旁路 |

## 小白常见疑问
**Q1：rsLoRA 的参数量和 LoRA 一样吗？**
一样。`count_adapter_parameters` 只看 `rank` 和模块形状，不看变体；rsLoRA 改的只是 `scaling`。测试 `test_parameter_budget_and_scaling` 同时断言 LoRA 的 `scaling=2.0`、rsLoRA 的 `scaling=4.0`，而参数量公式共用。

**Q2：为什么 DoRA 可能让可用秩从 4 降到 3？**
因为预算要先扣掉每个输出通道的幅值参数。`q_proj (16,16)` 加 `v_proj (16,8)` 的固定幅值一共 24 个；预算 224 时，LoRA 还能用 $224/56=4$，DoRA 只剩 $(224-24)/56=3.57$，向下取整为 3。`test_budget_aligned_rank` 直接断言 `tighter_dora["max_rank"] == 3`。

**Q3：参数上限相同，为什么不能保证显存和速度相同？**
DoRA 多了方向归一化和幅值学习的前向/反向计算，参数只是账本的一部分；notebook 的原话是“相同参数上限只能控制参数预算，不能直接控制峰值显存或更新时间”。GPU 结果里也能看到 DoRA 候选的更新时间更长。

**Q4：`max_rank_for_budget` 返回的秩一定“刚好装下”吗？**
不一定刚好用满，但一定不会再高一阶。`test_budget_maximality` 对预算 1、55、56、79、80、224、248、1000 逐一检查：当前秩的参数量不超过预算，而 `rank+1` 一定超过预算；若连秩 1 都装不下，则返回 `reject`。

**Q5：为什么 `trainable_params` 只统计目标模块，不统计整个模型？**
因为 LoRA/DoRA 只训练注入到目标模块的适配器，基座参数冻结。PEFT 核对里也是用 `requires_grad=True` 的参数总数来对齐这个口径，`base_parameters_frozen` 在 GPU 实验里是单独的检查项。

## 动手验证：怎么知道自己做对了
所有测试在 CPU 上直接运行，全部通过后打印 `PASSED: LoRA变体缩放、参数预算及最大可用秩核对通过。`。
- `test_parameter_budget_and_scaling`：LoRA 224、DoRA 248；LoRA `scaling=2.0`、rsLoRA `scaling=4.0`。
- `test_budget_aligned_rank`：预算 248 时 LoRA 与 DoRA 都是秩 4（分别用 224、248）；预算 224 时 DoRA 降到秩 3，且三个 `trainable_params` 都不超过各自预算。
- `test_invalid_variant_inputs`：`rank=0`、未知 `k_proj`、重复 `q_proj`、`(16,0)` 维度、未知变体名、空 `target_modules` 分别抛 `ValueError` 或 `KeyError`。
- `test_insufficient_budget`：`(16,16)` 单模块、预算 31 返回 `max_rank=0, trainable_params=0, decision='reject'`。
- `test_integer_and_finite_configuration`：`1.5/True/"4"` 的秩、`nan/inf/0` 的 alpha、`31.5/True/0` 的预算、2.5 的维度全部被拒绝。
- `test_budget_maximality`：对多组预算验证“当前秩装得下、再加一阶必超限”，这是对取整逻辑最强的检查。

PEFT 核对与 GPU 实验是可选项。PEFT 单元要求环境里装有 `peft` 与 `transformers`，否则打印 `skipped`；GPU 单元默认 `RUN_MODE="dry_run"`，只保存计划 JSON，不会真正训练。
