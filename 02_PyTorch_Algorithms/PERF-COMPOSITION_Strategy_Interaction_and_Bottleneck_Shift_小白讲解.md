# PERF-COMPOSITION. 策略相互作用与瓶颈转移 — 小白讲解

> 对应 notebook：[`PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb`](./PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb)

## 一句话总结
本节要回答“两个各自有效的优化合在一起，还能不能继续变快”。`validate_order` 先检查组合的顺序、依赖与冲突，`measured_margins` 用两个不同分母算下降比例，`decide_composition` 只在“合格且比基线快”的单项里选回退参考，再判断组合是否值得保留。核心结论是：单项收益不能直接相加，组合必须实测；组合相对“已经较好的单项”的新增收益才是是否值得增加复杂度的依据。

## 为什么需要它：痛点是什么
假设降低计算精度让延迟从 100 ms 降到 80 ms，双缓冲又让延迟从 100 ms 降到 85 ms。直觉上两项一起用应该接近 65 ms，但真实结果可能是 76 ms：两项减少了同一段工作，或者新增了调度、传输和资源争用。更麻烦的是，计算变短后，原本不显眼的固定等待（例如 20 ms 的数据搬运）占比反而上升，瓶颈从“算得慢”转移到“等得久”。

生活类比：上班通勤。原来开车 40 分钟、等红灯 20 分钟，总 60 分钟；换一条更快的路后开车 15 分钟，但红灯还是 20 分钟，总 35 分钟。开车时间下降了 62.5%，总时间只下降 41.7%，因为剩下的 20 分钟等待没有变。两项优化如果都作用在“开车”上，收益会重叠；如果一项减少开车、一项减少等待，才可能接近相加。

初学者常犯的错：把单项百分比相加、只看组合相对基线的收益而忽略“相对最佳单项还剩多少”、以及组合失败时随便回退到一个“看起来最快但其实没通过正确性检查”的单项。

## 核心问题
- 两个策略组合前，要检查哪些依赖、冲突和顺序？
- 为什么“相对基线的下降”和“相对最佳单项的下降”要用不同分母？
- 组合失败或收益不足时，应该回退到哪个配置？
- 为什么“最快”的单项不一定能当回退目标？
- 组合通过后，怎样判断瓶颈是否已经转移？
- 真实 GPU 上，BF16 和双缓冲的组合在两档 token 负载下分别得到什么结论？

## 解决思路：一步步看懂

**第 1 步：组合前先检查能不能一起执行。** `validate_order(order, specs)` 的 `specs` 是“候选名 → `{requires, conflicts}`”的字典。函数先对**所有** spec 做静态检查：`requires`/`conflicts` 必须是字符串集合，不能引用不存在的候选，也不能引用自己；`order` 不能是字符串或 bytes（否则会被当成字符序列误遍历）。然后按顺序逐个加入：

```python
seen, validated = set(), []
requires = set(specs[name].get("requires", set()))
conflicts = set(specs[name].get("conflicts", set()))
reverse_conflicts = {prior for prior in seen if name in specs[prior].get("conflicts", set())}
if not requires <= seen:
    raise ValueError(f"候选 {name} 的依赖尚未满足")
if conflicts & seen or reverse_conflicts:
    raise ValueError(f"候选 {name} 与已有候选冲突")
seen.add(name)
validated.append(name)
```

`requires <= seen` 表示“所有前置依赖都已经出现过”。冲突要看两个方向：当前候选声明的 `conflicts & seen`，以及**已经加入的候选是否反向声明了当前候选**（`reverse_conflicts`）。这样单向声明也有效，不要求两边都填。注意这里只禁止“显式冲突”，普通的带宽/算力竞争不算不兼容；竞争仍允许组合，只是净收益要靠测量。
测试 `test_order_and_asymmetric_conflicts` 用 `_specs()`：`format` 无依赖，`kernel` 依赖 `format`、冲突 `eager_only`，`eager_only` 无声明。`["format","kernel"]` 通过；`["kernel","format"]` 因依赖未满足失败；`["format","kernel","eager_only"]` 因 kernel 直接冲突失败；`["eager_only","format","kernel"]` 因反向冲突失败；重复项和未知项也失败。`test_missing_references_and_cycles` 再检查未知引用、自依赖、`a⇄b` 循环，以及把字符串当 order 传入。
容易写错成只检查当前候选的 `conflicts`，漏掉 `reverse_conflicts`；或者把 `requires <= seen` 写成 `requires == seen`，后者会误伤“依赖已满足但还有别的候选”的情况。

**第 2 步（TODO 2）：用两个分母算边际。** 设同负载基线延迟为 $T_0$，组合延迟为 $T_{AB}$，通过检查且优于基线的最低单项延迟为 $T_*$，则

$$
g_0=\frac{T_0-T_{AB}}{T_0},
\qquad
g_*=\frac{T_* - T_{AB}}{T_*}.
$$

$g_0$ 是“相对基线下降了多少”，分母是 $T_0$；$g_*$ 是“在最佳单项基础上还能再降多少”，分母是 $T_*$。两者不能混用。notebook 的例子：基线 100 ms、可用单项 80 ms、组合 76 ms，则 $g_0=(100-76)/100=24\%$，$g_*=(80-76)/80=5\%$。看起来“降了 24%”，但其中 20 个百分点是单项本来就有的，组合只额外贡献 5%。

参考答案：

```python
best_single_name = min(singles, key=lambda name: (singles[name], name))
best_single_value = singles[best_single_name]
gain_vs_baseline = (baseline - combined) / baseline
gain_vs_best_single = (best_single_value - combined) / best_single_value
```

`min` 的 key 是 `(数值, 名字)`，数值相同时按名字字典序取，保证并列时结果稳定。测试 `test_measured_margins_and_ties` 用 `measured_margins(100, {"a":80,"b":85}, 76)` 断言 `best_single=="a"`、`gain_vs_baseline==0.24`、`gain_vs_best_single==0.05`；再用 `{"b":80,"a":80}` 检查并列时仍返回 `"a"`。
输入校验要求 `baseline`、`combined`、所有单项值都是有限正数，且不能是 `bool`（Python 里 `True` 会被当成 1）。`test_nonfinite_measurements` 用 0、-1、`True`、`nan`、`inf` 逐项验证，并检查 `singles` 为空也报错。
注意 `measured_margins` 只做数值计算，不判断单项质量，也不决定保留谁；这个分工留给下一步。

**第 3 步（TODO 3）：决定保留、用单项还是回退基线。** `decide_composition(baseline, singles, combined, guardrails_passed, minimum_marginal_gain, *, single_guardrails)` 先调用一次 `measured_margins` 做数值校验，再检查 `guardrails_passed` 是布尔、`single_guardrails` 的键与 `singles` 完全一致且值都是布尔、最低边际在 $[0,1)$ 内。
参考答案：

```python
eligible = {name: value for name, value in singles.items()
            if single_guardrails[name] and value < baseline}
best_single = min(eligible, key=lambda name: (eligible[name], name)) if eligible else None
reference_value = eligible[best_single] if best_single else baseline
reference_name = best_single if best_single else "baseline"
gain_vs_baseline = (baseline - combined) / baseline
gain_vs_reference = (reference_value - combined) / reference_value
combination_passed = guardrails_passed and gain_vs_baseline > 0
worthwhile = gain_vs_reference > 0 and gain_vs_reference >= minimum_marginal_gain
```

三层筛选要分清：`eligible` 只保留“通过单项检查**且**严格快于基线”的候选；`best_single` 从合格集合里取最低延迟；如果没有合格项，参考就退回基线，`reference_name="baseline"`。`combination_passed` 要求组合自己的检查通过且严格快于基线；`worthwhile` 要求组合严格快于参考，并且新增收益达到门槛。两者同时为真才 `keep_combination`；否则有合格单项就 `keep_single`，没有就 `rollback_baseline`。返回里的 `rollback_to` 与 `reference` 相同，告诉你失败时该回到哪里。
手算 `test_three_decisions`（基线 100，单项 `a=80,b=85`，门槛 0.05）：
- 组合 72、guard 通过：`gain_vs_reference=(80-72)/80=10%≥5%`，keep_combination。
- 组合 79：`(80-79)/80=1.25%<5%`，keep_single，回退 a。
- 组合 72 但 guard 失败：keep_single，回退 a。
- 单项只有 `a=105`、组合 110、guard 失败：a 比基线还慢，不算 eligible，参考是基线；`gain_vs_baseline=(100-110)/100<0`，rollback_baseline。
`test_ineligible_fastest_single` 进一步检查“最快但不合格”的情况：`fast_bad=50` 检查失败，`safe=80` 检查通过；组合 70、guard 失败时，合格集合只有 safe，所以回退 safe，而不是回退更快的 fast_bad。两个单项都不合格时返回 `best_single=None`、`rollback_to="baseline"`。`test_marginal_equalities` 则验证“等于门槛可以通过”（`5%≥5%`）、“等于参考不保留组合”（增益 0）、“组合等于基线也不保留”。
容易写错成用 `value <= baseline` 把“和基线一样快”的单项也算合格；参考答案用严格 `< baseline`，因为不比基线快的单项没有回退价值。

**第 4 步：组合后重新看瓶颈。** Step 3 提醒：阶段占比由“自身耗时 ÷ 总耗时”决定。计算从 40 ms 缩到 15 ms、固定等待仍是 20 ms 时，等待占比从 `20/60≈33%` 升到 `20/35≈57%`。瓶颈转移不是“某个阶段变快了”这么简单，而是“限制总完成速度的因素换了一个”。判断时要先看各阶段的**绝对耗时**，再结合依赖、重叠和请求竞争，确认下一项改动还能不能继续缩短总时间。
notebook 的 GPU 实验用 `dominant_service_time` 标记组合里计算和 H2D 搬运谁的中位服务时间更长。保存的结果里，32 token 时 `h2d=0.0119 ms`、`compute=0.1816 ms`；256 token 时 `h2d=0.0762 ms`、`compute=0.5403 ms`，两档都标记 `compute`。这不等于“计算占了总时间的这个比例”（两个服务时间不是简单串行相加），而是提示下一步应该沿时间线检查计算依赖，而不是继续优化搬运。

**第 5 步：GPU 组合矩阵怎么设计、怎么读。** 5.1 固定一个 SwiGLU MLP（隐藏维 2048、中间维 5504），每次处理 8 批，token 负载 `[32, 256]`，预热 12 轮、正式 24 轮，四条路径**逐轮轮换顺序**以减少先后偏差：

```python
variants=["baseline_fp32_serial", "bf16_serial", "fp32_overlap", "bf16_overlap"]
```

- `baseline_fp32_serial`：FP32、串行搬运 → 计算。
- `bf16_serial`：只换 BF16。
- `fp32_overlap`：只加双缓冲重叠。
- `bf16_overlap`：两者组合。

对每个 token 负载，先从 `bf16_serial`、`fp32_overlap` 中筛“相似度 ≥ 0.999、P95 相对基线 ≤ 1.05、中位数严格快于基线”的单项，取最低延迟作参考；再算 `gain_vs_baseline` 和 `gain_vs_reference`。组合还要求自己的余弦相似度和 P95 通过。决策规则与 CPU 版一致：`quality_passed and tail_passed and gain_vs_baseline>0 and gain_vs_reference>0 and gain_vs_reference>=0.03` 才 `keep_combination`。
notebook 保存的是旧运行 `rtx5070ti_composition_bf16_overlap_v1`（schema v1）：32 token 时 `best_single=bf16_serial`，`gain_vs_baseline≈0.6847`（68.47%），`gain_vs_best_single≈0.1513`（15.13%），余弦 `0.999967`，决策 `keep_combination`；256 token 时 `gain_vs_baseline≈0.6589`，但 `gain_vs_best_single≈-0.0126`（组合比 BF16 单项还慢 1.26%），决策 `keep_single`，回退 `bf16_serial`。
5.4 讨论的是另一份 `rtx5070ti_composition_rotating_v2` 结果：24 轮轮换、两档 BF16 串行都是最佳单项，组合在 32/256 token 上分别再降约 5.6%/12.2%，24/24 轮都更快，P95 比约 0.73/0.89，两档都保留组合。读结果时要把“保存输出对应的 run_label”和“5.4 讨论的 run_label”分开，不能把两轮数字混着引用；notebook 也说明历史 schema v1 保持原样、新决策分支尚未做 GPU 复测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 策略组合 | 同时使用两个或多个优化手段 |
| 前置依赖 | 后一项必须先有前一项的结果才能执行 |
| 显式冲突 | 两项对同一对象要求互斥状态，任一方声明即不可组合 |
| 资源竞争 | 同时抢带宽、算力或临时空间，但仍可能共存 |
| 边际收益 | 在已有方案上再加一项，还能省多少 |
| $g_0$ / $g_*$ | 相对基线 / 相对最佳单项的下降比例 |
| 合格参考 | 通过检查且严格快于基线的单项 |
| 回退配置 | 组合不适用时使用的单项或基线 |
| 瓶颈转移 | 限制总速度的因素从一段工作换到另一段 |
| 服务时间 | 某个操作自身耗时，不含等待和其他阶段 |
| P95 | 95% 的样本不超过的耗时位置 |
| 逐轮轮换 | 每轮改变四路径执行顺序，抵消先后偏差 |

## 小白常见疑问
**Q1：为什么不直接把两个单项的延迟下降百分比相加？**
因为两项可能减少同一段工作，也可能新增调度或争用。notebook 的例子 100→80→76 说明：单项下降 20%，组合只再降 5%，不是 20%+? 的线性叠加。`measured_margins` 用实测的 `combined` 做分子，就是为了让结果来自测量而不是推算。

**Q2：`eligible` 为什么要排除“等于基线”的单项？**
回退的目标应该是“比基线更好”的配置。如果单项和基线一样快，回退到它没有收益，还会多一层复杂度。参考答案用 `value < baseline`；`test_three_decisions` 里 `a=105` 比 100 慢，就直接落选，最终 `rollback_baseline`。

**Q3：组合只比最佳单项快 3%，门槛也是 3%，算通过吗？**
算。`worthwhile` 用的是 `gain_vs_reference >= minimum_marginal_gain`，`test_marginal_equalities` 专门验证“等于阈值可通过”。但“组合等于参考”（增益 0）不通过，因为 `gain_vs_reference > 0` 这一条先失败。

**Q4：既然 `fast_bad` 只要 50 ms，为什么组合失败时不能回退到它？**
因为它没通过正确性/质量检查。`single_guardrails={"fast_bad": False, "safe": True}` 时，`eligible` 只含 `safe`，所以回退 `safe`。性能优化不能以牺牲正确性为代价，这也是 `decide_composition` 把“检查状态”和“数值大小”分开的理由。

**Q5：`combination_cosine_vs_fp32` 出现 `1.000032`，超过 1 正常吗？**
这是浮点计算和余弦实现的数值误差，不是“相似度超过完美”。门槛只检查下界 `≥0.999`，所以超过 1 的小偏差不影响通过；但如果要做严格的逐元素比较，还需要像 notebook 说的那样补完整双缓冲流水线的输出对照，不能只靠一次独立前向的余弦。

## 动手验证：怎么知道自己做对了
CPU 测试按顺序运行 8 个函数，最后打印 `✅ 组合依赖、实测边际与合格回退测试通过`。关键断言：
- `test_order_and_asymmetric_conflicts`：`["format","kernel"]` 通过；依赖倒置、显式冲突、反向冲突、重复、未知都抛 `ValueError`。
- `test_missing_references_and_cycles`：未知引用、自依赖、`a⇄b` 循环、字符串 order 都报错。
- `test_measured_margins_and_ties`：`(100,{a:80,b:85},76)` 得 `best_single="a"`、`g0=0.24`、`g*=0.05`；并列时取 `"a"`。
- `test_nonfinite_measurements`：0、-1、`True`、`nan`、`inf` 和空 `singles` 都被拒绝。
- `test_three_decisions`：72→keep_combination；79→keep_single；guard 失败→keep_single；单项 105、组合 110→rollback_baseline。
- `test_ineligible_fastest_single`：最快但不合格的单项不能当回退；全部不合格时回退基线；组合 guard 通过且相对基线有 30% 收益时，即使没有合格单项也可 keep_combination。
- `test_marginal_equalities`：恰好等于 5% 门槛通过；组合等于参考、等于基线都不保留。
- `test_guardrail_and_threshold_inputs`：最低边际为负、≥1、`True`、`nan`、`inf` 都报错；`single_guardrails` 缺键、多键、非布尔都报错。

GPU 单元格默认 `RUN_MODE="dry_run"`，只打印配置，不导入 CUDA 运行时；改为 `real_gpu` 前要先通过 `PERF-CUDA-MECHANISM` 预检，且 `RUN_LABEL` 必须是不含目录分隔符的唯一值，结果文件已存在会 `FileExistsError`。保存输出是 schema v1 的历史运行，5.4 讨论的是 v2 轮换运行；读 5.3 时以 `read_composition_result` 重算的 `comparisons` 为准，不要手抄旧百分比。
