# PERF-COMPOSITION. Strategy Interaction and Bottleneck Shift | 策略相互作用与瓶颈转移 · 小白入门

> 配套 notebook：[PERF-COMPOSITION. 策略相互作用与瓶颈转移](../PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

两个各自有效的提速办法，合在一起不一定更好：可能互相冲突，也可能抢同一种资源。这本 notebook 教你先检查能不能组合，再实测比较，决定保留组合还是退回单个办法。

## 为什么需要它

你手里有两个各自有效的提速办法。比如 CPU 预取（提前把下一批数据准备好）和异步 H2D 复制（host to device，把数据从主机内存拷贝到显存，同时不让计算干等），各自都可能减少等待。直觉是一起打开、收益叠加，但它们也可能争用带宽和缓冲空间。只靠直觉会出三种错：

- 不检查就叠加：格式转换还没做完，下游的计算内核就去读它产出的布局。
- 把单项收益相加：基线延迟是 100，单项 A 降到 80，单项 B 降到 85，就以为一起用能把两份收益加起来。
- 失败时退回最快的那个：它最快，但可能没通过正确性、质量或资源检查。

这本 notebook 先按候选之间声明的依赖和冲突检查能不能组合，再在同一个工作负载上实测基线、单项和组合，看组合比最佳合格单项又快了多少，最后保留组合或回退到合格的方案。它不自动推断新的瓶颈。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 基线 / 单项 / 组合（baseline / single / combination） | 基线是不做改动的做法，单项是只启用一个改动，组合是把几个改动一起启用 |
| `requires` | 前置依赖：这个候选启用之前，名单里的候选必须已经启用（测试里 `kernel` 依赖 `format`） |
| `conflicts` | 明确不兼容：这个候选和名单里的候选不能同时启用，任一方声明就算数 |
| 资源竞争 | 几个候选同时使用带宽、计算资源或临时空间。它不是禁令：组合保留，靠测量看争用有多大 |
| 合格单项（`eligible`） | 通过了正确性、质量与资源检查，并且比基线更快的单项。只有它能当回退方案 |
| 边际收益（marginal gain） | 组合比最佳合格单项又多降低的比例，用来衡量多出来的复杂性值不值 |
| 瓶颈转移（bottleneck shift） | 一个阶段变短之后，限制整体完成时间的环节换成了别的阶段 |

## 核心思路

输入是候选规格（每个候选的 `requires` 和 `conflicts` 名单）和组合顺序、同一工作负载下实测的延迟（基线、各单项、组合，越低越好）、各项有没有通过检查、最低边际门槛。程序分三步：验证顺序、算两个下降比例、选回退参考并做决定，输出 `keep_combination`、`keep_single` 或 `rollback_baseline`。

第一步只认声明。候选加入之前，它 `requires` 里的候选必须已经出现；依赖成环、名字重复或不存在都要拒绝。冲突要看两个方向：它自己声明的 `conflicts`，以及之前已启用候选声明的 `conflicts` 里有没有它。资源竞争不在这步里，要靠测量。第二步用两个式子：

```text
g_0 = (T_0 - T_AB) / T_0      组合相对基线的下降比例
g_* = (T_* - T_AB) / T_*      组合相对最佳合格单项的下降比例，也就是边际收益
```

`T_0` 是基线延迟，`T_AB` 是组合的实测延迟，`T_*` 是合格单项里最低的延迟。两个式子分母不同，不能混用，也不能把两个单项的下降比例相加。看小例子：

```python
T0, singles, T_ab = 100, {"a": 80, "b": 85}, 76     # 同一工作负载下的延迟，越低越好

best = min(singles, key=singles.get)                # 实测值最低的单项
T_star = singles[best]
g0 = (T0 - T_ab) / T0                               # 组合相对基线
g_star = (T_star - T_ab) / T_star                   # 组合相对最佳单项
each = [round((T0 - t) / T0, 2) for t in singles.values()]   # 两个单项各自的下降比例
print(best, g0, g_star)
print(each, each[0] + each[1])


def pick_reference(singles, passed):
    eligible = {n: t for n, t in singles.items() if passed[n] and t < T0}
    if not eligible:
        return "baseline", T0                       # 没有合格单项，用基线
    name = min(eligible, key=eligible.get)
    return name, eligible[name]


print(pick_reference(singles, {"a": True, "b": True}))
print(pick_reference(singles, {"a": False, "b": True}))
print(pick_reference(singles, {"a": False, "b": False}))
```

```text
a 0.24 0.05
[0.2, 0.15] 0.35
('a', 80)
('b', 85)
('baseline', 100)
```

前两行是边际收益：两个单项各自只降了 0.2 和 0.15，相加是 0.35，组合实测只降了 0.24，所以组合收益不能由单项收益相加得到；换成相对最佳单项，组合只再多降 0.05，这才是判断“多一层组合值不值”的依据。

后三行是回退参考（reference，组合不保留时改用的方案）的选择。`a` 最快，但没通过检查就不能用，改选合格的 `b`；一个合格单项都没有时用基线。第三步按下面的规则决定，参考就是上面选出的回退参考：

```text
组合通过检查，比基线快，边际收益大于 0 且不低于最低门槛
                                      → keep_combination：保留组合，同时保存回退方案
其余情况，有合格单项                  → keep_single：改用合格单项
其余情况，没有合格单项                → rollback_baseline：恢复基线
```

组合和参考一样快时边际收益是 0，这时不保留组合，即使门槛设成 0 也一样。组合生效后，某个阶段的时间占比升高可能只是别的阶段变短了，不能直接认定它是新瓶颈，确认要靠时间线或受控改动；CPU 练习不推断瓶颈。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 哪些依赖与冲突会改变组合结果：前置依赖、明确不兼容、资源竞争、顺序敏感四种关系怎样识别、怎样处理 | 读 |
| Step 2 | 怎样计算组合相对最佳单项的新增收益：两个分母不同的下降比例，以及四组对照各自回答什么问题 | 读，对照上面的两个式子 |
| Step 3 | 组合生效后怎样重新判断下一步：四种比较结果对应的动作，以及为什么最长的阶段不等于新瓶颈 | 读，对照上面的规则 |
| Step 4 · TODO 1 | 验证组合顺序 | 在 `validate_order` 里写 4 行：`seen` 和 `validated` 两个累加器，循环里的 `requires`、`conflicts`、`reverse_conflicts` |
| Step 4 · TODO 2 | 计算实测边际 | 在 `measured_margins` 里写 4 行：选出实测值最低的单项，再分别算对基线、对最佳单项的下降比例 |
| Step 4 · TODO 3 | 选择保留或回退 | 在 `decide_composition` 里写 8 个赋值：筛出合格单项，选回退参考，算两个比例，最后得到 `combination_passed` 和 `worthwhile` |
| 测试 | 检查顺序是否合法、两个比例、非法测量值、三种决策，以及不合格的最快单项不能成为回退目标 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上同一次运行里测量 FP32 串行、BF16 串行、FP32 双缓冲和 BF16 双缓冲四条路径，算组合相对基线和相对合格参考的下降 | 有 GPU 时再做，默认是 `RUN_MODE = "dry_run"` |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `PERF-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，这时 5.2 不导入 GPU 运行时；改成 `real_gpu` 才执行真实组合矩阵。

## 容易踩的坑

- **冲突只查了一半**：除了当前候选声明的 `conflicts`，还要看之前已启用候选声明的 `conflicts` 里有没有它。任一方声明不兼容就拒绝，不能指望两边都登记。
- **把资源竞争当成禁止共存**：同时使用带宽、计算资源或临时空间只是可能争用，合法的组合要保留，靠测量看争用和峰值。
- **两个比例混用，或者把单项收益相加**：对基线的分母是 `T_0`，对最佳单项的分母是 `T_*`。组合收益只能来自组合的实测值，测量值还必须是有限的正数。
- **回退到最快但不合格的单项**：先按检查状态和“比基线快”筛出合格单项，再选最低的；一个都没有就回到基线。
- **把最长的阶段当成新瓶颈**：占比升高可能只是别的阶段变短了，确认新瓶颈要靠时间线或受控改动。

## 读完后自查

- [ ] 能说出前置依赖、明确不兼容、资源竞争三者的区别和处理方式。
- [ ] 基线 100、最佳合格单项 80、组合 76，能算出两个下降比例。
- [ ] 最快的单项没通过检查时，能说出回退参考选谁。
- [ ] 能说出组合和参考一样快时，为什么不保留组合。

## 下一步

- 打开 notebook：[PERF-COMPOSITION. 策略相互作用与瓶颈转移](../PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md)
- 建议先读：[PERF-CANDIDATE. 从资源约束到候选改动与新增代价 · 小白入门](./PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md)、[PERF-REGRESSION. 配对基准与性能回归阈值 · 小白入门](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md)、[PERF-TRACE. 执行轨迹依赖与关键路径 · 小白入门](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.md)
- 专题页：[性能优化 · 07 端到端性能工程与决策](../../topic_discussion/performance_optimization/07_end_to_end_performance_engineering.md)
- 做完之后：[DISTILL-OPD. 在策略蒸馏与 GKD · 小白入门](./DISTILL-OPD_On_Policy_Distillation_and_GKD.md)
