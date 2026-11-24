# PERF-REGRESSION. Paired Benchmark and Regression Thresholds Beginner | 配对基准与性能回归阈值 · 小白入门

> 配套 notebook：[PERF-REGRESSION. 配对基准与性能回归阈值](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

判断一版改动有没有让程序变慢，不能各跑一次就比时间。这本 notebook 把新旧两版在同一轮的运行配成一对，从每对延迟比例估出范围，再按事先定好的“允许慢多少”给出结论。

## 为什么需要它

你改了一版实现，想知道它是快了还是慢了。最直接的做法是旧版跑一次、新版跑一次，比时间。但一次运行更快，可能只是因为运行顺序、机器温度或后台负载碰巧不一样，和你的改动无关。

notebook 测试里有一个具体的场景：基线跑了 3 轮，延迟是 10、11、9 毫秒；候选只有 2 轮有记录，其中 1 轮执行失败。真正能拿来比较的只剩 1 对。如果只看剩下的成功样本，或者把失败的那轮当作没发生过，结论就不可靠。

几种常见做法会出问题：

- 先后顺序固定。基线永远先跑、候选永远后跑，顺序带来的偏差就一直偏向一边。
- 失败和缺失不记账。失败的那轮被悄悄丢掉，剩下的样本里候选更快，就直接采用。
- 只报一个中位比例，不说这个估计本身波动多大。
- 边界事后定。看到结果以后，再决定慢多少才算变慢。

这本 notebook 把规则定清楚：同一轮的新旧运行配成一对，失败和缺失单独登记；从逐对延迟比例估出一个范围；再按事先设定的允许变慢幅度，给出 `accept`、`tune` 或 `reject`。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 基线 / 候选（baseline / candidate） | 基线是原来的做法，候选是改动后的做法 |
| 轮次 / 试验编号（`trial_id`） | 同一轮里基线和候选各跑一次，编号相同的两条记录配成一对；一侧没记录叫缺失，任一侧报错叫失败 |
| 工作负载签名（`workload_signature`） | 一段文字，记下输入、形状、模型和环境等固定条件；签名不同的运行不能混在一起比 |
| 延迟比例 / 中位比例 / 变慢占比 | 逐对用候选延迟除以基线延迟得到延迟比例；把它们排好取中间一个是中位比例；延迟比例大于 1 的配对所占的比例是变慢占比 |
| 自助重采样（bootstrap）/ 90% 自助区间 | 从已有比例里有放回地再抽同样多个、算一个中位数，重复多次后取 5% 和 95% 分位，得到区间 `[L, U]` |
| 回归边界（`regression_boundary`） | 事先同意候选最多慢多少。允许慢 5% 时，边界是 1 + 0.05 = 1.05 |
| `accept` / `tune` / `reject` | 三种结论：通过回归检查、需要补齐或复测、拒绝 |

## 核心思路

输入是基线和候选各自的运行记录，每条记录有试验编号、工作负载签名、延迟和执行状态。TODO 1 按编号把两侧配成对，分成成功、缺失、失败三类。TODO 2 对成功的配对逐对求延迟比例，算出中位比例和变慢占比，再用重采样得到区间。TODO 3 先查正确性、失败和缺失，再把区间和回归边界比较，给出结论。输出是三种结论之一，以及理由。

```text
r_i = c_i / b_i                           # 第 i 对的延迟比例，大于 1 表示这一轮候选更慢
中位比例 = median(r_1, ..., r_N)           # 变慢占比 s = (r_i > 1 的对数) / N
[L, U] = 重采样中位数的 5% 和 95% 分位      # 每次从 N 个比例里有放回抽 N 个，求中位数，重复 K 次
tau = 1 + delta                           # 回归边界，delta 是事先允许的相对延迟增加量

有执行失败或正确性未通过      → reject
有缺失配对或有效配对数不足    → tune
L > tau                     → reject
U <= tau                    → accept
其他位置，区间跨过边界        → tune
```

`N` 是成功配对数，`b_i` 和 `c_i` 是第 `i` 对的基线延迟和候选延迟，`K` 是重采样次数，`tau` 是回归边界，`L` 和 `U` 是区间的下界和上界。分位数按排序后最接近的位置取值。

下面用测试里的 3 对延迟走一遍统计和最后一步判断。数字很小，可以心算。

```python
from random import Random
from statistics import median

pairs = [(10.0, 9.0), (10.0, 10.0), (10.0, 11.0)]   # 每对是（基线延迟, 候选延迟）
ratios = [c / b for b, c in pairs]
print(ratios, median(ratios), sum(r > 1 for r in ratios) / len(ratios))

b, c = [1.0, 10.0], [2.0, 10.0]                     # 换两组延迟做对比
print(median([y / x for x, y in zip(b, c)]), median(c) / median(b))

def quantile(values, p):                            # 排序后取最接近的位置
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * p))]

rng = Random(7)                                     # 固定种子，结果可复现
resampled = [median(rng.choices(ratios, k=len(ratios))) for _ in range(200)]
print(quantile(resampled, 0.05), quantile(resampled, 0.95))

def decide(L, U, delta):                            # 只看区间和边界这一步
    tau = 1 + delta
    return "reject" if L > tau else "accept" if U <= tau else "tune"

print(decide(0.9, 1.1, 0.05), decide(1.08, 1.10, 0.05), decide(0.9, 1.05, 0.05))
```

```text
[0.9, 1.0, 1.1] 1.0 0.3333333333333333
1.5 1.0909090909090908
0.9 1.1
tune reject accept
```

3 对延迟的比例是 0.9、1.0、1.1，中位比例是 1.0，变慢占比约 0.33，也就是 3 对里有 1 对变慢。换两组延迟 `[1, 10]` 和 `[2, 10]`：逐对比例的中位数是 1.5，两组延迟中位数之比却只有约 1.09，所以这两个量不能混用。接着做 200 次重采样，每次有放回地抽 3 个比例、取中位数，再取这 200 个中位数的 5% 和 95% 分位，得到区间 `[0.9, 1.1]`。种子固定，所以每次运行的结果相同。

最后一步拿区间和边界比较。允许慢 5% 时边界是 1.05：`[0.9, 1.1]` 跨过了边界，结论是 `tune`，要固定条件复测；`[1.08, 1.10]` 整体超过边界，结论是 `reject`；`[0.9, 1.05]` 的上界正好等于边界，结论是 `accept`。上面的小函数只演示了最后这一步，实际判断要先查正确性和执行失败，再查缺失和配对数量。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 哪些运行可以配成一对：试验编号、工作负载签名、延迟、执行状态、计划与完成轮数各自的要求 | 读 |
| Step 2 | 怎样从逐对延迟得到中位比例区间：延迟比例、中位数、变慢占比和自助重采样 | 读，对照上面的小例子 |
| Step 3 | 区间相对允许回归边界怎样决定结果：按顺序判断，给出 `reject`、`tune` 或 `accept` | 读 |
| Step 4 · TODO 1 | 建立配对运行 | 在 `pair_benchmark_runs` 里写 3 行：`missing_baseline`、`missing_candidate`、`failed_pair` |
| Step 4 · TODO 2 | 汇总比例分布 | 在 `summarize_paired_ratios` 里写 4 行：`ratios`、`bootstrap_medians`、`ci_low`、`ci_high` |
| Step 4 · TODO 3 | 判断性能回归 | 在 `decide_performance_regression` 里写 4 行：`quality_failed`、`insufficient_pairs`、`confirmed_regression`、`confirmed_safe` |
| 测试 | 检查乱序配对、缺失和失败记录、重复编号与签名冲突、比例与固定种子复现、三种决策与边界等号 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在同一个微型多层感知机、相同权重与输入上，交替运行普通执行和 `torch.compile`，做配对复测 | 需要单张 CUDA GPU 和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，改成 `real_gpu` 后才执行真实配对复测 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，只检查配置，不执行真实配对复测。

## 容易踩的坑

- **把失败或缺失的轮次悄悄丢掉**：重复编号要报错，不能用字典覆盖；失败的轮次要保留编号和原因。当前规则对执行失败零容忍，有失败就拒绝，不能只看剩下的成功样本。
- **只给决策函数传成功样本的区间**：失败数和缺失数要单独传给 `decide_performance_regression`。没有成功配对时直接报告无法统计，不把空数据当成没有回归。
- **统计对象弄错**：先逐对求比例，再取比例的中位数，不是两组延迟各取中位数再相除。重采样抽的是比例，每次抽的个数等于原配对数，取的也是中位数而不是平均值。
- **以为固定种子能消除偏差**：固定种子只让计算可重复，不会消除持续漂移和轮次相关的影响。增加重采样次数，也不能代替新增真实配对。
- **把通过当成加速**：区间上界不超过边界，只说明没有确认超过允许的变慢幅度，候选仍可能略慢，不能写成“已证明加速”。

## 读完后自查

- [ ] 能说出同一轮的基线和候选要满足哪些条件才能配成一对。
- [ ] 3 对延迟是 10/9、10/10、10/11 时，能算出延迟比例、中位比例和变慢占比。
- [ ] 区间是 `[1.08, 1.10]`、边界是 1.05 时，能说出结论和理由。
- [ ] 能解释为什么通过回归检查不等于证明加速。

## 下一步

- 打开 notebook：[PERF-REGRESSION. 配对基准与性能回归阈值](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md)
- 需要补基础：[Part 00 · 20 性能证据与实验设计](../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md)
- 想读深一点：[性能分析 · 02 可重复基线](../topic_discussion/performance_analysis/02_reproducible_baseline.md)、[性能分析 · 回归、噪声与阈值](../topic_discussion/performance_analysis/06_benchmark_design_and_regression_validation.md)
- 做完之后：[PERF-MEMORY-TIMELINE. 显存驻留与峰值重叠 · 小白入门](./PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap_Beginner.md)
