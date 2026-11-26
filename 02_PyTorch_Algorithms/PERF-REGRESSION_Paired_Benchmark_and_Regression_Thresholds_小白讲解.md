# PERF-REGRESSION. 配对基准与性能回归阈值 — 小白讲解

> 对应 notebook：[`PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb`](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb)

## 一句话总结
本节用 `pair_benchmark_runs`、`summarize_paired_ratios`、`decide_performance_regression` 三个 TODO，把"同一轮基线与候选配成一对 → 算逐对延迟比例 → 自助重采样求中位比例的 90% 区间 → 和允许回归边界比较"固化成一套可测试流程。核心结论：区间上界不超过 `1 + 允许回归量` 只说明"没有超过允许的变慢"，既不等于加速，也不能拿两组各自的中位数之比来代替配对比例。

## 为什么需要它：痛点是什么
改完代码跑一次 benchmark，发现候选慢了 3%，这算回归吗？不一定——机器温度、前后台负载、执行顺序都会让同一份代码每次略有不同。如果基线全在上午跑、候选全在下午跑，时间差里混进了环境差，根本分不清是代码变慢还是机器变慢。
生活类比：比较两位厨师做同一道菜。不配对就像一位在中午灶台最热时做、另一位在清晨做，谁快谁慢说不清。配对就是让两人在同一时段、同一口锅、同样的食材下轮流做，记录每一轮的差值；多轮后看差值的典型水平，而不是看两人各自的最好成绩。

## 核心问题
- 哪些运行可以配成一对？单侧缺失、执行失败、编号重复分别怎么处理？
- 为什么要用逐对比例的中位数，而不是"候选中位数 ÷ 基线中位数"？
- 自助重采样（Bootstrap）重采样的是什么？90% 区间怎么算？
- 允许慢 5% 时，区间落在哪里判 `accept`、`tune`、`reject`？
- 失败和缺失为什么要先于性能判断，且失败不能被"更多好样本"掩盖？
- GPU 上 `torch.compile` 的首次调用成本和稳态比例为什么要分开？

## 解决思路：一步步看懂

**第 1 步：配对、缺失与失败（TODO 1）。** 每条记录形如 `{"trial_id", "workload_signature", "latency_ms", "failure"}`。`_index_runs` 先做严格校验：`trial_id` 必须是非负 `int` 且不能是 `bool`，同侧不能重复；`workload_signature` 必须是非空字符串；必须显式有 `failure` 字段；无失败时 `latency_ms` 必须是有限正数（`_finite_latency` 也拒绝 `bool`）。`pair_benchmark_runs` 再把两侧签名合起来，超过一种就拒绝混比，然后按编号并集遍历：

```python
missing_baseline = trial_id not in baseline
missing_candidate = trial_id not in candidate
failed_pair = (not missing_baseline and not missing_candidate
               and bool(baseline[trial_id]["failure"] or candidate[trial_id]["failure"]))
```

三者按缺失基线→缺失候选→失败的顺序分流到 `excluded`，只有两侧都成功才进入 `pairs`。`failed_runs` 单独列出失败明细，因为"失败"不是"没记录"：前者要修复或回退，后者要补测。`observed_trials` 用编号并集计数，保留所有见过的轮次。

`test_pairing_and_exclusions` 的夹具很典型：基线 latency `[10, 11, 9]`（编号 0/1/2），候选 `[10.5, 12]` 且 `failed_trial=1`。编号 0 成功配对；编号 1 是候选侧 `oom`；编号 2 只有基线。结果 `pairs` 只有编号 0，`excluded == {"missing_baseline": [], "missing_candidate": [2], "failed": [1]}`，`observed_trials == 3`。把两侧对调，编号 2 变成缺失基线。空输入则不报错，`observed_trials == 0`。

**第 2 步：逐对比例与中位数（TODO 2 前半）。** 设第 $i$ 对基线、候选延迟为 $b_i,c_i$，比例与变慢占比：
$$r_i=\frac{c_i}{b_i},\qquad \hat r=\mathrm{median}(r_1,\dots,r_N),\qquad s=\frac{1}{N}\sum_{i=1}^N \mathbf{1}[r_i>1]$$
$1.1$ 表示慢 10%；$s$ 只回答"多少轮更慢"，不回答慢多少。代码先对每对做 `_finite_latency`，并额外检查 `candidate_value / baseline_value` 仍有限——两侧各自有限仍可能在相除时上溢或下溢，测试专门用 `(1e-308, 1e308)` 和 `(1e308, 1e-308)` 验证会被拒绝。

为什么不能拿两组中位数相除？`test_known_ratios_and_reproducibility` 给了反例：三对 `(1,100)`、`(10,1)`、`(100,10)`，比例是 100、0.1、0.1，逐对中位数是 **0.1**；但候选中位数 10 ÷ 基线中位数 10 = **1.0**。配对把"同一轮的快慢"保留下来，跨轮的整体漂移被消掉了。

**第 3 步：自助区间（TODO 2 后半）。** 从已观测的 $N$ 个比例中有放回抽 $N$ 个，求一次中位数，重复 $K$ 次，再取这些中位数的 5% 和 95% 分位：
$$L=Q_{0.05}(\hat r^\ast_1,\dots,\hat r^\ast_K),\qquad U=Q_{0.95}(\hat r^\ast_1,\dots,\hat r^\ast_K)$$
代码是 `rng.choices(ratios, k=len(ratios))` 加 `statistics.median`，`_quantile` 用 `round((len-1)*p)` 取最近的离散位置，和 GPU 侧约定一致。测试要求固定种子可复现：`summarize_paired_ratios(pairs, 200, 7)` 两次结果完全相同；比例 `[0.9, 1.0, 1.1]` 得到 `ci_low == 0.9`、`ci_high == 1.1`；单样本时上下界都等于 `0.9`。注意重采样的是**整对比例**，不是把基线、候选两侧分别打乱，也不能改成重采样均值。

**第 4 步：决策顺序（TODO 3）。** 设允许相对回归 $\delta\ge0$，边界 $\tau=1+\delta$；默认 $\delta=0.05$ 即 $\tau=1.05$。代码：

```python
quality_failed = not quality_passed or failed_pairs > 0
insufficient_pairs = missing_pairs > 0 or summary["pair_count"] < minimum_pairs
confirmed_regression = low > boundary
confirmed_safe = high <= boundary
```

分支顺序是质量/执行失败→缺失/样本不足→确认回归→确认安全→其余。这带来两条容易写错的规则：
- 失败零容忍：`test_failure_counts_reach_decision` 用 21 轮、20 个成功且都更快（比例 0.9），第 21 轮候选失败。若只看成功样本会误放行；把 `failed_pairs=1` 传进决策后必须 `reject`，reason 为 `execution_failed`。
- 边界等号：`test_decision_regions_and_equalities` 在阈值 0.05 下检查 `(0.98,1.03)→accept`、`(1.08,1.12)→reject`、`(1.01,1.08)→tune`、`(1.0,1.05)→accept`、`(1.05,1.06)→tune`、`(1.05,1.05)→accept`。`U==τ` 算通过，`L==τ<U` 仍要复测。

**第 5 步：GPU 配对实验。** Step 5 用固定种子的 `TinyMLP(hidden_size=2048)`（`up: 2048→4096`、`down: 4096→2048`、中间 `silu`、无 bias），在 `TOKEN_COUNTS=[1,32,256]` 三档上比较 eager 与 `torch.compile(eager, dynamic=False)`。协议是每档先记录首次调用（同步墙钟），再每路径 10 次预热、30 对交替顺序复测（`trial_id` 偶数 eager 先跑、奇数 compile 先跑），稳态用 CUDA event；2000 次 Bootstrap、阈值 0.05、最少 20 对。首次调用单独存进 `first_execution`，因为它包含编译准备，混进稳态会污染比例。

notebook 保存的结果（5.4 汇总）是：三档首次输出都 `allclose(atol=1e-3, rtol=1e-3)`、无执行失败；首次调用 eager/compile 分别是 1 Token 7.27/404.44 ms、32 Token 0.46/29.00 ms、256 Token 1.11/245.67 ms；稳态中位比例与区间分别是 1 Token `1.089 [1.082, 1.100]`、32 Token `1.055 [1.050, 1.063]`、256 Token `1.017 [1.014, 1.020]`。读法：1 Token 下界 1.082 > 1.05，确认回归→`reject`；32 Token 区间跨过 1.05→`tune`；256 Token 上界 1.020 ≤ 1.05→`accept`，但它仍然比 eager 慢约 1.7%，**通过不等于加速**。三档总决策任一 reject 即 reject，所以 `overall_decision` 是 `reject`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 配对试验（paired） | 同一轮里基线和候选各跑一次，用同一编号绑在一起 |
| workload_signature | 输入、形状、模型、环境等固定条件的标识，不同签名不能合并 |
| 逐对比例 r_i | 候选延迟 ÷ 基线延迟，1.1 表示慢 10% |
| 中位比例 | 所有 r_i 的中位数，代表"典型一轮"的快慢 |
| 变慢占比 s | 成功配对中 r_i > 1 的比例，只讲次数不讲幅度 |
| Bootstrap | 有放回重采样整对比例，重复求中位数，估计区间 |
| 90% 区间 [L,U] | Bootstrap 中位数的 5% 与 95% 分位，不是单次延迟范围 |
| 回归边界 τ | 1 + 允许回归量，默认 1.05 |
| `accept` / `tune` / `reject` | 上界不超边界 / 区间跨界或样本不足 / 失败或确认回归 |
| 首次调用成本 | 含编译、缓存预热的那一次，不能和稳态混算 |
| minimum_pairs | 作出结论前要求的最少成功配对数，默认 20 |

## 小白常见疑问
**Q1：为什么 `U <= 1.05` 就 `accept`，明明候选中位比例是 1.017、还是更慢？**
因为这里检查的是"是否超过允许的回归幅度"，不是"是否变快"。1.017 的区间上界 1.020 仍在 1.05 内，所以通过；要证明加速，得另拿比例和 1 比较。notebook 在 5.4 明确写了"256 Token 的通过与加速是不同判断"。

**Q2：缺失一轮和失败一轮，为什么要分开？**
处理动作不同：失败要实现或回退（而且当前规则零容忍），缺失要补测。`test_quality_failure_and_missing_priority` 里 `failed_pairs=1` 得 `execution_failed`，`missing_pairs=1` 得 `incomplete_pairs` 且决策是 `tune`。把它们都当"少一条数据"会漏掉真正的 bug。

**Q3：重采样 2000 次能让区间更可信吗？**
只能让区间估计本身更稳定，不能替代新增真实配对。notebook 的表格写明"增加计算次数不能代替新增真实配对"；`BOOTSTRAP_SAMPLES=2000` 只是降低蒙特卡洛噪声，样本量 N 才是信息量的来源。

**Q4：为什么 `pair_benchmark_runs` 要检查两侧签名？**
签名编码输入、形状、模型和环境。如果基线和候选在不同签名下测得，比例里混入了工作负载差异，不再是"同一任务的快慢变化"。`test_duplicate_and_signature_checks` 用 `signature="other"` 验证会抛 `ValueError`。

**Q5：`_quantile` 为什么用 `round((len-1)*p)` 而不是插值？**
notebook 选择"排序后最接近的离散位置"，并注明与 GPU 统计保持同一约定。这样 CPU 练习和 GPU 复算得到完全一致的边界值，便于复现；它不是统计软件里常见的线性插值分位数。

## 动手验证：怎么知道自己做对了
`run_perf_regression_tests()` 依次跑 8 个测试，全过打印 `✅ 配对完整性、自助区间与回归判断测试通过`。TODO 未完成会抛 `NotImplementedError`。
- `test_pairing_and_exclusions`：乱序基线也能配出编号 0；`excluded == {"missing_baseline": [], "missing_candidate": [2], "failed": [1]}`；`observed_trials == 3`；失败明细为 `{"side": "candidate", "trial_id": 1, "failure": "oom"}`；对调后缺失基线为 `[2]`；空输入观测 0 轮。
- `test_duplicate_and_signature_checks`：重复编号、签名不同、`trial_id` 为 `True`/`0.5`/`-1`、空签名都抛 `ValueError`。
- `test_known_ratios_and_reproducibility`：`[0.9,1.0,1.1]` 的 `median_ratio == 1`、`slower_fraction == 1/3`、区间 `[0.9,1.1]`；固定种子两次结果相等；单样本三值都为 0.9；反例中配对中位数 0.1 而两组中位数之比 1.0。
- `test_invalid_statistical_inputs`：`0`、`-1`、`True`、`NaN`、`inf`、空列表、非法 `bootstrap_samples`（0/-1/True/1.5）都拒绝；极端比值溢出也拒绝。
- `test_decision_regions_and_equalities` 与 `test_quality_failure_and_missing_priority`：覆盖上面六个区间样例和 `quality_guard_failed`、`execution_failed`、`incomplete_pairs`、`insufficient_pairs` 四种 reason。
- `test_failure_counts_reach_decision`：20 个 0.9 的成功比例 + 1 次失败，最终必须 `reject`。
- `test_invalid_decision_inputs`：`NaN`/`inf`/`-0.1`/`True` 阈值、`pair_count=1.5`、`ci_low=NaN`、`ci_high<ci_low`、`minimum_pairs=0` 都拒绝。
- GPU 单元保存了三档负载的原始配对与首次调用输出；5.3 的读取器会从原始配对重算并与记录核对。若本地重跑，绝对毫秒会变，但"先正确性、再失败/缺失、再区间"的判断顺序和 1.05 边界不变。
