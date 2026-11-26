# 80. MoE 专家并行基准 — 小白讲解

> 对应 notebook：[`80_MoE_Expert_Parallel_Benchmark.ipynb`](./80_MoE_Expert_Parallel_Benchmark.ipynb)

## 一句话总结
MoE 层把每个词元送给 Top-k 个专家，专家分散到多个 rank 后就要 All-to-All 分发与返回。本节先用三个 TODO 做“路由记录筛选器”：用实际完成的 `effective_tokens` 而不是原计划词元数算速率，比较候选相对基线的通信、偏斜、溢出与成功比例，再按 `ready_for_gpu / tune / reject` 三条路径决定是否进入多卡复测；最后给出双卡 balanced / skewed 路由的 GPU 实验口径。核心结论是：偏斜路由可能因为丢弃 280 个路由项而“跑得更快”，必须把完成量、溢出和通信代价一起看，短时间本身不是收益。

## 为什么需要它：痛点是什么
初学者看 MoE 基准时容易只盯“耗时”：skewed 6000 ms 比 balanced 8000 ms 快，就以为偏斜路由更好。但 skewed 的 720 个有效路由项比 balanced 的 1000 个少了 28%，它是靠丢工作换来的短时间；真实系统里这些被丢掉的路由项会影响模型质量，或者需要在后续步骤重试。另一个误区是把“词元数”和“路由项数”混为一谈：100 个词元、每个选 2 个专家，会产生 200 个分配项，不是 100 个。

生活类比：快递分拣。100 个包裹、每个要送 2 个目的地，就是 200 个“包裹—目的地”任务。某个分拣口容量只有 180 件，超出的 20 件要么退回、要么改送、要么延迟；如果你只数“今天处理了 100 个包裹”就宣布吞吐高，实际上有 20 件没送到。比较两个分拣方案，必须同时看“实际送成多少件”“哪个口最堵”“跨区运输花了多少时间”。

## 核心问题
- 一个词元选 Top-k 个专家，为什么会产生 k 个路由项？“原始词元数”“分配路由项数”“已接收路由项数”分别是什么？
- 专家容量和溢出是什么？丢弃、重路由、延迟处理三种策略对吞吐比较有什么影响？
- 为什么 `effective_tokens / duration_ms * 1000` 才是路由项每秒，而不是用原计划词元数？
- 候选相对基线的差值方向怎么定义？`effective_throughput_gain == 25` 为什么是绝对差不是 25%？
- `recommend_moe_parallel_run` 的三条决策路径分别要求什么？
- GPU 实验 balanced / skewed 只改变什么？为什么“返回形状正确、数值有限”不足以证明模型质量？

## 解决思路：一步步看懂

**第 1 步：从词元到专家的三段路。** notebook 的 Step 1 把 MoE 层拆成：路由器为每个词元选 Top-k 个专家（得分最高的 k 个分支）；专家并行把专家分布到不同进程，目标专家在远端时就要传词元状态；All-to-All 让各进程按目标分组发送，专家算完再把结果返回来源进程；完整层还要恢复词元顺序、按路由权重合并多个专家的结果。路由均衡时各进程有机会同时结束；流量集中到少数专家时，最忙的进程会拖长本轮。Step 1 的表把路由形态分成“本地路由”“均衡跨卡路由”“偏斜跨卡路由”，分别观察各专家分配量、传输字节/各进程耗时、超额分配/最慢进程耗时。

**第 2 步：五个数量的账本。** Step 2 的表是本节最重要的口径：

| 数量 | 计算方式 | 解释什么 |
|---|---|---|
| 原始词元数 | 输入词元数量 | 模型收到多少输入 |
| 分配路由项数 | 词元数 × 每个词元选择的专家数 | 各专家合计收到多少分配 |
| 专家容量 | 单个专家允许接收的路由项上限 | 本轮可接收多少工作 |
| 溢出路由项数 | 超出各专家容量的分配之和 | 哪些分配未按原计划执行 |
| 已接收路由项数 | 总分配减去被丢弃的溢出项 | 当前丢弃策略下进入专家计算的数量 |

例子：100 个词元、每个选 2 个专家，总分配 200 项；若 20 项超过容量，已接收 180 项。但“完成的原始词元数”还要看每个词元是否保留了它需要的专家结果——如果某个词元的一个专家被丢，它可能只完成了一半，不能简单说“完成了 180 个词元”。notebook 的 `effective_tokens` 沿用原字段名，在本练习里表示“已接收并完成专家计算的路由项数”，所以它除以秒数得到的是“路由项每秒”，不是“原始词元每秒”。

**第 3 步：TODO 1——汇总多种路由形态。** `summarize_moe_parallel_runs(runs)` 要返回 `run_count`、`total_effective_tokens`、`mean_comm_ms`、`max_overflow`、`best_effective_throughput_run`。空列表走提前返回，避免 `max()` 对空序列报错。参考实现：

```python
for item in runs:
    if item.get('duration_ms', 0.0) <= 0:
        raise ValueError('duration_ms 必须大于 0')
best = max(runs, key=lambda item: item['effective_tokens'] / item['duration_ms'])
return {
    'run_count': len(runs),
    'total_effective_tokens': sum(item.get('effective_tokens', item.get('tokens', 0.0)) for item in runs),
    'mean_comm_ms': sum(item.get('comm_ms', 0.0) for item in runs) / len(runs),
    'max_overflow': max(item.get('overflow', 0.0) for item in runs),
    'best_effective_throughput_run': best.get('name', 'run'),
}
```

几个细节：`duration_ms` 必须大于 0，否则速率除零或负值；`best` 用的是 `effective_tokens / duration_ms` 的原值（不乘 1000 也不影响谁最大）；`total_effective_tokens` 用 `get('effective_tokens', get('tokens', 0.0))` 兼容旧字段名；`mean_comm_ms` 是算术平均，不是加权平均；`max_overflow` 直接取最大值，用来回答“最坏的一组丢了多少”。

**第 4 步：TODO 2——候选减基线。** `compare_moe_parallel_to_baseline(baseline, candidate)` 先算两边速率：

```python
baseline_throughput = baseline['effective_tokens'] / baseline['duration_ms'] * 1000
candidate_throughput = candidate['effective_tokens'] / candidate['duration_ms'] * 1000
```

乘 1000 是把“每毫秒路由项”换成“每秒路由项”。然后返回五个差值，方向统一是“候选减基线”：`comm_delta_ms`、`imbalance_delta`（保留 4 位小数）、`effective_throughput_gain`、`overflow_delta`、`successful_step_ratio_delta`（保留 4 位小数）。正负号的含义要背下来：速率差为正表示收益；通信、偏斜、溢出差为正表示代价增加。测试里 baseline 是 `local`（1000 项 / 10000 ms = 100 项/秒），candidate 是 `cross_rank_balanced`（1000 项 / 8000 ms = 125 项/秒），所以 `effective_throughput_gain == 25`，是 25 项/秒的绝对差，不是 25%。

**第 5 步：TODO 3——三条决策路径。** `recommend_moe_parallel_run` 的默认参数是 `max_overflow=0.0`、`max_comm_increase_ms=12.0`，调用方还要给 `max_imbalance` 和 `min_successful_step_ratio`。参考实现先算 `comparison`，然后：

```python
if (comparison['effective_throughput_gain'] > 0
        and candidate.get('imbalance', 10**9) <= max_imbalance
        and candidate.get('successful_step_ratio', -10**9) >= min_successful_step_ratio
        and candidate.get('overflow', 10**9) <= max_overflow
        and comparison['comm_delta_ms'] <= max_comm_increase_ms):
    return {'decision': 'ready_for_gpu', ...}
if comparison['effective_throughput_gain'] > 0 and candidate.get('successful_step_ratio', -10**9) >= min_successful_step_ratio:
    return {'decision': 'tune', ...}
return {'decision': 'reject', ...}
```

三条路径的逻辑：
- `ready_for_gpu`：速率增加，且负载偏斜、成功比例、溢出、通信增量全部达标；
- `tune`：速率增加、成功比例也够，但容量/偏斜/通信至少有一项不达标，需要调路由、容量或专家放置；
- `reject`：速率没增加，或者成功比例不达标（连“稳定跑完”都保证不了），退回基线。

`candidate.get('imbalance', 10**9)` 和 `get('successful_step_ratio', -10**9)` 的默认值是“最坏情况”，字段缺失时不会被误判为通过。`overflow` 缺失同样默认 `10**9`，所以缺字段的候选不可能拿到 `ready_for_gpu`。

**第 6 步：用测试数据走一遍。** `test_moe_parallel_decision` 给了三组记录：

| 记录 | comm_ms | imbalance | duration_ms | success | overflow | effective_tokens |
|---|---|---|---|---|---|---|
| `local` 基线 | 40 | 0.25 | 10000 | 0.82 | 0 | 1000 |
| `cross_rank_balanced` | 48 | 0.12 | 8000 | 0.80 | 0 | 1000 |
| `cross_rank_skewed` | 47 | 0.72 | 6000 | 0.80 | 280 | 720 |

汇总：`run_count == 3`；三组速率分别是 100、125、120 项/秒，所以 `best_effective_throughput_run == 'cross_rank_balanced'`；`total_effective_tokens == 1000+1000+720 == 2720`；`max_overflow == 280`。比较 balanced 相对 baseline：`comm_delta_ms == 8`、`imbalance_delta == -0.13`（0.12−0.25）、`effective_throughput_gain == 25`、`overflow_delta == 0`、`successful_step_ratio_delta == -0.02`（0.80−0.82）。

决策：`recommend_moe_parallel_run(baseline, candidate, max_imbalance=0.15, min_successful_step_ratio=0.78)` 时，速率 +25 > 0、偏斜 0.12 ≤ 0.15、成功 0.80 ≥ 0.78、溢出 0 ≤ 0、通信增量 8 ≤ 12，返回 `ready_for_gpu`、`next_action='run_cluster_evaluation'`。skewed 的速率 +20 仍 > 0、成功 0.80 也够，但偏斜 0.72 > 0.15、溢出 280 > 0，所以落到 `tune`。把 balanced 的 `successful_step_ratio` 改成 0.60 后，速率收益仍在，但成功比例低于 0.78，第二个 `if` 也不成立，返回 `reject`。

**第 7 步：skewed 的“快”为什么不等于“好”。** skewed 的 6000 ms 比 balanced 的 8000 ms 短，但它的有效路由项只有 720，速率 120 项/秒，仍低于 balanced 的 125；同时它丢了 280 个路由项、偏斜 0.72。如果只报告“耗时缩短 25%”，就掩盖了“少做 28% 工作”的事实。notebook 的 Step 3 要求同时列出已接收路由项、溢出数量、分发与返回字节、最慢进程耗时，再判断热点是否真正缓解。偏斜还要区分“专家之间”和“进程之间”：同一进程上的两个专家一忙一闲，进程总负载可能仍和其他进程接近，只看进程级指标会漏掉专家级热点。

**第 8 步：GPU 实验设计。** 5.1 的固定配置：`WORLD_SIZE=2`、`EXPERTS_PER_RANK=1`、`TOKENS_PER_RANK=256`、`HIDDEN_SIZE=1024`、`DTYPE='bf16'`、`WARMUP=2`、`REPEATS=5`；`ROUTE_MODES=['balanced', 'skewed']`，只改路由负载分布，其余保持一致。三组资产：`G0` 是 balanced，`G1` 是 skewed，`G2` 是可选的真实 MoE backend（需要原词元输出、质量和端到端时间）。5.4 用 `torchrun --standalone --nproc_per_node=2 tools/run_parallel_mechanism_benchmark.py --mode ep_dispatch --routing <mode>` 分别跑两组，把原始 JSON 写到 `benchmarks/results/80_moe_dispatch/<mode>.json`，再用 `tools/aggregate_distributed_evidence.py --project 80` 汇总成 `summary.json`。5.5 用 `validate_record` 按 `distributed-benchmark/v1` 校验，并生成给 81 节消费的资产引用 `summary_reference.json`。

默认 `RUN_GPU_EXPERIMENT=False`，5.2 的 `GPU_READY` 也是 False，所以 notebook 只打印配置，不会真的启动多卡。要跑的话至少需要两张可见 CUDA GPU 和 `torchrun`，预检会检查 `torch.cuda.device_count() >= WORLD_SIZE`。

**第 9 步：指标怎么读、实验不能证明什么。** 先看运行状态，区分启动失败、显存不足和 All-to-All 错误；再核对输入数量、进程数、专家数、Top-k、容量系数和精度是否一致；然后同时看各专家容量前分配量、接收路由项和溢出比例，以及最慢进程耗时、跨卡分发/往返字节、峰值显存。notebook 明确说当前简化路径没有为每个专家配置独立权重，也没有恢复原词元顺序和完成 Top-k 加权合并；返回形状和有限值检查只对应这段往返计算是否正常完成。要谈模型质量、原始词元吞吐和成功迭代比例，得进入完整模型复测。`EXPERTS_PER_RANK` 也还没接入命令，实际专家总数、Top-k 和容量系数以脚本输出为准。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MoE（混合专家） | 每个词元只激活少数几个专家，而不是走同一个 FFN |
| 路由器 | 给每个词元打分并选 Top-k 个专家的模块 |
| 路由项 | 一个词元选中一个专家产生的一项分配，k 个专家就有 k 项 |
| 专家容量 | 单个专家本轮最多接收多少路由项 |
| 溢出 | 超过专家容量、未按原计划执行的分配 |
| All-to-All | 各进程按目标分组互相发送数据，专家算完再返回 |
| `effective_tokens` | 本练习中表示已接收并完成专家计算的路由项数 |
| `imbalance` | 输入记录给定的负载偏斜指标，越大越不均 |
| `successful_step_ratio` | 成功完成的迭代数 ÷ 尝试次数 |
| 路由项每秒 | `effective_tokens / duration_ms * 1000` |
| `ready_for_gpu` | 收益与各项约束都达标，进入多卡复测 |
| `tune` / `reject` | 有收益但需调整 / 收益或稳定性不达标 |

## 小白常见疑问

**Q1：为什么偏斜路由 6000 ms 比均衡路由 8000 ms 快，却不能直接说它更好？**
因为它只完成了 720 个路由项，而均衡完成了 1000 个；它是靠丢弃 280 项换来更短时间。按速率算 720/6000×1000 = 120 项/秒，仍低于均衡的 125 项/秒。notebook 的 Step 3 说“运行更快未必更有效”，必须同时看已接收路由项、溢出数量、字节和最慢进程耗时。

**Q2：`effective_throughput_gain == 25` 为什么不是“提升 25%”？**
因为它是绝对差：candidate 125 项/秒 − baseline 100 项/秒 = 25 项/秒。比例是 25/100 = 25%，但字段名和解析都强调“单位为路由项每秒”。测试直接断言 `== 25`，不是 `== 0.25`。

**Q3：`recommend_moe_parallel_run` 里 `max_overflow` 默认 0.0，是不是太严格？**
这是本练习的默认门槛：只要候选有溢出，就拿不到 `ready_for_gpu`，最多 `tune`。真实系统可能允许一定溢出（配合重路由或延迟处理），那时可以显式传更大的 `max_overflow`；但 notebook 想强调的是“溢出必须被看见并被显式允许”，不能默默忽略。

**Q4：为什么“返回形状正确、数值有限”不能证明 MoE 输出质量？**
因为当前 GPU 路径没有每个专家的独立权重，也没有恢复词元顺序、按路由权重合并 Top-k 结果；形状和有限值只能说明 All-to-All 往返没有崩。真正的质量要拿完整模型的输出与原词元输出对照，notebook 把 `G2 real_moe_backend` 留作这一步。

**Q5：`EXPERTS_PER_RANK=1` 写了但没接入命令，会不会影响结论？**
会影响你对“专家放置”的解读。notebook 的 5.1 明确说这个配置尚未接入命令，专家总数、Top-k、容量系数用脚本默认值，运行前要核对输出工作负载里的实际值。所以 CPU 示例里的 `local` / `cross_rank_balanced` / `cross_rank_skewed` 是筛选逻辑的练习，不是多卡实测的配置镜像。

## 动手验证：怎么知道自己做对了

测试区只有一个函数 `test_moe_parallel_decision`，直接调用，TODO 未填会抛 `NotImplementedError("请先完成 TODO 代码！")`，全过才打印 `✅ 路由项速率、容量溢出与候选决策通过基础校验。`

断言逐条：
- `summary['run_count'] == 3`；
- `summary['best_effective_throughput_run'] == 'cross_rank_balanced'`（三组速率 100 / 125 / 120 项/秒）；
- `summary['total_effective_tokens'] == 2720`（1000 + 1000 + 720）；
- `summary['max_overflow'] == 280`；
- `comparison['comm_delta_ms'] == 8`（48 − 40）；
- `comparison['imbalance_delta'] == -0.13`（0.12 − 0.25）；
- `comparison['effective_throughput_gain'] == 25`（125 − 100）；
- `comparison['overflow_delta'] == 0`；
- `comparison['successful_step_ratio_delta'] == -0.02`（0.80 − 0.82）；
- `recommend_moe_parallel_run(baseline, candidate, max_imbalance=0.15, min_successful_step_ratio=0.78)` 返回 `decision == 'ready_for_gpu'`、`next_action == 'run_cluster_evaluation'`；
- 同一个函数对 `skewed` 返回 `decision == 'tune'`（收益 +20 > 0、成功 0.80 ≥ 0.78，但偏斜 0.72 > 0.15、溢出 280 > 0）；
- 把 candidate 的 `successful_step_ratio` 改成 0.60 后返回 `decision == 'reject'`。

GPU 部分默认关闭：5.1 会打印 `'run': False`、`world_size=2`、`route_modes=['balanced','skewed']`，5.2 会打印 `{'run': False, 'required_gpus': 2}`，5.4 会打印 `GPU benchmark 默认关闭。`，5.5 在没有 `summary.json` 时会打印“尚无 dispatch companion”。notebook 没有保存任何多卡结果，所以本节以三个实现函数的断言和路由账本为准；如果你真的开了双卡，先确认 `G0/G1` 的输入数量、进程数、专家数、Top-k、容量系数和精度一致，再比较接收路由项、溢出、最慢进程耗时与跨卡字节，别只报一个总时间。
