# 74. Profiling Driven End to End Optimization | profiling 驱动的候选准入与交接 · 小白入门

> 配套 notebook：[74. profiling 驱动的候选准入与交接](../74_Profiling_Driven_End_to_End_Optimization.ipynb) ｜ 难度：Hard ｜ 环境：CPU 可完成报告收口；GPU 用于真实 trace 与候选复测

## 一句话说清

优化训练的手段测出了一堆数据，各自的实验条件不同，结论不能混着用。这本 notebook 教你给每份结论标明适用范围，并且只用完整训练的结果做项目总决策。

## 为什么需要它

你给训练加了一个优化，比如激活检查点（checkpoint），想知道它值不值得采用。测完之后手上会有三类数据，它们的实验条件并不一样。完整训练的重复测量用 AdamW、`seq_len=768` 训练，记录单步耗时、吞吐、峰值显存和验证损失。激活显存专项换成无动量 SGD，把序列长度一段一段加长再测。时间线追踪（trace）和阶段测量用性能分析器（profiler）看时间花在哪个阶段、哪个算子上。

看这些数据时容易犯三种错：

- 拿专项结果当整体结论。专项在 `seq_len=1280` 时通过了，就说“可以采用”。可完整训练那组配置下，激活检查点的显存节省没达到门槛，结论是 `tune`。
- 拿解释当证据。追踪里看到重算花了很多时间，就据此决定用还是不用。追踪只能解释变化从哪里来，不能替代没有性能分析器时的测量。
- 没跑的长度当作能跑。更长的序列因为可用显存余量不够没有运行，就说“再长也没问题”。

解决办法是一句话：每份结论只在自己的证据范围内成立，项目总决策只取完整训练范围。这本 notebook 让你动手做其中的三件事：核对范围配置、保留各范围的结论和限制、用完整训练范围定出总决策。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 候选准入 | 判断一个优化候选能不能被采用，本节看的是激活检查点这类显存策略 |
| 证据范围 | 一份证据对应的实验条件，比如用哪种优化器、序列多长。结论只在这些条件下成立 |
| 采用证据 / 归因证据 | 采用证据用来决定候选用不用，是完整训练的重复测量；归因证据只解释变化来自哪里，是时间线追踪和阶段测量 |
| 配对测量 | 基线和候选在相同条件下各测一遍，再比较两者的差 |
| OOM（out of memory） | 显存不够用。OOM 边界指序列长到多少才会发生 |
| 安全准入 / 安全停止 | 跑更长序列之前先检查可用显存余量，余量不够就不跑。停下来不等于已经 OOM |
| 交接（handoff） | 把结论、证据路径和限制整理成文件，交给后面的综合项目（Part 05）使用 |

## 核心思路

输入是一份报告。报告顶层有一个 `decision`，`decision_scopes` 里是三个范围各自的结论、证据状态和实验条件：

```text
范围                         实验条件                  结论
full_training_adoption      AdamW，seq_len=768        tune
activation_memory_mechanism 无动量 SGD，seq_len=1280   accept_for_activation_memory_scope
profiler_attribution        基线和激活检查点的配对追踪  explain_only
```

三行是三组条件不同的实验，所以三份结论各管各的。处理分三步。

第一步核对范围配置。拿一张预期配置表，对每个范围、每个字段，和报告里的实际值逐项比较。整个范围不存在，记成 `范围名:missing`；某个字段不相等，记成 `范围名:字段名_mismatch`。有任何一个问题，这份报告就不能用。

第二步保留各范围的结论。每个范围留下 `decision`、`evidence_status`、`limitations` 三个字段，限制列表复制一份。专项范围即使在已测长度内通过，也要原样保留 `blocked` 状态和限制，不往上提升。

第三步形成项目总决策。总决策只取 `full_training_adoption` 的 `decision`，不读报告顶层的值；这个范围缺失就报错，不生成总决策。

用一份缩小的报告试一下：

```python
report = {'decision': 'accept'}   # 报告顶层的旧值
scopes = {
    'full_training_adoption': {'optimizer': 'AdamW', 'seq_len': 768, 'decision': 'tune'},
    'activation_memory_mechanism': {'optimizer': 'SGD', 'seq_len': 1280,
                                    'decision': 'accept_for_activation_memory_scope'},
}
expected = {'full_training_adoption': {'optimizer': 'AdamW', 'seq_len': 768},
            'activation_memory_mechanism': {'optimizer': 'SGD', 'seq_len': 1280}}

def find_issues(scopes, expected):
    return [f'{scope}:{field}_mismatch' for scope, config in expected.items()
            for field, want in config.items() if scopes[scope].get(field) != want]

print(find_issues(scopes, expected))
print(report['decision'], scopes['full_training_adoption']['decision'])

scopes['activation_memory_mechanism']['seq_len'] = 1536
print(find_issues(scopes, expected))
```

```text
[]
accept tune
['activation_memory_mechanism:seq_len_mismatch']
```

第一行是空列表，两个范围的条件都和预期一致。第二行里顶层写着 `accept`，完整训练范围却是 `tune`，项目总决策要取后者；专项范围的通过只在它自己的范围内算数。第三行把专项的序列长度改成 1536，和预期的 1280 对不上，被查了出来。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 按实验范围整理已有证据：完整训练、激活显存隔离、性能分析归因三种范围各能支持什么结论 | 读，对照上面的三行表 |
| Step 2 | 区分采用证据与归因证据：四类证据各自的观察内容和使用边界 | 读 |
| Step 3 | 按证据范围形成结论：三种范围各自的触发条件和后续动作 | 读 |
| Step 4 · TODO 1 | 核对证据范围配置 | 在 `validate_evidence_scopes` 里补 `field_matches`：拿报告里的实际字段值和预期值比较 |
| Step 4 · TODO 2 | 采用完整训练范围作为项目总决策 | 在 `build_project_decision` 里补 `project_decision`：取 `full_training_adoption` 的 `decision`，不读报告顶层 |
| 测试 | 检查配置错配或范围缺失时是否拒绝、摘要是否保留各范围的状态和限制且不改原报告、总决策是否只听完整训练范围 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（项目实验） | 读取 73、76 两本 notebook 的结果，采集配对时间线追踪和阶段显存，生成收口报告和交接文件 | 需要 `PERF-CUDA-MECHANISM` 环境和单卡至少 8 GiB 显存；默认是 `RUN_STAGE = 'prepare'`，只做只读检查，`profile` 和 `phase` 阶段才使用 GPU，`close` 只读收口 |

Step 1–4 只需要 CPU。Step 5 的 `profile` 和 `phase` 阶段需要 CUDA GPU（单卡显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境，`prepare` 和 `close` 阶段不需要 GPU。

## 容易踩的坑

- **把专项结论当成整体结论**：激活显存专项是在无动量 SGD、长度阶梯下测的，通过也只算这个范围内通过。完整训练的结论不能由专项结果提升得到。
- **把归因证据当成采用证据**：时间线追踪和阶段测量只解释变化来自哪里，不能替代没有性能分析器的重复测量，也不单独决定候选采用。
- **读了报告顶层的 `decision`**：顶层可能留着和范围结论冲突的旧值。总决策只取 `full_training_adoption` 的 `decision`；缺少这个范围时要报错，不能生成总决策。
- **把不同条件的结果拼在一起**：两类数据的优化器和序列长度不同，不能拼成同一组候选的端到端收益。先逐项核对范围配置，字段对不上就拒绝使用这份证据。
- **把安全停止当成 OOM 边界**：更长的序列因为可用显存余量不够而没有运行，只能记为未测，不能外推成“能跑”或“已经找到 OOM 边界”。

## 读完后自查

- [ ] 能说出三个证据范围的名字，以及各自能支持什么结论。
- [ ] 能说明采用证据和归因证据的区别。
- [ ] 报告顶层写 `accept`、完整训练范围写 `tune` 时，能说出项目总决策是什么、为什么。
- [ ] 能解释“安全停止不等于 OOM”是什么意思。

## 下一步

- 打开 notebook：[74. profiling 驱动的候选准入与交接](../74_Profiling_Driven_End_to_End_Optimization.ipynb)
- 建议先读：[73. 训练性能分析 · 小白入门](./73_Training_Performance_Analysis.md)、[76. Activation / Checkpoint / Offload 对比项目 · 小白入门](./76_Activation_Checkpoint_Offload_Benchmark.md)
- 需要补基础：[性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
- 做完之后：[79. 训练状态并行基准 · 小白入门](./79_Distributed_Parallel_Benchmark.md)
