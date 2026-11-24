# 28. Pipeline Parallelism MicroBatch | Pipeline 并行微批次 · 小白入门

> 配套 notebook：[28. Pipeline 并行微批次](../28_Pipeline_Parallelism_MicroBatch.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

模型太大放不进一张 GPU 时，可以把层切给几张卡接力计算。接力时前面的卡在工作、后面的卡在等，这本 notebook 教你把一批数据切成小份来减少这种等待。

## 为什么需要它

完整模型放不进一张 GPU 时，可以把连续的模型层分给多张卡，每张卡负责一段，数据依次流过去。这就是流水线并行（Pipeline Parallelism）。

麻烦在于等待。如果一个批次必须整份从第一段流到最后一段，流完才轮到下一个批次，那么第一段工作时，后面的段都在等；最后一段工作时，前面的段已经闲下来了。这像车间流水线上只有一件产品在走，大部分工位都空着。对应到本节就是，stage 之间频繁等待，设备利用率被拉低。

办法是把一个大批次切成多个微批次（micro-batch），让它们一个接一个进入流水线。第一个微批次到了第二段时，第一段已经在处理第二个微批次，几段交错工作。但开头和结尾仍然有空档：开始时后面的段还没收到数据，结束时前面的段已经没有新工作。这些空档就是流水线气泡（bubble）。

这本 notebook 在 CPU 上把这条时间线画出来，数一数空档，再比较两种训练调度对显存的影响。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| stage | 流水线里的一段，也就是一张 GPU 负责的那几层，编号从 0 开始 |
| 微批次（micro-batch） | 把一个大批次切成的小份，每一份单独流过所有 stage |
| 时间步 / 槽位（time step / slot） | 把时间切成等长的小格。一个槽位就是“某个 stage 在某个时间步” |
| 流水线气泡（bubble） | 没有被计算占用的槽位 |
| 灌入 / 排空（fill / drain） | 开始时数据逐个 stage 流进去的那几步，和结束时逐个 stage 流完的那几步 |
| activation | 前向传播留下的中间结果，反向传播要用它。留着它就占显存 |
| fill-drain / 1F1B | 两种训练调度。fill-drain 先做完所有前向再做反向；1F1B 让前向和反向交错 |

## 核心思路

输入是 stage 数 `p` 和微批次数 `m`，输出是一条时间线：每个时间步里，哪些 stage 在处理哪个微批次。简化时间线假设每个微批次在每个 stage 上花一样长的时间。构造方法只有一句话：在时间步 `t`，stage `s` 处理的是编号为 `t - s` 的微批次；编号不在 `0` 到 `m - 1` 之间，这个 stage 就空着。时间步一共 `p + m - 1` 个。

气泡占比用槽位数来算：

```text
总槽位   = 时间步数 × p = p × (m + p - 1)     # 每个 stage 每个时间步都满载时的格子数
活跃槽位 = m × p                             # 每个微批次经过每个 stage 各一次
bubble  = (总槽位 - 活跃槽位) / 总槽位 = (p - 1) / (m + p - 1)
```

`m` 远大于 `p` 时，比例近似为 `(p - 1) / m`。用 `p = 3`、`m = 4` 画一下：

```python
p, m = 3, 4                                  # stage 数、微批次数
grid = []
for stage in range(p):
    row = []
    for t in range(p + m - 1):               # 时间步 t = 0 … 5
        micro_idx = t - stage                # 这个 stage 在时间步 t 处理的微批次编号
        row.append(f"F{micro_idx}" if 0 <= micro_idx < m else "--")
    grid.append(row)
    print(f"stage {stage}", row)

active_slots = sum(cell != "--" for row in grid for cell in row)
total_slots = len(grid) * len(grid[0])       # 时间步数 × stage 数
print(active_slots, total_slots, round(1 - active_slots / total_slots, 4))
print([(m2, round((p - 1) / (m2 + p - 1), 4)) for m2 in (4, 8, 16)])   # 公式 (p - 1) / (m + p - 1)
```

```text
stage 0 ['F0', 'F1', 'F2', 'F3', '--', '--']
stage 1 ['--', 'F0', 'F1', 'F2', 'F3', '--']
stage 2 ['--', '--', 'F0', 'F1', 'F2', 'F3']
12 18 0.3333
[(4, 0.3333), (8, 0.2), (16, 0.1111)]
```

`F0` 表示正在处理 0 号微批次的前向，`--` 表示空着。左下角是灌入，数据还没流到后面的 stage；右上角是排空，前面的 stage 已经没有新活。两个三角形一共 6 个空格，占 18 个槽位的 `0.3333`，和公式 `(p - 1) / (m + p - 1)` 算出的 `2 / 6` 一致。`p` 不变，微批次增加到 8 个和 16 个，气泡占比降到 `0.2` 和 `0.1111`。

这个比例只是均衡时间线上的理想下界，不直接等于真实的训练步时间。训练时还有反向传播，有两种排法：

```text
fill-drain：所有微批次的前向全部做完，才开始反向
1F1B：     反向一旦就绪就优先做反向，前向和反向交错
```

依赖关系是：一个 stage 的前向，要等上游 stage 做完同一个微批次的前向；反向要等本 stage 的前向做完，还要等下游 stage 的反向做完。前向留下的 activation，要等到对应微批次的反向完成才释放。所以 fill-drain 下靠前的 stage 会同时留着较多 activation，1F1B 能更早释放它们。微批次也不是越多越好：气泡变小了，同时存活的 activation 片段却变多了。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | stage、微批次和跨阶段依赖：一个微批次怎样从前走到后，再反向返回 | 读，对照上面的时间线 |
| Step 2 | fill-drain 和 1F1B 两种训练调度、推理流水线，以及气泡占比公式 | 读，对照上面的算式 |
| Step 3 | 增加微批次、增加 stage、stage 耗时不均时，气泡、显存和通信怎么变 | 读 |
| Step 4 · TODO 1 | `p`、`m`、时间步和 stage 索引 | 在 `build_pipeline_timeline` 里写 2 行：算出当前 stage 的微批次编号，判断它是否在有效范围内 |
| Step 4 · TODO 2 | `timeline`、`active_slots`、`total_slots` | 在 `compute_bubble_ratio` 里写 3 行：数活跃槽位和总槽位，算出未占用的比例 |
| Step 4 · TODO 3 | `ready_forward`、`ready_backward`、`policy`、`completed` | 在 `select_pipeline_event` 里写 3 行：按策略排候选事件，取第一个，没有就返回 `None` |
| Step 4 · TODO 4 | 前向 / 反向完成集合、`peak_live_activations` | 在 `summarize_pipeline_schedule` 里写 3 行：累加活跃槽位，把事件记进对应的完成集合，数出每个 stage 存活的 activation |
| 测试 | 检查时间轴长度和覆盖、活跃槽位和气泡公式、前反向依赖，以及两种策略的 activation 驻留差异 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上复测单 stage 与两 stage 流水线的显存和时间 | 需要 `PAR-CUDA-MECHANISM` 环境和两张 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要两张 GPU，默认不执行真实实验。

## 容易踩的坑

- **时间轴没按对角线推进**：时间步 `t` 上，stage `s` 处理的是 `t - s` 号微批次，编号不在 `0` 到 `m - 1` 之间时这个 stage 空着。直接用 `t` 当编号，所有 stage 会在同一时间步处理同一个微批次。
- **气泡比例直接套公式**：题目要求先用时间轴数出活跃槽位，再用 `len(timeline) * p` 做分母，不写成只依赖 `p`、`m` 的孤立公式。
- **把理想比例当成真实步时间**：它只描述均衡时间线上的空槽位。真实调度还要检查前反向依赖、stage 不均衡和边界通信，较小的气泡不自动等于更高的吞吐。
- **1F1B 的优先级写反了**：1F1B 要优先已就绪的反向，为的是尽早释放 activation。前向没做完就一直做前向的，是 fill-drain。
- **activation 的存活时间算错**：它从前向完成后开始存活，到对应微批次在本 stage 的反向完成后才释放。

## 读完后自查

- [ ] 能说出 `p = 3`、`m = 4` 时时间线一共有几个时间步。
- [ ] 能数出上面时间线里的空槽位，并算出气泡占比。
- [ ] 能写出气泡占比公式，并说出微批次变多时它怎么变。
- [ ] 能解释 1F1B 为什么比 fill-drain 更早释放 activation。

## 下一步

- 打开 notebook：[28. Pipeline 并行微批次](../28_Pipeline_Parallelism_MicroBatch.md)
- 需要补基础：[并行策略决策框架](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)、[通信拓扑与分布式基石](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)
- 做完之后：[29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim.md)
