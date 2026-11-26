# 28. Pipeline 并行微批次 — 小白讲解

> 对应 notebook：[`28_Pipeline_Parallelism_MicroBatch.ipynb`](./28_Pipeline_Parallelism_MicroBatch.ipynb)

## 一句话总结
模型放不进一张 GPU 时，可以把层切给多个 stage；但每个 batch 都从头走到尾会让 stage 互相等待。本节先用 `build_pipeline_timeline` 画出"灌入—排空"的对角线时间轴，用 `compute_bubble_ratio` 算出理想气泡率，再用 `select_pipeline_event` 和 `summarize_pipeline_schedule` 比较 fill-drain（GPipe 风格）与简化 1F1B 的前反向依赖和 activation 驻留。最后可选在 PyTorch GPipe 与 DeepSpeed 上核对真实语义。

## 为什么需要它：痛点是什么
初学者听到"micro-batch 越多，流水线气泡越小"，容易直接得出"那就把 batch 切得越碎越好"。但切碎有代价：每个未完成的 micro-batch 的 activation 都要留在显存里等反向，调度事件也更多；而且 stage 之间负载不均时，最慢的 stage 决定整条流水线的节奏。气泡小不等于端到端吞吐高。

生活类比：流水线包饺子。stage 0 擀皮、stage 1 包馅。如果一次只做一个饺子，擀皮的人做完要等包馅的人腾出手；把一批饺子分成多个 micro-batch 交错做，等待少了，但每个未包完的饺子都要占一个托盘（activation）。1F1B 就是"包完一个立刻让下一个人擀，别囤一桌子皮"。

## 核心问题
- stage、micro-batch 与前向/反向依赖分别是什么？时间轴怎么读？
- 气泡率公式 $\rho=(S-1)/(m+S-1)$ 是怎么从槽位守恒推出来的？
- fill-drain 和 1F1B 的区别在哪里？为什么 1F1B 能更早释放 activation？
- 为什么增加 micro-batch 不总是更好？还要看哪些代价？
- 训练 Pipeline 与推理 Pipeline 的依赖和代价有何不同？

## 解决思路：一步步看懂

**第 1 步：理解事件与时间轴（Step 1）。** 设 stage 数为 $S$（代码里叫 `p`），micro-batch 数为 $m$。一个 micro-batch 的前向必须按 `stage 0 → stage 1 → …` 推进，反向沿相反方向返回。当前 stage 把前向结果交给下游后，才有机会接收下一个 micro-batch。时间轴 `timeline[t]` 的每个元素是第 $t$ 步活跃的 `(stage, micro_batch)` 列表；没有被计算占用的格子就是气泡。

**第 2 步：TODO 1——构造时间轴。** 参考实现：

```python
total_steps = p + m - 1
for t in range(total_steps):
    active = []
    for stage in range(p):
        micro_idx = t - stage
        is_active = 0 <= micro_idx < m
        if is_active:
            active.append((stage, micro_idx))
    timeline.append(active)
```

`micro_idx = t - stage` 是核心：同一个 micro-batch 每经过一个 stage 就晚一个时间步，形成对角线；`0 <= micro_idx < m` 过滤掉灌入阶段（micro 还没轮到）和排空阶段（micro 已走完）。总步数是 $p+m-1$。

测试 `test_pipeline_timeline_structure` 用 `p=3, m=5` 检查：时间轴长度 7；所有坐标满足 `0 <= stage < 3`、`0 <= micro < 5`；`(stage, micro)` 对恰好 15 个且不重复；每个 micro 出现 3 次、每个 stage 出现 5 次。`test_pipeline_progression_order` 用 `p=4, m=3` 断言每个事件满足 `t == stage + micro`，即 micro 每次只跨一个 stage。`test_pipeline_active_slots` 验证总活跃槽位等于 $p \times m = 15$。

**第 3 步：TODO 2——从槽位算气泡率。** 参考实现：

```python
timeline = build_pipeline_timeline(p, m)
active_slots = sum(len(step) for step in timeline)   # 真正干活的槽位 = p*m
total_slots = len(timeline) * p                      # 所有 stage 在所有时间步的槽位
bubble_ratio = 1 - active_slots / total_slots
```

代入 $S=p$、$m$：`total_slots = (p+m-1)p`，`active_slots = pm`，所以

$$
\rho_{\mathrm{bubble}}=1-\frac{pm}{p(p+m-1)}
=\frac{p(m+p-1)-pm}{p(m+p-1)}
=\frac{p-1}{m+p-1}.
$$

`test_pipeline_bubble` 用 `p=8, m=32` 断言 `abs(ratio - 7/39) < 1e-12`。手算：$7/39\approx0.179487$，即约 17.9% 的槽位是气泡；槽位视角是 $312-256=56$ 个空槽，$56/312$ 同样约 17.9%。`test_pipeline_bubble_boundary` 断言 `compute_bubble_ratio(3,8) <= compute_bubble_ratio(3,4)`，即增加 micro-batch 不会放大理想气泡。`test_pipeline_input_contract` 检查 $p$ 或 $m$ 为 0、负数时抛 `ValueError`。

**第 4 步：TODO 3——选择下一个事件。** 参考实现：

```python
if policy == 'fill_drain':
    candidates = ready_forward if forward_remaining else ready_backward
else:
    candidates = ready_backward or ready_forward
selected_event = candidates[0] if candidates else None
```

`fill_drain` 的规则是：只要还有前向没完成，就只排前向；全部前向结束后才排反向——这就是 GPipe 风格，"先灌满再排空"。简化 `1f1b` 则优先取已就绪的反向，没有反向才取前向，让前向和反向交错，从而更早释放对应 activation。`candidates[0]` 取的是列表里第一个就绪事件，依赖关系由调用方 `build_training_schedule` 预先筛好：

- 前向 `('F', stage, micro)` 就绪条件：本 stage 未完成，且 `stage == 0` 或上游 `('F', stage-1, micro)` 已完成；
- 反向 `('B', stage, micro)` 就绪条件：本 stage 的前向已完成，且 `stage == p-1` 或下游 `('B', stage+1, micro)` 已完成。

每个 stage 在每个时间步最多选一个事件，所以 `step_events` 最多 $p$ 个。如果某一步一个事件都选不出来，说明依赖写错，代码会抛 `RuntimeError('调度没有可执行事件，前反向依赖可能错误')`。

`test_training_schedule_dependencies` 对两种策略、`p=3, m=4` 断言 24 个事件齐全且唯一，并检查：`F(stage,micro)` 早于 `B(stage,micro)`；stage>0 时 `F(stage-1,micro)` 早于 `F(stage,micro)`；stage<2 时 `B(stage+1,micro)` 早于 `B(stage,micro)`。这正好对应"前向顺流、反向逆流"。

**第 5 步：TODO 4——统计 activation 驻留。** 参考实现：

```python
for step_events in timeline:
    active_slots += len(step_events)
    for direction, stage, micro in step_events:
        completed_events = completed_forward if direction == 'F' else completed_backward
        completed_events.add((stage, micro))
    for stage in range(p):
        live_count = sum(
            (stage, micro) in completed_forward and (stage, micro) not in completed_backward
            for micro in range(m)
        )
        peak_live_activations[stage] = max(peak_live_activations[stage], live_count)
```

activation 的寿命是"前向完成 → 对应反向完成"：前向做完就产生，反向做完才释放。`live_count` 数的是当前已完成前向但还没完成反向的 micro-batch 个数，每个时间步取峰值。`summarize_pipeline_schedule` 最后返回 `time_steps`、`bubble_ratio = 1 - active_slots/total_slots`（这里是含反向的完整训练时间线，不是前面纯前向的公式）和 `peak_live_activations`。

`test_schedule_activation_tradeoff` 比较 `p=3, m=6` 的两种策略：断言 `fill_drain` 每层峰值都大于 0，且 `max(1f1b peaks) <= max(fill_drain peaks)`。这就是 1F1B 的核心收益——不是一定减少气泡，而是更早释放 activation、降低峰值驻留。

**第 6 步：框架核对与可选 GPU 实验。** Step 14 用两个本地 Gloo 进程和 PyTorch 的 `pipeline`、`SplitPoint.BEGINNING`、`ScheduleGPipe(n_microbatches=2)` 把 4×4 输入切成两个 micro-batch，跨两个 stage 执行。`check_pipeline_framework_conformance` 检查：rank 0 的 source stage 返回 `None`、目标 stage 输出形状 `[4, 2]`、与同权重 dense 前向的 `max_abs_error < 1e-7`。这一步验证的是语义和形状，不测 GPU 气泡或跨卡带宽。

Step 5 是可选的 DeepSpeed 实验：`MODEL_WIDTH=1024`、`MODEL_LAYERS=8`、`MICRO_BATCH=1`、`MICRO_BATCHES=4`、`SEQ_LEN=256`、BF16，对比 `single_stage_baseline`（1 stage）和 `two_stage_pipeline`（2 stage）。默认 `RUN_MODE='dry_run'` 不执行。5.4 的表格"当前结果"全部是"运行后填写"，notebook 没有保存 GPU 数值，不能编造耗时或扩展效率。读结果时先确认 `failure` 为空、模型形状与 micro-batch 一致，再看每 rank 峰值显存是否解决容量约束、最慢 stage/rank 时间和 bubble 或吞吐代理，最后决定调 stage 切分还是调 micro-batch。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Pipeline 并行 | 把模型层切成多个 stage 分给不同设备 |
| stage | 流水线的一段，持有连续的若干层 |
| micro-batch | 把一个 batch 切小后的执行单元 |
| 灌入 / 排空 | 流水线开始只有部分 stage 有活 / 结束只有部分 stage 还有活 |
| 气泡（bubble） | 时间轴上没有被计算占用的槽位 |
| 气泡率 | 空槽位占总槽位的比例，理想前向为 $(p-1)/(m+p-1)$ |
| fill-drain（GPipe） | 先做完所有前向，再统一开始反向 |
| 1F1B | 反向就绪就优先做，让前向反向交错 |
| activation 驻留 | 前向产出后要留到反向才能释放的中间结果 |
| `peak_live_activations` | 每个 stage 同时存活的 activation 峰值 |
| `ScheduleGPipe` | PyTorch 分布式流水线的 GPipe 调度实现 |

## 小白常见疑问

**Q1：micro-batch 越多，气泡率越低，为什么不直接取很大？**
理想公式只描述均衡时间轴上的空槽位。实际还要看 activation 显存（未完成反向的 micro-batch 都要留状态）、调度事件开销，以及 stage 是否均衡。notebook 的 Step 3 表格把"增加 micro-batch"的代价列为"更多 activation 片段与调度事件"，并建议先检查单卡显存和 activation 生命周期。

**Q2：1F1B 一定比 fill-drain 气泡更少吗？**
不一定。简单前向时间轴的公式只由 $p$、$m$ 决定，两种策略都要经历灌入和排空；1F1B 的主要差异是 activation 更早释放。测试 `test_schedule_activation_tradeoff` 断言的是峰值 activation 不增加，而不是气泡率下降。

**Q3：为什么时间轴总步数是 $p+m-1$？**
第一个 micro-batch 要走 $p$ 个 stage，之后每个新 micro-batch 让流水线多走一步，一共 $m-1$ 个额外步，所以 $p+(m-1)$。`p=3, m=5` 时是 7 步，测试直接断言 `len(timeline) == 7`。

**Q4：反向事件的依赖为什么是"下游 stage 的反向"先完成？**
反向传播的梯度从最后一层往前传。`('B', stage, micro)` 需要本 stage 前向已完成，并且 `stage+1` 的同一个 micro-batch 反向已完成（把梯度传回来）。代码里的条件是 `stage == p-1 or ('B', stage + 1, micro) in completed`。

**Q5：推理 Pipeline 和训练 Pipeline 有什么不同？**
推理没有反向，只需要前向 micro-batch 的跨 stage 传递；公式仍可描述一批 micro-batch 的灌入排空，但在线服务还要区分单请求延迟和跨请求吞吐，持续到达的请求会引入排队和 stage 间 handoff。notebook 明确说本节只分析既定请求进入 Pipeline 后的 stage 执行，请求选择属于 Serving 调度。

## 动手验证：怎么知道自己做对了

`run_pipeline_test` 依次运行八个测试：

- `test_pipeline_timeline_structure`：`p=3, m=5` 时间轴长度 7，15 个 `(stage, micro)` 唯一，每个 micro 出现 3 次、每个 stage 出现 5 次。
- `test_pipeline_active_slots`：活跃槽位总数 `sum(len(step)) == 15`。
- `test_pipeline_progression_order`：`p=4, m=3` 时每个事件满足 `t == stage + micro`。
- `test_pipeline_bubble_boundary`：`ratio(3,8) <= ratio(3,4)`。
- `test_pipeline_input_contract`：`(0,4)`、`(3,0)`、`(-1,4)` 都必须抛 `ValueError`。
- `test_training_schedule_dependencies`：两种策略各 24 个事件，前向/反向依赖顺序全部满足。
- `test_schedule_activation_tradeoff`：`p=3, m=6` 时 fill-drain 峰值全部大于 0，且 1F1B 的最大峰值不超过 fill-drain。
- `test_pipeline_bubble`：`compute_bubble_ratio(8, 32)` 与 `7/39` 的误差小于 `1e-12`。

notebook 的 CPU 部分没有保存执行输出，以这些断言为准；全部通过后 `run_pipeline_test` 不会额外打印成功信息，若失败会打印 `❌ Pipeline 机制测试失败` 并抛出。框架核对单元若运行，会返回 `status: passed`、`world_size: 2` 和三个 `checks` 字段；它需要 Gloo 环境，属于可选路径。GPU 的 DeepSpeed 实验默认 `dry_run`，5.3 在结果不存在时只打印"尚无真实 Pipeline 结果"，不会给出虚构指标。
