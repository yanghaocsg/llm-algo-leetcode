# 29. CUDA Stream 高级调度 — 小白讲解

> 对应 notebook：[`29_CUDA_Stream_Advanced_Scheduling.ipynb`](./29_CUDA_Stream_Advanced_Scheduling.ipynb)

## 一句话总结
这一节把“多开几个 CUDA Stream 就能重叠”拆成可计算的依赖图：每个任务有耗时、所在队列和显式前驱，同队列还有隐含的提交顺序。`schedule_stages` 求出最早开始/结束时间，再据此算出单批次依赖链无法缩短、两批次流水线的理想节省是 22 us，以及提交开销或资源争用怎样把这份节省吃掉。

## 为什么需要它：痛点是什么
初学者常把 Stream 当成“多线程”：以为把拷贝和计算放进不同 Stream，它们就会自动并行。实际有两个容易忽略的约束。第一，**同一 Stream 内的任务是严格按提交顺序执行的**，后面的任务不能越过前面的任务，哪怕它依赖的数据已经就绪。第二，跨 Stream 并不会自动建立依赖，必须用 Event 显式表达“消费者要等生产者做到哪一步”。忽略任何一条，算出来的时间线都是错的。

生活类比：银行大厅的几个窗口。每个窗口（Stream）前有一队人，队内严格先来后到；但不同窗口的队伍互不相干。如果你要先去 A 窗口盖章、再拿章去 B 窗口办下一步，就得让 B 窗口的柜员等你——这张“已盖章”的回执就是 Event。多开窗口不等于业务就能并行，关键看业务之间有没有这种先后关系。

## 核心问题
- 怎样把“任务 + 队列 + 显式依赖”变成一条满足所有约束的最早完成时间线？
- 同一批次的 H2D → Kernel → D2H → 主机后处理，拆成多个 Stream 后为什么仍然是 70 us？
- CUDA Graph 回放优化的是依赖关系，还是重复提交的开销？
- 两批独立任务为什么能重叠？理想节省的 22 us 由哪几段构成？
- 提交开销、共用拷贝队列、计算资源争用分别会怎样抵消重叠收益？
- 哪些条件下应该放弃“理想重叠”的估算？

## 解决思路：一步步看懂

### Q1：用 DAG 把队列顺序和显式依赖一起算进去
`TaskStage` 是一个 `@dataclass`，四个字段：`name`、`duration_us`、`stream`、`depends_on`（默认空元组）。它只描述“哪个任务、在哪个逻辑队列、耗时多少、显式等待谁”，不涉及真实 CUDA API。

`schedule_stages(stages)` 做三件事：

1. **校验输入。** 任务名非空且不重复、队列名非空、耗时是有限非负数、`depends_on` 由任务名组成；任何一项不满足都抛 `ValueError`。
2. **补上同队列的隐含依赖。** 代码维护 `previous_on_stream` 字典，每遇到一个任务，就把该 Stream 上一个任务加入它的前驱集合：

   ```python
   dependencies = set(stage.depends_on)
   if stage.stream in previous_on_stream:
       dependencies.add(previous_on_stream[stage.stream])
   predecessors[stage.name] = dependencies
   previous_on_stream[stage.stream] = stage.name
   ```

   这一步是整节的灵魂：**你以为只写了 `depends_on`，其实每个任务还欠同队列前一个任务一笔“顺序债”**。
3. **反复推进就绪任务。** 每轮扫描所有未排程任务，当前驱都已在 `timeline` 里，就令 `start = max(前驱的 end_us, default=0.0)`，`end = start + duration`。如果一整轮没有任何任务能推进，说明显式依赖和同队列顺序合起来形成了环，抛 `ValueError('显式依赖与同队列顺序形成了环')`。

配套三个函数：`sequential_time_us` 把所有任务耗时直接相加（前提是任务集合本身合法），`pipelined_time_us` 取时间线里最大的 `end_us`，`overlap_ratio` 返回 $1 - \frac{\text{pipelined}}{\text{sequential}}$，并校验 `0 <= pipelined <= sequential`，避免把“流水线比串行还慢”算成正收益。

整套模型假设各逻辑队列有独立执行资源、阶段耗时固定、没有提交开销和资源争用，所以得到的是**理想时间线**，不是实测 CUDA 时间；忽略跨队列依赖、只取各队列累计耗时的最大值，会把仍需等待的工作误算成可重叠工作。

Q1 的例子是：`a(10 us, stream="one")`、`b(20 us, stream="one")`、`c(25 us, stream="two")`。a 和 b 同队列，b 必须等 a，所以 b 从 10 开始、30 结束；c 在另一队列，从 0 开始、25 结束。最晚完成是 30 us，而全部串行是 $10+20+25=55$ us。断言直接检查 `timeline["b"]["start_us"] == 10`、`timeline["c"]["start_us"] == 0`、`pipelined_time_us == 30`、`sequential_time_us == 55`。

### Q2：Event 依赖链不会因为“换了 Stream”而变短
`build_dependency_edges(stages)` 只返回显式边：遍历每个任务，把 `(dep, stage.name)` 收集起来。它先调用一次 `schedule_stages`，等于借排程器完成合法性校验。`has_cross_stream_dependency(stages)` 则检查是否存在某条显式边连接了两个不同 Stream 的任务。

notebook 的四阶段例子是：

| 阶段 | 耗时 | Stream | 显式依赖 |
|---|---:|---|---|
| H2D（主机→设备） | 12 us | copy | 无 |
| Kernel | 40 us | compute | H2D |
| D2H（设备→主机） | 10 us | copy | Kernel |
| Post（主机后处理） | 8 us | host | D2H |

`build_dependency_edges` 返回 `[("H2D","Kernel"), ("Kernel","D2H"), ("D2H","Post")]`；`has_cross_stream_dependency` 为真。排出来的时间线是 0–12、12–52、52–62、62–70，`pipelined_time_us == sequential_time_us == 70`。

为什么拆成三个 Stream 一点没省？因为**依赖是一条首尾相接的链**：Kernel 要等 H2D 搬完，D2H 要等 Kernel 算完，Post 要等 D2H 回传完。Stream 只决定“谁和谁排队”，不创造数据。唯一能重叠的是彼此独立的工作；这条链上没有。注意 Post 被建模在 `host` 逻辑队列里，它不是 CUDA Stream 上的设备操作，但同样受依赖约束。

### Q3：CUDA Graph 省的是提交，不是依赖
Graph 把一段已经捕获的执行结构反复回放，主要减少每次启动大量小内核时的**主机提交开销**。但捕获不会改变数据依赖：Q2 的链即使被 Graph 捕获，消费者仍然不能在生产者的数据写好前执行。notebook 用一句话点明：“Graph 回放改变了依赖，还是提交成本？”答案是后者。

`graph_suitability(is_fixed_shape, is_repeated_path, has_many_branches)` 只做保守筛选：

```python
return is_fixed_shape and is_repeated_path and not has_many_branches
```

三个参数都必须是 `bool`，否则抛 `ValueError`。四个对照用例是：固定且重复、无分支 → `True`；形状不固定 → `False`；只跑一次的路径 → `False`；分支很多 → `False`。打印时只有第一行说“继续检查捕获条件”，其余提示“先整理稳定子图或考虑普通提交”。

这里必须强调返回 `True` 的含义：**只是“值得继续检查捕获支持”，不代表捕获一定成功，更不代表会变快**。真实捕获还要看内存地址是否稳定、资源生命周期、算子是否支持等条件。变长服务也可以按形状分桶，只捕获稳定子图，不能因为“输入会变”就一刀切说 Graph 不可用。

### Q4：两批流水线的重叠窗口从哪来
把两批相同任务摆进三条队列：`copy_in`、`compute`、`copy_out`。每批是 12 us 搬入 → 40 us 计算 → 10 us 回传，依赖关系批内串联，但**第二批的搬入与第一批的计算没有依赖**，这是重叠的来源。

`pipeline_comparison(stages, exposed_overhead_us=0.0)` 返回六个量：`serial_us`、`ideal_pipeline_us`、`ideal_saved_us`、`exposed_overhead_us`、`estimated_elapsed_us`、`net_saved_us = serial - ideal - overhead`。它会校验开销是有限非负数。

排出来的时间线（断言逐项核对）：

| 任务 | 起止（us） | 说明 |
|---|---|---|
| in0 | 0–12 | 第一批搬入 |
| in1 | 12–24 | 与 compute0 并行 |
| compute0 | 12–52 | 等 in0 |
| compute1 | 52–92 | 等 in1，且同 compute 队列要排在 compute0 后 |
| out0 | 52–62 | 与 compute1 并行 |
| out1 | 92–102 | 等 compute1，同 copy_out 队列排在 out0 后 |

串行总时间 $2 \times (12+40+10) = 124$ us；理想流水线 102 us；节省 $124-102=22$ us。这 22 us 正好是“第二批搬入 12 us”加“第一批回传 10 us”两段被藏起来的时间。`compute1` 明明在 24 us 就等到了输入，却要到 52 us 才开始，因为它和 `compute0` 共用计算队列——这是同队列顺序债的又一次体现。

然后 notebook 演示三种“收益消失”的方式：

1. **暴露开销。** 传入 `exposed_overhead_us=3`，净节省 $22-3=19$；传 22，净节省 0；传 25，净节省 $-3$，流水线反而更慢。
2. **共用拷贝队列。** 把 `copy_in` 和 `copy_out` 合并成 `copy`，按原任务列表顺序，`in1` 会排到 `out0` 后面，队头阻塞使完成时间回到 124 us，与串行持平。
3. **资源争用。** 保持两批工作量不变，只把计算阶段从 40 us 改成 52 us，完成时间变成 126 us；而原串行基线仍是 124 us，所以“相对原串行”是 $-2$ us。notebook 特意提醒：比较同一批工作要保留原路径耗时，不能拿争用后的阶段时间重新定义基线，否则会把变慢算成变快。

### Q5：哪些条件会让理想重叠落空
notebook 用一张表把“模型里的假设”和“执行时要核对的事实”分开：

| 条件 | 对模型结果的影响 | 实际执行时要核对 |
|---|---|---|
| 消费者等待生产者 | 跨队列任务仍串行 | Event 与输入可用时刻 |
| 同队列提交顺序 | 后续任务不能越过前项 | 队头等待是否阻塞独立工作 |
| 多队列共用执行资源 | 理想并行时间偏乐观 | 拷贝引擎、计算资源与带宽争用 |
| 提交或同步开销暴露 | 抵消部分甚至全部隐藏时间 | 主机提交与设备完成时间 |
| Graph 捕获条件不满足 | 需调整子图或执行路径 | 形状、地址与操作支持 |

还有一条容易忽略：**Stream 优先级只影响内核调度偏好**，不保证内存传输顺序，也不保证抢占已经在执行的工作。所以计算器不按优先级生成固定收益，而是先算依赖时间线，再把“假设的暴露开销”单独加上去。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Stream（流） | 一条按提交顺序执行的队列，队内任务不能互相越过 |
| Event（事件） | 标记“某段工作做到哪了”，供其他 Stream 等待 |
| 依赖图 / DAG | 任务作节点、先后关系作边的有向无环图 |
| 前驱 / 后继 | 必须先完成的任务 / 之后才能开始的任务 |
| 关键路径 | 决定整体完成时间的那条最长依赖链 |
| 串行时间 | 所有任务一个接一个执行的总耗时 |
| 流水线时间 | 满足全部依赖的最早完成时间 |
| 重叠收益 | 串行时间减流水线时间 |
| 队头阻塞 | 队首任务没完成，后面的独立任务也过不去 |
| 暴露开销（exposed overhead） | 没被计算藏住、实打实加到总时间上的那部分 |
| CUDA Graph | 把一段执行结构捕获后反复回放，主要省提交开销 |
| Graph 捕获 | 把操作序列记录成图；不改变数据依赖 |

## 小白常见疑问
**Q1：为什么 `b` 依赖 `a`？我明明没写 `depends_on=("a",)`。**
因为 a、b 同在 `stream="one"`。`schedule_stages` 会自动把同队列的上一个任务加进前驱集合。这正是 CUDA Stream 的语义：同流内的提交顺序就是执行顺序。

**Q2：Q2 里 H2D 和 D2H 都在 `copy` 流，会不会因为同队列而多等？**
这条链本来就首尾依赖：D2H 要等 Kernel，Kernel 要等 H2D。即使把它们放到不同流，只要 Event 依赖存在，时间线仍然是 0–12–52–62–70。同队列在这里没有额外增加等待，真正锁死时间的是数据依赖。

**Q3：两批流水线为什么是 102 us，而不是 124 的一半 62 us？**
重叠只发生在“后一批搬入”和“前一批计算/回传”之间，并不是所有阶段都能并行。计算队列本身仍然串行：compute0 占 12–52，compute1 只能 52–92；批内 12→40→10 的链条也没变。把时间线逐段加起来，最后一笔在 102 us 结束。

**Q4：Graph 能把 70 us 的依赖链变短吗？**
不能。Graph 主要减少重复提交的主机开销，依赖边仍然成立。`graph_suitability` 只筛“形状固定、路径重复、分支少”的候选；即便返回 `True`，也只是“继续检查捕获条件”，不承诺加速。

**Q5：暴露开销 25 us 时净节省是 −3 us，为什么还要算？**
因为它是“候选方案相对原路径”的真实账。正收益才值得做；负收益说明重叠带来的好处被开销吃光还不够。notebook 保留负值，就是为了避免只报喜不报忧。

## 动手验证：怎么知道自己做对了
notebook 没有独立的 `test_*` 函数，所有检查都是模块级 `assert`；也没有保存任何执行输出，运行时以是否抛 `AssertionError` 为准。

- Q1：`timeline["b"]["start_us"] == 10`、`timeline["c"]["start_us"] == 0`、`pipelined_time_us(example) == 30`、`sequential_time_us(example) == 55`，并打印“独立队列的理想时间：30 us；全部串行：55 us”。
- Q2：`build_dependency_edges` 必须精确等于 `[("H2D","Kernel"), ("Kernel","D2H"), ("D2H","Post")]`；`has_cross_stream_dependency` 为真；`timeline` 的四个起止时间必须精确等于 `(0,12)`、`(12,52)`、`(52,62)`、`(62,70)`，且 `pipelined_time_us == sequential_time_us == 70`。
- Q3：四个 `graph_suitability` 结果必须等于 `{"fixed_repeated": True, "unbucketed_variable_subgraph": False, "one_time_path": False, "branching_subgraph": False}`；传入非布尔值应抛 `ValueError`（可自己加一行验证）。
- Q4：六段起止必须等于 `(0,12)`、`(12,24)`、`(12,52)`、`(52,92)`、`(52,62)`、`(92,102)`；`serial_us == 124`、`ideal_pipeline_us == 102`、`ideal_saved_us == 22`；开销 3 时 `net_saved_us == 19`，22 时为 0，25 时为 −3；合并拷贝队列后 `pipelined_time_us == 124`；计算阶段改成 52 us 后 `contended_elapsed_us == 126`，且 `124 - 126 == -2`。
- Q1–Q4 的末尾都有 `print`，会打印每个任务的起止时间、独立队列理想完成时间、计入开销后的时间、合用搬运队列的 124 us，以及争用后的 126 us。

如果所有断言都通过，你会看到这些时间线逐行打印出来；只要有一段起止对不上，就说明同队列顺序或跨队列依赖有一边没算对。
