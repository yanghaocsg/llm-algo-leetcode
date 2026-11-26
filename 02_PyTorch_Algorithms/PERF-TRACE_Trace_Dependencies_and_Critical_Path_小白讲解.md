# PERF-TRACE. 执行轨迹依赖与关键路径 — 小白讲解

> 对应 notebook：[`PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb`](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb)

## 一句话总结
执行轨迹（trace）只告诉你"每个事件什么时候开始、什么时候结束、在哪个通道上跑"，不直接告诉你"谁必须等谁"。本节用 `normalize_trace_events`、`build_trace_dependencies`、`critical_path_report` 三个 TODO，把一张事件表补成依赖图，按拓扑序算出每个事件的就绪时刻、就绪后间隔和完成时刻，再回溯出通向任务结束的完成链。样例时间线的完成链是 `load → prepare → h2d → compute`，总长 12 ms，其中 0.5 ms 和 1.0 ms 是"已知前驱早已结束、事件却没开始"的待解释间隔；日志支路 5 ms 结束，不在关键路径上。Step 5 用 GPU 对照"每步同步"与"末尾同步"，登记结果是 25.89 ms 对 25.32 ms（1.023×），但 10 对样本只有 5 对更快，不能认定为稳定加速。

## 为什么需要它：痛点是什么
初学者做性能分析常犯一个错：打开 profiler，按总耗时排序，看到某个矩阵乘法占了 40%，就认为优化它能让整个任务快 40%。但被优化的算子可能在一个并行支路上——把它从 5 ms 压到 1 ms，另一条支路仍然要跑 12 ms，端到端一点没变。反过来，GPU 上出现一段 1 ms 的空闲，也不一定是某个 kernel 慢，可能是主机提交晚了、同步点放错了，或者它在等一个还没完成的拷贝。

生活类比：厨房出菜。凉菜、热菜、汤是三条并行支路，最后要等三样都齐了才能上桌。你把凉菜从 5 分钟压到 1 分钟，如果汤本来就要 12 分钟，客人不会早 4 分钟吃到。真正决定上菜时间的是"最长的那条必须等待的链"，以及链上那些"材料早就备好、厨师却没开始"的空档。

## 核心问题
- trace 里一个事件的持续时间、就绪时刻、完成时刻分别怎么定义？
- 为什么同一条串行通道内的先后关系不能强加到不同 CUDA 流上？
- "实际开始晚于前驱完成"的那段时间，为什么才是真正要解释的间隔？
- 完成链怎么从最后完成的事件回溯出来？为什么并行支路不能简单相加？
- 末尾同步一定更快吗？10 对样本 5 对更快说明了什么？
- 为什么说关键路径是"解释这一次运行"，而不是"预测改动后的性能"？

## 解决思路：一步步看懂

**第 1 步：先定义事件和通道。** 每个事件是一个字典，至少含 `event_id`、`resource`、`start_ms`、`end_ms`，可选 `predecessors`（显式前驱列表）。`resource` 表示一条**串行通道**：同一通道上的事件按时间先后执行，不同通道可以重叠。样例里的通道有 `cpu:0`（主机线程）、`copy:0`（拷贝流）、`cuda:0`（计算流）。注意 `resource` 不是"设备类别"：两块 GPU 或两条 CUDA 流都属于 GPU，但彼此不串行；把 `cuda:0` 和 `cuda:1` 当成同一条通道，就会凭空造出依赖。

依赖有两类：
- **数据依赖**：后继要读前驱的输出，必须等前驱完成。这类依赖需要显式写在 `predecessors` 里，因为光看时间线看不出来。
- **资源串行**：同一通道上的相邻事件，后一个必须等前一个结束。这类依赖可以由通道和时间自动补齐，不需要手写。

还有一种容易混淆的是**同步依赖**：比如 `host_sync` 等待设备工作完成。在 trace 里它表现为一个事件或一个同步点，建模时要把它当成"必须等谁完成"的边，而不是一条普通的数据边。

选粒度是建模前的第一步。profiler 的 trace 里，父级注解（比如 `record_function('sync_each_step.h2d_copy')`）会包含底下的真实 CUDA/CPU 算子；如果父级和子级都当事件，同一份工作会被算两次，持续时间和依赖边都会翻倍。Step 2 说"父级注解常包含内部算子，应先选定同一粒度"，Step 5 的实验也是先给每条路径的 `h2d_copy`、`cuda_matmul`、`host_sync` 各建一个注解，再用 `key_averages()` 按名字筛，避免混层。选粒度没有唯一答案：想看"同步位置的整体影响"就用路径级注解，想看"到底是哪个 kernel 慢"就下钻到算子级；关键是整张图统一，不能一半父级一半子级。

数据依赖和同步依赖的区别在归因时很关键。数据依赖说"我必须等你的输出"，间隔通常对应真实的流水线空档；同步依赖说"我必须等某段工作完成才能越过这个点"，间隔可能来自主机提交晚、同步点放错，甚至来自测量本身。只看到"compute 在 7 ms 开始、h2d 在 6 ms 结束"，不能直接断定这 1 ms 是拷贝慢；它可能是主机 6.5 ms 才提交 compute，也可能是同步点把两条本可重叠的流隔开了。Step 1 的表格把"事件区间""数据与同步依赖""最终完成链""单条件对照"分成四层，就是要逼你先分层再解释。

**第 2 步：三个时刻的公式。** 设事件 $i$ 的实际开始、结束、持续时间分别为 $s_i,e_i,d_i$，前驱集合为 $\mathcal P(i)$，前驱 $j$ 的完成时刻为 $C_j$，已知依赖决定的就绪时刻为 $r_i$：

$$
d_i=e_i-s_i,\qquad
r_i=\max\left(\{C_j:j\in\mathcal P(i)\}\cup\{0\}\right),\qquad
s_i\ge r_i.
$$

逐项解释：$d_i$ 是观测到的持续时间；$r_i$ 是"所有已知前驱都完成之后，这个事件最早可以开始的时刻"，没有前驱时取 0（窗口起点）；$s_i\ge r_i$ 是一条一致性约束——如果实际开始早于前驱完成，说明时间记录或依赖图至少有一个是错的，不能硬算。

就绪之后仍未开始的部分记为 $w_i$，完成时刻记为 $C_i$，窗口最终完成时刻为 $T$：

$$
w_i=\max(s_i-r_i,0),\qquad
C_i=r_i+w_i+d_i,\qquad
T=\max\left(\{C_i:i\in V\}\right).
$$

把 $w_i$ 代进 $C_i$ 可以看到：当 $s_i\ge r_i$ 时，$C_i=s_i+d_i=e_i$，也就是完成时刻等于观测到的结束时刻。那为什么还要绕一圈写 $r_i+w_i+d_i$？因为这样能把"等了多久"显式拆出来。$w_i$ 就是"已知依赖解释不了的那段空档"，它才是接下来要去查主机提交、资源争用、隐藏同步的入口。直接用 $e_i$ 当完成时刻，数字一样，但丢掉了归因。

**第 3 步：TODO 1 校验并规范化事件。** 参考答案两行：

```python
valid_interval = isfinite(start) and isfinite(end) and 0.0 <= start <= end
duration_ms = end - start
```

三个条件缺一不可：两个时刻都必须是有限值（`NaN` 和 `inf` 都不行）；起点不能为负（时间轴以窗口起点为 0）；终点不能早于起点。零时长是合法的，`start == end` 表示一个瞬时事件，`test_normalize_trace_events` 专门用一个 `point` 事件断言 `duration_ms == 0.0`。

在 TODO 之前，骨架已经检查了 `event_id` 非空且唯一、`resource` 非空、`predecessors` 不含自己且都在已知集合里。这些检查的顺序很重要：先确认所有 id 的集合，才能判断前驱是否未知；先确认 id 唯一，后面建图才不会出现同名节点。规范化后的输出按 `(start_ms, event_id)` 排序，`test_normalize_trace_events` 把样例倒序输入，要求输出顺序是 `['load', 'prepare', 'h2d', 'log', 'compute']`——`log` 的 4.2 ms 排在 `h2d` 的 4.0 ms 之后、`compute` 的 7.0 ms 之前。

**第 4 步：TODO 2 补串行边。** 参考答案：

```python
serialization_edges = {
    (str(left["event_id"]), str(right["event_id"]))
    for left, right in zip(ordered, ordered[1:])
}
edges.update(serialization_edges)
```

先把显式前驱变成边集合，再按 `resource` 分组，每组按 `(start_ms, event_id)` 排序，只连**相邻**两个事件。为什么只连相邻？因为串行是传递的：A→B、B→C 已经蕴含 A→C，把所有两两组合都加进去会让边数从 $O(n)$ 涨到 $O(n^2)$，还会让"前驱里最晚完成的是谁"变得难以判断。

补边之前有一个重叠检查：`if left["end_ms"] > right["start_ms"]: raise ValueError`。同一通道上两个事件时间重叠，说明"串行"这个前提被违反了，不能悄悄接受。注意用的是 `>` 而不是 `>=`：前一个在 4.0 ms 结束、后一个在 4.0 ms 开始是允许的（首尾相接），样例里的 `prepare` 4.0 结束、`h2d` 4.0 开始就是这种情况。

三个测试分别钉住三种情况：
- `test_serial_edges_without_explicit_predecessors`：两个事件都没有 `predecessors`，同一 `cpu:0` 上 `a`（0–1）在前、`b`（2–3）在后，必须补出 `[("a", "b")]`。这条测试防的是"只信显式前驱、忘了补串行边"。
- `test_independent_lanes_can_overlap`：`a` 在 `cuda:0` 跑 0–5，`b` 在 `cuda:1` 跑 0–3，时间重叠但通道不同，边集必须为空，`critical_path_ms` 是 5.0；把 `b` 改成 `cuda:0` 后必须抛 `ValueError`。这条测试防的是"看到同一块 GPU 就强加串行"。
- `test_trace_dependencies`：样例的边集必须恰好是 `{("load","prepare"), ("prepare","h2d"), ("prepare","log"), ("h2d","compute")}`，且没有重复边。`prepare→log` 既是显式前驱又是同通道串行边，集合去重后只算一条。

**第 5 步：TODO 3 重建完成链。** 骨架已经用 Kahn 算法做了拓扑排序，并在 `len(order) != len(event_map)` 时抛"图必须无环"。TODO 的三行：

```python
ready_at = max((completion[item] for item in predecessors[event_id]), default=0.0)
exposed_wait[event_id] = max(event["start_ms"] - ready_at, 0.0)
completion[event_id] = ready_at + exposed_wait[event_id] + event["duration_ms"]
```

`default=0.0` 处理没有前驱的事件；`max(..., 0.0)` 保证间隔非负（在一致性检查之前，负值会被压成 0，但随后 `start_ms < ready_at` 会抛错，不会静默通过）。拓扑序保证算到某个事件时，它所有前驱的 `completion` 都已经算好。

拿样例手算一遍（单位 ms）：

| 事件 | 通道 | 区间 | 前驱 | 就绪 $r_i$ | 间隔 $w_i$ | 完成 $C_i$ |
|---|---|---|---|---|---|---|
| load | cpu:0 | 0–2 | 无 | 0 | 0 | 2.0 |
| prepare | cpu:0 | 2.5–4 | load | 2.0 | 0.5 | 4.0 |
| h2d | copy:0 | 4–6 | prepare | 4.0 | 0 | 6.0 |
| compute | cuda:0 | 7–12 | h2d | 6.0 | 1.0 | 12.0 |
| log | cpu:0 | 4.2–5 | prepare | 4.0 | 0.2 | 5.0 |

关键路径的回溯规则是：从 `completion` 最大的事件出发，每次选"完成时刻最晚的前驱"作为父节点，直到没有父节点，再反转。样例里最大的是 `compute`（12.0），它的唯一前驱是 `h2d`（6.0），再到 `prepare`（4.0）、`load`（2.0），得到 `['load', 'prepare', 'h2d', 'compute']`，`critical_path_ms = 12.0`。`log` 的完成时刻是 5.0，虽然也在 `prepare` 之后，但不是任何最大完成事件的前驱，所以不在链上。

读这组数字要同时看三件事：
- **链上总时长 12 ms**，链上各事件持续时间之和是 2 + 1.5 + 2 + 5 = 10.5 ms，差额 1.5 ms 就是 0.5 + 1.0 两段间隔。间隔不是"某个事件慢"，而是"依赖已经满足、事件却没开始"。
- **`log` 支路 5 ms 结束**，比主链早 7 ms。把它优化到 0，整次任务仍然是 12 ms。`test_critical_path_report` 断言 `"log" not in report["critical_path"]`，就是在防"把并行支路也算进关键路径"。
- **所有事件持续时间之和是 11.3 ms**，比 12 ms 小。这说明"总耗时 = 各事件耗时相加"本身就是错的：并行事件不能相加，间隔又必须计入。

**第 6 步：为什么说是"解释"而不是"预测"。** 公式里的 $s_i$、$e_i$ 都是观测值，$r_i$ 是根据依赖图推出来的。你如果把 `compute` 从 5 ms 优化到 2 ms，新的完成时刻不保证就是 9：主机可能更早提交下一个任务，拷贝流可能和计算重叠，调度器可能换一种顺序。notebook 的 Step 3 特意写了"这里固定了实际开始时刻，因此减少等待后的新调度需要重新运行才能确认"。关键路径的用途是：告诉你这一次的时间花在哪、哪些间隔值得查；改完之后必须重新采一条 trace。

**第 7 步：GPU 实验的设计。** Step 5 要验证的问题是"主机同步是否把设备工作暴露到关键路径"。自变量是同步位置，只有两个取值：
- `sync_each_step = True`：每步 `output[0, 0].item()`，强制主机等这一步的矩阵乘法算完；
- `sync_each_step = False`：循环里只提交，循环结束后取一次 `.item()`，让主机尽量往前跑。

控制变量写得很细：2048×2048 的 FP32 矩阵、固定的 `host_left`（pinned memory）、`non_blocking=True` 的 H2D、同一个 `@` 运算、同样的步数。测量协议是 5 次预热、10 轮配对、每轮交替两条路径的顺序（`if repeat % 2: order = tuple(reversed(order))`），每条路径记录 10 步 CPU/CUDA 活动。交替顺序是为了抵消频率漂移、发热和缓存状态随轮次变化的系统性偏差；如果总是先跑 A 再跑 B，B 可能永远吃亏或永远占便宜。

计时和 profiler 采集是分开的：`measure_path` 只用 `time.perf_counter()` 和 `torch.cuda.synchronize()` 量墙钟，profiler 单独再跑一遍。原因是 profiler 本身有开销，混在一起测会把插桩成本算进延迟。trace 完整性用两组硬指标检查：
- `required_events` 必须包含 8 个注解：`sync_each_step`、`sync_each_step.h2d_copy`、`sync_each_step.cuda_matmul`、`sync_each_step.host_sync`，以及 `sync_at_end` 的四个同名项；
- CPU 注解计数必须匹配：两条路径的 `h2d_copy`、`cuda_matmul` 都是 `PROFILE_STEPS = 10` 次；`host_sync` 在逐步同步路径是 10 次，在末尾同步路径是 1 次——因为末尾同步只发生一次。这个 10 对 1 的差异，就是"同步位置"在 trace 里的直接证据。

**第 8 步：结果怎么读。** 5.4 登记的是 `rtx5070ti_trace_v3`（工作负载版本 v4）：逐步同步 25.89 ms，末尾同步 25.32 ms，`sync_at_end_speedup = 25.89 / 25.32 ≈ 1.023`，延迟下降约 2.20%。轨迹文件 150585 bytes，没有缺失事件，也没有 CPU 计数不匹配，所以 `decision = ready_for_trace_interpretation`，意思是"这份 trace 可以拿来做解释了"，不是"这个优化可以上线了"。

关键的一句是"10 对样本仅 5 对更快"。中位数低 2.2% 看起来是收益，但配对样本里一半赢一半输，符号检验上和抛硬币没有区别。可能的解释有很多：2048 的三次方矩阵乘法本身够长，单次同步的固定开销占比小；主机提交队列本来就不深，末尾同步也没能提前多少；测量噪声和 GPU 频率波动覆盖了 2% 的差异。所以 5.4 的结论是"当前用它观察同步位置与时间关系，策略采用仍需补测"，并指出还缺完整输出误差与自动 CUDA 依赖核对。要判断是否稳定，需要看逐轮配对差异和回归阈值，这正是相关阅读里 PERF-REGRESSION 那节的工作。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 执行轨迹（trace） | 一张记录每个事件起止时间与执行通道的表 |
| 执行通道（resource） | 一条串行队列，如主机线程、拷贝流、计算流 |
| 数据依赖 | 后继要读前驱的输出，必须等它完成 |
| 资源串行 | 同一通道上的相邻事件天然有先后 |
| 同步依赖 | 等某段工作完成才能越过同步点 |
| 就绪时刻 $r_i$ | 所有已知前驱都完成后，事件最早能开始的时刻 |
| 就绪后间隔 $w_i$ | 实际开始比就绪晚的那段，已知依赖解释不了 |
| 完成时刻 $C_i$ | 就绪时刻 + 间隔 + 持续时间 |
| 完成链（关键路径） | 从最后完成事件沿最晚前驱回溯出的那条链 |
| 拓扑排序 | 把有向无环图排成"前驱一定在后继之前"的顺序 |
| 暴露等待（exposed wait） | 没有被其他并行工作覆盖掉的等待时间 |
| 配对测量 | 两条路径在同一轮里各跑一次，逐轮比较 |

## 小白常见疑问

**Q1：为什么不能把每个事件的耗时加起来当作总时间？**
因为并行事件会重叠，而且事件之间还有间隔。样例里五个事件的持续时间之和是 2 + 1.5 + 2 + 5 + 0.8 = 11.3 ms，但实际完成是 12 ms；`log` 和 `h2d` 在 4.2–5.0 ms 之间是重叠的，简单相加会把重叠算两遍，又会漏掉 1.5 ms 间隔。正确做法是先建依赖图，再取所有完成时刻的最大值。

**Q2：`log` 明明也有 0.2 ms 的就绪后间隔，为什么不在关键路径上？**
因为关键路径只关心"限制最终完成"的那条链。`log` 在 5.0 ms 就结束了，比 `compute` 的 12.0 ms 早得多，把它压到 0 也不会让 `T` 变小。`exposed_wait_ms` 会给所有事件都算间隔，包括 `log` 的 0.2 ms；但 `critical_path` 只回溯最大完成事件的前驱链。间隔"存在"和间隔"在关键路径上"是两件事。

**Q3：两个事件在同一块 GPU 上、时间重叠，为什么不算错误？**
因为"GPU"是设备，不是串行通道。`cuda:0` 和 `cuda:1` 是两条独立的流，可以真正并行。`test_independent_lanes_can_overlap` 就是用这个例子断言边集为空、总完成时间是 5.0 ms。只有 `resource` 相同（同一条流或同一个线程）时，重叠才是错误。

**Q4：为什么 `completion` 不直接用 `end_ms`？两者数值不是一样吗？**
在数据合法时数值确实一样，但公式把"就绪 + 间隔 + 持续"三段拆开了，`exposed_wait_ms` 才有定义。更重要的是，这种写法把依赖图放到了主位：完成时刻是从前驱推出来的，不是从观测值抄来的。一旦 `start_ms < ready_at`，就能立刻发现"时间记录和依赖图矛盾"，而不是拿一个看起来正常的 `end_ms` 继续算下去。

**Q5：10 对样本 5 对更快，为什么不能算 1.023× 加速？**
1.023× 是两组中位数的比值，描述的是这 10 轮的总体位置；"5 对更快"描述的是逐对方向。中位数略低可能由少数几轮的大幅领先拉动，而另一半轮次其实是变慢的。要声称稳定加速，至少要看逐对差异的分布、符号检验或置信区间，并固定更多控制条件。notebook 因此把结论写成"收益不稳定，策略采用仍需补测"。

**Q6：同一通道里两个事件起点相同，排序稳定吗？**
稳定。`normalize_trace_events` 和 `build_trace_dependencies` 都用 `(start_ms, event_id)` 作为排序键，起点相同就按 id 字典序，结果与输入顺序无关。这也是为什么 `event_id` 必须唯一且非空：它既是节点标识，也是并列时的决胜键。如果没有这个第二键，同一毫秒的两个事件顺序会依赖输入顺序，补出来的串行边和关键路径就可能在不同运行间漂移。

**Q7：`.trace.json` 和摘要 `.json` 各有什么用？**
`prof.export_chrome_trace` 写出的 `.trace.json` 是给 Chrome Trace Viewer / Perfetto 看的原始事件流，包含每个注解和算子的详细区间；摘要 `.json` 只放统计量、计时样本和完整性检查。两者分开保存，摘要可以快速比较，原始轨迹可以下钻。5.3 读取时先看摘要里的 `missing_required_events` 和 `cpu_annotation_count_mismatches`，确认采集完整，再打开 `.trace.json` 看具体时间线。

## 动手验证：怎么知道自己做对了
`run_perf_trace_tests()` 依次跑 8 个测试，全过才打印 `✅ 执行事件、串行通道、完成链与异常边界测试通过`。测试用 `_trace_fixture()` 构造固定的五事件时间线，用 `_assert_value_error` 统一要求非法输入抛 `ValueError`，而不是返回一份"看起来还行"的部分报告。

- `test_normalize_trace_events`：倒序输入后 `event_id` 顺序是 `['load', 'prepare', 'h2d', 'log', 'compute']`；`load` 的 `duration_ms == 2.0`；零时长事件 `point` 的 `duration_ms == 0.0`；空列表返回 `[]`。
- `test_invalid_trace_events`：重复事件、空 `event_id`、空 `resource`、`end_ms = -1.0`、`start_ms = nan`、`end_ms = inf`、未知前驱、自己当前驱，八种输入全部抛 `ValueError`。
- `test_trace_dependencies`：边集恰好等于 `{("load","prepare"), ("prepare","h2d"), ("prepare","log"), ("h2d","compute")}`，且 `len(edges) == len(set(edges))`。
- `test_serial_edges_without_explicit_predecessors`：没有显式前驱的同通道两事件，必须补出 `[("a","b")]`。
- `test_independent_lanes_can_overlap`：`cuda:0` 与 `cuda:1` 重叠时边集为空、`critical_path_ms == 5.0`；改成同一通道后抛 `ValueError`。
- `test_critical_path_report`：`critical_path == ['load','prepare','h2d','compute']`，`critical_path_ms == 12.0`，`exposed_wait_ms['prepare'] == 0.5`，`exposed_wait_ms['compute'] == 1.0`，`"log"` 不在链上。
- `test_dependency_failures`：未知边 `[("missing","load")]`、二元环 `[("load","prepare"),("prepare","load")]`、以及前驱 5.0 ms 才完成但后继 4.0 ms 就开始，三种情况都抛 `ValueError`。
- `test_empty_trace_report`：空事件、空边返回 `{"critical_path": [], "critical_path_ms": 0.0, "exposed_wait_ms": {}}`；空事件但带边必须抛错，不能返回零报告。

GPU 部分默认 `RUN_MODE = "dry_run"`，只打印"未执行 GPU Trace"，不产生文件。改成 `real_gpu` 需要唯一 `RUN_LABEL`（不能是 `default`）、`PERF-CUDA-MECHANISM` 预检通过、CUDA 可用，结果写到 `benchmarks/results/perf_trace/perf-trace-critical-path_<RUN_LABEL>.json` 和同名 `.trace.json`；同标签已存在会抛 `FileExistsError`，不覆盖旧结果。5.3 用 `read_performance_result` 从原始计时样本重算统计，并打印两条路径的中位时间、全部计时样本、8 个注解的 CPU/GPU 计数、缺失事件和计数不匹配。通过条件就是 5.4 表格里的两条：轨迹文件非空、8 个必需事件齐全、CPU 计数与 `PROFILE_STEPS` 一致；性能结论则要自己看逐对样本，不能只看 1.023×。
