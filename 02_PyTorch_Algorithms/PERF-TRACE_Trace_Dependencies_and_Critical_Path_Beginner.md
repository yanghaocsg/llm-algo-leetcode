# PERF-TRACE. Trace Dependencies and Critical Path Beginner | 执行轨迹依赖与关键路径 · 小白入门

> 配套 notebook：[PERF-TRACE. 执行轨迹依赖与关键路径](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

程序慢，往往不是算得慢，而是有地方在干等。这本 notebook 教你从一份“每件事几点开始、几点结束”的记录里，找出哪几段执行和等待连成了拖长总时间的那条链。

## 为什么需要它

你用性能分析工具录下一段程序，看到 GPU 利用率不高。这只能说明设备没有一直忙碌，没说它在等什么。主机提交太晚、数据还在复制、同步卡住了，都会让计算晚开始。

notebook 的测试里有 5 个事件，单位是毫秒：读取 0–2、准备 2.5–4、复制 4–6、计算 7–12，另有一条日志支路在 4.2–5。盯着这条时间线，会看到两个空洞：

- 读取在 2 ms 结束，准备到 2.5 ms 才开始，中间空了 0.5 ms。
- 复制在 6 ms 结束，计算到 7 ms 才开始，中间空了 1 ms。

光看图，说不清空洞是在等数据、在排队，还是程序自己提交晚了。几种常见的误判是：只因为两个事件在图上挨得近，就认定后者在等前者；把整块 GPU 或所有 CPU 线程当成一条队，看到时间重叠就觉得不可能；把空洞直接叫作“同步开销”或“I/O 开销”。

这本 notebook 的做法是：先把“谁必须等谁”写成一张依赖图，从最后完成的事件往回追出一条完成链，再算出链上每个事件“条件都满足了却还没开始”的间隔。Step 1–4 在 CPU 上把这套计算写对；真实 GPU 轨迹怎么采集、怎么对照，放在可选的 Step 5。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 执行轨迹（Trace） | 性能分析工具导出的一份记录，里面每个事件（一次读取、一次复制、一次计算）都带着开始时刻、结束时刻和所在的通道 |
| 执行通道（`resource`） | 按顺序执行、不能重叠的一串事件所在的位置，比如一个 CPU 线程或一个串行 CUDA 流；不同通道的事件可以在时间上重叠 |
| 前驱 / 后继（predecessor / successor） | 后继必须等前驱先完成。计算要等复制完成，复制就是计算的前驱 |
| 依赖 | 前驱必须先于后继的理由。notebook 分三种：数据依赖（后者要用前者的输出）、资源串行（排在同一条通道里）、同步依赖（要等同步点前的工作做完） |
| 就绪时刻（`ready_at`） | 所有已知前驱都完成的时刻；没有前驱时记为 0 |
| 暴露等待（`exposed_wait`） | 实际开始比就绪时刻晚的那一段，也就是“条件齐了却还没开始” |
| 完成链（关键路径，critical path） | 从最后完成的事件往回追“哪个前驱完成得最晚”得到的一串事件，它的结束时刻就是最终完成时刻 |

## 核心思路

输入是一组事件，每个事件带有开始时刻 `start_ms`、结束时刻 `end_ms`、所在的通道 `resource`，以及显式写出的前驱 `predecessors`。TODO 1 先检查时间区间合不合法，再算出持续时间。TODO 2 把同一通道里相邻的两个事件连成“串行边”，和显式前驱合并成一张依赖图。TODO 3 按依赖图算出每个事件的就绪时刻、暴露等待和完成时刻，再从最后完成的事件往回追出完成链。输出是完成链、它的毫秒长度，以及每个事件的暴露等待。

```text
d_i = e_i - s_i                      # 持续时间
r_i = max(C_j, j 取遍 i 的所有前驱)    # 就绪时刻；没有前驱时取 0
w_i = max(s_i - r_i, 0)              # 暴露等待
C_i = r_i + w_i + d_i                # 完成时刻
T   = 所有事件 C_i 的最大值            # 最终完成时刻
```

`s_i` 和 `e_i` 是实际开始和结束时刻，`d_i` 是持续时间，`C_j` 是前驱 `j` 的完成时刻，`r_i` 是就绪时刻，`w_i` 是暴露等待，`C_i` 是完成时刻，`T` 是最终完成时刻。单位都是毫秒，时间从观测窗口的起点 0 算起。

下面用 notebook 测试里的时间线算一遍。为了简短，代码只写了显式前驱，并且事件已经按先后排好。

```python
events = [
    {"id": "load",    "start": 0.0, "end": 2.0,  "pred": []},
    {"id": "prepare", "start": 2.5, "end": 4.0,  "pred": ["load"]},
    {"id": "h2d",     "start": 4.0, "end": 6.0,  "pred": ["prepare"]},
    {"id": "compute", "start": 7.0, "end": 12.0, "pred": ["h2d"]},
    {"id": "log",     "start": 4.2, "end": 5.0,  "pred": ["prepare"]},
]
pred = {e["id"]: e["pred"] for e in events}
dur = {e["id"]: e["end"] - e["start"] for e in events}
ready, wait, done = {}, {}, {}
for e in events:                      # 这个列表恰好按先后排列
    ready[e["id"]] = max([done[p] for p in e["pred"]], default=0.0)
    wait[e["id"]] = max(e["start"] - ready[e["id"]], 0.0)
    done[e["id"]] = ready[e["id"]] + wait[e["id"]] + dur[e["id"]]
    print(e["id"], "就绪", ready[e["id"]], "等待", f"{wait[e['id']]:.1f}", "完成", done[e["id"]])

tail = max(done, key=done.get)
chain = [tail]
while pred[tail]:                     # 往回追：每步挑完成得最晚的前驱
    tail = max(pred[tail], key=done.get)
    chain.append(tail)
chain.reverse()
print(chain, done[chain[-1]], sum(wait[i] + dur[i] for i in chain))
```

```text
load 就绪 0.0 等待 0.0 完成 2.0
prepare 就绪 2.0 等待 0.5 完成 4.0
h2d 就绪 4.0 等待 0.0 完成 6.0
compute 就绪 6.0 等待 1.0 完成 12.0
log 就绪 4.0 等待 0.2 完成 5.0
['load', 'prepare', 'h2d', 'compute'] 12.0 12.0
```

读取没有前驱，就绪时刻是 0，一开始就做，没有等待。准备要等读取：读取 2 ms 完成，准备 2.5 ms 才开始，暴露等待是 0.5 ms。复制紧跟着准备的结束时刻开始，等待是 0。计算 6 ms 就就绪了，7 ms 才开始，暴露等待是 1 ms。日志同样等到了准备完成，但它在 5 ms 就结束了。

从最后完成的计算往回追：计算的前驱是复制，复制的前驱是准备，准备的前驱是读取。于是完成链是 `load → prepare → h2d → compute`，长度 12 ms。链上 5 段执行时间加 2 段等待，合计正好也是 12.0。日志支路虽然也有 0.2 ms 的等待，但它 5 ms 就结束，没有延长 12 ms 的最终完成，所以不在链上。

还有两点要记住。第一，依赖图不能有环，也不能出现后继早于前驱完成就开始的情况，遇到了要报错。第二，0.5 ms 和 1 ms 只是“已知前驱解释不了的间隔”，可能来自隐藏依赖、提交延迟、排队或资源争用，要再靠单变量对照（同一工作量下只改一个提交或等待条件）去验证。这套算法固定了实际开始时刻，只解释已经发生的记录，不预测删掉等待之后会快多少。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从事件时间线找到等待关系：要看事件区间、数据与同步关系、完成链，再用单变量对照验证 | 读 |
| Step 2 | 串行通道和数据依赖怎样限制开始时间：三种依赖，以及就绪时刻和完成时刻的公式 | 读，对照上面的公式 |
| Step 3 | 完成链上哪些间隔需要继续解释：执行时间和等待间隔分开算，每种观察能下什么结论 | 读，对照上面的小例子 |
| Step 4 · TODO 1 | 校验时间区间 | 在 `normalize_trace_events` 里写 2 行：`valid_interval` 和 `duration_ms` |
| Step 4 · TODO 2 | 补齐资源串行边 | 在 `build_trace_dependencies` 里写 2 行：同一通道相邻事件的边集合，并入 `edges` |
| Step 4 · TODO 3 | 重建完成链与等待 | 在 `critical_path_report` 里写 3 行：`ready_at`、`exposed_wait[event_id]`、`completion[event_id]` |
| 测试 | 检查排序和零时长、非法事件被拒绝、串行边与并行流、完成链和 0.5/1 ms 间隔、空图、环和未知边 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用 `torch.profiler` 记录 CPU 准备、主机到设备复制、CUDA 计算和主机同步，比较逐步同步与末尾同步两条路径的时间线 | 需要单张 CUDA GPU 和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，改成 `real_gpu` 后才生成真实 Trace |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不生成真实 Trace。

## 容易踩的坑

- **把整块 GPU 当成一条通道**：串行边只能加在同一 `resource` 的相邻事件之间。不同的 CUDA 流即使时间重叠也合法，不能强加先后；同一通道里两个事件时间重叠才是错误，要报错。
- **只凭时间上挨得近就认定依赖**：数据依赖的依据是后继使用了前驱的输出，不是两者在时间线图上相邻。
- **把就绪后的间隔直接叫作同步或 I/O**：它只是当前依赖图下解释不了的残余间隔，可能来自隐藏依赖、提交延迟、排队或资源争用，要靠单变量对照验证。
- **时间不合法还算持续时间**：两个时刻必须是有限数、非负，并且结束不早于开始。`nan`、无穷、结束早于开始都要报错；开始等于结束的零时长事件可以保留。
- **图里有环还接着算**：环说明先后关系自相矛盾，必须报错。后继早于前驱完成就开始，同样要报错，不能返回一份看似有效的报告。

## 读完后自查

- [ ] 能说出数据依赖、资源串行、同步依赖三种关系各自的依据。
- [ ] 前驱在 6 ms 完成、事件 7 ms 开始、持续 5 ms，能算出它的就绪时刻、暴露等待和完成时刻。
- [ ] 能解释日志支路为什么不在完成链上。
- [ ] 能说明为什么 1 ms 的暴露等待不能直接说成同步或 I/O。

## 下一步

- 打开 notebook：[PERF-TRACE. 执行轨迹依赖与关键路径](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb)
- 需要补基础：[Part 00 · 17 PyTorch 性能分析基础](../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)、[Part 01 · 17 CUDA 流与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb)
- 想读深一点：[性能分析 · 时间线与关键路径](../topic_discussion/performance_analysis/03_time_breakdown_and_trace_reading.md)
- 做完之后：[PERF-ROOFLINE. 算术强度与资源上界 · 小白入门](./PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling_Beginner.md)
