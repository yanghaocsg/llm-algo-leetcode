# 17. CUDA Stream 与异步执行 — 小白讲解

> 对应 notebook：[`17_CUDA_Stream_and_Asynchrony.ipynb`](./17_CUDA_Stream_and_Asynchrony.ipynb)

## 一句话总结
本节用三个时间账本把「异步就能重叠」这句口号拆开：`stream_timeline` 比较独立与相依两种依赖（10 ms 搬运 + 20 ms 计算 + 4 ms 同步，独立时 24 ms、相依时 34 ms，隐藏 10 ms）；`overlap_candidate` 逐项撤掉锁页内存、独立数据、并发资源三个条件（缺一项隐藏时间归零，1 ms 事件开销后净省 9 ms，缺条件时变成 −1 ms）；`classify_timeline` 把观察窗口划成互斥的计算、暴露搬运、暴露同步、空闲四类，用占比判断下一步该查什么。

## 为什么需要它：痛点是什么
新手最常见的误解是「开了多个 stream 就会自动并行」。实际上同一个 stream 内严格按提交顺序执行，不同 stream 之间还要满足数据依赖和资源条件；把下一批数据的搬运和当前批计算放在不同 stream，只是「有机会」重叠，不是「一定」重叠。另一个常见错误是算时间时把多个 stream 的完整事件时长直接相加，或者把 24 ms 当成「单批请求从 34 ms 加速到 24 ms」。

生活类比：洗衣机和烘干机。洗下一批（搬运）可以和烘当前批（计算）同时进行，但前提是两批衣服互相独立、两台机器都有空；如果烘干机必须等洗衣机把水排干（数据依赖），就只能串行。把衣服从洗衣机挪到烘干机（同步/事件）也要花时间，而且这个动作可能把省下的时间吃光。

## 核心问题
- 同一 stream 内和不同 stream 之间，约束有什么不同？
- 搬运和计算什么时候能重叠，最多能藏掉多少时间？
- 事件（Event）、锁页内存（pinned memory）、缓冲区生命周期各解决什么问题？
- 事件开销大于或等于隐藏时间时，净收益会变成什么？
- 怎样从一条时间线判断瓶颈在计算、搬运还是等待？

## 解决思路：一步步看懂

**第 1 步：只改依赖，比较完成时间（Q1）。** `stream_timeline(copy_ms, compute_ms, sync_ms, independent=True)` 的模型只有两个阶段加一个末尾同步：

```python
serial_ms = copy_ms + compute_ms + sync_ms
elapsed_ms = (max(copy_ms, compute_ms) if independent else copy_ms + compute_ms) + sync_ms
"hidden_ms": serial_ms - elapsed_ms
"speedup": serial_ms / elapsed_ms if elapsed_ms else None
```

独立时，两段并行，总时长取较长的那段，再加同步；相依时只能串行相加。代入 `copy_ms=10`、`compute_ms=20`、`sync_ms=4`：

- 独立：$\max(10,20) + 4 = 24$ ms，串行基线 34 ms，隐藏 10 ms；
- 相依：$10 + 20 + 4 = 34$ ms，隐藏 0。

`assert independent["serial_ms"] == 34`、`independent["estimated_elapsed_ms"] == 24`、`independent["hidden_ms"] == 10`、`dependent["estimated_elapsed_ms"] == 34`、`dependent["hidden_ms"] == 0`。隐藏量就是 `min(copy, compute)`——被较长那段盖住的部分。同步 4 ms 在两种情况下都串行加在末尾，因为它是最后一道等待。

两个边界：`copy_ms=0` 时串行 24 ms、完成 24 ms，`speedup == 1`；三段全 0 时 `elapsed_ms` 为 0，`speedup` 返回 `None`，避免除零。校验要求三个阶段都是有限非负实数，`independent` 必须是布尔。

这里必须读清楚「24 ms 与 34 ms」的口径。notebook 的 markdown 明确说这是**跨批窗口**：假设 A 批输入已经就绪，下一批的 10 ms 搬运与当前批的 20 ms 计算重叠。它不能直接当成「单批请求快了 10 ms」的结论；完整两批工作量的对照在 29 节计算。把跨批流水线的稳态时间和单批延迟混为一谈，是这类分析里最常见的错。

**第 2 步：重叠需要哪些条件（Q2）。** `overlap_candidate(copy_ms, compute_ms, event_overhead_ms=0.0, pinned_memory=True, independent=True, concurrent_resources=True)` 把三个前提做成布尔开关：

```python
conditions = {"pinned_memory": pinned_memory, "independent": independent,
              "concurrent_resources": concurrent_resources}
missing = [name for name, value in conditions.items() if not value]
hidden = min(copy_ms, compute_ms) if not missing else 0.0
"estimated_net_saved_ms": hidden - event_overhead_ms
```

逻辑很硬：只要缺任何一项，理想隐藏时间直接归零，不是打折。`pinned_memory` 指锁页内存——固定在主机物理内存、可供异步传输直接使用的缓冲区；普通可分页内存的搬运可能要先拷贝到临时锁页缓冲，异步性和带宽都会受影响。`independent` 指计算不依赖这次搬运的数据。`concurrent_resources` 指设备能同时提供搬运和计算所需的资源（例如拷贝引擎与计算单元不打架）。

以 10 ms 搬运、20 ms 计算、1 ms 事件开销为例：条件齐全时 `ideal_hidden_ms == 10`、`estimated_net_saved_ms == 9`。逐个把条件置 `False`，都得到 `ideal_hidden_ms == 0`、`net == -1`、`missing_conditions` 恰好是那一个名字。事件开销等于隐藏时间（10 ms）时净省 0；开销更大时，比如 `copy_ms=1, compute_ms=3, event_overhead_ms=3`，净省 $1-3 = -2$ ms——重叠不但没赚，还倒贴。

事件在这里的角色是「设备侧的进度标记」：搬运流在传输后 `record` 一个事件，计算流 `wait` 这个事件再使用输入。这样等待只约束相关消费者，主机不必阻塞到整个设备空闲。但主机如果立刻要读回传结果，仍然得确认对应传输完成。缓冲区寿命同样要管：主机输入缓冲在传输完成前必须保持有效且内容不变，设备输入缓冲要保留到消费者用完；只有「生命周期」和「输入就绪」都满足，才能安全复用。

开销等于或超过隐藏时间时，净节省可能为零或负数——notebook 用 `-1` 和 `-2` 两个断言把这句话量化了。异步不是免费的，事件记录、流同步、主机提交都有成本。

**第 3 步：把时间线切成互斥区间（Q3）。** `classify_timeline(kernel_ms, exposed_copy_ms, exposed_sync_ms, idle_ms)` 只接受**已经去掉重叠**的区间长度：

```python
parts = {"kernel": kernel_ms, "copy": exposed_copy_ms, "sync": exposed_sync_ms, "idle": idle_ms}
total_ms = sum(parts.values())
dominant = max(parts, key=parts.get)
fractions = {name: value / total_ms for name, value in parts.items()}
```

四个字段分别是：内核计算、暴露出来的搬运（没被计算盖住的部分）、暴露的同步等待、完全空闲。`exposed` 这个词很关键——已经和计算重叠的搬运不算，否则会和 `kernel` 重复计数。校验要求四个值都是有限非负实数，总和为 0 时抛 `ValueError("时间线总时间须大于0")`。

四个场景的占比读法：

| 场景 | kernel | copy | sync | idle | 总长 | 最长类 |
|---|---|---|---|---|---|---|
| `kernel_longest` | 18 | 2 | 1 | 1 | 22 | kernel |
| `sync_longest` | 6 | 2 | 12 | 1 | 21 | sync |
| `copy_longest` | 5 | 14 | 2 | 1 | 22 | copy |
| `idle_longest` | 1 | 1 | 1 | 9 | 12 | idle |

`kernel_longest` 的 `fractions["kernel"] == 18 / 22`，四个占比之和为 1（用 `math.isclose` 断言）。最长类只是「下一项检查的线索」，不是已确认的瓶颈：计算时间长要回去核对计算量和效率；空闲时间长要追查输入是否就绪、依赖是否解开、主机是否没提交；同步时间长要看事件和流同步是否过多；暴露搬运长要看带宽、锁页内存和重叠条件。

还有一个并列的细节：`max(parts, key=parts.get)` 在并列时返回字典顺序里的第一个。`classify_timeline(kernel_ms=1, exposed_copy_ms=1, exposed_sync_ms=0, idle_ms=0)` 的 kernel 和 copy 都是 1，结果 `dominant == "kernel"`，因为 `kernel` 在 `parts` 里先出现。这不是「计算更重要」，只是实现顺序。

**第 4 步：这套估算的边界。** 所有函数都是给定布尔和毫秒数的算术模型，不执行真实事件记录或传输。真实系统里，重叠是否发生要看执行轨迹；估算的收益要用固定输入、重复端到端计时核对。notebook 在「继续检查重叠条件」里说得很清楚：分别记录输入就绪、任务提交、设备完成时刻，并注明所用资源；完整依赖链仍需等待，独立任务是否真正重叠要从轨迹判断。这里讨论的是主机与设备之间的跨批搬运；单个内核内部从全局内存预取到共享内存的流水是另一套机制，见 Part04·09。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| CUDA Stream | 设备上的操作队列，同一队列内按提交顺序执行 |
| 异步提交 | 主机把任务交给设备后立刻返回，不等待完成 |
| 事件（Event） | 流里的进度标记，用来让别的流等待某个点 |
| 锁页内存（pinned） | 固定在主机物理内存、可被异步传输直接使用的缓冲 |
| 数据依赖 | 后一个任务必须等前一个任务产出的数据 |
| 并发资源 | 设备能同时提供搬运和计算所需的硬件资源 |
| 隐藏时间 | 被另一段时间盖住、不增加总时长的部分 |
| 暴露时间（exposed） | 没被盖住、真实计入总时长的那部分 |
| 时间线占比 | 互斥区间长度 ÷ 总长度，用来定位下一项检查 |
| 缓冲区生命周期 | 缓冲在被使用完之前不能释放或改写 |

## 小白常见疑问
**Q1：用了不同 stream，搬运和计算就一定会重叠吗？**
不一定。`overlap_candidate` 要求三个条件同时成立：锁页内存、数据独立、资源可并发。缺任何一项，`ideal_hidden_ms` 直接是 0。不同 stream 只是「允许」并发，是否发生还取决于依赖和资源。

**Q2：为什么缺了锁页内存，隐藏时间就变成 0，而不是少一点？**
这是本节刻意采用的布尔模型：把三个前提当作必要条件，缺一项就不建模部分重叠。真实系统里非锁页内存可能仍能异步，但效率更低、行为更依赖驱动；notebook 不去猜那个中间值，而是用 0 表示「这个模型下不成立」。读结论时要记住这是保守近似。

**Q3：事件开销 10 ms、隐藏时间也是 10 ms，净省为什么是 0？**
`estimated_net_saved_ms = hidden - event_overhead_ms = 10 - 10 = 0`。重叠省下的时间刚好被事件记录与等待吃掉。开销更大时会是负数，例如 `copy_ms=1, compute_ms=3, event_overhead_ms=3` 得 −2；这时候应该减少同步、换更大批次，或者重新审视重叠是否值得。

**Q4：24 ms 能不能说成「比 34 ms 加速了 1.42 倍」？**
不能直接这么说。24 ms 是跨批窗口的稳态估算，前提是 A 批输入已就绪，只把「下一批搬运」和「当前批计算」重叠；它描述的是流水线稳态，不是单批请求的端到端延迟。单批延迟还要看输入就绪、提交、输出回传；两批完整工作量在 29 节算。

**Q5：`classify_timeline` 里 kernel 和 copy 一样长，为什么 `dominant` 选 kernel？**
因为 `max(parts, key=parts.get)` 遇到并列返回最先出现的那一项，而 `parts` 的插入顺序是 `kernel, copy, sync, idle`。这只是展示上的约定，不代表 kernel 更值得优化；遇到并列应该两类都查。

## 动手验证：怎么知道自己做对了
没有测试入口，也没有保存输出；所有检查是单元末尾的 `assert`，失败抛 `AssertionError`。

- Q1 断言 `independent["serial_ms"] == 34`、`estimated_elapsed_ms == 24`、`hidden_ms == 10`，`dependent["estimated_elapsed_ms"] == 34`、`hidden_ms == 0`；`copy_ms=0` 时 `speedup == 1`，全 0 时 `speedup is None`；打印两种依赖的完成时间与隐藏量。
- Q2 断言条件齐全时 `ideal_hidden_ms == 10`、`estimated_net_saved_ms == 9`；三个条件逐个置 `False` 都得到 `ideal_hidden_ms == 0`、`net == -1`、`missing_conditions == [condition]`；开销 10 ms 时净省 0；`copy=1, compute=3, overhead=3` 时净省 −2。
- Q3 断言四个场景的 `dominant` 分别是 `kernel`、`sync`、`copy`、`idle`，每个 `fractions` 之和 `isclose(..., 1.0)`；`kernel_longest` 的 `total_ms == 22`、`fractions["kernel"] == 18 / 22`；并列时 `dominant == "kernel"`。

可以改 `sync_ms` 从 4 到 0，观察独立与相依的差距是否仍为 10 ms；再把 `event_overhead_ms` 调到 9，净省恰好 1 ms。notebook 没有保存执行输出，所有数值以你自己跑出来的结果为准。
