# PERF-IO. Artifact State and Output IO Critical Path | 模型产物、状态传输与输出回写的 I/O 关键路径 · 小白入门

> 配套 notebook：[PERF-IO. 模型产物、状态传输与输出回写的 I/O 关键路径](../PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

读写数据花的时间，并不都会拖慢程序，有的能和计算同时进行。这本 notebook 教你算三件事：一次读写要多久，其中多少真的被等到，写回队列满了会怎样拖住生产。

## 为什么需要它

你在性能分析结果里看到读写数据花了不少时间，想去优化它。但读写不是都值得优化：模型文件一般只在启动时读一次，输入每一批都要搬，运行时状态会在设备之间迁移，输出和检查点还要写回去。它们发生的频率不同，对完成时间的影响也不同。更要紧的是，只有“后面的工作必须等它”的那部分读写，才会拖长完成时间。

看一个具体的数：一次读写要 12 毫秒，旁边有 5 毫秒的独立工作可以和它同时开始。读写结束时，那 5 毫秒的工作也做完了，真正多出来的等待只有 7 毫秒（下面的代码会算出来）。反过来，如果后面的计算必须等这次读写做完，12 毫秒就全部要等。

不区分这些，容易出三种错：

- 把一次读写的时间直接加到总时间里，不管它是不是和计算重叠。
- 小对象、高频次的读写只数字节，忘了每次操作都有的固定开销。
- 输出写回的队列满了，生产者被卡住，却只盯着带宽看。

这本 notebook 把问题拆成三个可以算的小块：一次读写要多久（服务时间），其中有多少真的被等到（暴露尾部），队列满了以后生产者被推迟多久（背压）。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| I/O 对象 | 被读写的东西，分四类：输入（每批次搬运）、模型产物（启动时加载的模型文件）、运行时状态（在设备间迁移的状态）、输出（写回的词元、日志、检查点） |
| 服务时间（`service_ms`）/ 固定开销（`fixed_latency_ms`） | 一次读写从开始到做完的时间：字节除以带宽，再加每次操作都有的固定开销。搬 0 字节也可能有固定开销 |
| 重叠窗口（`overlap_window_ms`） | 和这次读写同时开始、可以并行去做的独立工作有多长 |
| 阻塞依赖（`blocking_dependency`） | 后面的计算必须等这次读写做完才能开始，没有可以并行的独立工作 |
| 暴露尾部（`exposed_ms`） | 服务时间里没被并行工作盖住、后面真正要等的那一段 |
| 有限队列 / 队列容量（`queue_capacity`） | 输出写回之前排队的地方。容量包括正在写的对象和排队等待的对象 |
| 背压（backpressure，`queue_wait_ms`） | 队列满了，生产者要等最早的对象写完才能入队，这段等待就是背压 |

## 核心思路

输入有三组数：字节量、带宽和固定开销，用来算一次读写的服务时间；服务时间、重叠窗口和是否阻塞，用来算暴露尾部；每个对象的生产间隔、写回的服务时间和队列容量，用来模拟队列。TODO 1 算服务时间。TODO 2 先算能被并行工作盖住的部分，再从服务时间里减掉，剩下的就是暴露尾部。TODO 3 逐个对象推进队列，算出入队时刻、背压等待和服务开始时刻。骨架里还有一个账本函数 `io_critical_path_ledger`，按对象类别和发生次数把这些量累加起来。

```text
S = a + D / (10^9 * B) * 10^3     # 服务时间（毫秒）。D 是字节数，B 是带宽（GB/s，十进制），a 是固定开销（毫秒）

E = S                # 后续计算必须先等这次读写：全部暴露
E = max(0, S - W)    # 有独立工作和读写同时开始、可以并行：只暴露超出窗口 W 的部分

r_i = u_(i-1) + p_i  # 第 i 个输出对象的就绪时刻：上次入队以后，再生产 p_i
u_i = r_i            # 队列没满，立刻入队
u_i = 最早占用者的完成时刻   # 队列已满，等它完成才能入队
v_i = max(u_i, d_(i-1))      # 服务开始：入队了，还要等前一个写完（只有一条写回通道）
d_i = v_i + s_i              # 服务完成
q_i = u_i - r_i              # 背压等待
```

`D`、`B`、`a` 分别是字节量、带宽和固定开销，`S` 是服务时间，`W` 是重叠窗口，`E` 是暴露尾部。队列部分：`r_i` 是就绪时刻，`u_i` 是实际入队时刻，`v_i` 是服务开始时刻，`d_i` 是服务完成时刻，`s_i` 是这个对象的服务时间，`p_i` 是生产它的耗时，`q_i` 是背压等待。

下面的代码是三个 TODO 的最小版本，数字都取自 notebook 的测试。

```python
def service_ms(num_bytes, bandwidth_gbps, fixed_ms=0.0):    # 带宽单位 GB/s，1 GB = 10^9 字节
    return num_bytes / (bandwidth_gbps * 1_000_000_000) * 1000 + fixed_ms

print(service_ms(1_000_000_000, 1.0, 2.0), service_ms(0, 10.0, 0.5))

def exposed_ms(service, window, blocking):                  # blocking 为 True 表示后续计算必须等这次读写
    hidden = 0.0 if blocking else min(service, window)
    return service - hidden

print(exposed_ms(8.0, 20.0, True), exposed_ms(8.0, 20.0, False), exposed_ms(12.0, 5.0, False))

def queue_waits(intervals, services, capacity):             # 容量包括正在写的对象
    departures, waits, clock = [], [], 0.0
    for interval, service in zip(intervals, services):
        ready = clock + interval                            # 这个对象就绪的时刻
        departures = [d for d in departures if d > ready]   # 已写完的对象离队
        admitted = min(departures) if len(departures) >= capacity else ready   # 满了就等最早写完的
        waits.append(max(0.0, admitted - ready))
        start = max([admitted, *departures])                # 只有一条通道：等前面写完
        departures = [d for d in departures if d > admitted]
        departures.append(start + service)
        clock = admitted                                    # 入队以后生产者才继续生产
    return waits

print(queue_waits([2.0, 2.0, 2.0], [5.0, 5.0, 5.0], 1), queue_waits([2.0, 2.0, 2.0], [5.0, 5.0, 5.0], 2))
```

```text
1002.0 0.5
8.0 0.0 7.0
[0.0, 3.0, 3.0] [0.0, 0.0, 1.0]
```

第一行是服务时间：10^9 字节按 1 GB/s 要搬 1000 毫秒，加上 2 毫秒固定开销是 1002.0；0 字节只剩 0.5 毫秒的固定开销。第二行是暴露尾部：服务时间 8 毫秒，后续计算必须等它时是 8.0；旁边有 20 毫秒的独立工作可以同时做时，8 毫秒全被盖住，暴露 0.0；服务 12 毫秒、窗口只有 5 毫秒时，盖住 5 毫秒，暴露 7.0。

第三行是背压：生产者每 2 毫秒产出一个对象，写回一个要 5 毫秒。容量为 1 时，只放得下正在写的那一个，第 2、3 个对象各等 3 毫秒。容量为 2 时多了一个排队的位置，第 2 个不用等，第 3 个等 1 毫秒。容量变大能吸收短时的突发，但写回一个对象仍然要 5 毫秒，长期的写回能力没有变。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 先按对象和生命周期拆分 I/O：输入、模型产物、运行时状态、输出各自的路径和首要证据 | 读 |
| Step 2 | 字节量、带宽与操作次数怎样决定服务时间：服务时间公式，GB/s 按十进制算 | 读，对照上面的小例子 |
| Step 3 | 重叠窗口与有限队列怎样改变等待：暴露尾部的两种情况，以及队列满时的入队和服务时刻 | 读 |
| Step 4 · TODO 1 | 计算 I/O 服务时间 | 在 `io_service_time_ms` 里写 2 行：`transfer_ms` 和 `service_ms` |
| Step 4 · TODO 2 | 计算暴露 I/O | 在 `exposed_io_time_ms` 里写 2 行：`hidden_ms` 和 `exposed_ms` |
| Step 4 · TODO 3 | 推进有限队列 | 在 `simulate_output_backpressure` 里写 3 行：`admitted_ms`、`queue_wait_ms`、`service_start_ms` |
| 测试 | 检查单位换算、零字节、暴露尾部的三种情况、账本按类别和频率累计、容量 1 和 2 下的背压，以及非法输入 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 分开测量本地保存、热缓存加载、双向搬运和回写，再用相同输入比较串行与双缓冲的复制—计算流水线 | 需要单张 CUDA GPU 和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，改成 `real_gpu` 后才执行真实 I/O 矩阵 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不读写临时张量，也不执行真实的 I/O 实验。

## 容易踩的坑

- **GB 用了二进制换算**：公式里的 GB 是十进制的 10^9 字节。字节除以带宽得到的是秒，再乘 1000 才是毫秒。
- **固定开销漏掉或重复计入**：搬 0 字节的操作也可能有固定开销。编解码或格式转换可以另列一个阶段，不能既算进实测带宽，又重复加进固定开销。
- **阻塞路径也想隐藏**：后续计算必须等读写做完时，暴露尾部等于整个服务时间。可以重叠时，隐藏量最多是服务时间和窗口中的较小值，结果不能是负数，也不能超过服务时间。
- **把局部尾部加起来当总延迟**：`total_exposed_ms` 只是各事件尾部的加和估算。只有尾部互不重叠、依赖关系允许相加时，才能当作总延迟。
- **队列容量算错**：容量包括正在服务的对象。队列满时，要等最早占用者完成才能入队；入队以后，还要等前面的服务写完才开始写。

## 读完后自查

- [ ] 能说出四类 I/O 对象，以及它们各自的发生频率。
- [ ] 搬 10^9 字节、带宽 1 GB/s、固定开销 2 毫秒时，能算出服务时间。
- [ ] 服务时间 12 毫秒、重叠窗口 5 毫秒时，能分别算出可以并行和必须等待两种情况下的暴露尾部。
- [ ] 能解释队列容量变大为什么不能提高写回的长期能力。

## 下一步

- 打开 notebook：[PERF-IO. 模型产物、状态传输与输出回写的 I/O 关键路径](../PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb)
- 建议先读：[PERF-INPUT. 数据加载、装箱与输入搬运性能 · 小白入门](./PERF-INPUT_Training_Input_Pipeline_Performance.md)
- 需要补基础：[Part 00 · 04 配置、路径与数据入口](../../00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb)、[Part 01 · 07 CPU/GPU 异构调度](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb)
- 想读深一点：[性能优化 · 05 计算、通信与 I/O](../../topic_discussion/performance_optimization/05_compute_communication_and_io.md)
