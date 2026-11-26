# PERF-IO. 模型产物、状态传输与输出回写的 I/O 关键路径 — 小白讲解

> 对应 notebook：[`PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb`](./PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb)

## 一句话总结
模型服务里的 I/O 不止"读数据"一种：加载模型产物会推迟首次可用，搬运运行时状态会卡住 GPU，写回输出队列满了会反压上游。本节先用 `io_service_time_ms` 算单次读写的理想服务时间，用 `exposed_io_time_ms` 区分"完全阻塞"和"能被独立计算掩盖的尾部"，用 `simulate_output_backpressure` 模拟有限队列的回压；再用一个 GPU 实验对比串行复制与双缓冲重叠，读已保存的结果判断收益是否成立。

## 为什么需要它：痛点是什么
初学者看到"带宽 1 GB/s"，就以为传 1 GB 一定花 1 秒，于是忽略了固定开销、方向差异和排队；看到"传输 8 ms、计算 5 ms"，就以为总时间一定是 13 ms，其实两者重叠时只需 8 ms（计算被完全掩盖），而计算必须等 I/O 时则要先等满 8 ms。更常见的误判是把加载、保存、H2D、D2H 几个独立测量直接相加，当成端到端时间。

生活类比：洗衣机洗衣服（I/O）和做饭（计算）。如果饭必须等衣服洗完才能做，8 分钟全暴露；如果可以先做饭再晾衣服，衣服程序结束时饭已经做好 5 分钟，只剩 3 分钟尾部要等。队列则像晾衣杆：杆满了，下一桶洗好也没地方挂，洗衣机只能停着——这就是回压。

## 核心问题
- 输入、模型产物、运行时状态、输出四类 I/O 的生命周期和主要路径有何不同？
- 单次服务时间怎么由字节数、带宽和固定开销算出？单位怎么统一？
- 什么情况下 I/O 时间会全部暴露？什么情况下只暴露"尾部"？
- 有限队列的容量如何计入正在服务的对象？回压怎样推迟后续生产？
- GPU 上双缓冲重叠到底省了多少？分段测量能不能直接相加？

## 解决思路：一步步看懂

**第 1 步：先按对象拆 I/O（Step 1）。** notebook 用一张表区分四类对象：输入每批次/每请求走"存储→CPU→GPU"；模型产物在冷启动或版本切换时走"存储→CPU→GPU"；运行时状态（KV Cache、激活、元数据）在每步、每词元或阶段交接时在 GPU↔GPU/CPU/远端之间搬；输出按 token、请求或周期写回网络/存储。对象和频率不明确时，单个带宽数字解释不了端到端影响。

**第 2 步：TODO 1——服务时间模型（Step 2）。** 公式是

$$
S=a+\frac{D}{10^9 B}\times 10^3,
$$

其中 $D$ 是这次操作实际搬运的字节数（非负整数），$B$ 是路径有效带宽（GB/s，十进制 $10^9$ Byte/s），$a$ 是固定开销（毫秒），$S$ 是服务时间（毫秒）。参考实现：

```python
transfer_ms = num_bytes / (bandwidth_gbps * 1_000_000_000) * 1000
service_ms = transfer_ms + fixed_latency_ms
```

除以 $10^9$ 得到秒，再乘 1000 换成毫秒；固定开销即使零字节也存在。测试 `test_service_time_units_and_invalid_input` 断言 `io_service_time_ms(1_000_000_000, 1.0, 2.0) == 1002.0`：1 GB ÷ 1 GB/s = 1 s = 1000 ms，加 2 ms 固定开销；`io_service_time_ms(0, 10.0, 0.5) == 0.5`；带宽为 0 必须抛 `ValueError`。骨架还会拒绝 `bool`、非整数字节、`nan`/`inf` 和负固定延迟。

**第 3 步：TODO 2——局部暴露尾部（Step 3）。** 设服务时间 $S$，与它同时开始、可独立执行的工作持续 $W$：

$$
E=\begin{cases}
S, & \text{必须先等 I/O 完成才能继续},\\
\max(0,S-W), & \text{独立工作与 I/O 同时开始且可并行}.
\end{cases}
$$

参考实现：

```python
hidden_ms = 0.0 if blocking_dependency else min(service_ms, overlap_window_ms)
exposed_ms = service_ms - hidden_ms
```

阻塞时 `hidden=0`，全部 $S$ 暴露；可重叠时最多隐藏 $\min(S,W)$，只剩尾部。测试 `test_exposed_time`：`exposed_io_time_ms(8, 20, blocking_dependency=True) == 8`；同样参数非阻塞得 0（20 ms 窗口足够覆盖 8 ms）；`exposed_io_time_ms(12, 5, blocking_dependency=False) == 7`（12−5）。函数还要求 `blocking_dependency` 必须是真正的 `bool`，时间不能为负或非有限。

**第 4 步：TODO 3——有限队列与回压（Step 3）。** 设第 $i$ 个对象就绪于 $r_i$、入队于 $u_i$、服务开始/完成于 $v_i,d_i$，服务时间 $s_i$，容量 $K$ 同时计入正在服务和排队的对象：

$$
q_i=u_i-r_i,\qquad v_i=\max(u_i,d_{i-1}),\qquad d_i=v_i+s_i.
$$

参考实现的关键三行：

```python
admitted_ms = min(departures) if len(departures) >= queue_capacity else ready_ms
queue_wait_ms = max(0.0, admitted_ms - ready_ms)
service_start_ms = max([admitted_ms, *departures])
```

`departures` 保存尚未完成对象的结束时刻。容量满时，新对象要等最早的一个完成才能入队，所以 `admitted_ms` 取 `min(departures)`；`service_start_ms` 还要等所有更早的对象服务完（单通道串行），所以取 `admitted_ms` 与所有 `departures` 的最大值。`producer_clock` 在入队后推进到 `admitted_ms`，因此下一次 `ready_ms` 会被回压推迟。

用 `test_backpressure_capacity_and_overload` 手算 `[2,2,2]` 的生产间隔、`[5,5,5]` 的服务时间、容量 1：

- 第 0 个：ready 2，队列空，admitted 2，服务 2→7，等待 0；
- 第 1 个：ready 4，但容量 1 且前一个 7 ms 才结束，admitted 7，等待 3，服务 7→12；
- 第 2 个：producer_clock 已是 7，ready 9，容量仍满，admitted 12，等待 3，服务 12→17。

所以 `queue_wait_ms == [0.0, 3.0, 3.0]`。容量 2 时前两个可以同时占位，等待变成 `[0.0, 0.0, 1.0]`，总等待更小。`test_empty_ledger_and_backpressure_propagation` 进一步核对 ready `[2,4,9]`、admitted `[2,7,12]`、结束 `[7,12,17]`，证明回压确实把后续生产推迟了。`test_zero_time_and_exact_service_boundary` 检查零耗时全为 0，以及"恰好完成服务时无需等待"（第二个对象 ready 7、前一个结束也是 7，等待 0）。

**第 5 步：账本与频率。** `io_critical_path_ledger` 按 `VALID_IO_OBJECTS = {"input","artifact","state","output"}` 分类，把每次事件的服务时间和暴露时间乘以 `frequency` 累加。测试 `test_ledger_frequency_and_objects` 中：artifact 1 GB、1 GB/s、阻塞 → 1000 ms，暴露 1000 ms；state 100 MB、10 GB/s → 10 ms，非阻塞且窗口 6 ms → 暴露 4 ms，频率 4 → 服务 40 ms、暴露 16 ms。总字节 1.4 GB。注意汇总里的 `total_exposed_ms` 是各事件尾部**相加**，只有当尾部依次发生时才近似累计等待；尾部重叠时必须按完整时间线递推。

**第 6 步：GPU 实验设计与已有结果。** 5.1 固定：64 MiB FP32 张量、8 个流水线批次、hidden 1024、5 次预热、20 轮交替顺序、`minimum_speedup=1.05`。分段部分测量保存、热缓存加载、普通 H2D、锁页非阻塞 H2D、D2H＋保存；流水线部分用 `_serial_pipeline_ms` 在默认流上依次 H2D 和矩阵乘，用 `_overlapped_pipeline_ms` 建 `copy_stream`、双缓冲和 `copy_done`/`compute_done` 事件，让下一批复制与当前批计算重叠。

已保存的 v2 结果（RTX 5070 Ti Laptop）是：

| 阶段 | 中位数 ms | P95 ms |
|---|---:|---:|
| `artifact_save` | 39.75 | 42.72 |
| `artifact_load_warm_cache` | 14.40 | 15.79 |
| `pageable_h2d` | 6.44 | 6.61 |
| `pinned_non_blocking_h2d` | 4.68 | 4.70 |
| `output_d2h_and_save` | 49.08 | 55.87 |

流水线：串行中位数 7.066 ms、P95 7.149 ms；双缓冲中位数 4.965 ms、P95 5.073 ms；`median_speedup=1.423`，`median_exposed_reduction_ms=2.101`，P95 减少 2.076 ms，20/20 轮都更快，`decision=accept_overlap_for_local_pipeline`。净时间下降约 1−4.965/7.066≈29.7%。怎么读：分段表里各项是独立测量，**不能相加**成端到端；流水线只比较同一输入的串行与双缓冲路径，"净下降"不等于"暴露的搬运时间全部消失"；锁页非阻塞 H2D 4.68 ms 比普通 6.44 ms 快，是 `pin_memory` 减少页锁定/暂存开销的体现。notebook 还明确标注 v2 只比较了源张量与锁页设备副本回传的张量，未完成完整文件重载核对；v3 增加了输入/输出文件重载检查，但当前历史结果仍是 v2。

GPU 实验的决策门槛是：正确性通过、中位数加速比至少 1.05、且 P95 下降，才 `accept`；校验失败则 `reject_correctness_failure`；只有小幅改善则 `tune_overlap_small_gain`；没有改善则 `keep_serial_path`。计时不包含新增文件内容校验。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 模型产物（artifact） | 保存下来的权重、配置等文件 |
| 冷启动 | 从没加载模型到能执行计算的启动过程 |
| 服务时间 $S$ | `固定开销 + 字节/带宽`，单位毫秒 |
| 暴露尾部 $E$ | 独立工作结束后仍需等待的 I/O 时间 |
| 重叠窗口 $W$ | 与 I/O 同时开始、可独立执行的工作时长 |
| 回压（backpressure） | 下游队列满，等待向上游传播 |
| 队列容量 $K$ | 同时计入正在服务和排队对象的数量上限 |
| 热缓存加载 | 文件已在页缓存中，不代表冷存储速度 |
| 锁页内存（pinned） | 不会被换出的主机内存，DMA 传输更快 |
| 双缓冲 | 两块缓冲区交替复制/计算，掩盖 H2D 时间 |
| `minimum_speedup` | 采用重叠路径要求的最低中位数加速比 |

## 小白常见疑问

**Q1：8 ms 的 I/O 加 5 ms 的计算，为什么可能只要 8 ms？**
因为两者可以同时执行。`exposed_io_time_ms(8, 5, blocking_dependency=False)` 返回 3，表示计算结束后只剩 3 ms 的 I/O 尾部，整段完成时间由 8 ms 的 I/O 决定。只有计算必须等 I/O 时才是 8+5。

**Q2：为什么 `io_service_time_ms` 除以 $10^9$ 而不是 $2^{30}$？**
因为 notebook 约定 GB 用十进制 $10^9$ Byte（`bandwidth_gbps * 1_000_000_000`），这样 1 GB/s 下 1 GB 恰好 1000 ms，测试断言 1002 ms。用 $2^{30}$ 会得到约 953.67 ms，和测试对不上。

**Q3：队列容量为什么把正在服务的对象也算进去？**
因为单通道写回时，正在服务的对象仍占用缓冲位置。`simulate_output_backpressure` 的 `departures` 里包含尚未完成的服务；容量 1 时第二个对象必须等第一个服务结束才能入队，所以等待是 3 ms 而不是 0。

**Q4：分段测量的 39.75、14.40、6.44、4.68、49.08 ms 能加起来当总时间吗？**
不能。它们是不同路径、不同计时方式的独立测量：保存和热加载是墙钟，H2D 是 CUDA 事件，D2H＋保存又混合了两种范围；notebook 在 5.1 明确写"各项不相加为一次端到端时间"。要比较端到端，应看流水线实验的串行/双缓冲总时间。

**Q5：中位数加速比 1.423，能说"搬运时间减少了 42.3%"吗？**
不能。1.423 是整段流水线时间的比值，包含了复制、计算和同步；`median_exposed_reduction_ms=2.101` 才是串行与双缓冲的中位数之差。notebook 5.4 也提醒"不能把差值全部认定为暴露搬运减少"，更不能用它反推单次 H2D 的加速。

**Q6：为什么历史结果是 v2，代码却写着 v3？**
`WORKLOAD_VERSION="v3"` 增加了输入/输出文件重载检查，但历史登记结果 `rtx5070ti_io_v2` 是 v2，未完成完整文件往返验证。5.3 的读取器在版本为 v2 时会打印提示，不能把 v2 的 `artifact_roundtrip_passed=True` 当成完整文件往返证据。

## 动手验证：怎么知道自己做对了

测试区直接调用八个测试，全过打印 `✅ 服务时间、局部尾部与有限队列测试通过`。关键断言：

- `test_service_time_units_and_invalid_input`：`(1_000_000_000, 1.0, 2.0)` 得 1002.0；`(0, 10.0, 0.5)` 得 0.5；零带宽抛 `ValueError`。
- `test_exposed_time`：`(8,20,True)=8`、`(8,20,False)=0`、`(12,5,False)=7`。
- `test_ledger_frequency_and_objects`：总字节 1.4e9；artifact 暴露 1000；state 服务 40、暴露 16。
- `test_backpressure_capacity_and_overload`：容量 1 等待 `[0,3,3]`；容量 2 等待 `[0,0,1]`，且总等待更小。
- `test_empty_ledger_and_backpressure_propagation`：ready `[2,4,9]`、admitted `[2,7,12]`、结束 `[7,12,17]`。
- `test_zero_time_and_exact_service_boundary`：零耗时全 0；第二个对象 ready 7 等于前一个结束 7 时等待 0；`exposed_io_time_ms(5,5,False)=0`。
- `test_nonfinite_values_and_strict_counts` 与 `test_backpressure_invalid_input`：`nan`/`inf`/负数/`bool`/非法容量必须抛 `ValueError`。

GPU 部分只有 `RUN_MODE="real_gpu"` 才执行；默认 `dry_run` 只打印配置。notebook 保存了上述 v2 输出，但运行时仍应以当前结果文件和 `correctness` 字段为准，不要照抄历史数字到新环境。
