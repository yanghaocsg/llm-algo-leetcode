# 16. Warp、Block 与 Shared Memory 基础 — 小白讲解

> 对应 notebook：[`16_Warp_Block_SharedMemory_Basics.ipynb`](./16_Warp_Block_SharedMemory_Basics.ipynb)

## 一句话总结
这一节用三个 CPU 计算器回答三个问题：48 个线程的块为什么要占 2 个 Warp、两个这样的块为什么是 4 个 Warp 而不是 3 个；同一块内把 32 KiB 数据用 4 次，逻辑全局读取怎样从 128 KiB 降到 32 KiB；以及容量刚好够、只差 1 字节、同步条件不成立时，这个“省读取”的候选还能不能继续往下做。

## 为什么需要它：痛点是什么
初学者容易把“线程数”直接当“Warp 数”：96 个线程就是 3 个 Warp，所以两个 48 线程的块当然也是 3 个。实际 Warp 是**按块内连续 32 个线程分组**的，两个块各自从 0 开始数，不能把两个块剩下的 16 个线程拼成一个满 Warp。另一类误解是“用了共享内存就一定更快”：共享内存只改变数据从哪里读，不保证容量装得下，也不保证写的顺序对；容量和同步任何一个不满足，省下来的读取量都只是纸面数字。

生活类比：会议室里的共享白板。一个项目组（Block）共用一块白板，组员（线程）围坐在每排 32 个座位的区域（Warp）。48 人的组坐满一排再坐第二排的 16 个座位，空着 16 个；另一组再开会，必须另开一排，不能和上一组剩下的 16 个座位拼在一起。白板的使用还要讲顺序：写的人写完，大家才能读；所有人都读完，才能擦掉写下一轮——这就是两次同步。

## 核心问题
- 48 线程的块需要几个 Warp？两个这样的块为什么是 4 个而不是 3 个？
- 线程槽利用率 75% 说明了什么、又不说明什么？
- 同一块内复用 4 次，逻辑全局读取减少多少？8 个块各自复用呢？
- 容量“刚好等于”数据块大小算不算满足？差 1 字节呢？
- 为什么共享内存需要两次同步，而不是一次？

## 解决思路：一步步看懂

### Q1：Warp 按块分组，不能跨块拼座位
`execution_coverage(num_threads, warp_size=32)` 返回四个量：

```python
warps = (num_threads + warp_size - 1) // warp_size          # 向上取整
thread_slots = warps * warp_size
last_warp_assigned_threads = num_threads - (warps - 1) * warp_size
thread_slot_utilization = num_threads / thread_slots
```

代入 48：`warps = ceil(48/32) = 2`；槽位 $2×32=64$；最后一个 Warp 实际分配 $48-32=16$ 个线程；利用率 $48/64=0.75$。断言要求返回的字典精确等于 `{"threads": 48, "warps": 2, "last_warp_assigned_threads": 16, "thread_slot_utilization": 0.75}`。

两个这样的块：$2×2=4$ 个 Warp。若把 96 个线程合成一个块，`execution_coverage(96)` 得到 3 个 Warp、利用率 1.0。**总线程数一样，Warp 数却不同**，因为分组边界变了；同时块内协作范围和单块资源需求也跟着变，所以不能只凭“都是 96 个线程”判断性能相同。

几个边界用例把取整规则钉死：1 个线程 → 1 个 Warp、最后 Warp 分配 1 个；32 → 1 个、分配 32 个；33 → 2 个、最后分配 1 个；65 → 3 个、最后分配 1 个。注意 `last_warp_assigned_threads` 是“最后一个 Warp 真正分到几个线程”，不是“空了几个”；65 个线程时第三个 Warp 只坐 1 个人，空 31 个。

还要区分三种“没坐满”：**线程槽利用率**只统计已分配线程占 Warp 槽的比例；分支和边界条件会让某条指令的活跃 Lane 更少，那是另一回事；SM 占用率则涉及驻留 Warp 数与设备上限，又是第三个指标。72% 的利用率不代表性能打七五折，只代表座位分配比例。

### Q2：共享内存复用，省的是“逻辑全局读取”
`shared_reuse_gain(tile_bytes, reuse_count, blocks)` 的模型很简单：

```python
baseline_global_read_bytes = tile_bytes * reuse_count * blocks
shared_global_read_bytes = tile_bytes * blocks
saved_global_read_bytes = baseline - shared
global_read_reduction_fraction = 1 - shared / baseline
```

代入主例子：`tile_bytes = 32 KiB = 32,768` 字节，单块、使用 4 次。

- 不复用：$32{,}768 × 4 = 131{,}072$ 字节 = 128 KiB；
- 先搬入共享内存、块内读副本：$32{,}768 × 1 = 32{,}768$ 字节 = 32 KiB；
- 节省 $131{,}072 - 32{,}768 = 98{,}304$ 字节 = 96 KiB，比例 0.75，即 75%。

这组数字被写成精确断言：`reports[4] == {"baseline_global_read_bytes": 131072, "shared_global_read_bytes": 32768, "saved_global_read_bytes": 98304, "global_read_reduction_fraction": 0.75}`。使用 2 次时节省 32,768 字节；使用 1 次时节省 0。

Q2 的扩展只改块数：8 个块、每块仍用 4 次。逻辑读取基线 $32{,}768×4×8 = 1{,}048{,}576$ 字节，共享后 $32{,}768×8 = 262{,}144$ 字节，节省 **786,432 字节**。关键认知是：**共享内存是块私有的，每个块都要独立搬一份**，不能把“块内复用”当成“跨块共享”。8 个块的总读取仍然是单块的 8 倍。

最后是口径提醒：96 KiB 是**逻辑读取请求**的减少量。真实设备流量取决于原方案是否已经命中缓存——如果重复读取本来就被缓存接住了，DRAM 字节数可能没这么多。所以“读取请求减少”和“实际内存流量减少”要分别观察。

### Q3：容量与同步，是两个独立前提
`shared_memory_candidate(tile_bytes, reuse_count, blocks, shared_limit_bytes, synchronization_valid)` 先调用 `shared_reuse_gain` 算出潜在收益，再检查两个前提：

```python
reasons = []
if tile_bytes > shared_limit_bytes: reasons.append("shared_capacity")
if not synchronization_valid:       reasons.append("synchronization")
```

返回里既有 `capacity_and_sync_conditions_met`（是否两个前提都满足）、`reasons`，也有 `has_logical_read_reduction`（收益是否大于 0）和完整的流量账。注意 `reasons` 的顺序固定：先容量、后同步。

六组对照写得很细：

| 用例 | 改动 | 结果 |
|---|---|---|
| `given_conditions` | 32 KiB 数据、64 KiB 预算、同步有效 | 前提满足，节省 98,304 B |
| `at_capacity` | 预算改成正好 32 KiB | 满足（等于不算超） |
| `short_one_byte` | 预算 32,768 − 1 | `reasons == ["shared_capacity"]` |
| `sync_miss` | `synchronization_valid=False` | `reasons == ["synchronization"]` |
| `both_miss` | 容量差 1 字节且同步无效 | `reasons == ["shared_capacity", "synchronization"]` |
| `no_reuse` | 只用 1 次 | 前提满足，但 `has_logical_read_reduction=False`，节省 0 |

这张表想教的是“**收益、前提、正确性要分开判断**”。`no_reuse` 这一行尤其重要：容量和同步都没问题，函数仍返回 `capacity_and_sync_conditions_met=True`，但因为只用一次、没有复用，逻辑读取减少为 0。反过来，容量不够时，函数仍会把“完整驻留假设下”的 98,304 字节算出来给你看，但那只是假设值，不能当成已实现收益。

为什么同步要两次？notebook 的说明是：一组线程协作写入当前数据块，**完成所需同步后**其他线程才能读；所有消费者读完后，**再同步**才能覆盖成下一块。前一次同步防止“读到还没写完的数据”，后一次防止“较慢的消费者读到下一轮的数据”。如果块内并非所有线程都能到达同一个块级屏障（比如分支导致部分线程提前退出），还要调整控制流或同步范围，否则屏障本身就会出问题。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Thread（线程） | 执行局部计算的最小单位 |
| Block（线程块） | 一组线程，共享内存和块级同步的协作边界 |
| Warp（线程束） | 块内连续 32 个线程组成的执行组 |
| Lane | 线程在 Warp 内的位置（0–31） |
| 线程槽利用率 | 已分配线程数 ÷（Warp 数 × 32） |
| 尾 Warp | 线程数不是 32 倍数时，最后那个没坐满的 Warp |
| 共享内存 | 同一 Block 内线程共享的片上存储 |
| Tile（数据块） | 一次搬入共享内存的一块数据 |
| 逻辑全局读取 | 按程序请求统计的读取字节，不等于 DRAM 实际流量 |
| 块级屏障 | 让块内线程在此会合、保证写入可见的同步点 |
| 生产者 / 消费者 | 写入数据块的线程 / 读取数据块的线程 |
| 容量前提 / 同步前提 | 数据装得下 / 读写顺序正确，二者独立 |

## 小白常见疑问
**Q1：两个 48 线程的块，为什么不是 3 个 Warp？**
Warp 只在块内按连续 32 个线程分组，不能跨块拼。每个 48 线程块是 2 个 Warp（32 + 16），两块共 4 个；只有把任务真正合并成一个 96 线程块，才会变成 3 个 Warp。`execution_coverage(96)["warps"] == 3` 与 `2 × execution_coverage(48)["warps"] == 4` 是两组不同的事实。

**Q2：48 线程的利用率是 0.75，是不是性能也只有 75%？**
不是。0.75 只是“已分配线程占 Warp 槽”的比例，第二个 Warp 有 16 个槽空着。实际性能还取决于指令活跃 Lane、访存、同步和占用率；notebook 明确把“分配比例”和“分支导致的活跃 Lane 数”“SM 占用率”分开。

**Q3：用了共享内存，全局读取就真的少了 96 KiB 吗？**
减少的是逻辑读取请求。如果原来的重复读取本来就命中缓存，实际 DRAM 流量可能没那么多。`shared_reuse_gain` 算的是请求字节，notebook 提醒要再观察缓存行为。

**Q4：共享内存预算正好 32 KiB，数据也 32 KiB，为什么算满足？**
检查条件是 `tile_bytes > shared_limit_bytes`，只有严格大于才报 `shared_capacity`。`at_capacity` 用例就是 32,768 对 32,768，返回前提满足；差 1 字节（32,767）才失败。

**Q5：只用一次的数据，还需要搬进共享内存吗？**
不需要。`no_reuse` 用例显示：前提都满足，但 `saved_global_read_bytes=0`、`has_logical_read_reduction=False`。没有复用就没有收益，搬入反而多一次写。

## 动手验证：怎么知道自己做对了
notebook 没有 `test_*` 函数，全部是模块级断言，也没有保存执行输出；运行时以是否抛 `AssertionError` 为准。

- Q1：`execution_coverage(48)` 必须精确等于 `{"threads": 48, "warps": 2, "last_warp_assigned_threads": 16, "thread_slot_utilization": 0.75}`；`2 × 2 == 4`；`execution_coverage(96)` 的 `warps == 3` 且利用率为 1.0；边界 `(1,32,33,65)` 分别得到 `(1,1)`、`(1,32)`、`(2,1)`、`(3,1)` 的 `(warps, last_warp_assigned_threads)`。
- Q2：`reports[4]` 的四个字段必须是 131,072、32,768、98,304、0.75；使用 2 次时节省 32,768；使用 1 次时节省 0；8 个块时 `shared_global_read_bytes == 262_144` 且 `saved_global_read_bytes == 786_432`。
- Q3：`given_conditions` 与 `at_capacity` 的前提为真；`short_one_byte` 的原因恰为 `["shared_capacity"]`；`sync_miss` 恰为 `["synchronization"]`；`both_miss` 恰为 `["shared_capacity", "synchronization"]`；`given_conditions` 节省 98,304；`no_reuse` 前提为真但 `has_logical_read_reduction` 为假、节省 0。
- 每个单元末尾会打印每个块的 Warp 数与利用率、单块/8 块加载量，以及每个用例的“前提是否满足 / 潜在读取减少 KiB / 未满足项”。

断言全过，说明你已经把“Warp 分组边界、逻辑读取与实际流量、容量与同步”这三组容易混淆的概念分开了。
