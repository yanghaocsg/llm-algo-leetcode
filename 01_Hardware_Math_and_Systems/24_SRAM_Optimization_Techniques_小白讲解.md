# 24. SRAM 优化技术 — 小白讲解

> 对应 notebook：[`24_SRAM_Optimization_Techniques.ipynb`](./24_SRAM_Optimization_Techniques.ipynb)

## 一句话总结
本节用四个纯 CPU 计算器，把「共享内存优化」从口号变成可核对的账：`reuse_traffic_ledger` 算出 4 KiB 数据块用 4 次时全局读取从 16384 B 降到 4096 B、省 12288 B，但增加 4096 B 共享写入和 16384 B 共享读取；`bank_access_report` 说明 32 个存储体下步长 32 会退化到 32 路冲突，填充一个字后冲突度回到 1，而同字广播本来就不算冲突；`tile_resource_ledger` 与 `register_resource_ledger` 再把分块、双缓冲、填充和寄存器需求换算成每 SM 能驻留几个块。

## 为什么需要它：痛点是什么
初学者第一次听说「把数据放进 shared memory 会快」，很容易变成无脑加共享内存：结果共享内存把每个 SM 的驻留块数压到 1 块，占用率暴跌；或者二维数组按列访问，32 个线程全撞在同一个 bank 上排队；又或者寄存器用量涨到 80 个/线程，编译器开始把值往本地内存溢。更常见的是把「全局读取少了 12 KiB」直接说成「快了 3 倍」——省下的是逻辑字节，不是时间。

生活类比：办公室只有一张公共复印桌。一本 4 KiB 的资料，四个人轮流看，每人跑一趟档案室就是 4 次往返（16 KiB 全局读取）；复印一份放桌上只需跑一趟（4 KiB），但要复印（4 KiB 写桌）、四个人还要轮流拿（16 KiB 读桌），而且桌子被占着，能同时办公的人就少了。如果四个人同时抢同一页（bank conflict），还得排队；如果看的是同一页（broadcast），反而可以一起看。

## 核心问题
- 复用之后全局读取省了多少，片上又增加了多少读写，同步次数算不算字节？
- 为什么步长 32 会 32 路冲突，而 32 个线程读同一个字只算 1 路？
- 行尾填充一个 word，为什么能把冲突度从 32 降到 1？
- 分块变大、开双缓冲、加填充，各自如何改变共享内存用量和驻留块数？
- 寄存器需求怎么变成另一个驻留上限，多个上限如何合并？

## 解决思路：一步步看懂

**第 1 步：复用账本，先分清「省的是什么」（Q1）。** `reuse_traffic_ledger(tile_bytes, uses, sync_points=1)` 接收三个参数并做类型校验：`tile_bytes` 与 `uses` 必须是正整数，`sync_points` 必须是非负整数，否则抛 `ValueError`。返回六个字段：

```python
'global_read_bytes_without_reuse': tile_bytes * uses,   # 不复用：每用一次读一次
'global_read_bytes_with_reuse': tile_bytes,             # 复用：只读一次
'global_read_bytes_saved': tile_bytes * (uses - 1),     # 省下的全局读取
'shared_write_bytes': tile_bytes,                       # 搬进共享内存写一次
'shared_read_bytes': tile_bytes * uses,                 # 每个消费者仍要读一次
'sync_points': sync_points,
```

以 `TILE_BYTES = 4096` 为例，四次使用：全局读取 16384 → 4096，省 12288 B；共享写入 4096 B（只写一次）；共享读取 16384 B（四次，每次完整读一份）。使用 1 次时 `global_read_bytes_saved == 0`，因为本来就没有重复；使用 2 次时省 4096 B。这个账本回答了一个常被忽略的问题：复用减少的是**全局**读取，片上读取次数一次都没少——消费者还是要从共享副本里把数据读出来。

`sync_points` 是单独一列，只记录协作开销，不参与字节计算。`more_sync = reuse_traffic_ledger(..., uses=4, sync_points=3)` 与 `reports[4]` 只差 `sync_points` 一个字段，其余完全相同：

```python
assert more_sync == reports[4] | {"sync_points": 3}
```

notebook 特意强调「不能把字节数与同步次数相减得到耗时」：全局读写字节、共享读写字节、同步次数单位不同，不能相加成「显存流量」，也不能相减成「节省的时间」。这里还假设消费者每次完整读取共享副本，没有计缓存命中，也没有计指令级优化。

**第 2 步：存储体映射，冲突看的是「不同字」而不是「线程数」（Q2）。** 模型是 32 个 bank、每个 32 位字映射到一个 bank，字地址除以 32 的余数就是 bank 号。`bank_access_report(stride, banks=32, threads=32, padding_words=0, access_mode="strided")` 先生成每个 lane 的字地址：

```python
effective_stride = stride + padding_words
word_addresses = ([0] * threads if access_mode == "broadcast" else
                  [lane * effective_stride for lane in range(threads)])
bank_ids = [address % banks for address in word_addresses]
```

然后按 bank 收集「不同字地址」的集合，`max_conflict_degree` 取每个 bank 里不同字个数的最大值；`max_lanes_per_bank` 取的是同一 bank 上的线程数，两者不是一回事。结果对照：

| 步长 | 不同 bank 数 | 每 bank 不同字数（冲突度） | 说明 |
|---|---|---|---|
| 1 | 32 | 1 | 每线程一个 bank，无冲突 |
| 2 | 16 | 2 | 两个线程同 bank、不同字，2 路 |
| 16 | 2 | 16 | 地址 0,16,32,… 只在 bank 0 和 16 间交替，各 16 个字 |
| 32 | 1 | 32 | 所有地址都是 32 的倍数，全撞 bank 0 |
| 33 | 32 | 1 | 33 与 32 互质，lane×33 mod 32 = lane |

步长 32 时 lane 0…31 的地址是 0,32,64,…,992，全部 `% 32 == 0`，32 个不同字挤在一个 bank，只能分 32 轮服务。加一个填充字后 `effective_stride = 33`，地址变成 0,33,66,…，模 32 后正好遍历 0…31，冲突度变 1：

```python
assert bank_access_report(stride=32, padding_words=1)["max_conflict_degree"] == 1
```

同地址广播是另一个概念。`access_mode="broadcast"` 把 32 个 lane 的地址都设成 0，`max_lanes_per_bank == 32`，但 `max_conflict_degree == 1`、`serialized_risk` 为 `False`：

```python
broadcast = bank_access_report(stride=32, access_mode="broadcast")
assert broadcast["max_lanes_per_bank"] == 32
assert broadcast["max_conflict_degree"] == 1 and not broadcast["serialized_risk"]
```

硬件可以把同一个字同时发给多个线程，所以「32 个线程」不等于「32 路冲突」。判断冲突看的是同一 bank 上有几个**不同**字，而不是几个线程。

这个模型有明确边界：一次只模拟单 Warp、每线程读一个 32 位字，`threads > 32` 直接抛 `ValueError`；更宽的向量访问要按实际指令拆成多个 phase 分析，不能直接套这个结论。

**第 3 步：分块、双缓冲、填充如何吃掉驻留块数（Q3）。** 矩阵乘的一个线程块要同时放左矩阵的 $M\times K$ 子块和右矩阵的 $K\times N$ 子块。设每元素 $b$ 字节、右矩阵每行填充 $p$ 个元素、缓冲份数 $q$，共享内存用量：

$$S_{\mathrm{block}} = q\,b\,\big[MK + K(N+p)\big]$$

其中 $M,N,K$ 是分块维度，$q=2$ 表示给两组数据预留双缓冲，$p$ 改变右矩阵行步长。`tile_resource_ledger` 就按这个式子算 `shared_bytes_per_block`，再用三个上限取最小值：

```python
limits = {
    'shared_memory': smem_per_sm_bytes // shared_bytes,
    'threads': max_threads_per_sm // threads_per_block,
    'blocks': max_blocks_per_sm,
}
```

代入 $M=N=K=32$、$b=4$、$q=1$：$4\times(32\times32 + 32\times32) = 4\times2048 = 8192$ B，即 8 KiB。默认预算 `smem_per_sm_bytes=65536`、`max_threads_per_sm=2048`、`threads_per_block=256`、`max_blocks_per_sm=16`，三个上限是共享内存 8、线程 8、块数 16，取最小得到 8 块。开双缓冲 $q=2$ 后每块 16384 B，共享内存只够 4 块。加一个填充字 $p=1$：

$$4\times\big[1024 + 32\times(32+1)\big] = 4\times(1024+1056) = 8320\ \text{B}$$

比 8192 多 128 B，`padding_bytes_per_block` 就是 $q\,b\,K\,p = 1\times4\times32\times1 = 128$。共享内存上限变成 $65536\div8320 = 7$（7 块用 58240 B，8 块要 66560 B 超预算），驻留块数从 8 降到 7。

两个边界值得注意。`tile_resource_ledger(128, 128, 128)` 的共享内存是 $4\times(16384+16384) = 131072$ B，超过 65536 B 预算，`resident_blocks_upper_bound` 是 **0** 而不是 1——这组假设下连一块都放不下。预算改成 8192 时 8 KiB/块刚好放 1 块，改成 8191 时又是 0，说明整除的向下取整是硬边界。

`active_thread_fraction_upper_bound = resident_blocks * threads_per_block / max_threads_per_sm` 把驻留块换算成线程占用比例：8 块是 1.0，4 块是 0.5，7 块是 0.875。notebook 的文档字符串特意说明「预算是教学输入，不代表特定 GPU」；`resident_blocks` 是已经在 SM 上分到资源、可以推进的线程块，不是整个网格的块数。

**第 4 步：寄存器是第二道上限（Q4）。** `register_resource_ledger(registers_per_thread, threads_per_block=256, per_thread_budget=64, registers_per_sm=65536, ...)` 先算每块寄存器总量：

```python
registers_per_block = registers_per_thread * threads_per_block
'excess_registers_per_thread': max(registers_per_thread - per_thread_budget, 0)
'resident_blocks_upper_bound': min(
    registers_per_sm // registers_per_block,
    max_threads_per_sm // threads_per_block,
    max_blocks_per_sm,
)
```

32 寄存器/线程：每块 8192，65536 够 8 块；64 寄存器/线程：每块 16384，够 4 块，且 `excess_registers_per_thread` 为 0，说明刚好在 64 的预算内；80 寄存器/线程：每块 20480，65536 只够 3 块，`excess_registers_per_thread` 为 16；96 寄存器/线程：每块 24576，只够 2 块。最后把共享内存上限 `SHARED_LIMIT = 8` 和寄存器上限取最小：

```python
record['combined_resident_blocks_upper_bound'] = min(SHARED_LIMIT, record['resident_blocks_upper_bound'])
```

所以 80 寄存器时是 3 块，96 时是 2 块——此时瓶颈已经变成寄存器，再加共享内存优化也不会提高驻留。

这里必须强调「超出预算」不等于「已经溢出」。超出 64 只是说这个假设的分配量超过了预算，编译器可能缩短变量生命周期、重算某些值，或者把值放进本地内存；真实的溢出字节和耗时要看编译报告与执行指标，notebook 不推算这些。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| SRAM | 片上静态存储的电路实现；本节讨论的是线程块显式使用的共享内存 |
| 共享内存（shared memory） | 一个线程块内所有线程都能读写的小块片上空间 |
| 全局内存（global memory） | 容量大但离计算单元远的显存；每次读取都要走更长的路 |
| 逻辑字节 | 按算法数出来的读写量，不等于实际 HBM 流量或耗时 |
| 存储体（bank） | 共享内存被切成的并行服务单元，本节模型是 32 个 |
| 存储体冲突（bank conflict） | 同一 bank 上出现多个不同字，只能分轮次服务 |
| 广播（broadcast） | 多个线程读同一个字，硬件一次发给所有人，不算冲突 |
| 填充（padding） | 在每行末尾多留几个字，改变地址映射以打散 bank |
| 双缓冲（double buffering） | 为两组数据预留空间，容量条件是重叠搬运与计算的前提 |
| 驻留块（resident block） | 已在 SM 上分到资源、可以推进的线程块 |
| 占用率（active thread fraction） | 驻留块 × 每块线程数 ÷ 每 SM 最大线程数 |
| 寄存器压力 | 每线程寄存器需求过高导致能驻留的块数下降 |

## 小白常见疑问
**Q1：明明省了 12288 B 全局读取，为什么 `shared_read_bytes` 还是 16384 B？**
因为复用只改变「从哪里读」，不改变「读几次」。四次使用每次都要从共享副本读一份 4096 B，合计 16384 B；省掉的是四次全局读取里的三次。`reuse_traffic_ledger` 的字段名把这件事写得很清楚：`global_read_bytes_saved` 与 `shared_read_bytes` 是两个独立的量。

**Q2：32 个线程读同一个字，为什么 `max_lanes_per_bank` 是 32 却不算冲突？**
冲突度看的是同一 bank 上不同字的个数。广播模式下所有 lane 的地址都是 0，bank 0 上只有一个不同字，所以 `max_conflict_degree == 1`、`serialized_risk` 为 `False`。`max_lanes_per_bank` 只统计线程数，正好用来对比「线程多」和「冲突多」不是一回事。

**Q3：为什么加一个填充字就能解决步长 32 的冲突？**
因为 `effective_stride` 从 32 变成 33，而 33 与 32 互质。地址 `lane × 33 mod 32` 随 lane 逐一遍历 0…31，每个线程落到不同 bank。填充改变的是地址映射，不是数据本身；代价是每块多 128 B 共享内存，驻留块数从 8 降到 7。

**Q4：双缓冲开了就一定会重叠搬运和计算吗？**
不一定。`tile_resource_ledger` 只算容量：双缓冲让每块共享内存从 8 KiB 涨到 16 KiB，驻留块数从 8 降到 4。是否真的重叠还取决于执行调度，notebook 的表格把「缓冲份数」和「调度是否真正重叠」列成两件事。

**Q5：为什么 128×128×128 的分块驻留块数是 0，而不是至少 1？**
因为每块要 131072 B 共享内存，而假设预算只有 65536 B，`65536 // 131072 == 0`。这个函数不替调用者兜底成 1，而是如实报告「这组假设下放不下」；想让它变成 1，要么缩小分块，要么提高预算。

## 动手验证：怎么知道自己做对了
这个 notebook 没有测试运行器，也没有保存执行输出，检查全部是 cell 末尾的 `assert`，失败会抛 `AssertionError`；以代码和断言为准。

- Q1：`reports[1]["global_read_bytes_saved"] == 0`；`reports[4]` 的六个字段必须精确等于 16384、4096、12288、4096、16384、1；`more_sync == reports[4] | {"sync_points": 3}` 证明只改同步不动字节。
- Q2：`stride=1` 冲突度 1，`stride=2` 为 2，`stride=32` 为 32，`stride=32, padding_words=1` 为 1；广播的 `max_lanes_per_bank == 32` 且 `max_conflict_degree == 1`、`serialized_risk` 为假。
- Q3：`single['shared_bytes_per_block'] == 8192`、`double['shared_bytes_per_block'] == 16384`、`single` 驻留 8 块、`double` 驻留 4 块、`padded['padding_bytes_per_block'] == 128` 且驻留 7 块；`tile_resource_ledger(128,128,128)` 驻留 0；预算 8192 时驻留 1，预算 8191 时驻留 0。
- Q4：32 寄存器驻留 8 块，64 驻留 4 块且 `excess_registers_per_thread == 0`，80 的 `excess_registers_per_thread == 16` 且驻留 3 块，96 驻留 2 块；循环把 `combined_resident_blocks_upper_bound` 与共享内存上限取最小后打印。

想验证自己对机制的理解，可以改 `TILE_BYTES` 为 8192 观察各字段是否按比例翻倍，或把 `bank_access_report` 的步长改成 31、33 看冲突度是否都为 1。注意每次改完都要重新看断言是否仍成立，notebook 没有保存任何历史输出可供对照。
