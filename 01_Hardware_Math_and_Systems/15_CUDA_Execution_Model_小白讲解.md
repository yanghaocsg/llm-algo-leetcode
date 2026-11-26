# 15. CUDA 执行模型 — 小白讲解

> 对应 notebook：[`15_CUDA_Execution_Model.ipynb`](./15_CUDA_Execution_Model.ipynb)

## 一句话总结
这一节从“1000 个元素、每线程处理一个”出发，用三个 CPU 计算器把 CUDA 的执行层级算清楚：要开多少个 Block、每个 Block 分成几个 Warp、尾块里有多少线程不干活、又有多少 Warp 槽位根本没人坐；随后算共享内存复用把 256 KiB 逻辑读取降到 64 KiB，最后用一张资源预算表判断单块配置是否超限。全节不启动内核，只做数量级推演。

## 为什么需要它：痛点是什么
初学 CUDA 最常见的误解，是把 `grid`、`block`、`warp` 当成“GPU 里真实存在的硬件分区”，以为写 1000 个线程就有 1000 个核心在跑；或者把“尾部越界线程”和“Warp 里空着的槽位”当成同一个数。实际写代码时就会困惑：1000 个元素、每块 256 线程，为什么启动了 1024 个线程？那 24 个线程会不会读越界？为什么 66 个元素、每块 33 线程，明明没有越界线程，却还有 62 个槽位空着？

生活类比：包车出游。一辆车固定 32 个座位（Warp），一个旅行团（Block）可以订几辆车，但**不能和另一个团拼车**。48 人的团要订 2 辆车，第二辆只坐 16 人、空 16 座；两个 48 人的团各自订车，一共 4 辆，永远不会把两个团剩下的 16 人拼成第三辆满车。座位空着是“Warp 槽位未分配”，而“有人买了票却没来景点”（尾部越界线程）是另一回事。

## 核心问题
- 1000 个元素、每块 256 线程，需要几个 Block、几个 Warp、多少无效线程？
- 为什么 66 个元素、每块 33 线程没有尾部越界线程，却有 62 个未分配的 Warp 槽？
- 块数从 1 变成 8，总读取量怎么变？每块申请的共享内存会跟着变吗？
- 每线程寄存器数、每块线程数、共享内存三者，怎样分别卡住一个 Block？
- “在给定预算内”为什么不等于“能在真实 GPU 上启动”？

## 解决思路：一步步看懂

### Q1：从元素数推 Grid、Block、Warp
`launch_geometry(num_elements, threads_per_block, warp_size=32)` 假设每个线程处理一个元素，返回六个量：

```python
blocks = (num_elements + threads_per_block - 1) // threads_per_block
warps_per_block = (threads_per_block + warp_size - 1) // warp_size
launched_threads = blocks * threads_per_block
unassigned_warp_slots = blocks * (warps_per_block * warp_size - threads_per_block)
```

逐项解释：

- `blocks` 是向上取整的块数。1000 个元素、每块 256 线程：$(1000+255)//256 = 4$ 块。向上取整保证输入被完全覆盖。
- `warps_per_block` 是每块按 32 个一组能分成几个 Warp。256 线程正好 8 个；33 线程要 2 个（第二个只有 1 个线程）。
- `launched_threads = 4 × 256 = 1024`，比 1000 多 24，这就是 `inactive_tail_threads = 24`：最后一块里有 24 个线程的编号超出了输入范围，必须靠 `if (idx < n)` 之类的边界判断让它们不干活，否则会越界读写。
- `warp_thread_slots` 是按 Warp 打包后的总座位数；`unassigned_warp_slots` 是“座位有、但没分配线程”的数量。对 256 线程/块，每块 8×32=256 个座位、正好 256 个线程，所以每块 0 个空槽，四块合计 0。

换成分块 128 线程：`blocks=8`、`warps_per_block=4`、总 Warp 数 $8×4=32$，和 256 线程/块的 $4×8=32$ 一样；`launched_threads` 仍是 1024，尾部仍是 24。**总线程数和总 Warp 数都没变，但块内协作范围和每块资源需求变了**，所以不能仅凭这两个数判断耗时相同。

真正区分两种“没干活”的是 66 个元素、每块 33 线程这个例子：

- `blocks = ceil(66/33) = 2`，两块正好覆盖 66 个元素，`inactive_tail_threads = 66 - 66 = 0`，没有越界线程；
- 但每块 33 线程要占 2 个 Warp、64 个座位，空 31 个；两块合计 `unassigned_warp_slots = 2 × (2×32 − 33) = 62`。

所以“没有尾部越界线程”和“没有 Warp 空槽”是两件事。notebook 还补了两个边界：`launch_geometry(256, 256)` 的尾部为 0；`launch_geometry(1, 32)` 启动了 32 个线程、只有 1 个有效，尾部 31。

最后要记住：这些数字是**程序组织任务的单位**，不是芯片出厂时就固定拥有的 Grid/Block 数量。SM（流式多处理器）是真正承载 Block 的硬件单元，一个 SM 可以驻留多个满足资源约束的 Block，Block 编号也不指定某个固定 SM。

### Q2：块数变了，总读取和单块共享内存要分开算
`block_reuse_traffic(blocks, tile_elements, bytes_per_element, reuse_count)` 把账拆成“整个 Grid 的逻辑读取”和“单块共享内存申请”两栏：

```python
tile_bytes = tile_elements * bytes_per_element
baseline_global_read_bytes = blocks * tile_bytes * reuse_count   # 每次都重读全局内存
tiled_global_read_bytes = blocks * tile_bytes                    # 每块只从全局搬一次
```

代入 notebook 的配置：每块 4096 个两字节元素，`tile_bytes = 4096 × 2 = 8192` 字节 = 8 KiB；8 个块，使用 4 次。

- 不复用：$8 × 8192 × 4 = 262{,}144$ 字节 = 256 KiB；
- 每块先搬一次到共享内存：$8 × 8192 = 65{,}536$ 字节 = 64 KiB；
- 节省 $262{,}144 - 65{,}536 = 196{,}608$ 字节 = 192 KiB。

使用 2 次时节省 $8×8192 = 65{,}536$ 字节；使用 1 次时节省 0——只用一次就没有复用可言。断言 `reports[4]["shared_bytes_per_block"] == 8192` 和 `reports[4]["tiled_global_read_bytes"] == 8 * one_block["tiled_global_read_bytes"]` 把两个口径钉死：**总读取随块数线性增长，单块共享内存申请始终是 8 KiB**。

这里还有两个“别把模型当实测”的提醒。第一，64 KiB 是 8 个块各搬一次的总和，不等于某个 SM 同时占用了 64 KiB；实际峰值要看同时驻留多少块。第二，192 KiB 是“逻辑全局读取”的减少量，真实设备流量还取决于缓存是否已经命中——缓存可能早就把重复读取吸收了，所以逻辑请求字节数和实际 DRAM 字节数要分开观察。

### Q3：单块资源门槛，四个约束分别检查
`block_resource_gate(threads_per_block, shared_bytes_per_block, registers_per_thread, limits)` 用一份显式的 `limits` 字典做四项检查：

```python
required = ("max_threads_per_block", "max_shared_bytes_per_block",
            "max_registers_per_thread", "max_register_bytes_per_block")
register_bytes = threads_per_block * registers_per_thread * 4
```

四个 `reasons` 按固定顺序追加：线程数超限记 `"threads_per_block"`，共享内存超限记 `"shared_memory"`，每线程寄存器超限记 `"registers_per_thread"`，每块寄存器总字节超限记 `"register_file"`。寄存器按每个 4 字节计算，所以 256 线程 × 64 寄存器 × 4 B = 65,536 B，正好等于示例预算 `max_register_bytes_per_block = 64 * 1024`，`within_stated_limits` 为真——**等于上限不算超**。

示例预算是：每块最多 1024 线程、64 KiB 共享内存、每线程最多 128 个寄存器、每块寄存器总量最多 64 KiB。注意这些数字只是“给定预算”，notebook 明确说不绑定某个 GPU 型号。逐组对照：

| 用例 | 改动 | 触发原因 |
|---|---|---|
| `at_register_budget` | 256 线程、64 寄存器、32 KiB | 无，寄存器刚好 65,536 B |
| `over_threads` | 1025 线程、0 寄存器 | `threads_per_block` |
| `over_shared` | 共享 65,537 B | `shared_memory` |
| `over_per_thread` | 32 线程、129 寄存器 | `registers_per_thread` |
| `over_register_total` | 256 线程、65 寄存器 | `register_file`（66,560 B） |
| `at_threads_and_shared` | 1024 线程、16 寄存器、64 KiB | 无，寄存器 65,536 B |
| `at_per_thread` | 32 线程、128 寄存器 | 无，寄存器 16,384 B |

`over_threads` 特意把寄存器设成 0，是为了让原因只剩线程数一条；`over_register_total` 的每线程寄存器 65 没超过 128，却因为总量 66,560 B 超过 64 KiB 而触发 `register_file`——这说明“每线程不超”不等于“每块不超”。组合用例把线程改成 1025、共享改成 65,537、寄存器保持 64，结果 `reasons == ["threads_per_block", "shared_memory", "register_file"]`，三项同时报出，顺序与代码检查顺序一致。

最后一个限定：这个函数只回答“在给定单块预算内是否超标”，不计算分配粒度和多块驻留，也不代表真实 GPU 一定能启动或达到高吞吐。实际还要看编译后的资源用量、启动配置和数值结果。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Thread（线程） | 执行一次局部计算的最小单位 |
| Block（线程块） | 一组线程，有共享内存和块级同步的协作边界 |
| Grid（网格） | 一次内核启动的全部 Block |
| Warp（线程束） | NVIDIA 把块内连续 32 个线程分成的执行组 |
| Lane | 线程在 Warp 中的位置（0–31） |
| SM | 流式多处理器，真正承载并执行 Block 的硬件单元 |
| 尾部无效线程 | 为凑整块而启动、但编号超出输入范围的线程 |
| Warp 未分配槽 | 块内线程数不是 32 的倍数时，Warp 里空着的座位 |
| 共享内存 | 同一 Block 内线程共享的片上存储，用来复用数据 |
| Tile（数据块） | 一次搬入共享内存的一块输入 |
| 逻辑读取字节 | 按程序请求统计的读取量，不等于实际 DRAM 流量 |
| 寄存器 / 寄存器文件 | 每线程私有存储 / 每块可用的寄存器总预算 |

## 小白常见疑问
**Q1：1000 个元素，为什么要启动 1024 个线程？多出来的 24 个会不会算错？**
因为 Block 是整数个，`blocks = ceil(1000/256) = 4`，`4×256=1024`。多出的 24 个线程必须由内核里的边界判断挡住，不参与读写；`inactive_tail_threads` 统计的就是这 24 个。它们不是“Warp 空槽”，是真实分配了线程但任务无效。

**Q2：66 个元素、每块 33 线程，没有越界线程，为什么还说浪费？**
两块刚好覆盖 66 个元素，`inactive_tail_threads=0`。但每块 33 个线程要占 2 个 Warp（64 个槽），空 31 个；两块共 `unassigned_warp_slots=62`。Warp 按 32 对齐，33 这种不是 32 倍数的块大小一定会产生槽位浪费。

**Q3：Q2 里 8 个块共读 64 KiB，是不是说明 8 个块能共用同一份共享内存？**
不是。64 KiB 是“每块各搬 8 KiB、共 8 块”的总逻辑读取量。共享内存是 Block 私有的，每个块都要独立搬入自己的副本；`shared_bytes_per_block` 始终是 8 KiB。跨块共享不在这个模型里。

**Q4：每线程 64 个寄存器没超，为什么还会报 `register_file`？**
因为还有“每块寄存器总量”这一项。256 线程 × 64 寄存器 × 4 B = 65,536 B；把每线程改成 65 后，总量 66,560 B 超过 64 KiB。两个约束要分别过。

**Q5：资源检查都通过了，内核就一定跑得快吗？**
不一定。`block_resource_gate` 只看单块是否超过给定预算，既不算分配粒度，也不算一个 SM 能同时驻留几块；“预算内”不等于“能启动”，更不等于高吞吐。真实性能还要看占用率、访存和同步。

## 动手验证：怎么知道自己做对了
notebook 没有 `test_*` 函数，全部是模块级断言，也没有保存任何执行输出；运行后不抛 `AssertionError` 即通过。

- Q1：`reports[256]` 必须精确等于 `{"grid_blocks": 4, "warps_per_block": 8, "total_warps": 32, "launched_threads": 1024, "inactive_tail_threads": 24, "warp_thread_slots": 1024, "unassigned_warp_slots": 0}`；`reports[128]` 的块数为 8、每块 4 个 Warp、尾部仍是 24；`launch_geometry(66, 33)` 的 `unassigned_warp_slots == 62` 且 `inactive_tail_threads == 0`；`(256, 256)` 尾部为 0；`(1, 32)` 尾部为 31。
- Q2：`reports[4]` 必须等于 `{"baseline_global_read_bytes": 262144, "tiled_global_read_bytes": 65536, "global_read_bytes_saved": 196608, "shared_bytes_per_block": 8192}`；使用 2 次时节省 65,536；使用 1 次时节省 0；1 个块与 8 个块的 `shared_bytes_per_block` 都是 8192，而 8 块的 `tiled_global_read_bytes` 是 1 块的 8 倍。
- Q3：`at_register_budget` 的 `register_bytes_per_block == 65_536`；`at_register_budget`、`at_threads_and_shared`、`at_per_thread` 的 `within_stated_limits` 为真；四个单项超限用例的 `reasons` 分别精确等于 `["threads_per_block"]`、`["shared_memory"]`、`["registers_per_thread"]`、`["register_file"]`；组合用例的 `reasons` 等于前三项。
- 各单元末尾会打印每块线程数、块数、Warp 数、尾部线程数，以及“1 块/8 块：总加载 8/64 KiB，单块申请均为 8 KiB”和每个用例的寄存器字节数。

如果这些断言全过，说明你对“块数向上取整、Warp 按 32 打包、尾部线程与空槽分开、总读取与单块资源分开”这四个口径已经对齐。
