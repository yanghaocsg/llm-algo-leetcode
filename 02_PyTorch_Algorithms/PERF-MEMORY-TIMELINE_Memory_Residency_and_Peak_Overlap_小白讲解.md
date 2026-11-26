# PERF-MEMORY-TIMELINE. 显存驻留与峰值重叠 — 小白讲解

> 对应 notebook：[`PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb`](./PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)

## 一句话总结
用一张"哪个对象、在哪一毫秒、创建或释放了多少字节"的事件表，亲手算出每一刻真正被占用的内存，并找出显存峰值的那一刻是哪几块存储叠在一起。全程 CPU 可做；GPU 实验只用来比较"提前释放再重算"能省多少峰值、多花多少时间。

## 为什么需要它：痛点是什么
不少人以为 `del` 掉一个大张量，显存就会立刻下降；结果要么看到占用没变，要么训练在"明明已经释放"的地方报显存不足，却无从查起。根本原因是三件事被当成了一件：Python 变量消失、底层存储变得可复用、内存真正归还给设备驱动。中间还夹着两个捣乱者——切片视图（view，只改变访问方式、不复制数据的张量）会悄悄握着原存储；分配器（allocator，负责向设备申请和回收内存的管家）会把释放的块留在缓存里备用。更隐蔽的是"峰值"：它不是各类内存最大值的相加，而是某一刻碰巧同时活着的存储之和。

类比合租房：名字从租约上划掉，不等于行李已经搬走（室友手里那把钥匙就是视图）；物业也不会马上收回房间（分配器缓存）。想知道"最多同时住过几个人"，不能拿"卧室最高入住数 + 客厅最高入住数"来加，得挑同一个时刻数全屋的人。

## 核心问题
- 张量变量删掉之后，它背后的存储什么时候才算不再被占用？
- 视图和原张量共享存储时，字节怎么数才不会重复？
- 显存峰值为什么必须从"同一个状态"读取，而不是各类别最大值相加？
- 时间戳相同的"先释放再分配"和"先分配再释放"，为什么会给出不同的峰值？

## 解决思路：一步步看懂
**第 1 步：把内存变化写成事件流。** 张量对象只描述形状、步长和存储位置，真正装数据的是底层存储（storage）；视图与原对象指向同一块存储，这些对象互称别名。一份存储"仍被持有"的时间段叫驻留区间。notebook 用事件来记录：每条事件带毫秒时间 `time_ms`、对象标识 `object_id`、动作 `action`（`allocate` 创建引用，`free` 释放引用）、整块存储字节 `bytes`、类别 `category`，可选的 `storage_id` 表示共享存储，不写就默认每个对象独占一块。注意 allocate/free 说的是"引用"的生死：真实设备上，最后一个引用消失后，CUDA 流上没跑完的使用还可能推迟复用，可复用的块也可能留在分配器缓存。窗口开始前就存在的对象，要用一条起点处的创建记录登记。

**第 2 步：按存储去重，不按对象名累加。** 12 个 FP32 元素占 48 字节，从里面切出 4 个元素的视图只访问 16 字节，却仍握着那 48 字节的整块存储：删掉原对象、留着视图，存储照样要算；克隆（`clone()`）才会新建存储，要另外计数。设时刻 $t$ 存活的对象集合为 $O(t)$，对象 $o$ 指向的存储为 $g(o)$，去重后的独立存储集合为 $S(t)$，存储 $s$ 的容量为 $b_s$ 字节：

$$S(t)=\{g(o):o\in O(t)\},\qquad M(t)=\sum_{s\in S(t)}b_s$$

$M(t)$ 就是这一刻的活动字节。参考实现只有两行：以 `storage_id` 为键把存活对象压成字典，再把字典里的字节相加。

**第 3 步：峰值要从同一个状态里读。** 设 $\mathcal T$ 是按真实事件顺序排好的状态序号，第 $k$ 个事件之后的活动字节为 $M_k$：

$$M_{\mathrm{peak}}=\max_{k\in\mathcal T}M_k,\qquad k^{\ast}=\min\{k\in\mathcal T:M_k=M_{\mathrm{peak}}\}$$

拿 notebook 的事件亲手算一遍：0 ms 创建 weights（100 字节，model），1 ms 创建 input（20，batch），2 ms 创建 activation（60，activation），3 ms 释放 input，4 ms 释放 activation。活动字节依次是 100 → 120 → 180 → 160 → 100，峰值 180 字节出现在 2.0 ms，此刻活着的是 activation、input、weights 三个对象，类别明细 {model: 100, batch: 20, activation: 60}。如果 5 ms 再分配 80 字节的 workspace，末尾也回到 180，但并列时取首次，峰值仍是 2.0 ms 的状态，里面没有 workspace。若改成"各类别最大值相加"，会得到 100+60+20+80=260 字节——一个从未真实出现过的数字。

**第 4 步：同一毫秒也要分先后。** 先分配 z（100 字节），然后在同一毫秒里"先释放 z、再分配 a"：活动字节 100 → 0 → 100，峰值 100；换成"先分配 a、再释放 z"：100 → 200 → 100，峰值 200。所以 `normalize_memory_events` 用 `(time_ms, event_order)` 排序——时间相同就保留输入先后，而不是把所有 free 统一挪到 allocate 前面。采集时丢了顺序，真实瞬时峰值就无法恢复。

**第 5 步：三个 TODO 的参考答案。** `live` 是"当前存活对象"的字典，`timeline` 是每个事件之后的一份快照。

```python
duplicate_allocation = event["action"] == "allocate" and object_id in live
unknown_free = event["action"] == "free" and object_id not in live
unique_storages = {item["storage_id"]: item for item in live.values()}
active_bytes = sum(item["bytes"] for item in unique_storages.values())
peak_bytes = max(point["active_bytes"] for point in timeline)
peak = next(point for point in timeline if point["active_bytes"] == peak_bytes)
```

**第 6 步：到 GPU 上验证取舍。** `run_path(recompute)` 的两条路径共用常驻张量 `persistent`（8192×2048，约 64 MiB）。重叠路径：`activation = persistent * 2.0`，`workspace = activation.square()`；重算路径：先 `del activation`，再 `workspace = persistent.clone()` 并 `workspace.mul_(2.0).square_()`。每轮先同步并重置峰值统计，用 CUDA 事件计时，记录 `latency_ms`、`peak_allocated_bytes`、`peak_reserved_bytes` 和 `checksum`；预热 5 次、每条路径测 20 次。5.4 登记的 `rtx5070ti_memory_v1` 结果：分配峰值约 192 → 128 MiB（降约 33.3%），延迟中位数 0.80 → 1.49 ms（约 1.86 倍），保留峰值两条路径都是 194 MiB——省了活动峰值，付出了时间，设备预留并没有减少。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 张量对象 / 存储 | 对象只记形状、步长和位置；数据真正放在底层存储里 |
| 视图 / 别名 | 只改变访问方式的张量，与原对象指向同一块存储 |
| 驻留区间 | 一份存储从创建到最后一个引用释放的时间段 |
| `storage_id` | 事件里"同一块存储"的标记，缺省等于 `object_id` |
| 活动字节 $M(t)$ | 某一刻去重后的存储容量之和 |
| 峰值驻留区间 | 活动字节达到最高值并持续的那段时间 |
| allocate / free | 事件流里引用的创建与释放，不等于设备真实的分配与归还 |
| 分配器缓存 | 分配器留着已释放块备用，所以预留量可能不降 |
| 已分配 / 已预留 | 张量正占用的字节 / 分配器向驱动要来的字节 |
| 类别 `category` | model、batch、activation、workspace 等标签，用来给峰值归因 |

## 小白常见疑问
**Q1：`del` 掉大张量后显存没降，是出 bug 了吗？**
不一定。先查是否还有视图（别名）握着同一块存储；即使引用全部消失，CUDA 流上没跑完的使用可能推迟复用，可复用的块也可能留在分配器缓存。"活跃占用下降而预留不变"正是分配器缓存的典型表现。

**Q2：释放了那个撑起峰值的大对象，只看释放那一刻够不够？**
不够。notebook 要求释放后仍扫描整个窗口：别的时刻可能冒出新的最高值。`peak_overlap_report` 也是对整条时间线取最大值，再取首次达到它的那个状态。

**Q3：共享同一存储的两个对象，`bytes` 填得不一样行不行？**
不行。字节和类别必须一致，不能用局部视图的大小冒充整块存储，否则报 "shared storage metadata must agree"；释放时的 `storage_id`、`bytes`、`category` 也要和创建时相同，否则报 "free metadata does not match allocation"。

**Q4：GPU 实验里保留峰值没降，是不是说明优化白做了？**
看目的。教学模型关心"逻辑上同时活着的字节"，实验里活动分配峰值确实降了约 33.3%，但延迟变成约 1.86 倍；保留峰值不变说明设备层面的归还和碎片没有变化，要继续查得看分配块与空闲块的记录。记录里的 `decision` 只针对峰值下降（`accept_for_peak_reduction`），时间代价要结合 `latency_ratio` 自己判断。

## 动手验证：怎么知道自己做对了
`run_memory_timeline_tests` 依次跑 8 个测试，全过才打印"✅ 事件顺序、共享存储与峰值归因测试通过"。事件由 `_memory_events()` 构造，非法输入统一用 `_assert_memory_error` 要求抛出 `ValueError`。关键断言：

- `test_residency_and_peak`：活动字节序列必须是 `[100, 120, 180, 160, 100]`，`peak_bytes == 180`、`peak_time_ms == 2.0`，`live_object_ids == ["activation", "input", "weights"]`，类别字节之和等于峰值。
- `test_shared_storage_release`：owner 与 view 共用存储 `s`，先释放 owner 再释放 view，字节序列为 `[100, 100, 100, 0]`；第 1 个快照里两个对象都在，但 `live_storage_ids == ["s"]`。
- `test_same_time_event_order`：两种同刻顺序分别得到 `[100, 0, 100]`（峰值 100）与 `[100, 200, 100]`（峰值 200）。
- `test_peak_ties_and_category_maxima`：加入 80 字节 workspace 后末尾也是 180，但峰值仍取 2.0 ms，`category_bytes == {"activation": 60, "batch": 20, "model": 100}`。
- `test_invalid_memory_events`：非法动作、空标识、字节为 0 / 1.5 / True、时间为负 / NaN / inf、重复创建、未创建就释放、释放元数据不符、共享存储元数据冲突，都必须报错。
- 另有 `test_normalize_memory_events`（倒序输入按时间排好）、`test_reallocation_after_release`（全部释放后同名重分配 200 字节，序列 `[100, 0, 200]`）、`test_empty_memory_timeline`（空输入可规范化和重建，但 `peak_overlap_report([])` 必须报错）。

框架一致性核对（可选）`check_shared_storage_framework_conformance`：用真实的 CPU 张量 `torch.arange(12, dtype=torch.float32)` 和偏移视图 `[2:6]`，通过 `untyped_storage().data_ptr()` 与 `nbytes()` 读取存储标识和容量，转成 Step 4 的事件流。7 项检查的键名是 `offset_view_shares_owner_storage`、`view_elements_smaller_than_backing_storage`、`view_survives_owner_reference_removal`、`clone_uses_independent_storage`、`reference_table_bytes_match_expected`、`teaching_timeline_matches_tensor_storage`、`peak_state_matches_coexisting_storages`。期望按元素计为 `[12, 12, 12, 16, 4, 0]`，即 48、48、48、64、16、0 字节；峰值 64 字节，共存对象 `["clone", "view"]`。没装 PyTorch 时结果标为 `skipped`。它只核对引用表对应的存储计量，不测 Python 对象回收、进程内存或 CUDA 分配器的真实释放。

GPU 实验默认 `RUN_MODE = "dry_run"`，只打印"⏭️ RUN_MODE=dry_run；未执行 GPU 显存时间线实验。"改成 `real_gpu` 需要唯一的 `RUN_LABEL` 并通过 `PERF-CUDA-MECHANISM` 预检，结果写入 `benchmarks/results/perf_memory_timeline/perf-memory-peak-overlap_<RUN_LABEL>.json`，同标签已存在就报 `FileExistsError`，不覆盖。5.3 的读取单元用 `read_performance_result` 从原始样本重算，打印两条路径的延迟中位数、分配峰值、保留峰值、派生对比（`allocated_reduction_ratio`、`reserved_reduction_ratio`、`latency_ratio`）和逐轮样本。读法：前两个比值越大省得越多，`latency_ratio` 大于 1 表示变慢；`quality.checksum_close` 只说明两条路径最后一轮校验和的相对差在千分之一以内，不等于逐元素 `allclose`。
