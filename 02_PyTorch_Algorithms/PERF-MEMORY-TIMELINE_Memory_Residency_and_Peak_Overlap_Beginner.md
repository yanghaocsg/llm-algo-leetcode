# PERF-MEMORY-TIMELINE. Memory Residency and Peak Overlap Beginner | 显存驻留与峰值重叠 · 小白入门

> 配套 notebook：[PERF-MEMORY-TIMELINE. 显存驻留与峰值重叠](./PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

训练时显存不够，常常是几块内存在同一时刻叠在了一起。这本 notebook 教你用“创建 / 释放”记录算出每个时刻占着多少字节，找出最高的那一刻是哪几块叠出来的。

## 为什么需要它

训练跑到某一步，显存突然不够用。你看代码，发现一个很大的张量用完就删掉了，显存却没有降下来。一个常见的原因是：另一个变量是它的视图，还指着同一块底层存储。删掉的只是一个名字，存储本身还被别的对象用着。

notebook 的框架核对里有一个可以亲手验证的例子：12 个 `float32` 元素的张量，取其中 4 个元素做切片。视图自己只占 16 字节，背后的存储还是整块的 48 字节，而且和原张量是同一块（下面的代码会算出来）。

想找出峰值是哪几块内存叠出来的，还有两种容易犯的错：

- 把每个变量的字节直接相加。几个名字指向同一块存储时，就重复算了。
- 把各类别各自的最大值加起来当峰值。参数、激活各有自己的峰值时刻，可能从来没有同时出现过。

这本 notebook 用一串“创建 / 释放”事件搭一个教学模型：每个事件之后记下还有哪些对象活着、它们指向哪些独立存储、一共多少字节，再从中找出字节最多、并且最早出现的那个状态。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 对象引用（`object_id`） | 代码里的一个张量变量，比如 `activation`。事件里的 `allocate` 和 `free` 指的是引用的创建和释放 |
| 底层存储（`storage_id`） | 真正占着字节的那块内存。几个对象可以指向同一块存储 |
| 视图 / 别名（view / alias） | 不复制数据、直接指向已有存储的新对象，比如对张量做切片得到的结果 |
| 事件后状态 | 处理完一个创建或释放事件之后的快照，记下活着的对象、独立存储和各类别的字节 |
| 活动字节（`active_bytes`） | 某个状态里，活着的对象指向的独立存储，容量加起来一共多少字节 |
| 峰值（`peak_bytes`） | 所有事件后状态里活动字节的最大值。并列时取第一次达到峰值的状态 |
| 分配器（allocator）的已分配量与保留量 | 分配器是 PyTorch 管理显存的那一层。对象释放后，块可能还留在它的缓存里，没有还给驱动 |

## 核心思路

输入是一串事件，每个事件带毫秒时间、对象标识、动作（`allocate` 或 `free`）、存储字节和类别，还可以带可选的 `storage_id` 表示共享存储。TODO 1 校验事件，拒绝重复创建和未知释放。TODO 2 在每个事件之后，把存活对象按存储去重，再把容量相加，得到一个事件后状态。TODO 3 从所有状态里选活动字节最大、且第一次达到的那个，输出峰值时间、峰值字节，以及那一刻活着的对象、存储和各类别字节。

```text
S(t)    = { o 指向的存储 : o 是时刻 t 还活着的对象 }     # 去重后的独立存储集合
M(t)    = sum( 存储 s 的容量 b_s , s 属于 S(t) )       # 活动字节：每块存储只加一次
M_peak  = max(M_k , k 取遍所有事件后状态)               # 峰值
k*      = 第一个满足 M_k = M_peak 的状态序号             # 峰值时间、对象、类别都从 k* 这一个状态里读
```

`t` 是观测时刻；`S(t)` 是当时活着的对象共同指向的独立存储集合；`b_s` 是存储 `s` 的容量；`M(t)` 是活动字节；`k` 是事件后状态的序号，`M_k` 是状态 `k` 的活动字节；`M_peak` 是峰值，`k*` 是首次达到峰值的状态序号。下面先用真实张量看一眼视图和存储的关系，再用 notebook 测试里的事件流算一遍。事件流里多加了一个共享存储的 `act_view`，省略了类别字段。

```python
import torch
owner = torch.arange(12, dtype=torch.float32)
view = owner[2:6]                                   # 切片得到的视图
print(view.numel() * view.element_size(), view.untyped_storage().nbytes(),
      view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr())
events = [  # (时间, 对象, 动作, 字节, 底层存储)
    (0.0, "weights",    "allocate", 100, "s_w"),
    (1.0, "input",      "allocate",  20, "s_i"),
    (2.0, "activation", "allocate",  60, "s_a"),
    (2.5, "act_view",   "allocate",  60, "s_a"),   # 和 activation 共用存储
    (3.0, "input",      "free",      20, "s_i"),
    (3.5, "activation", "free",      60, "s_a"),   # 只释放一个别名
    (4.0, "act_view",   "free",      60, "s_a"),   # 最后一个引用
]
live, timeline = {}, []
for time, name, action, size, storage in events:
    if action == "allocate":
        live[name] = (storage, size)
    else:
        del live[name]
    unique = {item[0]: item for item in live.values()}  # 同一存储只留一份
    row = (time, sum(item[1] for item in unique.values()), sorted(live))
    timeline.append(row)
    print(row)
peak_bytes = max(row[1] for row in timeline)
peak = next(row for row in timeline if row[1] == peak_bytes)
print(peak_bytes, peak[0], peak[2])
```

```text
16 48 True
(0.0, 100, ['weights'])
(1.0, 120, ['input', 'weights'])
(2.0, 180, ['activation', 'input', 'weights'])
(2.5, 180, ['act_view', 'activation', 'input', 'weights'])
(3.0, 160, ['act_view', 'activation', 'weights'])
(3.5, 160, ['act_view', 'weights'])
(4.0, 100, ['weights'])
180 2.0 ['activation', 'input', 'weights']
```

第一行说明视图只有 16 字节，底层存储却有 48 字节，两者是同一块存储。后面每行是一个事件后状态：时间、活动字节、活着的对象。前三个事件之后活动字节依次是 100、120、180。接着创建 `act_view`，它和 `activation` 指向同一块存储，活着的对象多了一个，字节还是 180。释放 `input` 后降到 160。再释放 `activation`：这只是两个别名中的一个，`act_view` 还指着那块存储，字节仍是 160。直到最后一个引用也释放，才减到 100。

峰值是 180 字节，在 2.0 ms 和 2.5 ms 各出现一次，报告取第一次，也就是 2.0 ms 的状态，当时活着的对象是 `activation`、`input`、`weights`。按 notebook 的类别标注，这时 `model` 是 100、`batch` 是 20、`activation` 是 60。时间、对象和类别都来自同一个状态，不能分别挑各自的最大值来拼。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 对象引用如何进入和离开驻留集合：创建、视图、释放各改变了什么，分配器状态为什么要分开看 | 读 |
| Step 2 | 共享存储怎样计入活动字节：先找出仍有引用的独立存储，再把容量相加 | 读，对照上面的小例子 |
| Step 3 | 哪些驻留重叠形成峰值：峰值和首次达到峰值的状态怎样定义 | 读 |
| Step 4 · TODO 1 | 检查引用状态转移 | 在 `normalize_memory_events` 里写 2 行：`duplicate_allocation` 和 `unknown_free` |
| Step 4 · TODO 2 | 按独立存储计算驻留量 | 在 `build_residency_timeline` 里写 2 行：`unique_storages` 和 `active_bytes` |
| Step 4 · TODO 3 | 选择同一峰值状态 | 在 `peak_overlap_report` 里写 2 行：`peak_bytes` 和 `peak` |
| 测试 | 检查排序、非法事件、重复引用和未知释放、共享存储的释放、同刻事件顺序、重复峰值取首次、空时间线报错 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用真实的 CPU 张量、切片视图和克隆，核对教学模型的存储计量 | 需要 PyTorch，同样在 CPU 上运行 |
| Step 5（可选） | 让激活与工作区重叠，或者在工作区创建前释放激活，比较真实显存峰值和重算的时间代价 | 需要单张 CUDA GPU 和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，改成 `real_gpu` 后才执行真实显存峰值对照 |

Step 1–4 只需要 CPU，框架核对用同一个 BASE-CPU 环境，但要装有 PyTorch。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不执行真实显存峰值对照。

## 容易踩的坑

- **删了一个变量就以为显存释放了**：多个别名共享同一存储时，释放一个别名，活动字节不减；最后一个引用释放后才减去。
- **用视图的大小当存储容量**：事件里的 `bytes` 要用整块存储的容量，不能用局部视图的大小冒充。共享同一存储的引用，字节和类别必须一致。
- **同刻事件重新排序**：时间戳相同的事件按输入先后处理，不能按对象名字或动作类型重排。先释放再申请和先申请再释放，瞬时峰值不同。
- **把各类别的最大值相加当峰值**：峰值字节、类别和对象必须来自同一个事件后状态，多次达到峰值时取第一次。
- **只看分配器数字就下结论**：已分配量下降而保留量不变，块可能还留在分配器缓存里。窗口结束时模型参数仍然存活是正常状态，不能仅凭缺少释放事件判断泄漏。

## 读完后自查

- [ ] 能说出释放一个别名和释放最后一个引用，对活动字节的影响有什么不同。
- [ ] 权重 100 字节、输入 20 字节、激活 60 字节同时存活时，能算出活动字节，并说出峰值出现在哪个状态。
- [ ] 能解释为什么峰值的时间、对象和类别必须从同一个状态里读。
- [ ] 能说出已分配量下降而保留量不变时，为什么不能断言显存已经归还。

## 下一步

- 打开 notebook：[PERF-MEMORY-TIMELINE. 显存驻留与峰值重叠](./PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)
- 建议先读：[43. 统一训练内存账本 · 小白入门](./43_Unified_Training_Memory_Ledger_Beginner.md)
- 需要补基础：[Part 00 · 18 显存分析](../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb)
- 做完之后：[PERF-IO. 模型产物、状态传输与输出回写的 I/O 关键路径 · 小白入门](./PERF-IO_Artifact_State_and_Output_IO_Critical_Path_Beginner.md)
