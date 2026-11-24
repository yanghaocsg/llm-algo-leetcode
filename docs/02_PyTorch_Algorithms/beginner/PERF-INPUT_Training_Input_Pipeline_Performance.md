# PERF-INPUT. Training Input Pipeline Performance | 数据加载、装箱与输入搬运性能 · 小白入门

> 配套 notebook：[PERF-INPUT. 数据加载、装箱与输入搬运性能](../PERF-INPUT_Training_Input_Pipeline_Performance.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

GPU 常在等数据：一批数据要先读取、整理，再从 CPU 搬到 GPU。这本 notebook 教你把短样本装进同一个盒子少补空位，再算清提前准备、边搬边算各能省多少等待。

## 为什么需要它

训练时 GPU 可能在空转，因为它在等下一批数据。一个批次要经过读取和解码、整理成张量、从 CPU 内存搬到 GPU，最后才轮到模型计算。这些步骤一个接一个串行做，GPU 每一批都要干等。CPU 其实可以在 GPU 算当前批时先把下一批准备好；搬运和计算同时进行，则还需要别的条件。

另一处浪费在批次里面。样本长短不一，填充的做法是把样本补到容器的长度；顺序装箱的做法是把相邻的短样本放进同一个容器。notebook 测试里的五条样本长度是 6、3、7、2、1，容器容量是 10。装箱后只有 1 个填充位置（下面的代码会算出来）。

几种常见的错：

- 打开了锁页内存或预取开关，就认为搬运和计算已经重叠；或者把第一批的等待当成供给不足。
- 看到利用率提高，就认为耗时按同样的比例下降。
- 为了腾出空间，拆分、丢弃或重排样本。

这本 notebook 做三件事：按顺序把样本装进容器，算出有效位置、容器位置和填充位置；再用两组开关分别控制“CPU 提前准备下一批”和“复制与计算重叠”，算出每一批什么时候开始计算、GPU 要等多久。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 装箱（packing）/ 容器（pack） | 把几条相邻的短样本放进同一个固定容量的盒子，容量是 `C` 个词元。不拆分样本，也不改变顺序 |
| 填充（padding）/ 填充位置 | 把样本补到容器长度时加上的无效位置。容器里没被真实样本占用的位置数就是填充位置 `Z` |
| 有效位置 / 容器位置 / 利用率（`efficiency`） | 真实样本占的位置数 `U` / 容器数乘以容量 `A` / 两者的比 |
| 预取（prefetch） | GPU 还在算当前批时，CPU 提前准备下一批 |
| H2D | 主机到设备的复制，把准备好的批次从 CPU 内存搬到 GPU |
| 锁页内存 / 非阻塞提交 / 独立复制流（`pinned`、`non_blocking`、`separate_copy_stream`） | 复制要和计算同时进行需要的三个条件。三个开关都为真，教学模型才允许复制与计算重叠 |
| 暴露等待（`exposed_wait_ms`） | 本批计算比上一批计算结束晚开始多久。第一批的等待来自流水线填充，后面的空隙才反映供给不足 |

## 核心思路

输入有两部分。第一部分是样本长度列表和容器容量：TODO 1 按顺序装箱，TODO 2 统计有效位置、容器位置、填充位置和利用率。第二部分是每批的准备、复制、计算耗时，加上四个开关：TODO 3 逐批算出准备、复制、计算各在什么时刻开始，以及计算比上一批晚开始多久。四个开关里，`prefetch` 决定准备要不要等上一批计算结束；`pinned`、`non_blocking`、`separate_copy_stream` 三个都为真时，复制才被允许和计算重叠。无论哪种情况，本批计算都必须等本批复制完成、上一批计算结束。

```text
U   = sum(l_i)                 # 有效位置：所有样本的长度之和
A   = J * C                    # 容器位置：非空容器数 J 乘以每个容器的容量 C
Z   = A - U                    # 填充位置
eta = U / A                    # 利用率；空输入时约定为 1
S_i^p = P_(i-1)                # 准备开始：预取时，上一批准备完就开始准备下一批
S_i^p = max(P_(i-1), C_(i-1))  # 不预取时，还要等上一批计算结束
S_i^h = max(P_i, H_(i-1))      # 复制开始：允许复制与计算重叠
S_i^h = max(P_i, H_(i-1), C_(i-1))   # 不允许时，复制也要等上一批计算结束
S_i^c = max(H_i, C_(i-1))      # 计算开始：本批复制完成，并且上一批计算结束
P_i = S_i^p + p_i   H_i = S_i^h + h_i   C_i = S_i^c + c_i   # 三个阶段的结束时刻
W_i = max(0, S_i^c - C_(i-1))  # 暴露等待
```

`l_i` 是第 `i` 条样本的长度，`J` 是非空容器数，`C` 是容器容量，`U`、`A`、`Z` 分别是有效位置、容器位置、填充位置，`eta` 是利用率。流水线部分：`p_i`、`h_i`、`c_i` 是第 `i` 批的准备、复制、计算耗时，`S_i^p`、`S_i^h`、`S_i^c` 是三个阶段的开始时刻，`P_i`、`H_i`、`C_i` 是三个阶段的结束时刻，`W_i` 是暴露等待，下标 `i-1` 表示上一批。下面用测试里的数字把装箱和流水线各算一遍。

```python
def greedy_packs(lengths, capacity):                      # 按顺序装箱，不拆分样本
    packs, current = [], []
    for length in lengths:
        if current and sum(current) + length > capacity:  # 当前包装不下：封箱，开新包
            packs.append(current)
            current = []
        current.append(length)
    return packs + [current] if current else packs
packs = greedy_packs([6, 3, 7, 2, 1], 10)
useful, container = sum(sum(pack) for pack in packs), len(packs) * 10
print(packs, useful, container, container - useful, useful / container)

def schedule(prepares, copies, computes, prefetch, copy_overlap):
    prepare_end = copy_end = compute_end = 0.0
    rows = []
    for p, h, c in zip(prepares, copies, computes):
        prepare_start = prepare_end if prefetch else max(prepare_end, compute_end)   # 不预取：等上一批算完
        prepare_end = prepare_start + p
        ready = [prepare_end, copy_end] + ([] if copy_overlap else [compute_end])     # 不重叠：复制也等上一批算完
        copy_start, copy_end = max(ready), max(ready) + h
        compute_start = max(copy_end, compute_end)                                  # 本批复制完、上一批算完才开始算
        rows.append((prepare_start, compute_start, compute_start - compute_end))
        compute_end = compute_start + c
    return rows

for flags in ((False, True), (True, False), (True, True)):
    print(flags, schedule([4, 4], [2, 2], [10, 10], *flags))
```

```text
[[6, 3], [7, 2, 1]] 19 20 1 0.95
(False, True) [(0.0, 6.0, 6.0), (16.0, 22.0, 6.0)]
(True, False) [(0.0, 6.0, 6.0), (4.0, 18.0, 2.0)]
(True, True) [(0.0, 6.0, 6.0), (4.0, 16.0, 0.0)]
```

第一行是装箱：6 和 3 装进第一个容器；7 放不下了（6 + 3 + 7 超过 10），封箱开新包，得到 `[7, 2, 1]`。2 个容器共 20 个位置，真实样本占 19 个，填充 1 个，利用率 0.95。

后面三行是流水线，每批准备 4 毫秒、复制 2 毫秒、计算 10 毫秒。括号里是两个开关：是否预取、是否允许复制与计算重叠。每批的三个数是准备开始、计算开始、相对上一批计算结束的等待。第一批都要等 4 + 2 = 6 毫秒才能开始计算，这是流水线填充。看第二批：不预取时，准备要等第一批算完（16 毫秒）才开始，计算到 22 毫秒才开始，GPU 空等 6 毫秒。只预取时，准备在 4 毫秒就开始，但复制还要等第一批算完，GPU 空等 2 毫秒。两个都打开时，复制 8 毫秒开始、10 毫秒完成，第一批一算完，第二批在 16 毫秒立刻开始，等待是 0。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 输入对象如何到达 GPU：一个批次经过读取与预处理、批次整理、主机缓冲与搬运、GPU 消费四个阶段 | 读 |
| Step 2 | 装箱怎样改变有效位置与填充比例：有效位置、容器位置、填充位置和利用率的公式 | 读，对照上面的小例子 |
| Step 3 | 预取与复制—计算重叠怎样分别减少等待：三阶段流水线的开始时刻和暴露等待 | 读 |
| Step 4 · TODO 1 | 更新 pack 状态 | 在 `build_greedy_packs` 里写 2 行：把样本加进 `current`，并累加 `current_tokens` |
| Step 4 · TODO 2 | 建立词元守恒账本 | 在 `packing_accounting` 里写 4 行：`useful_tokens`、`container_tokens`、`padding_tokens`、`efficiency` |
| Step 4 · TODO 3 | 递推阶段时间 | 在 `schedule_input_pipeline` 里写 3 行：`prepare_start`、`copy_start`、`compute_start` |
| 测试 | 检查装箱顺序和尾包、容量守恒、完全串行 / 仅预取 / 完整重叠 / 供给不足四种调度，以及非法输入 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用 `TensorDataset` 和 `DataLoader` 检查装箱计划交给真实批次接口后，样本编号和长度是否完整、有序地到达 | 需要 PyTorch，同样在 CPU 上运行 |
| Step 5（可选） | 固定已生成的主机张量和前向矩阵计算，比较普通内存、锁页内存、多进程预取三种输入供给配置 | 需要单张 CUDA GPU 和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，改成 `real_gpu` 后才执行真实矩阵 |

Step 1–4 只需要 CPU，框架核对用同一个 BASE-CPU 环境，但要装有 PyTorch。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不导入 CUDA 框架，也不执行真实矩阵。

## 容易踩的坑

- **拆分、丢弃或重排样本**：顺序装箱只是把相邻样本放进同一个容器，不拆分、不丢样本、不改变顺序。超过容量的样本要报错，不能偷偷改短；循环结束后，最后一个没装满的容器也要保存进结果。
- **填充位置算出负数，或容器数为 0 时除零**：填充位置等于容器位置减有效位置；空输入时利用率约定为 1，这只是避免除零，不代表产生了训练吞吐。
- **以为打开预取就能让复制重叠**：预取只让 CPU 准备不用等上一批计算。复制要和计算重叠，需要锁页内存、非阻塞提交和独立复制流三个条件同时满足；任何一个开关本身都不能证明重叠已经发生。
- **去掉消费依赖**：任何路径下，本批计算都必须等本批复制完成，不能为了重叠去读还没准备好的数据。

## 读完后自查

- [ ] 能说出填充和顺序装箱的区别。
- [ ] 长度 6、3、7、2、1 的样本、容量 10 时，能写出装箱结果，并算出有效位置、容器位置、填充位置和利用率。
- [ ] 能说出复制要与计算重叠，需要哪三个条件同时满足。
- [ ] 只开预取、不允许复制重叠时，能说出第二批的复制为什么要等第一批计算结束。

## 下一步

- 打开 notebook：[PERF-INPUT. 数据加载、装箱与输入搬运性能](../PERF-INPUT_Training_Input_Pipeline_Performance.md)
- 需要补基础：[Part 00 · 12 数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)、[Part 00 · 17 PyTorch Profiling 基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md)、[Part 01 · 07 CPU/GPU 异构调度](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.md)
- 想读深一点：[性能优化 · 05 计算、通信与 I/O](../../topic_discussion/performance_optimization/05_compute_communication_and_io.md)
- 做完之后：[PERF-CANDIDATE. 从资源约束到候选改动与新增代价 · 小白入门](./PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md)
