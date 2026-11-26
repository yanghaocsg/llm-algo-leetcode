# 23. Tensor Core 深度剖析 — 小白讲解

> 对应 notebook：[`23_TensorCore_Deep_Dive.ipynb`](./23_TensorCore_Deep_Dive.ipynb)

## 一句话总结
这一节不做真实矩阵乘，而是用四个 CPU 小工具把“用没用上 Tensor Core”拆成四层可核对的事实：逻辑子问题有多少 FLOPs、精度三元组是否被目标声明、形状/布局/寄存器/占用率各自是什么状态、以及指令计数与调用时间是否同时支持“更快”的结论。核心结论是：**“不是 16 的倍数”“占用率低”都不足以判定没用 Tensor Core，看到矩阵指令也不等于完整调用更快**。

## 为什么需要它：痛点是什么
关于 Tensor Core 的传言特别多：有人说“矩阵维度必须是 16 的倍数，否则用不了”；有人说“占用率低于 50% 就不会走 Tensor Core”；还有人跑 profiler 看到 `HMMA` 就宣布“已经用上 Tensor Core，所以快了”。这些说法把三件不同的事混成了一件：**逻辑工作量**、**实现是否满足某条指令路径的条件**、**实际执行有没有收益**。一旦混在一起，就会出现“M=130 所以肯定没用”“指令出现了所以一定更快”这类错误结论。

生活类比：用 16×16 的托盘运货。一个 128×128 的订单正好铺满若干整托盘，直接装；130×130 的订单多出两行两列，要么单独处理尾货，要么补到 144×144 再装，多出来的托盘位是空的。但“这个仓库用不用叉车（矩阵指令）”还要看叉车是否支持这种货物、通道能不能进；而“用了叉车是否更省钱”又是另一笔账——可能叉车调度费更贵。

## 核心问题
- 16×16×16 的逻辑矩阵块对应多少 FLOPs？它等于一条 MMA 指令吗？
- 判断一条精度路径，为什么要同时看输入存储、乘法模式和累加类型？
- M=130 时补齐到多少？多出多少输出槽？低占用率能否直接排除矩阵指令？
- 指令计数为 `None`、为 0、为正数，分别能得出什么结论？
- 为什么“看到矩阵指令”与“完整调用更快”必须分开报告？

## 解决思路：一步步看懂

### Q1：先算逻辑工作量，别把它当成指令数
`mma_flops(m, n, k)` 只有一行：返回 $2mnk$。系数 2 来自“一次乘加 = 一次乘法 + 一次加法”两次浮点运算。代入：

- $16×16×16$：$2×16×16×16 = 8192$ FLOPs；
- K 翻倍成 32：$2×16×16×32 = 16{,}384$ FLOPs；
- 小例子 $2×3×4$：$2×2×3×4 = 48$ FLOPs。

函数里没有任何设备查询，也没有说“这就是一条指令”。notebook 反复强调：8192 描述的是**逻辑子问题的工作量**，实际实现还要把它分解成目标架构支持的指令形状，并安排线程协作。WMMA（Warp Matrix Multiply Accumulate）是一个 Warp 协作完成矩阵乘加的编程接口；fragment 是矩阵片段在协作线程中的表示。逻辑 Tile、指令块、线程块是三个不同层级，不能用一张统一的 16×16 图代表所有设备。

### Q2：精度路径是一个三元组，不是“输入 dtype”
`tensorcore_dtype_path(input_dtype, multiplication_dtype, accumulation_dtype, numel, supported_paths)` 返回三样东西：

```python
dtype_bytes = {"FP32": 4, "FP16": 2, "BF16": 2, "FP8": 1}
path = (input_dtype, multiplication_dtype, accumulation_dtype)
return {"precision_path": path,
        "input_storage_bytes": numel * dtype_bytes[input_dtype],
        "declared_path_supported": path in supported_paths}
```

三个参数分别回答：数据**怎么存**、乘法**用什么精度算**、结果**累加到什么类型**。notebook 给一个假设目标声明了三组路径：

- `("FP16", "FP16", "FP32")`
- `("BF16", "BF16", "FP32")`
- `("FP32", "TF32", "FP32")`

`numel=1024` 时，BF16 路径的输入载荷是 $1024×2=2048$ 字节，TF32 路径是 $1024×4=4096$ 字节——**TF32 不是一种需要单独保存的模型 dtype，输入仍按 FP32 存 4 字节，只是乘法内部走 TF32 路径**。这正是“存储格式和计算路径要分开看”。

两个反例也很有信息量：`("FP32","FP32","FP32")` 不在声明集合里，`("BF16","BF16","BF16")` 也不在。前者说明“普通 FP32 乘法”未必等于“Tensor Core 路径”；后者说明“输入是 BF16”不代表“累加也是 BF16”。代码用完整三元组做 `in supported_paths` 精确匹配，而不是只看 `input_dtype`。同时 notebook 声明：这个集合是**假设目标**的配置，未声明只代表此配置没提供，不代表所有 GPU 都不支持。

还有一句重要的边界：高精度累加（比如 FP32 累加）能减轻累积误差，但**不能恢复输入转成 BF16 时已经丢掉的精度**。累加类型和输出类型也要分别记录。

### Q3：把几何覆盖、布局、寄存器、占用率分开观察
`inspect_matrix_configuration(m, n, k, tile=16, layout_contiguous=True, occupancy=1.0, register_pressure=32, register_budget=64)` 返回六个观察值：

```python
padded = tuple((dim + tile - 1) // tile * tile for dim in (m, n, k))
"shape_aligned_for_example": padded == (m, n, k),
"padded_shape_mnk": padded,
"extra_output_slots": padded[0] * padded[1] - m * n,
"layout_adaptation_needed": not layout_contiguous,
"given_occupancy": occupancy,
"register_budget_met": register_pressure <= register_budget,
```

它做的事很克制：只记录，不把占用率乘成一个“利用率分数”，也不把它当指令路径的门槛。

- **M=128**：128 能被 16 整除，补齐形状 `(128,128,128)`，额外输出槽 0。
- **M=130**：向上补齐到 144，形状 `(144,128,128)`，额外输出槽 $144×128 - 130×128 = 14×128 = 1792$。这 1792 个槽是补齐带来的“空托盘位”。
- **转置/非连续输入**：`layout_adaptation_needed=True`，需要相应地址计算或布局转换。
- **占用率 0.3**：`shape_aligned_for_example` 仍为真、`given_occupancy` 仍是 0.3，函数不会因为低占用率就判“不能用矩阵指令”。
- **寄存器**：每线程 64 个、预算 64 个时 `register_budget_met=True`（等于算满足）；65 个时为 `False`。

notebook 明确限定：这是一个“按 16×16×16 分块、只直接处理整除形状和连续输入”的示例内核的能力范围。库或其他内核可能处理不整除、非连续输入；不能把这个示例的限制当成 Tensor Core 的统一限制。同理，SM 驻留占用率只是驻留 Warp 数相对上限的比例，低占用率既不自动排除矩阵指令，也不直接预测耗时。

### Q4：指令观察和时间改善是两列，不能互相补写
`summarize_tensorcore_observation(baseline_ms, candidate_ms, tensor_instruction_count)` 把一次构造记录拆成：

```python
"tensor_instructions_observed": None if tensor_instruction_count is None
                               else tensor_instruction_count > 0,
"time_ratio": baseline_ms / candidate_ms,
"saved_ms": baseline_ms - candidate_ms,
```

三个状态的语义必须分清：

- `None`：**没有采集**计数器，不能推断“有”或“没有”；
- `0`：采集了，但本次记录**没观察到**所选指令；
- 正数：观察到了。

四个用例都以 `baseline_ms=2` 为基准：

| 用例 | 候选耗时 | 指令计数 | 观察结果 | 节省 |
|---|---:|---:|---|---:|
| `instructions_but_slower` | 3 ms | 64 | `True` | −1 ms（时间比 0.667） |
| `instructions_and_faster` | 1.6 ms | 64 | `True` | 0.4 ms（时间比 1.25） |
| `counter_not_collected` | 1.6 ms | `None` | `None` | 0.4 ms |
| `no_matching_instructions` | 3 ms | 0 | `False` | −1 ms |

第一行就是本节最想纠正的场景：**采到了矩阵指令，但调用更慢**——说明这条路径存在，却没带来净收益，可能被布局转换、尾块或其它开销吃掉。第三行则相反：时间变快了，但没采指令计数，只能报告“时间改善”，不能补写“用了 Tensor Core”。另外 `baseline_ms == candidate_ms` 时 `saved_ms == 0`，时间比 1.0，属于“没变化”，不是“更快”。

真实测量还要固定工作量与精度要求，做预热和多次重复，并把布局转换等必要开销算进候选路径。`HMMA` 只是部分架构上的指令名称，不是所有 Tensor Core 路径的统一名字，计数器要按目标架构和实际内核选择。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MMA | 矩阵乘加，一次完成一小块矩阵的乘与累加 |
| FMA | 标量融合乘加，一次算一个数的 $a×b+c$ |
| FLOPs | 浮点运算次数，$2mnk$ 里的 2 表示一乘一加 |
| 逻辑 Tile | 你希望一起处理的子矩阵，如 16×16×16 |
| 指令块 | 硬件指令真正接受的分块形状，可能小于逻辑 Tile |
| WMMA | 一个 Warp 协作做矩阵乘加的编程接口 |
| fragment | 矩阵片段在协作线程中的分布表示 |
| 精度三元组 | 输入存储 / 乘法模式 / 累加类型 |
| TF32 | 输入仍按 FP32 存，乘法走较低精度的 Tensor Core 路径 |
| 补齐（padding） | 把 130 补到 144，凑成 tile 的整数倍 |
| 占用率（occupancy） | 驻留 Warp 数相对设备上限的比例 |
| HMMA | 部分架构上的矩阵乘加指令名，不是 Tensor Core 的统称 |

## 小白常见疑问
**Q1：16×16×16 就是一条 MMA 指令吗？**
不是。`mma_flops(16,16,16)` 只告诉你这个逻辑子问题有 8192 FLOPs。真实实现可能用多条指令、不同指令形状，或由一个 Warp 的多个 fragment 协作完成；逻辑 Tile 与指令块必须分开核对。

**Q2：M=130 是不是就“用不了 Tensor Core”了？**
不能这么说。`inspect_matrix_configuration` 只描述这个示例内核：它按 16 分块、只直接处理整除形状，所以 130 要补到 144，多出 1792 个输出槽，或者单独处理尾块。其它库/内核可能支持尾块处理。notebook 明确说“不能把这个示例的范围当成 Tensor Core 的统一限制”。

**Q3：占用率只有 0.3，是不是肯定没用矩阵指令？**
不是。`inspect_matrix_configuration(occupancy=0.3)` 仍然返回 `shape_aligned_for_example=True`，占用率只是被记录下来。低占用率可能影响延迟隐藏，但不自动排除矩阵指令，也不直接预测耗时。

**Q4：profiler 里看到 64 条矩阵指令，能说“加速了”吗？**
不能。看 `instructions_but_slower`：候选 3 ms、基线 2 ms，指令计数 64，`saved_ms=-1`。指令存在只证明路径被走到，净收益要另看完整调用时间。

**Q5：TF32 路径的输入为什么算 4096 字节，不是更少？**
因为 TF32 不是单独保存的模型 dtype。`dtype_bytes` 里 FP32 是 4 字节，`tensorcore_dtype_path(input_dtype="FP32", ...)` 的 `input_storage_bytes` 就是 $1024×4=4096$。变的只是乘法模式。

## 动手验证：怎么知道自己做对了
notebook 没有 `test_*` 函数，全部是模块级断言，也没有保存执行输出；运行时看是否抛 `AssertionError`。

- Q1：`mma_flops(16,16,16) == 8192`、`mma_flops(16,16,32) == 16_384`、`mma_flops(2,3,4) == 48`；非法维度（0 或负数）应抛 `ValueError`。
- Q2：`bf16["declared_path_supported"]` 为真且 `input_storage_bytes == 2048`；`tf32` 的为真且 `input_storage_bytes == 4096`；`("FP32","FP32","FP32")` 和 `("BF16","BF16","BF16")` 的 `declared_path_supported` 都为假。每个结果还会打印精度三元组、是否已声明和输入载荷字节。
- Q3：128 的 `shape_aligned_for_example` 为真且 `extra_output_slots == 0`；M=130 的 `padded_shape_mnk == (144,128,128)`、`extra_output_slots == 1792`；`layout_contiguous=False` 时 `layout_adaptation_needed` 为真；`occupancy=0.3` 时形状仍对齐且 `given_occupancy == 0.3`；`register_pressure=64` 时预算满足、65 时不满足。
- Q4：`instructions_but_slower` 的 `tensor_instructions_observed is True` 且 `saved_ms == -1`；`instructions_and_faster` 的 `time_ratio == 1.25`；`counter_not_collected` 为 `None`；`no_matching_instructions` 为 `False`；候选与基线都为 2 ms 时 `saved_ms == 0`。

四个单元都会打印各自的构造记录：逻辑子问题的 FLOPs、每条精度路径的输入字节、补齐形状与额外输出槽，以及每条观察记录的“指令观察 / 节省 ms / 时间比”。断言全过，说明你没有把“工作量、路径声明、实现条件、执行收益”这四层混为一谈。
