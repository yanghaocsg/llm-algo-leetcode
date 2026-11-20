# 06. Triton 持久化 GEMM 与任务分配 — 小白讲解

> 对应 notebook：[`06_Triton_Persistent_GEMM_and_Tile_Scheduling.ipynb`](./06_Triton_Persistent_GEMM_and_Tile_Scheduling.ipynb)
> NPU 版：[`06_Triton_Persistent_GEMM_and_Tile_Scheduling_NPU.ipynb`](./06_Triton_Persistent_GEMM_and_Tile_Scheduling_NPU.ipynb)

## 一句话总结

这个 notebook 教你让少量 Program（可以理解成一个工人线程块）"包干"多个输出块的矩阵乘法：做完一块领下一块，而不是一块任务雇一个工人。

## 为什么需要它：痛点是什么

矩阵乘法 $C=AB$ 切成输出 tile（小块）后，直觉写法是"一个 tile 一个 Program"。这就像搬家时来多少件家具就雇多少个搬家工：活儿再多也不怕，但工人进场退场（GPU 上的块调度）本身也有开销。

于是有人想：能不能只雇固定一小队工人，每人排队领任务，做完一件再领下一件？这就是"持久化（Persistent）"内核。但马上冒出三个新问题：任务怎么分才不重不漏？上一个任务算了一半的部分和要不要带进下一个任务？工人少了活儿会不会反而干得更慢？本节就用同一个 GEMM 对照两种分工回答这些问题。

## 核心问题

- 共有 $T$ 个输出 tile、启动 $P$ 个 Program 时，每个 Program 该领哪些 tile，才能不遗漏、不重复？
- 一个 Program 换到新 tile 时，FP32 累加器什么时候清零？
- 减少 Program 数量，到底更快还是更慢？

## 解决思路：一步步看懂

1. **两种分工，同一种计算。** 输入 $A$ 为 $M\times K$、$B$ 为 $K\times N$，把 $C$ 按 32 行 × 64 列切块。普通分工启动 $T$ 个 Program 各干一块；持久化分工只启动 $P$ 个，编号 $p$ 的 Program 依次处理 $p,\ p+P,\ p+2P,\ldots$。就像 3 个工人轮流派单：0 号工人拿第 0、3、6 件，1 号拿 1、4，2 号拿 2、5。因为任意 tile 编号除以 $P$ 的余数唯一，每件任务恰好归一个工人。

2. **CPU 端先做计划：`plan_tiles`。** 参考实现就一行核心代码：

   ```python
   workers = min(total_tiles, requested_workers)      # 任务少就把网格裁剪到任务数
   assignments = [list(range(worker, total_tiles, workers)) for worker in range(workers)]
   # 工人 w 的任务清单：w, w+workers, w+2*workers...（轮流派单，天然不重不漏）
   ```

   零任务时直接返回空列表。

3. **kernel 里把编号翻译成块坐标。** tile 编号是"排平"的，要按行优先映射回（块行，块列）：

   ```python
   tile_m = tile_id // num_n   # 除数是 tile 列数（不是矩阵列数！）
   tile_n = tile_id % num_n    # 余数是块列
   ```

4. **换任务必须清零累加器。** 参考实现用 `while tile_id < total_tiles:` 包住整个流程，每个新 tile 都执行 `accumulator = tl.zeros((BM, BN), dtype=tl.float32)`，再进入 K 循环用 `tl.dot` 累加。清零位置在 tile 循环内、K 循环外——同一 tile 沿 K 累加，不同 tile 各算各的，上一块的结果绝不能当代下一块的初值（否则就"串块"了）。

5. **领下一件任务：`tile_id += workers`。** 步长是实际 Program 数 `tl.num_programs(0)`，不是 1。步长写成 1，所有工人会挤去抢同一件任务，即使数值碰巧对（大家重复写同样的数），访问记录也会露馅。

6. **更少的 Program 不一定更快。** Step 5 的实测（RTX 5070 Ti Laptop，46 个 SM）：在 1024×1536×512 这组形状上，普通网格 0.073 ms，而 46 / 92 个 Program 的持久化版本分别是 0.254 / 0.137 ms——更慢。因为可并行的块变少了，每个 Program 的循环变长了。持久化不是单向优化。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| Program | Triton 的执行实例，可理解为一个工人线程块 |
| Persistent kernel | 少量 Program 循环领任务、长时间存活的内核写法 |
| tile | 把大矩阵切成的小块，本节输出块为 32×64 |
| `tl.num_programs(0)` | 当前网格里实际的 Program 数量，即"工人总数" |
| 行优先映射 | tile 编号先除以 tile 列数得块行、取余数得块列 |
| 累加器（accumulator） | 存部分和的 FP32 寄存器块，换 tile 必须清零 |
| SM | 流式多处理器，GPU 上承载线程块的硬件单元 |
| VISITS / OWNERS | 记录每个 tile 被访问次数和归属 Program 的调试缓冲 |
| CUDA Graph | 预先捕获、之后整段重放的执行图，用于剔除 Python 提交开销 |

## 小白常见疑问

- **Q：为什么 `tile_id += workers` 而不是 `+= 1`？** 步长是工人总数，才能形成"轮流派单"（0 号拿 0、P、2P…）。`+= 1` 会让每个工人都把所有任务做一遍，重复劳动。
- **Q：累加器不清零会怎样？** 上一块的积分数会加进下一块，结果被"污染"。测试专门让一个 Program 领多个 tile 来抓这种错误。
- **Q：Program 数量设成 SM 数是不是就绑定了 SM？** 不是。Program 编号不等于 SM 编号，实际驻留由设备调度；预算只是上限。
- **Q：持久化既然可能更慢，为什么还要学？** 它把"任务归属"和"块内计算"解耦了，是理解任务调度、后续融合内核的基础；收益要靠测量说话，不能凭感觉。

## 动手验证：怎么知道自己做对了

三层验证：CPU 端核对 `plan_tiles` 的任务列表（7 项 3 工人、余数归属、零任务）；GPU 端读取 kernel 写出的 VISITS/OWNERS 缓冲，确认每个 tile 恰好被访问 1 次且归属符合轮转计划；数值上与普通"一 Program 一块"路径逐元素一致，PyTorch 仅作数值参考。性能实验固定 32×64×32 tile、4 warp、2 流水级，用 CUDA Graph 承载 50 次固定地址启动、CUDA event 计时，正逆序配对比较。

## 与 NPU 版的区别

NPU 版的算法、TODO 结构、tile 划分完全相同，差异集中在设备相关代码：设备检查换成 `is_npu` / `torch.npu.is_available()`，张量建在 `device="npu"` 上，上下文用 `torch.npu.device`；Step 5 实验里 CUDA Graph 换成 `torch.npu.NPUGraph`、CUDA event 换成 `torch.npu.Event`，SM 数从 `torch.npu.get_device_properties(0).multi_processor_count` 读取，并额外 import `torch_npu`。
