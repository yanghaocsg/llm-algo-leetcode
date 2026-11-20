# 05 Triton 矩阵乘法（GEMM）— 小白讲解

> 对应 notebook：[`05_Triton_GEMM_Tutorial.ipynb`](./05_Triton_GEMM_Tutorial.ipynb)
> NPU 版：[`05_Triton_GEMM_Tutorial_NPU.ipynb`](./05_Triton_GEMM_Tutorial_NPU.ipynb)

## 一句话总结

矩阵乘法是 GPU 的主食：一个 Program 负责一块输出 tile，沿 K 维分轮"搬砖累加"，最后还能顺手把偏置和 SiLU 激活一起做掉（epilogue），省掉中间矩阵的来回搬运。

## 为什么需要它：痛点是什么

$C_{M\times N}=A_{M\times K}\cdot B_{K\times N}$：算一个输出元素要拿 A 的一行乘 B 的一列、沿 K 全长求和；而相邻输出会反复用到许多相同的数据。朴素做法（每个元素单独算，或整行整列地读）会让同样的数据从显存搬无数遍。

用 PyTorch 的 `torch.matmul` 你碰不到这些问题，但它是个黑盒调优怪兽。自己写一遍才知道 GPU 上 GEMM 真正的结构：**分块（tiling）+ 分轮累加 + 尾块掩码**。类比工厂：与其每个工人独自跑遍全仓库领料，不如把仓库切成小料箱，每个工人守一个工位，料箱轮流送来，工人只往自己的半成品上装——料箱在工位间复用，半成品（部分和）攒在手里不出门。

## 核心问题

- 一个 Program 怎么负责 BLOCK_M×BLOCK_N 的输出块，沿 K 维分轮读取 A、B 子块并累加？
- K/M/N 三个方向都可能不满整块，掩码怎么分别保护？
- 矩阵乘之后加偏置、SiLU 的"尾部处理"为什么要等 K 归约全部完成，舍入位置怎么保留？
- 输出块按分组（GROUP_SIZE_M）排列 Program ID，图的是什么？

## 解决思路：一步步看懂

1. **一个 Program 一块输出**：每轮读一个 BLOCK_M×BLOCK_K 的 A 子块和 BLOCK_K×BLOCK_N 的 B 子块，`tl.dot` 乘加进同一个累加器（accumulator，保存输出块中间结果的变量），K 走完才写回。部分和用 FP32 保存，避免每轮舍回 FP16。例：块取 16×16×16 时，一轮的 256 个 A 元素各参与 16 个输出，共 4096 次乘加——这就是块内数据复用。
2. **沿 K 推进 + 尾块**：K=35、B_K=16 要三轮（0–15、16–31、32–34），最后一轮只有 3 个有效位置、其余补零。每轮结束指针前进 `BLOCK_SIZE_K * stride_ak`（A）和 `BLOCK_SIZE_K * stride_bk`（B），不能回头重读。M、N 方向同理：加载 A 时保护 M 行和 K 段，加载 B 时保护 K 段和 N 列，写回时行列都保护。

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    k_mask = offs_k < (K - k * BLOCK_SIZE_K)            # 本轮 K 有效位置
    a = tl.load(a_ptrs, mask=(offs_am[:,None] < M) & (k_mask[None,:]), other=0.0)
    b = tl.load(b_ptrs, mask=(k_mask[:,None]) & (offs_bn[None,:] < N), other=0.0)
    accumulator += tl.dot(a, b)                          # 累加，不重置
    a_ptrs += BLOCK_SIZE_K * stride_ak                   # 两路指针各按自己的步长前进
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

3. **尾部融合（epilogue）**：偏置按输出列广播、SiLU 作用在加完偏置的结果上，都必须等全部 K 段加完再做一次——若两段部分和是 2 和 −3，完整结果是 −1、SiLU≈−0.269；逐段激活再相加≈1.619，完全不同。参考 `gemm_reference` 的舍入顺序是"矩阵乘→舍 FP16→加偏置→舍 FP16→SiLU"，内核里照抄：`accumulator.to(fp16).to(fp32)` → 加 bias → `.to(fp16).to(fp32)` → `output * tl.sigmoid(output)`。省掉的是中间矩阵写回再读取，不是精度要求。
4. **分组映射**：输出块不是按行扫完再下一行，而是 GROUP_SIZE_M=8 行一组，组内先沿 M 排、再进下一列。这样相邻的 Program ID 会用同一批 B 子块，给缓存留复用机会（实际命中率还取决于访问次序、缓存容量与硬件执行，需另测）。
5. **自动调优**：`@triton.autotune` 按 M、N、K（及偏置/激活模式）在 8 个候选配置（块大小、`num_warps`、`num_stages`）里计时挑最快的。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| GEMM | 通用矩阵乘法，GPU 负载的大头 |
| tile（分块） | 一个 Program 负责的矩形输出块及其输入子块 |
| K 维（归约维） | A 的列 = B 的行，乘加求和的方向 |
| accumulator / 部分和 | 跨 K 轮保留的 FP32 中间累加结果 |
| `tl.dot` | 块级矩阵乘，通常落到 Tensor Core 指令 |
| epilogue（尾部处理） | K 归约结束后、写回前的偏置/激活等收尾 |
| GROUP_SIZE_M | 分组映射的组高，让相邻 Program 复用 B 子块 |
| `num_warps` / `num_stages` | 每个 Program 的 warp 数 / 软件流水级数 |
| `@triton.autotune` | 按形状自动在候选配置中计时选择 |

## 小白常见疑问

**Q1：为什么不用一个 Program 算一个输出元素？**
那样 A 的一行和 B 的一列会被无数 Program 重复读取。分块让一次加载服务整块输出，把访存摊薄——这是 GEMM 优化一切的起点。

**Q2：累加器为什么放 K 循环外面？**
放里面每轮清零，最后只剩最后一段的乘积，结果直接错。K 超过一个子块的测试用例（如 K=35、K=61）专门抓这个错。

**Q3：尾块补零会不会让乘积变大？**
不会——补的零乘什么都得零，加进累加器无影响；真正要防的是越界地址被访问，所以 mask 照戴。

**Q4：加偏置、做激活能不能塞进 K 循环里省事？**
不行。偏置会被重复加、非线性会被作用在部分和上（见上面 2 与 −3 的例子），数学上就不对。

## 动手验证：怎么知道自己做对了

`test_gemm` 与 `test_gemm_epilogue` 覆盖纯矩阵乘和四种尾部模式（none / bias / silu / bias+silu），形状从 (1,1,1)、(17,19,35) 到 (127,193,61)、(1024,1536,768) 专门命中非整除和多轮 K 的尾块，FP16 下与 `torch.matmul`/`gemm_reference` 对照（atol=rtol=2e-2），并断言输入不变、非法输入被拒。还精心构造了两个 FP16 舍入探针：1024.5（应先舍到 1024，加 −1024 得 0）和 1+2⁻¹¹（先舍为 1），漏掉任何一处舍入都会跨过 FP16 格点被查出。性能用 `benchmark_gemm`（1024³ FP16）对比 `torch.matmul` 并报告 TFLOPS；另有可选单元读取 autotuner 选定配置的编译产物（寄存器、共享内存、PTX 摘要）。

## 与 NPU 版的区别

NPU 版是同一分块累加框架的精简版：只保留加载、累加、指针推进三个 TODO，去掉了偏置/SiLU 尾部融合（标题即为"分块与累加"），autotune 的 key 也相应只有 M、N、K。设备换成 `npu`、测试函数改用 `torch.npu.is_available()`，分组映射与 K 循环算法与 CUDA 版一致。

