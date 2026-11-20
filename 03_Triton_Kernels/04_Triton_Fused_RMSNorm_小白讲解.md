# 04 Triton 融合 RMSNorm — 小白讲解

> 对应 notebook：[`04_Triton_Fused_RMSNorm.ipynb`](./04_Triton_Fused_RMSNorm.ipynb)
> NPU 版：[`04_Triton_Fused_RMSNorm_NPU.ipynb`](./04_Triton_Fused_RMSNorm_NPU.ipynb)

## 一句话总结

从"每个位置独立算"升级到"一行共用一个尺度"：RMSNorm 要先对整行做归约（reduction，把多个数汇总成一个），再把尺度广播回每一列并乘各自权重——这一篇教你一个 Program 怎么吃完一整行。

## 为什么需要它：痛点是什么

RMSNorm 是 LLM 每层都要过的一关：对一行 N 个特征，先算均方值，得到整行共享的均方根倒数，再逐列乘权重 γ。公式：

$$s=\frac{1}{N}\sum_{j=0}^{N-1}x_j^2,\qquad r=(s+\epsilon)^{-1/2},\qquad y_i=x_i\cdot r\cdot \gamma_i$$

文字版：每个元素的平方加起来除以 N 得到均方值；加一个很小的正数 ε（epsilon，防止全零行除零）后开方取倒数，得到整行的"缩放旋钮" r；每个元素乘 r 再乘自己那一列的权重。

用朴素 PyTorch 写是好几条算子（平方、mean、rsqrt、乘、乘权重），中间结果反复进出显存；更要命的是 PyTorch 看不到"这本来就是一行一算"的结构。而自己写 kernel 时会撞上三个新问题：多维张量怎么映射到"逻辑行"？块宽补齐的零会不会污染分母？FP16 的数一平方会不会溢出？这三个问题任何一个答错，整行都被错误缩放。

## 核心问题

- 多维张量（如 (2,3,4)）怎么拆成"多行 × 固定列宽"，行地址怎么算？
- 补齐列（padding 出来的零）为什么不能进入归约分母？
- 为什么平方与累加必须用 FP32，权重怎么和列正确配对？

## 解决思路：一步步看懂

1. **一个 Program 一行**：把形状 (2,3,4) 的连续张量看成 6 行 × 4 列，行起点依次是 0、4、8、12、16、20。kernel 里 `row = tl.program_id(0)`，行地址按真实列数 N 推进：`row * N + columns`。注意原始首维步长 12 跨过的是三行，不能拿来当行间距。wrapper `triton_rmsnorm` 会先把输入 `.contiguous()`，行宽上限 4096。
2. **读行、读权重**：权重只随列号变化，和行无关——同一份 γ 作用在每一行上：

```python
columns = tl.arange(0, BLOCK_SIZE)              # BLOCK_SIZE = next_power_of_2(N)
mask = columns < N
x = tl.load(X_ptr + row * N + columns, mask=mask, other=0.0)  # 当前行，补零
w = tl.load(W_ptr + columns, mask=mask, other=0.0)            # 权重按列取
```

3. **FP32 归约，除以真实 N**：

```python
x_f32 = x.to(tl.float32)              # 先升精度再平方
sum_sq = tl.sum(x_f32 * x_f32, axis=0)  # 沿行归约：N 个数汇总成一个平方和
inverse_rms = tl.rsqrt(sum_sq / N + eps)  # 除以真实列数 N，再加 eps
```

两个易错点：行宽 N=3、块宽 4 时加载的是 (1,2,3,0)，补零不影响平方和，但分母必须是 3 而不是 4；ε 加在均方值上（除完 N 再加），不是加在平方和上。FP16 的 1000 平方是 10⁶，超出 FP16 表示范围，所以平方前必须转 FP32。
4. **广播缩放写回**：`x_f32 * inverse_rms * w`——每列共享行尺度 r，但乘自己的权重；只对 `mask` 内的有效列 `tl.store`。零权重列输出零，负权重列翻转符号。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| 归约（reduction） | 把一行多个数汇总成一个结果，如求平方和 `tl.sum(..., axis=0)` |
| 逻辑行 | 最后一维是 N 个特征，前面所有维度每种索引组合算一行 |
| 广播（broadcast） | 整行共享的标量尺度自动作用于每一列 |
| RMSNorm | 均方根归一化：不减均值，只除均方根（区别于 LayerNorm） |
| epsilon（ε） | 加在分母里的正的稳定项，避免全零行除零 |
| `tl.rsqrt` | 计算 1/√x，即均方根倒数 |
| `next_power_of_2` | 把行宽向上取成 2 的幂作为 BLOCK_SIZE |
| 溢出（overflow） | FP16 最大约 65504，1000²=10⁶ 直接爆掉 |

## 小白常见疑问

**Q1：为什么不一个 Program 算一个元素，像 01 篇那样？**
每个输出都依赖整行的平方和，单元素算不了归约。让一个 Program 吃一整行，读一次、片上累加、写一次，天然是融合的。

**Q2：补齐的零会不会让分母变大？**
补零只影响加载块的形状，平方和确实多加了 0（没影响），但分母必须用真实特征数 N——用 `sum_sq / N` 而不是 `sum_sq / BLOCK_SIZE`，否则整行都被错误缩放。

**Q3：行宽很大怎么办，一个 Program 装得下吗？**
本题限定最后一维 ≤ 4096（练习约束，不是硬件通用上限）。行宽越大单个 Program 需要的资源越多；数值公式本身不规定行宽上限。

**Q4：归约对了结果就一定对吗？**
不一定。γ=(1,0,−1) 时第二个输出应为零、第三个变号——权重配错列、行地址偏移都会给出"归约对但输出错"的结果，两者要分别检查。

## 动手验证：怎么知道自己做对了

`test_triton_rmsnorm` 覆盖一维/二维/三维（含 (8,127) 非整除尾块、(2,4096) 上限行宽）、非连续输入（先经 wrapper 连续化）、零输入/大幅值/两档 ε（1e-6 和 0.25）、以及用 `torch.linspace(-1.5, 2.0, N)` 构造的非均匀权重（能抓出权重配对错误），与 `reference_rmsnorm` 逐元素对照（FP16: atol=2e-3, rtol=2e-3；FP32: atol=1e-6, rtol=1e-5），并断言输出属性、输入和权重不变。性能上 `benchmark_rmsnorm`（1024×1024 FP16，预热 10ms/计时 50ms/5 轮交替次序）对比参考与 Triton 完整调用。

## 与 NPU 版的区别

NPU 版把设备换成 `npu`（`torch.npu.is_available()`、`device="npu"`），wrapper 的连续化、行映射、FP32 归约和逐列权重算法完全一致（参考解里用 `1/tl.sqrt(...)` 表达同一数学）。测试形状、ε 档位、容差与 CUDA 版相同，benchmark 也用同样的 `triton.testing.do_bench` 交替协议，只是结果里的设备名来自 `torch.npu.get_device_name`。

