# 03 Triton 融合 SwiGLU — 小白讲解

> 对应 notebook：[`03_Triton_Fused_SwiGLU.ipynb`](./03_Triton_Fused_SwiGLU.ipynb)
> NPU 版：[`03_Triton_Fused_SwiGLU_NPU.ipynb`](./03_Triton_Fused_SwiGLU_NPU.ipynb)

## 一句话总结

把"SiLU 激活再乘门控"两步合成一个 kernel，省掉中间张量的一次写和一次读——但融合不只是拼代码，还要保住浮点数中间精度，结果才能和分步算的对上。

## 为什么需要它：痛点是什么

SwiGLU 是 LLM 里常见的激活：两路线性投影的输出 x 和 y，先算 SiLU(x)（SiLU(x)=x·σ(x)，σ 是 Sigmoid，把值压到 0 和 1 之间当"阀门"），再逐元素乘 y。

朴素 PyTorch 写法是两步：

```python
u = torch.nn.functional.silu(x)   # 第一步：算出中间张量 u，写进显存
z = u * y                         # 第二步：把 u 读回来，再乘 y
```

u 一写一读看着无害，但按逐元素读写计数，分离路径共 5N 次访存（读 x、写 u、读 u 和 y、写 z），融合后只要 3N 次（读 x、读 y、写 z）。显存带宽是 GPU 最贵的资源，中间结果马上就被消费，何必去显存绕一圈？这就像做菜时切好的菜不必先放进冰箱再拿出来下锅——直接递给下一道工序就行。

但融合有个隐蔽的坑：**省掉中间张量 ≠ 可以改变中间计算精度**。参考路径的 SiLU 结果是与输入同类型的（比如 FP16），你在内核里一直用 FP32 算到底，舍入位置变了，逐位结果就可能不同。

## 核心问题

- 两步计算怎么放进同一个 kernel，读写次数从 5N 降到 3N？
- 同一位置的两路输入（x 和 y）怎么保证配对读取、配对写回？
- FP16 输入下，中间精度怎么处理才能与参考路径的舍入顺序一致？

## 解决思路：一步步看懂

1. **沿用一维分块**：把连续张量按存储顺序当长度 N 的向量，Program p 的全局位置 `i = p*B + r`（r 是块内偏移；wrapper `triton_fused_swiglu` 取 `block_size = 1024`，offsets 和 mask 的写法与 01 篇相同）。
2. **两路同位读取**：x 和 y 必须用**同一个** offsets 和 mask。若两路用了不同起点，结果形状照样正确，却把不同位置的门控值乘在了一起——这种错误数值对照才能抓出来。
3. **保留中间舍入顺序（本篇灵魂）**：

```python
x_f32 = x.to(tl.float32)                 # 先升到 FP32 算激活，避免中途舍入
silu = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)  # SiLU 算完立刻转回输入类型（关键舍入点）
out = silu * y                           # 第二次乘法用转回后的 SiLU 结果
tl.store(out_ptr + offsets, out, mask=mask)      # 只写有效位置
```

参考 `reference_swiglu` 是 `silu(x) * y`：SiLU 张量先落成输入类型、再乘 y。内核里照抄这个顺序——FP32 算 Sigmoid 和第一次乘法、转回输入类型、再做第二次乘法。若把 Sigmoid 提前舍入到 FP16，后面的结果就可能变。
4. **尾块照旧**：无效位置补零，`x_i = 0` 时输出恰好为零；读和写的 mask 不能互相替代。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| 算子融合（fusion） | 把多个连续算子放进同一个 kernel，省中间显存读写 |
| SwiGLU | SiLU(x) 逐元素乘 y 的门控激活 |
| SiLU / Sigmoid | SiLU(x)=x·σ(x)；σ 把输入压到 (0,1) 当阀门 |
| 中间张量 | 分步实现里暂存到显存的 u，融合后被消灭 |
| 舍入（rounding） | 低精度格式表示不了的数被推到最近的可用值 |
| FP16 / FP32 | 16 位 / 32 位浮点格式，FP16 精度低但省带宽 |
| atol / rtol | 绝对/相对容差，允许误差不超过 atol + rtol·\|参考值\| |
| eager / 图编译 | PyTorch 逐条执行 vs torch.compile 整图编译执行 |

## 小白常见疑问

**Q1：融合后数值会不会和 PyTorch 不一样？**
会有微小差异，但本篇刻意让舍入顺序与参考一致（SiLU 完成后转回输入类型再乘 y），在共同容差内对得上。"保持中间精度"和"要求逐位相同"是两个不同条件。

**Q2：都是逐元素运算，直接把两行 Python 拼进一个 `@triton.jit` 就行了吧？**
结构上是的，但两路输入的 offsets/mask 必须严格同位，且精度链路要对齐参考路径——这正是 TODO 1–3 各自考察的点。

**Q3：为什么用 FP32 算 Sigmoid，FP16 不行吗？**
可以算，但提前舍入会改变后续乘法的结果，偏离参考路径的中间舍入位置；先 FP32 算完整个 SiLU 再转回，才是与"分步得到 FP16 SiLU 张量"等价的做法。

**Q4：融合一定更快吗？**
不一定。Step 5 的三路径实验（eager / torch.compile / 手写）显示：RTX 5070 Ti 记录里多数小输入上 eager 完整调用最快，手写路径快于图编译，FP32 最大输入才出现局部收益。减少设备调用数不等于完整调用更快。

## 动手验证：怎么知道自己做对了

- **正确性**：`test_fused_swiglu` 在长度 1023/1024/1025（覆盖不满块、整块、尾块）和多维连续输入、FP16/FP32 下与 `reference_swiglu` 逐元素对照，并包含正、负、零和大幅值输入，同时断言输出属性与输入不变。
- **性能（Step 5，可选）**：固定负载（4096/65536/1048576 元素 × 两种类型，种子 42），比较 eager、Inductor 图编译（固定形状、fullgraph）与手写融合的完整调用耗时；预热 5 轮、每窗口 50 次、共 6 轮且三种路径排列平衡，另用 profiler 核对设备上的 kernel 数（eager 两个、compile 和手写各一个）。

## 与 NPU 版的区别

NPU 版设备换成 `npu`（导入 `torch_npu`），kernel 算法与舍入顺序完全相同；Step 5 的三路径实验按同一协议在昇腾 Ascend 910 上复测（记录 `swiglu_three_path_ascend910_20261116_v1.json`）。两点 NPU 特有差异：一是 NPU 不支持 float64，设备端 `.double()` 会被静默降级为 FP32，所以高精度数学参考改在 CPU 上用真 FP64 计算；二是设备活动来自 torch_npu profiler 导出的 chrome trace（eager 对应 aclnnMul + aclnnSilu 两个 kernel）。结论与 GPU 版一致：eager 完整调用最快，手写路径稳定快于图编译。

