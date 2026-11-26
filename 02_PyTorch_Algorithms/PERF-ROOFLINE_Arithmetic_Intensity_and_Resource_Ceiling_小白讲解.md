# PERF-ROOFLINE. 算术强度与资源上界 — 小白讲解

> 对应 notebook：[`PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb`](./PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb)

## 一句话总结
本节用 `matmul_workload`、`roofline_bound`、`roofline_report` 三个 TODO 建立 Roofline（屋顶线）模型：由矩阵形状算运算量 $2mnk$ 与最低搬运字节，得到算术强度；再用 $\min(P, IB)$ 判断算子受带宽还是计算限制；最后用实测吞吐除以资源参考得到 $\eta$。结论：Roofline 给的是"先查搬运还是先查计算"的假设，不是实测硬件流量；$\eta>1$ 要复核测量条件，不能把比值截成 1。

## 为什么需要它：痛点是什么
看到矩阵乘法慢，第一反应常是"GPU 算力不够"或"内存太慢"，但这两个判断需要不同证据。大矩阵每个读入元素能参与很多次乘加，通常卡在计算；小矩阵还没把数据复用几次就结束了，通常卡在搬运。只看 FLOPs 总数或只看显存带宽，都可能优化错方向。
生活类比：厨房出餐速度。食材（数据）从仓库搬到灶台（内存到计算单元）要时间，厨师切配烹饪（计算）也要时间。如果菜谱让每份食材只做一道小菜，瓶颈在跑腿；如果每份食材要做很多道工序，瓶颈在厨师。Roofline 就是先算"每搬一趟食材要做多少道工序"，再判断该加人手还是加快跑腿。

## 核心问题
- 矩阵乘法的运算量与"最低搬运量"分别怎么由 $m,n,k$ 和元素字节数算出来？
- 算术强度 $I$ 的单位是什么？为什么它连接了计算与带宽？
- 机器平衡点 $I_\ast=P/B$ 怎么定？带宽侧和计算侧的边界在哪？
- 两项相等时为什么归到计算侧？这只是标签约定吗？
- 实测吞吐比参考高（$\eta>1$）时应该相信还是复核？
- 经验参考曲线与严格硬件上界差在哪？

## 解决思路：一步步看懂

**第 1 步：工作量模型（TODO 1）。** 对 $C=AB$，$A$ 为 $m\times k$、$B$ 为 $k\times n$、$C$ 为 $m\times n$，每个元素占 $d$ 字节。每个输出元素约需 $k$ 次乘加，每次乘加算 2 次浮点运算，所以
$$F\approx 2mnk,\qquad Q_{\min}=d(mk+kn+mn),\qquad I_{\mathrm{model}}=\frac{F}{Q_{\min}}$$
$F$ 单位 FLOPs，$Q_{\min}$ 单位 Byte，$I$ 单位 FLOPs/Byte。$Q_{\min}$ 假设两个输入各读一次、输出写一次，不读旧输出。参考答案：

```python
flops = float(2 * m * n * k)
minimum_bytes = float((m * k + k * n + m * n) * dtype_bytes)
```

测试 `matmul_workload(2, 3, 4, 2)`：$F=2\times2\times3\times4=48$；字节 $=(2\times4+4\times3+2\times3)\times2=(8+12+6)\times2=52$；强度 $48/52\approx0.923$。方阵 `(8,8,8,4)`：$F=2\times8^3=1024$，字节 $=3\times8^2\times4=768$。`test_dtype_scaling` 再验证：元素从 4 字节换 2 字节，FLOPs 不变、字节减半、强度翻倍——这就是"同样计算，低精度更容易落到计算侧"的原因。

**第 2 步：两类资源限制（TODO 2）。** 设计算吞吐上界 $P$（FLOPs/s）、存储层带宽 $B$（Byte/s），即使计算和搬运完全重叠，时间也不小于两者较大值：
$$T\ge\max\left(\frac{F}{P},\frac{Q}{B}\right),\qquad R(I)=\min(P,IB),\qquad I_\ast=\frac{P}{B}$$
代码先算 `bandwidth_bound = arithmetic_intensity * bandwidth_bytes_per_s`，再 `bound_flops = min(peak_flops, bandwidth_bound)`；区间判定是 `"bandwidth_bound" if bandwidth_bound < peak_flops else "compute_bound"`。注意严格小于才标带宽侧，所以两项相等时归 `compute_bound`——这只是标签约定，不代表带宽限制消失。

`test_resource_regimes`：`roofline_bound(2.0, 1000.0, 100.0)` 的机器平衡点是 $1000/100=10$，带宽限制 $2\times100=200<1000$，所以 `bound_flops=200`、`regime="bandwidth_bound"`；强度 20 时带宽限制 2000>1000，取 1000、`compute_bound`。`test_balance_point` 用强度 10 让两项都等于 1000，断言 `regime == "compute_bound"`。输入必须是有限正数，`1e308` 强度乘 `1e308` 带宽会溢出，代码检查 `isfinite(bandwidth_bound)` 后拒绝。

**第 3 步：相对吞吐（TODO 3）。** 实测时间 $T_{\mathrm{meas}}$、实测吞吐 $P_{\mathrm{meas}}=F/T_{\mathrm{meas}}$，相对屋顶线：
$$\eta=\frac{P_{\mathrm{meas}}}{R(I)}$$
参考答案只有一行 `efficiency = measured_flops / machine["bound_flops"]`，同时返回 `measurement_review_required = efficiency > 1.0`。为什么允许大于 1？因为 $P,B$ 可能来自经验参考（例如用一次矩阵乘法和复制测得），参考本身会随形状、缓存、采样波动变化；超过参考说明"这次比参考工作负载还快"，应复核计时、运算计数、形状和精度，而不是硬截断成 1 掩盖异常。

`test_relative_throughput` 固定机器参考：强度 2.0、带宽限制 200 FLOPs/s（测试里单位是抽象的）。实测 0→0、100→0.5、200→1.0、202→1.01；只有 202 的 `measurement_review_required` 为 `True`。注意 `measured_flops` 允许 0（还没跑或空算子），但不允许负数、`NaN`、`inf`。

**第 4 步：怎么用模型排查。** notebook 的 Step 3 给了四种读法：
- 低强度且贴近斜线：先怀疑搬运限制，减少流量或提高复用后看时间是否下降；
- 高强度但远低于平台：计算侧还有差距，查计时范围、内核效率和启动等待；
- 超过经验参考：重测参考与候选，核对形状、精度和缓存；
- 局部吞吐改善：只说明该算子执行变了，还要用时间线确认它是否在端到端关键路径上。

**第 5 步：GPU 经验参考实验。** Step 5 用 `COPY_ELEMENTS=64*1024*1024` 个 FP32 元素做设备内复制：源和目标各 256 MiB，所以 `copy_bytes = 2 * COPY_ELEMENTS * 4 = 512 MiB`，除以复制中位时间得到经验带宽。计算参考用 `PEAK_MATMUL_SIZE=4096` 方阵，候选用 `CANDIDATE_MATMUL_SIZE=2048`，`MATMUL_PRECISION="high"`，10 次预热、每类 20 次采样，CUDA event 取中位数。`_gpu_matmul_workload` 与 CPU 公式一致，方阵下 `minimum_bytes = 3 * size**2 * dtype_bytes`。

notebook 保存的结果是：经验带宽约 394.76 GB/s、经验计算参考约 20.04 TFLOP/s，机器平衡点约 `50.75 FLOPs/Byte`；2048 方阵候选的强度是 `2×2048³ / (3×2048²×4) ≈ 341.33 FLOPs/Byte`，远高于平衡点，所以 `regime="compute_bound"`；候选实测约 20.12 TFLOP/s，相对参考 `1.004`，即高约 0.4%。读法：模型把候选划入计算侧，下一步优先查计算路径；但候选和参考用了不同矩阵尺寸，这点差异还不能单独归因于形状、缓存或采样波动，更不能说端到端一定变快。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| FLOPs | 浮点运算次数，一次乘加算 2 次 |
| 算术强度 I | 每搬运 1 字节做了多少次浮点运算，单位 FLOPs/Byte |
| 最低搬运量 Q_min | 两个输入各读一次、输出写一次的理论字节数 |
| 计算吞吐 P | 所选精度路径每秒能做的 FLOPs 上界 |
| 带宽 B | 所分析存储层每秒能搬的字节数 |
| 机器平衡点 I* | P/B，计算限制与带宽限制相等处的强度 |
| 带宽侧 / 计算侧 | I<I* 时时间由搬运主导 / I>I* 时由计算主导 |
| 屋顶线 R(I) | min(P, I·B)，理想情况下的吞吐上界 |
| 相对吞吐 η | 实测吞吐 ÷ R(I) |
| 经验参考 | 用本机复制和 matmul 测出的 P、B，不是硬件计数器或硬上界 |
| 分块 / 复用 | 让一块数据参与更多计算，从而改变真实流量 |

## 小白常见疑问
**Q1：为什么 $F$ 是 $2mnk$，不是 $mnk$？**
每个输出元素要做 $k$ 次乘法、$k-1$ 次加法，notebook 把乘法和加法各记一次，近似为 $2mnk$。真实硬件用 FMA（乘加融合）时指令数不同，但按 FLOPs 计账这个近似是通行口径。

**Q2：算术强度低就一定慢吗？**
不一定。低强度只说明"每字节的运算少"，是否成为瓶颈还要看 $I$ 与 $I_\ast$ 的相对位置，以及真实流量、缓存命中。Roofline 给的是限制项假设，Step 3 的表格要求再用实际耗时和流量验证。

**Q3：两项相等时为什么标成 `compute_bound`？**
代码用 `bandwidth_bound < peak_flops` 判带宽侧，相等落到 else。notebook 明说"两项相等时只是标签约定，不意味着带宽限制消失"；在这种点上，计算和搬运所需时间相同，任何一侧改善都可能有效。

**Q4：$\eta=1.004$ 是不是说明候选比参考快 0.4%？**
只能说明这次测得的吞吐比参考高 0.4%。参考来自 4096 方阵、候选是 2048 方阵，尺寸不同，缓存行为也不同；notebook 明确"差异来源尚未单独验证"。正确的动作是复核测量条件，而不是宣布加速。

**Q5：这个实验能证明真实模型的端到端收益吗？**
不能。5.4 的表格最后一行写得很清楚：本页只形成算子级资源假设，仍需结合执行时间线确认该矩阵乘法是否位于目标工作负载的关键路径。Roofline 是定位工具，不是端到端结论。

## 动手验证：怎么知道自己做对了
`run_roofline_tests()` 依次跑 8 个测试，全过打印 `✅ 工作量、资源分界与相对吞吐测试通过`。TODO 未完成会抛 `NotImplementedError`。
- `test_matmul_workload`：`(2,3,4,2)` 的 `flops == 48`、`minimum_bytes == 52`、强度误差小于 `1e-12`；`(8,8,8,4)` 的 `flops == 1024`、`minimum_bytes == 768`。
- `test_dtype_scaling`：2 字节版与 4 字节版 FLOPs 相同，字节是后者一半，强度是后者两倍。
- `test_invalid_workload_inputs`：四个参数分别代入 `0`、`-1`、`1.5`、`True`、`NaN`、`inf` 都抛 `ValueError`。
- `test_resource_regimes`：强度 2 得 `machine_balance == 10`、`bound_flops == 200`、`bandwidth_bound`；强度 20 得 `bound_flops == 1000`、`compute_bound`。
- `test_balance_point`：强度 10 时 `bandwidth_bound_flops == bound_flops == 1000`，标签仍是 `compute_bound`。
- `test_invalid_resource_inputs`：三个输入各自非法值都拒绝；`(1e308, 1000.0, 1e308)` 因乘积溢出拒绝。
- `test_relative_throughput`：`(0.0, 0.0)`、`(100.0, 0.5)`、`(200.0, 1.0)`、`(202.0, 1.01)`；只有 202 触发复核标记。
- `test_invalid_report_inputs`：实测吞吐 `-1/True/NaN/inf` 拒绝；强度 0、`bound_flops` 0、未知 `regime` 都拒绝。
- GPU 单元保存了 `rtx5070ti_roofline_v1` 的原始 20 次采样和汇总（带宽 394.76 GB/s、计算 20.04 TFLOP/s、强度 341.33、η 1.004）。`read_performance_result` 会从原始样本重算并核对，若字段缺失、数值非法或汇总不一致会停止。本地重跑时绝对数值随硬件变化，判断顺序不变。
