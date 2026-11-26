# 13. 性能分析与瓶颈定位 — 小白讲解

> 对应 notebook：[`13_Profiling_and_Bottleneck_Analysis.ipynb`](./13_Profiling_and_Bottleneck_Analysis.ipynb)

## 一句话总结
本节拿一条构造好的性能记录（400 亿次浮点运算、80 亿字节 DRAM 搬运、内核 20 ms，另有 3 ms 搬运、2 ms 同步、1 ms 启动的暴露时间），用 `derive_resource_metrics` 算出算术强度 5 FLOP/B、2 TFLOP/s、400 GB/s；再用参考上界 120 TFLOP/s 与 3000 GB/s 通过 `attribute_bottleneck` 得到机器平衡点 40 FLOP/B 和 Roofline 候选；最后用 `build_discrimination_experiment` 把候选假设变成"只改一个变量"的对照实验。核心结论是：先统一记录的范围和单位，再判断该改计算、访存还是等待，不能看到内核慢就直接换硬件。

## 为什么需要它：痛点是什么
初学者拿到一段 profiling 结果，常见的反应有三种：看到内核跑了 20 ms，就说"太慢了，要换卡"；看到 GPU 利用率低，就认定 kernel 写得差；或者把内核时间、搬运时间、同步时间直接相加，却不知道这些区间在真实时间线上是否重叠。更隐蔽的错误是单位不统一：FLOPs 按十进制、带宽按 GiB/s、时间按微秒，算出来的数看着合理，结论却是错的。

生活类比：体检报告。验血单上每个指标都有单位和参考范围，医生不会因为某一项超出参考值就立刻下诊断，而是先确认采样条件是否一致，再看这个指标和症状是否对得上，最后安排一项只改变一个条件的复查。性能分析也是这样：原始计数器只是"化验单"，算术强度和 Roofline 是"参考范围"，单变量实验才是"复查"。

本节所有数字都是手工构造的教学输入，不是某张真实显卡的实测值。notebook 明确把 `time_model` 限定为 `nonoverlapping_exposed_intervals`，意思是你给的搬运、同步、启动时间已经是"暴露在关键路径上、彼此不重叠"的净时间，才能直接相加。这个前提不成立时，正确做法是先按依赖关系求关键路径，而不是套公式。

## 核心问题
- 怎样把一条原始记录（FLOPs、DRAM 字节、内核时间、各类等待）换算成可比较的算术强度、实际吞吐、实际带宽和暴露时间占比？
- 机器平衡点、Roofline 上界和"计算受限 / 访存受限 / 两者相当"的判定之间是什么关系？
- 记录里的实际吞吐离参考上界很远时，应该先怀疑什么？筛查线 70% 和容差 5% 分别管什么？
- 为什么内核时间减半、阶段总时间却可能一点没变？怎样设计一个只改一个变量的判别实验？
- 观察到"搬运时间最长"时，为什么不能直接说"瓶颈就是搬运"？

## 解决思路：一步步看懂

### 第 0 步：先分清证据来自哪一层
notebook 开头给了一张证据表，三种工具回答的问题不同：

| 证据层级 | 典型来源 | 本节用到的字段 |
|:---|:---|:---|
| 阶段与算子 | `torch.profiler` | 阶段耗时、算子耗时、调用次数 |
| 系统时间线 | Nsight Systems | 搬运、同步、启动与暴露等待 |
| Kernel 指标 | Nsight Compute | FLOPs、DRAM 搬运字节、实际吞吐与停顿原因 |

这张表的用意是：`torch.profiler` 告诉你是谁慢，Nsight Systems 告诉你在等谁，Nsight Compute 告诉你内核内部为什么慢。三者不能互相替代。本节的计算器只处理已经整理好的标量记录，不负责采集。

采集时还要固定三个口径：同一工作负载、同一测量区间、同一数据类型。换 batch、换精度、把 warmup 算进去、或者只截取时间线的一段，都会让 FLOPs 和字节数对不上。notebook 把"理论 FLOPs、最低搬运量与硬件计数器实测值应分别标注"写在导读里，就是为了避免把不同来源的数混进同一个公式。

### 第 1 步：Q1 `derive_resource_metrics`——把记录变成四个可比指标
函数先做严格的输入检查，再算四个派生量。检查部分值得逐条看，因为它定义了"什么记录才算可比"：

```python
required = {'flops', 'dram_bytes', 'kernel_ms', 'transfer_ms',
            'sync_ms', 'launch_ms', 'time_model'}
if set(record) != required:            # 键必须不多不少
    raise ValueError(...)
if record['time_model'] != 'nonoverlapping_exposed_intervals':
    raise ValueError(...)              # 只接受已去重、互不重叠的暴露区间
```

注意这里是 `!=`，不是"至少包含"。多给一个字段、少给一个字段都会报错，目的是防止不同口径的记录混在一起比较。接着要求六个数值都是有限的 `int`/`float`，`flops`、`dram_bytes`、`kernel_ms` 必须大于 0，三类暴露时间不能为负。

四个派生量的定义是：

$$\text{算术强度}=\frac{\text{FLOPs}}{\text{DRAM bytes}},\qquad
\text{实际吞吐}=\frac{\text{FLOPs}}{\text{kernel time}},\qquad
\text{实际带宽}=\frac{\text{DRAM bytes}}{\text{kernel time}}$$

代码里吞吐再除以 $10^{12}$ 换成 TFLOP/s，带宽除以 $10^9$ 换成 GB/s：

```python
'achieved_tflops': record['flops'] / (record['kernel_ms'] / 1000) / 1e12,
'achieved_dram_gb_s': record['dram_bytes'] / (record['kernel_ms'] / 1000) / 1e9,
```

用示例记录 `profile_record` 手算一遍：

- 算术强度 $=4.0\times10^{10}\div8.0\times10^{9}=5$ FLOP/B。意思是每从 DRAM 搬 1 字节，平均做 5 次浮点运算。
- 实际吞吐 $=4.0\times10^{10}\div(20\div1000)\div10^{12}=2$ TFLOP/s。
- 实际带宽 $=8.0\times10^{9}\div0.02\div10^{9}=400$ GB/s。
- 阶段总时间 $=20+3+2+1=26$ ms；暴露占比分别是 $3/26$、$2/26$、$1/26$。

这里最容易写错的是分母：吞吐和带宽都只用 `kernel_ms`，因为分子是内核完成的工作；而暴露占比用四段时间之和 `total_ms`。如果把总时间当吞吐分母，会得到 $4\times10^{10}/0.026/10^{12}\approx1.54$ TFLOP/s，这不是内核交付的吞吐，而是整段流程的平均值，含义完全不同。

`exposed_share` 只返回三类等待的占比，不含内核本身：搬运 $3/26\approx11.5\%$、同步 $2/26\approx7.7\%$、启动 $1/26\approx3.8\%$，三者合计约 23.1%，剩下约 76.9% 才是内核时间。这个设计是有意的：内核占比通常最大，把它也放进去会稀释等待项，不容易看出"等待是否在关键路径上"。另外，占比的分母是 `total_ms` 而不是 `kernel_ms`，所以三项之和不会等于 1；如果你想让它们相加为 1，得自己把内核项补上。

还要注意 `total_ms` 的直接相加只在 `nonoverlapping_exposed_intervals` 下成立。真实时间线里，内核执行期间可能同时有搬运；这时四段时间不是互斥区间，先把它们相加会重复计算重叠部分。正确做法是按依赖关系找关键路径，或者只把"确实暴露、没有被隐藏"的时间填进记录。notebook 用显式的 `time_model` 字段把这个前提写死，就是为了让误用尽早报错。

### 第 2 步：Q2 `attribute_bottleneck`——用参考上界形成假设
有了四个指标，下一步是回答"这段 kernel 更像计算受限还是访存受限"。函数接收同一设备、同一精度路径的参考上界 `reference_tflops=120` 和 `reference_dram_gb_s=3000`，先算机器平衡点：

$$\text{machine balance}=\frac{\text{参考计算吞吐}\times1000}{\text{参考带宽}}
=\frac{120\times1000}{3000}=40\ \text{FLOP/B}$$

再算这段记录在 Roofline 上的上界：

$$\text{roofline limit}=\min\left(\text{参考吞吐},\ \frac{\text{算术强度}\times\text{参考带宽}}{1000}\right)$$

示例记录是 $\min(120,\ 5\times3000/1000)=\min(120,15)=15$ TFLOP/s。也就是说，在 3000 GB/s 的带宽下，算术强度只有 5 的工作最多只能跑到 15 TFLOP/s；120 TFLOP/s 的计算峰值根本喂不饱。

然后判定类别，注意用的是严格不等号：

```python
bound = "memory" if intensity < machine_balance else (
    "compute" if intensity > machine_balance else "balance")
```

算术强度 5 < 40，所以是 `memory`。效率是实际吞吐除以上界：

$$\text{efficiency}=\frac{2}{15}\approx13.3\%$$

低于默认筛查线 `efficiency_screen=0.70`，于是内核假设被标成 `kernel_underutilized`。这一步的语义是"连这类上界的 70% 都没到，先别急着归因到计算或访存，检查布局、驻留、指令和工作量估算"。如果效率 ≥ 70%，标签才会按算术强度变成 `kernel_memory`、`kernel_compute` 或 `kernel_balance`。

`exposed_ms` 把内核时间和三类等待放在一起：

```python
exposed_ms = {
    kernel_hypothesis: record["kernel_ms"],   # 键名随上一步变化
    "transfer": record["transfer_ms"],
    "synchronization": record["sync_ms"],
    "launch": record["launch_ms"],
}
primary = "recheck_reference" if inconsistent else max(exposed_ms, key=exposed_ms.get)
```

示例里内核 20 ms 最长，所以 `primary_hypothesis` 是 `kernel_underutilized`。但"标签"和"最长的暴露项"是两件事：在搬运主导的例子里，效率只有 44.4%，内核标签仍是 `kernel_underutilized`，而 `max` 选出来的是 14 ms 的 `transfer`，最终主假设是 `transfer`。这正是 notebook 想强调的：效率筛查和关键路径归因要分开看。

最后一个保护是参考一致性。若实际吞吐超过参考上界的 $1+\text{tolerance}=1.05$ 倍，说明记录和参考至少有一个口径不对（单位、计数范围、精度路径），函数返回 `reference_consistency="recheck_reference"`，主假设也变成复查参考，而不是继续归因。notebook 用 `reference_tflops=1, reference_dram_gb_s=100` 构造出这个情形：上界是 $\min(1, 5\times100/1000)=0.5$ TFLOP/s，实际 2 TFLOP/s，效率 4 倍，明显不合理。

把五个构造用例的结果列成表，读起来更清楚：

| 用例 | 关键改动 | 算术强度 | Roofline 上界 | 效率 | 主假设 |
|:---|:---|:---:|:---:|:---:|:---|
| 原始记录 | 无 | 5 | 15 | 13.3% | `kernel_underutilized` |
| 带宽侧 | kernel 3 ms，等待 0.3/0.2/0.1 | 5 | 15 | 88.9% | `kernel_memory` |
| 计算侧 | FLOPs 4e12，kernel 40 ms | 500 | 120 | 83.3% | `kernel_compute` |
| 交点 | FLOPs 3.2e11，kernel 3.2 ms | 40 | 120 | 83.3% | `kernel_balance` |
| 搬运主导 | kernel 6 ms，transfer 14 ms | 5 | 15 | 44.4% | `transfer` |

交点这个例子值得手算：$3.2\times10^{11}\div8\times10^{9}=40$，正好等于机器平衡点，所以 `roofline_bound` 是 `"balance"`；此时计算上界和带宽上界都是 120 TFLOP/s，实际 100 TFLOP/s，效率 83.3%。

这张表不只是五个结论，它对应五种不同的"下一步"。读懂每一行的走向，比记住标签更重要：

- **原始记录（`kernel_underutilized`）**：效率只有 13.3%，说明内核离两类上界都很远。此时既不能说是计算受限，也不能说是访存受限；应该先查线程块大小、占用率、数据布局、指令选择和实际工作量估算。notebook 给的 `block_size` 和 `occupancy` 观察指标就是为这种情况准备的。
- **带宽侧（`kernel_memory`）**：效率 88.9%，已经贴近带宽上界；下一步看数据布局、缓存命中、实际 DRAM 搬运量是否还能下降。注意算术强度 5 没变，变的是内核时间从 20 ms 降到 3 ms，所以实际带宽从 400 GB/s 升到约 2667 GB/s（$8\times10^9/0.003/10^9$），但仍低于参考 3000 GB/s。
- **计算侧（`kernel_compute`）**：效率 83.3%，贴近计算上界；下一步看矩阵指令是否用满、精度路径是否正确、寄存器/共享内存是否限制了驻留。此时再优化访存收益有限，因为算术强度 500 已经远高于平衡点 40。
- **交点（`kernel_balance`）**：两条上界在 120 TFLOP/s 重合，计算和访存都可能解释现象。notebook 允许改 `tile_size` 或 `compute_kernel`，但如果只改一个变量后阶段时间仍不变，`revisit_when` 会提示"两条路径都解释不了阶段时间"，需要继续找第三个原因。
- **搬运主导（`transfer`）**：内核本身只跑了 6 ms、效率 44.4%，但 14 ms 的暴露搬运才是关键路径。下一步应查输入准备、依赖关系、主机提交和锁页内存，而不是继续压内核。

这也解释了为什么 `primary_hypothesis` 和 `kernel_hypothesis` 要分成两个字段：前者回答"关键路径上最长的是谁"，后者回答"内核相对资源上界处于什么状态"。两者可以不一致，读报告时必须一起看。

### 第 3 步：Q3 `build_discrimination_experiment`——把假设变成单变量实验
得到主假设还不够，notebook 要求为它生成一个"只改一个条件"的实验计划。函数里有一张 `plans` 表，每个假设对应三样东西：允许改的变量、同时观察的指标、出现什么现象时要回头复查。

| 假设 | 允许的变化 | 观察指标 | 复查信号 |
|:---|:---|:---|:---|
| `kernel_memory` | `tile_size` / `layout` | `dram_bytes`、`kernel_ms`、`stage_ms`、`max_abs_error` | 字节降了但内核时间没变 |
| `kernel_compute` | `compute_kernel` | `achieved_tflops`、`kernel_ms`、`stage_ms`、`max_abs_error` | 内核快了但阶段时间没变 |
| `kernel_balance` | `tile_size` / `compute_kernel` | DRAM 字节、内核时间、阶段时间、误差 | 两条路径都解释不了阶段时间 |
| `kernel_underutilized` | `block_size` | `kernel_ms`、`occupancy`、阶段时间、误差 | 占用率变了但内核时间没变 |
| `transfer` | `pin_memory` / `async_copy` | `exposed_transfer_ms`、阶段时间、主机内存、误差 | 重叠变多但阶段时间没变 |
| `synchronization` | `sync_placement` | `sync_ms`、阶段时间、误差 | 同步降了但阶段时间没变 |
| `launch` | `graph_capture` | `launch_count`、`launch_ms`、阶段时间、误差 | 启动少了但阶段时间没变 |

这些变量名不是随便起的，每个都对应一种真实的实现改动：

- `tile_size`：一次处理/搬入多少数据。分块变大通常减少重复搬运，但可能压低占用率；分块变小则相反。
- `layout`：数据在内存里的排列（行主序、转置、交错等）。它直接影响合并访问和缓存命中，所以同时观察 `dram_bytes` 与 `kernel_ms`。
- `compute_kernel`：换一种矩阵乘或激活实现，可能改变指令组合、精度路径和寄存器用量，所以主看 `achieved_tflops`。
- `block_size`：线程块大小，影响一个 SM 上能驻留多少 Warp。`kernel_underutilized` 只允许改这个，是因为在没贴近上界之前，先查占用率最合理。
- `pin_memory`：把主机侧的传输缓冲放进锁页内存，减少额外的页锁定/拷贝，直接作用于暴露搬运时间。
- `async_copy`：异步拷贝，让搬运和计算有机会重叠；它与 `pin_memory` 的组合是传输优化的常见入口。
- `sync_placement`：同步点放在哪里。放得太频繁会把本可并行的阶段串起来，所以要同时看 `sync_ms` 和阶段时间。
- `graph_capture`：用 CUDA Graph 捕获重复的启动序列，减少每次的主机提交开销；观察 `launch_count` 和 `launch_ms`。

函数会校验两件事：主假设必须在表里（`recheck_reference` 不在，直接报"不支持的瓶颈假设"）；`single_change` 必须是该假设允许的精确字符串（`'tile_or_layout'` 这种合并写法不行，`memory_case` 也不能改 `pin_memory`）。每个计划都保留 `max_abs_error`，因为任何性能改动都要先确认数值没算错。

notebook 最后用一条构造记录演示"内核变快、阶段不变"：内核从 20 ms 降到 10 ms，但为了让数据可用，搬运从 3 ms 涨到 13 ms，总时间仍是 26 ms。此时吞吐从 2 TFLOP/s 变成 4 TFLOP/s，内核时间比是 2.0，但阶段收益为零。`revisit_when` 提示继续检查新增的 10 ms 花在哪，而不是宣布优化成功。

为什么这个对照仍然算"单变量"？因为输入、输出、精度路径和阶段目标都没变，只换了"计算内核"这一个实现条件；搬运增加到 13 ms 不是同时改的第二个实验变量。但它是内核改动带来的副作用，所以必须记录到 `stage_ms` 里。如果你只记录 `kernel_ms`，会得到"快了 2 倍"的结论；把 `stage_ms` 一起看，才发现收益被抵消。这正是 notebook 在计划里同时要求 `kernel_ms` 和 `stage_ms` 的原因。

还要注意这个计算器的边界：它不估算 speedup，也不证明你换的内核在真实设备上就是这样。`reference_tflops` 和 `reference_dram_gb_s` 必须来自同一设备、同一精度路径的参考值；真实记录还要核对测量区间、调用次数、是否包含数据准备，以及理论 FLOPs/最低搬运量与硬件计数器实测值的区别。把这些前提带进后续的完整负载复测，单变量实验才有意义。

## 关键概念速查表
| 术语 | 大白话解释 |
|:---|:---|
| Profiling（性能剖析） | 记录程序实际怎么执行：每段花了多久、搬了多少数据、谁在等谁 |
| 算术强度 | 每搬 1 字节 DRAM 做多少次浮点运算，单位 FLOP/B |
| 机器平衡点 | 参考计算峰值 ÷ 参考带宽，Roofline 两条上界的交点 |
| Roofline 上界 | 给定算术强度和带宽时，理论能达到的最高吞吐 |
| 实际吞吐 / 实际带宽 | 用记录里的 FLOPs、字节数除以内核时间得到的交付速度 |
| 暴露时间（exposed time） | 没有被计算隐藏、真实拖长关键路径的搬运/同步/启动时间 |
| 关键路径 | 决定整体完成时间的那条最长依赖链，重叠区间不能直接相加 |
| 效率筛查线 | 默认 0.70：实际吞吐不到上界 70% 就先查利用率，不急着归因 |
| 参考容差 | 默认 0.05：超过参考上界 5% 视为口径不一致，需要重查 |
| 单变量判别实验 | 只改一个实现条件，同时看目标指标、阶段时间和数值误差 |
| `revisit_when` | 实验后出现该现象就要继续解释，不自动等于假设被推翻 |

## 小白常见疑问

**Q1：算术强度只有 5，为什么效率是 13.3%，而不是 5/40？**
`5/40` 是算术强度相对平衡点的位置，不是效率。效率的分子是实际吞吐 2 TFLOP/s，分母是这条记录在 Roofline 上的上界 15 TFLOP/s，所以是 `2/15`。如果带宽更高、上界变成 120，同样的 2 TFLOP/s 效率就只有 1.67%，说明"离上界多远"取决于参考条件。

**Q2：内核 20 ms、搬运 3 ms、同步 2 ms、启动 1 ms，总时间为什么是 26 ms？**
因为 `time_model` 声明这些都是"互不重叠、已经去重"的暴露区间，串行相加就是 26 ms。真实时间线里搬运可能和计算重叠，那时不能直接加，要先按依赖求关键路径；notebook 用 `overlapping_intervals` 测试这种输入会被 `derive_resource_metrics` 拒绝。

**Q3：效率超过 100% 不是好事吗，为什么反而要复查？**
实际交付不可能超过同一设备、同一精度路径的理论上界。超过 1.05 倍通常意味着记录和参考不同源：比如 FLOPs 把稀疏算成了稠密、带宽按 GiB 而不是 GB、或者参考值是另一张卡。函数返回 `recheck_reference`，是让你先统一口径，而不是庆祝"超常发挥"。

**Q4：搬运时间最长，为什么主假设不一定是搬运？**
`primary_hypothesis` 取的是 `exposed_ms` 里最大的项，所以 14 ms 的搬运确实会成为主假设；但效率标签仍可能是 `kernel_underutilized`，因为 6 ms 内核只跑到上界的 44.4%。两个字段回答不同问题：一个说"关键路径上谁最长"，另一个说"内核离资源上界有多远"，要一起读。

**Q5：内核时间减半，为什么阶段时间可以完全不变？**
Q3 的构造记录就是这样：内核 20→10 ms，搬运 3→13 ms，总时间仍是 26 ms。内核收益被新增的数据准备开销吃掉了。所以判别实验必须同时记录 `kernel_ms` 和 `stage_ms`，只看前者会把"局部变快、端到端没变"误判成优化成功。

**Q6：`reference_tflops` 和 `reference_dram_gb_s` 应该填什么？填错了会怎样？**
应该填同一设备、同一精度路径、同类工作负载的参考上界，例如该卡在 FP16 下的峰值吞吐和可持续 DRAM 带宽。填得过高，效率会被压低，容易误判成 `kernel_underutilized`；填得过低，效率可能超过 $1+\text{tolerance}$，函数返回 `recheck_reference`。所以返回值里专门保留了 `reference_consistency`，提醒你先核对参考，而不是继续归因。筛查线 0.70 和容差 0.05 都是可改的示例条件，换场景时可以和参考值一起调整，但要在报告里写清楚。

## 动手验证：怎么知道自己做对了
本节没有留空的 TODO，三个函数都是完整参考实现，验证靠单元末尾的 `assert`。notebook 没有保存执行输出，所以以断言和函数返回值为准；运行后每个单元会打印一行摘要。

`derive_resource_metrics` 的断言：
- `arithmetic_intensity_flop_per_byte == 5`、`achieved_tflops == 2`、`achieved_dram_gb_s == 400`、`total_ms == 26`；
- `exposed_share == {"transfer": 3/26, "sync": 2/26, "launch": 1/26}`；
- 把 `time_model` 改成 `overlapping_intervals` 必须抛 `ValueError`，否则断言失败并提示"重叠区间不能直接相加"。

`attribute_bottleneck` 的断言：
- 原始记录：`primary_hypothesis == "kernel_underutilized"`，且 `reference_efficiency` 与 `2/15` 满足 `math.isclose`；
- 带宽侧 `kernel_memory`、计算侧 `kernel_compute`、交点 `roofline_bound == "balance"` 且主假设 `kernel_balance`、搬运主导 `transfer`；
- 把 `sync_ms` 改成 30 得 `synchronization`，把 `launch_ms` 改成 30 得 `launch`；
- 参考改成 `reference_tflops=1, reference_dram_gb_s=100` 时得 `recheck_reference`。

`build_discrimination_experiment` 的断言：
- `memory` 计划的 `single_change == "tile_size"`；
- `compute` 计划的 `target_metrics == ["achieved_tflops", "kernel_ms", "stage_ms", "max_abs_error"]`，`revisit_when == "kernel_faster_but_stage_ms_unchanged"`；
- `balance` 的观察指标包含 `max_abs_error`，`transfer` 的包含 `host_memory_mib`；
- 三种非法输入（`tile_or_layout`、给 memory 假设配 `pin_memory`、给 `recheck_reference` 配 `tile_size`）都必须抛 `ValueError`。

最后的构造对照断言 `profile_record["kernel_ms"] / candidate_record["kernel_ms"] == 2.0`、`candidate_metrics["achieved_tflops"] == 4.0`、`candidate_metrics["total_ms"] == resource_metrics["total_ms"] == 26.0`，并打印"内核 20→10 ms，暴露搬运 3→13 ms，阶段仍为 26 ms"。这些全过，说明你已经能把记录、上界、归因和判别实验串起来了。
