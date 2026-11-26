# PERF-INPUT. 训练输入流水线性能 — 小白讲解

> 对应 notebook：[`PERF-INPUT_Training_Input_Pipeline_Performance.ipynb`](./PERF-INPUT_Training_Input_Pipeline_Performance.ipynb)

## 一句话总结
训练时 GPU 等数据，是因为一个批次要经过“读取/预处理 → 组批 → 主机缓冲 → H2D 搬运 → GPU 计算”好几道工序；本节先用三个 TODO 把“顺序装箱的容量账本”和“预取、复制、计算三阶段的时间线”写成可测试的公式，再用真实 GPU 上的三条 DataLoader 配置（普通内存、锁页单进程、2 进程预取）比较取批等待、H2D、步骤时间、P95 与吞吐。保存的输出里预取配置取批等待从 0.141 ms 降到 0.038 ms、吞吐从 13,612 样本/s 升到 20,303 样本/s，但锁页单进程的步骤中位数最低（0.379 ms），说明“典型延迟”和“持续吞吐/尾延迟”不是同一个指标。

## 为什么需要它：痛点是什么
初学者常把“DataLoader 慢”笼统归因成一个原因，然后随便加 `num_workers`。实际等待至少有三种来源：样本准备太慢（CPU 忙不过来）、搬运太慢（H2D 复制排队）、以及复制和计算不能重叠（默认流里复制必须等上一批算完）。这三种来源对应三个不同的开关，改错了就只是换一种等法。

生活类比：餐厅出菜。后厨（CPU 准备）要先切配，传菜员（H2D 复制）把菜从后厨送到餐桌（GPU），客人（计算）才能吃。没有提前备菜，客人吃完一道才通知后厨，下一道就要干等；传菜通道只有一条且要等客人吃完才允许走，菜就会凉在门口。加一个传菜员（预取）能减少“通知后厨”的等待，但如果后厨本身很慢，或者传菜通道和餐桌不能并行，等待仍然存在。

## 核心问题
- 一个训练批次从 Dataset 到 GPU 要经过哪些阶段？哪个阶段的等待会暴露成 GPU 空闲？
- 装箱后“有效词元、容器位置、填充位置、利用率”怎么算？顺序装箱为什么不等于最优装箱？
- 预取、锁页内存、非阻塞复制、独立复制流，四个条件分别控制什么？为什么“预取”和“复制—计算重叠”是两件事？
- 三阶段时间线怎么递推？“暴露等待” $W_i$ 的定义是什么？
- GPU 实验里 `batch_wait_median_ms`、`h2d_median_ms`、`step_median_ms`、`step_p95_ms`、`samples_per_second` 各自量什么？
- 保存结果里为什么“锁页单进程步骤中位数最低”，最终却把预取配置记为候选？

## 解决思路：一步步看懂

**第 1 步：把流水线拆成四段。** notebook 的 Step 1 表把输入路径分成：读取与预处理（原始样本、Token 序列，吃存储/CPU/内存）、批次整理（填充或装箱后的张量，吃 CPU/主机内存）、主机缓冲与搬运（锁页或普通内存，吃内存带宽/PCIe）、GPU 消费（设备张量，吃 GPU 计算与显存）。关键结论是：各阶段有独立资源且允许并行时，稳态供给受最慢阶段限制；串行执行则把各段耗时累加。队列只能吸收短时抖动，不能消除长期供给不足。锁页内存（pinned memory）是操作系统暂不换出的主机内存，可供设备异步搬运，但它仍在主机侧，不是显存。下游消费持续落后、队列填满后上游也必须等，这叫回压（背压）。

**第 2 步：TODO 1——顺序装箱。** `build_greedy_packs(lengths, capacity)` 先做输入校验：`capacity` 必须是正整数且不能是 `bool`（因为 Python 里 `True` 是 `int` 的子类），每个长度也必须是正整数；任何样本超过容量直接抛 `ValueError('存在超过 pack 容量的样本')`，不在这里偷偷截断。循环里：

```python
if current and current_tokens + length > capacity:
    packs.append(current)
    current, current_tokens = [], 0
# TODO 1
current.append(length)
current_tokens += length
```

`if current` 这个条件不能少：第一批时 `current` 是空列表，如果只判断 `current_tokens + length > capacity`，`0 + length` 不会超，倒也无事；但语义上“空容器不触发封包”更清楚。`current_tokens + length > capacity` 用严格大于，所以恰好装满（如 `[10, 10]`、容量 10）会各自成包，测试 `test_strict_lengths_and_packs` 就断言 `build_greedy_packs([10, 10], 10) == [[10], [10]]`。循环结束别忘 `if current: packs.append(current)`，否则尾包会丢；测试用 `[6, 3, 7, 2, 1]`、容量 10 得到 `[[6, 3], [7, 2, 1]]`，并断言把所有包摊平后仍等于原 `lengths`，保证“不重不漏、顺序不变”。

**第 3 步：TODO 2——容量守恒。** 公式是

$$U=\sum_i \ell_i,\qquad A=JC,\qquad Z=A-U,\qquad \eta=\frac{U}{A}\quad(A>0)$$

其中 $U$ 是有效输入位置（所有样本长度之和），$J$ 是非空容器个数，$C$ 是每个容器的容量，$A$ 是容器位置总数，$Z$ 是填充位置，$\eta$ 是利用率。参考实现：

```python
useful_tokens = sum(sum(pack) for pack in packs)
container_tokens = len(packs) * capacity
padding_tokens = container_tokens - useful_tokens
efficiency = useful_tokens / container_tokens if container_tokens else 1.0
```

用测试数据 `packing_accounting([[6, 3], [7]], 10)` 手算：$U=6+3+7=16$，$J=2$，$A=2\times10=20$，$Z=20-16=4$，$\eta=16/20=0.8$，和断言完全一致。空输入 `packing_accounting([], 10)` 时 $A=0$，不能除零，约定返回 `efficiency == 1.0`；但有效词元数仍是 0，所以吞吐要另按“完成词元数 ÷ 时间”算，不能拿 1.0 当“真的满载”。输入校验里 `any(not pack or ... or sum(pack) > capacity for pack in packs)` 同时拒绝空包、非正整数元素和超容量包，防止负数或空包制造出大于 1 的虚假利用率。

**第 4 步：TODO 3——三阶段时间线。** 记第 $i$ 批的准备、复制、计算耗时为 $p_i,h_i,c_i$，结束时刻为 $P_i,H_i,C_i$，初始全为 0。notebook 给的递推是：

$$S_i^p=P_{i-1},\quad P_i=S_i^p+p_i,\qquad
S_i^h=\max(P_i,H_{i-1}),\quad H_i=S_i^h+h_i,$$

$$S_i^c=\max(H_i,C_{i-1}),\quad C_i=S_i^c+c_i,\qquad
W_i=\max(0,S_i^c-C_{i-1}).$$

代码里先算 `copy_overlap = pinned and non_blocking and separate_copy_stream`，三个条件缺一不可。参考实现：

```python
prepare_start = prepare_end if prefetch else max(prepare_end, compute_end)
copy_start = (max(prepare_start + prepare_ms, copy_end) if copy_overlap
              else max(prepare_start + prepare_ms, copy_end, compute_end))
compute_start = max(copy_start + h2d_ms, compute_end)
```

逐项解释：`prepare_start` 只用两个候选——有预取时，上一批准备一结束就能准备下一批（`prepare_end`），可以与前一批的复制/计算重叠；没有预取时，必须等上一批计算结束（`compute_end`），准备才会开始。`copy_start` 先受“本批准备完成”约束（`prepare_start + prepare_ms`），再受“复制资源上一批用完”（`copy_end`）约束；只有 `copy_overlap` 为真时，复制才可以不等上一批计算；否则还要 `max(..., compute_end)`。`compute_start` 必须等本批复制完（`copy_start + h2d_ms`），也要等上一批计算完（`compute_end`）。`W_i` 是“本批计算开始时刻减去上一批计算结束时刻”，也就是计算资源真正空转等输入的时间。

**第 5 步：用测试反推时间线。** `test_serial_and_overlapped_pipeline` 用 `[4,4]` 准备、`[2,2]` 复制、`[10,10]` 计算：
- 串行（`prefetch=False`，但三个复制条件为真）：第 1 批准备 0→4、复制 4→6、计算 6→16；第 2 批准备要等计算结束，16→20、复制 20→22、计算 22→32。所以 `compute_start_ms` 是 `[6.0, 22.0]`，两次 `exposed_wait_ms` 是 `[6.0, 6.0]`——第一批的 6 ms 是流水线填充，第二批的 6 ms 是“准备+复制”串行造成的真实等待。
- 开启预取：第 2 批准备 4→8，复制 8→10，计算仍要等上一批 16 结束，于是 16→26。`compute_start_ms` 是 `[6.0, 16.0]`，第二次 `exposed_wait_ms == 0.0`，说明输入等待被完全藏住了。

**第 6 步：供给不足与“假重叠”。** `test_input_starvation_and_missing_stream` 把准备改成 `[12,12]`：第 1 批准备 0→12、复制 12→17、计算 17→27；第 2 批准备 12→24、复制 24→29，计算只能 29 开始，等待 `2.0`。这说明预取不是万能：准备本身比计算还慢时，等待照样暴露。另一个用例把 `separate_copy_stream=False`：第 2 批准备 4→8，但复制必须等上一批计算 16 结束，复制 16→18，计算 18 开始，等待也是 `2.0`。`test_prefetch_independent_of_copy_options` 进一步逐个关掉 `pinned`、`non_blocking`、`separate_copy_stream`，断言第 2 批 `prepare_start_ms == 4`（预取仍然生效）、`h2d_start_ms == 16`（复制仍被串行化）、`exposed_wait_ms == 2`。这就是“预取独立于复制开关”的精确含义：预取只让 CPU 提前干活，不能让 H2D 与计算并行。

**第 7 步：框架一致性核对。** 这一段可选，但很能说明“教学实现”和“成熟框架”的边界。`TensorDataset(torch.arange(5), torch.tensor([4,1,1,3,2]))` 配 `DataLoader(batch_sampler=batch_indices, num_workers=0)`，把 `build_greedy_packs([4,1,1,3,2], 6)` 得到的 `[[4,1,1],[3,2]]` 变成两批样本编号；核对项包括编号顺序完整、长度计划保持、有效词元总数 11、`drop_last=False` 时批量 `[2,2,1]`、`drop_last=True` 时只剩 `[[0,1],[2,3]]`。它只证明“计划交给真实采样接口后不被重排/丢失”，不证明锁页、预取速度或复制重叠。

**第 8 步：GPU 实验的自变量与指标。** 固定 `batch_size=8`、`seq_len=128`、`feature_dim=512`、`warmup_steps=10`、`repeat_steps=80`、`seed=42`，三条路径：

| 配置 | workers | pin_memory | non_blocking | 作用 |
|---|---|---|---|---|
| `pageable_single` | 0 | False | False | 基线：普通内存、单进程 |
| `pinned_single` | 0 | True | True | 只加锁页+非阻塞，区分“搬运”与“多进程” |
| `prefetched_pinned` | 2 | True | True | 再加 2 个 worker、`prefetch_factor=2` |

因变量：`batch_wait_median_ms` 是 `next(iterator)` 的墙钟时间（取批等待），`h2d_median_ms` 是包在 `batch.to('cuda', non_blocking=...)` 两侧的 CUDA Event 时间，`step_median_ms` 是整步墙钟（含取批、复制、矩阵乘、校验值回传和 `torch.cuda.synchronize()`），`step_p95_ms` 是 80 步的 P95，`samples_per_second = repeat_steps × batch_size ÷ 总步时间`，`peak_allocated_mb` 是 `max_memory_allocated`。`_p95` 的实现是 `ordered[int(0.95 * (len(ordered) - 1))]`，是“最近秩”而不是插值。

**第 9 步：结果怎么读。** 5.4 的登记值（`rtx5070ti_input_v1`）：

| 配置 | 取批等待 ms | H2D ms | 步骤中位 ms | 吞吐 样本/s |
|---|---|---|---|---|
| `pageable_single` | 0.141 | 0.229 | 0.567 | 13,612.1 |
| `pinned_single` | 0.139 | 0.152 | 0.379 | 18,842.3 |
| `prefetched_pinned` | 0.038 | 0.162 | 0.394 | 20,302.6 |

读法要分三层。第一，锁页把 H2D 从 0.229 ms 降到 0.152 ms（约 34%），步骤中位数从 0.567 降到 0.379 ms，说明“搬运”这一段确实变快了；第二，多进程预取把取批等待从 0.141 降到 0.038 ms（约 73.1%），吞吐升到 20,303 样本/s，说明“准备”这一段被提前做了；第三，预取的步骤中位数 0.394 ms 反而略高于锁页单进程的 0.379 ms，但 P95 约 0.488 ms 更好，说明多进程有启动/调度抖动，典型步不一定最快，尾延迟和持续吞吐才是它的优势。notebook 因此把 `pinned_single` 和 `prefetched_pinned` 分别对应“关注典型步骤延迟”和“关注持续吞吐与尾部”两种场景，而不是简单说“预取全面获胜”。

三条路径的累计校验值相对差是 0.0，`correctness.passed` 为真，说明这段往返计算在数值上一致；但它只是一个累计标量和，不是逐元素对照，也不能推出完整训练质量不变。原始决策登记为 `accept_prefetch_candidate`，对应本次吞吐与尾延迟的改善；5.2 代码里满足条件时写入的字符串是 `accept_prefetch_for_throughput_and_tail_latency`，读结果时以保存的 JSON 和 5.4 的登记为准。

**第 10 步：这份实验不能证明什么。** notebook 的 5.1 和 5.4 把边界写得很清楚：数据已经在内存里，不含外部下载、磁盘读取或分词；默认流 H2D、每步同步，没有构造“复制—计算并发”的时间线，所以不能宣称已经隐藏了 H2D；GPU 路径没有装箱、没有反向传播，不能据此证明装箱收益或完整训练加速；只保存了汇总统计，没有逐步样本，复测时应保留逐步时间并轮换配置顺序，才能估计运行波动。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 输入流水线 | 读取→组批→主机缓冲→H2D→GPU 计算这条供给链 |
| 填充（padding） | 把短样本补到固定长度，补出来的位置不携带有效词元 |
| 装箱（packing） | 把多个短样本合进一个容器，减少填充位置 |
| 有效词元 / 容器位置 | 真正来自样本的 token 数 / 容器数 × 容量 |
| 利用率 $\eta$ | 有效词元 ÷ 容器位置，空输入约定为 1.0 |
| 锁页内存（pinned） | 不被操作系统换出的主机内存，可配合异步复制 |
| 非阻塞复制 | `non_blocking=True`，让 H2D 提交后 CPU 可以继续 |
| 独立复制流 | 复制和计算跑在不同 CUDA stream，才可能真正重叠 |
| 预取（prefetch） | 当前批还在算时，提前准备下一批 |
| 暴露等待 $W_i$ | 计算开始时刻减去上一批计算结束时刻，>0 就是 GPU 空等 |
| 背压 | 下游处理不过来，队列满后把等待传回上游 |
| P95 | 95% 的步骤不超过这个时间，用来看尾延迟 |

## 小白常见疑问

**Q1：为什么 `prepare_start = prepare_end if prefetch else max(prepare_end, compute_end)`，不是永远 `max(prepare_end, compute_end)`？**
`prefetch=False` 的语义是“上一批算完才准备下一批”，所以要等 `compute_end`。开启预取后，CPU 准备和 GPU 计算是不同资源，可以并行，只要上一批准备结束（`prepare_end`）就能开始下一批。测试里第 2 批 `prepare_start_ms` 在预取时为 4、串行时为 16，就是这个区别。

**Q2：`copy_overlap` 为什么要三个条件同时为真？只开 `non_blocking=True` 不行吗？**
`pinned=True` 保证主机缓冲可被设备异步访问；`non_blocking=True` 只是允许调用立即返回；`separate_copy_stream=True` 才让复制和计算不在同一条流上排队。缺任何一个，代码都把 `copy_start` 退化为 `max(..., compute_end)`，测试 `test_prefetch_independent_of_copy_options` 逐个关掉三个开关都要求第 2 批 `h2d_start_ms == 16`、`exposed_wait_ms == 2`，就是在防止“只看一个 flag 就宣称重叠”。

**Q3：装箱利用率 0.95 很高，为什么时间不一定按同样比例下降？**
利用率只说明“容器里的无效位置少”，不说明计算量同比下降。注意力、内核实现、变长形状的处理都可能让实际耗时和有效词元数不成正比；notebook 的 Step 2 表专门写了“利用率提高不等于时间按相同比例下降”。而且本节的 GPU 实验根本没有装箱，装箱结论只在 CPU 账本和框架核对里成立。

**Q4：`test_input_starvation_and_missing_stream` 里准备 12 ms、计算 10 ms，为什么第 2 批只等 2 ms，而不是 12 ms？**
因为预取让第 2 批的准备与第 1 批的计算并行，等待只来自“供给周期比计算周期长出来的差”。按测试的 `[12,12]` 准备、`[5,5]` 复制、`[10,10]` 计算算：第 1 批准备 0→12、复制 12→17、计算 17→27；第 2 批准备在 12 就开始，12→24，复制 24→29，而上一批计算 27 就结束了，所以计算 29 开始，`29 - 27 = 2`。稳态下准备资源每批要 12 ms、计算每批只要 10 ms，准备成了瓶颈，所以每批稳定暴露 2 ms；5 ms 的复制能塞进 12 ms 的准备周期里，不是等待来源。这个例子说明“预取能藏住一部分，但准备长期慢于计算时仍会暴露等待”。

**Q5：既然预取吞吐最高，为什么锁页单进程的步骤中位数最低，最终该选哪个？**
看目标。如果关心“单步典型延迟”，锁页单进程的 0.379 ms 更好；如果关心“持续吞吐和尾延迟”，预取的 20,303 样本/s 和更低 P95 更好。notebook 没有替你选，而是把两条都保留，并要求在目标训练任务里复测。单次 80 步、固定顺序的汇总也不足以排除运行波动。

## 动手验证：怎么知道自己做对了

测试区有 8 个函数，全部直接调用，TODO 未填会抛 `NotImplementedError`，全过才打印 `✅ 装箱守恒、独立预取与阶段依赖测试通过`。

- `test_greedy_packs_and_invalid_lengths`：`build_greedy_packs([6,3,7,2,1], 10) == [[6,3],[7,2,1]]`；摊平后等于原 `lengths`；`[0,3]` 与 `[11]` 必须抛 `ValueError`。
- `test_accounting_and_padding`：`packing_accounting([[6,3],[7]], 10)` 严格等于 `{'useful_tokens': 16, 'container_tokens': 20, 'padding_tokens': 4, 'efficiency': 0.8}`；`useful + padding == container`；空输入 `efficiency == 1.0`。
- `test_serial_and_overlapped_pipeline`：串行 `compute_start_ms == [6.0, 22.0]`、`exposed_wait_ms == [6.0, 6.0]`；预取 `compute_start_ms == [6.0, 16.0]`、第二批等待 0.0。
- `test_input_starvation_and_missing_stream`：准备 `[12,12]` 时第二批等待 2.0；`separate_copy_stream=False` 时第二批等待 2.0；三个阶段长度不一致必须抛 `ValueError`。
- `test_prefetch_independent_of_copy_options`：分别关闭 `pinned`、`non_blocking`、`separate_copy_stream`，第二批 `prepare_start_ms == 4`、`h2d_start_ms == 16`、`exposed_wait_ms == 2`。
- `test_strict_lengths_and_packs`：容量 `0`、`-1`、`True`、`1.5` 都抛 `ValueError`；`[[True]]`、`[[-1]]`、`[[]]`、`[[11]]` 在容量 10 下都抛 `ValueError`；`build_greedy_packs([], 10) == []`；`[10,10]` 得到 `[[10],[10]]`。
- `test_nonfinite_stage_times`：`nan`、`inf`、`-1`、`True` 出现在任一阶段都必须抛 `ValueError`。
- `test_zero_times_and_dependencies`：三阶段全 0 时时间线所有字段为 0，最后 `compute_end_ms == 0`。

框架核对 `check_input_data_interface` 额外断言：样本编号完整为 `[0,1,2,3,4]`、批次长度为 `[[4,1,1],[3,2]]`、有效词元总数 11、保留尾批为 `[2,2,1]`、丢弃尾批为 `[[0,1],[2,3]]`。GPU 部分没有断言，只有 5.4 的保存记录：三条配置校验值相对差 0.0、`correctness.passed: True`、`decision` 为 `accept_prefetch_candidate`。notebook 保存的输出里，取批等待 0.141 / 0.139 / 0.038 ms、步骤中位 0.567 / 0.379 / 0.394 ms、吞吐 13,612.1 / 18,842.3 / 20,302.6 样本/s；你自己复跑时，这些数值会随机器和负载变化，先看趋势与守恒关系，不要死记绝对值。
