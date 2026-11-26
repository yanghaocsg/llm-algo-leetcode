# 43. 统一训练内存账本 — 小白讲解

> 对应 notebook：[`43_Unified_Training_Memory_Ledger.ipynb`](./43_Unified_Training_Memory_Ledger.ipynb)

## 一句话总结
训练峰值不是"参数 + 梯度 + 优化器状态 + 激活"各自最大值的相加，而是某一训练阶段里仍驻留在 GPU 上的那些状态之和。本节用 `PHASES` 把一次训练切成 `load → forward → loss → backward → update → zero_grad` 六个阶段，用四个 TODO 实现"校验状态生命周期 → 汇总每阶段驻留量 → 找峰值与贡献对象 → 比较基线与候选账本"。样例账本算出的峰值是 backward 阶段的 2200 MB；Checkpoint 和 Offload 两个候选都省 300 MB，但峰值分别落在 forward 和 backward，说明"省了多少"和"在哪里省、付出什么代价"是两个问题。

## 为什么需要它：痛点是什么
初学者常见的两个错误，一是把各类状态的峰值直接相加：参数 800、优化器 600、激活 300、梯度 400、workspace 200、通信缓冲 100，加起来 2400 MB，就宣称"峰值 2400"。可这 2400 MB 从来没有在同一时刻出现过——workspace 在 forward 结束就释放了，梯度在 backward 才产生，激活在 update 前已经消失。二是认为"offload 一定降峰值"：把激活搬到 CPU，确实不再计入 GPU 账本，但代价是反向前要搬回来，D2H/H2D 的时间和 host 容量都要另算。

生活类比：公司统计"同时在岗人数"。不能把每个部门历史最高人数相加，那会得到一个从没发生过的数字；正确做法是拿一张排班表，看每个时间点谁还在岗。账本里的"阶段"就是时间点，"驻留区间"就是某人的排班起止，"device"就是他在不在本公司上班（CPU 相当于外派，不计入本公司工位）。

## 核心问题
- 一条训练状态记录至少要有哪些字段，才能放到统一时间轴上？
- 阶段总量 $M_p$ 和理论峰值 $M_{peak}$ 怎么算？为什么 CPU 上的状态不计入 GPU 峰值？
- 生命周期区间为什么按闭区间处理，起止阶段都算驻留？
- 并列峰值时为什么取最早阶段，而不是任意一个？
- Checkpoint 和 Offload 都让峰值降 300 MB，为什么峰值阶段不同？这说明账本还缺什么？
- 这个理论账本不包含什么，为什么不能直接当成 `nvidia-smi` 的数字？

## 解决思路：一步步看懂

**第 1 步：先把七类状态排到时间轴上。** 一条记录是字典，必备字段在 `validate_state_lifecycle` 里写死：`name`、`category`、`size_mb`、`start_phase`、`end_phase`、`device`、`persistent`。`category` 必须属于 `VALID_CATEGORIES` 的七类之一：`parameter`、`master_weight`、`activation`、`gradient`、`optimizer_state`、`workspace`、`communication_buffer`；阶段必须属于 `PHASES = ['load', 'forward', 'loss', 'backward', 'update', 'zero_grad']`。

notebook 的 Step 1 表格解释了每类状态的生命周期，几个容易混淆的点：
- **普通 activation 和 saved tensors 不是一回事。** 普通激活是前向产生的中间结果，最后一次被消费就可以释放；saved tensors 是 autograd 为了反向的局部 VJP 明确保留的那部分，才是真正跨越到反向的激活子集。
- **optimizer state 是延迟创建的。** 第一次 `update` 之后才出现，之后长期驻留。样例账本为了教学简单，把它写成从 `load` 开始，这一点和 Step 1 的文字描述不同，读样例数字时要以 `_sample_states()` 的字段为准。
- **gradient 跨 micro-batch 保留**，直到 `zero_grad` 才释放；**workspace** 只在当前算子内短期存在；**通信缓冲区**只在 collective 的窗口内驻留。

展开说两类最容易搞混的状态。普通 activation 的终点是"最后一次消费者完成"，所以它可能在前向内部就被释放，不必等到 backward；saved tensors 是 autograd 为了算局部 VJP 明确保留的那一小部分，它的终点才是对应的 backward 节点。notebook 提到第 10 节的 $Q/K/V/P$ 就是 Attention 的实例：这四个张量里哪些会被 autograd 保留到反向，取决于反向公式是否需要它们；不需要的就不进 saved tensors，不能一概而论。这里不用展开第 10 节，只要记住：账本里的 `activation` 是粗粒度类别，`start_phase`/`end_phase` 是这一类状态的简化区间，不是每个张量的精确生死时刻。

optimizer state 的"延迟创建"也值得单独说。Step 1 说它第一次 update 后才创建，之后长期驻留；样例 `_sample_states()` 却把它写成 `load → zero_grad`，从 `load` 就开始占 600 MB。这是教学样例的简化，不是真实训练的时间线。读样例数字时以字段为准；如果要做真实训练账本，应该把 `start_phase` 设成 `update`，峰值阶段和总量都可能改变。

**第 2 步：用指示量写阶段总量。** 设状态 $j$ 的容量为 $m_j$（MB），它在阶段 $p$ 是否驻留 GPU 用 $I_{j,p}\in\{0,1\}$ 表示，则

$$
M_p=\sum_j m_j I_{j,p},\qquad M_{peak}=\max_p M_p.
$$

$M_p$ 回答"此刻有哪些状态重叠"，$M_{peak}$ 回答"理论峰值出现在哪个阶段"。$I_{j,p}$ 由三个条件共同决定：`device == 'gpu'`、`PHASES.index(start_phase) <= p`、`p <= PHASES.index(end_phase)`。注意第三个条件用的是 `<=`，所以起止阶段都算驻留（闭区间）。这不是随便定的：状态在 `start_phase` 开始就占着空间，到 `end_phase` 结束时才释放，端点自然都包含。

**第 3 步：TODO 1 校验区间，先保证"排班表"合法。** 骨架已经做了字段缺失（`KeyError`）、`size_mb` 类型与负值、类别、设备、阶段名的检查，TODO 1 只差最后一步：

```python
if PHASES.index(state['start_phase']) > PHASES.index(state['end_phase']):
    raise ValueError('start_phase 不能晚于 end_phase')
```

为什么用 `PHASES.index` 而不是字符串比较？因为阶段名不是字典序：`'backward' < 'forward' < 'load' < 'loss' < 'update' < 'zero_grad'`，字符串排序和真实训练顺序完全对不上。必须用同一份有序阶段表取得下标再比大小。`start_phase == end_phase` 是合法的，表示状态只在一个阶段内存在，比如样例里的 `forward_workspace` 是 `forward → forward`。

校验顺序也有讲究：先查必需字段，再查 `size_mb` 的类型（并单独排除 `bool`，因为 `True` 也是 `int`），再查负值、类别、设备、阶段名，最后才查区间顺序和 `persistent` 是否为布尔。这样每条非法输入都能得到最具体的错误类型，而不是笼统地报"记录非法"。举几个具体的错误路径：`size_mb = -1.0` 命中 `ValueError('size_mb 不能为负数')`；`device = 'tpu'` 命中 `ValueError('device 必须是 cpu 或 gpu')`；倒置区间命中 `ValueError('start_phase 不能晚于 end_phase')`；缺少字段则命中 `KeyError`，消息里会列出缺失的字段名。测试 `test_lifecycle_contract` 用四组篡改验证：`size_mb = -1.0`、`category = 'unknown'`、`device = 'tpu'`、`start_phase = 'backward'` 且 `end_phase = 'forward'`，每一组都必须抛 `TypeError` 或 `ValueError`。注意它只捕获这两类，`KeyError` 属于"结构不完整"，不在这四组的覆盖范围里。

为什么 `size_mb` 要单独排除 `bool`？因为 Python 里 `True == 1`、`False == 0`，如果不排除，一条 `size_mb=True` 的记录会被当成 1 MB 悄悄通过，后面所有汇总都带着这个错误。同理，`persistent` 必须严格是 `bool`，传 `0`/`1` 也会被拒绝。

**第 4 步：TODO 2 汇总每个阶段的 GPU 驻留量。** 参考答案是一个列表推导：

```python
active = [
    state for state in states
    if state['device'] == 'gpu'
    and PHASES.index(state['start_phase']) <= phase_index <= PHASES.index(state['end_phase'])
]
```

然后同时记录四样东西：`phase`、`total_mb`（容量和）、`active_states`（名字列表）、`state_sizes_mb`（名字到容量的映射）。后两样是为峰值归因准备的——只知道"backward 是 2200 MB"不够，还要知道"是谁撑起来的"。为什么两个都要记？`active_states` 给人看，`state_sizes_mb` 给程序算贡献排序；如果只记名字，排序时还要回原表查容量，而原表的过滤条件和汇总时的可能不一致，容易出现"名单里有、数字里没有"的错位。两个字段都从同一个 `active` 列表生成，保证名单和数字同源。

拿 `_sample_states()` 手算一遍（单位 MB）：

| 阶段 | 参数 800 | 优化器 600 | 激活 300 | 梯度 400 | workspace 200 | 通信 100 | CPU 激活 250 | 合计 |
|---|---|---|---|---|---|---|---|---|
| load | ✓ | ✓ | | | | | | 1400 |
| forward | ✓ | ✓ | ✓ | | ✓ | | | 1900 |
| loss | ✓ | ✓ | ✓ | | | | | 1700 |
| backward | ✓ | ✓ | ✓ | ✓ | | ✓ | | 2200 |
| update | ✓ | ✓ | | ✓ | | | | 1800 |
| zero_grad | ✓ | ✓ | | | | | | 1400 |

`offloaded_activation` 的 `size_mb` 是 250，但 `device = 'cpu'`，所以六个阶段一次都没被计入。测试 `test_phase_residency` 断言 `forward = 1900.0`、`backward = 2200.0`、`update = 1800.0`，并检查 `offloaded_activation` 不在 backward 的 `active_states` 里、`activations` 不在 update 的 `active_states` 里。后两条比总数更能说明"释放时机"被正确建模：如果只写 `total_mb` 而忘记按区间过滤，总数可能碰巧对，但对象列表会露馅。

阶段总量不是单调递增的：load 1400 → forward 1900 → loss 1700 → backward 2200 → update 1800 → zero_grad 1400。forward 到 loss 降了 200，因为 workspace 只在 forward 存在；backward 到 update 降了 400，因为 activations 和 collective_buffer 都在 backward 结束后释放（300 + 100 = 400）。峰值必须扫完六个阶段再取最大，不能只看 backward 或 update 中的某一个，也不能假设"后面的阶段一定更大"。

如果把 TODO 2 的闭区间写成开区间 `start < p < end`，会发生什么？`forward_workspace` 是 `forward → forward`，开区间下它一个阶段都不算；`collective_buffer` 是 `backward → backward`，同样消失；`activations` 是 `forward → backward`，两端都不算，只在 loss 出现。峰值会从 backward 的 2200 变成 loss 的 1700。闭区间不是风格问题，而是"状态在起止阶段确实存在"这个语义的直接表达：起点的第一刻已经占着空间，终点的最后一刻还没释放。

如果把 `device == 'gpu'` 这个条件漏掉，offload 候选就完全失效：候选里的 `activations` 会被算进 GPU 阶段，峰值不再下降；基线也会把 `offloaded_activation` 的 250 MB 算进去，整体数字一起偏大，`memory_saving_mb` 变成 0。测试 `test_candidate_peak_comparison` 期望 offload 省 300，漏掉设备过滤就会失败。这也是为什么 `summarize_phase_memory` 的文档字符串写的是"汇总 GPU 上仍驻留的状态"，而不是"汇总所有状态"。

**第 5 步：TODO 3 找峰值，并列时取最早。** 参考答案一行：

```python
peak = max(phase_records, key=lambda item: float(item['total_mb']))
```

`phase_records` 是按 `PHASES` 顺序生成的，而 Python 的 `max` 在多个元素并列最大时返回**第一个**，所以自然得到"最早出现的峰值阶段"。如果先按容量排序再取最后一个，或者用 `>=` 手写循环，并列时就会取到更晚的阶段，`peak_phase` 就和测试对不上。测试数据里 backward 2200 是唯一最大值，但 `test_candidate_peak_comparison` 里的 checkpoint 候选会出现 forward 和 backward 都是 1900 的并列，正好检验这条规则。

贡献对象用 `sorted(sizes, key=lambda name: (-float(sizes[name]), name))`：先按容量降序，容量相同再按名字升序。第二个键是为了让结果稳定可复现，不受字典插入顺序影响。样例的 `contributors` 是 `['parameters', 'optimizer_state', 'gradients', 'activations', 'collective_buffer']`，对应 800、600、400、300、100，`forward_workspace` 不在其中，因为它在 backward 之前已经释放。

**第 6 步：TODO 4 用同一口径比较候选。** 关键是"重算"，而不是在基线的结果上做加减：

```python
for name, states in candidate_states.items():
    if not name:
        raise ValueError('候选名称不能为空')
    candidate_peak = find_peak_overlap(summarize_phase_memory(states))
    comparisons[name] = {
        'peak_phase': candidate_peak['peak_phase'],
        'peak_mb': candidate_peak['peak_mb'],
        'memory_saving_mb': baseline_peak['peak_mb'] - candidate_peak['peak_mb'],
    }
```

每个候选都要重新走 `summarize_phase_memory` 和 `find_peak_overlap`，因为候选可能改变的不只是某个容量，还可能改变驻留区间，峰值阶段会跟着搬家。如果只拿基线的 2200 去减某个局部数字，就会漏掉"峰值换阶段"这种情况。

如果候选峰值比基线高，`memory_saving_mb` 会是负数，函数不会报错，表示这个候选在这套账本下更差；Step 3 的 $\Delta M_{peak}>0$ 只是"峰值下降"的判据，不是函数的前置条件。测试里的两个候选恰好演示了另一个重点：
- **checkpoint**：把 `activations` 的 `end_phase` 从 `backward` 改成 `forward`。forward 仍是 1900，loss 变 1700，backward 变成 800+600+400+100 = 1900，与 forward 并列；取最早，峰值阶段是 `forward`，`peak_mb = 1900.0`，节省 300。
- **offload**：把 `activations` 的 `device` 改成 `cpu`。forward 变成 1600，backward 变成 1900，峰值阶段是 `backward`，`peak_mb = 1900.0`，同样节省 300。

两者 `memory_saving_mb` 都是 300，但一个峰值在 forward、一个在 backward。这就是账本的价值：它告诉你"省在哪里"。checkpoint 的代价是反向重算时间，offload 的代价是 D2H/H2D 传输和 host 容量，这些都不在 `memory_saving_mb` 里，需要另测。Step 3 的表格也是这个意思：策略只有改变峰值阶段的驻留对象、容量或区间，才可能降低理论峰值；重算、传输、通信的代价必须单独测量。

这一步还隐含了一个实现要求：每个候选必须重新走完整的 `summarize_phase_memory`，不能拿基线的阶段总量做局部修补。因为候选可能同时改变多个状态，峰值阶段会跟着搬家；如果只对基线峰值做减法，得到的数字可能对应一个候选里根本不存在的阶段。`find_peak_overlap` 返回的 `peak_phase` 就是用来核对这一点的：checkpoint 的 `peak_phase` 是 `forward`，offload 是 `backward`，两者不能混用。

**第 7 步：框架一致性核对只看"边界"，不看真实 GPU。** `check_memory_ledger_framework_conformance` 用 `torch.profiler` 在 CPU 上跑一段 `value = torch.ones((128, 128), requires_grad=True)`、`loss = value.square().sum()`、`loss.backward()`，用 `record_function` 标出 `forward_phase` 和 `backward_phase`。四项检查：
- `phase_ranges_present_and_ordered`：两个阶段都被记录，且 forward 的起始时间早于 backward；
- `cpu_memory_activity_reported`：profiler 记录的正向 CPU 内存活动不少于张量本身的字节数。128×128 个 FP32 元素是 16384 × 4 = 65536 字节，即 0.0625 MiB；
- `cpu_state_excluded_from_gpu_ledger`：把这个 CPU 张量做成一条 `device = 'cpu'` 的状态记录，`summarize_phase_memory` 的每个阶段 `total_mb` 都必须是 0；
- `backward_gradient_present`：`value.grad` 非空且有限。

还要注意 profiler 的父子注解：`forward_phase` 和 `backward_phase` 是外层范围，内部还会有 `square`、`sum`、反向节点等算子事件。核对只筛名字完全等于 `forward_phase` / `backward_phase` 的事件，不做前缀匹配，避免把子事件也算进阶段范围。这个"先选粒度再统计"的原则，和 PERF-TRACE 里选择事件粒度是同一个道理：父级注解常包含内部算子，混着算会把同一份工作数两遍。

`positive_cpu_bytes` 是把 profiler 所有事件的 `cpu_memory_usage` 取正后求和，再和 `tensor_bytes` 比较。它不是精确的峰值，只是"profiler 确实报告了不少于这个张量的 CPU 内存活动"。`cpu_state` 的 `size_mb` 用 `tensor_bytes / (1024 ** 2)`，即 0.0625 MiB；把它交给 `summarize_phase_memory`，六个阶段都必须是 0，因为设备是 cpu。这个核对只证明"阶段事件能对上、CPU 状态确实被 GPU 账本排除"，不证明 CUDA 分配器的真实行为。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 生命周期（lifecycle） | 一个状态从哪个阶段开始占空间、到哪个阶段释放 |
| 驻留（residency） | 某个阶段里状态是否还在 GPU 上 |
| 阶段总量 $M_p$ | 该阶段所有仍在 GPU 上的状态容量之和 |
| 理论峰值 $M_{peak}$ | 六个阶段总量里的最大值 |
| 指示量 $I_{j,p}$ | 状态 $j$ 在阶段 $p$ 是否驻留，取 0 或 1 |
| 闭区间 | 起止阶段都算驻留，`start <= p <= end` |
| 贡献对象（contributors） | 峰值阶段里按容量排序的状态名单 |
| saved tensors | autograd 为反向明确保留的激活子集 |
| Checkpoint | 不长期保存部分激活，反向时重算 |
| Offload | 把状态搬到 CPU，用时再取回 |
| `state_sizes_mb` | 阶段内"名字 → 容量"的映射，用来归因 |
| 分配器预留 | allocator 额外占着但账本没算的显存 |

## 小白常见疑问

**Q1：为什么不能直接把各类状态的最大值相加得到峰值？**
因为不同状态的最大值不在同一时刻出现。样例里 workspace 只在 forward 存在（200 MB），梯度只在 backward 到 update 存在（400 MB），两者从不同时驻留。相加会得到 2400 MB，而真实峰值是 backward 的 2200 MB。`M_p` 的定义就是对"同一阶段"求和，`M_peak` 再对这些和取最大，顺序不能反。这也说明账本必须先定义阶段、再定义状态：没有阶段，"此刻"就无从谈起，峰值也只能退化成一张对不上的汇总表。

**Q2：CPU 上的 offload 状态为什么完全不计入？它不是还占着内存吗？**
它占的是 host 内存，不是 GPU 显存。本节账本回答的是"GPU 峰值"，所以 `summarize_phase_memory` 先过滤 `device == 'gpu'`。offload 的 host 容量和传输代价要在策略比较里另算，Step 3 表格的"需要另行测量的代价"一列写的就是这个。

**Q3：并列峰值取最早阶段，会不会掩盖更晚的那个峰值？**
会丢掉"并列"这个信息，但不会丢掉峰值容量。规则的目的不是完整性，而是确定性：同样输入必须给同样答案，否则测试和复现都无从谈起。样例的 checkpoint 候选就是并列（forward 1900、backward 1900），`peak_phase` 固定为 `forward`。如果你关心"还有哪些阶段也达到了峰值"，需要自己再扫一遍 `summarize_phase_memory` 的结果。

**Q4：`memory_saving_mb` 是负数可以吗？**
可以。代码只做减法 `baseline_peak['peak_mb'] - candidate_peak['peak_mb']`，没有夹到 0。负数表示候选账本的理论峰值反而更高，是一个有效结论。Step 3 的公式 $\Delta M_{peak}>0$ 只是说"正数才说明峰值下降"，并没有说负数非法。不过返回字典里没有单独的"是否变差"标志，读的人要自己看符号；下游如果要做自动选择，应该在 `memory_saving_mb > 0` 之上再加质量和时间门槛。

**Q5：`summarize_phase_memory` 为什么对每条状态都重新调用 `validate_state_lifecycle`？**
因为它是一个对外函数，调用方可能直接传入未校验的列表（测试就是这么做的）。在入口统一校验，能让函数对任何输入都给出明确错误，而不是在求和时因为缺字段或 `size_mb` 是字符串而抛出难以定位的异常。代价是 O(状态数) 的重复检查，相对于六个阶段的汇总可以忽略。`find_peak_overlap` 也做了入口检查：`phase_records` 为空直接抛 `ValueError`，避免 `max()` 在空序列上报一个不直观的错误。

**Q6：这个账本能替代 `torch.cuda.max_memory_allocated()` 吗？**
不能。理论账本只统计你显式登记的状态，不含 allocator 的额外预留、未知 workspace、其他进程、碎片和缓存。Step 2 明确写了"理论账本不包含 allocator 额外预留、未知 workspace 或其他进程，这些差异仍需 GPU memory snapshot 或 profiler 解释"。它的用途是解释"峰值由谁形成"，不是预测真实显存数字。

**Q7：`persistent` 字段为什么没有参与求和？**
因为驻留区间已经表达了"什么时候在"，`persistent` 只是给读者和下游工具的一个标签，说明这条状态是否跨训练 step 长期存在（样例里 `parameters` 和 `optimizer_state` 是 `True`，激活、梯度、workspace 是 `False`）。`summarize_phase_memory` 只按阶段和设备过滤，不按 `persistent` 过滤；如果你想单独看"跨 step 常驻状态占多少"，可以自己用这个字段筛，但那是另一个口径，不是峰值。`validate_state_lifecycle` 仍然要求它是布尔值，缺失或类型不对会报 `TypeError`。

**Q8：候选名称重复会怎样？**
`candidate_states` 是字典，重复键在传进函数之前就已经被 Python 合并了，函数内部看不到重复；`compare_peak_candidates` 只拒绝空字符串名称。所以"名称唯一"是调用方的责任，不是这个函数能检查的。测试用 `{'checkpoint': ..., 'offload': ...}` 两个唯一键，正好避开这个问题。

## 动手验证：怎么知道自己做对了
`run_unified_memory_tests()` 依次跑四个测试，全过才打印 `✅ 生命周期、阶段驻留、峰值重叠与候选对照测试通过`；TODO 没填会抛 `NotImplementedError`。

- `test_lifecycle_contract`：合法记录 `validate_state_lifecycle(valid)` 不抛异常；四组非法记录（负容量、未知类别、`tpu` 设备、倒置区间）必须抛 `TypeError` 或 `ValueError`，否则 `AssertionError('非法生命周期记录应明确拒绝')`。
- `test_phase_residency`：`summarize_phase_memory(_sample_states())` 的结果里，`forward['total_mb'] == 1900.0`、`backward['total_mb'] == 2200.0`、`update['total_mb'] == 1800.0`；`offloaded_activation` 不在 backward 的 `active_states`，`activations` 不在 update 的 `active_states`。
- `test_peak_overlap`：`peak_phase == 'backward'`、`peak_mb == 2200.0`，`contributors == ['parameters', 'optimizer_state', 'gradients', 'activations', 'collective_buffer']`。这个列表的顺序验证了"按容量降序"；"同容量按名字升序"这条规则在样例里没有被触发，因为五个容量互不相同。
- `test_candidate_peak_comparison`：基线 `peak_mb == 2200.0`；checkpoint 候选严格等于 `{'peak_phase': 'forward', 'peak_mb': 1900.0, 'memory_saving_mb': 300.0}`；offload 候选严格等于 `{'peak_phase': 'backward', 'peak_mb': 1900.0, 'memory_saving_mb': 300.0}`。

框架核对 `check_memory_ledger_framework_conformance` 返回 `status = 'passed'`、`check_kind = 'framework_conformance'`，并记录 `torch` 版本、`tensor_bytes = 65536`、`profiled_positive_cpu_bytes`。四项 `checks` 全为 `True` 才算通过，否则 `AssertionError` 会列出失败项。notebook 没有保存这段输出，实际数值以你本机运行打印的字典为准。

最后提醒两个读测试的小细节。`test_peak_overlap` 的 `contributors` 顺序同时验证了"按容量降序"：样例的五个容量互不相同，排序键的第二项（名字）这次没有真正被触发；要验证同容量时的名字排序，可以自己把 `collective_buffer` 的 100 改成 300，此时 activations 和 collective_buffer 都是 300，排序结果应先 `activations` 后 `collective_buffer`。`test_candidate_peak_comparison` 用字典严格相等断言三个字段，多一个或少一个键都会失败，这比逐个字段比较更能锁住返回结构。和 35 一样，这些测试是直接调用的，任何一个失败都会立即中断后面的测试。
