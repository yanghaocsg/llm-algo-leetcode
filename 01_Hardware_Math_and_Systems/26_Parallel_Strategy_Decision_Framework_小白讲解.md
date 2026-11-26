# 26. 并行策略决策框架 — 小白讲解

> 对应 notebook：[`26_Parallel_Strategy_Decision_Framework.ipynb`](./26_Parallel_Strategy_Decision_Framework.ipynb)

## 一句话总结
本节把「选并行策略」拆成四步可核对的账：`build_resource_ledger` 先算单卡缺口（13 GiB 状态 + 4 GiB 激活 + 2 GiB 预留 = 19 GiB，24 GiB 卡剩 5 GiB）；`communication_ledger` 固定总载荷比较发送频率（256 MiB 一次与 32 MiB 八次传输都是 2.5 ms，加 5 μs 固定开销后分别是 2.505 和 2.54 ms）；`evaluate_candidate` 按容量、任务模式、模型结构、放置域筛掉不合适的候选；`validate_parallel_composition` 只在进程网格轴彼此独立时把度数相乘（DP=2、TP=4、PP=2 需要 16 个 rank），否则要求先给出进程组映射。

## 为什么需要它：痛点是什么
初学者背得出 DP、TP、PP 的中文名，却做不了决策：24 GiB 的卡到底能不能用 DP？为什么 TP 通常只放在机内？为什么总通信量相同，拆成多次发送会更慢？为什么不能把 DP×TP×PP×EP 直接乘起来算 rank 数？还有一个基础但致命的坑：把 GiB 和 GB 混着算，13 GiB 直接当 13 GB 用。

生活类比：搬家。数据并行是几组人各搬一份行李，每个人都要一份完整清单，搬完还要对账（同步梯度）；张量并行是把一件大家具拆成几块，几个人抬，抬的时候要不停对接口（交换激活）；流水线并行是接力，一人负责一段，段与段之间要把东西递过去并等对方；专家并行是按物品类型分工，前提是「真有分类的专家」；副本并行是开多个窗口接待不同客户，各管各的队列。用哪种，取决于你要分摊的是「东西太多」「一件太大」还是「客人太多」。

## 核心问题
- 单卡放不下时，先看哪些数字？状态、激活、运行预留各占多少？
- 同样的总通信量，为什么发送次数会影响时间？
- 哪些条件不满足时应该先排除候选，而不是继续比较性能？
- DP/TP/PP 的度数什么时候可以直接相乘，什么时候必须先给组映射？
- EP 和 Replica 各有什么结构前提，为什么不能和训练 DP 混着算？

## 解决思路：一步步看懂

**第 1 步：先算单卡缺口（Q1）。** `build_resource_ledger(state_gib, activation_gib, runtime_reserve_gib, device_memory_gib)` 把三块需求相加，再和显存比：

```python
required_gib = state_gib + activation_gib + runtime_reserve_gib
headroom_gib = device_memory_gib - required_gib
'fits_one_device': headroom_gib >= 0,
```

校验规则是：所有输入非负，设备显存必须大于 0 GiB。`fits = build_resource_ledger(13.0, 4.0, 2.0, 24.0)` 得到 `required_gib == 19`、`headroom_gib == 5`；`does_not_fit = build_resource_ledger(22.0, 5.0, 2.0, 24.0)` 得到 29 GiB、余量 −5 GiB，`fits_one_device` 为 `False`。

单位必须说清楚：本节全程用 GiB（$2^{30}$ B），而 06 节的显存账本用十进制 GB（$10^{9}$ B），换算要乘 $10^9/2^{30}$。13 GiB 约等于 13.96 GB，直接当 13 GB 会在边界上算错。`round(..., 2)` 只影响展示，不改变判断。

这一步只回答「一张卡够不够」。不够，才有后面的切分问题；够了，也不代表单卡吞吐达标——容量和性能是两条线。

**第 2 步：先选切分对象，再谈策略。** notebook 的表格把六种策略的「切什么」和「引入什么工作」列在一起：

| 策略 | 切分对象 | 引入的工作 |
|---|---|---|
| DP | 不同进程处理不同批次 | 同步梯度；和 ZeRO 组合时还会分片训练状态 |
| TP | 层内张量与计算 | 交换局部激活或归约局部结果 |
| PP | 不同网络层 | 阶段间传递激活，等待相邻阶段 |
| CP | 序列位置（上下文） | 交换注意力所需的状态 |
| EP | 不同专家 | 分发 token、合并输出、处理负载不均 |
| Replica | 独立请求（服务副本） | 管理队列、缓存局部性与尾延迟 |

这张表的用法是「先问要分摊什么」：状态太大 → DP/ZeRO/TP/PP；单层算不动 → TP；序列太长 → CP；模型是 MoE → 才轮到 EP；在线服务 → Replica。不要把「策略名」当答案，策略只是手段。

notebook 还提醒：机内（intra_node）与跨节点（cross_node）表示位置，不是固定的速度等级；互连快慢取决于实际带宽、时延、消息大小和频率。CP 既能分摊长序列状态，也能改善计算分工，值不值得要比较通信和执行时间。

**第 3 步：固定总载荷，比较发送频率（Q2）。** `communication_ledger(payload_mib, calls_per_step, bandwidth_gib_s, latency_us)` 的模型是「每次调用的传输时间 + 固定时延，再乘调用次数」：

```python
transfer_ms_per_call = payload_mib / 1024 / bandwidth_gib_s * 1000
latency_ms_per_call = latency_us / 1000
total_ms = calls_per_step * (transfer_ms_per_call + latency_ms_per_call)
```

`payload_mib / 1024` 先把 MiB 换成 GiB，除以带宽得到秒，再乘 1000 得到毫秒。`latency_us / 1000` 把微秒换成毫秒。两组对比如下：

- 低频大载荷：256 MiB 一次。$256/1024 = 0.25$ GiB，$0.25/100 \times 1000 = 2.5$ ms 传输；固定开销 $5/1000 = 0.005$ ms；总计 2.505 ms。
- 高频小载荷：32 MiB 八次。每次 $32/1024/100\times1000 = 0.3125$ ms，八次 2.5 ms；固定开销 $0.005\times8 = 0.04$ ms；总计 2.54 ms。

```python
assert low_frequency['total_payload_mib'] == high_frequency['total_payload_mib'] == 256
assert low_frequency['transfer_ms'] == high_frequency['transfer_ms'] == 2.5
assert low_frequency['lower_bound_ms'] == 2.505
assert high_frequency['lower_bound_ms'] == 2.54
```

两者总字节一样、传输项一样，差 0.035 ms，正好是多出来的七次固定开销。这个差值随步数累积；TP、EP 这类每层都要交换的策略，调用次数是主要成本之一。

三个限定词不能丢：`payload_mib` 是**每 rank 实际发送量**，已经包含所选通信算法的放大系数（Ring 等算法的换算在 20 节）；时间是**串行相加**，没有算重叠；返回值叫 `lower_bound_ms`，是下界，不是实测。TP、EP 是否合算还要在 27 节结合就绪时间和可重叠计算判断。

**第 4 步：先排除，再比较（Q3）。** `evaluate_candidate(candidate, environment)` 检查四个条件，每个条件对应一条固定的拒绝原因：

```python
if required > capacity:
    reasons.append('per-rank memory exceeds device memory')
if environment['mode'] not in candidate['modes']:
    reasons.append('workload mode is unsupported')
if environment['model_kind'] not in candidate['model_kinds']:
    reasons.append('model structure is unsupported')
if environment['link_scope'] not in allowed:
    reasons.append('placement domain is not allowed')
```

候选声明 `memory_per_rank_gib`、`modes`、`model_kinds`、`allowed_link_scopes`；环境给出 `memory_per_device_gib`、`mode`、`model_kind`、`link_scope`。放置域只允许 `{"none", "intra_node", "cross_node"}`，非法值抛 `ValueError("需给定有效放置域及允许集合")`。

在 24 GiB、训练、稠密模型、机内的环境下：

| 候选 | 每 rank 需求 | 结果 | 原因 |
|---|---|---|---|
| DP | 29 GiB | 排除 | 超出设备显存 |
| ZeRO-3 | 18 GiB | 通过 | — |
| TP | 16 GiB | 通过 | 只允许机内，环境正好是机内 |
| EP | 14 GiB | 排除 | 只支持 MoE，模型是 dense |

断言 `reports[0]['reasons'] == ['per-rank memory exceeds device memory']`、`reports[2]['conditions_met']`、`reports[3]['reasons'] == ['model structure is unsupported']`。读法很重要：筛选通过只表示列出的条件满足，不表示它更快；跨节点候选是否更快，取决于第 3 步的通信成本和实测。容量数字也要随批量、序列长度和分片方式更新，不能一次算完就锁死。

**第 5 步：并行度什么时候能相乘（Q4）。** `validate_parallel_composition(degrees, available_ranks, model_kind, *, independent_axes)` 先校验维度名只能来自 `{'DP','TP','PP','CP','EP','Replica'}`，rank 数必须是正整数，各维度度数必须 ≥1，`independent_axes` 必须是布尔。然后：

```python
if not independent_axes:
    return {'required_ranks': None, 'valid': False, 'issues': ['需要实际进程组映射']}
required_ranks = math.prod(degrees.values())
```

只有在「DP、TP、PP 是彼此独立的进程网格轴」这个前提下，才能把度数相乘。`{'TP': 4, 'PP': 2, 'DP': 2}` 得到 $4\times2\times2 = 16$ 个 rank，16 个可用 rank 时 `valid` 为真；只有 8 个时 `issues` 恰好一条「requires 16 ranks but only 8 are available」。

另外两条结构检查：`EP` 度数大于 1 时模型必须是 MoE，否则报 `'EP degree greater than 1 requires an MoE model'`（`{'EP': 2}` 配 dense 就是这条）；`Replica` 与训练 DP 同时大于 1 时报 `'Replica and training DP must use distinct execution roles'`，因为服务副本和同步训练 DP 是两种执行角色。

`shared_groups = validate_parallel_composition({'DP': 4, 'EP': 2}, 4, 'moe', independent_axes=False)` 返回 `required_ranks` 为 `None`、`issues` 为「需要实际进程组映射」——EP 可能复用 DP 已经建好的进程组，这时度数不能直接相乘，必须先确定真实组映射。这就是「先声明轴是否独立」这个关键字参数存在的原因。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| DP（数据并行） | 每张卡跑不同批次，梯度要同步 |
| TP（张量并行） | 把一层内的矩阵切开放到多卡，前向反向都要交换局部结果 |
| PP（流水线并行） | 不同层放不同卡，像接力一样传激活 |
| CP（上下文并行） | 按序列位置切分，注意力要交换跨块状态 |
| EP（专家并行） | 不同专家放不同卡，只适用于 MoE |
| Replica（副本并行） | 多个服务副本各接各的请求 |
| ZeRO | 把优化器状态、梯度、参数分片，降低每卡状态 |
| rank | 分布式里的一个进程编号，通常对应一张设备 |
| 进程组 | 一组能互相通信的 rank；不同并行维度可能复用同一组 |
| 放置域 | 允许放在机内、跨节点还是不限制 |
| 每 rank 状态 | 单个进程需要保存的模型/优化器/激活总量 |
| GiB / GB | $2^{30}$ B / $10^{9}$ B，本节用 GiB |

## 小白常见疑问
**Q1：256 MiB 一次和 32 MiB 八次，总字节完全一样，为什么慢 0.035 ms？**
因为每次调用都有固定时延。`communication_ledger` 把 `latency_us` 乘进调用次数：一次是 0.005 ms，八次是 0.04 ms，差 0.035 ms。传输项两边都是 2.5 ms。这个模型假设调用串行相加，真实系统如果能把固定开销藏进计算，差距会变小。

**Q2：DP 只差 5 GiB（29 对 24），为什么直接排除，不给个「挤一挤」的机会？**
`evaluate_candidate` 的第一条就是 `required > capacity`，超过就是 `per-rank memory exceeds device memory`。容量是硬条件，不像通信时间可以靠重叠优化；挤一挤意味着改批量、改序列长度或换分片方式，那已经是一个新候选了。notebook 明确说「容量数字需要随批量、序列长度与分片方式更新」。

**Q3：16 个 rank 是怎么来的？为什么不能永远这么乘？**
$4\times2\times2=16$，前提是 `independent_axes=True`。如果 EP 或 CP 复用了 DP 的进程组，度数相乘会重复计数。`shared_groups` 传 `independent_axes=False` 时直接返回 `required_ranks=None` 和「需要实际进程组映射」，拒绝猜一个数字。

**Q4：EP 为什么不能用在稠密模型上？**
EP 切的是「专家」，稠密模型的 FFN 没有专家这个概念，切了也没有对应的计算单元。`{'EP': 2}` 配 `model_kind='dense'` 会得到 `'EP degree greater than 1 requires an MoE model'`。同理，EP 候选的 `model_kinds` 只写 `{'moe'}`，dense 环境直接不匹配。

**Q5：2.5 ms 是实测的通信时间吗？**
不是。它是「每 rank 载荷 ÷ 带宽 + 固定时延」的串行估算，返回值名字就叫 `lower_bound_ms`。真实时间还受拓扑、拥塞、消息大小、算法放大系数和重叠影响；notebook 只提供比较候选的下界。

## 动手验证：怎么知道自己做对了
没有测试运行器，也没有保存输出；检查是单元末尾的 `assert`，失败抛 `AssertionError`。

- Q1 断言 `fits['required_gib'] == 19`、`fits['headroom_gib'] == 5`、`fits['fits_one_device']` 为真，`does_not_fit['fits_one_device']` 为假；打印两份账本。
- Q2 断言两组 `total_payload_mib` 都是 256、`transfer_ms` 都是 2.5、`lower_bound_ms` 分别是 2.505 和 2.54；打印「低频大载荷」「高频小载荷」两份字典。
- Q3 断言 DP 的 `reasons` 只有容量一条、TP 的 `conditions_met` 为真、EP 的 `reasons` 只有 `'model structure is unsupported'`；循环打印四个候选的完整报告。
- Q4 断言 `valid_plan['valid']` 且 `required_ranks == 16`、`rank_shortage['issues']` 长度为 1、`wrong_structure['issues'] == ['EP degree greater than 1 requires an MoE model']`、`shared_groups['required_ranks'] is None`；循环打印四份结果。

想练手，可以把 `environment` 的 `memory_per_device_gib` 改成 32，观察 DP（29 GiB）是否从排除变成通过；再把 `link_scope` 改成 `cross_node`，看只允许机内的 TP 是否被拒。注意没有保存的历史输出，结论以你自己跑出的字典为准。
