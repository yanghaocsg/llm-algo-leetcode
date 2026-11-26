# 49. 并行策略选型 — 小白讲解

> 对应 notebook：[`49_Parallelism_Strategy_Selection.ipynb`](./49_Parallelism_Strategy_Selection.ipynb)

## 一句话总结
并行不是一张"选一个"的菜单：ZeRO 切训练状态、TP 切层内张量、PP 切模型层、CP 切长序列、EP 切专家，Serving 副本则只分发独立请求。本节把七种观测信号归一化成 0–1 的压力，三个 TODO 依次完成"找主压力和次要风险""按硬约束与压力相关度排序候选""组合兼容方案并给出复测字段"，最后用 CPU 测试覆盖训练状态、MoE、长上下文和 Serving 四条路径。

## 为什么需要它：痛点是什么
初学者容易把"显存不够"直接等同于"上 TP"。但显存不够可能只是优化器状态太大，这时 DP + ZeRO 更合适；模型放不下也可能是单层矩阵太大，TP 才对；长上下文把单卡序列状态撑爆，则要用 CP。切分对象判断错，新增的 collective、气泡或 all-to-all 会掩盖原问题。

生活类比：搬家时东西太多，可以按抽屉分给不同人（ZeRO/DP）、把大衣柜拆成板（TP）、按房间分批搬（PP）、把长地毯裁段（CP），MoE 还要按专家分工（EP）。不能因为"人多"就随便拆，得先看是哪一类东西装不下。

第二个痛点是"可行"和"值得"是两件事。$P=P_{\mathrm{DP}}P_{\mathrm{TP}}P_{\mathrm{PP}}P_{\mathrm{CP}}P_{\mathrm{EP}}$ 只是进程总数的必要条件，每一维还要满足层数、Head 数、序列长度、专家数和设备网格的整除/复制规则；即使组合合法，也要用扩展效率和通信暴露判断收益。

## 核心问题
- 七种观测信号分别对应哪类切分对象？为什么只能有一个主压力？
- TP 为什么要求"大线性层 + 强互联"两个条件同时成立？
- 主压力记 3 分、次要风险记 1 分，会带来什么排序结果？
- 兼容组合表怎么读？为什么 `dp_zero` 能和 `tp` 组合，却不能和 `replica` 组合？
- 什么时候给 `accept`、`tune` 或 `collect_more_evidence`？
- 每个候选的 `required_measurements` 为什么不同？拿到数后怎么判断是否回调策略？

## 解决思路：一步步看懂

**第 1 步：先识别压力来自哪里。** Step 1 把信号和候选方向对应起来：

| 观察到的信号 | 要追问的问题 | 常见候选 |
|---|---|---|
| optimizer state 或参数显存高 | 状态能否按 rank 分片？ | DP + ZeRO |
| 单层矩阵过大且互联强 | 张量维能否拆到多卡？ | TP |
| 模型放不进单卡、stage 时长不同 | 层能否分段、气泡是否可接受？ | PP |
| MoE token 分布不均 | 专家负载和 dispatch 能否均衡？ | EP |
| 长上下文导致单 rank 序列状态过大 | 序列维是否要跨 rank 切？ | CP / Sequence Parallel |
| Serving 队列或副本负载不均 | 请求能否按队列/缓存/会话路由？ | 模型副本 + 请求路由 |

`classify_parallel_bottleneck` 接收七个键：`state_memory_pressure`、`activation_memory_pressure`、`comm_exposed_ratio`、`pipeline_bubble_ratio`、`rank_imbalance_ratio`、`long_context_pressure`、`serving_queue_pressure`，值都在 0–1。TODO 1 的参考答案：

```python
primary_signal, severity = max(signals.items(), key=lambda item: item[1])
primary = signal_to_bottleneck[primary_signal]
secondary_risks = [
    signal_to_bottleneck[name]
    for name, value in signals.items()
    if name != primary_signal and value >= 0.50
]
```

`max` 取最强的信号作为唯一主压力，`severity` 保留原始强度。次要风险的门槛写死为 0.50：排除主信号后，不低于 0.50 的信号都保留并映射成机制名称。阈值不能再低，否则 0.1、0.2 的弱信号会让候选列表膨胀、让 `tune` 无谓触发。测试 `test_parallel_bottleneck_classification` 用 `state=0.85`、`activation=0.20`、`comm=0.40`、`bubble=0.10`、`rank_imbalance=0.55`，断言主压力 `state_memory`、强度 0.85、次要风险只有 `['load_imbalance']`；0.40 的通信和 0.20 的激活被门槛挡掉。空输入或越界值（如 `1.1`）抛 `ValueError`，避免"负压力"进入排序。

**第 2 步：TODO 2——先过硬约束，再算相关度。** 每个候选的 `eligible` 是硬门槛：

- `dp_zero`：主压力必须是 `state_memory`。
- `tp`：`large_linear_layer` 且 `strong_interconnect`，两个都为真。
- `pp`：主压力是 `activation_memory` 或 `pipeline_bubble`，**或者** `model_fits_single_gpu` 为假。后一条是"模型根本放不下"的兜底。
- `ep`：`has_moe` 为真。
- `cp`：`long_context` 为真。
- `replica`：`serving_workload` 为真且 `model_fits_single_gpu` 为真。

TP 的两个条件缺一不可：只有大线性层才值得拆矩阵；互联弱时 all-reduce 会把收益吃掉。PP 的 `or not model_fits_single_gpu` 则说明"放不下"是硬约束，即使主压力不是激活/气泡也要考虑按层切分，但 PP 有气泡和 stage 不均衡的代价，后面必须复测。通过硬门槛后：

```python
primary_match = primary in profile['pressures']
secondary_matches = len(secondary & profile['pressures'])
score = 3 * int(primary_match) + secondary_matches
```

主压力匹配记 3 分，每个次要风险匹配记 1 分，表达"先解决最痛的问题，再顺手处理次要问题"。排序键是 `(-score, name)`，同分按名字字母序，保证确定性；分数为 0 的候选直接丢弃。风险字段不是"候选天生有风险"，而是"当前已知压力与候选风险集合有交集"才标记：

```python
known_pressures = {primary} | secondary
risk = profile['risk'] if known_pressures & profile['risk_pressures'] else None
```

`test_training_state_candidate_gate` 用主压力 `state_memory`、无次要风险、`model_fits_single_gpu=True`、其余约束为假，结果必须精确等于 `[{'name': 'dp_zero', 'score': 3, 'risk': None}]`：TP 因互联/大层不满足被排除，PP 因模型放得下且主压力不是激活/气泡被排除。`test_specialized_candidate_ranking` 再验证：MoE 场景（`load_imbalance` + `communication`、`has_moe=True`）EP 得 $3+1=4$ 分排第一；长上下文场景（`long_context=True`、模型放不下）CP 得 4 分排第一，PP 虽 eligible 但两个压力都不匹配，得 0 被丢弃；Serving 场景 replica 得 3 分排第一。

**第 3 步：TODO 3——兼容组合与复测决策。** 先处理"没有候选"：

```python
if not ranked or ranked[0]['score'] <= 0:
    return {'decision': 'collect_more_evidence', 'plan': None,
            'reason': '没有足够强的约束支持某个并行候选',
            'compatible_plans': [],
            'required_measurements': ['peak_memory_mb', 'step_time_ms']}
```

证据不足时不要硬选，先补峰值显存和步时。否则取 `ranked[0]`，查兼容表：

```python
compatibility = {
    'dp_zero': {'tp', 'pp'}, 'tp': {'dp_zero', 'pp', 'cp', 'ep'},
    'pp': {'dp_zero', 'tp'}, 'ep': {'dp_zero', 'tp'},
    'cp': {'dp_zero', 'tp'}, 'replica': {'tp', 'pp', 'cp', 'ep'},
}
compatible_plans = [item['name'] for item in ranked[1:]
                    if item['name'] in compatibility[candidate['name']]]
```

这是"可以叠加"的关系，不是"相似"的关系。`dp_zero` 切训练状态，可以和 TP（层内）或 PP（层间）组合，但不能和 `replica` 组合，因为副本是 Serving 层请求分发，不参与训练切分；`replica` 反而可以和 TP/PP/CP/EP 组合，因为"每个副本内部怎么切"和"副本之间怎么路由"是两层决策。测试 `test_parallel_followup_compatibility_and_missing_evidence` 用 `dp_zero`（4 分）、`tp`（3 分）、`replica`（2 分）验证 `compatible_plans == ['tp']`：`replica` 不在 `dp_zero` 的兼容集合里，分数再高也不能进。

然后决定 `tune` 还是 `accept`：

```python
needs_tuning = candidate['risk'] is not None or bool(summary['secondary_risks'])
decision = 'tune' if needs_tuning else 'accept'
```

首选候选有风险，**或者**还有任何次要风险没解决，就进 `tune`。次要风险可能不在首选候选的处理范围内，但仍然要复测或另行处理；直接 `accept` 会把"已知但未解决"的压力藏起来。`test_parallel_followup_accept_and_tune` 用无风险的 `dp_zero` 得到 `accept`，用带 `inspect_expert_load_and_all_to_all` 风险且有通信次要风险的 EP 得到 `tune`。

`required_measurements` 按候选不同：

| 候选 | 必测字段 | 读法 |
|---|---|---|
| `dp_zero` | `peak_memory_mb_per_rank`、`payload_bytes_per_rank`、`median_max_rank_ms` | 显存降了没？同步时间涨没涨？ |
| `tp` | `payload_bytes_per_rank`、`median_max_rank_ms`、`max_abs_error` | all-reduce 暴露多少？数值误差可接受吗？ |
| `pp` | `bubble_ratio`、`stage_time_ms`、`peak_memory_mb` | 气泡多大？哪个 stage 拖后腿？ |
| `ep` | `cross_rank_dispatch_bytes`、`load_skew_ratio`、`p95_max_rank_ms` | all-to-all 多贵？专家负载偏不偏？ |
| `cp` | `kv_exchange_bytes_per_rank`、`median_max_rank_ms`、`peak_memory_mb_per_rank` | 序列通信有没有抵消容量收益？ |
| `replica` | `queue_wait_p95_ms`、`cache_hit_rate`、`throughput_requests_per_s` | 吞吐涨了但尾延迟/缓存命中变差没？ |

读到数后按三步判断：先看容量和正确性（能不能跑、`max_abs_error` 是否可接受），再看扩展效率 $\eta_P$ 和最大 rank 时间，最后看暴露通信/气泡/负载偏斜是否把收益吃掉。其中

$$
S_P=\frac{T_1}{T_P},\qquad \eta_P=\frac{S_P}{P},
$$

$T_1$ 是单设备基线时间，$T_P$ 是 $P$ 个设备上的时间，$\eta_P=1$ 是线性扩展。notebook 还提醒：只有真实 Trace 提供 `exposed_comm_ms` 和 step time 时，才计算 `comm_exposed_ratio`，不要把"通信总量"直接当"暴露时间"。

**第 4 步：组合并行的进程约束。** 上面的 $P$ 乘积只是必要条件；每一维还要满足层数能被 PP 的 stage 数整除、Head 数能被 TP 并行度整除、序列长度能配合 CP 切分、专家数能配合 EP 放置，设备网格和进程组也要能映射到实际拓扑。notebook 没把这些规则写成代码，因为取值依赖具体模型和集群；本节只负责"从信号到候选"，合法性要在目标框架里验证。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| DP | 数据并行：每张卡放一份模型，分不同的数据 |
| ZeRO | 把参数、梯度或优化器状态按 rank 分片，降低单卡驻留 |
| TP | 张量并行：把单层矩阵的维度拆到多张卡 |
| PP | 流水线并行：把模型层切成若干 stage，分批喂 micro-batch |
| CP | 上下文/序列并行：把长序列的状态切到多个 rank |
| EP | 专家并行：把 MoE 的专家分配到不同卡 |
| Replica routing | 模型副本 + 请求路由，不切模型，只分发独立请求 |
| 主压力 | 归一化信号里最强的那一个，决定首选方向 |
| 次要风险 | 排除主信号后仍不低于 0.50 的压力 |
| 硬约束 | 模型结构/互联/workload 条件，不满足直接排除 |
| 兼容组合 | 可以叠加的并行维，比如 DP+ZeRO 与 TP |
| 扩展效率 $\eta_P$ | 加速比除以设备数，衡量新增设备有多少转化成速度 |

## 小白常见疑问

**Q1：显存压力最大，为什么不能无脑选 TP？**
TP 的 `eligible` 要求 `large_linear_layer` 和 `strong_interconnect` 同时为真。显存压力可能来自优化器状态而非单层矩阵；即使单层很大，互联不够强时 all-reduce 也会吃掉收益。`test_training_state_candidate_gate` 就是在"只有状态压力、互联和大层都不满足"时断言只剩 `dp_zero`。

**Q2：次要风险为什么定在 0.50？**
这是 notebook 写死的教学阈值，不是硬件常数。它用来区分"值得同时关注"和"噪声"：测试里 0.55 的负载不均被保留，0.40 的通信被排除。真实项目里应结合 SLO 和测量噪声调整；调了之后要同步更新测试期望，不能只改代码不改断言。

**Q3：PP 明明不是主压力，为什么有时也进候选？**
因为 `pp` 的 `eligible` 有 `or not model_fits_single_gpu`。模型放不进单卡是硬约束，即使当前最痛的是长上下文或负载不均，也必须把 PP 纳入考虑。长上下文测试里 PP 确实 eligible，但两个压力都不匹配，得 0 分被丢弃——"可行"不等于"入选"。

**Q4：首选没风险，但还有次要风险，为什么结果是 `tune`？**
`needs_tuning` 同时检查首选候选的 `risk` 和全局 `secondary_risks`。次要风险可能不在首选候选的处理范围内，但仍然要复测或另行处理；直接 `accept` 会隐藏已知压力。EP 案例就是首选带风险 + 通信次要风险，结果 `tune`，必测字段含 `load_skew_ratio`。

## 动手验证：怎么知道自己做对了

notebook 依次直接调用六个测试，最后打印 `parallel strategy selection passed`。没有保存执行输出，以下以代码与断言为准；TODO 没填时三个函数会抛 `NotImplementedError`，测试在第一个调用处失败。

- `test_parallel_bottleneck_classification`：输入 `{'state_memory_pressure': 0.85, 'activation_memory_pressure': 0.20, 'comm_exposed_ratio': 0.40, 'pipeline_bubble_ratio': 0.10, 'rank_imbalance_ratio': 0.55}`，返回值必须精确等于 `{'primary': 'state_memory', 'severity': 0.85, 'secondary_risks': ['load_imbalance']}`。
- `test_parallel_bottleneck_validation`：空字典和 `{'state_memory_pressure': 1.1}` 都必须抛 `ValueError`。
- `test_training_state_candidate_gate`：主压力 `state_memory`、无次要风险、`{'model_fits_single_gpu': True, 'strong_interconnect': False, 'large_linear_layer': False, 'has_moe': False}`，排序结果必须精确等于 `[{'name': 'dp_zero', 'score': 3, 'risk': None}]`。
- `test_specialized_candidate_ranking`：三组用例分别要求首选 `ep`（`load_imbalance` + `communication`、`has_moe=True`）、`cp`（`long_context` + `communication`、模型放不下、`long_context=True`）、`replica`（`serving_queue`、`serving_workload=True`）。
- `test_parallel_followup_accept_and_tune`：`dp_zero` 无风险 → `accept`，且 `peak_memory_mb_per_rank` 在必测字段里；EP 带 `inspect_expert_load_and_all_to_all` 风险且有通信次要风险 → `tune`，且 `load_skew_ratio` 在必测字段里。
- `test_parallel_followup_compatibility_and_missing_evidence`：`dp_zero`（4 分）+ `tp`（3 分）+ `replica`（2 分）→ `compatible_plans == ['tp']`；空候选 → `collect_more_evidence`、`plan is None`、`compatible_plans == []`。

跑通测试只说明"信号→候选"的规则实现正确，不说明真实集群一定加速。运行环境说明提到的多 GPU 实测不在本页正文；本页交给下一步的复测清单就是 TODO 3 返回的 `required_measurements`。如果显存降了但同步时间显著增长，或者吞吐涨了但 P95/P99 变差，就要回调策略，而不是只盯一个平均指标。
