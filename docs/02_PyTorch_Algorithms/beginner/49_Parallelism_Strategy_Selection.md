# 49. Parallelism Strategy Selection | 并行策略选型 · 小白入门

> 配套 notebook：[49. 并行策略选型](../49_Parallelism_Strategy_Selection.md) ｜ 难度：Medium ｜ 环境：CPU-first

## 一句话说清

模型装不下或者跑不快时，有好几种把活分给多张卡的办法，各自解决不同的麻烦。这本 notebook 教你从观测到的信号出发挑出最合适的一种，并列出下一步要复测什么。

## 为什么需要它

模型训练或推理会遇到几种麻烦：单卡装不下、某一层的矩阵太大、长序列的状态太大、流水线有空等、各卡负载不均。它们都可以靠“把活分给多张卡”来缓解，但分的东西不一样：有的分训练状态，有的分层里的矩阵，有的分模型的层，有的分序列，有的分专家，Serving 则是把请求分给多个副本。

麻烦在于，分错了不会报错，只会用新的通信或路由开销把原来的问题盖住。比如显存压力来自优化器状态，你却去拆某一层的矩阵：矩阵是小了，状态还占着原来的显存，又多出一轮通信。

而且几个信号常常同时偏高。notebook 的测试里有三个：状态显存压力 0.85、通信暴露比例 0.40、rank 负载不均衡 0.55。先处理哪一个，其余的要不要管，需要一条规则。

这本 notebook 要做的是：从信号里找出唯一的主压力，保留达到阈值的次要风险，再结合模型和集群的条件生成候选并排序，最后给出首选方案、能和它组合的候选，以及复测的决策。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 切分对象 | 要分到多张卡上的那样东西：训练状态、一层里的矩阵、模型的层、序列、专家，或者请求 |
| DP + ZeRO | 把参数、梯度、优化器状态切开存到各卡上，用来缓解状态显存压力 |
| 张量并行（TP） | 把单层里的大矩阵拆到多卡上一起算 |
| 流水线并行（PP） | 把模型按层分成几段，分放到不同的卡上 |
| 专家并行（EP） | MoE（混合专家）模型里，把不同专家的计算分给不同的卡 |
| 上下文并行（CP）/ 序列并行 | 把长序列切到多卡上 |
| `signals` / `constraints` | 观测到的压力信号，每个值在 0 到 1 之间；以及模型和集群的条件，比如模型能否放进单卡 |

## 核心思路

输入有两样。一是 `signals`：几个归一化的信号，每个值在 0 到 1 之间。二是 `constraints`：模型和集群的条件，比如模型能否放进单卡、有没有 MoE、是不是 Serving 场景。处理分三步：从信号里找出唯一的主压力，保留达到阈值的次要风险；让每个候选先过硬约束，再按它和主压力、次要风险的匹配程度打分排序；取第一名作为首选方案，列出能和它组合的候选，给出决策。输出是首选方案、兼容候选和复测要测的量。

每个信号对应一种压力：

```text
state_memory_pressure        → state_memory       优化器状态或参数的显存压力
activation_memory_pressure   → activation_memory  激活带来的显存压力
comm_exposed_ratio           → communication      通信暴露在关键路径上的比例
pipeline_bubble_ratio        → pipeline_bubble    流水线里空等时间的比例
rank_imbalance_ratio         → load_imbalance     各 rank 负载不均衡
long_context_pressure        → long_context       长序列带来的压力
serving_queue_pressure       → serving_queue      Serving 请求排队的压力
```

三步用到的规则如下，阈值取自 notebook 的参考答案：

```text
主压力   = 最大的信号对应的压力
次要风险 = 其余强度不低于 0.50 的信号对应的压力
硬约束   = 候选的 eligible 条件，不满足就不进入列表
得分     = 3 ×（主压力在候选能处理的压力里）+ 次要风险里命中候选压力的个数；按得分降序排，并列时按名字
兼容候选 = 排在其后、且能与首选方案组合的候选
决策     = 首选方案有已知风险，或有次要风险 → tune；否则 → accept；没有候选 → collect_more_evidence
```

下面是最小例子，把三步连起来走一遍：

```python
to_pressure = {'state_memory_pressure': 'state_memory', 'comm_exposed_ratio': 'communication',
               'rank_imbalance_ratio': 'load_imbalance'}
signals = {'state_memory_pressure': 0.85, 'comm_exposed_ratio': 0.40, 'rank_imbalance_ratio': 0.55}

# TODO 1：最大的信号是主压力，其余不低于 0.50 的信号是次要风险
primary_signal, severity = max(signals.items(), key=lambda item: item[1])
primary = to_pressure[primary_signal]
secondary = {to_pressure[k] for k, v in signals.items() if k != primary_signal and v >= 0.50}
print(primary, severity, sorted(secondary))

# TODO 2：两个满足硬约束的候选，各自是（能处理的压力，已知的风险压力，可组合的候选）
profiles = {'dp_zero': ({'state_memory'}, {'communication'}, {'tp', 'pp'}),
            'tp': ({'state_memory', 'communication'}, {'communication'}, {'dp_zero', 'pp', 'cp', 'ep'})}
ranked = []
for name, (pressures, risk_p, _) in profiles.items():
    score = 3 * int(primary in pressures) + len(secondary & pressures)
    risk = sorted(({primary} | secondary) & risk_p)
    ranked.append((name, score, risk))
ranked.sort(key=lambda item: (-item[1], item[0]))
print(ranked)
# TODO 3：首选方案的兼容候选和决策
name, _, risk = ranked[0]
compatible = [n for n, _, _ in ranked[1:] if n in profiles[name][2]]
needs_tuning = bool(risk) or bool(secondary)
print(name, compatible, 'tune' if needs_tuning else 'accept')
```

```text
state_memory 0.85 ['load_imbalance']
[('dp_zero', 3, []), ('tp', 3, [])]
dp_zero ['tp'] tune
```

第一行是 TODO 1 的结果：最大的信号对应主压力 `state_memory`；`rank_imbalance_ratio` 是 0.55，不低于 0.50，成为次要风险 `load_imbalance`；通信暴露比例 0.40 没到阈值，不进入结果。第二行是排序：`dp_zero` 和 `tp` 都能处理主压力，各得 3 分，并列时按名字排。两者的已知风险都是通信，而当前的压力里没有通信，所以 risk 为空。第三行是 TODO 3：首选 `dp_zero`，排在后面的 `tp` 能和它组合；首选方案没有已知风险，但存在次要风险，所以决策是 `tune`。例子假设 `tp` 的硬约束（有大线性层、互联强）都满足；notebook 里每个候选的 `risk` 是一个提示字符串，或者 `None`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从工作负载信号识别切分对象：六种信号各先追问什么问题，常见的候选方向是什么 | 读 |
| Step 2 | 用约束生成并行候选：六种候选各自的适用约束、主要收益和首先要复测的风险 | 读，对照上面的词表 |
| Step 3 | 比较风险并设计复测：每种方案排到前列时要保留什么证据，出现什么情况要回调 | 读 |
| Step 4 · TODO 1 | classify_parallel_bottleneck 已提供信号到瓶颈的映射 | 写 `classify_parallel_bottleneck`：3 行，取最大信号，映射成主压力，再筛出不低于 0.50 的次要风险 |
| Step 4 · TODO 2 | rank_parallel_candidates 已提供候选的适用条件、目标压力和风险 | 写 `rank_parallel_candidates`：3 行，算出 `primary_match`、`secondary_matches`、`score` |
| Step 4 · TODO 3 | recommend_parallel_followup 已提供兼容关系和每类候选的测量字段 | 写 `recommend_parallel_followup`：3 行，算出 `compatible_plans`、`needs_tuning`、`decision` |
| 测试 | 检查主压力和次要风险的筛选、非法信号是否报错、不可行候选是否被排除、兼容组合和三种决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |

整本 notebook 只需要 CPU。

## 容易踩的坑

- **信号越界没有拦住**：`signals` 不能为空，每个值都要在 0 到 1 之间，否则抛出 `ValueError`。空字典和 1.1 这样的值都在测试里。
- **保留了多个主压力**：只有最大的信号是主压力。其余信号要不低于 0.50 才进入次要风险，并且要排除主信号本身。
- **不可行候选进了列表**：先检查候选的硬约束，不满足的直接跳过；得分为 0 的也不进入列表。
- **次要风险的权重超过了主压力**：主压力匹配计 3 分，次要风险每项只计 1 分。并列时按名字排序。
- **把排名当成结论**：候选排序只是开始。要对每个候选固定同一个模型、批次、序列长度和精度再复测；有已知风险或次要风险时，决策是 `tune` 而不是 `accept`。

## 读完后自查

- [ ] 能说出六种并行候选各自切分的是什么。
- [ ] 给一组信号，能找出主压力和次要风险。
- [ ] 能算出一个候选的得分，说出主压力匹配和次要风险匹配各占多少。
- [ ] 能说出什么情况下决策是 `accept`，什么情况下是 `tune`。

## 下一步

- 打开 notebook：[49. 并行策略选型](../49_Parallelism_Strategy_Selection.md)
- 建议先读：[27. ZeRO 优化器模拟 · 小白入门](./27_ZeRO_Optimizer_Sim.md)、[28. Pipeline 并行微批次 · 小白入门](./28_Pipeline_Parallelism_MicroBatch.md)、[29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim.md)、[PAR-CONTEXT. 上下文与序列并行 · 小白入门](./PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- 相关 notebook：[79. 训练状态并行基准](../79_Distributed_Parallel_Benchmark.md)、[80. MoE 专家并行基准](../80_MoE_Expert_Parallel_Benchmark.md)
