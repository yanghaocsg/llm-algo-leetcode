# 80. MoE Expert Parallel Benchmark Beginner | MoE 专家并行基准 · 小白入门

> 配套 notebook：[80. MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

混合专家模型里，每个词元只交给少数几个专家处理，专家分散在不同卡上时词元要跨卡传送。这本 notebook 教你汇总几种传送方式的测量结果，再按收益、容量和稳定性决定采不采用。

## 为什么需要它

混合专家模型里有很多个专家，每个词元只交给其中少数几个处理，由路由器（Router）来挑。专家并行（EP）把专家分散到不同的卡上以后，词元选中的专家可能在别的卡上。这时要先把词元发过去，叫分发（dispatch）；专家算完再收回来，按原顺序拼好，叫合并（combine）。

路由有三种形态，各有各的风险：

- 本地路由：词元只去本卡的专家，不用跨卡传，但本地专家的容量可能不够。
- 均衡跨卡路由：词元均匀地发到多张卡的专家，避开了单个专家过热，但多了一笔固定的跨卡通信。
- 偏斜跨卡路由：少数远端专家收到大量词元，热点、长尾延迟和溢出叠在一起。

比较这三种形态时，最容易犯的错是只看吞吐。偏斜路由的原始吞吐可能更高，但超出容量的词元会被丢弃、回退或延迟处理，真正完成的词元变少了。notebook 的测试里，偏斜路由的有效吞吐最高，却有 280 个词元溢出。

所以要分三步：把几次运行的记录汇总成一张表，再和基线逐项对比，最后按收益、容量、通信和稳定性给出 `accept`、`tune` 或 `reject`。这本 notebook 让你把这三步各写成一个函数。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 专家（expert） | 混合专家模型里的小子网络，有很多个。每个词元只交给其中少数几个处理 |
| 路由器（Router） | 给每个词元挑选专家的那一层 |
| 专家并行（EP） | 把专家分散放到不同的卡上。rank 是多卡训练里的一个成员，可以理解成一张卡 |
| 分发（dispatch）/ 合并（combine） | 词元要去别的卡上的专家时，先把它发过去叫分发；算完收回来、按原顺序拼好叫合并 |
| 容量（capacity）/ 溢出（overflow） | 容量是一个专家一步里最多能接收的词元数，超出的词元数叫溢出 |
| 有效吞吐 | 按真正处理完的词元算的吞吐，扣掉了溢出的部分 |
| 负载不均衡（imbalance）/ 稳定性（stability） | 记录里的两个指标：不均衡越低越好，稳定性越高越好，决策时都设了门槛 |

## 核心思路

输入是几条运行记录，每条有这些字段：名字 `name`、通信时间 `comm_ms`、负载不均衡 `imbalance`、有效吞吐 `effective_throughput`、稳定性 `stability`、溢出 `overflow`、有效词元数 `effective_tokens`。决策时还要给几个门槛：`max_imbalance`、`min_stability`、`max_overflow`（默认 0.0）和 `max_comm_increase_ms`（默认 12.0）。

先看溢出和有效词元是怎么来的。每个专家一步里只能接收有限的词元：

```text
overflow         = max(分到的词元数 - capacity, 0)
effective tokens = 各个专家的（分到的词元数 - overflow）加起来
```

处理分三步。第一步汇总。把所有记录的有效词元数加起来，通信时间取平均，溢出取最大值，再找出有效吞吐最高的一条，记下它的名字。

第二步对比。用候选减去基线，得到吞吐收益，以及通信时间、溢出、不均衡、稳定性四个变化量。方向要统一：有效吞吐和稳定性越高越好，通信、不均衡、溢出越低越好。

第三步决策：

```text
吞吐收益大于 0，并且 imbalance、stability、overflow、通信增量都在门槛内   → accept
吞吐收益大于 0，stability 达标，但别的门槛没过                           → tune
其余                                                                    → reject
```

用 notebook 测试里的三条记录试一下：

```python
runs = [
    {'name': 'local',               'comm_ms': 40, 'imbalance': 0.25, 'effective_throughput': 100, 'stability': 0.82, 'overflow': 0,   'effective_tokens': 1000},
    {'name': 'cross_rank_balanced', 'comm_ms': 48, 'imbalance': 0.12, 'effective_throughput': 128, 'stability': 0.80, 'overflow': 0,   'effective_tokens': 1000},
    {'name': 'cross_rank_skewed',   'comm_ms': 47, 'imbalance': 0.72, 'effective_throughput': 132, 'stability': 0.80, 'overflow': 280, 'effective_tokens': 720},
]
base, balanced, skewed = runs

print(sum(r['effective_tokens'] for r in runs), max(r['overflow'] for r in runs))   # 汇总
print(max(runs, key=lambda r: r['effective_throughput'])['name'])
print(balanced['comm_ms'] - base['comm_ms'], balanced['effective_throughput'] - base['effective_throughput'],
      round(balanced['imbalance'] - base['imbalance'], 4), balanced['overflow'] - base['overflow'])   # 对比

print(skewed['effective_throughput'] > base['effective_throughput'], skewed['imbalance'] <= 0.15,   # 决策的五个条件
      skewed['stability'] >= 0.78, skewed['overflow'] <= 0.0, skewed['comm_ms'] - base['comm_ms'] <= 12.0)
```

```text
2720 280
cross_rank_skewed
8 28 -0.13 0
True False True False True
```

汇总后三条记录一共有 2720 个有效词元，最大溢出是 280，有效吞吐最高的是偏斜路由。均衡跨卡比基线多花 8 毫秒通信，换来 28 的吞吐收益，不均衡降了 0.13，没有新增溢出。最后一行是偏斜路由的五个决策条件：不均衡 0.72 超过上限 0.15，溢出 280 超过上限 0，另外三项通过，所以它是 `tune`，不是 `accept`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从 Router 选择到跨 rank 专家执行：本地、均衡跨卡、偏斜跨卡三种路由形态的风险和要记录的证据 | 读 |
| Step 2 | 容量与 overflow 如何改变有效工作量：分到的词元数、容量、溢出、有效词元四个量怎么算 | 读，对照上面的公式 |
| Step 3 | 把通信、负载与吞吐放到同一对照中：为什么要用有效吞吐，以及 06、07、46、47 几本 notebook 的证据怎样衔接进来 | 读 |
| Step 4 · TODO 1 | 汇总 `token`、通信与 `overflow` | 在 `summarize_moe_parallel_runs` 里补 4 个值：有效词元总数、平均通信时间、最大溢出、有效吞吐最高的运行名 |
| Step 4 · TODO 2 | 计算吞吐、通信、负载与 `overflow` 的变化 | 在 `compare_moe_parallel_to_baseline` 里补 5 个差值：吞吐收益、通信时间差、溢出差、不均衡差、稳定性差 |
| Step 4 · TODO 3 | 按容量、稳定性和收益作出决策 | 在 `recommend_moe_parallel_run` 里补 `capacity_ok`、`communication_ok` 和 `decision`：五项条件同时满足才 `accept` |
| 测试 | 检查汇总的数值和最高吞吐候选、相对基线的差值、均衡跨卡得到 `accept` 而偏斜跨卡得到 `tune` | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在多卡上实测全对全通信（all-to-all）的分发、本地专家计算和合并，比较均匀与偏斜两种路由 | 需要 CUDA、至少 2 张 GPU 和 `torchrun`；默认是 `RUN_GPU_EXPERIMENT = False`，关闭时只打印所需条件，不运行实验 |

Step 1–4 只需要 CPU。Step 5 需要多张 GPU，默认关闭。

## 容易踩的坑

- **把最高吞吐的运行当成结论**：`best_effective_throughput_run` 只是帮你定位最值得回看的候选。测试里它指向偏斜路由，可偏斜路由最后是 `tune`。
- **差值的方向不统一**：所有差值都用候选减基线。有效吞吐和稳定性越大越好，通信、不均衡、溢出越小越好，所以收益为正才算好，代价类的变化为负才算好。
- **只看吞吐就 `accept`**：吞吐有收益、不均衡不超上限、稳定性不低于下限、溢出不超上限、通信时间增量不超过 `max_comm_increase_ms`，五项同时成立才能 `accept`。
- **把 `tune` 和 `reject` 弄混**：吞吐有收益、稳定性也达标，只是容量、负载或通信还没过关，是 `tune`。吞吐没有收益或稳定性不达标，才是 `reject`。
- **漏了题目注释里没列的字段**：汇总的返回值要带 `run_count`，对比的返回值要带 `stability_delta`，测试都会检查。

## 读完后自查

- [ ] 能说出三种路由形态的名字，以及各自先出现的风险。
- [ ] 能说出溢出和有效词元各怎么算。
- [ ] 能说出 `accept` 要同时满足哪五项条件。
- [ ] 吞吐有收益、稳定性达标，但不均衡超过上限时，能说出决策是什么。

## 下一步

- 打开 notebook：[80. MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
- 建议先读：[06. MoE 路由器 · 小白入门](./06_MoE_Router_Beginner.md)、[07. MoE 负载均衡损失 · 小白入门](./07_MoE_Load_Balancing_Loss_Beginner.md)
- 做完之后：[81. 分布式推理逻辑验证 · 小白入门](./81_Distributed_Inference_Project_Beginner.md)
