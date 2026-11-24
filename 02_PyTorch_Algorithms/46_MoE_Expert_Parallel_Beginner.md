# 46. MoE Expert Parallel Beginner | MoE 专家并行 · 小白入门

> 配套 notebook：[46. MoE 专家并行](./46_MoE_Expert_Parallel.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

混合专家模型里每个词元只经过少数几个专家，而专家分散在不同的 GPU 上。这本 notebook 教你算清楚：多少词元挤不进专家，多少词元要跨卡传输。

## 为什么需要它

MoE（混合专家）层里有很多个专家，但每个词元只经过其中少数几个，这叫稀疏激活。专家并行（Expert Parallel）再把专家分给不同的 GPU。

只看单机，路由只是“这个词元去哪几个专家”。专家分散到多张卡以后，多出三件麻烦事：

- 词元和专家可能不在同一张卡上。词元的表示要先送过去，专家算完还要送回来，按原来的词元顺序合并。
- 每个专家能接收的路由数有上限。太多词元涌向同一个专家，超出的部分放不进去。
- 专家摆在哪张卡，决定了有多少词元要跨卡传输。

所以同样数量的词元，路由是否均匀、专家怎么放置，会产生不同的溢出、通信字节和长尾风险。这本 notebook 在 CPU 上把这几本账分开算清楚，再给出 `accept`、`tune` 或 `reject` 的判断。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 专家（expert）/ 路由（route） | MoE 层里的小子网络叫专家，用编号表示。一条路由是“某个词元去了某个专家”这一件事 |
| 路由器（router）/ top-k | 路由器给每个词元挑专家，top-k 是每个词元去几个专家 |
| 分发（dispatch）/ 合并（combine） | 分发是把词元的表示送到专家所在的卡；合并是把专家的结果送回来，按原词元顺序加权合起来 |
| 容量（capacity）/ 溢出（overflow） | 容量是一个专家最多接收多少条路由，超出的部分就是溢出 |
| `capacity_factor` | 容量相对平均负载的放大倍数：容量等于平均负载乘它，再向上取整 |
| 本地 / 跨 rank（local / cross-rank） | 词元来源的卡正好是专家所在的卡，叫本地；否则要跨 rank 传输，靠 `all-to-all` 完成，也就是每张卡同时给其他卡各发一份数据。rank 是一张 GPU 的编号 |
| 负载比（balance ratio） | 最忙的专家的负载除以平均负载，1.0 表示完全均匀 |

## 核心思路

输入是每个词元的路由 `routes`（这个词元选中了哪几个专家）、专家数、每个词元来自哪张卡、每个专家放在哪张卡。输出是三本账：

```text
路由账本   routes → 每个专家的负载 → 容量、溢出、负载比
局部性账本 routes + 词元来源 rank + 专家所在 rank → 本地路由数、跨 rank 路由数、跨卡字节
决策       溢出、负载比、跨 rank 比例 → accept / tune / reject
```

前两本账的算法：

```text
平均负载 = 总路由数 / 专家数
容量     = ceil(平均负载 × capacity_factor)        # 没有路由时为 0
溢出     = 各专家 max(负载 - 容量, 0) 之和
负载比   = 最大负载 / 平均负载
跨卡字节 = 跨 rank 路由数 × hidden_size × 每个元素的字节数   # 去程；合并是回程，往返是它的 2 倍
```

`ceil` 是向上取整，`hidden_size` 是每个词元向量的长度。下面用 2 个专家、4 个词元试一下，每个词元去 1 个专家：

```python
from collections import Counter
import math

def ledger(routes, num_experts, capacity_factor=1.0):       # 路由账本
    counts = Counter(e for r in routes for e in r)           # 每个专家被选中了几次
    loads = [counts[e] for e in range(num_experts)]          # 专家负载
    average = sum(loads) / num_experts                       # 平均负载
    capacity = math.ceil(average * capacity_factor)          # 单个专家最多接收多少条路由
    overflow = sum(max(load - capacity, 0) for load in loads)
    return loads, capacity, overflow, max(loads) / average   # 最后一个是负载比

balanced, skewed = [[0], [1], [0], [1]], [[0], [0], [0], [1]]   # 4 个词元各去 1 个专家
print(ledger(balanced, 2))                                  # 均匀
print(ledger(skewed, 2))                                    # 偏斜：3 个词元都去了专家 0
print(ledger(skewed, 2, capacity_factor=1.5))               # 把容量放宽

source = [0, 0, 0, 1]                                       # 偏斜路由里，每个词元的来源 rank
for owner in ({0: 0, 1: 1}, {0: 1, 1: 0}):                  # 专家 0 放在 rank 0，或者换到 rank 1
    cross = sum(owner[e] != src for r, src in zip(skewed, source) for e in r)
    print(cross, cross * 8 * 2, cross / 4)                  # 跨 rank 路由数、去程字节、跨 rank 比例
```

```text
([2, 2], 2, 0, 1.0)
([3, 1], 2, 1, 1.5)
([3, 1], 3, 0, 1.5)
0 0 0.0
4 64 1.0
```

前 3 行是路由账本。均匀时两个专家各收到 2 条路由，平均负载是 2，容量是 2，没有溢出，负载比是 1.0。偏斜时 3 个词元都去了专家 0：负载是 `[3, 1]`，容量还是 2，专家 0 多出 1 条路由，溢出 1，负载比升到 1.5。把 `capacity_factor` 放宽到 1.5，容量变成 3，溢出消失，代价是预留了用不上的空间。

后 2 行是局部性账本，路由不变。3 个词元来自 rank 0，1 个来自 rank 1。专家 0 放在 rank 0、专家 1 放在 rank 1 时，没有路由需要跨卡。把两个专家对调，4 条路由全部要跨 rank，去程 64 字节，跨 rank 比例是 1.0。

最后是决策。`accept` 要同时满足：没有溢出，负载比不超过 1.2，跨 rank 比例不超过 0.5。没达到时，如果溢出比例不超过 0.25、负载比不超过 1.8、跨 rank 比例不超过 0.9，就是 `tune`，否则 `reject`。套到偏斜路由上：溢出 1 条，4 条路由里占 1 条，没有超过调优阈值；负载比 1.5 也没有。专家 0 放在 rank 0 时跨 rank 比例是 0.0，结果是 `tune`。对调之后跨 rank 比例是 1.0，超过 0.9，结果是 `reject`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 词元被路由到不同专家、专家又分布在不同设备上，中间多了哪些环节 | 读 |
| Step 2 | 路由、容量和分发的关系：溢出怎么来，哪些路由才需要跨卡 | 读，对照上面的算式和小例子 |
| Step 3 | 路由分布、容量系数、专家放置各改变什么，要一起观察哪些量 | 读 |
| Step 4 · TODO 1 | 计算容量与溢出，判断 capacity 是否容纳当前路由 | 在 `summarize_expert_parallel` 里写 2 行：由平均负载和 `capacity_factor` 算容量，再汇总各专家超出容量的路由数 |
| Step 4 · TODO 2 | 判断路由局部性，区分本地计算与跨 rank dispatch | 在 `summarize_dispatch_locality` 里写 3 行：比较来源 rank 和专家所在的 rank，分别累加本地和跨 rank 路由数 |
| Step 4 · TODO 3 | 联合容量、负载与通信风险形成决策 | 在 `recommend_moe_parallel_plan` 里写 2 行：按阈值算出 `accept_ready` 和 `tune_ready` |
| 测试 | 检查均匀、偏斜、空路由和非法配置下的路由账本，本地命中率与跨卡字节，以及接受、调优、拒绝三种决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实 NCCL 环境里执行分发、专家计算、合并，比较三类路由 | 需要 `PAR-CUDA-MECHANISM` 环境和至少两张 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要至少两张 GPU，默认不执行真实实验。

## 容易踩的坑

- **容量没按平均负载算**：容量是“平均负载 × `capacity_factor`”向上取整，对每个专家是同一个上限。没有路由时容量是 0。
- **溢出算成总路由数减总容量**：要把每个专家超出容量的部分分别截断再相加。例子里专家 1 的负载没到容量，但省下的名额抵不了专家 0 的溢出。
- **把全部路由都算成跨卡通信**：只有专家不在词元来源 rank 上的路由才需要跨 rank 传输。全部路由的规模是逻辑工作量，要和真正的网络传输分开记录。
- **往返只算了一遍**：分发是去程，合并是回程，往返字节是去程的 2 倍。
- **决策只看溢出**：`accept` 要同时满足无溢出、负载比和跨 rank 比例三个阈值；空路由直接 `reject`。

## 读完后自查

- [ ] 能说出专家分散到多张卡以后，一条路由要经过哪两步通信。
- [ ] 4 个词元里 3 个去了专家 0、1 个去了专家 1，`capacity_factor` 为 1.0，能算出容量、溢出和负载比。
- [ ] 能解释为什么专家放置会改变跨卡字节，以及放得更本地时要同时观察什么。
- [ ] 能说出 `accept` 要同时满足的三个条件。

## 下一步

- 打开 notebook：[46. MoE 专家并行](./46_MoE_Expert_Parallel.ipynb)
- 建议先读：[06. MoE 路由器 · 小白入门](./06_MoE_Router_Beginner.md)、[07. MoE 负载均衡损失 · 小白入门](./07_MoE_Load_Balancing_Loss_Beginner.md)
- 做完之后：[47. NCCL 通信剖析 · 小白入门](./47_Communication_Profiling_with_NCCL_Beginner.md)
