# 46. MoE 专家并行 — 小白讲解

> 对应 notebook：[`46_MoE_Expert_Parallel.ipynb`](./46_MoE_Expert_Parallel.ipynb)

## 一句话总结
MoE 的路由只是"每个 token 选 top-k 专家"；一旦专家分散到不同 GPU，就要回答 token 怎么送过去、送多少、热门专家塞不下怎么办。本节用 `summarize_expert_parallel` 算每专家负载、容量和溢出，用 `summarize_dispatch_locality` 区分本地与跨 rank 路由，用 `recommend_moe_parallel_plan` 给出 `accept / tune / reject`，再用 `combine_expert_outputs` 把专家结果按原 token 顺序加权合并。

## 为什么需要它：痛点是什么
初学者容易把 MoE 理解成"参数更多、激活更稀疏，所以更省钱"。但稀疏计算只省了 FLOPs，没有自动省通信：token 被路由到别的 GPU 上的专家时，必须先做一次 all-to-all 把隐藏状态送过去，算完再送回来。热门专家还会让某个 rank 变成长尾，其他 rank 干等；capacity 设小了会丢 token，设大了会预留一堆用不上的槽位。

生活类比：一个跨楼层办公的公司。每个员工（token）把任务交给擅长的人（专家），但专家分布在 4 个楼层（rank）。如果任务正好在本层，走两步就到（local route）；如果在外层，就要坐电梯送过去再送回来（all-to-all 往返）。电梯容量有限（capacity），某层突然收到太多任务就会排队或退件（overflow），而最忙那层决定所有人几点下班（最慢 rank 时间）。

## 核心问题
- 每个专家的容量上限怎么由 token 数、top-k、专家数和 `capacity_factor` 算出来？
- overflow 统计的是 token 还是路由？一个 token 路由到两个专家、只丢一个，算不算丢？
- 同样的路由，为什么"专家放置"会改变跨 rank 字节，却不改变逻辑计算量？
- 本地命中率高就一定好吗？为什么还要同时看负载偏斜？
- 决策阈值 `1.2 / 0.5 / 0.25 / 1.8 / 0.9` 分别在什么位置起作用？combine 为什么用 `index_add_`？

## 解决思路：一步步看懂
### 第 1 步：先算容量与溢出（TODO 1）
`summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0)` 先做参数校验（专家数、hidden、字节数、capacity factor 都必须为正），再遍历 `routes` 统计每个专家收到多少条路由，同时检查 `expert_id` 是否在 `[0, num_experts)` 内。`routes` 是"每个 token 的 top-k 专家列表"，所以 `total_tokens = len(routes)`、`total_routes = sum(len(r) for r in routes)`，通常 `total_routes = k × total_tokens`。

容量公式是

$$C=\left\lceil c\frac{kT}{E}\right\rceil,\qquad O=\sum_{e=1}^{E}\max(0,n_e-C).$$

$T$ 是 token 数，$k$ 是 top-k，$E$ 是专家数，$c$ 是 `capacity_factor`，$n_e$ 是专家 $e$ 实际收到的路由数，$C$ 是单专家容量，$O$ 是总溢出路由数。实现只有两行：

```python
capacity = math.ceil(average_load * capacity_factor) if total_routes else 0
overflow = sum(max(load - capacity, 0) for load in expert_loads)
```

其中 `average_load = total_routes / num_experts`。**为什么用 `ceil`？** 容量必须是整数个槽位，2.5 个槽位没有意义；向上取整保证平均负载本身不会被截断。**为什么空路由时单独返回 0？** 因为 `0 / E = 0`，`ceil(0)=0` 其实也对，但显式判断让"没有路由"和"容量算出来是 0"在语义上更清楚。

`logical_route_bytes = total_routes * hidden_size * bytes_per_elem` 是"所有路由的表示总量"，属于逻辑工作量；它不等于网络传输量，因为本地路由不需要过网。

用手算测试：`routes=[[0,1],[1,2],[0,2]]`、3 个专家、hidden 4、2 字节。总路由 6，平均负载 2，`capacity=ceil(2×1.0)=2`，负载 `[2,2,2]`，溢出 0，逻辑字节 `6×4×2=48`，`balance_ratio = max(2)/2 = 1.0`。

### 第 2 步：区分本地与跨 rank（TODO 2）
`summarize_dispatch_locality(routes, token_source_ranks, expert_owner_ranks, hidden_size, bytes_per_elem=2)` 里，`token_source_ranks[i]` 是第 i 个 token 当前所在的 rank，`expert_owner_ranks[e]` 是专家 e 被放在哪个 rank。核心判断：

```python
is_local = source_rank == owner_rank
local_routes += int(is_local)
cross_rank_routes += int(not is_local)
```

`total_routes = local_routes + cross_rank_routes` 必须守恒。字节账本：

- `cross_rank_dispatch_bytes = cross_rank_routes × hidden_size × bytes_per_elem`：去程，把 token 送到专家所在 rank；
- `cross_rank_roundtrip_bytes = 去程 × 2`：combine 时专家输出还要送回原 rank，所以往返是两倍；
- `local_hit_ratio = local_routes / total_routes`。

测试：`routes=[[0],[1],[0,1]]`、来源 `[0,1,0]`、owner `{0:0, 1:1}`。逐条看：token0 的专家 0 在 rank0，本地；token1 的专家 1 在 rank1，本地；token2 的专家 0 在 rank0（本地）、专家 1 在 rank1（跨 rank）。于是 `local_routes=3`、`cross_rank_routes=1`、去程 `1×8×2=16`、往返 32、命中率 `3/4=0.75`。

**为什么专家放置只改变跨 rank 比例？** 因为路由分布（谁被选中）和逻辑计算量（每个专家算多少 token）没变，变的只是"这些计算发生在哪个 rank、要不要过网"。所以 Step 3 的表把"路由分布 / 容量上限 / 专家放置"列为三个独立变量。

### 第 3 步：联合决策（TODO 3）
`recommend_moe_parallel_plan` 先算两个比例：

```python
overflow_ratio = summary.overflow / summary.total_routes
cross_rank_ratio = locality.cross_rank_routes / locality.total_routes  # 没有 locality 时为 0
```

然后两级判断：

```python
accept_ready = (
    summary.overflow == 0
    and summary.balance_ratio <= max_accept_balance      # 1.2
    and cross_rank_ratio <= max_accept_cross_rank)       # 0.5
tune_ready = (
    overflow_ratio <= max_tune_overflow_ratio            # 0.25
    and summary.balance_ratio <= max_tune_balance        # 1.8
    and cross_rank_ratio <= max_tune_cross_rank)         # 0.9
```

`accept` 要求**零溢出**，是硬条件；`tune` 允许最多 25% 的路由溢出、负载比到 1.8、跨 rank 比例到 0.9；超过任一条就是 `reject`。`summary.total_routes == 0` 时直接 `reject`，因为没有任何可评估的工作负载。

把测试逐个算一遍：
- `balanced = summarize_expert_parallel([[0],[1]], 2, 8)`：2 条路由、平均 1、容量 1、负载 `[1,1]`、溢出 0、负载比 1.0；无 locality 时跨 rank 比例 0 → `accept`。
- `tuned = summarize_expert_parallel([[0],[0],[0],[1]], 2, 8)`：4 条路由、平均 2、容量 2、负载 `[3,1]`、溢出 1；`overflow_ratio=1/4=0.25` 刚好等于阈值，`balance_ratio=3/2=1.5` → `tune`。
- `rejected = summarize_expert_parallel([[0]]*10 + [[1]], 2, 8)`：11 条路由、平均 5.5、容量 `ceil(5.5)=6`、负载 `[10,1]`、溢出 4；`4/11≈0.364 > 0.25`，且 `10/5.5≈1.818 > 1.8` → `reject`。
- `high_cross_rank = DispatchLocalitySummary(10, 2, 8, 128, 256, 0.2)`：`cross_rank_ratio = 8/10 = 0.8`，超过 accept 的 0.5、但在 tune 的 0.9 内；溢出 0 → `tune`。注意这里用的是 `cross_rank_routes / total_routes`，不是构造时传的 `local_hit_ratio=0.2`。

### 第 4 步：combine 为什么用 `index_add_`
`combine_expert_outputs(token_indices, expert_outputs, routing_weights, num_tokens)` 是 dispatch 的反向：同一个 token 的 top-k 专家输出先乘各自路由权重，再累加回该 token 的位置。

```python
combined = torch.zeros(num_tokens, expert_outputs.size(-1), ...)
combined.index_add_(0, token_indices, expert_outputs * routing_weights.unsqueeze(-1))
```

`index_add_` 对 `token_indices` 中相同的索引做累加，正好对应"一个 token 有多个专家输出"。测试里 `token_indices=[0,0,1]`、输出 `[[1,0],[3,0],[0,2]]`、权重 `[0.25,0.75,1.0]`：第 0 行 = 0.25×[1,0] + 0.75×[3,0] = [2.5, 0]；第 1 行 = 1.0×[0,2] = [0,2]。如果用普通赋值 `combined[token_indices] = ...`，后一个专家会覆盖前一个，结果就错了。

### 第 5 步：框架核对与 GPU 实验
框架核对 `check_expert_dispatch_framework_conformance` 用两个本地 Gloo 进程跑 `dist.all_to_all_single`。payload 是 `[rank*10 + i for i in range(4)]`，每个 rank 4 个元素、切成 2 份：rank0 发 `[0,1]` 给 rank0、`[2,3]` 给 rank1；rank1 发 `[10,11]` 给 rank0、`[12,13]` 给 rank1。所以 rank0 收到 `[0,1,10,11]`、rank1 收到 `[2,3,12,13]`。检查三项：分桶结果与 `chunk` 期望一致、全局元素和守恒（6+46=52）、各 rank 输出长度相等。它验证的是 all-to-all 语义，不含真实专家 kernel、NCCL 带宽或 GPU 吞吐。

Step 5 的 GPU 基准固定 `WORLD_SIZE=2`、`NUM_EXPERTS=4`、`TOP_K=2`、`CAPACITY_FACTOR=1.25`、每 rank 256 token、hidden 1024、BF16，跑 `local_heavy / balanced / skewed` 三种路由，warmup 5、repeats 20。按公式手算这组配置：总 token 512、总路由 1024、平均每专家 256、容量 `ceil(1.25×256)=320`。均衡路由下每个专家 256 条，溢出 0；偏斜路由若把超过 320 条压到某个专家，多出的部分就是 overflow。每条跨 rank 路由单向 1024×2=2048 字节，往返 4096 字节。

读结果时按 5.4 的表：`failure` 非空先修环境；local-heavy 的跨 rank 字节下降说明放置改善了局部性，但要回头检查有没有引入新的负载偏斜；skewed 的 overflow、最慢 rank 时间或负载离散度升高，说明热点进入了关键路径。三组字段完整且重复稳定，才算拿到 EP 机制级证据，之后再去 80 号项目页复测真实模型吞吐。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Expert Parallel（EP） | 把不同专家放到不同 rank/GPU 上 |
| top-k 路由 | 每个 token 选 k 个专家，产生 k 条路由 |
| `capacity_factor` | 单专家容量相对平均负载的放大系数 |
| 容量 `C` | 每个专家最多接收多少条路由 |
| overflow | 超过容量、被丢掉的路由数 |
| dispatch | 把 token 送到专家所在 rank 的去程 |
| combine | 把专家输出按 token 顺序加权合并的回程 |
| all-to-all | 每个进程向所有进程发送不同分块的集合通信 |
| 本地命中率 | 不需要跨 rank 的路由占比 |
| 负载偏斜 `balance_ratio` | 最忙专家负载 ÷ 平均负载 |
| 逻辑路由字节 | 所有路由的表示总量，不等于网络传输量 |
| 往返字节 | dispatch 去程 + combine 回程 |

## 小白常见疑问
**Q1：容量公式为什么是 `ceil(c × kT/E)`？直接用 `T/E` 不行吗？**
`kT` 是总路由数，除以专家数 `E` 得到"平均每个专家该收多少"，再乘 `capacity_factor` 留出缓冲。直接用 `T/E` 会漏掉 top-k 的倍数：top-2 时总路由是 2T，用 T/E 会把容量算小一半，导致大量正常路由被当成 overflow。测试里 6 条路由、3 个专家，平均正好 2，容量也是 2。

**Q2：overflow 能不能按 token 数算？**
不建议。一个 token 路由到两个专家，可能只有其中一个专家的槽位满了，另一个仍能计算；这个 token 只损失了一部分 expert 贡献。notebook 的 `overflow = sum(max(load - capacity, 0))` 统计的是**路由条数**，和 `total_routes` 同口径，后面的 `overflow_ratio` 也才有意义。

**Q3：local-heavy 路由让跨 rank 字节下降，为什么还要看负载偏斜？**
因为"本地"是按专家放置算的，不是按负载算的。如果某个 rank 上刚好放了很多热门专家，路由当然本地命中高、通信少，但这个 rank 的计算会变成瓶颈，其他 rank 空闲。所以 Step 3 的表把"本地命中率、跨卡 bytes、负载是否反而偏斜"放在一起观察，决策函数也同时检查 `balance_ratio` 和 `cross_rank_ratio`。

**Q4：`high_cross_rank` 里 `local_hit_ratio=0.2`，为什么决策函数算出的跨 rank 比例是 0.8？**
因为 `recommend_moe_parallel_plan` 不看 `local_hit_ratio` 字段，而是重新用 `locality.cross_rank_routes / locality.total_routes` 计算：8/10=0.8。字段只是用来展示的摘要，决策以原始计数为准。这也提醒我们：改摘要字段不会影响判定。

**Q5：既然每个 token 只算 top-k 个专家，为什么还要 `combine` 权重？**
路由权重表示这个 token 对该专家的信任程度。top-k 输出必须先乘权重再相加，才能保持"各专家贡献按比例混合"；如果直接相加，等于把每个专家都当成同等重要，和 router 的打分不一致。`index_add_` 负责在同一个 token 位置上累加这些加权输出。

## 动手验证：怎么知道自己做对了
测试单元顺序执行五个测试，最后打印 `moe expert parallel passed`（注意不是 emoji 版本）。TODO 未填时 `summarize_expert_parallel` 等函数会抛 `NotImplementedError`。
- `test_moe_load_capacity_contract`：`summarize_expert_parallel([[0,1],[1,2],[0,2]], 3, 4)` 必须得到 `total_tokens=3`、`total_routes=6`、`expert_loads=[2,2,2]`、`capacity=2`、`overflow=0`、`logical_route_bytes=48`、`balance_ratio=1.0`。
- `test_moe_empty_and_invalid_contract`：空路由时 `total_routes=0`、`capacity=0`、`overflow=0`；`([[2]], 2, 8)`、`([[0]], 0, 8)`、`([[0]], 2, 0)` 三组非法配置都抛 `ValueError`。
- `test_moe_dispatch_locality_contract`：4 条路由中 3 本地、1 跨 rank，`cross_rank_dispatch_bytes=16`、`cross_rank_roundtrip_bytes=32`、`local_hit_ratio=0.75`。
- `test_moe_decision_contract`：均衡配置 `accept`，均衡 + 高跨 rank 比例 `tune`，轻度不均衡 `tune`，严重热点 `reject`。
- `test_moe_combine_contract`：`index_add_` 的结果必须等于 `[[2.5, 0.0], [0.0, 2.0]]`。

框架核对 `check_expert_dispatch_framework_conformance` 的门槛是 `dispatch_matches_expected`、`global_element_conservation`、`equal_output_sizes` 三项全真，`world_size=2`。dump 里 Step 5 是 `RUN_MODE='dry_run'`，只打印配置、不启动 torchrun，也没有保存 benchmark JSON；GPU 数字要等你改成 `real_gpu` 并在至少两张卡上跑完后，从 `benchmarks/results/46_expert_parallel/<RUN_LABEL>/` 下读取。
