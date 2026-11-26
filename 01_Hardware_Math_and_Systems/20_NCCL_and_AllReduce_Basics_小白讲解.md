# 20. NCCL 与 AllReduce 基础 — 小白讲解

> 对应 notebook：[`20_NCCL_and_AllReduce_Basics.ipynb`](./20_NCCL_and_AllReduce_Basics.ipynb)

## 一句话总结
这一节用四个 CPU 小工具建立“局部梯度 → All-Reduce → 环形两阶段 → 延迟与带宽成本”的完整链条：先算通信有多少能被计算藏住，再验证 AllReduce 的输出不变量，然后用 $2(P-1)S/P$ 估算 8 卡各发 448 MiB、耗时 4.515 ms，最后用 4×4 的 chunk 矩阵逐步模拟 Ring Reduce-Scatter 与 All-Gather 的归约与所有权变化。

## 为什么需要它：痛点是什么
多卡训练时，“加卡就该变快”是最容易落空的期待。8 张卡并行算梯度，算完还要把 8 份局部结果合成一份一致的全局结果；如果每张卡都先把完整张量发给所有卡，通信量会随卡数平方增长，加卡反而可能更慢。初学者还容易在单位上翻车：把 256 MiB 当成 256 MB、把 GiB/s 当成 GB/s、把“发送 + 接收”加总后除以单向带宽，算出来的时间能差 7% 甚至更多。

生活类比：8 个小组各有一张完整的成绩表，要合成一张“每科总分表”并让 8 个组都拿到。笨办法是每组把整张表复印 7 份发给别人；环形办法是大家围成一圈，每人只把表切成 8 段，一段一段往右传：先转 7 圈，每圈负责把某一段的分数累加齐；再转 7 圈，把已经算好的段传回所有人。每人每圈只递一小段，总发送量反而更可控。

## 核心问题
- 一步训练的耗时，为什么不是“计算时间 + 通信时间”简单相加？
- AllReduce 的输出形状和一致性不变量是什么？求和与求平均差在哪？
- Ring AllReduce 每个 rank 到底发送多少字节？为什么是 $2(P-1)S/P$？
- 256 MiB、8 卡、100 GiB/s 时，4.515 ms 是怎么一步步算出来的？
- Ring 的两个阶段各自在移动什么、每个 chunk 最后归谁所有？

## 解决思路：一步步看懂

### Q1：通信只有“没被藏住”的部分才加进关键路径
`training_step_time(compute_ms, collective_ms, overlap_ms=0.0)` 的模型只有三行：

```python
hidden_ms = min(overlap_ms, collective_ms, compute_ms)
exposed_ms = collective_ms - hidden_ms
return {"step_ms": compute_ms + exposed_ms,
        "hidden_collective_ms": hidden_ms,
        "exposed_collective_ms": exposed_ms}
```

$T_{\text{step}} = T_{\text{compute}} + (T_{\text{collective}} - T_{\text{hidden}})$。`hidden_ms` 取三者最小值，是因为重叠不能超过通信本身，也不能超过可用来重叠的计算。notebook 的三组输入是计算 10 ms、通信 3 ms：

- `overlap=0`：一点没藏住，`exposed=3`，步时 13 ms；
- `overlap=1.5`：藏住一半，`exposed=1.5`，步时 11.5 ms；
- `overlap=3`：通信完全被计算覆盖，`exposed=0`，步时仍是 **10 ms**，不是 13 ms。

断言检查 `reports[0]["exposed_collective_ms"] == 3.0` 和 `reports[-1]["exposed_collective_ms"] == 0.0`。这说明“通信时间”与“通信暴露时间”是两回事：前者是集合通信本身耗时，后者是真正拖长训练步的尾部等待。模型假设通信与独立计算可以重叠，真实重叠窗口还要从执行记录里看。

### Q2：AllReduce 的不变量是“形状不变、各 rank 一致”
`allreduce_sum(local_vectors)` 先校验：列表非空，且所有 rank 的向量等长；然后用 `zip(*local_vectors)` 按位置把各 rank 的分量相加，最后为每个 rank 返回一份**独立的完整副本**。

```python
reduced = [sum(items) for items in zip(*local_vectors)]
return [reduced.copy() for _ in local_vectors]
```

用四组局部梯度 `[[1,2], [10,20], [100,200], [1000,2000]]` 验证：第 0 列 $1+10+100+1000=1111$，第 1 列 $2+20+200+2000=2222$，四个 rank 都拿到 `[[1111.0, 2222.0], ...]`。这抓住了 AllReduce 的两个不变量：**输出形状与局部输入相同；所有 rank 的输出完全一致**。

数据并行常用它聚合梯度。求和与求平均只差一个“除以 world_size”的缩放，但这一步必须和损失函数的归约方式、优化器语义对齐；否则等价学习率会悄悄变掉。notebook 只实现求和，就是为了不让缩放混进原语本身。

### Q3：Ring 的发送量是 $2(P-1)S/P$，不是 $2S$
`ring_cost(payload_bytes, world_size, alpha_ms, bandwidth_gib_s)` 的输入是每 rank 完整载荷 $S$、进程数 $P$、每轮启动时延 $\alpha$、单向带宽。核心公式：

```python
rounds = 2 * (world_size - 1)                       # Reduce-Scatter + All-Gather 各 P-1 轮
send_bytes = rounds * (payload_bytes // world_size) # 每轮发 S/P
latency_ms = rounds * alpha_ms
transfer_ms = send_bytes / (bandwidth_gib_s * 2**30) * 1000
```

推导一下：每个阶段有 $P-1$ 轮，每轮每个 rank 把 $S/P$ 字节递给右邻居，所以单 rank 单阶段发送 $(P-1)S/P$；两个阶段合计：

$$\text{send per rank} = \frac{2(P-1)S}{P}$$

接收量相同，收发合计是它的 2 倍。**估算单向传输时间时只用发送量**，因为发送和接收可以同时发生，不能把收发加起来再除以同一个单向带宽。

代入 notebook 的配置：`PAYLOAD_BYTES = 256 * 1024 * 1024`，即 $S = 256 \text{ MiB} = 268{,}435{,}456$ 字节，$P=8$，$\alpha=0.01$ ms，带宽 100 GiB/s。

- 轮数：$2×(8-1)=14$；
- 每轮发送：$256/8=32$ MiB；
- 单 rank 发送：$14×32=448$ MiB，即 `448 * 2**20 = 469,762,048` 字节；接收也是 448 MiB；收发合计 896 MiB；
- 传输时间：$448 \text{ MiB} \div 1024 = 0.4375$ GiB，再除以 100 GiB/s 得 0.004375 s = **4.375 ms**；
- 启动时延：$14×0.01=0.14$ ms；
- 合计：$4.375+0.14=4.515$ ms。

把带宽换成 12.5 GiB/s，传输时间变成 $0.4375/12.5×1000=35$ ms，加 0.14 ms 得 35.14 ms。断言用 `abs(total_ms - 4.515) < 1e-12` 和 `abs(total_ms - 35.14) < 1e-12`，精度要求很高，说明公式里的 $2^{30}$ 不能换成 $10^9$：GiB 是二进制单位，$1 \text{ GiB}=2^{30}$ 字节。

另外，`ring_cost` 要求 `payload_bytes % world_size == 0`，因为等长分片要求载荷能被进程数整除；还要求 `world_size >= 2`，否则没有“环”可言。

### Q4：用 4×4 chunk 矩阵把两个阶段走一遍
`ring_allreduce(local_chunks)` 接收一个 $P×P$ 的矩阵：第 $i$ 行是 rank $i$ 的输入，第 $j$ 列是第 $j$ 个 chunk。它用同步快照模拟，避免“本轮读到本轮刚写的数据”。

**Reduce-Scatter 阶段**跑 $P-1$ 轮。每轮每个 rank 从 `(rank - 1) % P` 收数据，处理的 chunk 是 `(rank - step - 1) % P`：

```python
source = (rank - 1) % world_size
chunk = (rank - step - 1) % world_size
partial[rank][chunk] += snapshot[source][chunk]
```

转完 $P-1$ 轮后，每个 rank 有一个**完全归约好**的 chunk，归属是 `(rank + 1) % world_size`。用 4 个 rank、`local_chunks[rank][chunk] = rank*10 + chunk` 代入，输入矩阵是：

```
rank 0: [0, 1, 2, 3]
rank 1: [10, 11, 12, 13]
rank 2: [20, 21, 22, 23]
rank 3: [30, 31, 32, 33]
```

每个 chunk 的全局和是 $0+10+20+30=60$、$1+11+21+31=64$、$68$、$72$。Reduce-Scatter 结束后，`owned_after_reduce_scatter` 正是 `{0: 60, 1: 64, 2: 68, 3: 72}`，但键是 chunk 编号、不是 rank 编号：chunk 0 的 60 由 rank 3 持有，chunk 1 的 64 由 rank 0 持有，chunk 2 由 rank 1，chunk 3 由 rank 2，即 rank $r$ 持有 chunk $(r+1) \bmod P$。

**All-Gather 阶段**再跑 $P-1$ 轮，每轮从 `(rank - 1) % P` 收 chunk `(rank - step) % P`，把已归约的 chunk 传播给所有 rank。最后每个 rank 的 `outputs` 行都等于 `[60, 64, 68, 72]`，`trace` 总长度是 $2(P-1)=6$（两个阶段各 3 轮），每个 `step` 记录 4 条 `{'from','to','chunk'}` 传输。

代码在返回前自己做了断言：

```python
expected = [sum(local_chunks[rank][chunk] for rank in range(world_size))
            for chunk in range(world_size)]
assert all(row == expected for row in outputs)
```

也就是说，无论中间怎么转，最终每个 rank 都必须拿到逐 chunk 的全局和。这是对算法状态机的正确性检查，不是性能测量。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 集合通信（collective） | 一组进程一起完成的通信，如 AllReduce、AllGather |
| rank / world_size | 进程编号 / 参与的进程总数 |
| AllReduce | 归约所有 rank 的输入，把相同完整结果发给每个 rank |
| Reduce-Scatter | 先归约，再把结果按 chunk 分散到各 rank |
| All-Gather | 把分散在各 rank 的 chunk 拼回完整结果并发给所有人 |
| Ring（环） | 每个 rank 只和左右邻居传数据，$P$ 个 rank 围成一圈 |
| chunk | 把完整载荷切成 $P$ 份后的一份 |
| 每轮启动时延 $\alpha$ | 一轮通信固定要付的延迟，与数据量无关 |
| 带宽（GiB/s） | 每秒传多少 $2^{30}$ 字节；注意不是 $10^9$ |
| 暴露通信时间 | 没被计算藏住、真正拖长训练步的那部分 |
| 数据并行（DP） | 各卡算不同数据、用 AllReduce 聚合梯度 |
| NCCL | NVIDIA GPU 上的集合通信库，负责实现这些原语 |

## 小白常见疑问
**Q1：256 MiB 的梯度，为什么每个 rank 要发 448 MiB，而不是 256 MiB？**
因为 Ring 不是一次性把整份数据发给一个人。它分两个阶段、共 $2(P-1)=14$ 轮，每轮发 $S/P=32$ MiB，所以单 rank 发送 $14×32=448$ MiB。这个量比“每卡把完整张量发给其他 7 卡”的 $7×256=1792$ MiB 小得多，而且不随 $P$ 平方增长。

**Q2：8 张卡，训练速度能快 8 倍吗？**
不一定。Q1 的 `training_step_time` 显示，通信只有被计算藏住才不拖慢步时；没藏住的部分要加到关键路径上。此外还有启动时延、带宽争用、负载不均。加卡增加的是并行计算能力，同时也增加了要协调的数据和参与者。

**Q3：为什么 Ring 要分 Reduce-Scatter 和 All-Gather 两个阶段？**
因为一次全量聚合如果让某张卡承担全部传输，负载会很不均。Reduce-Scatter 把“归约”摊到 $P-1$ 轮、每轮只动一个 chunk，让每张卡都只负责一部分；All-Gather 再把已经算好的 chunk 转一圈。好处是每轮每卡只收发 $S/P$，负载均匀；代价是轮数变成 $2(P-1)$。

**Q4：求和和求平均，对 AllReduce 有区别吗？**
原语本身只是逐元素归约；`allreduce_sum` 实现的是求和。求平均等于求和后除以 `world_size`，这个缩放必须和损失归约、优化器语义一致，不能想当然地加在通信原语里，否则等价学习率会变。

**Q5：真实 NCCL 一定用 Ring 吗？**
不一定。`ring_allreduce` 只是教学模拟，验证的是 chunk 所有权与归约状态；notebook 明确说真实 NCCL 会根据拓扑、消息规模和配置选择 Ring、Tree 等算法，实际性能要在多卡环境测。

## 动手验证：怎么知道自己做对了
notebook 没有 `test_*` 函数，全部是模块级断言，也没有保存任何执行输出；运行时以是否抛 `AssertionError` 为准。

- Q1：`training_step_time(10.0, 3.0, 0.0)` 的 `exposed_collective_ms == 3.0`；`overlap=3.0` 时 `exposed_collective_ms == 0.0`；三组分别打印 `step_ms`、`hidden_collective_ms`、`exposed_collective_ms`。
- Q2：`allreduce_sum([[1,2],[10,20],[100,200],[1000,2000]])` 的每个 rank 输出都必须是 `[[1111.0, 2222.0], ...]`；不等长或空输入应抛 `ValueError`。
- Q3：`ring_cost(256 MiB, 8, 0.01, 100)` 的 `rounds == 14`、`send_bytes_per_rank == 448 * 2**20`、`receive_bytes_per_rank` 相同、收发合计 `896 * 2**20`、`total_ms` 与 4.515 的误差小于 `1e-12`；带宽 12.5 GiB/s 时 `total_ms` 与 35.14 的误差同样小于 `1e-12`。
- Q4：`ring_allreduce` 的 `outputs` 必须是四行 `[60, 64, 68, 72]`；`trace` 长度为 6；`owned_after_reduce_scatter` 精确等于 `{0: 60, 1: 64, 2: 68, 3: 72}`。函数内部还会用 `expected` 再断言一次每个 rank 的输出。
- 运行后会打印三组训练步时间、两个带宽下的 ring 成本字典、Reduce-Scatter 后的所有权映射、rank 0 的输出和总轮数。

断言全过，说明你已经把“通信暴露时间、AllReduce 不变量、环形发送量、两阶段 chunk 归属”这四件事对齐了。
