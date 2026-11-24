# 79. Distributed Parallel Benchmark | 训练状态并行基准 · 小白入门

> 配套 notebook：[79. 训练状态并行基准](../79_Distributed_Parallel_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

用多张卡训练时，参数、梯度和优化器状态可以每张卡都存一份，也可以切开分着存。这本 notebook 教你算每种存法每张卡占多少，再用显存、吞吐和通信三个门槛选出一种。

## 为什么需要它

模型大到一张卡放不下时，就用多张卡一起训练。训练时每张卡的显存里都要放着三样东西：参数、梯度、优化器状态。怎么放，有两种思路。一种是复制：每个 rank 都存一份完整的，最基础的 DDP 就是这样，每张卡各自算，再把梯度同步。另一种是分片：把三样东西切开，每个 rank 只存自己那一片。ZeRO 和 FSDP 走这条路，全局进程数（world size）越大，每个 rank 常驻的状态越少。

分片看起来省显存，但比较这些策略时容易犯三种错：

- 只看常驻的状态。账本上分片后每张卡只剩四分之一，可前向传播前要把参数临时拼回完整的，还有激活值和通信缓冲区，峰值显存并不等于账本上的数。
- 认为显存最小的策略最好。分片更深的策略可能要用频繁的聚合来换容量，吞吐未必更高。
- 只给出“行”或“不行”。候选被拒绝时只有一个 `False`，分不清是显存超了、吞吐不够，还是运行时显存溢出。

所以要先算清常驻状态，再用门槛筛选，最后在合格候选里按吞吐、通信占比、峰值显存的顺序选。这本 notebook 让你把这三步各写成一个函数。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| rank | 多卡训练里的一个成员，本节可以理解成其中一张卡。“每 rank 驻留量”就是每张卡上一直放着的状态有多大 |
| 全局进程数（world size） | 一共有多少个 rank 一起训练。它是 4 时，分片的状态就切成 4 份 |
| 训练状态 | 训练时要放在显存里的三样东西：参数、梯度、优化器状态 |
| 复制 / 分片 | 复制是每个 rank 都存完整的一份；分片是切开，每个 rank 只存自己那一片 |
| 账本 | 本节指一张记录表：某个策略下，参数、梯度、优化器状态在每张卡上各占多少 |
| 吞吐 | 单位时间里训练了多少数据，越高越好 |
| 通信占比 | 训练时间里花在卡与卡之间传数据上的比例，本节给它设了上限 |

## 核心思路

输入分两部分。一部分是三样状态各有多少 MB，以及一共有多少个 rank。另一部分是每种策略的测量记录：峰值显存 `peak_memory_mb`、吞吐 `throughput`、通信占比 `communication_ratio`，还有记录运行是否失败的 `failure`。

处理分三步，输出是一个决策：`accept` 并给出选中的策略，或者 `reject`。

第一步算账本。每种策略对三样状态各有“切成几份”，查表得到份数，用大小除以份数，再相加：

```text
每 rank 驻留量 = 参数大小 / 参数份数 + 梯度大小 / 梯度份数 + 优化器状态大小 / 优化器状态份数

策略                      参数份数    梯度份数    优化器状态份数
DDP                       1           1           1
ZeRO-1                    1           1           world size
ZeRO-2                    1           world size world size
ZeRO-3 / FSDP full shard  world size  world size  world size
```

份数表来自 notebook，顺序是参数、梯度、优化器状态。notebook 的表里还有一列“关键通信”，列的是每种策略要多做的卡间数据交换，比如全归约（all-reduce），本节不要求实现它们。

第二步准入。四道门槛逐项检查：`failure` 不为空记 `runtime`，峰值显存超过上限记 `memory`，吞吐低于下限记 `throughput`，通信占比超过上限记 `communication`。失败原因全部留下，一个都没有才算合格。

第三步选型。只在合格候选里选，先比吞吐，高的优先；相同再比通信占比，低的优先；再相同比峰值显存，低的优先。

用 notebook 测试里的数字试一下：

```python
shard = {'ddp': (1, 1, 1), 'zero1': (1, 1, 4), 'zero2': (1, 4, 4), 'zero3': (4, 4, 4)}   # 4 个 rank
for name, (p, g, o) in shard.items():
    print(name, 400 / p + 400 / g + 800 / o)   # 参数 400 MB、梯度 400 MB、优化器状态 800 MB

cands = [
    {'strategy': 'ddp',   'eligible': False, 'throughput': 130, 'communication_ratio': 0.12, 'peak_memory_mb': 15000},
    {'strategy': 'zero2', 'eligible': True,  'throughput': 112, 'communication_ratio': 0.20, 'peak_memory_mb': 9000},
    {'strategy': 'zero3', 'eligible': True,  'throughput': 105, 'communication_ratio': 0.31, 'peak_memory_mb': 7000},
]
eligible = [c for c in cands if c['eligible']]
best = sorted(eligible, key=lambda c: (-c['throughput'], c['communication_ratio'], c['peak_memory_mb']))
print(best[0]['strategy'])
```

```text
ddp 1600.0
zero1 1000.0
zero2 700.0
zero3 400.0
zero2
```

4 个 rank 时，DDP 每张卡常驻 1600 MB，ZeRO-3 只有 400 MB。但账本只描述常驻状态，前向传播前的参数聚合、激活值和通信缓冲区仍会参与峰值。选型部分里 DDP 的吞吐最高，可它没通过门槛；合格的两条里 `zero2` 吞吐更高，所以选它，尽管 `zero3` 的峰值显存更低。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从复制状态走向分片状态：DDP、ZeRO-1/2/3 和 FSDP 各分片哪一类状态，关键通信是什么 | 读，对照上面的份数表 |
| Step 2 | 沿训练生命周期识别峰值与通信：加载、前向传播和反向传播、优化器更新、保存 checkpoint 四个位置各要看什么代价 | 读 |
| Step 3 | 用门槛筛选训练状态策略：先过显存上限，再比吞吐和通信占比，显存最低不一定最好 | 读 |
| Step 4 · TODO 1 | 训练状态分片机制——分别计算参数、梯度和优化器状态的每 rank 驻留量 | 在 `training_state_ledger` 里写 4 行：从 `shard_factors` 取出三个份数，三项大小各除以份数 |
| Step 4 · TODO 2 | 候选准入机制——逐项检查 failure、显存、吞吐与通信占比 | 在 `evaluate_candidate` 里补 `failures`：四道门槛各追加一个失败原因 |
| Step 4 · TODO 3 | 策略选择机制——优先吞吐，其次通信占比，再比较峰值显存 | 在 `select_training_state_strategy` 里补 `selected`：对合格候选按这个顺序排序，取第一个 |
| 测试 | 检查账本的数值、非法 `world_size` 和未知策略是否报错、门槛的临界值和多个失败原因、选型顺序和没有候选时的结果 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在多卡上实测 DDP 和 ZeRO 各阶段的每 rank 显存、步时、吞吐和通信证据 | 需要 CUDA、至少 2 张 GPU、NCCL 和 `torchrun`；默认是 `RUN_GPU_EXPERIMENT = False`，关闭时只打印配置，不运行实验 |

Step 1–4 只需要 CPU。Step 5 需要多张 GPU，默认关闭。

## 容易踩的坑

- **把账本当成了峰值显存**：账本只算常驻的参数、梯度和优化器状态。临时的参数聚合、激活值和通信缓冲区仍会参与峰值，不能当成已经消失。
- **门槛的比较方向写反**：峰值显存和通信占比是超过上限才失败，吞吐是低于下限才失败，正好等于门槛值算通过。
- **只返回一个布尔值**：候选没通过时要把所有失败原因都留下来，这样才能分清是显存超限、吞吐不足、通信占比过高，还是运行时显存溢出。
- **让不合格的候选参与排序**：先排除没通过门槛的候选。测试里 DDP 的吞吐最高，但它不合格，最后选中的是 `zero2`。
- **把当前排序顺序当成普遍规律**：先吞吐、再通信占比、再峰值显存，表达的是这个项目的偏好。目标变了要改排序规则，不能去改测量结果。

## 读完后自查

- [ ] 能说出 DDP、ZeRO-1、ZeRO-2、ZeRO-3 各分片哪几类训练状态。
- [ ] 参数、梯度、优化器状态分别是 400、400、800 MB，4 个 rank 时，能算出 ZeRO-2 的每 rank 驻留量。
- [ ] 能说出候选准入的四道门槛，以及没通过时要留下什么。
- [ ] 能说出选型时比较三项指标的先后顺序。

## 下一步

- 打开 notebook：[79. 训练状态并行基准](../79_Distributed_Parallel_Benchmark.md)
- 建议先读：[27. ZeRO 优化器模拟 · 小白入门](./27_ZeRO_Optimizer_Sim.md)
- 需要补基础：[通信拓扑与分布式基石](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)
- 做完之后：[PAR-MODEL-INTERNAL. 模型内部并行基准 · 小白入门](./PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md)
