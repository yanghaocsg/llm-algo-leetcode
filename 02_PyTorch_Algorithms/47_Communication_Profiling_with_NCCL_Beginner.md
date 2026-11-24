# 47. Communication Profiling with NCCL Beginner | NCCL 通信剖析 · 小白入门

> 配套 notebook：[47. NCCL 通信剖析](./47_Communication_Profiling_with_NCCL.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

多张显卡一起训练时要互相传数据，等数据的时间会拖慢每一步。这本 notebook 教你从一条时间线里算出：哪类通信没有被计算盖住，真正在拖时间。

## 为什么需要它

多卡训练里，ZeRO、流水线并行、张量并行都会让多个 rank 协同完成一次训练或推理，每一步都要互相传数据。只看每张卡的利用率判断不了快慢：有的 rank 在等通信结束，有的 rank 还在算。通信到底拖了多少时间，要把通信和计算排到同一条时间线上才看得出来。

看 notebook 测试里的一段时间线：一段 `all_reduce` 从第 1 毫秒传到第 2.5 毫秒，一共 1.5 毫秒；计算只占着 0 到 2 毫秒，所以有 1 毫秒被计算盖住，剩下 0.5 毫秒里计算没活可干，在专门等通信。

读这条时间线时容易出三种错：

- 只记“有没有重叠”。上面这段通信的答案是“有”，但真正拖慢一步的是没盖住的 0.5 毫秒，一个“是”把它丢了。
- 几段计算区间自己也有重叠时，把交集直接相加，同一段通信被数了两遍。
- 按通信总时长挑最该处理的对象。总时长长的通信可能大部分被计算盖住了，并不占关键路径的时间。

这本 notebook 要做的是：从一条时间线算出每类集体通信没有被盖住的时间，并把最大的那个挑出来，作为优先检查的热点。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| rank | 多卡协同干活时，每个参与方的编号 |
| 集体通信（collective） | 必须多个 rank 一起参加的通信，比如全归约、广播 |
| 全归约（all-reduce） | 所有 rank 把数据汇总，每个 rank 都拿到相同的结果，比如汇总后的梯度 |
| payload | 一次通信要搬的数据量，单位是字节 |
| 关键路径（critical path） | 决定一步训练何时结束的那条时间链，落在它上面的时间会直接拖长这一步 |
| 有效重叠时间（overlap）/ 暴露通信时间（exposed time） | 通信时长里落在计算区间内的部分是前者，没被盖住的部分是后者，后者用来决定先检查哪个集体通信 |

## 核心思路

输入是同一条时间线上的两类区间：计算区间有名字、开始、结束，通信区间还多一个集体通信的名字和传输的字节数。处理分四步：用公式估算一类通信的成本；对每段通信算出被计算盖住了多久；用时长减去盖住的部分，得到暴露时间；按类型汇总，挑出暴露时间最大的一个。输出是热点的名字、暴露时间和字节数。

成本估算用的是 notebook 里的近似公式：

```text
single_collective_time ≈ startup_latency + payload_bytes / effective_bandwidth
total_communication_time ≈ sync_count × single_collective_time
```

`startup_latency` 是启动一次集体通信的固定成本，`payload_bytes` 是要搬的字节数，`effective_bandwidth` 是单位时间能搬的字节数（已经包含拓扑、竞争和实现效率的影响），`sync_count` 是同一种通信重复了几次。小 payload 更容易被启动成本主导，大 payload 更容易被带宽主导；单次不慢但同步次数太多，加起来同样会占用关键路径。

时间线这部分有三个量：通信时长、有效重叠时间、暴露时间，关系是 `暴露时间 = 通信时长 - 有效重叠时间`。有效重叠时间在通信总时长里的占比叫重叠比例。下面是最小例子，数字取自 notebook 的测试：

```python
compute = [(0.0, 2.0), (3.0, 5.0)]      # 两段计算区间（开始，结束），单位毫秒
comms = [('all_reduce', 1.0, 2.5), ('broadcast', 2.6, 3.0),
         ('reduce_scatter', 3.5, 4.3), ('all_reduce', 4.5, 5.5)]

def covered(s, e):                       # 通信区间 [s, e] 被计算盖住的长度
    cuts = sorted((max(s, a), min(e, b)) for a, b in compute if max(s, a) < min(e, b))
    merged = []
    for a, b in cuts:                    # 和上一段相交就并进去
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return round(sum((b - a for a, b in merged), 0.0), 1)

by_op = {}
for op, s, e in comms:
    dur, hidden = round(e - s, 1), covered(s, e)
    print(op, dur, hidden, dur - hidden)       # 时长、被盖住、暴露
    item = by_op.setdefault(op, {'count': 0, 'time': 0.0, 'exposed_time': 0.0})
    item['count'] += 1
    item['time'] += dur
    item['exposed_time'] += dur - hidden
for op, item in by_op.items():
    print(op, item)
compute = [(0.0, 2.0), (1.0, 3.0)]      # 换成两段自己也有重叠的计算区间
naive = sum(min(2.5, b) - max(0.5, a) for a, b in compute if max(0.5, a) < min(2.5, b))
print(round(naive, 1), covered(0.5, 2.5))       # 交集直接相加 对比 合并之后
```

```text
all_reduce 1.5 1.0 0.5
broadcast 0.4 0.0 0.4
reduce_scatter 0.8 0.8 0.0
all_reduce 1.0 0.5 0.5
all_reduce {'count': 2, 'time': 2.5, 'exposed_time': 1.0}
broadcast {'count': 1, 'time': 0.4, 'exposed_time': 0.4}
reduce_scatter {'count': 1, 'time': 0.8, 'exposed_time': 0.0}
3.0 2.0
```

前四行是每段通信的时长、被盖住的时长、暴露时间：`broadcast` 落在两段计算之间，全部暴露；`reduce_scatter` 完全落在计算里，暴露时间是 0。中间三行是按类型汇总的结果，`all_reduce` 出现 2 次，暴露时间 1.0 毫秒，是三类里最大的，所以它是优先热点。最后一行说明为什么要合并：两段计算 0 到 2 和 1 到 3 自己就重叠，通信 0.5 到 2.5 只有 2.0 毫秒长，交集直接相加却得到 3.0；先合并再求长度，得到的 2.0 才是对的。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 通信为什么会进入关键路径：只看 GPU 利用率判断不了，以及一次集体通信的成本由哪几部分组成 | 读 |
| Step 2 | 通信时间线需要记录什么：计算区间、通信区间、执行上下文、聚合指标四类字段 | 读 |
| Step 3 | 时间重叠、暴露通信时间与热点归因：四个概念，以及为什么要先合并计算区间再算 | 读，对照上面的小例子 |
| Step 4 · TODO 1 | 计算启动成本与 payload / bandwidth 的传输成本，并乘同步次数 | 写 `estimate_collective_time`：2 行，先算一次的时间，再乘 `sync_count` |
| Step 4 · TODO 2 | 合并重叠的交集区间，得到有效 overlap 时长 | 写 `_overlap_duration`：把排好序的交集区间合并，返回合并后的总长度 |
| Step 4 · TODO 3 | 按 collective 汇总 count、time、bytes 和 exposed_time | 写 `summarize`：用 `setdefault` 取出统计桶，再累加四个字段 |
| Step 4 · TODO 4 | 选择 exposed_time 最大的 collective，并返回 op、exposed_time 与 bytes | 写 `find_priority_hotspot`：2 行，从 `summary['by_op']` 选最大项并返回三个字段 |
| 测试 | 成本模型、时间线、归因和边界条件分别验证，比如零同步、非法带宽、多段计算盖住同一段通信 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实多卡上比较不同集体通信、payload 和算法（Auto、Ring、Tree）下的时间与有效带宽 | 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验。

## 容易踩的坑

- **只记“有没有重叠”**：`overlap_with_compute` 只是个是非值，说明不了拖了多久。要算的是通信区间和计算区间的交集长度，暴露时间等于时长减去它。
- **几段计算区间重复计数**：计算区间自己重叠时，交集直接相加会把同一段通信数两遍。先把相交的交集区间合并，再求总长度。
- **按总时长挑热点**：总时长大的集体通信不一定最拖时间。`find_priority_hotspot` 要选 `exposed_time` 最大的，不是 `time` 最大的。
- **非法输入没有拦住**：`payload_bytes`、`startup_latency`、`sync_count` 必须非负，`bandwidth` 必须为正，否则要抛出 `ValueError`。测试会专门传一个 0 带宽进来。
- **单位不统一**：本节事件的 `start / end / duration` 都是毫秒，接入真实的时间线追踪记录时必须先统一单位。

## 读完后自查

- [ ] 能说出一次集体通信的时间由哪两部分组成，`sync_count` 起什么作用。
- [ ] 给一段通信区间和几段计算区间，能算出它的有效重叠时间和暴露时间。
- [ ] 能解释为什么计算区间自己有重叠时，不能把交集直接相加。
- [ ] 能说明优先热点为什么按暴露时间选，而不是按通信总时长选。

## 下一步

- 打开 notebook：[47. NCCL 通信剖析](./47_Communication_Profiling_with_NCCL.ipynb)
- 建议先读：[29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim_Beginner.md)
- 需要补基础：[通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)、[NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)
- 做完之后：[48. 通信热点与缓解策略](./48_Communication_Hotspots_and_Mitigation.ipynb)、[PAR-REPLICA. Serving 副本与请求路由 · 小白入门](./PAR-REPLICA_Serving_Replica_Routing_Beginner.md)
