# 48. Communication Hotspots and Mitigation | 通信热点与缓解策略 · 小白入门

> 配套 notebook：[48. 通信热点与缓解策略](../48_Communication_Hotspots_and_Mitigation.md) ｜ 难度：Medium ｜ 环境：CPU-first

## 一句话说清

多卡程序慢在通信上时，原因可能不止一种。这本 notebook 教你把通信记录按类型汇总，找出最该处理的那一类，再按症状挑一个缓解动作，并决定下一步是接受还是继续调。

## 为什么需要它

多卡程序出现通信等待时，直觉是把单次耗时最长的那次通信当成热点去优化。但同样是“通信慢”，根因可能不同：

- 每次搬的数据太多。
- 单次并不慢，但同步太频繁，或者几乎没有被计算盖住，时间全露在外面。
- 个别的 rank 到得明显更晚，别人都在等它。

修错了地方，时间花了，等待还在。另外，同一种通信一步里会出现好几次，只看单次会低估它：两次分别暴露 8 和 6 毫秒的 `all_reduce`，加起来是 14 毫秒，才能和一次 20 毫秒的 `all_to_all` 放在一起比较。

这本 notebook 做的是三步：把证据按通信类型汇总并选出热点，按热点的症状匹配缓解动作，再给出下一轮复测的决策。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 时间线追踪记录（trace） | 性能分析工具导出的事件记录，这里每条事件描述一次集体通信：类型、payload（要搬的字节数）、暴露时间等 |
| 集体通信（collective） | 必须多个 rank 一起参加的通信，`all_reduce`、`all_to_all` 是不同的种类 |
| 暴露通信时间（exposed time） | 通信里没有被计算盖住、真正占着关键路径的时间 |
| 重叠比例（overlap ratio） | 被计算盖住的通信时间占通信总时长的比例，越低说明通信越露在外面 |
| rank 不均衡（rank imbalance） | 各 rank 到达或完成通信时间的差异，个别 rank 明显更晚就是不均衡 |
| `all_to_all` | 一种集体通信，各 rank 按目的地互相交换数据，流量大时要检查专家路由和跨 rank 的词元数 |

## 核心思路

输入是一组时间线追踪记录，每条事件描述一次集体通信，带五个字段：类型 `collective`、暴露时间 `exposed_comm_ms`、传输量 `payload_bytes`、重叠比例 `overlap_ratio`、rank 不均衡 `rank_imbalance_ratio`。处理分三步：按类型汇总，挑出暴露时间最大的一项作为优先热点；按优先热点的症状选一个缓解动作；按暴露时间的大小给出决策。输出是汇总结果、缓解动作和决策。

汇总和判断的规则如下，阈值取自 notebook 的参考答案：

```text
汇总：次数、同步次数、通信时间、暴露时间、payload 相加；重叠比例取最小值，rank 不均衡取最大值
热点：暴露时间最大的一项是优先热点
动作：rank 不均衡 > 0.20               → rebalance_ranks_or_experts
      重叠比例 < 0.30 且暴露时间 > 0   → increase_overlap_or_delay_sync
      类型是 all_to_all 且 payload > 0 → reduce_routing_traffic
      其他                            → inspect_collective_payload_and_schedule
决策：没有追踪记录 → collect_more_evidence；暴露时间 <= 5.0 → accept；否则 → tune
```

选动作的三道检查有先后顺序：rank 不均衡优先级最高，其次是露在外面的低重叠通信，最后才是 `all_to_all` 的路由流量。下面是最小例子，事件和两种症状取自 notebook 的测试：

```python
# 每条事件：(collective, 暴露时间, payload 字节数, 重叠比例, rank 不均衡)
events = [('all_reduce', 8.0, 1024, 0.60, 0.00), ('all_reduce', 6.0, 2048, 0.45, 0.05),
          ('all_to_all', 20.0, 65536, 0.60, 0.05), ('broadcast', 5.0, 512, 0.90, 0.00)]

by_collective = {}
for c, exposed, payload, overlap, imbalance in events:
    b = by_collective.setdefault(c, {'count': 0, 'exposed_comm_ms': 0.0, 'payload_bytes': 0,
                                     'overlap_ratio': 1.0, 'rank_imbalance_ratio': 0.0})
    b['count'] += 1
    b['exposed_comm_ms'] += exposed
    b['payload_bytes'] += payload
    b['overlap_ratio'] = min(b['overlap_ratio'], overlap)                  # 重叠比例保留最差值
    b['rank_imbalance_ratio'] = max(b['rank_imbalance_ratio'], imbalance)  # 不均衡保留最大值
for c, b in by_collective.items():
    print(c, b)

def mitigation(c, exposed, payload, overlap, imbalance):   # 三道检查有先后顺序
    if imbalance > 0.20:
        return 'rebalance_ranks_or_experts'
    if overlap < 0.30 and exposed > 0:
        return 'increase_overlap_or_delay_sync'
    if c == 'all_to_all' and payload > 0:
        return 'reduce_routing_traffic'
    return 'inspect_collective_payload_and_schedule'

print(mitigation('all_to_all', 20.0, 65536, 0.60, 0.05))    # 汇总出的优先热点
print(mitigation('all_reduce', 16.0, 4096, 0.10, 0.00))     # 重叠比例很低
print(mitigation('all_to_all', 12.0, 8192, 0.60, 0.35))     # rank 不均衡明显
print('tune' if 20.0 > 5.0 else 'accept')                   # 优先热点的暴露时间是 20.0
```

```text
all_reduce {'count': 2, 'exposed_comm_ms': 14.0, 'payload_bytes': 3072, 'overlap_ratio': 0.45, 'rank_imbalance_ratio': 0.05}
all_to_all {'count': 1, 'exposed_comm_ms': 20.0, 'payload_bytes': 65536, 'overlap_ratio': 0.6, 'rank_imbalance_ratio': 0.05}
broadcast {'count': 1, 'exposed_comm_ms': 5.0, 'payload_bytes': 512, 'overlap_ratio': 0.9, 'rank_imbalance_ratio': 0.0}
reduce_routing_traffic
increase_overlap_or_delay_sync
rebalance_ranks_or_experts
tune
```

前三行是汇总结果。两条 `all_reduce` 合成了一桶：次数 2，暴露时间 14.0，payload 3072；重叠比例取两条里较小的 0.45，rank 不均衡取较大的 0.05。三类里 `all_to_all` 的暴露时间 20.0 最大，它是优先热点。后四行依次是：对它的判断、测试里另外两种症状的判断、对它的决策。rank 不均衡 0.05 没超过 0.20，重叠比例 0.60 没低于 0.30，类型是 `all_to_all` 且 payload 大于 0，所以动作是减少路由流量；暴露时间 20.0 大于 5.0，决策是 `tune`。例子省略了“rank 不均衡是否采集到”这个标记，notebook 里有。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从时间线建立热点证据：暴露通信时间、payload 与同步频率、重叠比例、rank 不均衡四个信号 | 读 |
| Step 2 | 识别热点属于哪种模式：传输量或路由压力、暴露通信压力、rank 不均衡 | 读 |
| Step 3 | 选择缓解动作并设计复测：每种模式的候选动作、复测指标和可能的代价 | 读 |
| Step 4 · TODO 1 | 为当前 collective 建立或取得统计桶，并更新同步频率等热点证据 | 写 `summarize_comm_hotspots`：用 `setdefault` 建桶，五个量相加，重叠比例取最小、不均衡取最大，最后选出暴露时间最大的 `priority` |
| Step 4 · TODO 2 | 根据 rank_imbalance_ratio、overlap_ratio、collective 和 payload_bytes 决定策略 | 写 `choose_comm_mitigation`：按“不均衡、低重叠、`all_to_all` 流量”的顺序判断，返回 `strategy` 和 `next_measurements` |
| Step 4 · TODO 3 | 无 trace、暴露通信较低、暴露通信明显三种情况分别给出决策和原因 | 写 `recommend_comm_followup`：3 个分支，返回 `decision`、`reason`、`next_measurements` |
| 测试 | 检查重复的 collective 是否先汇总再选热点、三种症状是否导向不同动作、决策是否区分三种状态 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用合成矩阵计算和真实 NCCL `all_reduce`，比较串行调度和异步重叠调度下的最大 rank 步时间 | 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验。

## 容易踩的坑

- **只看单次耗时最大的通信**：同一种 collective 一步里会出现多次。要先累计同步次数、暴露时间和 payload，再比较热点。
- **重叠比例和不均衡取了平均**：重叠比例保留最差值，rank 不均衡保留最大值，平均值会掩盖最差路径。
- **判断顺序颠倒**：rank 不均衡的优先级最高，其次是低重叠的暴露通信，最后才是 `all_to_all` 的路由流量。
- **把缓解动作当成结论**：动作只是下一轮对照实验的候选。每次调整要保持工作负载不变，记录它可能带来的新代价，决策里带上下一轮的测量项。
- **没采集的不均衡当成 0**：rank 不均衡未采集时要保持 unknown，不能当成“没有不均衡”。

## 读完后自查

- [ ] 能说出判断热点要同时看的四个信号。
- [ ] 给几条同类型的通信事件，能算出汇总后的次数、暴露时间和 payload。
- [ ] 能说出三种症状各对应什么缓解动作，以及判断的先后顺序。
- [ ] 能说出没有追踪记录、暴露通信较低、暴露通信明显三种情况下的决策。

## 下一步

- 打开 notebook：[48. 通信热点与缓解策略](../48_Communication_Hotspots_and_Mitigation.md)
- 建议先读：[47. NCCL 通信剖析 · 小白入门](./47_Communication_Profiling_with_NCCL.md)
- 需要补基础：[通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)、[NCCL 与 AllReduce 基础](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)
- 做完之后：[49. 并行策略选型 · 小白入门](./49_Parallelism_Strategy_Selection.md)
