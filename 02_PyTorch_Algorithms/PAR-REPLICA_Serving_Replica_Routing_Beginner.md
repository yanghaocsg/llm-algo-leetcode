# PAR-REPLICA. Serving Replica Routing Beginner | Serving 副本与请求路由 · 小白入门

> 配套 notebook：[PAR-REPLICA. Serving 副本与请求路由](./PAR-REPLICA_Serving_Replica_Routing.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

线上服务常常同时开着好几份一样的模型，每个请求都要决定送去哪一份。这本 notebook 教你三种分法，以及怎样在“排队长短”和“是否已有缓存”之间取舍。

## 为什么需要它

一个模型上线提供服务时，常常不只开一份，而是开好几份一模一样的，每一份叫一个副本。请求从入口进来，要有人决定送去哪一个副本，这个决定叫路由。

路由分得不好，会出三种事：

- 轮流分发，看上去公平。但请求的成本有高有低，轮到谁全凭顺序，某个副本可能攒下一堆高成本请求，形成热点。
- 只挑最闲的副本。排队确实短了，但这次请求开头那段内容，其实在某个副本上已经算过、缓存好了，换一个副本就要重算一遍。
- 哪里有缓存就送哪里。那个副本正忙，请求为了命中一直在等，尾延迟反而变长。

notebook 的测试里有一个具体例子：0 号副本已经缓存了这次请求要用的前缀，但要等 20 毫秒才能接；1 号副本没有缓存，等 5 毫秒就能接。这本 notebook 要做的是：给“为了缓存最多愿意多等多久”设一个门槛，再在同一个请求流上模拟三种路由，比较它们的排队时间和吞吐。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 副本（replica） | 同一个模型拷贝出来的一份。各副本独立执行请求，路由层决定请求进哪个副本的队列 |
| 前缀缓存（Prefix Cache） | 副本上存着的、前面已经算过的内容。命中就不用重复预填充（prefill） |
| 轮询（Round-Robin） | 按请求序号轮流分发，不读任何运行状态 |
| 队列感知（Queue-Aware） | 选预计等待最短的副本 |
| 局部性感知（Locality-Aware） | 优先选已经缓存了这段前缀，或者上次处理过同一个会话的副本 |
| 等待容忍度（slack） | 为了用上缓存，最多愿意比最短等待多等多久，代码里是 `locality_slack_ms` |
| 尾延迟（P95 / P99） | 95% 或 99% 的请求都不超过的那个排队时间，用来看少数等得最久的请求 |

## 核心思路

输入是一串请求和一组副本。每个请求带编号和四个字段：到达时间 `arrival_ms`、服务时间 `service_ms`、前缀 `prefix_key`、会话 `session_id`。每个副本记着自己的编号、什么时候空闲（`available_at_ms`）、缓存了哪些前缀、接过几个请求。

模拟按到达顺序一个一个处理请求：先用路由策略选一个副本；再算这个请求什么时候开始、有没有命中缓存、实际要算多久；然后更新副本的空闲时间和缓存。所有请求处理完，再汇总成指标。几种策略各跑一遍，用的必须是同一串请求。

关键量的算法如下：

```text
waits[i]             = max(0, replica[i].available_at_ms - arrival_ms)
start_ms             = max(arrival_ms, replica.available_at_ms)
queue_wait_ms        = start_ms - arrival_ms
effective_service_ms = service_ms * (1 - cache_hit_speedup)     # 命中缓存时；没命中就是 service_ms
throughput           = request_count * 1000 / window_ms
load_skew_ratio      = (max(counts) - min(counts)) / mean_count
```

`waits[i]` 是请求到达时，在副本 `i` 上预计要等多久。`start_ms` 取到达时间和副本空闲时间里较晚的那个，所以同一个副本上的请求不会重叠。`cache_hit_speedup` 默认是 `0.30`，表示命中时服务时间减少的比例。`window_ms` 是观测窗口，从最早的到达时间到最后一个请求结束；`counts` 是每个副本接到的请求数。

局部性感知策略选副本时多一道门槛：有缓存的副本，等待不能比最短等待多出 `locality_slack_ms`，超过就放弃缓存，回到最短队列。下面是最小例子，数字取自 notebook 的测试：

```python
import math

# 1. 选副本：0 号缓存了前缀 a，但要等 20 毫秒；1 号没有缓存，等 5 毫秒
replicas = [{'available_at_ms': 20.0, 'cached_prefixes': {'a'}},
            {'available_at_ms': 5.0, 'cached_prefixes': set()}]
arrival_ms, prefix_key = 0.0, 'a'
waits = [max(0.0, r['available_at_ms'] - arrival_ms) for r in replicas]
shortest = min(waits)
cached = [i for i, r in enumerate(replicas) if prefix_key in r['cached_prefixes']][0]
for slack in (20.0, 5.0):
    if waits[cached] <= shortest + slack:          # 有缓存的副本，等待不能比最短等待多出 slack
        print(shortest + slack, cached, 'prefix_locality')
    else:
        print(shortest + slack, waits.index(shortest), 'queue_fallback')

# 2. 排队时间的平均值与 P95
queue_waits = sorted([0.0, 1.0, 2.0, 10.0])
index = max(0, math.ceil(0.95 * len(queue_waits)) - 1)
print(sum(queue_waits) / len(queue_waits), queue_waits[index])
```

```text
25.0 0 prefix_locality
10.0 1 queue_fallback
3.25 10.0
```

前两行是门槛的作用，每行依次是门槛、选中的副本、路由原因。等待容忍度是 20 时，门槛是 25，0 号副本的等待 20 没有超过，请求送去 0 号用缓存；容忍度是 5 时，门槛只有 10，于是放弃缓存，送去 1 号。最后一行说明为什么要看尾延迟：4 个请求的排队时间是 0、1、2、10，平均只有 3.25，P95 却是 10，那个等了很久的请求只有尾延迟才看得见。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 区分训练数据并行与 Serving 副本：两者都复制模型，但协同的对象和均衡目标不同 | 读 |
| Step 2 | 比较三种路由信号：轮询、队列感知、局部性感知各自的收益和容易误判的场景 | 读，对照上面的小例子 |
| Step 3 | 用负载、局部性与尾延迟共同判断：路由结果至少要同时记录哪几项证据 | 读 |
| Step 4 · TODO 1 | 应用局部性等待门槛 | 写 `select_replica`：只保留等待不超过“最短等待 + `locality_slack_ms`”的候选，按会话、前缀的顺序选，都不行就回到最短队列 |
| Step 4 · TODO 2 | 推进请求与副本时间线 | 写 `simulate_replica_routing`：3 行，算出 `start_ms`、`cache_hit`、`effective_service_ms` |
| Step 4 · TODO 3 | 计算服务指标 | 写 `summarize_replica_service`：3 行，算出 P95 排队时间、吞吐和负载偏斜 |
| 测试 | 检查三类策略各读自己的信号并遵守等待容忍度、同一副本上的请求不重叠、三种策略用同一请求流比较 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 把同一组不均匀的请求成本分给两个真实 GPU rank，比较 `round-robin` 与 `least-loaded` 的最大 rank 完成时间 | 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要双卡 GPU 环境，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验。

## 容易踩的坑

- **为了命中一直等**：只认缓存的策略必须设置等待容忍度。有缓存的副本等待超过“最短等待 + `locality_slack_ms`”时，要放弃缓存，回到最短队列。
- **同一副本上的请求重叠了**：`start_ms` 要同时受到达时间和副本空闲时间约束，取较大的那个。处理完还要把副本的 `available_at_ms` 更新成 `finish_ms`。
- **比较时请求流不一样**：三种策略必须复用相同的请求流和副本数，吞吐必须用统一的观测窗口，否则指标不可比。
- **只看平均延迟**：少量被错误粘住的长等待请求会被平均值掩盖。要同时看 P95 / P99、命中率、每副本请求数和忙碌时间。
- **空输入的口径没保留**：`records` 为空时要返回各项都是 0 的指标。骨架已经写好了这个分支，补指标时不要破坏它。

## 读完后自查

- [ ] 能说出轮询、队列感知、局部性感知三种策略各自看什么信号。
- [ ] 给两个副本的等待时间和一个等待容忍度，能判断请求会被送到哪个副本。
- [ ] 能写出开始时间、排队时间和命中缓存后有效服务时间的算式。
- [ ] 能解释为什么比较策略时不能只看平均排队时间。

## 下一步

- 打开 notebook：[PAR-REPLICA. Serving 副本与请求路由](./PAR-REPLICA_Serving_Replica_Routing.ipynb)
- 建议先读：[24. Prefix Cache 匹配与复用 · 小白入门](./24_Prefix_Cache_Matching_and_Reuse_Beginner.md)、[36. 解码调度 · 小白入门](./36_Decode_Scheduling_Beginner.md)、[70. 推理服务调度基准 · 小白入门](./70_Serving_Scheduler_Benchmark_Beginner.md)
- 做完之后：[48. 通信热点与缓解策略 · 小白入门](./48_Communication_Hotspots_and_Mitigation_Beginner.md)
- 相关 notebook：[49. 并行策略选型](./49_Parallelism_Strategy_Selection.ipynb)、[81. 分布式推理逻辑验证](./81_Distributed_Inference_Project.ipynb)
