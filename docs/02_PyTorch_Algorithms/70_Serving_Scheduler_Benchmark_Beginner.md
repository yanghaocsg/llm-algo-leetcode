# 70. Serving Scheduler Benchmark Beginner | 推理服务调度基准 · 小白入门

> 配套 notebook：[70. 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

推理服务里请求要排队，排队规则决定谁先执行。这本 notebook 用一个小队列模型比较几种排队规则，并判断新的规则值不值得采用。

## 为什么需要它

推理服务同一时刻会收到很多请求，它们长短不一：有的要生成很久，有的很短。这些请求排在同一个队列里，总得有个规则决定谁先执行。

规则不同，结果差别很大。常见的做法有三种，每种都有代价：

- 先到先服务（FIFO）。按到达顺序执行，短请求排在长请求后面就得久等。
- 最短优先（`shortest`）。先跑短的，平均等待降了，但长请求被推到后面。
- 业务优先级（`priority`）。数值越小越先执行，交互类请求能更早完成，但等待可能被转移给其他请求。

还有两点容易看漏。一是吞吐变高，仍可能让一部分请求等得更久。二是从发出请求到收到第一个词元的时间里，只有一部分是真正的排队。

所以比较调度策略，要把高优先级请求的收益和低优先级请求里等得最久的那部分放在一起看。notebook 登记的一次真实对照采用了优先级策略，但结论只限当前那种优先级混合的请求流：低优先级请求的端到端延迟（从发请求到完成）第 95 百分位增加了约 6.9%，只是仍在设定上限内。

这本 notebook 的主线在 CPU 上用一个单 worker 的队列模型把三种策略走一遍，再练熟三个判断：两组运行能不能比、请求完成清单可不可信、候选能不能采用。真实 vLLM 服务上的对照放在可选的 Step 5。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 调度器（scheduler） | 决定排队请求里谁下一个被执行的那部分逻辑 |
| `pending` / `active` / `completed` | 请求的三个状态：已到达在排队、已被选进当前批次正在执行、已完成并离开队列 |
| 先到先服务（FIFO） | 按到达顺序处理请求，notebook 里拿它当基线 |
| 排队等待（`queue_wait_ms`） | 从请求到达到获准执行的等待时间 |
| 首词元延迟（TTFT）、单词元时间（TPOT） | 前者是从请求发出到收到第一个词元的时间；后者是之后每个词元平均要等多久 |
| P95 | 第 95 百分位：把一批请求的延迟从小到大排好，排在第 95 百分位的那个值 |
| 公平性（fairness） | 本题按等待差距构造的教学分数，在 0 到 1 之间，不是通用的公平性定义 |

## 核心思路

输入是一批请求，每个带到达时间、服务时间（要跑多久）、输出词元数和优先级。CPU 模型只有一个 worker，每一轮做三件事：把已到达的请求放进队列，按策略选出下一批（最多 `batch_size` 个），等这一批跑完再回到第一步。批里最慢的请求决定这一批要跑多久，批内请求在窗口结束时一起退出。

输出有两层：每个请求得到排队等待、完成时间等记录，整组请求得到吞吐、利用率、公平性等指标。三种策略的差别，用一个最小模型就能看出来：

```python
requests = [{'id': 'r1', 'arrival': 0, 'service': 10},   # 先到的长请求
            {'id': 'r2', 'arrival': 0, 'service': 2},    # 同时到的短请求
            {'id': 'r3', 'arrival': 1, 'service': 3}]    # 稍后到的短请求

def run(policy):                                   # 单 worker，一次只跑一个请求
    todo, now, waits = sorted(requests, key=lambda r: r['arrival']), 0, {}
    while todo:
        ready = [r for r in todo if r['arrival'] <= now]        # 只有已到达的才能被选
        if not ready:
            now = todo[0]['arrival']
            continue
        key = {'fifo': lambda r: r['arrival'], 'shortest': lambda r: r['service']}[policy]
        pick = min(ready, key=key)
        waits[pick['id']] = now - pick['arrival']               # 排队等待
        now += pick['service']
        todo.remove(pick)
    return waits

for policy in ('fifo', 'shortest'):
    waits = run(policy)
    print(policy, waits, sum(waits.values()) / len(waits))
```

```text
fifo {'r1': 0, 'r2': 10, 'r3': 11} 7.0
shortest {'r2': 0, 'r3': 1, 'r1': 5} 2.0
```

字典里是每个请求的排队等待（毫秒），最后的数是平均等待。先到先服务时，r1 不用等，r2 和 r3 分别等了 10 ms 和 11 ms。换成最短优先，r2 先跑，r1 要等到 r2 和 r3 都跑完，等了 5 ms。平均等待从 7 ms 降到 2 ms，但先到的长请求从不用等变成了等 5 ms：等待没有消失，只是换了人。

再看时间。请求有四个不同的事件，排队等待和首词元延迟停在不同的位置：

```text
事件 1  请求到达
事件 2  获准执行      排队等待   = 事件 2 - 事件 1
事件 3  收到首词元    首词元延迟 = 事件 3 - 事件 1
事件 4  完成          端到端延迟 = 事件 4 - 事件 1
```

CPU 模型只有整批服务窗口，没有事件 3，所以结果里的 `ttft_ms` 只是 `queue_wait_ms` 的兼容别名，不是真实的首词元延迟。

最后是三个判断，顺序和参考代码一致：

```text
固定字段有缺失或不一致、策略没变、运行失败、重复或丢失请求  → reject
缺请求轨迹、完成清单或关键指标                             → tune
TTFT 和 TPOT 的 P95 不超过上限，吞吐增益、完成率、公平性都达标 → accept，否则 tune
```

固定字段是 `model_id`、`revision`、`backend`、`request_stream_id`、`measurement_key`；候选的 `scheduler_policy` 必须和基线都非空而且不同。公平性只用于本题，算法是 `max(0, 1 - 等待极差 / max(平均等待, 1 ms))`，等待差距越大分数越低。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 请求从排队到执行再到完成的状态链；三种选择策略；CPU 模型和真实后端的区别 | 读 |
| Step 2 | CPU 模型要检查的三条不变量；真实后端对照时固定什么、只变什么 | 读 |
| Step 3 | 到达、获准执行、收到首词元、完成是四个事件；排队等待和首词元延迟的区别；六个指标的定义 | 读，对照上面的例子 |
| Step 4 · TODO 1 | 固定条件一致且调度策略确实变化，才允许解释策略收益 | 写 `validate_scheduler_contract`：补 `fixed_ok` 和 `changed_once`，返回 `contract_ok` 和 `issues` |
| Step 4 · TODO 2 | 契约错配、重复或已知丢失为硬失败；未采集轨迹和指标为缺证 | 写 `build_scheduler_trace_evidence`：补轨迹完整性检查，把硬失败和缺证分开装进两个列表 |
| Step 4 · TODO 3 | 硬失败 reject；缺证或部分门槛未过 tune；全部通过 accept | 写 `decide_scheduler_candidate`：补 `latency_ok` 和 `service_ok`，先处理失败与缺证，再比数值 |
| 测试 | 先查 CPU 模型的状态与时间关系，再查配对条件、完成清单和三种决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在单卡 vLLM 上用同一请求流配对比较 FCFS 与 priority，读取逐请求证据并形成采用判断 | 需要单张 12 GB NVIDIA GPU 和 `INF-VLLM-011-CU128` 环境；默认是 `RUN_REAL_BACKEND = False`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `INF-VLLM-011-CU128` 环境，默认关闭。

## 容易踩的坑

- **固定字段缺失当成一致**：`model_id` 这类固定字段，任一边为空或两边不同都算错配。两边的 `scheduler_policy` 还要都非空而且不同，否则策略收益无从解释。
- **重复完成或丢了请求还去比数字**：完成集合必须和请求集合一致，每个请求只完成一次。重复或已知丢失是硬失败，返回 `reject`。
- **没采集到轨迹当作没问题**：请求清单或完成清单为空、缺关键指标，是缺证，返回 `tune`，不能拿空值去比较大小。
- **把 CPU 模型的等待当成首词元延迟**：CPU 模型只有整批服务窗口，没有首词元事件。结果里的 `ttft_ms` 只是 `queue_wait_ms` 的兼容别名。
- **只看吞吐**：延迟门槛（TTFT、TPOT 的 P95）和服务能力门槛（吞吐增益、完成率、公平性）要同时达标才 `accept`。

## 读完后自查

- [ ] 能说出请求的三个状态，以及 CPU 模型每一轮做的三件事。
- [ ] 给出三条请求的服务时间，能分别写出先到先服务和最短优先下的排队等待。
- [ ] 能解释为什么 CPU 模型结果里的 `ttft_ms` 不能当作真实的首词元延迟。
- [ ] 能列出采用前必须同时满足的两组门槛。

## 下一步

- 打开 notebook：[70. 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)
- 建议先读：[36. 解码调度 · 小白入门](./36_Decode_Scheduling_Beginner.md)、[37. KV Cache 生命周期与容量控制 · 小白入门](./37_KV_Cache_Lifecycle_and_Capacity_Control_Beginner.md)、[38. Prefill/Decode 调度 · 小白入门](./38_Prefill_Decode_Scheduling_Beginner.md)
- 做完之后：[72. 单卡推理部署与 Backend 对比项目 · 小白入门](./72_Multi_Model_Deployment_Comparison_Beginner.md)
