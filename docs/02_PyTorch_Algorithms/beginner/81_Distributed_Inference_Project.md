# 81. Distributed Inference Project | 分布式推理逻辑验证 · 小白入门

> 配套 notebook：[81. 分布式推理逻辑验证](../81_Distributed_Inference_Project.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

大模型对外提供推理服务时，可以多放几个副本，可以把模型切到多张卡，也可以把预填充和解码分开。这本 notebook 先在 CPU 上估算每种做法的时间，再判断值不值得搬到多卡上实测。

## 为什么需要它

你训练好一个大模型，要让它对外提供推理服务。请求一多，或者模型大到一张卡放不下，就得扩展。扩展有三种形态：

- 多放几个完整副本，把新请求发给比较空闲的那个。并发上去了，但副本之间的负载可能不均，每个副本都要占一份模型的显存。
- 把模型切到多张卡上，比如张量并行（TP）或流水线并行（PP）。大模型能跨卡运行，但卡与卡之间要传数据。
- 把预填充（prefill）和解码（decode）分开，放在不同的 worker 上。worker 是专门负责这一段工作的服务进程。两种阶段的计算特点不同，分开后互不干扰，但要把 KV 状态从预填充 worker 交给解码 worker，这一步叫交接（handoff）。

选哪种，不能只看总时间。把通信成本合进总时延会把问题盖住：首词元快了，但交接很慢；吞吐高了，但少数请求等得特别久。直接上多卡去试又很费事。

这本 notebook 先在 CPU 上估算一遍：把一条请求的时间拆成几段记成账本，模拟把一批请求分给几个副本，再按完成时间、负载和通信给出 `accept`、`tune` 或 `reject`，帮你决定哪一组值得带到真实多卡上测。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 预填充（prefill）/ 解码（decode） | 一条请求的两个阶段：先处理提示词，再一个词元一个词元往后生成 |
| 副本（replica）与路由 | 副本是模型的一份完整部署；路由是决定新请求发给哪个副本 |
| PD 分离 / handoff | PD 分离是把预填充和解码放在不同的 worker 上；handoff 是把预填充得到的 KV 状态交给解码 worker |
| KV Cache | 推理时把已经算过的中间结果存下来，后面生成时不用重算 |
| makespan / queue wait | makespan 是一批请求从开始到全部做完的时间；queue wait 是请求排队等待的时间 |
| 首词元延迟（TTFT）/ 单词元时间（TPOT） | TTFT 是从请求到来到收到第一个词元的时间；TPOT 是之后平均每个词元要等多久 |
| 代理（proxy） | 用简单公式估出来的数，不是实测值。本节用它表示通信成本和吞吐 |

## 核心思路

整个流程是三个函数，前一个的输出是后一个的输入：

```text
一批请求 + 配置 → 单请求账本 → 按副本模拟分配 → accept / tune / reject
```

第一步算单请求的账本。一条请求的时间拆成五段，加起来是 `total_ms`：

```text
prefill_ms   = prompt_tokens * prefill_ms_per_token
decode_ms    = generate_tokens * decode_ms_per_token
comm_ms      = max(tp_degree - 1, 0) * comm_ms_per_token * (prompt_tokens / 128 + generate_tokens / 64)
pipeline_ms  = max(pp_stages - 1, 0) * pipeline_penalty_ms
handoff_ms   = 0；PD 分离时等于 handoff_base_ms + handoff_bytes / handoff_bytes_per_ms
total_ms     = prefill_ms + decode_ms + comm_ms + pipeline_ms + handoff_ms
```

`prompt_tokens` 和 `generate_tokens` 是提示词的词元数和要生成的词元数，几个 `_per_token` 是配置里给定的单位耗时。`tp_degree` 是张量并行的份数，`pp_stages` 是流水线的段数，份数是 1 时这两项都是 0。`handoff_bytes` 等于 `prompt_tokens * kv_bytes_per_token`，只有 PD 分离才计算交接时间。

第二步模拟分配。每来一条请求，看哪个副本当前的负载最低，就分给它，这叫最低负载（least-loaded）路由。全部分完后，负载最高的那个副本的负载就是 `makespan_ms`，也就是这批请求全部完成的时间。

第三步决策：

```text
makespan 小于单副本时间，并且不均衡比、通信比都在门槛内   → accept
makespan 小于单副本时间，但有门槛没过                     → tune
其余                                                      → reject
```

单副本时间是把这些请求在一个副本上顺序跑完的总时间。通信比是 `comm_cost_ms` 除以单副本时间。

用 notebook 测试里的两条请求试一下：

```python
print(128 * 0.01, 64 * 0.02, 1 * 0.03 * (128 / 128 + 64 / 64))   # 预填充、解码、通信代理

totals = [2.62, 1.31]                       # 两条请求各自的 total_ms
loads = [0.0, 0.0]                          # 两个副本当前的负载
for t in totals:
    chosen = min(range(2), key=lambda i: loads[i])   # 挑负载最低的副本
    loads[chosen] += t
    print(chosen, loads)

makespan, shortest, mean = max(loads), min(loads), sum(loads) / len(loads)
print(round(makespan, 2), round(len(totals) / makespan, 4), round((makespan - shortest) / mean, 4))
print(round(sum(totals), 2), round((0.06 + 0.03) / 3.93, 4))   # 单副本顺序完成时间、通信比
```

```text
1.28 1.28 0.06
0 [2.62, 0.0]
1 [2.62, 1.31]
2.62 0.7634 0.6667
3.93 0.0229
```

第一行是第一条请求的预填充、解码和通信代理，三段加起来是 2.62。两条请求的时间分别是 2.62 和 1.31。第一条请求到达时两个副本都空着，分给 0 号；第二条到达时 1 号副本负载最低，分给 1 号。这批请求的完成时间是 2.62，而单个副本顺序跑完要 3.93。两个副本并没有把时间减半，因为两条请求一大一小，不均衡比是 0.6667。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从请求阶段选择分布式服务形态：完整副本路由、TP / PP 服务实例、PD 分离三种形态的收益和先要观察的代价 | 读 |
| Step 2 | 定义可比较的请求 workload：提示词词元数、生成词元数、到达顺序等字段，对哪些对照必须一致 | 读 |
| Step 3 | 用阶段指标判断收益来自哪里：TTFT、TPOT、排队、makespan、交接各回答什么问题 | 读 |
| Step 4 · TODO 1 | 阶段成本账本 | 在 `estimate_request_cost` 里算出 `prefill_ms`、`decode_ms`、通信代理和流水线惩罚，再加成 `total_ms` |
| Step 4 · TODO 2 | 请求路由与系统账本 | 在 `simulate_distributed_inference` 里按最低负载路由分配请求并累积副本负载，算出 `makespan_ms`、吞吐代理和不均衡比 |
| Step 4 · TODO 3 | 策略决策 | 在 `recommend_distributed_inference_run` 里补 `candidate_makespan`、`comm_ratio`、`imbalance_ratio` 和 `decision`：三项都过才 `accept` |
| 测试 | 检查单请求账本的数值、PD 分离时的交接字节数和时间、两个副本下的分配结果、完成时间、吞吐代理、不均衡比，以及迁移决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上用真实服务的 backend 比较单副本、多副本路由和 PD 分离 | 需要 CUDA、至少 2 张 GPU、`torchrun` 和 vLLM 或 SGLang 这类 backend；默认是 `RUN_GPU_EXPERIMENT = False`，而且每组的 `command` 默认是空的，要先填入启动与压测命令再打开 |

Step 1–4 只需要 CPU。Step 5 需要多张 GPU，默认关闭。

## 容易踩的坑

- **把通信成本并进总时延里看**：通信和交接要单独看。首词元延迟改善但交接过慢，或者吞吐提升但长尾上升，都是要继续调优的信号，不能直接接受。
- **把 `single_replica_time_ms` 当成迁移后的时间**：它是一个副本顺序跑完这批请求的基线。迁移后的完成时间是 `makespan_ms`，也就是负载最高的那个副本的负载。
- **比较时请求不一样**：提示词词元数、生成词元数、到达顺序和并发必须在各对照之间一致，比较 PD 分离还要固定 `kv_bytes_per_token`。否则时间变化说不清来自服务形态，还是来自请求分布。
- **有加速就直接 `accept`**：`makespan_ms` 小于单副本时间之后，还要看不均衡比和通信比是否在门槛内。都在门槛内才是 `accept`，否则是 `tune`；没有加速是 `reject`。
- **把 CPU 账本当成真实 GPU 时间**：通信代理和流水线惩罚是逻辑估算，用来判断下一轮真实 benchmark 该测哪一组候选，不是精确运行时。

## 读完后自查

- [ ] 能说出三种服务形态各自先要观察的代价。
- [ ] 一条请求有 128 个提示词词元、64 个生成词元，能说出 `total_ms` 由哪几段组成。
- [ ] 能解释 `single_replica_time_ms` 和 `makespan_ms` 的区别。
- [ ] 能说出 `accept` 要同时满足哪三个条件。

## 下一步

- 打开 notebook：[81. 分布式推理逻辑验证](../81_Distributed_Inference_Project.md)
- 建议先读：[66. 单卡推理测量基线 · 小白入门](./66_Inference_Performance_Comparison.md)、[70. 推理服务调度基准 · 小白入门](./70_Serving_Scheduler_Benchmark.md)、[49. 并行策略选型 · 小白入门](./49_Parallelism_Strategy_Selection.md)、[PAR-REPLICA. Serving 副本与请求路由 · 小白入门](./PAR-REPLICA_Serving_Replica_Routing.md)
- 做完之后：[82. 量化产物评估项目 · 小白入门](./82_Quantization_Artifact_Evaluation_Project.md)
