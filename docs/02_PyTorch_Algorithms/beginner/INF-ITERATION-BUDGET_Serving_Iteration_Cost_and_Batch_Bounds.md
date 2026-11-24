# INF-ITERATION-BUDGET. Serving Iteration Cost and Batch Bounds | Serving 迭代成本与批次容量上界 · 小白入门

> 配套 notebook：[INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界](../INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

大模型在线服务是一轮一轮处理请求的。这本 notebook 教你算清一轮要花多久、一批最多能放几个请求，以及批次加到多大就不值得再加。

## 为什么需要它

大模型对外提供生成服务时，通常不是来一个请求就单独跑一次，而是每一轮把若干请求凑成一批一起算。于是有个很自然的想法：一批里多塞几个请求，是不是就更快？

不一定。一轮的时间里有两笔开销，随批次变大时走向相反：

- 读模型权重。一轮处理的词元越多，这份时间可能被它们分摊得越薄。
- 读历史 KV Cache。每个请求都要读自己上下文的 KV，请求越多、上下文越长，这部分越大。

一边被摊薄，一边在涨。而且每个正在生成的请求都在等下一步结果，一轮拖得太久，它们都跟着变慢。所以“容量还装得下”和“装得下还够快”是两件事，只看容量会误判。

这本 notebook 把两件事分开算：先用一本账估出一轮的时间，再用五类上界算一批最多能放几个请求，最后沿批次从小到大扫描，找到值得继续复测的位置。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 一轮（iteration） | 调度器选出一批请求，执行引擎跑一次前向、采样并更新状态，这样一个来回叫一轮 |
| 预填充（prefill）/ 解码（decode） | 预填充处理还没算过的输入词元；解码利用已有 KV 往下生成 |
| 摊销（amortize） | 多个请求共用同一份开销，每个请求分到的变少 |
| 带宽（bandwidth）/ 算力 | 带宽是每秒搬多少字节（GB/s）；算力是每秒做多少浮点运算（TFLOP/s） |
| 暴露开销（exposed overhead） | 调度、提交、同步等没被藏进搬运或计算里的时间 |
| 服务等级目标（SLO） | 对延迟或成功率定下的可接受要求 |
| 运行拐点（operating knee） | 满足延迟门槛的前提下，吞吐开始饱和之前的那个批次 |

## 核心思路

输入是本轮的工作量（多少字节、多少浮点运算）和硬件的两个有效速率。先分别估出搬运时间和计算时间，取较大的一个，再加上暴露开销。

```text
T_memory   = 字节数 / 有效带宽
T_compute  = 浮点运算量 / 有效算力
T_iter     ≈ max(T_memory, T_compute) + T_overhead
```

- `T_memory`：访存时间。字节数是权重读取、历史 KV 读取、新 KV 写入之和。
- `T_compute`：计算时间，即每词元计算量乘本轮词元数，再除以算力。
- `T_overhead`：暴露开销。没被前两项盖住的调度、提交、同步时间，同一份不能重复加。

取较大值而不是相加，是因为计算和搬运如果充分重叠，一轮至少要等较慢的一侧。带宽按十进制 GB/s 算，算力按 TFLOP/s 算，结果换算成毫秒。

下面动手算。前两段用 notebook 测试里的数字，第三段是假设的一组扫描结果：

```python
# 1) 一轮的时间
weight_bytes, kv_per_token, contexts, scheduled = 1_000_000_000, 1_000, [100, 200], 16 + 4
memory_bytes = weight_bytes + kv_per_token * sum(contexts) + kv_per_token * scheduled   # 权重 + 读 KV + 写 KV
memory_ms = memory_bytes / (100.0 * 1e9) * 1e3        # 带宽 100 GB/s
compute_ms = 1e9 * scheduled / (10.0 * 1e12) * 1e3    # 每词元 1e9 FLOPs，算力 10 TFLOP/s
print('tokens', scheduled, 'bytes', memory_bytes)
print('memory_ms', memory_ms, 'compute_ms', compute_ms)
print('iteration_ms', round(max(memory_ms, compute_ms) + 0.2, 4))   # 暴露开销 0.2 ms

# 2) 五类上界
kv_per_request = kv_per_token * (9 + 1)               # 上下文 9 个词元 + 预留输出 1 个
limits = {'slot': 8, 'kv': 80_000 // kv_per_request, 'token': 8 // 1, 'slo': 8, 'backend': 8}
final = min(limits.values())
print(final, [name for name, v in limits.items() if v == final])

# 3) 相邻吞吐增益
throughput = {1: 100.0, 2: 180.0, 4: 200.0, 8: 210.0}  # 批次大小 → 吞吐
for prev, cur in [(1, 2), (2, 4), (4, 8)]:
    print(prev, '->', cur, round((throughput[cur] - throughput[prev]) / throughput[prev], 3))
```

```text
tokens 20 bytes 1000320000
memory_ms 10.0032 compute_ms 2.0
iteration_ms 10.2032
8 ['slot', 'kv', 'token', 'slo', 'backend']
1 -> 2 0.8
2 -> 4 0.111
4 -> 8 0.05
```

第 1 段：本轮处理 20 个词元，访存要约 10 毫秒，计算只要 2 毫秒，所以这一轮由访存主导，加上 0.2 毫秒暴露开销是 10.2032 毫秒。

第 2 段回答“一批最多放几个”。五类上界是序列槽位、KV 容量、单轮词元、延迟目标和执行后端，各算一个数再取最小值。例子里五类上界都是 8，全部是并列约束。如果可用 KV 是 0 字节，`kv` 上界就是 0，表示无法为一个请求预留足够状态。

第 3 段回答“值不值得再加”。沿批次从小到大扫描，每档看两件事：单轮时间是否超过门槛，吞吐比上一档涨了多少。阈值是 0.10 时，从 4 到 8 只涨 0.05，这一档已经饱和，应停在上一个选中的批次 4。这是保守筛选，不是在找全局最优。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一轮里请求队列、调度器、执行引擎各自拿到什么、产出什么 | 读 |
| Step 2 | 访存时间和计算时间怎样估，为什么取较大值再加暴露开销 | 读，对照上面的公式 |
| Step 3 | 五类上界各需要什么输入，零上界意味着什么 | 读 |
| Step 4 · TODO 1 | 估算单轮计算与访存下界 | 在 `estimate_iteration_cost` 里补 4 行：`memory_bytes`、`memory_ms`、`compute_ms` 和主导项 `dominant` |
| Step 4 · TODO 2 | 合并批次的五类上界 | 在 `compute_batch_bounds` 里补 3 行：上界字典 `limits`、最小值 `final_limit`、并列约束名列表 |
| Step 4 · TODO 3 | 从批量扫描中选择运行拐点 | 在 `select_operating_knee` 的循环里补 2 行：相邻吞吐增益、是否已饱和 |
| 测试 | 三个测试检查主导项与单位换算、上下文变长后 KV 上界收缩、并列约束与零上界、饱和与延迟越界 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用真实模型在 GPU 上测不同批次和上下文下的预填充、单步解码、峰值显存和 KV 字节 | 需要 `INF-CUDA-MECHANISM` 环境和一张 12 GB 以上显存的 GPU；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，可选，默认是 `RUN_MODE = 'dry_run'`。

## 容易踩的坑

- **把访存时间和计算时间相加**：公式取两者的较大值，再加暴露开销，同一份提交或通信时间不能重复加。
- **单位换算出错**：带宽用十进制 GB/s，算力用 TFLOP/s，结果要换算成毫秒。测试里访存时间应为 `10.0032`。
- **只记一个瓶颈名**：几类上界并列最小时要全部记下来。测试里 `tied` 的 `binding_constraints` 有五个约束名。
- **越界后继续往后找**：首次延迟越界或增益低于阈值就停止，保留上一档，不跳过越界点往后找。
- **把上界当成延迟保证**：取最小值只是检查必要条件。容量公式算不出首词元延迟或尾部延迟，满足 SLO 的批次上限要靠实测或外部估算。

## 读完后自查

- [ ] 能写出 `iteration_ms` 的算式，并说出访存和计算为什么取较大值而不是相加。
- [ ] 能列出五类上界，并说出其中任意一类为 0 时表示什么。
- [ ] 给相邻两档的吞吐，能算出相对增益，并判断它是否低于 `0.10`。
- [ ] 能说出“一批最多放几个”和“放几个才划算”为什么要分开算。

## 下一步

- 打开 notebook：[INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界](../INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md)
- 建议先读：[INF-REQUEST-LIFECYCLE. 推理请求生命周期与核心指标 · 小白入门](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)、[37. KV Cache 生命周期与容量控制 · 小白入门](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
- 需要补基础：[Part 01 · 11：KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- 做完之后：[36. 解码调度 · 小白入门](./36_Decode_Scheduling.md)
