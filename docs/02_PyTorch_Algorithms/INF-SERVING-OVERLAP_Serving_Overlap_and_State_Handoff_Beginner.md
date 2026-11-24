# INF-SERVING-OVERLAP. Serving Overlap and State Handoff Beginner | 服务重叠与状态交接 · 小白入门

> 配套 notebook：[INF-SERVING-OVERLAP. 服务重叠与状态交接](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

把数据搬运和计算同时进行可以省时间，但能省多少要看计算能盖住多少搬运。这本 notebook 教你算真正省下的时间，并判断交接过来的状态能不能安全使用。

## 为什么需要它

预填充算完一个请求的状态后，解码一侧要用这些状态，中间要把数据搬一次。搬运需要时间，最简单的做法是等搬完再开始解码。

流水线的想法是：搬这批状态的同时，让 GPU 去算别的请求。这样搬运的时间就“藏”在计算里，等待方不用单独为它等。能和传输同时进行的独立计算，下面叫覆盖窗口。

但有两个容易想错的地方：

- 重叠不等于全部藏起来。传输要 10 毫秒，独立计算只有 6 毫秒，剩下的 4 毫秒照样要等。两件事如果争用同一份资源，时间线上看着并行，实际上谁也没变快。
- 搬完不等于能用。解码要的不只是 KV 字节，还有块表、序列位置、所有权这些信息。只看到“传输已经提交”就去读，可能读到没写完的状态。

这本 notebook 做两件事：算出传输时间里真正暴露在等待路径上的部分；检查交接状态齐不齐，再决定解码是启动、等待、重算还是回退。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 关键路径（critical path） | 从开始到出结果，必须依次等待的那串工作。重叠能缩短它，但不能绕过数据依赖 |
| 覆盖窗口 / 可重叠窗口 | 能和传输同时进行的独立计算所占用的时间段 |
| 隐藏（hidden） | 传输时间和别的计算同时进行，等待方不用单独为它等 |
| 暴露时间（exposed time） | 传输时间里没被窗口盖住、仍然要等的部分 |
| 预填充—解码交接（handoff） | 预填充算出的 KV 状态连同元数据，交给解码一侧使用 |
| 所有权（ownership） | 这份状态归谁管、由谁释放。解码一侧看不到它，就不能安全读取 |
| 期限（deadline） | 解码最晚可以等到什么时候 |

## 核心思路

输入是两样：传输占用的时间段，以及可以和它同时进行的几段独立计算。

```text
1. 裁剪：每段独立计算只保留落在传输区间内的部分，完全在外面的丢掉
2. 合并：裁剪后按起点排序，互相重叠的并成一段
3. 相减：被隐藏时间 = 合并后各段长度之和
          暴露时间   = 传输总时间 - 被隐藏时间
```

notebook 给的近似式是：暴露时间 ≈ `max(传输时间 - 可重叠窗口, 0)`。这是理想下界，同步等待、带宽争用和任务粒度都会让有效窗口变短。

下面用 notebook 测试里的数字算一遍：

```python
start, end = 0.0, 10.0                            # 传输区间，单位毫秒
covers = [(1.0, 4.0), (3.0, 7.0), (12.0, 14.0)]   # 可以同时进行的独立计算
clipped = sorted((max(start, a), min(end, b)) for a, b in covers if min(end, b) > max(start, a))
merged = []
for left, right in clipped:                       # 合并互相重叠的窗口
    if not merged or left > merged[-1][1]:
        merged.append([left, right])
    else:
        merged[-1][1] = max(merged[-1][1], right)
hidden = sum(b - a for a, b in merged)
print(clipped, merged, hidden, end - start - hidden)
print(sum(b - a for a, b in clipped))             # 不合并直接相加会多算

bytes_per_ms = 10.0 * 1_000_000 / 8.0             # 10 Gbit/s 的链路，每毫秒传多少字节
transfer_ms = 1.0 + 125_000_000 / bytes_per_ms    # 固定延迟 1 毫秒 + 125 MB 载荷
print(bytes_per_ms, transfer_ms, transfer_ms - 6.0)
```

```text
[(1.0, 4.0), (3.0, 7.0)] [[1.0, 7.0]] 6.0 4.0
7.0
1250000.0 101.0 95.0
```

第一行：`(12, 14)` 完全在传输区间之外，被丢掉。`(1, 4)` 和 `(3, 7)` 有重叠，合并成 `(1, 7)`，长度是 6。传输总共 10 毫秒，暴露 4 毫秒。第二行是不合并直接相加的结果 7，把重叠的 1 毫秒多算了一次。

第三行是交接时间。带宽的单位是 Gbit/s，先换成每毫秒多少字节：乘 `1_000_000` 再除以 8。125 MB 的载荷在 10 Gbit/s 的链路上要搬 100 毫秒，加上 1 毫秒固定延迟，一共 101 毫秒。能盖住 6 毫秒，暴露 95 毫秒。

算完时间还要检查状态。三个依赖 `payload_ready`、`metadata_ready`、`ownership_ready` 都就绪，并且 `request_id` 和 `version` 都对得上，解码才能开始：

```text
身份或版本不兼容                 → recompute
依赖齐全                         → ready
依赖未齐，就绪时刻不晚于期限     → wait
依赖未齐，就绪时刻晚于期限       → fallback
```

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 哪些工作可以重叠，重叠需要什么条件 | 读 |
| Step 2 | 总传输时间和暴露时间为什么不同 | 读，对照上面的裁剪、合并、相减 |
| Step 3 | KV 状态交接除了字节还包含哪些元数据，分哪几个阶段 | 读 |
| Step 4 · TODO 1 | 计算未被隐藏的传输区间 | 在 `compute_exposed_interval` 里补 2 行：`clipped` 是裁剪并排序后的交集，`hidden_ms` 是合并后的长度之和 |
| Step 4 · TODO 2 | 换算完整状态交接时间 | 在 `estimate_handoff` 里补 2 行：`bytes_per_ms` 和 `transfer_ms` |
| Step 4 · TODO 3 | 检查状态可消费条件 | 在 `validate_handoff_state` 里补 `missing` 和 `incompatible` 两个列表 |
| Step 4 · TODO 4 | 选择解码启动动作 | 在 `plan_decode_start` 里补 1 行 `can_wait`：就绪时刻不晚于期限 |
| 测试 | 检查重叠窗口的合并、字节和带宽换算、固定延迟、缺失依赖、版本冲突和期限边界 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用 PyTorch 的 CUDA Stream 让复制和计算并发提交，对照串行和重叠两条时间线 | 需要 `INF-CUDA-MECHANISM` 环境和一张 GPU；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认是 `RUN_MODE = 'dry_run'`。

## 容易踩的坑

- **重叠窗口被算了两次**：先裁剪到传输区间，再合并重叠部分，最后才相加。测试里 `(1, 4)` 和 `(3, 7)` 合并后隐藏时间是 `6.0`，暴露时间是 `4.0`。
- **区间外的窗口没丢掉**：完全落在传输区间之外的窗口不进入列表。测试里 `(12, 14)` 在 `(0, 10)` 之外，对结果没有贡献。
- **带宽单位和固定延迟弄错**：带宽是 Gbit/s，要换成每毫秒字节数。传输时间是固定延迟加搬运时间，测试里合起来是 `101.0` 毫秒。
- **缺失和不兼容混在一起**：三个依赖没就绪记在 `missing` 里，`request_id` 或 `version` 对不上记在 `incompatible` 里，要分开报告。不兼容时必须重算，等待没有用。
- **期限边界写错**：就绪时刻恰好等于期限时仍然可以等待，所以判断用“不晚于”。测试里就绪时刻 `8.0`、期限 `10.0` 得到 `wait`，`12.0` 得到 `fallback`。

## 读完后自查

- [ ] 传输 10 毫秒，独立计算是 `(1, 4)` 和 `(3, 7)`，能算出被隐藏时间和暴露时间。
- [ ] 能把 10 Gbit/s 换成每毫秒字节数，并写出传输时间的算式。
- [ ] 能说出交接状态要检查哪三个依赖，哪两个字段对不上时要重算。
- [ ] 就绪时刻是 12.0、期限是 10.0，能说出 `plan_decode_start` 返回什么。

## 下一步

- 打开 notebook：[INF-SERVING-OVERLAP. 服务重叠与状态交接](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md)
- 建议先读：[38. Prefill/Decode 调度 · 小白入门](./38_Prefill_Decode_Scheduling_Beginner.md)
- 做完之后：[39. 异构 PD 与服务分层 · 小白入门](./39_Hetero_PD_and_Serving_Tiers_Beginner.md)
