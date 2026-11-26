# INF-SERVING-OVERLAP. 服务重叠与状态交接 — 小白讲解

> 对应 notebook：[`INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.ipynb`](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.ipynb)

## 一句话总结
本节把“并发能不能省时间”拆成两个可计算的问题：一是传输区间里有多少时间被独立计算真正覆盖（`compute_exposed_interval`、`estimate_handoff`），二是预填充产生的 KV 状态要满足什么条件才算“可以安全交给解码”（`validate_handoff_state`、`plan_decode_start`）。核心结论是：重叠不是把所有传输时间都藏起来，只有与计算窗口相交的部分才算数；而“复制已提交”和“解码可以开始”是两个不同事件，中间还差数据就绪、元数据可见、所有权登记和版本匹配。

## 为什么需要它：痛点是什么
服务推理时，GPU 一边算当前批次，CPU 一边准备下一批，复制引擎一边搬运 KV 状态。三件事提交到不同队列后，初学者容易产生两个误解：第一，以为“放进了另一个流”就一定并行、一定更快；第二，以为“复制操作返回了”就代表数据已经能被解码使用。

生活类比：餐厅后厨。灶台（GPU 计算）在炒当前这桌菜，服务员（复制引擎）从仓库搬下一桌的食材。搬食材和炒菜可以同时进行，但如果灶台只要 3 分钟、搬货要 10 分钟，能省下的最多只有 3 分钟，剩下 7 分钟还是要等。而且食材搬到门口不等于厨师能直接用：还要核对订单号（`request_id`）、确认是这一批（`version`）、登记谁负责保管（`ownership`），任何一个不对都不能开火。

## 核心问题
- 传输总时长由什么决定？Gbit/s 怎样换算成“毫秒能传多少字节”？
- 多个独立计算窗口与传输区间重叠时，能隐藏多少时间？为什么必须先合并区间？
- 暴露时间、可隐藏时间和重叠比例分别怎么定义？
- 解码消费 KV 状态前，要检查哪些数据、元数据和身份条件？
- 依赖没齐时，什么情况下可以等、什么时候必须回退或重算？
- GPU 上“串行复制 → 计算”和“双流重叠”实测差多少？为什么传输越大相对收益反而可能越小？

## 解决思路：一步步看懂

**第 1 步：先分清哪些工作有依赖。** notebook 把工作分成控制面（CPU 安排请求、准备批次、管理状态）、设备侧计算、复制引擎/链路搬运。只有**没有数据依赖**的工作才有同时推进的机会：下一个独立请求的 KV 可以边传边算当前批次；但当前请求的解码必须等它自己的 KV 到达。完成全部必需工作的最早时间由关键路径决定——从输入就绪到结果完成，依赖链里耗时最长的那条。两项工作即使在不同队列提交，也可能争用显存带宽或计算资源，所以不能只看“队列不同”就断定会变快，要核对实际开始、结束和等待事件。

**第 2 步（TODO 1）：算传输区间被覆盖了多少。** `compute_exposed_interval(transfer_interval, cover_intervals)` 的输入是一个传输区间 `(start, end)` 和若干独立工作窗口。它先把每个窗口裁剪到传输区间内：

```python
clipped = sorted(
    (max(start, a), min(end, b))
    for a, b in cover_intervals
    if min(end, b) > max(start, a)
)
```

`max(start, a)`、`min(end, b)` 把窗口截到传输范围内；条件 `min(end,b) > max(start,a)` 过滤掉完全在范围外或只有端点接触的空交集。然后合并重叠区间：

```python
for a, b in clipped:
    if not merged or a > merged[-1][1]:
        merged.append([a, b])
    else:
        merged[-1][1] = max(merged[-1][1], b)
```

如果当前窗口的起点晚于上一段的终点，就新开一段；否则说明相交或相接，把上一段的终点扩展到两者较大值。TODO 1 还要补 `hidden_ms`：合并后各段长度之和。写成公式（传输区间 $[s,e]$，窗口 $[a_i,b_i]$）：

$$
\text{hidden}=\sum_{\text{merged }[l,r]}(r-l),
\qquad
\text{exposed}=(e-s)-\text{hidden}.
$$

为什么必须先合并？测试 `test_interval_union_and_exposed_time` 给 `(0,10)` 和窗口 `[(1,4),(3,7),(12,14)]`：`1–4` 与 `3–7` 重叠，合并后是 `1–7`，覆盖 6 ms；如果逐段相加会得到 `3+4=7 ms`，把 `3–4` 重复算了一次。`12–14` 完全在传输区间外，被过滤掉。最终 `hidden=6`、`exposed=4`、`total=10`。

**第 3 步（TODO 2）：把带宽换算成时间。** `estimate_handoff(payload_bytes, bandwidth_gbps, fixed_latency_ms, start_ms, cover_intervals)` 的骨架先校验载荷非负、带宽为正、固定延迟非负；参考答案保留了前两项检查。核心换算是：

```python
bytes_per_ms = bandwidth_gbps * 1_000_000 / 8.0
transfer_ms = fixed_latency_ms + payload_bytes / bytes_per_ms
```

推导：1 Gbit/s $=10^9$ bit/s。1 秒 $=1000$ ms，所以每毫秒 $10^6$ bit；1 字节 $=8$ bit，所以每毫秒 $10^6/8=125{,}000$ 字节。`bandwidth_gbps * 1_000_000 / 8.0` 就是“每毫秒可传多少字节”。这里用的是十进制 Gbit/s（$10^9$），不是 $1024^3$，单位和网络链路的习惯一致。
然后用总传输时间构造区间 `(start_ms, start_ms + transfer_ms)`，交给 `compute_exposed_interval` 算暴露时间。返回的 `overlap_ratio` 是 `hidden_ms / transfer_ms`，传输时长为 0 时约定为 1.0。
测试 `test_handoff_estimate`：`125_000_000` 字节、`10 Gbit/s`、`1 ms`、起点 0、覆盖 `(0,6)`。`bytes_per_ms=10×10^6/8=1.25×10^6`；传输 `125,000,000/1,250,000=100 ms`；加固定延迟得 `transfer_ms=101 ms`。覆盖窗口 `0–6` 全部落在 `0–101` 内，`hidden=6`、`exposed=101-6=95`。这与解析里“总长 101 ms、覆盖 6 ms 后暴露 95 ms”一致。
容易写错成用 `1024×1024` 做 MiB 换算，或者忘记把固定延迟加在载荷时间之外。notebook 用的是十进制字节数和 Gbit/s，`125 MB` 对应 `125,000,000` 字节。

**第 4 步（TODO 3）：检查状态能不能被消费。** `validate_handoff_state(handoff_state, expected_request_id, expected_version)` 定义三类必需依赖：

```python
required = ['payload_ready', 'metadata_ready', 'ownership_ready']
missing = [name for name in required if not handoff_state.get(name, False)]
incompatible = []
if handoff_state.get('request_id') != expected_request_id:
    incompatible.append('request_id')
if handoff_state.get('version') != expected_version:
    incompatible.append('version')
```

`missing` 和 `incompatible` 要分开报告，因为恢复动作不同：缺依赖可能只是还没到，可以等；身份或版本冲突说明拿到的状态不是这一份，等待没有意义，要重算。`payload_ready` 表示缓存数据到了；`metadata_ready` 表示块地址、序列位置、精度/缩放信息可见；`ownership_ready` 表示谁持有引用、谁负责释放已经登记，防止还在读的块被回收。`ready` 只有两个列表都为空才为真。
测试 `test_handoff_protocol_and_decision`：`metadata_ready=False` 时 `missing=['metadata_ready']`、`incompatible=[]`；把 version 从 3 改成 2、请求 id 保持 `r1`，则 `incompatible=['version']`；三项都就绪且版本匹配时 `ready=True`。

**第 5 步（TODO 4）：选解码动作。** `plan_decode_start(validation, ready_at_ms, deadline_ms)` 的参考答案：

```python
if validation['incompatible']:
    return 'recompute'
if validation['ready']:
    return 'ready'
return 'wait' if ready_at_ms <= deadline_ms else 'fallback'
```

顺序很重要：身份/版本冲突优先判 `recompute`，即使预计就绪时间在期限内也不等，因为等到的可能是错误状态；已经就绪直接 `ready`；只有“缺依赖但预计能在期限内补齐”才 `wait`，否则 `fallback` 到其他执行路径。`ready_at_ms <= deadline_ms` 包含等号：恰好卡在期限上仍允许等待。
测试覆盖四种分支：缺元数据、预计 8 ms 就绪、期限 10 ms → `wait`；预计 12 ms → `fallback`；版本冲突 → `recompute`；完整状态 → `ready`。

**第 6 步：GPU 实验怎样测“物理重叠”。** 5.1 固定三档主机到设备复制量 `[16, 64, 128] MiB` 和同一个 2048×2048 矩阵乘，预热 5 次、正式 20 次，正式轮交替执行先后顺序。`time_case` 用 CUDA Event 记录时间：
- 串行路径：当前流里先 `device_target.copy_(host_source, non_blocking=True)`，再 `left @ right`。
- 重叠路径：建 `copy_stream` 和 `compute_stream`，两者都 `wait_event(start)` 后独立提交复制和矩阵乘，当前流再 `wait_event(copy_done)`、`wait_event(compute_done)`，最后记录 `end`。
锁页内存（`pin_memory=True`）是非阻塞 H2D 复制的前提；`distinct_streams` 检查两个流的句柄不同；`path_evidence` 同时记录 `host_memory_pinned`、`distinct_streams`、`non_blocking_copy`，三者都为真才算路径条件成立。
每个档位计算：`serial_ms`、`overlap_ms` 的中位数和 P95；`copy_median_ms`、`compute_median_ms` 来自串行路径的分项；理论下界

$$
T_{\text{lower}}=\max(T_{\text{copy}},T_{\text{compute}}),
$$

因为复制和计算完全重叠时，总时间不可能低于两者中更长的那个。可隐藏时间 `hideable = max(serial - lower, 0)`，实际隐藏 `hidden = max(serial - overlap, 0)`，重叠效率 `efficiency = min(hidden / hideable, 1.0)`。收益门槛是 `latency_ratio <= 1 - 0.02` 且 `p95_ratio <= 1.05`。
保存的结果 `rtx5070ti_overlap_v1` 里：16 MiB 的串行 1.5529 ms、重叠 1.6210 ms，比值 1.0439（重叠反而更慢），`hidden=0`；64 MiB 比值 0.9336、隐藏约 0.332 ms；128 MiB 比值 0.9611、隐藏约 0.377 ms。尽管 16 MiB 这一档没通过收益门槛，另外两档通过，`accepted_cases>0` 且三项正确性都成立，所以整体决策仍是 `accept_physical_overlap`。
5.4 讨论的是 `rtx5070ti_overlap_v2`：三档中位数比为 0.788/0.934/0.965，即改善约 21.2%/6.6%/3.5%；固定计算窗口约 0.32–0.34 ms，传输量增加后可减少的绝对时间仍约 0.32–0.33 ms，所以相对改善随传输量增大而下降。隐藏时间估算约 0.318–0.333 ms，效率约 95.7%–97.6%。这说明“重叠比例”会骗人：大传输下比例看着小，但绝对隐藏时间可能已经接近计算窗口的上限。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 控制面 | CPU 安排请求、准备批次、管理状态的活 |
| 关键路径 | 依赖链里耗时最长、决定最早完成时间的那条 |
| 传输区间 | 一次状态搬运从开始到结束的时间范围 |
| 覆盖窗口 | 可与传输同时进行的独立计算时间段 |
| 区间合并 | 把相交的覆盖段并成一段，避免重复计时 |
| 暴露时间 | 总传输时间里没有被覆盖、仍需等待的部分 |
| 重叠比例 | 被覆盖时间 ÷ 总传输时间 |
| KV 状态 | 预填充算出的键值缓存，解码要继续使用 |
| 元数据 | 块地址、序列位置、精度/缩放等描述信息 |
| 所有权 | 谁持有引用、谁负责释放缓存块 |
| 状态版本 | 区分同一请求不同写入批次的编号 |
| CUDA Stream / Event | 提交队列 / 记录完成点并建立跨流依赖 |

## 小白常见疑问
**Q1：两个覆盖窗口重叠时，为什么不能把各自长度直接相加？**
因为同一段时间只能被隐藏一次。`(1,4)` 和 `(3,7)` 在 `3–4` 上重叠，直接相加得 7 ms，但实际覆盖只有 `1–7` 共 6 ms。`compute_exposed_interval` 先裁剪、排序再合并，最后只累加合并后每段的长度；`exposed_ms` 用总时长减隐藏时长，保证两者相加等于总时长。

**Q2：为什么理论下界是 `max(copy, compute)`，不是 `copy + compute`？**
因为两条路径完全重叠时，复制和计算同时进行，总时间由较慢的那个决定。如果复制 5 ms、计算 3 ms，理想重叠后约 5 ms；`copy+compute=8 ms` 是串行的时间。`theoretical_lower_bound_ms = max(copy_median_ms, compute_median_ms)` 就是这个意思，后面的 `hideable_ms` 用它来估算“最多能藏多少”。

**Q3：复制已经 `non_blocking=True` 返回了，为什么解码还不能开始？**
非阻塞只表示 CPU 提交后不必等复制完成，数据可能还在链路里。解码需要数据真正到达、元数据可见、所有权登记完成，并且请求 id 和版本匹配；`validate_handoff_state` 的三个 `*_ready` 和两个身份字段就是这道门。notebook 反复强调“复制已提交”和“解码可以安全开始”是不同事件。

**Q4：版本冲突时，明明预计 8 ms 就能就绪、期限是 10 ms，为什么不能等？**
因为等到的状态版本不对，继续等只会浪费时间，甚至读到错误请求的 KV。`plan_decode_start` 把 `incompatible` 放在最前面，直接返回 `recompute`；只有“缺依赖但身份正确”才比较 `ready_at_ms` 与 `deadline_ms`。测试里 version 从 3 改成 2 就触发 `recompute`。

**Q5：GPU 实验里 `compute_correct` 只比较两个结果的第一个元素，够吗？**
不够，notebook 自己也在 5.4 说明这是“采样核对”，完整输出一致性还要另补。`torch.equal(device_target.cpu(), host_source)` 对复制是完整张量比较；矩阵乘只比 `product[0,0]`（`checksum`，容差 `1e-2`），用于确认两条路径算的是同一份输入，不能替代逐元素对照。

## 动手验证：怎么知道自己做对了
CPU 测试依次运行 3 个函数，最后打印 `✅ Serving Overlap 机制测试通过`。关键断言：
- `test_interval_union_and_exposed_time`：`compute_exposed_interval((0,10), [(1,4),(3,7),(12,14)])` 必须精确等于 `{'total_ms':10.0,'hidden_ms':6.0,'exposed_ms':4.0}`；`12–14` 的范围外窗口被排除，`3–4` 的重叠只算一次。
- `test_handoff_estimate`：`estimate_handoff(125_000_000, 10.0, 1.0, 0.0, [(0.0,6.0)])` 的 `transfer_ms` 约为 101.0、`exposed_ms` 约为 95.0（`rel_tol=1e-6`）。
- `test_handoff_protocol_and_decision`：缺 `metadata_ready` 时 `missing=['metadata_ready']`、`incompatible=[]`；`ready_at=8, deadline=10` → `wait`；`12 > 10` → `fallback`；version 2≠3 → `recompute`；三项齐全 → `ready`。
骨架对非法输入（`end < start`、负载荷、带宽 ≤0、负固定延迟）会抛 `ValueError`；参考答案实际保留的是 `end < start`、负载荷和带宽 ≤0 三项，测试没有单独覆盖所有分支，可以自己补一组来确认。

GPU 单元格默认 `RUN_MODE='dry_run'`，只登记计划和结果路径，不导入 CUDA 运行时；改为 `real_gpu` 需要 `INF-CUDA-MECHANISM` 环境和可用的 CUDA。保存的结果是 `rtx5070ti_overlap_v1`，5.4 讨论的是 v2；两轮的档位数字不要混用。判断整体通过要看 `copy_correct`、`compute_correct`、`path_confirmed` 以及至少一档 `benefit_gate_passed`，而不是只看平均值。
