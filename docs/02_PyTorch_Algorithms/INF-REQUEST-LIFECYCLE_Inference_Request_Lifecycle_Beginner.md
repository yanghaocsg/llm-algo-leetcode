# INF-REQUEST-LIFECYCLE. Inference Request Lifecycle Beginner | 推理请求生命周期与核心指标 · 小白入门

> 配套 notebook：[INF-REQUEST-LIFECYCLE. 推理请求生命周期与核心指标](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

你给大模型发一条请求，从提交到回答结束要经过排队、准备、出第一个词、继续输出几步。这本 notebook 教你把每一步的时间点记下来，再算出各个延迟指标。

## 为什么需要它

你向大模型服务提交一条提示词，等了一会儿才看到第一个词，之后文字一个一个往外冒，最后结束。中间可能在排队，可能在处理你的输入，也可能在生成。如果只记下“这条请求一共花了多久”，有两类问题回答不了：

- 时间花在了哪一段？同样是 12 ms，是排队久，还是处理输入慢，还是输出慢？
- 中途出事的请求怎么算？一条请求在第一个词出来之前就被取消了，它的“耗时”能和正常完成的请求放在一起比吗？

看 notebook 里的两条请求。请求 A 正常完成，端到端 12 ms。请求 B 同样在 0 ms 到达，5 ms 时被取消，一个词都没输出。只看总耗时，B 的 5 ms 比 A 的 12 ms 短，可它什么都没有回答。如果把 B 缺失的首词元延迟填成 0 再和 A 求平均，会得到 3.5 ms，而真实的首词元延迟是 7 ms。

这本 notebook 的做法是把到达、准入、输出、终止这些事件按时间逐一记下来，先还原每条请求经历了哪些阶段，再规定每项统计用哪些请求做样本。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 事件 / 状态 / 轨迹 | 事件是某一时刻发生的变化（比如“准入”）；状态是两次事件之间请求所处的阶段（比如“排队”）；一条请求的全部事件按时间排好叫轨迹 |
| 准入 | 允许这条请求占用所需的执行和缓存资源。准入之前模型还没有开始计算 |
| 词元（token） | 模型处理输入和生成输出的基本单位，不等于一个汉字或一个单词 |
| 预填充（prefill）/ 解码（decode） | 预填充处理输入词元，并建立后续生成要读取的 KV Cache；解码继续读取和追加这些历史状态 |
| 首词元延迟（TTFT） | 从请求到达到第一个词元出现的时间 |
| 单词元时间（TPOT） | 相邻两个输出词元之间的平均间隔，notebook 里叫“每输出词元时间” |
| 端到端延迟（E2E） | 从请求到达到完成或异常终止的时间 |

## 核心思路

输入是一串按时间排好的事件，每个事件有名字和时间，比如 `{'event': 'admit', 'timestamp': 2.0}`。notebook 用四个函数依次做四件事：

- 回放：从“未到达”开始，按事件表一步步推进状态，同时记下关键时间戳和每个输出词元的时间。
- 算指标：用两个时间戳相减得到排队、输入准备、预填充、TTFT 和 E2E；相邻输出词元的间隔取平均，得到 TPOT。
- 看占比：把正常完成请求的四段时长相加做分母，算出每段的占比。最大占比达到 45% 才贴上这个阶段的标签，否则是 `mixed`。
- 做汇总：完成率用全部终止请求做分母，TTFT 均值只用已经产生首词元的请求。

下面是 notebook 里请求 A 的轨迹：

```text
时间（ms）   0      2      3      7      9     11     12
事件        到达   准入   就绪   首词元  词元   词元   完成
            排队 2 → 输入准备 1 → 预填充 4 → 首词元之后 5
```

预填充这一段在 notebook 里叫“观察区间”，从输入就绪到首个词元，可能包含设备计算之外的等待和返回。拿这条轨迹把几个指标算一遍：

```python
from statistics import mean

token_times = [7.0, 9.0, 11.0]                # 请求 A 的三个输出词元的时间
gaps = [b - a for a, b in zip(token_times, token_times[1:])]
print(gaps, mean(gaps))                       # 相邻间隔，以及 TPOT
print(7.0 - 0.0, 12.0 - 0.0)                  # TTFT：到达 → 首个词元；E2E：到达 → 完成

phases = {'queue': 2.0, 'preprocessing': 1.0, 'prefill': 4.0, 'decode': 5.0}
total = sum(phases.values())
shares = {name: value / total for name, value in phases.items()}
top = max(shares, key=shares.get)
print(total, shares)
print(top if shares[top] >= 0.45 else 'mixed')

ttft = [7.0, None]                            # A 的 TTFT；B 没有首词元，所以缺失
valid = [value for value in ttft if value is not None]
print(len(valid), mean(valid), mean([value or 0.0 for value in ttft]))
```

```text
[2.0, 2.0] 2.0
7.0 12.0
12.0 {'queue': 0.16666666666666666, 'preprocessing': 0.08333333333333333, 'prefill': 0.3333333333333333, 'decode': 0.4166666666666667}
mixed
1 7.0 3.5
```

三个词元只有两个间隔，TPOT 是 2 ms。四段时长里最大的是“首词元之后”的 5 ms，占 5/12，约 41.7%，没到 45%，所以标签是 `mixed`。最后一行是 A 和 B 放在一起汇总：TTFT 只有 1 个有效样本，均值是 7 ms；如果把缺失值当成 0，就会错成 3.5 ms。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 跟踪请求从到达到终止：事件、状态，以及哪个状态允许哪个事件 | 读，对照上面的轨迹 |
| Step 2 | 用事件时间计算等待与生成指标：每个指标由哪两个事件相减得到，边界不完整时怎么处理 | 读 |
| Step 3 | 比较单／多请求的等待与统计样本：加入请求 C 之后 A 的变化，以及哪些请求进入哪项统计 | 读 |
| Step 4 · TODO 1 | 推进合法请求状态 | 写 `replay_request_events`：补 1 行 `next_state`，在 `TRANSITIONS` 里按当前状态和事件查下一状态 |
| Step 4 · TODO 2 | 计算相邻词元间隔 | 写 `compute_request_metrics`：补 2 行 `token_gaps` 和 `tpot` |
| Step 4 · TODO 3 | 按阶段占比归因请求 | 写 `classify_request_outcome`：补 4 行 `explained_ms`、`phase_shares`、`dominant_stage`、`dominant_share` |
| Step 4 · TODO 4 | 固定工作负载统计分母 | 写 `summarize_workload`：补 2 行 `completed_count` 和 `ttft_values` |
| 测试 | 检查完整轨迹、部分轨迹、异常终止、非法迁移、阶段描述与统计分母 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |

整本 notebook 只需要 CPU。

## 容易踩的坑

- **把缺失的 TTFT 填成 0**：首词元之前就终止的请求没有 TTFT，缺失项保持为空。填成 0 会让没得到回答的请求被统计成最快的请求，A 和 B 的均值会从 7 ms 错成 3.5 ms。
- **TPOT 用词元数当分母**：TPOT 是相邻输出词元间隔的均值，3 个词元只有 2 个间隔。不足两个词元时保持缺失，不能以零替代。
- **把 E2E 算到最后一个词元**：端到端延迟是到达到完成或异常终止。请求 A 是 12 ms，不是最后词元所在的 11 ms。
- **把 `decode_time` 当成解码计算时间**：这个字段是首词元至终止的观察时长，包含结束尾部。A 是 5 ms，其中 1 ms 在最后一个词元之后，不是 GPU 解码计算时间。
- **看到占比就下瓶颈结论**：45% 只是练习里的教学阈值，不是通用瓶颈门槛。阶段占比描述时间怎样分布，实际瓶颈还要结合区间内部的计算与等待判断。

## 读完后自查

- [ ] 给请求 A 的事件时间，能算出排队时间、TTFT、TPOT 和 E2E。
- [ ] 能说出 3 个输出词元为什么只有 2 个间隔。
- [ ] 能说出请求 B 的哪些指标可以算、哪些缺失，以及为什么不能把缺失的 TTFT 填成 0。
- [ ] 能说出完成率和 TTFT 均值分别用哪些请求做样本。

## 下一步

- 打开 notebook：[INF-REQUEST-LIFECYCLE. 推理请求生命周期与核心指标](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)
- 需要补基础：[Part 00 · 20：性能证据与实验设计](../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md)
- 做完之后可以对照：[66. 单卡推理测量基线](./66_Inference_Performance_Comparison.md)、[INF-CONTINUOUS-BATCHING. 连续批处理](./INF-CONTINUOUS-BATCHING_Continuous_Batching.md)
- 做完之后：[22. vLLM 分页注意力 · 小白入门](./22_vLLM_PagedAttention_Beginner.md)
