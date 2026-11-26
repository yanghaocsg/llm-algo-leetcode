# INF-REQUEST-LIFECYCLE. 推理请求生命周期与核心指标 — 小白讲解

> 对应 notebook：[`INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb`](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb)

## 一句话总结
本节把一条推理请求拆成「未到达 → 排队 → 输入准备 → 预填充 → 解码 → 完成/取消/超时/失败」的事件轨迹，用 `replay_request_events` 回放状态，用 `compute_request_metrics` 算出排队、预处理、预填充、TTFT、TPOT、解码观察时长和 E2E，再用 `classify_request_outcome` 与 `summarize_workload` 做阶段归因和工作负载汇总。最关键的结论是：**缺失的指标要保留为 `None`，不能用 0 顶替；完成率和 TTFT 均值的统计分母不是同一个集合。**

## 为什么需要它：痛点是什么
初学者看推理服务指标，常把几个名词混成一锅：

- 把「首词元延迟 TTFT」当成「整个请求的耗时」。其实 TTFT 只到第一个输出词元，后面还有一大段解码。
- 用「平均每输出词元时间 TPOT」除以总时间，忘了 TTFT 不参与 TPOT 的分母。
- 请求在输入准备阶段被取消，报表里却给它记 `TTFT = 0 ms`，于是"没得到回答的请求"反而成了最快请求。
- 算成功率时用「已完成请求数 ÷ 产生过首词元的请求数」，把取消的请求从分母里悄悄抹掉。

生活类比：去医院看病。从进门（到达）到挂号（准入）是排队；到分诊台建档案（输入准备）是预处理；医生第一次叫号（首词元）之前是预填充；之后每次复诊（输出词元）之间的间隔是 TPOT；看完离开（完成）才是 E2E。如果病人还没见到医生就退号了，你不能把他的"见到医生时间"记成 0 秒，也不能把他从"今天来了多少人"里删掉。

## 核心问题
- 一条正常请求的状态如何从 `not_arrived` 一步步走到 `completed`？哪些事件在哪些状态是非法的？
- TTFT 为什么等于排队 + 输入准备 + 预填充观察区间？E2E 又为什么等于 TTFT + `decode_time`？
- TPOT 的分母到底是什么？为什么三个词元只产生两个间隔？
- 请求在首词元之前取消，哪些指标还能算、哪些必须为空？
- 阶段占比超过 45% 就一定是瓶颈吗？`mixed` 标签在说什么？
- 为什么成功率用「完成数 ÷ 全部终止请求」，而 TTFT 均值只用「已产生首词元的请求」？

## 解决思路：一步步看懂

### 第 1 步：状态机与事件表
notebook 开头给出 `TRANSITIONS` 和 `TERMINAL_EVENTS`：

```python
TERMINAL_EVENTS = {'complete': 'completed', 'cancel': 'cancelled',
                   'timeout': 'timed_out', 'fail': 'failed'}
TRANSITIONS = {
    'not_arrived': {'arrive': 'queued'},
    'queued': {'admit': 'preprocessing'},
    'preprocessing': {'input_ready': 'prefill'},
    'prefill': {'first_token': 'decode'},
    'decode': {'token': 'decode', 'complete': 'completed'},
}
TERMINAL_STATES = {'completed', 'cancelled', 'timed_out', 'failed'}
```

逐行读：

- `not_arrived` 只接受 `arrive`，进入 `queued`。这是每条轨迹的起点，且只能出现一次。
- `queued` 只接受 `admit`（获得执行与缓存资源），进入 `preprocessing`。准入之前不能执行模型。
- `preprocessing` 只接受 `input_ready`，进入 `prefill`。输入没准备好就不能进模型。
- `prefill` 只接受 `first_token`，进入 `decode`。首个词元标志着观察阶段切换。
- `decode` 接受 `token` 并**留在 `decode`**（自环），或接受 `complete` 进入 `completed`。
- 四个终止事件 `cancel` / `timeout` / `fail` / `complete` 分别映射到四个终止状态；其中前三个可以在到达之后的任意未终止阶段发生，`complete` 只允许从 `decode` 走。

Step 1 的表格还规定：终止后不再追加词元，每次迁移都应符合当前状态允许的事件。这为后面的非法迁移测试埋下伏笔。

时间戳不是每个事件都记。`replay_request_events` 里的映射是：

| 事件 | 写入的时间戳键 | 是否进入 `token_times` |
|---|---|---|
| `arrive` | `arrival` | 否 |
| `admit` | `admitted` | 否 |
| `input_ready` | `input_ready` | 否 |
| `first_token` | `first_token` | 是（首词元也算一个输出词元） |
| `token` | 无 | 是 |
| `complete` | `terminal` | 否 |
| `cancel` / `timeout` / `fail` | `terminal` | 否 |

这张表解释了两件事：为什么 `duration('input_ready','first_token')` 能算预填充，而 `duration('admitted','first_token')` 也能算但语义是"输入准备 + 预填充"；以及为什么 `token_times` 包含首词元，`streamed_tokens` 统计的是"首词元 + 后续词元"的总数。notebook 说"一次输出事件对应一个词元"，真实流式内容块可能一条消息含多个词元，那种情况下这张表要按后端事件重新定义。

预填充观察区间的结束边界选在 `first_token`，而不是 `input_ready` 或"模型内部最后一层算完"。Step 1 明确说这个区间"可能包含设备计算之外的等待与返回"；也就是说，`prefill = 7 - 3 = 4 ms` 里既有真正的矩阵计算，也可能有调度、传输、缓冲。想拆出纯设备计算，需要后端更细的时间线，不能只靠这四个事件。

### 第 2 步：TODO 1 `replay_request_events`——查表推进状态
骨架已经做了三件事：空列表抛 `ValueError('events 不能为空')`；时间戳必须单调（`timestamp < previous_time` 抛 `ValueError('事件时间必须单调')`）；终止状态后继续追加事件抛 `ValueError('终止状态后不能继续追加事件')`。你要补的是合法迁移：

```python
next_state = TRANSITIONS.get(state, {}).get(event)
if next_state is None:
    raise ValueError(f'状态 {state} 不接受事件 {event}')
state = next_state
```

为什么用 `.get(state, {}).get(event)` 而不是 `TRANSITIONS[state][event]`？因为当前状态可能不在字典里（比如已经到终止状态，或拼错的状态名），直接下标会抛 `KeyError`，而题目要求统一抛带上下文的 `ValueError`。两层 `.get` 的默认值 `{}` 保证第一层缺失时第二层也不会炸。

推进之后，代码按事件记录时间戳：

```python
timestamp_key = {
    'arrive': 'arrival', 'admit': 'admitted',
    'input_ready': 'input_ready', 'first_token': 'first_token',
    'complete': 'terminal',
}.get(event)
```

注意 `token` 不在这张表里——它只被追加到 `token_times`，不单独存时间戳键。`first_token` 同时做两件事：写入 `timestamps['first_token']`，并追加到 `token_times`；`token` 只追加。所以请求 A 的 `token_times` 是 `[7.0, 9.0, 11.0]`，包含首词元。

异常终止分支单独处理：

```python
if event in {'cancel', 'timeout', 'fail'}:
    if state == 'not_arrived':
        raise ValueError('请求到达前不能终止')
    failure_stage = state          # 记住终止发生在哪个阶段
    state = TERMINAL_EVENTS[event]
    timestamps['terminal'] = timestamp
```

`failure_stage` 是本节的关键设计：取消发生在 `preprocessing`，就记 `'preprocessing'`；超时发生在 `decode`，就记 `'decode'`。没有它，后面 `classify_request_outcome` 只能说"失败了"，说不出"卡在哪"。

函数最后检查 `state not in TERMINAL_STATES` 就抛 `ValueError('轨迹缺少终止事件')`，保证返回的每条轨迹都有明确结局。

**容易写错的地方**：把 `token` 的下一状态写成 `'completed'` 或新状态；把 `complete` 也塞进异常终止分支（它其实由 `TRANSITIONS['decode']` 正常推进）；忘记把 `first_token` 追加进 `token_times`，导致 TPOT 少一个样本。

### 第 3 步：TODO 2 `compute_request_metrics`——用事件边界算指标
参考实现只有两行核心代码：

```python
token_gaps = [current - previous for previous, current in zip(token_times, token_times[1:])]
tpot = mean(token_gaps) if token_gaps else None
```

`zip(token_times, token_times[1:])` 把相邻元素配成 `(7,9)`、`(9,11)`，相减得到 `[2.0, 2.0]`，均值 `2.0`。三个词元只有 **两个** 间隔，这是 TPOT 的定义决定的：它衡量"上一个词元到下一个词元"的速度，不是"词元数 ÷ 总时间"。不足两个词元时（例如只产生了首词元就超时），`token_gaps` 为空，`tpot` 必须是 `None`，不能填 0。

其余指标都通过局部函数 `duration(start, end)` 计算：

```python
def duration(start, end):
    return timestamps[end] - timestamps[start] if start in timestamps and end in timestamps else None
```

只有两端时间戳都存在才相减，否则返回 `None`。这解释了为什么"部分轨迹保留已有指标、缺失项为空"能自动成立。代入请求 A 的完整轨迹：

| 指标 | 计算 | 结果 |
|---|---|---|
| `queue_wait` | `admitted - arrival = 2 - 0` | 2.0 ms |
| `preprocessing` | `input_ready - admitted = 3 - 2` | 1.0 ms |
| `prefill` | `first_token - input_ready = 7 - 3` | 4.0 ms |
| `ttft` | `first_token - arrival = 7 - 0` | 7.0 ms |
| `tpot` | `mean([9-7, 11-9])` | 2.0 ms |
| `decode_time` | `terminal - first_token = 12 - 7` | 5.0 ms |
| `token_span` | `11 - 7` | 4.0 ms |
| `e2e` | `terminal - arrival = 12 - 0` | 12.0 ms |
| `streamed_tokens` | `len([7,9,11])` | 3 |

两条恒等式值得记住：

$$
\mathrm{TTFT} = \mathrm{queue\_wait} + \mathrm{preprocessing} + \mathrm{prefill} = 2+1+4=7
$$

$$
\mathrm{E2E} = \mathrm{TTFT} + \mathrm{decode\_time} = 7+5=12
$$

第二条里的 `decode_time` 是"首词元 → 终止"的观察时长（notebook 说字段名沿用 `decode_time`），它不仅包含 4 ms 的词元跨度，还包含最后词元 11 ms 返回后到 12 ms 终止的 1 ms 收尾。所以 `decode_time = token_span + 结束尾部 = 4 + 1 = 5`。notebook 特意提醒：这段观察区间里除了设备计算，还可能含调度、传输或缓冲，要靠后端事件再拆。

### 第 4 步：TODO 3 `classify_request_outcome`——占比归因与 45% 阈值
函数先处理异常终止：`report['status'] != 'completed'` 时直接返回 `{'outcome': status, 'stage': failure_stage, 'dominant_share': None}`，不参与占比计算。正常完成才继续，并校验 `0.0 < dominance_threshold <= 1.0`，否则抛 `ValueError('dominance_threshold 必须位于 (0, 1]')`。

接着收集四个阶段时长，过滤掉 `None`：

```python
phase_durations = {'queue': metrics['queue_wait'],
                   'preprocessing': metrics['preprocessing'],
                   'prefill': metrics['prefill'],
                   'decode': metrics['decode_time']}
available = {name: value for name, value in phase_durations.items() if value is not None}
```

TODO 部分参考实现：

```python
explained_ms = sum(available.values())
if explained_ms <= 0:
    raise ValueError('可解释阶段时长必须为正')
phase_shares = {name: value / explained_ms for name, value in available.items()}
dominant_stage = max(phase_shares, key=phase_shares.get)
dominant_share = phase_shares[dominant_stage]
```

逐项解释：

- `explained_ms` 是四个阶段之和，也就是 E2E（对完整轨迹而言）。它必须为正，否则占比会出现除零。
- `phase_shares` 把每段时长除以总时长，四个占比之和为 1。
- `max(..., key=phase_shares.get)` 取占比最大的阶段名。
- 最后返回时：`dominant_share >= dominance_threshold` 才把 `stage` 记成该阶段，否则记成 `'mixed'`。

代入请求 A：`queue=2`、`preprocessing=1`、`prefill=4`、`decode=5`，总和 12。占比分别是 $2/12\approx16.7\%$、$1/12\approx8.3\%$、$4/12\approx33.3\%$、$5/12\approx41.7\%$。最大是 decode 的 41.7%，小于默认阈值 0.45（45%），所以 `stage='mixed'`，但 `dominant_share` 仍返回 `5/12`。测试断言：

```python
assert outcome['stage'] == 'mixed'
assert abs(outcome['dominant_share'] - 5.0 / 12.0) < 1e-9
```

把 `first_token` 从 7 ms 推迟到 9 ms，后两个词元改成 10、11 ms，重新构造 `prefill_heavy`：此时 `prefill = 9 - 3 = 6`，`decode_time = 12 - 9 = 3`，总和仍是 12，prefill 占比 $6/12 = 50\% \ge 45\%$，所以 `stage='prefill'`。测试只断言这一项。

**这个 45% 是什么？** notebook 明确说它是"给定的 45% 教学阈值"，用于描述示例中的耗时分布；未达到时返回 `mixed`。它**不是**判断真实瓶颈的物理定律。阶段占比高只说明时间花在哪，不说明内部是计算、等待还是传输；实际瓶颈要按执行时间线分析。

分母也值得注意：`explained_ms` 是 `available` 里所有非空阶段之和，不是固定的 E2E。对正常完成的请求，四段都有，和就是 E2E，四个占比之和为 1；如果某段缺失（比如异常轨迹），该段不会以 0 进入分母，而是从 `available` 里删掉，剩余阶段重新归一化。好在 `classify_request_outcome` 在 `status != 'completed'` 时提前返回，不会对异常轨迹算占比，所以这个"缺段归一化"主要影响的是代码复用到其他场景时的语义。四段占比中任意一个达到 45%，意味着它几乎占了可解释时间的一半；`mixed` 则表示没有单段超过这条线。

### 第 5 步：TODO 4 `summarize_workload`——两套分母
参考实现：

```python
completed_count = sum(report['status'] == 'completed' for report, _ in request_results)
ttft_values = [metrics['ttft'] for _, metrics in request_results
               if metrics['ttft'] is not None]
```

然后返回：

```python
{'total_requests': total_requests,
 'completed_requests': completed_count,
 'success_rate': completed_count / total_requests if total_requests else None,
 'ttft_sample_count': len(ttft_values),
 'mean_ttft': mean(ttft_values) if ttft_values else None}
```

两个关键点：

1. **成功率的分母是 `total_requests`（全部终止请求）**，不是"产生过首词元的请求数"。请求 B 在 `preprocessing` 阶段取消，仍然计入总数，所以 A、B 两条请求的完成率是 $1/2=0.5$。
2. **TTFT 均值只收集 `metrics['ttft'] is not None` 的记录**，B 没有首词元，被排除；样本数 1，均值 7.0。如果把 B 的缺失 TTFT 填 0，均值会变成 $7/2=3.5$，正好复现 Step 3 警告的错误。

为什么两套分母不能合并？因为它们回答的是不同问题。成功率问"所有提交的请求里有多少正常完成"，分母必须是全部终止请求；TTFT 问"能看到首词的请求首词有多快"，分母只能是产生过首词元的请求。notebook 还提到另一种口径：已产生首词元、后来才超时的请求也可以进入 TTFT 统计，但要保留超时状态，并说明这与"只统计正常完成请求"不同。本节实现选择的是"非缺失 TTFT 即可"，所以超时请求只要产生过首词元也会被计入。

空工作负载时，`total_requests=0`，`success_rate` 和 `mean_ttft` 都返回 `None`，避免除零。

### 第 6 步：并发请求怎样改变 A 的时间线
Step 3 构造了另一轮运行：A 的输入输出词元数不变，请求 C 在 8 ms 到达并参与资源竞争，A 的后续输出推迟到 12、14 ms，15 ms 完成。结果：

- A 的 `TTFT` 仍是 7 ms（首词元在竞争前已经返回）。
- `TPOT` 从 2 ms 变成 3.5 ms：间隔变成 $[12-7, 14-12]=[5, 2]$，均值 $7/2=3.5$。
- `E2E` 从 12 ms 变成 15 ms。

这段对照说明：**并发竞争影响的不一定是首词元，可能主要落在解码阶段。** notebook 同时区分两个概念：并发数（同一时刻已到达、尚未结束的请求数）和执行批次（本轮实际被选中处理的请求集合）。没被选中的请求仍在等待，所以并发数高不等于批次大。要解释 A 的延后，还需要批次成员和执行区间记录；本节只给定时间线，不推断调度原因。

### 第 7 步：异常轨迹的指标边界
三个异常用例覆盖了不同的"部分轨迹"：

| 轨迹 | 事件 | 可算指标 | 为空指标 | 归因 |
|---|---|---|---|---|
| 首词元前取消 | arrive 0、admit 2、cancel 5 | `queue_wait=2`、`e2e=5` | `preprocessing`、`prefill`、`ttft`、`tpot`、`decode_time`、`token_span` | `cancelled` / `preprocessing` |
| 解码中超时 | 前 5 个事件 + timeout 10 | `queue_wait=2`、`preprocessing=1`、`prefill=4`、`ttft=7`、`tpot=2`、`decode_time=3`、`token_span=2`、`e2e=10`、`streamed_tokens=2` | 无（首词元已产生） | `timed_out` / `decode` |

取消用例里 `preprocessing` 为什么是 `None`？因为 `duration('admitted','input_ready')` 需要 `input_ready` 时间戳，而该事件从未发生。`e2e` 为什么还能算？因为 `arrival` 和终止事件写入的 `terminal` 都存在：$5-0=5$。`token_span` 为 `None` 因为 `token_times` 为空。notebook 用这个用例强调「若将缺失 TTFT 填成 0 ms，未得到回答的请求反而会被统计成最快的请求」。

超时用例里 `token_times=[7,9]`，所以 `streamed_tokens=2`、`tpot=2`、`token_span=9-7=2`；`decode_time=10-7=3`，`e2e=10`。`failure_stage` 在超时发生时的状态是 `decode`，所以归因阶段为 `decode`。这验证了"已产生的词元不会因为超时被抹掉"。

这里还要防止一个读数陷阱：取消请求的 `e2e=5 ms` 比正常完成的 12 ms 小，但它并不"更快"。notebook Step 3 的原话是「取消更早不代表生成更快」；E2E 必须和终止状态、`failure_stage` 一起读。把 `completed`、`cancelled`、`timed_out` 的 E2E 混在一张榜上排名，会得出"取消越多服务越快"的荒谬结论。正确做法是分开报告状态计数，再在同类状态内部比较时间。

正常完成的请求，`failure_stage` 是 `None`：它只在 `cancel` / `timeout` / `fail` 分支被赋值，`complete` 走 `TRANSITIONS['decode']`，不会设置这个字段。`classify_request_outcome` 对 `completed` 的请求也不读它，而是算四个阶段占比。所以 `failure_stage is None` 对正常轨迹是预期行为，不是漏填。

### 第 8 步：把五个时间量串成一条链
设一条正常完成的请求产生 $n$ 个输出词元（`streamed_tokens = n`），首词元时间戳为 $t_1$，最后词元为 $t_n$，终止事件为 $t_{end}$。则：

$$
\mathrm{token\_span}=t_n-t_1,\qquad
\mathrm{TPOT}=\frac{\mathrm{token\_span}}{n-1}\quad(n\ge2),
$$

$$
\mathrm{decode\_time}=t_{end}-t_1=\mathrm{token\_span}+\mathrm{tail},
$$

$$
\mathrm{E2E}=\mathrm{TTFT}+\mathrm{decode\_time}
=\mathrm{queue\_wait}+\mathrm{preprocessing}+\mathrm{prefill}+\mathrm{token\_span}+\mathrm{tail}.
$$

对照请求 A：$n=3$，$t_1=7$，$t_n=11$，$t_{end}=12$。于是 `token_span=4`、`TPOT=4/(3-1)=2`、`decode_time=5`、`tail=1`、`E2E=7+5=12`。这组公式解释了为什么"三个词元只有两个间隔"：分母是 $n-1$，不是 $n$。

边界要单独记：

- $n=0$（首词元前终止）：`token_span=None`、`TPOT=None`、`decode_time=None`，但 `E2E` 仍可由 `arrival → terminal` 算出。
- $n=1$（只产生首词元）：`token_span = t_1-t_1 = 0`，但 `TPOT` 仍为 `None`，因为没有第二个词元可减。
- $n\ge2$：`TPOT` 才有定义，且它只反映解码阶段，不含 TTFT。

### 第 9 步：四个统计口径的样本集合
同一个工作负载，四个指标用的样本集合并不相同。notebook 的 Step 3 表格可以整理成下面这张"谁进分母"的表：

| 指标 | 样本条件 | A、B 两条请求里谁被计入 |
|---|---|---|
| 正常完成率 | 全部已终止请求 | A 完成、B 取消，共 2 条；完成 1 条 → 0.5 |
| TTFT 均值 | 已产生首词元（`ttft is not None`） | 只有 A 的 7 ms；样本数 1 |
| TPOT 均值 | 至少两个输出词元 | 只有 A 的两个间隔；样本数 1 条请求 |
| E2E | 到达与终止都记录 | A 的 12 ms、B 的 5 ms，各自保留终止状态 |
| 阶段占比 | 正常完成且四段边界完整 | 只有 A；B 保留 `cancelled` 和 `preprocessing` |

这套口径的直接后果是：`success_rate` 的分母和 `mean_ttft` 的分母可以不同，报告里必须同时写 `total_requests`、`completed_requests`、`ttft_sample_count`。代码返回这三个字段，就是为了让读报表的人知道"7 ms 是 1 条请求的均值，不是 2 条"。

还有一个容易混的口径：notebook 提到"已产生首词元、后来才超时的请求，也可以进入'已产生首词元'的 TTFT 统计，但要保留超时状态"。本节的 `summarize_workload` 用的是"`ttft is not None`"，所以解码中超时的请求只要产生过首词元，其 TTFT 会进入均值；这与"只统计正常完成请求"的严格口径不同，读结果时要看代码而不是猜。

举例：`test_failed_decode_keeps_streamed_tokens` 的超时轨迹有首词元 7 ms，因此 `ttft=7`；如果把它和正常完成的 A 放在一起汇总，`mean_ttft` 会把这条超时请求算进去，而 `success_rate` 只把 A 算作完成。报表里同时出现"完成率 1/2"和"TTFT 均值包含 2 条请求"并不矛盾，因为它们回答的是不同问题；但必须显式写出 `ttft_sample_count` 和状态分布，读者才不会误以为两个 7 ms 都来自成功请求。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 事件 / 状态 | 某一时刻发生的变化 / 两次事件之间请求所处的阶段 |
| 准入（admit） | 请求获准占用执行与缓存资源，可以开始输入准备 |
| 预填充（prefill） | 处理输入词元并建立 KV Cache 的阶段 |
| 解码（decode） | 逐个生成输出词元、读取并追加历史状态的阶段 |
| TTFT | 从到达（或提交）到首个输出词元的时间 |
| TPOT | 相邻输出词元时间戳间隔的平均值 |
| E2E | 从到达（或提交）到完成/异常终止的端到端时间 |
| `decode_time` | 本节字段名，实际是"首词元 → 终止"的观察时长 |
| 结束尾部 | 最后一个词元返回到终止事件之间的收尾时间 |
| `failure_stage` | 请求被取消/超时/失败时所处的状态 |
| 阶段占比 | 某阶段时长 ÷ 可解释阶段总时长，用于描述时间分布 |
| 统计分母 | 每个指标实际参与统计的样本集合大小 |

## 小白常见疑问

**Q1：为什么 TTFT 要拆成排队 + 输入准备 + 预填充，而不是直接首词元时间减到达时间？**
两种算法在数值上等价：`duration('arrival','first_token')` 就是 7 ms，而 $2+1+4=7$。拆开是为了看清时间花在哪一段：排队等资源、输入准备、还是模型预填充。测试 `test_completed_trace_and_metrics` 专门断言 `metrics['ttft'] == metrics['queue_wait'] + metrics['preprocessing'] + metrics['prefill']`，以及 `metrics['e2e'] == metrics['ttft'] + metrics['decode_time']`，就是检查这几条边界没有接错。

**Q2：TPOT 为什么是 2 ms，而不是 12 ms ÷ 3 个词元 = 4 ms？**
TPOT 衡量的是"输出词元之间的速度"，不是"平均每个词元分摊多少总时间"。三个词元 `[7,9,11]` 只有两个相邻间隔，都是 2 ms，所以 TPOT=2 ms。12 ms 里还包含排队、输入准备、预填充和结束尾部，这些不属于词元间隔。测试期望的 `tpot` 正是 `2.0`。如果实现成 `token_span / streamed_tokens`，会得到 $4/3\approx1.33$，测试立刻失败。

**Q3：阶段占比最大是 decode 41.7%，为什么标签是 `mixed` 而不是 `decode`？**
默认 `dominance_threshold=0.45`，41.7% < 45%，所以返回 `mixed`。阈值比较的是最大占比，不是"最大阶段是谁"。把首词元推迟到 9 ms 后，prefill 占比变成 $6/12=50\%$，标签才变成 `prefill`。这个 45% 是教学阈值，notebook 明确说它只用于描述示例耗时分布，不代表真实瓶颈判定。

**Q4：为什么不能把缺失 TTFT 填成 0？**
因为 0 是一个"极快"的合法数值，会污染均值。A、B 汇总时，正确口径是 TTFT 样本只有 A 的 7 ms，均值 7 ms；填 0 后会变成 $(7+0)/2=3.5$ ms，让取消的请求看起来最快。实现上用列表推导过滤 `metrics['ttft'] is not None`，并额外返回 `ttft_sample_count`，就是为了让读者知道均值背后有几个样本。

**Q5：`decode_time` 和 `token_span` 有什么区别？**
`token_span` 是首词元到最后一个词元的时间差，请求 A 是 $11-7=4$ ms；`decode_time` 是首词元到终止事件的时间差，A 是 $12-7=5$ ms。两者相差 1 ms，就是最后词元返回后的结束尾部。notebook 特意说明 `decode_time` 沿用原名，实际保存的是"首词元至终止的观察时长"。

## 动手验证：怎么知道自己做对了
测试区在 cell 9，依次调用六个测试函数，全部通过后打印 `✅ 请求生命周期机制测试通过`。TODO 未填时先抛 `NotImplementedError`。

- `test_completed_trace_and_metrics`：完整轨迹的 `status == 'completed'`，`token_times == [7.0, 9.0, 11.0]`，整个 `metrics` 字典严格等于 `{'queue_wait': 2.0, 'preprocessing': 1.0, 'prefill': 4.0, 'ttft': 7.0, 'tpot': 2.0, 'decode_time': 5.0, 'token_span': 4.0, 'e2e': 12.0, 'streamed_tokens': 3}`，并断言两条恒等式 `ttft == queue_wait + preprocessing + prefill`、`e2e == ttft + decode_time`。
- `test_stage_share_attribution`：原轨迹 `outcome == 'completed'`、`stage == 'mixed'`、`dominant_share` 与 `5/12` 的差小于 `1e-9`；把首词元改到 9 ms 后 `stage == 'prefill'`。
- `test_partial_trace_preserves_missing_metrics`：取消轨迹的 `queue_wait == 2.0`、`ttft is None`、`e2e == 5.0`，归因严格等于 `{'outcome': 'cancelled', 'stage': 'preprocessing', 'dominant_share': None}`。
- `test_failed_decode_keeps_streamed_tokens`：解码中超时的 `streamed_tokens == 2`，`outcome == 'timed_out'` 且 `stage == 'decode'`。
- `test_workload_denominators`：两条请求汇总后 `total_requests == 2`、`completed_requests == 1`、`success_rate == 0.5`、`ttft_sample_count == 1`、`mean_ttft == 7.0`。
- `test_invalid_transitions`：四类非法序列都必须抛 `ValueError`——未到达就 `admit`、未到达就 `cancel`、时间倒序 `1.0 → 0.5`、完成后再追加 `token`。任何一类不抛，循环就 `raise AssertionError('非法事件序列应被拒绝')`。

这个 notebook 没有保存任何执行输出（dump 里没有 output），所以不要期待看到打印的指标表；判断标准就是上述断言。跑通后最后一行是 `✅ 请求生命周期机制测试通过`。
