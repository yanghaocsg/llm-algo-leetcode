# 66. 单卡推理测量基线 — 小白讲解

> 对应 notebook：[`66_Inference_Performance_Comparison.ipynb`](./66_Inference_Performance_Comparison.ipynb)

## 一句话总结
同一组提示词两次测得的延迟不同，未必是模型变快或变慢，可能是模型版本、缓冲方式或实际完成的请求数变了。本节用四个 TODO 搭出“基线核验器”：`check_baseline_identity` 核对模型身份，`check_baseline_request` 核对单条流式请求的时间、词元数与停止条件，`check_baseline_totals` 用原始记录复算整轮计数与吞吐，`decide_baseline_readiness` 把证据归为 `ready` / `tune` / `reject`。核心结论是：先证明“测的是同一件事、真的测完了”，再谈性能比较。

## 为什么需要它：痛点是什么
初学者常把“吞吐更高”直接当成“执行更快”。但推理测量有三个坑：输入长度、输出长度、停止位置不同，做了不等量的工作；流式客户端先看到的是“内容块”，一块可能含多个词元，把首内容时间叫成词元级 TTFT 并不准确；汇总表可能声称完成 30 条请求，原始记录却只有 29 条，或只跑两轮却按三轮汇报。

生活类比：给跑步计时。必须确认跑的是同一条赛道（模型与请求固定）、计时器起止位置一致（时间边界）、每个人真的跑完计划圈数（请求数与轮次完整）。少一条，成绩都不能直接横比。

## 核心问题
- 首内容延迟、内容间隔代理、端到端时间、结束尾部，这四个指标各自从哪个事件算到哪个事件？
- 为什么内容间隔代理要除以“输出词元数减一”？只有一个输出词元时为什么不算？
- 整轮输出吞吐的分子和分母分别是什么？失败请求算不算进分母？
- 什么情况判 `reject`、什么情况判 `tune`、什么情况才算 `ready`？为什么“有失败又有缺失”必须判 `reject`？
- 一轮 30 条、重复 3 轮的计划，只跑了两轮，即使每轮内部自洽，为什么仍不能引用？

## 解决思路：一步步看懂

**Step 1：固定基线到底固定了什么。** notebook 列出四类条件：模型身份（权重、分词器、精度、实际解析版本）、请求负载（提示词、顺序、生成上限、并发、重复次数）、测量边界（首内容、最后内容、流结束）、基线核验（原始请求、汇总、停止与资源记录）。本节采用**单并发**：前一条结束后再发下一条，不含排队和组批影响；高负载服务需另设固定到达方式。正式测量在服务就绪和预热后开始，模型准备与服务启动时间单独记录，避免冷启动混进稳态延迟。这里不是多方案加速对比：固定模型和请求是前提，因变量是逐请求时间/停止状态和逐轮吞吐，CPU 题先核验这些量是否自洽。

**Step 2：把“看到内容”和“请求结束”分开。** 设一个构造请求在 0 ms 发出，首个内容块 20 ms 到达，最后内容块 70 ms 到达，流结束 80 ms：

- 首内容延迟 = 20 − 0 = **20 ms**，边界是“发出 → 首次非空内容”，不是词元级 TTFT。
- 内容跨度 = 70 − 20 = **50 ms**。
- 端到端时间 = 80 − 0 = **80 ms**。
- 结束尾部 = 80 − 70 = **10 ms**，是最后内容之后的结束与传输尾巴。

如果后端准确报告生成 4 个词元，内容间隔代理 = 50 ÷ (4−1) ≈ **16.7 ms**；若用“流结束 − 首内容”再除以 3，则得 (80−20)/3 = **20 ms**，把 10 ms 尾部也算进去了。前者是 `content_tpot_proxy_ms`，后者是含结束尾部的平均时长，不能混用。只有一个输出词元时没有“后续间隔”，两者都不定义。

吞吐按整轮窗口算：成功输出 120 个词元、从开始发送到全部结束 3 秒，吞吐 = 120 ÷ 3 = **40 词元/秒**。分子只算成功输出，分母是整个窗口；按每条请求速率取平均、或删掉请求间等待，都会改变分母。失败请求仍计入计数，因为同样占用了窗口时间。

**Step 3：TODO 1——身份必须“完整且匹配”。** 七个键是 `model_id`、`model_revision`、`tokenizer_revision`、`workload_sha256`、`profile_id`、`dtype`、`timing_protocol`。骨架已经算出 `missing`（任一方的值为空）和 `mismatches`（双方都有值但不等）。参考答案：

```python
identity_ok = not missing and not mismatches
```

两类问题必须同时为空：缺字段说明“证据不全”，错配说明“不是同一条件”。`test_identity_requires_complete_match` 分别验证完整匹配通过、`model_revision` 从 `r1` 改成 `r2` 进入 `mismatches`、`tokenizer_revision=None` 进入 `missing`。常见错误是只判断 `mismatches`，漏掉“字段缺失”这种无法比较的情况。

**Step 4：TODO 2——单条请求的时间、长度与停止。** 参考答案：

```python
time_order_ok = 0 <= submitted <= first <= last <= completed
token_count_ok = integer_count and 1 <= tokens <= max_tokens
stop_ok = reason == "stop" or (reason == "length" and tokens == max_tokens)
```

四个时刻必须按 `submit → first_content → last_content → completion` 排列；`tokens` 必须是非 bool 的整数，且在 1 到 `max_tokens` 之间；停止原因要么是自然 `stop`，要么是 `length` 且确实达到上限。骨架还会在 `stream_done_received is not True` 时记 `incomplete_stream`，在 `estimated_output_tokens is not False` 时把 `exact_output_tokens` 放进 `missing`。注意 `missing` 和 `failures` 是两回事：前者是“证据不足、需要补采”，后者是“已经确认违反规则”。

`test_request_order_length_and_stop` 覆盖了这些分支：默认请求（4 个词元、`stop`、80 ms 结束）通过；`output_tokens=1, last_content=20` 通过（单词元合法，只是不算间隔）；`output_tokens=64, finish_reason='length'` 通过；`output_tokens=4, finish_reason='length'` 失败（说截断却没达到上限）；`last_content=10 < first=20` 失败；`output_tokens=65 > 64` 失败；空 `output_text` 失败；`estimated_output_tokens=True` 让 `exact_output_tokens` 进入 `missing`；`{'ok': False}` 直接返回 `request_failed`；`completion_offset_ms=nan` 返回 `nonfinite_time`。

**Step 5：TODO 3——整轮统计必须用同一批样本和同一个窗口。** 参考答案：

```python
counts_ok = (len(rows) == expected_requests
             and metrics.get("successful_requests") == len(successful)
             and metrics.get("failed_requests") == failed_count)
tokens_ok = total_tokens is not None and metrics.get("output_tokens") == total_tokens
throughput_ok = (finite_rate and window_covers_records and expected_rate is not None
                 and math.isclose(throughput, expected_rate, rel_tol=5e-4, abs_tol=0.001))
```

其中 `total_tokens` 是成功记录的 `output_tokens` 之和，`expected_rate = total_tokens / duration`；`window_covers_records` 要求 `duration * 1000 + 1.0 >= max(completion_offset_ms)`，即测量窗口至少要覆盖最后一条请求的完成时刻（留 1 ms 容差）。舍入容差是相对 `5e-4`、绝对 `0.001`。`test_totals_use_all_samples_and_the_same_window` 的一轮是“1 条成功 + 1 条失败”，成功输出 4 个词元、窗口 2 秒，所以声明吞吐必须是 2.0；把 `failed_requests` 改成 0、把 `output_tokens` 改成 5、把 `duration_s` 改成 1.0、只给 1 条记录、或把窗口改成 0，都会让 `ok` 变 False。

这里的关键设计是：吞吐的分母是**完整轮次窗口**，不是“成功请求的总耗时”。失败请求虽然不进分子，但仍然计入 `failed_requests` 并占用窗口；否则失败越多，吞吐看起来越高。

**Step 6：TODO 4——状态优先级。** 参考答案：

```python
must_reject = bool(failures)
can_reference = not failures and not missing
status = "reject" if must_reject else ("ready" if can_reference else "tune")
```

规则是“已确认的硬失败优先于缺失证据”：有 `failures` 就一定 `reject`，哪怕同时还有 `missing`；没有失败但缺证据，判 `tune`，先补采；两者都空才 `ready`。`test_readiness_failure_precedes_missing` 用三组输入断言：空/空 → `ready`，空/`['peak']` → `tune`，`['runtime']`/`['peak']` → `reject`。常见错误是写成“有 missing 就 tune”，导致明明已经 OOM 的运行也被标成“待补证据”。

**Step 7：多轮与统计口径。** 计划每轮 30 条、重复 3 轮，就应保留 3 组各 30 条记录；只完成两轮，即使每轮内部自洽，也没完成计划。P95 也有口径差异：三轮各自 P95 取平均，和合并 90 条后重算 P95 不是一回事；保存时要同时保留统计方法与采样对象。设备总显存含桌面和其他进程，应以设备级指标记录。

**Step 8：可选的 vLLM 实测。** 默认 `RUN_STAGE='prepare'`，只有 `measure` 使用 GPU；固定 Qwen2.5-0.5B-Instruct、每轮 30 条、单并发、最多 64 词元、预热 5 次、重复 3 轮。notebook 登记表记录了一份历史结果：RTX 5070 Ti Laptop、vLLM、30/30 成功、旧协议 `ready`；但缺少首末内容的独立时间戳，新协议尚未实测，只读核对为 `tune`。这正说明旧结论不能直接套到新口径。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 首内容延迟 | 发出请求到第一次收到非空内容的时间 |
| 内容间隔代理 | 首末内容跨度 ÷（输出词元数 − 1），估计后续词元的平均间隔 |
| 端到端时间 | 发出请求到流读取结束 |
| 结束尾部 | 流结束减去最后内容到达，是收尾与传输开销 |
| 输出吞吐 | 本轮成功输出词元总数 ÷ 完整测量窗口 |
| 基线身份 | 模型、分词器、负载哈希、精度、计时协议等一组固定条件 |
| `ready` / `tune` / `reject` | 可引用 / 缺证据待补采 / 已确认失败不可引用 |
| P95 | 95% 分位延迟，描述尾部；不同合并方式不可直接比较 |
| 单并发 | 前一条请求结束后才发下一条，排除排队影响 |
| 校准/评测切分 | 本例里指固定的请求集合同样用于采集与复算 |

## 小白常见疑问
**Q1：首个内容块到了，为什么不能直接叫 TTFT？**
一个内容块可能携带多个词元，客户端看到“第一块内容”不等于后端生成“第一个词元”。字段因此叫 `ttft_content_ms`，表格里明确写“不直接称作词元级 TTFT”；词元级时间要靠后端准确计数或额外埋点。

**Q2：内容间隔为什么除以“词元数减一”？**
4 个词元之间只有 3 段间隔，首内容到末内容的跨度覆盖这 3 段，所以除以 3；除以 4 会把首内容之前那段不存在的间隔也算进去。单词元没有后续间隔，因此不定义。

**Q3：吞吐为什么用整轮窗口，不用每条请求耗时之和？**
服务串行发请求，请求之间的等待也是真实成本。按每条请求速率取平均会忽略这些空隙、得到偏高数字；整轮窗口才能回答“这段时间实际吐出多少词元”。失败请求仍占时间，所以保留在计数里。

**Q4：既有失败又有缺失，为什么不是 `tune`？**
`tune` 的前提是“没有已知硬失败，只是证据不全”。运行报错、输出为空、时间倒序等已确认失败必须先修，补采也救不回来；`must_reject = bool(failures)` 就是这个优先级。

**Q5：三轮 P95 取平均，和合并重算 P95 差别大吗？**
统计过程不同：前者先对每轮求分位数再平均，后者把所有样本放在一起排序取分位数；分布不均时结果会不同，不能当成同一个指标。

## 动手验证：怎么知道自己做对了
测试在 cell 9 直接调用四个函数，全部通过打印 `PASSED: 身份、请求、逐轮统计与基线状态核验通过。`
- `test_identity_requires_complete_match`：完整身份 `ok=True`；`model_revision='r2'` 进 `mismatches`；`tokenizer_revision=None` 进 `missing`。
- `test_request_order_length_and_stop`：默认、单词元、`64 + length` 通过；`4 + length`、末内容 10 ms、65 词元、空文本失败；估算词元数进 `missing`；`{'ok': False}` 返回 `request_failed`；`nan` 返回 `nonfinite_time`。
- `test_totals_use_all_samples_and_the_same_window`：`1 成功 + 1 失败`、4 词元、2 秒、2.0 tok/s 通过；改失败数、词元数、窗口、样本数或窗口置 0 都失败。
- `test_readiness_failure_precedes_missing`：空/空 → `ready`；空/缺 → `tune`；失败/缺 → `reject`。

notebook 没有保存 CPU 测试的执行输出，也没有保存新协议的 vLLM 实测 JSON；登记表里的 30/30、RTX 5070 Ti 是历史记录，不能当作本次运行结果。跑通后可以自己构造边界：把 `_request(completion_offset_ms=79.9)` 传给 `check_baseline_request` 仍应通过（时间顺序只看相对大小），但若把它放进 `check_baseline_totals` 且 `duration_s=0.05`，窗口只覆盖 50 ms，`window_covers_records` 会失败。
