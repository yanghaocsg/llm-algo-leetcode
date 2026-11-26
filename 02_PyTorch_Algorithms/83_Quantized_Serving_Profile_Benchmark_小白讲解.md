# 83. 量化服务 Profile 基准 — 小白讲解

> 对应 notebook：[`83_Quantized_Serving_Profile_Benchmark.ipynb`](./83_Quantized_Serving_Profile_Benchmark.ipynb)

## 一句话总结
同一个量化模型，单请求很快，并发一上来首个回答和后续内容都可能变慢；交互场景和离线吞吐场景该看的指标完全不同。本节把服务场景拆成 `latency`、`balanced`、`throughput` 三类，用三个 TODO 补全“部署来源校验、Profile 契约校验、同场景候选选择”的规则，并用 `test_deployment_lineage`、`test_serving_profile_contract`、`test_profile_candidate_selection` 检查来源、指标方向和阈值方向不会被静默放过。

## 为什么需要它：痛点是什么
初学者容易把“量化后模型很小、单条请求很快”直接等同于“服务更快”，但真实服务里还有排队、并发、批处理、首个 token 的等待和每个 token 的间隔。另一个常见坑是看错方向：TTFT 是“越低越好”的上限指标，吞吐是“越高越好”的下限指标，如果统一按“数值越大越好”排序，就会选到延迟更高的配置。

生活类比：评价一家餐厅。外卖用户关心“多久能拿到第一口”（TTFT），堂食翻台率关心“每小时能出多少份”（吞吐），而“每道菜之间的间隔”（TPOT）决定吃得顺不顺。你不能用“每小时出餐量最高”去证明“第一口最快”，也不能把三家餐厅的数据混在一张榜上比。

## 核心问题
- 为什么比较服务配置前必须先固定量化产物、后端、硬件和请求来源？
- `latency`、`balanced`、`throughput` 三类场景各自的主指标和 SLO 是什么？
- 延迟类指标为什么用 P95，多轮重复的 P95 为什么要再求均值？
- `direction` 为 `max` 和 `min` 时，候选到底该取更小值还是更大值？
- 怎样先过滤“不可比记录”，再按 SLO 筛“合格记录”，最后选主指标最优？
- 主指标持平时，“归一化余量之和”怎么算，为什么还要分别读延迟和吞吐？

## 解决思路：一步步看懂

**第 1 步：固定部署对象，再改变请求压力。** 服务场景（Profile）是一组“请求压力 + 主指标 + 服务目标”的约定，它和环境 Profile 的依赖配置不是同一个东西。比较三类场景前，要先沿用已经验证过的量化产物、基座版本和执行路径，只调整请求数、并发和生成上限，观察服务表现怎样变化。notebook 的表格要求记录四类固定对象：模型与产物（产物编号、版本、文件摘要、来源）、执行路径（格式、后端、回退、日志）、设备环境（硬件、运行时、后端版本）、请求来源（请求内容、分词和生成规则）。`validate_deployment_lineage` 检查的正是上游 67 部署记录：`stage == "quantized_deployment"`、`decision.decision == "accept"`、`artifact_id` 与 `path` 都存在、`lineage.artifact_gate_reference` 与 `artifact_gate_sha256` 都存在。

**第 2 步：三类场景看不同指标。** `latency` 是低并发、短输出，主指标 TTFT P95，越低越好，同时约束首响应上限和失败率；`balanced` 是中等并发和生成长度，主指标 TPOT P95，越低越好，同时保留输出吞吐下限和失败率；`throughput` 是较高并发、较长输出，主指标输出词元/秒，越高越好，同时约束吞吐下限和失败率。历史字段沿用 `ttft_p95_ms`、`tpot_p95_ms` 这些名字，具体内容分块和流结束如何计时，要看当前采集协议。

**第 3 步：P95 的口径。** P95 表示单轮请求延迟中约 95% 不超过的位置。本项目不是把所有请求合并后算一个 P95，而是先对每一轮重复算 P95，再对多轮的 P95 求均值，得到重复运行的汇总。读结果时要记住这个口径，不能把它解释成“全部请求合并后的 P95”。同一场景内按固定请求压力比较，跨场景分别保留结论，不能把 `latency` 的记录拿去和 `throughput` 的记录混排。

**第 4 步：TODO 1——检查部署血缘。** 参考答案一行：

```python
lineage_ready = accepted and artifact_ready and gate_lineage_ready
```

三个条件不可互相替代：`accepted` 说明 67 节已经接受这次部署；`artifact_ready` 说明有产物身份和路径；`gate_lineage_ready` 说明带上了 82 节产物闸门的引用和 SHA256。少任何一个，后续场景比较都失去同一个上游对象。函数同时返回 `issues` 列表，分别用 `deployment_not_accepted`、`artifact_identity_or_path_missing`、`artifact_gate_lineage_missing` 说明缺什么。常见错误是只检查 `decision == "accept"` 就放行，或者把 `artifact_id` 存在当成血缘完整。

**第 5 步：TODO 2——检查 Profile 契约。** 参考答案三个布尔量：

```python
workload_ready = bool(workload.get("id")) and _positive_workload(workload)
slo_ready = bool(slo) and _valid_slo_rules(slo)
primary_ready = profile.get("primary_metric") in slo
```

`_positive_workload` 要求 `num_prompts`、`concurrency`、`max_tokens`、`repeats` 四个字段都是正整数；`_valid_slo_rules` 要求每条 SLO 都有合法的 `direction`（只能是 `max` 或 `min`）和有限、非负的 `limit`；`primary_ready` 要求主指标确实出现在当前 Profile 的 SLO 里，否则“主指标最优”无从谈起。函数还检查 `profile["name"]` 是否在 `SUPPORTED_PROFILES = {"latency", "balanced", "throughput"}` 中，并把问题分别记成 `unsupported_profile`、`invalid_workload`、`invalid_slo`、`primary_metric_missing_from_slo`。注意 `bool` 是 `int` 的子类，测试特意用 `True` 作为非法计数，靠 `_positive_int` 的类型检查把它挡掉。

**第 6 步：方向语义——`max` 是“上限”，不是“越大越好”。** 这是本节最容易读反的地方。`SUPPORTED_DIRECTIONS = {"max", "min"}` 描述的是 SLO 的类型：

- `direction == "max"`：这个指标有上限，值必须 `<= limit`，所以候选值**越小越好**。例如 `ttft_p95_ms: {"direction": "max", "limit": 500.0}` 表示首次响应不得超过 500 ms。
- `direction == "min"`：这个指标有下限，值必须 `>= limit`，所以候选值**越大越好**。例如 `throughput_tps: {"direction": "min", "limit": 70.0}` 表示吞吐不得低于 70 token/s。

`_meets_slo` 就是按这个规则逐项检查：任一项不是有限非负数、`max` 项超过上限、`min` 项低于下限，整条记录都不合格。

**第 7 步：TODO 3——主指标排序与持平处理。** 参考答案：

```python
primary_is_better = candidate_value < best_value if direction == "max" else candidate_value > best_value
primary_is_tied = candidate_value == best_value
candidate_is_better = primary_is_better or (
    primary_is_tied
    and _slo_headroom(candidate["metrics"], slo) > _slo_headroom(best["metrics"], slo)
)
```

主指标方向为 `max`（上限型）时取更小值，为 `min`（下限型）时取更大值。主指标完全相等时，才比较 `_slo_headroom`：对每条 SLO 算归一化余量，`scale = max(abs(limit), 1.0)`，上限型用 `(limit - value) / scale`，下限型用 `(value - limit) / scale`，再求和。余量越大，说明离各项门槛越远。这个量只用于当前规则下的并列候选比较，具体延迟和吞吐仍要分别读取，不能合成一个“总分”。

**第 8 步：三层过滤与选型结果。** `select_serving_candidate` 先调用 `validate_serving_profile`，契约不完整直接抛 `ValueError("Profile 契约不完整")`；再筛“可比记录”：`status == "ok"`、`artifact_id` 与期望产物一致、`profile_name` 与当前场景一致、`workload_id` 与当前工作负载一致、`evidence_level == "real_project_benchmark"`；然后用 `_meets_slo` 筛“合格记录”；最后在合格记录里调用 `choose_primary_record`。返回字典包含 `decision`（选中为 `accept`，否则 `tune`）、`selected`、`comparable_count`、`feasible_count` 和 `excluded`（被排除记录的名称）。骨架已经把失败、错配和未达标记录都排除在外，所以 `choose_primary_record` 只需要做排序。

**第 9 步：实验阶段 prepare → measure → close。** CPU 部分只验证规则；真实测量才需要 `INF-VLLM-011-CU128`、兼容 GPU 和已经通过部署对照的量化产物。`RUN_LABEL` 默认是 `replace_with_unique_label`，`RUN_STAGE` 默认 `prepare`，`UPSTREAM_RUN_LABEL` 默认 `replace_with_upstream_label`。`prepare` 只读预检，`measure` 才启动服务，`close` 只消费已保存证据生成报告和 SHA256 回执。总结论规则是：三类都完成且至少一类达标为 `accept`（只登记实际达标的场景）；三类都完成但没有一类达标为 `tune`；场景间后端/硬件/精度身份变化为 `reject`；测量不完整或关键指标缺失为 `pending`。notebook 明确说明还没有本页新协议的真实 GPU 结果，所以真实数字要等测量后按 `profile_decisions` 读取。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 服务场景（Profile） | 一组请求压力、主指标和服务目标的约定 |
| SLO | 服务等级目标，规定某项指标的上限或下限 |
| TTFT | 首个 token 的到达时间，交互场景的核心等待指标 |
| TPOT | 后续每个 token 的输出间隔，影响流式输出的顺畅度 |
| 吞吐（tokens/s） | 单位时间输出的词元数，离线场景更关心 |
| P95 | 约 95% 的请求不超过的延迟位置；本项目对多轮 P95 再求均值 |
| `direction="max"` | 上限型约束，值必须不超过 `limit`，越小越好 |
| `direction="min"` | 下限型约束，值必须不低于 `limit`，越大越好 |
| 可比记录 | 同产物、同场景、同工作负载、真实运行的记录 |
| 归一化余量（headroom） | 已合格记录离各项 SLO 还有多远，用于主指标持平时比较 |
| 部署血缘 | 产物身份、接受状态、上游闸门引用和 SHA256 的完整链条 |

## 小白常见疑问

**Q1：`direction="max"` 不是“最大化”的意思吗？为什么延迟反而取更小值？**
这里的 `max` 描述的是“指标上限”，不是优化方向。`ttft_p95_ms` 的规则是 `{"direction": "max", "limit": 500.0}`，含义是“最多 500 ms”，所以 360 ms 比 420 ms 好。`choose_primary_record` 里 `direction == "max"` 时用 `candidate_value < best_value`，正是这个意思。真正“越大越好”的吞吐用 `direction="min"`（下限 70 token/s）。读代码时不要按字面把 `max` 当成“取最大值”。

**Q2：为什么测试里传 `"smaller"` 会失败？**
因为 `SUPPORTED_DIRECTIONS` 只允许 `max` 和 `min`。`test_serving_profile_contract` 把 `ttft_p95_ms` 的方向改成 `"smaller"`，`_valid_slo_rules` 返回 `False`，整个 Profile 不 `ready`。这是为了让方向字段保持封闭枚举，避免出现“看起来像方向、实际没被实现”的字符串。

**Q3：为什么 `failure_rate` 的规则是 `{"direction": "max", "limit": 0.0}`？**
上限为 0 表示不允许失败。`_meets_slo` 中 `max` 项要求 `value <= limit`，所以只有 `failure_rate == 0.0` 才通过；负数或非有限值也会在更前面被拒绝。这样“质量更好”不能抵消“失败率超限”。

**Q4：`comparable_count` 和 `feasible_count` 有什么区别？**
`comparable` 只要求记录“能放在一起比”：状态成功、产物一致、场景一致、工作负载一致、证据等级是真实项目基准；`feasible` 还要每条 SLO 都达标。一个记录可能可比但不可行（比如 TTFT 420 ms 超过 500 ms 之外的其它约束），也可能因为产物不一致而根本不可比。测试里 `contaminated` 额外插入 `wrong-artifact`，它不会进入可比集合。

**Q5：主指标相同，为什么用余量之和而不是直接看第二指标？**
notebook 的规则是：主指标相同时，比较各项约束的归一化余量之和，上限指标越低、下限指标越高，余量越大。`test_profile_candidate_selection` 的持平用例里，`narrow` 和 `roomy` 的 `tpot_p95_ms` 都是 60.0，但 `roomy` 的 `throughput_tps=30.0` 高于 `narrow` 的 22.0，余量更大，所以选中 `roomy`。这个量只用于当前规则下的并列比较，不是统计显著性，也不替代分别读延迟和吞吐。

## 动手验证：怎么知道自己做对了
notebook 在题目区末尾调用 `run_serving_profile_tests()`，依次跑三个测试，全部通过后打印 `✅ 部署血缘、Profile 契约和方向化 SLO 选择均已验证。`
- `test_deployment_lineage`：`_deployment_record()` 返回 `stage="quantized_deployment"`、`artifact_id="qwen-awq-r1"`、`path="/tmp/qwen-awq-r1"`、`artifact_gate_reference="82_ref.json"`、`artifact_gate_sha256="abc123"`、`decision="accept"`，断言 `ready` 为真；把 `lineage` 换成空字典后断言 `ready` 为假。
- `test_serving_profile_contract`：`_latency_profile()` 的 workload 是 8 条请求、并发 1、`max_tokens=64`、`repeats=3`，主指标 `ttft_p95_ms`，SLO 为 `ttft_p95_ms <= 500.0` 和 `failure_rate <= 0.0`；断言完整 Profile `ready`；把方向改成 `"smaller"` 后不 `ready`；把 `repeats` 依次改成 `True`、`0`、`-1`、`1.5`，四种都必须不 `ready`。
- `test_profile_candidate_selection`：`c1` 的 TTFT 是 420.0，`c2` 是 360.0，`failed` 状态为 `failed`；额外插入 `wrong-artifact`（产物为 `other`）。断言选中的是 `c2`、决策为 `accept`；把指标换成 `NaN`、`Inf`、`True`、`-1.0` 时决策不能是 `accept`。吞吐场景把主指标换成 `throughput_tps`、方向 `min`、下限 70.0，两个候选 80.0 和 95.0，断言选中 `c2`。持平用例的 SLO 是 `tpot_p95_ms <= 80.0`、`throughput_tps >= 20.0`，`narrow=(60.0, 22.0)`、`roomy=(60.0, 30.0)`，断言选中 `roomy`。
notebook 没有保存这些测试的执行输出，以代码和断言为准；真实服务部分默认 `RUN_STAGE="prepare"`，不会启动后端，`measure` 才需要 GPU 环境和上游 67 部署标签。
