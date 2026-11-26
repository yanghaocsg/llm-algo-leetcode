# 72. 单卡推理部署与 Backend 对比项目 — 小白讲解

> 对应 notebook：[`72_Multi_Model_Deployment_Comparison.ipynb`](./72_Multi_Model_Deployment_Comparison.ipynb)

## 一句话总结
本节把「这个服务配置能不能上线」拆成四道可审计的关卡：固定身份是否一致、候选改了什么是否已声明、启用的专项策略证据是否齐全、质量与服务指标是否过线。四个 TODO 分别实现配对比较、证据分类、门槛判断和 `accept / tune / reject` 决策，最后用一份已保存的 vLLM 基线记录演示怎样读 TTFT、TPOT、吞吐、峰值显存和回滚条件。

## 为什么需要它：痛点是什么
初学者容易把「请求都返回了 HTTP 200」当成「可以部署了」。但在真实服务里，请求成功只说明进程没崩：回答可能答错，P95 延迟可能超过用户忍耐上限，显存可能只剩一点点余量，启用的量化或投机解码也可能在加载时悄悄回退成普通路径。只看一个漂亮的吞吐数字就上线，等于把「能跑」误当成「可交付」。

生活类比：飞机起飞前的检查单。引擎响了不等于能起飞，要逐项确认油量、襟翼、舱门、天气；任何一项红牌都必须停飞，不能因为「今天风速很理想」就把红牌盖过去。本节的门槛顺序也是这个逻辑：硬失败先于缺证据，缺证据先于「性能看起来不错」。

## 核心问题
- 两次服务运行要固定哪些字段，才能被称为「同一个部署对象的对比」？
- 候选换了后端、并发或最大长度，为什么必须和 `declared_changes` 完全对上？
- 量化、投机解码、前缀缓存、调度四类专项结果，缺失、待补、已拒绝怎样分开记录？
- 质量、成功率、TTFT、TPOT、吞吐、峰值显存和回滚，哪些是上限、哪些是下限？
- 为什么 `reject` 优先于 `tune`，`tune` 又优先于 `accept`？
- 真实记录里成功率 1.0、吞吐 175.2218 token/s、峰值约 10.206 GiB，为什么最终结论仍是 `tune`？

## 解决思路：一步步看懂

**第 1 步：什么叫一个「部署候选」。** notebook 的 Step 1 把候选定义成一组可以实际启动的模型、加载方式、后端配置和请求处理规则。比较之前先固定模型版本、分词器、聊天模板、请求集和生成规则；候选只能改动被允许的那几项，并且要逐项声明。否则你测的可能是「另一个模型 + 另一批请求 + 另一个后端」的混合结果，任何指标差异都无法归因。

**第 2 步：核对部署配对（TODO 1）。** `compare_deployment_pair` 把字段分成两类：

```python
fixed_fields = ('model_id', 'revision', 'tokenizer_revision',
                'artifact_digest', 'dtype', 'workload_key')
controlled_fields = ('backend', 'enabled_strategies', 'max_model_len', 'concurrency')
```

固定字段必须完全相等：`model_id` 只说明「哪一类模型」，`revision` 和 `artifact_digest` 才锁定实际加载的文件，`tokenizer_revision` 会影响分词，`dtype` 会影响数值和显存，`workload_key` 锁定同一份请求清单。受控字段则允许变化，但变化集合必须和候选声明的集合一致。参考答案：

```python
identity_ok = not mismatches
changes_ok = bool(changed_fields) and sorted(changed_fields) == declared_changes
capacity_ok = candidate.get('capacity_status') == 'ready'
```

三个点容易写错。第一，`changes_ok` 必须要求 `changed_fields` 非空：一个「什么都没改」的运行不是候选，只是基线的重复。第二，比较前要 `sorted`，因为受控字段按元组顺序收集，而声明列表的顺序不保证一致。第三，容量状态必须是 `ready`，显存装不下时性能数字没有意义。`test_deployment_pair_contract` 用三个例子验证：只把 `backend` 从 `vllm` 改成 `sglang` 并声明 `['backend']` 时可比；改 `artifact_digest` 后不可比；把 `concurrency` 改成 4 却不声明时也不可比（实际变化集合变成 `['backend','concurrency']`）。注意测试里的 `sglang` 只是一个字段值，单元测试不会启动任何后端，也没有测 SGLang 的性能；真实记录链路读取的是 vLLM 基线。

**第 3 步：汇总专项证据（TODO 2）。** `summarize_strategy_evidence` 只处理四个受支持的策略名：`quantized_profile`、`speculative`、`prefix_cache`、`scheduler`。不在集合里的名字进 `unknown_strategies`；启用但 `evidence_by_strategy` 里没有记录的进 `missing_evidence`；有记录但 `decision == 'reject'` 的进 `hard_failures`；其余不是 `accept` 也不是 `reject` 的状态（例如 `tune`、缺失 decision）进 `incomplete_evidence`。参考答案：

```python
rejected = sorted(name for name, status in statuses.items() if status == 'reject')
incomplete = sorted(name for name, status in statuses.items() if status not in {'accept', 'reject'})
```

为什么要分成三堆？因为后续动作不同：缺记录要补测，`tune` 要等上游调整，`reject` 是硬失败、必须修复或换方案。`test_strategy_evidence_states` 分别验证：两个都 `accept` 时三堆都空；只启用 `scheduler` 却没有记录时 `missing_evidence == ['scheduler']`；`speculative` 是 `tune`、`prefix_cache` 是 `reject` 时，前者进 `incomplete_evidence`，后者进 `hard_failures`。

**第 4 步：检查部署门槛（TODO 3）。** `evaluate_deployment_gates` 返回四个布尔值，每个都混合了「方向」和「缺失值处理」：

```python
quality_ok = (candidate.get('quality_status') == 'passed'
              and candidate.get('success_rate', 0.0) >= limits['min_success_rate'])
service_ok = (candidate.get('ttft_p95_ms', float('inf')) <= limits['max_ttft_p95_ms']
              and candidate.get('tpot_p95_ms', float('inf')) <= limits['max_tpot_p95_ms']
              and candidate.get('throughput_tokens_per_s', 0.0) >= limits['min_throughput_tokens_per_s'])
capacity_ok = (candidate.get('peak_memory_gb', float('inf')) <= limits['max_peak_memory_gb']
               and candidate.get('memory_headroom_gb', 0.0) > 0.0)
rollback_ok = candidate.get('rollback_ready') is True
```

方向要记清：延迟和峰值显存是**上限**，越小越好；成功率和吞吐是**下限**，越大越好；显存余量必须严格大于 0，回滚必须显式 `is True`。缺失值故意用了会失败的方向（延迟缺省 `inf`、吞吐缺省 `0.0`、峰值缺省 `inf`），这样「没测到」不会被当成「通过」。测试里的 `_limits()` 是 `min_success_rate=0.99`、`max_ttft_p95_ms=120`、`max_tpot_p95_ms=15`、`min_throughput_tokens_per_s=100`、`max_peak_memory_gb=11.0`；`_ready_candidate()` 是成功率 1.0、TTFT 90、TPOT 12、吞吐 120、峰值 9.5、余量 1.5，四项全部通过。把 TTFT 改成 130、吞吐改成 90、峰值改成 11.5，分别只能让 `service_ok` 或 `capacity_ok` 失败。

**第 5 步：形成部署状态（TODO 4）。** 决策不是「分数加权」，而是固定优先级：

```python
if hard_failures:
    decision, next_action = 'reject', 'repair_failed_candidate'
elif evidence_gaps or not all_gates_passed:
    decision, next_action = 'tune', 'collect_evidence_or_adjust_candidate'
else:
    decision, next_action = 'accept', 'write_deployment_handoff'
```

`hard_failures` 由三部分组成：已启用策略的 `reject`、候选的 `failure`（记为 `backend`）、配对不可比（记为 `deployment_pair`）。`evidence_gaps` 是未知策略、缺失证据、待补证据三张列表的拼接。`test_deployment_decision_priority` 验证优先级：完整证据 + 全门槛通过得到 `accept`；把 `prefix_cache` 改成 `tune` 后即使性能数字不变也变成 `tune`；把候选加上 `failure='backend_start_failed'` 后即使其余全过也变成 `reject`。这正是「好看的性能不能覆盖先出现的失败」。

**第 6 步：真实记录链路。** Step 5 不是重新跑模型，而是读一份已登记的共享基线 `benchmarks/results/66_g0_vllm_baseline.json`。5.1 的配置里 `RUN_FRESH_BENCHMARK=False`、`RUN_LABEL='rtx5070ti_single_gpu_delivery_v1'`、`PROFILE_ID='INF-VLLM-011-CU128'`，真实门槛是 `min_success_rate=1.0`、`max_ttft_p95_ms=300`、`max_tpot_p95_ms=40`、`min_throughput_tokens_per_s=1.0`、`max_peak_memory_gib=11.0`。5.2 的预检会检查六类问题：基线合同版本不是 `inference-baseline/v2`、请求文件缺失、本地模型快照缺失、快照名与记录 revision 不一致、六项指标不完整、基线运行质量不是 `passed`；任一出现就进入 `blocked`。

5.3 把上游专项记录分成四种处置：同模型同负载且 `accept` 的 `eligible_for_primary_candidate`；同模型但不足以进入主候选的 `retain_as_specialized_profile`；`reject` 的 `exclude_rejected`；其余 `hold_for_more_evidence`。5.4 只有在显式开启新鲜复测时才启动 vLLM，并且明确关闭前缀缓存、使用 `fcfs` 调度，避免把缓存收益混进基线；若同一 `RUN_LABEL` 已有结果，会抛 `FileExistsError` 而不是覆盖。

5.5 把记录换算成统一口径：`success_rate = successful / (successful + failed)`；`ttft_p95_ms` 和 `tpot_p95_ms` 从指标字典里取 `p95`；吞吐取 `output_token_throughput_per_s`；峰值从 MB 除以 1024 换成 GiB。`service_gates` 有七个键：`success_rate`、`ttft_p95`、`tpot_p95`、`throughput`、`peak_memory`、`operational_quality`、`independent_task_quality`。前五个是服务指标，`operational_quality` 要求成功率恰为 1.0 且基线质量状态为 `passed`，最后一项要求独立质量报告存在且通过。

**第 7 步：结果怎么读。** notebook 正文登记的这次评估是：模型 `Qwen2.5-0.5B-Instruct`、版本 `7ae5576`、并发 1、复用原 vLLM 结果；成功率 1.0、TTFT P95 46.1171 ms、TPOT P95 5.4064 ms、吞吐 175.2218 token/s、峰值约 10.206 GiB。对着门槛看：46.1171 ≤ 300、5.4064 ≤ 40、175.2218 ≥ 1、10.206 ≤ 11，成功率与 `operational_quality` 也都通过；但 `QUALITY_EVIDENCE_PATH=None`，没有独立任务质量记录，所以 `failed_gates == ['independent_task_quality']`，`project_decision` 是 `tune`，`next_action` 是 `provide_independent_task_quality_evidence`，不会生成交接清单。专项结果方面，投机记录被拒绝、前缀缓存待调整、调度记录保留为不同负载的参考，主候选没有启用专项策略。

峰值 10.206 GiB 距离 11 GiB 上限只剩约 0.794 GiB（约 7.2%），notebook 的 `observed` 里专门算 `headroom_to_limit_gib` 供人看；不过当前代码的 `peak_memory` 门槛只检查「不超过上限」，并不要求固定比例的余量。读结果时要把「代码判过关」和「工程上是否安心」分开：前者是布尔值，后者要结合更长上下文、更高并发再测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 部署候选 | 一组可启动的模型、加载方式、后端配置和请求规则 |
| 固定身份字段 | 两次运行必须完全一致的模型、版本、摘要、dtype、负载标识 |
| 受控字段 | 允许候选改动的后端、策略、最大长度、并发 |
| `declared_changes` | 候选自己声明的改动清单，必须与实际变化完全一致 |
| `workload_key` | 锁定同一份请求清单的标识 |
| TTFT | Time To First Token，从发请求到收到第一个内容的时间 |
| TPOT | Time Per Output Token，首个内容之后每个输出 token 的平均间隔 |
| P95 | 95% 的请求不超过该值，用来看尾延迟而不是平均值 |
| 专项证据 | 量化、投机、前缀缓存、调度四类策略各自的评估记录 |
| `accept / tune / reject` | 可采用 / 补证据或调配置 / 修复硬失败 |
| 回滚基线 / 交接清单 | 失败时能恢复的上一版配置 / `accept` 后写给下游的交付记录 |

## 小白常见疑问
**Q1：请求成功率 1.0，不就是服务没问题吗？**
不是。成功率只说明请求完成了，不说明回答正确。notebook 把「请求完成」和「任务质量」分成两个条件：5.5 的 `operational_quality_ok` 看成功率与基线运行质量，`independent_task_quality` 则要读独立质量报告。登记的这次评估正是服务全过、但因为缺少独立质量记录而停在 `tune`。

**Q2：候选只改了后端，为什么还要 `declared_changes`？直接比较不就行了？**
因为「只改后端」是候选自己的说法，代码必须验证。`compare_deployment_pair` 会把受控字段的实际差异收集成 `changed_fields`，再和排序后的声明列表比较；如果候选偷偷把 `concurrency` 从 1 改成 4，实际变化就多了一项，`changes_ok` 为假。`test_deployment_pair_contract` 的第三个断言就是这种情况。

**Q3：为什么延迟用 P95，不用平均值？**
平均值会被大量快请求拉低，掩盖少数用户的糟糕体验。notebook 的 `p95` 从指标字典里取 `p95` 字段，门槛写成 `<= max_ttft_p95_ms`。GPU 微基准里还会同时看中位数和 P95，因为只看中位数可能漏掉抖动。

**Q4：为什么 `reject` 不能因为吞吐高就改成 `accept`？**
`reject` 对应硬失败：后端起不来、配对不可比、已启用策略被上游拒绝。这些是「结论不可信」或「方案不成立」，不是「数字不够好」。`decide_single_gpu_deployment` 先检查 `hard_failures`，再检查 `evidence_gaps` 和门槛，测试专门用「其余全过 + `failure`」验证仍然 `reject`。

**Q5：峰值 10.206 GiB 离 11 GiB 还有余量，为什么不能算高枕无忧？**
代码只要求 `peak_memory_gib <= 11.0`，所以这次确实过线；但余量只有约 0.794 GiB。真实服务还要面对更长上下文、更高并发、KV Cache 增长和临时工作区，Step 2 明确要求「峰值在预算内且留有余量」。因此登记结果里仍把它当作需要继续观察的证据，而不是容量结论。

## 动手验证：怎么知道自己做对了
`test_deployment_pair_contract`：`_deployment_pair()` 构造的 `vllm → sglang` 候选必须 `comparable`；把 `artifact_digest` 改成 `sha256:b` 后必须不可比；把 `concurrency` 改成 4 但不声明时必须不可比。`test_strategy_evidence_states`：两个策略都 `accept` 时 `missing_evidence`、`incomplete_evidence`、`hard_failures` 都为空；只启用 `scheduler` 且无记录时 `missing_evidence == ['scheduler']`；`speculative=tune`、`prefix_cache=reject` 时 `incomplete_evidence == ['speculative']`、`hard_failures == ['prefix_cache']`。

`test_deployment_gate_directions`：`_ready_candidate()` 配 `_limits()` 时四项全真；`ttft_p95_ms=130` 和 `throughput_tokens_per_s=90` 只让 `service_ok` 为假；`peak_memory_gb=11.5` 只让 `capacity_ok` 为假。`test_deployment_decision_priority`：完整证据 + 全门槛通过得到 `accept`；`prefix_cache` 改为 `tune` 得到 `tune`；候选带 `failure='backend_start_failed'` 得到 `reject`。

`run_deployment_project_tests` 依次跑这四个测试，全过才打印 `✅ 72：部署配对、专项证据、服务门槛与采用状态均已验证。`。notebook 没有保存单元格输出；5.x 读取的是仓库里已有的 JSON，若文件不存在或预检失败，会打印 `blocked` 或抛 `FileNotFoundError`，以你本地的 `preflight`、`strategy_portfolio` 和 `saved` 字典为准。登记结果可对照 `benchmarks/results/72_single_gpu_deployment/72_evaluation_rtx5070ti_single_gpu_delivery_v1.json` 中的 `observed`、`gates`、`decision` 字段阅读。
