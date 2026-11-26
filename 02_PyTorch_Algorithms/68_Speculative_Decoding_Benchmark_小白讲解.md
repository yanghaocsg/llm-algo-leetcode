# 68. 投机解码机制选型基准 — 小白讲解

> 对应 notebook：[`68_Speculative_Decoding_Benchmark.ipynb`](./68_Speculative_Decoding_Benchmark.ipynb)

## 一句话总结
本节把"要不要采用某个投机解码方案"拆成一条可检查的流水线：先用 `classify_speculative_strategy` 认家族和资产，再用 `admit_speculative_candidate` 判断能不能启动，接着用 `build_speculative_pair_evidence` 在同目标、同请求、同采样配置下算加速比，最后用 `decide_speculative_candidate` 按质量优先、性能门槛的方向给出 `accept` / `tune` / `reject`。CPU 部分是可跑通的规则与测试，真实 GPU 部分则登记了 G0–G5 六组候选的执行、证据和决策。

## 为什么需要它：痛点是什么
初学者容易把"投机解码能加速"当成一句通用结论，但同一套策略在重复文本上可能明显变快，在普通问答上却因为草稿被频繁拒绝而得不偿失。更麻烦的是，EAGLE、DFlash、DSpark 这类候选依赖与目标模型配对的草稿权重，MTP 依赖目标模型自带预测层，N-gram 则不需要额外权重。于是常见困惑是：这个候选到底能不能装？装上了有没有真的走投机路径？跑出来的数和基线能不能直接比？

生活类比：公司招人。第一步先看候选人有没有必要的证件和工位（准入），第二步让他在**同一个岗位、同一批任务、同一套考核标准**下试工（同目标配对），第三步才按 KPI 决定录用、再培训还是淘汰（决策）。跳过前两步直接看"简历上写的速度"，很容易招错人。

## 核心问题
- 五类 proposer 家族（N-gram、MTP、EAGLE、DFlash、DSpark）各自需要什么模型资产？
- 一个候选从"能启动"到"值得采用"，中间要经过哪些状态？
- 为什么必须固定目标模型、请求集合和采样配置，比值才可解释？
- 吞吐、TTFT、E2E P95 和峰值显存的门槛方向分别是什么？质量失败能不能被更快的速度抵消？
- 真实 benchmark 里接受率、平均接受长度、`algorithm_observed` 这类证据应该怎么读，哪些不能当作采用结论？

## 解决思路：一步步看懂

**Step 1：先分清候选从哪里来。** `PROPOSER_FAMILIES` 是一张静态契约表，每个家族记录 `asset_kind`（资产种类）、`dependency`（依赖方式）和 `telemetry`（需要采集的遥测字段）。例如 `ngram` 的 `asset_kind` 是 `'none'`、依赖 `'retrieval'`；`mtp` 是 `'target_native'`、依赖 `'model_native'`；`eagle` 是 `'paired_checkpoint'`、依赖 `'feature_autoregressive'`；`dspark` 多了一个 `verified_tokens` 遥测。`classify_speculative_strategy('N-Gram')` 能工作的关键在 `_normalize_family`：先 `str(...).strip().lower()`，再把 `-` 和 `_` 都删掉，所以 `N-Gram`、`n_gram`、`ngram` 会落到同一个键。别名表还额外接受 `eagle3` 并映射到 `eagle`；未知名字直接抛 `ValueError`，避免把资产要求不同的方法混成一类。返回值用 `{'family': canonical, **PROPOSER_FAMILIES[canonical]}` 新建字典，而不是把原表交出去，防止调用方改坏全局配置。

**Step 2：按固定顺序判断"能不能启动"。** `admit_speculative_candidate` 是一个短状态机，检查顺序写在代码里就是优先级：

```python
if not candidate.get('enabled', False):        # 先看开关
    return {'run_status': 'skipped', ...}
if not candidate.get('backend_supported', False):
    return {'run_status': 'unsupported', ...}
if not candidate.get('asset_ready', False):
    return {'run_status': 'asset_missing', ...}
if not candidate.get('compatible', False):
    return {'run_status': 'incompatible', ...}
if estimated > limit:                          # 最后看容量
    return {'run_status': 'capacity_blocked', ...}
return {'run_status': 'runnable', ...}
```

为什么顺序重要？测试 `test_admission_states` 同时把 `backend_supported` 和 `asset_ready` 设为 `False`，断言结果必须是 `unsupported` 而不是 `asset_missing`——因为后端根本不支持这个方法时，准备资产也没有意义。顺序还保证只返回**第一个**阻断原因，便于运维按原因修。`_candidate()` 默认 `estimated_memory_gb=6.0`、`memory_limit_gb=10.5`，改成 12.0 就触发 `capacity_blocked`。

**Step 3：只有"可比配对"才能算比值。** `build_speculative_pair_evidence` 先看两条记录是否都 `run_status == 'completed'`，否则返回 `evidence_status='not_evaluated'`。接着检查三个固定字段 `target_snapshot`、`workload_key`、`sampling_key` 是否完全一致；不一致返回 `incomparable` 并列出 `mismatches`。然后检查关键指标 `throughput_tokens_per_s`、`ttft_p95_ms`、`e2e_p95_ms`、`peak_memory_mb`、`quality_ok` 是否缺失，再要求基线路径是 `target_only`、候选路径是 `speculative`，且 `mechanism_active is True`（候选确实提议过草稿）。任何一项不满足返回 `incomplete`。只有全部通过才进入 `comparable`，此时：

$$\text{throughput\_speedup}=\frac{\text{candidate.throughput}}{\text{baseline.throughput}},\qquad
\text{ttft\_ratio}=\frac{\text{candidate.ttft\_p95}}{\text{baseline.ttft\_p95}}$$

E2E 比值同理。注意 speedup 越大越好，而两个 ratio 越小越好，方向相反，这正是后面门槛要分开写的原因。

**Step 4：决策是"质量一票否决 + 性能逐项过线"。** `decide_speculative_candidate` 先处理证据状态：`not_evaluated` 原样返回；`incomplete` 返回 `tune`（去补证据）；`comparable` 但 `quality_ok` 为假则返回 `reject`，理由是 `quality_gate_failed`。只有质量过了才看四道门：

```python
'throughput': speedup >= min_throughput_speedup,
'ttft':       ttft_ratio <= max_ttft_regression_ratio,
'e2e_p95':    e2e_p95_ratio <= max_e2e_p95_regression_ratio,
'memory':     candidate_peak_memory_mb <= max_peak_memory_mb,
```

全部通过才 `accept`，否则 `tune`。`test_adoption_threshold_directions` 特意用**等号**验证边界：`speedup=1.1`、`ttft_ratio=1.1`、`e2e_p95_ratio=1.05`、显存恰好 10500.0 时判 `accept`；把任一项改成 1.09、1.11、1.06、10501.0 都只能 `tune`。这提醒你门槛使用的是 `>=` / `<=`，不是严格大于或小于。

**Step 5：真实 benchmark 的固定口径与结果怎么读。** 5.1 节把可变量收紧：`TARGET_MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'`、`DTYPE='bfloat16'`、`MAX_MODEL_LEN=2048`，`NUM_PROMPTS=8`、`WARMUP=2`、`REPEATS=1`、`MAX_TOKENS=64`、`TEMPERATURE=0.0`、`COMMON_SPECULATIVE_TOKENS=5`，并发只取 `[1]`。两类 workload 分别是 `speculative_general.jsonl` 和 `speculative_repetition.jsonl`，前者看通用生成，后者专门暴露 N-gram 的适用边界。六个候选身份固定为 `g0_target_only`、`g1_ngram`、`g2_mtp`、`g3_eagle`、`g4_dflash`、`g5_dspark`；G0 是唯一基线，其余是候选。`RUN_REAL_EXPERIMENT=False` 时只做预检，不会启动服务。

真实执行时，`_read_speculative_counters` 从 vLLM 的 Prometheus `/metrics` 读取 `vllm:spec_decode_num_drafts`、`..._num_draft_tokens`、`..._num_accepted_tokens`，用测量前后的增量算：

$$\text{acceptance\_rate}=\frac{\text{accepted}}{\text{proposed}},\qquad
\text{mean\_accepted\_length}=1+\frac{\text{accepted}}{\text{drafts}}$$

接受率是**诊断项**，notebook 明确说它不是独立采用门槛：接受率高不代表端到端变快，还要看草稿开销、验证成本和实际吞吐。5.5 节在 `TEMPERATURE=0.0` 时用 `Counter` 比较两条路径的 `(prompt, output_text)` 多重集合，重复项数量也要一致，这就是 `quality_ok` 的来源。5.6 节登记的正式结果 `rtx5070ti_speculative_ngram_smoke_v3` 是 `reject`：N-gram 有性能信号，但输出一致性未过；后续 `v5` 的 G1 后端未就绪、超时，没有候选性能数据，不能替代 v3 的结论。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| proposer（候选生成器） | 先猜几个后续词元的组件，猜完交给目标模型验证 |
| target（目标模型） | 真正决定输出分布的模型，也是基线和候选共同比较的对象 |
| N-gram / MTP / EAGLE / DFlash / DSpark | 五种候选来源：检索重复片段 / 模型自带预测层 / 特征草稿器 / 块扩散 / 半自回归块 |
| `asset_kind` | 候选需要的资产类别：`none`、`target_native`、`paired_checkpoint` |
| 准入（admission） | 启动前逐项检查开关、后端、资产、兼容和容量 |
| 同目标配对 | 目标版本、请求集合、采样配置相同，才能比较两条路径 |
| `mechanism_active` | 证据表明候选确实执行了投机提议，而不是普通生成 |
| TTFT / TPOT / E2E | 首词元延迟 / 每输出词元时间 / 端到端延迟 |
| 接受率 | 被目标模型接受的草稿词元数 ÷ 提议数，用于解释验证成本 |
| P95 | 样本中第 95 百分位，关注尾延迟而不是平均值 |
| `accept` / `tune` / `reject` / `not_evaluated` | 采用 / 调整后复测 / 拒绝 / 证据不足暂不评估 |

## 小白常见疑问

**Q1：为什么 `classify_speculative_strategy('unknown')` 要抛 `ValueError`，而不是返回一个"未知"字典？**
因为家族决定资产要求。如果未知名字被默认为某种通用家族，后续准入可能错误地认为"不需要草稿权重"，把一个根本跑不起来的方案放进 benchmark。测试 `test_family_classification` 用 `unknown` 明确检查这个失败路径。

**Q2：候选吞吐是基线的 1.25 倍，为什么还可能被拒绝？**
`test_pair_evidence_and_decision` 构造了 `throughput_tokens_per_s=125.0`、基线 100.0，`throughput_speedup` 四舍五入为 1.25，但把候选的 `quality_ok` 改成 `False` 后决策是 `reject`。代码顺序是先判质量、后看性能，质量失败不会被更快的速度抵消。

**Q3：为什么 `target_snapshot`、`workload_key`、`sampling_key` 任一不同就判 `incomparable`，而不是只提醒一下？**
不同目标模型、不同请求集合或不同温度/长度，算出来的延迟和吞吐没有共同分母。`test_pair_boundaries_and_missing_evidence` 逐个改这三个字段，断言状态都是 `incomparable`。若候选必须换目标模型，正确做法是为新目标另建基线，只比较组内 speedup。

**Q4：`mechanism_active=False` 为什么算 `incomplete` 而不是 `reject`？**
它说明"这次运行没有观察到投机提议"，可能是配置没生效、统计没导出，也可能是短测采样问题。证据不完整时先补测，而不是直接否定方案；只有质量或性能门槛明确失败才 `reject`。测试把 `execution_path='target_only'`、`mechanism_active=False`、`ttft_p95_ms=None` 都归入 `incomplete`。

**Q5：接受率很高，是不是就可以直接 `accept`？**
不是。接受率只解释"提议里有多少被接受"，不包含草稿生成、验证和调度开销。notebook 在 5.1 节注释"接受率是诊断项，不是独立采用门槛"，决策仍然只看质量、吞吐、TTFT、E2E P95 和显存。`test_adoption_threshold_directions` 的 evidence 里根本没有接受率字段。

## 动手验证：怎么知道自己做对了
先跑 `test_speculative_selection_template()`，它依次调用五个测试，全部通过后打印 `测试通过：proposer 分类、准入状态、配对证据和采用决策均符合契约。`。
- `test_family_classification`：`classify_speculative_strategy('N-Gram')['asset_kind'] == 'none'`、`'MTP'` 为 `target_native`、`'DSpark'['dependency'] == 'semi_autoregressive'`，`unknown` 必须抛 `ValueError`。
- `test_admission_states`：默认候选 `runnable`；分别破坏后端、资产、兼容、容量、开关，得到 `unsupported`、`asset_missing`、`incompatible`、`capacity_blocked`、`skipped`；后端与资产同时坏时仍是 `unsupported`。
- `test_pair_evidence_and_decision`：候选吞吐 125、基线 100，`round(throughput_speedup, 2) == 1.25`；门槛为 1.10 / 1.10 / 1.05 / 10500 时 `accept`；质量失败 `reject`；候选 `run_status='unsupported'` 时 `not_evaluated`。
- `test_pair_boundaries_and_missing_evidence`：三个固定字段分别改成 `'different'` 都得 `incomparable`；路径回退、机制未激活、TTFT 缺失都得 `incomplete`。
- `test_adoption_threshold_directions`：四个指标恰好等于门槛时 `accept`，任一越界变 `tune`，`evidence_status='incomplete'` 也是 `tune`。

notebook 的 CPU 单元没有保存输出，dump 里也没有真实 GPU 运行结果，因此以断言和 `run_status` / `decision` 的返回值为准，不要从本节推断某个家族的实际加速数字。5.6 节登记的 `v3` 结论是 `reject`、`v5` 是失败记录，二者都不是 CPU 测试跑出来的。
