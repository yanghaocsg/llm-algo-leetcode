# 69. 前缀缓存基准 — 小白讲解

> 对应 notebook：[`69_Prefix_Caching_Benchmark.ipynb`](./69_Prefix_Caching_Benchmark.ipynb)

## 一句话总结
多个请求共用一段长系统提示时，后面的人可以复用前面已经算好的 K/V，省掉部分预填充。本节先在 CPU 上实现分块、完整前缀键、连续命中、LRU 驱逐和指标统计，再补三个工程判断：`validate_cache_namespace` 核对缓存身份，`build_prefix_cache_pair_evidence` 区分"硬失败"和"缺证据"，`decide_prefix_cache_candidate` 用命中率、TTFT 和吞吐三个门槛共同决定是否采用。最后用 vLLM 的 G0（关闭缓存）/G1（开启缓存）配对实验验证，历史登记结论是 `tune`。

## 为什么需要它：痛点是什么
初学者看到两个请求开头都是同一段系统提示，就以为一定能复用缓存。实际至少有三道坎：块的内容相同但**历史不同**不能复用；模型权重、分词器、聊天模板、位置编码、缓存精度或租户不同，算出的 K/V 不能共享；后端到底有没有命中，不能靠 TTFT 变快反推，必须读命中计数。更麻烦的是，性能变好不代表可以上线——输出是否等价、重复运行是否稳定，必须先过。

生活类比：图书馆的复印机缓存。你带着同一本书的前 10 页来，复印机可以跳过已经复印过的部分；但如果上次那 10 页是另一个版本（历史不同），或者复印机换了墨盒/纸张规格（命名空间不同），就不能直接复用。而且"这次复印快了"不等于"印出来的内容对"。

## 核心问题
- 缓存命名空间包含哪些字段？为什么字段缺失也不能当成"一致"？
- 分块键为什么要带上"从开头到当前块"的完整历史？
- 连续前缀命中、请求级命中率、词元复用比例分别怎么算？有什么区别？
- LRU 容量和驱逐如何影响后续请求？为什么尾部残块不参与匹配？
- 硬失败、缺证据、输出不等价、性能不达标，分别应该给出什么决策？
- 真实 G0/G1 实验的自变量、因变量和读法是什么？

## 解决思路：一步步看懂

**第 1 步：把比较条件固定住（Step 1–2）。** G0 关闭前缀缓存、G1 开启，除此之外模型 revision、分词器、workload、生成长度、并发、重复次数全部一致，唯一变量是 `cache_policy`。notebook 用表格列出"缓存关闭与开启""输入分布""逻辑缓存模型""真实服务记录"四组对照，强调只有开关不同，结果才能归因于缓存。

**第 2 步：分块与"历史敏感"的键。** `complete_token_blocks(tokens, block_size)` 只切完整块：`range(0, len(tokens) - block_size + 1, block_size)`，尾部不足一块的 token 被丢掉。`_prefix_block_key(blocks, block_index)` 返回 `(block_index, tuple(blocks[:block_index+1]))`，即"块位置 + 从第 0 块到当前块的全部内容"。

notebook 的例子最能说明问题：每块 2 个 token，先处理 `[1,2,3,4]`，再处理 `[9,9,5,6]`，新请求 `[1,2,5,6]`。新请求的第 0 块 `[1,2]` 能命中，但第 1 块 `[5,6]` 的键是 `(1, ((1,2),(5,6)))`，而缓存里那条 `(1, ((9,9),(5,6)))` 历史不同，所以只能复用开头 2 个 token。`test_prefix_keys_do_not_mix_histories` 断言 `reused_tokens == 2`、`hit_rate == 1/3`。

**第 3 步：连续命中与 LRU。** `count_continuous_prefix_hits` 从第 0 块开始逐块查，遇到第一个 miss 就 `break`，后面偶然命中不计入——因为前缀复用必须是从头连续的一段。`write_blocks_with_lru(cache, blocks, clock, capacity_blocks)` 每次写入把时钟加 1，已存在的块也刷新访问时间；`len(cache) > capacity_blocks` 时反复删除 `min(cache, key=cache.get)`（最久未访问），并累计 `eviction_count`。测试 `test_lru_write_and_eviction` 用容量 2 写入两块触发一次驱逐，再访问旧块确认时钟变大且不再驱逐；`simulate_prefix_cache(..., block_size=2, capacity_blocks=4)` 断言复用 6 个 token 且至少发生一次驱逐；尾部残块案例 `[[1,2,3,4,5],[1,2,3,4,9]]` 只复用 4 个 token。

**第 4 步：`simulate_prefix_cache` 的指标定义。** 它对每条请求累加 `total_prompt_tokens`，用连续命中块数乘 `block_size` 得到 `reused_tokens`，再写回缓存。返回：

- `hit_rate = hit_requests / request_count`：**请求级**命中率，只要命中至少一块就算一次；
- `token_reuse_rate = reused_tokens / total_prompt_tokens`：**词元级**复用比例；
- `prefill_work_reduction` 与 `token_reuse_rate` 同值，表示可省的预填充比例；
- `miss_tokens`、`eviction_count`、`cache_size_blocks` 分别记录未复用词元、驱逐次数和当前缓存块数。

两个分母不能混用：命中 1 次但只复用 2 个 token，请求命中率可能很高，词元复用率却很低。notebook 在 Step 3 专门提醒这一点。

**第 5 步：TODO 1——命名空间门槛。** 参考实现：

```python
namespace_ok = not mismatches
change_ok = baseline.get('prefix_cache_enabled') is False and candidate.get('prefix_cache_enabled') is True
issues = list(mismatches) + ([] if change_ok else ['prefix_cache_enabled'])
return {'contract_ok': namespace_ok and change_ok, 'issues': issues}
```

`mismatches` 在骨架里已经算好，覆盖八个字段：`model_revision`、`tokenizer_id`、`chat_template`、`adapter_id`、`position_key`、`cache_dtype`、`attention_key`、`tenant_key`。判定条件是"任一侧为 `None` 或空串，或两侧不相等"——所以**两边都缺字段也算不一致**，`test_missing_identity_and_quality_cannot_pass` 逐字段 `pop` 后断言 `contract_ok` 为假。`change_ok` 要求基线明确关闭、候选明确开启，不能两边都开或两边都关。

**第 6 步：TODO 2——区分硬失败与缺证据。** 参考实现：

```python
hard_failures = list(contract['issues']) + failures
missing_evidence = list(missing)
if candidate.get('backend_cache_hit_observed') is not True:
    missing_evidence.append('backend_cache_hit')
```

`required_metrics` 是 `hit_rate`、`ttft_ms`、`throughput_tps`、`output_complete`、`cache_capacity_blocks`、`output_equivalent`、`repeat_stable`，两侧任一字段为 `None` 就记成 `baseline.xxx` 或 `candidate.xxx`。两者的语义不同：命名空间问题或运行 `failure` 是**硬失败**，直接 `reject`；缺指标、缺后端命中证据是**缺证据**，应该 `tune` 去补采，不能补成 0。返回值里还算了 `ttft_gain_ms = baseline.ttft_ms - candidate.ttft_ms`（越小越好，所以 `reverse=True`）、`throughput_gain = candidate - baseline`，以及 `output_ok`、`output_equivalent`、`repeat_stable`。

**第 7 步：TODO 3——三门槛合取。** 参考实现：

```python
reuse_ok = (evidence['hit_rate'] >= min_hit_rate
            and evidence['ttft_gain_ms'] >= min_ttft_gain_ms
            and evidence['throughput_gain'] >= min_throughput_gain)
```

`decide_prefix_cache_candidate` 的判定顺序是：有硬失败 → `reject`；有缺证据 → `tune`；`output_ok` 为假 → `reject`；输出不等价或重复不稳定 → `tune`；最后三项性能门槛同时满足才 `accept`，否则 `tune` 并检查 workload 与缓存成本。

测试 `test_cache_project_contract_and_evidence` 用基线 `hit_rate=0.0, ttft=130, throughput=80`、候选 `0.65, 95, 84`，门槛 `(0.5, 20.0, 2.0)`：命中 0.65≥0.5，TTFT 改善 35≥20，吞吐改善 4≥2，所以 `accept`。把 `backend_cache_hit_observed` 改成 `False` → 缺证据 → `tune`；把 `output_complete` 改成 `False` → `reject`。`test_missing_identity_and_quality_cannot_pass` 再验证：单项性能不足（TTFT 125，改善只有 5；或吞吐 81，改善只有 1）都只能 `tune`，不能因为另一项好就放行。

**第 8 步：候选排序。** `compare_prefix_cache_to_baseline` 要求两条记录 `workload_id` 相同，否则 `ValueError`；`prefix_cache_rank_key` 返回 `(-hit_rate_gain, ttft_delta_ms, overhead_delta)`，即先要复用增益大，再要 TTFT 低、开销小。`select_prefix_cache_candidate` 先筛 `evidence_ok and quality_ok`，再排序。测试里 `large` 命中增益 0.7、`small` 0.5，即使 `large` 开销更高也排第一；`blocked` 即使伪造 `hit_rate_gain=99`，因证据/质量不合格仍被排除，`selected is None`。

**第 9 步：真实 G0/G1 实验与怎么读。** 5.1 固定：模型 `Qwen/Qwen2.5-0.5B-Instruct`，workload `prefix_reuse_long.jsonl`，`NUM_PROMPTS=30`、`CONCURRENCY=1`、`REPEATS=3`、`WARMUP=5`、`MAX_TOKENS=64`、`MAX_MODEL_LEN=2048`、`GPU_MEMORY_UTILIZATION=0.80`。`observe_prefix_cache` 从 `/metrics` 里筛含 `prefix` 和 `cache` 的行，用正则读 `prefix_cache_queries_total`、`prefix_cache_hits_total`、`num_gpu_blocks`，并解析 `enable_prefix_caching="True|False"`；日志里再找 `Prefix cache hit rate: x%`。`read_prefix_result` 把 TTFT/TPOT 的 p50/p95、输出吞吐、E2E p95、成功/失败请求数规整成统一字段；`read_prefix_manifest` 检查两组配置签名是否唯一（`shared_config_ok`）。

最终决策 `decide_prefix_cache_evidence` 的顺序是：证据不可读 → `tune`；有 backend 失败 → `reject`；两组或共同配置不全 → `tune`；请求未完整返回 → `reject`；输出等价性未通过（`exact_output_equivalence.status != 'passed'` 或 `matched != total`）→ `tune`；重复稳定性未通过 → `tune`；实际开关不是"关/开" → `reject`；缺命中率、命中来源或容量 → `tune`；最后才比较 `ttft_p95` 下降且吞吐上升 → `accept`，两者同时退化 → `reject`，信号不一致 → `tune`。

历史登记的 `rtx5070ti_prefix_cache_v4` 结论是 `tune`：直接命中与性能改善已记录，但 30 组输出只有 12 组相同，缓存开启后 5 个重复提示组中仅 1 组输出保持一致，需要先定位差异；容量压力、独立显存峰值和驱逐压力仍待测。这个例子说明"性能收益不能替代正确性核对"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 前缀缓存 | 复用请求开头已算好的 K/V，跳过重复预填充 |
| 缓存命名空间 | 决定 K/V 能否共享的一组身份字段 |
| 完整块 | 长度恰好等于 `block_size`、可参与匹配的 token 块 |
| 完整前缀键 | "块位置 + 从开头到当前块的全部内容"，防止跨历史误复用 |
| 连续前缀命中 | 从第 0 块开始不间断的命中，遇到首个 miss 即停止 |
| LRU | 容量满时淘汰最久未被访问的条目 |
| 请求级命中率 | 命中请求数 / 总请求数 |
| 词元复用比例 | 复用词元数 / 总提示词元数 |
| 硬失败 | 命名空间不一致或运行失败，直接拒绝比较 |
| 缺证据 | 指标或后端命中没采到，应补测而不是补零 |
| TTFT | 请求发出到首次返回内容的时间 |
| `shared_config_ok` | G0/G1 的模型、backend、dtype、workload 等配置签名一致 |

## 小白常见疑问

**Q1：两个请求第 1 块内容都是 `[5,6]`，为什么不能复用？**
因为键里带了完整历史。`_prefix_block_key` 是 `(block_index, blocks[:block_index+1])`，前一块不同，整个键就不同。`test_prefix_keys_do_not_mix_histories` 用 `[1,2,3,4]`、`[9,9,5,6]`、`[1,2,5,6]` 验证只能复用 2 个 token。

**Q2：为什么尾部不足一个块的 token 不参与匹配？**
因为完整块缓存按固定块大小组织和索引，残块没有稳定的块边界，notebook 的 `complete_token_blocks` 直接把它排除。测试 `[[1,2,3,4,5],[1,2,3,4,9]]` 在 `block_size=4` 下只复用第一块，共 4 个 token。

**Q3：命中率 0.65 和词元复用率有什么区别？**
前者按请求数算，后者按词元数算。一个请求命中一块短前缀，请求命中率可能很高，但省下的预填充比例很小。`simulate_prefix_cache` 同时返回 `hit_rate` 和 `token_reuse_rate`，读结果时要说明用的是哪个分母。

**Q4：字段两边都缺失，为什么不算"一致"？**
因为无法证明一致。`validate_cache_namespace` 把 `None`/空串直接记入 `mismatches`，`test_missing_identity_and_quality_cannot_pass` 对 `model_revision`、`cache_dtype`、`tenant_key`、`attention_key` 逐个删除后断言契约不通过。缺证据应补测，而不是默认相同。

**Q5：缓存让 TTFT 变快、吞吐变高，为什么历史结论还是 `tune`？**
因为输出等价性和重复稳定性没过。`decide_prefix_cache_evidence` 在看到 `exact_output_equivalence` 或 `repeat_stability` 不通过时直接返回 `tune`；v4 登记里 30 组只有 12 组输出相同，5 个重复提示组只有 1 组稳定。性能收益不能替代正确性。

**Q6：`prefix_cache_hit_rate` 能从 TTFT 反推吗？**
不能。notebook 5.2 明确写"不能由 TTFT 或吞吐反推命中率"，所以 `observe_prefix_cache` 必须读 `/metrics` 的 `prefix_cache_queries_total`/`prefix_cache_hits_total` 或服务日志，并记录 `hit_rate_evidence` 来源。

## 动手验证：怎么知道自己做对了

`test_prefix_cache_benchmark_template` 依次运行六个测试，全过打印 `测试通过：缓存命名空间、后端证据与采用状态均通过。`；TODO 未完成会打印 `请先完成 TODO 部分的代码！` 并抛出。

- `test_continuous_block_hits`：已有两块缓存时，`[(1,2),(3,4),(5,6)]` 返回 2；`[(9,9),(3,4)]` 因第 0 块未命中返回 0。
- `test_lru_write_and_eviction`：容量 2 写入触发 1 次驱逐；重复访问刷新时钟且不再驱逐；`_prefix_cache_sequences()` 在 `block_size=2, capacity_blocks=4` 下 `reused_tokens == 6` 且 `eviction_count > 0`；尾部案例复用 4；非法 `block_size=0`、`capacity_blocks=0` 抛 `ValueError`。
- `test_candidate_ranking_and_contract`：`large` 被选中；`blocked` 即使命中增益 99 也 `selected is None`；`summarize_prefix_cache` 返回 `best_hit_rate_run='small'`；不同 `workload_id` 比较抛 `ValueError`。
- `test_cache_project_contract_and_evidence`：正常配对 `contract_ok`；改 `chat_template` 后不通过；门槛 `(0.5,20,2)` 下 `accept`；无后端命中 → `tune`；`output_complete=False` → `reject`。
- `test_prefix_keys_do_not_mix_histories`：`reused_tokens == 2`、`hit_rate == 1/3`。
- `test_missing_identity_and_quality_cannot_pass`：缺身份不通过；`output_equivalent`/`repeat_stable` 为假 → `tune`；`baseline.ttft_ms` 为 `None` 时 `missing_evidence` 含该字段；TTFT 只改善 5 或吞吐只改善 1 都 → `tune`。

CPU 测试没有保存执行输出，以断言为准。真实 G0/G1 实验默认 `RUN_REAL_BACKEND=False`，不启动 vLLM；只有改为 `True` 且完成 `INF-VLLM-011-CU128` 环境后才运行。5.4 写出的 manifest 里的 `decision` 是运行前占位，最终结论应以 5.6 的派生决策和单独的证据记录为准，不能把 manifest 或原始 JSON 改写成结论。
