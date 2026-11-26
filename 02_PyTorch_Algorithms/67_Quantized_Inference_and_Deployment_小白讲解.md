# 67. 量化推理与部署 — 小白讲解

> 对应 notebook：[`67_Quantized_Inference_and_Deployment.ipynb`](./67_Quantized_Inference_and_Deployment.ipynb)

## 一句话总结
量化把权重存得更小，但真实请求不一定更快：加载器要解码、计算 kernel 可能更换，收益还受批量和生成长度影响。本节用 `validate_deployment_pair` 检查"同源、同请求、同环境"的配对契约，用 `summarize_deployment_pair` 把 TPOT、吞吐、显存、质量换算成方向明确的比值，再用 `decide_quantized_deployment` 组合执行证据、质量和资源门槛，输出 `pending`、`reject`、`tune` 或 `accept`。

## 为什么需要它：痛点是什么
初学者常把"4-bit 量化"直接等同于"显存省 4 倍、速度也快 4 倍"。实际部署时会遇到反例：权重文件确实从 16 GB 变成 4 GB，但服务启动后加载器要反量化，计算 kernel 走的是另一条路径，首词元延迟可能更高；小 batch、短输出时解码开销占比大，速度甚至可能回退。更麻烦的是，有人拿量化模型和另一个浮点模型比，结果模型版本、分词器、并发数、硬件全都不同，这样的"加速"根本不能归因于量化。

生活类比：把羽绒服抽真空打包。箱子确实变小了（权重字节少），但每次穿之前要拆开、拍松（反量化/换 kernel），赶时间出门时不一定更快。要比"打包是否值得"，必须同一件衣服、同一个出门场景、同一辆车，只改变"是否抽真空"。

## 核心问题
- 什么条件下浮点基线与量化候选才算"可配对的部署实验"？
- TPOT、吞吐、显存、质量这四个指标，比值方向分别是什么？哪个越大越好、哪个越小越好？
- 显存缺失时为什么保留 `None`，而不是当成 0？
- `pending`、`reject`、`tune`、`accept` 四种状态按什么优先级判断？
- 为什么"质量通过 + 资源至少满足一项"就可以进入下一轮，而不是要求三项全赢？

## 解决思路：一步步看懂

**Step 1：配对契约先于性能数字。** `validate_deployment_pair` 接收三份记录：上游 82 的准入结果 `gate`、浮点基线 `baseline`（G0）、量化候选 `candidate`（G1）。先算四个原子事实：`gate_ready` 要求 `gate['decision'] == 'accept'`；`artifact_matches` 要求候选的 `artifact_id` 与门禁一致；`model_revision_matches` 要求候选的 `model_revision` 与门禁一致；`fixed_mismatches` 遍历 `FIXED_PAIR_FIELDS`（`model_revision`、`tokenizer_revision`、`workload_id`、`backend`、`hardware_id`、`cache_policy`），只要基线该字段为空或两边不相等就记下来。

**Step 2：TODO 1——把血缘和固定条件合起来。** 参考答案：

```python
lineage_ready = gate_ready and artifact_matches and model_revision_matches
fixed_fields_ready = not fixed_mismatches
...
'ready': lineage_ready and fixed_fields_ready
```

`issues` 仍然逐项记录：门禁没接受记 `artifact_gate_not_accepted`，产物 ID 不一致记 `artifact_id_mismatch`，模型 revision 不一致记 `model_revision_mismatch`，固定字段漂移记 `fixed_field_mismatch:<字段>`。为什么要拆成 `lineage_ready` 和 `fixed_fields_ready` 两个变量？因为测试要能分别定位"来源不对"和"运行条件漂移"：`test_deployment_pair_contract` 把 `hardware_id` 改成 `gpu-b`，要求 `fixed_mismatches == ['hardware_id']`；把门禁的 `model_revision` 改成 `other-revision`，要求 `issues` 含 `model_revision_mismatch`；把两边的 `workload_id` 都设为 `None`，要求 `ready=False`。

**Step 3：TODO 2——四个指标的方向。** 参考答案先把每个指标变成候选/基线的比值：

```python
tpot_ratio = _positive_ratio(float(candidate["tpot_ms"]), float(baseline["tpot_ms"]))
throughput_ratio = _positive_ratio(float(candidate["throughput_tps"]), float(baseline["throughput_tps"]))
memory_reduction_ratio = 1.0 - _positive_ratio(float(candidate["peak_memory_mb"]), float(baseline["peak_memory_mb"]))
quality_ratio = _positive_ratio(float(candidate["quality"]), float(baseline["quality"]))
```

手算测试数据：TPOT $18/20=0.9$，小于 1 表示候选更快；吞吐 $56/50=1.12$，大于 1 表示候选更吞吐；显存 $1-5600/8000=1-0.7=0.3$，表示降低 30%；质量 $10.4/10=1.04$。这里质量用的是"越低越好"的 NLL 一类指标，所以比值大于 1 表示退化；如果换成准确率这种"越高越好"的指标，判断方向要反过来。`_positive_ratio` 要求基线严格大于 0、候选有限且非负，`NaN`、`inf`、`True` 都会被拒绝——`test_paired_metric_ledger` 专门测了这三种非法输入。

**Step 4：显存缺失为什么是 `None`。** 如果基线或候选任一 `peak_memory_mb` 为 `None`，`memory_reduction_ratio` 就保持 `None`。这不是"省了 0%"，而是"没测到，无法判断"。测试断言 `summarize_deployment_pair(baseline, {**candidate, "peak_memory_mb": None})["memory_reduction_ratio"] is None`。后面 `decide_quantized_deployment` 里 `memory_ready` 会因此为假，但如果延迟或吞吐达标，仍可能 `accept`。

**Step 5：TODO 3——三种门槛和状态优先级。** 参考答案：

```python
evidence_ready = backend_observed and quantized_path_observed and no_fallback
quality_ready = comparison.get("quality_ratio", float("inf")) <= max_quality_ratio
resource_ready = latency_ready or throughput_ready or memory_ready
```

`evidence_ready` 是合取：必须在真实后端跑过（`real_backend is True`）、确实走了量化路径（`quantized_path is True`）、没有回退（`fallback is False`）。`quality_ready` 是质量比值不超过 `max_quality_ratio`；`resource_ready` 是延迟、吞吐、显存三项至少一项达标。默认门槛是 `max_tpot_ratio=1.20`、`min_throughput_ratio=0.90`、`min_memory_reduction_ratio=0.20`、`max_quality_ratio=1.10`，也就是说允许 TPOT 最多慢 20%、吞吐最多降 10%、显存至少降 20%、质量最多退化 10%。

判断顺序固定为：配对不通过 → `reject`；执行证据不全 → `pending`；质量超预算 → `reject`；资源一项都没达到 → `tune`；全过 → `accept`。注意 `pending` 排在质量之前：没有真实执行证据时，连"这个数值是不是量化路径测出来的"都不知道，不能直接判质量失败。`test_deployment_state_machine` 覆盖四条路径：完整证据 `accept`；`real_backend=False` 变 `pending`；`quality_ratio=1.3` 变 `reject`；`tpot_ratio=1.4`、`throughput_ratio=0.7`、`memory_reduction_ratio=0.05` 三项全不达标变 `tune`。

**Step 6：真实实验怎么配对、怎么读。** Step 5 的设计是浮点 G0 与量化 G1 跑同一份固定请求集，固定模型与分词器版本、后端、请求、缓存策略和硬件，只改变权重产物和对应的低精度加载路径。为了减少"谁先跑谁占便宜"的顺序偏差，三轮先后是浮点→量化、量化→浮点、浮点→量化；每组取 TPOT 的 P50 算配对比值，再取三轮比值的中位数；质量沿用上游 82 的候选评测；当前服务适配器没有提供可用峰值，所以显存降幅保留为空。`prepare` 只读预检、不启动 GPU；`measure` 才执行固定矩阵并保存原始请求、日志和失败；`close` 只消费已保存证据，内容一致就复用、冲突就停止。只有 `accept` 才生成 67 的部署引用交给 83。notebook 的登记结果明确说"尚无本页新协议的真实 GPU 结果"，所以这里不引用任何实测加速数字。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 量化产物 | 已经量化好的权重文件及其配置，必须能追溯到同一基座模型 |
| G0 / G1 | 浮点基线 / 量化候选 |
| 配对契约 | 两组必须在模型、分词器、请求、后端、硬件、缓存上一致 |
| 血缘（lineage） | 产物 ID 和模型 revision 能对上上游 82 的准入记录 |
| TPOT | 首词元之后平均每词元耗时，越小越好 |
| 吞吐比 | 候选 tokens/s ÷ 基线 tokens/s，越大越好 |
| 显存降幅 | $1-\text{候选峰值}/\text{基线峰值}$，越大越好 |
| 质量比 | 候选质量 ÷ 基线质量；对 NLL 这类指标越大越差 |
| 回退（fallback） | 量化路径没走成，退回浮点计算 |
| `pending` | 证据还不完整，暂时不能判断 |
| `tune` | 执行和质量没问题，但当前负载下资源门槛没过 |
| `accept` | 质量、执行路径和至少一项资源门槛都成立 |

## 小白常见疑问

**Q1：为什么质量比值对 NLL 是越大越差，对准确率却要反过来？**
因为比值本身没有方向，方向来自指标语义。题目明确说"本题和默认项目采用负对数似然（NLL）一类越低越好的指标，候选与基线的比值越大表示退化越多"；`max_quality_ratio=1.10` 就是"最多退化 10%"。如果换成准确率，应该用基线/候选或者改判断规则，不能直接套 `<= 1.10`。

**Q2：资源三项只要满足一项就能 `accept`，会不会太宽松？**
这是本项目的阶段目标：先判断"量化值不值得进入下一轮复测"，不是宣布"全面更优"。notebook 允许一定回退（TPOT 比不高于 1.20、吞吐比不低于 0.90），只要延迟、吞吐、显存有一项明显改善就算过线；真正是否加速还要读比值相对 1 的方向和所有服务约束。

**Q3：为什么 `pending` 排在质量判断前面？**
因为没有真实执行证据时，你无法确认这些指标是量化路径跑出来的，还是悄悄回退到浮点了。`evidence_ready` 要求 `real_backend`、`quantized_path`、`fallback is False` 同时成立；证据不全就先 `pending`，补测后再判断质量。

**Q4：显存没测到，为什么不能按 0 降幅算？**
按 0 算会把"未知"误判成"没有收益"，从而错误地卡掉一个可能合格的候选。代码把 `memory_reduction_ratio` 保留为 `None`，`memory_ready` 自然为假；如果 TPOT 或吞吐达标，仍然可以 `accept`。测试专门断言了缺测时该字段是 `None`。

**Q5：`validate_deployment_pair` 里为什么要同时检查 `artifact_id` 和 `model_revision`？**
`artifact_id` 保证候选就是上游 82 准入的那个量化文件，`model_revision` 保证它和门禁记录来自同一版基座模型。只查一个都可能出现"拿 A 模型的量化文件配 B 模型的 revision"的错配，`test_deployment_pair_contract` 用 `other-revision` 专门测了这种情况。

## 动手验证：怎么知道自己做对了
`run_quantized_deployment_tests` 依次调用三个测试，全过才打印 `✅ 配对契约、指标方向和部署状态机均已验证。`
- `test_deployment_pair_contract`：默认配对 `ready=True`；`hardware_id` 改成 `gpu-b` 后 `fixed_mismatches == ['hardware_id']`；门禁 model revision 改成 `other-revision` 后 `issues` 含 `model_revision_mismatch`；两边 `workload_id=None` 时 `ready=False`。
- `test_paired_metric_ledger`：结果严格等于 `{'tpot_ratio': 0.9, 'throughput_ratio': 1.12, 'memory_reduction_ratio': 0.3, 'quality_ratio': 1.04}`；候选显存设 `None` 时显存降幅为 `None`；`_positive_ratio` 对 `nan`、`inf`、`True` 都抛 `ValueError`。
- `test_deployment_state_machine`：完整证据 `accept`；`real_backend=False` 为 `pending`；`quality_ratio=1.3` 为 `reject`；TPOT 1.4、吞吐 0.7、显存降幅 0.05 为 `tune`。

这些测试单元没有保存执行输出，Step 5 默认 `RUN_STAGE='prepare'` 也只做预检。判断是否做对以断言和打印文案为准；真实 GPU 结论要以新 `RUN_LABEL` 生成的记录为准，不能拿其他标签的历史结果替代。
