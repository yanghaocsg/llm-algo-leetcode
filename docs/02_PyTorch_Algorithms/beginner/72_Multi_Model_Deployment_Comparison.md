# 72. Multi Model Deployment Comparison | 单卡推理部署与 Backend 对比项目 · 小白入门

> 配套 notebook：[72. 单卡推理部署与 Backend 对比项目](../72_Multi_Model_Deployment_Comparison.md) ｜ 难度：Hard ｜ 环境：CPU 完成候选审计与决策；NVIDIA 单 GPU 完成可选 backend benchmark

## 一句话说清

把一个模型部署到一张显卡上，要同时确认模型身份、显存够不够、各项指标达标。这本 notebook 教你把这些检查串起来，给出采用、调优或拒绝的结论。

## 为什么需要它

前面几本 notebook 各留下一类结果：66 的共享推理基线，67 的量化结果，以及 68 投机解码、69 前缀缓存、70 服务调度各自的候选结论。真要部署时，问题变成另一个：把模型放到一张显卡上，这一套配置能不能稳定运行，出了问题能不能退回去。

判断时常见的错法有：

- 看模型名字就当成同一个部署对象。候选换了版本、分词器或模型产物，其实已经不是同一个东西。
- 看后端名字就以为策略生效了。只有加载记录、启动配置和运行指标都和声明一致，才算数。
- 启用了一项策略，却没有对应的测量结果，照样拿来比较。
- 只问“哪个后端最快”。

比如候选顺手把并发从 1 改成 4，却没有声明这项变化。这时基线和候选的差别不只来自后端，结果没法归因到一处。

notebook 的说法是：项目输出不是“最快后端”标签，而是一份能说明采用条件、缺失证据和回滚入口的交接资产。这本 notebook 的主线在 CPU 上用简化的记录，把四个判断练熟：两组运行能不能配对、上游证据属于哪一类、门槛过没过、最后状态是什么。Step 5 再读取真实的基线和专项证据做汇总。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 模型产物（artifact） | 实际拿去部署的权重文件或量化产物。模型名字相同，产物可能不同，所以还要核对版本、摘要和格式 |
| 执行后端（backend） | 真正运行推理的服务实现。vLLM 是 notebook 里默认可运行的后端，SGLang 是可选对照 |
| 工作负载（workload） | 固定的请求集合和生成设置，包括请求集、生成长度、采样、并发和重复次数 |
| 容量账本 | 给一张卡的显存记账：常驻的模型产物、KV Cache、运行时工作区和安全余量各占多少 |
| 上游证据 | 前面专项 notebook 留下的结果，比如 68 的投机解码记录、69 的前缀缓存记录、70 的调度记录 |
| 回滚基线 | 新候选出问题时可以退回去的那套参考配置 |
| `accept` / `tune` / `reject` | 三种部署状态：采用、调优（补证据或调整候选）、拒绝 |

## 核心思路

输入有三样：基线和候选两份部署记录、每个已启用策略的上游结果、一组服务目标。处理分四步，和四个 TODO 一一对应：

```text
配对能不能比 → 上游证据属于哪一类 → 门槛过没过 → 最终状态
```

第一步是配对。有六个固定字段必须一致：`model_id`、`revision`、`tokenizer_revision`、`artifact_digest`、`dtype`、`workload_key`。允许变化的是四个受控字段：`backend`、`enabled_strategies`、`max_model_len`、`concurrency`，但实际变化的集合必须和候选声明的 `declared_changes` 完全一致，而且不能为空。另外候选的 `capacity_status` 必须是 `ready`。

```python
baseline = {'model_id': 'm', 'revision': 'c1', 'backend': 'vllm', 'concurrency': 1}

def compare(candidate, declared):
    fixed, controlled = ('model_id', 'revision'), ('backend', 'concurrency')
    changed = [f for f in controlled if baseline[f] != candidate[f]]
    identity_ok = all(baseline[f] == candidate[f] for f in fixed)
    changes_ok = bool(changed) and sorted(changed) == sorted(declared)
    return changed, identity_ok and changes_ok

print(compare({**baseline, 'backend': 'sglang'}, ['backend']))
print(compare({**baseline, 'backend': 'sglang', 'concurrency': 4}, ['backend']))
print(compare({**baseline, 'revision': 'c2', 'backend': 'sglang'}, ['backend']))
```

```text
(['backend'], True)
(['backend', 'concurrency'], False)
(['backend'], False)
```

例子只保留了两个固定字段和两个受控字段。第一种只换了后端并且做了声明，可以比较。第二种顺手把并发从 1 改成 4 却没声明，实际变化的集合和声明不一致。第三种换了模型版本，固定字段对不上，字段再相似也不算配对。

第二步是给上游证据分类。对每个启用的策略看它的记录：没有记录的进 `missing_evidence`，决策是 `reject` 的进 `hard_failures`，决策既不是 `accept` 也不是 `reject` 的进 `incomplete_evidence`。

第三步是门槛，注意每个量的方向：

```text
质量   quality_status 为 passed，且 success_rate 不低于下限
服务   TTFT、TPOT 的 P95 不超过上限，吞吐不低于下限
容量   峰值显存不超过上限，显存余量大于 0
回滚   rollback_ready 为 True
```

TTFT 是首词元延迟，TPOT 是单词元时间，P95 是第 95 百分位。缺了某个指标时，参考答案让对应的门槛不通过。

第四步按顺序出状态，前面的条件优先：

```text
有硬失败（上游拒绝、后端失败、配对不成立）          → reject
有证据缺口（未知策略、缺记录、待补），或有门槛没过  → tune
以上都没有                                      → accept
```

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从固定模型产物到交接清单的五个阶段；输出不是“最快后端”标签，而是一份交接资产 | 读 |
| Step 2 | 五个检查对象各用什么证据、怎样才算通过；容量账本把显存分成哪几份 | 读 |
| Step 3 | 硬失败、证据缺口、门槛的先后顺序；每一层证据在什么情况下 `tune` 或 `reject` | 读 |
| Step 4 · TODO 1 | 核对部署配对 | 写 `compare_deployment_pair`：补 3 行，`identity_ok`、`changes_ok`、`capacity_ok` |
| Step 4 · TODO 2 | 汇总专项证据 | 写 `summarize_strategy_evidence`：补 2 行，`rejected` 是决策为 `reject` 的策略，`incomplete` 是决策不属于这两种的策略 |
| Step 4 · TODO 3 | 检查部署门槛 | 写 `evaluate_deployment_gates`：补 4 个布尔值，质量、服务、容量、回滚各一个，注意上下限方向 |
| Step 4 · TODO 4 | 形成部署状态 | 写 `decide_single_gpu_deployment`：补一个三分支判断，按顺序给出 `decision` 和 `next_action` |
| 测试 | 检查配对里的未声明变化、四类上游证据状态、指标方向、容量边界和决策优先级 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（复测可选） | 读取 66 的共享基线和 68、69、70 的专项证据做汇总，必要时做一次新鲜复测 | 默认是 `RUN_FRESH_BENCHMARK = False`，只做只读汇总，不执行真实 GPU 实验；改为 `True` 需要 NVIDIA 单 GPU 和 `INF-VLLM-011-CU128` 环境 |

Step 1–4 只需要 CPU。Step 5 的可选复测需要 NVIDIA 单 GPU；默认路径只做只读汇总，不占用 GPU。

## 容易踩的坑

- **字段相似就当成配对**：模型、版本、分词器版本、产物摘要、数据类型、工作负载这六个固定字段必须一致，差一个产物摘要也不行。
- **变化没有声明**：实际变化的字段集合必须和 `declared_changes` 完全一致，而且不能为空。改了并发却没声明，配对就不成立。
- **缺记录、待补、已拒绝混在一起**：没有记录表示还没测，`tune` 表示证据不全，`reject` 表示已确认失败。三者要分别放进 `missing_evidence`、`incomplete_evidence`、`hard_failures`。
- **门槛方向写反**：延迟和峰值显存用上限，吞吐、成功率和显存余量用下限。质量状态和回滚入口必须显式通过。
- **顺序颠倒**：先看硬失败，再看证据缺口和门槛。有硬失败时，其他项都过也是 `reject`。

## 读完后自查

- [ ] 能列出配对时必须一致的六个固定字段。
- [ ] 候选只声明换后端，却同时改了并发，能判断配对是否成立。
- [ ] 能说出缺记录、`tune`、`reject` 三种上游证据分别进哪个列表。
- [ ] 能说出延迟、吞吐、成功率、峰值显存、显存余量各自用的是上限还是下限。

## 下一步

- 打开 notebook：[72. 单卡推理部署与 Backend 对比项目](../72_Multi_Model_Deployment_Comparison.md)
- 建议先读：[66. 单卡推理测量基线 · 小白入门](./66_Inference_Performance_Comparison.md)、[67. 量化推理与部署 · 小白入门](./67_Quantized_Inference_and_Deployment.md)
- 建议先读：[68. 投机解码机制选型基准 · 小白入门](./68_Speculative_Decoding_Benchmark.md)、[69. 前缀缓存基准 · 小白入门](./69_Prefix_Caching_Benchmark.md)
- 做完之后：[73. 训练性能分析 · 小白入门](./73_Training_Performance_Analysis.md)
