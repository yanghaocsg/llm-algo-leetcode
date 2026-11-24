# 68. Speculative Decoding Benchmark Beginner | 投机解码机制选型基准 · 小白入门

> 配套 notebook：[68. 投机解码机制选型基准](./68_Speculative_Decoding_Benchmark.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

投机解码先猜出几个后续词元，再交给目标模型验证，希望生成得更快。这本 notebook 教你判断一种猜法能不能启动、是否真的更快，以及能不能采用。

## 为什么需要它

大模型平常一个词元一个词元往下写。投机解码换了个做法：候选生成器先提出几个可能的后续词元，目标模型来验证，接受的留下。同一套策略，可能在重复文本上减少生成时间，却在普通问答上得不偿失。有些方法还依赖特定的模型结构，或者要加载与目标模型配对的草稿权重。选型时常见的错法有：

- 候选没真正走投机路径，拿到数字就去比。
- 只看接受率。接受率只能解释成本，不能单独证明加速。
- 候选有性能收益，但输出变了。质量不通过时，性能再好也不能采用。

notebook 登记的一次真实结果就是最后这种：N-gram 候选有性能信号，但输出一致性未通过，结论是 `reject`。

这本 notebook 的主线解决其中“判断”的部分：在 CPU 上用模拟记录，把分类、准入、配对、决策四步规则练熟。真实 GPU 上的采集放在可选的 Step 5。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 候选生成器（proposer） | 提出可能的后续词元的那一方。来源随家族不同：检索重复片段、用目标模型内置的预测层，或加载配对的草稿权重 |
| 目标模型（target） | 负责验证候选词元的模型。基线和候选要用同一个目标模型才能配对 |
| 执行后端（backend） | 真正运行推理的实现，决定哪些候选家族可以启动。有通用投机接口，不等于每个家族都支持 |
| 基线 / 候选（baseline / candidate） | 基线是只用目标模型单独生成的参照，记作 G0；候选是五类投机方案，记作 G1–G5 |
| 配对（pair） | 基线和候选各跑一次得到的两条记录，是比较的基本单位 |
| 工作负载（workload） | 固定的输入请求集合。notebook 用普通问答和重复文本两种 |
| `accept` / `tune` / `reject` / `not_evaluated` | 四种决策：采用、补测或调整、拒绝、不评估（连可比的证据都还没有） |

## 核心思路

输入有四样：候选家族表、每个候选的检查事实、基线和候选的两条运行记录、一组门槛。处理分四步，每步只回答一个问题，最后输出四种决策之一和原因。

```text
家族是什么 → 能不能启动 → 证据能不能比 → 能不能采用
```

第一步看候选从哪里来。notebook 列了五类：N-gram 从提示或历史输出里找重复片段，不需要额外草稿权重；原生多词元预测（MTP）用目标模型内部的预测层；EAGLE、DFlash、DSpark 各自需要与目标版本匹配的草稿权重。

第二步按固定顺序检查，遇到第一个不满足的条件就停下，并返回对应的运行状态。下面是参考答案里的顺序：

```text
enabled 为假                              → skipped（未启用）
backend_supported 为假                    → unsupported（不支持）
asset_ready 为假                          → asset_missing（缺资产）
compatible 为假                           → incompatible（不兼容）
estimated_memory_gb 大于 memory_limit_gb  → capacity_blocked（容量受阻）
全部通过                                  → runnable（可尝试启动）
```

`failed`（执行失败）不在这张表里：它是启动之后才可能出现的状态，静态检查给不出来。

第三步核对基线和候选的两条记录能不能比。先看两边是否都已完成，再看三个固定字段是否一致：`target_snapshot`（目标模型的固定版本）、`workload_key`（请求集合）、`sampling_key`（采样配置）。最后看候选是否真的走了投机路径、关键指标齐不齐。

第四步做决策，质量排在性能前面。比值都是“候选 ÷ 基线”，但方向不同。首词元延迟（TTFT）是等到第一个词元的时间，端到端延迟（E2E）是整个请求的总时间；带 `p95` 的字段取的是第 95 百分位。

```python
baseline  = {'throughput_tokens_per_s': 100.0, 'ttft_p95_ms': 50.0, 'e2e_p95_ms': 900.0}
candidate = {'throughput_tokens_per_s': 120.0, 'ttft_p95_ms': 55.0,  'e2e_p95_ms': 990.0}

speedup = candidate['throughput_tokens_per_s'] / baseline['throughput_tokens_per_s']
ttft_ratio = candidate['ttft_p95_ms'] / baseline['ttft_p95_ms']
e2e_ratio = candidate['e2e_p95_ms'] / baseline['e2e_p95_ms']
print(speedup, ttft_ratio, e2e_ratio)

print(speedup >= 1.10, ttft_ratio <= 1.10, e2e_ratio <= 1.05)
```

```text
1.2 1.1 1.1
True True False
```

吞吐加速比越大越好，要达到下限；两个延迟比越小越好，不能超过上限（门槛数值取自 Step 5 的配置）。这里吞吐和首词元延迟通过，端到端延迟比 1.1 超过上限 1.05，没通过。决策里一共四道门槛：吞吐、首词元延迟、端到端延迟、峰值显存，例子省略了最后一道；只要有一道没过，结果就是 `tune`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 五类候选家族的来源、需要的模型资产和主要检查项；比较前要固定哪些身份和条件 | 读 |
| Step 2 | 七种运行状态的判定依据；静态检查通过之后还要确认什么 | 读，对照上面的检查顺序 |
| Step 3 | 运行成功、证据可比、值得采用是三个判断；五种证据情况各自对应哪种决策 | 读 |
| Step 4 · TODO 1 | 规范化 family，拒绝未知家族，并返回一份不可修改原表的副本 | 写 `classify_speculative_strategy`：去掉大小写、空格和连字符后查别名表，查不到就抛 `ValueError` |
| Step 4 · TODO 2 | 状态只能来自 skipped / unsupported / asset_missing / incompatible / capacity_blocked / runnable，并保留具体 reason | 写 `admit_speculative_candidate`：按启用、后端、资产、兼容、容量的顺序返回首个状态和 `reason` |
| Step 4 · TODO 3 | 运行未完成、固定字段不一致、路径未生效和关键指标缺失要分开记录 | 写 `build_speculative_pair_evidence`：先分出 `not_evaluated`、`incomparable`、`incomplete`，再算吞吐加速比和两个延迟比 |
| Step 4 · TODO 4 | 未完成或不可比返回 not_evaluated；质量失败 reject；其余按门槛给出 accept 或 tune | 写 `decide_speculative_candidate`：质量优先，再查吞吐、首词元延迟、端到端延迟和显存四道门槛 |
| 测试 | 检查家族分类、未知家族、准入顺序、固定条件、缺指标、路径回退、门槛方向和阈值等号 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上用 vLLM 后端逐个启动 G0–G5，采集真实证据并形成候选决策 | 需要 GPU 和 notebook 指定的 vLLM 环境；默认是 `RUN_REAL_EXPERIMENT = False`，只做预检，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU，用的是模拟记录，不是真实后端测量。Step 5 需要 GPU 和 notebook 指定的 vLLM 环境。

## 容易踩的坑

- **跳过了检查顺序**：后端不支持、资产也没就绪时，要先返回 `unsupported`。单项失败和多项同时失败，测试都会查返回的是不是顺序里的第一个。
- **三种证据问题混成一个**：有一边没跑完是 `not_evaluated`，固定字段不一致是 `incomparable`，缺指标、路径回退或没有投机活动是 `incomplete`。要分开返回，只有 `comparable` 才计算比值。
- **把接受率当采用条件**：接受率来自后端原生统计，只用于解释成本。采用看的是质量、吞吐、延迟和显存门槛。
- **比值方向写反**：吞吐加速比要达到下限，延迟比和峰值显存不能超过上限；正好等于门槛也算通过。
- **用性能抵消质量失败**：`quality_ok` 为假时直接 `reject`，吞吐加速比再高也一样。

## 读完后自查

- [ ] 能说出 G0 和 G1–G5 各指什么。
- [ ] 一个候选后端不支持、资产也没就绪，能说出返回的运行状态。
- [ ] 能列出配对时必须一致的三个固定字段。
- [ ] 能说出吞吐加速比和延迟比各自的比较方向，以及质量不通过时的决策。

## 下一步

- 打开 notebook：[68. 投机解码机制选型基准](./68_Speculative_Decoding_Benchmark.ipynb)
- 想先弄清候选的模型结构：[35. 原生多 Token 预测](./35_Multi_Token_Decoding.ipynb)、[61. 模型架构审计](./61_Model_Architecture_Exploration.ipynb)
- 想深究接受情况与候选长度：[INF-PROPOSAL. 动态候选生成与验证预算](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb)
- 做完之后：[69. 前缀缓存基准 · 小白入门](./69_Prefix_Caching_Benchmark_Beginner.md)
