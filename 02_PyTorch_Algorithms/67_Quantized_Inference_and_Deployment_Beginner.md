# 67. Quantized Inference and Deployment Beginner | 量化推理与部署 · 小白入门

> 配套 notebook：[67. 量化推理与部署](./67_Quantized_Inference_and_Deployment.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

量化后的模型能不能上线，要和它自己的浮点版本在同样条件下比一比。这本 notebook 教你把两次运行配成一对，算清快了多少、省了多少、质量掉了多少，再得出部署结论。

## 为什么需要它

82 节检查的是量化产物本身合不合格。但 notebook 提醒，加载成功只说明产物能进入后端，还不能直接进入服务配置。它要和同一个模型的浮点版本比过，才知道低精度这条路在你的请求负载下站不站得住。

比的时候有三种做法会让结论没法解释：

- 两次运行条件不同。浮点版本在一张卡上跑，量化版本换到另一张卡，快出来的部分说不清是量化的功劳还是硬件的功劳。
- 指标方向搞混。生成每个词元花的时间是越小越好，每秒输出量是越大越好。用同一条“大于”规则去判，就会有一个判反。
- 缺的数当成零。没测到量化版本的显存，就把“显存降了多少”记成 0，看起来像量化没带来收益，其实只是没测。

看 notebook 测试里的一对记录。浮点版本 G0 生成每个词元要 20.0 毫秒，量化版本 G1 只要 18.0 毫秒。但 G1 的质量分数是 10.4，比 G0 的 10.0 高，而这里质量越低越好。快了多少、质量掉了多少，要换算成方向统一的比值才能一起判断。

这本 notebook 做三件事：先确认两组结果可以比较，再把各项指标换算成方向明确的比值，最后按固定顺序得出部署结论。通过之后，交给 83 去检验不同请求压力下能不能达到服务目标。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 配对运行 | 把浮点版本和量化版本在相同条件下各跑一遍，结果成对比较。浮点版本叫 G0，量化版本叫 G1 |
| 产物血缘（lineage） | 这个量化产物从哪里来：82 节是否接受了它，属于哪个模型版本 |
| 漂移（drift） | 两组运行里本该相同的一个字段变得不一样了，比如 G0 和 G1 跑在不同的卡上 |
| 单词元时间（TPOT） | 解码时平均每个词元花的时间，记录在 `tpot_ms`，越小越好 |
| 吞吐（throughput） | 每秒输出多少词元，记录在 `throughput_tps`，越大越好 |
| 执行路径 | 模型实际按什么方式计算。要从量化加载参数和后端日志追溯到低精度路径；回退（fallback）指后端没走量化路径，退回了别的算法 |
| 指标账本 | 把延迟、吞吐、显存、质量四项换算成比值后记成的一张表，每一项都标明方向 |

## 核心思路

输入是三样东西：82 节的门禁结论、G0 的记录、G1 的记录。处理分三步，正好对应三个 TODO。

```text
lineage_ready          = gate_ready and artifact_matches and model_revision_matches
fixed_fields_ready     = 六个固定字段都没有漂移

tpot_ratio             = G1 的 tpot_ms / G0 的 tpot_ms                     越小越好
throughput_ratio       = G1 的 throughput_tps / G0 的 throughput_tps       越大越好
memory_reduction_ratio = 1 - G1 的 peak_memory_mb / G0 的 peak_memory_mb    越大越好，缺测时为 None
quality_ratio          = G1 的 quality / G0 的 quality                    越小越好

evidence_ready = backend_observed and quantized_path_observed and no_fallback
quality_ready  = quality_ratio <= max_quality_ratio
resource_ready = latency_ready or throughput_ready or memory_ready
```

`gate_ready` 是 82 节的结论为 `accept`。后两项要求 G1 记录里的产物编号、模型版本和 82 节的记录对得上。六个固定字段列在 `FIXED_PAIR_FIELDS` 里。`latency_ready`、`throughput_ready`、`memory_ready` 是三个资源目标，默认门槛分别是 1.20、0.90、0.20。

注意最后一行的连接词：执行证据三项要同时成立，用 `and`；三类资源目标只要达到一个，用 `or`。状态按固定顺序得出：

```text
配对契约不一致               → reject
比值里有无效数值             → pending
执行证据不完整               → pending
质量退化超过预算             → reject
三类资源目标一个都没达到     → tune
其余                         → accept
```

用测试里的一对记录算一下：

```python
baseline  = {'tpot_ms': 20.0, 'throughput_tps': 50.0, 'peak_memory_mb': 8000.0, 'quality': 10.0}
candidate = {'tpot_ms': 18.0, 'throughput_tps': 56.0, 'peak_memory_mb': 5600.0, 'quality': 10.4}

tpot_ratio = candidate['tpot_ms'] / baseline['tpot_ms']
throughput_ratio = candidate['throughput_tps'] / baseline['throughput_tps']
memory_reduction = 1.0 - candidate['peak_memory_mb'] / baseline['peak_memory_mb']
quality_ratio = candidate['quality'] / baseline['quality']
print(round(tpot_ratio, 6), round(throughput_ratio, 6), round(memory_reduction, 6), round(quality_ratio, 6))
print(tpot_ratio <= 1.20, throughput_ratio >= 0.90, memory_reduction >= 0.20, quality_ratio <= 1.10)
```

```text
0.9 1.12 0.3 1.04
True True True True
```

第一行是四个比值，第二行是它们和默认门槛的比较，全部通过。单词元时间是 G0 的 0.9 倍，吞吐是 1.12 倍，显存降了 0.3，质量比值 1.04 没超过 1.10。

notebook 对结论留了余地。质量分数沿用上游挑选候选时的评测，不是新的独立测试。加载成功不是实际量化内核的取证，执行路径没核实之前保持 `pending`。Step 5 的真实配对跑三轮，比较比值的中位数；只有 `accept` 才生成交给 83 的引用。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从合格产物进入配对部署：产物准入、浮点基线 G0、量化候选 G1、部署判断四个阶段 | 读 |
| Step 2 | 配对实验要求一致的五个可比条件 | 读 |
| Step 3 | 哪些证据按什么方向比较，不通过时对应什么状态 | 读，对照上面的比值和状态表 |
| Step 4 · TODO 1 | 已提取的门禁、产物血缘和固定字段漂移 | 在 `validate_deployment_pair` 里写 2 行：`lineage_ready`（三项用 `and`）和 `fixed_fields_ready` |
| Step 4 · TODO 2 | G0/G1 的 TPOT、吞吐、显存和质量 | 在 `summarize_deployment_pair` 里写 4 个比值：三个用候选除以基线，显存降幅用 1 减去显存比值，缺测时保留 `None` |
| Step 4 · TODO 3 | 已提取的执行路径、质量和三类资源目标 | 在 `decide_quantized_deployment` 里写 3 行：`evidence_ready`、`quality_ready`、`resource_ready`，后面的状态分支骨架已经写好 |
| 测试 | 硬件和模型版本漂移会被识别；四个比值方向正确、缺失显存保持为空；`pending`、`reject`、`tune`、`accept` 四条路径 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 用 vLLM 在同一环境下交替运行浮点基线和量化候选，三轮配对后形成部署引用 | 需要 `INF-VLLM-011-CU128` 环境、NVIDIA GPU 和已准备好的量化产物。默认是 `RUN_STAGE = 'prepare'`，只读预检；`prepare` 和 `close` 只检查磁盘证据，不启动后端 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。Step 5 需要 `INF-VLLM-011-CU128` 环境、NVIDIA GPU 和已准备好的量化产物，默认停在只读预检的 `prepare`，不启动后端。

## 容易踩的坑

- **两次运行条件不同还硬比**：模型版本、分词器版本、请求负载、后端、硬件、缓存策略六个固定字段必须一致。测试里把 G1 的 `hardware_id` 改成 `gpu-b`，就要判为不可比较。
- **用同一个不等式处理所有指标**：单词元时间和质量的比值越小越好，吞吐比值和显存降幅越大越好。方向写反，状态就判反。
- **显存缺失写成零**：没测到显存时，`memory_reduction_ratio` 保持 `None`。写成 0 等于伪造了“量化没有收益”。
- **执行证据和资源目标用错连接词**：后端、量化路径、无回退三项必须同时成立，用 `and`；延迟、吞吐、显存三个资源目标至少达到一个，用 `or`。
- **把加载成功当成量化生效**：量化启动参数和加载成功不是实际量化内核的取证。执行路径没核实时保持 `pending`，不能仅凭格式名声明低精度执行。

## 读完后自查

- [ ] 能说出配对实验要求一致的五个可比条件。
- [ ] 给定 G0 和 G1 的四个数，能算出四个比值，并说出每个比值是越大越好还是越小越好。
- [ ] 能解释为什么 `evidence_ready` 用 `and`，`resource_ready` 用 `or`。
- [ ] 能说出 `pending`、`reject`、`tune`、`accept` 各对应哪种情况。

## 下一步

- 打开 notebook：[67. 量化推理与部署](./67_Quantized_Inference_and_Deployment.ipynb)
- 建议先读：[82. 量化产物评估项目 · 小白入门](./82_Quantization_Artifact_Evaluation_Project_Beginner.md)、[66. 单卡推理测量基线 · 小白入门](./66_Inference_Performance_Comparison_Beginner.md)
- 做完之后：[83. 量化服务 Profile 基准 · 小白入门](./83_Quantized_Serving_Profile_Benchmark_Beginner.md)
