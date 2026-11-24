# 82. Quantization Artifact Evaluation Project Beginner | 量化产物评估项目 · 小白入门

> 配套 notebook：[82. 量化产物评估项目](./82_Quantization_Artifact_Evaluation_Project.md) ｜ 难度：Hard ｜ 环境：CPU 练习，NVIDIA GPU 实验

## 一句话说清

量化后的权重文件能保存下来，不等于能拿去部署。这本 notebook 教你检查一份量化后的模型文件包：文件齐不齐、质量记录可不可信、能不能真正加载，再决定放行还是拒绝。

## 为什么需要它

量化是把模型的权重从高精度数字换成低精度数字。量化做完，权重、配置和分词器文件都能存到磁盘上。但 notebook 开头就提醒：能保存下来，不代表它已经是可部署的产物。

一份候选包可能带着四种毛病：

- 文件不完整。权重文件在，分词器文件却不在。
- 校准数据和评测数据重叠。调量化参数时用过的样本又出现在打分的样本里，质量数字偏乐观。
- 质量记录方向写错。有的指标越低越好，却按“越高越好”判了通过。
- 目标后端加载不了这个格式。文件没坏，只是跑不起来。

看第三种。测试里的候选包记录了：指标越低越好，基线值 10.0，候选值 10.5，允许的退化比率是 1.10。方向看反，结论就整个反过来。

这本 notebook 把这些风险排成一条准入链：先核对候选包的身份、文件和质量证据，再用真实加载结果补上执行证据。它只负责“候选能不能进入部署对照实验”这一段，通过之后交给 67 做部署对照。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 量化（quantization） | 把模型权重从高精度数字换成低精度数字（比如 4 位），目的是省显存 |
| 候选包 | 一份记录，把基座模型、量化方式、数据来源、评测结果和实际文件绑到同一个身份上 |
| 门禁（gate） | 一道检查关卡，不过关就不能进入下一步 |
| 哈希（hash） | 把文件内容算成一串固定长度的摘要（比如 SHA256）。内容改了一点，摘要就完全不同，用来发现文件被替换或损坏 |
| 校准集 / 评测集 | 校准集是量化时用来看模型数值分布的一小批样本；评测集是给量化后的模型打分的样本。两者要来自同一份数据的不同划分（比如 `train` 和 `validation`） |
| 指标方向 | 这个指标是越低越好（`lower_is_better`）还是越高越好（`higher_is_better`） |
| 合取 | 几个条件用“并且”连起来，全部成立才算成立。代码里就是 `and` |

## 核心思路

输入是一份候选包，外加一份可选的真实加载结果。骨架已经把细项检查算成一个个布尔值，你要做的是把它们组合成两道 CPU 门禁和一道加载判断，再按顺序得出状态。

```text
contract_ready = identity_ready and quantization_ready and files_ready
quality_ready  = quality_inputs_ready and 比率按方向满足门槛
evidence_ready = split_isolated and sample_hashes_distinct and quality_ready
loader_ready   = load_complete and roundtrip_complete
memory_ready   = budget_valid and (max_memory_mb == 0 or memory_within_budget)
```

`identity_ready`、`quantization_ready`、`files_ready` 是产物身份、量化配置、文件完整性三项检查。`split_isolated` 表示校准和评测登记的是不同划分，`sample_hashes_distinct` 表示两者的样本摘要不同。`quality_inputs_ready` 表示质量数值有限，并通过了一致性检查。`load_complete` 是加载成功，`roundtrip_complete` 是全部固定提示词都返回了非空生成。`max_memory_mb` 是显存上限，`0` 表示不启用。

门禁按顺序消费这些结果，状态有四种：

```text
契约或质量没通过                 → reject（拒绝）
没有真实加载记录                 → pending（等待证据）
加载或往返生成没完成             → reject
已启用预算，但没有有效显存记录   → pending
加载成功，但显存记录超过预算     → tune（调整后复测）
全部通过                         → accept（进入部署对照）
```

用测试里的质量数字试一下，再看合取是什么意思：

```python
baseline, candidate, limit = 10.0, 10.5, 1.10   # direction = 'lower_is_better'
ratio = candidate / baseline
print(ratio, ratio <= limit)                    # 越低越好：比率不超过门槛才通过

ratio = 12.0 / baseline                         # 把候选值改成 12.0
print(ratio, ratio <= limit)

identity_ready, quantization_ready, files_ready = True, True, False   # 分词器缺失
print(identity_ready and quantization_ready and files_ready)
```

```text
1.05 True
1.2 False
False
```

候选值 10.5 对应的比率是 1.05，不超过 1.10，通过；换成 12.0 就是 1.2，不通过。最后一行里三项只错了一项，合取的结果就是 `False`。

notebook 对结论留了余地。样本摘要不同只能说明两份记录不同，不能证明两个集合没有交集。40 节的验证集已经用来挑选候选，这里复核的是挑选时的证据，不是新的独立测试。非空生成只说明调用跑通了。所以 `accept` 只表示“可以进入部署基准实验”。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 候选包要保留哪五类信息，每一类排除什么问题 | 读，对照“先认识几个词”里的候选包 |
| Step 2 | 静态契约、候选质量与真实加载三层证据各自证明什么 | 读，对照上面的五条合取式 |
| Step 3 | 门禁状态如何决定候选的下一步 | 读，对照上面的状态表 |
| Step 4 · TODO 1 | 组合静态门槛 | 在 `validate_candidate_contract` 里写 1 行：把三个原子状态用 `and` 连起来 |
| Step 4 · TODO 2 | 质量数值先通过有限性和一致性检查，再按指标方向比较 | 在 `evaluate_candidate_evidence` 里写 2 行：按方向比较比率得到 `quality_ready`，再和两个数据条件用 `and` 连起来 |
| Step 4 · TODO 3 | 组合加载与资源判断，缺测分支由骨架保留为 `pending` | 在 `decide_artifact_adoption` 里写 2 行：`loader_ready` 和 `memory_ready`，后面的状态分支骨架已经写好 |
| 测试 | 完整候选包能通过；缺分词器、数据重叠、质量失败、加载失败、超预算、非法数值要被识别 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 用 40 节获选候选的同一份文件做真实加载，生成准入报告和登记回执 | 需要 `INF-VLLM-011-CU128` 环境和单张兼容的 NVIDIA GPU。默认是 `RUN_STAGE = 'prepare'`，只读；`measure` 执行或恢复采集；`close` 校验并补齐报告与交接 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。Step 5 需要 `INF-VLLM-011-CU128` 环境和单张兼容的 NVIDIA GPU，默认停在只读的 `prepare`，不启动模型或后端。

## 容易踩的坑

- **三个条件互相替代**：身份、量化配置、文件完整性必须同时成立，要用 `and` 连接。测试里把分词器清空、把文件哈希去掉，都要判为不通过。
- **按错误方向判断质量**：先看 `direction`。`lower_is_better` 要求比率不超过门槛，`higher_is_better` 要求比率不低于门槛。`quality_inputs_ready` 不成立时，不能直接进入比较。
- **数据重叠被总状态掩盖**：校准和评测用了同一个划分，或者样本摘要相同，都要分别记进问题列表，不能只返回一个 `False`。
- **把缺测当成超预算**：已启用显存上限但还没有有效显存记录，是 `pending`；只有有效记录超过预算才是 `tune`。另外 Step 5 保存的 `peak_memory_mb` 是设备空闲显存差，不是过程峰值。
- **把通过当成质量证明**：非空生成只说明往返调用成功，`accept` 只表示可以进入部署基准实验，不能拿它替代任务质量的评测。

## 读完后自查

- [ ] 能说出候选包要保留的五类信息，以及每一类要排除的问题。
- [ ] 指标越低越好、基线 10.0、门槛 1.10 时，能判断候选值 10.5 和 12.0 通不通过。
- [ ] 能说出 `pending`、`reject`、`tune`、`accept` 各对应哪种情况。
- [ ] 能解释为什么 `accept` 不等于“模型质量已经被证明”。

## 下一步

- 打开 notebook：[82. 量化产物评估项目](./82_Quantization_Artifact_Evaluation_Project.md)
- 建议先读：[40. GPTQ 与 AWQ 权重量化 · 小白入门](./40_GPTQ_and_AWQ_Weight_Quantization_Beginner.md)
- 做完之后：[67. 量化推理与部署 · 小白入门](./67_Quantized_Inference_and_Deployment_Beginner.md)
