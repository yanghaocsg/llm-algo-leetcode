# 61. 模型架构审计 — 小白讲解

> 对应 notebook：[`61_Model_Architecture_Exploration.ipynb`](./61_Model_Architecture_Exploration.ipynb)

## 一句话总结
这个项目不训练模型，只做"架构审计"：给定基线和候选两份 Decoder 配置，先确认到底改了哪一个字段，再用公式估算注意力/前馈/词嵌入参数和每 token 的 KV 缓存字节，最后按改动类型指向 60、71、77、78 或 80 号专项实验，并根据参数预算与证据状态给出 `accept / tune / reject`。五个 TODO 串起"读配置 → 核对单一改动 → 算账本 → 生成审计卡 → 决定下一步"。

## 为什么需要它：痛点是什么
初学者比较模型结构时，常把"配置文件里改了一个数"直接当成"模型变强/变快"。比如把 16 个查询头、4 个 KV 头里的 KV 头减到 2 个，参数量确实少了、每 token 缓存也确实小了，但答案质量会不会掉、请求会不会更快？配置文件本身回答不了，必须有同任务、同请求条件的实测。

生活类比：装修前改图纸。你可以立刻算出"拆掉这堵墙能多出 3 平方米、少用 200 块砖"（参数与缓存估算），但"住起来是否更舒服、采光是否更好"必须实地量（任务评测）。审计报告的作用就是把"图纸上已经确定的"和"还需要现场验证的"分栏写清楚，避免把估算当结论。

## 核心问题
- 一份 Decoder 配置里，哪些字段决定它是 MHA、GQA 还是 MQA？哪些维度必须整除？
- 怎样证明候选"只改了一个字段"，而不是嘴上说改 KV 头、实际还动了前馈宽度？
- 参数量和每 token KV 缓存字节怎么由配置算出来？减少 KV 头会影响哪些矩阵？
- 不同结构改动应该去哪个专项实验找证据？为什么不能互相替代？
- 参数增量、质量、成本、失败记录这四类信息，如何共同决定 `accept / tune / reject`？

## 解决思路：一步步看懂
### 第 1 步：TODO 1——提取并校验结构配置
`extract_architecture_profile(config)` 先检查必填字段 `hidden_size`、`num_hidden_layers`、`num_attention_heads`、`intermediate_size`、`vocab_size` 是否存在，缺了就返回 `{'ready': False, 'issues': [...]}`。然后：

```python
kv_heads = int(config.get('num_key_value_heads', attention_heads))  # 缺省等于查询头数
if hidden_size % attention_heads: issues.append(...)   # 隐藏维要能被头数整除
if attention_heads % kv_heads:    issues.append(...)   # 查询头数要能被 KV 头数整除
attention_type = 'MHA' if kv_heads == attention_heads else ('MQA' if kv_heads == 1 else 'GQA')
head_dim = hidden_size // attention_heads
```

三个判断的含义：
1. `hidden_size % num_attention_heads == 0` 才能把头维切成整数，`head_dim` 才有意义；
2. `num_attention_heads % num_key_value_heads == 0` 才能把若干个查询头分给同一个 KV 头，这是 GQA 的前提；
3. 头数关系决定类型：KV 头 = 查询头是 MHA（每个查询头一套 K/V），KV 头 = 1 是 MQA（所有查询头共用一套），介于两者之间是 GQA（分组共享）。

返回字典还带 `attention_pattern`（默认 `dense`）、`state_representation`（默认 `kv_cache`）、`num_local_experts`、`num_experts_per_tok`、`capacity_factor`，这些字段默认值让同一套审计逻辑也能覆盖稀疏注意力和 MoE 配置。

测试用的基线是 `hidden_size=16, layers=2, heads=4, kv_heads=4, intermediate=32, vocab=100`，所以 `head_dim=4`。改成 kv=2 得 GQA，kv=1 得 MQA；kv=3 时 `4 % 3 != 0`，`ready` 为 False。

### 第 2 步：TODO 2——只允许一个变量
`validate_single_variable_candidate(baseline, candidate)` 把 11 个结构字段逐个比较：

```python
fields = ('hidden_size', 'num_attention_heads', 'vocab_size', 'num_key_value_heads',
          'intermediate_size', 'num_hidden_layers', 'attention_pattern',
          'state_representation', 'num_local_experts', 'num_experts_per_tok', 'capacity_factor')
changes = [field for field in fields if baseline.get(field) != candidate.get(field)]
```

然后要求 `len(changes) == 1`，并且 `candidate['changed_field']` 恰好等于这个字段；候选自身还必须 `ready`。为什么这么严？因为审计卡后面要按 `changed_field` 去选专项实验。如果实际同时改了 `intermediate_size` 和 `num_key_value_heads`，却只声明后者，就会出现"拿 KV 实验的结论去证明前馈改动"的错配。

`test_single_variable_change` 先只改 kv 头，通过；再偷偷把 `intermediate_size` 改成 48、`hidden_size` 改成 32、`num_attention_heads` 改成 8、`vocab_size` 改成 200，四种都被拒绝。

### 第 3 步：TODO 3——参数与缓存账本
`build_architecture_ledger(profile, dtype_bytes=2, seq_len=1024)` 先校验 `ready`、`dtype_bytes > 0`、`seq_len > 0`，然后逐项算：

```python
attention_per_layer = h * h + 2 * h * kv_heads * head_dim + h * h
attention_params = layers * attention_per_layer
mlp_params = layers * 3 * h * intermediate
embedding_params = vocab * h
norm_params = layers * 2 * h
total_params = attention_params + mlp_params + embedding_params + norm_params
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
```

逐项解释：
- `h*h` 是 Q 投影（$h \times h$）；`2 * h * kv_heads * head_dim` 是 K 和 V 两个投影，每个输出宽度是 `kv_heads * head_dim`（不是 `h`）；第二个 `h*h` 是输出投影 O。GQA 只缩小 K/V，Q 和 O 不变。
- `3 * h * intermediate` 对应 SwiGLU 的 gate、up、down 三个矩阵（gate 和 up 合计 2 个，down 1 个）。
- 词嵌入 `vocab * h`，每层两个 RMSNorm 共 `2*h`。
- KV 缓存前面的系数 `2` 是 K 和 V 各一份；乘 `layers` 是所有层都要存；`kv_heads * head_dim` 是每个 token 每层真正写入的宽度。

手算测试里的两组配置（默认 `dtype_bytes=2`）：

| 项目 | MHA（kv=4） | GQA（kv=2） |
|---|---|---|
| 每层注意力参数 | 256 + 2×16×4×4 + 256 = 1024 | 256 + 2×16×2×4 + 256 = 768 |
| 注意力参数（2 层） | 2048 | 1536 |
| MLP 参数 | 2×3×16×32 = 3072 | 3072 |
| 词嵌入 | 100×16 = 1600 | 1600 |
| 归一化 | 2×2×16 = 64 | 64 |
| 总参数 | 6784 | 6272 |
| 每 token KV 字节 | 2×2×4×4×2 = 128 B | 2×2×2×4×2 = 64 B |
| `seq_len=64` 的状态字节 | 8192 B | 4096 B |

所以把 KV 头从 4 减到 2，参数量少 512（约 7.5%），每 token 缓存直接减半。注意这只是"稠密 Decoder + KV Cache"的简化账本：不含 bias、LM head、MoE 专家参数，也不含 MLA 的压缩状态；notebook 明确说这些要由 71/80 等项目另测。

### 第 4 步：TODO 4——生成审计卡
`build_architecture_audit_card(baseline, candidate, evidence=None)` 先调用 `validate_single_variable_candidate`，不通过就返回 `{'ready': False, 'issues': ...}`；通过后分别算两份 ledger 和 `param_delta`，再按 `changed_field` 选专项项目：

| `changed_field` | `primary_project` |
|---|---|
| `intermediate_size` / `num_hidden_layers` | `60_Decoder_Block_Stability_Benchmark` |
| `num_key_value_heads` | `71_MLA_KV_Cache_Architecture_Benchmark` |
| `attention_pattern` | `78_Attention_Access_Pattern_Benchmark` |
| `state_representation` | `77_Long_Sequence_Memory_Architecture_Benchmark` |
| `num_local_experts` / `num_experts_per_tok` / `capacity_factor` | `80_MoE_Expert_Parallel_Benchmark` |
| 其他 | `workload_specific_training_or_inference_benchmark` |

`required_evidence['fields']` 固定为 `('workload', 'quality', 'cost', 'failure', 'evidence_level')`。`evidence_record` 把调用方传入的证据补成这五个键；没传时 `evidence_level` 默认 `cpu_structure_ledger`，表示"只有 CPU 结构估算，没有任务实测"。

### 第 5 步：TODO 5——按预算和证据决策
`recommend_audit_action(card, *, max_param_delta)` 是一条有顺序的判定链：

1. 卡不 `ready` → `reject`，`next_action='repair_candidate'`；
2. `param_delta > max_param_delta` → `reject`，`next_action='reduce_structure_change'`；
3. `evidence['failure']` 非空 → `tune`，下一步去 `primary_project` 修；
4. `evidence_level != 'same_family_benchmark'` → `tune`，表示缺少同族、同条件的实测；
5. `quality.passed` 或 `cost.passed` 不为真 → `tune`，`next_action='expand_or_remeasure'`；
6. 全过 → `accept`，`next_action='document_adoption'`。

`test_audit_decision` 给了一份"完整且达标"的证据：`quality` 是 `retrieval_accuracy=0.92`、阈值 0.90、`passed=True`；`cost` 是 `prefill_ms=12.0`、`tpot_ms=1.1`、`peak_memory_mb=512.0`、`passed=True`；`evidence_level='same_family_benchmark'`；`failure=None`。这份证据下 GQA 的 `param_delta=-512 <= 0`，得到 `accept`。把 `failure` 改成 `'cache interface unavailable'` 后退回 `tune`。

### 第 6 步：Step 5 项目实验——从配置到上游证据
CPU 测试只是夹具；真正的审计在 Step 5：
- `RUN_STAGE` 取 `prepare / audit / close`，`audit` 和 `close` 必须给唯一 `RUN_LABEL`，否则抛 `ValueError`。
- 输入清单是 `benchmarks/project_inputs/61_architecture_audit.json`，里面有 `baseline_model`、`candidate_model`、`revision`、`changed_field`、`producer_results`。
- `profile_from_hf_config` 用 `AutoConfig.from_pretrained(..., local_files_only=True)` 只读配置、不加载权重；`model_type` 含 `mamba` 时把 `state_representation` 记成 `recurrent_state`，否则 `kv_cache`。
- `audit` 阶段算出的卡以 `checkpoint_config_audit` 为证据级别，`max_param_delta=0`，用 `save_json_record_once` 落盘，不会覆盖同名结果。
- 5.4 从清单读取 60/71/77/78 等 producer 结果，用 `validate_record` 检查 schema、`semantic_id`、`role == 'producer'`，再用 `make_evidence_reference`/`verify_evidence_reference` 校验文件哈希。
- 5.6 `close` 调用 `decide_audit(config_decision, required_semantic_id, producer_records, reference_issues, external_required)`，把配置审计和上游实测合并成最终记录；对应项目不在 `ARCH_PROJECT_TO_SEMANTIC_ID`（60/71/77/78）里的，比如 80，会被标成 `external_required`。

读结果时先看 `changed_field` 和 `primary_project` 是否匹配，再看 `decision` 是 `accept` 还是 `tune`。只有 `cpu_structure_ledger` 或 `checkpoint_config_audit` 时，结论只能是"结构上可行，等待实测"，不能写成质量结论。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MHA / MQA / GQA | 每个查询头一套 K/V / 所有查询头共用一套 / 分组共用 |
| `head_dim` | 每个注意力头的宽度，等于 `hidden_size / num_attention_heads` |
| KV 头 | 真正写进缓存的 Key/Value 头数，GQA 里少于查询头 |
| 结构账本（ledger） | 按配置估算出的参数、权重字节和缓存字节 |
| `param_delta` | 候选总参数减基线总参数，可为负 |
| 单变量改动 | 候选相对基线只允许一个字段不同 |
| 审计卡（audit card） | 记录改动、账本、专项项目和证据字段的中间产物 |
| `primary_project` | 按改动类型指定的专项实验编号 |
| `evidence_level` | 证据强度，如 `cpu_structure_ledger`、`same_family_benchmark` |
| `external_required` | 所需专项项目不在本仓库 60/71/77/78 映射里，需要外部结果 |
| 上游证据（producer） | 由专项实验生成、被本审计引用的结果 JSON |

## 小白常见疑问
**Q1：为什么注意力参数里有两个 `h*h`，K/V 却用 `h * kv_heads * head_dim`？**
两个 `h*h` 分别是 Q 投影和输出投影 O，它们的输入输出都对齐隐藏维。K/V 投影的输出宽度只需要覆盖 KV 头，即 `kv_heads * head_dim`；GQA 正是靠缩小这两个矩阵来省参数和缓存，Q/O 不受影响。

**Q2：每 token 的 KV 字节为什么用 `kv_heads` 而不是 `num_attention_heads`？**
缓存只存 K 和 V，不存 Q。MHA 里每个查询头都有独立 K/V，所以两者相等；GQA/MQA 里多个查询头共享一套 K/V，真正写进显存的只有 `kv_heads * head_dim` 宽。用查询头数算会把缓存高估成 `heads / kv_heads` 倍。

**Q3：`max_param_delta=0` 是不是意味着任何改动都被拒绝？**
不是。它限制的是"参数增量"，GQA 把 KV 头减半后 `param_delta = 6272 - 6784 = -512`，负增量当然不超过 0，所以能走到质量与成本判断。会拒绝的是"参数变多"的改动，比如把 `intermediate_size` 从 32 加到 48，增量 2×3×16×16 = 1536，超过 0 就被 `reject`，提示 `reduce_structure_change`。

**Q4：为什么只允许改一个字段？现实中不是经常同时调好几个吗？**
可以同时调，但那就不能归因。审计卡的 `primary_project` 是按单个 `changed_field` 选的：改 KV 头看 71，改前馈宽度看 60，改专家数看 80。如果一次动三个，质量和成本的变化无法分配给某一个专项，报告就失去意义。要比较组合改动，应该拆成多次单变量实验。

**Q5：`extract_architecture_profile` 里的报错文案说 "hidden_size must divide num_attention_heads"，但我记得是头数整除隐藏维？**
以代码条件为准：这里写的是 `hidden_size % attention_heads`，即要求 `attention_heads` 整除 `hidden_size`（16 能被 4 整除），文案的措辞方向不太准确。同理下一行是 `attention_heads % kv_heads`，要求 KV 头数整除查询头数。读代码时先看 `%` 两边是谁，不要只背提示语。

## 动手验证：怎么知道自己做对了
测试单元按顺序调用五个函数，最后打印 `✅ 模型架构项目：配置比较、大小估算和选择规则通过测试。`；TODO 未填时 `NotImplementedError` 直接冒泡。
- `test_profile_contract`：同一份 `hidden_size=16, layers=2, heads=4, kv=4/2/1` 配置分别得到 `MHA / GQA / MQA`；把 kv 改成 3 后 `ready` 为 False。返回 `(mha, gqa)` 供后续测试复用。
- `test_single_variable_change`：`{**gqa, 'changed_field': 'num_key_value_heads'}` 的 `ready` 为真；额外改 `intermediate_size=48`、`hidden_size=32`、`num_attention_heads=8`、`vocab_size=200` 都必须 `ready` 为假。
- `test_architecture_ledger`：`seq_len=64` 下，GQA 的 `kv_bytes_per_token < MHA`，且 `attention_params < MHA`。手算值是 64 B < 128 B、1536 < 2048。
- `test_audit_card_evidence_contract`：改 KV 头的卡 `primary_project == '71_MLA_KV_Cache_Architecture_Benchmark'`；`required_evidence['fields']` 恰好是五个字段的集合；未传证据时 `evidence['evidence_level'] == 'cpu_structure_ledger'`；改 `intermediate_size` 指向 60；加 `num_local_experts=8` 指向 80；该卡 `max_param_delta=0` 时决策为 `tune`。
- `test_audit_decision`：带完整达标证据时 `accept`；把 `failure` 设为 `'cache interface unavailable'` 后变 `tune`。

notebook 的 Step 5 默认 `RUN_STAGE='prepare'`、`RUN_LABEL='replace_with_unique_label'`，只会打印清单缺失项，不读模型、不写最终报告；dump 里没有保存 Step 5 的执行输出，所以复跑时以 `benchmarks/results/61_architecture_exploration/config_audit_<label>.json` 里的 `audit_card` 和 `decision` 为准。
