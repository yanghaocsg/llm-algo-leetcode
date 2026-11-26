# 71. MLA 与 KV Cache 结构基准 — 小白讲解

> 对应 notebook：[`71_MLA_KV_Cache_Architecture_Benchmark.ipynb`](./71_MLA_KV_Cache_Architecture_Benchmark.ipynb)

## 一句话总结
本节先用两个公式分别估算 GQA 和简化 MLA 的 KV 状态字节数，再用 `validate_comparison_contract` 判断两组模型是否真的可比，用 `qualify_architecture_evidence` 区分"执行失败、证据缺失、指标不达标"，最后用 `select_architecture_candidate` 按"先显存、后 TPOT、再模型 ID"选出方案。测试里 GQA 的 8,388,608 字节和 MLA 的 2,359,296 字节都是可以手算核对的。

## 为什么需要它：痛点是什么
KV Cache 会随着输入变长而增长，这是推理服务最直接的显存压力来源。同学通常记住两个名词：GQA 通过共享 KV 头来省显存，MLA 把内容压到低维 latent 里再单独存位置分量。但一到实际选型就会问：到底省了多少？为什么两个模型都号称用了 MLA/GQA，跑同一批请求的延迟和峰值显存却差很多？为什么"显存更小"不等于"端到端更快"？如果只记名词，这些问题都答不了。

生活类比：两家仓库都要为每个订单保留凭证。GQA 是"每类货架共用一本账，但每笔订单仍逐条记 K 和 V"；MLA 是"把订单内容压缩成一张低维摘要卡，再单独贴一张位置标签"。摘要卡确实更小，但取货时还要先解码摘要，实际搬运时间取决于叉车和通道，不是账本越薄就一定越快。

## 核心问题
- GQA 和简化 MLA 每个词元、每层分别保存多少字节？长度翻倍时谁翻倍？
- 两组结果在什么条件下算 `controlled`，什么时候只能算 `observational`？
- 请求失败、指标缺失、质量或显存越界，分别应该返回什么状态？
- 多个候选都合格时，为什么先比峰值显存而不是先比速度？
- 本地模型检查阶段，怎么判断一个 checkpoint 是 GQA 还是 MLA，以及能不能装进显存预算？

## 解决思路：一步步看懂

**Step 1：比较条件不是看名字，而是看字段。** `validate_comparison_contract` 先要求两组的 `model_id`、`revision`、`representation` 都存在；再逐项比较 7 个请求字段 `prompt_set_id`、`batch_size`、`sequence_length`、`max_tokens`、`concurrency`、`warmup`、`repeats`，任一不同就记进 `issues`，此时 `valid=False`，连配对都不可用。然后检查训练记录（`data_revision`、`recipe_revision`、`train_tokens`、`seed`）和结构字段（`hidden_size`、`num_heads`、`num_layers`）是否齐全且两组相同，以及 `dtype_bytes` 是否相同、`representation` 是否真的不同。

**Step 2：TODO 1——什么叫受控对照。** 骨架已经算好 `same_training`、`same_structure`、`same_storage_type`、`representation_changed`，参考答案是四个条件的合取：

```python
conditions_match = (same_training and same_structure
                    and same_storage_type and representation_changed)
comparison_mode = 'controlled' if not issues and conditions_match else 'observational'
```

`representation_changed` 不能省：如果两组都是 GQA，就没有"换表示"这个自变量。`test_comparison_contract` 把训练记录清空、把 `data_revision` 设为 `None`、把 `hidden_size` 改成 2048、把 `dtype_bytes` 改成 4、把候选表示改回 `gqa`，每一种都必须是 `observational`；而把 `batch_size` 或 `sequence_length` 改掉，`valid` 必须为 `False`。注意 `valid` 和 `comparison_mode` 是两件事：请求不一致是"不可配对"，训练记录不一致是"可配对但不能归因"。

**Step 3：TODO 2——GQA 与 MLA 的字节账本。** 设批大小 $B$、序列长度 $L$、层数 $N$、每元素字节数 $s$。GQA 每层每个词元保存 K 和 V 两份，每份有 $H_{kv}$ 个头、每头 $D_h$ 维，所以

$$\text{bytes\_per\_token\_layer}_{GQA}=2\times H_{kv}\times D_h\times s$$

MLA 保存一份维度为 $D_{latent}$ 的内容状态，再加一份维度为 $D_{rope}$ 的位置分量：

$$\text{bytes\_per\_token\_layer}_{MLA}=(D_{latent}+D_{rope})\times s$$

整段历史再乘 $B\times L\times N$。参考答案：

```python
bytes_per_token_layer = 2 * model['num_kv_heads'] * model['head_dim'] * model['dtype_bytes']  # GQA
bytes_per_token_layer = (model['latent_dim'] + model['rope_dim']) * model['dtype_bytes']        # MLA
state_bytes = workload['batch_size'] * workload['sequence_length'] * model['num_layers'] * bytes_per_token_layer
```

代入测试常量：GQA 是 $2\times8\times128\times2=4096$ 字节/词元/层，$1\times1024\times2\times4096=8{,}388{,}608$；MLA 是 $(512+64)\times2=1152$，$1\times1024\times2\times1152=2{,}359{,}296$。两者比值 $1{,}152/4{,}096=0.28125$，也就是 MLA 约为 GQA 的 28%，省掉约 72%。把 `sequence_length` 改成 2048，两边都必须精确翻倍。非法维度（非正整数）或未知 `representation` 要抛 `ValueError`。

**Step 4：TODO 3——把失败、缺测、越界分开。** 先生成 `failures`：`workload_comparable` 为假加 `workload_mismatch`，`backend_capability_observed` 为假加 `backend_capability`。再收集缺失：`quality_score` 或 `peak_memory_mb` 为 `None` 就进 `missing`。参考答案的优先级是：

```python
if failures:
    status = 'reject'          # 请求失败或后端能力没观察到
elif missing or record.get('comparison_mode') != 'controlled':
    status = 'tune'            # 缺指标，或条件没配齐
elif record['quality_score'] < limits['quality_min'] or record['peak_memory_mb'] > limits['peak_memory_mb_max']:
    status = 'reject'          # 指标越界
else:
    status = 'accept'
```

为什么失败优先于缺测？因为请求都没跑成功，补指标也没意义。`test_evidence_gate` 用完整记录（质量 0.84、峰值 6200 MB、TPOT 8.0）得到 `accept`；把质量设为 `None` 得 `tune`；注入 `request_failed` 得 `reject`；质量 0.79（低于 0.80）或峰值 12001（高于 12000）也得 `reject`。

**Step 5：TODO 4——合格候选怎么排序。** 先过滤 `status == 'accept'`；空列表返回 `reject`，有候选但没人合格返回 `tune`。排序键是三元组：

```python
selected = min(accepted, key=lambda item: (item['peak_memory_mb'], item['tpot_ms'], item['model_id']))
```

第一关键字是峰值显存，所以"显存 6000、TPOT 8.0"的 b 会赢过"显存 7000、TPOT 7.0"的 a——这正是 `test_candidate_selection` 想验证的：本项目优先解决装不装得下的问题，速度只在显存相同时比较。第三关键字 `model_id` 是为了让并列结果稳定，不随输入顺序变化。

**Step 6：本地模型检查阶段做了什么。** `inspect_local_model` 从 `config.json` 判断表示类型：有 `kv_lora_rank` 和 `qk_rope_head_dim` 就是 `mla`；否则若有 `num_key_value_heads`，且它小于 `num_attention_heads` 就是 `gqa`，相等则是 `mha`。显存估算很粗暴但明确：把 `*.safetensors` 和 `*.bin` 的大小加起来得到 `weight_gb`，再算 `estimated_runtime_gb = weight_gb * 1.2 + 1.0`，后者与 `GPU_MEMORY_LIMIT_GB = 10.5` 比较得到 `capacity_ok`。基线是 `Qwen/Qwen2.5-0.5B-Instruct`，候选是 `deepseek-ai/DeepSeek-V2-Lite`，dtype 固定 `bfloat16`；准入还要求上游机制证据 `decision` 以 `accept_` 开头，否则记 `mechanism_evidence_not_accepted`。

**Step 7：真实对照实验怎么读。** 固定 workload 是 4 条 prompt、`max_tokens=64`、`temperature=0.0`、并发 1、warmup 2、repeats 3。`measure` 阶段依次启动 GQA 和 MLA 的 vLLM 服务，读三类指标：`ttft_ms`（首词元延迟）、`tpot_ms`（首词元之后平均每词元耗时）、`output_token_throughput_per_s`，以及归一化结果里的 `peak_memory_mb`。读法要分清层次：`tpot_ratio = 候选 TPOT / 基线 TPOT`，小于 1 才更快；显存差要看绝对值与相对降幅；而理论 `state_bytes` 只是账本，不等于实测峰值。notebook 当前登记结果写得很直白：没有真实 MLA backend 测量，模型检查阶段记录了文件不足，结论是 `tune`（运行状态 `blocked`）。所以不要从这一节编造任何加速比。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KV Cache | 推理时缓存历史词元的 Key/Value，避免每步重算 |
| GQA | 多个查询头共享少量 KV 头，减少要保存的 K/V 份数 |
| MLA | 把内容压到低维 latent，再单独保存位置分量 |
| `D_latent` / `D_rope` | 内容状态的维度 / 位置分量的维度 |
| `representation` | 本节的表示类型标记，取 `gqa` 或 `mla` |
| `controlled` | 训练、结构、存储类型一致，只有表示改变 |
| `observational` | 条件不齐，只能分别报告，不能归因于表示 |
| TTFT | Time To First Token，首词元延迟 |
| TPOT | Time Per Output Token，首词元之后平均每词元耗时 |
| 峰值显存 | 实际运行中的最大占用，包含权重、激活和 workspace |
| 准入（admission） | 模型文件、表示类型、显存预算的前置检查 |
| `tune` / `reject` | 继续补证据 / 当前方案不可用 |

## 小白常见疑问

**Q1：GQA 的公式为什么要乘 2，MLA 的为什么不乘？**
GQA 要同时保存 Key 和 Value 两份历史，所以乘 2；本题的简化 MLA 只保存一份内容状态和一份位置分量，二者相加即可。notebook 的 TODO 2 注释写的是"GQA 保存 K/V 两份状态；MLA 保存潜在状态与位置分量"，不要给 MLA 再额外乘 2。

**Q2：MLA 字节数更小，是不是就一定更快？**
不一定。`build_state_ledger` 只算"保存对象"的大小；真实速度还取决于 kernel、是否回退到非量化路径、批大小和生成长度。notebook 把理论账本标成 `cpu_theoretical_ledger`，并要求真实实验单独读 TTFT、TPOT 和峰值显存。`select_architecture_candidate` 也只在已 `accept` 的候选里排序，不会拿理论字节数直接当性能。

**Q3：为什么 `comparison_mode` 是 `observational` 时，指标达标也只能 `tune`？**
`qualify_architecture_evidence` 的顺序是"失败 → 缺测/条件 → 阈值 → 接受"。条件不是 `controlled` 时，即使质量和显存都过线，也只能说明"这个模型当前可用"，不能说明"是 MLA 表示带来的收益"，所以要继续补匹配记录，而不是 `accept`。

**Q4：为什么候选排序先看峰值显存，而不是先看 TPOT？**
这是本节项目目标决定的：先解决"装不装得下"，再在装得下的方案里比速度。`test_candidate_selection` 故意构造了显存与速度优劣相反的 a、b 两个候选，答案选显存更低的 b，就是为了固定这个优先级。

**Q5：本地检查里的 `estimated_runtime_gb = weight_gb * 1.2 + 1.0` 是精确值吗？**
不是。它只是一个保守的准入估算：权重之外预留约 20% 再加 1 GB。真正的峰值显存要靠 `measure` 阶段的后端记录；`capacity_ok` 只表示"按这个粗估没有超过 10.5 GB"。

## 动手验证：怎么知道自己做对了
四个测试函数直接调用，全过才打印 `测试通过：比较条件、保存量与方案选择规则通过测试。`
- `test_comparison_contract`：默认两组返回 `{'valid': True, 'issues': [], 'comparison_mode': 'controlled'}`；改训练方案、清空训练记录、把 `data_revision` 设为 `None`、改 `hidden_size`、改 `dtype_bytes`、把表示改回 `gqa`，都必须是 `observational`；改 `batch_size=2` 或 `sequence_length=2048` 必须 `valid=False`；只改 `max_tokens=32` 时 `issues` 里要有 `workload.max_tokens`。
- `test_state_ledger`：GQA 恰好 `8388608` 字节，MLA 恰好 `2359296` 字节；MLA 长度改成 2048 后恰好是原来的 2 倍。
- `test_evidence_gate`：完整记录 `accept`；质量 `None` 为 `tune`；带 `request_failed` 为 `reject`；质量 0.79 或峰值 12001 为 `reject`。
- `test_candidate_selection`：a(7000, 7.0) 与 b(6000, 8.0) 中选 b；空列表返回 `{'decision': 'reject', 'candidate': None}`。

notebook 每个测试单元都没有保存执行输出；Step 5 的 `prepare` 也只打印准入 JSON。判断是否做对，以这些断言和打印文案为准，不要引用未记录的 GPU 数字。
