# 65. QLoRA 选型项目 — 小白讲解

> 对应 notebook：[`65_QLoRA_Selection_Project.ipynb`](./65_QLoRA_Selection_Project.ipynb)

## 一句话总结

本节把"要不要改用 QLoRA"做成一个可复查的选型项目：固定模型、数据、LoRA 适配器和更新预算，只让冻结底座的表示（以及随之而来的低比特训练准备）在 BF16 与 NF4 之间变化，然后用同一套门槛同时检查显存、可行微批次、吞吐和验证损失。三个 TODO 分别负责"两条训练路径是否可比""候选是否达标""最终采用、补测还是拒绝"，最后用三个随机种子的配对训练和容量阶梯给出结论。历史 v1 实测（RTX 5070 Ti、1.5B 模型）显示 QLoRA 峰值显存低 23.4%、最大微批次从 4 提到 8，但中位步时慢 65.2%，因此只建议用于容量受限场景。

## 为什么需要它：痛点是什么

初学者常见两种相反的口诀：一种说"QLoRA 省显存，无脑上"；另一种说"QLoRA 慢，别用"。两种说法都只对一半。QLoRA 把冻结权重压成 4-bit，确实能让你在同样显存里塞下更大的批次；但每次前向都要把 4-bit 权重反量化回计算精度，单步可能更慢。于是"省显存"和"更快"经常不能同时成立，选型就变成了一个带约束的决策问题，而不是一句口号。

生活类比：搬家时大货车一次能装完但进不了窄巷，小货车能进巷子却要多跑几趟。你不能只问"哪辆车好"，得先问"巷子多宽、时间多紧"。notebook 的 Step 1 就是在问这两件事：当前目标是"在现有批次下缩短训练时间"，还是"在显存上限内扩大批次"？目标不同，结论可以完全相反。

另一个痛点是"看起来公平的比较"其实不公平。如果 QLoRA 那一侧开了梯度检查点、换了有效批次或改了 LoRA 秩，显存和速度的差异就不知道来自量化还是来自这些改动。notebook 用 `same_workload_identity` 把模型提交、数据清单哈希、适配器签名、有效批次、最大长度、更新步数和种子组逐项锁死，只允许"冻结底座表示"这一项不同。

## 核心问题

- 两条训练路径要比较，哪些条件必须完全相同？少检查一个字段会出什么问题？
- QLoRA 候选要同时满足哪四项门槛？为什么门槛里的不等号方向有的是 `<=`、有的是 `>=`？
- 在什么条件下可以 `accept`，什么条件下只能 `tune`，什么条件下必须 `reject`？三者的区别是什么？
- 多个候选都可行时，`candidate_rank_key` 为什么把显存放在第一位？这会导致什么反直觉的结果？
- 固定批次比较和容量阶梯分别在回答什么问题？为什么容量探测会故意改变有效批次？
- 历史实测里"显存降 23.4%、步时升 65.2%"应该怎么读？为什么结论只限"容量受限"？

## 解决思路：一步步看懂

### Step 1：先把"比较什么"固定下来

notebook 的输入表列了四类固定项：数据切分（样本、训练/验证划分、文本模板）、更新条件（秩、缩放、目标层、学习率、有效批次）、资源预算（峰值显存上限、所需微批次）、产物要求（验证损失、低比特加载状态、适配器与哈希）。这四类合起来就是一次"可复查实验"的最小描述。

这里有两个名词要先说清。NF4（NormalFloat 4-bit）是一组专门为近似正态分布的权重设计的 4-bit 编码值，比普通的线性量化更贴合神经网络权重的分布。双重量化（double quantization）则继续压缩"量化缩放因子"这份元数据：第一层量化把权重压成 4-bit，但每组权重还要存一个缩放常数；双重量化把这些常数再量化一次，进一步降低附加开销。代码里对应 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, ...)`。

Step 2 特别提醒：QLoRA 侧还会调用 `prepare_model_for_kbit_training`，把部分非量化参数转成更高精度，并关闭梯度检查点（两侧都关）。所以实验比较的不是"一个矩阵乘法的快慢"，而是两条完整训练路径，差异可能同时来自量化、反量化和训练准备。这一点如果不说清，后面所有数字都会被误读。

### Step 2：两个实验，各自回答一个问题

第一个实验是**固定批次配对训练**：同一模型提交、同一数据、同一组种子、同一 LoRA 配置、同一有效批次，BF16 LoRA 与 NF4 QLoRA 各跑一遍，记录验证损失、峰值显存、步时和吞吐。它回答"在同样工作负载下，质量、速度、显存各差多少"。

第二个实验是**容量阶梯**：保持序列长度、累积次数和短训练步数不变，逐档增大微批次，直到某一档 OOM（显存不足）。它回答"当前这张卡最多能跑多大的微批次"。注意，微批次一变大，有效批次也会变大：

$$\text{有效批次}=\text{微批次}\times\text{累积次数}\times\text{设备数}$$

所以容量探测故意改变了有效批次，它不能拿来和固定批次比较质量，只能用来判断"哪一档能跑"。notebook 把这两类测量分开记录，就是为了避免把"批次变大"误当成"模型更好"。

### Step 3：显存账本——先分解，再对账

`build_memory_ledger` 把训练显存拆成六项：`base_weight_mb`（冻结底座）、`trainable_param_mb`（适配器参数）、`gradient_mb`（梯度）、`optimizer_state_mb`（优化器状态）、`activation_mb`（激活）、`quant_metadata_mb`（量化元数据）。函数先校验每项都是有限非负数，再求和得到 `estimated_total_mb`；如果传入了 CUDA 实测峰值 `peak_memory_mb`，还会算一个对账差额：

```python
estimated_total_mb = sum(float(value) for value in values.values())
report = {**values, 'estimated_total_mb': round(estimated_total_mb, 3), 'evidence': evidence}
if peak_memory_mb is not None:
    report['reconciliation_gap_mb'] = round(float(peak_memory_mb) - estimated_total_mb, 3)
```

测试里的手算：100 + 10 + 10 + 40 + 200 + 5 = 365 MB；实测峰值 380 MB，于是 `reconciliation_gap_mb` 是 15 MB。这个差额不是"算错了"，而是账本只列了六类对象，实际还有临时缓冲、CUDA 上下文、评测时的额外分配等。`evidence` 字段（默认 `'estimated'`）用来标记这份账本是估算还是实测，防止把估算值当成实测值引用。

### Step 4：三个 TODO 的分工

**TODO 1 `same_workload_identity`：先证明两条路可比。**

函数从 `baseline['identity']` 和 `candidate['identity']` 里取七个字段：

```python
required = ('model_commit', 'data_manifest_hash', 'adapter_signature',
            'effective_batch', 'max_length', 'update_steps', 'seed_set')
```

参考答案是一行：

```python
return all(baseline_identity.get(key) not in (None, '', [], {})
           and baseline_identity.get(key) == candidate_identity.get(key)
           for key in required)
```

每个字段要满足两件事：baseline 侧存在且非空，并且两侧相等。为什么不能只写 `==`？因为如果两侧都缺同一个字段，`None == None` 会返回 `True`，两条其实信息不全的运行就被误判成可比。为什么用 `not in (None, '', [], {})` 而不是只判 `is not None`？因为空字符串、空列表、空字典同样代表"这个字段没填"。测试 `test_workload_identity_mechanism` 先断言正常 fixture 返回 `True`，再把 QLoRA 的 `max_length` 从 256 改成 512，断言返回 `False`——只改一个字段就足以让比较失效。

**TODO 2 `candidate_passes_gates`：四项门槛取合取。**

骨架已经把四个布尔量算好了：

```python
memory_ok = candidate['memory_mb'] <= budget['memory_cap_mb']
scale_ok = candidate['max_micro_batch'] >= budget['min_micro_batch']
throughput_ok = candidate['tokens_per_s'] >= budget['min_tokens_per_s']
quality_ok = candidate['val_loss'] <= quality_floor['max_val_loss']
```

TODO 只需 `return memory_ok and scale_ok and throughput_ok and quality_ok`。方向要记清：显存和验证损失是"不能超过上限"，所以用 `<=`；微批次和吞吐是"不能低于下限"，所以用 `>=`。写反了不会报错，只会悄悄放行一个不该放行的候选。

测试给的门槛是 `memory_cap_mb=12000.0`、`min_micro_batch=1`、`min_tokens_per_s=18.0`、`max_val_loss=1.20`。QLoRA fixture 是 9800 MB、2、19.0、1.16，四项全过。`test_gate_mechanism` 另造了一个 `too_small`：把基线候选的显存改成 11000（低于上限）、微批次改成 0（低于下限），断言整体 `False`——说明只要一项失败就整体失败。`test_qlora_fail_closed_boundaries` 还检查了边界值本身应当通过（显存正好等于上限、微批次正好等于下限、吞吐正好等于下限、损失正好等于上限），以及 NaN、Inf、负数、`1.5` 这种非整数微批次、字符串 `'1'` 都必须被拒绝。

**TODO 3 `choose_qlora_action`：把"可行性"和"证据强度"分开。**

参考答案是一条决策链：

```python
if summary['feasible_count'] == 0:
    return 'reject'
if not summary.get('pair_complete') or summary.get('baseline_evidence_level') != 'real_gpu_paired_repeat' or summary.get('qlora_quantization') != 'nf4':
    return 'tune'
if not summary.get('identity_match') or not qlora_evaluation or not qlora_evaluation['feasible']:
    return 'tune'
if summary.get('qlora_evidence_level') != 'real_gpu_paired_repeat' or not summary.get('qlora_low_bit_active'):
    return 'tune'
return 'accept' if summary['best_strategy'] == 'qlora' else 'tune'
```

逐层看：

1. 一个可行候选都没有，返回 `reject`。这不是"再调调参"，而是当前预算和质量要求下这条路根本走不通，对应动作是放宽预算或先把质量做上去。
2. 配对不完整（不是"恰好一条 LoRA + 一条 QLoRA + 名字唯一"）、基线证据不是 `real_gpu_paired_repeat`、或者量化类型不是 `nf4`，返回 `tune`。这一层在说"比较本身不成立"。
3. 身份不匹配、QLoRA 评估缺失或不可行，返回 `tune`。
4. QLoRA 侧证据不是配对 GPU 实测、或者 `low_bit_active` 不为 `True`，返回 `tune`。这一层防止"以为加载了 4-bit，其实还是 BF16"。
5. 最后只有 `best_strategy == 'qlora'` 才 `accept`。也就是说，光"QLoRA 可行"不够，它还得在排序里赢过所有可行候选。

`test_project_decision_mechanism` 用三组 fixture 覆盖三种结果：正常配对、真实证据、QLoRA 达标 → `accept`；两侧都是 `estimated` 且 QLoRA 验证损失 1.28 超过 1.20 → `tune`；只有一个 QLoRA 且显存 13000、微批次 0、吞吐 17、损失 1.28 → `reject`。

这里还要注意一个容易误解的地方：fixture 里的基线 `lora_bf16` 显存是 14500 MB，超过 12000 MB 的上限，所以它自己并不可行；`accept` 的含义不是"两条路都能跑、QLoRA 更好"，而是"在当前预算下只有 QLoRA 可行，而且它的证据链完整"。如果两条路都可行，排序仍然优先显存，谁排第一取决于实测值，不能预设；结论要按 `best_strategy` 读，不能脑补成"BF16 基线被全面击败"。

### 骨架里的辅助函数也要看懂

`validate_qlora_candidate` 检查八个必填字段是否存在，并给错误打标签：`missing:xxx`、`non_numeric:xxx`、`non_finite:xxx`、`negative:xxx`、`non_integer:max_micro_batch`、`empty:xxx`。`validate_budget_and_quality` 检查预算三项和 `max_val_loss` 是否完整、有限、合法。`_candidate_failure_reasons` 按 `memory`、`scale`、`throughput`、`quality` 的顺序给出未通过原因，并原样写进 `summarize_low_resource_candidates` 的 `rejected` 列表。测试没有逐项断言这个顺序，但保持稳定顺序能让报告读者直接看出失败类型，也方便不同运行之间对比。

`candidate_rank_key` 返回四元组：

```python
return (candidate['memory_mb'], -candidate['max_micro_batch'],
        -candidate['tokens_per_s'], candidate['val_loss'])
```

Python 的元组比较是从左到右的字典序，不是加权打分。显存越小越靠前；显存相同才看微批次（取负号让大的靠前）；再看吞吐（取负号）；最后看损失（小的靠前）。`test_candidate_ranking_mechanism` 故意让 `qlora_rank_16`（9600 MB、1、20、1.18）和 `qlora_rank_32`（9800 MB、2、23、1.12）对比，断言 `best_candidate == 'qlora_rank_16'`。这是反直觉的：rank_32 的可行批次更大、吞吐更高、损失更低，但显存多 200 MB，所以在"显存优先"的排序里输了。这个测试不是在说 rank_16 真的更好，而是在锁定"本项目的排序优先级就是显存第一"。

`summarize_low_resource_candidates` 把逐候选评估、可行列表、拒绝原因、配对完整性、QLoRA 量化类型、`identity_match`、`qlora_low_bit_active` 汇总成一个字典。`decide_qlora_project` 再把三个动作映射成原因和下一步：

| decision | reason | next_action |
|---|---|---|
| `accept` | `qlora_is_best_feasible_option` | `promote_to_training_run` |
| `tune` | `qlora_needs_rank_or_quant_tuning` | `adjust_rank_or_quantization_bits` |
| `reject` | `no_candidate_meets_budget_and_quality` | `relax_budget_or_improve_quality` |

### Step 5：真实 GPU 实验的设计

Step 5 用 `RUN_STAGE` 控制三个阶段：`prepare` 只读准备，核对合同与数据来源；`measure` 检查训练依赖和 GPU，真正加载模型做配对训练；`close` 读取原始记录并登记。`RUN_LABEL` 必须是唯一标签，`measure` 阶段如果发现同名结果或 adapter 目录已存在会直接 `FileExistsError`，避免覆盖旧证据。

固定工作负载来自 `benchmarks/contracts/qlora_selection_v2.json`，数据来自 64 节的准入清单。`load_admitted_workload` 会检查清单 `decision == 'accept'`、tokenizer id 和 `max_length` 与 65 一致、重建后的样本 ID 与快照逐条相等，任何一项不符都不允许启动对照。这是一个很实在的防呆：选型实验不能偷偷换数据。

训练函数 `run_qlora_variant` 的关键控制点：

- 同一 `seed` 同时设 `torch.manual_seed` 和 `torch.cuda.manual_seed_all`；
- QLoRA 侧用 `BitsAndBytesConfig` 加载 NF4 + 双重量化，并用 `getattr(model, 'is_loaded_in_4bit', False)` 确认低比特真的生效，否则抛 `RuntimeError`；
- `prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)`，两侧 `gradient_checkpointing=False`；
- LoRA 配置来自合同（`r`、`lora_alpha`、`lora_dropout`、`target_modules`、`bias='none'`）；
- `CountingTrainer.compute_loss` 只在 `model.training` 为真时累加 `causal_supervised_token_count(inputs["labels"])`，统计"实际被训练消费的监督词元数"；
- `step_time_s = elapsed / update_steps`，`supervised_tokens_per_s = supervised_tokens_seen / elapsed`；
- `peak_memory_mb` 来自 `torch.cuda.max_memory_allocated()`，`peak_reserved_mb` 来自 `max_memory_reserved()`，口径是 `train_and_eval`；
- 每个成员单独保存 adapter 并计算目录哈希。

指标定义值得逐条读：

| 指标 | 定义 | 读的时候要注意 |
|---|---|---|
| `val_loss` | 每批一条样本，先求每条回答的平均 token loss，再对样本等权平均 | 分母是回答 token，不是整段输入 |
| `step_time_s` | 训练总时长 ÷ 更新步数 | 含 Trainer 启动和计数开销，不含评测与导出 |
| `supervised_tokens_per_s` | 实际消费的监督 token 数 ÷ 训练总时长 | 不是"总 token 数"，padding 和 prompt 不算 |
| `peak_memory_mb` | 训练与评测期间的最大已分配显存 | 与账本估算不是同一口径，看 `reconciliation_gap_mb` |
| `max_micro_batch` | 容量阶梯中最后一档未 OOM 的微批次 | 阶梯会同时改变有效批次，不能拿去比质量 |

配对顺序在种子间交替：`pair_index % 2` 为真时把 `[lora_baseline, qlora_candidate]` 反过来。这样做是为了避免某条路径总是先跑，把"顺序"带来的系统性影响集中在一边；notebook 没有进一步拆解具体是哪些环境因素，只要求顺序在种子间轮换。

把实验设计拆成"自变量 / 因变量 / 控制变量"会更清楚：

| 角色 | 内容 | 为什么重要 |
|---|---|---|
| 自变量 | 冻结底座表示：BF16 还是 NF4 双重量化 | 这是唯一允许变化的实验条件 |
| 因变量 | 验证损失、峰值显存、步时、监督词元吞吐、最大可行微批次 | 分别回答质量、资源、速度、容量 |
| 控制变量 | 模型 commit、数据清单、LoRA 秩/缩放/目标层、有效批次、长度、步数、种子组 | 任何一项变了，比较就不再是配对 |
| 基线 | BF16 LoRA（`lora_baseline`） | 所有差额都相对它计算 |
| 配对单位 | 同一个种子下的两次运行 | 消除种子之间的随机波动 |

为什么 QLoRA 可能更慢？notebook 没有把原因写死，只说"差异可能来自量化、反量化计算和训练准备"。可以确定的是：冻结权重以 4-bit 存储，每次前向和反向都要做反量化；`prepare_model_for_kbit_training` 还会调整部分非量化参数的精度。这两件事都不在 BF16 路径里。再加上同样的更新步数、更大的单步开销，最终就表现为"每步更慢，但能塞下更大批次"。至于具体慢多少，必须用 `step_time_s` 实测，不能用理论推断。

### 结果怎么读：`_compare_reports` 的四道闸

`_compare_reports` 先做事实校验：四个指标必须是有限非负数，baseline 的显存和步时必须为正（否则除零），三个阈值必须有限非负，种子至少两个、互不相同且为非负整数。任何一项不满足，直接返回 `tune`，`failed_checks=['invalid_metrics_thresholds_or_seeds']`。

通过后计算：

$$\text{memory\_saving\_ratio}=\frac{-(\text{candidate\_peak}-\text{baseline\_peak})}{\text{baseline\_peak}}$$

$$\text{step\_time\_regression\_ratio}=\frac{\text{candidate\_step}}{\text{baseline\_step}}-1$$

质量条件是 `candidate_loss <= baseline_loss + QUALITY_TOLERANCE`，其中容忍带是 0.1；速度条件是步时回退不超过 `MAX_STEP_TIME_REGRESSION_RATIO=0.15`（15%）；资源收益条件是容量变大，或者显存节省比例不低于 `MIN_MEMORY_SAVING_RATIO=0.1`（10%）。

最终有两条接受路径：

- `fixed_workload_accept`：质量通过、显存确实下降、有资源收益、速度达标、证据完整（配对种子一致、QLoRA 低比特已验证、adapter 哈希齐全）。这对应"同样批次下也值得换"。
- `capacity_limited_accept`：质量通过、显存下降、容量确实变大、证据完整。这对应"虽然每步更慢，但能跑更大批次"。

`selection_scope` 会标成 `both`、`fixed_workload`、`capacity_limited` 或 `none`。`test_visible_qlora_comparison_gate` 构造的场景是：质量 1.05（基线 1.0，容忍 0.1 内）、显存 80 对 100（省 20%）、步时 1.65 对 1.0（回退 65%，超过 15%）、容量 8 对 4。结果必须是 `accept`、`selection_scope == 'capacity_limited'`、`speed_ok` 为 `False`。另一个 `fast` 场景把步时改回 1.0、容量探测传空，结果应是 `fixed_workload`。测试还遍历了容量报告缺一条、重复、为空，以及四个指标取 NaN/Inf/-1 的情况，全部不允许 `accept`——这就是"fail closed"：证据不完整时宁可 `tune`，不能默认通过。

### 历史结果与当前状态

notebook 的 5.6 表格保留的是 v1 协议结果，标签是 `rtx5070ti_qlora_15b_capacity_v1`：

| 项目 | 数值 | 读法 |
|---|---|---|
| QLoRA 峰值分配显存 | 4,572.63 MiB，对照 5,968.06 MiB | 低 23.4%（(5968.06−4572.63)/5968.06） |
| 最大 micro-batch | 4 → 8 | 容量翻倍 |
| 验证损失差 | +0.0386 | 在 0.1 容忍带内，质量通过 |
| 中位步时 | +65.2% | 速度未通过 |
| 项目决策 | `accept` | 仅限 1.5B、12 GB、容量受限路径 |

所以正确结论不是"QLoRA 更好"，而是"在这张 12 GB 卡上，如果你需要更大的可行批次，QLoRA 值得；如果目标是缩短训练时间，继续用 BF16 LoRA"。表格上方还明确写了：新 v2 协议修正了监督词元计数、评测批大小和梯度检查点控制，尚未采集新的 GPU 结果，不能直接沿用历史通过结论。读到这句话就应该知道，历史数字只能当"方法演示"，不能当"当前合同下的有效证据"。

容量阶梯的实现也值得看一眼。它对每个标签依次尝试 `MICRO_BATCH_CANDIDATES` 里的档位，捕获 `RuntimeError` 后只处理消息里含 `out of memory` 的情况，其他错误继续抛出；遇到 OOM 就把该档记为 `{'status': 'oom'}` 并 `break`，最后取所有非 OOM 档位的最大值：

```python
max_micro_batch = max((item['fixed_micro_batch'] for item in attempts
                       if item.get('status') != 'oom'), default=0)
```

这个写法有两个细节：OOM 是"预期结果"而不是失败，所以记录后停止，不是让整个实验崩掉；`default=0` 保证连第一档都 OOM 时也有确定值，`_compare_reports` 里 `max_micro_batch > 0` 的检查会把它判为容量证据无效。

`close` 阶段会重新读取原始 JSON，核对运行标签、合同、上游数据、两侧种子矩阵和逐成员指标，再由逐成员记录重算中位数。`_build_artifact_manifest` 生成的清单包含基础模型 id 与 revision、上游数据清单、每个种子的 adapter 路径与 `sha256`、量化配置、质量报告和 `status`（`ready` 或 `pending_artifact_export`）。清单里还有一句很重要的备注：`quantization_artifact_note` 明确说"此 adapter 不能替代 GPTQ/AWQ/GGUF 量化 artifact"。也就是说，QLoRA 训练出来的是 LoRA 适配器，不是可直接部署的量化模型文件，别把两者混为一谈。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| QLoRA | 冻结底座用 4-bit 存，训练时反量化回计算精度，再挂 LoRA 适配器 |
| NF4 | 为近似正态权重设计的一组 4-bit 编码值，比线性量化更贴合分布 |
| 双重量化 | 把量化缩放因子再压一次，减少元数据开销 |
| `prepare_model_for_kbit_training` | 低比特训练准备，会把部分非量化参数转成更高精度 |
| 微批次 / 有效批次 | 单次前向的样本数 / 一次参数更新实际覆盖的样本数 |
| 容量阶梯 | 逐档增大微批次直到 OOM，用来找"最大能跑多大" |
| 峰值已分配显存 | `max_memory_allocated` 记录的张量峰值，不含缓存池里未用的部分 |
| 峰值保留显存 | `max_memory_reserved` 记录的缓存池峰值，通常大于已分配 |
| workload identity | 模型提交、数据哈希、适配器签名、批次、长度、步数、种子的组合 |
| evidence level | 证据强度，如 `estimated` 与 `real_gpu_paired_repeat` |
| 配对训练 | 同一种子下两条路径各跑一次，成对比较 |
| 质量容忍带 | 允许 QLoRA 比基线差多少验证损失，这里是 0.1 |

## 小白常见疑问

**Q1：既然 QLoRA 显存低 23.4%，为什么还会得出"速度敏感时别用"的结论？**

因为显存和速度是两个指标。历史结果里中位步时慢了 65.2%，超过 `MAX_STEP_TIME_REGRESSION_RATIO=0.15` 的容忍线，所以 `speed_ok` 为 `False`，`fixed_workload_accept` 不成立；只有容量从 4 提到 8 让 `capacity_benefit` 为真，才走通 `capacity_limited_accept`。`next_action` 字段对应的意思就是"容量受限训练用 QLoRA，速度敏感继续 BF16 LoRA"。

**Q2：测试里 `qlora_rank_32` 批次更大、吞吐更高、损失更低，为什么排序反而输给 `qlora_rank_16`？**

因为 `candidate_rank_key` 是字典序：先比 `memory_mb`，9600 < 9800，第一项就分出胜负，后面三项根本没机会参与。这是项目明确选择的优先级——显存是硬约束，其他指标只在显存打平时才比较。测试断言 `candidate_rank_key(candidates[0]) < candidate_rank_key(candidates[1])`，锁定的就是这个规则，而不是"rank_16 质量更好"。

**Q3：为什么 `accept` 要求 QLoRA 的 `low_bit_active` 为 `True`？**

因为 `load_in_4bit=True` 只是"请求"，不一定真的生效。`run_qlora_variant` 里专门用 `getattr(model, 'is_loaded_in_4bit', False)` 检查，如果 QLoRA 分支没激活 4-bit 就直接 `RuntimeError`。如果这里不查，可能拿一个其实是 BF16 的模型去和 BF16 基线比，然后得出"QLoRA 没省显存"的错误结论。

**Q4：`same_workload_identity` 为什么要求字段"非空且相等"，只写相等不行吗？**

只写 `==` 时，两侧都缺同一个字段会得到 `None == None` 为 `True`，于是一对信息不全的运行被当成可比。测试把 `max_length` 改成 512 只能覆盖"值不同"的情况，所以代码还要用 `not in (None, '', [], {})` 覆盖"值为空"的情况。这是防御式编程，不是多此一举。

**Q5：`build_memory_ledger` 算出的 365 MB 和实测 380 MB 差 15 MB，是不是实现错了？**

不是。账本只包含六类显式对象，`reconciliation_gap_mb` 就是用来暴露这个差额的。notebook 的 Step 3 说得很清楚：实际峰值取决于这些对象在训练和验证中何时同时存在，反量化还可能有临时缓冲。账本的价值是解释"节省来自哪里"，不是替代 profiler。

## 动手验证：怎么知道自己做对了

测试不是 pytest，而是直接调用函数；TODO 没填时抛 `NotImplementedError`。`run_qlora_selection_tests` 依次跑七个测试，全过才打印 `✅ QLoRA 选型机制测试通过：门槛、排序、策略决策与完整链路均已验证。`

- `test_memory_ledger_contract`：`build_memory_ledger(100, 10, 10, 40, 200, quant_metadata_mb=5, peak_memory_mb=380, peak_reserved_mb=420)` 的 `estimated_total_mb == 365.0`、`reconciliation_gap_mb == 15.0`；传 `-1.0` 抛 `ValueError`。
- `test_candidate_validation_contract`：缺 `strategy` 的候选错误列表含 `missing:strategy`；`memory_mb=float('nan')` 含 `non_finite:memory_mb`；合法预算返回 `is_valid is True`。
- `test_workload_identity_mechanism`：正常 fixture 为 `True`；`max_length` 改 512 后为 `False`。
- `test_gate_mechanism`：QLoRA fixture 通过；`memory_mb=11000` 且 `max_micro_batch=0` 的候选不通过。
- `test_candidate_ranking_mechanism`：`best_candidate == 'qlora_rank_16'`，且 `candidate_rank_key(candidates[0]) < candidate_rank_key(candidates[1])`。
- `test_project_decision_mechanism`：正常配对 → `accept`；两侧 `estimated` 且损失超限 → `tune`；唯一候选全面不达标 → `reject`。
- `test_qlora_selection_integration`：`summary['qlora_evaluation']['feasible'] is True`，`result['decision'] == 'accept'`，`result['next_action'] == 'promote_to_training_run'`。
- `test_qlora_fail_closed_boundaries`（单独调用）：NaN 显存、Inf 损失、负损失、`1.5` 微批次、字符串 `'1'` 全部不通过；边界值本身通过；预算任一项为 NaN 不通过；显存上限 +1、吞吐下限 −1、损失上限 +0.01 都不通过；重复候选、单候选、`estimated` 基线、`quantization='fp32'` 都不能 `accept`。
- `test_visible_qlora_comparison_gate`：从 notebook 源码里用 `ast` 抽出 `_compare_reports` 编译执行，注入 `SEEDS=(17, 29, 43)`、`QUALITY_TOLERANCE=.1`、`MAX_STEP_TIME_REGRESSION_RATIO=.15`、`MIN_MEMORY_SAVING_RATIO=.1`；断言容量场景 `accept` + `capacity_limited` + `speed_ok False`，快速场景 `accept` + `fixed_workload`，容量报告缺失/重复/为空或指标非有限时都不接受。

notebook 里保存的输出只有一处：5.2 预检打印了 GPU 信息（RTX 5070 Ti Laptop GPU、torch 2.11.0+cu128、CUDA 12.8、Python 3.12.15、transformers 5.18.0、peft 0.21.2、bitsandbytes 0.50.2、datasets 5.1.0），没有保存测试输出或训练结果。因此测试是否通过，以你本地运行时是否打印那行 ✅ 为准；历史表格里的 4,572.63 MiB、5,968.06 MiB、+0.0386、+65.2% 是 v1 协议记录，不要把它们当成当前 v2 合同的新结果。
