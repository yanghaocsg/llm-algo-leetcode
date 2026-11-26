# 76. 激活重算、卸载与预算比较 — 小白讲解

> 对应 notebook：[`76_Activation_Checkpoint_Offload_Benchmark.ipynb`](./76_Activation_Checkpoint_Offload_Benchmark.ipynb)

## 一句话总结
本节先在一个小型 CPU 网络上确认 `baseline / checkpoint / offload / hybrid` 四种策略的输出、loss 和全部参数梯度一致，再用四个 TODO 搭出「条件核对 → 配对差异 → 预算筛选 → 敏感性结论」的分析链；GPU 项目则在 Qwen2.5-0.5B、batch 1 × 序列 768、FP32 + AdamW 上重复 3 次比较四种策略，并用无动量 SGD 的长度阶梯单独观察激活显存。登记的 `rtx5070ti_activation_memory_v1` 结论是顶层 `tune`：checkpoint 在已测激活显存范围内达到收益与吞吐门槛，但安全余量检查提前停止了更长的序列。

## 为什么需要它：痛点是什么
初学者常把「省显存」当成单指标：看到 checkpoint 峰值更低就想全量采用；看到 offload 把张量搬到 CPU，就以为显存直接归零；或者把一次 OOM 当成「省了 100%」。这些都会导致错误决策。真实比较至少要同时看：同一任务是否可比、实际执行路径是否生效、省了多少显存、损失了多少吞吐、验证损失有没有变差，以及这些结论在更紧的显存/吞吐预算下是否还成立。

生活类比：搬家选车。小货车省油但一趟装不完（checkpoint 省显存、多花时间）；大车一趟拉完但占道贵（baseline 占显存、速度快）；先把家具存进仓库再分批取（offload）要付搬运费和等待时间；混合方案则是「一部分随车、一部分寄存」，代价由实际路线决定，不是两种收益简单相加。选哪套取决于限高、时限和预算，而且必须搬同一批家具、走同一条路线才可比。

## 核心问题
- checkpoint 与 offload 分别改变了什么状态？为什么 CPU 上的 offload 只能验证数值语义，不能证明 GPU 省显存？
- 比较前为什么要核对 `workload_id`、`hardware_id`、`repeat_count` 和执行路径？
- `memory_saving_mb`、`throughput_ratio`、`quality_delta` 三个差值的方向分别是什么？
- 预算筛选的可行条件和排序规则是什么？基线会不会被选中？
- `stable`、`threshold_sensitive`、`no_feasible_candidate` 三种结论各表示什么，不表示什么？
- G0 和 G1 为什么用不同优化器？安全边界是怎么估的？
- 为什么激活省了，整步峰值却可能不降？

## 解决思路：一步步看懂

**Step 1：固定训练条件，只改保存/恢复方式。** notebook 要求模型、输入、批量、长度、精度、优化器和更新过程保持一致，只改变激活的保存与恢复。checkpoint 是「少保存中间张量、反向重算」，和「把模型权重存到磁盘」不是一回事；offload 是把需要保留的张量放到主机内存（CPU 一侧），反向需要时再搬回 GPU。记录四类东西：训练任务基线、实际执行的保存策略、整步耗时/吞吐/峰值、以及输出/梯度/评估损失和失败状态。

**Step 2：追踪从前向结束到反向使用的状态。** 完整保存让中间结果留在 GPU；重算只保留边界输入，延后生成中间结果；卸载把保存期的容量移到主机，但反向要等数据搬回；混合策略同时使用重算与搬运，代价由实际包围的计算区域和保存张量决定，**不是两种收益相加**。notebook 还提醒：减少激活不一定降低整步峰值——前向峰值降了，更新阶段的参数、梯度、optimizer state 和临时张量仍可能决定最大占用；比较时要确认重算/搬运真的发生，并看峰值出现在哪个阶段。

**Step 3：指标口径与预算筛选。** `peak allocated` 是测量窗口内张量分配占用的最高值；`peak reserved` 还包括 allocator 留待复用的块，两者要分别记录。吞吐保留率是候选吞吐除以基线吞吐，`0.8` 表示只保留 80% 的速度；评估损失用来检查当前输入下的数值差异，长期质量仍要训练任务评估。筛选顺序是：先排除执行失败、路径未生效、数值不合格的候选；再按显存上限、吞吐下限、损失上限取可行集合；在可行集合里按目标选择。改变阈值后复用同一批测量重算，不重新跑模型；多个预算得到同一选择，才说明结论在这些预算内稳定。

**Step 4：四个 TODO 的参考写法。**

TODO 1 在 `validate_strategy_experiment` 里核对身份：

```python
identity_keys = ('workload_id', 'hardware_id', 'repeat_count')
...
identity_matches = candidate.get(key) == baseline.get(key)
if not identity_matches:
    issues.append(f'{name}:{key}_mismatch')
```

三个字段分别代表「同一训练任务」「同一设备」「同一重复次数」，任一不同就不能直接相减。函数还会记录 `missing_name`、`duplicate_name`、`execution_path_unverified`（要求 `execution_path_verified is True`），以及基线指标是否有限；返回 `is_valid` 和 `issues`。

TODO 2 在 `build_strategy_pair_evidence` 里计算三个方向不同的差值：

```python
memory_saving_mb = baseline['peak_memory_mb'] - candidate['peak_memory_mb']
throughput_ratio = candidate['samples_per_s'] / baseline['samples_per_s']
quality_delta = candidate['eval_loss'] - baseline['eval_loss']
```

显存节省用「基线减候选」，正值表示候选更省；吞吐比用「候选除基线」，小于 1 表示变慢；损失差用「候选减基线」，正值表示变差。三者不能混用同一个正负号去判断「更好」。函数先处理 `status` 为 `oom` / `failed` 的记录：只计数、只保留 `{'name', 'status'}`，**不用缺失数值构造收益**；指标非有限的记为 `invalid`；其余才算 `comparable`，并带上 `repeat_count` 和 `execution_path_verified`。

拿测试 fixture 手算：基线 `peak_memory_mb=11000.0`、`samples_per_s=8.0`、`eval_loss=1.06`；checkpoint 候选是 `8200.0`、`6.8`、`1.08`。于是 `memory_saving_mb = 11000 − 8200 = 2800.0`，`throughput_ratio = 6.8 / 8.0 = 0.85`，`quality_delta = 1.08 − 1.06 = 0.02`。offload 候选是 `status='oom'`，只进 `oom_count`，不产生任何「节省」。

TODO 3 在 `decide_strategy_sensitivity` 里按预算筛选：

```python
feasible_names = [item['name'] for item in feasible]
selected_candidate = feasible[0]['name'] if feasible else None
```

`feasible` 的构造已经给出：同时满足 `peak <= memory_cap_mb`、`samples_per_s >= min_samples_per_s`、`eval_loss <= max_val_loss`，并按 `(peak_memory_mb, -samples_per_s)` 排序——先选显存更低的，再在显存相同时选更快的。选项里包含基线本身，所以基线也可以是最终选择。每个情景必须提供三个有限数值，否则抛 `ValueError`。

TODO 4 判断结论稳定性：

```python
sensitivity_status = (
    'no_feasible_candidate' if any(item is None for item in selected)
    else 'stable' if len(set(selected)) == 1
    else 'threshold_sensitive'
)
```

只要任一情景没有可行项，就是 `no_feasible_candidate`；所有情景选同一项才是 `stable`；选择发生变化是 `threshold_sensitive`。

用测试里的两个情景走一遍：`reference` 的显存上限 `12000`、最低吞吐 `1.0`，基线和 checkpoint 都可行，排序后 checkpoint 的 `8200 MB` 更小，故选 `checkpoint`；`throughput_tighter` 把最低吞吐提到 `7.0`，checkpoint 的 `6.8` 不再可行，只剩基线，选择变成 `baseline`，所以整体是 `threshold_sensitive`。把同一个 `reference` 传两次，选择都为 `checkpoint`，得到 `stable`；把最低吞吐提到 `20.0`，没有任何可行项，得到 `no_feasible_candidate`。

**Step 5：CPU 上验证四种策略的 autograd 语义。** `run_cpu_strategy(strategy, observe_saved_tensors=False)` 用两个线性层 `8 → 16 → 4`、输入 `torch.randn(2, 8)`、目标 `torch.randn(2, 4)`，固定 `torch.manual_seed(42)`：

- `offload` / `hybrid` 用 `torch.autograd.graph.save_on_cpu(pin_memory=False)` 包住前向和反向；
- `checkpoint` / `hybrid` 用 `checkpoint(first, x, use_reentrant=False)`；
- 只有 `baseline` / `checkpoint` 用 `saved_tensors_hooks(pack, unpack)` 观察保存张量的形状、元素数和设备。

测试断言：baseline 的 `saved_count > 0` 且 `saved_count == unpacked_count`；checkpoint 同样有保存/取回记录；offload / hybrid 的 `saved_count is None`，因为 `save_on_cpu` 自己接管了钩子，不能再挂一层观察器。四种策略的输出、loss 和全部参数梯度都要与 baseline 在 `atol=1e-6, rtol=1e-5` 下一致；未知策略必须抛 `ValueError`。cell 末尾打印 `CPU correctness test passed; this does not measure real GPU memory saving.`，这一句就是边界：它只证明数值语义，不证明 GPU 上省了多少。

**Step 6：GPU 项目怎么设计。** 固定配置是 `PROJECT_ID='76'`、`SEMANTIC_ID='PERF-MEMORY-STRATEGY'`、模型 `Qwen/Qwen2.5-0.5B-Instruct`（本地缓存）、`DTYPE_MODE='fp32'`、`WORKLOAD='pressure'`（batch 1、seq 768、warmup 2、iters 5）、`REPEATS=3`、`MEMORY_CAP_MB=11200.0`、`MIN_SAMPLES_PER_S=1.0`。`MAX_VAL_LOSS=None` 时，代码用基线 `eval_loss × 1.02` 作为质量上限；预算里另有 `min_memory_saving_mb=512.0` 和 `min_throughput_ratio=0.70`。

运行分四个阶段，只改 `RUN_STAGE`：

| 阶段 | 做什么 | 用 GPU |
|---|---|---|
| `prepare` | 只读校验固定配置、输入清单和已登记结果 | 否 |
| `g0` | AdamW 完整训练，重复测量四种策略 | 是 |
| `g1` | 在同标签 G0 基础上跑无动量 SGD 的长度阶梯 | 是 |
| `close` | 读取 G0/G1，生成分范围决策 | 否 |

G0 的每条策略都重新加载一份模型，用 `measure_training_run` 测 `step_time_ms`、`samples_per_s`、`loss`、`eval_loss`、`peak_memory_mb`、`peak_reserved_mb`；`run_strategy_repeated` 跑 3 次，`aggregate_strategy_runs` 取成功样本的均值并记录极差（`stability`），部分 OOM 记为 `partial_oom`。`run_strategy` 外面包了 `torch.cuda.OutOfMemoryError` 捕获，OOM 只记录状态和错误首行，不伪造指标。

G1 改用 `torch.optim.SGD(lr=1e-4, momentum=0.0)`，原因是去掉 Adam 的两个动量状态，让显存差异更集中地反映激活本身；notebook 明确说这个结论的 scope 是 `activation_memory_isolation`，**不能直接推广到 AdamW 完整训练**。每个更长序列开始前先做安全门禁：

```python
safety_margin_mb = max(2048.0, total_memory_mb * 0.15)
projected_peak_mb = previous_baseline['peak_memory_mb'] * boundary_seq_len / previous_seq_len
safe_process_limit_mb = free_memory_mb - safety_margin_mb
admitted = projected_peak_mb <= safe_process_limit_mb
```

`projected_peak_mb` 用上一档的峰值按序列长度线性外推；`safety_margin_mb` 取 2 GiB 与总显存 15% 的较大值，留出余量。不满足就记录 `blocked` 并停止，**不用实际 OOM 去试探设备极限**。长度阶梯是 `[1024, 1280, 1536, 1792]`，全部大于 G0 的 768。

`close` 阶段生成两个决策范围：`full_training_adoption`（基于 G0 的 AdamW 结论）和 `activation_memory_mechanism`（基于 G1 的激活专项结论），并用五个预算情景重算选择：`reference`、`memory_cap_5pct_tighter`（显存上限 ×0.95）、`throughput_floor_10pct_higher`（吞吐下限 ×1.10）、`quality_floor_1pct_tighter`（损失上限 ×0.99）、`combined_tighter`（显存上限 ×0.95、吞吐下限 ×1.10、损失上限 ×0.99 三者同时收紧）。所有来源文件都要通过 SHA256 核对，同名决策资产内容不一致会直接报错，不允许覆盖。

**结果怎么读。** 登记的 `rtx5070ti_activation_memory_v1` 在 5.6 写成：顶层结论 `tune`；checkpoint 在已测激活显存范围内达到收益与吞吐门槛；G1 状态是 `blocked`，安全余量检查在更长序列前停止，**实际显存不足点未知**；激活专项结论仅适用于已测范围，且 `promotable_to_full_training=False`。读顺序建议是：先看 `failure` / 各候选 `status` 和 `successful_runs`，排除 OOM、失败和重复不完整的项；再看 `peak_memory_mb` 与 `peak_reserved_mb`，并确认峰值出现在哪个阶段；接着看 `throughput_ratio`、`quality_delta` 和 `stability` 极差；最后才看预算情景。顶层 `tune` 的含义不是「策略没用」，而是「完整训练采用尚未成立，需要按安全预算调整测试范围或策略」。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 激活（activation） | 前向产生、反向还要用的中间张量 |
| checkpointing | 少存中间结果，反向按区段重算 |
| offload | 把要保留的张量搬到主机内存，用时再搬回 |
| hybrid | 一部分重算、一部分卸载的组合策略 |
| peak allocated | 真实张量分配占用的峰值 |
| peak reserved | allocator 预留（含可复用块）的峰值 |
| 配对差异 | 同一任务、设备、重复次数下的候选减基线 |
| `memory_saving_mb` | 基线峰值减候选峰值，正数才表示更省 |
| `throughput_ratio` | 候选吞吐除以基线吞吐，`0.85` 表示保留 85% |
| `quality_delta` | 候选验证损失减基线，正数表示更差 |
| 可行集合 | 同时满足显存、吞吐、损失三个门槛的候选 |
| `threshold_sensitive` | 换个预算，最优选择就变了 |
| G0 / G1 | 完整训练对照 / 激活显存隔离长度阶梯 |
| 安全余量 | 预留 2 GiB 或总显存 15%，避免用 OOM 试探极限 |

## 小白常见疑问

**Q1：CPU 测试里 offload 的 `saved_count` 为什么是 `None`，是不是没生效？**
不是。`save_on_cpu` 自己接管了 autograd 的保存/取回钩子，如果再套一层 `saved_tensors_hooks` 就会冲突，所以代码只对 `baseline` / `checkpoint` 挂观察器，`offload` / `hybrid` 的观察字段直接记 `None`，并在 `note` 里说明「用 GPU profiling 看实际放置」。CPU 上的 offload 只验证输出和梯度语义，不验证数据真的搬到了哪里。

**Q2：offload 候选 OOM 了，为什么不能把它的显存节省记成 11000 MB？**
因为 OOM 时根本没有得到有效的 `peak_memory_mb`、`samples_per_s`、`eval_loss`。`build_strategy_pair_evidence` 对 `status in {'oom','failed'}` 只记录 `{'name', 'status'}` 并计数，正是为了避免「用缺失值构造收益」。把 OOM 当成 100% 节省，会让预算筛选选中一个根本跑不完的策略。

**Q3：checkpoint 显存更低，为什么 fixture 里收紧吞吐后反而选基线？**
因为决策不是「唯显存论」。fixture 里 checkpoint 的 `throughput_ratio` 是 `0.85`，当最低吞吐从 `1.0` 提到 `7.0` 时，它的 `6.8 samples/s` 不再可行，只剩基线；这正是 `threshold_sensitive` 的含义。真实项目里预算还包含 `min_memory_saving_mb=512` 和 `min_throughput_ratio=0.70`，同样要一起看。

**Q4：G1 为什么用无动量 SGD，结论能直接用于 AdamW 训练吗？**
用 `momentum=0.0` 的 SGD 是为了去掉 Adam 的一阶、二阶动量状态，让不同策略的显存差异更集中地反映激活本身。notebook 在 `g1_decision` 里固定写上 `scope='activation_memory_isolation'`、`promotable_to_full_training=False`，并在 `limitations` 里列出「优化器状态被有意移除」「结果不构成 AdamW 完整训练采用依据」。所以它只能回答「激活这一项怎么变」，不能替代 G0。

**Q5：`stable` 是不是说明结果统计显著、可以外推到别的预算？**
不是。`stable` 只表示**所列出的这些情景**选了同一项；`decide_strategy_sensitivity` 不估计测量置信度，也不外推未列出的预算。notebook 还提醒，判断计时波动对选择的影响要保留重复样本重新比较；本项目用 `REPEATS=3` 和 `stability` 极差记录波动，但 3 次并不等于严格统计检验。

**Q6：为什么序列长度阶梯要提前停，不能跑到 OOM 为止？**
因为项目把「安全」放在「测到极限」之前：`safety_margin_mb` 预留 2 GiB 或总显存 15%，投影峰值超过 `free_memory - margin` 就记 `blocked` 并停止。所以登记的 G1 状态是 `blocked`，实际 OOM 点未知。这个设计避免一次实验把机器打到不可恢复，也提醒读者不要把「没测到」当成「不会发生」。

## 动手验证：怎么知道自己做对了
CPU 部分有三个独立测试，全部通过后测试 cell 保存的输出是 `所有测试通过！`。

- `test_strategy_identity_gate`：fixture 原样通过（`is_valid is True`）；把 `hardware_id` 改成 `gpu-b`，必须 `is_valid is False` 且 `issues` 含 `checkpoint:hardware_id_mismatch`；重复传入同名候选，必须含 `checkpoint:duplicate_name`；`execution_path_verified=False` 必须含 `checkpoint:execution_path_unverified`。
- `test_pair_evidence_preserves_failures_and_computes_deltas`：断言 checkpoint 的 `memory_saving_mb == 2800.0`、`round(throughput_ratio, 2) == 0.85`、`round(quality_delta, 2) == 0.02`；`oom_count == 1`、`comparable_count == 1`；offload 的比较记录必须恰好等于 `{'name': 'offload', 'status': 'oom'}`；把 `eval_loss` 改成 `NaN` 的候选必须让 `invalid_count == 1`。
- `test_budget_sensitivity_changes_selected_candidate`：两个情景下 `status == 'threshold_sensitive'`，`reference` 选 `checkpoint`，`throughput_tighter` 选 `baseline`；同一情景重复两次必须 `stable`；最低吞吐 `20.0` 时必须 `no_feasible_candidate`。

CPU 正确性 cell 另外断言：baseline 保存张量数大于 0 且保存/取回数量相等；`checkpoint`、`offload`、`hybrid` 三种策略的输出、loss、全部参数梯度都与 baseline 在 `atol=1e-6, rtol=1e-5` 下一致；未知策略抛 `ValueError`；最后打印 `CPU correctness test passed; this does not measure real GPU memory saving.`。

GPU 侧没有保存本次执行输出（默认 `RUN_STAGE='prepare'`，`RUN_LABEL='replace_with_unique_label'`）；`g0` / `g1` / `close` 只在完成环境预检、输入清单 SHA256 和 73 基线口径核对后才会写文件。5.6 的 `rtx5070ti_activation_memory_v1` 是登记结果：顶层 `tune`、checkpoint 在已测激活范围达到收益与吞吐门槛、G1 `blocked`、实际 OOM 点未知。复跑时先核对 `review_strategy_reports` 的 `issues`、`g0_complete`、`g1_ready`、`ready_for_decision`，再看 `summary` 和 `decision_scopes`，不要只截一个 `peak_memory_mb` 就下结论。
