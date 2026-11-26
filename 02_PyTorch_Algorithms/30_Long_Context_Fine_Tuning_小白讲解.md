# 30. 长上下文微调机制 — 小白讲解

> 对应 notebook：[`30_Long_Context_Fine_Tuning.ipynb`](./30_Long_Context_Fine_Tuning.ipynb)

## 一句话总结
本节回答一个很实际的问题：一条微调样本经过对话模板和分词器后，到底能不能完整塞进目标窗口？`summarize_length_distribution` 看长度分布，`audit_context_fit` 按“监督保留量 + 允许损失比例”把样本判成 fit / truncate / overflow，`pack_sample_lengths` 用 first-fit decreasing 做拼接，`build_boundary_isolated_packing` 在拼接时重置位置、屏蔽样本首标签并隔离跨样本注意力，`evaluate_length_curriculum_stage` 用一组门槛决定要不要进入更长窗口的阶段。两个 TODO 分别补“窗口策略”和“边界隔离的注意力掩码”。

## 为什么需要它：痛点是什么
初学者常把“长上下文微调”理解成“把 `max_length` 调大就行”。真跑起来会碰到三件事：第一，样本可能超过窗口，直接截断会把回答末尾甚至整个监督信号删掉；第二，大量短样本各自 padding 到最长长度，算力浪费在填充 token 上；第三，把多条短样本拼进一个窗口（packing）后，后一条样本会“看到”前一条样本的内容，造成上下文泄漏和损失污染。

生活类比：装箱。窗口是固定尺寸的箱子，样本是大小不一的物品。直接硬塞（截断）可能把物品的关键部件削掉；每件物品单独装一个箱子（padding）又浪费空间；把多件物品塞进同一个箱子（packing）时，必须放隔板（样本边界隔离），否则两件物品会混在一起。本节就是一套“先量尺寸、再决定削不削、最后装箱且放隔板”的流程。

## 核心问题
- 一条样本的真实 token 长度怎么统计？P50、P95 和溢出比例分别说明什么？
- 窗口放不下时，怎样在“保留多少监督”和“允许多大损失”之间做取舍？
- 为什么“完整进入窗口”不等于“可以训练”？最低监督量起什么作用？
- Packing 怎样提高利用率，又怎样保证样本之间不串味？
- 长度课程（length curriculum）进入下一阶段需要同时满足哪些门槛？
- GPU 上序列变长时，时间、每 token 时间和显存各自怎么变？Packing 的收益从哪里来？

## 解决思路：一步步看懂

**第 1 步：先把长度口径统一。** notebook 强调，长度必须从“固定分词器 + 固定对话模板”渲染出的完整序列上测量，不能把提示、回答分别分词后直接相加，否则特殊 token 和模板边界会对不上。本例采用“只监督回答”的口径，并把回答末尾的 EOS 计入监督：

$$
L_{\text{sup},i}=L_{\text{response},i}+L_{\text{EOS},i}.
$$

设 $T_i$ 为样本填充前的实际输入 token 数，目标窗口为 $C$，则完整长度 $L_{\text{total},i}=T_i$，数据集溢出比例为

$$
r_{\text{overflow}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[L_{\text{total},i}>C].
$$

其中 $i$ 是样本编号，$N$ 是有效样本数，$\mathbf{1}[\cdot]$ 在条件成立时取 1。EOS 被标签屏蔽时仍占输入位置，只是不计入监督。

`summarize_length_distribution(lengths, short_threshold, long_threshold)` 做三件事：按 `short`（≤ short_threshold）、`medium`（介于两者）、`long`（> long_threshold）分桶；用最近秩法算分位数：

```python
p50_tokens = ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)] if ordered else 0
```

例如测试 `test_context_fit_preserves_supervision` 输入 `[64, 128, 300, 900]`、阈值 `128/512`：排序后 `64,128` 是 short（2 条），`300` 是 medium（1 条），`900` 是 long（1 条）；`ceil(0.5×4)-1=1`，取 `ordered[1]=128`；`ceil(0.95×4)-1=3`，取 `900`。所以结果是 `{"short": 2, "medium": 1, "long": 1}`、`p50_tokens=128`、`p95_tokens=900`。

**第 2 步（TODO 1）：`audit_context_fit` 的窗口策略。** 函数签名是 `audit_context_fit(samples, target_context_len, template_tokens, min_kept_supervised_tokens=1, max_supervision_loss_ratio=0.0)`。对每条样本，先算提示加模板占用的位置：

```python
prompt_and_template_tokens = template_tokens + prompt_tokens
available_supervision_slots = max(0, target_context_len - prompt_and_template_tokens)
```

然后参考答案三行：

```python
kept_supervised_tokens = min(supervised_tokens, available_supervision_slots)
loss_ratio = (supervised_tokens - kept_supervised_tokens) / supervised_tokens
admitted = kept_supervised_tokens >= min_kept_supervised_tokens and loss_ratio <= max_supervision_loss_ratio
```

逐项解释：`available_supervision_slots` 是窗口里还能留给监督的位置数，不能为负；`kept_supervised_tokens` 是实际留下的监督位置，取“原有监督”和“可用位置”的较小值；`loss_ratio` 是被截掉的比例，范围 $[0,1]$；`admitted` 要求两个门槛同时通过——留下的监督不少于 `min_kept_supervised_tokens`，且损失比例不超过 `max_supervision_loss_ratio`。默认 `max_supervision_loss_ratio=0.0`，意味着只要监督被截掉就拒绝，这是“宁可不训，也不训半截回答”的保守策略。
函数随后给每条记录打标签：不通过就是 `overflow`；通过且 `kept == supervised_tokens` 是 `fit`；通过但有截断是 `truncate`。被拒样本的 `kept_supervised_tokens` 记为 0，避免误读成“还留了一些”。
手算测试数据（`target_context_len=2048, template_tokens=32`）：
- `short`：`prompt=300`，`300+32=332`，可用 1716，监督 80 全部留下，`loss_ratio=0`，fit。
- `boundary`：`prompt=1760`，`1760+32=1792`，可用 256，监督 256 全部留下，`total_tokens=2048`，fit。
- `overflow`：`prompt=1761`，`1761+32=1793`，可用 255，只能留 255，损失 `1/256≈0.0039>0`，默认阈值 0，拒绝，`kept=0`。
`test_controlled_truncation` 则把阈值放宽：`prompt=80, template=8, target=112`，可用 `112-88=24`，监督 40，留 24，损失 `16/40=0.4`；`min_kept=20`、`max_loss=0.4` 时允许截断，`max_loss=0.1` 时拒绝。
还有一个容易忽略的点：`test_full_fit_still_requires_supervision_minimum` 用 `prompt=2, supervised=1, target=16, template=1, min_kept=2`。完整进入窗口只留下 1 个监督，仍低于最低量 2，所以判 `overflow`，不能标成 `fit`。这说明“放得下”和“够不够训”是两件事。

**第 3 步：`pack_sample_lengths` 做容量拼接。** 它用 first-fit decreasing：先把长度从大到小排序，再依次放进第一个“剩余空间够”的箱子，放不下就开新箱。

```python
for length in sorted(lengths, reverse=True):
    target_bin = next((b for b in bins if sum(b) + length <= max_tokens_per_bin), None)
    if target_bin is None:
        bins.append([length])
    else:
        target_bin.append(length)
```

利用率是 `sum(lengths) / (len(bins) * max_tokens_per_bin)`。测试 `[7, 6, 4, 3]`、容量 10：7 先开箱，6 放不进 7 的箱，开第二箱；4 放进 6 的箱（6+4=10）；3 放进 7 的箱（7+3=10），得到 `[[7,3],[6,4]]`，总容量 20、总长度 20，利用率 1.0。函数还会拒绝长度 ≤0、超过单箱容量、或 `max_tokens_per_bin` 不是正整数的情况。

**第 4 步（TODO 2）：`build_boundary_isolated_packing` 的边界隔离。** 它把多条已准入样本的 `input_ids`、`labels` 首尾相接，同时生成 `segment_ids`（每条样本一个编号）和 `position_ids`（每条样本从 0 重新数）。循环里已经做了一件事：

```python
sample_labels[0] = -100  # 屏蔽样本起点，避免移位损失借用前一条样本的输出
```

为什么只屏蔽首 token？因为语言模型损失通常用位置 $i$ 的输出预测位置 $i+1$ 的标签。拼接后，前一条样本的最后一个位置会去预测后一条样本的第一个 token；把后一条样本首 token 的标签设为 `-100`，就切断了这次跨样本预测。注意代码先 `list(sample["labels"])` 复制，再改副本，所以不会污染调用方的源样本——`test_packing_masks_cross_sample_prediction` 专门断言 `samples[1]["labels"][0] == 21` 仍成立。
TODO 2 要补的是注意力掩码：

```python
attention_allowed = [
    [query_segment == key_segment and key_position <= query_position
     for key_segment, key_position in zip(segment_ids, position_ids)]
    for query_segment, query_position in zip(segment_ids, position_ids)
]
```

这是一个 $[总token数, 总token数]$ 的布尔矩阵。两个条件缺一不可：`query_segment == key_segment` 保证只看同一条样本；`key_position <= query_position` 保证因果性（只能看当前和更早的位置）。位置重置和注意力掩码职责不同：前者描述“位置语义”，后者控制“可见范围”，两者都要做。
用测试数据手算：样本 A 是 `[11,12,13]`、标签 `[-100,12,13]`，样本 B 是 `[21,22]`、标签 `[-100,22]`。拼接后 `input_ids=[11,12,13,21,22]`，`labels=[-100,12,13,-100,22]`，`segment_ids=[0,0,0,1,1]`，`position_ids=[0,1,2,0,1]`。第 2 行（token 13）可以看 `[T,T,T,F,F]`；第 3 行（token 21，B 的起点）只能看自己 `[F,F,F,T,F]`；第 4 行（token 22）可以看 `[F,F,F,T,T]`。

**第 5 步：`evaluate_length_curriculum_stage` 的晋级门槛。** 它比较基线和候选阶段，先做字段与数值校验（分数有限、监督 token 是正整数、失败次数是非负整数、资源是非负有限数），再算三项证据：

```python
"token_budget_matches": baseline["supervised_tokens_per_optimizer_step"] == candidate[...],
"long_context_gain": candidate["long_context_score"] - baseline["long_context_score"],
"short_context_regression": baseline["short_context_score"] - candidate["short_context_score"],
```

然后逐项检查 8 个 gate：token 预算是否一致、长上下文收益是否达到 `min_long_context_gain`、短上下文回退是否超过 `max_short_context_regression`、loss 和梯度是否有限、失败次数是否超限、峰值显存和每监督 token 成本是否超预算。全部通过返回 `decision="advance"`，否则 `"hold"` 并列出 `failed_gates`。
测试里的基线是 `long=0.56, short=0.82, tokens=8192`，门槛是 `gain≥0.04, regression≤0.02, failed≤0, peak≤10.5, cost≤1.4`。候选 `long=0.63, short=0.81, tokens=8192, peak=9.8, cost=1.25`：长上下文收益 `0.07≥0.04`，短上下文回退 `0.01≤0.02`，全部通过，advance。
另一个候选 `long=0.64, short=0.77, tokens=4096, peak=11.2` 同时触发三项：`token_budget_mismatch`（4096≠8192）、`short_context_regression`（`0.82-0.77=0.05>0.02`）、`peak_memory_budget`（11.2>10.5），所以 hold。把 `loss_finite/gradients_finite` 改成 `False`、`failed_runs=1`，则额外触发 `nonfinite_loss`、`nonfinite_gradients`、`failed_runs`。这正是“收益、稳定性、资源”三类门槛必须一起过的意思。

**第 6 步：可选 GPU 实验怎么读。** 5.1 固定 `Qwen/Qwen2.5-0.5B`、`batch=1`、长度 `[128, 256, 512]`、warmup 2、repeat 10、显存波动容差 `0.005`。三档长度都记录 `median_step_ms`、`p95_step_ms`、`median_ms_per_token`、loss、梯度范数、峰值显存。保存的结果来自 `rtx5070ti_length_packing_v4`，`decision=accept_mechanism`，三档中位数约 `29.16 / 47.06 / 79.52 ms`，每 token 约 `0.228 / 0.184 / 0.155 ms`——序列变长后单步更贵，但每 token 成本下降，这是批内并行度提高的表现，不是“越长越便宜”的普遍结论。峰值分配约 `3.10 GB`，长度趋势在 0.5% 容差内通过。
Packing 对照固定四条样本长度 `[48, 96, 32, 64]`（合计 240 token），padding 路径把它们补到 4×96=384 token。结果：处理 token 从 384 降到 240（少 37.5%），中位时间从 `63.4 ms` 降到 `48.8 ms`（约 23%），有效吞吐从 `3724` 升到 `4835 token/s`（约 30%）；两条路径的有效监督 token 都是 236，说明性能差异来自“少处理填充”，不是偷偷少监督。四项 Packing 检查（有效监督一致、起点标签屏蔽、位置重置、跨样本注意力阻断）全部通过，才允许解释吞吐差异。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 词元（token） | 分词器切出来的最小输入单位 |
| 目标窗口 $C$ | 一次序列最多能放多少个 token |
| 监督 token | 参与损失计算的位置，本例是回答加 EOS |
| 溢出（overflow） | 样本长度超过窗口，或监督保留量不达标被拒 |
| 受控截断 | 保留模板/提示、从回答尾部裁剪，并同时满足两个门槛 |
| Packing | 把多条短样本拼进一个窗口，减少填充浪费 |
| first-fit decreasing | 先放大样本、依次塞进第一个放得下的箱子 |
| 样本边界隔离 | 用位置重置、标签屏蔽和注意力掩码防止跨样本串味 |
| P50 / P95 | 一半样本不超过 / 95% 样本不超过的长度 |
| 长度课程 | 逐阶段提高长样本比例或窗口，每阶段都要过门槛 |
| 有效监督 token | 真正参与损失的 token，不含 padding 和被屏蔽位置 |

## 小白常见疑问
**Q1：既然默认 `max_supervision_loss_ratio=0.0`，为什么还要保留“截断”这个分支？**
默认策略确实倾向拒绝，但函数支持调用方放宽。`test_controlled_truncation` 用 `max_loss=0.4`、`min_kept=20` 允许把 40 个监督裁到 24 个；只要两个门槛同时满足，就标 `truncate` 而不是 `overflow`。这给了“回答很长但我愿意保留大部分”的场景一个受控入口。

**Q2：Packing 时已经重置了 `position_ids`，为什么还要注意力掩码？**
两者管的事不同。位置编号告诉模型“这是第几个位置”，但不会阻止它去 attend 前面的物理 token；如果不加掩码，后一条样本仍能通过注意力看到前一条样本的内容。`test_boundary_isolated_packing` 直接检查第 3 行（B 的起点）只有自己可见，就是在验证这层隔离。

**Q3：为什么只屏蔽每条样本的**第一个** label，而不是整条样本？**
因为跨样本污染只发生在“边界那一次预测”：前一条样本的最后一个位置会预测后一条样本的第一个 token。把后一条样本首 token 的标签设为 `-100`，这次预测就不产生损失；样本内部的其他标签仍然正常参与训练。`test_packing_masks_cross_sample_prediction` 还检查所有有效标签对 `(i-1, i)` 都属于同一 segment。

**Q4：GPU 结果里 256→512 的显存只涨了一点点，为什么仍算通过？**
绝对峰值受分配器缓存和复用影响，不一定随长度线性增长。配置里专门设了 `MEMORY_TREND_TOLERANCE=0.005`，只要求后一档不低于前一档的 99.5%，避免把正常的分配器波动误判成“机制失败”。真正判断资源压力要看完整的 `peak_allocated_bytes` 和 `peak_reserved_bytes`，而不是单看是否严格递增。

**Q5：为什么长度课程要求“每步有效监督 token 数”一致？**
如果候选阶段偷偷把每步监督 token 变少，训练成本和质量都会变，长上下文收益就无法归因。`token_budget_matches` 把这条作为第一个 gate：预算不一致时，后面的分数比较没有意义，测试里的 4096 vs 8192 就因此被判 hold。

## 动手验证：怎么知道自己做对了
notebook 在 CPU 上依次运行 10 个测试，最后打印 `PASSED: 窗口准入、监督起点、注意力隔离与阶段门槛核对通过。`。关键断言：
- `test_context_fit_preserves_supervision`：分桶 `{"short":2,"medium":1,"long":1}`、P50=128、P95=900；`fit_names=["short","boundary"]`、`overflow_names=["overflow"]`；boundary 的 `total_tokens=2048`；overflow 的 `kept_supervised_tokens=0`。
- `test_controlled_truncation`：放宽阈值时判 `truncate` 且保留 24；收紧到 0.1 时判 `overflow`。
- `test_packing_preserves_capacity_and_samples`：`bins=[[7,3],[6,4]]`、每箱不超 10、样本不丢、利用率 1.0。
- `test_boundary_isolated_packing`：`labels=[-100,12,13,-100,22]`、`position_ids=[0,1,2,0,1]`，以及上面三行注意力掩码。
- `test_length_curriculum_stage_decision`：第一个候选 `advance`、第二个 `hold` 且 `failed_gates` 恰好是三项、非有限候选触发三项稳定性 gate。
- `test_invalid_length_inputs` / `test_invalid_discrete_lengths_and_nonfinite_evidence`：负长度、零监督、空样本、超长样本、错位标签、浮点长度、`True` 当容量、NaN 指标（代码用 `math.isfinite` 检查，Inf 同样会被拒）、`failed_runs=0.5` 都必须抛 `ValueError`。
- `test_full_fit_still_requires_supervision_minimum`：完整放得下但监督不足，仍判 `overflow`。
- `test_packing_masks_cross_sample_prediction`：拼接不修改源样本，所有有效标签对同属一个 segment。
- 另外 `test_eos_truncation_and_masked_occupancy` 验证 EOS 随右侧截断被删除、审计不会把它补回，被屏蔽的 EOS 仍占输入位置。
GPU 单元格默认 `RUN_MODE="dry_run"`，只登记计划和路径，不加载模型；保存的结果文件与 5.3 的读取单元格给出了上文引用的实测数字。不要在没有 GPU 环境时强行改成 `real_gpu`。
