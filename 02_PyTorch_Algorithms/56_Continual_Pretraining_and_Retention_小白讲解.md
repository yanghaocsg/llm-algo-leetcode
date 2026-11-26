# 56. 持续预训练与能力保留 — 小白讲解

> 对应 notebook：[`56_Continual_Pretraining_and_Retention.ipynb`](./56_Continual_Pretraining_and_Retention.ipynb)

## 一句话总结
持续预训练不是「拿领域文本再训几轮」这么简单。本节用四个 TODO 搭出一条判断链：先构造全序列监督标签，再对齐「当前位置预测下一个 token」的损失，然后检查领域训练集、领域评测集、通用保留集是否两两隔离，最后按 checkpoint 完整性、数值有限性、保留损失阈值和领域收益的顺序给出 `continue_with_evaluation`、`tune_or_rollback`、`tune_data_or_schedule` 或 `blocked_*`。核心结论是：训练 loss 下降只能说明模型拟合了训练文本，必须同时看独立领域集和通用保留集。

## 为什么需要它：痛点是什么
初学者最常犯的错是盯着训练 loss：它从 2.3 降到 1.2，就认为「领域能力提升了」。但训练 loss 是在参与更新的那批文本上算的，模型完全可能靠记忆把 loss 压下去；与此同时，它在通用文本上的损失可能明显变差，也就是常说的灾难性遗忘。更隐蔽的是数据泄漏：如果领域评测文本混进了训练集，评测分数会虚高，你却以为模型真的学会了。

生活类比：准备竞赛的补习班。只做竞赛题，竞赛模考分数可能上涨，但高考基础题可能手生。要判断这个补习班值不值得继续，至少要有三份卷子：平时练习题（训练集）、从没发过的竞赛模考（独立领域评测）、以及综合基础模考（通用保留集）。而且还要能退回上一次的状态，否则一次失败的补习就把之前的水平也搭进去了。

## 核心问题
- 持续预训练的损失对哪些 token 计算？它和 SFT 的 response-only mask 差在哪？
- 标签里的 `-100` 表示什么？为什么不能靠「token 编号是否为 0」判断 padding？
- `logits[:, :-1]` 与 `labels[:, 1:]` 位移后，哪些位置参与损失？分母为什么是「全批次有效目标数」？
- 领域训练集、领域评测集、通用保留集为什么要两两隔离？代码怎样检查交集？
- 判定继续、调整还是回滚时，checkpoint 字段、有限值、保留阈值、领域收益的优先级是什么？
- GPU 小实验里的 `domain_loss_change` 与 `retention_loss_change` 能说明什么、不能说明什么？

## 解决思路：一步步看懂

**第 1 步：持续预训练的目标函数。** 它沿用自回归语言建模：给定前面的 token，预测下一个 token。设有效预测位置集合为 $\mathcal V$，平均损失是

$$
L_{CLM}=-\frac{1}{|\mathcal V|}\sum_{t\in\mathcal V}\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

其中 $t$ 是当前预测位置，$x_{\le t}$ 是当前及之前的上下文，$x_{t+1}$ 是下一个 token 目标，$p_\theta$ 是参数为 $\theta$ 的模型给出的条件概率，$|\mathcal V|$ 决定分母。与 SFT 的 response-only mask 不同：SFT 只监督助手回答区间，持续预训练让每个非 padding token 都参与下一个 token 预测。原因是前者要学「怎么回答」，后者要改变模型对整个领域 token 序列的概率分布，包括领域术语、句式和行文习惯。

**第 2 步：构造全序列标签（TODO 1）。** `build_continual_pretraining_labels` 的输入是 `input_ids` 和 `attention_mask`，输出同形状的新标签张量。骨架先校验形状一致、掩码只含 0/1，参考答案只有一行：

```python
labels = input_ids.masked_fill(attention_mask == 0, -100)
```

`-100` 是 PyTorch 交叉熵的 `ignore_index`，表示这个位置不参与损失、也不回传梯度。为什么要按 `attention_mask` 而不是按 token 值判断？因为 token 编号 0 可能是一个合法的词元（例如某些 tokenizer 的 `<unk>` 或特殊符号），不能因为它等于 0 就当成 padding。`test_labels_follow_mask_without_mutation` 专门用 `input_ids=[[0, 5, 7, 9]]`、`attention_mask=[[1, 1, 1, 0]]` 验证：输出必须是 `[[0, 5, 7, -100]]`，第一个 0 保留；同时原 `input_ids` 不能被改写，且 `labels.data_ptr() != input_ids.data_ptr()`，说明返回的是新张量。`masked_fill` 本身返回新张量，不会原地修改输入。

**第 3 步：对齐下一 token 监督（TODO 2）。** `continual_pretraining_loss` 收到的 `logits` 形状是 `[batch, seq, vocab]`，`labels` 是 `[batch, seq]`。位置 $t$ 的 logits 应该预测位置 $t+1$ 的标签，所以参考答案是：

```python
shift_logits = logits[:, :-1].contiguous()   # 丢掉最后一个位置的预测
shift_labels = labels[:, 1:].contiguous()    # 丢掉第一个位置的标签
valid_count = int(shift_labels.ne(-100).sum().item())
loss_sum = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='sum',
)
return loss_sum / valid_count
```

这里有两个位置被自然排除：`logits` 的最后一个位置没有「下一个 token」可预测，`labels` 的第一个位置不是任何位置的预测目标。分母用 `valid_count`，也就是位移后所有非 `-100` 目标的个数，而不是 `batch × (seq-1)`。`test_effective_token_loss` 用 `labels=[[2, 4, 6, -100]]` 验证：只有位置 0 预测 4、位置 1 预测 6 两个目标，损失等于这两项负对数概率的平均，`atol=1e-10`。`test_unequal_target_counts_and_ignored_gradients` 用两条样本，第一条有 2 个有效目标、第二条只有 1 个，期望值是三项一起取平均（不是两条样本各算平均再平均）；反向之后还断言被忽略位置的梯度为零：`logits.grad[0, 2:]` 和 `logits.grad[1, 1:]` 全零，而有效位置有非零梯度。

为什么这个「全批次 token 平均」重要？如果每个样本先各自平均再对 batch 平均，短样本的每个 token 会被赋予更大权重；而语言建模通常希望每个有效 token 等权。代码用 `reduction='sum'` 再除以总有效数，口径明确，也方便在日志里单独记录 `loss_sum` 和 `valid_count`。

**第 4 步：检查三组数据是否隔离（TODO 3）。** `audit_continual_splits` 接收三个样本 ID 集合，骨架先拒绝任何空集合，然后要算三个两两交集：

```python
train_domain_overlap = train_ids & domain_eval_ids
train_retention_overlap = train_ids & retention_ids
domain_retention_overlap = domain_eval_ids & retention_ids
```

三组缺一不可。只查「训练 vs 领域评测」会漏掉「训练 vs 保留集」的污染：保留集本来是衡量通用能力有没有退化的，如果它的文本被拿来训练，退化就会被掩盖。任何交集非空都抛 `ValueError`，错误信息里带上三组重叠集合；全部干净时返回 `{'clean': True, 'counts': {...}}`。`test_continual_split_audit` 先验证三个单元素集合返回正确计数，再验证 `{'shared'}` 同时出现在训练和领域评测时抛错。`test_every_split_pair_and_empty_group` 更细：分别制造三种两两重叠，再分别传入三个空集合，六种情况都必须抛 `ValueError`。

**第 5 步：比较领域收益与保留代价（TODO 4）。** 设更新前后的领域评测损失为 $L^{before}_{domain}$、$L^{after}_{domain}$，保留损失为 $L^{before}_{retain}$、$L^{after}_{retain}$，则

$$
\Delta_{domain}=L^{after}_{domain}-L^{before}_{domain},\qquad
\Delta_{retain}=L^{after}_{retain}-L^{before}_{retain}.
$$

$\Delta_{domain}<0$ 表示领域损失降低，$\Delta_{retain}>0$ 表示保留集退化。给定允许的最大保留损失增量 $\tau_{retain}\ge0$，接受条件是 $\Delta_{domain}<0$ 且 $\Delta_{retain}\le\tau_{retain}$。`decide_continual_pretraining` 的骨架已经按固定顺序处理了前两层：先检查 `checkpoint_fields` 是否包含 `{'model', 'optimizer', 'scheduler', 'rng', 'global_step'}`，缺项返回 `blocked_unrecoverable` 和 `missing_fields`；再检查 `update_finite` 与两个损失变化是否有限，不满足返回 `blocked_nonfinite`。这里的 `checkpoint_fields` 只是一份字段清单，用来发现「续训所需状态有没有登记」，notebook 并不实际保存或加载权重；真实的 checkpoint 保存与恢复要另行验证。TODO 4 只补两个布尔：

```python
retention_exceeded = retention_loss_change > max_retention_increase
domain_improved = domain_loss_change < 0.0
```

注意这里是严格大于和严格小于：保留损失增量恰好等于阈值时**不算**越界，领域损失变化恰好为 0 时**不算**改善。测试 `test_continual_boundaries_and_blocking_priority` 用 `retention_loss_change=0.05`、`max_retention_increase=0.05` 验证等号可通过，用 `domain_loss_change=0.0` 验证会落到 `tune_data_or_schedule`。优先级也很重要：同一组输入里如果既缺 checkpoint 字段又有 `NaN`，返回的是 `blocked_unrecoverable` 而不是 `blocked_nonfinite`——先证明「能恢复」，再谈「数值可解释」。骨架还对 `max_retention_increase` 做前置校验：必须是有限非负数，传 `-1.0` 或 `nan` 直接抛 `ValueError`。

**第 6 步：四种决策分别是什么意思。** 代码只返回四种字符串：`blocked_unrecoverable`（checkpoint 状态不全，无法安全续训/回退）、`blocked_nonfinite`（更新或指标出现 NaN/Inf，先查数值和输入）、`tune_or_rollback`（保留损失越界，降步数/学习率或回退）、`tune_data_or_schedule`（保留没越界但领域没改善，调数据配比或日程）、`continue_with_evaluation`（两项都过，可以进入更完整评测）。注意最后一个并不等于「能力已提升」，只是「值得继续验证」。

**第 7 步：和 Transformers 的因果损失对账。** `check_continual_pretraining_framework_conformance` 用 `GPT2Config(vocab_size=16, n_positions=8, n_embd=24, n_layer=1, n_head=2)` 在 CPU 上建一个微型模型，把同一组 `input_ids`、`attention_mask` 和本节构造的 `labels` 喂进去，比较三件事：padding 位置标签是否为 `-100`、非 padding 位置标签是否等于原 token、`continual_pretraining_loss` 是否与模型内置 `output.loss` 在 `atol=1e-6` 内一致。它验证的是「位移 + ignore_index + 有效目标平均」这套语义与成熟框架相同，不涉及真实模型能力。

**第 8 步：GPU 实验的设计与读法。** Step 5 用 `Qwen/Qwen2.5-0.5B`、FP32、`seq_len=128`、`train_steps=8`、`warmup_steps=1`、`learning_rate=1e-4`、`seed=42`。数据角色分三份：两条领域训练文本、一条独立领域评测文本、一条通用保留文本（例如 `'A small function returns the sum of two integers.'`）。自变量是「是否执行这 8 步全序列领域更新」（`base` 对 `continual`），因变量是更新前后的领域评测 loss、保留 loss、训练 loss、梯度范数、step 中位数/P95 和峰值显存。

执行时先用 `model.eval()` + `torch.no_grad()` 测 before，再训练 1 步 warmup 和 8 步计时更新，最后再测 after；`encode` 会把 `input_ids` 克隆成 labels 并把 padding 置为 `-100`，和 TODO 1 的语义一致。结果里 `domain_loss_change = after.domain_eval_loss - before.domain_eval_loss`、`retention_loss_change = after.retention_loss - before.retention_loss`；前者为负才是领域改善，后者为正表示保留退化。`finite_all`、`gradient_norm_median`、`median_step_ms`、`p95_step_ms`、`peak_allocated_mb` 和 `checkpoint_fields` 一起写进 JSON。

读结果时先看 `finite_all` 和 `checkpoint_fields`：任一失败就不要解释 loss 变化。再看两个 delta：领域降了、保留增量在阈值内，才只是「这条测量链跑通」。notebook 明确说明内置文本只用于验证方法，不能外推为真实领域能力结论；真实项目要换成版本化领域语料、独立领域任务和预先声明的阈值。默认 `RUN_MODE='dry_run'` 只打印核对信息，不加载模型。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 持续预训练 | 用新领域文本继续训练已有语言模型，改变其 token 分布 |
| 全序列监督 | 每个非 padding token 都作为下一个 token 的预测目标 |
| response-only mask | SFT 只监督助手回答部分，不监督用户输入 |
| `-100` | PyTorch 交叉熵的 `ignore_index`，表示该位置不参与损失 |
| 下一 token 位移 | 位置 $t$ 的 logits 对齐位置 $t+1$ 的标签 |
| 有效目标数 | 位移后标签不为 `-100` 的位置总数，是损失的分母 |
| 领域评测集 | 不参与训练、用来判断域内是否真的改善的独立文本 |
| 通用保留集 | 不参与训练、用来观察原有能力是否退化的文本 |
| 数据隔离 | 三组样本 ID 两两不相交，防止评测污染 |
| $\Delta_{domain}$ / $\Delta_{retain}$ | 领域损失变化 / 保留损失变化，负/正分别代表好坏 |
| `checkpoint_fields` | 续训和回退所需的状态清单：模型、优化器、调度器、随机数、全局步 |
| 灾难性遗忘 | 学新领域时把原有通用能力学坏 |

## 小白常见疑问
**Q1：为什么持续预训练用全序列监督，而 SFT 只监督回答？**
目标不同。SFT 想让模型学会「给定问题怎么回答」，用户输入不需要被预测；持续预训练想让模型适应领域的语言分布，领域术语、句式和上下文都值得学习，所以每个非 padding token 都是目标。notebook 的 Step 1 用一张监督范围对照图强调这一点。

**Q2：`labels` 里为什么用 `-100`，用 0 不行吗？**
因为 0 可能是合法 token。`cross_entropy(ignore_index=-100)` 会跳过这个值；如果用 0，模型会被要求预测编号 0 的 token，padding 就参与了损失。`test_labels_follow_mask_without_mutation` 特意把有效 token 设为 0，验证它必须保留，而 padding 位置才变成 `-100`。

**Q3：损失为什么不是每个样本先平均再对 batch 平均？**
因为样本长度不同。若先按样本平均，短样本里的每个 token 权重会更大；`continual_pretraining_loss` 用 `reduction='sum'` 累加所有有效目标，再除以全批次 `valid_count`，让每个有效 token 等权。`test_unequal_target_counts_and_ignored_gradients` 用 2 个目标和 1 个目标的两条样本验证了这一点。

**Q4：保留损失增量刚好等于阈值，为什么算通过？**
代码写的是 `retention_loss_change > max_retention_increase`，只有严格越界才回滚；等号表示「正好在允许上限内」。同理 `domain_loss_change < 0.0` 才叫改善，等于 0 不算。`test_continual_boundaries_and_blocking_priority` 把这两个边界都写成了断言。

**Q5：为什么 checkpoint 缺字段要排在 NaN 前面处理？**
因为指标再干净，不能恢复状态也无法安全续训或回退。`decide_continual_pretraining` 先查 `required_fields.issubset(checkpoint_fields)`，缺项直接 `blocked_unrecoverable`，此时即使 `domain_loss_change=nan` 也不会走到数值检查。测试里 `checkpoint_fields={'model'}` 且 `domain_loss_change=nan` 的结果正是 `blocked_unrecoverable`，`missing_fields` 为 `['global_step', 'optimizer', 'rng', 'scheduler']`。

## 动手验证：怎么知道自己做对了
`test_full_sequence_labels`：`input_ids=[[5, 6, 7, 0]]`、`attention_mask=[[1, 1, 1, 0]]` 必须得到 `[[5, 6, 7, -100]]`；形状不一致或掩码含 2 必须抛 `ValueError`。`test_effective_token_loss`：`logits` 形状 `(1, 4, 10)`、`labels=[[2, 4, 6, -100]]` 时，损失等于位置 0 预测 4、位置 1 预测 6 两项负对数概率的平均，`atol=1e-10`；全 `-100` 标签必须抛 `ValueError`。

`test_continual_split_audit`：`({'train-1'}, {'domain-1'}, {'retain-1'})` 返回 `{'clean': True, 'counts': {'train': 1, 'domain_eval': 1, 'retention': 1}}`；训练与领域评测共享 `'shared'` 时抛 `ValueError`。`test_labels_follow_mask_without_mutation`：有效位置的 token 0 保留，原输入不被改写，标签张量与输入不共享内存。

`test_unequal_target_counts_and_ignored_gradients`：两条样本的 3 个有效目标一起平均，`torch.testing.assert_close` 对照手工 `log_softmax`；反向之后被忽略位置梯度全零、有效位置有非零梯度。`test_every_split_pair_and_empty_group`：三种重叠 + 三种空集合共 6 种情况全部抛 `ValueError`。

`test_continual_decision_priority`：`domain=-0.2, retention=0.01, max=0.05` 得到 `continue_with_evaluation`；retention 改成 0.08 得到 `tune_or_rollback`；domain 改成 0.01 得到 `tune_data_or_schedule`；checkpoint 只有 `model` 得到 `blocked_unrecoverable`；domain 为 `nan` 得到 `blocked_nonfinite`。`test_continual_boundaries_and_blocking_priority` 再验证 retention 等号通过、domain 为 0 需调整、`update_finite=False` 与非有限值优先、缺项优先于 NaN，以及 `max_retention_increase` 为 `-1.0` 或 `nan` 时抛 `ValueError`。

两个测试块合计 8 个测试，全过打印 `✅ 8 项持续预训练机制测试通过`。notebook 没有保存任何单元格输出；框架核对通过时返回 `status: 'passed'` 和三项布尔检查。GPU 实验默认 `dry_run`，真实运行时读取 `benchmarks/results/train_continual_pretrain/train-continual-pretrain_<run_label>.json` 里的 `metrics.before`、`after`、`domain_loss_change`、`retention_loss_change`、`finite_all`、`median_step_ms`、`peak_allocated_mb` 和 `decision`。
