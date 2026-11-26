# 15. SFT 有效更新与训练闭环 — 小白讲解

> 对应 notebook：[`15_End_to_End_Fine_Tuning_Experiment.ipynb`](./15_End_to_End_Fine_Tuning_Experiment.ipynb)

## 一句话总结
本节用 `TinyCausalLM` 和两个 TODO 把 SFT 训练闭环接起来：`evaluate_loss` 在不污染模型模式和已有梯度的前提下复用训练损失；`run_finetuning_experiment` 用整个逻辑批次的有效监督词元总数做**共同分母**，把多个微批次的损失总和累积成一次有效更新，调度器、评估和 `history` 都按这次更新编号推进。notebook 还保存了一次 Qwen2.5-0.5B 的真实运行结果，用来示范"机制跑通"和"模型变好"是两回事。

## 为什么需要它：痛点是什么
SFT 的损失函数本身不难，难的是"时钟"对齐。初学者常见的四个坑：

- **梯度累积时除以 `accum_steps`。** 当各微批次的回答长度不同，短回答里的每个词元会被算得更重；正确做法是除以整个逻辑批次的有效词元总数。
- **评估时忘了 `model.eval()`，或者忘了恢复。** Dropout 还在工作，下一次训练结果漂移；更隐蔽的是评估时没关 `torch.no_grad()`，显存悄悄涨上去。
- **调度器按 micro step 推进。** 一个逻辑批次里 `scheduler.step()` 被调用多次，学习率多走了好几步。
- **日志里的 step 是微批次计数。** 记录的"第 10 步"和参数真正写回的第 10 次对不上，曲线没法解释。

生活类比：一桌人吃饭 AA 制。正确做法是先算全桌总账（所有有效词元的损失总和），再除以总人数（总有效词元数）；如果每桌各自算人均、再把几张桌的人均一平均，人少的那桌每个人就会被算得更重。梯度累积的"共同分母"就是这张总账。

## 核心问题
- 训练和评估的损失为什么应该相等？什么情况下数值会不同？
- 梯度累积时，为什么不能每个微批次各取平均再平均？
- `evaluate_loss` 怎样保证正常返回、异常退出都不污染模型状态？
- 一次有效更新里，参数、优化器、调度器、评估记录各推进几次？
- 保存的真实 GPU 结果能说明什么、不能说明什么？

## 解决思路：一步步看懂

**第 1 步：训练与评估共享损失，但隔离状态。** 对固定批次 $B$ 和参数 $\theta$，两条路径都先累加有效监督位置的损失，再除以目标总数，所以

$$\mathcal{L}_{\mathrm{train}}(B;\theta)=\mathcal{L}_{\mathrm{eval}}(B;\theta)=\mathcal{L}_{\mathrm{SFT}}(B;\theta)$$

这个等式的前提是"相同参数、输入和确定性的前向"。Dropout 在训练时随机屏蔽特征、评估时关闭，所以同一批数据在两种模式下数值可能不同；但监督位置、预测与标签的对齐方式、归约分母必须一致。训练路径保留计算图、产生梯度并写回参数；评估路径只读参数，算完恢复调用前的模式，并且不能改动已有梯度或优化器状态。

**第 2 步：一次有效更新会推进哪些状态。** Step 2 的表格列了五类：累计梯度在更新边界前由多个微批次共同形成，中途被清零或出现非有限值就是异常；模型参数每个逻辑批次只写回一次；优化器状态与参数写回同步推进一次；学习率状态与有效更新编号保持一致；评估与记录使用同一次更新后的模型，记录编号也按有效更新而不是微批次。

**第 3 步：共同分母的公式。** 记第 $u$ 次更新的第 $k$ 个微批次有 $N_{u,k}$ 个有效监督词元、损失总和为 $S_{u,k}$，逻辑批次总数 $N_u=\sum_k N_{u,k}>0$。累计梯度应为

$$g_u=\sum_k\nabla_\theta\frac{S_{u,k}}{N_u},\qquad \theta_{u+1}=\mathrm{OptimizerStep}(\theta_u,g_u)$$

注意每个微批次除的都是**同一个** $N_u$，不是自己的 $N_{u,k}$。举例：两个微批次各有两条样本，但有效监督词元分别是 4 和 10，总分母是 14；如果先各自取平均、再各乘 $1/2$，短回答每个词元的权重是 $1/(2\times4)=1/8$，长回答是 $1/(2\times10)=1/20$，而正确值都应是 $1/14$——短回答被放大 1.75 倍，长回答被压到 0.7 倍，两者权重比从 1:1 变成 2.5:1。只有各微批次有效目标数相等时，等权平均才和共同分母等价。没有有效目标的微批次贡献零梯度；整个逻辑批次都没有目标时，平均损失无定义。

**第 4 步：`build_sft_batch` 与损失接口。** 测试用的 `TinyCausalLM` 是 `Embedding + GRU + Linear` 的微型模型，只负责把接口跑通，不是真实 Transformer。每条样本拼成 `prompt + response + eos`，`labels` 在 prompt 段填 `-100`（不参与损失），response 和 eos 保留原 id；超过 `max_len` 就截断，截断后如果 `labels[1:]` 里没有任何非 `-100`，直接抛 `ValueError("截断后没有有效监督 token")`。`count_supervised_targets` 做下一词元对齐：`targets = labels[..., 1:]`，统计非 `-100` 且 `attention_mask[..., 1:] != 0` 的位置。`compute_sft_loss_terms` 用 `logits[..., :-1, :]` 对 `targets` 做 `cross_entropy(..., ignore_index=-100, reduction="sum")`，返回损失总和与目标数；`compute_sft_loss` 再把总和除以目标数，得到平均损失。

**第 5 步：TODO 1——评估状态隔离。** 参考答案：

```python
was_training = model.training
try:
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch.get("attention_mask"))
        loss = compute_sft_loss(logits, batch["labels"], batch.get("attention_mask"))
finally:
    model.train(was_training)
return float(loss.item())
```

三个设计点。第一，先保存 `was_training`，最后用 `model.train(was_training)` 恢复原模式：如果调用方本来就在评估模式，评估完仍然是评估模式。第二，`torch.no_grad()` 关掉计算图，否则评估也会建图、占显存。第三，用 `finally` 而不是顺序执行，这样 `compute_sft_loss` 抛异常（例如整批没有有效词元）时也能恢复模式。最后把损失转成 Python `float` 返回，调用方比较、记录都更方便，也不会把张量带进后续计算。

**第 6 步：TODO 2——一次有效更新。** `run_finetuning_experiment` 先做输入校验：`accum_steps`、`num_updates`、`eval_every` 必须是正整数（用 `type(value) is not int` 连 `bool` 也拒绝）；逻辑批次必须有有效词元；`batch_size % accum_steps == 0`。然后记录初始 train/val 损失，按 `micro_size = batch_size // accum_steps` 切微批次。核心循环：

```python
optimizer.zero_grad()
for idx in range(accum_steps):
    mb = {key: value[idx*micro_size:(idx+1)*micro_size] ...}
    if count_supervised_targets(mb["labels"], mb.get("attention_mask")) == 0:
        continue
    logits = model(...)
    loss_sum, _ = compute_sft_loss_terms(logits, mb["labels"], ...)
    scaled_loss = loss_sum / logical_token_count   # 整个逻辑批次的总目标数
    scaled_loss.backward()
optimizer.step()
if scheduler is not None:
    scheduler.step()
```

`logical_token_count` 在循环外对**整个** `train_batch` 算一次，所以每个微批次共享同一个分母。空监督微批次直接 `continue`，不产生梯度，但也不会改变分母。参数只在 `accum_steps` 个微批次都处理完后写回一次，`scheduler.step()` 也只在同一个边界推进一次。评估记录的条件是 `step == 1 or step % eval_every == 0 or step == num_updates`，所以哪怕 `num_updates` 不是 `eval_every` 的倍数，最后一次也会被记录。函数最终返回 `report`，字段包括 `initial_train_loss`、`initial_val_loss`、`final_train_loss`、`final_val_loss` 和 `history`；`history` 里每一行是 `{'step': 有效更新编号, 'train_loss': ...}`，传了 `val_batch` 时还会多一个 `val_loss`。

**第 7 步：测试怎么把"时钟"钉死。** `_build_sft_case` 用 4 条相同的训练样本 `([1,2,3], [4,5,6,7])` 和 2 条验证样本 `([1,2,3], [4,5,6,8])`，`max_len=9`，拼出来是 `(4, 9)`。

- `_assert_sft_batch_contract`：前 3 列 `labels` 必须是 `-100`；第 9 列 `attention_mask` 为 0、`labels` 为 `-100`；`build_sft_batch([1,2,3,4], [5,6], max_len=5)` 的 `labels` 必须恰好是 `[-100,-100,-100,-100,5]`——注意是截断后的最后一个 `5`，不是 `6`。
- `_assert_next_token_loss`：固定种子 9，`logits` 形状 `(1,4,8)`，`labels=[-100,2,3,-100]`，`mask=[1,1,1,0]`；把 `compute_sft_loss` 的结果和"先移位、再按 mask 填 `-100`、再用 `CrossEntropyLoss(ignore_index=-100)`"的手算结果比较，检查下一词元对齐和 padding 屏蔽。
- `_assert_evaluation_state`：先训练模式前向 + `backward()`，保存参数和梯度；调 `evaluate_loss` 后，返回值必须是 `float`、`model.training` 仍为 `True`、已有梯度逐元素不变；再从 `model.eval()` 进入，调用后仍是 `False`；最后传一个全 `-100` 的非法批次，必须抛 `ValueError`，而模式仍恢复到训练、参数不变。
- `_assert_gradient_accumulation_equivalence`：两个 `TinyCausalLM` 共享初始权重，SGD `lr=0.01`；一个用 `accum_steps=2, num_updates=1`，另一个对完整批次做一次损失反传和 `step()`；参数必须在 `atol=1e-6, rtol=1e-5` 下 `allclose`。这组样本各微批次目标数相等（每条 5 个），所以共同分母和等权平均恰好等价。
- `_assert_token_weighted_accumulation`：两条短回答 `([1,2,3],[4])` 加两条长回答 `([1,2,3],[4,5,6,7])`，断言两个微批次的目标数是 `[4, 10]`。完整批次按 14 做分母；教学循环也必须得到同样的梯度和参数；再把"每个微批次各自取平均、各乘 1/2"的旧写法跑一遍，断言它和完整批次**不一致**，用反例证明共同分母不是可有可无。
- `_assert_training_update_and_report`：AdamW `lr=0.05`，`LambdaLR` 的 `lr_lambda = 1.0 / (1.0 + 0.01 * update)`，`accum_steps=2`、`num_updates=30`、`eval_every=10`；断言参数确实变了、`optimizer.state` 非空、`scheduler.last_epoch == 30`、`history` 的 `step` 恰好是 `[1, 10, 20, 30]`、最终训练损失低于初始、验证损失有限。
- `_assert_invalid_update_inputs`：全 `-100` 的逻辑批次，以及 `accum_steps`/`num_updates`/`eval_every` 取 `0`、`-1`、`1.5`、`True` 的组合，都必须在参数更新前抛 `ValueError`，且参数保持原样。

**第 8 步：可选的框架一致性核对。** notebook 用 Accelerate 对照"累积边界"：`Accelerator(cpu=True, gradient_accumulation_steps=2)`，每个微批次里把损失写成 `loss_sum * accelerator.gradient_accumulation_steps / logical_token_count` 再 `accelerator.backward(...)`。为什么要乘回 `gas`？因为 `accelerator.backward` 会自动把损失除以累积步数；乘回来再除共同分母，才和教学实现等价。`accelerator.sync_gradients` 在两个微批次上应该是 `[False, True]`，第一个微批次后参数不变，第二个之后才变；`scheduler.step()` 只在 `sync_gradients` 为真时调用。`check_sft_trainer_path` 再用一个 1 层、1 头、`n_embd=16` 的 `GPT2LMHeadModel` 和内存数据集跑 `SFTTrainer.train()`，断言 `global_step == 1` 且参数发生变化——它核对的是"更新接线"，词元加权的数值等价由上一项负责。框架没装时返回 `status: "skipped"`，提示用 `PT-CPU` 环境。

**第 9 步：Step 5 的真实 GPU 实验。** 固定配置是 `Qwen/Qwen2.5-0.5B` + `tatsu-lab/alpaca`，`stable_sample` 稳定抽 64 条候选、保留前 32 条有有效监督的样本，其中 24 条训练、8 条验证；`MAX_LENGTH=256`、`BATCH_SIZE=1`、`UPDATES=8`、`LEARNING_RATE=2e-5`、`SEED=42`，支持 BF16 就用 BF16，否则 FP16（`GradScaler` 只在 FP16 时启用）。监督只覆盖 response 和 EOS：`labels = [-100] * len(prompt_ids) + response_ids`，截断到 256 后若 `supervised == 0` 就丢弃该样本。评估函数把每个 batch 的 `model(**batch).loss`（默认按有效词元取平均）乘以该 batch 的有效词元数，再除以总词元数，得到**词元加权**的数据集损失。训练循环只做 8 次有效更新，每次只取一个 batch；24 条训练数据、batch 1 时 `train_batches` 有 24 个，`step % len(train_batches)` 只用到前 8 条。notebook 明确说明：这个实验只保存结果 JSON，不保存可部署 checkpoint，也不把少量更新外推成任务能力提升。

**第 10 步：保存的结果怎么读。** 已保存的 `rtx5070ti_sft_v1` 输出是：模型 commit `060db649...`、数据 commit `dce01c9b...`；训练集 1445 个监督词元、验证集 351 个，截断 1 条；baseline 的 train/eval 词元加权损失是 `1.5518 / 1.6544`，8 次更新后是 `1.3401 / 1.6343`；8 次更新所用 batch 的损失依次为 `1.6813, 1.7007, 1.8364, 1.5817, 0.3919, 0.7067, 0.5577, 1.4130`；中位更新时间 `98.478 ms`、P95 `173.030 ms`；峰值已分配 `5,006,659,072` 字节（约 `4.66 GiB`）、峰值保留 `5,624,561,664` 字节（约 `5.24 GiB`）；`failure` 为空，`evidence_level` 是 `real_model_training_mechanism_benchmark`，`decision` 是 `accept_mechanism`。

读法：先确认 `failure` 为空、模型和数据都解析到固定 commit；再看监督词元（1445/351）和截断数（1），确认监督链路有效；然后看训练前后损失——train 从 1.552 降到 1.340，eval 从 1.654 降到 1.634，且最终 eval 是有限值；最后看资源和时间，它们只对"0.5B、长度 256、batch 1、8 次更新"这个矩阵有效。注意单次更新损失是波动的（`0.39` 到 `1.84`），不能用某一步的下降判断趋势；`accept_mechanism` 只表示"训练损失低于初始且最终验证损失有限"，**不表示验证损失通过了回退门槛，也不表示模型能力提升**。notebook 5.4 也写了：采用决策还要另一套项目实验来设验证损失回退门槛、检查 checkpoint 交付和独立任务指标。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 逻辑批次 | 累积完共同参与一次参数更新的全部样本 |
| 微批次 | 逻辑批次里一次前向/反向处理的子集 |
| 有效更新 | 优化器真正写回一次参数；时钟按它计数，不按微批次 |
| 有效监督词元 | 参与损失的词元：非 `-100`、非 padding，且移位后仍是目标 |
| 共同分母 | 整个逻辑批次的有效词元总数，所有微批次共用 |
| 梯度累积 | 多个微批次的梯度先累加，最后只 `step()` 一次 |
| 评估隔离 | 评估不建图、不改参数/梯度，结束后恢复调用前模式 |
| `was_training` | 调用前的 `model.training`，用于精确恢复模式 |
| 更新时钟 | 参数、优化器、调度器、日志共享的更新编号 |
| `history` | 按有效更新记录的 `step`、`train_loss`、`val_loss` |
| 词元加权损失 | 先算每批平均损失，再按该批有效词元数加权汇总 |
| `accept_mechanism` | 仅表示训练损失下降且验证损失有限，不是质量结论 |

## 小白常见疑问

**Q1：为什么两个微批次要除以 14，而不是各除以 4 和 10？**
因为损失是"整个逻辑批次的平均"，分母必须是全批次的 14。各除以自己的目标数，等于让每个微批次内部先归一化，短回答（4 个词元）和长回答（10 个词元）就被赋予了相同总权重，短回答里的每个词元被放大。`_assert_token_weighted_accumulation` 专门构造了 `[4, 10]` 这组数，并断言旧的"各取平均再平均"写法得到的梯度和参数与完整批次不一致。

**Q2：空监督微批次直接 `continue`，会不会让梯度少算？**
不会。它本来就没有有效目标，`compute_sft_loss_terms` 会抛 `ValueError`，跳过它是正确语义；它的贡献是 0，而分母仍然是整个逻辑批次的 `logical_token_count`（在循环外算好），所以其他微批次的权重不会被改变。`_assert_token_weighted_accumulation` 的 `empty_first=True` 分支就是把前两条样本的 `labels` 全设成 `-100`，再断言结果仍与完整批次一致。

**Q3：`evaluate_loss` 为什么用 `finally`，直接写 `model.train()` 不行吗？**
直接写只能覆盖正常路径；如果 `compute_sft_loss` 因为"当前批次没有有效监督词元"抛 `ValueError`，后面的 `model.train()` 就不会执行，模型会停在 `eval` 模式，下一次训练的前向就错了。`finally` 保证异常路径也恢复。`_assert_evaluation_state` 用全 `-100` 的非法批次专门验证这一点：既要求抛 `ValueError`，又要求 `model.training` 仍为 `True`。

**Q4：`scheduler.last_epoch == 30` 说明了什么？**
说明调度器只按有效更新推进。循环里 `scheduler.step()` 在 `optimizer.step()` 之后调用，位置在微批次循环之外，所以 30 次更新对应 30 次调度；如果误把 `scheduler.step()` 放进微批次循环，`last_epoch` 会变成 60。`history` 的 `[1, 10, 20, 30]` 同理：`eval_every=10`，第 1 步、每 10 步和最后一步记录，编号是有效更新编号，不是微批次编号。

**Q5：训练损失下降、验证损失有限，是不是说明模型变好了？**
不是。保存的结果里 train 从 1.552 降到 1.340，eval 只从 1.654 降到 1.634，而且只训练了 8 次更新、用了 24 条训练样本中的前 8 条。`decision` 的 `accept_mechanism` 只检查两个条件：`train_decreased` 和 `evaluation_finite`。notebook 5.3 和 5.4 都明确说：这不表示验证损失通过回退门槛，也不表示任务能力提升；质量结论要靠独立任务集和项目实验。

## 动手验证：怎么知道自己做对了

测试区先调用 `test_no_shifted_target_is_rejected()`，再调用 `test_end_to_end_finetuning()`；全过时打印 `PASSED: 有效词元加权、评估隔离及更新时钟核对通过。`。任何断言失败都会直接抛异常，不会打印 PASSED。

- `test_no_shifted_target_is_rejected`：`build_sft_batch([], [20], max_len=1)` 必须抛 `ValueError`——只有 1 个位置、移位后没有目标，构造阶段就要拒绝。
- `test_end_to_end_finetuning` 内部依次跑七个断言组：`_assert_sft_batch_contract`（形状 `(4,9)`、prompt/padding 的 `-100`、截断标签 `[-100,-100,-100,-100,5]`）；`_assert_next_token_loss`（与手工交叉熵一致）；`_assert_evaluation_state`（返回 `float`、模式恢复、梯度和参数不变、异常也恢复）；`_assert_gradient_accumulation_equivalence`（等大微批次与完整批次在 `atol=1e-6, rtol=1e-5` 下一致）；`_assert_token_weighted_accumulation`（`[4,10]` 目标数、变长与空监督微批次都和完整批次一致、旧的等权平均必须产生差异）；`_assert_invalid_update_inputs`（空监督和非法参数在更新前失败，参数不变）；最后 `_assert_training_update_and_report`（参数改变、优化器状态建立、`scheduler.last_epoch == 30`、`history` 编号 `[1,10,20,30]`、最终 train 损失下降、val 损失有限）。
- 可选的框架核对：全部通过时打印 `✅ Accelerate 更新控制与 TRL SFTTrainer 路径一致。`；`check_accelerate_training_control` 的五个检查项是 `sync_boundary_match`、`no_early_parameter_update`、`parameter_updated_after_accumulation`、`final_parameters_match`、`scheduler_update_count_match`；`check_sft_trainer_path` 检查 `single_step_completed` 和 `parameters_updated`。环境缺依赖时状态是 `skipped`，不会误报失败。

GPU 部分的结果以保存的 JSON 为准：`rtx5070ti_sft_v1` 的 `failure` 为 `null`、`decision` 为 `accept_mechanism`、`train_supervised_tokens=1445`、`evaluation_supervised_tokens=351`、`truncated_samples=1`、`median_update_ms=98.478`、`p95_update_ms=173.030`，训练损失 `1.5518 → 1.3401`、验证损失 `1.6544 → 1.6343`。这些数字只对应 `Qwen2.5-0.5B`、长度 256、batch 1、8 次更新的当前矩阵，换模型、数据或长度都要重新测量。
