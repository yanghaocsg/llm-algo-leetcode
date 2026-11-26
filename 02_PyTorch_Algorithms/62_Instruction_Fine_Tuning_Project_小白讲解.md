# 62. SFT 训练交付项目 — 小白讲解

> 对应 notebook：[`62_Instruction_Fine_Tuning_Project.ipynb`](./62_Instruction_Fine_Tuning_Project.ipynb)

## 一句话总结
本节回答两个工程问题：训练到一半保存的文件，能不能接着训练？最终保存的 LoRA 适配器，换一个新模型实例加载后，能不能复现同样的验证损失？CPU 部分用三个 TODO 把"选型清单可消费、检查点可恢复、产物可交付"写成硬门槛；GPU 部分按 `prepare → measure → close` 跑一次 Qwen2.5-0.5B 的 24 步 SFT，中途在第 12 步保存、同进程重建模型/优化器/调度器后继续，最后保存并重载适配器。已保存的 `sft_delivery_v2_1` 是历史协议下的结果，决策为 `accept`，`final_eval_loss` 为 1.7079639425501227，`reload_loss_abs_diff` 为 0.0；但它没有当前 v4 门禁要求的 `measurement_version` 和三类校验标志，所以这个 `accept` 不能当成 v4 已通过，v4 仍在等待真实 GPU 运行登记。

## 为什么需要它：痛点是什么
初学者容易把"保存了模型"当成"保存了训练"。只存适配器权重，恢复时模型参数是对的，但优化器的一阶矩、二阶矩没了，学习率调度器的进度没了，随机数状态和"下一条样本读到哪"也没了，继续训练出来的轨迹和原训练不是同一条。另一类困惑是"训练完了、损失也降了，为什么还不能交付"：因为还没有证明新实例加载适配器后能复现保存前的损失，也没有证明文件哈希与清单一致。

生活类比：打游戏的中途存档和通关存档。检查点（checkpoint）是中途存档，必须包含角色状态、关卡进度、背包和随机种子，读档后才能从原地继续；适配器（adapter）是通关存档，只保留最终角色能力，用来开新游戏。你不能拿通关存档去续关，也不能只看"通关了"就说明中途读档没问题。

## 核心问题
- 选型清单里的"接受"为什么不够？为什么还要求至少两次、不同随机种子的复测，而且必须是同一个候选？
- 恢复训练除了适配器权重，还要保存哪些状态？"续训首步 = 保存步 + 1"检查的是什么？
- 训练损失下降、验证损失达标、重载损失一致，这三件事分别说明什么？
- 为什么 `decide_sft_delivery` 用合取（所有门槛都过）而不是打分？哪些情况必须 `reject`？
- GPU 实验里第 12 步保存、第 24 步完成，`saved_step`、`resumed_from_step`、`completed_step`、`training_update_count` 各是什么？
- 历史 `sft_delivery_v2_1` 的 `accept` 和 v4 协议"待验证"矛盾吗？

## 解决思路：一步步看懂

**第 1 步：先分清四类产物。** 选型清单（selection manifest）记录 63 节选出的 LoRA 配置、复测证据和适配器路径；检查点保存继续训练所需的适配器、优化器、调度器、随机数、训练步和下一条样本位置；适配器是训练结束后的可加载权重；产物清单（manifest）记录这些文件的位置和 SHA256。哈希的作用是检查内容有没有变，不能证明训练质量。四者的关系是：选型清单决定"训练什么"，检查点保证"能接着训练"，适配器和交付清单保证"能复现和交付"。

**第 2 步：TODO 1，校验选型清单。** 骨架提供了 `valid_selected_config(config)`：`name` 必须是非空字符串；`rank`、`alpha` 必须是正整数；`dropout` 必须有限且落在 $[0,1)$；`target_modules` 必须是非空、元素非空且不重复的字符串列表。`validate_selection_manifest(manifest)` 先构造六个检查：

```python
checks = {
    "accepted": manifest.get("decision") == "accept",
    "config_complete": required_config.issubset(selected_config) and valid_selected_config(selected_config),
    "repeat_complete": len(repeat_results) >= 2 and all(item.get("status") == "ok" for item in repeat_results),
    "same_candidate": False,
    "independent_seeds": False,
    "artifact_present": bool(manifest.get("artifact", {}).get("adapter_path")),
}
```

要补的两项是候选归属和种子独立性：

```python
checks["same_candidate"] = bool(selected_name) and all(
    item.get("name") == selected_name and ("config" not in item or item["config"] == selected_config)
    for item in repeat_results)
checks["independent_seeds"] = (
    all(type(seed) is int and seed >= 0 for seed in repeat_seeds)
    and len(repeat_seeds) == len(set(repeat_seeds)))
```

`same_candidate` 不仅比较名字，还在复测记录带 `config` 时比较整份配置：同名但 rank 从 8 漂到 16 也必须失败。`independent_seeds` 用 `len(list) == len(set)` 查重，并用 `type(seed) is int` 排除布尔值。最后：

```python
failed_checks = [name for name, passed in checks.items() if not passed]
consumable = not failed_checks
```

测试覆盖四种情况：完整清单 `consumable` 为真；只留一条复测记录时失败；混入 `rank16` 时 `failed_checks == ["same_candidate"]`；两条都用 seed 42 时 `failed_checks == ["independent_seeds"]`。注意测试只要求这些构造下恰好出现那一个失败项，所以其他检查必须都通过。

**第 3 步：TODO 2，检查恢复条件。** `check_resume_contract(checkpoint, expected)` 先算三项：`state_complete` 要求 `adapter_present`、`optimizer_state_present`、`scheduler_state_present`、`rng_state_present` 全部 `is True`；`progress_valid` 要求 `global_step` 和 `total_steps` 都是 `int` 且 `0 < global_step < total_steps`；然后补两项：

```python
identity_keys = ("model_commit", "data_hash", "config_hash")
identity_matches = all(
    isinstance(expected.get(key), str) and bool(expected[key].strip())
    and checkpoint.get(key) == expected[key] for key in identity_keys)
continuation_valid = (
    progress_valid and type(expected.get("first_resumed_step")) is int
    and expected["first_resumed_step"] == checkpoint["global_step"] + 1)
resumable = identity_matches and state_complete and progress_valid and continuation_valid
```

`identity_matches` 要求"期望值本身是非空字符串"且"检查点与它完全相等"：如果 `expected` 里根本没有 `model_commit`，`checkpoint.get(...)` 与 `None` 比较也不会误判为真。`continuation_valid` 检查的是"恢复后的第一步必须紧接保存步"，而不是只检查"保存步小于总步数"；两个条件都满足才叫连续。测试里 expected 是 `total_steps=20`、`first_resumed_step=11`，checkpoint 的 `global_step=10` 且四项状态齐全，必须 `resumable`；把 `data_hash` 改成 `d2` 或把 `scheduler_state_present` 改成 False 都必须失败；把 `first_resumed_step` 改成 12 时 `continuation_valid` 必须为假。

**第 4 步：TODO 3，交付判断是合取。** `decide_sft_delivery(evidence, gates)` 构造八个检查：`execution`（`status == 'ok'`）、`resume`（`saved_step` 是整数、位于 0 和目标之间、且等于 `resumed_from_step`）、`completed_budget`（`final_step == target_step` 且目标为正）、`finite_metrics`（三个指标和两个门槛都有限非负）、`quality`（`final_eval_loss <= initial_eval_loss + maximum_eval_loss_regression`）、`reload`（`reload_loss_abs_diff <= maximum_reload_loss_abs_diff`）、`update_dynamics`（`finite_all is True`）、`artifact`（`sha256` 非空）。参考答案只有一句：

```python
decision = "accept" if all(checks.values()) else "reject"
```

为什么用合取？因为交付不是"平均分及格"，而是"每一项都必须成立"：预算没跑完、验证损失退化、重载对不上、产物没哈希，任意一项都足以让下游无法信任这份基线。`quality` 的门槛是"相对初始损失不退化"，测试和 GPU 合同里 `maximum_eval_loss_regression=0.0`，所以最终损失只要比初始高一点就失败；`reload` 的容差是 `1e-4`。测试的正常证据是 initial 1.9、final 1.8、reload 1e-6、saved 10、target 20、artifact 有哈希，应 `accept`；把 reload 改成 0.01 或把 artifact 改成 `{}` 都应 `reject`。

**第 5 步：`test_sft_fail_closed_boundaries` 在补哪些洞。** 这组测试专门检查"默认值、负数、NaN、Inf、布尔值"这些容易漏的边界：`expected` 只给 `total_steps=20`、`first_resumed_step=11`，checkpoint 没有三个身份字段，必须不可恢复；把 `checkpoint` 换成 `{}`、`reload_loss_abs_diff` 改成 -1、`final_eval_loss` 改成 NaN、`initial_eval_loss` 改成 Inf、`target_step` 改成 `True`，都必须 `reject`；把任一门槛改成 `inf`，也必须 `reject`；复测记录重复同一条（seed 相同）或带 `config rank16`，都必须不可消费。这里的关键是 `type(value) is int` 和 `finite_number` 都不接受布尔值：Python 里 `True` 是 `int` 的子类，不显式排除就会被当成步数 1。

**第 6 步：v4 的可见运行门禁。** 5.5 的 `decide_sft_delivery_for_run(evidence, gates, target_steps)` 比 CPU 版更严：除了上面的检查，还要求 `measurement_version == 'sft-delivery-recovery/v4'`；`checkpoint` 里 `identity_verified`、`files_sha256_verified`、`sample_cursor_verified` 全为真；`optimizer_state_present`、`scheduler_state_present`、`rng_state_present` 全为真；`completed_step == target_steps`；`training_update_count == completed_step`；`quality_absolute` 要求 `final_eval_loss <= gates['maximum_eval_loss']`；`quality_relative` 要求相对初始不退化。测试 `test_visible_sft_run_gate` 不执行 GPU，它读取 notebook 的 JSON，用 `ast.parse` 解析 id 为 `2ababa5c` 的单元，只 `exec` 其中的函数定义，再构造 v4 证据：正常情况 `gate(e, limits, 24)` 为 `accept`，目标步改成 30 为 `reject`，reload 改成 NaN/Inf/-1 为 `reject`，五个 checkpoint 标志任一为 False 也为 `reject`。

**第 7 步：GPU 实验的固定条件。** 5.1 写明模型 Qwen2.5-0.5B、长度 256、LoRA，配置来自 63 节，数据来自 64 节；5.2 从 `benchmarks/project_registry/part02.json` 找到 `PT-SFT-DELIVERY`，再读合同 `sft_delivery_v4.json`，并核对选型清单、扫描结果和数据清单的 SHA256。5.3 的门槛是 `maximum_eval_loss=2.0`、`maximum_eval_loss_regression=0.0`、`maximum_reload_loss_abs_diff=1e-4`；`checkpoint_step=12`、`total_steps=24`，且代码先检查 `0 < checkpoint_step < total_steps`。训练按列表顺序逐条取样，单卡 micro-batch 1、累积 1；检查点记录模型提交、实际分词后的样本顺序摘要、LoRA 配置和下一条样本位置，重建前核对文件哈希；保存的状态包括 Torch CPU/CUDA 随机数和常数学习率调度器。

**第 8 步：评测口径与探针。** 5.3 写明：先计算每条样本"回答部分"的平均 token loss（包含保留的 EOS，排除提示和 padding），再对样本等权平均。梯度范数覆盖全部梯度；参数增量只探测首个可训练张量的前 4096 个元素；有限性检查覆盖每次更新后的全部可训练参数。这三者分工不同：梯度范数看更新尺度，参数探针看"参数到底动没动"，全参数有限性看数值有没有坏掉，不能互相替代。

**第 9 步：已保存的运行结果怎么读。** notebook 保存的是 `sft_delivery_v2_1` 的历史结果：5.2 输出 `{'preflight': 'ok', 'selected': 'rank4'}`；5.4 保存 evidence 和 artifact，artifact 的 SHA256 前缀是 `53204548...`；5.5 的决策是 `{'decision': 'accept', 'failed_checks': []}`，`initial_eval_loss` 为 1.772523831576109、`final_eval_loss` 为 1.7079639425501227，`reloaded_eval_loss` 与最终值相同，`reload_loss_abs_diff` 为 0.0；checkpoint 的 `saved_step=12`、`resumed_from_step=12`、`completed_step=24`、`training_update_count=24`，`optimizer_state_present=True`，checkpoint SHA256 前缀是 `629455a0...`；`peak_allocated_bytes=1893894144`，约 1.76 GiB。5.6 的登记输出 `{'decision': 'accept', 'failed_checks': []}`，并生成 manifest 路径。

但 5.6 的文字同时说明：`sft_delivery_v2_1` 是旧交付链，缺少 v4 的检查点文件哈希、样本顺序、下一条样本位置和每次更新的全参数有限性检查，因此要按"历史协议"解读。对照 5.5 当前展示的 `decide_sft_delivery_for_run`，这一点可以静态验证：这份 evidence 里没有 `measurement_version`，`checkpoint` 里也只有 `optimizer_state_present` 等旧字段，没有 `identity_verified`、`files_sha256_verified`、`sample_cursor_verified`；按当前函数定义，`recovery_protocol` 和 `recovery_checks` 会失败。也就是说，dump 里 5.5 的 `accept` 是旧协议下保存的输出，不是当前 v4 门禁对这份记录的结论；不要把两者混在一起。读结果时先看 `measurement_version` 和 `failed_checks`，再看 `final_eval_loss`、`reload_loss_abs_diff` 和 `completed_step`，最后看产物哈希是否存在。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| SFT | 监督微调：用输入和标准回答更新模型 |
| LoRA adapter | 冻结基座，只训练低秩投影得到的小权重 |
| checkpoint | 中途存档：适配器、优化器、调度器、随机数、步数和数据位置 |
| selection manifest | 选型清单：选中的配置、复测记录和适配器路径 |
| `global_step` | 检查点保存时已经完成的训练步数 |
| `first_resumed_step` | 恢复后第一次更新应处的步数，必须等于保存步加一 |
| 重载一致性 | 新模型实例加载适配器后，验证损失与保存前一致 |
| `reload_loss_abs_diff` | 重载前后损失的绝对差，门槛 1e-4 |
| 合取门禁 | 所有检查都通过才 `accept`，任一失败即 `reject` |
| 样本游标 | 恢复到"下一条该读哪条样本"的位置 |
| fail-closed | 输入缺失、NaN、Inf、布尔值一律按失败处理 |
| 观察性证据 | 只证明当前条件下跑通，不足以推广到其他配置 |

## 小白常见疑问

**Q1：保存了适配器权重，为什么还不能恢复训练？**
因为一次更新不只依赖权重。AdamW 为每个参数维护一阶矩和二阶矩，调度器决定当前学习率，随机数状态影响 dropout 等随机操作，数据游标决定下一条样本。缺任何一项，恢复后的梯度统计、学习率和样本顺序都可能与原训练不同。`check_resume_contract` 要求四项状态全部存在，还要求模型提交、数据哈希、配置哈希与当前训练一致；适配器只是其中一份文件。

**Q2：为什么复测至少要两次，而且种子不能重复？**
一次运行无法区分"配置好"和"这次运气好"。两次以上、不同种子，才能看到配置在随机性下的表现。但如果两次记录用同一个种子，它们不是独立复测，只是同一次结果的复制，所以 `independent_seeds` 用 `len(repeat_seeds) == len(set(repeat_seeds))` 查重。测试里两条 seed 42 的记录会让 `failed_checks == ["independent_seeds"]`。

**Q3：`final_eval_loss` 比初始低，为什么还可能 `reject`？**
因为交付有多个独立条件。验证损失只是其中一项：预算没跑完（`final_step != target_step`）、重载差超过 1e-4、`update_dynamics.finite_all` 不为真、产物没有 SHA256，任意一项都会 `reject`。代码用 `all(checks.values())` 而不是加权平均，就是要表达"短板决定结论"。测试专门把 reload 改成 0.01、artifact 改成 `{}`，验证这两种情况都会被拒绝。

**Q4：第 12 步保存、第 24 步完成，为什么 `training_update_count` 也是 24？**
训练从基座开始先做 12 次更新，保存检查点；然后在同一进程里重建模型、优化器和调度器，从第 12 步继续做 12 次更新，累计 24 次。`saved_step=12` 是保存位置，`resumed_from_step=12` 是恢复起点，`completed_step=24` 是最终位置，`training_update_count=24` 是总更新次数。`decide_sft_delivery_for_run` 要求 `completed_step > resumed_from_step` 且 `training_update_count == completed_step`，用来排除"只跑了一半却说完成了"。

**Q5：`reload_loss_abs_diff` 为 0.0，是不是说明保存和加载绝对没有问题？**
只能说明这次测量里，新实例加载适配器后的验证损失与保存前完全相同（在记录的精度下）。它证明保存/加载路径在当前配置下是一致的，不证明模型质量、泛化能力或安全性。notebook 把"训练完成""验证质量""重载一致性"分成三件事：`final_eval_loss` 看任务质量，`reload_loss_abs_diff` 看保存加载是否一致，`completed_step` 看预算是否完成；三者都过才形成 SFT 基线，但基线仍然不等于偏好训练或通用能力评测的结论。

## 动手验证：怎么知道自己做对了
CPU 测试单元先跑 `test_selection_manifest_contract`、`test_resume_contract`、`test_sft_delivery_decision`，打印 `所有测试通过！`；然后定义并运行 `test_sft_fail_closed_boundaries`，再用 `_load_visible_run_gate` 从 notebook JSON 里提取 v4 门禁函数，运行 `test_visible_sft_run_gate`。dump 中没有保存这些测试的输出，以断言为准。注意 `print("所有测试通过！")` 在单元中间就出现，后面还有两组测试，这是单元书写顺序导致的，不代表后面两组没跑。

- `test_selection_manifest_contract`：完整清单的 `consumable` 为真；只剩一条复测记录时为假；混入 `rank16` 时 `failed_checks == ["same_candidate"]`；两条 seed 42 时 `failed_checks == ["independent_seeds"]`。
- `test_resume_contract`：`global_step=10`、`total_steps=20`、`first_resumed_step=11` 且四项状态齐全时 `resumable` 为真；`data_hash='d2'` 或 `scheduler_state_present=False` 时为假；`first_resumed_step=12` 时 `continuation_valid` 为假。
- `test_sft_delivery_decision`：initial 1.9、final 1.8、reload 1e-6、saved 10、target 20、artifact 有哈希时 `accept`；reload 0.01 或 artifact `{}` 时 `reject`。
- `test_sft_fail_closed_boundaries`：缺身份字段不可恢复；checkpoint `{}`、reload -1、final NaN、initial Inf、target `True` 全部 `reject`；门槛为 `inf` 也 `reject`；重复种子或配置漂移不可消费。
- `test_visible_sft_run_gate`：v4 证据在 `target_steps=24` 时 `accept`，目标改成 30 时 `reject`；reload 为 NaN/Inf/-1 时 `reject`；`scheduler_state_present`、`rng_state_present`、`identity_verified`、`files_sha256_verified`、`sample_cursor_verified` 任一为 False 都 `reject`。

GPU 侧按阶段自查：`prepare` 应打印 `stage`、`profile`、`selection_label` 和 `evidence_path`，不训练；`measure` 应保存 evidence、checkpoint 和 adapter，并打印 artifact 的路径与 SHA256；`close` 应打印 `{'decision': 'accept', 'failed_checks': []}` 并生成 `sft_delivery_manifest_<RUN_LABEL>.json`。已保存的 `sft_delivery_v2_1` 结果是历史协议：`final_eval_loss=1.7079639425501227`、`reload_loss_abs_diff=0.0`、`completed_step=24`；如果你的新标签跑出 v4 记录，还要确认 `measurement_version` 是 `sft-delivery-recovery/v4`、三类 SHA256 校验标志为真，才能按新协议解读。
