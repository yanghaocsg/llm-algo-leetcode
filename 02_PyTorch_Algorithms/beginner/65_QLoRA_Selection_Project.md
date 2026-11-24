# 65. QLoRA Selection Project | QLoRA 选型项目 · 小白入门

> 配套 notebook：[65. QLoRA 选型项目](../65_QLoRA_Selection_Project.ipynb) ｜ 难度：Hard ｜ 环境：CPU + 单卡 GPU

## 一句话说清

QLoRA 把 LoRA 微调里不训练的权重压成 4 位来存。这本 notebook 教你判断它能否替代普通 LoRA：先确认其他条件相同，再看证据够不够。

## 为什么需要它

用 LoRA 微调时，真正更新的只有一小部分参数，其余的预训练权重原样放着、不训练。显存不够时，一个办法是把这些不训练的权重压成 4 位来存，这就是 QLoRA。

但“省显存”只是一半。看测试样例里的两条训练路径：

- BF16 LoRA 的验证损失更好（1.10），但显存 14500 超过了上限 12000，进不了预算。
- 换成 NF4 QLoRA，显存是 9800，验证损失变成 1.16。这点损失能不能接受，要靠上限来判。
- 你想拿纸面估算或者只跑一次的结果下结论。notebook 的态度是：这种证据只能记为 `tune`，不能写成选型结论。

这本 notebook 解决的是“怎样判断”：先确认两条路径除了冻结底座以外其他条件都相同，再逐项检查证据，最后给出采用、调优或拒绝。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 冻结底座 | 微调时不更新的那部分预训练权重，LoRA 只训练挂在它旁边的适配器 |
| 适配器（adapter） | LoRA 训练出来的那部分小参数，保存下来就是本项目的产物 |
| BF16 / NF4 | 两种存数字的格式：BF16 用 16 位，NF4 用 4 位。两条路径分别用它们存冻结底座 |
| 工作负载（workload） | 一次实验固定的那套条件，比如数据、长度、步数、随机种子 |
| 微批次（micro-batch） | 一次前向传播同时喂给模型的样本数。显存不够会报 OOM（显存用尽） |
| 显存账本 | 把训练时占显存的几大项逐项加起来的估算表，只是估计，不是实测 |
| `accept` / `tune` / `reject` | 决策的三种结果：采用、补证据或调参后再看、拒绝 |

## 核心思路

输入是两条训练路径各自的记录：基线是 BF16 LoRA，候选是 NF4 QLoRA。每条记录里有身份、显存、最大微批次、吞吐、验证损失和证据等级。你依次做三次判断，输出是 `accept`、`tune` 或 `reject`。

```text
身份相同吗 → 四项门槛都过吗 → 证据够不够 → accept / tune / reject
```

notebook 里有两个式子，写成纯文本是：

```text
账本估算总量 = 底座权重 + 可训练参数 + 梯度 + 优化器状态 + 激活 + 量化元数据
对账差值     = 实测峰值显存 - 账本估算总量
```

账本估算总量是把占显存的六项加起来，对账差值是实测峰值减去这个估算。两者不一定相等，所以账本只能决定一个候选值不值得进入 GPU 对照，不能当交付证据。

身份指 7 个字段：`model_commit`、`data_manifest_hash`、`adapter_signature`、`effective_batch`（每次更新参数时汇总的样本数）、`max_length`、`update_steps`、`seed_set`。四项门槛是：显存不超过上限、最大微批次不小于最小值、吞吐不低于下限、验证损失不超过上限。看一个小例子：

```python
# 身份：7 个字段都必须存在，而且完全相同
required = ("model_commit", "data_manifest_hash", "adapter_signature", "effective_batch",
            "max_length", "update_steps", "seed_set")
lora_id = {"model_commit": "m1", "data_manifest_hash": "d1", "adapter_signature": "r8-qv",
           "effective_batch": 2, "max_length": 256, "update_steps": 12, "seed_set": "42,123,2024"}
qlora_id = {**lora_id, "max_length": 512}     # 只把长度改成 512
print(all(lora_id[k] == qlora_id[k] for k in required))

# 门槛：四项同时通过才算可行
budget = {"memory_cap_mb": 12000.0, "min_micro_batch": 1, "min_tokens_per_s": 18.0}
floor = {"max_val_loss": 1.20}
lora = {"memory_mb": 14500.0, "max_micro_batch": 1, "tokens_per_s": 20.0, "val_loss": 1.10}
qlora = {"memory_mb": 9800.0, "max_micro_batch": 2, "tokens_per_s": 19.0, "val_loss": 1.16}

def passes(c):
    checks = {
        "memory": c["memory_mb"] <= budget["memory_cap_mb"],
        "scale": c["max_micro_batch"] >= budget["min_micro_batch"],
        "throughput": c["tokens_per_s"] >= budget["min_tokens_per_s"],
        "quality": c["val_loss"] <= floor["max_val_loss"],
    }
    return checks, all(checks.values())

print(passes(lora))
print(passes(qlora))

# 账本：把占显存的几项加起来，再和实测峰值对照
print(100.0 + 10.0 + 10.0 + 40.0 + 200.0 + 5.0, 380.0 - 365.0)
```

```text
False
({'memory': False, 'scale': True, 'throughput': True, 'quality': True}, False)
({'memory': True, 'scale': True, 'throughput': True, 'quality': True}, True)
365.0 15.0
```

第一行：只改了长度这一个字段，身份就不相同，两条路径不能比。中间两行：LoRA 的验证损失更低，但显存超限，所以不可行；QLoRA 四项都过，算可行。最后一行：账本估算是 365，实测峰值 380，对账差值是 15。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 固定任务与预算，明确要比较什么：不是“QLoRA 是否更省显存”，而是它能否在显存上限内保住可接受的验证质量 | 读 |
| Step 2 | 保持更新语义，只改变冻结底座表示：两条路径哪些必须固定，哪里是主要变化 | 读，对照上面的身份字段 |
| Step 3 | 从资源账本走到可交付训练证据：账本只能估计，真实训练还要确认哪些证据 | 读，对照上面的账本公式 |
| Step 4 · TODO 1 | `same_workload_identity`：比较两条路径的模型、数据、adapter 与有效 batch 身份 | 在 `same_workload_identity` 里补 1 行：7 个身份字段都存在且完全相同才返回真 |
| Step 4 · TODO 2 | `candidate_passes_gates`：把资源、规模、速度和质量合成硬门槛 | 在 `candidate_passes_gates` 里补 1 行：四项检查同时通过 |
| Step 4 · TODO 3 | `choose_qlora_action`：从两条可比较路径的证据决定交付动作 | 在 `choose_qlora_action` 里补 5 行 `return`：没有可行候选是 `reject`，证据不足是 `tune`，QLoRA 可行且资源最优才是 `accept` |
| 测试 | 分别验证门槛组合、候选排序和项目决策，最后检查完整链路 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上配对比较 BF16 LoRA 与 NF4 QLoRA，再跑容量阶梯 | 需要单卡 GPU、`Q-QLORA` 环境和 bitsandbytes；默认是 `RUN_STAGE = "prepare"`，只做只读准备，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU，不加载模型。Step 5 需要 GPU，并先准备 `Q-QLORA` 环境。

## 容易踩的坑

- **一个身份字段漂移还继续比**：任一字段漂移都会改变训练语义或工作负载，差异不能归因于 NF4。测试把 `max_length` 从 256 改成 512，结果必须是假。
- **把账本估算当实测**：账本只能估计容量，作用只是决定候选值不值得进入 GPU 对照。只有估算或单次 smoke（只跑一次的冒烟测试）的证据，决策只能是 `tune`。
- **把门槛通过当成可以交付**：TODO 2 的通过只代表“可以测”。最终交付还要求低比特路径生效、配对 GPU 重复和 artifact 证据；CPU 简化摘要不核验磁盘 artifact。
- **用质量抵消显存超限**：显存、微批次、吞吐和验证损失必须同时通过。上面的例子里 LoRA 的验证损失更好，但显存超限，照样不可行。
- **把所有差异都归因于 4 位表示**：真实 NF4 路径还包含 k-bit preparation，要同时记录它的 `dtype` 和梯度检查点行为。容量采用和速度采用也是两个判断，不能用资源排序代替速度收益。

## 读完后自查

- [ ] 能说出两条路径的身份要比较哪 7 个字段。
- [ ] 账本六项是 100、10、10、40、200、5，实测峰值是 380，能算出估算总量和对账差值。
- [ ] 给一份候选的显存、最大微批次、吞吐和验证损失，能判断四项门槛是否都通过。
- [ ] 能说出什么时候是 `reject`、什么时候是 `tune`、什么时候才是 `accept`。

## 下一步

- 打开 notebook：[65. QLoRA 选型项目](../65_QLoRA_Selection_Project.ipynb)
- 建议先读：[14. 低秩适配教程 · 小白入门](./14_LoRA_Tutorial.md)、[11. 梯度累积 · 小白入门](./11_Gradient_Accumulation.md)、[15. SFT 有效更新与训练闭环 · 小白入门](./15_End_to_End_Fine_Tuning_Experiment.md)、[26. QLoRA 与 4-bit 量化 · 小白入门](./26_QLoRA_and_4bit_Quantization.md)
- 做完之后：[84. DPO 偏好优化项目 · 小白入门](./84_DPO_Preference_Project.md)
- 相关项目：[67. 量化推理与部署](../67_Quantized_Inference_and_Deployment.ipynb)、[76. Activation / Checkpoint / Offload 对比项目](../76_Activation_Checkpoint_Offload_Benchmark.ipynb)
