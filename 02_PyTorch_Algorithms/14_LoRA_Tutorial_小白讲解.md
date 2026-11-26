# 14. LoRA 低秩适配 — 小白讲解

> 对应 notebook：[`14_LoRA_Tutorial.ipynb`](./14_LoRA_Tutorial.ipynb)

## 一句话总结
本节先用一个最小 `LoRALinear` 把 LoRA 的三件事讲透：冻结基座、只训练低秩旁路 `A`/`B`、把 `(α/r)BA` 合并回基座；再用 CPU 测试核对参数形状、零初始化下的梯度顺序、前向公式与幂等合并。最后用 PEFT 的 `get_peft_model` / `merge_and_unload` 做框架一致性检查，并给出在 Qwen2.5-0.5B 上扫描三种目标模块集合的真实 GPU 结果。

## 为什么需要它：痛点是什么
全参微调要更新模型里的每一个权重，显存账本会同时多出梯度、优化器状态和激活，7B 模型很快就从"权重 14 GB"变成几十上百 GB。可很多业务只需要在已有能力上做小幅适配，并不想重写整个模型。初学者的困惑通常集中在：LoRA 到底训练了哪些参数？为什么 `B` 初始化成零？`α/r` 是干什么的？合并之后输出为什么还能对得上？

生活类比：给一本已经写好的书做批注。全参微调像把整本书重印一遍；LoRA 像在页边贴一张很小的便利贴，便利贴只能写少量独立方向的修改（低秩），原书内容不动。部署时可以把便利贴的内容誊抄进正文（合并），也可以让读者同时看正文和便利贴（未合并推理）。

## 核心问题
- 冻结基座后，LoRA 新增的参数到底有多少？形状为什么是 `[r, d_in]` 和 `[d_out, r]`？
- 为什么 `B` 初始化为零，训练第一步 `A` 的梯度却是零？这算梯度断了吗？
- 缩放系数 `α/r` 在前向和合并时分别怎么用？
- 合并（merge）怎样保证与未合并推理等价，且重复调用不会重复叠加？
- 真实 GPU 上扩大挂载范围会怎样改变命中层数、可训练参数、更新时间和合并误差？

## 解决思路：一步步看懂

**Step 1：先确定改哪里。** 注意力里有 `q_proj`、`k_proj`、`v_proj`、`o_proj`，前馈网络里有 `gate_proj`、`up_proj`、`down_proj`。挂载位置越多，可训练容量越大，但参数量、激活和通信也会增加。notebook 给出三档起点：只挂查询与值投影（参数预算紧）、挂全部注意力投影（注意力适配容量更大）、再挂 MLP（前馈容量更大）。这不是"越多越好"的结论，而是后续 GPU 实验的自变量。

**Step 2：由维度推出形状与参数量。** 设输入 $x\in\mathbb{R}^{d_{in}}$，输出 $h\in\mathbb{R}^{d_{out}}$，冻结权重 $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$。LoRA 旁路先把输入压到秩 $r$，再升回输出维：

$$\Delta W=sBA,\qquad h=W_0x+\Delta Wx=W_0x+\frac{\alpha}{r}BAx,\qquad s=\frac{\alpha}{r}$$

其中 $A\in\mathbb{R}^{r\times d_{in}}$、$B\in\mathbb{R}^{d_{out}\times r}$，$\alpha$ 是控制增量幅度的超参数。可训练参数量为：

$$N_{\mathrm{LoRA}}=r d_{in}+d_{out}r=r(d_{in}+d_{out})$$

以测试用的 `in_dim=128`、`out_dim=256`、`r=8` 为例：$8\times(128+256)=3072$。冻结基座是 $128\times256=32768$ 个参数，所以 LoRA 只训练约 8.6% 的参数。注意 `bias=False`：基座没有偏置，旁路也不引入偏置，合并时只需处理权重矩阵。

**Step 3：TODO 1——冻结基座，建可训练旁路。** 参考答案：

```python
self.linear = nn.Linear(in_features, out_features, bias=False)
self.linear.weight.requires_grad = False          # 基座冻结
self.lora_A = nn.Parameter(torch.empty(r, in_features))
self.lora_B = nn.Parameter(torch.empty(out_features, r))
```

`reset_parameters` 由骨架提供：`kaiming_uniform_` 初始化基座和 `A`，`zeros_` 初始化 `B`。为什么 `B=0`？因为 $\Delta W=BA=0$，训练开始时模块输出严格等于冻结基座，不会一上来就把预训练能力扰动掉。容易写错的是把 `requires_grad=False` 写到 `bias` 上，或者忘了把 `A`/`B` 包成 `nn.Parameter`——后者会导致它们根本不在 `module.parameters()` 里，`count_trainable_parameters` 直接少算。

**Step 4：TODO 2——前向叠加旁路。** 参考答案：

```python
adapter_input = self.lora_dropout(x)
adapter_delta = (adapter_input @ self.lora_A.T) @ self.lora_B.T * self.scaling
return result + adapter_delta
```

`x` 形状是 `[..., d_in]`，`x @ A.T` 得到 `[..., r]`，再 `@ B.T` 得到 `[..., d_out]`，最后乘 `scaling`。dropout **只作用在旁路输入**，主分支不能被随机屏蔽；`merged=True` 时函数提前返回 `self.linear(x)`，旁路和 dropout 都不再执行。测试用 `manual = layer.linear(x) + (x @ A.T) @ B.T * scaling` 逐元素对照，容差 `atol=1e-6`。

这里有个反直觉但重要的梯度现象。零初始化时：
- `B` 的梯度来自 $\partial L/\partial B$，它与 `A` 的输入有关，所以第一步非零；
- `A` 的梯度含因子 `B`，`B=0` 时整条梯度为零；
- 基座权重 `requires_grad=False`，`linear.weight.grad is None`。

这不是链路断开。`_assert_gradient_route` 先检查 `B.grad` 非零且 `A.grad` 全零，然后手动把 `B` 填成 `normal_(0, 0.02)`，再反传一次，`A` 就获得非零梯度。实际训练中优化器会在第一步更新 `B`，下一步 `A` 自然开始学习。

**Step 5：TODO 3——合并回基座。** 参考答案：

```python
delta_weight = (self.lora_B @ self.lora_A) * self.scaling
with torch.no_grad():
    self.linear.weight.add_(delta_weight)
self.merged = True
```

`B @ A` 的形状是 `[d_out, r] @ [r, d_in] = [d_out, d_in]`，与基座一致。合并后 $W_{\mathrm{merged}}=W_0+(\alpha/r)BA$，所以普通线性层就能复现"基座 + 旁路"。两个细节：写回必须在 `torch.no_grad()` 里，否则会往计算图里塞一个叶子张量操作；函数开头 `if self.merged: return` 保证同一增量只写一次，测试会保存合并后的权重、再调用一次 `merge_weights()`，断言权重没变。`merged` 只是普通 Python 属性，不随 `state_dict` 保存——所以 `test_lora_reload_paths` 分两条路验证：未合并时按相同秩和缩放重载 LoRA 参数；合并后保存 `linear.state_dict()`，加载到普通 `nn.Linear` 里输出一致。

**Step 6：可选框架核对与 GPU 实验。** PEFT 路径用 `LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none")` 包一个只有 `proj` 的 `_TinyProjectionModel`，再 `merge_and_unload()`。`check_lora_framework_conformance` 的 8 项检查覆盖：基座冻结、A/B 形状一致、`scaling_match`、可训练参数数量、初始输出、受控 A/B 下的旁路输出、合并输出、合并权重。注意它先把同一组 `controlled_A` / `controlled_B` 写进教学实现和 PEFT，避免随机初始化掩盖公式差异。

GPU 实验固定 `Qwen/Qwen2.5-0.5B`、`MAX_LENGTH=128`、`LORA_RANK=8`、`LORA_ALPHA=16`、`UPDATE_STEPS=3`、`LEARNING_RATE=1e-3`，只改 `TARGET_MODULE_SETS` 三档。判定分成两层：训练机制看 `matched_module_count > 0`、`base_frozen`、`initial_max_abs_diff <= 1e-3`、`trainable_params > 0`、loss 和梯度范数有限；合并质量看 `cosine_similarity >= MIN_MERGE_COSINE=0.999`、`top1_agreement >= MIN_MERGE_TOP1_AGREEMENT=0.98`、`merge_loss_abs_diff <= MAX_MERGE_LOSS_DIFF=0.02`。已保存的 `rtx5070ti_lora_targets_v3` 结果：三档命中 48 / 96 / 168 层，可训练参数 540672（`attention_qv`），训练峰值约 1.67 / 1.70 / 1.82 GiB，合并峰值约 1.97 / 1.98 / 2.02 GiB，中位更新时间 35.2 / 39.2 / 58.7 ms；`decision=accept_mechanism`、`merge_decision=tune`。`attention_qv` 的合并 loss 差为 0.0228，超过 0.02，所以保留未合并 adapter，而不是放宽门槛。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| PEFT | 参数高效微调：只训练少量参数来适配大模型 |
| LoRA | 冻结基座，在旁边加一条低秩可训练分支 |
| 秩 $r$ | 低秩空间的宽度，决定旁路容量和参数量 |
| `lora_A` / `lora_B` | 降维矩阵 `[r, d_in]` / 升维矩阵 `[d_out, r]` |
| `scaling = α/r` | 控制旁路增量相对基座的幅度 |
| 零初始化 | `B=0`，训练起点输出等于冻结基座 |
| 合并（merge） | 把 `(α/r)BA` 写回基座，部署时只跑普通线性层 |
| 幂等 | 重复合并不会重复叠加增量 |
| dropout | 只作用于旁路输入，训练时随机屏蔽，评估时关闭 |
| 目标模块（target modules） | 哪些线性层挂 LoRA，如 `q_proj`、`v_proj` |
| 可训练比例 | 可训练参数 ÷ 总参数，衡量适配的"便宜程度" |

## 小白常见疑问

**Q1：`B=0` 时 `A` 的梯度是零，这不是白训练了吗？**
不是。第一步 `B` 有非零梯度并更新，`B` 离开零后 `A` 的梯度就恢复。`_assert_gradient_route` 明确断言"零初始化时 B 应先获得非零梯度""B 为零时 A 的首步梯度应为零"，然后再把 `B` 填充为 `normal_(0, 0.02)` 验证 `A` 变为非零。这个顺序是零初始化的预期行为。

**Q2：为什么参数量是 `r(d_in + d_out)`，不是 `r × d_in × d_out`？**
因为 LoRA 不学完整矩阵，而是学两个细长矩阵。`A` 有 `r × d_in` 个元素，`B` 有 `d_out × r` 个。完整增量的秩最多为 `r`，表达能力受限，参数量也从平方级降到线性级。

**Q3：`α/r` 到底乘在前向还是合并里？两边都要乘吗？**
两边都必须用同一个 `self.scaling`。前向是 `(x @ A.T) @ B.T * scaling`，合并是 `(B @ A) * scaling`，所以合并前后数学上完全等价。测试 `_assert_forward_and_merge` 先用 `atol=1e-6` 对照前向公式，合并后用 `atol=1e-5` 对照输出，并检查第二次 merge 不改变权重。

**Q4：合并后为什么不能继续训练这个 LoRA？**
`merge_weights` 把增量写进了冻结基座，并把 `merged` 设为 `True`。此时前向直接返回 `self.linear(x)`，旁路不再参与；如果再合并一次也不会重复加。要恢复训练，应重新加载未合并的 LoRA 参数，而不是在已合并权重上继续叠加。

**Q5：GPU 结果里 `decision=accept_mechanism` 但 `merge_decision=tune`，到底算成功吗？**
算"训练机制被接受，合并精度仍需调整"。两层决策是分开的：三档训练路径都通过了命中、冻结、初始等价和有限性检查；但 `attention_qv` 的合并 loss 差 0.0228 超过 0.02，所以合并层给 `tune`。notebook 明确说不放宽门槛，保留未合并 adapter，交付前再复测。

## 动手验证：怎么知道自己做对了
CPU 测试直接调用 `test_lora_reload_paths()` 和 `test_lora()`。`test_lora()` 组合六个断言函数，全部通过打印 `PASSED: LoRA参数、梯度路径和合并不变量核对通过。`。
- `_assert_parameter_contract`：`layer.linear.weight.requires_grad` 为假；`count_trainable_parameters(layer) == 8 * (128 + 256) = 3072`；`lora_A.shape == (8, 128)`、`lora_B.shape == (256, 8)`。
- `_assert_gradient_route`：首步 `B.grad` 非零、`A.grad` 全零、`linear.weight.grad is None`；把 `B` 设为 `normal_(0, 0.02)` 后 `A.grad` 非零。
- `_assert_forward_and_merge`：初始 `layer(x)` 与 `layer.linear(x)` 全等；有随机 `B` 时与手写公式 `atol=1e-6` 一致；`layer.eval(); layer.merge_weights()` 后与未合并输出 `atol=1e-5` 一致；再次 merge 权重不变。
- `_assert_lora_boundaries`：`r=0`、`r=1.5`、`r=True`、`lora_alpha=0`、`inf`、`nan`、`lora_dropout=1.1` 都必须抛 `ValueError`；dropout 层在 `eval()` 下两次输出相同，在 `train()` 下两次不同。
- `_assert_frozen_base_input_gradient`：输入 `x` 仍能收到梯度，期望值是 `layer.linear.weight.detach().sum(dim=0)`，同时基座权重没有梯度。
- `test_lora_reload_paths`：`state_dict` 保存/重载后输出一致，且 `restored.merged` 为假；合并后把 `linear.state_dict()` 载入普通 `nn.Linear`，输出仍一致。

notebook 保存了 GPU 结果：`attention_qv` 的 `trainable_ratio` 约 0.001093，`initial_max_abs_diff` 为 0.0，`merge_agreement` 的 `cosine_similarity` 约 0.999748、`top1_agreement` 0.988、`max_abs_diff` 0.84375。这些数字只对应 `rtx5070ti_lora_targets_v3` 这一次运行，换模型、rank 或模块集合后要重新测，不能直接复用。
