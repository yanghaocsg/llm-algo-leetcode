# 26. QLoRA 与 4-bit 量化 — 小白讲解

> 对应 notebook：[`26_QLoRA_and_4bit_Quantization.ipynb`](./26_QLoRA_and_4bit_Quantization.ipynb)

## 一句话总结
QLoRA 把冻结的底座权重存成 4-bit 的 NF4 索引，只训练浮点的 LoRA 旁路。本节用三个 TODO 分别完成"查表还原底座权重""合并底座与 LoRA 两条前向分支""只统计可训练适配器的状态字节"，再用可选的 BF16 LoRA / NF4 QLoRA 配对矩阵比较显存、步时和测试损失。归档的 `default2` 实测里，三档序列长度的峰值显存下降 14.8%–26.7%，测试损失增加约 3.8%–4.2%，单步训练时间增加 18.1%–41.4%，所以结论是"容量受限时可用"，不是"QLoRA 全面更快"。

## 为什么需要它：痛点是什么
初学者对 QLoRA 最常见的误解是把它当成一句乘法口诀："4-bit 就是每个参数 0.5 字节，所以 7B 模型只要 3.5 GB。"这句话只描述了权重索引的理想存储，漏掉了缩放因子、码点表、计算时反量化出来的浮点权重、激活、LoRA 梯度和优化器状态，也解释不了"为什么 4-bit 底座还能正常训练"。

第二个困惑是"底座都冻结了，为什么还要给它传梯度"。很多人以为 `requires_grad=False` 等于"这条路径对反传完全透明"，于是把整个底座分支放进 `torch.no_grad()`，结果上游 LoRA 层拿不到输入梯度，训练信号被悄悄切断。

生活类比：把底座想成一箱压缩好的换季衣服，NF4 是真空压缩袋，袋子上贴着 0–15 的编号；穿衣服前必须按编号查表、乘上该袋的缩放系数，把衣服"还原"出来。压缩袋让衣柜更省空间，但你每次穿都要先拆袋，所以"省空间"和"更省时间"并不是同一件事。QLoRA 的训练显存账本也一样：省下的是底座驻留，代价是反量化与额外的训练工作区。

## 核心问题
- NF4 的 16 个码点是怎么来的？为什么它比 INT4 更贴合权重分布？
- "4-bit"描述的是存储还是计算？训练时到底哪些对象还是浮点？
- 底座冻结后，为什么输入 `x` 仍然需要梯度？切断底座分支会怎样？
- LoRA 旁路的 `A`、`B` 和缩放系数 `alpha / r` 怎样与底座输出相加？
- 统计可训练状态时为什么只累加 `requires_grad=True` 的参数？`packed_nf4_bytes` 的向上取整怎么算？
- GPU 配对实验的自变量、控制变量和五个指标分别是什么？为什么显存节省比例会随序列长度变小？

## 解决思路：一步步看懂

**第 1 步：先把"参与训练的对象"列清楚。** notebook 的 Step 1 用一张表把 QLoRA 拆成五类对象：基座权重（NF4 索引 + 缩放元数据，冻结）、码点与缩放因子（查表信息，不更新）、LoRA 旁路（浮点低秩矩阵，更新）、适配器优化器状态（随优化器保存）、输入激活（参与两条分支，需要时接收梯度）。这张表是后面所有账本的基础：说"QLoRA 省显存"时，省的是第一、二项的驻留；说"训练还有开销"时，开销来自第四、五项以及反量化出来的临时浮点权重。4-bit 只描述权重怎么存，不表示矩阵乘法、激活、适配器梯度或优化器状态也用 4-bit。

**第 2 步：NF4 是"按概率密度铺点"，不是等间隔。** 标准正态分布的累积分布函数记作 $\Phi$，它的反函数 $\Phi^{-1}$ 能把一个概率 $p$ 映射回对应的分位点。notebook 给出的教学近似是

$$
q_i=\Phi^{-1}(p_i),\qquad p_i=\frac{i-0.5}{16},\qquad i=1,2,\dots,16.
$$

其中 $i$ 是码点编号，$p_i$ 是第 $i$ 个代表概率，$q_i$ 是对应的标准正态分位点。取 $i=1$ 时 $p_1=0.03125$，$q_1\approx-1.86$；取 $i=8$ 时 $p_8=0.46875$，$q_8\approx-0.078$；取 $i=9$ 时 $p_9=0.53125$，$q_9\approx0.078$。可以看到 0 附近两个码点只差约 0.16，而两端拉得很开——这正是"0 附近更密、尾部更疏"的来源。教学意义上的 INT4 则是把 $[-1,1]$ 等分成 16 格，每个格子一样宽，不会专门照顾权重集中的区域。

notebook 里的 `create_nf4_lookup_table()` 返回的是一张教学表：

```python
nf4_values = [
    -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
    0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0,
]
return torch.tensor(nf4_values)
```

它把分位点思路截断、归一化到 $[-1,1]$，并显式放了 `0.0` 和两个端点。所以**不要拿上面的公式逐项去对这张表**：notebook 自己说明真实 bitsandbytes NF4 codebook 还包含端点、归一化和零值处理。理解重点是"索引只有 0–15，真正权重 = 码点 × 该组缩放因子"。例如一个权重先被归一化成 `0.20`，最近码点是 `0.161`，再乘分组缩放因子就得到近似权重；NF4 保存的是索引，不是这个浮点结果。

双重量化（Double Quantization）继续压缩"缩放因子"这份元数据：第一层把权重压成 4-bit，每组还要存一个缩放常数；第二层把这些常数再量化一次。它降低的是底座驻留，不会自动降低激活、LoRA 梯度或优化器状态的峰值。

**第 3 步：冻结底座，但不切断输入梯度。** 记输入 $x\in\mathbb R^{b\times d_{\mathrm{in}}}$，反量化后的底座近似权重 $\hat W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$，LoRA 的 $A\in\mathbb R^{r\times d_{\mathrm{in}}}$、$B\in\mathbb R^{d_{\mathrm{out}}\times r}$，则

$$
y=x\hat W^{\top}+(xA^{\top})B^{\top}\frac{\alpha}{r}.
$$

其中 $b$ 是批量大小，$d_{\mathrm{in}}$、$d_{\mathrm{out}}$ 是输入输出维度，$r$ 是秩，$\alpha$ 是缩放超参数。第一项是冻结底座变换，第二项是 LoRA 增量。冻结 $\hat W$ 省掉的是"对 $\hat W$ 的梯度"和"$\hat W$ 的优化器状态"；但 $y$ 仍然通过 $\hat W$ 依赖 $x$，所以对 $x$ 的偏导仍然是 $\hat W$ 的函数。若把整个底座分支包进 `torch.no_grad()`，上游适配器就收不到这部分梯度，notebook 的梯度测试会直接抓出这个错误。

**第 4 步：TODO 1——把 0–15 的索引还原成浮点权重。**

```python
indices = self.weight_nf4_indices.long()          # 索引必须转成 int64
dequantized_base_weight = self.nf4_table[indices] * self.weight_scale
```

两个细节都不能省。第一，PyTorch 的高级索引要求索引张量为 `torch.long`；`weight_nf4_indices` 为了模拟 4-bit 存储被注册成 `int8` 缓冲区，直接拿它索引会报错，所以参考答案先 `.long()`。第二，查表只得到 $[-1,1]$ 附近的码点，必须乘 `weight_scale` 才恢复这一组的数值范围，结果形状与 `weight_nf4_indices` 相同，都是 `[out_features, in_features]`。忘了乘缩放，权重幅度会整体偏小；忘了 `.long()`，代码根本跑不起来。

**第 5 步：TODO 2——两条分支分别算，再相加。**

```python
base_out = F.linear(x, dequantized_base_weight)          # x @ W_hat.T
lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
return base_out + lora_out
```

`F.linear(x, W)` 内部做的正是 $x\hat W^{\top}$，输出最后一维是 `out_features`。LoRA 的 `lora_A` 形状是 `[r, in_features]`，所以 `x @ lora_A.T` 得到 `[..., r]`；再乘 `lora_B.T`（`[r, out_features]`）得到 `[..., out_features]`，最后乘 `scaling`（构造时注册为 `alpha / r`）。测试 fixture 用 `alpha=8.0, r=2`，所以 `scaling=4.0`。为什么先 `A` 后 `B` 且都转置？因为 `nn.Linear` 的权重约定是 `[out, in]`，而 LoRA 论文常把 `A` 写成降维矩阵 `[r, in]`、`B` 写成升维矩阵 `[out, r]`，这样写与公式一一对应。

`__init__` 里 `lora_B` 初始化为全零，`lora_A` 用 `0.01` 倍的小随机数。这样初始 LoRA 增量为 0，模型从底座原本的行为出发，不会一上来就被随机旁路扰动。`test_lora_branch_contract` 正是利用这一点：先算纯底座输出 `base_out`，再断言 `layer(x)` 与它**不**全等（`atol=1e-6`），证明旁路确实参与了计算，而不是被漏掉。

**第 6 步：TODO 3——训练状态只算可训练参数。**

```python
trainable_parameters = sum(
    parameter.numel() for parameter in layer.parameters() if parameter.requires_grad
)
```

`layer.parameters()` 默认会遍历所有 `nn.Parameter`，但 QLoRA 的索引、缩放、码点表和 `scaling` 都是用 `register_buffer` 注册的缓冲区，本来就不在 `parameters()` 里；真正需要按 `requires_grad` 过滤的是"部分冻结"的场景。notebook 的 `test_partial_freeze_and_odd_storage` 手动把 `lora_A.requires_grad_(False)`，此时账本必须只算 `lora_B` 的 6 个元素，而不是 14 个。

后半段由骨架提供，读一遍能看清两个口径：

```python
index_container_bytes = layer.weight_nf4_indices.numel() * layer.weight_nf4_indices.element_size()
scale_metadata_bytes = layer.weight_scale.numel() * layer.weight_scale.element_size()
packed_nf4_bytes = (layer.weight_nf4_indices.numel() + 1) // 2 + scale_metadata_bytes
```

`index_container_bytes` 按 INT8 容器算，12 个索引就是 12 字节；`packed_nf4_bytes` 才估算"两个 4-bit 索引共用一个字节"的真实打包成本，奇数个索引要向上取整，所以 `(numel + 1) // 2`。fixture 里 `in_features=4, out_features=3`，索引总数 12，缩放因子是 1 个 float32（4 字节）：

- `index_container_bytes = 12`
- `scale_metadata_bytes = 4`
- `base_storage_bytes = 12 + 4 = 16`
- `packed_nf4_bytes = (12 + 1) // 2 + 4 = 6 + 4 = 10`
- 可训练参数 `= lora_A.numel() + lora_B.numel() = 2×4 + 3×2 = 8 + 6 = 14`
- 默认 `optimizer_bytes_per_parameter=8`，优化器状态 `= 14 × 8 = 112` 字节

测试断言 `packed_nf4_bytes < base_storage_bytes`（10 < 16），就是在提醒：**INT8 容器不等于真实 4-bit 驻留**，两个口径必须分开写清楚。账本也没有计共享码点表，真实后端的 4-bit 驻留还要看 packing、padding 和元数据布局。

**第 7 步：可选的 GPU 配对矩阵——只改"底座表示"。** Step 5 用 `RUN_MODE="dry_run"` 做默认保护，确认环境后才改成 `real_gpu`。实验的自变量只有一个：底座是 BF16 还是 NF4 + double quant；控制变量包括同一模型 commit、同一数据划分、同一 LoRA rank（8）、同一目标模块（`q_proj`、`v_proj`）、同一 `alpha=16`、同一学习率 `2e-4`、同一 `MAX_STEPS=20`、同一随机种子 42。模型是 `Qwen/Qwen2.5-1.5B-Instruct`，数据是 WikiText-2 的 train/validation/test 固定快照，序列长度取 `(64, 128, 256)` 三档，每档跑 BF16 LoRA 和 NF4 QLoRA 各一次。

注意 NF4 路径还会调用 `prepare_model_for_kbit_training`，所以这里比较的是两条完整训练路径，差异不只来自一次矩阵乘法，还可能来自量化准备、反量化和训练工作区。每个矩阵点记录：`initial_validation_loss`、`final_validation_loss`、`final_test_loss`、`train_loss`、`training_ms_per_step`、`base_allocated_bytes`、`peak_allocated_bytes`、`incremental_peak_allocated_bytes`、`trainable_parameters`、`total_parameters`、`params4bit_count`、`path_confirmed`。这里的峰值显存来自 `torch.cuda.max_memory_allocated()`，统计的是 PyTorch 已分配显存，不是整卡总显存，也不包含其他进程占用。其中 `path_confirmed` 的定义很关键：

```python
path_confirmed = (
    base_parameters_frozen
    and ((params4bit_count > 0) if quantized else (params4bit_count == 0))
)
```

它同时要求"底座参数被冻结"和"量化路径真的出现了 `Params4bit`、非量化路径真的没有"。只看到 `load_in_4bit=True` 不够，必须证明真实 4-bit 参数状态存在。`trainable_parameter_match` 再要求两条路径的可训练参数量相等，否则 LoRA 配置可能被悄悄改过，显存和损失的差异就不可归因。

**第 8 步：结果怎么读。** 归档的 `default2` 三档比较如下（`peak_memory_reduction_ratio` 是相对 BF16 的峰值显存降幅，`incremental_training_peak_ratio` 是训练增量工作区相对 BF16 的倍数，`step_time_ratio` 是单步耗时倍数，`final_test_loss_ratio` 是测试损失倍数）：

| 序列长度 | 峰值显存降幅 | 省下字节 | 增量工作区倍数 | 步时倍数 | 测试损失倍数 |
|---:|---:|---:|---:|---:|---:|
| 64 | 26.66% | 906,521,600 | 2.92 | 1.352 | 1.0416 |
| 128 | 22.00% | 807,660,544 | 2.17 | 1.414 | 1.0389 |
| 256 | 14.84% | 629,500,928 | 1.74 | 1.181 | 1.0375 |

三个数字放在一起才看得出取舍。第一，**峰值显存降幅随序列变长而缩小**：64 token 时底座驻留占比大，省下 26.7%；256 token 时激活和训练工作区变大，固定的底座节省被摊薄，只剩 14.8%。第二，**增量工作区倍数大于 1**，说明 QLoRA 训练时额外占用的工作区反而是 BF16 的 1.74–2.92 倍；它省的是"常驻底座"，不是"训练全过程都更省"。第三，**步时变慢 18%–41%**，因为每次前向都要反量化 4-bit 权重。第四，测试损失只增加 3.8%–4.2%，低于 `MAX_TEST_LOSS_RATIO = 1.10` 的门槛。

最终决策 `accept_qlora_candidate` 不是只看显存，而是五个条件同时成立：所有矩阵点 `path_confirmed`、所有长度 `trainable_parameter_match`、所有长度显存都有改善、所有长度测试损失倍数不超过 1.10、六个 adapter 的配置和权重文件都完整。缺一项就降为 `tune_qlora_protocol`。notebook 还特别记录了一个反例：`default1` 的 0.5B 模型虽然也确认了真实 4-bit 路径和 adapter 交付，但总峰值、步时和测试损失都没过准入，说明"模型放大后 NF4 的常驻节省才覆盖额外工作区"，不能把小模型结论直接外推。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| NF4 | 4-bit NormalFloat：用 16 个按正态分位点挑出的码点表示权重 |
| 码点（code point） | 查找表里的一个代表值，索引 0–15 对应一个浮点数 |
| 查找表 / codebook | 16 个码点组成的表，查表把索引变成近似权重 |
| 反量化（dequantization） | 索引查表再乘缩放因子，恢复出计算用的浮点近似权重 |
| 分组缩放因子 | 一组权重共享的系数，用来恢复这组的数值范围 |
| 双重量化 | 把缩放因子这份元数据再量化一次，进一步压缩底座 |
| 冻结底座 | `requires_grad=False`，不更新底座，但仍向后传输入梯度 |
| LoRA 的 A / B | 降维矩阵 `[r, in]` 与升维矩阵 `[out, r]`，二者相乘形成低秩增量 |
| `alpha / r` | LoRA 增量的缩放系数，控制旁路强度 |
| 存储精度 vs 计算精度 | 权重怎么存（4-bit）与矩阵乘法/激活用什么算（BF16）是两件事 |
| 驻留显存 vs 峰值显存 | 权重常驻占用 vs 训练过程中同时占用的最大值 |
| `packed_nf4_bytes` | 按两个索引一字节估算的打包成本，奇数向上取整 |

## 小白常见疑问

**Q1：QLoRA 把权重压到 4-bit，为什么训练显存不是直接除以 4？**
因为 4-bit 只压缩底座权重的存储。`qlora_trainable_state_ledger` 明确把底座存储和适配器状态分开：fixture 里底座打包后是 10 字节，可训练参数 14 个，优化器状态 112 字节，后者反而是大头。真实训练还要加激活、反量化出来的临时浮点权重和 CUDA 工作区。GPU 结果也印证了这一点：峰值只降 14.8%–26.7%，不是 75%。

**Q2：底座都冻结了，为什么 `test_frozen_base_gradient_contract` 还要求 `x.grad` 等于完整有效权重的列和？**
因为前向是 $y=x\hat W^{\top}+\dots$，对 $x$ 的偏导包含 $\hat W$。测试先构造 `effective = restored + (B @ A) * scaling`，再算 `effective.detach().sum(dim=0).expand_as(x)` 作为期望梯度。如果把底座分支放进 `no_grad`，`x.grad` 会少掉 `restored` 这一项，测试立刻失败。冻结的是"要不要更新 $\hat W$"，不是"要不要让梯度穿过它"。

**Q3：`indices = self.weight_nf4_indices.long()` 里的 `.long()` 能不能省？**
不能。`weight_nf4_indices` 是 `torch.int8` 缓冲区，PyTorch 的张量索引要求 `int64`，直接 `self.nf4_table[self.weight_nf4_indices]` 会抛类型错误。参考答案显式转 `long`，测试 `test_nf4_lookup_and_dequantization` 也先做同样的转换再比较恢复形状，等于把这个约定写进了契约。

**Q4：`packed_nf4_bytes` 为什么是 `(numel + 1) // 2`，而不是 `numel // 2`？**
两个 4-bit 索引正好塞进一个字节，但索引个数可能是奇数。`(numel + 1) // 2` 是向上取整：12 个索引得 6 字节，9 个索引得 5 字节。`test_partial_freeze_and_odd_storage` 用 3×3 的层（9 个索引）断言 `packed_nf4_bytes == 5 + weight_scale.element_size()`，就是在检查这个奇数边界。写成 `numel // 2` 会把最后一个索引漏掉。

**Q5：GPU 结果里 QLoRA 明明更慢，为什么还 `accept_qlora_candidate`？**
因为准入标准不是"更快"，而是"在质量可接受的前提下降低峰值显存"。`MAX_TEST_LOSS_RATIO = 1.10` 约束质量，`peak_memory_reduction_ratio > 0` 约束显存，`step_time_ratio` 只被记录、不被当成拒绝条件。notebook 的结论也写得很克制：只支持"容量受限的适配训练候选"，长周期收敛、adapter 合并和部署加载都还要另测。读这份结果时不能把"accepted"理解成"任何场景都推荐"。

## 动手验证：怎么知道自己做对了

先跑 `test_qlora_frozen_buffers_reload()`，再跑 `run_qlora_tests()`；后者依次执行七个测试，全部通过才打印 `✅ QLoRA 机制测试通过：NF4、LoRA 旁路、冻结基座与梯度流向均已验证。`。TODO 没填时，`forward` 会先抛 `NotImplementedError('TODO 1：恢复 NF4 底座权重')`，账本函数会抛 `NotImplementedError('TODO 3：统计可训练状态')`。

`test_qlora_frozen_buffers_reload` 先把 fixture 的 `state_dict` 存进 `io.BytesIO`，再加载到一个新的 `QLoRALinearSim(4, 3, r=2, alpha=8.0)` 里，断言恢复后的前向输出与原来一致、`weight_nf4_indices` / `weight_scale` / `nf4_table` / `scaling` 四个缓冲区数值相等且 `requires_grad` 为假，最后确认 `lora_A`、`lora_B` 仍然可训练。这个测试检查的是"冻结状态能被完整保存和恢复"，不是训练效果。

- `test_nf4_lookup_and_dequantization`：`create_nf4_lookup_table()` 形状必须是 `(16,)`，且 `table[0] < 0 < table[-1]`；用 fixture 的索引和缩放恢复后，形状与 `weight_nf4_indices` 相同，且 `torch.isfinite(restored).all()` 为真。
- `test_lora_branch_contract`：输入 `x` 形状 `(1, 2, 4)`，输出必须是 `(1, 2, 3)`；并且 `layer(x)` 与纯底座 `F.linear(x, base_weight)` 在 `atol=1e-6` 下**不**全等，证明 LoRA 旁路真的改变了结果。
- `test_qlora_trainable_state_ledger_contract`：fixture 的可训练参数必须是 `lora_A.numel() + lora_B.numel() = 14`，优化器状态 `14 × 8 = 112`；`base_storage_bytes > 0`，且等于 `index_container_bytes + weight_scale.element_size()`（12 + 4 = 16）；`packed_nf4_bytes < base_storage_bytes`（10 < 16）。
- `test_frozen_base_gradient_contract`：`x` 需要梯度，前向求和反传后，`x.grad` 必须等于 `effective.detach().sum(dim=0).expand_as(x)`；`lora_A.grad`、`lora_B.grad` 非空；`weight_nf4_indices.grad`、`weight_scale.grad` 都是 `None`，且 `weight_nf4_indices.requires_grad` 为假。
- `test_qlora_output_contract`：`layer(x)` 必须与显式参考式 `F.linear(x, reference_weight) + (x @ lora_A.T) @ lora_B.T * scaling` 在 `atol=1e-5` 内一致。
- `test_partial_freeze_and_odd_storage`：`QLoRALinearSim(3, 3, r=2)` 冻结 `lora_A` 后，可训练参数只剩 `lora_B` 的 6 个；`packed_nf4_bytes == 5 + weight_scale.element_size()`；优化器状态 `6 × 8 = 48`。
- `test_invalid_qlora_configuration`：`r=0`、`r=1.5`、`alpha=nan`、`alpha=inf` 构造时必须抛 `ValueError`；`optimizer_bytes_per_parameter` 取 `0`、`True`、`2.5` 时也必须抛 `ValueError`（注意 `True` 虽然是 `int` 的子类，但 `type(True) is not int` 会把它挡下）。

GPU 实验没有 CPU 测试，结果以 5.2 保存的 JSON 为准。归档输出里 `decision` 是 `accept_qlora_candidate`，`evidence_level` 是 `matched_bf16_lora_nf4_qlora_sequence_matrix`，`failure` 为 `None`，六个 adapter 全部有 `adapter_config.json` 和权重文件；`resolved_model_commit` 是 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。如果你在自己的环境复跑，先确认 `RUN_MODE` 已从 `dry_run` 改成 `real_gpu`、`RUN_LABEL` 唯一，并且显存足够容纳 1.5B 模型的 BF16 与 NF4 两条路径；只看单个序列长度不能替代完整矩阵。
