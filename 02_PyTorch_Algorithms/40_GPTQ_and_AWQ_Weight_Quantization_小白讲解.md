# 40. GPTQ 与 AWQ 权重量化 — 小白讲解

> 对应 notebook：[`40_GPTQ_and_AWQ_Weight_Quantization.ipynb`](./40_GPTQ_and_AWQ_Weight_Quantization.ipynb)

## 一句话总结
这一节用 `WeightQuantizerSim` 在 CPU 上搭一个"教学代理"：先从校准激活算出逐通道 RMS 和 Hessian 对角近似（TODO 1），再为 AWQ 标出组内显著通道（TODO 2），最后为每个分组算出安全的对称缩放因子（TODO 3）。它刻意不做完整的 GPTQ 误差补偿和 AWQ 缩放搜索，而是把"哪些权重误差更疼""保护谁""怎么定缩放"拆开给你看；之后可选地用 LLM Compressor 生成真实 W4A16 产物，并用独立 NLL 比较 RTN、GPTQ、AWQ。

## 为什么需要它：痛点是什么
刚学量化时最容易把"4-bit 就是每个数少存几位"当成全部。真到 W4A16 才发现：同一个分组里，把所有权重误差都压到一样小，并不等于层输出误差一样小——输入经常激活的方向上，同样的权重误差会被放大；而少数幅度特别大的输入通道，其对应权重一旦量化粗了，输出就偏得厉害。

生活类比：一个班级算总分，语文和数学都扣 5 分，对排名的影响并不一样。普通 MSE 只看"总共扣了几分"，GPTQ 想按"这门课有多重要"加权，AWQ 则先把"主科"（显著通道）挑出来单独保护。notebook 里的 CPU 版本只模拟"怎么挑主科、怎么打分"，真实的"补课方案"（GPTQ 的逐步补偿、AWQ 的缩放搜索）交给成熟框架。

## 核心问题
- 一个 4-bit 权重量化产物，除了 `qweight` 还必须记录哪些东西才能被恢复？
- 为什么权重 MSE 相同的两个候选，层输出误差可能不同？`H = XᵀX/N` 在这里扮演什么角色？
- "Hessian 对角代理"具体怎么算？为什么它等于 RMS 的平方？
- AWQ 用什么统计量找显著通道？保护掩码为什么必须限制在组内？
- 分组缩放因子为什么要除以 `qmax`，全零分组和尾组怎么保证不出 `NaN`/越界？
- GPU 实验里 RTN / GPTQ / AWQ 的可比条件是什么，NLL 比值和校准时间该怎么读？

## 解决思路：一步步看懂

**第 1 步：量化产物是一份"可恢复的契约"。** notebook 的 Step 1 把流程拆成校准、分组量化、层输出复核、产物生成四段；产物必须固定源模型版本、校准样本、`bits`、`group_size`、缩放因子、对称约定、打包布局和目标模块，否则加载器无法把整数码还原成浮点。`artifact_manifest()` 是这份契约的教学版：`schema_version` 为 `weight-quant-artifact/v1`，`packing_layout` 为 `teaching_int8_container`，`packed_low_bit_weights` 为 `False`——提醒你这里并没有真的把两个 4-bit 码打包进一个字节。

**第 2 步：为什么权重的普通 MSE 不够。** 设校准输入 $X\in\mathbb R^{N\times d}$、浮点权重 $W$、量化扰动 $\Delta W=\widehat W-W$，线性层输出扰动是 $\Delta Y=X\Delta W^{\mathsf T}$。定义输入方向的二阶代理 $H=X^{\mathsf T}X/N$，则

$$
\frac1N\lVert\Delta Y\rVert_F^2=\mathrm{tr}(\Delta W H\Delta W^{\mathsf T}).
$$

$N$ 是校准样本数，$d$ 是输入通道数，$\mathrm{tr}$ 是迹（对角线元素之和）。直白说：误差 $\Delta W$ 会被 $H$ 按输入方向"称重"再求和。两个候选的 $\lVert\Delta W\rVert^2$ 一样，只要误差落在高激活方向上，$\mathrm{tr}(\Delta W H\Delta W^{\mathsf T})$ 就更大。notebook 用对角代理 $H_{jj}=\frac1N\sum_n X_{nj}^2$ 近似完整 $H$，忽略通道之间的相关性，所以只是"敏感度打分"，不是完整 GPTQ。

**第 3 步：AWQ 为什么盯住激活 RMS。** 令 $a_j=\sqrt{\frac1N\sum_n X_{nj}^2}$ 为第 $j$ 个输入通道的激活 RMS。$a_j$ 大，说明这个通道经常"使劲"，对应权重列的误差更容易传到输出。notebook 的 Step 3 还给出一个教学显著性 $I_j=a_j\lVert W_{:j}\rVert_2$，但**代码里的 `_awq_protection_mask` 实际只接收组内的 `activation_rms` 切片**，没有乘权重列范数——这是代理与真实 AWQ 的差别之一。真实 AWQ 会做缩放搜索；CPU 代理为了让你看见保护效果，直接把被保护位置的浮点权重存进 `protected_weight`，这会让存储变大，不能当压缩产物。

**第 4 步：TODO 1 —— 一次算出两个统计量。** 骨架先把激活 reshape 成 `(N, in_features)` 并校验最后一维等于权重输入维，参考答案只有两行：

```python
hessian_diag = flat.square().mean(dim=0)   # 每个输入通道的 mean(X²)
activation_rms = hessian_diag.sqrt()       # RMS = sqrt(mean(X²))
return activation_rms, hessian_diag
```

`dim=0` 是在样本维上求平均，结果形状 `[in_features]`；`sqrt` 与 `square` 互为逆运算，所以测试才敢断言 `rms.shape == hessian.shape == (8,)` 且 `torch.allclose(rms.square(), hessian, atol=1e-6)`。容易写错的是在 `dim=1` 上求平均（那就变成逐样本统计），或者先开方再平方导致形状对但语义错。`flat` 用 `reshape(-1, last_dim)` 而不是 `mean(dim=(0,1))`，是为了兼容 `(3, 5, 8)` 这种带中间维的校准激活。

**第 5 步：TODO 2 —— 只在组内挑 top-k。** 保护掩码先全 `False`；当 `method != "awq"`、组为空或 `protect_ratio == 0` 时直接返回空掩码。真正的答案是一行：

```python
mask[topk] = True   # topk 是组内局部下标，不是全局通道号
```

`count = min(numel, max(1, int(round(numel * protect_ratio))))` 保证至少保护一个，又不超过组宽。因为 `importance` 传进来的是 `self.activation_rms[start:end]`，`topk` 的下标天然是 0 到组宽减一，写回 `protected_mask[row, start:end]` 时不会串到别的组。`test_awq_protection_contract` 用 8 个通道、`group_size=4`、`protect_ratio=0.25`，每组保护 1 个：第 3 列激活被设成 20，落在第一组，于是 `awq.protected_mask[:, 3].all()` 为真；而 `gptq` 分支根本不创建保护位，断言 `not gptq.protected_mask.any()`。

**第 6 步：TODO 3 —— 分组缩放因子。** 骨架已经算好 `quantized_source`（未被保护的那部分；全被保护时退回整个 chunk），参考答案：

```python
scale = (quantized_source.abs().max() / self.qmax).clamp_min(self.eps)
```

`qmax = 2 ** (bits - 1) - 1`，4-bit 时是 7。用绝对最大值除以 7，是为了让最大的那个数刚好落在对称整数范围 $[-7, 7]$ 内；`clamp_min(eps)` 处理全零分组——`0 / 7 = 0` 会把缩放因子变成 0，后面除零就出 `Inf`，所以兜到 `1e-8`。量化时 `torch.clamp(torch.round(chunk / scale), -qmax, qmax)` 再转 `int8`，保证不会出现 8。`dequantize()` 按 `qweight * scale` 逐组还原，再用 `protected_mask` 把浮点旁路值填回去。测试 `test_group_zero_range_and_input_guards` 用全零权重检查 `scales > 0`、`count_nonzero(qweight) == 0`、`qweight.abs().max() <= 7`；`test_group_partition_and_tail_contract` 用 `(4,10)` 权重、`group_size=4` 得到 `scales.shape == (4,3)`，第三组只有 2 列也必须存在。

**第 7 步：误差指标与"代理边界"。** `mse()` 是普通均方误差；`weighted_reconstruction_error()` 把每个输入通道的平方误差乘上 `hessian_diag` 再平均，正好对应第 2 步公式的对角版本。`test_proxy_scope_and_weighted_error` 是最能说明"CPU 版不是 GPTQ"的测试：同一权重分别用 `torch.eye(4)` 和 `2 * torch.eye(4)` 校准，`qweight` 和 `scales` 完全相等（因为量化只看向量本身），但加权误差变成 4 倍（因为 `H` 变成 4 倍）；同时清单里的 `implementation_scope` 是 `rtn_with_diagonal_error_proxy`，`packed_low_bit_weights` 是 `False`。也就是说，校准统计只改变"评分"，没有改变码值，不能宣称算法改善了量化。

**第 8 步：可选框架探测。** `check_gptq_framework_conformance()` 用微型 `LlamaForCausalLM` 的 `q_proj` 挂 forward pre-hook 抓真实层输入，一边跑本地代理，一边用 LLM Compressor 的 `GPTQModifier`（`scheme="W4A16"`、`block_size=64`、忽略 `lm_head`）走 `oneshot`。7 项检查只看接口：本地统计通道数、两侧缩放有限、框架状态含 `quantization_scheme`/`weight_scale`/`weight_zero_point`、前向有限且发生变化等；返回 `"mechanism_equivalence_claimed": False`、`"check_kind": "interface_probe"`，不证明补偿过程一致。

**第 9 步：可选 GPU 实验怎么读。** 三个候选共享 `Qwen/Qwen2.5-0.5B-Instruct`、W4A16、WikiText-2 的 train 校准 64 条、validation 评测 64 条、最大长度 256、`SEED=42`，只改变校准算法：RTN 用 `QuantizationModifier`，GPTQ 用 `GPTQModifier`，AWQ 先 `AWQModifier` 再 `QuantizationModifier`。因变量是 Token 加权 NLL（`total_nll / total_tokens`）、`perplexity=exp(min(20,nll))`、`calibration_seconds`、`peak_incremental_bytes` 和 `nll_ratio_vs_bf16`。门槛 `MAX_NLL_RATIO_VS_BF16=1.10`，合格候选里再选 NLL 最小者。notebook 5.4 记录的 `default11`（RTX 5070 Ti Laptop GPU）为：BF16 NLL 3.393920；GPTQ/AWQ/RTN 比值 1.039188/1.058645/1.094336；校准时间约 9.21/22.12/2.67 秒。三者都在 1.10 内，GPTQ 质量最好，因此进入加载器与后端检查；RTN 最快但最差，AWQ 这次比 GPTQ 慢也略差。`memory_comparison_valid=false`，因为三个候选同进程先后跑，显存峰值不能排名。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 校准激活 | 用代表性输入喂给目标层时记录下来的输入值 |
| Hessian 对角代理 | 每个输入通道的 `mean(X²)`，近似"这个方向上的误差有多重要" |
| 激活 RMS | 通道激活的均方根，等于 Hessian 对角代理的平方根 |
| 显著通道 | 激活幅度大、量化误差更容易伤到输出的输入通道 |
| `group_size` | 多少个输入通道共享一个缩放因子 |
| 对称量化 | 码值围绕 0 对称，范围 $[-qmax, qmax]$，不另存零点 |
| `qmax` | `2 ** (bits - 1) - 1`，4-bit 时为 7 |
| 尾组 | 输入维不能整除 `group_size` 时剩下的那一组 |
| 浮点旁路 | AWQ 教学代理把被保护权重按浮点另存一份 |
| NLL 比值 | 候选的 Token 加权 NLL 除以 BF16 基线，越接近 1 越好 |
| 量化产物清单 | 记录位宽、分组、缩放、打包布局和实现范围的元数据 |

## 小白常见疑问
**Q1：既然叫 GPTQ，为什么说它没有执行 GPTQ？**
因为 CPU 路径只做"逐组直接舍入（RTN）＋用对角 Hessian 给误差打分"。真正的 GPTQ 会按列逐步量化并用 $H^{-1}$ 补偿剩余列的误差；这里 `qweight` 在两种校准下都完全一样（`test_proxy_scope_and_weighted_error` 直接断言 `torch.equal`）。清单里的 `implementation_scope` 写成 `rtn_with_diagonal_error_proxy` 就是在提醒这一点。

**Q2：`hessian_diag` 为什么就是 `mean(X²)`，和 $X^{\mathsf T}X/N$ 什么关系？**
完整矩阵 $H=X^{\mathsf T}X/N$ 的第 $j$ 个对角元素是 $\frac1N\sum_n X_{nj}^2$，正是第 $j$ 列的平方均值。notebook 只保留对角线，丢掉通道间协方差，所以叫"对角代理"。`test_calibration_statistics_contract` 用 `rms.square() == hessian` 检查两者关系。

**Q3：AWQ 的保护掩码会不会跨组？**
不会。`_awq_protection_mask` 每次只收到 `activation_rms[start:end]`，`topk` 是局部下标，写回时也落回 `protected_mask[row, start:end]`。所以"每组保护几个"是按组独立计算的，`protect_ratio=0.25`、组宽 4 时每组 1 个，而不是全层 25%。

**Q4：全零分组为什么还需要正的 `scale`？**
因为 `dequantize()` 会用 `qweight * scale` 还原。如果 `scale=0`，后续任何除法/比对都可能出问题，而且清单里"缩放因子必须有限正数"是契约。`clamp_min(self.eps)` 让全零组也保留一个正的占位值，反正码值全是 0，乘出来还是 0。

**Q5：GPU 实验里 RTN 的 NLL 比值 1.094336 也低于 1.10，为什么没选它？**
门槛只决定"有没有资格"，选择还要在合格候选里取 NLL 最小。`selected_name = min(quality_candidates, key=... token_weighted_nll)`，GPTQ 的比值 1.039188 最小，所以被选中。RTN 仍是有效候选，只是质量排序靠后。

## 动手验证：怎么知道自己做对了
`test_calibration_statistics_contract` 断言 `rms.shape == hessian.shape == (8,)` 且 `rms.square()` 与 `hessian` 在 `atol=1e-6` 内相等；`test_group_partition_and_tail_contract` 断言 `(4,10)` 权重、`group_size=4` 时 `scales.shape == (4,3)`；`test_awq_protection_contract` 断言 GPTQ 分支掩码全假、AWQ 第 3 列全真；`test_weighted_error_contract` 只要求两个误差有限；`test_artifact_manifest_contract` 要求清单含 8 个字段且 AWQ `scale_shape == [4,2]`；`test_proxy_scope_and_weighted_error` 检查两种校准码值相同、加权误差 4 倍、`packed_low_bit_weights is False`、`protect_ratio=1` 时恢复等于原权重；`test_group_zero_range_and_input_guards` 检查全零组正缩放、码值不超 7，并确认 `bits=9`、`group_size=1.5`、`eps=0`、空校准、含 `NaN` 校准都抛 `ValueError`。全部通过后打印 `✅ GPTQ/AWQ 机制测试通过：统计、分组、显著通道、误差代理与 artifact 均已验证。` notebook 没有保存执行输出；框架探测和 GPU 实验以你自己运行得到的字典/JSON 为准，5.4 的数字是 `default11` 参考，不是本地复现保证。
