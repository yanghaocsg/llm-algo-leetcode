# 53. 激活量化与 SmoothQuant — 小白讲解

> 对应 notebook：[`53_Activation_Quantization_and_SmoothQuant.ipynb`](./53_Activation_Quantization_and_SmoothQuant.ipynb)

## 一句话总结
W8A8 要同时把权重和激活压到 8 位，但两者的数值分布并不对称：权重通常较平稳，激活却常在少数通道出现远大于其他通道的异常值。逐张量量化只能共用一个缩放因子，异常值会挤占普通通道的整数码点。SmoothQuant 用恒等式 $XW^{\mathsf T}=(XD_s^{-1})(WD_s)^{\mathsf T}$，按 $s_j=a_j^{\alpha}/w_j^{1-\alpha}$ 把量化难度从激活迁移一部分到权重。本节用 `SmoothQuantSimulator` 的 `channel_scale`、`smooth`、`select_alpha` 三个 TODO 复现这条链路，并用 7 个测试验证等价性、异常通道响应和误差选择。

## 为什么需要它：痛点是什么
初学者常听到“权重量化到 INT8 几乎不掉点”，于是以为激活也一样。实际把激活也压到 INT8 后，输出误差可能明显变大。原因不是激活“整体很大”，而是**通道间差异**：某一列绝对值可能是其他列的几十倍，逐张量 scale 必须覆盖最大值，普通通道可用的整数格点就变少了。

生活类比：一把刻度总数固定的尺子。队伍里有一个 1 米高的人，尺子按他定刻度，其他 10 厘米高的小人只能用很少格子，误差自然大。SmoothQuant 把“高度”重新分配：异常通道的一部分数值搬到权重那边，让各列“身高”更接近，两边都更好量化。搬家不改变浮点总账。

## 核心问题
- 为什么权重 INT8 误差小，激活也 INT8 后输出误差却可能变大？异常通道怎么识别？
- 恒等式 $XW^{\mathsf T}=(XD_s^{-1})(WD_s)^{\mathsf T}$ 为什么成立？它保证了什么、又不保证什么？
- $s_j=a_j^{\alpha}/w_j^{1-\alpha}$ 中 $\alpha$ 取 0、0.5、0.9 时，激活和权重的范围各自怎么变？
- 为什么“激活峰值下降”不等于“输出误差下降”？应该看哪个指标？
- 校准阶段选出的 `alpha` 和逐通道因子，为什么必须在独立样本上固定复用？

## 解决思路：一步步看懂

**Step 1：W8A8 与逐张量对称量化。** W8A8 是权重 8 位、激活 8 位。`_quantize_dequantize_per_tensor` 是教学用的对称逐张量量化：

```python
absmax = tensor.detach().float().abs().max().clamp_min(self.eps)
scale = absmax / self.qmax                        # qmax 默认 127
quantized = torch.clamp(torch.round(tensor.float() / scale), -self.qmax, self.qmax).to(torch.int8)
return quantized.float() * scale                  # 反量化
```

`scale` 由**整个张量**的 `absmax` 决定；`qmax=127` 是 INT8 正半轴最大码点，`eps=1e-8` 防止全零除零。`quantized_mse` 衡量量化-反量化结果与浮点参考的差距。这是“先恢复浮点再算矩阵乘”的教学对照，不执行整数内核。

**Step 2：异常通道为什么伤害逐张量量化。** 激活 $X$ 形状 `[tokens, in_features]`，权重 $W$ 形状 `[out_features, in_features]`；$Y=XW^{\mathsf T}$ 里两者最后一维是同一组输入通道。`channel_scale` 沿样本维统计激活的逐通道 `absmax`，沿输出维统计权重的逐通道 `absmax`：

```python
act_max = activations.detach().float().abs().reshape(-1, activations.shape[-1]).amax(dim=0).clamp_min(self.eps)
weight_max = weight.detach().float().abs().amax(dim=0).clamp_min(self.eps)
```

`reshape(-1, last_dim)` 把前置维压平成样本维，再 `amax(dim=0)` 得到每个输入通道的最大值；两者形状都是 `[in_features]`。夹具里激活第 1 列 `[120,-90,80]` 的 `act_max[1]=120`，权重第 1 列 `[0.25,0.5]` 的 `weight_max[1]=0.5`；第 0 列对应值是 2 和 1。第 1 列就是异常通道，范围是第 0 列的 60 倍。

**Step 3：TODO 1——计算平滑因子。** 参考答案：

```python
scale = (act_max.pow(alpha) / weight_max.pow(1.0 - alpha)).clamp_min(self.eps)
```

即 $s_j=a_j^{\alpha}/w_j^{1-\alpha}$，其中 $a_j$ 是激活第 $j$ 列的 `absmax`，$w_j$ 是权重第 $j$ 列的 `absmax`，$0\le\alpha\le1$。`clamp_min(self.eps)` 给零范围通道一个正下限，避免除以 0 或得到 0。测试在夹具、`alpha=0.5` 下断言 `scale.shape == (4,)`、全部有限且为正、`scale[1] > scale[0]`：第 1 列是 $\sqrt{120/0.5}=\sqrt{240}\approx15.49$，第 0 列是 $\sqrt{2/1}\approx1.41$。常见错误是沿错误维度取 `amax`（比如对整张量取一个全局最大值，那就退化成逐张量）、忘记 `clamp_min`（零通道产生 0 或 inf）、或把比值写反成 $w^{1-\alpha}/a^\alpha$。

**Step 4：TODO 2——执行等价迁移。** 参考答案：

```python
smooth_activations = activations.float() / scale          # X / s
smooth_weight = weight.float() * scale.unsqueeze(0)       # W * s
return smooth_activations, smooth_weight, scale
```

把 $D_s=\mathrm{diag}(s)$ 代入：$XW^{\mathsf T}=(XD_s^{-1})(WD_s)^{\mathsf T}$，右边展开为 $XD_s^{-1}D_sW^{\mathsf T}=XW^{\mathsf T}$。代码里 `activations / scale` 是 $XD_s^{-1}$，`weight * scale.unsqueeze(0)` 是 $WD_s$；`unsqueeze(0)` 把 `[in_features]` 变成 `[1,in_features]`，广播到权重输入维。`test_smoothing_preserves_float_product` 用随机 `(2,3,6)`/`(4,6)`、`alpha=0.6` 断言 `equivalence_mse < 1e-10`。常见错误是激活乘、权重除（输出变成 $XD_s^2W^{\mathsf T}$），或广播到输出维。

**Step 5：$\alpha$ 到底在分配什么。** 变换后激活第 $j$ 列范围变成 $a_j/s_j=(a_jw_j)^{1-\alpha}$，权重第 $j$ 列变成 $w_js_j=(a_jw_j)^{\alpha}$。$\alpha$ 接近 0 时激活保留大部分乘积、权重范围趋近 1；接近 1 时激活趋近 1、权重承接大部分乘积；中间值让两侧分担。这不是“越大越好”，只是把范围从一边搬到另一边。`test_smoothing_reduces_activation_peak_for_outlier_fixture` 对比 `alpha=0.0` 和 `0.8`，断言前者 `activation_absmax` 更大、后者 `weight_absmax` 更大——验证的是“迁移发生了”，不是“输出误差单调下降”。

**Step 6：TODO 3——按量化输出误差选 `alpha`。** 参考答案：

```python
records = [self.compare(activations, weight, alpha) for alpha in candidates]
best = min(records, key=lambda record: record['quantized_mse'])
return {'best': best, 'records': records}
```

`compare` 先 `smooth`，再由 `_record` 算浮点参考 `reference`、平滑后浮点 `smooth_float`、量化恢复输出 `quantized_output`，返回 `equivalence_mse`、两侧重建误差和 `quantized_mse`。选 `alpha` 的依据是 **W8A8 输出误差 `quantized_mse`**，不是激活恢复误差或 `activation_absmax`。测试在 `[0.0,0.5,0.8]` 里选，断言 `best['quantized_mse'] == min(errors)`；空候选集抛 `ValueError`，`alpha=1.1` 在 `channel_scale` 就抛错。

**Step 7：校准与独立评测必须分开。** `select_alpha` 在校准样本上选因子；`evaluate(activations, weight, channel_scale)` 在独立样本上复用同一因子，不重新统计范围。`test_frozen_scale_on_independent_inputs` 断言复用因子与原因子 `torch.equal`、`smooth_float == held_out @ weight.T`、`alpha is None`，且它**不等于**在 held-out 上重算的因子——防止评测统计量泄漏。

**Step 8：可选框架核对。** Q-CPU 环境里的 `check_smoothquant_framework_conformance` 检查两件事：教学线性层在 `alpha=0.5` 下变换前后输出等价（`atol=rtol=1e-5`）；LLM Compressor 的 `SmoothQuantModifier(smoothing_strength=0.5)` 处理微型 Llama 后，权重确实变了、有限，相同 token 的 logits 在 `atol=1e-5, rtol=1e-4` 内保持。它明确声明“两侧因子不等价”。

**Step 9：GPU 实验的变量与指标。** 默认 `RUN_MODE="dry_run"`，`real_gpu` 才下载模型并生成产物。固定 Qwen2.5-0.5B-Instruct、WikiText-2（固定 revision）、校准 64 条、评测 64 条、生成/范围审计各 8 条、`MAX_SEQ_LENGTH=256`、`MAX_NEW_TOKENS=32`。唯一自变量是“是否启用 SmoothQuant + `alpha`”，候选 `(0.5,0.7,0.8,0.9)` 加未平滑 `w8a8` 和 `bf16`，共 6 条路径。

recipe 是 `GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])`；启用平滑时在前面插 `SmoothQuantModifier(smoothing_strength=alpha)`。质量指标是 Token 加权 NLL：`logits[:, :-1]` 与 `input_ids[:, 1:]` 做交叉熵，只对有效位置求和再除以 token 数；`perplexity = exp(min(20, NLL))`。辅助指标有 `nll_ratio_vs_bf16`、`nll_ratio_vs_w8a8`、`mean_greedy_prefix_agreement`、`input_range_ratio_vs_bf16`、`peak_incremental_bytes` 和 `artifact_manifest`。

门槛是 `MAX_NLL_RATIO_VS_BF16=1.10` 和 `MAX_NLL_RATIO_VS_W8A8=0.995`。产物清单不完整 → `tune_or_reject_candidates`；平滑后满足 BF16 上限且（优于 W8A8 或未平滑不合格）→ `accept_smoothquant_candidate`；否则未平滑合格 → `prefer_unsmoothed_w8a8_candidate`；都不行 → 拒绝。

**Step 10：怎么读本地 v4 结果。** notebook 记录选中 `smoothquant_alpha_0.5`：相对 BF16 的 NLL 比率约 `1.0022`，相对未平滑 W8A8 约 `0.9903`（低于 0.995，约改善 1%），因此接受。`q_proj`、`gate_proj` 输入范围明显收缩，`down_proj` 基本不变，说明迁移有模块差异；贪心前缀一致率约 `0.1289`，比 NLL 敏感，只能作辅证。

显存要小心：五个候选的 `peak_incremental_bytes` 都约 `1.38 GB`，但 `allocated_after_cleanup_bytes` 每轮仍增加约 `0.99 GB`，`reset_session()`、`gc.collect()`、`empty_cache()` 都没完全释放。这说明递增的是压缩链的跨轮状态残留，不是“alpha 越大越费显存”；`empty_cache()` 只能归还未被引用的缓存块。所以 `memory_comparison_valid=False`，**不能横向比较绝对峰值**，严格比较要在独立进程里逐候选重跑。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| W8A8 | 权重 8 位、激活 8 位 |
| 逐张量量化 | 整个张量共用一个缩放因子 |
| 对称量化 | 以 0 为中心、正负用同一套码点，范围 `[-qmax, qmax]` |
| `absmax` | 张量或通道里绝对值最大的元素 |
| 反量化 | 把整数码点乘回 `scale`，恢复成浮点近似值 |
| 异常通道（outlier channel） | 某一输入通道的数值范围远大于其他通道 |
| SmoothQuant | 用等价对角变换把激活的量化难度迁移一部分到权重 |
| $\alpha$ | 控制范围往哪边搬：0 偏激活、1 偏权重 |
| 浮点等价 | 变换前后 `XWᵀ` 不变，只改数值分布 |
| `quantized_mse` | 量化输出与浮点参考的均方误差，是选 `alpha` 的依据 |
| 校准 / 留出 | 用一组样本选因子、在另一组样本固定复用 |
| Token 加权 NLL | 按有效 token 数平均的交叉熵，越小越好 |

## 小白常见疑问
**Q1：浮点输出完全一样，为什么量化后误差会变？**
浮点等价只说明没改变线性函数，但量化发生在变换后的 `smooth_x`、`smooth_w` 上。变换改变了每列动态范围，也就改变了逐张量 scale 和舍入误差；输出误差是两侧误差相乘后传播的结果，可能下降，也可能因权重侧误差变大而上升。

**Q2：为什么选 `alpha` 看 `quantized_mse`，不看激活恢复误差？**
最终关心线性层输出，不是中间激活。`activation_reconstruction_mse` 下降可能只是把误差搬到权重侧，乘起来反而更差。TODO 3 和 `test_alpha_selection_returns_lowest_error_candidate` 都以 `quantized_mse` 最小为准。

**Q3：$\alpha=0$ 时权重范围变成 1，是不是很好量化？**
范围 1 不代表误差小。$\alpha=0$ 时 $s_j=1/w_j$，激活被乘以 $w_j$，异常值仍在激活侧；夹具里 `alpha=0.0` 的 `activation_absmax` 是 60，远大于 `alpha=0.8` 的约 2.27。这是用“激活更难量化”换来的。

**Q4：为什么 `evaluate` 不允许重新算 `channel_scale`？**
校准因子必须在看到评测输入前固定，否则评测统计量会泄漏，结果偏乐观。`test_frozen_scale_on_independent_inputs` 断言复用因子不等于 held-out 上重算的因子，正是防止偷偷重拟合。

**Q5：`down_proj` 的范围为什么基本不变？** SmoothQuant 只平滑指定线性层输入，`q_proj`、`gate_proj` 的输入被变换，`down_proj` 不在同一路径上。范围审计正是用这种模块差异解释改善来源。

## 动手验证：怎么知道自己做对了
入口是 `run_smoothquant_tests()`，依次跑 7 个测试，全过打印 `✅ SmoothQuant 机制测试通过：范围统计、等价迁移、误差转移与 alpha 选择均已验证。`
- `test_channel_scale_contract_and_outlier_response`：`alpha=0.5` 时形状 `(4,)`、有限为正，异常列 `scale[1] > scale[0]`。
- `test_smoothing_preserves_float_product`：随机 `(2,3,6)`/`(4,6)`、`alpha=0.6`，`equivalence_mse < 1e-10`。
- `test_smoothing_reduces_activation_peak_for_outlier_fixture`：`alpha=0.8` 的激活峰值小于 `alpha=0.0` 时，权重峰值则更大。
- `test_alpha_selection_returns_lowest_error_candidate`：`best['alpha']` 属于 `[0.0,0.5,0.8]`，`quantized_mse == min(errors)`。
- `test_invalid_alpha_candidates_are_rejected`：空候选集和 `alpha=1.1` 抛 `ValueError`。
- `test_frozen_scale_on_independent_inputs`：复用因子，`smooth_float == held_out @ weight.T`，`alpha is None`，且不等于重算因子。
- `test_zero_channels_and_invalid_inputs`：全零输入两项 MSE 为 0；非法 `qmax`、`eps`、形状、NaN、`alpha=nan`、全零 `channel_scale` 都抛错。

notebook 的 dump 没有保存 CPU 测试或 GPU 矩阵的执行输出；Step 5 的 v4 数字来自 notebook 的解析文字（`1.0022`、`0.9903`、`0.1289`、`1.38 GB`、`0.99 GB`），可以作为“作者记录的本地结果”引用，但不能当成你本机跑出来的。自己复测时，先看 `run_mode` 和 `failure.stage`；`dry_run` 只会写出 `evidence_level="environment_preflight"` 的 JSON。真正的候选比较要确认 `candidate_matrix_complete`、`artifact_manifests_complete`，再看 `selected_smoothquant_candidate` 和两个 NLL 比率；显存只在独立进程复测时才有横比意义。
