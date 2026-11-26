# 54. KV Cache 量化策略 — 小白讲解

> 对应 notebook：[`54_KV_Cache_Quantization_Strategies.ipynb`](./54_KV_Cache_Quantization_Strategies.ipynb)

## 一句话总结
模型权重装得下，不代表长对话跑得动：每生成一个 Token，都要为它保存后续注意力会反复读取的 Key 和 Value，长上下文和高并发会让 KV Cache 成为持续的显存与带宽成本。本节用 `KVQuantizationStrategies` 在 CPU 上复现 KIVI 式"异轴量化"：Key 沿 Token 维聚合、得到 `[..., 1, head_dim]` 的逐通道缩放因子；Value 沿 `head_dim` 聚合、得到 `[..., tokens, 1]` 的逐 Token 缩放因子；最近的 `residual_length` 个 Token 留在高精度窗口。再用 Transformers + HQQ 做配对实验，看 4-bit 缓存在 512/2048/4096 上下文下把增量峰值降了约 18%，但每 Token 生成时间涨了 25%–34%，4096 时 NLL 增量 0.973 超过 0.5 的门槛而被拒绝。

## 为什么需要它：痛点是什么
初学者对 KV Cache 量化常有两个错觉。第一，"4-bit 就是把字节数除以 4"：实际还要存缩放因子（scale），而 scale 是 FP32，每个通道或每个 Token 一个，短片段里元数据可能比码值还大；如果只是把 4-bit 码塞进 `int8` 容器、不做位打包，占用甚至会比原来的 FP16/FP32 更大。第二，"缓存小了，生成一定更快"：读取时要先恢复、再拼接，恢复是额外的乘法和内存流量；容量收益和延迟收益是两本账，可能一个涨一个跌。

生活类比：把档案拍成缩微胶片。Key 像"按科目装订的成绩册"——同一科目的所有年份放在一起，用一把比例尺（逐通道 scale）就够；Value 像"每个人的体检报告"——每份报告内部的数值范围相近，但人和人之间差很多，所以每人一把比例尺（逐 Token scale）更合适。胶片很省柜子，但每次查阅都要上放大机；如果一份胶片只有一页，放大机的固定成本反而比省下的柜子更贵。

## 核心问题
- Key 和 Value 为什么要用不同的量化轴？各自的 scale 形状是什么，为什么能广播？
- 对称量化的编码、恢复公式怎么写？全零通道为什么不能产生无穷 scale？
- 高精度残留窗口怎么转存？`flush_tokens` 怎么算，为什么已量化分段不能重写？
- 为什么"容器字节""理论打包字节""物理存储字节"是三个不同的账本？
- 量化 KV 什么时候真的省显存？4096 上下文省得最多，为什么反而被质量门槛拒绝？
- 残留窗口从 16 增到 64，为什么延迟下降但增量峰值几乎不变？

## 解决思路：一步步看懂

**第 1 步：K 和 V 在 Attention 里的角色不同，分布也不同。** 每个 Decode 步都会追加新的 Key 和 Value，后续 Attention 里 Key 与当前 Query 做点积决定"看哪里"，Value 再按注意力权重汇聚。notebook 采用的教学形状是 `[batch, heads, tokens, head_dim]`，记作 $B\times H\times L\times D$。notebook 的表述是"KIVI 根据观察到的分布差异，采用 Key 逐通道、Value 逐词元的分组量化"：直观上，Key 与 Query 的点积让同一通道在不同 Token 间有可比较的范围，逐通道 scale 保留这种通道差异；Value 的数值范围更依赖具体 Token，逐 Token scale 更贴合。两个公式是

$$
s^K_{b,h,1,d}=\frac{\max_l|K_{b,h,l,d}|}{q_{\max}},\qquad
s^V_{b,h,l,1}=\frac{\max_d|V_{b,h,l,d}|}{q_{\max}}.
$$

逐项解释：$B$ 是批大小，$H$ 是 KV Head 数，$L$ 是缓存 Token 数，$D$ 是每个 Head 的维度；$q_{\max}$ 是量化码值的最大绝对值。$s^K$ 的下标里 Token 维被写成 1，表示它沿 $l$ 归约掉了，形状是 `[..., 1, head_dim]`；$s^V$ 的下标里通道维被写成 1，表示沿 $d$ 归约掉了，形状是 `[..., tokens, 1]`。保留长度为 1 的维度不是为了好看，而是为了能广播回原张量：`[B,H,1,D]` 可以和 `[B,H,L,D]` 逐元素相除，`[B,H,L,1]` 也可以。

为什么不能反过来？如果把 Key 沿 `head_dim` 聚合，scale 变成 `[..., tokens, 1]`，同一个 Token 的所有通道被迫共享一个范围，某个通道的极值会把其他通道的精度压没；如果把 Value 沿 Token 维聚合，scale 变成 `[..., 1, head_dim]`，一个 Token 的极值会污染整个历史。测试 `test_key_per_channel_scale_shape` 和 `test_value_per_token_scale_shape` 直接用形状把轴钉死：前者要求 `(1, 2, 1, 8)`，后者要求 `(1, 2, 5, 1)`，轴写反了形状就对不上。

**第 2 步：编码与恢复。** 给定浮点张量 $x$ 和恢复尺度 $s$，编码是

$$
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right),-q_{\max},q_{\max}\right),\qquad
\hat x=sq.
$$

`round` 是四舍五入到最近整数，`clip` 把码值限制在 $[-q_{\max},q_{\max}]$，恢复时乘回 $s$。这是对称量化：零点固定在 0，码值范围关于 0 对称。`KVQuantizationStrategies.__init__` 默认 `qmax=7`，注释写明"表示有符号 4-bit 的教学数值范围；张量仍以 int8 容器保存"。`code_bits = qmax.bit_length() + 1` 是根据码值反推需要的位数：7 的二进制是 `111`，`bit_length()` 是 3，加 1 个符号位得 4；`qmax=1` 得 2 位，`qmax=127` 得 8 位。测试 `test_scale_formula_and_separate_packing` 对 `(1, 2)`、`(7, 4)`、`(127, 8)` 三组做了断言。

`eps=1e-8` 是 scale 的正下限。全零通道的 `absmax` 是 0，$0/q_{\max}=0$，如果不加下限，编码时 $x/s$ 会得到 `inf` 或 `nan`；`clamp_min(eps)` 让 scale 保持有限且为正，代价是全零通道恢复出来仍然是 0（0 乘任何正数都是 0），语义正确。`test_zero_range_scale_is_finite` 用全零 K/V 断言 `torch.isfinite(scale).all()`。

**第 3 步：TODO 1，Key 逐通道量化。** 参考答案三行：

```python
key_absmax = key.float().abs().amax(dim=-2, keepdim=True)
key_scale = (key_absmax / self.qmax).clamp_min(self.eps)
key_q = self._quantize(key, key_scale)
```

`dim=-2` 是 Token 维，`keepdim=True` 保留它并写成 1；`amax` 取绝对值的最大值，不用 `max` 是因为要的是幅度而不是带符号的最大。`key.float()` 把 FP16/BF16 输入提升到 FP32 再统计，避免低精度求和/比较的误差。`_quantize` 内部再做一次 `x.float()`、除法、`round`、`clamp`，最后 `.to(torch.int8)`。

拿测试里的例子手算：`x = [[[[-14.0], [3.0]]]]`，形状 `(1,1,2,1)`，Token 维是 2。沿 Token 维取 absmax 得 14，scale $=14/7=2$，所以 `torch.testing.assert_close(scale, full_like(scale, 2.0))`。码值是 $\mathrm{round}(-14/2)=-7$ 和 $\mathrm{round}(3/2)=2$（PyTorch 的 `round` 遇到 .5 取偶数），恢复得 $-14$ 和 $4$，第二项的误差是 1，正好等于半个 scale。测试 `test_recovery_error_bounds_and_small_segment_metadata` 用 `|restore(codes, scale) - x| <= scale / 2 + 1e-6` 把这个上界写成了断言：四舍五入到最近的整数，误差最多半个格子。

**第 4 步：TODO 2，Value 逐 Token 量化。** 参考答案只把归约维换掉：

```python
value_absmax = value.float().abs().amax(dim=-1, keepdim=True)
value_scale = (value_absmax / self.qmax).clamp_min(self.eps)
value_q = self._quantize(value, value_scale)
```

`dim=-1` 是 `head_dim`，`keepdim=True` 得到 `[..., tokens, 1]`。测试例子 `[[[[3.0, -4.0]]]]` 每个 Token 有 2 个通道，absmax 是 4，scale $=4/7$，所以断言 `full_like(value_scale, 4.0 / 7.0)`。注意两个 TODO 都不能图省事复用同一个 helper：形状检查只是表面，真正的问题是归约轴决定了"谁和谁共享范围"，而 K 和 V 的分布假设正好相反。

**第 5 步：TODO 3，追加与转存。** 状态是一个字典，里面有四个分段列表 `key_q_segments`、`key_scale_segments`、`value_q_segments`、`value_scale_segments`，两个高精度尾部 `key_residual`、`value_residual`，以及 `quantization_config`、`residual_length`、`flushed_tokens`、`quantized_tokens`、`total_tokens`。每次追加先把旧尾部和新 Token 拼起来：

```python
key_residual_all = torch.cat((state["key_residual"], new_key), dim=-2)
...
flush_tokens = max(key_residual_all.shape[-2] - residual_length, 0)
```

`flush_tokens` 是"拼接后总长度超出窗口的部分"，且只在超出时为正。然后切三刀：前 `flush_tokens` 个是本次要量化的旧 Token，后面 `residual_length` 个留在高精度窗口。**只量化溢出的最旧一段**，已有的 `key_q_segments` 原样复制，不重写。这是"增量分段"策略，和"HQQ 到阈值时恢复并重新量化整个前缀"是两种不同的更新规则。

手算 `test_residual_window_partition_contract`：第一次追加 3 个 Token、`residual_length=2`，总长 3 超过 2，`flush_tokens = 1`，量化 1 个、残留 2 个；第二次再追加 2 个，残留 2+2=4，`flush_tokens = 2`，量化 2 个、残留 2 个。最终 `quantized_tokens = 1 + 2 = 3`，`key_residual.shape[-2] = 2`，总数 5。`test_append_state_contract` 用 `residual_length=4` 验证另一种情况：3 个 Token 不溢出（`flushed_tokens=0`），再加 2 个共 5 个，溢出 1 个，总数仍是 5。

三个不变量必须同时成立：
- **Token 守恒**：`total_tokens = quantized_tokens + key_residual.shape[-2]`，`_validate_state` 每次都会核对；
- **不重写历史**：`test_previous_quantized_segments_are_not_rewritten` 先保存第一段 `key_q_segments[0]` 的克隆，第二次追加后 `torch.equal` 必须仍为真，且 `len(key_q_segments) == 2`；
- **未溢出不能伪造分段**：`test_window_without_overflow_contract` 用 1+1 个 Token、窗口 4，要求 `quantized_tokens == 0`、`flushed_tokens == 0`、残留 2。

为什么尾部要 `.clone()`？`torch.cat` 产生的新张量里，尾部如果直接切片拿走，会继续引用整个拼接缓冲区；哪怕逻辑上只剩 2 个 Token，物理存储还占着 4 个 Token 的空间。`test_full_read_order_and_owned_residual_storage` 检查 `tail.untyped_storage().nbytes() == tail.numel() * tail.element_size()`，并断言 `not key.requires_grad`；最后把原始 `full_key` 全部填成 999，再读一次状态，结果必须不变——说明状态持有的是自己的副本，不是原张量的视图。

**第 6 步：三本字节账。** `state_bytes` 返回七个字段：`code_bits`、`metadata_bytes`、`residual_bytes`、`container_code_bytes`、`logical_state_bytes`、`physical_storage_bytes`、`minimum_packed_state_bytes`。它们的区别是：
- `container_code_bytes`：量化码实际放在 `int8` 容器里，每个码 1 字节。教学实现**没有做位打包**，`compare` 里 `packing_implemented` 恒为 `False`。
- `minimum_packed_state_bytes`：如果真把 $b$ 位码打包，码值需要 $\lceil \text{numel}\times b/8\rceil$ 字节，加上 scale 和 residual。
- `physical_storage_bytes`：按 `(device, data_ptr)` 去重后，所有张量底层存储的真实字节数。去重是为了避免同一个存储被两个视图重复计数。

手算两个例子（`qmax=7`，所以 4 位）：
- `1×1×1×1` 的 K/V：码值各 1 个元素，打包后各 1 字节，共 2 字节；scale 是 FP32，K 的 scale 形状 `(1,1,1,1)` 一个、V 的也是一个，共 8 字节；`packed_quantized_bytes = 2 + 8 = 10`。而原始 FP32 的 K/V 是 2 个元素 × 4 字节 = 8 字节。所以 `container_quantized_bytes`（不打包，码值 1+1 字节，scale 8 字节，共 10）比 `raw_bytes` 还大。`test_recovery_error_bounds_and_small_segment_metadata` 就用 `container_quantized_bytes > raw_bytes` 断言了这一点。
- `1×1×1×8` 的 K/V：码值各 8 个元素 × 4 位 = 32 位 = 4 字节，共 8 字节；K 的 scale 形状 `(1,1,1,8)` 是 8 个 FP32 = 32 字节，V 的 scale 形状 `(1,1,1,1)` 是 4 字节，元数据共 36 字节；总计 $2\times4 + 36 = 44$ 字节。测试写成 `2 * bits + 36`，对 `bits` 为 2、4、8 都成立。原始 FP32 是 16 个元素 × 4 = 64 字节，所以这个尺寸下打包才真正省。

这解释了为什么 notebook 反复强调"不能只比较名义位宽"。真实缓存容量还要加缩放、零点、打包填充和底层存储；一个小切片仍可能引用完整大张量。Step 3 给的容量公式是 $2BHL_qDb/8 + 2BHRD b_h/8$：因子 2 是 K 和 V 两份，$L_q$ 是量化前缀长度，$R$ 是高精度尾部长度，$b$ 是码值位宽，$b_h$ 是高精度位宽（FP32 是 32，BF16 是 16）。

**第 7 步：框架核对只核对生命周期，不核对数值等价。** `check_kv_cache_framework_conformance` 让教学状态机和 Transformers 的 `QuantizedCache` 共享同样的两次追加：先 3 个 Token，再 2 个，`residual_length=4`。教学路径第一次不溢出，第二次溢出 1 个；HQQ 路径从第一次 `update` 就建立 `_quantized_keys` 基段，第二次更新后 `framework_layer.keys` 保留最新 2 个 Token 作为残留。所以两边的分段时机不同，核对的是共同不变量，而不是 scale 或恢复值相等：
- Token 守恒：本地 `local_total == 5`，框架 `get_seq_length() == 5`；
- 量化基段：本地 `local_quantized_before == 0 and local_quantized_after > 0`，框架两次更新后都有 `_quantized_keys`；
- 残留上限：本地 4、框架 2，都不超过 4；
- 恢复完整性：两边形状都是 `(1, 2, 5, 8)`，且最新 2 个 Token 与输入完全相等、数值有限。

返回字典里 `quantization_algorithm_equivalence_claimed` 是 `False`，`check_kind` 是 `shared_input_lifecycle_invariants`。这句话很重要：它明确否认"两边算的是同一种量化"。

**第 8 步：GPU 实验的设计。** 实验用 `Qwen/Qwen2.5-0.5B-Instruct`、BF16、HQQ 4-bit、`q_group_size=64`、K 和 V 都用 `axis=1`，基线是 Transformers 的 `DynamicCache`，候选是 `QuantizedCache`。这里有一个必须说清的边界：**K/V 都取 `axis=1` 是本实验的配置，不是 KIVI 的异轴策略**。notebook 明确写"该矩阵比较同轴缓存的容量、质量和生成代价，没有比较 K/V 异轴策略是否更优"。CPU 题目讲的是"为什么可以异轴"，GPU 实验验证的是"量化缓存这套机制在真实模型上值不值"，两者不能互相代替。

实验分成两个矩阵，每次只动一个自变量：
- **容量矩阵**：上下文 512 / 2048 / 4096 Token，固定生成 64 Token、残留窗口 128，只改变上下文长度和缓存实现。它回答"从多长上下文开始，量化带来的增量峰值收益变得可观"。
- **窗口矩阵**：固定上下文 512、生成 128 Token，残留窗口取 16 / 32 / 64，只改变窗口大小。生成长度 128 大于最大窗口 64，保证每种窗口都真的发生多次转存；`expected_decode_window_flushes = 128 // residual_length` 分别是 8、4、2 次。

指标定义要看清：
- `incremental_peak_allocated_bytes` 是"生成过程的峰值已分配字节减去生成前基线"，衡量增量，不是模型常驻权重，也不是整机显存；
- `generation_ms_per_token` 是整段生成中位耗时除以生成的 Token 数，**包含 Prefill**，不是严格意义的 TPOT；
- 质量用固定 continuation 上的平均 NLL 和每步 Top-5 集合重合率。continuation 取自基线贪心输出的前 16 个 Token，这样两条路径评的是同一段文本，不会因为生成分歧越走越远；
- `token_agreement_rate` 和 `first_divergence_index` 只作诊断，不参与准入。

准入门槛写死在常量里：`MAX_MEAN_NLL_DELTA = 0.50`、`MIN_TOPK_OVERLAP = 0.60`、`MAX_LATENCY_RATIO = 2.00`。三个条件加上"显存确实有正收益"，才进入 `eligible_rows`。

**第 9 步：结果怎么读。** 容量矩阵（`default18`，RTX 5070 Ti Laptop 12 GiB，commit `7ae5576`）：

| 上下文 | 基线峰值 | 候选峰值 | 节省 | 降幅 | 延迟（基线→候选） | 延迟比 | NLL 增量 | Top-5 重合 |
|---|---|---|---|---|---|---|---|---|
| 512 | 25,049,600 B | 20,527,616 B | 4,521,984 B | 18.05% | 460.36→577.58 ms | 1.255 | 0.467 | 0.6125 |
| 2048 | 103,336,448 B | 85,248,512 B | 18,087,936 B | 17.50% | 495.84→637.29 ms | 1.285 | 0.375 | 0.70 |
| 4096 | 200,378,880 B | 164,203,008 B | 36,175,872 B | 18.05% | 565.19→758.50 ms | 1.342 | 0.973 | 0.55 |

先看容量：三档都省了约 18%，绝对节省从 4.3 MiB 涨到 34.5 MiB。再看质量：512 和 2048 的 NLL 增量分别是 0.467 和 0.375，都低于 0.50；4096 是 0.973，超了。Top-5 重合率 512 是 0.6125（刚过 0.60 的线），2048 是 0.70，4096 掉到 0.55。于是 4096 尽管省得最多，仍然被质量门槛拒绝——这就是"容量收益不能单独决定候选"。

NLL ratio 为什么不能当门槛？基线 NLL 很小（512 是 0.00552，2048 是 0.00139，4096 是 0.00153），因为 continuation 本来就是基线自己贪心生成的，基线对这段文本非常自信；候选的 NLL 只增加了 0.375，比值却是 271 倍。用比值会把一个绝对上很小的退化放大成"数量级崩坏"，所以 v5 协议改用绝对增量，ratio 只放在 `fixed_continuation_nll_ratio` 里做诊断。

窗口矩阵：

| 残留窗口 | 预期转存次数 | 延迟比 | 增量峰值节省 | Top-5 重合 | 贪心一致率 | 首次分歧 |
|---|---|---|---|---|---|---|
| 16 | 8 | 1.604 | 4,521,984 B | — | 0.125 | 8 |
| 32 | 4 | 1.409 | 4,521,984 B | — | 0.0625 | 8 |
| 64 | 2 | 1.252 | 4,521,984 B | — | 0.0625 | 8 |

窗口从 16 增到 64，延迟比从 1.60 降到 1.25，方向和"转存次数从 8 次降到 2 次"一致：窗口越小，越频繁地把旧 Token 量化、追加分段，恢复和元数据开销越多。但增量峰值三档几乎一样（都是 4,521,984 B）；一个合理的解释是，这次 512 Token 上下文的 Prefill、生成过程和固定开销占了大头，残留窗口那点字节差异被淹没了。窗口矩阵没有做固定 continuation 的质量评估，所以 NLL 和 Top-5 两列是 `None`，不能拿它去谈质量。

选中的候选是 2048 上下文、残留 128：`incremental_peak_saved_bytes = 18,087,936`，降幅 0.175，延迟比 1.285，NLL 增量 0.375，Top-5 0.70。`mechanism_metrics` 记录 `memory_reduction_onset_context = 512`（第一次看到正显存收益）、`eligible_crossover_context = 512`（第一次同时通过容量、质量、延迟门槛）、`eligible_candidate_count = 2`（512 和 2048）。`decision = accept_local_cache_candidate`。

但这里必须把话说全：2048 候选的 `token_agreement_rate` 只有 0.171875，`first_divergence_index = 2`——贪心输出从第 3 个 Token 就和基线分叉了。它通过的是"固定 16 Token 轨迹上的 NLL 和 Top-5"这道局部质量门，不是"开放式生成质量等价"。容量是生成增量峰值，不是缓存张量的驻留字节；计时含 Prefill，不是 TPOT；实验只覆盖 Transformers + HQQ，不代表 vLLM/SGLang 的 Serving 后端。所以 5.4 的结论是"可在目标推理后端中复测，尚不能证明部署可用"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| KV Cache | 每个 Token 的 Key/Value 状态，生成时反复读取，随上下文线性增长 |
| 量化轴 | 哪些数值共享一个缩放因子；本节 Key 按通道、Value 按 Token |
| absmax | 一组数绝对值的最大值，用来定这一组的范围 |
| 对称量化 | 零点固定在 0，码值范围 $[-q_{\max},q_{\max}]$ 关于 0 对称 |
| $q_{\max}$ | 码值最大绝对值，默认 7 表示 4-bit 的有符号范围 |
| scale（恢复尺度） | 编码时除、恢复时乘的那个正数 |
| 高精度残留窗口 | 最近 `residual_length` 个 Token 不量化，保留原始精度 |
| 转存（flush） | 把溢出窗口的最旧 Token 量化并追加为新分段 |
| 增量分段 | 只量化新溢出的部分，已有量化段不重写 |
| 位打包 | 把多个 4-bit 码塞进一个字节；本节教学代码不打包 |
| 增量峰值 | 生成过程峰值减去生成前基线，衡量缓存带来的增量 |
| 固定轨迹 NLL | 在同一段 continuation 上算平均负对数似然，避免生成分歧累积 |

## 小白常见疑问

**Q1：`qmax=7` 是 4-bit，为什么码值不是 -8 到 7？**
对称量化的码值取 $[-q_{\max},q_{\max}]$，7 对应 15 个整数（-7 到 7），4 位能表示 16 个值，剩下一个 -8 不用。这样做的好处是正负范围完全对称，零点没有偏移；代价是浪费一个码字。`code_bits = qmax.bit_length() + 1` 算的是"表示这个范围至少需要几位"，不是"容器类型占几位"。容器仍然是 `int8`，每个码 1 字节，所以位宽收益只有真正打包后才兑现。

**Q2：全零通道为什么不能直接让 scale 等于 0？**
恢复公式是 $\hat x=sq$，$s=0$ 时所有码值都会被乘成 0，看起来"全零通道恢复成零"没问题；但编码时 $x/s$ 会得到 `inf` 或 `nan`，`restore` 也会检查 `scale > 0` 并拒绝零尺度。`clamp_min(eps)` 用一个正的下限保证有限且可逆，`test_zero_range_scale_is_finite` 断言全零输入的 scale 仍然 `isfinite`。

**Q3：既然量化了，为什么字节数还可能变大？**
因为 scale 是 FP32 元数据，而且本节没有做位打包。`1×1×1×1` 的例子：原始两个 FP32 是 8 字节，量化后两个 int8 码是 2 字节，但两个 scale 要 8 字节，总共 10 字节，反而更大。只有元素数足够多、打包真正生效时，4-bit 才省。`test_scale_formula_and_separate_packing` 用 `1×1×1×8` 的例子算出 $2\times4+36=44$ 字节，比原始 64 字节小，就是"尺寸够大才划算"的边界。

**Q4：4096 上下文省了 34.5 MiB，为什么决策不选它？**
因为它没通过质量门槛：NLL 增量 0.973 超过 `MAX_MEAN_NLL_DELTA = 0.50`，Top-5 重合率 0.55 低于 `MIN_TOPK_OVERLAP = 0.60`。准入是"容量、质量、延迟"三条同时满足，4096 只满足了容量和延迟（1.342 < 2.00）。省得多不等于可以接受，质量退化可能让下游任务整体变差。

**Q5：CPU 题目讲 K/V 异轴，GPU 实验却让 K/V 都用 `axis=1`，是不是自相矛盾？**
不是，是两个不同的问题。CPU 题目回答"为什么可以异轴、异轴的 scale 形状怎么来"；GPU 实验回答"Transformers + HQQ 这套量化缓存机制在真实模型上容量、质量、延迟怎么权衡"。HQQ 的 `axis=1` 是本实验的配置，notebook 明说"不是 HQQ 仅支持这一轴"，也明说"没有比较 K/V 异轴策略是否更优"。框架一致性核对也只核对生命周期不变量，返回里 `quantization_algorithm_equivalence_claimed = False`。

**Q6：残留窗口是不是越大越好？**
不是单调的。窗口大，转存次数少、恢复开销低，延迟可能下降（16→64 时延迟比 1.60→1.25）；但窗口大意味着更多 Token 以高精度保存，容量收益可能变小。本实验里三档窗口的增量峰值几乎一样；一个合理的解释是 512 Token 的 Prefill 和固定开销占比太大，掩盖了窗口差异，但 notebook 没有逐次记录转存事件，不能从窗口长度直接算出实测转存频率。换成长上下文或更大模型，结论可能不同，需要重新配对测量。

## 动手验证：怎么知道自己做对了
`run_kv_strategy_tests()` 依次跑 12 个测试，全过才打印 `✅ KV Cache 策略测试通过：量化轴、追加状态、窗口边界与字节账本均已验证`。TODO 没填会抛 `NotImplementedError`（消息分别是 `TODO 1: quantize Key per channel`、`TODO 2: quantize Value per token`、`TODO 3: compute newly flushed token count`）。

- `test_key_per_channel_scale_shape`：`key_q.shape == key.shape`，`key_scale.shape == (1, 2, 1, 8)`。
- `test_value_per_token_scale_shape`：`value_q.shape == value.shape`，`value_scale.shape == (1, 2, 5, 1)`。
- `test_append_state_contract`：3 个 Token 加 2 个 Token、窗口 4，`total_tokens == 5`，量化数加残留数在 K/V 两侧都等于 5。
- `test_residual_window_partition_contract`：窗口 2 时，`quantized_tokens == 3`，两段 `key_q_segments` 合计 3，K/V 残留各 2。
- `test_previous_quantized_segments_are_not_rewritten`：两次追加后 `len(key_q_segments) == 2`，第一段与克隆完全相等，`flushed_tokens == 2`。
- `test_window_without_overflow_contract`：1+1 个 Token、窗口 4，`quantized_tokens == 0`、`flushed_tokens == 0`、残留 2。
- `test_restore_and_budget_contract`：`key_mse`/`value_mse` 有限，`packed_quantized_bytes < raw_bytes` 且 `< container_quantized_bytes`。
- `test_zero_range_scale_is_finite`：全零 K/V 的两个 scale 都有限。
- `test_scale_formula_and_separate_packing`：`[-14, 3]` 的 Key scale 为 2.0；`[3, -4]` 的 Value scale 为 $4/7$；`qmax` 1/7/127 对应 `code_bits` 2/4/8；单元素打包字节 `2+8`，八元素 `2*bits+36`。
- `test_full_read_order_and_owned_residual_storage`：三段 `(0,3)`、`(3,4)`、`(4,6)` 追加后总长 6，读取顺序与分段一致，每段误差不超过 `scale/2 + 1e-6`，最新两个 Token 与原始完全相等，尾部存储字节恰好等于 `numel × element_size`，`physical_storage_bytes == logical_state_bytes`，`minimum_packed_state_bytes <= logical_state_bytes`。
- `test_recovery_error_bounds_and_small_segment_metadata`：`(1,1,1,3)` 和 `(2,3,5,7)` 两种形状下，K/V 两种量化的恢复误差都受半个 scale 约束；单元素时容器字节大于原始字节。
- `test_invalid_inputs_and_cache_identity`：`qmax=0`、`qmax=128`、`eps=nan`、空 Token 维、含 `nan` 的输入、`residual_length=1.5`、把 `double` 张量续到已有缓存、把窗口从 2 改成 3、换 `qmax=3` 的策略续用旧状态，九种情况全部抛 `ValueError`。

框架核对 `check_kv_cache_framework_conformance` 的 12 项 `checks` 全为 `True` 才返回 `status = 'passed'`，键名包括 `local_tokens_are_conserved`、`framework_tokens_are_conserved`、`local_crosses_quantization_threshold`、`framework_has_quantized_base`、`framework_keeps_new_tokens_as_residual`、`both_residuals_are_bounded`、`framework_restore_shape_is_complete`、`local_restore_shape_is_complete`、`local_restore_is_finite`、`local_latest_tail_is_unchanged`、`framework_latest_tail_is_unchanged`、`framework_restore_is_finite`；`partition_observation` 会打印教学路径残留 4 个 Token、框架路径残留 2 个 Token。它需要 `transformers` 和 `hqq`，没装会直接导入失败，不是"跳过"。

GPU 部分默认 `RUN_MODE = "dry_run"`，只登记配置、不加载模型；改成 `real_gpu` 需要 `Q-CUDA-MECHANISM` 预检通过、CUDA 可用，结果写到 `benchmarks/results/54_transformers_quantized_cache_matrix_<RUN_LABEL>.json`。读取时先看 `resolved_model_commit`、`failure`、`evidence_level`，再按 `matrix_role` 分开看 `context_capacity` 和 `residual_window` 两类行；`failure` 非空时（例如依赖或后端报错）只能做故障定位，不能把失败记成量化收益。当前 `default18` 的 `failure` 为 `None`、`evidence_level = matched_transformers_quantized_cache_context_window_matrix`、`decision = accept_local_cache_candidate`，这是本机 12 GiB 显卡上的局部机制证据，不是生产部署结论。
