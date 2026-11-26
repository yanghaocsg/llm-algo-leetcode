# 20. FlashAttention 分块模拟 — 小白讲解

> 对应 notebook：[`20_FlashAttention_Sim.ipynb`](./20_FlashAttention_Sim.ipynb)

## 一句话总结
本节用 `flash_attention_forward_sim` 在二维 Q/K/V 上复现 FlashAttention 的前向数据流：不再先算出完整的 $N\times N$ 分数矩阵，而是把 Q 和 K/V 切成小块，用每条查询行的最大值 `m`、指数和 `l`、累计输出 `O` 做在线 Softmax 合并。四个 TODO 拼出这套递推，测试要求结果与完整注意力在 `1e-5` 内一致；最后用工作集代理和可选的 GPU SDPA 对照，说明"算法省工作集"和"实际变快"需要分别验证。

## 为什么需要它：痛点是什么
标准注意力只有三行：`scores = QKᵀ/√D`、`softmax`、`@V`。但 `scores` 有 $N^2$ 个元素：序列长度 4096 时是 1677 万个分数，单头 FP16 约 33.5 MB，8 个头叠加起来很可观，而且通常"写一遍、读一遍"，显存带宽成了瓶颈。

初学者第一次写分块版本，最直觉的错法是：对每个 K/V 分块单独 softmax，再把结果平均。这样每块的归一化分母只覆盖自己，拼出来和完整 softmax 不一样。真正要解决的是：**只看到一部分键时，怎样维护一个以后还能继续合并的"半成品"。**

生活类比：老师改卷子。完整做法是把所有得分写成一张总表再统一算平均；分块做法是每次只拿一叠卷子，边改边记"目前最高分"和"按它折算过的总分"，下一叠来了如果最高分变了，就把旧总分按新旧最高分的比例重算，再并入新一叠。最后结果和一次性算完全一样。

## 核心问题
- 为什么不能对每个分块单独做 Softmax 再拼接？差在哪个分母上？
- `m`、`l`、`O` 三个状态各存什么？为什么必须逐行保存？
- 重标定因子 $\alpha$ 为什么必须同时作用在历史指数和与历史输出上？
- 分块后的因果掩码为什么用全局位置，而不能用块内下标？
- 分块减少了什么、没减少什么？工作集代理和 GPU 实验各自能证明什么？

## 解决思路：一步步看懂

**第 1 步：标准路径 vs 分块路径。** 函数签名是 `flash_attention_forward_sim(q, k, v, block_size=2, causal=False)`，输入三个 `[seq_len, dim]` 张量，返回同样形状的 `[seq_len, dim]` 输出。完整路径算出 `[N, N]` 的 $S=QK^T/\sqrt{D}$，逐行 Softmax，再乘 $V$。分块路径固定 Q 分块 $Q_i$（最多 $B_q$ 行），依次和每个 $K_j,V_j$（最多 $B_k$ 行）算局部分数 $S_{ij}=Q_iK_j^T/\sqrt{D}$，形状只有 $|Q_i|\times|K_j|$。notebook 的 Step 1 表格概括为：分数从"完整"变"当前块"，Softmax 从"整行统一"变"跨块合并"，输出从"一次乘完"变"逐块累加"，工作集从 $O(N^2)$ 变成只跟块大小有关。这里的块是**算法分块**，不是 Transformer 模型块，也不是 CUDA thread block。

**第 2 步：在线 Softmax 的三个状态。** 对 $Q_i$ 的每一行维护：$m_i$ 是已处理分块的最大值（指数基准），$l_i$ 是已处理分块的指数和（未归一化分母），$O_i$ 是加权输出累计。初始 $m_i=-\infty$、$l_i=0$、$O_i=0$。收到新分块：

$$m_i'=\max(m_i,\mathrm{rowmax}(S_{ij})),\quad \alpha=\exp(m_i-m_i'),\quad W_{ij}=\exp(S_{ij}-m_i')$$

$$l_i'=\alpha l_i+\mathrm{rowsum}(W_{ij}),\qquad O_i'=\frac{\alpha l_iO_i+W_{ij}V_j}{l_i'}$$

$\alpha$ 把"按旧基准算的历史指数"换算到新基准；$W_{ij}$ 是当前块相对新基准的权重；$l_i'$ 把重标定后的历史指数和与当前块相加；分子里的历史输出也乘 $\alpha l_i$，和分母用同一套比例。

**第 3 步：一个手算例子。** 某一行三个键的分数是 `[1, 2]` 和 `[3]`。第一块：$m=2$，$l=e^{-1}+1=1.3679$。第二块：`new_max=3`，$\alpha=e^{-1}=0.3679$，$W=1$，$l'=1.3679\times0.3679+1=1.5032$。对照完整 softmax：分母 $e^1+e^2+e^3=30.1928$，除以 $e^3=20.0855$ 恰好是 $1.5032$；三个位置的权重为 `[0.0900, 0.2447, 0.6652]`，和为 1。分块没有改变数学，只是把"一次求分母"改成"边看边维护分母"。

**第 4 步：TODO 1——初始化。** 参考答案：

```python
m = torch.full((seq_len, 1), -float('inf'), device=q.device, dtype=q.dtype)
l = torch.zeros((seq_len, 1), device=q.device, dtype=q.dtype)
```

形状必须是 `[seq_len, 1]` 而不是 `[seq_len]`：`scores - new_max` 要按行广播，保留列维才能让 `[B_q, B_k]` 每一行减自己的最大值。`dtype=q.dtype` 必须跟输入走，`test_dtype_and_stability` 用 `float64` 检查类型不漂移。`m` 用负无穷，让第一块的 `history_scale=exp(-inf-finite)=0`；初始成 0 会把第一块错误地抬到不存在的基准上。

**第 5 步：TODO 2——局部分数与因果掩码。**

```python
scores = q_block @ k_block.transpose(-2, -1)
if causal:
    query_pos = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
    key_pos = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
    future_mask = key_pos > query_pos
    scores = scores.masked_fill(future_mask, -float('inf'))
```

`q_block` 进入内层循环前已乘过 `scale = 1.0 / math.sqrt(dim)`，这里不再除。因果判断用 `i`、`j` 这两个**全局起点**加偏移，不是块内 `0..B_q`；否则第二个 Q 分块会把本该可见的历史键当"未来"屏蔽。`masked_fill` 在求最大值之前，被屏蔽位置既不影响 `m`，也不分权重。

**第 6 步：TODO 3——重标定。**

```python
tile_max = torch.max(scores, dim=-1, keepdim=True)[0]
new_max = torch.maximum(m_i, tile_max)
history_scale = torch.exp(m_i - new_max)
tile_weights = torch.exp(scores - new_max)
```

`keepdim=True` 保住 `[B_q, 1]` 才能广播。`new_max` 取历史和当前的最大值，保证指数不超过 1、不上溢；`history_scale` 就是 $\alpha$，`tile_weights` 就是 $W_{ij}$。常见错误是只对当前块减 `tile_max`、忘了把历史搬到 `new_max`，两块基准不同，合并出的分母就错了。

**第 7 步：TODO 4——合并分母与输出。**

```python
new_l = l_i * history_scale + tile_weights.sum(dim=-1, keepdim=True)
out_i = out_i * (l_i * history_scale / new_l) + (tile_weights @ v_block) / new_l
m_i, l_i = new_max, new_l
```

第二行等价于 $O_i'=(\alpha l_iO_i+WV_j)/l_i'$：历史输出先乘 $\alpha l_i/l_i'$ 完成重标定和归一化，当前块贡献再除以同一个 $l_i'$。**分子和分母必须用同一个 $l_i'$**。`out_i` 是 `out` 切片的视图，重新绑定后要在 Q 分块结束时写回 `out[i:i+block_size]`。

**第 8 步：尾块与工作集。** `range(0, seq_len, block_size)` 的最后一块可能不足 `block_size`，切片自然变短，公式里的 $|Q_i|$、$|K_j|$ 就是实际长度。计算量仍是 $O(N^2)$（每个 Q 都要看所有 K），变的是工作集：完整分数矩阵随 $N^2$ 增长，分块后当前暂存只有 $B_q\times B_k$。所以 `test_working_set_proxy` 只断言分块元素数恒定，不声称 FLOPs 下降。

**第 9 步：Step 5 的 GPU 实验怎么设计。** 自变量只有"执行路径"：`_naive_attention` 显式物化分数矩阵，`SDPA` 调 `torch.nn.functional.scaled_dot_product_attention`；其余共用同一组 Q/K/V、`GPU_SEED=42` 和同样的 dtype。固定配置为 `GPU_BATCH_SIZE=1`、`GPU_NUM_HEADS=8`、`GPU_HEAD_DIM=64`、`GPU_SEQ_LENGTHS=[512,1024,2048,4096]`、`GPU_CAUSAL=False`；`GPU_WARMUP=5` 次预热不计，`GPU_ITERS=20` 次算 Median 和 P95。因变量分四类：

- 正确性：`output_close`（`atol=rtol=1e-2`）、`max_abs_error`、`relative_l2_error`（差向量范数 ÷ 基线范数）；
- 延迟：`median_ms`、`p95_ms`，用 CUDA Event 计时；
- 显存：`peak_allocated_mb`（实际分配）与 `peak_reserved_mb`（缓存池保留）；
- 执行路径：`_profile_attention_path` 过滤名字含 `attention` 或 `scaled_dot_product` 的算子，确认 SDPA 真的走了融合内核，而不是"接口叫 SDPA、底层回退稠密"。

由此算 `speedup = naive_median / sdpa_median` 和 `allocated_reduction = 1 - sdpa_peak_allocated / naive_peak_allocated`。OOM 会被捕获记成 `status: "OOM"`，不会让整轮实验崩掉。notebook 的 5.4 节记录了一组 RTX 5070 Ti、BF16、长度 512–4096 的参考结果：四档都通过数值检查，最大绝对误差约 `0.00195–0.00586`、相对 L2 误差约 `0.48%–0.50%`，加速比从 `2.13×` 增到 `6.18×`，已分配显存降幅从 `44.0%` 增到 `95.5%`，命中 `_scaled_dot_product_flash_attention` / `_flash_attention_forward`，决策为 `accept_for_mechanism_evidence`。这是 notebook 记录的参考运行，不是本讲解新跑出来的；本副本 5.2/5.3 没有保存执行输出，且 `RUN_MODE` 默认是 `dry_run`。

**第 10 步：结果怎么读。** 顺序不能反：先看 `failure` 和 `output_close`，数值不可比的长度档不能进入性能判断；再看 `speedup` 和 `allocated_reduction`；最后看 profiler 算子名。决策字段的取值共四种：没有可比较结果或正确性未全过 → `reject`；低长度通过、高长度 OOM → `tune_capacity_boundary`；正确性全过且至少一档有延迟或显存收益 → `accept_for_mechanism_evidence`；正确性过了但没有效率收益 → `tune`。`evidence_level` 固定为 `local_mechanism_benchmark`，不能外推成 FlashAttention-3/4 或服务吞吐。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 分数矩阵 $S=QK^T/\sqrt{D}$ | 每个查询对每个键的相关性得分，完整版有 $N^2$ 个元素 |
| 分块（tile） | 把 Q 和 K/V 各切一小段，只算这一小块局部分数 |
| 在线 Softmax | 边看新分块边维护分母，结果和完整 Softmax 相同 |
| $m_i$ | 每条查询行到目前为止的最大分数，用来防指数上溢 |
| $l_i$ | 每条查询行到目前为止的指数和，即未归一化分母 |
| $O_i$ | 每条查询行到目前为止的加权输出累计 |
| $\alpha$（`history_scale`） | 历史状态从旧基准搬到新基准的缩放因子 |
| $W_{ij}$（`tile_weights`） | 当前分块相对新基准的指数权重 |
| 尾块 | 序列长度不整除块大小时，最后一块更短 |
| 工作集 | 当前同时保留的中间数据量；分块把它从 $O(N^2)$ 压到 $O(B_qB_k)$ |
| SDPA | PyTorch 的 `scaled_dot_product_attention`，后端按设备/形状自动选内核 |
| Median / P95 | 20 次计时的中位数和第 95 百分位，后者更能反映抖动 |

## 小白常见疑问

**Q1：`m` 初始化为 `-inf`，`exp(-inf - (-inf))` 不会变成 NaN 吗？**
第一块到来时 `m_i=-inf`、`new_max` 是这一块的真实最大值（有限），所以 `history_scale=exp(-inf-finite)=0`，不会出现 `-inf-(-inf)`。只有"整行一个可见键都没有"时才会两者都是 `-inf`；本节非因果路径每行都有键、因果路径每行至少能看到自己，所以不会触发。`test_dtype_and_stability` 用全 100 的 Q/K 验证的正是"最大值基准把大分数压回 0 附近"。

**Q2：为什么不能每个分块各自 softmax，再按块平均？**
因为 Softmax 的分母是整行的指数和。第一块 2 个键、第二块 3 个键，各自归一化再平均，等于默认两块各占 1/2；完整 Softmax 里每个键的权重取决于它的分数和全行分母。Step 2 的原话是"同一 Q 行的归一化分母覆盖全部 K/V"，在线 Softmax 的 $\alpha$ 和 $l$ 就是在跨块维护这个共同分母。

**Q3：为什么 `history_scale` 要同时乘到 `l` 和 `O` 上？**
历史指数和与历史输出都按旧基准 $m_i$ 累计。换到新基准 $m_i'$ 后，每个历史指数都要乘 $\exp(m_i-m_i')$，所以"指数和"和"加权输出"必须乘同一个因子，否则分子分母不再对应同一个分布。notebook 的解析特意强调"只更新分母会破坏 Softmax 等价性"。

**Q4：`test_working_set_proxy` 能证明 FlashAttention 更快吗？**
不能，它只证明"局部分数的工作集不随序列长度平方增长"。测试固定 `block_size=8`，对 `seq_len=64/256/1024` 比较完整元素数 `N²` 和分块元素数 `64`，比值分别是 `64/1024/16384`；又对块大小 `4/8/16` 断言元素数为 `16/64/256`。真正的速度、显存和内核路径要看 Step 5，而且只在固定 GPU、dtype、形状下有意义。

## 动手验证：怎么知道自己做对了

`run_flash_attention_sim_tests` 依次跑五个测试，全过才打印 `✅ FlashAttention CPU 机制测试通过：数值等价、因果约束、稳定性、工作集代理与输入契约均符合预期。`；没填 TODO 会先打印"请先完成 TODO 部分的代码！"并抛出 `NotImplementedError`。

- `test_noncausal_equivalence`：三组 `(seq_len, dim, block_size, seed)` 为 `(8,4,2,42)`、`(5,3,3,7)`、`(3,2,1,123)`，其中 `(5,3,3)` 含尾块；断言与 `_reference_attention` 的最大绝对差 `< 1e-5`。
- `test_causal_equivalence`：固定种子 11，`q/k/v` 都是 `6×4`，块大小 2；断言因果分块结果与完整因果注意力 `allclose(atol=1e-5, rtol=1e-5)`。
- `test_dtype_and_stability`：`float64` 的 `4×3` 输入，断言输出 `dtype == torch.float64`、与参考在 `atol=1e-10, rtol=1e-10` 下一致；再把 Q/K 全设成 100.0，断言输出全部有限——没有减最大值的实现会在这里得到 `inf` 或 `nan`。
- `test_working_set_proxy`：断言 `tile_score_elements == 64`、小于完整元素数，且 `full // tile` 依次为 `[64, 1024, 16384]`；块大小 4/8/16 对应 `[16, 64, 256]`。
- `test_input_contract`：`block_size=0` 必须抛 `ValueError`。

函数入口还检查形状、device、dtype：非二维、形状不一致、device 不同、dtype 不同、`block_size <= 0` 都会在计算前报错。notebook 没有保存执行输出，判断标准就是五个测试不抛异常并打印最后那行 ✅。
