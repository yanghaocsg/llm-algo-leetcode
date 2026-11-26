# PAR-CONTEXT. 上下文与序列并行 — 小白讲解

> 对应 notebook：[`PAR-CONTEXT_Context_and_Sequence_Parallelism.ipynb`](./PAR-CONTEXT_Context_and_Sequence_Parallelism.ipynb)

## 一句话总结
上下文并行（Context Parallelism，CP）沿序列维把词元分给多个进程，每个进程只保存一段 Query 以及对应的 K/V 分片；但本地 Query 仍要访问全部历史 K/V，所以要么轮转分片，要么先收集。本节用四个 TODO 实现“连续等长切分”“局部 Softmax 统计”“跨分片在线合并”“Ring 发送字节估算”，并用 dense Attention 做数值等价检查；最后给出双卡 All-Gather 的 GPU 实验口径：只算本地 Q 对全量 K/V 的注意力，比较“K/V 已就绪只计算”与“先收集再计算”的耗时差。关键结论是：局部 Softmax 输出不能直接平均，必须先对齐最大值再合并分子/分母；Ring 单 rank 的发送量是 $(P-1)\times 2(S/P)Db$ 字节，而字节数不等于实际通信时间。

## 为什么需要它：痛点是什么
序列变长时，单卡要保存整段序列的 K/V 和注意力中间量；张量并行（TP）切的是注意力头或特征维，切完每个进程仍然持有整条序列，长上下文压力并没有消失。上下文并行换了一个切法：每个进程拿一段词元，存这段的 Q/K/V。但注意力是跨词元的——本地 Query 需要看到所有位置的 K/V，于是“计算本地”与“需要远端数据”同时存在。

生活类比：一本很长的账本分给几个人同时核对。每个人只拿自己那几页（本地 Query），但每一页的结论都要参考整本账（全局 K/V）。如果各自算完就把结论平均，会错：有的人那几页很重要（分母大），有的人不重要（分母小），平均等于把权重拉平。正确做法是每个人交回“本段最大值、指数和、加权和”三个统计量，汇总时先统一尺度再加权，最后才归一化。

## 核心问题
- CP 和 TP 到底切的是什么？为什么 CP 切完还需要通信，而 TP 也需要？
- 为什么不能把各分片的 Softmax 输出直接平均？局部最大值不同时怎么对齐？
- 在线 Softmax 的 `(m, ℓ, u)` 三个量分别是什么？合并公式怎么推出来？
- Ring Attention 每轮发多少字节、总共几轮？`cp_size=1` 时为什么是 0？
- GPU 实验里 `local_kv_bytes_per_rank`、`replicated_kv_bytes_per_rank`、`kv_exchange_bytes_per_rank`、`exposed_exchange_ms_proxy` 各是什么口径？
- 为什么“字节估算”不能直接当成“通信时间”或“峰值显存下降”？

## 解决思路：一步步看懂

**第 1 步：TP 与 CP 的分工。** notebook 的 Step 1 表写得很清楚：TP 每个进程主要持有“部分注意力头或特征”，主要通信用 All-Reduce / Reduce-Scatter，解决的是单层权重与矩阵计算的压力；CP 每个进程持有“一段序列的词元及其状态”，主要通信用 K/V 点对点轮转或集合收集，解决的是长上下文状态与注意力计算的压力；TP×CP 则是“部分头 × 部分序列”，两个进程组分别通信。注意“序列并行（SP）”这个词在不同框架里含义不同：Megatron 的 SP 主要拆 LayerNorm、Dropout 等激活，CP 才沿序列拆网络输入与各层激活，并处理注意力的跨分片依赖。本节讲的是后者。

**第 2 步：TODO 1——连续等长切分。** `split_sequence_contiguous(tensor, cp_size)` 先校验 `cp_size > 0`，再取 `sequence_length = tensor.shape[-2]`，要求它能被 `cp_size` 整除，否则抛 `ValueError('序列长度必须能被 cp_size 整除')`。参考实现：

```python
local_sequence_length = sequence_length // cp_size
return list(torch.split(tensor, local_sequence_length, dim=-2))
```

为什么切倒数第二维？因为注意力的张量形状通常是 `(..., seq, head_dim)`，最后一维是每个头的特征宽度，倒数第二维才是序列。`torch.split` 返回的是一组视图（tuple），`list(...)` 只是转成 list，方便按 rank 取。测试 `test_sequence_partition_contract` 用 `torch.arange(24).reshape(1, 6, 4)`、`cp_size=3`，断言三个分片的 `shape[-2]` 都是 2，并且 `torch.cat(shards, dim=-2)` 能逐元素恢复原张量——这就是“不重不漏、顺序不变”。再用 `cp_size=4` 触发不能整除的 `ValueError`。

**第 3 步：TODO 2——局部 Softmax 统计。** 对每个 K/V 分片，先算 `scores = Q @ K_shard^T * scale`，其中 `scale = 1/sqrt(query.shape[-1])`。然后：

```python
local_max = scores.max(dim=-1, keepdim=True).values
local_exp = torch.exp(scores - local_max)
local_denominator = local_exp.sum(dim=-1, keepdim=True)
local_numerator = torch.matmul(local_exp, value_shard)
```

三个量的含义：`local_max` 是本分片每一行（每个 Query 位置）的最大分数，用来数值稳定；`local_denominator` 是减完最大值后的指数和，对应 Softmax 的分母 $\ell_i$；`local_numerator` 是“指数权重 × Value”的和，对应 $u_i$。注意这里**不除以分母**，也不输出最终结果：局部归一化会把不同分片的尺度固化，后面无法正确合并。`keepdim=True` 让分母保持 `(..., seq_q, 1)`，方便广播相除。

**第 4 步：TODO 3——在线合并。** 设已累计的统计为 `(running_max, running_denominator, running_numerator)`，新分片为 `(local_max, local_denominator, local_numerator)`。先取

```python
merged_max = torch.maximum(running_max, local_max)
running_scale = torch.exp(running_max - merged_max)
local_scale = torch.exp(local_max - merged_max)
```

因为 `merged_max` 是两者较大值，所以两个 scale 都在 $(0,1]$，指数不会上溢。然后把两侧的分子、分母都乘上各自的 scale 再相加：

$$\ell=\ell_{\text{run}}e^{m_{\text{run}}-m}+\ell_{\text{local}}e^{m_{\text{local}}-m},\qquad
u=u_{\text{run}}e^{m_{\text{run}}-m}+u_{\text{local}}e^{m_{\text{local}}-m}.$$

为什么必须这样？假设两个分片的最大值相同，分母分别是 1 和 3，那么全局分母是 4，两个分片对结果的权重是 $1/4$ 和 $3/4$，不是各一半。直接平均局部输出会把权重变成 $1/2$ 和 $1/2$，结果错误。最大值不同时更糟：分片 A 的 $e^{m_A}$ 和分片 B 的 $e^{m_B}$ 不在同一个尺度上，必须先乘 $e^{m_i-m}$ 对齐。notebook 的 Step 2 用“分母 1 和 3”这个例子专门说明这一点。

**第 5 步：把四个函数串起来。** `context_parallel_attention_sim(query, key, value, cp_size)` 先切 K/V，再对每个分片算 `scores` 和局部统计，用 `merge_softmax_statistics` 累积，循环结束后才 `numerator / denominator`。这个顺序对应 Ring Attention 的整圈：本地 Q 不动，每收到一块 K/V 就算一次局部分数并更新累计统计，所有分片处理完才归一化。测试 `test_context_parallel_attention_equivalence` 用 `torch.manual_seed(7)` 生成 `query (2,4,8)`、`key (2,8,8)`、`value (2,8,6)`、`cp_size=4`，先算 dense 的 `softmax(QK^T/sqrt(8)) @ V`，再算分块版本，要求 `torch.allclose(..., atol=1e-5)`。`test_online_softmax_stability` 更狠：`query=[[[100.0, 0.0]]]`，分数很大，直接 `exp` 会溢出；因为先减最大值，结果仍然有限，并且与 dense 一致。

**第 6 步：TODO 4——Ring 发送字节。** Step 3 的公式是

$$B_{\mathrm{round}}=2\frac{S}{P}Db,\qquad B_{\mathrm{rank}}=(P-1)B_{\mathrm{round}},$$

其中 $S$ 是全局序列长度，$P$ 是并行进程数，$D$ 是一个位置全部 KV 头的合计宽度（MHA 里可等于隐藏维，GQA 里按 KV 头数算），$b$ 是每元素字节数，系数 2 表示 Key 和 Value 各一份。参考实现：

```python
local_sequence_length = sequence_length // cp_size
rounds = cp_size - 1
payload_per_round_bytes = 2 * local_sequence_length * hidden_size * dtype_bytes
total_payload_per_rank_bytes = rounds * payload_per_round_bytes
```

为什么是 $P-1$ 轮？Ring 里每个进程要把自己的 K/V 分片传给下一个进程，转一圈回到自己之前，其他 $P-1$ 个分片都经过它一次。测试 `test_ring_communication_ledger` 用 `(1024, 4096, 2, cp_size=4)`：本地长度 $1024/4=256$，轮数 3，每轮 $2\times256\times4096\times2=4{,}194{,}304$ 字节（4 MiB），单 rank 累计 $3\times4{,}194{,}304=12{,}582{,}912$ 字节（12 MiB）。`cp_size=1` 时轮数 $1-1=0$，累计 0，因为不需要远端交换。函数返回的字典键是 `local_sequence_length`、`rounds`、`payload_per_round_bytes`、`total_payload_per_rank_bytes`。

**第 7 步：框架一致性核对。** 这一段用 PyTorch 的 Gloo 后端启动两个 CPU 进程，实际做 All-Gather：每个 rank 把本地 K/V 分片 `dist.all_gather` 到两个缓冲区，`torch.cat` 成全局 K/V，用 `F.scaled_dot_product_attention(query, gathered_key, gathered_value)` 算参考输出，再用 `context_parallel_attention_sim(query, gathered_key, gathered_value, cp_size=world_size)` 算分块输出。检查项：`max_abs_error < 1e-10`（float64 下）、输出形状 `[1,2,4,8]`（全局序列 8、两个 rank 各 4）、以及独立的四分片账本 `estimate_ring_kv_communication(8, 8, 8, 4)`：本地长度 2、轮数 3、每轮 $2\times2\times8\times8=256$ 字节、累计 $3\times256=768$ 字节。注意这里的 `hidden_size=8` 是“单个位置全部 KV 头的合计宽度”，不是模型隐藏维；参数名沿用 `hidden_size` 容易误会，notebook 在 Step 4 专门解释了。

**第 8 步：GPU 实验设计。** 固定 `WORLD_SIZE=2`、`SEQ_LEN=2048`、`NUM_HEADS=8`、`HEAD_DIM=64`、`DTYPE='bfloat16'`、`WARMUP=3`、`REPEATS=10`，用 `torchrun --standalone --nproc_per_node 2` 调 `tools/run_parallel_mechanism_benchmark.py --mode cp_attention`。它比较两条路径：一条是“全量 K/V 已准备，仅计算注意力”，另一条是“先 All-Gather 收集 K/V，再计算注意力”。两条路径都保留全量 K/V 缓冲区，输出误差和两段耗时之差。指标口径要分清：
- `max_abs_error`：分块/收集后的注意力与参考实现的数值误差。
- `local_kv_bytes_per_rank`：每个 rank 本地那一段 K/V 的字节数。
- `replicated_kv_bytes_per_rank`：每个 rank 持有全量 K/V 副本时的字节数。
- `kv_exchange_bytes_per_rank`：按 5.4 表的说明，这个字段实际保存的是“一份本地 K/V 分片大小”，不是 All-Gather 的总传输量。
- `exposed_exchange_ms_proxy`：两条路径最慢进程平均耗时之差，是“加入收集后的附加耗时代理”，不是时间线上直接测到的等待。

按本节公式和固定配置可以手算一个参考：单 rank 本地序列 $2048/2=1024$，每个位置的 KV 宽度 $8\times64=512$，bf16 每元素 2 字节，本地 K/V 约 $2\times1024\times512\times2=2{,}097{,}152$ 字节（2 MiB），全量副本约 $2\times2048\times512\times2=4{,}194{,}304$ 字节（4 MiB）。notebook 没有保存这一步的运行结果，字段值要等你实际跑完 `real_gpu` 才会有；上面的数字只是按公式代入，不是实测。

**第 9 步：结果怎么读、不能读什么。** 先看 `failure` 与 `max_abs_error`，确认切分/拼接顺序没弄反；再比“仅计算”与“收集后计算”的同条件耗时。如果附加耗时很大，可能是 All-Gather 的同步成本；但 `exposed_exchange_ms_proxy` 只是两条路径耗时差，不能证明通信与计算没有重叠。字节字段是张量形状估算，两条路径都保留全量 K/V，不能据此说峰值显存下降了；真实峰值要另测。要评估 Ring 的实际性能，得用轮转内核和时间线；要评估容量，得测峰值显存。notebook 的 5.4 表把这些“下一步”逐项写了出来。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 上下文并行（CP） | 沿序列维把词元分给不同进程，每个进程存一段 Q/K/V |
| 张量并行（TP） | 切注意力头或特征维，每个进程仍持有整条序列 |
| 序列并行（SP） | 框架相关叫法，Megatron 里主要拆 LayerNorm/Dropout 等激活 |
| Ring Attention | K/V 分片沿进程环轮转，本地 Q 不动，逐块更新统计 |
| 在线 Softmax | 不一次性看全部分数，用 $(m,\ell,u)$ 增量合并 |
| 局部最大值 $m_i$ | 本分片每行的最大分数，用于数值稳定 |
| 局部分母 $\ell_i$ | 减最大值后的指数和，Softmax 的分母 |
| 局部分子 $u_i$ | 指数权重乘 Value 的和，未归一化 |
| 合并尺度 $e^{m_i-m}$ | 把不同分片的指数尺度对齐到全局最大值 |
| All-Gather | 每个进程收集所有分片，得到全量 K/V |
| payload / rounds | 每轮发送的字节数 / 远端交换轮数 $P-1$ |
| 暴露耗时代理 | 两条路径耗时差，不等于时间线上的真实等待 |

## 小白常见疑问

**Q1：为什么不能把每个分片算出的 `softmax(scores) @ V` 直接平均？**
因为每个分片的 Softmax 分母不同。notebook 的例子：两个分片最大值相同、分母分别是 1 和 3，全局分母是 4，正确权重是 $1/4$ 和 $3/4$；直接平均会变成 $1/2$ 和 $1/2$。如果最大值不同，还要先把指数尺度对齐。正确做法是保留未归一化的分子和分母，合并完再除。

**Q2：`local_softmax_statistics` 里为什么要先减 `local_max`？**
防止 `exp` 上溢。测试 `test_online_softmax_stability` 用 `query=[[[100.0, 0.0]]]`，分数很大；如果直接 `exp(scores)` 会溢出成 `inf`，先减每行最大值后，最大的指数是 $e^0=1$，其余都小于等于 1，结果有限且与 dense 一致。

**Q3：`merge_softmax_statistics` 里 `running_scale` 和 `local_scale` 为什么不会大于 1？**
因为 `merged_max = max(running_max, local_max)`，所以 `running_max - merged_max <= 0`、`local_max - merged_max <= 0`，两个 `exp` 的结果都在 $(0,1]$。这保证了合并时不会因为“把旧统计放大”而溢出。反过来，如果用了 `min` 或忘了取最大，就会出现 $e^{正数}$。

**Q4：GPU 实验里 `kv_exchange_bytes_per_rank` 为什么不是 All-Gather 的总传输量？**
notebook 的 5.4 表明确写：当前字段实际保存“一份本地 K/V 分片大小”，不是 All-Gather 总传输量。All-Gather 的传输量要按算法和实现核对（比如 ring 还是 tree），不能拿这个字段乘进程数就当成真实网络流量。同理，`exposed_exchange_ms_proxy` 是耗时差代理，不是通信等待。

**Q5：切了序列，显存一定下降吗？**
不一定。本节的两条 GPU 路径都保留全量 K/V 缓冲区，`local_kv_bytes_per_rank` 和 `replicated_kv_bytes_per_rank` 只是不同布局的字节估算；要得出“峰值显存下降”的结论，必须另测峰值。CP 的容量收益取决于实现是否真的只保留分片、是否把收集缓冲也算进去。

## 动手验证：怎么知道自己做对了

测试区有 4 个函数，直接调用，TODO 未填会抛 `NotImplementedError`，全过才打印 `✅ 序列分片、在线 Softmax 与 Ring 发送字节估算验证通过。`

- `test_sequence_partition_contract`：`torch.arange(24).reshape(1,6,4)` 用 `cp_size=3` 切出的三个分片 `shape[-2]` 都是 2；`torch.cat(shards, dim=-2)` 必须 `torch.equal` 原张量；`cp_size=4` 必须抛 `ValueError`。
- `test_context_parallel_attention_equivalence`：`seed=7`，`query (2,4,8)`、`key (2,8,8)`、`value (2,8,6)`、`cp_size=4`；分块结果与 dense `softmax(QK^T/√8) @ V` 满足 `torch.allclose(..., atol=1e-5)`。
- `test_online_softmax_stability`：大幅值 `query=[[[100.0, 0.0]]]`、4 个 key、`cp_size=2`；结果必须 `torch.isfinite(...).all()`，且与 dense 在 `atol=1e-5` 内一致。
- `test_ring_communication_ledger`：`(1024,4096,2,cp_size=1)` 的 `rounds==0`、`total==0`；`cp_size=4` 的 `local_sequence_length==256`、`rounds==3`、`payload_per_round_bytes == 2*256*4096*2`、`total == rounds * payload`。

框架核对 `check_context_parallel_framework_conformance` 额外要求：两个进程的 `max_abs_error < 1e-10`、输出形状 `[1,2,4,8]`、`estimate_ring_kv_communication(8,8,8,4)` 的 `rounds==3`、`payload_per_round_bytes == 2*2*8*8 == 256`、`total == 3*2*2*8*8 == 768`。GPU 部分的 `5.3` 读取单元只检查 JSON 是否含 `workload`、`hardware`、`metrics`、`evidence_level`、`failure`、`decision` 六个字段；notebook 没有保存这一步的输出，也没有 `real_gpu` 结果，所以以代码、公式和测试断言为准，别把上面的手算字节当成实测值。
