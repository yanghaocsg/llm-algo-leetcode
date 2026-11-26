# 29. Tensor 并行模拟 — 小白讲解

> 对应 notebook：[`29_Tensor_Parallelism_Sim.ipynb`](./29_Tensor_Parallelism_Sim.ipynb)

## 一句话总结
这一节用纯 CPU 的矩阵切片模拟 Tensor Parallelism（TP）：`tensor_parallel_column_sim` 沿输出维切权重再拼回（TODO 1），`tensor_parallel_row_sim` 沿输入维配对切分再求和（TODO 2），`tensor_parallel_mlp_sim` 把"Column → ReLU → Row"配成一条中间 activation 不落地的链路（TODO 3），`plan_attention_tp` 为 GQA 的 Query/KV Head 规划分片或复制（TODO 4）。结论：TP 的正确性靠"切分维对齐 + 在正确位置做 All-Gather / All-Reduce"，性能则取决于你是否保住了布局连续性。

## 为什么需要它：痛点是什么
单卡放不下一个 Linear 时，直觉是"把权重切成两块，各算一半再拼起来"。但切法有两种，通信位置完全不同：沿输出维切，每张卡算出输出的一部分，各卡结果拼起来就是完整输出；沿输入维切，每张卡只拿到输入的一部分，算出来的是"部分和"，必须相加才是完整输出。初学者常把这两种的恢复方式弄反——该拼的地方去加、该加的地方去拼，数值看起来形状对，结果却全错。

生活类比：两个人合写一本词典。Column Parallel 像按字母分册：A–M 一个人写、N–Z 一个人写，最后把两册并排放在书架上（拼接），内容不重叠。Row Parallel 像两个人各自写"同一批词条的一部分解释"，最后要把两份解释合并成一份（求和）。分册可以并排，解释必须合并；搞混就得到两本残缺的词典。

## 核心问题
- Column Parallel 和 Row Parallel 分别切哪个维度？各自局部结果的形状是什么？
- 为什么 Column 路径用 `torch.cat` 恢复，Row 路径必须用求和/All-Reduce？
- 配对 MLP 里，第二层为什么能直接消费第一层的 hidden 分片，而不需要中间 All-Gather？
- GQA 的 Query Head 和 KV Head 在 TP 下分别怎么处理？什么配置必须拒绝？
- 复制 KV Head 的代价是什么？它和切序列长度的 Context Parallel 有什么区别？
- GPU benchmark 里 `max_abs_error`、末端 payload、`max_rank_step_ms` 该怎么一起读？

## 解决思路：一步步看懂

**第 1 步：两种布局的数学。** 设输入 $X$ 形状 `(batch, in_dim)`、权重 $A$ 形状 `(in_dim, out_dim)`、输出 $Y=XA$ 形状 `(batch, out_dim)`，$P$ 是 TP 组进程数。

Column Parallel 把输出维切成 $A=[A_1,\ldots,A_P]$，每个 rank 算 $Y_i=XA_i$，再沿输出维拼成 $Y=[Y_1,\ldots,Y_P]$。每张卡都需要完整的 $X$，但只持有 $1/P$ 的输出列；如果下游正好要按输出维分片消费（比如 MLP 的 hidden），这次拼接可以推迟。

Row Parallel 把输入维切成 $X=[X_1,\ldots,X_P]$、权重按输入维切成 $A=[A_1^{\mathsf T},\ldots,A_P^{\mathsf T}]^{\mathsf T}$，每个 rank 算 $X_iA_i$，完整输出是

$$
Y=XA=\sum_{i=1}^{P}X_iA_i.
$$

所以末端必须求和（All-Reduce Sum），不能拼接——每个 $X_iA_i$ 都是 `(batch, out_dim)` 的完整形状，只是"贡献了一部分"。

**第 2 步：TODO 1 —— Column 的三行。** 骨架先断言 `out_features % num_gpus == 0`，参考答案：

```python
weight_chunks = torch.chunk(A, num_gpus, dim=1)          # 沿输出维切
local_outputs = [X @ weight_chunk for weight_chunk in weight_chunks]
Y_tp = torch.cat(local_outputs, dim=-1)                  # 沿输出维拼回
```

`torch.chunk(A, P, dim=1)` 把 `(8, 12)` 切成两个 `(8, 6)`；每个 rank 拿完整 `X`（`(2, 8)`）乘自己的 `(8, 6)`，得到 `(2, 6)`；`cat(dim=-1)` 恢复 `(2, 12)`。`test_column_parallel_contract` 用 `torch.manual_seed(42)`、`X=(2,8)`、`A=(8,12)`，断言与 `X @ A` 在默认容差内 `allclose`。容易写错的是 `dim=0` 切权重（那会切输入维，变成 Row 的切法）或最后 `sum` 而不是 `cat`。

**第 3 步：TODO 2 —— Row 必须切两样。** 断言 `in_features % num_gpus == 0` 后：

```python
input_chunks = torch.chunk(X, num_gpus, dim=1)    # 切输入特征
weight_chunks = torch.chunk(A, num_gpus, dim=0)   # 切权重输入维
partial_outputs = [x_chunk @ weight_chunk for x_chunk, weight_chunk in zip(input_chunks, weight_chunks)]
Y_tp = torch.stack(partial_outputs, dim=0).sum(dim=0)   # All-Reduce(Sum)
```

`X=(2,8)` 切成两个 `(2,4)`，`A=(8,12)` 切成两个 `(4,12)`，每个部分和都是 `(2,12)`，`stack` 成 `(2,2,12)` 后 `sum(dim=0)` 恢复 `(2,12)`。关键约束是**两边必须覆盖同一段输入范围**：`input_chunks[0]` 对应第 0–3 列，`weight_chunks[0]` 就必须对应第 0–3 行；用 `zip` 是按顺序配对，不能一个正序一个倒序。测试用 `atol=1e-5`，因为浮点求和顺序和 dense 的单个 GEMM 不完全一样。

**第 4 步：TODO 3 —— 配对 MLP 的"中间不拼接"。** 两层 MLP 是 $Y=\mathrm{ReLU}(XW_1)W_2$，`W1` 形状 `(in, hidden)`、`W2` 形状 `(hidden, out)`。骨架按 `hidden_per_rank = hidden_dim // num_gpus` 分 rank 循环，参考答案：

```python
w1_local = W1[:, start:end]          # 第一层输出维的一段
hidden_local = torch.relu(X @ w1_local)   # 只算这段 hidden
w2_local = W2[start:end, :]          # 第二层输入维的同一段
partial_output = hidden_local @ w2_local
```

最后 `torch.stack(partial_outputs, dim=0).sum(dim=0)` 模拟末端 All-Reduce。这里最妙的是 `start/end` 对两层是同一组下标：第一层第 `start:end` 列产生的 hidden，正好是第二层第 `start:end` 行要消费的输入。因此 `hidden_local` 从头到尾没被拼成完整 hidden，通信只发生在末端。`test_paired_tensor_parallel_mlp_contract` 用 `(3,8)` 输入、`W1=(8,16)`、`W2=(16,8)`、`num_gpus=2`，断言与 dense `relu(X @ W1) @ W2` 形状相同且 `allclose(atol=1e-5)`；再用 `W1[:, :15]`、`W2[:15]` 制造 hidden 不整除，断言抛 `ValueError`。

**第 5 步：TODO 4 —— GQA 的 Head 映射。** 先检查 `num_q_heads % tp_size == 0`，得到 `q_heads_per_rank = num_q_heads // tp_size`。然后分三种情况：

```python
if num_kv_heads % tp_size == 0:
    kv_plan = ('shard', num_kv_heads // tp_size, 1)
elif tp_size % num_kv_heads == 0:
    kv_plan = ('replicate', 1, tp_size // num_kv_heads)
else:
    raise ValueError('KV Head 无法在 TP rank 间均匀分片或复制')
```

读法：KV Head 够多（8 个、4 个 rank），每个 rank 分 2 个，`kv_replication=1`；KV Head 比 rank 少（2 个、4 个 rank），每个 rank 至少要有 1 个 KV Head，于是每个 KV Head 被复制 2 份，`kv_mode='replicate'`、`kv_replication=2`；如果两个方向都除不尽（比如 6 个 KV Head、4 个 rank，或 8 个 KV Head、3 个 rank），就没有稳定的一对多映射，必须在执行前失败。`test_attention_tp_head_mapping_contract` 分别断言 `32/8/4 -> shard, 2, 1` 和 `32/2/4 -> replicate, 1, 2`；`test_attention_tp_invalid_mapping` 用 `(30,8,4)`（Q 不整除）、`(32,6,4)`（KV 两个方向都除不尽）、`(32,8,0)`（零 rank）确认都抛 `ValueError`。

**第 6 步：为什么"布局连续"比"少算一点"更重要。** Step 3 的表格把代价写得很清楚：Column 的代价是输入复制和可选输出聚合；Row 的代价是输出归约；配对 MLP 如果中途恢复完整 hidden，就凭空多一次 All-Gather。Attention 同理：Q/K/V 投影按 Head 切，各 rank 本地做 Attention，输出投影再沿输入 Head 维切、末端 All-Reduce。TP 切的是 Head/特征维，不是序列长度——那是 Context Parallelism 的事。GQA 里复制 KV Head 虽然满足映射，却会让每个 rank 都存一份 KV 状态，显存收益变差。

**第 7 步：可选 DTensor 核对。** `_dtensor_framework_worker` 在 2 个 Gloo 进程里把 `torch.arange(24).reshape(4,6)` 用 `Shard(1)` 分布，检查每个 rank 的本地分片等于 `full.chunk(2, dim=1)[rank]`；再用 `redistribute(..., [Replicate()])` 恢复完整张量；最后 `all_gather` 后 `cat(dim=1)` 重建。三个检查项是 `local_shards_match`、`replicated_matches_full`、`shards_reconstruct_full`。它核对的是 DTensor 的布局语义，不测 NCCL 性能。

**第 8 步：可选 GPU benchmark 怎么读。** 固定 workload 是 `BATCH_SIZE=8`、`IN_DIM=4096`、`HIDDEN_DIM=11008`、`OUT_DIM=4096`、`DTYPE='bf16'`、`WORLD_SIZE=2`、`WARMUP=5`、`REPEATS=20`，跑 `tools/run_parallel_mechanism_benchmark.py --mode tp_mlp`。读结果先看 `max_abs_error`：正确性没过就停止解释性能。然后看每 rank 峰值显存是否真的降到约一半左右、`all_reduce_payload_bytes_per_rank` 是否等于"batch × out_dim × dtype 字节数"，再看 `max_rank_step_ms` 和 `tokens_per_s_proxy` 判断层内同步有没有吞掉局部计算收益。notebook 里这些字段的"当前结果"都写着"运行后填写"，默认 `RUN_MODE='dry_run'`，不会真的占 GPU。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Tensor Parallelism（TP） | 把同一层的权重/计算切到多张卡上共同完成 |
| Column Parallel | 沿权重输出维切，每卡算输出的一部分，需要时拼接 |
| Row Parallel | 沿输入维切输入和权重，每卡算部分和，末端求和 |
| All-Gather | 把各卡的分片收集起来拼成完整张量 |
| All-Reduce (Sum) | 把各卡的同形状张量按元素相加，结果广播给所有卡 |
| 布局连续性 | 上游的分片布局正好是下游能直接消费的布局 |
| hidden 分片 | 配对 MLP 中间层没有被拼回完整的激活 |
| GQA | Query 多、KV 少的注意力，KV Head 需要特殊映射 |
| KV 复制 | KV Head 少于 TP rank 时，每个 KV Head 被多个 rank 各存一份 |
| `Shard(1)` / `Replicate()` | DTensor 的两种放置方式：按第 1 维切 / 每卡一份完整副本 |
| payload | 一次集合通信实际搬运的字节数 |

## 小白常见疑问
**Q1：Column Parallel 每张卡都要完整输入，那不是没省显存吗？**
省的是权重和输出激活，不是输入。`X` 在每张卡上都有一份（输入复制），但权重只存 `1/P` 的输出列，局部输出也只有 `1/P` 宽。真正吃显存的大头通常是权重和中间激活，所以仍然划算；如果输入本身也巨大，就要结合序列并行等其他手段。

**Q2：Row Parallel 末端为什么不能用 `cat`？**
因为每个 `X_iA_i` 都是 `(batch, out_dim)` 的完整形状，只是数值上是总输出的一部分。`cat` 会得到 `(batch, P*out_dim)`，形状就不对了；必须 `sum` 才是 $Y=\sum_i X_iA_i$。`test_row_parallel_contract` 用 `allclose(X @ A)` 检查的就是这一点。

**Q3：配对 MLP 中间不拼接，数值上为什么还能对？**
因为 ReLU 是逐元素函数，$\mathrm{ReLU}([H_1,H_2])=[\mathrm{ReLU}(H_1),\mathrm{ReLU}(H_2)]$，先切后激活和先激活后切一样。第二层按同一段 hidden 行切分，$H_1W_2^{(1)}+H_2W_2^{(2)}$ 正好等于 $HW_2$。这就是"Column 输出直接喂给匹配的 Row 输入"的代数依据。

**Q4：为什么 `32/6/4` 这种配置要直接报错，不能凑合？**
`6 % 4 != 0`，KV Head 不能平均分给 4 个 rank；`4 % 6 != 0`，也没法让每个 rank 至少拿一个再均匀复制。硬凑会导致有的 rank 多一个 Head、有的少一个，本地 Attention 工作量和 KV 布局都不一致。`plan_attention_tp` 选择显式 `ValueError`，把问题挡在执行前。

**Q5：复制 KV Head 会增加多少开销？**
每个 rank 都存一份完整的被复制 KV Head。`32/2/4` 时 `kv_replication=2`，相当于这 2 个 KV Head 在 4 个 rank 上各有一份（每个 rank 1 个、每个 Head 被 2 个 rank 持有）。映射简单了，但 KV cache 的占用比"能分片"时高；notebook 的 Step 2 明确说复制"能满足映射约束，却会增加 KV 状态占用"。

## 动手验证：怎么知道自己做对了
`test_column_parallel_contract` 和 `test_row_parallel_contract` 用 `(2,8) @ (8,12)`、`num_gpus=2` 分别与 `X @ A` 做 `allclose`（Row 显式 `atol=1e-5`）。`test_tensor_parallel_shape_contract` 断言两条路径输出都是 `(2,12)`。`test_tensor_parallel_dimension_contract` 用输出维 11、输入维 7 确认不可整除时抛 `AssertionError`。`test_paired_tensor_parallel_mlp_contract` 用 `(3,8)`、`W1=(8,16)`、`W2=(16,8)` 与 dense 比较并检查 hidden 15 不可整除时抛 `ValueError`。`test_attention_tp_head_mapping_contract` 断言两组完整字典：`{'q_heads_per_rank':8,'kv_mode':'shard','kv_heads_per_rank':2,'kv_replication':1}` 和 `{'q_heads_per_rank':8,'kv_mode':'replicate','kv_heads_per_rank':1,'kv_replication':2}`。`test_attention_tp_invalid_mapping` 确认三组非法配置都抛 `ValueError`。notebook 没有保存执行输出，测试通过与否以这些断言为准；DTensor 的 `framework_check` 和 GPU JSON 需要你在对应环境自行运行，默认 `dry_run` 只打印配置。
