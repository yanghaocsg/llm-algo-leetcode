# 44. 局部、滑动窗口与稀疏注意力 — 小白讲解

> 对应 notebook：[`44_Local_Sliding_and_Sparse_Attention.ipynb`](./44_Local_Sliding_and_Sparse_Attention.ipynb)

## 一句话总结
本节不改 Q/K/V 的语义，只改"每个 Query 允许读哪些 Key"：用四个 TODO 分别构造局部因果掩码、块稀疏掩码、连接账本和局部窗口的最少传播层数，再用一组配对 GPU 微基准比较稠密因果 SDPA 与显式布尔滑动窗口掩码。CPU 测试确认访问语义正确；已保存的 GPU 结果则说明"连接数减少"并不自动等于"更快"——四档长度里显式掩码都更慢、峰值显存更高，决策停在 `investigate_mask_backend`。

## 为什么需要它：痛点是什么
标准因果注意力让第 $t$ 个 token 读取从 0 到 $t$ 的全部历史，连接数是 $L(L+1)/2$。序列一长，分数矩阵和访存都按平方增长，于是出现各种"只看附近"的方案。初学者最容易把三件事混为一谈：掩码里 `True` 变少了、理论连接数变少了、实际运行一定更快。前两者是语义和账本，最后一个是执行问题，中间隔着掩码表示、内核实现和后端选择。

生活类比：一间教室里的传纸条规则。稠密注意力是每个人可以读前面所有人的笔记；滑动窗口是只能读最近几排的笔记；块稀疏是老师额外指定"第 3 排可以读第 1 排"这类远程通道。规则改了只说明"允许读什么"，不代表传纸条的速度一定变快——如果每个人还要先领一张和全班人数一样大的"可读名单表"，光发表格就可能更慢。

## 核心问题
- 局部因果掩码要同时满足哪两个条件？只满足一个会泄漏什么？
- 块稀疏掩码里，块对、因果过滤和 `np.tri` 各负责哪一层？
- 连接账本 `visible_pairs / total_pairs` 能说明什么，不能说明什么？
- 局部窗口下，距离为 $d$ 的信息至少要走几层才能到达当前 token？
- 为什么显式布尔窗口掩码在连接数减少 76.57% 时反而更慢？结果 JSON 该怎么读？

## 解决思路：一步步看懂

**第 1 步：TODO 1，局部因果掩码。** `build_local_causal_mask(seq_len, window_size)` 先校验两个参数都是正整数，否则 `ValueError`。参考答案两行：

```python
query, key = np.indices((seq_len, seq_len))       # 行是 Query 位置，列是 Key 位置
mask = (key <= query) & ((query - key) < window_size)
```

`np.indices` 生成两个 `[L, L]` 矩阵：`query[i, j] = i`、`key[i, j] = j`。`key <= query` 是因果条件，禁止看未来；`query - key < window_size` 是窗口条件，只能看最近 `window_size` 个位置（包括自己）。两个条件用 `&` 逐元素与，缺一不可：只留因果会退化成稠密下三角，只留窗口会让每个 token 看到未来一个窗口。测试 `build_local_causal_mask(5, 2)` 断言 `mask[3,3]`、`mask[3,2]` 为真，`mask[3,1]`（窗口外）和 `mask[2,3]`（未来）为假。

**第 2 步：TODO 2，块稀疏掩码。** `build_block_sparse_mask(seq_len, block_size, active_block_pairs, causal=True)` 先把序列按 `block_size` 分块，`num_blocks = ceil(seq_len / block_size)`。骨架逐对检查 `(query_block, key_block)`：越界抛 `ValueError('active_block_pairs 包含越界 block')`；`causal=True` 且 `key_block > query_block` 时抛 `ValueError('因果 block 稀疏不能激活未来 block')`。参考答案：

```python
block_mask = np.ones((q_end - q_start, k_end - k_start), dtype=bool)
mask[q_start:q_end, k_start:k_end] = block_mask
```

也就是把被选中的块对整块置为可见。循环结束后，如果 `causal=True`，再执行 `mask &= np.tri(seq_len, dtype=bool)`：`np.tri` 是下三角（含对角线），把块内那些仍然属于"未来"的 token 过滤掉。块对决定"候选区域"，因果过滤决定"区域里哪些格子最终可见"，两层不能合并——同一个块对在自回归任务和双向任务里结果不同，测试 `test_block_sparse_causal_filter` 用 `causal=False` 显式保留了 `(0,1)` 这个未来块，得到 `bidirectional[0,2]` 为真。

注意最后一个块可能不满：`q_end`、`k_end` 都用 `min(..., seq_len)` 截断，所以块掩码的形状要按实际边界算，不能一律写 `block_size × block_size`。

**第 3 步：TODO 3，连接账本。** `summarize_access(mask)` 要求输入是方形 `bool` 数组，否则 `ValueError`。参考答案：

```python
visible_pairs = int(mask.sum())     # True 的个数
total_pairs = mask.size             # L*L
density = visible_pairs / total_pairs
```

`mask.sum()` 在布尔数组上数 `True` 的个数；`int(...)` 是为了让返回值是 Python `int`，而不是 `np.int64`。测试 `summarize_access(build_local_causal_mask(4, 2))` 必须精确等于 `{'visible_pairs': 7, 'total_pairs': 16, 'density': 7/16}`。手算：4 行每行可见数分别是 1、2、2、2，合计 7；总格子 16，密度 0.4375。这里 `density` 只是理论连接比例，不能写成"kernel 加速比"；notebook 的解析特意强调，要判断收益还得固定模型、张量形状和后端，测延迟、显存和任务质量。

**第 4 步：TODO 4，最少传播层数。** 局部窗口减少的是"单层直接访问范围"，不是"永远看不到远处"。若每层最多把信息向前传 `window_size - 1` 个位置，距离 $d$ 至少需要

$$\text{layers}=\left\lceil\frac{d}{w-1}\right\rceil$$

层。参考答案用整数向上取整：`(distance + step_per_layer - 1) // step_per_layer`，其中 `step_per_layer = window_size - 1`。边界由骨架先处理：`distance == 0` 返回 0；`window_size == 1` 时每层只能传 0 格，非零距离返回 `float('inf')`。测试覆盖 `(0,1) → 0`、`(1,2) → 1`、`(5,3) → 3`（$\lceil5/2\rceil=3$）、`(1,1) → inf`，以及负距离抛 `ValueError`。

notebook 的 Step 3 给了理论连接数对照：稠密因果是 $L(L+1)/2$；滑动窗口在 $L\ge w$ 时是 $wL-w(w-1)/2$；块稀疏由激活块对和因果过滤决定；选择性稀疏取决于内容或检索选了多少位置。这些数只是结构账本，实际速度还取决于张量形状、掩码表示、计算内核和后端。

**第 5 步：GPU 实验的自变量、因变量与固定条件。** 5.1 的固定配置是：`PROFILE_ID = 'ARCH-CUDA-MECHANISM'`、`WORKLOAD_ID = 'arch-window-sdpa-v2'`、batch 1、8 个注意力头、`head_dim=64`、FP16、长度阶梯 `[256, 512, 1024, 2048]`、窗口固定 256、warmup 10、repeats 100、`mask_representation='explicit_boolean_mask'`。自变量只有访问范围/掩码表示：baseline 调 `F.scaled_dot_product_attention(..., is_causal=True)`，candidate 调同一份 Q/K/V 但传 `attn_mask=window_mask`。因变量是 `mean_ms`、`peak_allocated_mb`、`output_finite`，外加账本字段 `dense_visible_pairs`、`window_visible_pairs`、`mask_storage_mb`、`last_token_can_read_first`。

`RUN_MODE` 默认是 `dry_run`，只登记计划；改成 `real_gpu` 且 CUDA 可用时才真正执行。每次测量先跑 warmup，`torch.cuda.synchronize` 后用 `torch.cuda.Event` 计时，重复 100 次取平均，并用 `torch.cuda.max_memory_allocated` 记录峰值。两条路径共用同一份随机 Q/K/V（`torch.manual_seed(44 + seq_len)`），只改掩码。

**第 6 步：已保存的 GPU 结果怎么读。** notebook 里保存了 `rtx5070ti_window_ladder_v2` 的结果（GPU 为 RTX 5070 Ti Laptop GPU），四档如下：

| 长度 | 稠密连接数 → 窗口连接数 | 掩码存储 | 延迟 baseline → candidate | 峰值显存 |
|---|---|---|---|---|
| 256 | 32896 → 32896 | 0.06 MiB | 0.016 → 0.036 ms（ratio 0.46） | 1.3 → 1.4 MiB |
| 512 | 131328 → 98432 | 0.25 MiB | 0.028 → 0.050 ms（ratio 0.57） | 2.8 → 3.2 MiB |
| 1024 | 524800 → 229504 | 1.00 MiB | 0.051 → 0.127 ms（ratio 0.40） | 6.0 → 8.0 MiB |
| 2048 | 2098176 → 491648 | 4.00 MiB | 0.149 → 0.526 ms（ratio 0.28） | 14.1 → 22.0 MiB |

`ratio` 是 `baseline.mean_ms / candidate.mean_ms`，小于 1 表示 candidate 更慢。读法分三步：先看 `output_finite`，四档都为真，说明数值没问题；再看连接数，L=256 时窗口等于全长，连接数没减少，L=2048 时从 2098176 降到 491648，减少 76.57%；最后看代价，L=256 已经没有连接收益，延迟却变成 2 倍左右，L=2048 时掩码本身要 4 MiB，延迟约 3.5 倍，峰值从 14.1 MiB 涨到 22.0 MiB。因此 `decision` 是 `investigate_mask_backend`，理由记录为"显式布尔 mask 没有同时改善时间和峰值显存；先检查 backend、mask 表示与 shape"。

关键理解：滑动窗口掩码仍然是一个 `L×L` 的布尔张量，存储随 $L^2$ 增长；SDPA 收到显式 `attn_mask` 后，未必能跳过被屏蔽的块，反而可能换到更慢的后端。`last_token_can_read_first` 只有 L=256 为真，其余为假，正好说明窗口确实截断了远程直接访问。`backend_note` 里也写明：`flash_sdp_enabled` 等开关不保证本次调用实际选中的后端，显式掩码可能改变执行路径。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 访问掩码 | 一张 `[L, L]` 的表，`True` 表示该 Query 可以读该 Key |
| 因果掩码 | 只允许看当前位置及以前，禁止看未来 |
| 滑动窗口 | 每个 Query 只看最近 $w$ 个位置 |
| 局部因果掩码 | 因果 + 窗口两个条件同时成立 |
| 块稀疏 | 先按 token 块选择块对，再在块内做因果过滤 |
| 连接账本 | 可见连接数、总格子数和密度，描述"允许看什么" |
| 密度 `density` | `visible_pairs / total_pairs`，理论比例，不是加速比 |
| 传播层数 | 局部信息跨过距离 $d$ 至少要经过的层数，$\lceil d/(w-1)\rceil$ |
| 显式布尔掩码 | 真正物化一张 `L×L` 的 bool 张量传给 SDPA |
| SDPA | PyTorch 的 `scaled_dot_product_attention`，会按条件选后端 |
| `is_causal` | 让 SDPA 用因果模式，不需要显式传掩码 |
| 配对微基准 | 两条路径共用 Q/K/V 和测量方法，只改一个变量 |

## 小白常见疑问

**Q1：`(query - key) < window_size` 用的是严格小于，窗口到底包含几个位置？**
包含当前位置和前面共 `window_size` 个位置。以 `window_size=2` 为例，第 3 行可见 `key=3`（差 0）和 `key=2`（差 1），差 2 的 `key=1` 不可见，所以每行最多 2 个。测试 `build_local_causal_mask(5,2)` 正是断言 `mask[3,2]` 真、`mask[3,1]` 假。如果写成 `<=`，窗口会多一格。

**Q2：块稀疏里已经有 `causal=True` 的块对检查，为什么最后还要 `mask &= np.tri(...)`？**
块对检查只保证"不激活整个未来块"，但被激活的块对内部仍然包含未来位置。例如 `(1,1)` 这个对角块，query 在第 2 行、key 在第 3 列时仍然是未来。`np.tri` 是全局下三角，负责把这些格子逐个滤掉。块对是粗粒度过滤，`np.tri` 是 token 级过滤，两层缺一不可。

**Q3：`summarize_access` 的 `density` 能直接拿来预测加速吗？**
不能。它只说明掩码里 `True` 的比例。notebook 的解析明确写了"`density` 只描述理论连接比例；要判断收益还要固定模型、张量形状与执行后端"。已保存的 GPU 结果就是反例：L=2048 时连接数少了 76.57%，但显式掩码路径延迟约 3.5 倍、峰值显存更高。真正的收益要看内核能否利用稀疏结构，而不是看账本。

**Q4：`min_local_propagation_layers(5, 3)` 为什么是 3？**
每层最多向前传 `window_size - 1 = 2` 个位置。距离 5 需要覆盖 5 格：一层覆盖 2，两层覆盖 4，仍不够；三层覆盖 6，才覆盖得了，所以 $\lceil5/2\rceil=3$。代码用 `(5 + 2 - 1) // 2 = 3`。这个层数是"信息能否间接到达"的下界，不代表模型一定学到了有效表示，也不替代长依赖任务上的质量验证。

**Q5：为什么 L=256 时连接数完全没减少，测试还要保留这一档？**
这一档是"无连接缩减时的 mask 开销"对照。窗口 256 等于序列长度，窗口掩码和因果掩码在语义上等价，连接数都是 32896；但 candidate 仍然要物化并传入一张布尔掩码，所以能单独观察到掩码本身带来的延迟和显存开销。结果里这一档延迟从 0.016 ms 变成 0.036 ms，正好说明开销不是来自"少看了哪些 token"，而是来自"多传了一张表"。

## 动手验证：怎么知道自己做对了
CPU 测试单元按顺序直接调用函数，最后打印 `✅ 局部与稀疏 Attention 访问机制测试通过。`。dump 中没有保存这一格的输出，以断言为准。

- `test_local_causal_visibility`：`build_local_causal_mask(5, 2)` 中 `mask[3,3]`、`mask[3,2]` 必须为真，`mask[3,1]`、`mask[2,3]` 必须为假。
- `test_block_sparse_pairs_and_bounds`：`build_block_sparse_mask(4, 2, {(0,0),(1,0)})` 中 `mask[1,0]`、`mask[3,0]` 为真，`mask[0,1]`、`mask[0,3]` 为假；空块对集合的结果必须全假；`(2,0)` 越界必须抛 `ValueError`。
- `test_block_sparse_causal_filter`：`(0,1)` 在 `causal=True` 时抛 `ValueError`；`causal=False` 时 `bidirectional[0,2]` 必须为真。
- `test_access_summary`：`build_local_causal_mask(4, 2)` 的账本必须精确等于 `{'visible_pairs': 7, 'total_pairs': 16, 'density': 7/16}`。
- `test_local_propagation_layers`：`(0,1) → 0`、`(1,2) → 1`、`(5,3) → 3`、`(1,1) → float('inf')`；负距离必须抛 `ValueError`。

GPU 部分的自查不看"是否更快"，而看证据是否自洽：先确认 `RUN_MODE`、`PROFILE_ID`、`WORKLOAD_ID` 和结果路径；再确认 `output_finite` 全为真、`dense_visible_pairs` 等于 `L(L+1)/2`、`window_visible_pairs` 等于 `mask.sum()`；最后看 `decision`。已保存的四档结果全部通过正确性检查，但因为 candidate 在所有长度上都更慢或更占显存，决策是 `investigate_mask_backend`。想复现就设置唯一 `RUN_LABEL` 并把 `RUN_MODE` 改成 `real_gpu`，不要覆盖已有 JSON。
