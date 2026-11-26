# INF-MOE-EXECUTION. MoE 推理中的词元分组与专家执行 — 小白讲解

> 对应 notebook：[`INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.ipynb`](./INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.ipynb)

## 一句话总结
MoE 推理里，同一个 token 会被 Top-K 路由分给 K 个专家；逐 token 调用和“按专家分组批量调用”在数学上等价，但后者把同专家的行拼成矩阵，可能减少小矩阵调用。本节用 `pack_expert_assignments` 建立可逆分组（同时搬动词元位置、专家编号和路由权重），用 `combine_expert_outputs` 按原位置加权累加，用 `expert_load_metrics` 量化填充率与负载偏斜。最后给出一套默认 `dry_run` 的 GPU 实验，比较三档词元数 × 两种路由下的调用次数、延迟与误差。

## 为什么需要它：痛点是什么
初学者容易以为“路由选完专家就结束了”。但执行阶段还要回答：哪些行送给哪个专家？同一 token 的两路贡献怎么合并？某专家这一轮只分到 1 行，矩阵乘还有意义吗？解码时每轮 token 少，大量专家可能只处理一两行；预填充时 token 多，又可能出现两三个专家包揽大部分行。直接按 token 循环调用会让每次矩阵很小、调用次数爆炸；把 token 按专家重排后，虽然矩阵变规整，却多了一次重排和一次恢复。

生活类比：快递分拣。每个包裹（token）要去两个目的地（专家）。逐 token 送货就是“一个包裹跑两趟”；按专家分组就是“先把同一目的地的包裹装一车，送完再按收件人拆回”。后者更省车次，但你需要两套标签：包裹属于谁（`token_ids`）和这趟该收多少运费（`route_weights`）。标签丢一个，最后拼回来的结果就错了。

## 核心问题
- 分组为什么要同时保存 `token_ids`、`expert_ids`、`weights` 三样东西？只保存行顺序够不够？
- `counts` 和 `offsets` 分别表示什么？空专家为什么要保留长度为 0 的区间？
- 合并时为什么必须用 `index_add_`，不能直接 `output[token_ids] = weighted`？
- 填充行数、有效行比例、偏斜度 $I$ 怎么算？为什么“偏斜大”不一定“填充多”？
- GPU 实验里 `path_confirmed` 检查的 `3 * token_count * top_k` 和 `3 * active` 是怎么来的？加速门槛 1.05 对应多少时间下降？

## 解决思路：一步步看懂

**Step 1：先写清“原序输出”的定义。** 设 $T$ 个 token、每个 token $D$ 个特征、每个 token 选 $K$ 个不同专家、共 $E$ 个专家。第 $t$ 个 token 的第 $j$ 路分配去专家 $e_{t,j}$，路由权重 $g_{t,j}\ge0$ 且 $\sum_j g_{t,j}=1$。专家函数 $F_e$ 输出仍是 $D$ 维：

$$
y_t=\sum_{j=0}^{K-1}g_{t,j}F_{e_{t,j}}(x_t),\qquad 0\le t<T.
$$

总共有 $TK$ 次分配。notebook 的例子：两个专家对同一 token 分别输出 2 和 6，权重 0.75 和 0.25，正确结果是 $0.75\times2+0.25\times6=3$；权重错配会得到 $0.25\times2+0.75\times6=5$；如果直接覆盖写回，只会留下某一路的 2 或 6。三种结果形状都可能对，但数值只有第一种是对的——所以检查不能只看形状。

**Step 2：TODO 1——按专家建立可逆分组。** 参考答案：

```python
order = torch.argsort(flat_experts, stable=True)      # 稳定排序，同专家内保持原顺序
token_ids = flat_tokens[order]                        # 排序后每行来自哪个 token
counts = torch.bincount(flat_experts, minlength=num_experts)  # 每个专家多少行
offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
```

`flat_experts` 是 `[T*K]`，`flat_tokens = torch.arange(T).repeat_interleave(K)` 让每个 token 的 K 次分配指向同一原位置。`argsort(stable=True)` 让同专家的行保持原顺序，测试因此断言 `token_ids[:4] == [0,1,3,4]`。`bincount(minlength=E)` 保留空专家（值 0），`offsets` 是长度为 $E+1$ 的前缀和；专家 $e$ 的区间是 `[offsets[e], offsets[e+1])`，空专家两边界相等。`inputs`、`weights`、`expert_ids` 用同一个 `order` 移动，这是“可逆”的关键。常见错误是只排序专家却忘了同步移动 token/权重，或用非稳定排序改变组内顺序。

夹具手算：5 token、K=2、E=4，`ids=[[2,0],[0,2],[2,1],[1,0],[0,1]]` 展平为 `[2,0,0,2,2,1,1,0,0,1]`；专家 0/1/2/3 分别出现 4/3/3/0 次，所以 `counts=[4,3,3,0]`、`offsets=[0,4,7,10,10]`。

**Step 3：TODO 2——恢复原序并合并。** 参考答案：

```python
weighted = expert_outputs * route_weights.to(expert_outputs.device, expert_outputs.dtype).unsqueeze(1)
output.index_add_(0, token_ids.to(expert_outputs.device), weighted)
```

`expert_outputs` 是 `[T*K, D]`，每行对应一次分配；`route_weights` 是 `[T*K]`，要先 `unsqueeze(1)` 变成 `[T*K,1]` 才能按行广播。`index_add_` 的语义是“对每个位置 `token_ids[i]`，把 `weighted[i]` 加到 `output[token_ids[i]]` 上”，所以同一 token 的多路贡献会累加，而不是后写覆盖前写。这正好实现 $y_t=\sum_j g_{t,j}F_{e_{t,j}}(x_t)$。

`test_combine_matches_token_reference` 用独立双重循环算期望值，再把三组行同时倒序，结果必须不变——说明合并只依赖“位置 + 权重”，不依赖分组顺序。常见错误：`output[token_ids] = weighted`（覆盖）、`scatter_add_` 索引/广播写错、忘记统一设备与 dtype。校验还要求位置在 `[0, token_count)`、权重有限非负、输出有限，空 token 返回 `(0, D)`。

**Step 4：TODO 3——量化填充与偏斜。** 参考答案：

```python
padded = ((counts + tile_rows - 1) // tile_rows) * tile_rows
utilization = total / int(padded.sum()) if total else 0.0
imbalance = float(counts.max()) / mean_load if total else 0.0
```

`padded` 把每个非空专家的行数向上取整到 `tile_rows` 的倍数，空专家 `0` 仍然是 `0`。`utilization` 是“有效分配数 ÷ 填充后的总行数”，回答“为了对齐块宽浪费了多少行”。`imbalance` 就是 notebook 的 $I=\max_e c_e / (TK/E)$：`mean_load = total / len(counts)` 正是 $TK/E$。

用测试数据手算：`balanced=[4,4,4,4]`、块宽 4，`padded` 不变，填充 0，$I=4/4=1$；`skewed=[16,0,0,0]`，$I=16/4=4$，但 `padded=[16,0,0,0]` 仍然填充 0；`tail=[5,3,0,0]`，块宽 4 时 `padded=[8,4,0,0]`，总填充 12，有效 8，`utilization=8/12`。这组测试特意保留“偏斜增加但填充不增加”的反例：skewed 的 $I$ 从 1 涨到 4，填充却仍是 0，说明偏斜和填充是两项独立观察。常见错误是把 `utilization` 的分母写成原始总数（会永远接近 1），以及 `total=0` 时直接除导致 `ZeroDivisionError`。

**Step 5：GPU 实验怎么设计。** 默认 `RUN_MODE="dry_run"`，只打印计划；`real_gpu` 才真正跑。固定 `NUM_EXPERTS=8`、`TOP_K=2`、`HIDDEN_SIZE=512`、`FFN_SIZE=1536`、`TILE_ROWS=16`、`WARMUP=3`、`ITERATIONS=5`、`REPEATS=3`、`ATOL=RTOL=0.02`、`MINIMUM_SPEEDUP=1.05`。自变量是 `TOKEN_COUNTS=(1,16,128)` 和 `ROUTING_CASES=("balanced","skewed")`，共 6 档；因变量是 `median_wall_ms`、`median_cuda_ms`、`max_extra_peak_mb`、`max_abs_error`、`linear_calls` 和 `speedup`。

`tokenwise()` 对每个 token、每个 slot 单独调用 `expert_forward`，线性调用次数是 $3\times T\times K$（gate、up、down 三个 `linear`）。`grouped()` 先重排，再对每个非空专家算一整批，最后 `index_add_`，调用次数是 $3\times\text{active}$，其中 `active` 是 `counts>0` 的专家数。`path_confirmed` 检查这两个等式和 CUDA 事件；只有路径确认且 `speedup >= 1.05` 才标 `accept`，否则 `tune`。

1.05 的加速比对应 $1-1/1.05\approx4.76\%$ 的时间下降，这正是配置表里写的门槛。`cpu_plan_once_ms`（CPU 构造计划时间）单独记录，不混进稳态计时；1 个 token 的低档可能因重排/合并固定开销没有收益，notebook 明确说“低档无收益也保留”。

**Step 6：结果怎么读。** 先看 `status` 是否为 `measured`、`failure` 是否为空；再看 `max_abs_error` 是否在 `atol=0.02, rtol=0.02` 内、`finite` 是否为真，不满足就直接拒绝，先查分组、专家计算或合并。然后看 `path_confirmed` 和 `traces`：逐 token 路径的 `linear_calls` 应等于 `3*T*K`，分组路径应等于 `3*active`。最后看 `speedup` 和 `decision`：每档独立决定，不能把“128 token 有收益”直接推广到“1 token 也有收益”。`cpu_plan_once_ms` 只说明计划构造开销，不代表每轮推理都要付这个成本——只有路由每轮变化、计划不能复用时才需要重算。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Top-K 路由 | 每个 token 选概率最高的 K 个专家 |
| 分配（assignment） | 一次“token → 专家”的投递，共 $T\times K$ 次 |
| 可逆分组 | 把分配按专家排序，同时保留原 token 位置和权重，算完能还原 |
| `token_ids` | 分组后每一行原本属于哪个 token |
| `counts` / `offsets` | 每个专家的行数 / 每个专家在打包张量里的半开区间边界 |
| `index_add_` | 按索引把值累加到目标位置，同一索引会相加而不是覆盖 |
| 分块填充 | 把行数向上补齐到块宽 `tile_rows` 的倍数 |
| 有效行比例 | 有效分配数 ÷ 填充后的总行数 |
| 负载偏斜 $I$ | 最忙专家行数 ÷ 全体专家平均行数 |
| 活跃专家 | 本轮至少分到一行的专家 |
| grouped GEMM | 把多个专家的矩阵任务交给专用内核组织，而不是逐个调用 |
| 固定路由计划 | 复用同一套 token→专家映射，不每轮重新路由 |

## 小白常见疑问
**Q1：按专家分组后数学等价，为什么还要保留原 token 位置？**
一个 token 会出现在多个专家批次里，最后要加回同一行。只保留“分组后的行顺序”不够，还必须知道每行原来属于哪个 token。测试把三者同时倒序后结果不变，就是在验证“位置 + 权重”的绑定。

**Q2：为什么用 `index_add_`，不用 `output[token_ids] += weighted`？**
后者在索引重复时只保留部分写入，同一 token 的多路贡献可能互相覆盖。`index_add_` 明确对重复索引累加，语义正好是 $\sum_j g_{t,j}F_{e_{t,j}}(x_t)$。两路输出 2 和 6、权重 0.75/0.25 时，覆盖写回只剩 2 或 6，而不是 3。

**Q3：分组路径的线性调用为什么是 `3 * active`，不是 `3 * E`？**
空专家不参与计算。`grouped()` 里 `if start != end` 才调用 `expert_forward`，只有 `counts > 0` 的专家执行 gate/up/down 三次线性映射。8 个专家里 4 个活跃就是 12 次，不是 24 次。

**Q4：`I=4` 一定比 `I=1` 慢吗？**
不一定。$I$ 只描述行数分布，不描述矩阵形状、调用次数、内核效率和跨卡等待。`skewed=[16,0,0,0]` 的填充是 0，但只有 1 个活跃专家，并行度很低。真实时间要分别测量。

**Q5：`dry_run` 会保存性能结果吗？**
不会。它只打印 `{"status": "planned", ...}`，不初始化 CUDA、不执行 GPU 矩阵、不写 JSON。只有 `real_gpu` 才要求唯一 `RUN_LABEL`、检查结果文件不存在，并在跑完 6 档后写 `moe_execution_<label>.json`。

## 动手验证：怎么知道自己做对了
测试在 cell 10 直接调用四个函数，全部通过打印 `PASSED: 专家分组、加权恢复、填充与偏斜、空输入和非法路由核对通过。`
- `test_dispatch_identity_and_empty_experts`：夹具 5 token、4 专家，断言 `counts == [4,3,3,0]`、`offsets == [0,4,7,10,10]`；分组的 `(token_id, expert_id, weight)` 三元组排序后必须和原始列表完全一致；`packed["inputs"]` 必须等于 `x[token_ids]`；`token_ids[:4] == [0,1,3,4]`。
- `test_combine_matches_token_reference`：用独立双重循环算期望输出，`torch.testing.assert_close(actual, expected)`；把三组行同时倒序后再合并，结果仍等于期望。
- `test_load_padding_and_skew`：`balanced` 的 `max_over_mean == 1`、`skewed == 4`，两者 `padding == 0`；`tail` 的 `padding == 4`、`utilization == 8/12`。
- `test_empty_input_and_invalid_routes`：空输入得到 `(0,3)` 输出和全零 `offsets`；全零 `counts` 的 `utilization == 0`；越界专家、重复专家、权重和不为 1、合并位置越界、负权重、NaN 输出、负计数、块宽 0 都必须抛 `ValueError`。

GPU 部分默认不执行，dump 里也没有保存任何实测 JSON，所以不能引用具体加速比。自己跑 `real_gpu` 时，以 `cases` 里每档的 `status`、`path_confirmed`、`max_abs_error`、`speedup` 和 `decision` 为准；再对照 `traces` 里的 `linear_calls` 是否满足 `3*T*K` 与 `3*active`。想验证分组逻辑，可以手工把 `ids` 改成 `[[0,1],[0,1],...]`，此时 `active=2`，`counts` 和 `offsets` 会随之变化，但 `counts.sum()` 始终等于 `T*K`。
