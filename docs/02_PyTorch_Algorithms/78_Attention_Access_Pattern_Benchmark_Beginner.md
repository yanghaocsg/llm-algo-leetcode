# 78. Attention Access Pattern Benchmark Beginner | Attention 访问模式基准 · 小白入门

> 配套 notebook：[78. Attention 访问模式基准](./78_Attention_Access_Pattern_Benchmark.md)

## 一句话说清

注意力默认让每个词元读遍前面所有词元，序列越长计算越多。这本 notebook 把“只读哪些历史”写成一张可见表，再数清每种规则省了多少计算、丢了多少远处的信息。

## 为什么需要它

注意力计算时，每个词元要给前面所有词元各打一次分。连接数随序列长度二次增长，序列一长，这部分计算就很可观。

省计算的办法是不读全部历史，只读一部分。这本 notebook 列了三种，各有各的风险：

- 滑动窗口。只看最近的几个词元。很早出现的信息可能再也读不到。
- 块稀疏。按块开放连接，跨块的信息要靠少数“锚点”位置传递。
- 选择性稀疏。在最近窗口之外，再保留几个选定的锚点。选错了，证据就丢了。

看一个小场景。序列有 8 个词元，全局因果注意力一共有 36 条连接。如果只看最近 3 个词元，连接数降到 21 条，但最后一个词元读不到下标 4 及更早的内容。远处那条线索还在序列里，只是这个位置够不着了。

所以换规则之前要先问两件事：远处的信息还能不能读到？计算到底省了多少？这本 notebook 把访问规则写成一张可以检查的表，从同一张表数出这两件事，再结合质量门槛和证据等级给出建议。它不比较模型能力榜单。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| query / key | 注意力里的两个角色。query 是当前正在处理的词元，key 是它可以读取的历史词元 |
| 掩码（mask） | 一个由 `True` 和 `False` 组成的方阵，行是 query 位置，列是 key 位置。`mask[q, k]` 为 `True` 表示位置 `q` 可以读位置 `k` |
| 因果（causal） | 一个位置只能读它自己和它前面的词元，不能读未来 |
| 滑动窗口（sliding window） | 只保留最近 `window_size` 个词元的规则 |
| 锚点（anchor） | 在 `anchor_positions` 里指定的少数历史位置，不随窗口滚动被淘汰 |
| SDPA | PyTorch 自带的注意力函数 `scaled_dot_product_attention`，可以传入掩码 |
| needle 检查 | 在序列远处放一条信息，看后面的位置还能不能读到它 |

## 核心思路

输入是一份候选规格 `spec`，里面有模式（`pattern`）、序列长度（`seq_len`）、窗口大小、块大小和锚点位置。处理分四步，最后输出一个采用建议：

```text
1. 校验规格     模式是四种之一，seq_len 为正；局部模式的 window_size、块稀疏的 block_size 为正；
                锚点必须是序列内的位置
2. 生成掩码     每个位置 q 先只考虑 k <= q 的 key（不能读未来），再按模式决定可见：
   global            全部可见
   sliding_window    k >= q - window_size + 1
   block_sparse      和 q 在同一个块里，或者是锚点
   selective_sparse  在最近窗口里，或者是锚点
3. 汇总账本     score_count = 掩码里 True 的个数
                density = score_count / causal_global_count（全局因果掩码的连接数）
                long_range_coverage = 超出局部窗口的连接里，仍然可见的比例
4. 做出决定     先筛出 quality >= min_quality 且 density <= max_density 的候选
                按 density 小的优先、quality 高的优先选出一个
                证据等级是 sparse_backend_benchmark 才 accept，否则 tune；没有合格候选是 reject
```

`window_size` 是最近历史的保留宽度，`block_size` 是块稀疏的分组粒度。账本里还有 `visible_keys`，是每个 query 行里可见的 key 数。`long_range_coverage` 可以叫长程覆盖，回答“远处的线索是否仍然可达”。

用测试里的规格试一下窗口掩码和稠密度：

```python
import numpy as np

seq_len, window_size = 8, 3
mask = np.zeros((seq_len, seq_len), dtype=bool)
for q in range(seq_len):
    for k in range(q + 1):                             # 只看自己和前面
        mask[q, k] = k >= max(0, q - window_size + 1)  # 最近 3 个

print(mask[7].astype(int))
score_count, causal_global_count = int(mask.sum()), seq_len * (seq_len + 1) // 2
print(score_count, causal_global_count)
print(round(score_count / causal_global_count, 4))
```

```text
[0 0 0 0 0 1 1 1]
21 36
0.5833
```

第一行是位置 7 的可见情况：位置 5、6、7 可见，位置 4 及更早都不可见。窗口掩码的连接数是 21，全局因果掩码是 36，所以稠密度约 0.58。

notebook 对结论留了余地。理论连接数少，不等于任务质量一定下降。给 PyTorch SDPA 传入稀疏掩码，也不等于自动调用了专用稀疏 kernel。CPU 部分只能证明访问规则和远处可达性，GPU 探针只记录同一 PyTorch 路径下的运行行为。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 四种访问候选各自能读哪些历史，收益和风险是什么 | 读 |
| Step 2 | 从同一张掩码数出可见范围、长程覆盖、连接数和稠密度 | 读，对照上面的“汇总账本” |
| Step 3 | CPU 掩码检查、连接账本、GPU 探针、真实后端基准各能支持什么结论 | 读 |
| Step 4 · TODO 1 | 校验访问模式规格 | 在 `validate_access_spec` 里确定 `pattern`、`seq_len`、`issues`：把不合法的窗口、块、锚点记进问题列表 |
| Step 4 · TODO 2 | 从候选规格生成因果访问 mask | 在 `build_access_mask` 里写循环：先保证 `key_index <= query_index`，再按四种模式决定 `visible` |
| Step 4 · TODO 3 | 把 mask 汇总为覆盖与成本代理 | 在 `summarize_access_pattern` 里算 `score_count`、`long_range_allowed`、`density` |
| Step 4 · TODO 4 | 在质量、成本和证据等级下推荐候选 | 在 `decide_access_pattern` 里写 `feasible`，选出最优候选，再按 `evidence_level` 给出 `decision` |
| 测试 | 四个测试依次检查规格、因果可见性、账本和决策 | 运行 |
| 参考代码与解析 | 参考答案和解析 | 卡住再看 |
| Step 5（可选） | 在 GPU 上用 PyTorch SDPA 跑同一组掩码，记录运行时间和峰值显存 | 需要 CUDA 版 PyTorch。默认是 `RUN_ACCESS_GPU_PROBE = False`，默认关闭，不会自动下载模型或启动服务 |

Step 1–4 只需要 CPU，只用到 `numpy`。Step 5 需要 CUDA 版 PyTorch，默认关闭。

## 容易踩的坑

- **锚点越界或窗口为 0**：测试里 `window_size` 取 0、`anchor_positions` 取 `[9]`（序列长度只有 8）都不通过。`sliding_window` 和 `selective_sparse` 要求 `window_size` 为正，`block_sparse` 要求 `block_size` 为正。
- **掩码读到了未来**：`key_index` 大于 `query_index` 的位置必须全是 `False`。测试用 `np.triu(mask, k=1)` 取出未来部分，要求里面没有任何 `True`。
- **把连接数当实际效率**：`score_count` 只适合比较理论规模，不能说明 kernel 的实际效率。`long_range_coverage` 也不能替代真实长文本任务得分。
- **证据不够就 `accept`**：只有真实稀疏后端的基准记录（`sparse_backend_benchmark`）才能 `accept`。测试里同一条记录把证据等级换成 `gpu_sdpa_mask_probe`，结论就从 `accept` 变成 `tune`。
- **把 SDPA 探针当成专用稀疏实现**：普通 SDPA 掩码结果不能写成专用实现的收益。要验证专用稀疏 kernel，必须另建候选，并保存实际后端、kernel、回退状态、版本与命令。

## 读完后自查

- [ ] 能说出四种候选分别允许当前词元读取哪些历史。
- [ ] 序列长度 8、窗口 3 时，能写出滑动窗口掩码最后一行的可见位置。
- [ ] 能说出 `score_count`、`density`、`long_range_coverage` 各自从掩码怎么数出来。
- [ ] 能解释为什么只有 CPU 掩码或 GPU SDPA 探针证据时不能 `accept`。

## 下一步

- 打开 notebook：[78. Attention 访问模式基准](./78_Attention_Access_Pattern_Benchmark.md)
- 建议先读：[04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA_Beginner.md)、[44. 局部、滑动窗口与稀疏注意力 · 小白入门](./44_Local_Sliding_and_Sparse_Attention_Beginner.md)
- 做完之后：[71. MLA 与 KV Cache 结构基准 · 小白入门](./71_MLA_KV_Cache_Architecture_Benchmark_Beginner.md)
