# 29. Tensor Parallelism Sim Beginner | Tensor 并行模拟 · 小白入门

> 配套 notebook：[29. Tensor 并行模拟](./29_Tensor_Parallelism_Sim.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

一层太大、一张 GPU 扛不住时，可以把它的权重矩阵切成几块，分给几张卡一起算。这本 notebook 教你两种切法，以及怎样让相邻两层接得上、少通信。

## 为什么需要它

当单个线性层的权重、计算量或中间结果一张设备扛不住时，可以把这一层沿张量的某个维度拆到多张卡上一起算，这就是张量并行（Tensor Parallelism）。它既用在 MLP 里，也用在注意力的投影层里。

切法有两种，各有一个要注意的地方：

- 切输出维（列切分）。每张卡拿到完整的输入，各算出输出的一段特征。
- 切输入维（行切分）。输入也要按同一范围切开，每张卡只算出一个部分和，最后要把各卡的结果加起来。

张量并行的核心不是孤立地切开一个矩阵，而是让相邻两层的切法接得上：前一层切输出维，后一层就切输入维，中间的结果一直保持切开的状态，不必先拼成完整的再重新切。这本 notebook 在 CPU 上实现这条路径，还要决定注意力里的 Query Head 和 KV Head 怎么分给各张卡。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 张量并行（Tensor Parallelism，TP） | 把同一层的权重矩阵沿某个维度切开，分给多张 GPU 一起算 |
| `Y = X @ A` | 一个线性层。`X` 是输入，形状 `[batch, in_dim]`；`A` 是权重，形状 `[in_dim, out_dim]` |
| 列切分（Column Parallel） | 切权重的列，也就是输出维。每张卡拿完整输入，算出输出的一段特征 |
| 行切分（Row Parallel） | 切权重的行，也就是输入维。输入也按同一范围切，每张卡算出一个部分和 |
| 分片 / 部分和（shard / partial sum） | 分片是整体里属于某张卡的那一段；部分和是只加了一部分的求和结果，各卡相加才是完整输出 |
| 集体通信 | 多张卡一起交换数据的操作，比如全归约（all-reduce）把各卡的结果加起来 |
| Query Head / KV Head | 注意力里的查询头和键值头。GQA（分组查询注意力）里 KV Head 比 Query Head 少 |

## 核心思路

先看一个线性层：输入 `X` 的形状是 `[batch, in_dim]`，权重 `A` 的形状是 `[in_dim, out_dim]`，输出 `Y = X @ A` 的形状是 `[batch, out_dim]`。两种切法如下：

```text
列切分（Column Parallel）：切 A 的列，也就是输出维
    A 切成 [A_0, A_1]。每张卡拿完整的 X，各算 Y_i = X @ A_i
    Y_i 是输出的一段特征。需要完整的 Y 时，把各卡的 Y_i 沿特征维拼接
行切分（Row Parallel）：切 A 的行，也就是输入维
    X 也按同一范围切成 X_i。各卡算部分和 Y_i = X_i @ A_i
    完整输出 Y = Y_0 + Y_1 + …，最后把各卡的结果按元素相加
```

`A_i` 是权重分给第 `i` 张卡的那一块，`X_i` 是输入里对应范围的那一段。用 2 张卡和一个 2 行 4 列的权重试一下：

```python
import torch

X = torch.tensor([[1.0, 2.0]])                                  # 输入，形状 [batch, in_dim]
A = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # 权重，形状 [in_dim, out_dim]
show = lambda t: t.flatten().tolist()                           # 打印成一维列表
print(show(X @ A))                                              # 不切分的结果
col = torch.chunk(A, 2, dim=1)                                  # Column：切 A 的列，每张卡拿完整输入
print([show(X @ a) for a in col])                               # 两个输出分片
print(show(torch.cat([X @ a for a in col], dim=-1)))            # 拼接，对应 All-Gather
xs, ws = torch.chunk(X, 2, dim=1), torch.chunk(A, 2, dim=0)     # Row：输入和权重按同一范围切
print([show(x @ w) for x, w in zip(xs, ws)])                    # 两个部分和
print(show(torch.stack([x @ w for x, w in zip(xs, ws)]).sum(dim=0)))   # 求和，对应 All-Reduce
W1 = torch.tensor([[1.0, 2.0, -1.0, 0.0], [0.0, 1.0, 2.0, 1.0]])       # 配对 MLP：W1 按列切
W2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])   # W2 按行切
print(show(torch.relu(X @ W1) @ W2))                            # 不切分的结果
partials = []
for start, end in [(0, 2), (2, 4)]:                             # 每个 rank 负责的 hidden 范围
    h = torch.relu(X @ W1[:, start:end])                        # 本地的 hidden 分片，不拼成完整的
    partials.append(h @ W2[start:end, :])
print([show(p) for p in partials])
print(show(torch.stack(partials).sum(dim=0)))
```

```text
[11.0, 14.0, 17.0, 20.0]
[[11.0, 14.0], [17.0, 20.0]]
[11.0, 14.0, 17.0, 20.0]
[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]]
[11.0, 14.0, 17.0, 20.0]
[2.0, 11.0]
[[1.0, 4.0], [1.0, 7.0]]
[2.0, 11.0]
```

输入是 `[1, 2]`，不切分的结果是 `[11, 14, 17, 20]`。第 2、3 行是列切分：两张卡各拿 2 列权重，各算出 2 个特征，拼起来就是完整输出。第 4、5 行是行切分：输入也切成 `[1]` 和 `[2]`，两张卡各算出一个部分和，形状和完整输出一样，加起来才是完整输出。最后三行是两层 MLP `relu(X @ W1) @ W2`：`W1` 按列切，`W2` 按行切，每张卡管 hidden 的一半。中间的 hidden 分片不拼成完整的，每张卡直接算出部分和，最后只求和一次。

这就是布局连续性：相邻两层的切法接得上，中间激活就不用先拼成完整的再重新切。中途恢复完整激活，数值上没错，却会多一次传输。

注意力用同样的思路：Q/K/V 投影沿输出维按 Head 切分，每张卡完成本地注意力，输出投影再沿输入维切分并归约。Query Head 数必须能被 TP 度整除；KV Head 数能整除就分片，KV Head 比 rank 少时可以复制。复制能满足映射约束，但会增加 KV 状态占用。另外要记住代价：张量并行降低每卡持有的权重与单层计算量，却在层内引入频繁的集体通信。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 列切分和行切分各切哪一维，完整输出分别怎样恢复 | 读，对照上面的公式和小例子 |
| Step 2 | 两层 MLP 和注意力里，相邻两层的切法怎样接得上 | 读 |
| Step 3 | 每条路径在哪里通信，主要代价是什么 | 读 |
| Step 4 · TODO 1 | `X`、沿输出维切分的权重 | 在 `tensor_parallel_column_sim` 里写 3 行：沿输出维切权重，各卡用完整输入算局部输出，沿特征维拼接 |
| Step 4 · TODO 2 | 沿同一输入维切分的 `X` 与权重 | 在 `tensor_parallel_row_sim` 里写 4 行：沿输入维切输入和权重，算部分和，按元素求和 |
| Step 4 · TODO 3 | `W1 / W2` 的匹配 hidden 范围、`partial_outputs` | 在 `tensor_parallel_mlp_sim` 里写 4 行：取本 rank 的 `W1` 列分片，算 hidden 分片，取同一范围的 `W2` 行分片，算部分和 |
| Step 4 · TODO 4 | `num_q_heads`、`num_kv_heads`、`tp_size` | 在 `plan_attention_tp` 的两个分支里各写 1 行：`num_kv_heads` 能被 `tp_size` 整除就 `shard`；反过来 `tp_size` 能被 `num_kv_heads` 整除就 `replicate`；其余情况报错 |
| 测试 | 检查两种切分和不切分实现（dense）的数值等价、输出形状、不能整除时是否报错、配对 MLP 的等价，以及 KV Head 在分片、复制、不兼容三种配置下的行为 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在两张 GPU 上运行真实的配对 MLP，和 dense 输出核对 | 需要 `PAR-CUDA-MECHANISM` 环境和两张 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要两张 GPU，默认不执行真实实验。

## 容易踩的坑

- **行切分只切了权重**：输入必须沿同一输入维范围切分，各卡才能算出完整输出的部分和。
- **行切分的末端用了拼接**：部分和要按元素求和，不能用拼接代替。靠拼接恢复输出的是列切分。
- **中间的 hidden 提前拼成完整的**：第一层按列切产生的 hidden 分片，直接喂给第二层同一范围的行分片。中途拼成完整张量，数值没错，但多一次传输。
- **切分维不能被卡数整除**：列切分要求输出维能整除，行切分要求输入维能整除，配对 MLP 要求 hidden 维能整除。不满足时应当显式失败。
- **Head 数不满足映射约束**：Query Head 数必须能被 TP 度整除。KV Head 既不能被 TP 度整除、TP 度也不能被 KV Head 数整除时，同样报错，不要硬凑映射。

## 读完后自查

- [ ] 能说出列切分和行切分各切权重的哪一维，完整输出分别靠什么恢复。
- [ ] 2 张卡、权重是 2 行 4 列，能说出列切分后每张卡算出的输出分片有几个特征。
- [ ] 能解释为什么两层 MLP 的第一层按列切，第二层就要按行切。
- [ ] 4 张卡、2 个 KV Head，能判断 KV Head 该分片还是复制。

## 下一步

- 打开 notebook：[29. Tensor 并行模拟](./29_Tensor_Parallelism_Sim.md)
- 需要补基础：[并行策略决策框架](../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)
- 做完之后：[PAR-CONTEXT. 上下文与序列并行 · 小白入门](./PAR-CONTEXT_Context_and_Sequence_Parallelism_Beginner.md)
