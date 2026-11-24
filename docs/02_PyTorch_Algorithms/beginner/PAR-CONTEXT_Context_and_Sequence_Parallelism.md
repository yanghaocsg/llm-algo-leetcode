# PAR-CONTEXT. Context and Sequence Parallelism | 上下文与序列并行 · 小白入门

> 配套 notebook：[PAR-CONTEXT. 上下文与序列并行](../PAR-CONTEXT_Context_and_Sequence_Parallelism.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

注意力要每个词元都看到全部词元。上下文并行把长序列切成几段分给几张卡，这本 notebook 教你切开以后怎样算出和不切一样的结果，以及要为此搬多少数据。

## 为什么需要它

上下文一长，每个词元都要保存自己的 Q/K/V 状态，序列状态随长度增长。张量并行把每一层沿 Head 或特征维切开，但它不会自动把 `S` 个词元分散出去，每张卡仍可能保存完整序列的状态。

上下文并行（Context Parallelism）换了一个方向：把序列切成连续的几段，每张卡只保存自己那段的 Q/K/V。麻烦出在注意力上。它要求每个 Query 和序列里所有 Key 算分数。序列切开后，本地的 Query 只能直接访问本地的 K/V，其余的 K/V 在别的卡上。不通信就算不出完整的结果。

所以要做三件事：把序列切成连续分片；每张卡对拿到的每个 K/V 分片各算一部分注意力统计，再正确地合并；最后算一算这样要搬多少 K/V。这本 notebook 在 CPU 上把三件事各做一遍。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 上下文并行（Context Parallelism，CP） | 把序列沿长度方向切成几段，每张卡只保存自己那一段词元的 Q/K/V |
| Query / Key / Value（Q / K / V） | 注意力的三个输入。每个 Query 要和所有 Key 算分数，再按分数加权汇总所有 Value |
| 序列分片（shard）/ rank | 分片是序列里连续的一段词元；rank 是一张 GPU 在并行训练里的编号 |
| Ring Attention | 让 K/V 分片在 rank 之间一轮一轮地传。每轮算一块分数，传完以后每个 Query 都看完了整个序列 |
| 三样统计 `m`、`l`、`o` | 一个分片的最大值、指数和、加权和。指数和最后做分母，加权和最后做分子 |
| 在线 Softmax（online softmax） | 每来一个分片，就把它的统计并进已有统计，而不是等所有分数凑齐再算 |
| payload | 一轮通信实际搬运的数据量，这里按字节算 |

## 核心思路

第一步，K 和 V 沿序列维切成 `P` 个连续等长的分片，每张卡只保存一段。第二步，每张卡用自己的 Query 对一个 K/V 分片算分数，从这个分片算出三样统计：最大值 `m`、指数和 `l`、加权和 `o`。第三步，把各分片的统计合并，最后相除得到注意力输出。

合并时不能把两边的数直接相加，要先对齐到新的最大值 `m' = max(m, m_i)`，再缩放、再相加：

```text
l' = l × exp(m - m') + l_i × exp(m_i - m')     # 指数和
o' = o × exp(m - m') + o_i × exp(m_i - m')     # 加权和
输出 = o' / l'
```

不带下标的是已经累计的统计，带 `_i` 的是新分片的统计。用两个小分片试一下：一个 Query，4 个 Key 的分数是 `[0, 1, 2, 3]`，前 2 个在分片 0，后 2 个在分片 1。

```python
import torch

r = lambda t: round(t.item(), 4)                   # 打印时保留 4 位小数

def stats(s, v):                                   # 一个分片的三样统计：s 是分数，v 是对应的 value
    m = s.max(dim=-1, keepdim=True).values         # 最大值
    e = torch.exp(s - m)                           # 先减去最大值，再取指数
    return m, e.sum(dim=-1, keepdim=True), e @ v   # 最大值、指数和、加权和

m0, l0, o0 = stats(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0], [1.0]]))   # 分片 0
m1, l1, o1 = stats(torch.tensor([[2.0, 3.0]]), torch.tensor([[2.0], [3.0]]))   # 分片 1
print(r(m0), r(l0), r(o0))
print(r(m1), r(l1), r(o1))

m = torch.maximum(m0, m1)                          # 新的最大值
l = l0 * torch.exp(m0 - m) + l1 * torch.exp(m1 - m)   # 两边先缩放到同一个最大值，再相加
o = o0 * torch.exp(m0 - m) + o1 * torch.exp(m1 - m)
print(r(m), r(l), r(o), r(o / l))                  # 最后一项是注意力输出

w = torch.softmax(torch.tensor([[0.0, 1.0, 2.0, 3.0]]), dim=-1)   # 不分片：直接对 4 个分数做 Softmax
print([round(x, 4) for x in w.flatten().tolist()], r(w @ torch.tensor([[0.0], [1.0], [2.0], [3.0]])))
a = torch.softmax(torch.tensor([[0.0, 1.0]]), dim=-1) @ torch.tensor([[0.0], [1.0]])
b = torch.softmax(torch.tensor([[2.0, 3.0]]), dim=-1) @ torch.tensor([[2.0], [3.0]])
print(r((a + b) / 2))                              # 错误做法：各分片单独归一化再平均
```

```text
1.0 1.3679 1.0
3.0 1.3679 3.7358
3.0 1.553 3.8711 2.4927
[0.0321, 0.0871, 0.2369, 0.6439] 2.4927
1.7311
```

两个分片各自减去自己的最大值之后，指数和都是 1.3679，看起来一样重要。合并时先对齐到新的最大值 3：分片 0 整体乘上 `exp(1 - 3)`，分片 1 乘 1。合并后的指数和是 1.553，输出 2.4927，和不分片直接算的结果一样。各分片单独归一化再平均，得到的是 1.7311，结果就错了。

真实场景里，K/V 分片在卡之间一轮一轮地传，这就是 Ring Attention：每轮拿到一块分片，算一块分数，合并一次统计。`P` 张卡传 `P - 1` 轮，每个 Query 就看完了整个序列。

通信账本这样估：本地长度是 `S / P`；每轮搬运的 K/V 字节约为 `2 × (S / P) × D × 每个元素的字节数`，开头的 2 是因为 K 和 V 各一份；单卡累计量是轮数乘每轮字节数。`S` 是序列长度，`D` 是隐藏维度。测试里的例子是序列长度 1024、隐藏维度 4096、每个元素 2 字节、`cp_size = 4`：本地长度 256，远端交换 3 轮。扩大 CP 能缩短本地序列，却增加远端轮数。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 张量并行为什么分不掉序列长度，上下文并行怎样沿序列维分片 | 读 |
| Step 2 | 本地 Query 怎样看见全部 K/V：K/V 分片轮转，分块 Softmax 在线合并 | 读，对照上面的公式和小例子 |
| Step 3 | 上下文并行的容量收益和通信代价各是什么，要保留哪些证据 | 读 |
| Step 4 · TODO 1 | 连续切分序列并保证词元不重不漏 | 在 `split_sequence_contiguous` 里写 2 行：算出本地序列长度，沿序列维返回有序分片 |
| Step 4 · TODO 2 | 计算局部 Softmax 的最大值、分母与加权和 | 在 `local_softmax_statistics` 里写 3 行：求最大值，指数求和，指数和 value 相乘 |
| Step 4 · TODO 3 | 在线合并跨分片统计 | 在 `merge_softmax_statistics` 里写 2 行：把两侧统计缩放到同一个 `merged_max` |
| Step 4 · TODO 4 | 建立 Ring K/V 通信账本 | 在 `estimate_ring_kv_communication` 里写 2 行：每轮 payload（K 和 V 各一份），乘上轮数得到单 rank 累计量 |
| 测试 | 检查分片拼接恢复和异常输入、与不分片的注意力（dense）数值等价、大幅值输入的稳定性，以及通信轮数和 payload 的增长规律 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在多张 GPU 上做真实的 K/V 交换，和预先复制全量 K/V 的路径比较 | 需要 `PAR-CUDA-MECHANISM` 环境和两张 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要两张 GPU，默认不执行真实实验。

## 容易踩的坑

- **各分片单独归一化再平均**：Softmax 的分母依赖全部分数。每个分片只能先算三样统计，对齐到同一个最大值之后再合并，最后才相除。
- **合并时没对齐最大值**：先取 `merged_max = max(m, m_i)`，两边分别乘上 `exp(自己的最大值 - merged_max)`，否则分母和加权和都算错。
- **分数没减最大值就取指数**：大幅值输入（比如测试里的 `100.0`）会溢出。先减去本分片的最大值。
- **序列长度不能整除 `cp_size`**：题目要求连续等长分片，不能整除时应当报错，不能悄悄丢掉尾部的词元。
- **每轮 payload 只算了 K 或者只算了 V**：每个分片同时包含 K 和 V，每轮 payload 是两份状态。`cp_size = 1` 时没有远端交换，轮数和累计量都是 0。

## 读完后自查

- [ ] 能说出上下文并行和张量并行各沿哪一维切分。
- [ ] 能写出 Softmax 的三样统计，以及合并两组统计的步骤。
- [ ] 分数 `[0, 1, 2, 3]` 切成两个分片，能解释为什么不能各算各的 Softmax 再平均。
- [ ] 能说出 `P` 张卡的 Ring 路径要远端交换几轮，每轮搬运多少 K/V。

## 下一步

- 打开 notebook：[PAR-CONTEXT. 上下文与序列并行](../PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- 建议先读：[04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA.md)、[29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim.md)
- 做完之后：[46. MoE 专家并行 · 小白入门](./46_MoE_Expert_Parallel.md)
