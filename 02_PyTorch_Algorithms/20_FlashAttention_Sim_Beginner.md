# 20. FlashAttention Sim Beginner | FlashAttention 模拟 · 小白入门

> 配套 notebook：[20. FlashAttention 模拟](./20_FlashAttention_Sim.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

注意力计算会先生成一张很大的中间分数表，序列越长它越大。这本 notebook 教你把这张表切成小块逐块计算，并且得到和完整计算一样的结果。

## 为什么需要它

注意力的输入是三个形状相同的矩阵 `Q`、`K`、`V`，每一行对应一个词元。标准做法分三步：先让 `Q` 的每一行和 `K` 的每一行两两算相似度，得到一张分数表；再对每一行做 Softmax，变成加起来等于 1 的权重；最后用这些权重对 `V` 的行加权平均。

麻烦出在分数表上。序列长度是 `N` 时，它是 `N × N` 的：`N` 等于 4 时只有 16 个数，`N` 等于 1024 时就有 1,048,576 个数。它不是最终输出，却可能作为一大块中间数据写进显存，再读出来做 Softmax。

能不能算一小块、用一小块？障碍是 Softmax 的分母要用一整行的分数，只拿到半行时算不出最终权重。对每一小块单独做 Softmax 再拼起来，每一块的分母只覆盖了自己的位置，权重就错了。FlashAttention 的办法是给每个查询行留三个量：目前见过的最大分数、指数和、累计输出。每来一个新小块，先把旧状态换算到新的基准上，再合并进去。这本 notebook 在 CPU 上把这套做法走一遍，并和完整注意力对照结果。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| `Q`、`K`、`V` | 注意力的三路输入，形状都是 `[seq_len, dim]`，每一行对应一个词元。`Q` 的行和 `K` 的行算出分数，分数变成权重后去加权 `V` 的行 |
| 分数矩阵 | `Q` 的每一行对 `K` 的每一行算出的相似度表，大小是 `N × N`。本节所有讨论都围绕它 |
| Softmax | 把一行分数变成一组加起来等于 1 的正权重：先每个数取指数，再除以这一行指数的总和 |
| 分块（tile） | 词元维度上的一小段，比如一次只取 2 个查询和 2 个键。它是算法上的切分，不是模型块，也不是 CUDA 线程块 |
| 逐行状态 | 每个查询行自己保存的三个量：最大值 `m`、指数和 `l`、累计输出 `O`。它们不是整个分块共用一个数 |
| 因果约束（causal） | 每个词元只能看到自己和之前的位置，之后的位置权重为零 |
| 工作集 | 计算过程中当前必须暂存的中间数据。完整算法要暂存整张分数矩阵，分块算法只暂存当前这一小块 |

## 核心思路

输入是三个 `[seq_len, dim]` 的张量。程序用两层循环：外层每次取一段查询，内层让这段查询依次去见每一段键值。每见一段，就算出一小块局部分数，并进逐行状态。全部键值见完，这段查询的输出就和完整注意力一样了。

每行记住三个量：目前的最大分数 `m`、到目前为止的指数和 `l`、累计输出 `O`，一开始分别是负无穷、0、0。新小块到来时按下面的式子更新，其中 `S_ij` 是当前的局部分数，`V_j` 是当前的值分块：

```text
m' = max(m, rowmax(S_ij))        # 新的最大值：历史的和当前小块的，取大的
α  = exp(m - m')                 # 历史贡献的缩放因子
W  = exp(S_ij - m')              # 当前小块每个分数的指数权重
l' = α * l + rowsum(W)           # 新的指数和：换算后的历史 + 当前小块
O' = (α * l * O + W @ V_j) / l'  # 新的输出：历史输出和当前贡献按同一比例合并
```

`rowmax` 和 `rowsum` 是对小块逐行取最大值、求和，并保留列维。`α` 用来换算基准：最大值从 `m` 换成 `m'` 以后，以前的指数都是相对旧基准的，要乘上 `exp(m - m')`。一直减最大值，是为了避免对很大的分数直接取指数而溢出。因果约束也在取最大值之前处理：先把未来位置的分数设成负无穷，它的指数权重就是 0。

下面看一个查询行、三个键的小例子。三个分数是 `1、2、3`，按 `block_size = 2` 切成两块：`[1, 2]` 和 `[3]`。每个键对应的值只有 1 个数：`10、20、30`。

```python
import torch

blocks = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]])]   # 一个查询行的分数，按 K 方向切成两块
v = torch.tensor([[10.0], [20.0], [30.0]])                      # 3 个键对应的值，每个只有 1 个数

m = torch.full((1, 1), -float('inf'))   # 这一行的最大值
l = torch.zeros(1, 1)                   # 这一行的指数和
out = torch.zeros(1, 1)                 # 这一行的输出
pos = 0
for scores in blocks:
    v_block = v[pos:pos + scores.shape[1]]
    pos += scores.shape[1]
    tile_max = torch.max(scores, dim=-1, keepdim=True)[0]
    new_max = torch.maximum(m, tile_max)
    history_scale = torch.exp(m - new_max)
    tile_weights = torch.exp(scores - new_max)
    new_l = l * history_scale + tile_weights.sum(dim=-1, keepdim=True)
    out = out * (l * history_scale / new_l) + (tile_weights @ v_block) / new_l
    m, l = new_max, new_l
    print(new_max.item(), round(new_l.item(), 4), round(out.item(), 4))

print(torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1) @ v)
print(torch.exp(torch.tensor([1000.0])), torch.exp(torch.tensor([1000.0]) - 1000.0))
```

```text
2.0 1.3679 17.3106
3.0 1.5032 25.7521
tensor([[25.7521]])
tensor([inf]) tensor([1.])
```

前两行每行三个数，依次是处理完这一块后的 `m`、`l`、`O`。第二块只有分数 3，比历史最大值 2 大，所以历史部分要先乘上 `exp(2 - 3)`，约等于 0.3679，再并进去。第三行是完整 Softmax 的结果，和第二行的 25.7521 一致。最后一行说明分数是 1000 时直接取指数会溢出，减掉最大值再取就是 1。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 注意力的中间工作集为什么大，分块怎样缩小它 | 读 |
| Step 2 | 在线 Softmax 怎样用逐行状态保持和完整 Softmax 数值等价 | 读，对照上面的式子和小例子 |
| Step 3 | 一个 Q 分块怎样依次吸收全部 K/V 分块，尾部分块可以更短 | 读 |
| Step 4 · TODO 1 | 初始化跨分块状态 | 在 `flash_attention_forward_sim` 里写 2 行：`m` 形状 `[seq_len, 1]`、初值负无穷，`l` 形状相同、初值 0 |
| Step 4 · TODO 2 | 形成局部分数与因果约束 | 同一函数里写 2 行：用 `q_block` 和 `k_block` 算 `scores`；因果时用全局位置写 `future_mask` |
| Step 4 · TODO 3 | 执行数值重标定 | 写 4 行：`tile_max`、`new_max`、`history_scale`、`tile_weights` |
| Step 4 · TODO 4 | 合并归一化输出 | 写 2 行：`new_l` 和 `out_i`，分子和分母用同一个重标定比例 |
| 测试 | 检查非因果和因果路径是否与完整注意力等价、大分数下输出是否有限、`dtype` 是否不变，以及工作集代理的计算 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上比较显式分数矩阵的基线和 PyTorch SDPA 候选，记录输出误差、延迟、峰值显存和实际算子路径 | 需要 `INF-CUDA-MECHANISM` 环境和一张 NVIDIA GPU；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。SDPA 指 PyTorch 的 `scaled_dot_product_attention`，它会按设备、数据类型、形状和运行时能力选择执行后端。

## 容易踩的坑

- **每块单独做 Softmax 再相加**：同一查询行的归一化分母覆盖全部 K/V，必须用逐行状态 `m`、`l`、`O` 跨分块合并。
- **状态写成分块共享的标量**：`m`、`l` 的形状是 `[seq_len, 1]`，每个查询行各有一个值。`m` 以负无穷开始，`l` 以 0 开始，并保留列维以便逐行广播。
- **因果掩码用了分块内部下标**：因果约束比较的是全局词元位置，即 `key_pos > query_pos`。而且要在取最大值之前就把未来位置设成负无穷。
- **分子和分母用了不同的重标定比例**：历史指数和、历史输出和当前权重必须迁移到同一个最大值基准，否则 Softmax 不再等价。
- **输出的数据类型变了**：测试用 `float64` 输入检查输出仍是 `float64`，并用全是 100 的 `Q`、`K` 检查输出有限。各状态都要沿用输入的 `dtype`。

## 读完后自查

- [ ] 能说出序列长度为 `N` 时分数矩阵有多少个元素，分块后当前暂存的是多大一块。
- [ ] 能说出每个查询行保留哪三个跨分块状态，以及它们的初始值。
- [ ] 新小块的最大值比历史最大值大时，能写出历史状态要乘的缩放因子。
- [ ] 能解释因果掩码为什么要用全局位置比较，并且要在取最大值之前生效。

## 下一步

- 打开 notebook：[20. FlashAttention 模拟](./20_FlashAttention_Sim.ipynb)
- 相关 notebook：[22. vLLM 分页注意力](./22_vLLM_PagedAttention.ipynb)
- 做完之后：[INF-EXECUTION-DISPATCH. 推理融合与后端分派 · 小白入门](./INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch_Beginner.md)
