# 00. PyTorch 热身 — 小白讲解

> 对应 notebook：[`00_PyTorch_Warmup.ipynb`](./00_PyTorch_Warmup.ipynb)

## 一句话总结
Part 02（解码器组件的基础实现）的热身题：动手写模型组件之前，先练三件最基本的事——按维度含义重排张量（tensor，多维数组）、用 token id（词元编号）到词表里查出向量、手写"线性变换 + ReLU"的前向和反向传播。它们串成本部分所有组件共用的那条路径：token id → 连续表示 → 沿模型主干前向 → 损失把梯度传回参数。

## 为什么需要它：痛点是什么
后面的每个组件（RMSNorm、激活、注意力）都要过同样三道关，而且错了往往不报错——形状对、数值错，是最难查的 bug。第一关是轴（axis，张量的某一维）的含义：`[B, C, H, W]` 直接 `reshape` 成 `[B, H×W, C]`，元素总数一样、程序照跑，但通道和空间位置已经被搅乱。第二关是词嵌入：不清楚它只是"按编号取一行"，后面改权重、做量化时就无从下手。第三关是梯度：只会调 `.backward()`，一旦要给自己的算子写反向，就不知道前向该留下什么。

生活类比：查字典 + 对账。tokenizer（分词器）把一句话切成 token 并编上号；Embedding 就是一本字典，编号是行号，翻到那一行把"解释"（向量）抄出来。反向传播像财务对账：发现总额（损失）不对，就沿每笔记录倒着查，算出每个经手人（参数）各该担多少责任——要能倒查，就得在前向时把单据留好。

## 核心问题
- 张量换形状时，怎样保证每个元素仍然带着原来的含义（哪个空间位置、哪个通道）？
- token id 怎样变成向量？`nn.Embedding` 和"直接用编号去索引权重"是不是一回事？
- 自己写 `torch.autograd.Function` 时，前向要保存哪些东西，反向才能算出输入、权重、偏置三种梯度？
- ReLU 的"门控状态"是什么，为什么反向传播离不开它？

## 解决思路：一步步看懂

**第 1 步：重排张量 = 先调轴的顺序，再合并轴。** 记 B（batch，一批几个样本）、C（channel，通道数）、H、W（特征图的高和宽）。目标是把 `[B, C, H, W]` 整理成 `[B, H×W, C]`：每个空间位置当成一个 token，该位置的 C 个通道值是它的特征。

```python
x_native = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
x_einops = einops.rearrange(x, 'b c h w -> b (h w) c')
```

`permute(0, 2, 3, 1)` 只调换轴的先后顺序，得到 `[B, H, W, C]`；再把 H、W 两轴合并成一根长 `H×W` 的序列轴。顺序是关键：这样展平后第 `h×W + w` 个 token 正是原特征图的 `(h, w)` 位置。拿测试输入 `torch.arange(2*3*4*5).reshape(2,3,4,5)` 手算：输出形状 `(2, 20, 3)`，第一个 token 是 `x[0, :, 0, 0]`，即 0 号空间位置上 3 个通道的值 0、20、40——测试正是用 `torch.equal(native[0,0], x[0,:,0,0])` 卡这一点。第二行的 `einops.rearrange` 用字符串 `'b c h w -> b (h w) c'` 写出同一件事（括号表示把两轴合并），必须与第一种逐元素一致。notebook 还点明多头表示同理：`[B, H, S, D] → [B, S, H×D]`，词元顺序不变，只合并注意力头和头维度（这一条只在表格里给了目标形状，没有对应的 TODO）。

**第 2 步：Embedding 就是按行号取向量。** `input_ids` 形状 `[B, S]`（S 是序列长度），类型必须是 `torch.long`，取值落在 `[0, V)`（V 是词表大小）；词嵌入权重形状 `[V, D]`（D 是隐藏维度），第 i 行就是第 i 个 token 的向量；查表结果 `[B, S, D]`，与输入位置一一对应。参考答案先建 `emb_layer = nn.Embedding(vocab_size, hidden_dim)`，用 `normal_(0, 0.1)` 填充权重，然后走两条路：`out_official = emb_layer(input_ids)` 走模块查表，`out_manual = emb_layer.weight[input_ids]` 用同一份权重按行做高级索引。两条路必须逐元素一致，因为 token id 的含义就是"词表行号"。越界编号在查表时会出错，所以已给出的 `validate_token_ids` 先查三件事：二维、`torch.long`、最小值 ≥ 0 且最大值 < vocab_size（空张量直接放行）。

**第 3 步：前向要为反向"留证据"。** 要实现的算子是 `relu(x @ W^T + b)`：x 是输入，W 是形状 `[out_features, in_features]` 的权重，b 是每个输出通道一个数的偏置，`F.linear(x, weight, bias)` 算的就是 `x @ W^T + b`。记激活前值为 z、输出 y = relu(z)。参考答案的前向只做三件事再存档：`z = F.linear(x, weight, bias)`，`y = F.relu(z)`，`mask = z > 0`，然后 `ctx.save_for_backward(x, weight, mask)`。mask 就是"激活门控状态"：relu 把 z ≤ 0 的位置截成 0，这些位置对损失没有影响，反向时梯度也必须挡成 0。反向只需要知道"通不通"，用不上具体的 z，所以存 mask 就够——保存的状态越多，反向越直接，但训练显存占用也越高。

用 `test_linear_relu_forward` 的固定输入手算：x = [[1, −2], [0.5, 3]]，W = [[1, 1], [−1, 0.5]]，b = [0, −0.5]。第一行 z = [1−2+0, −1−1−0.5] = [−1, −2.5]；第二行 z = [0.5+3, −0.5+1.5−0.5] = [3.5, 0.5]。于是 mask = [[False, False], [True, True]]，y = [[0, 0], [3.5, 0.5]]。

**第 4 步：反向按链式法则把梯度拆成三份。** 链式法则：损失对某个量的梯度 = 上游梯度 × 局部导数。上游梯度 `grad_output` 是损失对 y 的梯度，形状与 y 相同。

```python
grad_z = grad_output * mask                  # 门控：激活关闭的位置梯度为 0
grad_x = grad_z @ weight                     # 与 x 同形
grad_weight = grad_z.transpose(-1, -2) @ x   # 与 weight 同形
grad_bias = grad_z.sum(dim=0)                # 与 bias 同形；汇总 batch 维
```

为什么长这样：z = xWᵀ，输入的每个特征都会经过 W 的一列影响所有输出通道，所以输入梯度是 `grad_z @ W`；W 的第 (j, k) 个元素只进入输出通道 j，所以权重梯度是"输出通道 j 的梯度 × 输入特征 k"再跨样本求和，即 `(grad_z)ᵀ @ x`；b 是整体加到每个样本上的，梯度就是 grad_z 沿 batch 维求和。接着上例，对 `y.sum()` 反向（上游梯度全是 1）：grad_z = [[0, 0], [1, 1]]；grad_x 第二行 = [1−1, 1+0.5] = [0, 1.5]，第一行全 0（这一行的两个输出都被截断，梯度到不了输入）；grad_weight = [[0.5, 3], [0.5, 3]]（第 0 行样本没有贡献，第 1 行样本 [0.5, 3] 被两个输出通道各用一次）；grad_bias = [1, 1]。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 张量 / 轴 | 多维数组；每根轴要有明确含义（batch、通道、序列……） |
| B / C / H / W / S / D | 批次大小 / 通道数 / 高 / 宽 / 序列长度 / 隐藏维度 |
| token / token id | 词元（模型处理文本的最小单位）/ 它在词表里的行号 |
| tokenizer（分词器） | 把文本切成 token 并映射成编号 |
| Embedding（词嵌入） | `[V, D]` 的可训练查表：按 token id 取出对应那一行 |
| permute / einops.rearrange | 调换轴顺序 / 用字符串写明"轴怎么重排、怎么合并"的工具 |
| 前向 / 反向传播 | 从输入算输出 / 把损失对输出的梯度沿计算图传回输入和参数 |
| torch.autograd.Function | 自定义算子的基类：自己写 forward 和 backward |
| ctx.save_for_backward | 前向里登记"反向要用到的张量" |
| 上游梯度（grad_output） | 损失对本算子输出的梯度，是反向计算的起点 |
| 门控状态（mask） | 记录 relu 哪些位置是通的（激活前值 > 0） |

## 小白常见疑问

**Q1：为什么不直接 `x.reshape(B, H*W, C)`？**
元素总数对得上，不会报错，但语义全错：reshape 只是把排好的数字按新形状重新切分，而 `[B, C, H, W]` 里排在最前面的是整张通道 0 的特征图，切出来的"token"会把不同位置、不同通道的数混在一起。必须先用 permute 把通道轴挪到最后，再合并 H、W。

**Q2：`nn.Embedding` 和自己写 `weight[input_ids]` 有区别吗？**
在本节的用法里没有区别：两者读的是同一份 `emb_layer.weight`，`nn.Embedding` 只是把这张 `[V, D]` 的表包装成模块、登记为可训练参数。查表里没有乘法，就是把那一行拷出来；输出 `[B, S, D]` 的前两轴与 `input_ids` 一一对应，词元顺序原样保留。

**Q3：为什么不用保存偏置 b？它不需要梯度吗？**
需要梯度，但算它的梯度用不到 b 自己：b 是整体加到每个样本上的，局部导数是 1，`grad_bias` 就是 `grad_z` 沿 batch 维求和。判断"存不存"的原则是：反向公式里出现谁，才存谁。

**Q4：测试为什么先把 `.grad` 清成 None 再跑自己的反向？**
PyTorch 的 `.backward()` 默认把梯度累加到 `.grad` 上而不是覆盖。不清空的话，第二次得到的是"参考梯度 + 自己的梯度"，比较就失去了意义。

## 动手验证：怎么知道自己做对了

`run_warmup_tests` 依次跑五个测试，全过才打印 `✅ Warmup：输入、重排、查表、前向与反向契约测试通过`；有 TODO 没填会抛 `NotImplementedError`，先打印"请先完成 TODO 1–4，再运行测试。"再把异常抛出来。
- `test_token_id_contract`：`[[0, 2], [1, 3]]`（vocab_size=4）应通过；一维的 `[0, 1]`、浮点的 `[[0.0, 1.0]]`、越界的 `[[0, 4]]` 都必须抛 `ValueError`。
- `test_spatial_to_sequence_rearrangement`：两种重排形状都是 `(2, 20, 3)` 且 `torch.equal`；`native[0,0]` 等于 `x[0,:,0,0]`；三维输入 `torch.randn(2,3,4)` 被拒绝。
- `test_embedding_lookup_equivalence`：`embedding_warmup(torch.tensor([[0,3],[2,1]]), 4, 6)` 的两路输出形状 `(2, 2, 6)` 且 `torch.allclose`。
- `test_linear_relu_forward`：输出与 `F.relu(F.linear(x, weight, bias))` allclose；激活前值小于 0 的位置输出必须为 0。
- `test_linear_relu_backward`：x `randn(2,4)`、weight `randn(3,4)`、bias `randn(3)`，先让 Autograd 算出参考梯度，清空后跑 `LinearReLUFunction.apply(x,weight,bias).sum().backward()`，三份梯度分别 `torch.allclose`。以上比较都没显式给容差，用的是 `torch.allclose` 的默认值。
