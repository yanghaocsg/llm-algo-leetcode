# 05. LLaMA3 Block 教程 — 小白讲解

> 对应 notebook：[`05_LLaMA3_Block_Tutorial.ipynb`](./05_LLaMA3_Block_Tutorial.ipynb)

## 一句话总结
本节把 RMSNorm、注意力、SwiGLU 这些零件按真实解码器层的顺序组装起来：`LlamaDecoderLayer.forward` 先做注意力残差更新，再做 MLP 残差更新；`DecoderStack.forward` 让多个块沿同一条主状态顺序堆叠。三个 TODO 分别对应这两次更新的写回和层间传递，测试用结构替身检查调用顺序、形状与梯度是否连通。

## 为什么需要它：痛点是什么
单独看 RMSNorm、RoPE、GQA、SwiGLU 时，每个组件都能说上两句；一旦问“一个 LLaMA 解码器层到底先做什么、残差加在哪一步”，很多同学就只剩一张模糊的框图。常见错误是把归一化和残差的位置记反：以为先加残差再归一化，或者把 MLP 的输入接回块入口，而不是接在注意力更新之后。

生活类比：解码器块像一条有多条匝道的高速公路。主路（残差流）从入口一直通到出口，注意力是一条匝道，负责从别的车道借信息；MLP 是下一条匝道，负责在本地把货物重新打包。每条匝道都要先经过一个限速牌（Pre-Norm）再汇入主路，汇入时是“加上增量”，不是“替换主路”。如果顺序错了，车流就会在错误的入口汇合。

## 核心问题
- 一个 Pre-Norm 解码器块由哪两次“归一化 → 子层 → 残差相加”组成，顺序能不能交换？
- 设输入形状为 `[B, S, C]`，注意力更新和 MLP 更新各自读取什么、写回什么？
- 为什么第二次前置归一化必须读取“已经加过注意力更新”的状态？
- 组件升级（MHA→GQA/MLA、稠密 MLP→MoE、LayerNorm→RMSNorm）时，块外接口必须保持什么？
- 多个块堆叠时，后一层为什么必须读取前一层的完整输出？

## 解决思路：一步步看懂

**第 1 步：先认清主状态与两条更新路径。** 设块输入 $h$ 的形状为 $[B,S,C]$：$B$ 是批量大小，$S$ 是词元数，$C$ 是隐藏宽度。主状态就是每个词元当前的隐藏向量，残差流是它在各层之间持续传递的路径。一个块里有两类变换：注意力在当前词元与可见上下文之间交换信息；MLP（这里是 SwiGLU）只在单个词元内部重组通道。归一化负责把子层输入压到可控尺度，残差连接负责保留主状态并接收增量。notebook 的表格把每个组件的“读取什么 / 写回什么”列得很清楚：前置归一化读取当前主状态，注意力写回上下文增量，MLP 写回通道增量，残差主干把块输入和两个增量合成下一块的主状态。

**第 2 步：把两次更新写成公式。** 设 $h'$ 是第一次残差更新后的状态，$h_{next}$ 是块输出：

$$
h' = h + \mathrm{Attention}(\mathrm{Norm}_1(h)),\qquad
h_{next}=h' + \mathrm{MLP}(\mathrm{Norm}_2(h')).
$$

逐项解释：$\mathrm{Norm}_1(h)$ 是归一化后的子层输入，注意力读取它、产出与 $h$ 同形状的上下文增量；$h + \cdots$ 表示把增量加回**原始**主状态。第二次归一化读的是 $h'$，也就是已经包含注意力更新的中间状态，而不是最初的 $h$；MLP 产出通道增量后再加一次。两条更新不能合并成一次无序求和，因为注意力解决“token 之间”，MLP 解决“token 内部”，而且第二次归一化必须看到第一次更新后的结果。notebook 用不变量强调这一点：注意力更新后输出形状仍与 `h` 一致，MLP 必须读取第一次残差后的状态，块输出要能直接作为下一块输入。

**第 3 步：分清“接口不变”和“内部实现不变”。** 架构演进经常替换组件，但块外部仍要维持主状态的形状、数据类型和残差接口。MHA 换成 GQA 或 MLA，注意力输出仍要回到隐藏宽度；稠密 MLP 换成 MoE，每个 token 的输出仍要回到原位置和原通道；LayerNorm 换成 RMSNorm，归一化的输入输出形状不变。局部组件能产生同形状输出，只说明接口可连接，不代表堆叠后一定稳定——深度增加还会放大初始化、残差尺度、归一化位置和梯度传播之间的相互作用。这正是本节只验证“组装责任”，而把 RMSNorm 的数值机制留给第 01 节、把注意力和 RoPE 留给第 03/04 节的原因。

**第 4 步：TODO 1——写回注意力残差更新。** 参考答案两行：

```python
attention_delta = self.attention_update(self.norm_1(hidden_states))
hidden_states = hidden_states + attention_delta
```

第一行先归一化，再把归一化结果送进注意力替身；第二行把增量加回**未归一化**的主状态。常见写错法有三种：把 `self.norm_1(hidden_states)` 直接当成新状态（丢了残差主干）；写成 `hidden_states = attention_delta`（用增量覆盖主状态）；或者把归一化放到相加之后。`test_residual_update_order` 会把 `attention_update` 换成固定输出 2.0 的 `ConstantUpdate`，如果加错位置，MLP 看到的输入就不是 `x + 2.0`。

**第 5 步：TODO 2——写回 MLP 残差更新。** 参考答案：

```python
mlp_delta = self.mlp(self.norm_2(hidden_states))
hidden_states = hidden_states + mlp_delta
return hidden_states
```

这里的关键是 `self.norm_2(hidden_states)` 中的 `hidden_states` 已经是 TODO 1 更新后的 $h'$。如果把 `norm_2` 接回块入口的原始输入，第二次更新就变成了“基于旧状态算 MLP”，测试里 MLP 的 `seen_input` 会对不上 `x + 2.0`。最后必须 `return`，否则块输出是 `None`，堆叠测试会立刻失败。

**第 6 步：TODO 3——让多个块顺序更新同一条主状态。** 参考答案：

```python
for layer in self.layers:
    hidden_states = layer(hidden_states)
return hidden_states
```

`DecoderStack` 用 `nn.ModuleList` 保存 `num_layers` 个 `LlamaDecoderLayer`，构造函数里对 `num_layers <= 0` 抛 `ValueError("num_layers 必须为正整数")`。循环必须“覆盖”当前状态：第 $l+1$ 层读取第 $l$ 层的完整输出。常见错误是并行收集每层输出再求和，或者循环体里没有赋值，导致所有层都读同一个输入。`test_decoder_stack_order` 用三个分别加 1、2、3 的 `AdditiveBlock`，输入全 0，正确结果是 6；如果漏加某一层或并行覆盖，结果就不是 6。

**第 7 步：结构替身到底替了什么。** `StructureNormStub` 只保留逐通道缩放 `self.scale` 和调用计数 `self.calls`，前向是 `x * self.scale`，**不实现 RMSNorm 的均方根统计**；`AttentionUpdateStub` 只是一个无 bias 的 `nn.Linear(dim, dim)`，**不实现 causal mask、RoPE、MHA 或 GQA**。这样做的目的是把“块组装顺序”从数值细节里隔离出来：测试关心的是谁先被调用、谁读到哪一版状态，而不是注意力分数算得准不准。`LlamaMLP` 则是真实的 SwiGLU 最小实现：`gate_proj`、`up_proj`、`down_proj` 三个无 bias 线性层，前向为 `down_proj(silu(gate_proj(x)) * up_proj(x))`。

**第 8 步：可选延伸——接到最小语言模型。** `TinyLlamaLM` 的结构是 `Embedding → 1 × LlamaDecoderLayer → LM Head`：`embed_tokens` 把 token id 映射成隐藏向量，`decoder` 是刚组装好的块，`lm_head` 是 `Linear(hidden_size, vocab_size, bias=False)`。`build_toy_batch` 用 `torch.randint` 生成 `input_ids`，再把 `labels` 设为右移一位的版本（`torch.roll(input_ids, shifts=-1, dims=1)`），并把最后一个位置置 0 避免越界。`train_one_step` 取 `logits[:, :-1]` 和 `labels[:, 1:]` 做 next-token 交叉熵，然后执行 `zero_grad → backward → step`。`run_minimal_validation` 默认 `vocab_size=16, hidden_size=32, intermediate_size=80, batch_size=2, seq_len=4, num_steps=10, lr=1e-3`，断言 logits 形状为 `(2, 4, 16)`、loss 有限，并打印 loss 是否下降。这段只验证数据流和梯度连通，不是生产级模型。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 解码器块（Decoder Layer） | 一个注意力子层加一个 MLP 子层，各自带归一化和残差 |
| 主状态 / 残差流 | 每个 token 的隐藏向量，沿层一路传递并不断接收增量 |
| Pre-Norm | 先归一化再进子层，残差加回未归一化的主状态 |
| 子层增量（delta） | 注意力或 MLP 产生的“要加回主状态的那部分” |
| `[B,S,C]` | 批量 × 序列长度 × 隐藏宽度 |
| `StructureNormStub` | 只做逐通道缩放、记录调用次数的归一化替身 |
| `AttentionUpdateStub` | 只做线性映射的注意力替身，不实现 mask/RoPE/GQA |
| `LlamaMLP` | 最小 SwiGLU：gate/up/down 三个无 bias 投影 |
| `DecoderStack` | 用 `nn.ModuleList` 把多个块串起来顺序更新状态 |
| 组件替换 | 换注意力/MLP/归一化实现，但保持块外形状与残差接口 |

## 小白常见疑问

**Q1：为什么一定要 Pre-Norm，Post-Norm 不行吗？**
本节不比较两种归一化位置的优劣，只固定一种可验证的组装契约：`norm_1` 读块输入，`norm_2` 读第一次更新后的状态，两次都把增量加回主状态。`test_residual_update_order` 检查的正是这个顺序；如果改成先加残差再归一化，`ConstantUpdate` 记录到的输入和最终输出都会对不上。真实训练中归一化位置的影响还涉及初始化和深度，notebook 把它放在“堆叠后需要观察”的表格里，而不是本节下结论。

**Q2：MLP 为什么不能读最初的 `h`？**
因为第二次前置归一化的输入是 $h'$，即已经包含注意力更新的状态。公式 $h_{next}=h'+\mathrm{MLP}(\mathrm{Norm}_2(h'))$ 写得很明确。若读回最初的 `h`，相当于注意力更新对 MLP 不可见，`test_residual_update_order` 里 `layer.mlp.seen_input` 就不会等于 `x + 2.0`。

**Q3：用 `StructureNormStub` 和 `AttentionUpdateStub` 算不算“偷懒”，没有真正实现模型？**
这是刻意的分工。本节要验证的是块内调用顺序和残差接口，替身能提供同形状、可辨识的输出就够了；RMSNorm 的数值机制由第 01 节验证，RoPE 和 GQA 由第 03、04 节验证。替身还带 `calls` 计数器，正好用来断言每个组件在一次前向里只被调用一次。

**Q4：只要输出形状和输入一样，块就一定写对了吗？**
不一定。`test_block_shape_calls_and_gradients` 除了检查 `out.shape == x.shape`，还检查 `norm_1.calls == 1`、`attention_update.calls == 1`、`norm_2.calls == 1`，以及反传后所有参数都有梯度。形状对但顺序错、或者某条分支没参与计算，仍可能形状正确而语义错误。

**Q5：`DecoderStack` 为什么在构造时检查 `num_layers <= 0`？**
零层或负层无法构成“堆叠”，直接抛 `ValueError` 比等到前向返回原输入更早暴露配置错误。真正的前向循环只负责把每层输出交给下一层。

## 动手验证：怎么知道自己做对了
`run_tests()` 依次跑三个测试，全部通过后打印 `✅ Decoder Block：残差顺序、梯度连通与层堆叠测试通过`；如果 TODO 没填，会先捕获 `NotImplementedError` 并提示“请先按 TODO 1–3 完成 Block 组装”。
- `test_residual_update_order`：把 `norm_1`、`norm_2` 换成 `nn.Identity()`，`attention_update` 换成输出 2.0 的 `ConstantUpdate`，`mlp` 换成输出 3.0 的 `ConstantUpdate`，输入 `torch.ones(1, 2, 4)`。断言 `layer.mlp.seen_input` 等于 `x + 2.0`（MLP 读到第一次更新后的状态），且 `out` 等于 `x + 5.0`（两次增量按 Attention → MLP 顺序累加）。
- `test_block_shape_calls_and_gradients`：`torch.manual_seed(1)`，`hidden_size=8, intermediate_size=12`，输入 `torch.randn(2, 3, 8, requires_grad=True)`。断言输出形状与输入一致、三个组件各调用一次，`out.sum().backward()` 后 `layer.parameters()` 中每个参数的 `.grad` 都不为 `None`。
- `test_decoder_stack_order`：三层 `DecoderStack(hidden_size=4, intermediate_size=6, num_layers=3)`，把 `layers` 换成三个 `AdditiveBlock(1.0/2.0/3.0)`，输入全 0，断言输出全为 6.0，证明每层读取的是上一层输出而不是并行覆盖。
notebook 没有保存这些单元的执行输出，因此以代码和断言为准。可选的 `run_minimal_validation()` 会打印每一步 loss，并断言 logits 形状为 `(batch_size, seq_len, vocab_size)`、loss 有限；它只说明最小训练闭环能跑通，不代表模型质量。
