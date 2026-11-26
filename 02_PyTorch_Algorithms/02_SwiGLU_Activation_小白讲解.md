# 02. SwiGLU 激活 — 小白讲解

> 对应 notebook：[`02_SwiGLU_Activation.ipynb`](./02_SwiGLU_Activation.ipynb)

## 一句话总结
SwiGLU 是 LLaMA 系模型前馈层（FFN，也叫 MLP）用的门控激活：输入同时走两条分支，一条给出候选特征（up），另一条决定每个特征放行多少（gate），两者逐元素相乘后再投回隐藏维。本节用四个 TODO 搭出完整的 `SwiGLU_MLP`，顺带讲清一个工程细节——多了一条分支之后，中间维为什么要从 `4d` 改成约 `8d/3`。

## 为什么需要它：痛点是什么
Transformer 的前馈层一般是"升维 → 激活 → 降维"。激活函数（activation，对每个数单独做的非线性变换，如 ReLU、GELU）在这条路上只有一个固定规则：所有中间特征一起过同一种变换，模型没办法针对当前的 token 显式决定"这个特征该放大、那个该压住"。

生活类比：调音台。单路 FFN 像给所有声道装同一个固定均衡器，不管谁在唱都一样处理。SwiGLU 是给每个声道单独配一位调音师：歌手（up 分支）唱出内容，调音师（gate 分支）根据内容决定这个声道的推子推多少。而且这位调音师不止"开/关"两档——可以推得很高，也可以拉成反相。

## 核心问题
- SwiGLU 比普通 FFN 多了什么？门控值为什么不能当成"概率"？
- 中间维为什么从 `4d` 变成约 `8d/3`？这个数是怎么算出来的？
- 怎样把中间维向上对齐到 `multiple_of` 的倍数？
- 融合的 gate/up 投影是什么意思，前向时怎么切分？

## 解决思路：一步步看懂

**第 1 步：从单路 FFN 到双路门控。** 记 $d$ 为模型隐藏维、$h$ 为中间维（前馈层内部升到的宽度）。三种结构的中间路径：FFN 是 `up → activation → down`，所有中间维一起经过同一种激活；GLU（Gated Linear Unit，门控线性单元）增加一条门控分支，变成 `content × sigmoid(gate) → down`，让一组特征控制另一组候选内容；SwiGLU 再把有界的 Sigmoid 门换成平滑的 SiLU 门，变成 `up × SiLU(gate) → down`，每个中间特征获得独立的选择强度。写成式子：

$$\mathrm{out}=W_{down}\,\big(\mathrm{SiLU}(W_{gate}\,x)\odot W_{up}\,x\big)$$

其中 $x$ 是某个 token 的输入向量；$W_{gate}$、$W_{up}$ 是把 $x$ 投到中间维的两个权重矩阵（代码里合并在 `gate_up_proj` 里）；$W_{down}$ 把结果投回隐藏维（`down_proj`）；$\odot$ 表示逐元素相乘。括号里的 $W_{gate}\,x$ 就是代码里的 `gate`，$W_{up}\,x$ 就是 `up`。

**第 2 步：门控值不是概率，也不是开关。** SiLU 的定义是 $\mathrm{SiLU}(x)=x\cdot\sigma(x)$，$\sigma$ 是 Sigmoid（把任意实数压到 0 到 1 之间）。因为又乘了一个 $x$，门控值就不再被限制在 0 到 1：门的输入是 2 时，$\mathrm{SiLU}(2)=2\times0.881\approx1.76$，比 1 大；输入是 −1 时约为 −0.27，是负数。所以 notebook 强调门控值"可为负且不受 1 限制"，不能解释成概率或二值掩码——它是一个平滑的缩放系数。另一条约束是两条中间分支必须在同一个中间特征空间里逐元素对应：宽度相同，第 $i$ 个门只管第 $i$ 个候选特征，不是两组可以任意配对的特征。

**第 3 步：容量对齐——`8d/3` 是怎么来的。** 标准 FFN 有两个投影（up、down），参数量约 $2dh$；SwiGLU 有三个（gate、up、down），参数量约 $3dh$。如果沿用同样的中间宽度，参数量和计算量都会明显增加。公平比较两种结构要先固定"容量口径"（参数预算）：标准 FFN 取 $h=4d$，参数量 $2dh=8d^2$；让 SwiGLU 的 $3dh$ 也等于 $8d^2$，解得 $h\approx 8d/3\approx2.67d$。这就是 LLaMA 风格里中间维约等于 $8/3\times$ 隐藏维的由来。

**第 4 步：`calculate_intermediate_size`——先算理论值，再向上对齐。** 参考答案两行：`theoretical_size = int(hidden_size * 8 / 3)` 按 8/3 的容量关系得到理论中间维（`int` 直接舍去小数）；`aligned_size = ((theoretical_size + multiple_of - 1) // multiple_of) * multiple_of` 把它向上对齐到 `multiple_of` 的倍数（默认 256）。第二行是"向上取整到某倍数"的标准写法：先加 `multiple_of - 1` 再做整除，保证结果不小于理论值、又是 `multiple_of` 的倍数——notebook 的解析特意说明这里用向上取整，不是四舍五入。拿测试里的配置手算：
- `hidden_size=4096, multiple_of=256`：$4096\times8/3=10922.67\to10922$；$(10922+255)//256=43$；$43\times256=11008$。
- `hidden_size=64, multiple_of=16`：$64\times8/3=170.67\to170$；$(170+15)//16=11$；$11\times16=176$。

**第 5 步：`SwiGLU_MLP`——一次投影出两支，再切分。** `__init__` 和 `forward` 的参考答案：

```python
self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
...
gate_up = self.gate_up_proj(x)                 # [..., 2 * intermediate_size]
gate, up = torch.chunk(gate_up, 2, dim=-1)     # 各 [..., intermediate_size]
gated = torch.nn.functional.silu(gate) * up
return self.down_proj(gated)
```

两个 `nn.Linear` 都设 `bias=False`（无偏置）。"融合 gate/up 投影"是指不单独建两个矩阵，而是让一个线性层直接输出 `2 × intermediate_size` 宽，前向时用 `torch.chunk(..., 2, dim=-1)` 沿最后一维切成等宽的两半：前一半是 `gate`，后一半是 `up`。`silu(gate) * up` 是逐元素相乘，结果宽度是 `intermediate_size`；最后 `down_proj` 投回 `hidden_size`，MLP 输出恢复为模型隐藏维，才能继续进入残差路径（残差相加与归一化是模型块外层的关系，不在这个模块里）。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| FFN / MLP | 前馈层：升维 → 激活 → 降维，Transformer 里注意力之外的另一个子层 |
| 激活函数 | 对每个数单独做的非线性变换，如 ReLU、GELU、SiLU |
| 门控（gating） | 用一组数决定另一组数各自通过多少 |
| GLU | `content × sigmoid(gate)`：用 Sigmoid 当门的门控线性单元 |
| SwiGLU | 把 GLU 的 Sigmoid 门换成 SiLU 门 |
| SiLU | $\mathrm{SiLU}(x)=x\cdot\sigma(x)$：平滑，可为负，不限于 1 以内 |
| up / gate / down | 候选特征分支 / 门分支 / 把结果投回隐藏维的投影 |
| 融合投影（`gate_up_proj`） | 一个线性层同时算出 gate 和 up，输出宽度 2 × intermediate_size |
| `torch.chunk` | 沿指定维把张量切成若干等宽的块 |
| 容量对齐 | 调中间维让两种结构的参数量大致相同，便于公平比较 |
| `multiple_of` | 中间维要对齐到的倍数，默认 256 |

## 小白常见疑问

**Q1：门控值大于 1 或者是负数，不会出问题吗？**
这正是设计意图。门是平滑的缩放系数：大于 1 是放大，负数是反相，接近 0 是压住。相乘后的结果还要经过 `down_proj` 重新组合并恢复为模型隐藏维，残差相加与归一化由模型块的外层处理。

**Q2：gate 和 up 为什么融成一个投影，不分开写两个 `nn.Linear`？**
本节的契约就是融合写法：`test_swiglu_projection_contract` 直接检查 `gate_up_proj` 这个层的输入输出维度，分开写就没有这个属性，测试过不了。融合的动机属于算子融合的话题，notebook 把"算子融合导论"放在相关阅读里。

**Q3：`torch.chunk` 切出来的前半是 gate 还是 up？弄反了要紧吗？**
要紧。参考答案和测试都用 `gate, up = torch.chunk(gate_up, 2, dim=-1)`：前一半过 SiLU 当门，后一半当候选。弄反了算的是 `SiLU(up) * gate`，和测试里的参考数值对不上。

**Q4：中间维为什么一定要向上对齐，四舍五入不行吗？**
规则是"对齐后不小于理论维度，且是指定倍数"。四舍五入在理论值更靠近下方倍数时会把中间维压小，低于容量对齐的口径；题目把这一步叫"硬件倍数对齐"，并用 `4096 → 11008` 这样的已知配置来检查。

## 动手验证：怎么知道自己做对了

`run_swiglu_tests` 依次跑三个测试，全过才打印 `✅ SwiGLU：容量、投影与门控前向测试通过`。测试区是直接调用，TODO 没填会抛出 `NotImplementedError`。
- `test_swiglu_intermediate_size`：`calculate_intermediate_size(4096, multiple_of=256)` 必须等于 11008，`(64, multiple_of=16)` 必须等于 176；`(0, 16)` 和 `(64, 0)` 两组非法参数必须抛 `ValueError`。
- `test_swiglu_projection_contract`：`SwiGLU_MLP(hidden_size=8, intermediate_size=16)` 的 `gate_up_proj` 是 8 进 32 出，`down_proj` 是 16 进 8 出，两个层的 `bias` 都是 None。
- `test_swiglu_gated_forward`：固定 `torch.manual_seed(7)`，hidden 4、intermediate 6，输入 `torch.randn(2, 3, 4)`；用模块自己的投影按同一公式 `mlp.down_proj(silu(gate) * up)` 算参考值，断言 `output.shape == x.shape`、`torch.allclose(output, expected)`（未显式给容差），再 `output.sum().backward()` 检查 `x.grad is not None`。
