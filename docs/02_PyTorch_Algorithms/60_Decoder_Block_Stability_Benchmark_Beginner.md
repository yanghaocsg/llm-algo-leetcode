# 60. Decoder Block Stability Benchmark Beginner | Decoder Block 稳定性基准 · 小白入门

> 配套 notebook：[60. Decoder Block 稳定性基准](./60_Decoder_Block_Stability_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU-first；GPU 可选

## 一句话说清

Decoder Block 是大模型里重复堆叠的基本单元，里面的归一化、激活函数可以有不同的选法。这本 notebook 教你怎样公平地比较这些选法：一次只改一个地方，再检查结果。

## 为什么需要它

大模型由很多个 Decoder Block 叠起来。每个 Block 里有归一化、注意力、前馈网络和两条残差路径。零件的名字一样，选法却不一样：

- 归一化放在子层前面（Pre-Norm），还是放在残差相加之后（Post-Norm）。
- 归一化用 RMSNorm，还是 LayerNorm。
- 前馈网络的激活函数用 SwiGLU，还是 GELU。

这些选择会改变梯度的路径、激活值的大小和参数量。想比较它们，最容易犯的错是一次改了好几个地方。比如同时换了归一化的位置和类型，结果变好了，你也说不清是哪一个带来的。

这本 notebook 的做法是把每种选法写成一个候选，候选和基线只差一个字段，其余条件全部固定。这样结果的变化才能归因到那一个字段上。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| Decoder Block | 大模型里重复堆叠的一层，里面包含归一化、注意力和前馈网络 |
| 残差路径（residual path） | 把子层的输入直接加到输出上的那条通道，让信息可以跳过子层 |
| 基线 / 候选（baseline / candidate） | 基线是拿来对照的标准方案，候选是你要检验的改动 |
| 受控比较 | 除了要研究的那一项，其他条件全部保持不变 |
| 候选字段 | 这里指 `norm_position`、`norm_kind`、`activation` 三个可以选的结构项 |
| 证据等级 | 说明一条结论有多少依据，比如只有 CPU 探针，还是有 GPU 实测 |
| `accept` / `tune` / `reject` | 决策的三种结果：采用、继续调整、拒绝 |

## 核心思路

这本 notebook 用一张表记录一种 Block 设计，表里有三个候选字段。基线的选法是 `pre_norm`、`rmsnorm`、`swiglu`。

比较之前先检查：候选和基线是不是只差一个字段。

```python
baseline = {'norm_position': 'pre_norm',  'norm_kind': 'rmsnorm',  'activation': 'swiglu'}
one_change = {'norm_position': 'post_norm', 'norm_kind': 'rmsnorm', 'activation': 'swiglu'}
two_changes = {'norm_position': 'post_norm', 'norm_kind': 'layernorm', 'activation': 'swiglu'}

changed = [k for k in baseline if baseline[k] != one_change[k]]
print(changed, len(changed) == 1)

print([k for k in baseline if baseline[k] != two_changes[k]])
```

```text
['norm_position'] True
['norm_position', 'norm_kind']
```

第一个候选只改了归一化的位置，可以比较。第二个候选改了两处，结果无法归因，不应进入实验。

检查通过以后，整个流程分四步：

```text
校验规格 → 确认只改一个字段 → 运行稳定性探针 → 给出 accept / tune / reject
```

稳定性探针是在 CPU 上把几层 Block 叠起来跑一次前向传播和反向传播，记录损失、输入的梯度范数、激活值的标准差，以及结果是否都是有限值。它只能说明结构能不能跑、梯度能不能传回去，不能说明模型质量。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 把 Block 的选法写成候选，每个候选只改一个字段 | 读 |
| Step 2 | Pre-Norm 和 Post-Norm、RMSNorm 和 LayerNorm 分别改变了什么 | 读 |
| Step 3 | 质量、稳定性、成本要分开记录，CPU 能查什么，GPU 才能测什么 | 读 |
| Step 4 · TODO 1 | 校验 Block 规格 | 写 `validate_block_spec`：检查三个候选字段和维度，返回 `ready` 和 `issues` |
| Step 4 · TODO 2 | 识别唯一结构差异 | 写 `validate_single_block_change`：比较基线和候选，只允许一个字段不同 |
| Step 4 · TODO 3 | 运行稳定性探针 | 写 `run_block_stability_probe`：叠 4 层 Block，记录损失、梯度范数和激活值标准差 |
| Step 4 · TODO 4 | 汇总质量—成本决策 | 写 `decide_block_candidate`：返回 `accept`、`tune` 或 `reject` |
| 测试 | 四个测试依次检查规格、单变量比较、探针和决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用两个只差一个 Block 结构的 checkpoint，在 GPU 上补质量、步时和峰值显存 | 有 GPU 和匹配的 checkpoint 再做，默认关闭 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和一对匹配的 checkpoint，默认关闭。

## 容易踩的坑

- **一次改了两处**：同时换归一化的位置和类型，结果无法归因。候选必须只改一个字段，并且和声明的字段一致。
- **SwiGLU 的宽度是奇数**：SwiGLU 的门和值要等宽，`d_ff` 必须是偶数，否则规格不合法。
- **候选偷偷改了宽度**：`d_model` 和 `d_ff` 必须和基线相同，否则比较不公平。
- **把 CPU 探针当成质量结论**：只有 CPU 证据时，决策最多是 `tune`。要 `accept`，必须有 GPU 上匹配 checkpoint 的质量和成本证据。

## 读完后自查

- [ ] 能说出基线的三个候选字段和它们的取值。
- [ ] 给一个候选，能判断它和基线是不是只差一个字段。
- [ ] 能说出稳定性探针记录了哪四个量。
- [ ] 能解释为什么只有 CPU 证据时不能给出 `accept`。

## 下一步

- 打开 notebook：[60. Decoder Block 稳定性基准](./60_Decoder_Block_Stability_Benchmark.md)
- 建议先读：[01. RMSNorm 教程 · 小白入门](./01_RMSNorm_Tutorial_Beginner.md)、[05. LLaMA3 Block 教程 · 小白入门](./05_LLaMA3_Block_Tutorial_Beginner.md)
- 做完之后：[78. Attention 访问模式基准 · 小白入门](./78_Attention_Access_Pattern_Benchmark_Beginner.md)
