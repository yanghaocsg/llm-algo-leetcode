# 31. LoRA Variants Theory | LoRA 变体缩放与预算 · 小白入门

> 配套 notebook：[31. LoRA 变体缩放与预算](../31_LoRA_Variants_Theory.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

微调大模型时可以只训练一小块新增参数，这叫 LoRA，它有好几种改法。这本 notebook 教你算清每种改法训练多少参数、更新被放大多少倍，以及参数上限内能用多大的秩。

## 为什么需要它

给大模型做微调时，LoRA 的做法是把原模型冻住，只在旁边加一小块参数来学改动。这块参数有个旋钮叫秩：秩越大，可训练的参数越多。

后来出现了几种改法。rsLoRA 改了“更新放大多少倍”的规则；DoRA 在秩之外又加了一组幅值参数。想比较它们，最直接的办法是三种方案都把秩设成 8，各训练一次，看谁的结果好。

notebook 的导读提醒：这样看到的差异，可能来自更新尺度，也可能来自新增参数，而不只是方案名称。用 notebook 里的小数字看一下：

- 秩 4、`alpha` 8 时，LoRA 的缩放是 2，rsLoRA 是 4。低秩矩阵的形状完全相同，增量乘的系数却是 LoRA 的两倍。
- 同样是秩 4，两个目标层上 LoRA 训练 224 个参数，DoRA 是 248 个。

所以“哪个方案更好”要先说清在什么条件下比。这本 notebook 只做比较之前的准备：把秩、缩放、幅值三件事分开，算出每个方案的参数量，再反推预算允许的最大秩。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 低秩适配（LoRA） | 冻住原模型，只在旁边加两组小矩阵来学改动。这两组矩阵叫低秩矩阵 |
| 秩（rank） | 低秩矩阵中间那一维的宽度。秩越大，可训练的参数越多 |
| 缩放（scaling） | 低秩矩阵的结果要乘的系数，由超参数 `alpha` 和秩算出来。秩 4、`alpha` 8 时 LoRA 是 2 |
| 目标模块（target module） | 要加 LoRA 的线性层，比如 `q_proj`、`v_proj` |
| 幅值（magnitude） | DoRA 把一行权重看成一个箭头，箭头的长度叫幅值，指向叫方向 |
| 参数预算（parameter budget） | 允许训练的参数总数上限，代码里叫 `max_trainable_params` |

## 核心思路

输入是每个目标模块的形状（输入维度、输出维度）、秩、`alpha`、变体名，以及可选的参数上限。要做三步：数参数、算缩放、反推最大秩。输出是一份规格（`scaling` 和 `trainable_params`），或者预算允许的最大秩和 `accept` / `reject` 决策。

数参数和算缩放用下面几个公式：

```text
N_LoRA(T, r) = r * Σ (d_in + d_out)      # 所有目标模块的低秩矩阵
N_DoRA(T, r) = N_LoRA(T, r) + Σ d_out    # DoRA 每个输出通道再加一个幅值参数
s_LoRA       = alpha / r
s_rsLoRA     = alpha / sqrt(r)
r_max        = floor( (P - DoRA 幅值参数) / Σ (d_in + d_out) )
```

符号说明：`T` 是目标模块集合，`r` 是秩，`d_in` 和 `d_out` 是一个模块的输入、输出维度，`Σ` 表示对所有目标模块求和，`N` 是可训练参数总数，`alpha` 是缩放超参数，`s` 是增量系数，`P` 是参数上限，`floor` 是向下取整。

下面用 notebook 测试里的两个模块走一遍，最后两行是 DoRA 的一行权重：

```python
import math

shapes = {"q_proj": (16, 16), "v_proj": (16, 8)}   # 模块名: (输入维度, 输出维度)
rank = 4

lora = sum(rank * (d_in + d_out) for d_in, d_out in shapes.values())
dora = lora + sum(d_out for d_in, d_out in shapes.values())
print(lora, dora)                                    # LoRA、DoRA 的参数量
print(8 / rank, 8 / math.sqrt(rank))                 # alpha 为 8：LoRA、rsLoRA 的缩放

unit = sum(d_in + d_out for d_in, d_out in shapes.values())   # 每加一阶要的参数
fixed = sum(d_out for d_in, d_out in shapes.values())         # DoRA 的幅值参数
print(unit, fixed)
for budget in (248, 224):
    print(budget, budget // unit, (budget - fixed) // unit)   # 上限、LoRA 最大秩、DoRA 最大秩

row = (3.0, 4.0)                                      # 一行权重
length = math.sqrt(sum(x * x for x in row))           # 长度，也就是幅值
direction = tuple(round(x / length, 4) for x in row)  # 除以长度，得到方向
print(length, direction)
print(tuple(round(10 * x, 4) for x in direction))     # 幅值改成 10
```

```text
224 248
2.0 4.0
56 24
248 4 4
224 4 3
5.0 (0.6, 0.8)
(6.0, 8.0)
```

上限 248 时，LoRA 和 DoRA 都能取秩 4。上限 224 时，DoRA 先扣掉 24 个幅值参数，只剩 200，除以 56 得 3；LoRA 不用扣，仍是 4。DoRA 的可用秩有没有下降，取决于剩余预算，而不是方案名称。

DoRA 还改了有效权重的构造：`W' = m ⊙ (W + s * B @ A) / row_norm(W + s * B @ A)`。`W` 是冻结的原权重，`A`、`B` 是两组低秩矩阵，`s` 是缩放（DoRA 也用 `alpha / r`），`m` 是每个输出通道的幅值，`row_norm` 是按行算的长度，`⊙` 表示按输出通道逐个相乘。看输出的最后两行：行权重 `(3, 4)` 的长度是 5，除以长度得到方向 `(0.6, 0.8)`；幅值从 5 改成 10，权重变成 `(6, 8)`，方向不变。DoRA 就是用低秩增量调整方向，再用 `m` 学习幅值。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | LoRA、rsLoRA、DoRA 各改了什么：rsLoRA 改缩放规则，DoRA 把权重拆成幅值和方向 | 读 |
| Step 2 | 参数量公式，DoRA 的幅值参数怎样占预算，目标范围变大时预算怎么变 | 读，对照上面的公式 |
| Step 3 | 相同秩、相同参数上限两种比较条件各自能说明什么 | 读 |
| Step 4 · TODO 1 | 计算模块适配器参数 | 写 `count_adapter_parameters`：补 1 行，按模块累计低秩矩阵的参数，DoRA 再加上输出通道数的幅值参数 |
| Step 4 · TODO 2 | 建立变体规格 | 写 `build_lora_variant_spec`：补 2 行，算出 `scaling` 和 `trainable_params` |
| Step 4 · TODO 3 | 反推预算允许的 rank | 写 `max_rank_for_budget`：补 1 行，先扣掉幅值参数，再除以每阶的参数量并向下取整 |
| 测试 | 检查参数量、三种变体的缩放、预算内的最大秩，以及未知、重复、非法输入是否被拒绝 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用微型 Llama 模型，和 PEFT 的 LoRA、rsLoRA、DoRA 比较缩放值与可训练参数量 | 需要 `PT-CPU` 环境 |
| Step 5（可选） | 在真实模型上比较三种变体的缩放、参数量、更新时间、峰值显存和合并前后差异 | 需要 GPU 和 `PT-SFT-CUDA` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。框架核对需要另外的 `PT-CPU` 环境。Step 5 需要 GPU，默认不执行真实实验。

## 容易踩的坑

- **DoRA 漏算幅值参数**：DoRA 每个输出通道多一个幅值参数。数参数时要加上，反推最大秩时要先从预算里扣掉。
- **rsLoRA 用了 `alpha / rank`**：rsLoRA 的缩放是 `alpha / math.sqrt(rank)`，LoRA 和 DoRA 才是 `alpha / rank`。秩 4 时两者相差一倍。
- **反推秩时没有向下取整**：秩必须是整数，要用 `//`。结果小于 1 时，返回 `max_rank` 为 0、`decision` 为 `reject`。
- **把小数当成合法的秩**：骨架用 `type(rank) is not int` 检查，小数秩要报错，不能悄悄截断后返回自相矛盾的规格。
- **参数量接近就以为显存和速度接近**：相同参数上限只控制参数预算。DoRA 的方向归一化和幅值学习会增加计算和状态，峰值显存和更新时间要另外观察。

## 读完后自查

- [ ] 能说出 rsLoRA 和 DoRA 各自改变了 LoRA 的什么。
- [ ] 秩 4、`alpha` 8 时，能算出 LoRA 和 rsLoRA 的缩放。
- [ ] 两个目标层是 16/16 和 16/8，能算出秩 4 时 LoRA 和 DoRA 的参数量。
- [ ] 能解释上限 224 时，为什么 LoRA 能取秩 4，而 DoRA 只能取 3。

## 下一步

- 打开 notebook：[31. LoRA 变体缩放与预算](../31_LoRA_Variants_Theory.ipynb)
- 建议先读：[14. 低秩适配教程 · 小白入门](./14_LoRA_Tutorial.md)
- 做完之后可以做项目：[63. LoRA 变体选型项目](../63_LoRA_Variants_Benchmark.ipynb)、[65. QLoRA 选型项目](../65_QLoRA_Selection_Project.ipynb)
