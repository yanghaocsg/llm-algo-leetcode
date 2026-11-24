# 54. KV Cache Quantization Strategies Beginner | KV Cache 量化策略 · 小白入门

> 配套 notebook：[54. KV Cache 量化策略](./54_KV_Cache_Quantization_Strategies.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

模型生成时要为每个词元保存两份状态，也就是 KV Cache，对话越长占显存越多。这本 notebook 教你用更少的位保存它们：键和值各用一种量化方式，最新的词元先保持高精度。

## 为什么需要它

模型已经能装进显存，对话变长后仍可能耗尽空间。解码（decode）时每生成一个新词元，都要为它保存后续注意力会反复读取的键（Key，K）和值（Value，V）状态。这些状态合起来就是 KV Cache。长上下文、高并发时，它成了持续的显存和带宽成本。

把 K 和 V 用更少的位保存，是直观的办法。但随便选一种量化方式会遇到两个问题：

- 所有数共用一个范围。范围会被最大的那个数撑开，小数只能落在很粗的格子上，恢复误差变大。
- K 和 V 的分布不一样。它们一起参与注意力，但分工不同，用同一种缩放方式可能放大量化误差。

以 KIVI 为代表的方法从这种差异出发：Key 逐通道量化，Value 逐词元量化。这本 notebook 在 CPU 上实现它，并处理另一件事：新词元先留在一段高精度窗口里，溢出时才把最旧的部分转存（flush）成量化段。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| KV Cache | 推理时为每个词元保存的键和值状态。后面每生成一个词元，注意力都要读它一遍 |
| Key / Value / Query（K / V / Q） | 注意力的三个输入。Q 和 K 做点积，决定该关注哪些位置；V 再按这些权重加权汇聚 |
| 量化轴 | 沿哪一维把数归成一组、共用一个缩放范围。轴不同，缩放因子的形状就不同 |
| 逐通道 / 逐词元（per-channel / per-token） | 逐通道是同一通道的所有词元共用一个范围；逐词元是同一个词元的所有通道共用一个范围 |
| 恢复尺度（scale） | 编码时除以它、恢复时乘回它的那个数，这里记作 `s` |
| `qmax` | 量化码能取的最大绝对值。题目里默认是 7，代表有符号 4 位的取值范围 |
| 高精度残留窗口（residual window） | 最近的一段状态还没量化，保持高精度。它有长度上限，超出的旧词元会被转存成量化段 |

## 核心思路

notebook 用形状 `[batch, heads, tokens, head_dim]` 的张量表示一段 K 或 V，下面简写成 `[B, H, L, D]`。每个数除以恢复尺度，四舍五入成整数码，用的时候再乘回去。新东西是恢复尺度怎么来：

```text
s_K[b, h, 1, d] = max_l abs(K[b, h, l, d]) / q_max   # Key 逐通道：固定通道，沿词元维找最大
s_V[b, h, l, 1] = max_d abs(V[b, h, l, d]) / q_max   # Value 逐词元：固定词元，沿通道维找最大
q     = clip(round(x / s), -q_max, q_max)           # 编码：得到整数码
x_hat = s * q                                        # 恢复：码乘回恢复尺度
```

`B`、`H`、`L`、`D` 分别是批大小、KV Head 数、缓存词元数和 Head 维度。Key 的缩放因子形状是 `[..., 1, head_dim]`，Value 的是 `[..., tokens, 1]`，保留长度为 1 的那一维才能广播（broadcast）回原张量。

用一个 3 个词元、2 个通道的小张量试一下：

```python
import torch

x = torch.tensor([[[[1.0, 6.0], [4.0, 1.0], [3.0, 1.0]]]])   # [batch, heads, tokens, head_dim]
qmax = 7
for name, dim in [("Key 逐通道", -2), ("Value 逐词元", -1)]:
    absmax = x.float().abs().amax(dim=dim, keepdim=True)    # 沿一条轴求绝对最大值
    scale = (absmax / qmax).clamp_min(1e-8)                 # 恢复尺度
    q = torch.clamp(torch.round(x.float() / scale), -qmax, qmax).to(torch.int8)   # 量化码
    restored = q.to(scale.dtype) * scale                    # 恢复
    print(name, tuple(scale.shape), [round(v, 4) for v in scale.flatten().tolist()])
    print(q.flatten().tolist())
    print([round(v, 3) for v in restored.flatten().tolist()])
```

```text
Key 逐通道 (1, 1, 1, 2) [0.5714, 0.8571]
[2, 7, 7, 1, 5, 1]
[1.143, 6.0, 4.0, 0.857, 2.857, 0.857]
Value 逐词元 (1, 1, 3, 1) [0.8571, 0.5714, 0.4286]
[1, 7, 7, 2, 7, 2]
[0.857, 6.0, 4.0, 1.143, 3.0, 0.857]
```

每一行是一个词元，每一列是一个通道。按列看，两列的最大值是 4 和 6，所以 Key 得到 2 个缩放因子；按行看，三行的最大值是 6、4、3，所以 Value 得到 3 个。同样是第一行的 1.0：逐通道时所在列的最大值只有 4，恢复成 1.143；逐词元时它和 6 共用范围，恢复成 0.857。和谁共用范围，决定了一个数能分到多细的格子。

再看追加。缓存分成两段：已量化的旧词元，和高精度窗口里最近的词元。新词元先接在高精度窗口后面；合并后的长度超过上限，才把最前面超出的部分转存成量化段，已有的量化段不动：

```python
limit, residual, quantized = 4, 0, 0
for new in [3, 2, 4]:                        # 每轮追加的词元数
    flush = max(residual + new - limit, 0)   # 本轮转存数：超出上限的部分
    residual, quantized = residual + new - flush, quantized + flush
    print(flush, quantized, residual)
```

```text
0 0 3
1 1 4
4 5 4
```

三列依次是本轮转存数、累计已量化词元数、窗口里剩下的词元数。上限是 4：追加 3 个词元没溢出；再追加 2 个，多出 1 个；再追加 4 个，多出 4 个。总共 9 个词元，5 个已量化、4 个保持高精度。读取时按历史顺序恢复各段，再接上高精度尾部。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | K 和 V 在注意力里分工不同，为什么可以采用不同的量化策略 | 读 |
| Step 2 | 两种量化轴各沿哪一维求范围，缩放因子的形状是什么样 | 读，对照上面的公式和小例子 |
| Step 3 | 追加、转存与读取分别增加什么成本 | 读 |
| Step 4 · TODO 1 | 沿词元维估计 Key 的逐通道范围并生成量化码 | 在 `quantize_key_per_channel` 里写 3 行：沿词元维求绝对最大值，算缩放因子，调用 `_quantize` 生成量化码 |
| Step 4 · TODO 2 | 沿头维估计 Value 的逐词元范围并生成量化码 | 在 `quantize_value_per_token` 里写 3 行：换成沿 `head_dim` 维，同样求绝对最大值、算缩放因子、生成量化码 |
| Step 4 · TODO 3 | 计算本轮从高精度窗口转存的旧词元数 | 在 `append_and_partition` 里写 1 行：合并后的高精度长度减去上限，最小为 0 |
| 测试 | 检查两种缩放因子的形状、全零范围、多轮追加、已量化分段不变、窗口边界，以及逻辑字节与实际存储 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 和 Transformers 的 `QuantizedCache` 共享两次追加，核对生命周期不变量 | 需要 `Q-CPU` 环境，带 Transformers 和 HQQ |
| Step 5（可选） | 在 GPU 上比较浮点 KV Cache 与量化 KV Cache 的容量、质量和生成代价 | 需要 `Q-CUDA-MECHANISM` 环境和单张 NVIDIA GPU，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU，用 `BASE-CPU` 环境。框架核对需要 `Q-CPU` 环境，Step 5 需要 GPU，默认不执行真实实验。

## 容易踩的坑

- **Key 和 Value 沿了同一条轴**：Key 沿词元维聚合，缩放因子形状是 `[..., 1, head_dim]`；Value 沿 `head_dim` 聚合，形状是 `[..., tokens, 1]`。写反了，形状断言就过不了。
- **全零范围得到零尺度**：某个通道或某个词元全是 0 时，绝对最大值是 0。要给缩放因子设正下限，`restore` 要求它有限且为正。
- **转存数算错**：本轮转存数是“合并后的高精度长度减去上限”，没溢出时必须是 0。只转存溢出的最旧几个词元，已有量化分段不重写，词元总数才守恒。
- **把高精度窗口当成丢弃旧历史的滑动窗口**：旧词元没有丢，只是转成了量化段，读取时按历史顺序恢复。
- **只看名义位宽就下结论**：实际账本还要计入缩放等元数据；读取和转存都有开销，缓存更小，生成每个词元未必更快。

## 读完后自查

- [ ] 能说出 Key 和 Value 各沿哪一维求范围，缩放因子分别是什么形状。
- [ ] 高精度窗口上限是 4，依次追加 3 个和 2 个词元，能算出第二轮转存多少。
- [ ] 能解释为什么高精度窗口不是丢弃旧历史的滑动窗口。
- [ ] 能说出除了缓存大小，还要一起比较哪几项代价。

## 下一步

- 打开 notebook：[54. KV Cache 量化策略](./54_KV_Cache_Quantization_Strategies.ipynb)
- 建议先读：[52. 量化校准范围与裁剪 · 小白入门](./52_Quantization_Calibration_and_Error_Beginner.md)、[22. vLLM 分页注意力 · 小白入门](./22_vLLM_PagedAttention_Beginner.md)
- 需要补基础：[Part 01 · 11 KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
