# 19. Activation Checkpointing | 激活检查点 · 小白入门

> 配套 notebook：[19. 激活检查点](../19_Activation_Checkpointing.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

训练时为了算梯度，要把前向传播的中间结果都留在显存里。激活检查点只留下每一段的输入，等反向传播用到时再把这一段重新算一遍，用多算一次换更少的显存占用。

## 为什么需要它

一次训练步骤先做前向传播，再做反向传播。反向传播算梯度时要用到前向产生的中间结果，也就是激活值（activation）。所以普通做法是把它们全部留着，一直留到反向传播结束。

序列更长、批次更大、层数更多时，激活值越积越多，可能成为显存压力的来源。

激活检查点（activation checkpointing）换了一个思路：前向时不保留 Block 内部的激活值，只保留 Block 的输入；反向传播走到这里时，再把 Block 重新执行一遍，把中间结果算回来。这一步叫重算（recompute）。两条路径的差别是：

- 前向阶段：普通路径保存区段内部的所有中间状态，激活检查点路径只保存区段的边界。
- 反向阶段：普通路径直接读取保存的状态，激活检查点路径先把区段重新执行一遍，再读取结果。
- 结果：两条路径的输出、输入梯度和参数梯度应该一致。

这本 notebook 做三件事：给单个 Block 加上重算，把连续的 Block 分成区段，再验证重算前后的输出和梯度相同。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 激活值（activation） | 前向传播算出的中间结果。反向传播算梯度时要用，所以普通做法是一直留到反向结束 |
| 激活检查点（activation checkpointing） | 一种前向阶段的保存策略：不保留区段内部的激活值，只保留区段的边界，反向时把区段重算一遍 |
| Block | 模型里可以顺序执行的计算单元，比如一个 Transformer Decoder Layer。它不是 CUDA 里的线程块 |
| 区段（segment）与边界 | 连续几个 Block 组成一段。边界是这段的输入 `h_in` 和输出 `h_out`，只有边界会被保留 |
| 粒度（granularity） | 每段覆盖多少个 Block，公式里记作 `g`。每段 1 个最细，整条路径合成一段最粗 |
| 重算（recompute） | 反向传播时重新执行一遍区段的前向计算，把中间结果临时算回来 |
| `checkpoint` | PyTorch 提供的函数，把一段计算包起来，让它在反向时重算。本节固定传 `use_reentrant=False` |

## 核心思路

输入是模型的隐藏状态 `x`，也就是一层传给下一层的那串数字。激活检查点路径只保留每个区段的输入，反向传播走到这里时，用保存的输入把区段重新执行一遍，再算梯度。输出是和普通路径相同的最终隐藏状态、输入梯度和参数梯度。notebook 用两个等式表示这种一致：

```text
f_save(h_in) = f_recompute(h_in)                      输出相同
grad L_save = grad L_recompute  （对 h_in 和 θ）      输入梯度、参数梯度相同
```

`h_in` 是区段的输入，`θ` 是区段的参数，`f` 是区段函数，`save` 和 `recompute` 分别是普通路径和重算路径。

重算必须在和原前向相同的条件下执行：从同一边界状态按相同的 Block 顺序重跑，使用相同的参数；有 Dropout 这类随机层时，要恢复对应的随机状态；不能读取两次执行之间会变化的外部状态。

下面用 3 个小 Block 试一下，每个 Block 计算 `x * w + 1`，`w` 是参数。数一数两条路径里 Block 各被执行了几次：

```python
import torch
from torch.utils.checkpoint import checkpoint

calls = [0]
def block(x):                        # 假装它是一个带参数的模型 Block
    calls[0] += 1
    return x * w + 1

def run(x, use_checkpoint):
    for _ in range(3):               # 3 个 Block 依次执行
        x = checkpoint(block, x, use_reentrant=False) if use_checkpoint else block(x)
    return x

for use_checkpoint in (False, True):
    w = torch.tensor([2.0], requires_grad=True)
    x = torch.tensor([1.0], requires_grad=True)
    calls[0] = 0
    out = run(x, use_checkpoint)
    before = calls[0]
    out.sum().backward()
    print(f"use_checkpoint={use_checkpoint}: 输出 {out.item()}, 输入梯度 {x.grad.item()}, "
          f"参数梯度 {w.grad.item()}, 反向前调用 {before} 次, 反向后调用 {calls[0]} 次")
```

```text
use_checkpoint=False: 输出 15.0, 输入梯度 8.0, 参数梯度 17.0, 反向前调用 3 次, 反向后调用 3 次
use_checkpoint=True: 输出 15.0, 输入梯度 8.0, 参数梯度 17.0, 反向前调用 3 次, 反向后调用 6 次
```

两条路径的输出、输入梯度、参数梯度相同。区别在调用次数：普通路径前后都是 3 次；激活检查点路径在反向传播时把 3 个 Block 各重算了一遍，变成 6 次。

区段怎么分，由粒度决定。notebook 的例子是 5 个 Block、每段 2 个：

```text
[0, 2)   [2, 4)   [4, 5)      最后一段只有 1 个 Block，照样保留
```

段数大约是 `ceil(L / g)`，`L` 是 Block 总数，`g` 是每段的 Block 数。每段越小，边界越多，单次重算越短。notebook 提醒：粒度不会给出固定的显存节省比例，收益要实际测量。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 激活检查点路径在前向保存什么、反向重算什么，和普通路径比哪些量 | 读 |
| Step 2 | 重算之后输出和梯度为什么应该和普通路径相同，要保持哪些条件 | 读，对照上面的两个等式 |
| Step 3 | 每段覆盖多少个 Block，怎样影响边界数量和重算范围 | 读 |
| Step 4 · TODO 1 | 建立逐 Block 重算边界 | 在 `run_with_checkpointing` 的循环里补 1 行：对当前的 `block` 调用 `checkpoint`，结果写回 `x` |
| Step 4 · TODO 2 | 生成连续的 checkpoint 区段 | 在 `build_checkpoint_segments` 里补 1 行：从 0 起每 `segment_size` 个取一段，`end` 不超过 `num_blocks` |
| Step 4 · TODO 3 | 建立分段重算边界 | 在 `run_with_segment_checkpointing` 的循环里补 1 行：对已绑定的 `segment_forward` 调用 `checkpoint`，结果写回 `x` |
| 测试 | 检查区段边界是否连续并覆盖尾段，激活检查点路径的输出和梯度是否与普通路径一致，反向时 Block 是否真的再次执行 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 直接核对 PyTorch 激活检查点（`use_reentrant=False`，非重入）的输出、梯度和重算行为 | 需要 `TRAIN-CPU` 环境，可以先跳过 |
| Step 5（可选） | 在 GPU 上比较普通路径与逐 Block 激活检查点的输出、梯度、峰值显存和训练 step 时间 | 需要单张 CUDA GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `TRAIN-CPU` 环境。Step 5 需要 GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **忘了 `use_reentrant=False`**：本节的两条激活检查点路径都要求传这个参数，否则反向语义会变。
- **循环里的函数读到了错误的区段**：循环里定义的 `segment_forward` 要用默认参数绑定当前区段，也就是 `blocks=segment`。否则反向重算时读到的是循环结束后的最后一个区段。
- **尾段被丢掉**：区段用半开区间 `[start, end)`，要连续、不重叠。末段不足 `segment_size` 时仍然保留，空序列返回空列表。
- **重算的上下文和原前向不同**：区段里有 Dropout 这类随机层时，重算要恢复对应的随机状态，也不能读取两次执行之间会变化的外部状态。否则激活值不同，梯度对不上。
- **把某个粒度当成通用推荐**：`segment_size=2` 只是用来演示分段和尾段，不是通用的性能推荐。

## 读完后自查

- [ ] 能说出激活检查点路径在前向阶段保存什么、不保存什么。
- [ ] 能解释例子里激活检查点路径的反向调用次数为什么是 6。
- [ ] 5 个 Block、每段 2 个，能写出区段划分。
- [ ] 能说出重算要得到相同梯度，需要保持哪些条件。

## 下一步

- 打开 notebook：[19. 激活检查点](../19_Activation_Checkpointing.md)
- 建议先读：[09. 激活与损失反向传播 · 小白入门](./09_Activation_and_Loss_Backward.md)
- 需要补基础：[数据接口与训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)
- 做完之后：[42. 激活卸载 · 小白入门](./42_Activation_Offload.md)
