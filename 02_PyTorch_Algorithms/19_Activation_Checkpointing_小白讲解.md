# 19. 激活检查点 — 小白讲解

> 对应 notebook：[`19_Activation_Checkpointing.ipynb`](./19_Activation_Checkpointing.ipynb)

## 一句话总结
本节用 5 个 `SimpleTransformerBlock` 串成一条链，比较「普通保存」「逐 Block checkpoint」「不同 segment 粒度的分段 checkpoint」三种路径的输出、输入梯度和全部参数梯度，并用 `CountingBlock` 的 `forward_calls` 证明反向阶段确实重新执行了 Block。可选 GPU 实验在 Qwen2.5-0.5B、BF16、batch 2 × 序列 512 下测得：峰值 allocated 从 `4221.7 MB` 降到 `2799.6 MB`（约 `−33.7%`），median step 从 `139.56 ms` 升到 `166.42 ms`（约 `+19.2%`），判定 `accept`。

## 为什么需要它：痛点是什么
初学者容易把 activation checkpointing 理解成「少存一点激活」的一句口号，但马上会遇到三个问题：反向传播需要用的中间值不存了，从哪来？重算一遍为什么还能得到一模一样的梯度？`segment_size` 到底越大越省还是越小越省？如果只记住「用 checkpoint 省显存」，很容易忽略它用计算换容量，甚至误以为它会让训练更快。

生活类比：做一桌菜，不必把所有切好的配菜都摆满台面（占地方），只留整颗食材和步骤单（区段边界）；哪道菜要下锅，再按步骤单临时切（重算）。但如果有「随机撒盐」这种步骤（Dropout），第二次必须用同一份随机状态，否则两次味道不同——这就是重算能保持梯度一致的前提。

## 核心问题
- checkpoint 前向保存什么、反向重算什么？它和「把模型存到磁盘」有什么区别？
- 为什么重算后输出和梯度应当完全相同，什么会破坏这个等价？
- `segment_size` 如何改变边界数量和单次重算范围？它有没有固定的显存节省比例？
- `use_reentrant=False` 在本题中扮演什么角色？
- `segment_forward` 为什么要把 `blocks=segment` 写成默认参数？
- 怎样用代码证明「前向没有多跑、反向确实重算」？
- GPU 上省了多少峰值显存、付出多少时间，结果该怎样读？

## 解决思路：一步步看懂

**Step 1：checkpoint 是前向阶段的保存策略。** 一次训练先前向产生 activation，反向再用这些中间状态算梯度。序列、batch 或层数变大时，activation 会成为显存压力。checkpoint 把「保留区段内部 activation」改成「只保留区段边界，反向时重算区段内部」。这里的 Block 指可顺序执行的计算单元（例如 Transformer Decoder Layer），**不是 CUDA thread block**。

设区段接收边界状态 $h_{in}$，内部产生若干中间 activation，输出 $h_{out}$。普通路径保留内部状态；checkpoint 路径只保留重算所需的边界，反向时重新执行区段。

**Step 2：为什么重算能得到相同梯度。** 设区段函数为 $f_\theta$，要求

$$f_\theta^{save}(h_{in})=f_\theta^{recompute}(h_{in}),\qquad \nabla_{h_{in},\theta}L_{save}=\nabla_{h_{in},\theta}L_{recompute}.$$

$\theta$ 是区段参数。第一项约束输出，第二项约束输入梯度和参数梯度。要满足它，重算必须重放与原前向等价的计算上下文：同样的边界输入、同样的 Block 顺序、同样的参数、同样的随机状态，并且没有读取两次执行之间会变化的外部状态。notebook 用 Dropout 举例：如果重算时随机状态不同，新的 activation 就不同，梯度自然对不上。

**Step 3：粒度改变边界数和重算范围。** 设共 $L$ 个 Block、每段 $g$ 个，区段数约为 $N_{seg}=\lceil L/g\rceil$，尾部不足 $g$ 个也形成一个完整区段。$g$ 小 → 边界多、每次重算短；$g$ 大 → 边界少、每次重算范围大。notebook 特意强调：粒度**不会给出固定的显存节省比例**，参数、梯度、optimizer state 仍在账本里，实际收益必须固定模型、输入形状和环境去测。

**Step 4：三个 TODO 的参考写法。**

`SimpleTransformerBlock` 只是一个简化的残差 FFN：

```python
self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
self.norm = nn.LayerNorm(dim)
...
out = self.ffn(self.norm(x))
return x + out
```

`run_without_checkpointing` 是普通基线，按顺序执行所有 Block，作为输出和梯度的对照。

TODO 1 在 `run_with_checkpointing` 里对每个 Block 建立边界：

```python
x = checkpoint(block, x, use_reentrant=False)
```

注意输入是当前 `x`，输出写回 `x`；docstring 明确要求 `use_reentrant=False`。

TODO 2 在 `build_checkpoint_segments` 里生成半开区间：

```python
ranges = [(start, min(start + segment_size, num_blocks))
          for start in range(0, num_blocks, segment_size)]
```

`min(start + segment_size, num_blocks)` 负责保留不足一段的尾段；`num_blocks=0` 时 `range` 为空，返回 `[]`。函数开头已经校验 `num_blocks < 0` 和 `segment_size <= 0` 并抛 `ValueError`。手算：5 个 Block、`segment_size=2` → `(0,2)`、`(2,4)`、`(4,5)`；`segment_size=1` → 5 个单元素区间；`segment_size=6` → `(0,5)`。

TODO 3 在 `run_with_segment_checkpointing` 里对当前区段建立边界：

```python
for start, end in build_checkpoint_segments(len(blocks), segment_size):
    segment = blocks[start:end]

    def segment_forward(hidden, blocks=segment):
        return run_without_checkpointing(blocks, hidden)

    x = checkpoint(segment_forward, x, use_reentrant=False)
```

`blocks=segment` 是**默认参数绑定**，不是多此一举。Python 闭包按名字查找外层变量；如果写成 `def segment_forward(hidden): return run_without_checkpointing(segment, hidden)`，等反向真正调用时循环早已结束，所有区段都会读到最后一个 `segment`。把当前区段绑成默认值，才能保证「反向重算的是这一段，而不是最后一段」。

**Step 5：测试怎样证明重算发生。** `CountingBlock` 在 `forward` 里累加 `forward_calls`；`_run_and_collect` 在 `output.sum().backward()` 前后各取一次总调用次数。于是有两条判据：

- `calls_before_backward == num_layers`：前向阶段每个 Block 只执行一次，checkpoint 没有偷偷多跑；
- `calls_after_backward > calls_before_backward`：反向阶段确实重新执行了 Block。

普通路径则断言 `calls_after_backward == num_layers`，即反向不重算。测试用 `copy.deepcopy(template_blocks)` 给每条路径独立的权重副本，避免共享状态污染比较；输出、输入梯度、参数梯度都用 `atol=1e-5, rtol=1e-4` 对照基线。

**Step 6：框架一致性核对。** `check_checkpoint_framework_conformance` 用 dim 8 的 `CountedBlock`，输入 `torch.randn(2, 3, 8, requires_grad=True)`，把同一份 `state_dict()` 加载到两个独立模块，一条走普通路径、一条走 `run_with_checkpointing`。核对四项：输出一致、输入梯度一致、全部参数梯度一致（都是 `atol=1e-6, rtol=1e-6`），以及 `recompute.calls > direct.calls`。返回 `status: passed` 和两次调用次数。

**Step 7：GPU 实验的设计与结果。** 默认 `RUN_MODE = "dry_run"`；真实运行使用 `TRAIN-CUDA-MECHANISM`，下载 `Qwen/Qwen2.5-0.5B`，固定 batch 2、序列 512、BF16、seed 42，预热 1 次、重复 20 次。两条路径各加载一份模型：普通路径调用 `gradient_checkpointing_disable()`，候选路径调用 `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`。每个 step 用 CUDA Event 包住 `loss.backward()`，记录 loss、全局梯度 L2 范数、step 时间、`max_memory_allocated` 和 `max_memory_reserved`。

判据分两层：正确性要求两条路径的梯度都有限、`loss_abs_diff <= 5e-3`、梯度范数相对差 `<= 1e-3`；资源门槛要求 allocated 至少降低 `32 MB` 且至少降低 `1%`。决策是 `accept / tune / reject`。

已经保存的输出（`b2s512_default2`，RTX 5070 Ti Laptop GPU）是：

| 路径 | median | P95 | allocated | reserved |
|---|---:|---:|---:|---:|
| 普通路径 | 139.56 ms | 179.93 ms | 4221.7 MB | 4550.0 MB |
| 原生 checkpoint | 166.42 ms | 202.31 ms | 2799.6 MB | 4198.0 MB |
| 差值 | +26.86 ms | +22.38 ms | −1422.2 MB | −352.0 MB |

正确性三项差值都是 `0.0`，资源门槛 `memory_reduction_mb = 1422.15`、`memory_reduction_ratio = 0.3369`，判定 `accept`。5.4 把时间代价写成 median `+19.2%`、P95 `+12.4%`，容量收益写成 allocated `−33.7%`。

怎么读：先看正确性（这里完全一致），再看 allocated 是否越过双重门槛，最后才看时间。**不要只看 allocated**：reserved 只从 `4550.0` 降到 `4198.0`，约 `−7.7%`；allocator 会保留可复用的块，所以「活跃峰值省了 33.7%」不等于「整卡预算也省了 33.7%」。结论只说明容量—时间交换成立，不代表 checkpoint 会提升训练吞吐。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| activation | 前向产生、反向还要用的中间张量 |
| activation checkpointing | 只存区段边界，反向重算区段内部 |
| recompute | 反向时重新执行前向，恢复中间状态 |
| 区段边界 | 重算所需的输入状态 |
| segment / 粒度 | 一次 checkpoint 覆盖多少个连续 Block |
| `use_reentrant=False` | 使用非重入 checkpoint 实现，本题固定要求 |
| `CountingBlock` | 记录 forward 次数，用来证明反向是否重算 |
| peak allocated | 测量窗口内张量分配占用的最高值 |
| peak reserved | allocator 预留（含可复用块）的最高值 |
| 闭包晚绑定 | 循环结束后才读变量，导致所有区段指向最后一段 |
| 框架一致性核对 | 用同一断言对照 `torch.utils.checkpoint` 的行为 |

## 小白常见疑问

**Q1：用了 checkpoint，前向会不会执行两遍 Block？**
不会。测试断言每条 checkpoint 路径的 `calls_before_backward == num_layers`，前向仍然只跑一遍；重算发生在 `backward()` 里，所以 `calls_after_backward > calls_before_backward`。普通路径反过来，`calls_after_backward == num_layers`，反向不重算。

**Q2：重算一遍，为什么梯度能和普通路径完全一致？**
因为测试固定了相同权重、相同输入、相同执行顺序，残差 FFN 里也没有 Dropout 这类随机层；`_assert_matches_baseline` 用 `atol=1e-5, rtol=1e-4` 比较输出、输入梯度和每个参数梯度。真实模型那组更是三项差值都为 `0.0`。如果区段里有随机层或外部副作用，就必须额外恢复随机状态，否则等价会被破坏。

**Q3：`segment_size` 越大，是不是越省显存？**
不能这么说。$g$ 大时边界少，但单次重算范围大；$g$ 小时边界多，但每次重算短。notebook 明确写「粒度不会给出固定的显存节省比例」，参数、梯度、optimizer state 也不会因为 checkpoint 消失。要比较 `peak memory`、重算次数和 step time，得在固定模型与形状下实测。

**Q4：`segment_forward` 为什么必须写 `blocks=segment`？**
这是 Python 闭包的晚绑定问题。如果函数体直接引用外层的 `segment`，反向重算时循环已经跑完，`segment` 指向最后一个区段，于是所有区段都重算最后一段，数值自然对不上。把当前值绑成默认参数，等于在定义时就把这一段「拍照留存」。

**Q5：`use_reentrant=False` 到底改变了什么？**
notebook 在两处 docstring/注释里把它定为契约：TODO 1 和 TODO 3 都要求 `use_reentrant=False`，框架核对也直接调用这个实现。它让 checkpoint 走非重入路径，用保存张量钩子记录重算所需信息，因而能配合这里的闭包和分段写法；换成另一种语义可能改变反向行为，测试的一致性就无法保证。

**Q6：allocated 省了 33.7%，为什么 reserved 只省 7.7%？**
allocated 统计当前真实占用的张量，reserved 还包含 allocator 预留待复用的块。checkpoint 释放的激活块可能仍被缓存，所以 reserved 下降得更少。做容量规划时要同时看两者，必要时再看分配块大小与碎片。

## 动手验证：怎么知道自己做对了
主测试是 `test_activation_checkpointing_mechanism`，由 `run_checkpointing_test()` 调用；全部通过打印 `✅ CPU mechanism 测试通过：边界、输出、输入/参数梯度与 backward 重算均符合预期。` 它先调 `_assert_segment_plan(5)`：

- `build_checkpoint_segments(5, 2) == [(0,2),(2,4),(4,5)]`；
- `segment_size=1` 得到 `[(0,1),(1,2),(2,3),(3,4),(4,5)]`；
- `segment_size=6` 得到 `[(0,5)]`；
- `num_blocks=0` 得到 `[]`；
- `(5, 0)` 和 `(-1, 2)` 必须抛 `ValueError`。

随后固定 `torch.manual_seed(42)`、`dim=128`、`num_layers=5`、输入 `torch.randn(2, 32, 128)`，分别跑普通路径、逐 Block、`segment_size=2`、`segment_size=1`、`segment_size=6` 五条路径。每条 checkpoint 路径都要求前向调用次数等于 5、反向调用次数更多，并与 baseline 做输出/输入梯度/参数梯度的 `allclose`；最后额外断言 `segment_size=1` 与逐 Block 路径一致。

`check_checkpoint_framework_conformance` 是可选核对，返回字典里应看到 `status: passed`、四项 `checks` 全 True，以及 `checkpoint_calls` 大于 `direct_calls`。

GPU 单元保存了上面那张表的输出，所以可以直接对照：`correctness` 的 `loss_abs_diff`、`gradient_norm_abs_diff`、`gradient_norm_relative_diff` 都是 `0.0`，`resource_gate` 里的 `passed` 为 True，`decision` 为 `accept`。如果你的运行结果里时间或显存差得很远，先检查是否用了同一模型 commit、同一 batch/序列、同一 dtype 和 seed，再看 `peak_allocated_mb` 与 `peak_reserved_mb` 是否来自同一次测量窗口。
