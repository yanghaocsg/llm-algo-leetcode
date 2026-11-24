# 27. ZeRO Optimizer Sim | ZeRO 优化器模拟 · 小白入门

> 配套 notebook：[27. ZeRO 优化器模拟](../27_ZeRO_Optimizer_Sim.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

数据并行训练时，每张 GPU 都保存一份完整的训练状态，很浪费显存。ZeRO 把这些状态切开分给各张卡，这本 notebook 在 CPU 上模拟这个过程。

## 为什么需要它

模型大到单卡放不下时，常见的办法是数据并行：把一批数据切给多张 GPU，各算各的，再同步梯度。它有一个明显的浪费：每张卡都保存完整的参数、梯度和优化器状态。模型一大，真正压垮显存的往往不是计算本身，而是这些训练状态在每张卡上重复存储。

这笔账很具体。用 FP16 加 Adam 训练时，以参数量 `Φ` 为单位，参数占 `2Φ` 字节，梯度占 `2Φ` 字节，Adam 优化器状态占 `12Φ` 字节。8 张卡做数据并行，每张卡都存完整的 `16Φ`，其中优化器状态这一项，8 张卡存的是 8 份相同的内容。

ZeRO 的做法是把这些重复训练状态分给不同 GPU 维护，并且分三个阶段逐步切：ZeRO-1 先切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切。这本 notebook 在 CPU 上模拟这四种布局，再走一遍 ZeRO-1 的一次更新。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 数据并行（Data Parallel，DP） | 把一批数据切给多张 GPU 各算一部分，再同步梯度。每张卡都保存完整的训练状态 |
| 训练状态 | 训练时每张卡要保存的三样东西：参数、梯度、优化器状态 |
| 优化器状态 | 优化器为了更新参数而额外记住的量。账本里 Adam 的优化器状态占 `12Φ`，比参数的 `2Φ` 大得多 |
| 复制（replicated）/ 分片（shard） | 复制是每张卡存一份完整副本；分片是切成几段，每张卡只存一段 |
| rank / owner | rank 是一张 GPU 在并行训练里的编号；owner 是某个参数归哪个 rank 负责 |
| 集体通信 | 多张卡一起交换数据的操作，比如全归约（all-reduce），它让每张卡都拿到汇总结果 |
| `Reduce-Scatter` / `All-Gather` | 两种集体通信。`Reduce-Scatter` 把汇总结果按段分给各卡；`All-Gather` 把各卡的分片拼成完整的一份 |

## 核心思路

ZeRO 的输入是一份训练状态和 GPU 数量，输出是“每张卡保存哪些状态”的归属布局。四种布局只差一句话：阶段每升一级，就多切一类训练状态。题目区用参数编号表示状态，8 个参数分给 2 个 rank 时是这样：

```text
replicated = {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7]}
sharded    = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}

阶段    参数         梯度         优化器状态
dp      replicated   replicated   replicated
zero1   replicated   replicated   sharded
zero2   replicated   sharded      sharded
zero3   sharded      sharded      sharded
```

`replicated` 表示每张卡一份完整副本，`sharded` 表示按 rank 切成不重叠的几段。这张表只描述状态归谁管，不模拟真实的 GPU 张量，也不执行通信。

再看省下多少显存。`N` 是 GPU 数量，`14Φ` 是梯度 `2Φ` 加优化器状态 `12Φ`：

```text
Data Parallel   16Φ               不切
ZeRO-1          2Φ + 2Φ + 12Φ/N   只切优化器状态
ZeRO-2          2Φ + 14Φ/N        再切梯度
ZeRO-3          16Φ/N             再切参数
```

代入 `N = 8` 算一下。第二段是 ZeRO-1 的一次局部更新：每个 rank 只拿自己负责的参数、同步后的局部梯度和自己的优化器状态。更新规则是教学用的，不等同于 Adam：

```python
# 1) 显存账本：FP16 + Adam，参数量为 Φ
N = 8                                # GPU 数
params, grads, adam = 2, 2, 12       # 三类状态各占多少个 Φ
dp    = params + grads + adam        # 不分片
zero1 = params + grads + adam / N    # 只切优化器状态
zero2 = params + (grads + adam) / N  # 再切梯度
zero3 = (params + grads + adam) / N  # 连参数也切
print(dp, zero1, zero2, zero3)

# 2) ZeRO-1 的一次局部更新（教学用累积状态，不等同于 Adam）
param, state, grad, lr = 1.0, 0.0, 1.0, 0.1
state = state + grad                 # 新状态
param = param - lr * state           # 新参数
print(state, param)
```

```text
16 5.5 3.75 2.0
1.0 0.9
```

单卡状态从 `16Φ` 依次降到 `5.5Φ`、`3.75Φ`、`2Φ`。代价是通信变多：Data Parallel 的主要通信是全归约梯度；ZeRO-2 通常用 `Reduce-Scatter` 把同步后的梯度直接送到对应的 owner；ZeRO-3 在某一层参与计算前，才临时收集这一层需要的参数。所以切分越深，单卡状态越少，参数收集与释放却越频繁。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 数据并行为什么重复保存训练状态，ZeRO 三个阶段各多切哪一类状态 | 读，对照上面的归属布局 |
| Step 2 | 沿一次训练更新，看三个阶段分别由谁保存参数、梯度和优化器状态 | 读 |
| Step 3 | 用显存账本比较四种布局，以及各自的主要通信 | 读，对照上面的算式 |
| Step 4 · TODO 1 | `num_params`、`num_ranks`、`stage`、`replicated` / `sharded` 布局 | 在 `build_zero_stage_layout` 里写 3 行：为参数、梯度、优化器状态各选 `replicated` 或 `sharded` |
| Step 4 · TODO 2 | `model_params`、`num_gpus`、`params_per_rank` | 在 `__init__` 里写 3 行：算出当前 rank 的切片起点和终点，取一段连续的参数 |
| Step 4 · TODO 3 | `gpu_partitions`、参数对象 `id`、零初始化状态 | 在 `__init__` 里写 1 行：只为本地参数建零初始化状态，键是参数对象的 `id` |
| Step 4 · TODO 4 | `gradients_from_all_gpus`、`states`、`lr` | 在 `step` 里写 2 行：累加状态，再按学习率更新参数 |
| 测试 | 检查各阶段的归属长度、分片完整且不重叠、每个 rank 只持有自己的状态、局部更新只改自己的参数 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 用 DeepSpeed 在 GPU 上复测 DDP 与 ZeRO 各阶段的峰值显存和时间 | 需要 `PAR-CUDA-MECHANISM` 环境和两张 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要两张 GPU，默认不执行真实实验。

## 容易踩的坑

- **阶段和切分对象对不上**：ZeRO-1 只切优化器状态，ZeRO-2 再切梯度，ZeRO-3 再切参数，Data Parallel 三类都不切。写错一个阶段，长度断言就过不了。
- **分片有重叠或有遗漏**：被分片的状态必须覆盖所有参数编号，每个编号只能归属一个 rank。
- **参数数量不能被卡数整除**：题目要求均匀切分，不整除时直接报错。
- **给所有参数都建了优化器状态**：每个 rank 只为自己负责的参数建状态，这正是 ZeRO-1 省下的部分。
- **更新了不归自己管的参数**：每个 rank 只更新自己的参数和状态。传进来的梯度数量必须和本地参数数量一致，否则报错。

## 读完后自查

- [ ] 能说出 ZeRO-1、ZeRO-2、ZeRO-3 各多切了哪一类训练状态。
- [ ] 8 个参数分给 2 个 rank，能写出 `sharded` 的内容。
- [ ] `N = 8` 时，能算出 ZeRO-1 的单卡状态。
- [ ] 能说出参数本身放不下时该选哪个阶段，以及为什么不能只按省显存来选。

## 下一步

- 打开 notebook：[27. ZeRO 优化器模拟](../27_ZeRO_Optimizer_Sim.ipynb)
- 需要补基础：[PyTorch 优化器与损失函数](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)、[显存分析与优化](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb)、[NCCL 与 AllReduce 基础](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)
- 做完之后：[28. Pipeline 并行微批次 · 小白入门](./28_Pipeline_Parallelism_MicroBatch.md)
