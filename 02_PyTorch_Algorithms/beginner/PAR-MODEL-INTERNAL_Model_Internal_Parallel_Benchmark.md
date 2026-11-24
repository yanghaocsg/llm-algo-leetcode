# PAR-MODEL-INTERNAL. Model Internal Parallel Benchmark | 模型内部并行基准 · 小白入门

> 配套 notebook：[PAR-MODEL-INTERNAL. 模型内部并行基准](../PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

模型大到一张卡放不下时，可以按层、按矩阵、按序列三个方向切开，分给多张卡。这本 notebook 教你检查一种切法能不能排到卡上，再用显存、空转、通信和吞吐挑出一种组合。

## 为什么需要它

把模型切开分到多张卡上，有三个方向可以切。可以按层切，前几层放一组卡，后几层放另一组，数据算完一段再传给下一段。也可以把同一层拆开，一层里的大矩阵和注意力头（Head）分到几张卡上一起算。还可以按序列切，把输入的长序列切成几段，分给不同的卡。

三种切法可以组合，比如 8 张卡排成两个方向各 2 份、第三个方向也 2 份。组合的时候容易出三种错：

- 卡数对不上。三个方向上的份数乘起来必须等于总卡数，份数是 2、4、4 时需要 32 张卡，8 张卡排不出来。
- 切不开。24 层分给 2 段没问题，15 个注意力头分给 2 个 rank 就分不均。
- 以为切得越多越快。按矩阵切和按序列切会增加层内的同步，按层切会增加流水线的空转，三者还可能在同一时间线竞争链路。

所以要分三步：先检查卡数能不能分解、模型结构能不能整除；再用门槛筛掉不合格的组合；最后在合格的里面按吞吐选。这本 notebook 让你把三步各写成一个函数。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 切分轴 | 把模型切开的一个方向。本节有三种：按层、按单层内部的矩阵、按序列 |
| 流水线并行（PP） | 把连续的层分成几段（stage），每段放在不同的卡上 |
| 张量并行（TP） | 把单层里的大矩阵和注意力头拆到几张卡上一起算 |
| 上下文并行（CP） | 把输入序列切成几段，分给不同的卡 |
| Device Mesh | 把所有卡排成一个三维阵列，三个方向的边长就是 `pp`、`tp`、`cp` |
| rank | 多卡训练里的一个成员，本节可以理解成其中一张卡 |
| 气泡（bubble） | 流水线里有的卡在等上游、空转的时间占比，`bubble_ratio` 越小越好 |

## 核心思路

输入分两部分。一部分是布局配置：总卡数 `world_size`，三个方向的份数 `pp`、`tp`、`cp`，以及模型有多少层 `num_layers`、多少个注意力头 `num_heads`、序列多长 `sequence_length`。另一部分是每种组合的测量记录：峰值显存、气泡占比、通信占比、吞吐和运行是否失败。通信占比是训练时间里花在卡间传数据上的比例。

处理分三步。第一步校验布局。四项检查，哪一项不过就往 `errors` 里加一个名字：

```text
pp * tp * cp 不等于 world_size     → mesh_product
num_layers 除以 pp 有余数          → pipeline_layers
num_heads 除以 tp 有余数            → attention_heads
sequence_length 除以 cp 有余数      → context_blocks
```

`errors` 为空，布局才算 `valid`。

第二步准入。运行失败记 `runtime`，峰值显存超过 `limits['memory_mb']` 记 `memory`，气泡占比超过上限记 `bubble`，通信占比超过上限记 `communication`。失败原因全部留下，一个都没有才算合格。

第三步选择。只在合格候选里选，先比吞吐，高的优先；相同再比通信占比，再比峰值显存，都是低的优先。后两项只在吞吐持平时起作用。没有合格候选就返回 `reject`。

用 notebook 测试里的配置试一下第一步：

```python
def check(cfg):
    return (cfg['pp'] * cfg['tp'] * cfg['cp'] == cfg['world_size'],   # 三个方向的份数乘起来
            cfg['num_layers'] % cfg['pp'],                            # 余数：层数分给各段
            cfg['num_heads'] % cfg['tp'],                             # 余数：注意力头分给 TP
            cfg['sequence_length'] % cfg['cp'])                       # 余数：序列分给 CP

ok = {'world_size': 8, 'pp': 2, 'tp': 2, 'cp': 2, 'num_layers': 24, 'num_heads': 16, 'sequence_length': 4096}
print(check(ok))
print(check({**ok, 'world_size': 4, 'num_heads': 15}))
```

```text
(True, 0, 0, 0)
(False, 0, 1, 0)
```

第一行的布局合法：2 乘 2 乘 2 等于 8，24 层、16 个注意力头、4096 长的序列都能整除。第二行把卡数改成 4、注意力头改成 15 个，份数乘积对不上，注意力头也分不均，要记下两个错误。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从切分对象建立 Device Mesh：PP、TP、CP 各切什么，主要新增什么代价 | 读 |
| Step 2 | 检查布局兼容性：层数、注意力头数、序列长度分别要能被 PP、TP、CP 整除 | 读，对照上面的四项检查 |
| Step 3 | 用容量与关键路径比较组合：先满足单 rank 容量，再比气泡、暴露通信和吞吐 | 读 |
| Step 4 · TODO 1 | 模型内部切分机制——检查 mesh 乘积以及层、Head、序列的整除关系 | 在 `validate_mesh_layout` 里补 `errors`：四项检查各追加一个错误名 |
| Step 4 · TODO 2 | 候选准入机制——保留全部失败原因，不只返回布尔值 | 在 `evaluate_internal_parallel_candidate` 里补 `failures`：运行、显存、气泡、通信四道门槛各追加一个失败原因 |
| Step 4 · TODO 3 | 优先吞吐，其次通信占比和单 rank 峰值显存 | 在 `select_internal_parallel_candidate` 里补 `selected`：对合格候选按这个顺序排序，取第一个 |
| 测试 | 检查合法布局是否没有错误、卡数和注意力头数不对时是否报出对应的错误名、气泡超限时是否不准入、是否只从合格候选里选 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实多卡上验证单轴 TP、单轴 PP 和 PP 与 TP 的组合 | 需要 CUDA、NCCL、`torchrun` 和 Device Mesh API；默认是 `RUN_GPU_EXPERIMENT = False`，而且配置里每组的 `command` 还是空的，notebook 写明真实多卡验证留到统一的 GPU 环境阶段 |

Step 1–4 只需要 CPU。Step 5 需要多张 GPU，默认关闭，执行命令还没有登记。

## 容易踩的坑

- **只检查了卡数**：份数乘起来等于总卡数只是其中一项。层数、注意力头数、序列长度还要分别能被 `pp`、`tp`、`cp` 整除。两类错误分开记录，才好定位是资源分解失败还是结构映射失败。
- **只返回一个布尔值**：候选没通过时要把所有失败原因都留下来，分清是运行失败、显存超限、气泡超限，还是通信超限。
- **让没准入的高吞吐候选进入排序**：测试里 `pp2_tp2` 的吞吐是 130，比 `tp2` 的 120 高，但它的气泡超过上限，不能进入排序，最后选的是 `tp2`。
- **把排序规则当成最终证据**：排序规则表达的是当前工作负载的目标，不能替代真实多卡时间线和正确性证据。
- **把可整除当成充分条件**：可整除只是最低条件，负载均衡、填充（padding）和通信组仍会改变实际效率。

## 读完后自查

- [ ] 能说出 PP、TP、CP 各切模型的什么。
- [ ] 8 张卡、`pp=2`、`tp=2`、`cp=2`，24 层、16 个注意力头、序列长度 4096，能判断这个布局合不合法。
- [ ] 能说出候选准入的四道门槛。
- [ ] 能说出选型时三项指标的先后顺序。

## 下一步

- 打开 notebook：[PAR-MODEL-INTERNAL. 模型内部并行基准](../PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.ipynb)
- 建议先读：[28. Pipeline 并行微批次 · 小白入门](./28_Pipeline_Parallelism_MicroBatch.md)、[29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim.md)、[PAR-CONTEXT. 上下文与序列并行 · 小白入门](./PAR-CONTEXT_Context_and_Sequence_Parallelism.md)、[49. 并行策略选型 · 小白入门](./49_Parallelism_Strategy_Selection.md)
- 做完之后：[80. MoE 专家并行基准 · 小白入门](./80_MoE_Expert_Parallel_Benchmark.md)
