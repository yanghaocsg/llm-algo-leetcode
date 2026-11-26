# 27. ZeRO 优化器状态分片模拟 — 小白讲解

> 对应 notebook：[`27_ZeRO_Optimizer_Sim.ipynb`](./27_ZeRO_Optimizer_Sim.ipynb)

## 一句话总结
本节用一份"参数编号账本" `build_zero_stage_layout` 对比 Data Parallel、ZeRO-1、ZeRO-2、ZeRO-3 分别由哪个 rank 持有参数、梯度和优化器状态；再用 `ZeRO1_Optimizer_Sim` 在两个逻辑 rank 上走一遍"只更新自己负责的那片参数"的流程。配合 $16\Phi$ 的显存公式和可选的多卡/框架核对，说明每一级到底省了什么、额外付出什么通信。

## 为什么需要它：痛点是什么
初学者常把 ZeRO 理解成"把模型切开"，于是产生一连串混淆：

- ZeRO-1 都切了，为什么显存还是没降多少？——因为它只切优化器状态，参数和梯度仍是完整副本；
- ZeRO-2 和 ZeRO-3 到底差在哪？——差在"梯度是否分片"和"参数是否分片"；
- 状态切了，通信会变多还是变少？——省的是常驻显存，代价是额外的分发、Reduce-Scatter 或 All-Gather；
- 代码里为什么用 `id(param)` 当字典键，而不是参数名或张量值？

生活类比：小组共用一本教材。DP 是每个人复印一整本，人人都有、也人人都占地方；ZeRO-1 是正文大家都有，但笔记（优化器状态）每人只抄自己负责的章节；ZeRO-2 连作业本（梯度）也只留自己那章；ZeRO-3 连教材正文都只留自己那章，讲到自己那章时再向同学借（临时 All-Gather）。越往后，个人桌面越干净，但"借书"的次数越多。

## 核心问题
- Data Parallel 的 $16\Phi$ 是怎么从参数、梯度、优化器状态三项加出来的？
- ZeRO-1/2/3 的 owner 布局分别切哪几类状态？为什么每级只多切一类？
- 一次 ZeRO-1 更新里，哪些状态在本地、哪些来自同步后的局部梯度？
- 为什么优化器状态用 `id(param)` 索引？这个 CPU 模拟没有模拟什么？
- 显存账本什么时候选 ZeRO-1，什么时候才非 ZeRO-3 不可？

## 解决思路：一步步看懂

**第 1 步：先把账本口径统一。** 记 $\Phi$ 为参数量，$P$ 为数据并行进程数，$b_p,b_g,b_o$ 分别为每参数的参数、梯度、优化器状态字节数。notebook 用 FP16 参数、FP16 梯度、FP32 主权重加 Adam 状态，取 $b_p=2,b_g=2,b_o=12$（主权重 4 + Adam 一阶矩 4 + 二阶矩 4），于是：

$$M_{\mathrm{DP}}\approx\Phi(b_p+b_g+b_o)=16\Phi$$

$$M_{\mathrm{Z1}}\approx\Phi\left(b_p+b_g+\frac{b_o}{P}\right),\quad M_{\mathrm{Z2}}\approx\Phi\left(b_p+\frac{b_g+b_o}{P}\right),\quad M_{\mathrm{Z3}}\approx\frac{\Phi(b_p+b_g+b_o)}{P}=\frac{16\Phi}{P}$$

拿 $\Phi=7\times10^9$、$P=8$ 手算：DP 是 $16\times7=112$ GB；ZeRO-1 是 $14+14+12\times7/8=28+10.5=38.5$ GB；ZeRO-2 是 $14+14\times7/8=14+12.25=26.25$ GB；ZeRO-3 是 $16\times7/8=14$ GB。这些是**常驻状态下界**，参数 All-Gather 的临时缓冲、梯度桶、激活和显存碎片都要在实测峰值里另算。

**第 2 步：三个阶段到底切什么。** notebook Step 2 的定性描述是：ZeRO-1 每张卡仍看得到完整参数和梯度，只有优化器状态按 owner 保存并更新；ZeRO-2 梯度也只留在 owner，通常用 Reduce-Scatter 把同步后的梯度直接送到对应 owner；ZeRO-3 参数也按 owner 切分，某层参与计算前才临时收集这一层。切得越深，单卡常驻状态越少，但参数收集与释放越频繁，所以 ZeRO-3 对链路和调度更敏感。

**第 3 步：TODO 1——阶段 owner 布局。** 骨架已经准备好两份字典：

```python
replicated = {rank: param_ids[:] for rank in range(num_ranks)}          # 每个 rank 都持有全部编号
sharded = {rank: param_ids[rank*shard_size:(rank+1)*shard_size] ...}    # 连续切分
```

参考答案用三个条件表达式：

```python
parameter_owner   = sharded if stage == 'zero3' else replicated
gradient_owner    = sharded if stage in {'zero2', 'zero3'} else replicated
optimizer_owner   = sharded if stage in {'zero1', 'zero2', 'zero3'} else replicated
```

记忆口诀是"每升一级只多切一类"：ZeRO-1 只动优化器状态，ZeRO-2 再动梯度，ZeRO-3 再动参数。函数返回 `{'parameters': ..., 'gradients': ..., 'optimizer_states': ...}`，每个值都是 `{rank: [参数编号]}`。容易写错的是把 ZeRO-1 的参数也写成 `sharded`——那样显存公式会对不上，测试 `test_zero_stage_state_ownership` 里 `zero1['parameters']` 的长度会是 4 而不是 8。函数入口还检查 `num_params`、`num_ranks` 为正且能整除，以及 `stage` 必须是 `dp/zero1/zero2/zero3` 之一。

**第 4 步：TODO 2——参数 owner 切片。** `ZeRO1_Optimizer_Sim.__init__` 先算 `params_per_rank = len(self.params) // self.num_gpus`，再对每个 rank 取连续区间：

```python
start = rank * params_per_rank
end = start + params_per_rank
local_params = self.params[start:end]
```

为什么必须连续？因为测试 `test_zero1_parameter_visibility` 要求把所有 `gpu_partitions` 里的参数 `id` 摊平后，数量等于参数总数且无重复——连续切分天然满足"每个参数恰好属于一个 rank"。如果写成隔一个取一个，虽然也能覆盖，但和 `build_zero_stage_layout` 的账本语义就不一致了。

**第 5 步：TODO 3——只为 owner 建状态。** 参考答案是一行字典推导：

```python
local_states = {id(param): torch.zeros_like(param.data) for param in local_params}
```

键用 `id(param)`，是因为参数是 `nn.Parameter` 对象，身份比名字稳定，也不依赖 `named_parameters()` 的遍历顺序；值用 `zeros_like` 保证形状和 dtype 与参数一致。注意这里**只为 `local_params` 建状态**，不是为 `self.params` 全部建。`test_zero1_partition_state` 用 `SimpleModel(dim=4)`（两个 Linear、无 bias，共两个参数张量）配 `num_gpus=2`，断言每个 rank 的 `optimizer_states` 长度都是 1。

**第 6 步：TODO 4——局部更新。** `step` 接收的 `gradients_from_all_gpus` 被约定为"已经同步好的本地梯度"，结构是 `{rank: [该 rank 负责参数的梯度]}`。参考答案：

```python
updated_state = state + grad                    # 教学用累积状态
updated_parameter = param.data - self.lr * updated_state
states[id(param)] = updated_state
param.data = updated_parameter
```

`step` 原地修改 `param.data` 和 `states`，没有返回值。这是"累积梯度 + SGD"的教学替身，**不等同于 Adam**：`state` 每次累加梯度，参数按 `lr × 累积梯度` 更新。第一次 step 时 `state` 从 0 变成 `grad`，所以 `test_zero1_local_update_value` 用全 1 梯度、`lr=0.1` 断言每个参数的减小量恰好是 `0.1`，并且更新后的 state 恰好是全 1 张量。更新前代码会检查 `len(params) == len(grads)`，不等就抛 `ValueError`；`test_zero1_gradient_count_contract` 用 `{0: [], 1: [...]}` 触发这个分支。

**第 7 步：测试在读什么。** `test_zero_stage_state_ownership` 用 `num_params=8, num_ranks=2`（每片 4 个）检查：DP 的优化器状态每个 rank 长度 8；ZeRO-1 的参数、梯度长度 8、优化器状态长度 4；ZeRO-2 的梯度长度 4；ZeRO-3 的参数长度 4。`test_zero_stage_shards_cover_each_state_once` 进一步把被切分的那一类摊平，断言 `sorted(owned) == list(range(8))` 且无重复——即"完整覆盖、不重不漏"。

**第 8 步：可选的框架一致性核对。** notebook 用 `run_local_gloo_world` 起两个本地 Gloo 进程，跑 PyTorch 的 `ZeroRedundancyOptimizer`（`SGD, lr=0.05, momentum=0.9`），模型是 `4→4` 和 `4→2` 两个无 bias 线性层。四个核对项是：所有 rank 的本地参数张量数之和等于 2（`local_shards_cover_parameters`）；每个 rank 的本地状态数等于本地参数数（`local_states_match_shards`）；两个 rank 更新后的参数一致（`parameters_consistent`）；教学账本 `build_zero_stage_layout(num_params=2, num_ranks=2, stage="zero1")` 的优化器 owner 覆盖 `[0, 1]`（`teaching_owner_cover`）。它核对的是**语义**，不测 GPU 峰值、NCCL 通信或吞吐。

**第 9 步：Step 5 的 GPU 复测（可选）。** 配置默认 `RUN_MODE='dry_run'`，`STRATEGIES=['ddp','zero1','zero2','zero3']`，`WORLD_SIZE=2`，合成模型 `synthetic_transformer_mlp`，`MODEL_WIDTH=1024`、`MODEL_LAYERS=4`、`bf16`、`GLOBAL_BATCH_SIZE=4`、`MICRO_BATCH_SIZE=1`、`SEQ_LEN=512`、`WARMUP=2`、`REPEATS=5`。改成 `real_gpu` 后，5.2 会用 `torchrun` 调 `tools/run_zero_stage_benchmark.py`，每个 stage 写一份 JSON 到 `benchmarks/results/27_zero/<RUN_LABEL>/`。5.3 读取时要求字段齐全：`strategy`、`workload`、`hardware`、`metrics`、`evidence_level`、`failure`、`decision`；缺字段直接 `ValueError`。5.4 的解释表现在全是"运行后填写"，因为 notebook 没有保存真实 GPU 结果。读的时候先排除 `failure` 和不可比结果，再同时看"每 rank 状态/峰值显存"和"最慢 rank step time"——不能只按理论除法选阶段。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Data Parallel（DP） | 每张卡存完整参数、梯度、优化器状态，只把 batch 分开 |
| 训练状态 | 参数 + 梯度 + 优化器状态（Adam 的一阶矩、二阶矩等） |
| $\Phi$ | 参数量；7B 就是 $7\times10^9$ |
| $b_p,b_g,b_o$ | 每参数的参数、梯度、优化器状态字节数，本节取 2/2/12 |
| ZeRO-1 | 只把优化器状态按 rank 分片，参数和梯度仍复制 |
| ZeRO-2 | 优化器状态 + 梯度分片 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数都分片 |
| owner rank | 某个参数（或状态）归哪个进程负责 |
| Reduce-Scatter | 归约后把结果按片分发，每张卡只拿自己那份 |
| All-Gather | 把各卡的分片收集成完整张量，ZeRO-3 按层临时做 |
| 分片覆盖 | 所有编号恰好出现一次，不重不漏 |
| `id(param)` | Python 对象身份，用来把状态字典映射回参数对象 |

## 小白常见疑问

**Q1：ZeRO-1 参数还是完整副本，为什么能省这么多？**
因为优化器状态占大头。按本节口径，每参数 16 字节里有 12 字节是优化器状态（主权重 + 两个矩），参数和梯度各只有 2 字节。7B、8 卡时，DP 是 112 GB，ZeRO-1 把 12Φ 那部分除以 8，变成 38.5 GB；省下的 73.5 GB 几乎全部来自优化器状态。notebook Step 3 也提醒：若只是不想复制 Adam 状态，ZeRO-1 往往通信复杂度更低；只有参数本身放不下时才需要 ZeRO-3。

**Q2：状态字典为什么用 `id(param)` 而不是参数名？**
这是一个纯 CPU 的教学模拟，参数是普通的 `nn.Parameter` 对象，`id()` 能稳定地把"更新后的状态"写回同一个对象，不依赖命名或遍历顺序。代价是它只在本进程内有效；真实分布式训练会用参数分片和扁平缓冲区来管理，不会靠 `id` 做跨进程索引。

**Q3：`step` 只收本地梯度，通信去哪了？**
被抽象掉了。notebook 的注释明确说"本类接收的局部梯度已经完成同步"，所以 `step` 只演示"owner 用自己的那片梯度更新自己的那片参数和状态"。真实的 ZeRO-1 还需要把梯度同步/分发到对应 owner，并在更新后让各卡拿到更新后的完整参数；这些 collective 不在这个模拟里。

**Q4：这个模拟能证明 ZeRO-2 比 ZeRO-1 快吗？**
不能。`build_zero_stage_layout` 只是"所有权账本"，不执行 collective，也不测时间；`ZeRO1_Optimizer_Sim` 只跑 ZeRO-1 的更新语义。显存和速度结论要么按公式估算，要么看 Step 5 的真实 JSON；notebook 的 5.4 表在未运行时全部是"运行后填写"。

**Q5：为什么第一次 step 的参数变化恰好是 `lr × grad`？**
因为 `state` 初始化为全 0，`updated_state = 0 + grad = grad`，`updated_parameter = param - lr × grad`。测试用全 1 梯度、`lr=0.1`，所以每个元素的减小量是 `0.1`，同时 state 变成全 1。这也说明它只是"累积梯度版 SGD"，不是 Adam——换成 Adam 后同样的梯度不会得到同样的步长。

## 动手验证：怎么知道自己做对了

notebook 的测试区直接顺序调用六个函数，没有统一的 ✅ 打印；任何断言失败都会抛异常。全部通过就说明四类语义都对上了：

- `test_zero_stage_state_ownership`：`num_params=8, num_ranks=2`，DP 的 `optimizer_states` 每 rank 长度 8；ZeRO-1 的参数、梯度长度 8、优化器状态长度 4；ZeRO-2 的梯度长度 4；ZeRO-3 的参数长度 4。
- `test_zero_stage_shards_cover_each_state_once`：对 `('zero1','optimizer_states')`、`('zero2','gradients')`、`('zero3','parameters')` 三组，摊平后必须 `sorted(owned) == list(range(8))` 且长度等于去重后长度。
- `test_zero1_partition_state`：`SimpleModel(dim=4)` + `num_gpus=2`，每个 rank 的状态字典长度都是 1。
- `test_zero1_local_update_value`：全 1 梯度、`lr=0.1`，所有参数的减小量在 `allclose` 意义下等于 0.1，所有 state 等于全 1。
- `test_zero1_parameter_visibility`：所有 rank 的参数 `id` 摊平后，数量等于去重后数量，且等于 `len(list(model.parameters()))`。
- `test_zero1_gradient_count_contract`：`{0: [], 1: [fc2 梯度]}` 必须触发 `ValueError`。

notebook 没有保存这些测试的执行输出，所以以断言为准。可选的框架核对会额外检查 `local_shards_cover_parameters`、`local_states_match_shards`、`parameters_consistent`、`teaching_owner_cover` 四项，失败时打印 `框架一致性核对失败：['...']`；它需要 `torch.distributed` + Gloo，属于 CPU 多进程，不是 GPU 实验。
