# 87. 模型内部并行基准 — 小白讲解

> 对应 notebook：[`87_Model_Internal_Parallel_Benchmark.ipynb`](./87_Model_Internal_Parallel_Benchmark.ipynb)

## 一句话总结
本节先用四个纯 CPU 函数把「PP / TP / CP 三维 Device Mesh」的布局检查、单 rank 局部数量、候选门槛筛除和合格候选排序讲清楚，再给出一套固定 4 卡、固定模型与输入的基准实验设计：`G0_tp`=TP4、`G1_pp`=PP4、`G2_pp_tp`=PP2×TP2。它要验证的不是"哪种并行一定最快"，而是**先把不能均匀切、数值不对、显存超限、空转或通信超预算的候选全部排除，再在剩余候选里按吞吐排序**；同时 notebook 明确说明真实执行入口尚未接入，当前能跑的是 CPU 逻辑练习。

## 为什么需要它：痛点是什么
初学者第一次配并行，常把问题简化成"8 张卡怎么切最快"，然后直接抄一个 TP=8 或 PP=8 的配置。实际会连撞几堵墙：

- 模型有 24 层，PP 度数却设成 5——最后一阶段多出 4 层，负载根本分不均。
- 注意力有 16 个头，TP 度数设成 3——16 不能被 3 整除，头切不开。
- `pp × tp × cp` 算出来是 6，但实际只有 4 个进程——网格对不上。
- 切完能跑，但数值和单卡对不上，或者显存爆了，或者流水空转吃掉收益。

这些问题的共同点是：**它们不是"快慢"问题，而是"资格"问题**。先排序再筛选，会让一个错误配置因为数字好看而被选中。生活类比：招聘先设硬性门槛（学历、证书、体检），不合格的直接出局；剩下的人再按面试分数排序。你不能因为某人面试分最高，就忽略他没有执业资格。

## 核心问题
- `world_size`、`pp`、`tp`、`cp` 四者之间必须满足什么乘积关系？为什么？
- 为什么层数对 PP、Query 头数对 TP、序列长度对 CP 要做整除检查？不整除会怎样？
- 单 rank 的 `local_layers`、`local_heads`、`local_tokens` 怎么算？`state_units` 是不是显存？
- 候选筛选为什么要同时保留 `correctness`、`memory`、`bubble`、`communication` 等多个失败原因，而不是只返回一个布尔值？
- 排序时为什么先按吞吐降序，再看通信占比和峰值显存升序？"空转超预算但更快"的候选为什么不能入选？
- 这套 4 卡实验的自变量、因变量、baseline 分别是什么？结果应该怎么读？

## 解决思路：一步步看懂

### 第 1 步：三种并行分别切什么
notebook 的 Step 1 给出了三轴的分工表和代价表：

| 切分轴 | 分工对象 | 主要新增代价 |
|---|---|---|
| PP（流水线并行） | 不同层组成的阶段 | 阶段间激活传输、流水空转、阶段负载不均 |
| TP（张量并行） | 层内矩阵、特征或注意力头 | 多进程参与的归约与收集 |
| CP（上下文并行） | 不同序列位置 | 注意力所需的远端 K/V 交换 |

Device Mesh（设备网格）是把进程编号排成多维数组。8 个进程排成 $2\times2\times2$，每个 rank 有一个坐标 `(pp_index, tp_index, cp_index)`。沿 TP 维的进程一起完成层内计算，沿 PP 维传递阶段输出，沿 CP 维交换注意力需要的远端 K/V。因此第一层检查就是：

$$
\text{world\_size} = pp \times tp \times cp
$$

以 `BASE_CONFIG` 为例：`world_size=8, pp=2, tp=2, cp=2`，右边 $2\times2\times2=8$，通过。如果 `world_size=4` 而三轴乘积仍是 8，就多出 4 个进程没有坐标，记为 `mesh_product` 错误。

**不这样会怎样？** 网格乘积对不上时，后续所有"哪个 rank 属于哪个组"的推导都建立在错误编号上，通信组可能把不该通信的进程绑在一起，或者让某些进程永远收不到数据。所以它必须是第一个检查项。

### 第 2 步：TODO 1 `validate_mesh_layout`——一次收集所有布局错误
参考实现不是遇到第一个错误就 `return`，而是把错误追加进列表：

```python
errors = []
if config['pp'] * config['tp'] * config['cp'] != config['world_size']:
    errors.append('mesh_product')
if config['num_layers'] % config['pp'] != 0:
    errors.append('pipeline_layers')
if config['num_heads'] % config['tp'] != 0:
    errors.append('attention_heads')
if config['sequence_length'] % config['cp'] != 0:
    errors.append('context_blocks')
return {'valid': not errors, 'errors': errors}
```

四个检查项与错误名一一对应：

| 检查 | 条件 | 错误名 | 含义 |
|---|---|---|---|
| 网格乘积 | `pp*tp*cp == world_size` | `mesh_product` | 进程数与三轴度数不匹配 |
| PP 分层 | `num_layers % pp == 0` | `pipeline_layers` | 层数无法均匀分给阶段 |
| TP 分头 | `num_heads % tp == 0` | `attention_heads` | Query 头无法均匀分给 TP 进程 |
| CP 分序列 | `sequence_length % cp == 0` | `context_blocks` | 序列长度无法均匀分给上下文进程 |

为什么用"收集全部错误"而不是"快速失败"？测试 `test_mesh_layout_contract` 专门构造了一个同时犯两种错的配置：

```python
invalid = {**BASE_CONFIG, 'world_size': 4, 'num_heads': 15}
assert set(validate_mesh_layout(invalid)['errors']) == {'mesh_product', 'attention_heads'}
```

此时 `world_size=4` 但 $2\times2\times2=8$，触发 `mesh_product`；`15 % 2 = 1`，触发 `attention_heads`。如果实现里第一个 `if` 命中就 `return`，结果只会有一个错误，集合比较失败。这个测试就是用来抓"只报告首个问题"的偷懒写法。

容易写错的地方：用 `/` 而不是 `%`（`15/2=7.5` 不是整数，但判断方式不对）；把 `sequence_length` 的余数检查写成 `num_layers`；或者用 `errors.append` 后又把 `valid` 写成 `True`。注意 `valid` 的定义是 `not errors`，空列表为真，所以合法配置返回 `{'valid': True, 'errors': []}`。

### 第 3 步：TODO 2 `build_local_state_ledger`——三个维度各除一次
```python
local_layers = config['num_layers'] // config['pp']
local_heads  = config['num_heads'] // config['tp']
local_tokens = config['sequence_length'] // config['cp']
return {
    'local_layers': local_layers,
    'local_heads': local_heads,
    'local_tokens': local_tokens,
    'state_units': local_layers * local_heads * local_tokens,
}
```

代入 `BASE_CONFIG` 手算：

- `local_layers = 24 // 2 = 12`：PP=2，每个阶段拿一半层。
- `local_heads = 16 // 2 = 8`：TP=2，每个 TP 进程拿一半 Query 头。
- `local_tokens = 4096 // 2 = 2048`：CP=2，每个上下文进程拿一半序列位置。
- `state_units = 12 × 8 × 2048 = 196608`。

测试逐项断言 `12`、`8`、`2048`，再断言 `state_units == 12 * 8 * 2048`。

三个"容易误解"必须说清楚：

1. **`local_layers` 只受 PP 影响**，不受 TP/CP 影响。TP 切的是每层内部的矩阵，不是层数；CP 切的是序列位置，也不是层数。
2. **`local_heads` 用 Query 头数除 TP**，不是 KV 头数。notebook Step 2 还提醒运行真实模型时要"继续核对矩阵宽度、KV 头布局与后端对该切分方式的支持情况"——GQA 模型里 KV 头可能更少，不能假设和 Query 头一样能整除。
3. **`state_units` 是"局部数量示意"，不是显存字节数。** 解析里明确写「用三者乘积表示当前示例的局部数量，不能把它当作显存字节数；真实权重与激活大小另按具体张量计算」。`196608` 这个数字没有乘元素字节数，也没有区分权重、激活、梯度、优化器状态，所以不能拿它和 12 GB、80 GB 直接比。

### 第 4 步：TODO 3 `evaluate_internal_parallel_candidate`——逐项保留失败原因
```python
result = deepcopy(metrics)
failures = []
if metrics.get('failure'):
    failures.append('runtime')
if not metrics.get('correct', True):
    failures.append('correctness')
if metrics['peak_memory_mb'] > limits['memory_mb']:
    failures.append('memory')
if metrics['bubble_ratio'] > limits['bubble_ratio']:
    failures.append('bubble')
if metrics['communication_ratio'] > limits['communication_ratio']:
    failures.append('communication')
result['eligible'] = not failures
result['gate_failures'] = failures
return result
```

五个门槛与 `LIMITS` 的对应关系：

| 失败原因 | 触发条件 | `LIMITS` 默认值 |
|---|---|---|
| `runtime` | `metrics['failure']` 为真（运行报错/崩溃） | 无阈值，有值即失败 |
| `correctness` | `correct` 为 `False` | 无阈值，布尔判断 |
| `memory` | `peak_memory_mb > 12000` | `memory_mb=12000` |
| `bubble` | `bubble_ratio > 0.2` | `bubble_ratio=0.2` |
| `communication` | `communication_ratio > 0.3` | `communication_ratio=0.3` |

注意几个实现细节：

- **`deepcopy(metrics)`**：不修改调用方传进来的字典，避免后续比较时原始指标被 `eligible` / `gate_failures` 污染。这也是文件开头 `from copy import deepcopy` 的用途。
- **`metrics.get('failure')` 与 `metrics.get('correct', True)`**：前者用"有值即失败"，后者缺省按"正确"处理。但 `peak_memory_mb`、`bubble_ratio`、`communication_ratio` 是直接下标访问，缺字段会抛 `KeyError`——这说明契约要求候选必须带这三项指标。
- **严格大于**：`peak_memory_mb == 12000` 不算超限，`bubble_ratio == 0.2`、`communication_ratio == 0.3` 也算合格。测试里的 `13000 > 12000`、`0.25 > 0.2`、`0.35 > 0.3` 都严格越界。
- **不做短路**：五个 `if` 独立执行，所有失败原因都保留。解析里说「高吞吐不能抵消其中一项失败」。

测试 `test_candidate_gate_collects_failures` 构造的 `bad` 候选同时犯四项：`peak_memory_mb=13000`、`bubble_ratio=0.25`、`communication_ratio=0.35`、`correct=False`，`failure` 为 `None`。断言：

```python
assert result['eligible'] is False
assert set(result['gate_failures']) == {'correctness', 'memory', 'bubble', 'communication'}
```

用 `set` 而不是列表，是因为题目只关心"原因集合对不对"，不强制顺序；但你的实现应当稳定地按代码顺序追加，方便人读日志。这个测试专门抓两类错误：只记录第一个失败原因，或者把"任一指标合格"误写成通过。

还有一个边界值得注意：`memory` 用的是严格大于 `>`。`peak_memory_mb == 12000` 时算合格，`12001` 才失败。这不代表测量可以卡着上限跑——真实测量有噪声、分配器有缓存、不同批次峰值也会浮动，门槛通常要留余量。notebook 只是把比较规则写死成"超过才拒绝"，方便测试构造确定输入；生产上应把余量写进 `memory_mb` 的值里，而不是改比较符号。

### 第 5 步：TODO 4 `select_internal_parallel_candidate`——先过滤，再三级排序
```python
eligible = [item for item in candidates if item['eligible']]
if not eligible:
    return {'decision': 'reject', 'candidate': None}
selected = sorted(
    eligible,
    key=lambda item: (-item['throughput'],
                      item['communication_ratio'],
                      item['peak_memory_mb'])
)[0]
return {'decision': 'accept', 'candidate': selected['name']}
```

这是本节最值得细看的一段，因为它同时包含"资格"和"排序"两层逻辑：

1. **先构造 `eligible`**：只保留 `item['eligible']` 为真的候选。`evaluate_internal_parallel_candidate` 已经把五项门槛的结论写进 `eligible`，所以这里不再重新判断显存/空转。
2. **空集合返回拒绝**：`{'decision': 'reject', 'candidate': None}`。测试只断言 `select_internal_parallel_candidate([])['decision'] == 'reject'`。
3. **排序键三元组**：`-throughput` 实现吞吐降序（Python 的 `sorted` 默认升序，取负号翻转）；`communication_ratio` 升序；`peak_memory_mb` 升序。取 `[0]` 即第一名。
4. **只返回名字**：`{'decision': 'accept', 'candidate': selected['name']}`，状态沿用了题目给定的接口。

用测试数据走一遍：

- `tp2`：`peak_memory_mb=11000 ≤ 12000`、`bubble_ratio=0.0 ≤ 0.2`、`communication_ratio=0.22 ≤ 0.3`、`correct=True`、`failure=None`，所以 `eligible=True`，`gate_failures=[]`。
- `hybrid`（`pp2_tp2`）：`peak_memory_mb=9000`、`communication_ratio=0.28` 都合格，但 `bubble_ratio=0.25 > 0.2`，所以 `gate_failures=['bubble']`，`eligible=False`。
- 尽管 `hybrid` 的吞吐 130 高于 `tp2` 的 120，它已经出局；`select_internal_parallel_candidate([tp, hybrid])` 返回 `{'decision': 'accept', 'candidate': 'tp2'}`。这就是"空转超预算的候选即使更快也不参加排序"的直接证据。

再看并列排序测试 `equal_speed`：三个候选吞吐都是 120，通信占比分别是 `0.25`、`0.20`、`0.20`，峰值显存分别是 `8000`、`10000`、`9000`。第一轮比较通信占比，`more_comm`（0.25）被淘汰；第二轮比较显存，`less_memory`（9000）优于 `less_comm`（10000）。最终 `candidate == 'less_memory'`。这个用例检查了第二、第三排序条件，防止实现只写 `-throughput` 一个键。

**为什么排序键里没有 bubble？** 因为 bubble 已经是准入门槛（超过 0.2 直接出局），合格集合里所有候选的 bubble 都在预算内；再把它放进排序键会改变题目约定的优先级。题目明确写的是「优先吞吐，其次通信占比和单 rank 峰值显存」。

### 第 6 步：Step 5 的真实多卡实验设计
notebook 从 cell 14 开始给出一套 4 卡实验方案。虽然默认不执行，但设计值得逐项读懂。

**固定项（控制变量）**：

| 固定项 | 具体值/要求 |
|---|---|
| 模型 | `MODEL_ID = 'Qwen/Qwen3-0.6B'`，用于功能与通信路径验证，不代表生产规模收益 |
| 输入 | 相同编码输入、`SEQUENCE_LENGTH = 4096`、`MICRO_BATCH_SIZE = 1` |
| 设备 | `WORLD_SIZE = 4`，同一批 GPU 与互联 |
| 精度与软件 | 同一 PyTorch / CUDA / NCCL 版本，同一精度 |
| 计时 | `WARMUP = 3` 轮不统计，`REPEATS = 10` 轮正式重复 |

**自变量（三组切分）**：

| 组名 | mesh | 含义 |
|---|---|---|
| `G0_tp` | `pp=1, tp=4, cp=1` | 纯张量并行，4 卡切层内 |
| `G1_pp` | `pp=4, tp=1, cp=1` | 纯流水线并行，4 卡切层 |
| `G2_pp_tp` | `pp=2, tp=2, cp=1` | 2×2 混合，层和层内同时切 |

三组都用 4 个进程，CP 度数都是 1，因此 notebook 明说「当前组别的 CP 度数均为 1，没有执行 CP 组合」。这意味着本实验只能比较 TP、PP、PP×TP，不能得出 CP 的结论。

**因变量（要记录的指标）**：峰值显存 `peak_memory_mb`、空转占比 `bubble_ratio`、通信占比 `communication_ratio`、吞吐 `throughput`、数值是否一致 `correct`、运行是否失败 `failure`。Step 5.5 的核对表还要求记录模型版本、编码输入、批量、精度、GPU 与互联，以及「各组是否齐全，是否有失败、回退或不同计时方法」。

**baseline 是什么？** 这里有两层：

- 本轮实验内部，`G0_tp` 和 `G1_pp` 是两个单轴基线，`G2_pp_tp` 是组合候选。相同总 GPU 数（4 卡）下换切分方式，属于"配置比较"。
- 若要判断"加设备是否划算"（扩展效率），还需要不同设备数的基线。notebook 明确写「比较新增设备是否划算，还要记录设备数并另算扩展效率」，并说「当前四卡组别只能比较切分方案」。

**结果怎么读？** 按 Step 3 / Step 5.6 的顺序：

1. 先看布局和数值：布局不合法或输出与基线不一致，直接排除。
2. 再看峰值显存是否超过目标设备容量。
3. 再看空转和通信是否在本次预算内。
4. 最后只在合格集合里比较相同负载下的吞吐。
5. 结论分三档：采用（数值、容量、目标达标且收益可重复）、调整（能跑但空转/通信/吞吐不达标）、退回基线（布局、结果或运行失败，或收益不足）。

**最重要的现实约束**：`RUN_GPU_EXPERIMENT = False` 默认关闭；执行 cell 会先打印「GPU 实验默认关闭；确认至少 4 张 GPU、适配器和结果目录后再开启。」即使打开，也要先过 `GPU_PREFLIGHT` 五项检查（CUDA 可用、可见 GPU ≥ 4、NCCL 可用、`torchrun` 存在、`torch.distributed.device_mesh` 可导入），再检查 `ENTRYPOINT = 'benchmarks/model_internal_parallel.py'` 是否存在。notebook 原文说「运行入口尚未接入，因此本节目前可运行的是 CPU 布局与筛选练习」。所以本页保存的输出里没有性能数字是正常的，不要把它当成"跑出来很快/很慢"的证据。

### 第 7 步：两个比例指标到底在说什么
- **流水空转（bubble）**：某个阶段等待输入、下游或其他调度条件，没有做有效计算的时间。notebook 强调"它的占比需要说明统计区间"——分母是整段训练/推理时间还是稳态区间，会显著改变数值。
- **通信占比**：通信活动占多少时间。关键提醒是「通信与计算重叠时不等于未被隐藏的等待」。也就是说，通信占比 30% 不代表吞吐一定损失 30%；如果通信被计算掩盖，实际暴露的等待可能更少。

这两个指标都不能单独当结论，必须结合计时方法看。Step 3 的原话是「比较两类比例前，先核对它们的计时方法」。

### 第 8 步：把 Device Mesh 展开成坐标
`BASE_CONFIG` 的 8 个进程排成 $2\times2\times2$。如果按"最后一维变化最快"的行优先排法，一种可能的坐标分配是：

| rank | (pp, tp, cp) | 所属 TP 组（同 pp、同 cp） | 所属 PP 阶段 |
|---|---|---|---|
| 0 | (0, 0, 0) | {0, 2} | 0 |
| 1 | (0, 0, 1) | {1, 3} | 0 |
| 2 | (0, 1, 0) | {0, 2} | 0 |
| 3 | (0, 1, 1) | {1, 3} | 0 |
| 4 | (1, 0, 0) | {4, 6} | 1 |
| 5 | (1, 0, 1) | {5, 7} | 1 |
| 6 | (1, 1, 0) | {4, 6} | 1 |
| 7 | (1, 1, 1) | {5, 7} | 1 |

这张表只用于理解"沿某一维变化的进程组成一组"，具体 rank 到坐标的映射由框架的 mesh 构造决定，notebook 没有规定。关键点是：PP 组跨的是 rank 0→4 这种"阶段"方向，TP 组跨的是 0→2 这种"层内分片"方向，CP 组跨的是 0→1 这种"序列分片"方向。三种通信各走各的维，不能混。

再看几个 `validate_mesh_layout` 的边界例子（都由四条规则直接推出，不是实测）：

- `world=8, pp=2, tp=2, cp=2, layers=25`：乘积对，但 $25\bmod2=1$，只有 `pipeline_layers`。
- `world=8, pp=2, tp=2, cp=2, heads=15`：只有 `attention_heads`，因为其他三项都整除。
- `world=8, pp=2, tp=2, cp=2, seq=4095`：$4095\bmod2=1$，只有 `context_blocks`。
- `world=4, pp=2, tp=2, cp=2`：乘积 8 ≠ 4，触发 `mesh_product`；若同时把 heads 设成 15，就得到测试里的两个错误并存。

### 第 9 步：换切分与加设备是两种不同的问题
Step 3 和 Step 5.6 反复区分两类结论：

1. **相同总 GPU 数下换切分**：4 卡的 TP4、PP4、PP2×TP2 三组，比的是"同样资源怎么分工更划算"。这时不需要扩展效率，直接看合格候选的吞吐、显存、空转和通信。
2. **增加设备是否划算**：从 1 卡到 4 卡、8 卡，吞吐提升是否与设备数成比例。设 $N$ 卡相对 1 卡的吞吐倍率为 $S_N$，则扩展效率可写成 $E_N=S_N/N$。例如 4 卡吞吐是 1 卡的 2 倍，$E_4=2/4=0.5$，说明一半的算力被通信、空转或负载不均吃掉了。notebook 没有给出这个数字，只提醒"比较新增设备是否划算，还要记录设备数并另算扩展效率"。

因此，看到"`G2_pp_tp` 比 `G0_tp` 快"时先别下结论：它们都是 4 卡，属于配置比较；要回答"值不值得上 8 卡"，必须另做不同设备数的基线。Step 5.6 的结论表也把"采用 / 调整 / 退回基线"限定在"相同资源预算"内。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| PP（流水线并行） | 按层切成阶段，不同阶段放在不同进程，像流水线工位 |
| TP（张量并行） | 把层内的矩阵、特征或注意力头切开，多进程合作算一层 |
| CP（上下文并行） | 把一条长序列按位置切开，不同进程负责不同片段 |
| Device Mesh | 把进程编号排成多维网格，每一维对应一种并行 |
| `world_size` | 参与本次任务的总进程数 |
| 局部层/头/词元 | 单个 rank 实际负责的层数、Query 头数、序列位置数 |
| `state_units` | 三者乘积，只表示局部数量规模，不是显存字节 |
| 准入门槛（gate） | 运行、数值、显存、空转、通信五项硬条件，任一不过就出局 |
| 流水空转（bubble） | 某阶段没活干、干等上游/下游的时间占比 |
| 通信占比 | 通信活动耗时占比；与计算重叠时不一定等于暴露等待 |
| 扩展效率 | 增加设备后吞吐提升是否与设备数成比例，需要多设备数基线 |

## 小白常见疑问

**Q1：为什么 `pp × tp × cp` 必须等于 `world_size`，多几个进程不行吗？**
不行。Device Mesh 要求每个进程都能在三轴上找到唯一坐标；乘积小于 `world_size` 会有进程没有位置，乘积大于 `world_size` 会有坐标没有进程。测试把 `world_size` 从 8 改成 4 而三轴仍是 2×2×2，`validate_mesh_layout` 立刻返回 `mesh_product` 错误。真实系统里也可能用数据并行再乘一维，但那是另一个轴，notebook 明确说「增加数据并行等维度时，资源分解也要加入对应维度」。

**Q2：`local_heads` 为什么是 `num_heads // tp`，不是 `num_heads // (tp*cp)`？**
因为 CP 切的是序列位置，不是注意力头。TP 才负责把 Query 头分给不同进程。同理 `local_tokens` 只除 `cp`。测试断言 `local_layers=12`、`local_heads=8`、`local_tokens=2048`，就是分别对应 `24//2`、`16//2`、`4096//2`；如果错误地连乘或连除，三个数至少有一个对不上。

**Q3：`state_units=196608` 能直接当显存估算吗？**
不能。它只是 `12 × 8 × 2048` 的局部数量示意，没有乘每个元素的字节数，也没有区分权重、激活、梯度和优化器状态。notebook 解析的原话是「不能把它当作显存字节数；真实权重与激活大小另按具体张量计算」。真正的显存账本要看具体张量形状、dtype 和临时缓冲区。

**Q4：`hybrid` 吞吐 130 比 `tp2` 的 120 高，为什么最后选 `tp2`？排序键为什么不是三项加权？**
因为 `hybrid` 的 `bubble_ratio=0.25` 超过 `LIMITS['bubble_ratio']=0.2`，`evaluate_internal_parallel_candidate` 把它标成 `eligible=False`、`gate_failures=['bubble']`。`select_internal_parallel_candidate` 第一步就用列表推导过滤掉不合格候选，排序只发生在 `eligible` 集合内。排序用的是可解释的字典序 `(-throughput, communication_ratio, peak_memory_mb)`：先比吞吐，相同再比通信占比小的，仍相同再比显存低的；加权分数需要人为定权重，换一组权重结论就变。并列测试 `equal_speed` 专门验证第二、第三级 tie-break 最终选中 `less_memory`。

**Q5：这个 notebook 跑完能看到 `G0_tp`/`G1_pp`/`G2_pp_tp` 的实测吞吐吗？**
默认看不到。`RUN_GPU_EXPERIMENT = False`，执行 cell 只会打印"GPU 实验默认关闭"；而且 `ENTRYPOINT = 'benchmarks/model_internal_parallel.py'` 尚未接入，即使打开开关也会在 `Path(ENTRYPOINT).exists()` 处抛 `FileNotFoundError`。当前可靠可跑的是 CPU 上的四个逻辑测试和 `GPU_PREFLIGHT` 环境检查；预检只说明环境是否具备运行条件，本身不产生任何吞吐或显存数字，预检失败时应记录环境错误而不是生成性能结论。

## 动手验证：怎么知道自己做对了
测试区在 cell 9，用循环依次执行四个测试函数，全部通过后打印 `✅ Device Mesh、局部数量、候选筛选与组合选择通过基础校验。`。TODO 未填时先抛 `NotImplementedError`。

- `test_mesh_layout_contract`：`validate_mesh_layout(BASE_CONFIG)` 必须严格等于 `{'valid': True, 'errors': []}`；把 `world_size` 改成 4、`num_heads` 改成 15 后，错误集合必须等于 `{'mesh_product', 'attention_heads'}`。
- `test_local_state_ledger`：`build_local_state_ledger(BASE_CONFIG)` 的 `local_layers` 为 12、`local_heads` 为 8、`local_tokens` 为 2048、`state_units` 为 `12 * 8 * 2048`。
- `test_candidate_gate_collects_failures`：`bad` 候选的 `eligible is False`，`gate_failures` 集合等于 `{'correctness', 'memory', 'bubble', 'communication'}`；`runtime` 不在其中，因为 `failure=None`。
- `test_candidate_selection`：`hybrid['gate_failures'] == ['bubble']`；`[tp, hybrid]` 选出 `tp2`；空列表返回 `reject`；三个同速候选中选出 `less_memory`。

这个 notebook 没有保存任何执行输出（dump 里没有任何 output），所以没有可引用的实测吞吐或显存数字。判断标准就是上述断言；如果全部通过，会看到那行带 ✅ 的打印。真实多卡部分则要等 `benchmarks/model_internal_parallel.py` 接入、`RUN_GPU_EXPERIMENT=True` 且 `GPU_PREFLIGHT['ready']` 为真后，才会按 `WARMUP=3`、`REPEATS=10` 生成各组 JSON 和 `summary.json`。
