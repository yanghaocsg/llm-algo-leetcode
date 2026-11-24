# 76. Activation Checkpoint Offload Benchmark Beginner | Activation / Checkpoint / Offload 对比项目 · 小白入门

> 配套 notebook：[76. Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU 可完成正确性验证；GPU 用于策略 benchmark

## 一句话说清

训练时激活值要占显存，存不下时可以少存、用到时重算，也可以先搬到内存里。这本 notebook 教你在不同预算下比较这些做法，并检查结论稳不稳。

## 为什么需要它

训练时，前向传播算出的激活值要留给反向传播用。模型一大、序列一长，这些激活值可能放不下。前面的 notebook 介绍过两种省显存的办法：19 的激活检查点不保存全部激活，用到时再算一遍；42 的激活卸载把激活值搬到主机内存，用到时再搬回来。这本 notebook 把它们放到同一个训练任务里比较，再加上两者合用的混合策略。

比较时有两件事容易做错。

一是只看显存。显存最低的方案不一定最值得采用：重算要额外计算，搬运要传输和等待，这些都会体现在训练步耗时和吞吐上，验证损失也要一起看。要先确认训练任务和执行路径可比，再联合这些指标和失败状态一起判断。

二是只在一个门槛下选。预算门槛（显存上限、吞吐下限、验证损失上限）稍微一变，选出来的候选可能就换了。所以 notebook 最后一步是换几个预算情景重算选择，看结论是否稳定。

notebook 登记的真实结果是顶层 `tune`：激活检查点在已测的激活显存范围内达到了收益与吞吐门槛，但长度边界没有确认。

这本 notebook 的主线在 CPU 上用固定的测量样例练熟四个判断：候选和基线是不是同一个实验、候选相对基线差多少、预算之下选谁、换几个预算结论还稳不稳。真实模型上的策略测量放在 Step 5。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 激活值（activation） | 前向传播算出来、反向传播还要用的中间结果 |
| 激活检查点（activation checkpointing） | 代码里叫 `checkpoint`，本页称选择性重算：前向后只保留边界输入或部分激活，反向前重新执行选定的前向区域 |
| 激活卸载（activation offload） | 代码里叫 `offload`，本页称主机卸载：把保存的张量移到主机内存，反向前搬回设备并等待可用 |
| 峰值显存（peak allocated） | 张量实际占用的显存峰值 |
| 验证损失（val loss） | 在验证数据上算的损失，本页用它看训练质量有没有变差 |
| 预算情景 | 一组门槛：显存上限 `memory_cap_mb`、吞吐下限 `min_samples_per_s`、验证损失上限 `max_val_loss` |
| `stable` / `threshold_sensitive` / `no_feasible_candidate` | 结论的三种状态：各情景选中同一候选、选择随门槛变化、有的情景没有可行候选 |

## 核心思路

输入是一份基线记录和若干候选记录。每份记录带着身份（`workload_id`、`hardware_id`、`repeat_count`）、运行状态、峰值显存、吞吐和验证损失。另外还有几个预算情景。处理分四步，对应四个 TODO；输出是每个情景选中的候选，以及对结论稳定性的判断。

四种策略改变的是激活值在“前向结束到反向使用”之间的去向：

```text
完整保存    激活留在设备显存    反向时直接读取           代价：容量占用
选择性重算  只保留边界输入等    反向前重算选定的前向区域  代价：额外计算
主机卸载    张量移到主机内存    反向前搬回并等待         代价：传输与同步
混合策略    部分丢弃、部分搬走  重算和搬运一起恢复       代价：计算、传输与调度
```

比较之前先过 TODO 1 和 TODO 2。身份三个字段任一不同、名字重复、执行路径没有核验，都不能进入比较。候选相对基线的差异按固定方向计算：

```text
memory_saving_mb = 基线峰值显存 - 候选峰值显存    正值表示候选更省显存
throughput_ratio = 候选吞吐 / 基线吞吐            小于 1 表示候选更慢
quality_delta    = 候选验证损失 - 基线验证损失    正值表示候选损失更高
```

状态是 `oom` 或 `failed` 的候选只计数，不计算差值。

TODO 3 在每个预算情景里重做一次选择：先找出三项门槛都满足的候选（基线也参与），按“显存低者优先、再看吞吐高者”排序，取第一个。TODO 4 再比较各情景的选择。下面用测试里的两个候选走一遍：

```python
baseline   = {'name': 'baseline',   'peak_memory_mb': 11000.0, 'samples_per_s': 8.0, 'eval_loss': 1.06}
checkpoint = {'name': 'checkpoint', 'peak_memory_mb': 8200.0,  'samples_per_s': 6.8, 'eval_loss': 1.08}
options = [baseline, checkpoint]

scenarios = [
    {'name': '宽松',       'memory_cap_mb': 12000, 'min_samples_per_s': 6.0, 'max_val_loss': 1.10},
    {'name': '吞吐收紧',   'memory_cap_mb': 12000, 'min_samples_per_s': 7.0, 'max_val_loss': 1.10},
    {'name': '显存也收紧', 'memory_cap_mb': 10000, 'min_samples_per_s': 7.0, 'max_val_loss': 1.10},
]
selected = []
for s in scenarios:
    feasible = [o for o in options
                if o['peak_memory_mb'] <= s['memory_cap_mb']
                and o['samples_per_s'] >= s['min_samples_per_s']
                and o['eval_loss'] <= s['max_val_loss']]
    feasible.sort(key=lambda o: (o['peak_memory_mb'], -o['samples_per_s']))   # 显存低者优先，再看吞吐高者
    selected.append(feasible[0]['name'] if feasible else None)
    print(s['name'], [o['name'] for o in feasible], selected[-1])

status = 'no_feasible_candidate' if None in selected else 'stable' if len(set(selected)) == 1 else 'threshold_sensitive'
print(status)
```

```text
宽松 ['checkpoint', 'baseline'] checkpoint
吞吐收紧 ['baseline'] baseline
显存也收紧 [] None
no_feasible_candidate
```

宽松时两个候选都可行，激活检查点显存更低，排在前面被选中。吞吐下限提到 7.0 之后，激活检查点的 6.8 不够，只剩基线。再把显存上限收到 10000，基线的 11000 也放不下，没有可行候选。三个情景的选择是“激活检查点、基线、无”，所以结论是 `no_feasible_candidate`：只要有一个情景选不出候选，就是这种状态；各情景选中的不同是 `threshold_sensitive`；全部相同才是 `stable`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 怎样把训练基线变成策略候选实验：固定什么、只变什么，四类输入各产出什么证据 | 读 |
| Step 2 | 四种策略在“前向结束到反向使用”之间怎样处理激活值，各自的恢复动作和主要代价 | 读，对照上面的示意 |
| Step 3 | 六项指标各看什么；判定规则要求哪些条件同时成立 | 读 |
| Step 4 · TODO 1 | 核对候选与基线身份 | 写 `validate_strategy_experiment`：补 1 行，`identity_matches` 是当前身份字段与基线相同 |
| Step 4 · TODO 2 | 计算候选相对基线的差异 | 写 `build_strategy_pair_evidence`：补 3 行，显存节省是基线减候选，吞吐比是候选除以基线，损失差是候选减基线 |
| Step 4 · TODO 3 | 按预算选择候选并记录情景结果 | 写 `decide_strategy_sensitivity`：补 2 行，`feasible_names` 是可行候选名，`selected_candidate` 是排序后的首个 |
| Step 4 · TODO 4 | 比较情景选择并判断结论稳定性 | 写 `decide_strategy_sensitivity`：补 1 行，给出 `stable`、`threshold_sensitive` 或 `no_feasible_candidate` |
| 测试 | 检查身份门槛、失败记录是否被保留、差值方向，以及收紧吞吐门槛后选择是否变化 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（项目实验） | 在真实模型上比较四种策略（G0），再用长度阶梯单独观察激活显存（G1），运行顺序是 prepare、g0、g1、close | 需要 GPU（显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_STAGE = 'prepare'`，只读检查，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU，不测显存收益，也不需要下载模型或准备 CUDA。Step 5 的 `g0`、`g1` 两个阶段需要 GPU（显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境，默认停在 `prepare`。

## 容易踩的坑

- **把显存不足算成收益**：状态是 `oom` 或 `failed` 的候选只计数，记录里只留名字和状态，不计算差值。指标不是有限数的候选记为 `invalid`，同样不算收益。
- **差值方向写反**：显存节省是“基线减候选”，正值表示候选更省；吞吐比是“候选除以基线”；验证损失差是“候选减基线”。
- **身份不一致还配对**：`workload_id`、`hardware_id`、`repeat_count` 任一不同就不能配对。候选名重复、`execution_path_verified` 不是 `True`，也要记进 `issues`。
- **漏掉基线**：预算情景里基线也参与比较。只有状态可比、执行路径已核验的候选才进入选择。
- **只在一个预算下选**：任一情景没有可行候选是 `no_feasible_candidate`，各情景选中的不同是 `threshold_sensitive`，这两种都不能当作稳定结论。

## 读完后自查

- [ ] 能说出四种策略各自的恢复动作和主要代价。
- [ ] 能写出显存节省、吞吐比、损失差三个量的算式和方向。
- [ ] 给出一个预算情景，能判断哪些候选可行、按什么顺序选。
- [ ] 能说出 `stable`、`threshold_sensitive`、`no_feasible_candidate` 各在什么时候出现。

## 下一步

- 打开 notebook：[76. Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- 建议先读：[19. 激活检查点 · 小白入门](./19_Activation_Checkpointing_Beginner.md)、[42. 激活卸载 · 小白入门](./42_Activation_Offload_Beginner.md)
- 建议先读：[73. 训练性能分析 · 小白入门](./73_Training_Performance_Analysis_Beginner.md)
- 做完之后：[75. AdamW 执行路径基准 · 小白入门](./75_Optimizer_Execution_Path_Benchmark_Beginner.md)
