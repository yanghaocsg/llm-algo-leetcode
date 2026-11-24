# 43. Unified Training Memory Ledger | 统一训练内存账本 · 小白入门

> 配套 notebook：[43. 统一训练内存账本](../43_Unified_Training_Memory_Ledger.md) ｜ 难度：Medium ｜ 环境：CPU 主线

## 一句话说清

训练时显存的峰值，取决于某个阶段有哪些东西同时留在 GPU 上。这本 notebook 教你给每种状态记下多大、在哪个设备、从哪个阶段到哪个阶段，再按阶段算出峰值出现在哪、由谁造成。

## 为什么需要它

训练一步，GPU 上先后出现很多东西：参数、激活值（activation）、梯度、优化器状态、算子运行时的工作区、通信缓冲区。它们各自的大小都能查到，但把大小直接相加，既说不清峰值出现在哪，也说不清峰值是谁造成的。

原因是它们不在同一时间存在。参数从头到尾都在，梯度只在反向传播到参数更新之间存在，工作区只在一个算子运行时存在。峰值出现在同时在场的东西加起来最大的那个阶段。

几种显存优化策略，各自只改变了其中一小段时间：

- 梯度累积改变梯度保留多久。
- 激活检查点改变激活值怎么保存。
- 激活卸载改变激活值待在哪个设备。

要判断一种策略有没有降低峰值，得把所有状态放回同一条时间轴上比。这本 notebook 做的就是统一记账：每种状态记下它的生命周期（lifecycle），也就是从哪个阶段存在到哪个阶段释放；再按阶段汇总，找出峰值阶段和贡献最大的对象，最后比较基线（baseline）和候选（candidate）账本。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 生命周期（lifecycle） | 一个状态从哪个阶段开始存在，到哪个阶段可以释放。起止阶段都算在区间内 |
| 驻留（residency） | 某个阶段里，这个状态还留在设备上、占着空间 |
| 阶段（phase） | notebook 把一步训练拆成 6 个阶段：`load`、`forward`、`loss`、`backward`、`update`、`zero_grad` |
| 优化器状态（optimizer state） | 优化器为了更新参数自己保存的量，第一次更新后才出现，之后长期驻留 |
| 保存的张量（saved tensors） | 激活值里被自动求导明确留下来、供反向传播使用的那一部分 |
| 工作区（workspace）/ 通信缓冲区（communication buffer） | 算子运行时临时用的空间，以及多卡通信时临时放数据的空间，用完就释放 |
| 基线 / 候选（baseline / candidate） | 基线是拿来对照的账本，候选是用了某种策略之后的账本 |

## 核心思路

输入是一张状态记录表，账本里每条记录有容量、驻留设备、开始阶段和结束阶段。处理分四步：检查每条记录合法；对每个阶段，把这时还留在 GPU 上的状态容量加起来；取总量最大的阶段作为峰值，把在场状态按容量从大到小排；候选账本走同样的流程，用基线峰值减去候选峰值。输出是各阶段总量、峰值阶段、贡献对象和候选之间的比较。notebook 的公式是：

```text
M_p = Σ_j m_j * I_{j,p}             阶段 p 的总量
M_peak = max_p M_p                  理论峰值
ΔM_peak = M_peak - max_p Σ_j m'_j * I'_{j,p}     候选的理论节省量
```

`m_j` 是状态 `j` 的容量，`I_{j,p}` 表示阶段 `p` 时状态 `j` 是否留在 GPU 上，在是 1，不在是 0。`m'_j` 和 `I'_{j,p}` 是候选账本里对应的量。`ΔM_peak` 大于 0 只说明理论峰值下降，重算、传输、通信和数值代价要另外测量。

下面用 notebook 测试里的状态表试一下。候选账本把 `activations` 从 GPU 改到了 CPU：

```python
PHASES = ["load", "forward", "loss", "backward", "update", "zero_grad"]

def phase_totals(states):                     # states：(名字, MB, 开始阶段, 结束阶段, 设备)
    totals = {}
    for p in PHASES:
        i = PHASES.index(p)
        active = [s for s in states
                  if s[4] == "gpu" and PHASES.index(s[2]) <= i <= PHASES.index(s[3])]
        totals[p] = sum(s[1] for s in active)
    return totals

base = [
    ("parameters", 800, "load", "zero_grad", "gpu"),
    ("optimizer_state", 600, "load", "zero_grad", "gpu"),
    ("activations", 300, "forward", "backward", "gpu"),
    ("gradients", 400, "backward", "update", "gpu"),
    ("forward_workspace", 200, "forward", "forward", "gpu"),
    ("collective_buffer", 100, "backward", "backward", "gpu"),
    ("offloaded_activation", 250, "forward", "backward", "cpu"),
]
offloaded = [(n, mb, s, e, "cpu" if n == "activations" else d) for n, mb, s, e, d in base]

peaks = {}
for label, states in (("基线", base), ("候选", offloaded)):
    totals = phase_totals(states)
    peaks[label] = max(totals.values())
    print(label, totals, "峰值阶段", max(totals, key=totals.get))
print("理论节省", peaks["基线"] - peaks["候选"])
```

```text
基线 {'load': 1400, 'forward': 1900, 'loss': 1700, 'backward': 2200, 'update': 1800, 'zero_grad': 1400} 峰值阶段 backward
候选 {'load': 1400, 'forward': 1600, 'loss': 1400, 'backward': 1900, 'update': 1800, 'zero_grad': 1400} 峰值阶段 backward
理论节省 300
```

基线在 `backward` 阶段最高：参数、优化器状态、激活值、梯度、通信缓冲区同时在场。前向工作区只在 `forward` 阶段存在，所以 `loss` 阶段就少了 200。候选把激活值移到 CPU 后，峰值仍在 `backward`，但少了 300。放在 CPU 上的 `offloaded_activation` 从头到尾没有计入。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 参数、激活值、梯度、优化器状态、工作区、通信缓冲区各在什么时候产生和释放 | 读 |
| Step 2 | 怎样按阶段汇总仍在 GPU 上的状态，一条状态记录要有哪几项 | 读，对照上面的公式 |
| Step 3 | 梯度累积、激活检查点、激活卸载、分片、混合精度各自改变了账本的哪一部分 | 读 |
| Step 4 · TODO 1 | 判断生命周期区间是否有效 | 在 `validate_state_lifecycle` 里补 1 行：比较 `start_phase` 和 `end_phase` 在 `PHASES` 里的位置 |
| Step 4 · TODO 2 | 选择当前阶段仍驻留在 GPU 的状态 | 在 `summarize_phase_memory` 里补一个列表推导：设备是 `gpu`，并且区间覆盖当前阶段 |
| Step 4 · TODO 3 | 选择最早出现的峰值记录 | 在 `find_peak_overlap` 里补 1 行：按 `total_mb` 取最大的阶段记录 |
| Step 4 · TODO 4 | 比较候选账本与基线峰值 | 在 `compare_peak_candidates` 里补一个循环：每个候选重新汇总并找峰值，再用基线峰值减去候选峰值 |
| 测试 | 检查非法记录是否被拒绝、各阶段的驻留汇总、峰值阶段和贡献排序，以及候选比较 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| 框架一致性核对（可选） | 用同一段受控的 CPU 计算比较本节账本与 PyTorch profiler | 需要 `TRAIN-CPU` 环境，可以先跳过 |

整本 notebook 只需要 CPU（`BASE-CPU` 环境）。框架核对需要另外的 `TRAIN-CPU` 环境，可以先跳过。

## 容易踩的坑

- **把所有状态的容量直接相加**：峰值是某个阶段仍在 GPU 上的状态重叠出来的。要按阶段汇总，只算当前阶段还驻留的状态。
- **驻留区间漏掉了端点**：起止阶段都计入，是闭区间。比如梯度从 `backward` 到 `update`，这两个阶段都要算上。
- **把 CPU 上的状态算进 GPU 峰值**：设备是 `cpu` 的状态不计入 GPU 的阶段总量，但可以留在完整账本里。
- **并列峰值时取了靠后的阶段**：阶段记录按训练顺序生成，容量并列时保留最早的那个。
- **接受了倒置的区间**：开始阶段晚于结束阶段，就没有合法的驻留区间，要抛出 `ValueError`。容量为负、未知类别、未知阶段也同样要拒绝。

## 读完后自查

- [ ] 能说出一条状态记录至少要有哪几项。
- [ ] 能解释为什么峰值不是所有状态的容量相加。
- [ ] 用例子的状态表，能说出 `backward` 阶段有哪些状态在场，总量是多少。
- [ ] 能说出并列峰值时取哪个阶段，贡献对象按什么排序。

## 下一步

- 打开 notebook：[43. 统一训练内存账本](../43_Unified_Training_Memory_Ledger.md)
- 建议先读：[11. 梯度累积 · 小白入门](./11_Gradient_Accumulation.md)、[19. 激活检查点 · 小白入门](./19_Activation_Checkpointing.md)、[42. 激活卸载 · 小白入门](./42_Activation_Offload.md)
- 想补背景可以读专题页：[性能优化 · 训练状态与后训练资源账本](../../topic_discussion/performance_optimization/03_training_memory_ledger.md)
- 做完之后：[56. 持续预训练与能力保留 · 小白入门](./56_Continual_Pretraining_and_Retention.md)
