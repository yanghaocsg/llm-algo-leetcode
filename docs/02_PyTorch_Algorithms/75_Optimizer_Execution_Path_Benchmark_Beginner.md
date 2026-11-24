# 75. Optimizer Execution Path Benchmark Beginner | AdamW 执行路径基准 · 小白入门

> 配套 notebook：[75. AdamW 执行路径基准](./75_Optimizer_Execution_Path_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU 验证比较规则；单 GPU 测量执行路径

## 一句话说清

训练时每一步都要更新一次参数，这个更新有好几种执行写法，快慢不同。这本 notebook 教你判断某种写法值不值得换：整步是否真的变快，显存和参数结果有没有变差。

## 为什么需要它

训练一步要做三件事：前向传播、反向传播、优化器更新。最后一件由 AdamW 完成。同一个 AdamW 更新在 PyTorch 里可以走三条执行路径：`single_tensor` 逐个处理张量，`foreach` 把同类张量的运算批量提交，`fused` 把更新里的多个运算融合在一起。后两条有机会把重复的工作合并，但实际差多少，还受张量形状、数据类型和 PyTorch/CUDA 版本影响。

想换路径的人常犯三种错：

- 只测优化器调用。单独测快了 2 倍，就以为整个训练步也快 2 倍。可优化器更新只占一步的一部分，比如占 20%，整步其实只快约 11%。
- 比较时条件没固定。一条路径的学习率或权重衰减和别的不同，测出来的差异说不清是路径带来的，还是配置带来的。
- 只看速度。是快了，但峰值显存涨了不少，或者更新出来的参数和原路径对不上。

所以一条路径能不能用，要同时看完整训练步的耗时、显存增量和参数误差。这本 notebook 做的就是这件事：先核对三条候选的训练条件一致，再用三项门槛筛选。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 执行路径（execution path） | 同一个 AdamW 更新在 PyTorch 内部的实现路线，本节比较 `single_tensor`、`foreach`、`fused` 三条 |
| 契约 | 比较之前约定好的条件清单：三条候选只能差在执行路径上，其他训练条件必须相同 |
| 优化器调用 / 完整训练步 | 优化器调用只含“按梯度更新参数”这一步；完整训练步还包括前向传播和反向传播 |
| kernel 启动开销 | kernel 是 GPU 上运行的一段计算程序，每启动一次都有调度和启动的代价。逐张量处理时，这个代价要付很多次 |
| 峰值分配显存 | 一次测量期间，PyTorch 实际分配的显存的最高值 |
| 最大参数误差 | 换路径更新之后，把参数和单张量路径的结果逐项相减，取绝对值最大的那个 |
| `accept` / `tune` / `pending` | 决策的三种结果：采用、先不采用并回去调整、证据不够不能下判断 |

## 核心思路

输入是三条候选和它们的测量记录。每条候选写着名字、优化器、学习率、权重衰减、调度器，还有 `foreach`、`fused` 两个开关。输出是 `accept`、`tune`、`pending` 三者之一。

处理分三步。第一步核对契约：三条候选都是 AdamW，学习率、权重衰减和基准相同，调度器是 `None`；候选名和两个开关要对得上，比如 `fused` 的开关是 `{'foreach': None, 'fused': True}`。第二步核对证据可比：候选齐全、不重复，`workload_id` 和 `seed` 与基线相同，测量值齐全，否则返回 `pending`。第三步对每条候选算三个量，和基线比：

```text
加速比       = 基线训练步耗时 / 候选训练步耗时       至少 1.05
显存增量 MiB = 候选峰值显存 - 基线峰值显存           不超过 256
参数误差     = 候选相对单张量路径的最大参数误差       不超过 1e-4
```

三项同时通过才算合格。有合格候选就返回 `accept`，并选整步加速比最高的那条；一条都不合格返回 `tune`。记录里的耗时都是中位数，也就是把重复测量排好序后取中间那个。

为什么要看完整训练步？优化器更新的局部提速会被前向传播和反向传播稀释。设基线里优化器更新占训练步的比例是 `p`，它的局部加速比是 `s`，其余阶段耗时不变，那么整步加速比是：

```text
S = 1 / ((1 - p) + p / s)
```

`S` 是完整训练步加速比。代入两组数，再看一条候选怎样过门槛：

```python
def step_speedup(p, s):
    # p：基线里优化器更新占完整训练步耗时的比例
    # s：优化器更新的局部加速比
    return 1 / ((1 - p) + p / s)

print(round(step_speedup(0.2, 2), 3), round(step_speedup(0.05, 10), 3))

base = {'training_step_ms': 100.0, 'peak_allocated_mb': 1000.0}
cand = {'training_step_ms': 96.0, 'peak_allocated_mb': 1256.0, 'max_parameter_error': 1e-7}
speedup = base['training_step_ms'] / cand['training_step_ms']
memory_increase = cand['peak_allocated_mb'] - base['peak_allocated_mb']
print(round(speedup, 3), memory_increase,
      speedup >= 1.05, memory_increase <= 256, cand['max_parameter_error'] <= 1e-4)
```

```text
1.111 1.047
1.042 256.0 False True True
```

优化器更新占 20%、局部快 2 倍，整步只快到 1.111 倍。局部快 10 倍但只占 5%，整步是 1.047 倍，到不了 1.05 的门槛。下面的候选把训练步从 100 毫秒降到 96 毫秒，加速比 1.042；另外两项通过了，也不合格。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 固定训练条件，对比 AdamW 的执行路径：三个比较环节各固定什么、记录什么，以及三项采用门槛 | 读 |
| Step 2 | 张量布局（多小张量和少大张量）怎样改变更新成本 | 读 |
| Step 3 | 局部加速比怎样被稀释成整步加速比，四种证据各自怎么用 | 读，对照上面的公式 |
| Step 4 · TODO 1 | 核对共同训练配置 | 在 `validate_comparison_contract` 里补 `common_config_matches`：优化器、学习率、权重衰减、调度器四项都和基准一致 |
| Step 4 · TODO 2 | 核对候选执行路径 | 在同一个函数里补 `execution_path_matches`：把候选的两个开关和 `OPTIMIZER_PATHS` 里的定义逐项比较 |
| Step 4 · TODO 3 | 筛选可采用路径 | 在 `select_optimizer_path` 里补 `eligible`：加速比、显存增量、参数误差三项同时过门槛 |
| 测试 | 候选缺失、重复、配置漂移、路径错配时要判无效；局部更快但整步不达标、显存或误差超限时不能采用；证据缺失时返回 `pending` | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（项目实验） | 在单 GPU 上实测三种执行路径的优化器调用和完整训练步，登记 `accept` 或 `tune` | 需要 `PERF-CUDA-MECHANISM` 环境和单卡至少 8 GiB 显存；默认是 `RUN_STAGE = 'prepare'`，只检查环境和比较合同，改成 `measure` 才使用 GPU 测量，`close` 只读结果并登记决策 |

Step 1–4 只需要 CPU。Step 5 需要 CUDA GPU（单卡显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境。

## 容易踩的坑

- **只看优化器调用的耗时**：优化器单步耗时只用来说明局部开销，不参与准入。一条路径的优化器调用更快，但完整训练步没有快到 5%，也不能采用。
- **候选之间训练条件不一样**：优化器、学习率、权重衰减或调度器变了，耗时差异就不能归因到执行路径。三条候选必须用同一套配置。
- **候选名和开关对不上**：候选叫 `foreach`，开关就必须是 `{'foreach': True, 'fused': False}`。`fused` 路径的 `foreach` 开关是 `None`，不是 `False`。
- **三项门槛只看了一项**：整步加速比、显存增量、参数误差要同时达标，任何一项超限都不能采用。
- **证据不全还硬做决策**：候选缺失或重复、`seed` 和基线不同、测量值缺失，都要返回 `pending`，不能给出 `accept` 或 `tune`。

## 读完后自查

- [ ] 能说出三条执行路径的名字，以及 `fused` 路径的两个开关取什么值。
- [ ] 优化器更新占训练步的 20%、局部快 2 倍时，能算出整步加速比。
- [ ] 能说出三项采用门槛，以及哪一种耗时只作解释、不参与准入。
- [ ] 能说出证据不全时返回什么，以及它和 `tune` 的区别。

## 下一步

- 打开 notebook：[75. AdamW 执行路径基准](./75_Optimizer_Execution_Path_Benchmark.md)
- 建议先读：[12. 优化器更新与学习率调度 · 小白入门](./12_Optimizer_Updates_and_Learning_Rate_Scheduling_Beginner.md)、[73. 训练性能分析 · 小白入门](./73_Training_Performance_Analysis_Beginner.md)
- 做完之后：[74. profiling 驱动的候选准入与交接 · 小白入门](./74_Profiling_Driven_End_to_End_Optimization_Beginner.md)
