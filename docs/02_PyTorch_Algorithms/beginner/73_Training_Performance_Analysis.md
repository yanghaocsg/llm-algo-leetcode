# 73. Training Performance Analysis | 训练性能分析 · 小白入门

> 配套 notebook：[73. 训练性能分析](../73_Training_Performance_Analysis.md) ｜ 难度：Hard ｜ 环境：CPU 可完成代码与模板验证；有 GPU 时可补充真实运行数据

## 一句话说清

训练太慢或者显存不够时，先别急着改。这本 notebook 教你把一次训练的耗时和显存记成一份基线，再从里面读出瓶颈可能在哪里。

## 为什么需要它

训练慢或者显存不够时，最容易凭感觉改：看到反向传播耗时最长，就立刻去换算子、换精度；看到显存快满了，就加上激活检查点。改完再跑一次，快了一点就说有效。这样做有两个问题答不上来：改动前到底是多少？这次的变化是改动带来的，还是机器本来就在波动？

所以要先把一次训练的成本记清楚。这本 notebook 把它拆成三份证据：

- 显存对象账本：参数、梯度、优化器状态、激活和临时工作区各占多少显存，解释容量由什么构成。
- 训练步计时与吞吐：一步要多久、每秒处理多少样本，描述运行时看到的现象。
- 阶段时间线追踪（trace）：每个阶段各花多久，用来定位等待发生在哪里。

有了基线还要会读。重复测量的结果要足够稳定，才能继续找原因；看到某个阶段耗时最高，也只能先提出假设，再交给下一项实验去验证。

这本 notebook 的主线在 CPU 上练熟三个判断：一份基线报告能不能用来归因、重复样本怎样合成证据、证据指向哪一种瓶颈假设。真实的 GPU 测量放在 Step 5。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 训练步（training step） | 准备一个批次的数据，并完成一次参数更新 |
| 基线（baseline） | 固定模型、数据批次和更新配置后测出来的成本记录，后面的改动都和它比 |
| 显存对象账本 | 一张清单：参数、梯度、优化器状态、激活、临时工作区各占多少显存 |
| 优化器状态 | 优化器为了更新参数而额外保存的数据。测试里的 AdamW 在第一次 `optimizer.step()` 之后才创建 |
| 激活值（activation） | 前向传播算出来、反向传播还要用到的中间结果 |
| 变异系数（CV） | 标准差除以平均值，衡量重复测量的波动有多大 |
| 单变量实验 | 每次只改一个因素的对照实验，用来验证或推翻一个瓶颈假设 |

## 核心思路

输入是一份基线报告：它说明测的是哪个固定训练任务（模型、数据批次、更新配置都固定），运行状态如何，以及若干条重复样本。每条样本带着训练步耗时、吞吐、峰值显存和各阶段耗时。另外还有显存对象账本和显存容量。处理分三步，对应三个 TODO；输出是一条瓶颈假设和下一项实验。

先说训练步。notebook 的定义是“准备一个 batch，并完成一次参数更新”，下面五个阶段都算在内：

```text
输入准备 → 梯度清理 → 前向与损失计算 → 梯度计算 → 参数更新
```

两次测量必须从同一个位置开始、到同一个位置结束，否则没法比。

TODO 1 是准入：报告里要有 `workload_id`，`status` 必须是 `ok`，样本至少 3 条，每条样本是字典并且带齐 `step_time_ms`、`samples_per_s`、`peak_memory_mb`、`stages_ms` 四个字段。任何一项不满足，就记进 `issues`，停止归因。

TODO 2 把重复样本合成证据。下面是一个最小例子，三条样本各带五个阶段的耗时：

```python
import statistics

samples = [  # 同一个训练步重复测 3 次，每个阶段的时间单位是毫秒
    {'stages_ms': {'输入准备': 10, '梯度清理': 1, '前向与损失计算': 30, '梯度计算': 80, '参数更新': 10}},
    {'stages_ms': {'输入准备': 10, '梯度清理': 1, '前向与损失计算': 32, '梯度计算': 84, '参数更新': 10}},
    {'stages_ms': {'输入准备': 12, '梯度清理': 1, '前向与损失计算': 31, '梯度计算': 82, '参数更新': 11}},
]

step_times = [sum(s['stages_ms'].values()) for s in samples]
cv = statistics.pstdev(step_times) / statistics.fmean(step_times)
print(step_times, round(cv, 4), cv <= 0.10)

names = samples[0]['stages_ms'].keys()
stage_medians = {name: statistics.median(s['stages_ms'][name] for s in samples) for name in names}
dominant = max(stage_medians, key=stage_medians.get)
share = stage_medians[dominant] / sum(stage_medians.values())
print(stage_medians)
print(dominant, round(share, 4), share >= 0.40)
```

```text
[131, 137, 137] 0.021 True
{'输入准备': 10, '梯度清理': 1, '前向与损失计算': 31, '梯度计算': 82, '参数更新': 10}
梯度计算 0.6119 True
```

第一行是三次训练步的总耗时（毫秒）和变异系数，变异系数不超过 0.10，说明测量稳定。第二行是每个阶段在三次重复里的中位数。第三行是主导阶段和它的占比：梯度计算占全部阶段耗时的 0.6119，超过 0.40 的门槛。注意取值方式：阶段耗时、训练步耗时和峰值显存都取重复样本的中位数，只有变异系数的分母用平均值。

TODO 3 按下面的顺序给出假设，前面的条件优先：

```text
step_cv 大于 0.10                   → rerun_required：测量不稳，先复测
capacity_ratio 大于等于 0.90        → ready：假设是容量压力，下一项实验交给 76
dominant_stage_share 大于等于 0.40  → ready：假设是主导阶段在关键路径上
都不满足                           → needs_more_evidence：先补阶段时间线
```

`capacity_ratio` 是峰值显存除以显存容量。例子里的数据对应第三行：如果容量占用率没到 0.90，假设就是“梯度计算位于关键路径”，下一项实验是比较检查点、精度或算子候选。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 一份可用的基线证据由哪三部分组成：显存对象账本、训练步计时与吞吐、阶段时间线 | 读 |
| Step 2 | 一次训练步包含哪五个阶段，测量从哪里开始、到哪里结束 | 读 |
| Step 3 | 四种观察结果各自对应什么可验证假设，交给下一阶段什么动作 | 读，对照上面的例子 |
| Step 4 · TODO 1 | 核对每条样本的必需字段 | 写 `validate_training_baseline`：补 1 行，样本是字典且包含 `required` 里全部字段才算完整 |
| Step 4 · TODO 2 | 汇总阶段耗时并识别主导阶段 | 写 `build_training_baseline_evidence`：先补 `samples = report['samples']`，再补 2 行，每个阶段取重复样本的中位数，取中位数最高的阶段 |
| Step 4 · TODO 3 | 判断容量是否已形成瓶颈 | 写 `classify_training_bottleneck`：补 1 行，`capacity_ratio` 达到 `capacity_alert_ratio` 时为真 |
| 测试 | 先检查 CPU 计时和显存账本，再检查完整与缺失的报告、稳定性与阶段热点、复测和容量压力的先后 | 运行 |
| 参考代码与解析 | 参考答案和解析 | 卡住再看 |
| Step 5（项目实验） | 在 GPU 上各重复三次，比较 FP32 与 BF16 的单步耗时、吞吐和峰值显存，运行顺序是 prepare、measure、close | 需要 GPU（显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境；默认是 `RUN_STAGE = 'prepare'`，只读取和检查，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU（显存至少 8 GiB）和 `PERF-CUDA-MECHANISM` 环境，默认停在 `prepare`。

## 容易踩的坑

- **补 TODO 2 时漏了 `samples`**：题目区的 `build_training_baseline_evidence` 一上来就使用 `samples`，函数里却没有给它赋值。参考答案第一行是 `samples = report['samples']`，要一起补上。
- **被参考答案里的旧注释带偏**：答案里还留着“TODO 1 对应挖空：开始正式计时”这类注释。计时和差值函数在题目区已经写好，现在要补的只有 TODO 1–3 这三处。
- **拿不稳定的测量去归因**：`step_cv` 大于 0.10 说明重复测量波动过大，这时返回 `rerun_required`，先复测，不往下判断瓶颈。
- **报告不全就继续**：`workload_id` 缺失、`status` 不是 `ok`、样本少于 3 条、样本缺 `required` 里的字段，任何一项都要进 `issues`，`is_valid` 为假。
- **把 CPU 结果当成显存结论**：CPU 的 `measure_train_step` 把 `peak_mem_mb` 固定为 0.0，测试会检查这一点。真实峰值显存由 Step 5 采集。

## 读完后自查

- [ ] 能说出一次训练步包含的五个阶段。
- [ ] 能列出 TODO 1 里每条样本必须有的四个字段。
- [ ] 给出三条重复样本的阶段耗时，能算出主导阶段和它的占比。
- [ ] 能说出复测、容量压力、主导阶段、证据不足四种结论出现的先后顺序。

## 下一步

- 打开 notebook：[73. 训练性能分析](../73_Training_Performance_Analysis.md)
- 建议先读：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)
- 需要补基础：[优化器与损失](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)、[Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- 做完之后：[76. Activation / Checkpoint / Offload 对比项目 · 小白入门](./76_Activation_Checkpoint_Offload_Benchmark.md)
