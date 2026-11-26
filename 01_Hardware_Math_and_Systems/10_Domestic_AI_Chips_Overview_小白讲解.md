# 10. 加速器选型与软件栈适配 — 小白讲解

> 对应 notebook：[`10_Domestic_AI_Chips_Overview.ipynb`](./10_Domestic_AI_Chips_Overview.ipynb)

## 一句话总结
本节不评比任何一家芯片，而是用三个纯 CPU 函数把"这个模型能不能换到另一种加速器上跑"拆成可回答的问题：`accelerator_feasibility` 检查容量、算子、dtype 三个硬条件，`stack_readiness` 检查驱动、编译/产物、运行时、内核和通信五层软件是否齐备，`admit_accelerator_candidate` 再用迁移工期、供应风险和功率上限筛掉不合格候选。结论是：16 GB 的设备能接住 8 GB 需求，但接不住 20 GB；容量相同也不代表迁移工作量相同，缺 `softmax` 或缺目标编译产物都必须单独列出。

## 为什么需要它：痛点是什么
换加速器时最常见的三个误判：**"峰值算力更高就能跑"**（结果模型要的算子它没有）、**"Python 调用一样就是兼容"**（底层可能完全换了一套运行时）、**"显存一样大就能直接搬"**（模型能加载，但自定义内核没有目标产物）。这些问题的共同点是：它们都不是一个"综合分"能回答的，必须逐项说出缺什么。

notebook 的文件名虽然叫 `10_Domestic_AI_Chips_Overview.ipynb`，正文标题却是"加速器选型与软件栈适配"，而且刻意不点名任何品牌，也不按产地推算风险。它把"国产替代"还原成一个具体的工程问题：同一份模型产物换到另一套软硬件栈，容量、精度、算子、编译产物、通信、工期、功率逐项核对。这样得到的结论才可复核。

生活类比：把一家餐厅的菜谱搬到另一个厨房。灶的火力（峰值算力）只是一个数字；你还要看锅具尺寸（容量）、有没有对应的灶眼（算子）、食材规格（dtype）、菜谱能不能被这里的厨师读懂（编译/运行时）、有没有传菜通道（通信），以及装修要多久（迁移工期）、总电闸够不够（功率）。

## 核心问题
- 判断"能不能跑"时，容量、算子、dtype 三个条件的检查顺序和边界是什么？
- 为什么 16 GB 设备接得住 8 GB 需求，却接不住 20 GB？"等于"算不算够？
- 软件栈为什么要分成驱动、编译/产物、运行时、内核、通信五层？哪一层最容易漏？
- 单设备路径和多设备路径的组件清单差在哪？为什么不能套同一张安装清单？
- 迁移 6 周、风险 0.4、功率 0.5 kW 的候选为什么能过，而软件没补齐的候选不行？

## 解决思路：一步步看懂

### 第 1 步：Q1——容量、算子、dtype 三道硬门槛

`accelerator_feasibility(required_peak_bytes, device_memory_bytes, required_ops, supported_ops, required_dtype, supported_dtypes)` 只回答"声明上是否可行"，不加载模型、不读本机规格。先校验两个字节数都是有限正数：

```python
if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
       for value in (required_peak_bytes, device_memory_bytes)):
    raise ValueError('memory bytes must be positive')
```

然后按固定顺序收集原因：

```python
reasons = []
if required_peak_bytes > device_memory_bytes:
    reasons.append('insufficient_memory')
missing_ops = sorted(set(required_ops) - set(supported_ops))
if missing_ops:
    reasons.append('missing_ops')
if required_dtype not in supported_dtypes:
    reasons.append('unsupported_dtype')
return {'feasible': not reasons, 'reasons': reasons, 'missing_ops': missing_ops}
```

三个细节：

1. **容量是严格大于**：需求 20 GB、设备 16 GB 才记 `insufficient_memory`；需求正好 16 GB、设备 16 GB 是可行的。测试专门有一条 `accelerator_feasibility(16e9, 16e9, {'matmul'}, {'matmul'}, 'bf16', {'bf16'})['feasible']` 为真，验证这个边界。
2. **算子是差集**：`set(required_ops) - set(supported_ops)` 得到缺的算子，再 `sorted` 成列表。notebook 的 `ready` 需求是 `{matmul, softmax}`、候选也是 `{matmul, softmax}`，可行；`blocked` 候选只有 `{matmul}`，差集是 `['softmax']`。
3. **dtype 是成员判断**：需求只能是一个字符串，候选是集合。需求 `bf16`、候选 `{fp16}` 就记 `unsupported_dtype`。

`blocked` 的三项全踩：需求 20 GB、设备 16 GB；缺 `softmax`；只支持 `fp16`。断言 `reasons == ['insufficient_memory', 'missing_ops', 'unsupported_dtype']`，顺序就是代码里的检查顺序，不是字母序。`missing_ops == ['softmax']` 单独返回，是为了让使用者能说出"到底缺哪个算子"，而不是只知道"不可行"。

这里的关键区分：**容量需求是同一阶段的峰值工作集**，包括权重、状态和工作区，不只是权重文件大小。8 GB 的模型文件在推理时可能因为 KV Cache、临时缓冲而超过 8 GB；notebook 的输入叫 `required_peak_bytes`，就是提醒你填峰值而不是文件大小。另外，Q1 只做声明层面的比较：设备说支持 BF16，不代表这条路径的矩阵乘真的走 BF16、真的更快；那要在固定负载上测带宽和计算时间。

### 第 2 步：Q2——软件栈的五层清单

`stack_readiness(driver, compiler, runtime, kernel, communication, kernel_coverage=1.0, requires_communication=False)` 把软件路径拆成五层：

- **driver**：驱动，操作系统和设备对话的最底层；
- **compiler**：编译能力或已经生成的目标产物；
- **runtime**：执行框架，负责加载产物、安排算子；
- **kernel**：本任务关键算子有没有可用实现；
- **communication**：跨设备集合通信组件。

前四层是任何路径都需要的，第五层只在 `requires_communication=True` 时加入：

```python
required = ['driver', 'compiler', 'runtime', 'kernel']
if requires_communication:
    required.append('communication')
missing = [name for name in required if not layers[name]]
if kernel_coverage != 1:
    missing.append('kernel_coverage')
```

这个非对称很重要：单进程、无集合通信的任务不必先装通信库；只有当模型真的要跨设备传梯度或 token 时，缺 `communication` 才算缺项。notebook 的 `component_states` 是 `driver=True, compiler=True, runtime=True, kernel=True, communication=False`，所以 `single_device` 的 `declared_requirements_met` 为真，而 `multi_device` 的 `missing == ['communication']`。

第五个检查是**关键算子覆盖率**，默认 1.0。只要不等于 1（比如 0.9），就追加 `'kernel_coverage'`，而且排在所有缺失层之后。这条比布尔值更细：即使驱动、编译、运行时、内核都"有"，但本任务要的算子只覆盖了九成，仍然不能算齐备。

返回值有四个字段：`declared_requirements_met`（缺失列表为空）、`missing`（按固定顺序）、`first_reported_gap`（`missing[0]`，没有则为 `None`）、`kernel_coverage`。`first_reported_gap` 的用途是先修最前面那一层——驱动没装时，先去补内核没有意义。

测试覆盖了四种组合：全齐且不要通信 → 通过；全齐但要通信 → 缺 `['communication']`；`compiler=False` → 缺 `['compiler']`；`compiler=False, kernel=False` → 缺 `['compiler', 'kernel']`。顺序始终是 driver → compiler → runtime → kernel → communication，最后才是 `kernel_coverage`。

notebook 还给了一个很容易被忽略的提醒：Python 调用长得一样，不代表底层一样。PyTorch 的 ROCm 构建复用了许多 `torch.cuda` 接口，同一行调用在非 NVIDIA 设备上也可能照常执行；自定义扩展还要核对目标工具链与产物。所以"代码没改就能 import"不能当成"软件栈已适配"的证据。

### 第 3 步：Q3——工期、供应风险、功率

`admit_accelerator_candidate(feasible, stack_ready, migration_weeks, max_migration_weeks, supply_risk, max_supply_risk, power_kw, power_cap_kw)` 在完整 TCO 计算之前做硬约束筛选。先校验前两个是布尔值、其余是非负有限数、两个风险值都在 $[0,1]$，然后按顺序收集原因：

```python
if not feasible: reasons.append('device_constraints')
if not stack_ready: reasons.append('software_stack')
if migration_weeks > max_migration_weeks: reasons.append('migration_schedule')
if supply_risk > max_supply_risk: reasons.append('supply_risk')
if power_kw > power_cap_kw: reasons.append('power_cap')
```

三个数值门槛都是**严格大于**才算超：迁移 8 周对上限 8 周合法，风险 0.5 对上限 0.5 合法，功率 1.0 kW 对 1.0 kW 合法。

notebook 的三个候选都用同一个任务需求（`ready['feasible']` 为真、`single_device['declared_requirements_met']` 为真）：

| 候选 | 迁移周 | 供应风险 | 功率 | 软件栈 | 结果 |
|---|---:|---:|---:|---|---|
| `baseline` | 2（上限 8） | 0.2（上限 0.5） | 0.7 kW（上限 1.0） | 齐备 | 通过 |
| `alternative` | 6（上限 8） | 0.4（上限 0.5） | 0.5 kW（上限 1.0） | 齐备 | 通过 |
| `immature_stack` | 4（上限 8） | 0.3（上限 0.5） | 0.6 kW（上限 1.0） | 缺编译/内核 | 拒绝 |

断言 `admitted == ['baseline', 'alternative']`，`immature_stack['reasons'] == ['software_stack']`。`immature_stack` 的工期、风险、功率全在限内，唯独 `stack_ready=False`（它传入的是 `needs_target_artifacts['declared_requirements_met']`，即缺 `compiler` 和 `kernel`），所以被一条软件原因挡住。这正说明筛选是"一票否决"，不是打分求和。

关于 `supply_risk`，notebook 说得很克制：它是 0 到 1 的假设输入，不是从设备名称或产地推算出的真实故障概率，必须来自明确调查或场景假设。这一点很重要——把"国产"直接等同于"供应风险高"是偏见，不是数据。

### 第 4 步：三个函数怎样串成一条决策链

这三个函数不是并列的，而是有先后：

1. 先用 `accelerator_feasibility` 回答"硬件声明上能不能接住这个任务"——容量、算子、dtype 任意一项不满足，后面的软件和成本都不用谈。
2. 再用 `stack_readiness` 回答"这条具体执行路径的软件齐不齐"——同一块卡，单设备推理和跨设备训练需要的组件不同。
3. 最后用 `admit_accelerator_candidate` 回答"在工期、供应、功率的约束下，这个候选还值不值得进入验证"。

每一步的返回值都保留**具体原因列表**，而不是一个"可用/不可用"的布尔。这是本节的方法论：不要用一个综合分掩盖问题，要能指出"缺 `softmax`""缺 `communication`""迁移超 2 周"。保留下来的候选也只进入下一步验证（34 节的阶段交接、33 节的 TCO），不代表已经采用；采用还需要共同功能、质量和性能记录。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 峰值工作集 | 同一阶段真正要同时放下的权重、状态和工作区 |
| 算子支持 | 设备/框架有没有这个运算的实现 |
| dtype 支持 | 设备/框架能不能算这种数值格式 |
| 驱动（driver） | 操作系统和设备对话的最底层软件 |
| 运行时（runtime） | 加载产物、安排算子执行的框架 |
| 编译/产物 | 把模型或内核编成目标设备能执行的文件 |
| 算子覆盖率 | 本任务需要的算子里，有多少已有实现 |
| 集合通信 | 多设备之间交换数据的操作，如 All-Reduce |
| 迁移工期 | 把现有代码和产物搬到新栈所需的时间 |
| 供应风险 | 0 到 1 的假设分值，来自调查而非品牌 |
| 一票否决 | 任一硬条件不满足就拒绝，不做加权平均 |

## 小白常见疑问

**Q1：设备峰值算力更高，为什么还可能跑不了这个模型？**
因为 `accelerator_feasibility` 根本不看峰值算力，只看三件事：显存装不装得下峰值工作集、算子支不支持、dtype 支不支持。峰值再高，缺 `softmax` 就是缺，20 GB 需求放进 16 GB 就是放不下。算力只决定"跑得多快"，这三项决定"能不能跑"。

**Q2：容量需求正好等于设备容量，算通过吗？**
算。代码是 `required_peak_bytes > device_memory_bytes` 才拒绝，测试里 `16e9` 对 `16e9` 的 `feasible` 为真。但这是"声明上可行"，真实运行还要留出驱动、上下文和碎片，所以贴着容量上限通常并不稳妥；本节只做声明比较，不内置预留参数。

**Q3：为什么单设备路径不检查 `communication`？**
因为 `required` 列表只在 `requires_communication=True` 时追加这一项。单进程任务没有跨设备集合通信，先装通信库是浪费；但一旦模型要切到多卡，缺 `communication` 就必须列出来。同一个 `component_states`，`single_device` 通过、`multi_device` 缺一项，差别只在需求。

**Q4：`kernel_coverage=0.9` 时，`missing` 里是什么？**
如果五层都齐、但覆盖率是 0.9，`missing` 只有 `['kernel_coverage']`，因为覆盖率检查排在最后。它不会展开成"缺哪个算子"——本节只做覆盖率这一层的布尔判断，具体缺哪个算子要回到 Q1 的 `missing_ops` 或更细的算子清单。

**Q5：`immature_stack` 只缺软件，为什么不能"先通过了再补"？**
函数的设计就是一票否决：`stack_ready=False` 直接进 `reasons`，工期和功率再宽也没用。现实里当然可以先补软件再评估，但那属于下一轮候选；本节只回答"现在这个状态能不能进入验证"，答案是"不能，缺 `software_stack`"。

## 动手验证：怎么知道自己做对了

这个 notebook 没有保存执行输出，也没有 TODO 空位，以下结论以断言为准。

- **Q1**：`ready['feasible'] is True`；`blocked['feasible'] is False`；`blocked['reasons'] == ['insufficient_memory', 'missing_ops', 'unsupported_dtype']`；`blocked['missing_ops'] == ['softmax']`；`16e9` 对 `16e9` 的边界用例 `feasible` 为真。打印 `ready` 和 `blocked` 两个字典。
- **Q2**：`single_device['declared_requirements_met']` 为真；`multi_device['missing'] == ['communication']`；`compiler=False` 时 `missing == ['compiler']`；`compiler` 和 `kernel` 都缺时 `missing == ['compiler', 'kernel']`。打印三种情况。
- **Q3**：`admitted == ['baseline', 'alternative']`；`immature_stack['reasons'] == ['software_stack']`。打印三个候选的完整结果字典。
- 三个函数各有非法输入分支：Q1 传 0 或负数会抛 `ValueError('memory bytes must be positive')`；Q2 传非布尔组件或覆盖率超出 $[0,1]$ 会抛 `ValueError`；Q3 传负数或风险超出 $[0,1]$ 会抛 `ValueError`。

改参数做对照：把 `blocked` 的需求从 20 GB 改成 16 GB，`insufficient_memory` 会消失，但 `missing_ops` 和 `unsupported_dtype` 仍在；把 `stack_readiness` 的 `kernel_coverage` 改成 0.9，看 `missing` 是否只剩 `['kernel_coverage']`；把 `alternative` 的 `migration_weeks` 改成 9，看它是否从 `admitted` 掉出并多出 `migration_schedule`。这三组分别验证"条件是独立的""覆盖率是最后一层""门槛是严格大于"。
