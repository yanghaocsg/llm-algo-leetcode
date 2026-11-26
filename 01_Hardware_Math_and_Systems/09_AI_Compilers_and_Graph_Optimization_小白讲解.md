# 09. AI 编译器与计算图优化 — 小白讲解

> 对应 notebook：[`09_AI_Compilers_and_Graph_Optimization.ipynb`](./09_AI_Compilers_and_Graph_Optimization.ipynb)

## 一句话总结
本节用四个纯 CPU 函数把"编译器做了什么"拆成可核对的账本：`execution_plan_report` 判断哪些中间值必须跨组传递，`graph_cost` 把分组换算成逻辑读写字节和假设的启动微秒，`graph_rewrite_delta` 比较改写前后的净收益，`choose_backend` 先按算子、dtype、动态形状、工作区硬筛，再按假设延迟挑人。结论是：融合能省一次启动和 8 KiB 读写，但新增 8 KiB 布局转换就能把读写收益清零；能力不满足的候选哪怕延迟只有 0.5 ms 也必须先排除。

## 为什么需要它：痛点是什么
初学 `torch.compile` 的人常把三件事混成一件：**框架记下了什么计算**、**编译器改写成了什么**、**设备上真正跑了几个内核**。于是会问"加了 `torch.compile` 为什么没变快""融合了是不是一定省显存"，却不知道该死磕哪一层。另一个误区是把算子融合当成无条件的好事：融合少写一次中间结果、少启动一次内核，但为了满足新内核的布局要求，可能又要插一次转置或拷贝，这笔账必须分开记。

生活类比：三道工序分开做，每道之间要把食材装盘再端过去，每次换工序还要喊一声"开始"。合并两道能少端一次盘子、少喊一声；但如果合并后灶台摆放不合手，还得先把食材重新摆一遍（布局转换），省下的搬运可能又被这次重摆吃回去。

## 核心问题
- 程序里的 `X @ W + b` 再 ReLU，计算图、中间表示（IR）和设备实现各指什么？
- 把"加偏置"和"ReLU"合成一组后，哪些中间值还要写回内存？为什么 `T` 省不掉、`U` 可以？
- 融合后 16 KiB 变 8 KiB、15 μs 变 10 μs 是怎么算的？为什么这两个数不能相加？
- 新增布局读写 8 KiB 或 12 KiB 时净收益变多少？为什么"净读写没省"仍可能值得做？
- 后端候选的四个条件谁先谁后？为什么最快的候选反而落选？

## 解决思路：一步步看懂

### 第 1 步：Q1——把程序翻译成"节点 + 分组"

先固定一条最小链：矩阵乘产生 `T = X @ W`，加偏置产生 `U = T + b`，ReLU 把 `U` 的负值清零得到 `Y`。notebook 手工声明：

```python
node_inputs = {"T": ("X", "W"), "U": ("T", "b"), "Y": ("U",)}
node_ops = {"T": "matmul", "U": "add", "Y": "relu"}
external_inputs = {"X", "W", "b"}
```

`node_inputs` 是"谁依赖谁"，`external_inputs` 是图外输入。字典的书写顺序恰好是拓扑序，函数依赖这一点。

`execution_plan_report(node_inputs, groups, external_inputs)` 的 `groups` 是一份**假设的实现安排**：`[["T"], ["U"], ["Y"]]` 表示三个算子各占一个内核，`[["T"], ["U", "Y"]]` 表示后两个合成一个。函数返回 `users`、`materialized_intermediates`（哪些中间值必须写回内存）、`planned_calls`（等于 `len(groups)`）。

开头四道校验各挡一种建模错误：

```python
flattened = [name for group in groups for name in group]
if (not node_inputs or any(not group for group in groups)
        or len(flattened) != len(set(flattened)) or set(flattened) != set(node_inputs)):
    raise ValueError("各节点须恰好属于一个非空分组")
```

所有节点必须恰好出现一次：`len(flattened) != len(set(flattened))` 查重复，`set(flattened) != set(node_inputs)` 查遗漏和多写。第二道 `set(node_inputs) & external_inputs` 禁止节点结果与外部输入同名。

第三道按声明顺序查输入就绪：`available` 初始只有 `{X, W, b}`，逐节点消费并把自己加进去；若 `U` 写在 `T` 前面就报"节点输入须先于当前节点就绪"。这是教学简化，不是真正的图调度器。第四道用 `owner` 记录节点所属组，若 `owner[source] > owner[name]` 就报"分组执行顺序必须满足依赖"；允许 `owner[source] == owner[name]`，即同组内直接消费，这正是融合的关键。

最后算边界：

```python
boundaries = [name for name, readers in users.items()
              if any(owner[reader] != owner[name] for reader in readers)]
```

中间值要落地的唯一原因是**读者不在同一组**。`Y` 没读者，永远不是边界；`T` 的读者 `U` 在第 1 组、自己在第 0 组，是边界；`U` 在分离安排里读者 `Y` 在第 2 组，是边界，融合后与 `Y` 同组，就变成组内临时值。断言为：

```python
assert separate_plan["materialized_intermediates"] == ["T", "U"]
assert fused_plan["materialized_intermediates"] == ["T"]
assert separate_plan["planned_calls"] == 3 and fused_plan["planned_calls"] == 2
```

保存的输出是 `分离安排: 计划调用 3 次，组间中间值 ['T', 'U']` 和 `偏置与激活合并: 计划调用 2 次，组间中间值 ['T']`，两行结尾都强调"最终输出始终是Y"——分组改的是实现，不是语义。这一层没有生成代码、没有分配线程；notebook 只把 PyTorch 角色对应为 TorchDynamo 捕获、FX 图表示、TorchInductor 选实现，手写字典并不模拟真实捕获。

### 第 2 步：Q2——把分组换成字节账和启动账

`graph_cost` 的核心只有一行：

```python
materialized_boundaries = fused_groups - 1
intermediate_bytes = (
    materialized_boundaries * intermediate_elements * bytes_per_element * 2
)
```

三个因子：`fused_groups - 1` 是串行链的接缝数（$G$ 组有 $G-1$ 个接缝）；`intermediate_elements * bytes_per_element` 是一个中间值的大小；末尾乘 `2` 是因为落地要**写一次、读一次**，这个 2 最容易漏。

代入 Q1：`T`、`U` 各按 1024 个 FP32 元素算，一个值 $1024 \times 4 = 4096$ 字节 = 4 KiB。

- 分离安排 $G=3$：$(3-1) \times 1024 \times 4 \times 2 = 16384$ 字节 = 16 KiB。
- 融合安排 $G=2$：$(2-1) \times 1024 \times 4 \times 2 = 8192$ 字节 = 8 KiB。

启动项按"每组一次、每次 5 μs"的构造假设：3 组 15 μs，2 组 10 μs。`launch_overhead_us` 默认就是 `5.0`，真实开销随内核复杂度、驱动和队列状态变化，这里只是可比较的标尺。

函数用 `type(x) is not int` 而非 `isinstance`：Python 里 `True` 是 `int` 的子类，`type(True) is int` 为假，可以挡掉布尔值；`launch_overhead_us` 用 `math.isfinite` 挡掉 `nan`/`inf`。`num_ops` 只参与 `1 <= fused_groups <= num_ops` 的校验，不参与乘法，所以一组、零元素时 `total_extra_traffic_bytes == 0`。

第三个候选 `partial_with_layout` 在融合安排上加 `layout_bytes=8192`，`total_extra_traffic_bytes` 是 $8192 + 8192 = 16384$。返回字典把 `intermediate_traffic_bytes`、`layout_traffic_bytes` 分开列，`launch_overhead_us` 单独一个字段——代码刻意不把字节和微秒相加，因为单位不同。

### 第 3 步：Q3——净收益看"改写前减改写后"

`graph_rewrite_delta(before, after)` 只做减法，返回 `saved_launches`、`saved_launch_overhead_us`、`net_logical_traffic_saved_bytes` 和布尔值 `has_net_traffic_saving`。三个候选：

| 候选 | after 的额外读写 | 净省字节 | 读数 |
|---|---:|---:|---|
| `no_layout` | 8 KiB | 16384 − 8192 = 8192 | 真省 8 KiB |
| `equal_layout` | 8 KiB + 8 KiB = 16 KiB | 16384 − 16384 = 0 | 读写打平 |
| `more_layout`（`layout_bytes=12288`） | 8 KiB + 12 KiB = 20 KiB | 16384 − 20480 = −4096 | 反而多 4 KiB |

对应断言是 `== 8192`、`== 0`、`== -4096`。而启动维度上，`all(row["saved_launches"] == 1 ...)` 和 `all(row["saved_launch_overhead_us"] == 5 ...)` 说明三种情况都少一次启动、少 5 μs，因为分组数始终从 3 降到 2，布局字节不影响启动次数。

结论：**收益是多维的，不能用一个标量概括**。`more_layout` 的 `has_net_traffic_saving` 为 `False`，但启动收益仍在；反过来，若启动开销可忽略、带宽是瓶颈，`equal_layout` 这种白忙一场就不值得做。notebook 还提醒：寄存器、共享内存、实际调用要等生成代码和运行记录才能确认，"计划调用数"只是计划。

### 第 4 步：Q4——先过能力门槛，再比延迟

`choose_backend` 的需求有四项：`ops`、`dtype`、`dynamic_shape`、`workspace_limit_mib`；候选有 `ops`、`dtypes`、`dynamic_shape`、`workspace_mib`、`estimated_latency_ms`。筛选按固定顺序追加原因：

```python
if not requirements["ops"].issubset(candidate["ops"]):
    reasons.append("missing_ops")
if requirements["dtype"] not in candidate["dtypes"]:
    reasons.append("unsupported_dtype")
if requirements["dynamic_shape"] and not candidate["dynamic_shape"]:
    reasons.append("dynamic_shape")
if workspace_mib > limit_mib:
    reasons.append("workspace")
```

四个细节：运算是子集判断，需求 `{matmul, add, relu}` 对候选 `{matmul}` 会缺两项；dtype 是成员判断，需求一个字符串、候选一个集合；动态形状是**有条件**的，只有需求为 `True` 时才查候选，这个非对称很容易写反；工作区是**严格大于**，正好 512 MiB 在 512 上限下合法，测试专门用 `at_workspace` 验证边界。

通过全部条件的进入 `eligible`，然后才比延迟：

```python
selected = min(eligible, key=lambda name: candidates[name]["estimated_latency_ms"], default=None)
```

延迟相同时 `min` 保留先出现的那个；`default=None` 让空名单返回 `None` 而不抛异常。

测试的基准 `base` 是：支持三个算子、`{fp16, bf16}`、支持动态、256 MiB、3.0 ms。派生候选故意让"不合格者"更快，以证明能力优先于速度：

- `missing_op` 只支持 matmul，1.0 ms；`wrong_dtype` 只有 fp32，1.0 ms；
- `static_only` 不支持动态，1.0 ms；`over_workspace` 513 MiB，1.0 ms；
- `at_workspace` 正好 512 MiB，4.0 ms；`slower_supported` 6.0 ms；
- `multiple_misses` 四项全缺，0.5 ms。

结果 `selected == "supported"`，`eligible == ["supported", "at_workspace", "slower_supported"]`。`at_workspace` 虽比 `supported` 慢但合法，留在名单里；`multiple_misses` 的 0.5 ms 完全没用，`rejected` 记 `["missing_ops", "unsupported_dtype", "dynamic_shape", "workspace"]`，按检查顺序而非字母序排列。

边界断言：只有 `static_only` 时返回 `selected is None`、`eligible == []`；候选为空时返回 `{"selected": None, "eligible": [], "rejected": {}}`——`rejected` 是空字典，因为它的类型是"名字 → 原因列表"。

Q4 的动态形状只是简化布尔字段，notebook 明说它"不会捕获计算图或查询真实设备"；真正的编译版本复用要先过 guard，失败后可能查缓存、重编译或回退，留给 30 节。工作区同理，MiB 是声明值，真实占用要看运行时。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 计算图 | 用节点表示运算、连线表示"谁用谁的结果" |
| 图捕获 | 从 Python 程序里把依赖关系录下来 |
| IR（中间表示） | 编译器内部保存计算信息的形式，计算图是其中一种 |
| 后端 | 把图落实成库调用或生成代码的那一层 |
| materialized boundary | 必须写回内存、再被另一组读走的中间值 |
| 算子融合 | 把多个算子放进一次内核，组内中间值不落地 |
| 布局转换 | 为满足实现要求重排或复制数据，要单独记读写 |
| 启动开销 | 每发起一次内核调用要付的固定时间 |
| workspace | 执行期间临时用的显存，如矩阵乘的暂存区 |
| guard | 复用已编译版本前，检查输入是否仍符合要求 |

## 小白常见疑问

**Q1：为什么不把 `T` 也融进去，做成一个内核？**
可以。把分组写成 `[["T","U","Y"]]`，`planned_calls` 会是 1、`materialized_intermediates` 会是空列表，数学上完全合法。notebook 保留"矩阵乘库调用 + 融合偏置与激活"，是因为矩阵乘通常调用高度优化的库，盲目自己融合可能更慢。函数不判断哪种安排更好，只把成本算清楚。

**Q2：为什么中间值要乘 2 而不是 1？**
"落地"是写一次、读一次两个动作，公式末尾的 `* 2` 就是它。漏了这个 2，16 KiB 会被算成 8 KiB，正好和融合后的数字撞车。

**Q3：`partial_with_layout` 读写和原来一样多，为什么还说省 5 μs？**
启动次数和读写字节是两个独立维度。`graph_rewrite_delta` 同时返回 `saved_launch_overhead_us=5` 和 `net_logical_traffic_saved_bytes=0`。值不值得做，取决于当前瓶颈是启动延迟还是带宽。

**Q4：`at_workspace` 比 `supported` 慢，为什么还能进 `eligible`？**
`eligible` 的定义是"通过能力筛选"，不是"最快"。四项筛完后才按延迟取最小，`supported` 3.0 ms 胜出，`at_workspace` 4.0 ms 只是"合法但没被选"。如果 `supported` 不存在，它就上位。

## 动手验证：怎么知道自己做对了

本节没有 TODO 填空，断言直接写在单元格里，跑通即打印。这个 notebook 保存了执行输出，可以和下面的数字对照。

- **Q1**：`separate_plan["users"] == {"T": ["U"], "U": ["Y"], "Y": []}`；边界分别是 `["T", "U"]` 和 `["T"]`；`planned_calls` 为 3 和 2。输出第二行的组间中间值只有 `['T']`。
- **Q2**：`unfused` 的 `intermediate_traffic_bytes == 16384`、`launch_overhead_us == 15`；`partial` 对应 8192 和 10；`partial_with_layout["total_extra_traffic_bytes"] == 16384`。输出三行依次是 `16 KiB / 15 us`、`8 KiB / 10 us`、`16 KiB / 10 us`。
- **Q3**：三个 `net_logical_traffic_saved_bytes` 分别是 `8192`、`0`、`-4096`；所有候选 `saved_launches == 1`、`saved_launch_overhead_us == 5`；`more_layout["has_net_traffic_saving"]` 为假。输出第三行是 `净逻辑读写减少 -4 KiB`。
- **Q4**：`selected == "supported"`；`eligible == ["supported", "at_workspace", "slower_supported"]`；四个单原因候选的 `rejected` 分别是 `["missing_ops"]`、`["unsupported_dtype"]`、`["dynamic_shape"]`、`["workspace"]`；`multiple_misses` 是四元列表；单候选与空候选两种退化情况也各有断言。

自己改参数时只动一个量：把 `layout_bytes` 依次设为 0、8192、12288，观察净省字节变成 8192、0、−4096；或把 `workspace_limit_mib` 从 512 改成 511，看 `at_workspace` 是否掉进 `rejected` 并多出 `workspace` 原因。这两组实验分别验证"布局是独立成本"和"边界是严格大于"。
