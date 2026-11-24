# PERF-CANDIDATE. Constraint to Candidate and Cost Transfer | 从资源约束到候选改动与新增代价 · 小白入门

> 配套 notebook：[PERF-CANDIDATE. 从资源约束到候选改动与新增代价](../PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

给模型提速的一个改动，常常在省下一种资源的同时，在别处增加开销、拉低质量。这本 notebook 教你在实测之前把这几笔账分开记、分开比，再决定丢弃、调整还是去测。

## 为什么需要它

你想让模型跑得更快，想到一个改动，比如把权重换成低比特。效果往往不是单纯的“变好”：权重占的字节少了，但运行时可能要先反量化（dequantization，把低比特权重还原成高精度再计算）而多花时间，结果里还会多出误差。只看一面就去实测，容易白忙一场：

- 看名字下结论：候选叫“低比特权重”，就认定它一定减少搬运量，不去核对它到底改了哪个量。
- 把好处和坏处加成一个总分：搬运量少了 30、运算量多了 5，单位不同，加在一起说明不了问题，一项变好还会盖住另一项超限。
- 不知道的就不填：某个量没测过就不写，让它默认等于 0。

这本 notebook 把候选相对基线的变化拆到五个量上，各算各的，再逐项和门槛比较，得出 `reject`、`tune` 或 `measure`。它只负责实测之前的筛选，不决定最终是否采用。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 基线 / 候选（baseline / candidate） | 基线是现在的做法，候选是你想试的一个改动；所有比较都是候选相对基线 |
| 主约束（primary constraint） | 你认定当前最可能限制完成时间的那项资源，代码里叫 `constraint`，四类取值 `compute`、`bandwidth`、`capacity`、`critical_path` 分别对应计算、搬运、峰值容量、关键路径 |
| 资源轴 | 可以和基线比较的一个量，这里固定五个：运算量、搬运量、峰值容量、暴露时间、质量损失 |
| 变化量（Δx） | 候选值减基线值，负数表示减少（比如搬运量从 200 变成 180，变化量是 `-20`） |
| 目标减少量（R） | 目标轴上变化量的相反数，正数才表示真的减少了 |
| 新增开销（added cost） | 某个资源轴上增加的部分，减少的轴记 0 |
| 暴露时间（exposed time） | 没有被其他计算盖住、只能干等的等待时间，要靠依赖分析得到 |

## 核心思路

输入是三样：主约束是哪一类；候选在五个资源轴上的变化量；一组门槛，即允许的质量损失、目标最少减少量和每个轴的新增预算（budget）。程序先检查五个变化量齐全且都是有限数（不是 NaN 或无穷大），再取目标轴变化量的相反数作为目标减少量，最后逐项和门槛比较。前三行是前两步的式子，后三行是三个门槛：

```text
Δx = x_c - x_b        变化量：候选值减基线值
R  = -Δx_target       目标减少量：目标轴上变化量的相反数
A_x = max(0, Δx)      新增开销：只保留增加的部分

Δq <= q_max           质量变化不超过允许值
R  >= R_min           目标减少量不少于最低要求
A_x <= a_x            每个资源轴的新增开销不超过自己的预算，等号也算通过
```

`x_b` 是基线值，`x_c` 是候选值，`target` 是主约束对应的资源轴；`q_max`、`R_min`、`a_x` 就是上面三个门槛。质量轴也做减法，但约定这个量是“越大越差”的损失（比如误差），所以正数表示质量退化；越大越好的准确率不能直接放进来。

用小数字走一遍：主约束是搬运量，误差最多增加 1.0，搬运量至少减少 20，四个资源轴的新增预算分别是 20、5、5、1.0。

```python
axes = ("compute_flops", "memory_bytes", "peak_bytes", "exposed_ms", "quality_loss")
base = dict(zip(axes, (100, 200, 50, 4.0, 1.0)))   # 基线：运算量、搬运量、峰值、暴露时间、误差
cand = dict(zip(axes, (110, 180, 30, 4.5, 1.5)))   # 候选：同样五个量

delta = {k: cand[k] - base[k] for k in axes}       # 变化量 = 候选 - 基线
R = -delta["memory_bytes"]                         # 这次的目标轴是搬运量
A = {k: max(0.0, delta[k]) for k in axes[:4]}      # 四个资源轴的新增开销
print(delta)
print(R, A)


def decide(d, q_max, R_min, a):
    R = -d["memory_bytes"]
    if R <= 0 or d["quality_loss"] > q_max:
        return "reject"
    if R < R_min or any(max(0.0, d[k]) > a[k] for k in a):
        return "tune"
    return "measure"


a = {"compute_flops": 20.0, "memory_bytes": 5.0, "peak_bytes": 5.0, "exposed_ms": 1.0}
print(decide(delta, 1.0, 20, a))
print(decide({**delta, "memory_bytes": -10}, 1.0, 20, a))
print(decide({**delta, "memory_bytes": 30}, 1.0, 20, a))
print(decide({**delta, "quality_loss": 2.0}, 1.0, 20, a))
```

```text
{'compute_flops': 10, 'memory_bytes': -20, 'peak_bytes': -20, 'exposed_ms': 0.5, 'quality_loss': 0.5}
20 {'compute_flops': 10, 'memory_bytes': 0.0, 'peak_bytes': 0.0, 'exposed_ms': 0.5}
measure
tune
reject
reject
```

前两行是五个变化量（运算量多了 10，搬运量和峰值各少了 20，暴露时间多了 0.5 ms，误差多了 0.5）、目标减少量，以及各轴的新增开销（减少的轴记 0）。后四行是四种情况：

- `measure`（去做对照实测）：三项都满足，目标减少量正好等于最低要求 20，等号算通过。
- `tune`（调整后重新评估）：搬运量只少了 10，方向对但幅度不够。
- `reject`（拒绝）：搬运量反而多了 30，目标没有减少。
- `reject`：搬运量达标，但误差多了 2.0，超过 1.0。质量超限直接拒绝，目标上的收益不能抵消。

新增开销超预算同样是 `tune`。三个门槛各比各的，不合成加权总分；`measure` 只说明值得实测，不是采用结论。新增峰值预算也只是允许增加的字节数，没有基线峰值和设备容量时，不能据此判断会不会 OOM。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 候选改动怎样对应到受限资源：计算、搬运、峰值容量、关键路径四类约束各能改什么、要核对什么证据 | 读 |
| Step 2 | 怎样分别记录减少量和新增开销：变化量、目标减少量、新增开销的定义，以及五个变化轴的单位与方向 | 读，对照上面的式子 |
| Step 3 | 怎样按目标、质量和预算筛选候选：三个比较条件和 `reject`、`tune`、`measure` 三种状态 | 读，对照后面的四种情况 |
| Step 4 · TODO 1 | 检查变化轴 | 在 `normalize_effects` 里写 2 行：算出未知轴和缺失轴两个集合，后面的报错骨架已经写好 |
| Step 4 · TODO 2 | 读取目标减少量 | 在 `primary_reduction` 里写 2 行：按 `constraint` 查出目标轴，取变化量的相反数 |
| Step 4 · TODO 3 | 逐项判断预算 | 在 `decide_candidate` 里写 3 个判断：`quality_passed`、`added_costs_passed`、`primary_missed` |
| 测试 | 检查五轴是否完整、数值是否合法、四类约束下的符号、三种状态、恰好达标、各轴超限和非法预算 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上比较 FP32 串行、BF16 串行和 FP32 双缓冲三条路径的时间、对象字节和数值相似度 | 有 GPU 时再做，默认是 `RUN_MODE = "dry_run"` |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `PERF-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，这时 5.2 不导入 GPU 工作负载；改成 `real_gpu` 才执行真实候选矩阵。

## 容易踩的坑

- **少一个轴、多一个轴，或者填了非法的数**：五个变化量必须全给。多出的字段（比如综合分数）和缺失的字段都要报错；`True`、字符串、`None`、NaN、无穷值都要在比较前拦住，不能让未知开销靠省略字段变成 0。
- **按候选名称推断目标轴**：目标轴只能由 `constraint` 查 `CONSTRAINT_AXIS` 得到。减少量是变化量的相反数，符号写反会把“减少 30”算成“减少 -30”。
- **让一项收益抵消另一项超限**：质量超限或目标没有减少时直接 `reject`，其他资源改善再多也不抵消，也不要合成加权总分。
- **新增开销比错了对象**：只拿增加的部分和预算比，减少的轴记 0；等于预算算通过。预算必须是有限非负数，六个字段都要给齐。
- **把 `measure` 当成“已采用”**：它只说明通过了筛选、值得去做对照实测。

## 读完后自查

- [ ] 能说出五个资源轴的名字和各自的单位。
- [ ] 搬运量从 200 变成 180，能写出它的变化量、目标减少量和这个轴的新增开销。
- [ ] 能说出 `reject`、`tune`、`measure` 各自的触发条件。
- [ ] 能解释为什么 `measure` 不等于“已采用”。

## 下一步

- 打开 notebook：[PERF-CANDIDATE. 从资源约束到候选改动与新增代价](../PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md)
- 建议先读：[PERF-ROOFLINE. 算术强度与资源上界 · 小白入门](./PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md)
- 需要补基础：[Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md)、[Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- 做完之后：[PERF-COMPOSITION. 策略相互作用与瓶颈转移 · 小白入门](./PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md)
