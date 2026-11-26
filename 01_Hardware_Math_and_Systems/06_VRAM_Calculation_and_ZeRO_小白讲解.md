# 06. 显存计算与 ZeRO 优化 — 小白讲解

> 对应 notebook：[`06_VRAM_Calculation_and_ZeRO.ipynb`](./06_VRAM_Calculation_and_ZeRO.ipynb)

## 一句话总结
本节用一套纯 CPU 的账本函数，逐项算出训练时每张卡到底存了什么：`training_state_breakdown` 把参数、梯度、Adam 两个动量和 FP32 主权重拆开，7B、BF16 参数、Adam 的静态状态是 112 GB；`calculate_zero_memory` 按 ZeRO-1/2/3 分别分片"更新状态、梯度、参数"，8 卡上得到 38.5、26.25、14 GB；`state_capacity_limit_params` 再从 80 GB 卡、20% 预留反推能装多少参数，DDP 只有 4B，ZeRO-3 能到 32B。

## 为什么需要它：痛点是什么
"7B 模型 FP16 只要 14 GB，为什么我 80 GB 的卡训练还 OOM？"这是最典型的显存困惑。答案藏在"训练状态"里：推理只需要权重，训练还要梯度、优化器动量和一份高精度主权重。14 GB 只是**参数这一项**，7B 用 Adam 训练时静态状态是 112 GB，是推理的 8 倍。第二层困惑是 DDP：数据并行并不是"每张卡存一部分"，而是**每张卡都存完整副本**，所以 8 卡 DDP 每张卡仍然是 112 GB，一点没省。ZeRO 才是真正把这些状态切开。

生活类比：8 个人合作写一本 112 页的手册。DDP 是每人桌上都放一整套 112 页（改哪页都方便，但桌子占满）；ZeRO-1 是每人只负责 update 其中一部分页，但正文和草稿仍各有一份；ZeRO-3 是连正文都按人分，谁要用某页再临时借过来——省地方，但借来借去要花通信时间。

## 核心问题
- 7B 模型训练的 112 GB 是怎么一项一项加出来的？16 字节/参数从哪来？
- FP32 主权重什么时候保留、什么时候不保留？它和 Adam 的两个动量是什么关系？
- ZeRO-1、ZeRO-2、ZeRO-3 各切走哪一组状态？为什么 8 卡上分别是 38.5、26.25、14 GB？
- 80 GB 卡、预留 20%，各阶段最多能装多少参数？这个数字为什么不是"可训练上限"？
- 这些账本漏掉了哪些真实显存占用？

## 解决思路：一步步看懂

### 第 1 步：Q1——DDP 下每张卡的 112 GB

`training_state_breakdown(num_params_b, model_dtype='fp16', optimizer='adam', keep_master_weights=None)` 返回单份静态状态的十进制 GB。它先定每个参数的字节数：

```python
model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
moment_gb = num_params_b * 4 if optimizer == 'adam' else 0.0
components = {
    'parameters_gb': num_params_b * model_bytes,
    'gradients_gb': num_params_b * model_bytes,
    'adam_first_moment_gb': moment_gb,
    'adam_second_moment_gb': moment_gb,
    'master_weights_gb': num_params_b * 4 if keep_master_weights else 0.0,
}
```

五个组件：

1. **参数**：BF16 每参数 2 字节，7B 就是 $7 \times 2 = 14$ GB。
2. **梯度**：与参数同宽，也是 14 GB。
3. **Adam 一阶矩**：梯度的一阶移动平均，FP32 每参数 4 字节，28 GB。
4. **Adam 二阶矩**：梯度平方的移动平均，也是 28 GB。
5. **FP32 主权重**：4 字节，28 GB。

四项固定项相加 $14+14+28+28=84$ GB，加主权重 28 GB 得 112 GB。平均到每个参数是 $2+2+4+4+4=16$ 字节，这就是"7B 训练要 112 GB"的来历。

为什么需要主权重？参数更新是 $W_{new} = W_{old} - \mathrm{LR} \times G$。若 $W$ 只用 BF16 存，它的有效位数少，学习率乘以一个很小的梯度后，更新量可能小于 BF16 在该量级的最小间隔，直接被舍入吃掉。保留一份 FP32 副本，先在副本上做高精度减法，再转回 BF16 供前向使用，小更新才不会消失。notebook 的默认规则是：**FP16/BF16 参数默认保留主权重，FP32 参数默认不保留**，也可以显式传入布尔值覆盖：

```python
if keep_master_weights is None:
    keep_master_weights = model_dtype != 'fp32'
if type(keep_master_weights) is not bool:
    raise ValueError('keep_master_weights须为布尔值或None')
```

`optimizer='sgd'` 时两个动量都是 0，`moment_gb = 0.0`，`optimizer_state_gb` 也就是 0。函数还校验参数量非负、dtype 在 `{'fp32','fp16','bf16'}`、optimizer 在 `{'adam','sgd'}`。

`calculate_ddp_memory` 只是转调 `training_state_breakdown(...)['training_state_gb']`，保证两处用同一套口径。测试里的四个对照很能说明问题：

- `calculate_ddp_memory(7, keep_master_weights=False) == 84`：关掉主权重，正好少 28 GB。
- `calculate_ddp_memory(7, model_dtype='fp32') == 112`：FP32 参数每个 4 字节，参数和梯度各 28 GB，动量 28+28，默认不保留主权重，合计仍是 112。
- `calculate_ddp_memory(7, model_dtype='fp32', keep_master_weights=True) == 140`：再加一份 28 GB 的主权重。
- `calculate_ddp_memory(7, optimizer='sgd') == 56`：14+14，没有动量、默认 bf16 保留主权重 28，共 56。

注意第二个对照的巧合：BF16+Adam+主权重和 FP32+Adam 都是 112 GB，但组成不同（前者 2+2+4+4+4，后者 4+4+4+4）。只看总数会误判，所以函数把 `parameters_gb`、`gradients_gb`、两个动量、`master_weights_gb` 分开返回，`optimizer_state_gb` 只汇总两个动量（`2 * moment_gb`），主权重单列，避免重复相加。

### 第 2 步：Q2——ZeRO 按阶段切状态

ZeRO 把训练状态分成三组：**更新状态**（两个 Adam 动量 + 可选主权重）、**梯度**、**参数**。`calculate_zero_memory` 先取 Q1 的账本，再按阶段决定哪几组除以设备数：

```python
parameters = ledger['parameters_gb']
gradients = ledger['gradients_gb']
updates = ledger['optimizer_state_gb'] + ledger['master_weights_gb']
if zero_stage in (0, 'ddp'):
    return parameters + gradients + updates
if zero_stage == 1:
    return parameters + gradients + updates / num_gpus
if zero_stage == 2:
    return parameters + (gradients + updates) / num_gpus
if zero_stage == 3:
    return (parameters + gradients + updates) / num_gpus
raise ValueError('阶段使用0/ddp、1、2或3')
```

7B、BF16、Adam、保留主权重时，三组分别是 14、14、84。8 卡：

| 阶段 | 参数 | 梯度 | 更新状态 | 单卡合计 | 运行时新增的通信 |
|---|---:|---:|---:|---:|---|
| DDP / ZeRO-0 | 完整 14 | 完整 14 | 完整 84 | 112 | 梯度归约 |
| ZeRO-1 | 完整 14 | 完整 14 | 84/8 = 10.5 | 38.5 | 分片更新、参数同步 |
| ZeRO-2 | 完整 14 | 14/8 = 1.75 | 84/8 = 10.5 | 26.25 | 梯度分片、参数同步 |
| ZeRO-3 | 14/8 = 1.75 | 14/8 = 1.75 | 84/8 = 10.5 | 14 | 梯度分片、按需收集参数 |

ZeRO-1 只切更新状态：$14+14+10.5=38.5$。ZeRO-2 把梯度和更新状态一起切：$(14+84)/8 = 12.25$，加参数 14 得 26.25。ZeRO-3 连参数也切：$112/8=14$。测试逐阶段断言这四个数，并额外断言 `calculate_zero_memory(7, stage, 1) == 112`——单卡没有分片，四个阶段必须相同，这条用来防止公式把除以 1 写错。

还有一个容易忽略的对照：`calculate_zero_memory(7, 3, 8, keep_master_weights=False) == 10.5`。关掉主权重后更新状态从 84 降到 56，总量 $14+14+56=84$，除以 8 得 10.5。这说明主权重在 ZeRO-3 里也会被切，不是额外常驻的 28 GB。

notebook 强调这张表只描述**持续保存的状态**。ZeRO-3 在执行某一层时还要临时把该层参数收集到当前设备，这部分峰值不在 14 GB 里；实际还要加上激活、通信缓冲和分配器碎片。

### 第 3 步：容量反推——64 GB 预算能装多少参数

`state_capacity_limit_params(gpu_memory_gb, num_gpus, zero_stage, overhead_ratio=0.2, ...)` 做的是除法：

```python
state_budget_gb = gpu_memory_gb * (1 - overhead_ratio)
state_gb_per_1b = calculate_zero_memory(1, zero_stage, num_gpus, ...)
return state_budget_gb / state_gb_per_1b
```

思路是"先算每 1B 参数在当前阶段占多少 GB，再用预算去除"。80 GB 卡、预留 20%，预算是 $80 \times 0.8 = 64$ GB。

先把 7B 账本按比例缩到 1B：参数 2 GB、梯度 2 GB、更新状态 $84/7 = 12$ GB（两个动量各 4 + 主权重 4），合计 16 GB。再套四个阶段的公式：

- DDP：$2+2+12 = 16$ GB，$64/16 = 4$，只能装 4B。
- ZeRO-1：$2+2+12/8 = 5.5$ GB，$64/5.5 \approx 11.64$B。
- ZeRO-2：$2+(2+12)/8 = 3.75$ GB，$64/3.75 \approx 17.07$B。
- ZeRO-3：$16/8 = 2$ GB，$64/2 = 32$B。

这里 $12/8 = 1.5$ 而不是 1，是因为更新状态是每参数 12 字节而不是 8 字节；漏掉主权重的 4 字节会把 ZeRO-2 错算成 3.25、容量错算成约 19.7B，和测试的 26.25 GB / 3.75 对不上。

断言 `state_capacity_limit_params(80, 8, 'ddp') == 4`、`(80, 8, 3) == 32`、`(80, 8, 3, overhead_ratio=0) == 40`。最后一条：不预留时预算是 80 GB，$80/2=40$。还有一条回代校验 `calculate_zero_memory(state_capacity_limit_params(80, 8, 3), 3, 8) == 64`，即"32B 在 ZeRO-3 下的静态状态正好填满 64 GB 预算"。

这个容量数字**不是可训练上限**。它只除静态状态，没算激活值（随 batch、序列长度增长，常常是大头）、临时工作区、通信缓冲和显存碎片。notebook 的措辞是"当前假设下的可容纳规模"，真实能训多大要跑起来看。

### 第 4 步：Q3——把四个阶段放进同一张对照表

`compare_zero_strategies` 固定"每卡 80 GB、8 卡、预留 20%"的配置，对 1B 参数分别算三列：`per_gpu_state_gb_per_1b`（每 1B 的单卡静态状态）、`memory_saving_ratio`（相对 DDP 省的比例）、`static_capacity_limit_params_b`（能装多少 B），再附一句 `communication_work`。

| 策略 | 每 1B 单卡状态 | 相对 DDP 节省 | 静态容量 | 新增通信工作 |
|---|---:|---:|---:|---|
| DDP | 16 GB | 0% | 4B | 梯度归约 |
| ZeRO-1 | 5.5 GB | 65.6% | 11.64B | 分片更新、参数同步 |
| ZeRO-2 | 3.75 GB | 76.6% | 17.07B | 梯度分片、参数同步 |
| ZeRO-3 | 2 GB | 87.5% | 32B | 梯度分片、按需收集参数 |

节省比例是 $1 - \text{state}/\text{baseline}$：ZeRO-3 是 $1-2/16=0.875$，断言精确等于 `0.875`；ZeRO-1 是 $1-5.5/16=0.65625$，`round(...,3)` 后 0.656；ZeRO-2 是 $1-3.75/16=0.765625$，得 0.766。容量列是 `round(...,3)`，所以打印出来是 11.636 和 17.067 这样的数。

表里最后一列是提醒：省显存不是免费的。ZeRO-1 要在更新后同步参数，ZeRO-2 要归约并分发梯度，ZeRO-3 每次前向/反向用到某层参数时都要临时收集、用完释放，通信次数和依赖更复杂。容量筛选只是第一关，真实选择要固定模型、批量和序列长度，在多卡实验里量训练步耗时。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 训练状态 | 训练时长期驻留的参数、梯度、优化器状态等 |
| 静态状态 | 不算激活和临时缓冲的那部分常驻显存 |
| DDP | 数据并行，每个进程保存一份完整训练状态 |
| ZeRO | 按阶段把训练状态切片分给各卡，用时再通信 |
| 主权重 | 额外保留的 FP32 权重副本，专供高精度更新 |
| 一阶矩 / 二阶矩 | Adam 为每个参数记的梯度均值 / 平方均值 |
| 分片（shard） | 把一份完整状态按卡数切开，每卡只存一片 |
| 容量反推 | 用显存预算除以每参数状态量，估算能装多大 |
| 预留比例 | 给激活、工作区、碎片留出的显存百分比 |
| 参数同步 | 分片更新后把各卡参数重新对齐的通信 |

## 小白常见疑问

**Q1：为什么 FP32 参数默认不保留主权重，BF16 却要？**
因为 FP32 本身已经是高精度，更新量不会像 BF16 那样被轻易舍入掉，再存一份 FP32 副本纯属重复。BF16 只有 7 位尾数，小更新容易消失，才需要一份 FP32 主权重兜底。代码里 `keep_master_weights = model_dtype != 'fp32'` 就是这条规则。

**Q2：DDP 8 卡为什么每张卡还是 112 GB，数据并行不是分摊了吗？**
DDP 分摊的是**数据**，不是**状态**。每张卡拿不同的 batch 算梯度，但参数、梯度、优化器状态都各存一份完整副本；卡与卡之间只做梯度归约。所以单卡显存不随卡数下降，ZeRO 才把状态本身切开。

**Q3：ZeRO-2 的 26.25 GB 里，梯度到底占多少？**
梯度分片是 $14/8 = 1.75$ GB，更新状态分片是 $84/8 = 10.5$ GB，两者相加 12.25 GB，再加完整的参数 14 GB，得 26.25。公式写成 `parameters + (gradients + updates) / num_gpus`，是因为这两组在同一个阶段被分片，可以合并成一次除法，但拆开看仍然是 1.75 和 10.5。

**Q4：80 GB 卡反推出 32B，是不是说明我能训 32B 模型？**
不是。32B 只填满了 64 GB 的**静态状态**预算，激活值、临时工作区、通信缓冲和碎片都没算。序列长度一长，激活很容易超过静态状态。这个数字的用途是比较不同 ZeRO 阶段的相对容量，不是承诺可训练规模。

**Q5：主权重在 ZeRO-3 里会不会额外多占 28 GB？**
不会。`updates` 已经把 `optimizer_state_gb` 和 `master_weights_gb` 加在一起，ZeRO-3 再整体除以 8。测试里 `keep_master_weights=False` 时 7B 从 14 GB 降到 10.5 GB，差额 3.5 GB 正是 $28/8$，说明它是被切分的，不是额外常驻。

## 动手验证：怎么知道自己做对了

这个 notebook 没有保存执行输出，也没有 TODO 空位，以下数字以断言为准。

- `test_calculate_ddp_memory`：7B、BF16、Adam 时 `parameters_gb == gradients_gb == 14`、两个动量各 `28`、`optimizer_state_gb == 56`、`master_weights_gb == 28`、`training_state_gb == 112`；`keep_master_weights=False` 得 84；FP32 得 112；FP32+主权重得 140；SGD 得 56。全过打印 `静态状态组件和主权重增量核对通过`。
- `test_calculate_zero_memory`：8 卡时 `('ddp', 112)`、`(1, 38.5)`、`(2, 26.25)`、`(3, 14)`；同一组阶段在 1 卡时都等于 112；ZeRO-3 关主权重得 10.5。打印 `DDP与ZeRO各阶段的常驻状态核对通过`。
- `test_state_capacity_limit_params`：`(80, 8, 'ddp') == 4`、`(80, 8, 3) == 32`、`overhead_ratio=0` 时 `== 40`，并用 `calculate_zero_memory(32, 3, 8) == 64` 回代。打印 `静态状态容量反推核对通过`。
- `compare_zero_strategies`：`strategies[0]['static_capacity_limit_params_b'] == 4`、`strategies[-1] == 32`、`strategies[-1]['memory_saving_ratio'] == 0.875`，随后逐行打印四个字典。

改参数时建议只动一个：把 `keep_master_weights` 从 `None` 改成 `False`，看 112 是否变 84、ZeRO-3 是否从 14 变 10.5；把 `overhead_ratio` 从 0.2 改成 0，看容量是否从 32B 涨到 40B；把 `num_gpus` 从 8 改成 4，看 ZeRO-3 是否变成 28 GB、容量是否减半。这三组分别验证"主权重是独立项""预算是乘出来的""分片是按卡数均分"。
