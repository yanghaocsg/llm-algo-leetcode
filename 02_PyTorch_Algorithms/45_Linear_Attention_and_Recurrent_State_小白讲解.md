# 45. 线性注意力与递推状态 — 小白讲解

> 对应 notebook：[`45_Linear_Attention_and_Recurrent_State.ipynb`](./45_Linear_Attention_and_Recurrent_State.ipynb)

## 一句话总结
标准因果注意力每次都要拿当前 Query 与全部历史 Key 比较，显式保留随序列长度增长的 K/V；线性注意力先把 Key/Value 压成两个固定形状的累积状态 `S_t` 和 `z_t`，当前输出只需用 `φ(q_t)` 读取这两个状态。本节用 NumPy 补全 `positive_feature_map` 与 `linear_attention_scan`，验证状态形状、前缀因果性和输入契约，再用可选 GPU 实验比较“显式历史”与“固定递推状态”的状态字节、单步时间和峰值显存。

## 为什么需要它：痛点是什么
标准注意力的分数矩阵是 $S\times S$（$S$ 指序列长度），序列翻倍，计算和 KV 占用都明显增长；生成长文本时，显存里要一直保留每个历史 Token 的 K/V。线性注意力想换一种存法：不保留“每个 Token 的原始 K/V”，而是维护一个汇总账本，新 Token 来了就更新账本，输出时从账本里读。这样状态大小只由特征维和 Value 维决定，不随序列长度增长。

生活类比：你要回答“前面聊过哪些菜”。标准注意力像把聊天记录逐条翻出来，一条条和当前问题比对；线性注意力像维护一张“菜名 → 总评价”的汇总表，每来一条新消息就更新表，回答时查表。汇总表大小固定，但你也再也无法精确翻回某一条原始消息——这是它和 KV Cache 的本质差别。

## 核心问题
- 线性注意力把历史压缩成哪两个状态？它们各自的形状是什么？
- 输出公式 $\phi(q_t)^{\top}S_t/(\phi(q_t)^{\top}z_t)$ 里，分子和分母分别代表什么？
- 为什么特征映射要保证所有元素为正？`relu(x)+1` 在这里起什么作用？
- 为什么说它是“因果”的？修改未来 Token 为什么不能改变之前的输出？
- 线性注意力、KV Cache、SSM/Mamba 三者保存历史的方式有什么不同？
- 固定状态在短序列上为什么不一定省显存？GPU 实验的数字该怎么读？

## 解决思路：一步步看懂

**第 1 步：状态定义与输出公式。** 用正特征映射 $\phi$ 近似相似度后，历史被压成两个累积量：

$$
S_t=\sum_{i\le t}\phi(k_i)v_i^{\top},\qquad z_t=\sum_{i\le t}\phi(k_i),\qquad
o_t=\frac{\phi(q_t)^{\top}S_t}{\phi(q_t)^{\top}z_t}.
$$

$S_t$ 是“Key 特征与 Value 的联合状态”，形状为 `[feature_dim, value_dim]`；$z_t$ 是“归一化项”，形状为 `[feature_dim]`；$o_t$ 是当前输出，形状为 `[value_dim]`。分子 $\phi(q_t)^{\top}S_t$ 把当前查询与历史 Value 加权求和，分母 $\phi(q_t)^{\top}z_t$ 是权重之和，用来归一化。整个过程不显式构造 $S\times S$ 的分数矩阵，状态形状也与序列长度无关。

**第 2 步：TODO 1——构造正特征映射。** 参考答案只有一行：

```python
features = np.maximum(x, 0.0) + 1.0
```

先做 ReLU 再把整体加 1，保证所有特征严格大于 0（最小也是 1），从而归一化分母 $\phi(q_t)^{\top}z_t$ 不会变成 0 或负数。输入契约要求 `x` 是非空 `[seq_len, feature_dim]` 二维数组；空序列或非二维输入抛 `ValueError`。测试 `test_feature_map_contract` 用 `[[-2.0, 0.0]]` 检查输出全为正（−2 变 1，0 变 1），并确认空数组被拒绝。notebook 特意说明：`relu(x)+1` 只是教学用的正特征映射，它保证归一化项为正，但不代表某个真实线性注意力的完整核函数。

**第 3 步：TODO 2——按 Token 更新状态。** 在 `linear_attention_scan` 的循环里，每来一个 Token 就做：

```python
state = state + np.outer(key_feature, value)
normalizer = normalizer + key_feature
```

`np.outer(key_feature, value)` 是 `[feature_dim]` 与 `[value_dim]` 的外积，得到 `[feature_dim, value_dim]`，正好累加到 `S_t`；`normalizer` 则累加当前 Key 特征。两个状态都只增不减，形状固定。测试 `test_state_contract` 用 fixture 断言 `state.shape == (2, 1)`、`normalizer.shape == (2,)`、输出形状与 `values` 相同、全部有限且 `normalizer > 0`。

**第 4 步：TODO 3——读取当前输出。** 先算分母，再用同一个分母归一化分子：

```python
denominator = float(query_feature @ normalizer)
output = (query_feature @ state) / denominator
```

`query_feature @ normalizer` 是向量点积，得到标量；`query_feature @ state` 得到 `[value_dim]` 的输出。注意顺序：状态先更新当前 Token 的 K/V，再用当前 Query 去读，所以第 $t$ 步的输出能看到 $1..t$ 的历史，但看不到未来。测试 `test_prefix_causality` 把最后一个 Key 和 Value 都加 100，断言 `original[:-1]` 与 `changed[:-1]` 完全相同，这就是前缀因果性。

**第 5 步：手算三步，确认自己没有把公式写反。** fixture 是 `queries=[[1,-1],[0.5,2],[3,-0.5]]`、`keys=[[0,1],[2,-1],[1.5,1.5]]`、`values=[[1],[2],[4]]`，特征维 2、Value 维 1。
- 第 1 步：$\phi(k_1)=[1,2]$，$v_1=1$，所以 $S_1=[[1],[2]]$、$z_1=[1,2]$；$\phi(q_1)=[2,1]$，分子 $2\times1+1\times2=4$，分母 $2\times1+1\times2=4$，输出 1.0。
- 第 2 步：$\phi(k_2)=[3,1]$，$v_2=2$，$S_2=[[7],[4]]$、$z_2=[4,3]$；$\phi(q_2)=[1.5,3]$，分子 $1.5\times7+3\times4=22.5$，分母 $1.5\times4+3\times3=15$，输出 1.5。
- 第 3 步：$\phi(k_3)=[2.5,2.5]$，$v_3=4$，$S_3=[[17],[14]]$、$z_3=[6.5,5.5]$；$\phi(q_3)=[4,0.5]$，分子 $4\times17+0.5\times14=75$，分母 $4\times6.5+0.5\times5.5=28.75$，输出约 2.6087。

如果谁把 `state + outer(key, value)` 写成 `state + outer(query, value)`，或者把分母写成 `query_feature @ state`，手算结果立刻对不上；测试虽然不逐项核对这三个输出，但形状、有限性和因果性会先暴露结构错误。

**第 6 步：输入契约。** `linear_attention_scan` 要求 `queries` 与 `keys` 形状完全一致、都是二维、`values` 也是二维、三者序列长度相同且非空。`test_input_contract` 用 `Q=(2,2)`、`K=(2,3)` 检查维度不一致，用 `(0,2)/(0,2)/(0,1)` 检查空序列，二者都必须抛 `ValueError`。

**第 7 步：和 KV Cache、SSM 的关系。** notebook 用一张表把三者分开：标准注意力 + KV Cache 保存每个历史 Token 的 K/V，可以按 Token 精确读取，代价是状态和读取量随长度增长；线性注意力保存的是特征映射后的累积统计量，不能直接读回单个历史 Token，近似形式和数值稳定性需要验证；SSM / Mamba 也使用递推状态，但选择性更新规则和参数化方式不必经过 Q/K 相似度，不是标准注意力的简单等价替换。换句话说，线性注意力改变的是“历史以什么形式保存”，不是“模型是否依赖历史”。

**第 8 步：可选 GPU 实验怎么读。** 实验固定 `NUM_HEADS=8`、`HEAD_DIM=64`、`DTYPE='bfloat16'`、`WARMUP=10`、`REPEATS=100`，长度阶梯是 `[512, 2048, 8192]`，只改变“读取完整 K/V 历史”还是“更新固定递推状态”。保存的输出（cell 18 的 identity 是 `arch-linear-recurrence-v2` / `rtx5070ti_recurrent_v2`）显示：显式历史的 `state_bytes` 随长度从 1,048,576 → 4,194,304 → 16,777,216 字节增长（1 MiB → 4 MiB → 16 MiB），固定递推状态始终保持 65,536 字节（0.0625 MiB）；512 长度时显式历史单步 0.01117 ms、峰值 1.009 MB，递推路径单步 0.01759 ms、峰值 8.379 MB；2048 长度时两者单步约 0.01697 / 0.01703 ms，峰值 12.146 / 8.379 MB。决策是 `accept_fixed_state_growth_mechanism`。5.4 的文字还指出，8192 长度时递推路径耗时约为显式历史的 53.6%。读法：短序列上递推路径有固定的工作区开销，峰值反而更高；只有序列足够长，显式历史的内存增长才成为主导。这个实验只验证单层状态接口，不代表真实 Mamba、线性注意力 Kernel 或任务质量。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 特征映射 $\phi$ | 把 Q/K 变成非负特征的函数，本节用 `relu(x)+1` |
| 状态 $S_t$ | 历史 Key 特征与 Value 的联合累积，形状固定 |
| 归一化项 $z_t$ | 历史 Key 特征的累积，用来做分母 |
| 递推（scan） | 每来一个 Token 就更新状态，而不是重算全部历史 |
| 前缀因果性 | 第 $t$ 步输出只依赖 $1..t$，未来 Token 改不动它 |
| KV Cache | 按 Token 保存历史 K/V，可精确回读，但随长度增长 |
| SSM / Mamba | 另一类状态更新架构，不必经过 Q/K 相似度 |
| `state_bytes` | 保存历史所需的状态字节数 |
| `mean_step_ms` | 单个生成步的平均耗时 |
| `peak_memory_mb` | 实验过程中的峰值显存 |
| 固定状态 | 形状只由特征维和 Value 维决定，不随序列长度变 |

## 小白常见疑问
**Q1：为什么特征映射要“所有元素为正”，用 `relu(x)` 不行吗？**
`relu` 会把负数变成 0，如果某个 Key 的特征全为 0，归一化项可能不增长甚至接近 0，分母就不稳定。`relu(x)+1` 保证每个特征至少是 1，分母 $\phi(q_t)^{\top}z_t$ 始终为正。notebook 的解析也强调，这个映射只是教学简化，不代表真实核函数。

**Q2：分母一定要有吗？不归一化会怎样？**
没有分母时，输出是 $\phi(q_t)^{\top}S_t$，数值会随历史 Token 数量累加而变大，序列越长输出尺度越大；除以 $\phi(q_t)^{\top}z_t$ 相当于做加权平均，让输出尺度更稳定。代码里先用同一个 `denominator` 算分子，再整体相除，顺序不能反。

**Q3：为什么修改最后一个 Token 不会改变前两个输出？**
因为第 $t$ 步只读取“当前 Query + 截至当前的状态”，而状态是按顺序累积的；未来的 Key/Value 还没有进入状态。`test_prefix_causality` 把 `keys[-1]` 和 `values[-1]` 各加 100，断言 `original[:-1]` 与 `changed[:-1]` 完全一致，这就是递推因果性的直接证据。

**Q4：固定状态不随长度增长，为什么短序列反而更费显存？**
因为递推路径需要固定的工作区/临时缓冲，短序列时这部分开销占比高。保存的输出里，长度 512 时显式历史峰值 1.009 MB，而递推路径峰值 8.379 MB；到长度 2048 时显式历史涨到 12.146 MB，递推仍是 8.379 MB。notebook 的原话是“短序列仍受递推工作区影响，长序列才显出固定状态优势”。

**Q5：线性注意力能完全替代 Softmax 注意力吗？**
不能这样下结论。特征映射改变了相似度形式，固定大小状态也不天然保留所有精确的 Token 对关系；notebook 明确说它“不等于免费替代标准注意力”，近似误差和数值稳定性都需要验证。它和 KV Cache 的差别是“存聚合统计量”还是“存每个 Token 的 K/V”。

**Q6：这个实现能代表真实 Mamba 或线性注意力 Kernel 的性能吗？**
不能。GPU 实验只测单层状态接口的 `state_bytes`、`mean_step_ms`、`peak_memory_mb` 和输出有限性，notebook 明确说它不代表真实 Mamba、线性注意力 Kernel 或任务质量；真实 checkpoint 的长序列质量要由 77 节继续验证。

## 动手验证：怎么知道自己做对了
四个测试按顺序直接调用，全部通过后打印 `线性 Attention 递推状态测试通过。`。
- `test_state_contract`：`outputs.shape == values.shape`、`state.shape == (2, 1)`、`normalizer.shape == (2,)`，输出全部有限且 `normalizer > 0`。
- `test_prefix_causality`：把最后一个 Key/Value 各加 100 后，前两个输出必须 `np.allclose` 保持不变。
- `test_input_contract`：Q/K 形状不一致 `(2,2)` vs `(2,3)`、空序列 `(0,2)` 都必须抛 `ValueError`。
- `test_feature_map_contract`：`positive_feature_map([[-2.0, 0.0]])` 的结果全部大于 0；空序列必须抛 `ValueError`。

notebook 的 GPU 单元默认 `RUN_MODE='dry_run'`，只打印配置，不会真正执行；改 `real_gpu` 前要先完成 `ARCH-CUDA-MECHANISM` 预检，结果写入 `benchmarks/results/45_linear_recurrent_state_<RUN_LABEL>.json`。读取单元会检查 JSON 是否包含 `schema_version / identity / workload / hardware / baseline / candidates / metrics / evidence_level / failure / decision`，缺字段直接抛 `ValueError`；读数时先看 `failure` 和 `output_finite`，再比较不同长度下的 `state_bytes` 与 `peak_memory_mb`。
