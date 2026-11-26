# ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构 — 小白讲解

> 对应 notebook：[`ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.ipynb`](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.ipynb)

## 一句话总结
本节先用一张表区分全注意力、线性注意力和 SSM/Mamba 三种"历史保存方式"，再让读者补全三个函数：`build_hybrid_schedule` 生成注意力层与递推层的排列，`memory_state_ledger` 分别计算随长度线性增长的 KV 字节数和固定的递推状态字节数，`exact_history_gate` 检查这个排列是否保留了本题要求的逐词 KV 读取路径。最后用一个可选 GPU 实验观察三种排列在序列变长时的状态占用和单步耗时。

## 为什么需要它：痛点是什么
初学者常把"Attention 的显存随上下文线性涨"当成所有长上下文模型的通病，于是看到 Mamba 或混合模型宣称"恒定状态"就以为它天然能记住一切。实际是两种不同的取舍：注意力保存每个历史词元的 K/V，当前查询可以精确回看；递推层只维护一个固定大小的状态，读过的东西被压缩进状态里，能不能取回取决于更新方式和训练。

生活类比：注意力像把每次会议都录音存档，想问哪段就翻哪段，但档案室越堆越大；递推层像会议纪要，本子厚度固定，但纪要写得好不好、有没有漏掉关键决定，要另做任务测试。混合模型就是一部分楼层存档、一部分楼层只写纪要。

## 核心问题
- 全注意力、线性注意力和 SSM 各保存什么？随历史长度的增长方式有何不同？
- "层排列"为什么重要？同样数量的注意力层，放在前面和分散放置有区别吗？
- 显式 KV 和固定递推状态的字节数怎么算？为什么一个乘序列长度、一个不乘？
- "本题指定保留逐词 KV 读取路径"这个条件，怎样变成一个布尔检查？
- 状态字节数少，是否等于模型能正确找回远距离信息？

## 解决思路：一步步看懂

**第 1 步：三种历史保存方式（Step 1）。** notebook 的表格把差异压缩成三行：

| 保存方式 | 保存什么 | 随历史长度怎样变化 | 当前输出怎样取得 |
|---|---|---|---|
| 全注意力 | 各历史词元的 K/V | 线性增长 | 根据查询读取保存的位置 |
| 线性注意力 | 累计的特征统计 | 状态形状固定 | 从累计统计计算当前输出 |
| SSM／Mamba | 递推状态 | 状态形状固定 | 先更新状态，再计算输出 |

关键不是"谁更先进"，而是"保存量"和"读取能力"是两件事。notebook 反复强调，保存量要用字节账本核对，读取能力要用远距离检索或目标任务验证，不能互相替代。

**第 2 步：层排列不只是数个数（Step 2）。** 混合模型让部分层读历史 KV、其余层更新递推状态。hidden state 逐层传递，所以层的**顺序**决定了信息经过怎样的计算序列：同样两层注意力，放在最前面和分散在中间，后续递推层看到的状态不同。Step 2 还要求检查相邻层的隐藏维度和数据类型是否相容，因为残差相加和层间传递都要求形状、dtype 匹配。

**第 3 步：TODO 1——生成层排列。** 参考答案：

```python
schedule = ["ssm"] * num_layers if attention_every is None else [
    "attention" if index % attention_every == 0 else "ssm"
    for index in range(num_layers)
]
```

规则是：`attention_every=None` 表示全递推；否则**第 0 层以及所有索引能被 `attention_every` 整除的层**用注意力，其余用 `ssm`。注意 `%` 从 0 开始，所以 `attention_every=4` 时是第 0、4、8… 层，而不是第 4、8… 层。测试 `test_hybrid_schedule_contract` 给出三个基准：

- `build_hybrid_schedule(6, 2)` → `['attention','ssm','attention','ssm','attention','ssm']`；
- `build_hybrid_schedule(3, 1)` → 三层全是 `attention`；
- `build_hybrid_schedule(3, None)` → 三层全是 `ssm`；
- `num_layers=0` 必须抛 `ValueError`；骨架还会对非整数 `num_layers` 抛 `TypeError`。

**第 4 步：TODO 2——两类保存量。** 参考实现：

```python
kv_bytes = attention_layers * seq_len * num_kv_heads * head_dim * 2 * dtype_bytes
recurrent_bytes = ssm_layers * recurrent_state_dim * dtype_bytes
```

逐项解释：`attention_layers` 是排列中 `"attention"` 的个数；`seq_len` 是历史长度；`num_kv_heads × head_dim` 是每个词元每个注意力层的 K（或 V）宽度；乘 `2` 表示 K 和 V 各一份；`dtype_bytes` 默认 2，对应 BF16/FP16。所以 KV 部分与 `seq_len` 成正比。递推部分只有 `ssm_layers × recurrent_state_dim × dtype_bytes`，**没有 `seq_len`**，这就是"状态形状固定"的体现。

用 `test_memory_state_ledger` 的配置手算：`schedule = build_hybrid_schedule(4, 2)` 得到 `[attention, ssm, attention, ssm]`，即 2 层注意力、2 层递推；`seq_len=8, num_kv_heads=2, head_dim=4, recurrent_state_dim=16, dtype_bytes=2`：

- KV：2 × 8 × 2 × 4 × 2 × 2 = 512 字节；
- 递推：2 × 16 × 2 = 64 字节；
- 总计：512 + 64 = 576 字节。

测试断言的正是 `kv_bytes == 512`、`recurrent_bytes == 64`、`total_bytes == 576`。骨架的输入校验也值得看：`schedule` 非空且元素只能是 `attention`/`ssm`，五个维度参数都必须是正整数，否则 `ValueError`。这里算的是**单样本、理论下界**，不含激活、临时缓冲、分页对齐和框架开销。

**第 5 步：TODO 3——检查指定读取路径。** 参考实现一行：

```python
eligible = has_attention or not requires_exact_history
```

语义是：如果本题配置明确要求保留逐词 KV 读取路径（`requires_exact_history=True`），那么排列里至少得有一层 `attention`；如果没有这个要求，纯递推排列也可以继续。`reason` 字段给出可解释的原因：有注意力时是 `attention_path_available`，没有且要求精确历史时是 `exact_history_requires_attention`，没有但允许流式状态时是 `streaming_state_is_sufficient`。

`test_exact_history_gate` 用两组排列验证：`['attention','ssm']` 在 `True` 下通过；`['ssm','ssm']` 在 `True` 下不通过，在 `False` 下通过。解析里专门提醒：这个布尔检查**不评估模型能否记住或正确找回某条信息**，它只检查结构上有没有保留那条路径。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 全注意力 | 为每个历史词元保存 K/V，当前查询按位置读取 |
| 线性注意力 | 把历史压成固定形状的累计统计，从中算当前输出 |
| SSM / Mamba | 用状态转移矩阵逐步更新内部状态的递推模型 |
| 递推状态（recurrent state） | 固定大小的记忆，不随上下文长度增长 |
| 层排列（schedule） | 按顺序列出每层是 `attention` 还是 `ssm` |
| KV 字节账本 | `attention_layers × seq_len × heads × head_dim × 2 × dtype_bytes` |
| `dtype_bytes` | 每个元素占几字节，BF16/FP16 为 2 |
| `exact_history_gate` | 检查排列是否保留逐词 KV 读取路径的布尔门槛 |
| `attention_every` | 每隔多少层放一层注意力，`None` 表示全递推 |
| 理论下界 | 只算静态状态，不含激活、对齐和框架开销 |

## 小白常见疑问

**Q1：为什么 KV 字节要乘 2？**
因为注意力同时保存 Key 和 Value 两份张量，形状相同，所以字节数翻倍。代码里的 `* 2` 就是 K、V 各一份；如果只算 K 或只算 V，测试的 512 会对不上。

**Q2：纯递推排列在 `requires_exact_history=False` 时通过，是不是说明它和注意力一样好？**
不是。`eligible=True` 只表示"本题没有强制要求逐词 KV 读取路径"，是结构门槛，不是质量结论。notebook 在 Step 3 和解析里都写明，远距离读取能力要用任务输出检验。

**Q3：`attention_every=4` 到底是哪几层？**
索引从 0 开始，`index % 4 == 0` 的层是注意力，即第 0、4、8… 层。测试 `build_hybrid_schedule(6, 2)` 的结果 `[attention, ssm, attention, ssm, attention, ssm]` 直观展示了这个规则。

**Q4：`recurrent_state_dim` 为什么可以直接乘层数，不需要乘序列长度？**
因为题目把递推层建模成固定大小的状态：无论历史多长，每层只保存 `recurrent_state_dim` 个元素。真实 Mamba 的状态形状由 `d_state × d_inner` 等决定，但仍与序列长度无关；notebook 用简化维度表达这一性质，不模拟具体内部结构。

**Q5：GPU 实验里"混合状态少 74.7%"是怎么读出来的？**
5.4 的已有结果记录：长度 8192 时全注意力 `state_bytes` 是 64 MiB，混合排列是 16.19 MiB，混合约为全注意力的 25.3%，所以少约 74.7%。这是**状态字节数**的比较，不是任务质量或端到端延迟；同一张表还注明单步耗时约为全注意力的 32.8%，但那是简化实现的结果。

## 动手验证：怎么知道自己做对了

`run_arch_hybrid_tests` 依次运行三个测试，全过打印 `✅ 混合记忆：层排列、保存量与指定路径检查通过`。

- `test_hybrid_schedule_contract`：`(6, 2)`、`(3, 1)`、`(3, None)` 三种排列必须精确匹配；`num_layers=0` 必须抛 `ValueError`。
- `test_memory_state_ledger`：4 层、周期 2 的排列，`seq_len=8, num_kv_heads=2, head_dim=4, recurrent_state_dim=16` 时必须得到 `attention_layers=2`、`ssm_layers=2`、`kv_bytes=512`、`recurrent_bytes=64`、`total_bytes=576`。
- `test_exact_history_gate`：混合排列在 `True` 下 `eligible is True`；纯递推在 `True` 下 `False`，在 `False` 下 `True`。

notebook 的 CPU 部分没有保存执行输出，以这些断言为准。可选的 Step 5 默认 `RUN_MODE='dry_run'`，不会跑 GPU；只有改成 `real_gpu` 并完成 `ARCH-CUDA-MECHANISM` 预检后，才会调用 `tools/run_architecture_mechanism_benchmark.py --mode memory_growth`。5.3 的读取单元在结果文件不存在时会打印"尚无混合记忆结果"，不会伪造数字。历史登记结果 `rtx5070ti_hybrid_v1` 的 `decision` 是 `accept_hybrid_state_growth_mechanism`，含义只是状态增长机制通过检查，notebook 明确说明未测真实模型的长序列任务质量。
