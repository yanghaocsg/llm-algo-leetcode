# ARCH-HYBRID-MEMORY. Attention SSM Hybrid | Attention、SSM 与混合记忆架构 · 小白入门

> 配套 notebook：[ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构](../ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

有的层靠注意力逐条回看历史，有的层只维护一份固定大小的状态。这本 notebook 讲怎样安排这两种层各放在模型的哪一层，再算清状态占多少、够不够回看历史。

## 为什么需要它

处理序列的层，保存历史有两种办法。全注意力保留每个历史词元的 K 和 V，想回看哪一条都能取到，代价是缓存随长度线性增长。递推层只维护一份固定形状的状态，更新成本低，代价是历史被压缩过，不能像注意力那样按词元重新访问。

看下面的小例子：4 层的模型，2 层注意力、2 层递推。序列从 4 个词元变成 8 个词元时，注意力层的 K/V 状态从 64 字节涨到 128 字节，递推层的状态一直是 16 字节。

混合架构的做法是分工：一部分层保留注意力，负责精确回看历史；其余层用递推状态，压低长序列的状态成本。但这不只是把两种层交替摆一下，还要说清楚哪些层负责什么。

这本 notebook 把“哪一层是哪种”写成一张层级计划，算两类状态各占多少字节，再判断这份计划能不能满足任务对精确历史的要求。它只算 CPU 上的理论账本。混合是不是更高效，要靠真实模型在固定训练条件下的延迟和质量来回答。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 全注意力 | 保留每个历史词元（token）K 和 V 的层，可以随时回看具体某一条历史 |
| 递推层 / 状态空间层 | 只维护一份固定形状状态的层。线性注意力和 SSM 都属于这一类 |
| SSM | 状态空间模型。靠状态转移和输入相关的控制来更新内部状态，本节和 Mamba 写在一起 |
| 层级计划（schedule） | 一个列表，按层的顺序写明每一层是 `attention` 还是 `ssm` |
| 状态账本 | 两类状态各占多少字节：注意力层的 K/V 字节数，递推层的状态字节数 |
| 精确历史访问 | 任务要求能重新读到具体某个历史词元，而不只是它的汇总 |
| 流式（streaming） | 一边来一边更新的任务。对这类任务，不强制保留注意力层 |

## 核心思路

输入是层数 `num_layers` 和周期 `attention_every`。整个流程分三步：

```text
1. 层级计划：层编号从 0 数起，编号能被 attention_every 整除的层用 attention，其余用 ssm
             attention_every 为 None 时全部用 ssm，也就是纯递推
2. 状态账本（单样本）：
   K/V 字节数     = 注意力层数 × seq_len × num_kv_heads × head_dim × 2 × dtype_bytes
   递推状态字节数 = 递推层数 × recurrent_state_dim × dtype_bytes
   总字节数       = K/V 字节数 + 递推状态字节数
3. 精确历史访问：计划里有注意力层，或者任务不要求精确历史，这份计划才满足要求
```

`seq_len` 是序列长度，`num_kv_heads` 是 K/V 头数，`head_dim` 是每个头的维度，系数 2 表示 K 和 V 各一份，`recurrent_state_dim` 是每层递推状态的维度，`dtype_bytes` 是每个数占的字节数（默认 2）。只有 K/V 字节数这一项含 `seq_len`，所以只有它随序列长度增长。

用 4 层、每 2 层一个注意力层走一遍。K/V 头数 1，头维度 2，递推状态维度 4，每个数 2 字节：

```python
num_layers, attention_every = 4, 2
schedule = ["attention" if i % attention_every == 0 else "ssm" for i in range(num_layers)]   # TODO 1
print(schedule)

att, ssm = schedule.count("attention"), schedule.count("ssm")
for seq_len in (4, 8):                        # TODO 2：序列变长
    kv = att * seq_len * 1 * 2 * 2 * 2        # 注意力层数 × 长度 × K/V 头数 × 头维度 × 2 × 字节数
    rec = ssm * 4 * 2                         # 递推层数 × 状态维度 × 字节数
    print(seq_len, kv, rec, kv + rec)

for plan, need in ((schedule, True), (["ssm", "ssm"], True), (["ssm", "ssm"], False)):   # TODO 3
    has = "attention" in plan
    print(has, need, has or not need)
```

```text
['attention', 'ssm', 'attention', 'ssm']
4 64 16 80
8 128 16 144
True True True
False True False
False False True
```

第二、三行依次是序列长度、K/V 字节数、递推状态字节数、总字节数。序列从 4 变成 8，K/V 字节数从 64 翻到 128，递推状态一直是 16。后三行依次是“计划里有没有注意力层”“任务是否要求精确历史”“这份计划是否满足”。只有中间那一行不满足：纯递推的计划，任务又要求精确历史。

notebook 的测试用的是另一组数：4 层、序列长度 8、2 个 K/V 头、头维度 4、递推状态维度 16。算出来 K/V 是 512 字节，递推状态是 64 字节，总共 576 字节。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 全注意力、线性注意力、SSM 三类记忆接口各保存什么，状态随长度怎样变化 | 读 |
| Step 2 | 混合架构里注意力层、递推层、层间接口各自负责什么、要检查什么 | 读 |
| Step 3 | 判断架构要同时记录哪几项，CPU 账本之后还需要哪些真实证据 | 读 |
| Step 4 · TODO 1 | 生成混合层级计划 | 在 `build_hybrid_schedule` 里写 1 个表达式：`attention_every` 为 `None` 时全是 `ssm`，否则层编号能被它整除的层是 `attention` |
| Step 4 · TODO 2 | 计算混合记忆状态账本 | 在 `memory_state_ledger` 里写 2 行：`kv_bytes` 和 `recurrent_bytes` |
| Step 4 · TODO 3 | 判断精确历史访问条件 | 在 `exact_history_gate` 里写 1 行：`eligible`，由“有注意力层”和“是否需要精确历史”组合出来 |
| 测试 | 检查纯注意力、纯递推、周期混合的计划和非法层数；状态账本的层数与字节数；混合、纯递推、流式任务三种情况下的准入结果 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上对照全注意力、全递推和混合层计划的状态量、单步时间和峰值显存 | 需要单张 CUDA GPU 和 `ARCH-CUDA-MECHANISM` 环境；默认 `RUN_MODE = 'dry_run'`，只登记计划 |

Step 1–4 只需要 CPU。Step 5 是可选实验，需要单张 CUDA GPU 和 `ARCH-CUDA-MECHANISM` 环境；配置单元里 `RUN_MODE` 默认是 `'dry_run'`，预检通过后才能改成 `real_gpu`。

## 容易踩的坑

- **注意力层放错了位置**：周期混合的计划在第 0 层和后续的周期位置放注意力。从第 1 层开始数，结果就和测试对不上。
- **K/V 字节数漏乘序列长度或系数 2**：注意力层每个词元都要存 K 和 V，所以字节数里有 `seq_len`，也有系数 2。递推层的状态和长度无关。
- **层数不合法还继续算**：`num_layers` 必须是正整数，`attention_every` 要么是正整数，要么是 `None`。层数为 0 会抛出 `ValueError`。
- **计划里出现了别的层类型**：层级计划只能包含 `attention` 和 `ssm`，空计划也会被拒绝。
- **把状态小当成质量不变**：固定大小状态不等于固定质量。任务需要精确回看历史时，计划里至少要保留一层注意力。

## 读完后自查

- [ ] 给 6 层、每 2 层一个注意力层，能写出层级计划。
- [ ] 能写出 K/V 字节数和递推状态字节数的公式，并说出哪一个随序列长度增长。
- [ ] 能判断“纯递推计划加要求精确历史的任务”通不通过，并说出原因。
- [ ] 能说出 CPU 账本之后，还需要哪些真实证据才能回答“混合是否更高效”。

## 下一步

- 打开 notebook：[ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构](../ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md)
- 建议先读：[45. 线性注意力与递推状态 · 小白入门](./45_Linear_Attention_and_Recurrent_State.md)、[05. LLaMA3 Block 教程 · 小白入门](./05_LLaMA3_Block_Tutorial.md)
- 想深入可以看：[10 线性 Attention、SSM 与混合记忆](../../topic_discussion/llm_architecture_evolution/10_linear_attention_ssm_and_hybrid.md)
- 相关的 notebook：[77. 长序列记忆架构基准](../77_Long_Sequence_Memory_Architecture_Benchmark.md)
