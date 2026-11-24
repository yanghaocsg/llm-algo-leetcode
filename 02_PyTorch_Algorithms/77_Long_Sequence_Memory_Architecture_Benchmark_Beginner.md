# 77. Long Sequence Memory Architecture Benchmark Beginner | 长序列记忆架构基准 · 小白入门

> 配套 notebook：[77. 长序列记忆架构基准](./77_Long_Sequence_Memory_Architecture_Benchmark.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first；GPU 可选

## 一句话说清

处理长文本时，模型把读过的内容存成什么，决定了它记不记得住、要多占多少显存。这本 notebook 用一张状态账本和一个小测试，比较三种保存历史的办法。

## 为什么需要它

模型读一段长文本，前面的内容要以某种形式留着，后面才用得上。最常见的做法是 KV Cache：每个历史词元的 K 和 V 都存下来。好处是随时能查，坏处是文本越长存得越多。

notebook 比较三种保存历史的办法：

- 显式 KV。每个词元的 K/V 都保存。能按内容检索历史，但状态随长度线性增长。
- 递推状态。不存每个词元，把历史压成一份固定大小的状态。状态大小基本不变，但历史只能通过当前状态间接读取。
- 混合。一部分层保存 K/V，一部分层用递推状态，由层分工。

看 notebook 测试里的一对候选。一个有 4 层注意力、2 个 KV 头、每头 8 维，每个数占 2 字节：序列长度 16 时 KV 状态是 4096 字节，长度翻倍就是 8192 字节。另一个有 4 层递推，状态维度是 8 乘 8：不管序列多长，都是 512 字节。状态小了很多。但它还能不能读出远处那条信息，要另外检查。

这本 notebook 把状态大小、历史读取能力、长距离质量和服务成本分开记录，最后整理成可复查的架构选型记录。它还提醒：不同公开模型的数字通常只是观察性证据，不能把差异全部归因于记忆机制。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| KV Cache | 注意力为每个历史词元保存的 K 和 V。后面的词元算注意力时直接拿来用 |
| 递推状态（recurrent state） | 递推模型不保存每个词元，而是维护一份固定大小的状态，每读一个词元就更新一次 |
| 混合架构（hybrid architecture） | 一部分层用注意力，一部分层用递推，两类层各管一种记忆 |
| 状态账本 | 记录每种架构要保存多少状态的账。本 notebook 只算 KV 和递推状态两部分 |
| needle | 藏在长输入深处的一条信息，用来检查模型后面还能不能把它读出来 |
| 预填充（prefill）/ 解码（decode） | 预填充是处理输入的阶段，解码是一个一个生成新词元的阶段 |
| 观察性证据 | 拿来比较的模型不只差记忆机制，规模、训练数据和分词器都可能不同，差异说不清来自哪里 |

## 核心思路

输入是三类候选的结构规格 `spec`、序列长度，以及真实模型的评测记录。五个 TODO 依次是：校验结构、算状态账本、模拟读取历史、检查证据字段、给出决策。

先看状态账本（TODO 2）：

```text
kv_bytes              = 2 × B × L × N_a × H_kv × D_h × dtype_bytes
recurrent_state_bytes = B × N_r × state_dim × value_dim × dtype_bytes
```

`B` 是批大小，`L` 是序列长度，`N_a` 是注意力层数，`H_kv` 是 KV 头数，`D_h` 是每头的维度，`N_r` 是递推层数，`dtype_bytes` 是每个数占的字节数。第一行里的 2 来自每个词元要保存 K 和 V 两份。`L` 只出现在第一行，所以只有 KV 状态随长度增长。

再看历史怎样读取（TODO 3）。`simulate_memory_read` 是一个玩具接口，输入一串词元和一个目标位置：

```text
explicit_kv      精确读出目标词元                    read_mode = exact_history
recurrent_state  只回答“见过多少词元、是否包含目标”    read_mode = summary_only
hybrid           目标在最后 local_window 个里才读得出   read_mode = local_exact，否则 summary_only
```

最后是证据和决策（TODO 4、TODO 5）：

```text
合格记录  quality 有 task、eval_length、metric；cost 有 prefill_ms、decode_tpot_ms、peak_memory_mb；
          evidence 有 evidence_level、failure
可行候选  quality.metric >= min_quality 且 cost.peak_memory_mb <= max_peak_memory_mb
决策      在可行候选里选峰值显存最小、prefill 最短的
          evidence_level 是 same_family_variant 才 accept，否则 tune；没有可行候选是 reject
```

用测试里的规格算一下状态账本：

```python
dtype_bytes = 2   # 每个数占 2 字节

def state_bytes(attention_layers, recurrent_layers, seq_len):
    kv = 2 * seq_len * attention_layers * 2 * 8 * dtype_bytes   # 2 个 KV 头，每头 8 维
    recurrent = recurrent_layers * 8 * 8 * dtype_bytes           # state_dim 和 value_dim 都是 8
    return kv, recurrent

print(state_bytes(4, 0, 16), state_bytes(4, 0, 32))   # 显式 KV：长度 16 和 32
print(state_bytes(0, 4, 16), state_bytes(0, 4, 32))   # 纯递推：长度 16 和 32
print(state_bytes(2, 2, 32))                          # 混合：长度 32
```

```text
(4096, 0) (8192, 0)
(0, 512) (0, 512)
(4096, 256)
```

第一行是显式 KV 候选：长度从 16 到 32，KV 字节数翻倍。第二行是纯递推候选：两次都是 512。第三行是混合候选在长度 32 时的账本，总量介于前两种之间。

notebook 对结论留了余地。这个账本只解释状态趋势，不等于真实的峰值显存。玩具接口只演示三种记忆的差别，递推状态回答“见过”不等于模型在任务里能读出来，真实 SSM 的质量仍需任务评测。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 三类候选要在相同的长度阶梯、任务、批大小、`dtype`、后端和硬件下比较，混合架构还要记录层安排 | 读 |
| Step 2 | 状态账本与历史读取是两件事：三类候选各保存什么、怎样读取历史 | 读，对照上面的两个公式和读取方式 |
| Step 3 | 质量、成本、可比性、决策四个证据维度分别要记录什么 | 读 |
| Step 4 · TODO 1 | 结构契约：检查 `architecture_type`、层数分配与各类状态维度 | 在 `validate_memory_architecture` 里确定 `architecture_type` 和 `issues`，逐条检查层数和维度 |
| Step 4 · TODO 2 | 状态账本：KV 只由 `attention_layers` 随 `seq_len` 增长，递推状态由 `recurrent_layers` 和状态维度决定 | 在 `estimate_memory_state` 里按两个公式写 `kv_bytes` 和 `recurrent_state_bytes` |
| Step 4 · TODO 3 | 历史读取：`explicit_kv` 精确读出目标词元，`recurrent_state` 只返回是否见过的摘要，`hybrid` 只精确保留尾部 `local_window` | 在 `simulate_memory_read` 里按三种架构返回 `read_mode` 和 `recovered` |
| Step 4 · TODO 4 | 证据字段：`quality` 要有 `task`、`eval_length`、`metric`，`cost` 要有 `prefill_ms`、`decode_tpot_ms`、`peak_memory_mb` | 在 `validate_memory_evidence` 里把缺的段和字段记进 `issues` |
| Step 4 · TODO 5 | 决策门槛：先检查证据，再用 `quality.metric` 和 `cost.peak_memory_mb` 筛选候选 | 在 `recommend_memory_architecture` 里写 `feasible`，选出最优候选后按证据等级给出 `decision` |
| 测试 | 四个测试依次检查结构、长度翻倍时的状态增长、玩具接口的读取结果、证据字段与决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 真实模型的长度阶梯对照：在短、中、长三个长度档记录远程读取、prefill、TPOT 和显存 | 需要 CUDA 版 PyTorch、Transformers 和至少两个模型候选。开关 `RUN_MEMORY_ARCH_GPU` 由环境变量决定，未设置时为 `0`（即不执行） |

Step 1–4 只需要 CPU。Step 5 需要 GPU，默认关闭。

## 容易踩的坑

- **层数对不上**：`attention_layers + recurrent_layers` 必须等于 `num_layers`。显式 KV 不能含递推层，纯递推不能含注意力层。测试里把混合候选的注意力层数改成 3，就不通过。
- **维度和层数不匹配**：有注意力层就要有正的 `kv_heads` 和 `head_dim`，有递推层就要有正的 `state_dim` 和 `value_dim`。
- **把状态账本当峰值显存**：`estimate_memory_state` 算的是结构账本，不等于 allocator、activation 或真实峰值显存。长度翻倍时只有 KV 字节数翻倍，递推状态不变。
- **把玩具接口当质量结论**：递推状态返回“见过目标”，不等于模型在任务里能读出它。真实 SSM 的质量仍需任务评测。
- **跨模型数字直接宣称收益**：真实记录要同时有质量、成本、证据等级和失败状态，单个 `score` 判断不了长上下文能力。跨模型的观察即使数值更好，也只能 `tune`；同家族变体在质量和预算达标后才 `accept`。

## 读完后自查

- [ ] 能说出三类候选各保存什么、状态怎样随序列长度变化。
- [ ] 能写出 KV 状态和递推状态的两个公式，并解释公式里的 2。
- [ ] 能说出三种架构在玩具接口下的 `read_mode` 各是什么。
- [ ] 能解释为什么跨模型的观察结果只能 `tune`。

## 下一步

- 打开 notebook：[77. 长序列记忆架构基准](./77_Long_Sequence_Memory_Architecture_Benchmark.ipynb)
- 建议先读：[45. 线性注意力与递推状态 · 小白入门](./45_Linear_Attention_and_Recurrent_State_Beginner.md)、[ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构 · 小白入门](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid_Beginner.md)
- 想系统了解：[大模型架构 · 线性 Attention、SSM 与混合记忆](../topic_discussion/llm_architecture_evolution/10_linear_attention_ssm_and_hybrid.md)
- 做完之后：[61. 模型架构审计 · 小白入门](./61_Model_Architecture_Exploration_Beginner.md)
