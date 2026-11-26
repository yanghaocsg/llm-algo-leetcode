# INF-PROPOSAL. 动态候选与验证预算 — 小白讲解

> 对应 notebook：[`INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb`](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb)

## 一句话总结
投机解码不是“草稿器越快越好”，而是“每轮提多少候选、目标模型能连续接受多少、拒绝后修正要付多少”三笔账的平衡。本节用三个 TODO 写出“按近期接受率调整预算 `k`”“在首个分歧处截断连续接受前缀”“把接受收益与草稿/验证/修正成本记成账本”，再用 Qwen2.5 1.5B 目标模型 + 0.5B 助手模型跑同条件对照：固定预算把目标前向从 192 次降到 61 次却慢了 5.9%，启发式降到 68 次仍慢 0.6%，只有置信度动态候选以 73 次目标前向、186 次助手前向把中位时间从 2594.6 ms 降到 2216.9 ms（约 14.6%），相对延迟 0.854，输出与基线完全一致。

## 为什么需要它：痛点是什么
初学者容易把“目标模型调用次数减少”直接等同于“端到端变快”。可投机解码多了一个草稿器：它每轮要先跑若干次小模型，目标模型再做一次并行验证；如果候选提得太多、被接受得太少，草稿和验证白做，调用次数是少了，总时间反而更长。反过来，候选太保守，目标模型每轮只前进一两个词元，也拿不到收益。

生活类比：点外卖时的“凑单推荐”。平台（草稿器）先猜你想加什么，商家（目标模型）一次性确认。猜 5 样只中 1 样，你要为 5 样的推荐和 1 次确认都付时间；只猜 1 样虽然几乎不浪费，但每次确认只推进一样，也很慢。理想策略是看最近几次“猜中率”决定这次猜几样，猜中率高就多猜，低就少猜，并且一旦某样不对，后面整串都不要。

## 核心问题
- 候选预算 $k$、实际提议数、连续接受数、最终提交数之间是什么关系？为什么“提出更多”不保证“接受更多”？
- 近期接受率怎样映射成“加一 / 保持 / 减一”？阈值边界（`high=0.8`、`low=0.4`）用 `>=` 还是 `>` 有区别吗？
- 为什么验证只能保留“首个分歧之前的连续前缀”？目标序列比草稿短、草稿为空时怎么返回？
- 净收益公式里，草稿成本为什么按总提议数算、收益为什么按接受数算、修正成本为什么只在 $A<K$ 时加？
- GPU 实验里目标/助手前向次数怎么数？为什么调用次数少了延迟却可能更高？
- 三种候选（`constant`、`heuristic_transient`、`dynamic_confidence`）分别隔离了什么变量？

## 解决思路：一步步看懂

**第 1 步：一轮投机解码的四个阶段。** notebook 的 Step 1 表把一轮拆成：提出候选（用当前上下文和上限 $k$，实际提议数可以小于 $k$）、目标验证（对齐位置的目标证据，输出连续接受前缀或首个拒绝）、修正或追加（拒绝位置改用目标词元，或全部接受时追加一个目标词元，这个修正词元不计入“草稿接受长度”）、更新状态（只保留已提交词元，撤销未提交候选的缓存）。两条验证路径要分清：贪心验证比较候选与同位置目标最大分词元是否相同，拒绝时用目标选择；随机验证用接受概率和正残差修正，不能只比较两个词元是否相等。本节的前缀函数只实现贪心比较，不实现随机接受或模型缓存。

**第 2 步：TODO 1——用近期接受率调预算。** 函数签名是 `choose_proposal_budget(recent_acceptance, current_k, min_k=1, max_k=8, high=0.8, low=0.4)`。校验顺序：`1 <= min_k <= current_k <= max_k` 否则 `ValueError('候选预算边界非法')`；`0.0 <= low < high <= 1.0` 否则 `ValueError('接受率阈值非法')`；每个历史值都在 $[0,1]$ 否则 `ValueError('接受率必须位于 [0, 1]')`；空历史直接返回 `current_k`。然后 `rate = mean(recent_acceptance)`，要补：

```python
adjustment = 1 if rate >= high else -1 if rate <= low else 0
return min(max(current_k + adjustment, min_k), max_k)
```

三个分支的含义：平均接受率达到 0.8 就加一，降到 0.4 及以下就减一，中间保持。这里用 `>=` 和 `<=`，所以 `[0.8]` 返回 5、`[0.4]` 返回 3，测试专门卡这两个边界。最后的 `min(max(...))` 把结果夹在 `[min_k, max_k]`，所以 `[1]` 在 `current_k=8` 时仍是 8（已经到顶），`[0]` 在 `current_k=1` 时仍是 1（已经到底）。`[0, 0, 1]` 的平均是 $1/3 \approx 0.333 \le 0.4$，返回 3；`[0.6]` 落在中间，返回 4。

为什么不直接 `current_k = round(rate * max_k)`？因为那会让预算随单轮噪声大幅跳动，而且接受率和“合适的候选数”不是线性关系。notebook 采用的是增量式控制器：每轮最多动一格，历史用均值平滑。它只根据“已经发生的接受结果”调整，不预测未来；草稿置信度是另一路信号，可以让实际提议数小于预算上限。

**第 3 步：TODO 2——找连续接受前缀。** 参考实现只有一行生成器：

```python
rejected_at = next(
    (index for index, pair in enumerate(zip(draft_tokens, target_tokens))
     if pair[0] != pair[1]),
    None,
)
```

`zip` 只配对到较短的那条序列，所以它天然处理“一长一短”；`enumerate` 给的是从 0 开始的位置；`next(..., None)` 表示“全部配对都相同”时返回 `None`。后面两行是题目已给的：

```python
accepted = min(len(draft_tokens), len(target_tokens)) if rejected_at is None else rejected_at
if rejected_at is None and len(draft_tokens) > len(target_tokens):
    rejected_at = accepted
```

逐种情况读：
- `verify_candidate_prefix([1,2,3,4], [1,2,9,4])`：位置 2 首次不同，`rejected_at=2`，`accepted=2`，后缀 `3,4` 丢弃。
- `verify_candidate_prefix([1,2], [1,2])`：全部相同，`rejected_at=None`，`accepted=min(2,2)=2`。
- `verify_candidate_prefix([1,2,3], [1,2])`：`zip` 只比较前两个，全部相同，`rejected_at=None`，`accepted=2`；但草稿比目标长（3>2），于是把 `rejected_at` 补成 2，表示“目标证据只到位置 2，后面的草稿没有被验证”。这个分支容易漏，漏了就会把没验证过的第 3 个词元当成接受。
- `verify_candidate_prefix([], [1])`：`zip` 为空，`rejected_at=None`，`accepted=min(0,1)=0`；草稿长度 0 不大于目标长度 1，所以 `rejected_at` 保持 `None`。空候选返回“接受 0、无拒绝位置”，调用方据此直接走目标模型。

为什么必须在首个分歧处截断？因为一旦某个位置被拒绝，后面草稿 token 是基于“错误前缀”生成的，条件分布已经不对，不能继续接受。贪心验证下，拒绝位置的正确词元由目标模型给出；后续候选全部作废，回到已提交上下文重新生成。

**第 4 步：TODO 3——净收益账本。** 公式在 Step 3：

$$G_A=A t_b-\big[K t_d+t_v+\mathbf{1}_{A<K}t_c\big]$$

其中 $A$ 是接受的草稿数，$K$ 是总提议数，$t_b$ 是普通解码每词元成本，$t_d$ 是草稿每词元成本，$t_v$ 是本轮验证成本，$t_c$ 是拒绝修正成本，$\mathbf{1}_{A<K}$ 是“发生拒绝才为 1”的指示函数。参考实现：

```python
saved_ms = accepted * baseline_token_ms
extra_cost_ms = (proposed * draft_token_ms + verify_ms
                 + (correction_ms if accepted < proposed else 0.0))
```

注意三个口径：收益按“接受的草稿数”算，因为只有这些位置省掉了目标模型的逐词元生成；草稿成本按“总提议数”算，因为被拒绝的后缀也已经算过；修正成本只在 $A<K$ 时加一次，全部接受时不加。测试 `test_net_gain_ledger` 的四个断言正好覆盖这些：
- `estimate_net_gain(4, 4, 2.0, 0.2, 1.0)['net_gain_ms'] == 6.2`：收益 $4\times2=8$，成本 $4\times0.2+1=1.8$，净 6.2。
- `estimate_net_gain(0, 4, 2.0, 0.5, 2.0, 1.0)['net_gain_ms'] < 0`：一个都没接受，收益 0，成本 $2+2+1=5$，净 $-5$。
- `estimate_net_gain(4, 4, 2.0, 0.2, 1.0, 99.0)['net_gain_ms'] == 6.2`：全接受，`correction_ms=99` 被忽略，仍 6.2。这条专门防止把修正成本无条件加进去。
- `estimate_net_gain(2, 4, 2.0, 0.5, 1.0, 0.25) == {'saved_ms': 4.0, 'extra_cost_ms': 3.25, 'net_gain_ms': 0.75}`：收益 $2\times2=4$，成本 $4\times0.5+1+0.25=3.25$，净 0.75。

这个账本只算“接受草稿这一部分”的时间差，不把修正词元或追加词元的贡献算进去。notebook 说得很清楚：完整生成还要看总生成时间除以最终提交词元数，并分别记录草稿接受长度与目标调用次数。

**第 5 步：GPU 实验的自变量与固定条件。** 目标模型 `Qwen/Qwen2.5-1.5B-Instruct`，助手模型 `Qwen/Qwen2.5-0.5B-Instruct`，`float16`，同 tokenizer，3 条固定提示，每条最多新增 `MAX_NEW_TOKENS=64`，初始候选预算 `INITIAL_ASSISTANT_TOKENS=5`，`WARMUP_STEPS=1`、`REPEAT_STEPS=5`，`SEED=42`。基线是“无助手的目标模型贪心生成”，参考输出取第一次 baseline 运行的 `outputs`。三个候选隔离不同变量：

| 候选 | `schedule` | `confidence_threshold` | 想隔离什么 |
|---|---|---|---|
| `constant` | `constant` | 0.0 | 固定预算，不随接受率变 |
| `heuristic_transient` | `heuristic_transient` | 0.0 | Transformers 的启发式预算调度 |
| `dynamic_confidence` | `constant` | 0.4 | 草稿置信度低于 0.4 时提前停止 |

生成配置由 `greedy_generation_config` 复制模型默认配置后改成贪心：`do_sample=False`、`temperature=None`、`top_p=None`、`top_k=None`、`repetition_penalty=1.0`、`max_new_tokens=64`、`use_cache=True`，并把 `pad_token_id` 设为 `tokenizer.eos_token_id`。这样三种候选输出可以直接逐 token 比较，不受采样随机性干扰。

**第 6 步：指标怎么量。** `run_generation` 用 `register_forward_pre_hook` 给目标模型和助手模型各挂一个计数器，分别累加 `target_forward_calls` 和 `assistant_forward_calls`；时间用 `torch.cuda.Event(enable_timing=True)` 包住整组提示的分词、搬运、生成和输出回传，最后 `torch.cuda.synchronize()`；`generated_tokens` 是每条输出长度减去对应 prompt 的输入长度；`tokens_per_target_call = generated_tokens / max(1, target_calls)`；峰值显存用 `torch.cuda.max_memory_allocated()`。每个候选跑 5 次，取 `median_latency_ms`、`percentile_95`、中位前向次数、中位每次目标调用的 token 数，并检查 `outputs_match_baseline = all(item['outputs'] == reference_outputs)`。

**第 7 步：结果怎么读。** 5.4 的 `rtx5070ti_proposal_v2` 登记值：

| 候选 | 目标前向 | 助手前向 | 相对延迟 | 结论 |
|---|---|---|---|---|
| `target_only` 基线 | 192 | 0 | 1.0 | 中位 2594.6 ms |
| `constant` | 61 | 289 | 1.059 | 调用大减，端到端更慢 |
| `heuristic_transient` | 68 | 256 | 1.006 | 基本持平，略慢 |
| `dynamic_confidence` | 73 | 186 | 0.854 | 2216.9 ms，约降 14.6% |

先看正确性：三种候选的输出都与基线逐 token 一致，说明辅助生成没有改变贪心结果。再看调用次数：`constant` 把目标前向从 192 降到 61，降幅约 68%，但助手要跑 289 次，端到端反而慢 5.9%；`heuristic_transient` 目标 68 次，仍慢 0.6%；`dynamic_confidence` 目标 73 次、助手 186 次，延迟比 0.854，三条提示合计中位时间从 2594.6 降到 2216.9 ms，降幅 $(2594.6-2216.9)/2594.6 \approx 14.56\%$。这说明“目标调用次数”只是收益的一侧，助手成本和同步/调度开销会把收益吃掉；动态候选在“少提无用候选”和“保留足够并行验证”之间取得了更好的平衡。

决策规则写在代码里：`valid` 是输出一致且 `path_confirmed` 的候选，`faster` 是 `latency_ratio_vs_baseline < 0.95` 的候选；`accept` 只要存在 faster，否则 `tune`（有 valid），否则 `reject`。所以这里的 `accept` 是“本页规则下的采用标记”，不是“已证明在所有负载下都能加速”。notebook 的下一步要求补候选预热、交替测量和逐轮样本，确认收益可重复。

**第 8 步：这份实验的边界。** 只有 3 条提示、每条 64 个新词元、单机单卡、`float16`、5 次重复；基线只用首条提示预热 1 次，候选没有各自独立预热，也没有轮换配对顺序；只保存汇总，不保存逐步样本。因此它回答的是“在这套固定工作负载上，动态置信度候选能否保持输出一致并降低中位延迟”，不能外推成“所有模型/所有提示都能加速”。更换任一模型、revision、dtype、提示或生成长度后，不能与旧结果合并。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 投机解码 | 小模型先起草多个词元，大模型一次并行验证 |
| 草稿器 / 目标模型 | 提候选的模型 / 负责最终验证和输出的模型 |
| 候选预算 $k$ | 本轮最多提多少个候选，实际提议数可以更小 |
| 连续接受前缀 | 从第一个位置开始一路相同的最长前缀，遇到分歧就停 |
| 接受率 | 每轮已接受草稿数 ÷ 已提议数，空提议轮单列 |
| 动态预算 | 按近期接受率或置信度调整下一轮的 $k$ |
| 修正成本 | 拒绝后用目标词元改正所付的代价 |
| 净收益 $G_A$ | 接受省下的基线时间减去草稿、验证、修正成本 |
| 目标前向次数 | 目标模型被调用多少次，不等于生成的词元数 |
| `tokens_per_target_call` | 每次目标调用平均推进多少词元，衡量验证效率 |
| 相对延迟 | 候选中位延迟 ÷ 基线中位延迟，<1 才是加速 |
| 贪心验证 | 只比较候选与目标最大分词元是否相同 |

## 小白常见疑问

**Q1：为什么“提议 5 个、接受 4 个”和“提议 5 个、接受 1 个”成本差很多？**
草稿成本按总提议数 $K$ 算，收益按接受数 $A$ 算。前者收益 $4t_b$、成本 $5t_d+t_v$；后者收益只有 $1t_b$，成本仍是 $5t_d+t_v$，还要加修正成本。notebook 的 Step 3 用 $A=K=4$、$t_b=2$、$t_d=0.2$、$t_v=1$ 举例，净收益 6.2 ms；如果只接受 1 个，收益立刻缩水到 2 ms，账本可能转负。

**Q2：净收益公式里为什么修正成本要乘 $\mathbf{1}_{A<K}$？全接受时修正不是也要发生吗？**
全接受时，本轮所有草稿都被接受，不需要“改正”某个被拒绝的位置；目标模型只需在末尾追加一个额外词元，这个追加不计入“草稿接受长度”，也不在 $G_A$ 的修正项里。notebook 的测试用 `correction_ms=99` 且 $A=K=4$ 断言净收益仍是 6.2，就是在验证“全接受不扣修正”。

**Q3：`verify_candidate_prefix([1,2,3], [1,2])` 为什么 `rejected_at` 是 2，而不是 `None`？**
`zip` 只比较前两个，确实全部相同；但草稿还有第 3 个词元没有被目标证据覆盖。函数把“目标序列先结束”视为验证边界，返回 `rejected_at=2`，避免调用方把未验证的第 3 个词元当成接受。`accepted` 仍是 2。

**Q4：目标前向从 192 降到 61，为什么 `constant` 反而更慢？**
因为它把助手前向推到了 289 次，而且每轮验证的并行度/同步开销更高。`run_generation` 的延迟覆盖整组提示的分词、搬运、生成和回传，不是只算目标模型。notebook 的 5.4 明确写“调用更少但延迟更高时，说明助手成本或同步开销抵消了收益”；要定位差异，得把助手计算、目标验证和主机等待分开看。

**Q5：`accept` 是不是表示这个策略可以直接上线？**
不是。`accept` 的判定只要求“输出一致且中位延迟比 < 0.95”。它是在 3 条提示、5 次重复、单机 fp16 的局部机制基准上成立；notebook 要求补预热、交替测量、逐轮样本，再进入 68 节的后端复测。更换模型/提示/长度/dtype 后不能合并旧结果。

## 动手验证：怎么知道自己做对了

测试区有 4 个函数，直接调用，TODO 未填会抛 `NotImplementedError`，全过才打印 `✅ 动态候选机制测试通过`。

- `test_budget_adaptation`：`([1,1,1],4) == 5`；`([0,0,1],4) == 3`（均值 $1/3$ 落到低阈值）；`([],4) == 4`；`([1],8) == 8`（上界夹住）；`([0.8],4) == 5`（等于 high 也加一）；`([0.4],4) == 3`（等于 low 也减一）；`([0.6],4) == 4`；`([0],1) == 1`（下界夹住）。
- `test_budget_rejects_invalid_acceptance`：`[1.2]` 与 `[-0.1]` 都必须抛 `ValueError`。
- `test_contiguous_verification`：`([1,2,3,4],[1,2,9,4]) == {'accepted': 2, 'rejected_at': 2}`；`([1,2],[1,2]) == {'accepted': 2, 'rejected_at': None}`；`([1,2,3],[1,2]) == {'accepted': 2, 'rejected_at': 2}`；`([],[1]) == {'accepted': 0, 'rejected_at': None}`。
- `test_net_gain_ledger`：全接受 4/4 的净收益 6.2；0/4 且带修正 1.0 时净收益 < 0；全接受时 `correction_ms=99` 不影响结果仍为 6.2；部分接受 2/4 的字典严格等于 `{'saved_ms': 4.0, 'extra_cost_ms': 3.25, 'net_gain_ms': 0.75}`。

GPU 部分没有断言，只有保存的 JSON 与 5.4 的登记。notebook 保存的输出显示 `run_mode: 'real_gpu'`、`run_label: 'rtx5070ti_proposal_v2'`、目标模型 commit `989aa798...`、助手 commit `7ae55760...`、`failure: None`、`evidence_level: 'local_mechanism_benchmark'`、`decision: 'accept'`；5.4 记录基线中位 2594.6 ms、192 次目标前向，`constant` 61/289 次、延迟比 1.059，`heuristic_transient` 68/256 次、1.006，`dynamic_confidence` 73/186 次、0.854、2216.9 ms。你自己复跑时先确认三个候选的 `outputs_match_baseline` 都是真，再看延迟比是否小于 0.95；数值随机器变化，不要死记。
