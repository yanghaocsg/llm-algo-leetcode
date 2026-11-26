# 23. 投机解码 — 小白讲解

> 对应 notebook：[`23_Speculative_Decoding.ipynb`](./23_Speculative_Decoding.ipynb)

## 一句话总结
投机解码让便宜的小模型（草稿模型）一次提出 K 个候选 token，再让大模型（目标模型）用一次前向同时验证这些位置：候选以 $\min(1,p/q)$ 的概率被接受，第一次拒绝就从"正残差分布"采样一个修正 token，全部接受则额外采一个 bonus token。四个 TODO 分别实现接受概率、残差分布、接受判断和追加 token，测试用固定概率夹具逐条核对，保证最终输出分布仍然等于目标分布。

## 为什么需要它：痛点是什么
自回归生成一次前向只产出一个 token，大模型的大部分算力花在"逐个确认"上。直觉上可以"让小模型先写 5 个，大模型觉得没问题就全收"，但如果简单地"小模型写什么就收什么"，输出分布会变成小模型的分布；如果"大模型不同意就按大模型重新采一个"，分布也会偏。投机解码的难点不在"猜"，而在**猜错时如何修正，才能让最终分布严格等于目标模型**。

生活类比：主编（目标模型）让小助理（草稿模型）先写一段稿子。主编一次性通读 5 句：认可的句子直接采用；遇到第一句不认可，就按"主编觉得该写、但助理没写"的那部分灵感重写这一句，后面助理写的全部作废。这样既省了逐句讨论的时间，最终成稿的标准仍然是主编的标准。

## 核心问题
- 草稿分布 $q_i$、目标分布 $p_i$ 和候选 token 为什么要按位置一一对齐？目标模型为什么多给一行？
- 接受概率 $\alpha(x)=\min(1,p(x)/q(x))$ 是怎么保证分布不变的？
- 被拒绝后为什么不能直接从 $p$ 重采样，而要用正残差 $r(v)$？
- 第一次拒绝后，后面的候选为什么必须全部作废？
- 全部接受后，bonus token 从哪一行采？它和前 K 行有什么不同？

## 解决思路：一步步看懂
### 第 1 步：三份输入的形状契约
`speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None)` 接收：

- `draft_probs`: `[K, vocab_size]`，草稿模型在 K 个位置的分布；
- `target_probs`: `[K+1, vocab_size]`，目标模型在 K 个验证位置 **加 1 个额外位置**上的分布；
- `draft_tokens`: 长度 K，草稿实际提出的 token id。

`_validate_probability_contract` 会检查：两者都是二维；`target_probs.shape == (K+1, vocab_size)`；`len(draft_tokens) == K`；所有值有限、非负；每一行之和在 `atol=1e-5` 内等于 1；token id 是词表范围内的 `int`。第 K+1 行不是"多算了一行浪费"，而是给"全部接受"准备的：如果 K 个候选全过，目标模型已经知道第 K+1 个位置该是什么分布，可以免费再送一个 token。

为什么必须"同一前缀、同一位置、同一词表、各自声明的温度和过滤规则"？因为接受概率比的是同一个位置上的两个概率。如果草稿用了 top-p、目标没用，或者位置错开一位，$p(x)$ 和 $q(x)$ 就不是同一件事，比值没有意义。

### 第 2 步：TODO 1——接受概率
$$\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).$$

其中 $x$ 是草稿提出的那个 token，$p(x)$ 是目标模型给它的概率，$q(x)$ 是草稿模型给它的概率。实现：

```python
p, q = float(target_prob), float(draft_prob)
if q == 0.0:
    return 1.0 if p > 0.0 else 0.0
return min(1.0, p / q)
```

逐项解释：
- 当 $p \ge q$ 时，比值 ≥ 1，截断到 1，**必定接受**；
- 当 $p < q$ 时，只以 $p/q$ 的比例接受，接受得少的那部分质量，正好留给后面的残差修正；
- $q=0$ 时不能做除法。正常流程里候选是从 $q$ 采出来的，必有 $q(x)>0$；这个分支是防御性用例（Step 4 明确说"零草稿概率属于辅助函数的防御性用例"），返回 1.0 当且仅当 $p>0$。

关键恒等式：候选 $x$ 被接受时贡献的质量是 $q(x)\alpha(x)=\min(p(x),q(x))$。把所有 token 的这份质量加起来，就是 $\sum_v \min(p(v),q(v))$，剩下的 $1-\sum_v\min(p,q)=\sum_v\max(p(v)-q(v),0)$ 正好是拒绝后要补的质量。

### 第 3 步：TODO 2——正残差修正分布
$$r(v)=\frac{\max(p(v)-q(v),0)}{\sum_u\max(p(u)-q(u),0)}.$$

$v$ 遍历词表，$u$ 只是分母里的求和下标。实现：

```python
residual = torch.clamp(target_row - draft_row, min=0)
residual_mass = residual.sum()
if residual_mass <= 0:
    raise ValueError('residual 概率质量必须大于 0')
return residual / residual_mass
```

**为什么不能直接从 $p$ 重采样？** 因为被拒绝这件事本身携带信息：它说明"这个 token 在 $q$ 里被高估了"。如果直接按 $p$ 采，会把 $p$ 中已经被接受部分重复计算，最终分布偏大。正残差只取"目标比草稿多出来的那部分质量"，再归一化，刚好补上拒绝留下的缺口。

notebook 的两词元例子：$q=(0.8,0.2)$、$p=(0.4,0.6)$。
- 第 1 个 token 的接受概率 $\min(1,0.4/0.8)=0.5$，接受质量 $0.8\times0.5=0.4$；
- 第 2 个 token 的接受概率 $\min(1,0.6/0.2)=1$，接受质量 $0.2\times1=0.2$；
- 总接受质量 0.6，拒绝质量 0.4；
- 残差 $\max(p-q,0)=(0,0.4)$，归一化后 $(0,1)$，拒绝时只能修正成第 2 个 token；
- 最终分布：第 1 个 0.4，第 2 个 $0.2+0.4=0.6$，正好是 $p$。

如果残差总质量为 0（例如 $p=q$ 但候选却被拒，或两行完全相同），说明没有任何可补的质量，必须抛 `ValueError`，不能返回全 0 分布让 `multinomial` 崩掉。

### 第 4 步：TODO 3——逐位置接受，首次拒绝即停
```python
alpha = _accept_probability(p, q)
random_draw = torch.rand((), generator=generator).item()
accepted = random_draw < alpha
if accepted:
    accepted_tokens.append(int(token_id))
    continue
residual = _normalized_residual(target_probs[i], draft_probs[i])
correction = torch.multinomial(residual, 1, generator=generator).item()
return {'tokens': accepted_tokens + [correction],
        'accepted_count': len(accepted_tokens),
        'rejected': True,
        'target_positions_checked': i + 1}
```

三个细节：
1. **从左到右验证**。只有前缀全部接受，第 i 个位置的目标分布才是在"正确历史"下算出来的；一旦前面拒绝，后面的 $p_i$ 就对应了错误的前缀，不能再用。
2. **首次拒绝立即返回**。`accepted_tokens` 只保留拒绝之前的候选，拒绝位置用 correction 替换，后面的草稿全部丢弃。这就是"关闭候选后缀"。
3. **`target_positions_checked` 记 `i+1`**，表示本轮目标模型实际检查了几个位置，不是 K。

### 第 5 步：TODO 4——全部接受后的 bonus
```python
bonus = torch.multinomial(target_probs[K], 1, generator=generator).item()
return {'tokens': accepted_tokens + [bonus], 'accepted_count': K,
        'rejected': False, 'target_positions_checked': K}
```

全部 K 个候选都接受时，目标模型已经算过第 K+1 个位置的分布（`target_probs[K]`），直接从这一行采一个 token 作为"免费赠送"。**不能复用 `target_probs[K-1]`**：那一行是第 K 个候选位置的验证分布，位置已经用掉了；用它等于把同一个位置的分布采两次，分布会错。

### 第 6 步：把测试夹具走一遍
`_build_speculative_fixture()` 构造 5 词表、K=3：

- `draft_probs` 三行分别是 one-hot 的 token 0、1、2；
- `target_probs` 前三行分别是 one-hot 的 token 0、3、2，第 4 行 one-hot token 4；
- `draft_tokens = [0, 1, 2]`。

`test_rejection_and_residual`：
- i=0：token 0，$p=q=1$，α=1，接受；
- i=1：token 1，$p=0$、$q=1$，α=0，拒绝；残差 = target 行的 token 3 减去 draft 行的 token 1，归一化后只能采到 3；
- 返回 `tokens=[0, 3]`、`accepted_count=1`、`rejected=True`、`target_positions_checked=2`。

`test_all_accept_and_bonus`：把 `target_probs[:3]` 当草稿、token 传 `[0,3,2]`，每行 $p=q$ 所以 α=1 全接受，bonus 从第 4 行采到 4，返回 `[0,3,2,4]`、`accepted_count=3`、`rejected=False`。

`test_zero_q_acceptance`：draft `[[0,1,0]]`、target 两行 `[[0.5,0.5,0],[0,1,0]]`、token 0。$q(0)=0$、$p(0)=0.5>0$，走防御分支 α=1，接受；bonus 从第 2 行采到 1，`accepted_count=1`。

`test_zero_residual_failure`：draft `[[1,0]]`、target `[[1,0],[1,0]]`、token 1。$p(1)=q(1)=0$，α=0 拒绝；但残差为 0，必须抛 `ValueError`。

`test_probability_mass_recovery` 用 float64 做账：`alpha=[0.5,1.0]`、`accepted_mass=[0.4,0.2]`、`rejected_mass=0.4`、`residual=[0,1]`，`recovered=[0.4,0.6]`，`assert_close` 通过；同时断言"直接用 target 重采样"的 `accepted_mass + 0.4*target = [0.56,0.44]` 不等于 target，说明残差不是可选项。

### 第 7 步：GPU 实验怎么读
Step 5 用 `Qwen/Qwen2.5-3B-Instruct` 做目标模型、`Qwen/Qwen2.5-0.5B-Instruct` 做草稿模型，BF16，三条固定提示，贪心生成 64 token，每轮最多 5 个草稿候选，两条路径各自预热 3 次、正式重复 10 次。对照是"仅目标模型"与"目标 + 固定 5-token 草稿预算"，记录输出一致性、目标前向次数、每次目标调用推进的 token 数、Median/P95、TPOT、峰值显存和失败状态。

保存的 `rtx5070ti_qwen3b_05b_v2` 结果：Median 从 4661.6 ms 降到 3634.8 ms，P95 从 4742.8 ms 降到 3699.8 ms，约 $4661.6/3634.8\approx1.28$ 倍加速；目标模型前向次数从 192 降到 64（3 条提示 × 64 token）；峰值显存从 5.87 GiB 升到 6.81 GiB（+0.94 GiB，草稿模型和额外 KV 的开销）。每次目标前向平均推进 $192/64=3$ 个 token。

正确性上：三条提示里两条逐 token 一致；第三条在第 4 个生成 token 稳定分歧，目标 top-1/top-2 的概率差 margin 是 0.125，恰好等于一个 BF16 ULP。`decision` 因此是 `tune_numeric_equivalence`：机制和性能证据通过，但严格逐 token 复现没过。读结果时要区分这三层——**分布正确性**（数学上由接受/残差保证）、**数值等价性**（BF16 近并列时会翻转）、**端到端收益**（取决于草稿命中率和额外成本）。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 草稿模型（draft） | 便宜的小模型，负责一次提出多个候选 |
| 目标模型（target） | 要保留其分布的大模型，负责验证 |
| K | 一轮草稿提出的候选个数 |
| 接受概率 α | $\min(1,p/q)$，候选被目标分布接受的机会 |
| 正残差 $r(v)$ | 目标比草稿多出来的概率质量，归一化后用于修正 |
| correction token | 首次拒绝时从残差分布采出的替换 token |
| bonus token | 全部接受后从第 K+1 行目标分布额外采的 token |
| `accepted_count` | 本轮真正提交的连续候选数 |
| `target_positions_checked` | 目标模型本轮检查了几个位置 |
| 贪心路径 | 不采样，逐位置核对目标最大分词元 |
| BF16 ULP | BF16 下相邻可表示数的间隔，用来衡量数值分歧 |

## 小白常见疑问
**Q1：为什么接受概率要截断到 1，而不是直接用 $p/q$？**
因为 $p/q$ 可能大于 1，概率不能超过 1。当 $p \ge q$ 时，目标模型比草稿更认可这个 token，没有理由拒绝，直接以概率 1 接受。截断后，接受质量正好是 $\min(p,q)$，这是后面质量守恒的基础。

**Q2：草稿概率为 0 时为什么返回 1.0？这不是除零吗？**
正常流程里候选是从草稿分布采出来的，$q(x)=0$ 的 token 根本不会被提出，所以这是防御性分支。实现先判断 `q == 0.0`，避免除零；若目标概率 $p>0$，说明这个 token 在目标分布里有可能，接受它不会引入错误质量。测试 `test_zero_q_acceptance` 就是专门覆盖这条路径。

**Q3：拒绝后直接从目标分布 $p$ 采一个不就行了，为什么要算残差？**
直接按 $p$ 采会把"已经被接受的质量"再算一遍。以 $q=(0.8,0.2)$、$p=(0.4,0.6)$ 为例，接受质量已经是 $(0.4,0.2)$，若拒绝时再按 $p=(0.4,0.6)$ 采，最终第 1 个 token 会变成 $0.4+0.4\times0.4=0.56$，第 2 个变成 $0.2+0.4\times0.6=0.44$，不等于 $(0.4,0.6)$。`test_probability_mass_recovery` 里的第二个断言就是在检查这个错误。

**Q4：为什么第一次拒绝后，后面的候选不能继续验证？**
因为目标模型在第 i+1 个位置算出的分布，前提是"前 i 个 token 是草稿给的那串"。一旦第 i 个位置被替换成 correction，后面的历史就变了，之前算的 $p_{i+1}$ 对应的是错误前缀，不能再用来接受或修正。所以只能关闭后缀，等下一轮重新提议。

**Q5：bonus token 为什么一定从 `target_probs[K]` 采，不能用 `target_probs[K-1]`？**
`target_probs[K-1]` 是第 K 个候选位置的验证分布，那个位置已经被第 K 个草稿 token 占用；`target_probs[K]` 才是候选块之后的第一个新位置。用错行等于在同一个位置采两次，会破坏分布。notebook 的 TODO 4 注释特意写了"从额外的第 K 行目标分布采样，而不是复用验证位置"。

**Q6：为什么 `target_positions_checked` 在拒绝时是 `i+1`，全部接受时是 K？**
它记录的是"目标模型本轮实际检查/使用了几个位置"。拒绝发生在第 i 个位置（从 0 开始），所以检查了 i+1 个；全部接受时检查了 K 个验证位置，第 K+1 行只用来采 bonus，不算"验证位置"。

## 动手验证：怎么知道自己做对了
`test_speculative_decoding()` 先 `torch.manual_seed(42)`，再依次跑六个测试，全过打印 `✅ 投机解码机制测试通过：输入契约、接受、拒绝修正和追加 Token 均通过。`；TODO 未填时捕获 `NotImplementedError` 并提示"请先完成 TODO 代码"。
- `test_probability_contract`：把 `draft_probs[0,0]` 改成 -0.1，`speculative_decode_step` 必须抛 `ValueError`。
- `test_rejection_and_residual`：固定夹具返回 `tokens == [0, 3]`、`accepted_count == 1`、`rejected` 为真、`target_positions_checked == 2`。
- `test_all_accept_and_bonus`：`tokens == [0, 3, 2, 4]`、`accepted_count == 3`、`rejected` 为假。
- `test_zero_q_acceptance`：`accepted_count == 1` 且 `rejected` 为假。
- `test_zero_residual_failure`：残差质量为 0 时必须抛 `ValueError`。
- `test_probability_mass_recovery`：`accepted_mass + rejected_mass * residual` 与 `target` 在默认容差下 `assert_close`；而用 `target` 替代 residual 的版本必须不被 `allclose`。

Step 5 是 `RUN_MODE='dry_run'`，只加载配置、不跑真实生成；保存的 v2 结果（Median 4661.6→3634.8 ms、前向 192→64、显存 5.87→6.81 GiB、第三条提示 1 个 BF16 ULP 分歧）属于历史记录，复跑时以你自己 `benchmarks/results/inf_speculative_decoding_<RUN_LABEL>.json` 中的 `failure`、`decision` 和 `evidence_level` 为准。
