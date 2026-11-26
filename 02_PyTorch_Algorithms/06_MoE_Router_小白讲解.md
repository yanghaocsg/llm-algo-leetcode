# 06. MoE 路由器 — 小白讲解

> 对应 notebook：[`06_MoE_Router.ipynb`](./06_MoE_Router.ipynb)

## 一句话总结
本节用 `TopKRouter`、`build_capacity_plan` 和 `SparseMoEBlock` 把 MoE（Mixture of Experts，混合专家）的路由链一次跑通：先对全部专家算 Softmax 概率，再选 Top-K、把选中权重重归一化，最后按专家聚合 token 输出。四个 TODO 分别对应“完整概率、Top-K 选择、重归一化、容量计划”，测试会逐步检查概率和为 1、索引合法且不重复、权重和为 1，以及每个专家最多接纳 `capacity` 个分配。

## 为什么需要它：痛点是什么
稠密 Transformer 块里每个 token 都要经过同一组 MLP 参数：模型越大，参数越多，每个 token 的计算也越贵。可很多 token 并不需要“全套专家”一起上，只需要少数几条更对口的路径。MoE 的想法是：把 MLP 拆成多个专家，路由器只为每个 token 选少数几个，于是总参数容量可以变大，而每个 token 的实际计算仍由 Top-K 控制。

生活类比：快递分拣中心。每个包裹（token）先由分拣员（Router）看地址，打出一张“最可能去哪个格口”的分数表；只把包裹投进分数最高的 K 个格口（专家）。每个格口一轮最多处理 `capacity` 件（专家容量），超出就得记 overflow。包裹最后按各格口的分数加权合并，送回原来的传送带位置。

初学者常卡在三处：把“完整概率”和“Top-K 权重”混为一谈；以为选了 Top-K 就一定能全部执行，忘了容量会溢出；以及不知道为什么选完还要再归一化一次。

## 核心问题
- 路由分数经过什么变换才变成“每个 token 对全部专家的概率”？
- Top-K 的“值”和“索引”分别决定什么？为什么之后还要重归一化？
- 专家容量 `capacity` 怎样把“被选中”变成“被接纳”，溢出量怎么算？
- `SparseMoEBlock` 怎样用路由权重把多个专家的输出合并回原 token？
- 为什么完整概率要用 FP32 保存，而返回给模型前向的权重又要转回输入 dtype？

## 解决思路：一步步看懂

**第 1 步：先看清输入输出。** `TopKRouter.__init__` 保存 `num_experts`、`top_k`，并建一个无偏置线性层 `self.gate = nn.Linear(hidden_size, num_experts, bias=False)`。构造时就会检查 `0 < top_k <= num_experts`，非法配置直接抛 `ValueError`。
`forward` 要求输入是 `[batch, seq_len, hidden_size]` 的三维张量，先用 `reshape(-1, hidden_size)` 把 `B×T` 个 token 摊平成二维，再过一个 `gate` 得到 `router_logits`，形状 `[tokens, E]`。返回三个东西：完整概率 `routing_probs`、Top-K 重归一化权重 `routing_weights`、选中专家索引 `selected_experts`。

**第 2 步（TODO 1）：完整概率覆盖全部专家。** 参考答案只有一行：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)  # 在专家维归一化
```

对第 $i$ 个 token、第 $e$ 个专家，路由分数记为 $z_{i,e}$，则

$$
p_{i,e}=\frac{\exp(z_{i,e})}{\sum_{j=1}^{E}\exp(z_{i,j})},
\qquad \sum_{e=1}^{E}p_{i,e}=1.
$$

`dim=-1` 表示在“专家”这一维做 Softmax，所以每一行（每个 token）的概率和都是 1。先 `.float()` 是为了用 FP32 保存统计口径：后面即使把权重转回 BF16/FP16 去算，这份完整概率仍是稳定的 FP32，测试 `test_full_probability_distribution` 也直接断言 `probs.dtype == torch.float32`。
容易写错成 `dim=0` 或 `dim=1`：`[tokens, E]` 只有两维，`dim=-1` 才是专家维。

**第 3 步（TODO 2）：选 Top-K 的权重和索引。** 参考答案：

```python
routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
```

`torch.topk` 同时返回两样东西：每行最大的 K 个概率值（形状 `[tokens, K]`）和它们的位置编号（同样 `[tokens, K]`，取值在 `[0, E)`）。概率是单调变换后的分数，所以按概率取 Top-K 与按 logits 取 Top-K 选出的专家一致；但概率更容易做后续归一化和统计。
测试 `test_topk_contract_and_bounds` 检查三件事：形状必须是 `(6, 2)`（`2×3=6` 个 token、`K=2`）；索引都落在 `[0, 5)`；同一行的两个索引互不相同。`torch.topk` 本身保证不重复，不需要额外去重。

**第 4 步（TODO 3）：重归一化。** 完整概率的和是 1，但只截取 K 个之后，和通常小于 1。若不处理，专家输出的加权和会整体偏小，K 越小、分布越平，缩得越厉害。参考答案：

```python
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

即对 token $i$，设选中的专家编号为 $k_1,\dots,k_K$，则

$$
w_{i,k}=\frac{p_{i,k}}{\sum_{j=1}^{K}p_{i,k_j}},
\qquad \sum_{j=1}^{K}w_{i,k_j}=1.
$$

`keepdim=True` 让分母保持 `[tokens, 1]`，才能和 `[tokens, K]` 逐行相除。`test_renormalized_weights_match_selected_probability` 用 `probs.gather(dim=-1, index=indices)` 取出选中概率，再除以行和，要求与 `weights` 完全一致，并检查每行和为 1。
最后 `return` 时 `routing_weights.to(hidden_states.dtype)`：权重是要参与前向乘法的，跟随输入精度即可；完整概率则保留 FP32 供统计。

**第 5 步（TODO 4）：把“选中”变成“可执行”的容量计划。** `build_capacity_plan(selected_experts, num_experts, capacity)` 的输入是二维 `torch.long` 索引表 `[tokens, K]`。先做输入校验：维度、dtype、`num_experts`/`capacity` 为正、索引不越界。然后：

```python
expert_load = torch.bincount(selected_experts.flatten(), minlength=num_experts)
```

`bincount` 统计每个专家被选了多少次，得到长度为 `E` 的负载向量。接着逐个专家处理：

```python
expert_positions = torch.nonzero(selected_experts == expert_idx, as_tuple=False)
accepted_positions = expert_positions[:capacity]
```

`torch.nonzero(..., as_tuple=False)` 返回所有满足条件的位置坐标 `[行, 列]`。`nonzero` 按行优先的稳定顺序输出，所以 `[:capacity]` 表示“按稳定顺序接纳前 `capacity` 个分配”。把接纳位置写进布尔掩码 `accepted_mask`，最后算

$$
\text{overflow}_e=\max(0,\ \text{load}_e-\text{capacity}).
$$

这里有两个容易写错的地方。第一，不能用 `selected_experts == expert_idx` 直接当掩码，因为它是二维的；必须用 `nonzero` 定位到具体的 `(token, 第几个选择)`。第二，不要打乱顺序后再截断，测试要求的是稳定顺序。用测试里的例子手算：`selected = [[0,1],[0,1],[0,2]]`，三个 token 各选 2 个专家，共 6 个分配。`load = [3, 2, 1]`，`capacity=2` 时专家 0 接纳前两个、第三个被丢弃，所以 `accepted[:, 0] == [True, True, False]`，总接纳 5 个，`overflow = [1, 0, 0]`。

**第 6 步：`SparseMoEBlock` 怎样合并。** 它持有 `self.router = TopKRouter(...)` 和 `self.experts = nn.ModuleList([nn.Linear(h, h) ...])`。前向时对每个专家：

```python
token_idx, kth_expert = torch.where(selected_experts == expert_idx)
expert_output = expert(flat_hidden_states[token_idx])
expert_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
final_hidden_states[token_idx] += expert_output * expert_weight
```

`torch.where` 找出“哪些 token 把该专家选进了第几个位置”，然后只对这些 token 调用对应专家，再乘上路由权重累加。一个 token 被 K 个专家选中，就会收到 K 份加权输出，最后恢复成 `[B, T, h]`。
注意：这个演示块**没有调用** `build_capacity_plan`，所有被选中的 token 都会执行；容量计划是独立的审计工具，用来说明真实系统为什么还需要“丢弃/重路由/排队”。理解这一点，就不会误以为测试里的 overflow 会改变前向结果。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MoE | 把 MLP 拆成多个专家，每个 token 只走少数几条 |
| Router | 给每个 token 对全部专家打分的线性层 |
| 路由分数 / logits | 打分前的原始值 `[tokens, E]` |
| 完整概率 | 对全部专家做 Softmax 后的分布，每行和为 1 |
| Top-K | 每个 token 保留分数最高的 K 个专家 |
| 重归一化 | 把截取后的 K 个权重再除以它们的和，使总和回到 1 |
| 专家容量 `capacity` | 一个专家本轮最多接纳多少个分配 |
| overflow | 负载超过容量的部分，`max(0, load - capacity)` |
| 稳定顺序 | `torch.nonzero` 按行优先返回的确定顺序 |
| `torch.bincount` | 统计每个整数出现了多少次，用来算专家负载 |
| 稀疏激活 | 总参数很多，但每个 token 只算其中 K 个专家 |

## 小白常见疑问
**Q1：既然最后要把权重转回输入 dtype，为什么 TODO 1 还要 `.float()`？**
因为完整概率 `routing_probs` 是返回给外部做批次统计和负载分析的，测试明确要求它是 `torch.float32`；而 `routing_weights` 只参与前向乘法，转回 `hidden_states.dtype` 可以减少低精度训练时的类型摩擦。两个量的用途不同。

**Q2：Top-K 之后再 Softmax 一次，和“先 Softmax 再取 Top-K 再除行和”等价吗？**
在这个设定下数值上很接近：对选中的 K 个 logits 重新做 Softmax，等价于把原概率在 K 个专家上重新归一化。notebook 选择后者，因为要保留“完整概率”这个统计量，并且测试 `test_renormalized_weights_match_selected_probability` 直接按“选中概率 ÷ 行和”核对。

**Q3：`test_capacity_plan_tracks_acceptance_and_overflow` 里为什么总接纳是 5 而不是 6？**
6 个分配中专家 0 出现了 3 次，容量只有 2，多出的 1 个被记为 overflow，所以接纳 `2+2+1=5`。`accepted[:, 0]` 只看每个 token 的第一个选择，第三个 token 的第一选择是专家 0，但它在专家 0 的位置队列里排第三，因此被拒。

**Q4：`SparseMoEBlock` 里把专家权重设成单位阵，为什么输出等于输入？**
每个 token 的 Top-K 权重和为 1，每个专家都是恒等映射，所以 $\sum_i w_i x=x\sum_i w_i=x$。`test_sparse_aggregation_uses_router_weights` 正是用这个性质检查“加权合并有没有漏乘或错位”，容差 `atol=1e-5`。

**Q5：为什么非法 `top_k` 要在构造时抛错，而不是等到 forward？**
`TopKRouter.__init__` 的检查 `not 0 < top_k <= num_experts` 让配置错误尽早暴露；`test_router_config_contract` 用 `top_k=0` 和 `top_k=6`（专家数 5）验证两种越界都会被 `ValueError` 拒绝。

## 动手验证：怎么知道自己做对了
运行 `run_topk_router_tests()`，它会依次调用 6 个测试，全部通过后打印 `✅ Top-K Router：概率、选择、容量计划与聚合测试通过`。如果 TODO 没填，会先抛 `NotImplementedError` 并提示“请先完成 TODO 1–4”。
- `test_full_probability_distribution`：`TopKRouter(hidden_size=4, num_experts=5, top_k=2)` 输入 `torch.randn(2, 3, 4)`，要求 `probs.shape == (6, 5)`、`dtype == torch.float32`、每行和为 1。
- `test_topk_contract_and_bounds`：`weights.shape == indices.shape == (6, 2)`，索引都在 `[0, 5)`，同一行排序后相邻值不相等。
- `test_renormalized_weights_match_selected_probability`：`weights` 与“选中概率除以行和”一致，且每行和为 1。
- `test_capacity_plan_tracks_acceptance_and_overflow`：上面的 `load=[3,2,1]`、`overflow=[1,0,0]`、`accepted[:,0]=[True,True,False]`、总接纳 5。
- `test_sparse_aggregation_uses_router_weights`：把 5 个专家都设成单位阵后，输出与输入 `allclose(atol=1e-5)`。
- `test_router_config_contract`：`top_k` 为 0 或 6 时必须抛 `ValueError`。

notebook 没有保存执行输出，也没有 GPU 实验；以这些断言和你自己运行时打印的那行通过信息为准。
