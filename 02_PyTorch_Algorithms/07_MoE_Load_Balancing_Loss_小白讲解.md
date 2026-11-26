# 07. MoE 负载均衡损失 — 小白讲解

> 对应 notebook：[`07_MoE_Load_Balancing_Loss.ipynb`](./07_MoE_Load_Balancing_Loss.ipynb)

## 一句话总结
MoE 的 Top-K 路由器只保证“每个 token 选 K 个专家”，并不保证一个批次里的 token 会平均分给所有专家。本节用 `compute_load_balancing_loss` 把每个专家的平均路由概率 `P_i` 与实际分配比例 `f_i` 放进辅助损失 `alpha * E * Σ f_i P_i`，再用 `update_router_bias` 演示另一条不进入主损失的反馈路径：按负载误差调整下一轮专家选择偏置。四个 TODO 分别完成偏好统计、分配统计、辅助损失和无辅助偏置更新。

## 为什么需要它：痛点是什么
初学者常问：“路由器不是已经按概率选专家了吗，为什么还会不均衡？”因为概率是“软偏好”，Top-K 是“硬选择”，两者可以不一致；而且训练会自我强化——某个专家早期多拿样本、学得更好、又被选得更多，最后少数专家过载，其余长期闲置。MoE 名义上有很多参数，实际只用了几个，容量和并行收益都打折。

生活类比：食堂 8 个窗口，每人选 2 个。热门窗口一旦形成就会越来越挤，最后 2 个排长队、6 个没人。负载均衡就是给窗口装动态叫号牌：排队长的号码往后调，没人去的往前调。

## 核心问题
- `P_i`（平均路由概率）和 `f_i`（实际被 Top-K 选中的比例）分别怎么统计？为什么两者都要看？
- 辅助损失为什么写成 `alpha * E * Σ f_i P_i`？均匀路由时为什么恰好等于 `alpha`？
- 无辅助的偏置更新怎样做到“过载专家降偏置、欠载专家升偏置”，且均衡时不变化？
- 辅助损失和无辅助均衡各自把反馈送到哪里？为什么容量回退不能替代长期均衡？
- 损失下降是否等于路由语义变好？读实验时应该同时看哪些量？

## 解决思路：一步步看懂

**Step 1：先分清“偏好”和“实际分配”。** notebook 用三层观察拆开问题：单个 token 看它偏好哪些专家；一个 batch 看 `P_i`、`f_i` 和专家计数；多个 step 看热点是否持续。`P_i` 来自完整 softmax 概率，回答“路由器平均想把 token 送给谁”；`f_i` 来自 Top-K 索引计数，回答“实际送给了谁”。只看一项会漏掉两类故障：概率均匀但 argmax 集中；分配尚可但某个专家容量不够，执行时仍溢出。

**Step 2：写清三个统计量的公式。** 设 batch 有 $T$ 个 token、每个 token 选 $K$ 个专家、共 $E$ 个专家，$p_{t,i}$ 是第 $t$ 个 token 分给专家 $i$ 的概率：

$$
P_i=\frac{1}{T}\sum_{t=1}^{T}p_{t,i},\qquad
f_i=\frac{1}{TK}\sum_{t=1}^{T}\mathbf{1}\left[i\in\mathrm{TopK}(p_t)\right],\qquad
L_{\text{balance}}=\alpha E\sum_{i=1}^{E}f_iP_i.
$$

逐项解释：$P_i$ 是“所有 token 对专家 $i$ 的平均偏好”，因为每行概率和为 1，所以 $\sum_i P_i=1$。$f_i$ 是“专家 $i$ 拿到的分配次数 ÷ 总分配次数 $TK$”，所以 $\sum_i f_i=1$。注意分母是 $TK$ 不是 $T$：每个 token 贡献 K 次分配。$L_{\text{balance}}$ 把两个分布逐项相乘再求和；$\alpha$ 是辅助损失系数，$E$ 是把基线缩放到与专家数无关的尺度。

为什么乘 $E$？如果所有专家完全均匀，$P_i=f_i=1/E$，于是 $\sum_i f_iP_i=E\cdot(1/E^2)=1/E$，再乘 $E$ 得 1，损失正好是 $\alpha$。这样不同专家数的模型有一个共同基线，测试 `test_uniform_routing_statistics` 才敢直接断言 `loss == alpha`。反过来，如果路由集中到 2 个专家，比如 `probs[:, :2]=0.5`、其余为 0，且 Top-K 总是 `[0,1]`，则 $f_0=f_1=0.5$、$P_0=P_1=0.5$，$\sum f_iP_i=0.5$，损失是 $0.01\times8\times0.5=0.04$，正好是均匀基线 0.01 的 4 倍，满足 `concentrated > uniform * 2`。

**Step 3：TODO 1——统计平均偏好 `P_i`。** 参考答案只有一行：

```python
P_i = router_probs.mean(dim=0)   # [T, E] 沿 token 维平均 -> [E]
```

`router_probs` 的形状是 `[total_tokens, num_experts]`。要得到“每个专家在整个 batch 的平均概率”，必须沿第 0 维（token 维）求均值。容易写错成 `mean(dim=1)`，那会得到每个 token 的平均概率、形状是 `[T]`；或者写成 `sum(dim=0)`，虽然方向对但没除以 $T$，`P_i` 之和会变成 $T$ 而不是 1。

**Step 4：TODO 2——统计实际分配 `f_i`。** 参考答案三步：

```python
expert_mask = F.one_hot(selected_experts, num_classes=num_experts)  # [T, K, E]
tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()             # [E]
f_i = tokens_per_expert / (total_tokens * top_k)                    # 归一化
```

`selected_experts` 是 `[T, K]` 的 `torch.long`，`one_hot` 把它变成 `[T, K, E]` 的 0/1 张量；对前两维求和，就得到每个专家被选中多少次。除以 `total_tokens * top_k` 后，`f_i` 之和为 1。这里最容易错的是把 `f_i` 也当成概率质量：如果直接用 `P_i` 代替 `f_i`，`test_preference_and_dispatch_can_differ` 就抓不到“概率均匀但分配集中”的情况——该测试固定 `probs` 全为 `1/8`，却让 Top-K 永远是 `[0,1]`，要求 `f_i[:2] == [0.5, 0.5]`、其余为 0，而 `P_i` 仍然全是 `1/8`。另一个常见错误是只对 `dim=0` 求和，得到 `[K, E]`，后续乘 `P_i` 时广播会悄悄算错。

**Step 5：TODO 3——计算辅助损失。** 参考答案：

```python
aux_loss = alpha * num_experts * (f_i * P_i).sum()
```

顺序是“逐专家相乘再求和”，而不是先求和再相乘。`f_i` 和 `P_i` 都是 `[E]`，逐元素相乘得到每个专家对“偏好 × 分配”的贡献。`num_experts` 就是公式里的 $E$。如果把 `sum` 换成 `mean`，均匀基线会变成 `alpha / E`，测试里的 `torch.allclose(loss, torch.tensor(alpha))` 就会失败。`return_stats=True` 时返回 `(aux_loss, P_i, f_i)`，否则只返回损失。

**Step 6：TODO 4——无辅助的偏置更新。** 这条路径不把损失加进主目标，而是根据本轮的专家计数直接改“下一轮选择时给每个专家加的分”。参考答案：

```python
mean_load = expert_counts.float().mean()
load_error = (expert_counts.float() - mean_load) / mean_load.clamp_min(1.0)
updated_bias = router_bias - step_size * load_error.to(router_bias.dtype)
```

`load_error` 是“相对平均负载的偏差”：高于平均为正，低于平均为负。`mean_load.clamp_min(1.0)` 防止全零批次除以 0。更新用减号：过载专家 `load_error > 0`，偏置被调低；欠载专家 `load_error < 0`，偏置被调高，下一轮更容易被选中。函数返回 `(updated_bias, load_error)`。测试里 `counts=[8,4,2,2]`，均值是 4，误差为 `[1,0,-0.5,-0.5]`，`step_size=0.1` 时偏置变成 `[-0.1,0,0.05,0.05]`；而 `counts=[1,1,1,1]` 时误差全 0，偏置保持不变。常见错误是把减号写成加号（越忙越加，正反馈），或者除以计数总和而不是均值（误差尺度随 batch 大小漂移）。

**Step 7：两条反馈路径怎么选。** 辅助损失进入“训练总损失与梯度”，机制直接、统计清晰，但系数过大会干扰主任务；无辅助均衡进入“路由偏置或控制状态”，不改主损失，但引入额外状态、可能振荡；容量回退只在执行阶段重路由或丢弃，保护单步跑完，不能替代长期均衡。读结果时要把三者分开：辅助损失下降只说明“偏好和分配更均匀”，不说明“被选中的专家语义上更合适”。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MoE（混合专家） | 用路由器给每个 token 选几个专家网络，而不是全部参数都参与 |
| Top-K 路由 | 每个 token 只保留概率最高的 K 个专家 |
| `P_i` | 路由器对整个 batch 给专家 $i$ 的平均概率，是“软偏好” |
| `f_i` | 专家 $i$ 实际被 Top-K 选中的次数占总分配数的比例，是“硬分配” |
| 辅助损失（aux loss） | 额外加进训练目标、用来惩罚路由集中的一项损失 |
| `alpha` | 辅助损失的系数，越大越强调均衡、越可能干扰主任务 |
| 无辅助均衡 | 不改主损失，只根据历史负载调整专家选择偏置 |
| `load_error` | 专家计数相对平均负载的偏差 |
| 容量溢出 | 某个专家收到的 token 超过它能处理的上限，需要丢弃或重路由 |
| 路由偏置 | 下一轮选择前加到各专家分数上的修正量 |
| 专家并行 | 把不同专家放到不同设备上，热点会造成跨卡通信长尾 |

## 小白常见疑问
**Q1：辅助损失为什么一定要乘专家数 $E$？不乘不行吗？**
不乘也能惩罚集中，但基线会变成 `alpha / E`，不同专家数的模型之间没法直接用同一个阈值比较。乘 $E$ 后均匀路由的损失恒为 `alpha`，`test_uniform_routing_statistics` 才能用 `torch.allclose(loss, torch.tensor(alpha))` 做精确断言。

**Q2：`P_i` 已经很均匀了，还有必要看 `f_i` 吗？**
有必要，`test_preference_and_dispatch_can_differ` 就是反例：`probs` 全为 `1/8`，但 Top-K 永远选 `[0,1]`，此时 `P_i` 均匀、`f_i` 却是 `[0.5,0.5,0,...]`。只看概率会误以为路由很健康。

**Q3：偏置更新为什么不放进主损失？**
放进主损失会直接改梯度，系数大时干扰语言建模目标；偏置更新只改“选择倾向”。代价是要保存 `router_bias`、确定更新节奏，并监测它是否振荡——notebook 明确把“偏置更新可能振荡”列为风险。

**Q4：`load_error` 为什么除以平均负载，而不是总和？**
除以平均负载得到“相对平均值的比例”：8 对均值 4 是 +100%，2 是 -50%，不同 batch 规模下尺度一致。除以总和会随 batch 内 token 数变化，步长不再稳定。

**Q5：辅助损失越小，模型越好吗？**
不是。它只衡量负载是否均匀，不衡量专家是否学到有用分工。notebook 特意提醒“均衡不等于语义路由正确”；还要同时看容量溢出、主任务损失和路由稳定性。

## 动手验证：怎么知道自己做对了
测试入口是 `run_load_balance_tests()`，依次运行五个测试，全过才打印 `✅ 负载均衡：辅助损失、偏置反馈与输入契约测试通过`。TODO 没填时会先抛 `NotImplementedError`，外层 `try` 捕获后打印“请先完成 TODO 1–4，再运行测试”。
- `test_uniform_routing_statistics`：8 专家、Top-K=2、80 token，`probs` 全为 `1/8`；断言 `P_i`、`f_i` 都等于 `1/8`，损失等于 `alpha=0.01`。
- `test_concentrated_routing_has_larger_penalty`：token 都选 `[0,1]`、`probs[:, :2]=0.5`；断言集中损失大于均匀基线的 2 倍（0.04 对 0.01）。
- `test_preference_and_dispatch_can_differ`：概率均匀、分配集中；断言 `P_i` 仍均匀、`f_i[:2]` 为 `[0.5,0.5]`、其余为 0。
- `test_bias_feedback_moves_away_from_overloaded_experts`：`counts=[8,4,2,2]` 时 `error[0] > 0` 且 `updated[0] < 0`、`error[2] < 0` 且 `updated[2] > 0`；全 1 时偏置和误差都不变。
- `test_load_balance_input_contract`：专家维不匹配、`selected_experts` 非 `torch.long`、索引越界、`top_k=0` 四组非法输入都必须抛 `ValueError`。

notebook 没有保存执行输出，无法引用实际打印数值；以代码和断言为准。想额外观察，可把 `alpha` 从 0.01 调到 0.1 重跑集中路由：损失按比例放大，但 `P_i`、`f_i` 不变，说明 `alpha` 只缩放惩罚、不改变统计量。
