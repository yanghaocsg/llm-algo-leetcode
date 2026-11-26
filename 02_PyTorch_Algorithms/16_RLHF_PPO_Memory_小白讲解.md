# 16. RLHF / PPO 的显存与状态 — 小白讲解

> 对应 notebook：[`16_RLHF_PPO_Memory.ipynb`](./16_RLHF_PPO_Memory.ipynb)

## 一句话总结
本节先把 PPO 的更新信号拆成 `reward`、`value`、`advantage`、`ratio` 和 `clip`，再实现 `compute_actor_loss` 的四个 TODO：策略比率、双侧裁剪、两种归约口径、更新诊断。随后用「四类模型角色 + 单步驻留表」解释 PPO 的峰值为什么不是四个模型参数量简单相加；可选的 TorchRL `ClipPPOLoss` 单元只核对裁剪目标与裁剪比例，不启动 rollout、奖励模型或 Critic 训练。

## 为什么需要它：痛点是什么
初学者常把 PPO 记成一个 loss 公式，或者反过来以为「PPO 有四类模型，显存就是 4 份权重」。这两种理解都会卡住：公式里为什么要 `detach`？裁剪到底截哪一边？`token_mean` 和 `sequence_mean` 有什么区别？工程上为什么采样、评分、更新要分阶段？

生活类比：拍一部电影。演员（Actor）会按反馈改表演；剧本顾问（Reference）不许改剧本，只提供「别偏太远」的参照；影评人（Reward Model）给成片打分；场记（Critic）预估每一场能得多少分。四类人都参与，但不等于要同时租四个摄影棚——可以分时进场、共享布景，真正占地方的是同一时刻还在场的演员、道具和未剪素材（激活、优化器状态）。

## 核心问题
- PPO 的 Actor、Reference、Reward、Critic 各自产出什么，哪些会被更新？
- KL 约束为什么既可以写进奖励、也可以做独立损失项，但不能两边同时算？
- `ratio` 怎么由新旧策略的 log-prob 差得到，裁剪对正、负 advantage 分别截哪一侧？
- `token_mean` 与 `sequence_mean` 在变长回答上为什么会给出不同 loss？
- 近似 KL 与裁剪比例怎么算、怎么读？
- PPO 的单步峰值由哪些「同时驻留」的状态决定，分阶段加载/卸载改变了什么？

## 解决思路：一步步看懂

**Step 1：先分清四类角色和两类记录。** notebook 的表把「模型角色」和「记录」分开：策略模型（Actor）产生回答和当前对数概率，会被更新；参考模型（Reference）给出固定参考对数概率，不更新；奖励模型或验证器给终态奖励，不更新；价值模型（Critic）估状态价值，通常更新。另有「行为策略记录」保存采样时的旧对数概率——它**不是第五个模型**，只是回答范围内的逐 token 记录；「有效回答范围」则是回答掩码与终止状态。这个区分直接决定了后面显存账本怎么记。

**Step 2：奖励怎样变成受约束的更新信号。** 公式逐个看：

$$r_t^{\text{train}} = r_t^{\text{task}} - \beta_{KL}(\log \pi_{actor}-\log \pi_{ref}), \qquad A_t \approx R_t - V(s_t)$$

$r_t^{\text{task}}$ 是任务奖励，$\beta_{KL}$ 控制 KL 惩罚强度，$\log\pi_{actor}-\log\pi_{ref}$ 衡量当前策略相对参考策略的偏移；$R_t$ 是回报，$V(s_t)$ 是 Critic 给的状态价值，两者相减得到优势 $A_t$。接着：

$$\rho_t = \exp(\log \pi_{new}-\log \pi_{old}), \qquad L_t^{clip}=\min \left(\rho_t A_t,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)$$

$\rho_t$ 是当前策略与采样时行为策略的概率比，$\epsilon$ 是裁剪范围。Critic 自己用

$$\mathcal L_V=\frac{1}{2}\mathbb E_t\left[(V_\phi(s_t)-R_t)^2\right]$$

拟合回报。若 KL 作为独立损失项，完整目标写成

$$\mathcal L_{PPO}=-\mathbb E_t[L_t^{clip}]+c_V\mathcal L_V-c_H\mathbb E_t[\mathcal H(\pi_\theta(\cdot\mid s_t))]+\beta_{KL}\mathbb E_t[D_{KL}(\pi_\theta\Vert\pi_{ref})]$$

notebook 强调：KL 写进奖励或写成独立项都行，但同一次实验必须固定口径，不能重复计入。

**Step 3：峰值来自「同时驻留」，不是参数量相加。** 采样阶段要留策略模型、KV Cache、回答和行为策略记录；评分阶段要读参考、奖励和价值模型；更新阶段还要保留策略模型的激活、梯度与优化器状态。因此分阶段加载或卸载能降低峰值，但会增加模型搬运与等待时间。这是本节标题里「Memory」的真正含义。

**Step 4：四个 TODO 怎么写。**

输入校验已经给出：三个张量形状必须一致、必须是二维非空、必须全为有限数；`clip_range` 必须是有限非负数；`response_mask` 为 `None` 时默认全 True，否则必须是同形布尔张量且**每条回复至少有一个有效位置**（否则按 token 求均值会除以 0）；`reduction` 只能是 `token_mean` 或 `sequence_mean`。

TODO 1 计算逐 token 概率比：

```python
ratio = torch.exp((log_probs_new - log_probs_old.detach()).masked_fill(~response_mask, 0.0))
```

`log_probs_old.detach()` 冻结采样策略，梯度只回到当前策略；`masked_fill` 在 `exp` **之前**把无效位置置 0，避免「无效位置的两个极小 log-prob 相差上万」导致 `exp` 溢出。测试 `test_masked_extreme_scores` 专门用 `-10000` 和 `-20000` 检查这一点。

TODO 2 构造保守目标：

```python
surrogate = ratio * advantages.detach()
clipped_surrogate = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages.detach()
conservative = torch.minimum(surrogate, clipped_surrogate)
```

为什么取 `min`？对正优势，比率涨到 `1.2` 以上时裁剪项更小，梯度被切断，防止一次更新迈太大；对负优势，比率跌到 `0.8` 以下时裁剪项更「悲观」，同样阻止策略把坏动作的概率压得过猛。手算测试里的两个边界：

- `ratio = exp(log(2)) = 2`、`advantage = 1`：`min(2, 1.2) = 1.2`，loss 为 `-1.2`；
- `ratio = exp(log(0.5)) = 0.5`、`advantage = -1`：`min(-0.5, -0.8) = -0.8`，loss 为 `0.8`。

TODO 3 按口径聚合，只统计有效回答 token：

```python
if reduction == "token_mean":
    loss = -conservative[mask].mean()
else:
    counts = mask.sum(dim=-1)
    per_sequence = (conservative * mask).sum(dim=-1) / counts
    loss = -per_sequence.mean()
```

`token_mean` 让长回答贡献更多 token；`sequence_mean` 先对每条回答内部求平均，再对回答等权。测试数据 `advantage = [[1,1],[3,0]]`、`mask = [[True,True],[True,False]]`、`ratio = 1` 时：token mean 是 `(1+1+3)/3 = 5/3`，loss `-5/3`；sequence mean 是两条回答各得 `1` 和 `3`，再平均得 `2`，loss `-2`。两种口径都合理，但必须显式选择。

TODO 4 记录诊断值：

```python
log_ratio = (log_probs_new.detach() - log_probs_old.detach()).masked_fill(~response_mask, 0.0)
approx_kl = ((ratio.detach() - 1.0) - log_ratio)[mask].mean()
clip_fraction = ((ratio.detach() - 1.0).abs() > clip_range)[mask].float().mean()
```

`(ρ−1) − log ρ` 在 ρ 接近 1 时约为 `(ρ−1)²/2`，是 KL 的一个二阶近似；它恒非负，ratio 全为 1 时等于 0（测试断言 `approx_kl == 0.0`）。`clip_fraction` 是「比率偏离 1 超过 ε 的有效 token 比例」，用来判断这一步有多少 token 被裁剪影响。所有诊断都 `detach`，只记录、不进入梯度目标。

**Step 5：可选的 TorchRL 核对。** `check_ppo_clip_objective` 用 `TensorDictModule(torch.nn.Identity(), ...)` 造一个哑 Actor，并继承 `ClipPPOLoss` 覆写 `_log_weight`，把受控的 `log_probs_new - log_probs_old` 注入框架，避免真的跑 rollout。核对三件事：有效 token 数一致、`loss_objective` 与本地实现 `torch.isclose(..., atol=1e-6, rtol=1e-6)`、`clip_fraction` 差值小于 `1e-6`。缺少 TorchRL 时返回 `status: skipped`，不算失败。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Actor | 策略模型，生成回答并接收策略梯度 |
| Reference | 冻结的参考模型，限制策略长期漂移 |
| Reward Model | 给回答质量打分，不参与梯度更新 |
| Critic | 价值模型，估计状态价值，给优势做基线 |
| 行为策略记录 | 采样时的旧对数概率，不是独立模型 |
| KL 约束 | 惩罚当前策略偏离参考策略，可进奖励或独立损失 |
| advantage | 回报减价值基线，表示「比预期好多少」 |
| ratio（ρ） | 新旧策略在同一 token 上的概率比 |
| clip | 把 ratio 限制在 `[1−ε, 1+ε]`，只改目标不改实际策略 |
| token_mean | 所有有效 token 等权平均 |
| sequence_mean | 每条回答先内部平均，再对回答等权 |
| approx_kl | `(ρ−1) − log ρ` 的有效 token 均值，诊断用 |

## 小白常见疑问

**Q1：为什么 `log_probs_old` 和 `advantages` 都要 `detach`？**
PPO 的一轮更新把采样时的旧策略和已经算好的优势当作固定常数。测试 `_assert_ppo_gradient` 直接断言 `new.grad is not None`，而 `old.grad is None`、`advantage.grad is None`；不 detach 就会把梯度错误地送回采样记录或优势计算。

**Q2：`masked_fill` 为什么必须在 `exp` 之前？**
因为 `exp` 对差距极大的 log-prob 会溢出。`test_masked_extreme_scores` 构造了 `new = [-1.0, -10000.0]`、`old = [-1.0, -20000.0]`，无效位置差值为 `10000`，先 `exp` 会得到 `inf`；先填 0 再 `exp` 得到 `1`，loss 保持有限。测试还断言该位置梯度为 0、`old.grad is None`。

**Q3：裁剪之后，实际 ratio 会留在 `[0.8, 1.2]` 里吗？**
不会。裁剪只改变损失函数对 ratio 的取值，不会把策略参数或实际概率比限制在区间内。notebook 的表格特意写了「不保证实际比率留在区间」，`clip_fraction` 就是用来观察有多少 token 越界的。

**Q4：两种 reduction 到底什么时候差最多？**
回答长度差异大时。测试里 token mean 得 `5/3`，sequence mean 得 `2`，就是因为第一条回答只有 1 个有效 token、第二条有 2 个。`token_mean` 会让长回答的 token 数主导梯度，`sequence_mean` 则避免长回答「人多势众」。

**Q5：KL 写进奖励和写成独立损失项，能同时用吗？**
不能重复计入。notebook 说两种实现都可以，但同一次实验必须固定口径。既在奖励里扣 KL、又在 loss 里加 KL 项，相当于把约束放大了一倍，学习率和 β 的调参也会失去可比性。

## 动手验证：怎么知道自己做对了
测试单元先跑 `test_masked_extreme_scores()`，再跑 `test_ppo_actor_loss()`；后者串起五个断言函数，全部通过后打印 `✅ PPO ratio、clipping、变长回复聚合、诊断值、梯度隔离和输入契约测试通过。`

- `_assert_ppo_ratio_and_loss`：用参考公式 `-torch.minimum(ratio*adv, clamp(ratio,0.8,1.2)*adv)[mask].mean()` 与实现比较，要求 `torch.allclose`（默认容差）；断言 `metrics["valid_tokens"] == int(mask.sum())`。
- `_assert_ppo_clipping`：正优势用例 loss 必须等于 `-1.2`，负优势用例必须等于 `0.8`，分别覆盖上、下裁剪边界。
- `_assert_ppo_reduction_and_metrics`：`token_mean` 得 `-5/3`，`sequence_mean` 得 `-2`；同时断言 `approx_kl == 0.0`、`clip_fraction == 0.0`。
- `_assert_ppo_gradient`：`new.grad` 有限，`old.grad is None`，`advantage.grad is None`。
- `_assert_ppo_input_contract`：五组非法输入必须抛 `ValueError`——形状不一致、空张量、含 `NaN`、掩码不是布尔、`clip_range=inf`。
- `test_masked_extreme_scores`：两种 reduction 下 loss 有限，`new.grad[0, 1] == 0`，`old.grad is None`。

测试函数用 `try/except` 把 `AttributeError`、`NameError`、`TypeError`、`ValueError` 和 `AssertionError` 重新包装成 `NotImplementedError("请先完成 TODO 部分的代码！")`，所以「测试报这个错」通常意味着 TODO 还没填或字段名写错。dump 中没有保存任何运行输出；结果以断言和你自己跑出的 metrics 为准。可选的 TorchRL 单元在缺少依赖时返回 `skipped`，不影响 CPU 主线。
