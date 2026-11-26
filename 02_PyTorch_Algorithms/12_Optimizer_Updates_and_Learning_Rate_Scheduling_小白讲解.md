# 12. 优化器更新与学习率调度 — 小白讲解

> 对应 notebook：[`12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb`](./12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)

## 一句话总结
这一节实现 `WSD_Scheduler`：学习率按"有效参数更新次数"走 Warmup–Stable–Decay 三段曲线。TODO 1 把 Warmup 的步号映射成 0 到 1 的线性进度，TODO 2 把 Decay 的步号映射成从 1 平滑降到 0 的余弦系数。配套测试重点不是背公式，而是确认三件事：阶段边界接得上、单步阶段不会除零、梯度累积的 micro-batch 不会偷推进调度时钟。

## 为什么需要它：痛点是什么
训练一开始，权重是随机的，梯度的方向和尺度的统计都不稳；如果直接用大学习率，损失可能先炸一下。训练中段希望保持足够的步幅，别太早衰减；临近结束又要把步子收小，才能收敛到更细的谷底。这就是 WSD 的三段：Warmup 升、Stable 平、Decay 降。

另一个更隐蔽的坑是"步数怎么数"。你以为 `for batch in loader` 每循环一次就是一步，但开了梯度累积后，4 个 micro-batch 才合成 1 次参数更新；混合精度还可能因为梯度溢出跳过某次 `optimizer.step()`。如果调度器按 micro-batch 计数，学习率会在参数还没更新时就提前走完，训练后半段实际用的是已经衰减到底的学习率。

生活类比：爬一座长坡。Warmup 是刚起步先慢慢走，避免一脚踩空；Stable 是中间保持匀速；Decay 是快到山顶时小步挪，防止冲过头。计时器只在你"真的迈出一步"时加一，原地整理背包（micro-batch 反向、被跳过的更新）不算。

## 核心问题
- 为什么学习率调度要按"成功的优化器步骤"推进，而不是按 micro-batch 或参数差值？
- Warmup 为什么除以 `num_warmup_steps - 1` 而不是 `num_warmup_steps`？单步 Warmup 为什么要特判？
- 余弦 Decay 的进度和系数怎么算？首尾为什么必须正好接上基础学习率和最小学习率？
- 学习率为 0 时参数没变，为什么优化器内部状态仍然前进了一步？
- `get_lr()` 里的 `step = self._step_count - 1` 是怎么和 `scheduler.step()` 对上的？
- 跳过一次更新后，调度器和优化器怎么保持同步、怎么从 checkpoint 恢复？

## 解决思路：一步步看懂

**第 1 步：调度时钟是什么。** notebook 的 Step 1 把一次训练循环拆成四个事件：梯度形成（不推进）、有效性判断（不推进）、优化器步骤实际执行（推进一步）、跳过更新（保持不变）。注意"执行了 `optimizer.step()`"不等于"参数一定变了"：学习率为 0 时参数不动，但 AdamW 仍会更新 `step` 计数和一阶/二阶矩。所以计步依据是"实际执行"，不是"参数前后是否相等"。按样本数或词元数调度是另一种约定，本节不采用。

**第 2 步：WSD 公式。** 设基础学习率 $\eta_{max}$、最小比例 $r$、Warmup/Stable/Decay 的有效更新数 $T_w,T_s,T_d$、当前成功更新编号 $t$，notebook 采用端点对齐的写法：

$$
\eta_t=
\begin{cases}
\eta_{max}\dfrac{t}{T_w-1}, & 0\le t<T_w,\\[4pt]
\eta_{max}, & T_w\le t<T_w+T_s,\\[4pt]
r\eta_{max}+(1-r)\eta_{max}\dfrac{1+\cos(\pi u_t)}{2}, & T_w+T_s\le t<T_w+T_s+T_d,
\end{cases}
$$

其中 $u_t=\dfrac{t-T_w-T_s}{T_d-1}$ 是 Decay 阶段内的进度。三个区间首尾相接：Warmup 的 $t=0$ 得 0、$t=T_w-1$ 得 $\eta_{max}$；Stable 全程 $\eta_{max}$；Decay 的 $u=0$ 时 $\cos(0)=1$，系数为 1，得 $\eta_{max}$，$u=1$ 时 $\cos(\pi)=-1$，系数为 0，得 $r\eta_{max}$。分母用 `-1` 是为了让"首步"和"末步"正好落在两个端点上；代价是单步阶段（$T=1$）分母为 0，必须特判。

**第 3 步：TODO 1 —— Warmup 进度。** 骨架已经处理了 `num_warmup_steps == 1` 的情况（直接返回 `base_lr`），也保证了 `step < num_warmup_steps`，所以参考答案只需：

```python
warmup_progress = step / (self.num_warmup_steps - 1)
return base_lr * warmup_progress
```

`test_warmup_and_stable` 用 `warmup=3, stable=4, max_lr=3e-4` 收集 7 个学习率，断言 `lrs[0] == 0.0`、`lrs[2] == 3e-4`（`2/(3-1)=1`）、`lrs[3:7]` 全等于 `3e-4`。手算三个 warmup 步：$0/2=0$、$1/2=0.5$、$2/2=1$，学习率分别是 0、1.5e-4、3e-4。单步 Warmup 用 `_collect_wsd_lrs(1, 0, 0, max_lr=2e-4)`，断言 `single[0] == 2e-4`——如果写成 `step / (T_w - 1)` 而不特判，就会 `0/0`。

**第 4 步：TODO 2 —— 余弦衰减。** 骨架已经把全局步换算成阶段内步：`decay_step = min(step - stable_end, num_decay_steps - 1)`，并特判了 `num_decay_steps == 1` 和 `== 0`。参考答案：

```python
decay_progress = decay_step / (self.num_decay_steps - 1)
cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
return min_lr + (base_lr - min_lr) * cosine_factor
```

`decay_progress` 从 0 到 1；`cos(π·progress)` 从 1 降到 −1；`0.5*(1+cos)` 就得到从 1 平滑降到 0 的系数；最后在 `min_lr` 和 `base_lr` 之间插值。拿 `test_cosine_decay` 的配置手算：`warmup=2, stable=2, decay=3, max_lr=3e-4, ratio=0.1`，decay 三步的 `decay_step` 是 0、1、2，进度 0、0.5、1，系数 1、0.5、0，学习率 3e-4、1.65e-4、3e-5。测试断言首步 `3e-4`、末步 `3e-5`、整个序列单调不升。单步 Decay 用 `_collect_wsd_lrs(0, 0, 1, max_lr=2e-4, min_lr_ratio=0.2)`，断言 `decay_only[0] == 4e-5`（`2e-4 × 0.2`）。

**第 5 步：`get_lr` 和 `_step_count` 怎么对上。** 父类 `LRScheduler.__init__` 会先执行一次 `step()`，把 `_step_count` 置 1，并设置初始学习率；之后训练循环每次 `scheduler.step()` 让 `_step_count` 加一。`get_lr` 里 `step = self._step_count - 1`，所以第一次真正被训练使用的是第 0 步的值（Warmup 的 0），而不是第 1 步。这个偏移是 PyTorch `LRScheduler` 的约定，写错成 `_step_count` 会让整条曲线右移一格。

**第 6 步：为什么 0 学习率也会推进优化器状态。** `test_zero_lr_skip_and_resume` 直接演示：`WSD_Scheduler(optimizer, 2, 2, 3)` 的第一个有效步学习率是 0；执行 `optimizer.step()` 后 `torch.equal(parameter, original)` 为真，但 `optimizer.state[parameter]["step"].item() == 1`。也就是说参数没动，AdamW 的步数、矩估计照样更新。这解释了为什么不能靠"参数有没有变化"来判断这一步该不该推进调度器——必须用执行事实或框架的跳步状态。

**第 7 步：跳步与恢复。** 测试里用 `skipped = True` 模拟"已决定跳过"，不调用 `optimizer.step()` 也不调用 `scheduler.step()`，断言 `(scheduler.last_epoch, lr)` 完全不变。然后把 `(optimizer.state_dict(), scheduler.state_dict())` 深拷贝，加载到一组新的参数/优化器/调度器上，连续 3 步比较参数和学习率都相等。这说明调度器的状态就是"成功更新计数"，存下来就能续训；如果跳步时偷偷调了 `scheduler.step()`，恢复后曲线就会错位。

**第 8 步：可选框架核对。** `check_scheduler_framework_conformance()` 用 PyTorch `LambdaLR` 表达同一条曲线：`factor(step)` 里 Warmup 是 `step/(warmup-1)`、Stable 是 1.0、Decay 是 `min_ratio + (1-min_ratio)*0.5*(1+cos(pi*progress))`，与 `WSD_Scheduler._lr_for_step(1.0, step)` 逐点比较 `range(7)`。然后模拟 `(True, False, True)` 三次尝试：只有成功的那次才 `optimizer.step()` + `scheduler.step()`。三个检查项是 `piecewise_curve_match`、`skipped_update_holds_lr`（跳步前后学习率严格相等）、`clock_counts_successful_updates`（`last_epoch == successful_updates`，这里是 2）。

**第 9 步：配置校验。** `WSD_Scheduler.__init__` 拒绝三类非法配置：阶段长度不是非负整数（包括布尔值和负数）、三段全为 0（`total_steps == 0`）、`min_lr_ratio` 不在 `[0,1]`。`test_wsd_input_contract` 用 `(-1,1,1,0.1)`、`(0,0,0,0.1)`、`(1,1,1,1.2)` 三组确认都抛 `ValueError`，错误信息是 `"invalid WSD schedule configuration"`。min 比例为 0 是合法的，表示最后可以降到 0；比例大于 1 会让"最小学习率"反而大于基础学习率，必须拒绝。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Warmup | 训练开头把学习率从 0 线性升到基础值 |
| Stable | 中段保持基础学习率不变 |
| Decay | 尾段把学习率平滑降到最小比例 |
| WSD | Warmup–Stable–Decay 三段式学习率曲线 |
| 有效更新 | 真正执行且未被跳过的 `optimizer.step()` |
| 梯度累积 | 多个 micro-batch 的梯度合成一次参数更新 |
| micro-batch | 一次前向/反向的小批次，不单独推进调度器 |
| `base_lrs` | 优化器每个参数组的初始学习率列表 |
| `_step_count` | PyTorch 调度器内部的步数计数器，`get_lr` 里减 1 使用 |
| 跳步 | 因梯度无效等原因不执行参数更新，调度也停住 |
| `min_lr_ratio` | 最终学习率相对基础学习率的比例，取值 0 到 1 |
| `LambdaLR` | PyTorch 里用函数表达学习率曲线的调度器 |

## 小白常见疑问
**Q1：Warmup 为什么除以 `T_w - 1`，除以 `T_w` 不是更自然吗？**
因为本节约定"首步为 0、末步正好等于基础学习率"。`T_w=3` 时，步号是 0、1、2；除以 `T_w-1=2` 才能让第 2 步到 1。若除以 3，末步只有 2/3 的基础学习率，Warmup 结束接 Stable 时会突然跳一下。`test_warmup_and_stable` 的 `lrs[2] == 3e-4` 就是钉这个端点。

**Q2：单步 Warmup / Decay 为什么要特判？**
因为 `T-1=0`，公式会 `0/0`。单步阶段没有"中间进度"可言，约定直接取端点：单步 Warmup 直接给 `base_lr`（骨架里 `if self.num_warmup_steps == 1: return base_lr`），单步 Decay 直接给 `min_lr`。`test_warmup_and_stable` 和 `test_cosine_decay` 分别检查了这两个特判。

**Q3：梯度累积时，为什么 micro-batch 反向不能推进调度器？**
因为参数还没更新。`test_scheduler_update_clock` 连续做两次 `loss.backward()`，不调用 `optimizer.step()`，断言学习率仍是初始值；然后一次 `optimizer.step()` + `scheduler.step()`，学习率变成 1.0（Warmup 两步的第二步，进度 `1/1=1`）。如果每次 backward 都推进，4 个 micro-batch 就会吃掉 4 个学习率步，实际更新次数只有 1 次。

**Q4：学习率为 0，参数没变，为什么还说"更新成功"？**
因为"成功"指的是优化器步骤真的执行了，不是参数一定变了。`test_zero_lr_skip_and_resume` 用 `AdamW` 演示：第一步 lr=0，参数保持 `[1.0]`，但 `optimizer.state[parameter]["step"] == 1`。调度器跟随的是执行计数，所以这一步仍然推进；真正被跳过的更新（比如 AMP 溢出）才两边都不调用。

**Q5：为什么 `get_lr` 要用 `_step_count - 1`？**
PyTorch 的 `LRScheduler` 在构造时先做一次初始 `step()`，让 `_step_count` 从 1 开始；训练循环里第一次 `scheduler.step()` 后又会加一。若直接用 `_step_count`，第一条学习率会取到"第 1 步"，整条曲线右移，Warmup 的 0 就被跳过。本节的 `step` 语义始终是"已成功更新次数减一"，与公式里的 $t$ 对齐。

## 动手验证：怎么知道自己做对了
`test_wsd_input_contract` 断言 `(-1,1,1,0.1)`、`(0,0,0,0.1)`、`(1,1,1,1.2)` 都抛 `ValueError`。`test_warmup_and_stable` 断言 `warmup=3, stable=4` 时 `lrs[0] == 0.0`、`lrs[2]` 与 `3e-4` 差小于 `1e-8`、`lrs[3:7]` 全为 `3e-4`，单步 Warmup 为 `2e-4`。`test_cosine_decay` 断言 `warmup=2, stable=2, decay=3` 的 decay 首步 `3e-4`、末步 `3e-5`、序列单调不升，单步 Decay 为 `4e-5`。`test_scheduler_update_clock` 断言两次 micro-batch 反向不改学习率、一次有效更新后学习率到 1.0。`test_zero_lr_skip_and_resume` 断言零学习率下参数不变但 AdamW `step == 1`、跳过时调度器与学习率都不动、保存/加载后继续 3 步参数与学习率完全一致。全部通过后打印 `✅ WSD 配置、阶段边界和有效更新计数通过。` notebook 没有保存执行输出；可选的 `check_scheduler_framework_conformance()` 返回 `piecewise_curve_match`、`skipped_update_holds_lr`、`clock_counts_successful_updates` 三个布尔值，要自己运行确认。
