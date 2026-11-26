# DISTILL-OPD. 在策略蒸馏与 GKD — 小白讲解

> 对应 notebook：[`DISTILL-OPD_On_Policy_Distillation_and_GKD.ipynb`](./DISTILL-OPD_On_Policy_Distillation_and_GKD.ipynb)

## 一句话总结

普通蒸馏只让学生看"标准答案"，学生自己走错路之后遇到的前缀从来没被训练过。在策略蒸馏（on-policy distillation）改成：当前学生先自己生成回答，再把这条轨迹的每个前缀送给冻结教师，让教师给出"下一个词元应该是什么分布"，学生按分布差异更新。本节用四个 TODO 把 GKD 的四个动作拆开实现——按 $\lambda$ 分配轨迹来源、按 $\beta$ 计算广义 JSD、从序列边界构造完成区掩码、按有效 token 数归约损失——并用测试锁定端点 KL、掩码排除和教师/学生梯度隔离。

## 为什么需要它：痛点是什么

SFT 的监督是固定回答：数据里写什么，学生就学什么。问题在于推理时学生会自己生成，一旦它走到一个"数据里没出现过的前缀"，后面的行为就没有被训练过。这就像一个学生只背过标准答案，考试时第一步写错，后面整道题都不会了。

在策略蒸馏换了个思路：让学生先做一遍，教师在旁边逐步骤点评。学生写出的每一步（哪怕写错了）都构成一个真实前缀，教师在这些前缀上给出"如果是我会怎么选下一个词"的完整概率分布。学生不只是知道"正确答案是 A"，还知道"在这个位置上，B 和 C 各应该有多大把握"。生活类比：学开车时，教练坐在副驾看你实际操作，在你即将打错方向盘的那一刻告诉你"这里应该轻一点"；而不是只在事后给你一份标准路线图。

GKD（Generalized Knowledge Distillation）进一步把两件事拆开：**轨迹由谁生成**（$\lambda$）和**分布怎么比较**（$\beta$）。这两件事经常被混为一谈，notebook 的 Step 2 明确说"$\lambda$ 选择回答来源，$\beta$ 选择该位置的分布比较方式，两者各控制一件事"。把这两个旋钮分开，才能分别回答"学生该在哪些前缀上练"和"在这些前缀上该向教师靠多近"。

## 核心问题

- 一个蒸馏批次里，哪些回答由当前学生生成、哪些来自固定数据？$\lambda=0.5$ 且批次为 3 时到底选几条？
- 广义 JSD 怎么由教师分布、学生分布和混合分布组成？$\beta=0$ 和 $\beta=1$ 为什么分别变成两个方向的 KL？
- 为什么"完成区掩码"要同时排除 prompt、padding 和工具返回？只减掉 prompt 长度不够吗？
- 教师为什么要 `detach`，而学生在混合分布里的那一部分为什么还能拿到梯度？
- 损失为什么除以有效 token 数，而不是序列长度或 batch 大小？
- 这套实现和"把教师 logits 存下来做离线蒸馏"有什么区别？

## 解决思路：一步步看懂

### Step 1：轨迹来源——$\lambda$ 管什么

设输入为 $x$、完整回答为 $y$，固定轨迹分布为 $\mu_{\mathrm{off}}(y\mid x)$，当前学生策略为 $\pi_S(y\mid x)$，在策略比例为 $\lambda\in[0,1]$。蒸馏批次的轨迹分布是：

$$\mu_{\lambda}(y\mid x)=(1-\lambda)\mu_{\mathrm{off}}(y\mid x)+\lambda\pi_S(y\mid x)$$

$\lambda=0$ 表示整批都用固定轨迹（离线），$\lambda=1$ 表示整批都由当前学生生成（在策略）。注意这个比例描述的是"整批回答的来源"，不是"某个 token 怎么算"；它和 $\beta$ 是正交的两个旋钮。

工程上还有一个取整问题：小批次无法精确取一半。notebook 说"3 条回答取一半时只能选 1 条或 2 条，并记录采用的取整规则"。代码用的是"加 0.5 后取整"：

```python
on_policy_count = int(batch_size * on_policy_fraction + 0.5)
```

对 3 条、$\lambda=0.5$：$3\times0.5+0.5=2.0$，取 2 条；对 5 条、$\lambda=0.3$：$1.5+0.5=2.0$，取 2 条。这是四舍五入到最近整数（.5 向上），不是向下取整。

要特别强调：`build_on_policy_mask` **只分配"这条轨迹算不算学生生成"的标记，不负责真正生成回答**。真实采样在训练循环里完成（学生前向、按采样参数生成 token），notebook 的 TODO 1 只做"来源分配"这一层。Step 1 的表格要求记录学生版本、采样参数和轨迹身份，原因是：如果一批轨迹是用旧版学生采的，而更新时学生已经变了，那么"在策略"就名不副实，梯度会偏向旧策略访问过的前缀。这个"策略新鲜度"问题在预读的 `PT-ROLLOUT-LIFECYCLE` 里有专门讨论。

### Step 2：分布比较——$\beta$ 管什么

同一条轨迹的同一个前缀，分别送进冻结教师和可训练学生，得到 logits，再经 softmax 得到下一词元概率 $p_T$ 和 $p_S$。两者的每一维必须指向同一个词元，否则比较没有意义。

先形成混合分布：

$$m_{\beta}=\beta p_T+(1-\beta)p_S$$

再分别计算两个分布与混合分布的 KL，并按 $\beta$ 加权：

$$D_{\mathrm{GJS}}^{\beta}(p_T,p_S)=\beta D_{\mathrm{KL}}(p_T\Vert m_{\beta})+(1-\beta)D_{\mathrm{KL}}(p_S\Vert m_{\beta}),\quad 0<\beta<1$$

$\beta$ 的读法：

| 参数取值 | 对应计算 | 解释 |
|---|---|---|
| $\beta=0$ | $D_{\mathrm{KL}}(p_T\Vert p_S)$ | 以教师分布为参照惩罚学生差异（前向 KL） |
| $0<\beta<1$ | 上述加权 JSD | 教师、学生分别与混合分布比较 |
| $\beta=1$ | $D_{\mathrm{KL}}(p_S\Vert p_T)$ | 按学生分布衡量与教师的差异（反向 KL） |

这里有一个容易被忽略的数学细节：如果把上面的加权式直接取 $\beta\to0$，混合分布趋近学生分布，第一项被 $\beta$ 乘成 0，第二项是 $1\cdot D_{\mathrm{KL}}(p_S\Vert p_S)=0$，整体趋近 0；$\beta\to1$ 同理也趋近 0。也就是说，**内部加权式在端点附近趋向零，端点 KL 是单独选择的目标**，不是内部公式的连续极限。所以代码里必须写两个独立的 `if` 分支，不能只靠通式。

两个 KL 的方向也不同。$D_{\mathrm{KL}}(p_T\Vert p_S)$ 以教师为参照，学生只要在教师有概率质量的地方给 0，就会被重罚；$D_{\mathrm{KL}}(p_S\Vert p_T)$ 以学生为参照，学生更关注自己已经分配的众数区域。notebook 没有展开"mean-seeking / mode-seeking"这类术语，只要求你能从公式看出"方向改变时对概率差异的惩罚也会改变"。

拿一个两词表的小例子手算：教师 $p_T=(0.9,0.1)$，学生 $p_S=(0.5,0.5)$。前向 KL 是

$$0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}\approx0.3681$$

反向 KL 是

$$0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}\approx0.5108$$

同一对分布，换个方向数值就不同（这里反向更大），因为两项的权重从教师概率换成了学生概率。这就是"方向不是显示单位"的具体含义。

广义 JSD 还有一个对称性：把 $p_T$ 与 $p_S$ 互换、同时把 $\beta$ 换成 $1-\beta$，两项 KL 和权重会对调，结果不变。所以 $\beta=0.5$ 时它对教师和学生是对称的；$\beta\neq0.5$ 时它偏向某一方。notebook 把 $\beta$ 当作"分布比较方式"的旋钮，原因就在这里。

### Step 3：损失——掩码与归约

完成区（completion）指回答所占的序列位置；padding 是补齐不同长度序列用的填充。取值为 0 或 1 的掩码 $M_{b,t}$ 在批次 $b$、位置 $t$ 标记是否计算蒸馏目标：prompt、padding 以及指定排除的工具返回取 0，保留的回答位置取 1。对齐后的教师、学生分布记作 $p^T_{b,t}$、$p^S_{b,t}$，损失为：

$$\mathcal{L}_{\mathrm{distill}}=\frac{\sum_{b,t}M_{b,t}D_{\mathrm{GJS}}^{\beta}(p^T_{b,t},p^S_{b,t})}{\sum_{b,t}M_{b,t}}$$

分母是有效位置数，不是 $B\times T$。只保留 3 个回答 token 时就除以 3，而不是整段输入长度——否则 padding 越多，损失被稀释得越厉害。

Step 3 还列了四条数值约束：有效位置数为零时应停止本批计算；教师分布停止梯度，学生分布及其在混合分布中的部分参与求导；已经采样出的离散词元作为固定输入；输入必须有限，因为"掩码按位置选择损失，NaN 与零相乘仍会得到 NaN"。

### Step 4：TODO 1——`build_on_policy_mask`

输入是 `batch_size`、`on_policy_fraction`、`seed`，输出一个长度为 `batch_size` 的布尔张量，True 表示这条轨迹由当前学生生成。

参考答案：

```python
if batch_size <= 0:
    raise ValueError("batch_size 必须为正整数")
if not 0.0 <= on_policy_fraction <= 1.0:
    raise ValueError("on_policy_fraction 必须位于 [0, 1]")
on_policy_count = int(batch_size * on_policy_fraction + 0.5)
mask = torch.zeros(batch_size, dtype=torch.bool)
if on_policy_count:
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(batch_size, generator=generator)[:on_policy_count]
    mask[selected] = True
return mask
```

为什么用 `randperm` 而不是"取前 k 条"？如果批次是按任务或长度排序的，取前 k 条会让"在策略样本"永远集中在某一类任务上，来源和批次位置绑定，实验结论就被数据顺序污染了。用固定种子的随机排列，既打散位置，又保证同一 seed 可复现。

`if on_policy_count:` 的作用是处理 $\lambda=0$：此时不创建生成器，直接返回全 False。测试 `test_on_policy_mask` 断言 `build_on_policy_mask(4, 0.0, seed=7).tolist() == [False]*4`、`build_on_policy_mask(4, 1.0, seed=7).tolist() == [True]*4`；对 8 条、0.5、seed=7，`first.sum().item() == 4`，同 seed 的 `repeated` 与 `first` 完全相等，seed=11 的 `changed` 不相等；`1.1` 和 `batch_size=0` 都抛 `ValueError`。

### Step 5：TODO 2——`generalized_jsd_from_logits`

输入是形状相同的 `student_logits`、`teacher_logits`（通常是 `[batch, seq, vocab]`）和标量 `beta`，输出每个 token 位置的散度（形状 `[batch, seq]`）。

参考答案的结构：

```python
if student_logits.shape != teacher_logits.shape:
    raise ValueError("教师和学生 Logits 形状必须一致")
if not 0.0 <= beta <= 1.0:
    raise ValueError("beta 必须位于 [0, 1]")

teacher_probs = torch.softmax(teacher_logits.detach(), dim=-1)
student_probs = torch.softmax(student_logits, dim=-1)
tiny = torch.finfo(student_probs.dtype).tiny
safe_teacher = teacher_probs.clamp_min(tiny)
safe_student = student_probs.clamp_min(tiny)
if beta == 0.0:
    return torch.sum(teacher_probs * (torch.log(safe_teacher) - torch.log(safe_student)), dim=-1)
if beta == 1.0:
    return torch.sum(student_probs * (torch.log(safe_student) - torch.log(safe_teacher)), dim=-1)
mixture_probs = beta * teacher_probs + (1.0 - beta) * student_probs
safe_mixture = mixture_probs.clamp_min(tiny)
teacher_kl = torch.sum(teacher_probs * (torch.log(safe_teacher) - torch.log(safe_mixture)), dim=-1)
student_kl = torch.sum(student_probs * (torch.log(safe_student) - torch.log(safe_mixture)), dim=-1)
return beta * teacher_kl + (1.0 - beta) * student_kl
```

逐点解释：

- `teacher_logits.detach()` 在 softmax 之前就切断梯度。教师是冻结的，它的概率只作为目标；如果先 softmax 再 detach，图里仍然会保留教师 logits 的依赖，没必要。
- `torch.softmax(..., dim=-1)` 沿词表维归一化，得到每个词元的概率。
- `torch.finfo(dtype).tiny` 是该浮点类型的最小正正规数（不是 0），用 `clamp_min` 把概率托到它上面，避免 `log(0) = -inf`。为什么不用 `1e-8` 之类的常数？因为 `tiny` 会随 dtype 变化，FP16、BF16、FP32 各用各的下限，不会在低精度下把本来正常的数压成 0。
- 两个端点单独返回 KL，理由前面已经说过：加权式在端点趋近 0，不能拿来做目标。
- 混合分布 `mixture_probs` 里包含 `student_probs`，所以学生的梯度会同时经过 `student_kl` 和 `teacher_kl` 两条路径。这正是 GKD 想要的：学生不只在"自己像不像混合分布"上学习，也在"混合分布有没有靠近教师"上学习。

测试 `test_generalized_jsd_boundaries` 检查四件事：教师和学生传同一份 logits 时，散度形状是 `(1,1)` 且约等于 0（`atol=1e-7`）；学生 `[4,-2,-2]`、教师 `[-2,4,-2]`、`beta=0.4` 时散度大于 0.1；学生 `[2,0,-1]`、教师 `[0,2,-1]` 时，`beta=0.0` 的结果等于手算的 `forward_kl`，`beta=1.0` 等于 `reverse_kl`（这两个 KL 在这个对称例子里数值恰好都是 1.4592）；`beta=-0.1` 抛 `ValueError`。

用框架核对单元里的 logits 手算一遍（不是保存输出，是按公式算的）：学生 `[[1.2,0.1,-0.7],[0,1,-1],[0.4,-0.2,0.7]]`，教师 `[[0.2,1.1,-0.4],[0.8,0.1,-0.5],[-0.1,0.9,0.2]]`，`beta=0.35`，三个位置的散度约为 0.0903、0.0701、0.0626；掩码 `[0,1,1]` 只保留后两个，平均约 0.0664。

### Step 6：TODO 3——`build_completion_mask`

输入是 `attention_mask`（`[batch, seq]`）、`prompt_lengths`（`[batch]`）、可选的 `excluded_mask`（同形状），输出布尔掩码，True 表示该位置参与蒸馏。

参考答案与校验：

```python
if attention_mask.ndim != 2:
    raise ValueError("attention_mask 必须为二维张量")
batch_size, sequence_length = attention_mask.shape
if prompt_lengths.shape != (batch_size,):
    raise ValueError("prompt_lengths 形状必须等于批大小")
if torch.any(prompt_lengths < 0) or torch.any(prompt_lengths > sequence_length):
    raise ValueError("prompt_lengths 必须位于序列边界内")
if torch.any((attention_mask != 0) & (attention_mask != 1)):
    raise ValueError("attention_mask 必须为二值掩码")
if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
    raise ValueError("本题只支持无左填充的右侧 Padding")
positions = torch.arange(sequence_length, device=attention_mask.device).unsqueeze(0)
completion_mask = attention_mask.bool() & (positions >= prompt_lengths.unsqueeze(1))
if excluded_mask is not None:
    if excluded_mask.shape != attention_mask.shape:
        raise ValueError("excluded_mask 形状必须与 attention_mask 一致")
    completion_mask &= ~excluded_mask.to(device=attention_mask.device, dtype=torch.bool)
return completion_mask
```

三个要点：

1. `positions` 是 `[1, seq]`，`prompt_lengths.unsqueeze(1)` 是 `[batch, 1]`，广播后每个位置与自己的提示长度比较。`>= prompt_lengths` 表示"从提示结束的下一个位置开始"。
2. 为什么要再与 `attention_mask.bool()` 相与？因为右 padding 的位置虽然 `position >= prompt_length`，但它们是填充，不是回答。
3. 单调性检查 `attention_mask[:, 1:] > attention_mask[:, :-1]` 用来拒绝左 padding。左 padding 时序列是 `[0,0,1,1,...]`，某个位置会出现"前一个是 0、后一个是 1"，条件为真，直接报错。这个实现没有去猜 padding 在哪一侧，而是明确只支持右 padding，减少歧义。

`test_completion_mask` 的期望值可以逐格核对：`attention_mask=[[1,1,1,1,0],[1,1,1,1,1]]`、`prompt_lengths=[2,1]`、`excluded_mask=[[0,0,0,1,0],[0,0,1,0,0]]`。第一行：位置 0、1 是 prompt，位置 2、3 是回答，位置 3 被排除，位置 4 是 padding → `[F,F,T,F,F]`。第二行：位置 0 是 prompt，位置 1、2、3、4 是回答，位置 2 被排除 → `[F,T,F,T,T]`。测试用 `torch.equal` 做精确比较。报错路径分别检查左 padding（`[[0,1,1]]`）、`prompt_lengths=6` 越界、`excluded_mask` 形状 `(2,4)` 不匹配。

一个实现上的小提醒：`positions` 建在 `attention_mask.device` 上，`excluded_mask` 也显式 `.to(device=attention_mask.device)`，但 `prompt_lengths` 没有显式搬设备。真实使用时如果 attention 在 GPU、`prompt_lengths` 还在 CPU，广播比较会报设备不一致；调用方要先把它放到同一设备。

### Step 7：TODO 4——`masked_distillation_loss`

```python
per_token_loss = generalized_jsd_from_logits(student_logits, teacher_logits, beta)
if loss_mask.shape != per_token_loss.shape:
    raise ValueError("loss_mask 形状必须与逐 Token 损失一致")
if torch.any((loss_mask != 0) & (loss_mask != 1)):
    raise ValueError("loss_mask 必须为二值掩码")
float_mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
valid_tokens = float_mask.sum()
if valid_tokens.item() == 0:
    raise ValueError("蒸馏掩码没有有效 Token")
return (per_token_loss * float_mask).sum() / valid_tokens
```

几个设计选择：

- 先算逐 token 散度，再检查掩码形状。如果掩码形状不对，其实可以更早报错，但这样写保证"任何输入都先经过散度函数的形状校验"，错误信息更集中。
- 掩码必须是 0/1，不接受 0.5 这种软权重。测试专门用 `torch.tensor([[0.0, 0.5, 1.0]])` 要求抛 `ValueError`。如果将来要支持软权重，需要改契约，而不是悄悄接受。
- 除数是 `valid_tokens`（浮点掩码之和），不是 `loss_mask.numel()`，也不是 batch size。
- 空掩码直接报错，而不是返回 0。返回 0 会让训练"看起来正常"，但这一步其实没有任何学习信号；报错能让调用方发现数据或长度配置有问题。

`test_mask_and_gradient_isolation` 用学生 `[[2,0],[0,2],[1,-1]]`、教师 `[[0,2],[2,0],[-1,1]]`、掩码 `[[0,1,1]]` 检查三件事：把被屏蔽位置 0 的教师 logits 改成 `[100,-100]`，损失必须不变（`atol=1e-7`），说明掩码真的把该位置排除了；反向之后 `student.grad[:,1:,:]` 的绝对值之和大于 0，而 `teacher.grad is None`，说明梯度只进学生；全零掩码抛 `ValueError`。

四个函数拼起来，一个训练步的顺序大致是：先用 `build_on_policy_mask` 标记本批哪些轨迹要学生现采，采样得到序列；用 `build_completion_mask` 从 attention mask、提示长度和排除区算出 `loss_mask`（这一步只需要 token 级信息，可以在模型前向之前做）；学生前向得到 `student_logits`，冻结教师对同一批前缀前向得到 `teacher_logits`（真实实现通常把教师前向放在 `torch.no_grad()` 下）；最后 `masked_distillation_loss` 给出标量，`backward()` 只更新学生。notebook 没有提供完整训练循环，这四个函数是它的可测试内核；真实循环还要处理生成、tokenizer 对齐、教师 logits 的显存和批大小。

和序列级目标对比一下会更清楚差别：DPO 那种做法把一条回答压成一个总 logp，只比较"整条回答的偏好"；这里保留每个位置的整个词表分布，学生能学到"这个位置上第二候选应该有多大概率"这种更细的信息。代价是教师 logits 的形状是 `[batch, seq, vocab]`，比一个标量得分大得多，实际训练要按批大小和序列长度控制显存。

### 框架一致性核对：不只是对数值

可选单元用 `torch.distributions.Categorical` 和 `kl_divergence` 独立重算逐 token 散度，再按同一个布尔掩码取平均，`beta=0.35`、掩码 `[0,1,1]`，容差 `1e-6`。它还检查梯度边界：`local_loss.backward()` 后学生梯度有限、教师梯度为 `None`。

更有意思的是它对 TRL 接口的探测：

- `DistillationTrainer` 的签名里同时有 `model` 和 `teacher_model`；
- `DistillationConfig` 的 dataclass 字段里**没有** `lmbda`；
- `beta=0.0` 和 `beta=1.0` 的配置都能构造；
- `beta=1.1` 会抛 `ValueError`。

为什么检查"没有 `lmbda`"？notebook 没有展开这条检查的动机；从字段名看，它是在确认当前 TRL 配置把 $\beta$ 放在损失侧，而 $\lambda$ 不由这个配置类管理——也就是说，轨迹来源和分布比较是两套接口。这个观察只对应当前版本，接口变了要重新核对。notebook 也提醒"训练器实际损失由其版本实现决定，数值比较对象是 PyTorch 分布路径"，所以不要把 TRL 的内部实现当成唯一标准。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| 在策略（on-policy） | 训练数据由当前学生自己生成 |
| 离线（off-policy） | 训练数据来自固定数据集或旧模型 |
| 轨迹（trajectory） | 一条提示后接出的完整回答序列 |
| 前缀（prefix） | 生成到某个位置之前已经出现的 token |
| 教师 / 学生 | 冻结的提供分布的目标模型 / 正在训练的小模型 |
| $\lambda$ | 一批轨迹里学生生成的比例 |
| $\beta$ | 广义 JSD 里教师/学生两项 KL 的权重 |
| 广义 JSD | 两个分布分别与混合分布比较的加权 KL |
| 前向 KL / 反向 KL | $D_{\mathrm{KL}}(p_T\Vert p_S)$ / $D_{\mathrm{KL}}(p_S\Vert p_T)$ |
| 完成区掩码 | 只保留回答 token、排除 prompt/padding/工具结果的 0-1 掩码 |
| 有效 token 数 | 掩码为 1 的位置数量，是损失的分母 |
| 梯度隔离 | 教师 `detach` 不接收梯度，只有学生更新 |

## 小白常见疑问

**Q1：在策略蒸馏和强化学习是一回事吗？**

不是。这里没有奖励、没有优势、没有策略梯度；学生只是在自己的轨迹上逐 token 靠近教师的分布。采样只决定"在哪些前缀上比较"，更新仍然是监督式的分布匹配。notebook 的关键词是 `on-policy distillation`，不是 RL；把它叫"在策略"只是强调数据由当前学生产生。

**Q2：为什么 $\beta=0$ 不直接用加权 JSD 公式算，而要单独写一个分支？**

因为加权式在 $\beta=0$ 时混合分布等于学生，两项 KL 一项被 0 乘、一项是 0，结果是 0，不能表达"前向 KL"。端点 KL 是**另外定义**的目标，不是通式的连续极限。代码用 `if beta == 0.0` 和 `if beta == 1.0` 两个分支显式处理，测试也用 `forward_kl` / `reverse_kl` 手算值对照。

**Q3：教师 `detach` 了，为什么学生还能从混合分布里拿到梯度？**

`detach` 只切断教师那一支。`mixture_probs = beta * teacher_probs + (1-beta) * student_probs` 里仍然有 `student_probs`，它带着学生 logits 的计算图；`student_kl` 也直接用 `student_probs`。所以反向时梯度会经过"学生→学生概率→混合分布→两项 KL"回到学生参数，而教师那一支是常数。测试用 `teacher.grad is None` 和学生掩码位置梯度非零同时确认这一点。

**Q4：为什么掩码要排除 padding 和工具返回？损失又为什么除以有效 token 数？**

只减掉 prompt 长度不够：右 padding 会被当成回答，工具调用结果也会算进蒸馏目标。前者稀释损失，后者让学生去模仿不该由它生成的内容。notebook 的掩码同时与 `attention_mask` 相与、并减去 `excluded_mask`；测试里第一行位置 4 是 padding、位置 3 是排除区，两者都必须是 False。至于分母，如果除以序列长度，padding 越多分母越大，损失被稀释得越厉害，不同 batch 之间不可比；除以有效 token 数，等价于"每个参与蒸馏的回答 token 权重相同"。Step 3 的原话是"只保留 3 个回答词元时就除以 3，而不是整段输入长度"。

**Q5：输入 logits 需要自己左移吗？**

本题的输入已经对齐到"预测同一个目标 token"的位置，所以函数内部不再 shift。真实因果模型里，位置 $t$ 的 logits 预测的是 $t+1$ 的 token，必须先把 logits 和 labels 错开一位；notebook 在 Step 4 明确提醒了这一点。如果忘了 shift，教师和学生在"看同一个前缀、预测不同 token"的位置上比较，散度再小也没有意义。另外本题只支持右 padding，左 padding 会被单调性检查拒绝。

## 动手验证：怎么知道自己做对了

测试是直接调用，不是 pytest；TODO 未填会抛 `NotImplementedError`。四个测试全过打印 `✅ 在策略蒸馏轨迹、广义 JSD、掩码和梯度隔离测试通过。`

- `test_on_policy_mask`：`(4, 0.0)` 全 False；`(4, 1.0)` 全 True；`(8, 0.5, seed=7)` 的 `sum() == 4`，同 seed 完全相等，`seed=11` 不相等；`on_policy_fraction=1.1` 和 `batch_size=0` 抛 `ValueError`。
- `test_generalized_jsd_boundaries`：同一 logits 的散度形状 `(1,1)` 且约 0（`atol=1e-7`）；学生 `[4,-2,-2]`、教师 `[-2,4,-2]`、`beta=0.4` 时大于 0.1；`beta=0.0` 等于手算前向 KL，`beta=1.0` 等于手算反向 KL；`beta=-0.1` 抛 `ValueError`。
- `test_completion_mask`：两条变长序列的逐位置期望是 `[[F,F,T,F,F],[F,T,F,T,T]]`，用 `torch.equal` 精确比较；左 padding、`prompt_lengths=6` 越界、`excluded_mask` 形状 `(2,4)` 都抛 `ValueError`。
- `test_mask_and_gradient_isolation`：改被屏蔽位置的教师 logits 后损失不变（`atol=1e-7`）；非二值掩码 `[[0,0.5,1]]` 抛 `ValueError`；`loss.backward()` 后 `student.grad[:,1:,:].abs().sum() > 0` 且 `teacher.grad is None`；全零掩码抛 `ValueError`。

可选的 `check_gkd_distribution_conformance` 会返回 `status: passed`，并列出 `per_token_gjsd_match`、`masked_loss_match`、`student_only_gradient`、`stable_on_policy_interface`、`stable_config_has_no_lambda`、`beta_endpoints_supported`、`invalid_beta_rejected` 七项检查；如果没装 TRL，返回 `skipped`。notebook 没有保存任何执行输出，所以上面提到的 0.0903、0.0701、0.0626、0.0664 是按公式手算的演示值，不是你机器上的实测记录；判断实现是否正确，以测试断言和你运行时打印的 ✅ 为准。
