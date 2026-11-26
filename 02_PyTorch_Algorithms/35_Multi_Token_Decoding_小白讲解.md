# 35. 原生多 Token 预测 — 小白讲解

> 对应 notebook：[`35_Multi_Token_Decoding.ipynb`](./35_Multi_Token_Decoding.ipynb)

## 一句话总结
原生多 Token 预测（MTP）把"预测未来几个位置"的模块放进目标模型检查点内部，让多个预测深度各给出一个候选 Token，再由目标模型在同一轮里逐位置验证。本节用 `select_mtp_candidates`、`continuous_match_length`、`build_mtp_round` 三个函数复现"候选 → 验证 → 提交"这条链，核心结论是：候选数量只表示最多尝试几个位置，真正决定本轮推进量的是"从头开始连续匹配"的长度。Step 5 的 vLLM 配对实验在环境阶段失败，没有产出任何可比较的性能数字。

## 为什么需要它：痛点是什么
初学者最容易把多 Token 预测理解成"一次多生成几个 Token，所以一定更快"。真跑起来会发现：候选可能第一个就错，接受长度是 0，本轮只提交一个 Token，而候选计算和验证计算都白花了；也可能候选全对，一次拿到 `候选数 + 1` 个 Token，但每个 Token 的平均成本反而更高。

生活类比：让实习生先写一份后续邮件的草稿（MTP 候选），主管再逐句核对（目标模型验证）。主管从第一句开始读，读到哪里发现不对，就把那一句改掉、后面全部作废，只发"已确认的前缀 + 改好的那一句"。草稿写了五句，不代表最后能采纳五句；采纳几句取决于从第一句起连续对了几句。而且主管改完那一句之后，还会顺手多写一句自己的结尾——这就是"全部匹配时多拿一个 Token"的来源。

## 核心问题
- 原生 MTP 的候选从哪里来？为什么模型检查点里没有 MTP 层就不能启用这条路径？
- 为什么候选块必须按预测深度排序，不能重排或随便截断？
- 连续匹配长度为什么必须是"前缀"，中间跳过不匹配位置会破坏什么？
- 全部候选都接受时，本轮为什么能输出 `n + 1` 个 Token？
- 候选长度、接受长度、每轮输出长度和端到端加速比之间是什么关系？
- GPU 实验失败时，`failure`、`comparison`、`decision` 这几个字段应该怎么读？

## 解决思路：一步步看懂

**第 1 步（TODO 1）：候选是"每个深度取一个 argmax"。** 任务函数是 `select_mtp_candidates`。记预测深度数为 $D$、词表大小为 $V$，`depth_logits` 的形状是 `[depth, vocab]`，第 $d$ 行是第 $d$ 个预测深度对整个词表的分数。参考答案只有一行：

```python
candidates = scores.argmax(dim=-1)[:max_candidates].tolist()
```

`argmax(dim=-1)` 在词表维上取每一行分数最大的下标，得到长度 $D$ 的列表；`[:max_candidates]` 按深度顺序截断到预算；`.tolist()` 转成 Python 整数列表交给后面的比较逻辑。测试 `test_select_mtp_candidates` 给了三行分数 `[[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]]`，逐行 argmax 得到 `[1, 0, 2]`：预算为 2 时返回 `[1, 0]`，预算为 8（大于深度数）时返回全部 `[1, 0, 2]`。注意 `max_candidates` 的校验把 `bool` 单独排除了——`True` 在 Python 里是 `int` 的子类，不排除的话 `True` 会被当成 1 悄悄通过。

顺序不能重排，是因为第 $i$ 个候选对应的是"当前上下文之后第 $i+1$ 个位置"。如果按分数从高到低重排，候选就与待验证位置错位，后面 `continuous_match_length` 的逐位置比较全部失去意义。同理，截断只能从尾部砍掉更远的深度，不能从中间抽掉一个。

检查点契约是这条路径的硬前提。Step 1 的表格把"主模型状态""MTP 预测深度""有序候选块"三者的条件分开写：主模型状态必须与本轮目标验证共享同一模型语义；每个预测深度必须对应检查点里真实存在的内置模块；候选块顺序必须与待验证位置一一对应。运行参数只能决定"用不用、用几层"，不能凭空造出模块。GPU 实验里 `model_contract` 的 `mtp_num_hidden_layers = 1` 就是在核对这件事；如果它是 0，`SPECULATIVE_TOKENS = 1` 就没有对应的层可用，应该停在这一步而不是继续跑。反过来说，`select_mtp_candidates` 只按传入张量的第一维当作深度数，它自己无法知道检查点里有几层，所以调用方必须先完成契约检查，再把 `depth_logits` 交进来。

**第 2 步（TODO 2）：连续匹配长度从首位置开始数。** `continuous_match_length` 先用 `len(target_tokens) < len(candidates)` 保证目标结果覆盖所有候选位置（目标模型至少要给出同样多的验证结果，外加一个额外位置）。循环体里参考答案是：

```python
for candidate, target in zip(candidates, target_tokens):
    matches = int(candidate) == int(target)
    if not matches:
        break
    accepted_length += 1
```

`int(...)` 是防御性写法，避免 `torch.Tensor` 标量和 Python 整数比较时出现意外；一旦不匹配立刻 `break`，后面的候选即使碰巧和目标相同也不再计数。测试 `test_continuous_match_length` 用三组数据把这个性质钉死：
- `[4, 5, 6]` 对 `[4, 5, 9, 8]`：前两个相同、第三个不同，返回 2；
- `[4, 5]` 对 `[9, 5, 8]`：首位置就不同，返回 0——尽管第二个位置 5 是匹配的，也不能算；
- `[4, 5]` 对 `[4, 5, 8]`：全部匹配，返回 2。

第二组是最容易被写错的地方。如果写成"统计所有位置相同个数"，会得到 1；如果写成"遇到不同就跳过继续数"，也会把后面的匹配算进来。这两种写法都破坏了"接受的是候选块的一个连续前缀"这个不变量，而目标模型的输出分布只保证"在已接受前缀的条件下，当前位置的首选 Token 是可信的"。

还要注意这里比较的是"贪心首选 Token"，不是概率。Step 2 的状态表写的是"贪心路径下"：候选和目标都取 argmax，比较退化成确定性的整数相等。若换成采样，验证就不再是简单的 `==`，而要做接受概率的比较，本节不涉及那条路径。`int(candidate) == int(target)` 这层转换是为了让 `torch.Tensor` 标量和 Python 整数也能可靠比较；如果直接用张量做布尔判断，`if not matches` 在零维张量上虽然能工作，但多元素张量会直接报错。

Step 2 的验证状态表只有三行，但正好对应 `build_mtp_round` 的三个分支：当前位置匹配时接受长度加一、继续检查；首次不匹配时提交已接受前缀和目标修正 Token；全部匹配时提交候选块和额外目标 Token。把这三行翻译成代码，就是 `accepted_length` 在循环里累加、`emitted_tokens` 用 `candidates[:accepted_length]` 加 `target_tokens[accepted_length]`、`all_candidates_accepted` 用长度相等判断。读代码时可以拿这张表逐行核对，三者缺一不可。

还有一个观察量的区分：`candidate_count` 是这轮尝试了几个位置，`accepted_length` 是通过了几个。两者相除可以得到"本轮接受率"，但它仍然不是加速比；接受率描述的是候选质量，加速比还要除以候选和验证的成本。notebook 在返回字典里同时保留这两个字段，是为了让你在日志里能区分"候选开得太长"和"候选质量太差"：前者表现为 `candidate_count` 大而 `accepted_length` 长期偏小，后者表现为两者都小。

**第 3 步（TODO 3）：提交时只补一个目标 Token。** `build_mtp_round` 先检查 `len(target_tokens) == len(candidates) + 1`：目标模型验证的是 $n$ 个候选位置，再加一个"如果候选全对，下一位置该是什么"的额外位置。参考答案两行：

```python
emitted_tokens = list(candidates[:accepted_length]) + [int(target_tokens[accepted_length])]
all_candidates_accepted = accepted_length == len(candidates)
```

第一行是"接受前缀 + 一个修正或追加"：如果在中途第 `accepted_length` 个位置首次不匹配，就取目标模型在该位置的 Token 作为修正；如果全部匹配，`accepted_length == len(candidates)`，取的就是那个额外位置的目标 Token，于是 $n$ 个候选换来了 $n+1$ 个输出。测试里的拒绝路径 `build_mtp_round([4, 5, 6], [4, 5, 9, 8])` 得到 `accepted_length = 2`、`emitted_tokens = [4, 5, 9]`、`first_mismatch = 2`，候选 `6` 被丢弃；接受路径 `build_mtp_round([4, 5], [4, 5, 8])` 得到 `[4, 5, 8]`、`all_candidates_accepted = True`、`first_mismatch = None`。

`first_mismatch` 的语义也要注意：全部接受时它是 `None`，而不是 `len(candidates)`。这样调用方可以直接用 `if first_mismatch is not None` 判断是否发生过拒绝，不需要再比较长度。

**第 4 步：候选数、接受长度、推进量和加速比不是一回事。** 记第 $r$ 轮的接受长度为 $a_r$、候选数为 $n_r$，本轮真正提交的 Token 数是 $a_r + 1$。注意三个层次：
- **候选长度** $n_r$：本轮最多尝试验证多少个位置，由检查点里 MTP 层数和预算决定，不是收益。
- **接受长度** $a_r$：其中从首位置起连续通过验证的个数，后部候选即使内容正确也不能越过前面的失败。
- **每轮输出长度** $a_r + 1$：这才是这一轮对序列长度的实际贡献。

端到端加速还要看成本：每轮多跑了候选模块，目标模型也要对 $n_r + 1$ 个位置做一次前向验证。所以"接受长度长"只是必要条件；notebook 的 Step 3 表格特意写了三句"不能单独推出"：候选长度不等于实际推进量，连续接受长度不等于端到端加速比，每轮输出长度仍需结合候选和验证成本。如果第一处不匹配出现得很早，后部候选的计算基本是沉没成本。

把成本拆开看会更清楚：一轮里候选模块要对 $n_r$ 个深度各算一次词表分数，目标模型要对 $n_r+1$ 个位置给出验证结果。三种极端情况：
- 全部接受：一轮产出 $n_r+1$ 个 Token，目标模型只调用一次，这是最理想的情况；
- 第一处就错：一轮只产出 1 个 Token，候选模块和 $n_r+1$ 个位置的验证都白花，这时甚至可能比"不用 MTP、老老实实逐 Token 解码"更慢；
- 中间错：产出 $a_r+1$ 个 Token，$a_r$ 越大越接近理想情况。

所以"候选长度 $n_r$ 该开多大"本身是一个权衡：开大能提高潜在推进量，但候选计算、验证宽度和显存都会增加；如果模型的接受长度分布偏短，开大只会让沉没成本更高。notebook 没有给出最优 $n_r$ 的公式，Step 3 只强调"更长的候选块扩大潜在推进量，也会增加候选计算和验证成本"。

用一个三轮回放把"调用次数"和"端到端时间"区分开：假设 $n=3$，第 1 轮接受 2 个、第 2 轮接受 0 个、第 3 轮接受 1 个，三轮分别产出 3、1、2 个 Token，共 6 个；不用 MTP 时，基线需要 6 次目标模型调用。看起来"3 次对 6 次"是 2 倍，但每轮还要加上候选计算和更宽的验证，真实比例通常小于 2。这个回放只是用机制推出来的算术，不是实测加速比；要得到后者，必须像 Step 5 那样做同模型、同输入、同生成长度的配对测量。

**第 5 步：GPU 实验想验证什么。** Step 5 的设计是"同一模型检查点的仅目标模型 → MTP-1"配对：基线是只跑目标模型，候选是启用 1 层原生 MTP（`SPECULATIVE_TOKENS = 1`）。固定条件写得很细：`Qwen/Qwen3.5-0.8B`、BF16、批量 1、`max_model_len = 2048`、显存利用率 0.75、每条提示生成 64 Token、3 条提示、`seed = 42`、两条路径各自预热 2 次、正式重复 5 次、`enforce_eager = False`。要读的指标是输出一致性、接受长度、Median/P95 延迟、吞吐和峰值显存。实验设计的关键是"配对"：两条路径共用输入、数据类型、生成长度和随机种子，只改变是否启用 MTP，这样差异才能归因到 MTP 本身。

**第 6 步：这次实验失败了，怎么读。** 保存的输出里 `baseline`、`candidate`、`comparison` 全是 `None`，`failure` 是一个 `RuntimeError`，消息开头是 `baseline worker failed (exit=1)`，堆栈最后落在 Triton 的 `load_binary`，报 `Triton Error [CUDA]: device kernel image is invalid`；失败发生在 vLLM 的 `profile_run` 阶段，也就是基线还没开始正式生成。`model_contract` 倒是解析出来了：`model_type = qwen3_5`、`text_model_type = qwen3_5_text`、`mtp_num_hidden_layers = 1`、`resolution_source = huggingface_hub_api`，说明检查点确实带 1 层原生 MTP，问题不在模型契约。最终 `evidence_level = failed_real_backend_attempt`、`decision = reject_until_fixed`。`decision` 的语义是"修好之前拒绝采用"，不是"永久否定这个机制"；它记录的是这次运行的可信度，不是对 MTP 的最终判决。

读法只有一句：基线没有跑通，就不能解释任何性能字段，也不能拿"候选也没跑"去推断 MTP 没有收益。5.4 的表格把这次失败归到专用后端构建门禁（环境未满足要求的 vLLM 0.31.0 CUDA 12.9 构建），并明确"启动门禁失败不能用于推断性能或显存容量"。正确动作是先让 target-only 路径完成模型加载、内核初始化、预热和固定输入生成，再以新标签跑 MTP-1。

这里有一处需要读者自己核对：5.4 的文字说"本次尚未完成模型契约检查"、表格里写 `model_contract` 为 `null`，而 notebook 保存的 5.2/5.3 输出已经解析出 `mtp_num_hidden_layers = 1`。两处描述的不是同一份运行记录。读的时候以你手上 JSON 的 `failure`、`model_contract`、`comparison` 字段为准，不要用文字表格覆盖实际输出。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 原生 MTP | 预测未来多个位置的模块直接长在目标模型检查点里，不另训草稿模型 |
| 预测深度（depth） | 第几个未来位置；一个深度给出一行词表分数、一个候选 |
| 候选块 | 按深度顺序排好的一串候选 Token，顺序对应待验证位置 |
| 连续接受长度 | 从首位置开始，候选与目标首选 Token 连续相同的个数 |
| 首次不匹配 | 第一个候选与目标不同的位置，之后的后缀全部作废 |
| 修正 Token | 首次不匹配位置上目标模型给出的 Token，用来替代错误候选 |
| 额外目标 Token | 全部候选都接受时，目标模型多给的那个下一位置 Token |
| 有效推进量 | 本轮真正写进输出的 Token 数，等于接受长度加一 |
| target-only | 基线路径，只跑目标模型，不启用 MTP |
| MTP-1 | 候选路径，启用 1 层原生 MTP 预测 |
| `failure` | 记录失败阶段与异常的结构；非空时不能解释性能字段 |
| `decision` | 证据等级对应的动作，如 `reject_until_fixed` |

## 小白常见疑问

**Q1：为什么不能像投机解码那样，用一个独立的小模型当草稿？**
本节讲的是"原生"MTP：候选来自目标模型检查点内部的预测模块，多个深度复用主模型状态。它的前提是检查点包含对应数量的内置模块——`model_contract` 里的 `mtp_num_hidden_layers` 就是用来核对这一点的。独立草稿模型属于另一条路线（相关阅读里的 INF-EAGLE、INF-PARALLEL-DRAFTING），候选来源和成本结构都不同。

**Q2：候选块长度为 3，接受长度为 2，为什么输出是 3 个 Token 而不是 2 个？**
因为还要加上首次不匹配位置的目标修正 Token。`build_mtp_round([4, 5, 6], [4, 5, 9, 8])` 的输出是 `[4, 5, 9]`：前两个是接受的候选，第三个 `9` 是目标模型在位置 2 的修正。候选 `6` 没有被验证通过，不能进入输出。公式上每轮输出长度是 $a + 1$，$a$ 是接受长度。

**Q3：全部接受时为什么能多拿一个？这不是凭空多生成的吧？**
不是凭空。目标模型在验证时本来就要对 $n+1$ 个位置做前向（$n$ 个候选位置加一个额外位置），全部接受时那个额外位置的首选 Token 也是目标模型自己算出来的，所以可以一起提交。`build_mtp_round` 的输入要求 `len(target_tokens) == len(candidates) + 1` 就是在表达这个契约。

**Q4：`continuous_match_length` 里为什么先检查 `len(target_tokens) < len(candidates)`，而不是要求严格相等？**
因为这个函数只负责"数前缀"，它允许目标结果比候选更长（多出来的那个位置留给 `build_mtp_round` 用）。测试里 `continuous_match_length([4, 5], [4, 5, 8])` 返回 2，就是目标比候选多一个位置的正常情况。真正要求"恰好 $n+1$"的检查在 `build_mtp_round` 里。

**Q5：GPU 实验失败了，是不是说明 MTP 没用？**
不能这么推。失败发生在基线路径的 `profile_run`，`comparison` 是 `None`，连"仅目标模型"的延迟都没有测到。`evidence_level = failed_real_backend_attempt` 的含义是"这次真实后端尝试失败"，只能说明当前环境跑不起来，不能说明机制有没有收益。

**Q6：接受长度为 0 时，本轮还能输出东西吗？**
能，而且必须输出 1 个。`build_mtp_round` 的 `emitted_tokens = candidates[:0] + [target_tokens[0]]`，就是 `[target_tokens[0]]`。目标模型在首位置的首选 Token 是它自己算出来的，可信；候选一个都没通过，只是"草稿全废"。每轮输出长度统一是 $a+1$，$a=0$ 就是 1，不会出现空轮，这也是"有效推进量"的定义在边界上仍然成立的原因。

**Q7：为什么候选返回 Python 整数列表，而不是 `torch.Tensor`？**
因为后续要做的是逐位置切片、拼接和与目标 Token 比较，不是批量张量运算。`select_mtp_candidates` 末尾的 `.tolist()` 把张量转成 Python 整数，`build_mtp_round` 再用 `list(candidates[:accepted_length]) + [int(...)]` 拼输出，整个过程不依赖设备、不需要同步。测试也直接用列表相等断言，例如 `select_mtp_candidates(logits, 2) == [1, 0]`；如果返回张量，`==` 会得到逐元素布尔张量，断言方式就完全不同了。

## 动手验证：怎么知道自己做对了
notebook 在测试单元里直接调用三个测试函数，而不是等 pytest 收集；任何一个断言失败都会立刻中断，后面的测试不再执行，所以第一次跑就把三个 TODO 都填完整更省时间。全部通过后打印 `✅ 原生 MTP 候选与贪心验证测试通过`；TODO 1/2/3 没填时会先抛 `NotImplementedError`。

- `test_select_mtp_candidates`：`select_mtp_candidates(logits, 2) == [1, 0]`，`select_mtp_candidates(logits, 8) == [1, 0, 2]`；空张量 `torch.empty(0, 3)` 和一维张量 `torch.tensor([0.1, 0.9])` 都必须抛 `ValueError`（由 `_validate_depth_logits` 的二维、非空检查负责）。
- `test_continuous_match_length`：`([4, 5, 6], [4, 5, 9, 8]) == 2`、`([4, 5], [9, 5, 8]) == 0`、`([4, 5], [4, 5, 8]) == 2`。第二组专门防"统计所有匹配位置"的错误写法。
- `test_build_mtp_round`：拒绝路径的返回字典必须完全等于 `{"candidate_count": 3, "accepted_length": 2, "emitted_tokens": [4, 5, 9], "all_candidates_accepted": False, "first_mismatch": 2}`；接受路径要求 `emitted_tokens == [4, 5, 8]`、`all_candidates_accepted is True`、`first_mismatch is None`。

Step 5 没有可复现的通过条件：保存的输出显示 `failure` 非空、`comparison = None`。要真正验证性能，需要先在 `INF-VLLM-MTP` 环境通过预检，用新 `RUN_LABEL` 跑通 target-only，再跑 MTP-1，并确认 `failure` 为 `None` 后才读 Median/P95、吞吐和峰值显存。当前 notebook 里能确认的只有模型契约（1 层原生 MTP）和失败阶段，不能确认任何加速比。

调试时还会遇到几类输入错误，错误消息本身就说明了契约：
- `_validate_depth_logits`：不是二维、或任一维为 0，抛 `depth_logits 必须是非空二维张量 [depth, vocab]`；含 `NaN`/`Inf`，抛 `词表分数不能包含 NaN 或 Inf`。这两个检查在 TODO 之前执行，所以即使 TODO 没填，非法输入也会先报这里的错。
- `max_candidates`：不是 `int`、是 `bool`、或小于 1，抛 `max_candidates 必须是正整数`。注意 `8` 这种大于深度数的预算是合法的，返回全部深度，不算错误。
- `continuous_match_length`：目标结果比候选短，抛 `目标结果必须覆盖所有候选位置`，因为前缀比较需要每个候选位置都有对应的目标 Token。
- `build_mtp_round`：目标长度不等于候选数加一，抛 `目标结果必须包含候选位置和一个额外位置`。多了也不行，因为多出来的位置没有被候选覆盖，语义不明确。

如果测试只差一点，优先检查三处：候选是不是按深度顺序（不是按分数排序）、`break` 是不是在计数之前、`first_mismatch` 全部接受时是不是 `None`。这三处对应三个最典型的实现偏差。
