# 13. SFT 监督边界与损失 — 小白讲解

> 对应 notebook：[`13_SFT_Training_Loop.ipynb`](./13_SFT_Training_Loop.ipynb)

## 一句话总结
监督微调（SFT）的输入包含提示和回答，但损失只应该算在回答（以及可选的 EOS）上。本节用 `build_sft_data` 构造输入、注意力掩码和标签，用 `compute_sft_loss_terms` 做“位置 $t$ 预测位置 $t+1$”的错位对齐并累加有效损失，用 `compute_sft_loss` 按有效监督词元数求平均。四个 TODO 分别覆盖监督边界、右侧填充、下一词元对齐和 token 加权归约，最后用长短样本对照证明“按 token 加权”和“按样本等权平均”会得到不同结果。

## 为什么需要它：痛点是什么
初学者写 SFT 损失时最常犯三类错。第一，把提示也计入损失：模型于是花力气去预测用户的问题，而不是学习怎么回答。第二，忘记错位：位置 $t$ 的 logits 应该预测 $t+1$ 的标签，直接拿同位置对齐会让每个词都在“抄自己”。第三，用 `loss.mean()` 或对各微批次均值再平均：长短回答获得相同权重，短样本被放大、长样本被稀释，梯度尺度和完整批次不一致。
生活类比：老师批改作文。题目（prompt）是背景，不该打分；只给正文（response）打分。如果班里有 1 篇 100 字和 1 篇 1000 字的作文，按“每篇平均分再平均”会让 100 字那篇占一半权重；正确做法是把两篇的扣分总和除以总字数，按字数公平加权。

## 核心问题
- SFT 与预训练共享下一词元预测，但监督范围差在哪？
- 为什么 `labels` 里 prompt 和 padding 要写成 `-100`？`-100` 在哪里被忽略？
- 为什么有效监督数从 `labels[1:]` 开始统计，而不是从 `labels[0]`？
- 右侧填充时，`input_ids`、`attention_mask`、`labels` 三个序列怎样保持同步？
- 为什么完整批次的损失等于“各微批次损失总和相除”，而不是“各微批次均值再平均”？
- EOS 和 PAD 共用同一个 id 时，怎样靠位置而不是词元值区分？

## 解决思路：一步步看懂

**第 1 步：SFT 的目标是什么。** 对模板化后的词元序列 $x_{1:T}$，因果语言模型把联合概率分解为

$$
p_\theta(x_{1:T})=\prod_{t=0}^{T-1}p_\theta(x_{t+1}\mid x_{\le t}),
$$

其中 $T$ 是序列长度，$t$ 是当前位置，$\theta$ 是模型参数。代码在输入序列内部移位：用位置 $t$ 的输出预测 $x_{t+1}$，所以可配对的位置是 1 到 $T-1$。预训练通常监督全部非填充词元；SFT 通常只监督回答和可选 EOS，提示只提供上下文。对话模板、分词器配置和监督边界必须来自同一个模型版本，否则回答起点会偏移。

**第 2 步（TODO 1）：建立监督边界。** `build_sft_data` 先把 EOS 接在回答后面，再按 `max_len` 截断：

```python
response_with_eos = response_ids + ([] if eos_id is None else [eos_id])
input_ids = (prompt_ids + response_with_eos)[:max_len]
labels = ([-100] * len(prompt_ids) + response_with_eos)[:max_len]
valid_supervised = sum(label != -100 for label in labels[1:])
if valid_supervised < min_response_tokens:
    raise ValueError("截断后没有足够的 response token 参与监督")
```

`labels` 与 `input_ids` 等长，但 prompt 位置全部写成 `-100`，回答和 EOS 保留真实 token id。关键细节是 `labels[1:]`：`labels[0]` 对应序列第一个词，它前面没有位置可以产生预测，移位后不会进入损失，所以不能算作有效监督。如果提示为空、回答正好落在索引 0，`labels[0]` 会被丢掉；notebook 末尾用 `build_sft_data([], [20], eos_id=None, max_len=1)` 验证这种情况会因“移位后零目标”被拒绝。测试主用例 `prompt=[10,20,30]`、`response=[40,50,60,70]`、`eos=2`、`max_len=9` 得到 `labels=[-100,-100,-100,40,50,60,70,2,-100]`，`labels[1:]` 里有 5 个有效目标。

**第 3 步（TODO 2）：同步右侧填充。** 三个序列必须一起补齐，而且 padding 同时退出注意力和监督：

```python
padded_input_ids = input_ids + [pad_id] * pad_len
padded_attention_mask = attention_mask + [0] * pad_len
padded_labels = labels + [-100] * pad_len
```

只补 `input_ids` 会让三个张量长度不一致；只补 mask 不补 labels 会让 padding 位置参与交叉熵；给 padding 的 label 填 `pad_id` 而不是 `-100`，也会在 `ignore_index` 之外制造假目标。`test_eos_padding_and_truncation` 特意让 `pad_id=eos_id=2`：`build_sft_data([10], [20], pad_id=2, eos_id=2, max_len=5)` 得到 `ids=[10,20,2,2,2]`、`mask=[1,1,1,0,0]`、`labels=[-100,20,2,-100,-100]`。虽然第 3、4 个位置的值也是 2，但 mask 为 0，所以 `compute_sft_loss_terms` 统计到 2 个有效目标（`labels[1:]=[20,2,-100,-100]`）。另一个用例 `prompt=[10,11]`、`response=[20,21]`、`eos=2`、`max_len=4` 先加 EOS 再截断，EOS 被截掉但回答前缀保留，`labels=[-100,-100,20,21]`、有效数 2，说明“EOS 可以被截断，但回答不能全丢”。

**第 4 步（TODO 3）：下一词元对齐。** 参考实现只有两行移位，再加一行掩码同步：

```python
shift_logits = logits[..., :-1, :].contiguous()   # [B, T-1, V]
shift_labels = labels[..., 1:].contiguous()       # [B, T-1]
shift_attention_mask = attention_mask[..., 1:].contiguous()
shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
```

`logits` 形状是 `[batch, seq_len, vocab]`，去掉最后一个位置；`labels` 去掉第一个位置，于是 `shift_logits[t]` 与 `shift_labels[t]` 指向同一个目标词元 `x_{t+1}`。`attention_mask` 也要跟着去掉第一个位置：mask 描述的是“目标词元是否真实”，padding 在后半段，所以用 `[..., 1:]` 对齐后把对应 label 改成 `-100`。随后 `F.cross_entropy(..., ignore_index=-100, reduction="none")` 得到每个位置的损失，`-100` 位置贡献 0；`valid_token_count = (shift_labels != -100).sum()` 是分母。`test_sft_pipeline` 的 `_make_aligned_logits` 只在位置 2 到 6 把正确目标（40、50、60、70、2）的 logit 拉到 50.0，其余随机；正确的移位应该让损失小于 `0.01`。`_assert_supervision_scope` 还做三组扰动：改 prompt 位置的 logit 不改变损失；改 response 位置 `[0,2,40]` 会降低损失；把 padding 位置的 label 改成 99 也不改变损失（被 mask 掉）。

**第 5 步（TODO 4）：按有效 token 归约。** `compute_sft_loss_terms` 已经返回 `loss_sum` 与 `valid_token_count`，最后一步就是

```python
mean_loss = loss_sum / valid_token_count
```

它对应

$$
L_{\mathrm{SFT}}=\frac{\sum_{b,j}m_{b,j}\ell_{b,j}}{\sum_{b,j}m_{b,j}},
$$

$m_{b,j}$ 是监督掩码，$\ell_{b,j}$ 是目标位置的负对数似然。为什么不能对样本或微批次等权平均？`_assert_effective_token_reduction` 构造两条样本：第一条只有 1 个有效目标，第二条有 3 个，总共 4 个。完整批次损失是 `(S_1+S_2)/(N_1+N_2)`；如果先算每条样本均值再乘 0.5，短样本的 1 个目标会占一半权重，长样本的 3 个目标也只占一半，结果与完整批次不同。测试断言完整批次与“先拆分再按 `(short_sum+long_sum)/(short_count+long_count)` 合并”的损失、梯度都 `allclose`，同时断言 `unweighted_micro_mean` 与它们不相等——这正是错误写法会被抓住的地方。

**第 6 步：框架一致性核对。** 把两条受控样本先去掉本地右侧填充，交给 TRL 的 `DataCollatorForLanguageModeling(pad_token_id=0)` 重新组批；逐行检查 `input_ids`、`labels`、`attention_mask` 一致，padding 标签为 `-100`，prompt 段标签为 `-100`，response/EOS 段标签等于输入。有效 token 数应为 `(6-3)+(4-1)=6`。再用一个极小的 `GPT2Config`（`vocab_size=64, n_positions=16, n_ctx=16, n_embd=24, n_layer=1, n_head=2`）前向，比较 `framework_output.loss` 与本地 `compute_sft_loss`，要求 `atol=1e-6`。它验证的是“监督边界 + 位移 + 分母”三件事与成熟实现一致；没装 `transformers`/`trl` 时状态为 `skipped`，不能算通过。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| SFT | 用“提示—回答”标注数据继续训练预训练模型 |
| chat template | 把角色、消息和分隔标记排成模型约定的序列 |
| 词元（token） | 分词器切出的词、子词或符号 |
| 监督边界 | 哪些位置计入损失；SFT 通常只算回答和 EOS |
| label masking | 用 `-100` 把 prompt、padding 等位置排除出交叉熵 |
| next-token prediction | 用位置 $t$ 的输出预测位置 $t+1$ 的词元 |
| shift / 错位 | logits 去掉最后一位、labels 去掉第一位，使预测与目标对齐 |
| 有效监督词元数 | `shift_labels != -100` 的个数，是损失的分母 |
| 逻辑批次 / 微批次 | 一次参数更新用的全部样本 / 其中一次送进模型的子集 |
| 右侧填充 | 把短样本补在右边，保持 batch 内长度一致 |
| EOS | 回答结束标记，通常也参与监督 |
| token 加权归约 | 先汇总损失总和与有效数，再相除，而不是按样本平均 |

## 小白常见疑问
**Q1：为什么 prompt 的标签是 `-100`，但 prompt 还要作为输入？**
prompt 提供上下文，模型需要读它才能生成回答；但 SFT 不想让模型学习“预测用户的问题”，所以只把回答和 EOS 作为目标。`_assert_supervision_scope` 改 prompt 位置的 logit 后损失完全不变，就是在验证 prompt 不参与监督。

**Q2：为什么统计有效监督数要从 `labels[1:]` 开始？**
因为损失用的是移位后的标签。位置 0 的标签即使不是 `-100`，也没有位置 $-1$ 的 logit 来预测它，`shift_labels` 从索引 1 开始后就把它丢掉了。`build_sft_data([], [20], max_len=1)` 里唯一的目标在索引 0，移位后一个都不剩，所以函数在构造阶段就抛 `ValueError`，而不是等到损失计算时才发现除零。

**Q3：EOS 和 PAD 用同一个 id，不会混淆吗？**
不会，因为区分靠 `attention_mask` 和位置，不靠词元值。`test_eos_padding_and_truncation` 让 `pad_id=eos_id=2`，序列里三个 2 分别是 EOS 和两个 padding；mask 是 `[1,1,1,0,0]`，`labels` 是 `[-100,20,2,-100,-100]`，所以只有真正的 EOS 参与监督，padding 的 2 被 `-100` 排除。

**Q4：`loss.mean()` 和 `loss_sum / valid_token_count` 差在哪里？**
`reduction="none"` 时，`loss.mean()` 会把 `-100` 位置也算进分母（或者需要先过滤），而 `loss_sum / valid_token_count` 只除以真实目标数。`_assert_effective_token_reduction` 用 1 个目标与 3 个目标的样本证明：等权平均和 token 加权的结果、梯度都不一致，所以 SFT 必须用后者。

**Q5：为什么完整 batch 的损失不能由各微批次均值再平均？**
设微批次 $k$ 的损失总和为 $S_k$、有效目标数为 $N_k$。正确聚合是 $\sum_k S_k / \sum_k N_k$；先算 $S_k/N_k$ 再平均，相当于给每个微批次相同权重，短回答的微批次会被放大。这个分母在后续梯度累积里还会再次出现，必须从一开始就统一。

## 动手验证：怎么知道自己做对了
`test_sft_pipeline` 依次检查五件事：`_assert_sft_data_contract` 要求 `input_ids == [10,20,30,40,50,60,70,2,0]`、`attention_mask == [1,1,1,1,1,1,1,1,0]`、`labels == [-100,-100,-100,40,50,60,70,2,-100]`；`_assert_truncation_guard` 要求 `max_len=3` 把回答截空时抛 `ValueError`；`_assert_loss_alignment` 要求对齐 logits 的损失 `< 0.01`；`_assert_supervision_scope` 要求 prompt 扰动不变、response 扰动下降、padding 目标被屏蔽；`_assert_effective_token_reduction` 要求完整批次与 token 加权拆分的损失、梯度 `allclose`，且与等权微批次均值不同。通过后打印 `✅ SFT 监督边界、有效词元分母与梯度归约测试通过。`

`test_eos_padding_and_truncation` 再检查 EOS/PAD 同 id 的序列与有效数 2，以及“先加 EOS 再截断”时回答前缀保留、有效数 2。最后还有一条独立断言：空 prompt、单 token 回答、`eos_id=None`、`max_len=1` 必须在构造阶段抛 `ValueError`。三个检查都过，说明监督边界、填充同步、错位对齐和分母四件事都对了。notebook 没有保存这些单元的执行输出，因此以断言和打印文案为准；没有 GPU 代码，也不需要运行训练。
