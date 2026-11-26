# 08. 模型配置与参数归属 — 小白讲解

> 对应 notebook：[`08_Architecture_Tricks.ipynb`](./08_Architecture_Tricks.ipynb)

## 一句话总结
本节用三个小 TODO 把"配置字段"落到真实的 `nn.Module` 上：`GemmaRMSNorm.forward` 补上 `1 + weight` 缩放，`QwenTieEmbeddings.__init__` 让词嵌入和 LM Head 指向同一份 `Parameter`，`summarize_parameter_layout` 算清绑定与不绑定各占多少词表参数。最后用固定函数 `audit_model_config` 比较"配置声明"和"模块实际状态"，把参数身份、梯度归属和检查点语义串成一条可复查的证据链。

## 为什么需要它：痛点是什么
初学者读模型配置时，常把 `tie_word_embeddings: true` 当成一句注释，却说不清它到底省了多少参数；看到 Gemma 的 norm 权重初始化为全 0，也会疑惑"归一化层权重为 0，输出不就没了吗"。更隐蔽的坑是：把两份数值相同的词表矩阵复制一份，以为这就是"共享"。

生活类比：两家店共用同一个仓库，和两家店各自进了一批一模一样的货，账面数字一样，但前者补货一次两边都变，后者要分别补。权重绑定关心的是"是不是同一个仓库"（同一个 `Parameter`），不是"货长得像不像"。

## 核心问题
- Gemma 风格 RMSNorm 的 `1 + weight` 缩放为什么用零初始化？它和普通 RMSNorm 的 `weight` 有什么区别？
- 权重绑定（weight tying）到底绑的是数值还是参数身份？怎么用 `data_ptr()` 验证？
- 绑定和不绑定时，词表参数账本各是多少？省下的 `saved_params` 是什么？
- 配置文件说"绑定"，模型实际没绑定，审计函数如何分类？
- 做这些局部选择会改变哪些参数、梯度和检查点语义？

## 解决思路：一步步看懂

**第 1 步：先看配置审计的四个维度（Step 1–3）。** notebook 把配置选择分成参数归属、数值约定、残差与深度、注意力/MLP 四类，并反复强调审计要回答三件事：配置声称什么、模块实际是什么、这个选择改变了哪些参数或状态。Step 3 的表格把"权重共享""归一化变体"等逐项列出，本节只实现前两个可运行例子。

**第 2 步：TODO 1——Gemma 的 `1 + weight` 缩放。** 骨架已经在 FP32 里算好了 RMS 归一化：

```python
x_f32 = x.float()
variance = x_f32.pow(2).mean(-1, keepdim=True)      # 最后一维的均方值
x_norm = x_f32 * torch.rsqrt(variance + self.eps)   # x / sqrt(mean(x^2)+eps)
```

参考答案只有一行核心：

```python
output = x_norm * (1 + self.weight)   # 逐通道缩放，weight 初始化为 0
return output.type_as(x)              # 恢复输入 dtype
```

符号含义：`x_norm` 是归一化后的激活，形状与输入相同；`self.weight` 是长度为 `hidden_size` 的可学习向量，`__init__` 里用 `torch.zeros(hidden_size)` 初始化。乘以 `1 + weight` 后，初始时 `weight=0`，缩放因子恰好是 1，输出等于"不额外缩放"的 RMSNorm——这就是零初始化的意义：让新模块从恒等变换出发，不破坏训练初期的数值尺度。若照搬普通 RMSNorm 用全 1 初始化，初始缩放会变成 2，等于一上来就把每个通道放大一倍。

两个易错点：一是忘了 `+1`，直接写 `x_norm * self.weight`，零初始化会让输出恒为 0，测试 `test_gemma_rmsnorm_scale` 的第一条 `torch.allclose(norm(x), expected)` 立刻失败；二是忘了 `output.type_as(x)`，FP16 输入会返回 FP32，测试里 `norm(x.half()).dtype == torch.float16` 过不了。注意 RMS 本身始终在 FP32 里算，只有最后结果转回原 dtype，这是为了数值稳定。

**第 3 步：TODO 2——绑定的是同一个 `Parameter`。** 参考写法：

```python
shared_weight = self.embed_tokens.weight   # 取当前 Parameter 对象本身
self.lm_head.weight = shared_weight        # 让 Linear 的属性指向它
```

`nn.Embedding` 的 `weight` 形状是 `[vocab_size, hidden_size]`，`nn.Linear(hidden_size, vocab_size, bias=False)` 的 `weight` 形状正好也是 `[vocab_size, hidden_size]`，形状兼容才能直接赋值。关键在于赋值的是对象引用，不是 `shared_weight.clone()` 或 `shared_weight.detach().clone()`；后者只是数值副本，`data_ptr()` 不同，反向传播各算各的梯度。

为什么必须验证指针？`test_tied_parameter_identity_and_gradient` 做了三件事：`model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()` 证明存储同一块内存；对 embedding 权重 `add_(1.0)` 后 LM Head 权重也同步加 1，证明读写同步；对 `forward_lm_head(forward_embed(input_ids)).sum()` 反向，断言两边 `grad is not None` 且 `grad.data_ptr()` 相同，证明两条路径累积到同一份梯度。这就是"数值相同的两份拷贝不等于参数共享"的完整证据。

**第 4 步：TODO 3——词表参数账本。** 参考答案：

```python
embedding_params = vocab_size * hidden_size          # 一份 V×D 词表矩阵
lm_head_params = 0 if tied else embedding_params     # 绑定则 LM Head 不再单独持有
```

返回字典里 `total_params` 是两者之和，`saved_params` 在绑定时等于 `embedding_params`，否则为 0。用测试配置手算：`vocab_size=1000, hidden_size=64`，一份矩阵是 1000×64=64000；绑定后 `total_params=64000`、`saved_params=64000`；不绑定则 `total_params=128000`。换成真实模型的数量级，32k 词表、4096 隐藏维的一份矩阵就是 32000×4096≈1.31 亿参数，绑定省下的正是这一份。

输入检查在骨架里：`vocab_size <= 0 or hidden_size <= 0` 抛 `ValueError`，测试用 `(0, 64)` 和 `(1000, 0)` 两组非法值确认不会被静默接受。

**第 5 步：固定审计函数 `audit_model_config` 怎么读。** 它用 `model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()` 得到 `actual_tied`，再和 `config.get('tie_word_embeddings')` 比较：

- 任一字段缺失（`declared_tied is None or norm_scale_init is None`）→ `needs_checkpoint_review`，表示证据不全，不能下结论；
- 声明与实际相反 → `config_mismatch`；
- 一致 → `consistent`。

测试 `test_config_audit` 分别构造这三种配置，确认状态分类正确。这里的设计哲学是：缺字段不等于一致，必须显式核对。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 权重绑定（weight tying） | 输入词嵌入和输出 LM Head 共用同一个参数矩阵 |
| `Parameter` | PyTorch 里会被优化器更新、带梯度的张量对象 |
| `data_ptr()` | 张量实际内存块的首地址，地址相同才是真共享 |
| Gemma RMSNorm | 用 `1 + weight` 做逐通道缩放、weight 零初始化的 RMSNorm |
| 零初始化 | 权重全 0，使缩放因子初始为 1，模块从恒等变换起步 |
| 参数账本 | 按组件统计参数量，避免重复计算或漏算 |
| LM Head | 把隐藏状态投影回词表维度、产生 logits 的线性层 |
| `saved_params` | 绑定相对不绑定省下的参数数量，等于一份 `V×D` |
| `audit_model_config` | 对照配置声明与模块实际共享状态的审计函数 |
| 检查点语义 | 保存/加载时参数属于哪个模块、能否按预期恢复 |

## 小白常见疑问

**Q1：Gemma 的 norm 权重初始化为 0，输出真的不会全变成 0 吗？**
不会。公式是 `x_norm * (1 + weight)`，`weight=0` 时乘的是 1，输出就是归一化结果。测试第一条断言 `torch.allclose(norm(x), expected, atol=1e-4, rtol=1e-3)` 用的正是零权重；把 `weight.fill_(0.1)` 之后才断言输出"不再等于"原结果。

**Q2：我把 embedding 的权重 `clone()` 一份赋给 LM Head，数值完全一样，为什么不算绑定？**
因为参数身份不同。`clone()` 产生新内存，`data_ptr()` 不同，优化器会把它当两个参数分别更新；训练几步后数值就会分叉。`test_tied_parameter_identity_and_gradient` 检查的正是指针相等，而不是 `torch.equal`。

**Q3：绑定之后 `embed_tokens` 和 `lm_head` 还是两个模块，会不会重复计入参数量？**
模块对象仍是两个，但参数只有一份。`summarize_parameter_layout` 在 `tied=True` 时把 `lm_head_params` 记为 0，`total_params` 只算一份 64000；`audit_model_config` 也靠指针判断实际是否共享。

**Q4：`audit_model_config` 返回 `needs_checkpoint_review` 是什么意思？**
表示配置里缺少 `tie_word_embeddings` 或 `norm_scale_init`，无法判断一致性，需要回去检查检查点或配置，而不是默认它没问题。只有字段齐全且声明与实际相符才返回 `consistent`。

**Q5：这个 notebook 会验证绑定后模型效果更好吗？**
不会。它只验证参数身份、梯度归属和账本数字，不训练也不评测语言建模质量。绑定是否影响效果属于模型设计取舍，notebook 没有给出实验结论。

## 动手验证：怎么知道自己做对了

`run_architecture_trick_tests` 依次调用四个测试，全过才打印 `✅ 结构配置：缩放、参数共享、账本和审计测试通过`；TODO 未完成会抛 `NotImplementedError` 并提示先完成 TODO 1–3。

- `test_gemma_rmsnorm_scale`：hidden 16，输入 `torch.randn(2, 3, 16)`；零权重时 `torch.allclose(norm(x), expected, atol=1e-4, rtol=1e-3)`；权重填 0.1 后 `not torch.allclose(..., atol=1e-4)`；`norm(x.half()).dtype == torch.float16`。
- `test_tied_parameter_identity_and_gradient`：`QwenTieEmbeddings(vocab_size=32, hidden_size=8)`；两个权重 `data_ptr()` 相等；对 embedding 权重 `add_(1.0)` 后 LM Head 权重等于原值加 1；反向传播后两边梯度指针相同。
- `test_parameter_layout`：`(1000, 64, tied=True)` 必须得到 `embedding_params=64000, lm_head_params=0, total_params=64000, saved_params=64000`；`tied=False` 得到 `total_params=128000, saved_params=0`；`(0, 64)` 与 `(1000, 0)` 必须抛 `ValueError`。
- `test_config_audit`：同一模型分别传入声明 `True`、`False` 和缺 `norm_scale_init` 的配置，状态必须依次是 `consistent`、`config_mismatch`、`needs_checkpoint_review`。

notebook 没有保存执行输出，也没有 benchmark 单元格，因此以这些断言和测试打印为准；只要最后一行出现 ✅ 即表示三个 TODO 与固定审计逻辑都符合契约。
