# 08. Architecture Tricks | 模型配置与参数归属 · 小白入门

> 配套 notebook：[08. 模型配置与参数归属](../08_Architecture_Tricks.md) ｜ 难度：Easy ｜ 环境：CPU-first

## 一句话说清

模型的配置文件写着各种结构选择，它们到底怎样改变模型？这本 notebook 动手复现两个小例子：输入和输出两端是否共用同一份参数，以及归一化层的缩放初值。

## 为什么需要它

真实模型的配置文件里有很多开关，比如“输入端的词嵌入和输出端的输出头（LM Head）是否共享参数”“归一化的缩放怎样初始化”。只记住字段名还不够，还要知道它在模型里对应哪个对象、改变了什么。看一个具体的例子：词表里有 4 个词元，每个词元用 2 个数表示，一份词表矩阵就是 4 × 2 = 8 个参数。共享时只存这 8 个；不共享时，输出头再存一份 8 个，一共 16 个。配置里的一个开关，差别就是这些参数，以及训练时谁更新谁。

归一化的缩放也有约定。这本 notebook 的写法把缩放参数初始化成全 0，乘的是 `1 + weight`。如果写成直接乘 `weight`，输出全是 0。

这本 notebook 的做法是把配置声明对应到真实的模块对象和参数账本：亲手写一遍共享和缩放，再用一个固定的审计函数对照“配置声称的”和“模型实际的”。它只处理这两个局部选择，不涉及解码器块的主路径。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 词元（token） | 模型处理文本的最小单位。`V` 是词表里的词元个数，`D` 是每个词元对应向量的长度 |
| 词嵌入（embedding） | 模型输入端的查找表，每个词元编号对应一行向量，代码里叫 `embed_tokens` |
| 输出头（LM Head） | 模型输出端的投影层，把内部状态投影回词表，代码里叫 `lm_head` |
| 权重绑定（weight tying） | 让词嵌入和输出头用同一份参数 |
| 参数账本 | 数清楚某种结构下有哪些参数、各多少个，比如一份 `V × D` |
| `Parameter` | PyTorch 里可训练的权重。两个模块引用同一个 `Parameter` 才算共享，可以用 `data_ptr()` 查底层存储的地址来检查 |
| `dtype` | 数据类型。FP32 是 `float32`，FP16 是 `float16` |

## 核心思路

先看权重绑定。输入是词元编号，词嵌入拿编号去查表；模型处理完之后，输出头把内部状态投影回词表。绑定前，两边各有一份 `V × D` 的矩阵。绑定只要一行：让 `lm_head.weight` 指向 `embed_tokens.weight`。之后反向传播时，查表和投影两条路径算出的梯度会累加到这同一份参数上。

再看归一化缩放。对输入 `x` 做三步：

```text
x_norm = x_f32 * rsqrt(mean(x_f32^2) + eps)   # 1. 归一化，内部用 FP32 计算
out    = x_norm * (1 + weight)                # 2. 逐通道缩放
输出   = out.type_as(x)                       # 3. 转回输入的 dtype
```

`x_f32` 是转成 FP32 的输入，`eps` 是很小的正数（这里是 `1e-6`），`weight` 是初值全 0 的可学习参数，`rsqrt` 是先开根号再取倒数。

下面把两件事各走一遍：

```python
import torch
import torch.nn as nn

# 例 1：共用一份参数，还是各拿一份
V, D = 4, 2                                   # 词表 4 个词元，每个词元用 2 个数表示
embed = nn.Embedding(V, D)
head = nn.Linear(D, V, bias=False)
print(head.weight.data_ptr() == embed.weight.data_ptr())   # 刚建好：两个独立矩阵

head.weight = embed.weight                    # TODO 2 的做法：绑定
with torch.no_grad():
    embed.weight[0] = torch.tensor([10.0, 20.0])
print(head.weight.data_ptr() == embed.weight.data_ptr(), head.weight.detach()[0])

copy = nn.Parameter(embed.weight.detach().clone())         # 数值相同的拷贝
print(torch.equal(copy, embed.weight), copy.data_ptr() == embed.weight.data_ptr())

for tied in (True, False):                    # TODO 3 的账本
    emb = V * D
    print(tied, emb, 0 if tied else emb, emb + (0 if tied else emb), emb if tied else 0)

# 例 2：乘的是 1 + weight
x = torch.tensor([[3.0, 4.0]])
x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
w0 = torch.zeros(2)
w1 = torch.tensor([0.0, 1.0])
print(x_norm)
print(x_norm * w0)
print(x_norm * (1 + w0))
print(x_norm * (1 + w1))
```

```text
False
True tensor([10., 20.])
True False
True 8 0 8 8
False 8 8 16 0
tensor([[0.8485, 1.1314]])
tensor([[0., 0.]])
tensor([[0.8485, 1.1314]])
tensor([[0.8485, 2.2627]])
```

例 1 里，刚建好时两份矩阵的地址不同。绑定后地址相同，改 `embed.weight` 的第 0 行，`head.weight` 跟着变。`clone()` 出来的拷贝数值完全相同，地址却不同，所以不算共享。账本每行五个数依次是：是否绑定、词嵌入参数数、输出头参数数、总参数数、节省的参数数。例 2 里，直接乘全 0 的 `weight`，输出全是 0；乘 `1 + weight` 时，结果和归一化本身一样；第二个通道的 `weight` 变成 1，这个通道才放大到 2 倍。

最后是审计。固定函数 `audit_model_config` 把配置里的声明（`tie_word_embeddings`）和实际的地址比较结果放在一起：两者一致返回 `consistent`，相反返回 `config_mismatch`，配置里缺字段返回 `needs_checkpoint_review`。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 配置里的每一类选择对应模型里的哪个对象、带来什么直接后果 | 读 |
| Step 2 | 权重绑定后参数怎么存、梯度归谁 | 读，对照上面例 1 |
| Step 3 | 结构审计要回答的三件事：配置声称什么、实际是什么、改变了什么 | 读 |
| Step 4 · TODO 1 | 应用 Gemma 缩放 | 在 `GemmaRMSNorm.forward` 里写 1 行：`output = x_norm * (1 + self.weight)` |
| Step 4 · TODO 2 | 绑定共享参数 | 在 `QwenTieEmbeddings.__init__` 里写 1 行：取 `self.embed_tokens.weight` |
| Step 4 · TODO 3 | 计算词表参数账本 | 在 `summarize_parameter_layout` 里写 2 行：`embedding_params`，以及绑定时为 0 的 `lm_head_params` |
| 测试 | 检查零和非零缩放、FP16 输出的数据类型、参数和梯度的身份、两种配置的账本与非法维度、配置声明不一致 | 运行 |
| 参考代码与解析 | 参考答案、每个 TODO 的解释，以及固定审计函数的说明 | 卡住再看 |

整本 notebook 只需要 CPU。

## 容易踩的坑

- **绑定成了数值拷贝**：`shared_weight` 必须是 `embed_tokens` 当前持有的那个 `Parameter`，不能是 `clone()` 出来的张量。数值相同不代表共享，测试用 `data_ptr()` 检查参数和梯度的身份。
- **缩放直接乘了 `weight`**：这里 `weight` 的初值是 0，要乘的是 `1 + weight`。这样零初始化不改变归一化结果，非零权重才产生逐通道调整。
- **输出没有转回输入的 `dtype`**：内部用 FP32 计算，最后要用 `.type_as(x)` 转回去。测试会把输入转成 FP16，再检查输出的数据类型。
- **账本记错了份数**：词嵌入始终持有一份 `V × D`。绑定时输出头不再占参数，`saved_params` 正好是一份 `V × D`；不绑定时是两份，节省为 0。
- **维度不是正数还继续算**：`vocab_size` 或 `hidden_size` 不大于 0 时，要抛出 `ValueError`。

## 读完后自查

- [ ] 能说出绑定和不绑定两种情况下，词表矩阵各存几份，`saved_params` 各是多少。
- [ ] 能解释为什么数值相同的两份拷贝不算参数共享，测试用什么来判断。
- [ ] 能说出 `GemmaRMSNorm` 的 `weight` 全为 0 时，输出等于什么。
- [ ] 能说出结构审计要回答的三件事，以及 `config_mismatch` 在什么时候出现。

## 下一步

- 打开 notebook：[08. 模型配置与参数归属](../08_Architecture_Tricks.md)
- 建议先读：[01. RMSNorm 教程 · 小白入门](./01_RMSNorm_Tutorial.md)、[05. LLaMA3 Block 教程 · 小白入门](./05_LLaMA3_Block_Tutorial.md)
- 需要补基础：[PyTorch nn.Module 基础](../../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- 做完之后：[44. 局部、滑动窗口与稀疏注意力 · 小白入门](./44_Local_Sliding_and_Sparse_Attention.md)
