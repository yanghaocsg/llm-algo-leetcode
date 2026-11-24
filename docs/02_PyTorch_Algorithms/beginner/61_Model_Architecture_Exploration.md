# 61. Model Architecture Exploration | 模型架构审计 · 小白入门

> 配套 notebook：[61. 模型架构审计](../61_Model_Architecture_Exploration.md) ｜ 难度：Hard ｜ 环境：CPU-first；真实 checkpoint 配置审计可选

## 一句话说清

看到一份模型配置，怎样判断它和对照模型差在哪里、这个差别改变了什么、还缺哪些证据？这本 notebook 教你把一次结构改动写成一张审计卡，并给出下一步。

## 为什么需要它

大模型的配置文件里有一堆字段：隐藏维度、层数、注意力头数、前馈网络的宽度。两个模型的配置不一样，但你说不清它们到底差在哪里，更说不清这个差别带来了什么。

读配置时容易出三种错：

- 把好几个差异混在一起。两个模型的头数、宽度、层数都不同，就算一个的数值更好，也可能来自模型规模、训练数据等方面的差别，归不到某个结构字段上。
- 只看名词。配置里写了 GQA，不等于它和基线只差这一处。
- 拿估算当结论。按结构算出来的参数量、每个词元的 KV 字节数，只能说明“可能改变什么”，说明不了速度和质量。

看 notebook 测试里的例子：隐藏维度 16、2 层、4 个注意力头，把 KV 头数从 4 改成 2，就是从 MHA 改成 GQA。只改这一个字段，每个词元的 KV 字节数就从 128 降到 64。

这本 notebook 要做的是：确认候选只改了一个结构字段，算出它改变的参数和状态账本，写成一张审计卡，再按证据决定下一步是采用、继续调优，还是拒绝。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 基线 / 候选（baseline / candidate） | 基线是拿来对照的模型配置，候选是从基线改出来的另一份配置 |
| 结构档案（profile） | 从模型 config 里取出并校验过的一组结构字段，比如隐藏维度、层数、头数 |
| MHA / GQA / MQA | 三种注意力，由 KV 头数和注意力头数的关系区分：相等是 MHA，KV 头数是 1 是 MQA，其余是 GQA |
| `num_attention_heads` / `num_key_value_heads` | 注意力头数和 KV 头数。KV 头数更少，意味着多个注意力头共享 K/V |
| 账本（ledger） | 按结构字段算出来的参数量、权重字节数、每个词元的 KV 字节数 |
| KV Cache | 推理时为每个词元保存的 K 和 V，序列越长占得越多 |
| `accept` / `tune` / `reject` | 决策的三种结果：采用、继续调优、拒绝 |

## 核心思路

输入是一份基线配置和一份候选配置，都是字典。处理分五步：从配置里取出结构字段并校验，得到结构档案；核对候选是不是只改了一个字段；分别算出基线和候选的账本；把差分、账本和证据写成一张审计卡；最后按参数预算和证据状态给出决策。输出是审计卡，以及 `accept`、`tune` 或 `reject` 之一。

账本里和本节关系最大的几个量这样算，记隐藏维度为 `h`：

```text
head_dim               = hidden_size / num_attention_heads
attention_params       = num_hidden_layers * (h * h + 2 * h * kv_heads * head_dim + h * h)
mlp_params             = num_hidden_layers * 3 * h * intermediate_size
kv_bytes_per_token     = 2 * num_hidden_layers * kv_heads * head_dim * dtype_bytes
state_bytes_at_seq_len = kv_bytes_per_token * seq_len
```

`kv_heads` 是 `num_key_value_heads`。`attention_params` 括号里的三项依次是 Q、K 和 V、输出投影的权重，所以 KV 头数一变，这一项跟着变。`3` 是前馈网络里权重矩阵的个数。`dtype_bytes` 是每个元素的字节数，默认是 2；`kv_bytes_per_token` 开头的 2 表示 K 和 V 各一份；`seq_len` 是序列长度。账本里另外还记了词嵌入和归一化的参数量。

审计卡里的证据有五个字段：`workload`、`quality`、`cost`、`failure` 和 `evidence_level`（证据等级）。什么证据都没填时，等级是 `cpu_structure_ledger`，意思是只有 CPU 上算出来的结构账本。决策按顺序判断：结构不合法，或者参数增量超过预算，`reject`；有失败记录、证据不是同家族基准测试、质量或成本没通过，`tune`；都满足才 `accept`。

下面是最小例子，配置取自 notebook 的测试：

```python
config = {'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 4,
          'num_key_value_heads': 4, 'intermediate_size': 32, 'vocab_size': 100}

def attention_type(c):                          # 由注意力头数和 KV 头数的关系决定
    heads, kv = c['num_attention_heads'], c['num_key_value_heads']
    if min(c['hidden_size'], heads, kv) <= 0 or c['hidden_size'] % heads or heads % kv:
        return None                             # 维度不合法
    return 'MHA' if kv == heads else ('MQA' if kv == 1 else 'GQA')

def ledger(c):                                  # 只算注意力参数和每个词元的 KV 字节数
    h, layers, kv = c['hidden_size'], c['num_hidden_layers'], c['num_key_value_heads']
    d = h // c['num_attention_heads']
    attention_params = layers * (h * h + 2 * h * kv * d + h * h)
    return attention_params, 2 * layers * kv * d * 2      # 末尾的 2：每个元素 2 字节

for kv_heads in (4, 2, 1, 3):
    c = {**config, 'num_key_value_heads': kv_heads}
    print(kv_heads, attention_type(c), ledger(c) if attention_type(c) else None)

candidate = {**config, 'num_key_value_heads': 2}
fields = ('num_key_value_heads', 'intermediate_size', 'num_hidden_layers')
print([k for k in fields if config[k] != candidate[k]])
```

```text
4 MHA (2048, 128)
2 GQA (1536, 64)
1 MQA (1280, 32)
3 None None
['num_key_value_heads']
```

前四行依次是 KV 头数、注意力类型、账本里的两个量。KV 头数等于注意力头数 4 时是 MHA；改成 2 是 GQA，注意力参数从 2048 降到 1536，每个词元的 KV 字节数从 128 降到 64；改成 1 是 MQA，两个量再降。KV 头数为 3 时，4 不能被 3 整除，类型是 `None`，也就是结构档案不合法。最后一行是单变量检查：候选和基线只在 `num_key_value_heads` 上不同，可以通过。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从模型配置提出一个可检验的结构问题：基线、单变量候选、固定条件和证据出口各要写清楚什么 | 读 |
| Step 2 | 把结构差异落到参数、状态与访问接口：改 KV 头数、FFN 宽度、访问范围、状态表示各改变哪本账 | 读，对照上面的词表 |
| Step 3 | 区分结构假设、真实配置与专项性能证据：四个证据等级各能回答什么、不能推出什么 | 读 |
| Step 4 · TODO 1 | 提取并校验 Decoder profile | 写 `extract_architecture_profile`：从 config 取出结构字段，检查维度为正、头数能整除，再判断 MHA、GQA 或 MQA |
| Step 4 · TODO 2 | 核对候选是否是单变量结构改动 | 写 `validate_single_variable_candidate`：找出真实不同的字段，必须恰好一个，且等于声明的 `changed_field` |
| Step 4 · TODO 3 | 建立参数与状态账本 | 写 `build_architecture_ledger`：分别算注意力、MLP、词嵌入、归一化的参数量和每个词元的 KV 字节数 |
| Step 4 · TODO 4 | 生成带专项证据字段的审计卡 | 写 `build_architecture_audit_card`：整理证据记录和所需的专项项目，证据等级默认是 `cpu_structure_ledger` |
| Step 4 · TODO 5 | 根据预算和证据状态给出下一步 | 写 `recommend_audit_action`：按结构、预算、失败记录、证据等级、质量和成本的顺序判断 |
| 测试 | 检查结构档案的类型判断和非法维度、单变量检查、GQA 和 MHA 的账本差异、审计卡指向的专项项目和决策 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 只读取两个真实 checkpoint 的 config，生成审计卡并保存 JSON，再登记专项项目的结果 | 需要配好两个 checkpoint 的 ID；开关由环境变量决定，`RUN_ARCH_CONFIG_AUDIT` 未设置时为 `0`，不执行 |

Step 1–4 只需要 CPU。Step 5 是可选的真实 checkpoint 配置审计，只读 config，不加载权重、不测吞吐；开关由环境变量决定，`RUN_ARCH_CONFIG_AUDIT` 未设置时为 `0`。

## 容易踩的坑

- **一次改了不止一个字段**：候选必须只改一个结构字段，声明的 `changed_field` 要和真实差分一致。改了两处时审计卡不成立，应该拆分候选，而不是换一个结论。
- **头数不能整除**：`hidden_size` 要能被注意力头数整除，注意力头数要能被 KV 头数整除，否则结构档案不合法。测试里的 KV 头数 3 就不合法。
- **把结构账本当成性能结论**：账本只能说明“可能改变什么”。只有 CPU 结构账本时，决策最多是 `tune`；要 `accept`，需要同家族、同工作负载的基准测试，而且质量和成本都通过。
- **漏看预算和失败记录**：参数增量超过 `max_param_delta` 时直接 `reject`；证据里有失败记录时是 `tune`，不能 `accept`。
- **config 缺字段时直接崩**：必填字段缺失时，要返回 `ready` 为 False 和问题列表。`num_key_value_heads` 没写时，按等于注意力头数处理。

## 读完后自查

- [ ] 能说出 KV 头数和注意力头数是什么关系时，分别是 MHA、GQA、MQA。
- [ ] 给一个小 config，能算出每个词元的 KV 字节数。
- [ ] 能判断一个候选是不是单变量改动。
- [ ] 能说出 `accept` 需要同时满足哪几个条件。

## 下一步

- 打开 notebook：[61. 模型架构审计](../61_Model_Architecture_Exploration.md)
- 建议先读：[05. LLaMA3 Block 教程 · 小白入门](./05_LLaMA3_Block_Tutorial.md)、[08. 模型配置与参数归属 · 小白入门](./08_Architecture_Tricks.md)、[ARCH-MLA. MLA 与紧凑 KV 表示 · 小白入门](./ARCH-MLA_Multihead_Latent_Attention.md)、[ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构 · 小白入门](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md)
- 需要补背景：[大模型架构 · 代表模型与结构对照](../../topic_discussion/llm_architecture_evolution/08_representative_models.md)
- 做完之后：[64. SFT 数据质量项目 · 小白入门](./64_SFT_Data_Quality_Project.md)
