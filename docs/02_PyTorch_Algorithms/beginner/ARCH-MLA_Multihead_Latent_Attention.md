# ARCH-MLA. Multihead Latent Attention | MLA 与紧凑 KV 表示 · 小白入门

> 配套 notebook：[ARCH-MLA. MLA 与紧凑 KV 表示](../ARCH-MLA_Multihead_Latent_Attention.md) ｜ 难度：Hard ｜ 环境：CPU-first + 可选单卡 GPU

## 一句话说清

模型每生成一小段文本，都要把前面的信息缓存下来，而且越写缓存越大。这本 notebook 用同一组配置给三种缓存方式算账：每种占多少字节，谁最省。

## 为什么需要它

模型一个词元一个词元往外生成时，前面每个词元的键和值都要缓存下来，供后面的词元读取。这份缓存叫 KV Cache，序列越长它越大。

同样是缓存，保存的办法不止一种：

- MHA：每个查询头都配一份自己的 K 和 V。
- MQA / GQA：让查询头共用较少的 K/V 头。
- MLA：把历史内容写进一个较低维的潜在状态，位置相关的部分单独保存。

用 notebook 测试里的配置算一下：16 个查询头、4 个 K/V 头、单头维度 64、潜在维度 256、位置维度 32、每个数 2 字节。每个词元的历史状态，MHA 是 4096 字节，GQA 是 1024 字节，MLA 是 576 字节。

字节变少只是账面上的事。读取状态的办法也跟着变了：MLA 需要投影、吸收或重建的路径，理论字节下降不自动代表端到端更快。这本 notebook 负责算清 CPU 上的形状与字节账本：先校验配置合法，再算三种表示每个词元的字节数，最后给出相对 MHA 的比例和最紧凑的一种。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| KV Cache | 推理时缓存下来的每个历史词元（token）的 K 和 V |
| 查询头 / K/V 头（Head） | 注意力里并排工作的若干组。MHA 里两种头一样多，MQA 和 GQA 里 K/V 头更少 |
| MHA / MQA / GQA | 多头注意力，以及它两种共享 K/V 头的变体。靠减少 K/V 头数来省缓存 |
| MLA | 多头潜在注意力。压缩的办法不是减少 K/V 头，而是把历史内容压进低维的潜在状态 |
| 潜在状态（latent） | 隐藏状态经过低维投影得到的向量，长度是 `latent_dim` |
| RoPE 分量 | 单独保存的位置相关的键表示，给 RoPE（旋转位置编码）用，长度是 `rope_dim` |
| `dtype_bytes` | 一个数占几个字节。本节测试里是 2 |

## 核心思路

输入是一组配置：查询头数、K/V 头数、单头维度、潜在维度、RoPE 维度、每个数的字节数。整个账本分三步：

```text
1. 校验：六个字段齐全，都是正整数，查询头数能被 K/V 头数整除
2. 记账：先数每个词元要存多少个数，再乘 dtype_bytes 得到字节数
   MHA 字节数 = 2 × num_heads × head_dim × dtype_bytes       （K 和 V 各一份）
   GQA 字节数 = 2 × num_kv_heads × head_dim × dtype_bytes
   MLA 字节数 = (latent_dim + rope_dim) × dtype_bytes        （潜在状态 + 位置分量）
3. 对照：每种都除以 MHA 的字节数得到比例，再挑出字节数最小的一种
```

`num_heads` 是查询头数，`num_kv_heads` 是 K/V 头数，`head_dim` 是每个头里向量的长度。MHA 和 GQA 都同时保存 K 和 V，所以有系数 2。MLA 把每个历史词元的缓存拆成两部分：潜在状态承载可压缩的内容，RoPE 分量保留位置信息。两部分分开记账，因为位置旋转经过任意低维压缩后不再必然等价。

用一组很小的配置走一遍：4 个查询头、2 个 K/V 头、单头维度 2、潜在维度 3、RoPE 维度 1、每个数 2 字节。

```python
cfg = {'num_heads': 4, 'num_kv_heads': 2, 'head_dim': 2,
       'latent_dim': 3, 'rope_dim': 1, 'dtype_bytes': 2}

mha = 2 * cfg['num_heads'] * cfg['head_dim']      # K 和 V 各一份，所以乘 2
gqa = 2 * cfg['num_kv_heads'] * cfg['head_dim']   # 只保存较少的 K/V 头
mla = cfg['latent_dim'] + cfg['rope_dim']         # 潜在状态 + 位置分量
print(mha, gqa, mla)                              # 每个词元的元素数

bytes_by_mode = {k: v * cfg['dtype_bytes'] for k, v in {'mha': mha, 'gqa': gqa, 'mla': mla}.items()}
print(bytes_by_mode)
print({k: v / bytes_by_mode['mha'] for k, v in bytes_by_mode.items()})
print(min(bytes_by_mode, key=bytes_by_mode.get))

print(cfg['num_heads'] % cfg['num_kv_heads'] == 0, 3 % 2 == 0)   # 查询头数能否被 K/V 头数整除
```

```text
16 8 4
{'mha': 32, 'gqa': 16, 'mla': 8}
{'mha': 1.0, 'gqa': 0.5, 'mla': 0.25}
mla
True False
```

第一行是每个词元要存的元素数。MHA 存 4 个头的 K 和 V，每个头 2 个数，2 × 4 × 2 = 16。GQA 只存 2 个 K/V 头，是 8。MLA 存 3 + 1 = 4。第二行是乘上每个数 2 字节之后的字节数，第三行是相对 MHA 的比例，第四行是最紧凑的 `mla`。最后一行是整除检查：4 能被 2 整除，3 不能。

这本 notebook 里的 MLA 是简化版，只记潜在状态和位置分量两部分。读取状态时需要的投影、吸收或重建路径，留给 Step 5 的 GPU 实验去比较。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | MHA、MQA / GQA、MLA 三条路径各保存什么、靠什么压缩、读取时要付出什么 | 读 |
| Step 2 | MLA 把缓存拆成潜在状态和位置分量，两部分各从哪里来、读取时需要什么 | 读 |
| Step 3 | 训练和推理的差别，以及从 CPU 字节账本到后端基准，各层证据能回答什么 | 读 |
| Step 4 · TODO 1 | 校验表示配置 | 在 `validate_attention_state_config` 里写 2 行：取出六个字段的值 `values`，再算查询头数能否被 K/V 头数整除 `heads_divisible` |
| Step 4 · TODO 2 | 计算历史状态字节数 | 在 `cache_bytes_per_token` 里写 3 行：`mha_elements`、`gqa_elements`、`mla_elements` |
| Step 4 · TODO 3 | 生成同口径表示对照 | 在 `compare_attention_states` 里写 4 行：三种模式的字节数 `bytes_by_mode`、相对 MHA 的比例 `ratio_vs_mha`、最紧凑的模式 `most_compact` |
| 测试 | 检查字段缺失、非法维度和不能整除的配置被拒绝；三类字节数；比例、排序和同一数据类型下的对照 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上对照 MHA、GQA、潜在状态重建和吸收式路径的状态量、时间、峰值显存和输出一致性 | 需要单张 CUDA GPU 和 `ARCH-CUDA-MECHANISM` 环境；默认 `RUN_MODE = 'dry_run'`，只登记计划 |

Step 1–4 只需要 CPU。Step 5 是可选实验，需要单张 CUDA GPU 和 `ARCH-CUDA-MECHANISM` 环境；配置单元里 `RUN_MODE` 默认是 `'dry_run'`，预检通过后才能改成 `real_gpu`。

## 容易踩的坑

- **只检查了字段在不在**：六个字段齐全还不够，取值必须都是正整数，而且查询头数要能被 K/V 头数整除。测试用 15 个查询头配 4 个 K/V 头，确认这种配置会被拒绝。
- **MHA、GQA 忘了乘 2**：两种方式都同时保存 K 和 V，元素数里有一个系数 2。MLA 没有这个系数，它存的是潜在状态加位置分量。
- **MLA 只记了潜在状态**：位置分量也要单独记一笔，MLA 的元素数是 `latent_dim + rope_dim`。
- **用不同的数据类型做对比**：三个候选只有在同一数据类型下才能比较字节与相对比例。
- **把字节变少当成速度变快**：这本 notebook 的账本属于“CPU 形状与字节账本”这一层，只能说明历史表示更紧凑。理论字节下降不自动代表端到端更快。

## 读完后自查

- [ ] 给一组配置，能写出 MHA、GQA、MLA 三种表示每个词元的字节数公式。
- [ ] 能说出 MLA 的缓存拆成了哪两部分，为什么要分开记账。
- [ ] 能判断一组配置合不合法，比如查询头数不能被 K/V 头数整除时会怎样。
- [ ] 能说出为什么理论字节下降不等于端到端更快。

## 下一步

- 打开 notebook：[ARCH-MLA. MLA 与紧凑 KV 表示](../ARCH-MLA_Multihead_Latent_Attention.md)
- 建议先读：[04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA.md)
- 需要补基础：[Part 01 · 11 KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- 做完之后：[ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构 · 小白入门](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md)
