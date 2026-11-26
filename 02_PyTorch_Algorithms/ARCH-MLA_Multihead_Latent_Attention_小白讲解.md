# ARCH-MLA. 多头潜在注意力（MLA）— 小白讲解

> 对应 notebook：[`ARCH-MLA_Multihead_Latent_Attention.ipynb`](./ARCH-MLA_Multihead_Latent_Attention.ipynb)

## 一句话总结
本节用同一组形状参数，分别计算 MHA、GQA 和简化 MLA 在「单层、单样本、单 token」下要保存多少字节，并给出相对 MHA 的比例；再用可选的 GPU 实验比较「先恢复完整 K/V」与「矩阵吸收」两条内容计算路径。核心结论是：在这组配置里，MLA 每 token 只存 256 个内容元素加 32 个位置元素（576 字节），是 MHA 的 `0.140625`，但「存得最少」不等于「算得最快」。

## 为什么需要它：痛点是什么
初学者常把「MLA 省 KV Cache」当成一句口号，却说不清省在哪：MLA 是像 GQA 那样让多个查询头共享 K/V，还是干脆不存 K/V？为什么 MLA 的缓存里既有 `latent_dim` 又有 `rope_dim`，这两份能不能合并？更常见的误区是看到缓存变小，就默认推理一定更快。

生活类比：MHA 像每个部门各存一整套会议记录；GQA 像几个部门合用一套；MLA 则不再存原始记录，而是存一份「压缩摘要」（潜在状态 latent）加一张「日期索引」（位置分量）。摘要更小，但每次要用都得按规则解压；索引也不能塞进摘要，因为旋转位置编码（RoPE）和任意压缩投影通常不能交换顺序。省纸是真的，解压要花时间也是真的。

## 核心问题
- MHA、GQA、MLA 每个 token 分别保存什么？为什么 MHA/GQA 的元素数要乘 2，MLA 不用？
- 测试里的 `4096`、`1024`、`576` 字节和 `1.0`、`0.25`、`0.140625` 三个比例是怎么算出来的？
- 位置分量为什么必须单独保存，不能并进低维内容状态？
- 「矩阵吸收」省掉了什么，又增加了什么计算？
- 为什么配置校验要求 `num_heads` 能被 `num_kv_heads` 整除？
- 「最紧凑」的表示一定「最快」吗？GPU 实验用哪些字段分别检查存储和计算？

## 解决思路：一步步看懂

**Step 1：三种表示，存的不是同一种东西。** notebook 的对照表把差异压缩成三行：MHA 为每个查询头保存各自的 K/V；MQA/GQA 让多个查询头共享较少的 K/V 头；MLA 则保存「低维内容状态 + 位置分量」，从「少存几组」变成「改变每个 token 保存的内容」。这一步先建立一个直觉：前两者的压缩发生在「头的数量」上，MLA 的压缩发生在「每个 token 的表示维度」上。

**Step 2：内容与位置分家，读取时再算。** MLA 缓存的内容状态是隐藏状态的低维投影；位置分量保留键中参与 RoPE 的部分。notebook 明确写出原因：位置旋转与任意压缩投影通常不能交换顺序，所以两部分要分别保存、分别计算，缓存字节数也要相加。读取时有两种做法：

1. 先由潜在状态投影恢复出完整 K/V，再走普通注意力；
2. 把 K/V 的投影矩阵「吸收」进查询投影和输出映射，直接在低维状态上算注意力——这就是矩阵吸收（matrix absorption）。

矩阵吸收省掉了完整历史 K/V 的生成，但增加或改变了其他计算，所以 notebook 要求「分别测量耗时」，而不是直接宣布更快。

**Step 3：三件事分开检查。** 存储字节数由状态形状和元素字节数算出；计算路径要确认「是否真的生成了完整历史 K/V」；输出与局部耗时要在相同内容状态和投影下比较。最后，完整模型的答题质量与线上请求性能仍要用真实权重和任务测量，不能由这个机制实验外推。

**Step 4：三个 TODO 与手算过程。**

TODO 1 在 `validate_attention_state_config` 里补两行：

```python
values = [config[key] for key in required]
heads_divisible = config['num_heads'] % config['num_kv_heads'] == 0
```

`required` 是 `('num_heads', 'num_kv_heads', 'head_dim', 'latent_dim', 'rope_dim', 'dtype_bytes')`，缺字段会先抛 `ValueError(f'missing fields: {missing}')`。补完后统一检查「所有值都是正整数」，再检查整除。为什么必须整除？因为 GQA 要把查询头均匀分给 K/V 头；`16 % 4 == 0` 可以，测试里故意改成 15，`15 % 4 == 3`，必须被拒绝。

TODO 2 在 `cache_bytes_per_token` 里写三个元素数：

```python
mha_elements = 2 * config['num_heads'] * config['head_dim']
gqa_elements = 2 * config['num_kv_heads'] * config['head_dim']
mla_elements = config['latent_dim'] + config['rope_dim']
```

乘 2 是因为 K 和 V 各存一份；MHA 的每个 token 每层要存 `num_heads × head_dim` 个 K 元素和同样多的 V 元素。MLA 不乘 2，是因为这个简化模型只存一份内容状态和一份位置分量，K/V 由它们恢复或通过吸收直接使用。函数最后按 `mode` 取元素数，未知模式抛 `ValueError`，再乘 `dtype_bytes`。

拿 `BASE_CONFIG` 手算一遍（`dtype_bytes=2`）：

- MHA：`2 × 16 × 64 = 2048` 个元素，`2048 × 2 = 4096` 字节；
- GQA：`2 × 4 × 64 = 512` 个元素，`512 × 2 = 1024` 字节；
- MLA：`256 + 32 = 288` 个元素，`288 × 2 = 576` 字节。

TODO 3 在 `compare_attention_states` 里复用上面的函数：

```python
bytes_by_mode = {mode: cache_bytes_per_token(mode, config) for mode in modes}
mha_bytes = bytes_by_mode['mha']
ratio_vs_mha = {mode: value / mha_bytes for mode, value in bytes_by_mode.items()}
most_compact = min(bytes_by_mode, key=bytes_by_mode.get)
```

比例手算：`4096 / 4096 = 1.0`；`1024 / 4096 = 0.25`；`576 / 4096 = 0.140625`。`min(..., key=...)` 返回字节数最小的模式名，所以 `most_compact` 是 `'mla'`。注意它只回答「本组形状下谁存得少」，不代表运行最快。

**Step 5：GPU 机制实验怎么读。** 默认 `RUN_MODE = 'dry_run'`，只打印配置；预检通过后才改成 `'real_gpu'`，调用 `tools/run_architecture_mechanism_benchmark.py`，模式为 `kv_representation`。固定的身份是 `SEMANTIC_ID = 'ARCH-MLA-REPRESENTATION-MECHANISM'`、`WORKLOAD_ID = 'arch-mla-representation-v2'`、`PROFILE_ID = 'ARCH-CUDA-MECHANISM'`，标签变化只影响结果文件名。实验自变量是「历史长度」（`512 / 2048 / 8192`）和「保存/计算方式」（MHA、GQA、低维状态重建、合并投影后的内容计算）；因变量是保存量、额外生成的 K/V、内容输出余弦相似度、平均耗时和峰值显存。默认宽度为查询头 16、KV 头 4、头维 64、内容维 256、位置维 64，BF16，预热 10 次、重复 100 次。

notebook 登记的 `rtx5070ti_mla_v3` 结果可以这样读：

- GQA 比 MHA 少 75%，低维状态比 MHA 少 84.375%。后者来自 GPU 配置的 `rope_dim=64`：`(256 + 64) × 2 = 640` 字节，`640 / 4096 = 0.15625`，所以节省 `1 − 0.15625 = 0.84375`。注意它与 CPU 测试的 576 字节/`0.140625` 不是同一组 `rope_dim`，不能混用。
- 两条内容路径的余弦相似度为 `0.999988–0.999993`，说明输出接近。
- 合并投影路径的耗时约为重建路径的 `1.9–2.0` 倍，长度 8192 时峰值约低 `30.5%`。省了峰值，但更慢。
- 最终判断是 `accept_latent_representation_and_absorbed_path_mechanism`，只说明表示与内容计算检查通过，完整模型质量和服务性能要另测。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MHA | 每个查询头都有自己的一套 K/V |
| MQA / GQA | 所有头 / 一组头共享一套 K/V，压缩的是头的数量 |
| MLA | 不按头存 K/V，改存低维内容状态加位置分量 |
| 潜在状态（latent） | 隐藏状态压到低维后的内容摘要 |
| 位置分量（rope_dim） | 单独保留、参与 RoPE 的那部分键 |
| RoPE | 旋转位置编码，给 K/Q 注入位置信息 |
| 矩阵吸收 | 把 K/V 投影合并进 Q 投影和输出映射，直接算低维注意力 |
| `cache_bytes_per_token` | 算单层、单样本、单 token 的历史状态字节数 |
| `ratio_vs_mha` | 各表示字节数除以 MHA 字节数 |
| `most_compact` | 字节数最小的模式名，只代表省存储，不代表最快 |
| 恢复完整 K/V | 先投影出 K/V 再走普通注意力的路径 |
| 余弦相似度 | 衡量两条内容路径输出方向是否接近 |

## 小白常见疑问

**Q1：MLA 的 576 字节比 GQA 的 1024 字节还小，明明 GQA 只有 4 个 KV 头？**
因为两者压的不是同一个维度。GQA 仍按「4 个头 × 每个头 64 维 × K 和 V 两份」存，所以是 `2 × 4 × 64 = 512` 个元素；MLA 只存一份 `latent_dim + rope_dim = 256 + 32 = 288` 个元素。前者压缩头数，后者压缩每 token 的表示本身。

**Q2：位置分量能不能也并进 latent，进一步省字节？**
notebook 明确说不能直接合并：位置旋转与任意压缩投影通常不能交换顺序，所以内容状态和位置分量要分别保存、分别计算。硬合并会改变计算语义，不是免费的压缩。

**Q3：矩阵吸收省掉了完整 K/V，为什么反而可能更慢？**
它省的是「生成完整历史 K/V」这一步，但把投影搬到了查询或输出侧，改变了其他计算的形状和次数。登记结果显示这组 PyTorch 实现里吸收路径耗时约为重建路径的 `1.9–2.0` 倍，所以「省存储」和「省时间」必须分开测。

**Q4：`0.140625` 是怎么从 576 得来的？**
`576 / 4096 = 0.140625`。分子是 MLA 的 `(256 + 32) × 2`，分母是 MHA 的 `2 × 16 × 64 × 2`。两个 2 的含义不同：分子是字节数里的 `dtype_bytes=2`，分母同时含 K/V 两份和 `dtype_bytes=2`。

**Q5：为什么测试把 `num_heads` 改成 15 就应当报错？**
GQA 要把查询头均匀分给 K/V 头。`16` 个查询头、`4` 个 KV 头可以每组 4 个；`15 % 4 = 3`，分不匀，所以 `validate_attention_state_config` 必须抛 `ValueError`。测试用 `try/except ValueError` 检查这一点。

## 动手验证：怎么知道自己做对了
`run_arch_mla_tests()` 依次跑三个测试，全部通过后打印 `✅ MLA：配置、状态字节数与表示对照测试通过`。

- `test_attention_state_config_contract`：`BASE_CONFIG`（16 头、4 个 KV 头、头维 64、latent 256、rope 32、2 字节）必须通过校验；把 `num_heads` 改成 15 后必须抛 `ValueError`，否则测试自己抛 `AssertionError("num_heads 不能被 num_kv_heads 整除时必须拒绝")`。
- `test_cache_bytes_per_token`：断言 `cache_bytes_per_token("mha", BASE_CONFIG) == 4096`、`("gqa", ...) == 1024`、`("mla", ...) == 576`。这三个数分别防「漏掉 V」「忘了 GQA 只存 4 个头」「漏掉位置分量」。
- `test_attention_state_comparison`：断言 `most_compact == "mla"`，`ratio_vs_mha` 三项分别为 `1.0`、`0.25`、`0.140625`。

dump 里没有保存 CPU 测试的输出，也没有执行 GPU 单元；5.4 的余弦相似度、`1.9–2.0` 倍耗时和 `30.5%` 峰值都是已登记的 `rtx5070ti_mla_v3` 历史结果，不是本次运行产生的。复跑时先看 `identity` 与 `failure`，再看 `cache_bytes` / `materialized_kv_bytes`，最后才看 `cosine_vs_reconstruction`、`mean_step_ms` 和 `peak_memory_mb`。
