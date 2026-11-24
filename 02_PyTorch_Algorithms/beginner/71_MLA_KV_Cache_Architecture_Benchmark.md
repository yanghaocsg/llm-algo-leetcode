# 71. MLA KV Cache Architecture Benchmark | MLA 与 KV Cache 结构基准 · 小白入门

> 配套 notebook：[71. MLA 与 KV Cache 结构基准](../71_MLA_KV_Cache_Architecture_Benchmark.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first；GPU/backend 可选

## 一句话说清

MLA 是一种把注意力的历史记录压小了再保存的办法。这本 notebook 教你算清它和常见做法 GQA 各要存多少状态，再判断一个 MLA 模型有没有资格进入下一步评测。

## 为什么需要它

注意力在生成时会为每个历史词元保存 K 和 V，也就是 KV Cache。序列一长，这部分显存越占越多。MLA 的思路是不直接存每个词元的 K 和 V，而是存一份维度更低的“潜在状态”。

省空间听起来很直接，但有三处容易下错结论：

- 只看理论压缩比例就说更快。状态账本只描述状态表示，不包含分配器、临时张量和内核工作区。实际显存和时间要靠后端测量。
- 拿来比的两个模型差得太多。模型不同，权重和结构也不同，结果只能算观察，差异不能全算在 MLA 头上。
- 证据不全就放行。质量没测、峰值显存缺测时，不该给出采用结论。

看 notebook 测试里的两个玩具模型。每个词元在每一层要保存的状态，GQA 是 8 个 KV 头乘 128 维，K 和 V 各一份，共 4096 字节；MLA 是 512 维潜在状态加 64 维位置分量，共 1152 字节。数字小了很多，但 notebook 的目标不是这个比值，而是一份可追溯的架构比较记录。

这本 notebook 在固定请求工作负载下，先核对两个模型能不能比，再算状态账本，最后用证据门槛给出 `accept`、`tune` 或 `reject`，判断 MLA 候选是否具备继续评测的条件。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| GQA（分组查询注意力） | 这里指常见的保存方式：每个词元在每层保存 `num_kv_heads` 个 KV 头，每个头 `head_dim` 维，K 和 V 各一份 |
| MLA（多头潜在注意力） | 用低维潜在状态代替 K/V 来保存注意力历史的方法 |
| 潜在状态 / 位置分量 | MLA 每个词元每层保存的两部分：`latent_dim` 维潜在状态，加上 `rope_dim` 维位置分量 |
| 状态账本 | 理论上每个词元要保存多少字节的状态。它只描述状态表示 |
| 受控比较 / 观察性比较 | 受控：两个模型同架构族、同训练方案，只改了状态表示。观察性：还有别的差别，差异说不清来自哪里 |
| 后端（backend） | 真正运行模型的执行栈，比如推理框架和它调用的算子 |
| 单词元时间（TPOT） | 解码时平均每个词元花的时间，记录在 `tpot_ms` |

## 核心思路

输入是两个模型的结构配置、一份工作负载，以及运行记录。四个 TODO 对应四步：

```text
比较契约（TODO 1）
  controlled（受控）    = 架构族相同 且 训练方案相同 且 唯一变化是状态表示
  否则                 = observational（观察性）

状态账本（TODO 2）
  GQA 每词元每层字节数 = 2 × H_kv × D_h × s
  MLA 每词元每层字节数 = (D_latent + D_rope) × s
  理论状态字节数       = B × L × N × 每词元每层字节数

证据门禁（TODO 3）
  有失败记录（含工作负载不一致、后端能力未确认）            → reject
  缺质量或峰值显存，或者比较是观察性的                     → tune
  质量低于 quality_min，或峰值显存超过 peak_memory_mb_max   → reject
  其余                                                   → accept

候选决策（TODO 4）
  只在 accept 候选里选：先比峰值显存，再比 TPOT，最后比模型 ID
```

`B` 是批量，`L` 是序列长度，`N` 是层数，`s` 是每个元素的字节数（`dtype_bytes`）。`H_kv` 是 KV 头数，`D_h` 是每头维度，`D_latent` 是潜在维度，`D_rope` 是位置分量的维度。两边的 `B`、`L`、`N` 相同，差别只在“每词元每层字节数”。

用测试里的玩具模型算一下，再看受控比较怎么判断：

```python
dtype_bytes, layers, seq_len = 2, 2, 1024

gqa_per_layer = 2 * 8 * 128 * dtype_bytes          # K 和 V 各一份：8 个 KV 头，每个 128 维
mla_per_layer = (512 + 64) * dtype_bytes           # 512 维潜在状态 + 64 维位置分量
gqa_bytes = gqa_per_layer * seq_len * layers
mla_bytes = mla_per_layer * seq_len * layers
print(gqa_per_layer, mla_per_layer)
print(gqa_bytes, mla_bytes)
print(mla_per_layer * 2048 * layers == 2 * mla_bytes)

same_family, same_recipe, only_representation_changed = True, False, True   # 训练方案不同
print('controlled' if same_family and same_recipe and only_representation_changed else 'observational')
```

```text
4096 1152
8388608 2359296
True
observational
```

前两行是测试里断言的两个字节数。第三行说明 MLA 的账本随长度线性增长，长度翻倍，字节数翻倍，GQA 也是这样。最后一行里训练方案不同，所以只能标记为观察性比较。

notebook 对结论留了余地。状态账本只说明容量随长度变化的方向，推不出内核是否命中、质量是否保持。候选排序先比峰值显存，表达的是本项目容量优先的目标，不代表所有部署都应采用同一顺序。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从模型资产到架构比较记录：身份核对、固定比较、证据汇总、项目决策四个阶段 | 读 |
| Step 2 | 分开解释状态容量与运行成本：结构配置、状态账本、后端结果各能确认什么、不能推出什么 | 读，对照上面的状态账本公式 |
| Step 3 | 四种状态各自的触发条件、是否继续和下一动作 | 读，对照上面的证据门禁 |
| Step 4 · TODO 1 | 架构族与训练方案相同、唯一声明变化为表示时是受控比较；否则是观察性比较 | 在 `validate_comparison_contract` 里写 4 行左右：判断架构族、训练方案是否相同，是否只改了表示，再得到 `comparison_mode` |
| Step 4 · TODO 2 | GQA 保存 K/V 两份状态；MLA 保存潜在状态与位置分量 | 在 `build_state_ledger` 里按两种表示分别写 `bytes_per_token_layer`，维度必须是正整数 |
| Step 4 · TODO 3 | 硬失败为 reject；缺证据或观察性比较为 tune；完整且过门槛才 accept | 在 `qualify_architecture_evidence` 里写 `status` 的分支：先看失败，再看缺失和比较模式，最后比门槛 |
| Step 4 · TODO 4 | 按峰值显存、TPOT、模型 ID 依次排序，保证结果稳定 | 在 `select_architecture_candidate` 里写 `selected` 这 1 行，用三元组作为排序键 |
| 测试 | 身份和请求条件变化、字节数与长度翻倍、完整、缺失和失败记录、候选排序 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 消费 `ARCH-MLA` 的机制证据，检查真实模型资产能否准入，通过后顺序跑 GQA 和 MLA 的后端对照 | 需要 GPU 和本地已缓存的 GQA、MLA 模型快照。默认是 `RUN_BACKEND_IF_ADMITTED = True`：准入通过后才启动；默认只使用本地资产，准入不通过时不会启动服务 |

Step 1–4 只需要 CPU。Step 5 是可选的真实环境部分，需要 GPU 和本地已缓存的模型快照，不会因为运行 notebook 而下载模型。

## 容易踩的坑

- **模型身份或请求条件不一致还继续比**：`model_id`、`revision`、`representation` 缺失，或者工作负载的五个字段有任何一个不同，都会进问题列表。测试里把 `max_tokens` 从 64 改成 32，问题列表里就有 `workload.max_tokens`。
- **把观察性比较当成受控比较**：训练方案不同时，`comparison_mode` 必须是 `observational`。观察性比较即使运行成功，也只能 `tune`。
- **维度不是正整数还往下算**：批量、长度、层数、字节数，以及 GQA 的 KV 头数和头维度、MLA 的潜在维度和位置分量维度，都必须是正整数，否则抛 `ValueError`。
- **把缺证据当成失败**：加载或请求失败是硬失败，直接 `reject`。缺质量或峰值显存只是 `tune`，等着补证据。
- **用理论账本推速度**：状态账本不包含分配器、临时张量和内核工作区，推不出实际显存峰值和执行速度。跨模型的差异也不能全部归因于 MLA。

## 读完后自查

- [ ] 能说出受控比较要同时满足的三个条件。
- [ ] 能写出 GQA 和 MLA 的理论状态字节数公式，并算出测试里两个玩具模型的值。
- [ ] 能说出 `reject`、`tune`、`accept` 各由什么触发。
- [ ] 能说出多个 `accept` 候选按什么顺序选。

## 下一步

- 打开 notebook：[71. MLA 与 KV Cache 结构基准](../71_MLA_KV_Cache_Architecture_Benchmark.ipynb)
- 建议先读：[04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA.md)、[ARCH-MLA. MLA 与紧凑 KV 表示 · 小白入门](./ARCH-MLA_Multihead_Latent_Attention.md)
- 需要补基础：[Part 01: 04. Attention Memory Optimization](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb)、[11. KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
- 做完之后：[77. 长序列记忆架构基准 · 小白入门](./77_Long_Sequence_Memory_Architecture_Benchmark.md)
