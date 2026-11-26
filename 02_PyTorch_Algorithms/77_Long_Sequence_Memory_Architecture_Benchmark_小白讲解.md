# 77. 长序列记忆架构基准 — 小白讲解

> 对应 notebook：[`77_Long_Sequence_Memory_Architecture_Benchmark.ipynb`](./77_Long_Sequence_Memory_Architecture_Benchmark.ipynb)

## 一句话总结
本节先用纸笔公式估算三种长序列记忆方式要保存多少状态：显式 KV、固定形状的递推状态、以及两者混用的混合架构；再用 `validate_length_ladder_pair` 和 `decide_memory_candidate` 把"两组条件是否可比、每个长度档是否测全、所有档是否同时达标"变成可测试的规则。最后给出一个可选的 GPU 长度阶梯实验，用同一套记录格式采集答案、预填充耗时、单步解码耗时和峰值显存。

## 为什么需要它：痛点是什么
初学者常背两句口诀："Attention 的 KV Cache 随长度线性增长，SSM 的状态大小固定"。但真到选型时会卡住：固定状态是不是就一定能记住开头？显式 KV 更大是不是一定更慢？为什么两个模型都叫"长上下文模型"，跑同一个 8K 输入一个答对一个答错？只看架构名字，回答不了这些问题，因为状态大小只是"要存多少"，能不能找回信息、跑得多快、占多少显存是另外三件事。

生活类比：三个人读同一本长书。甲逐字抄笔记（显式 KV），想查哪页翻哪页，但笔记本越来越厚；乙只写一张固定大小的摘要卡（递推状态），本子不增厚，可摘要会漏掉细节；丙前几章抄笔记、后几章写摘要（混合架构），厚度取决于抄笔记的章节有多少。要比谁更适合"读完回答开头的问题"，不能只称本子重量，还得真的提问、计时、量桌面占用。

## 核心问题
- 给定层数、KV 头数、头维、状态维度，三种架构各要保存多少字节？长度翻倍时谁翻倍、谁不变？
- 两个模型的结果在什么条件下才能算"架构对照"，什么情况下只能"分别报告"？
- 每个长度档位需要哪些字段？`failure=None` 和指标为 `None` 有什么区别？
- 什么叫"所有长度都达标"？短档成功能否掩盖长档失败？
- GPU 实验里质量、耗时、显存分别怎么测？为什么单步解码耗时不能当成平均每词元耗时？

## 解决思路：一步步看懂

**Step 1：先固定比较条件，再谈架构。** notebook 的立场很明确：模型名、系列标签、"只改了记忆层"的声明都不能代替实际配置。`validate_length_ladder_pair` 先检查两组的 `candidate_id`、`model_id`、`revision`、`workload_id`、`tokenizer_revision` 是否齐全；再要求两组的 `workload_id` 和 `tokenizer_revision` 相同，否则直接记为 `gate_failures`，比较模式变成 `invalid`——连配对都成立不了。条件齐全后再看三组字段：训练条件（`data_revision`、`recipe_revision`、`train_tokens`、`seed`）、执行条件（`dtype`、`batch_size`、`backend`、`hardware`、`max_new_tokens`、`repeats`）、架构配置（`num_layers`、`hidden_size`、`memory_layers`）。前三组必须相同，`memory_layers` 必须不同，才算 `controlled`。

**Step 2：状态大小用形状算，不和实测显存混为一谈。** 设批大小 $B$、序列长度 $L$、注意力层数 $N_a$、KV 头数 $H_{kv}$、每头维度 $D_h$、递推层数 $N_r$、状态两个轴长 $D_s$ 与 $D_v$、每元素字节数 $s$。显式 KV 的字节数是

$$2 \times B \times L \times N_a \times H_{kv} \times D_h \times s$$

前面的 2 表示 K 和 V 各存一份；$L$ 在式子里，所以长度翻倍、KV 翻倍。递推状态是

$$B \times N_r \times D_s \times D_v \times s$$

没有 $L$，所以长度变化时它不变。`estimate_memory_state` 就是照这两个式子写的，返回 `kv_bytes`、`recurrent_state_bytes`、`total_state_bytes`，并标记 `evidence_level='cpu_state_ledger'`。

拿测试里的三组配置手算（`dtype_bytes=2`，`batch_size=1`）：
- 纯 KV：4 层、2 个 KV 头、头维 8，$L=16$ 时 $2\times1\times16\times4\times2\times8\times2=4096$ 字节；$L=32$ 时 8192 字节，正好 2 倍。
- 纯递推：4 层、状态 $8\times8$，$1\times4\times8\times8\times2=512$ 字节；$L=16$ 与 $L=32$ 都是 512。
- 混合：2 层 attention + 2 层递推，$L=32$ 时 KV 为 4096、递推为 $1\times2\times8\times8\times2=256$，合计 4352。

于是测试断言的不等式 $512 < 4352 < 8192$ 成立：递推最小，混合居中，纯 KV 最大。注意这只是"状态账本"，不含模型权重、激活、临时缓冲和 kernel workspace，所以不能拿它当峰值显存。

**Step 3：三种架构的合法性先校验。** `validate_memory_architecture` 要求 `architecture_type` 属于 `explicit_kv`、`recurrent_state`、`hybrid`；`attention_layers + recurrent_layers == num_layers`；纯 KV 不能有递推层，纯递推不能有注意力层，混合则两类层都必须大于 0；有注意力层就必须给正的 `kv_heads` 和 `head_dim`，有递推层就必须给正的 `state_dim` 和 `value_dim`。`test_architecture_contract` 把混合配置的注意力层改成 3（总数对不上）、改成 4+0（变成纯 KV）、改成 0+4（变成纯递推），三种都必须 `ready=False`。先校验再估算，可以避免用错误形状算出"看起来很合理"的字节数。

**Step 4：TODO 1——判断是否为受控比较。** 四个布尔量已经算好：`conditions_complete`（所有比较字段非空）、`training_matches`、`execution_matches`、`other_arch_matches`（除 `memory_layers` 外的架构字段都相同）、`memory_changed`（记忆层安排确实变了）。参考答案是：

```python
controlled = (conditions_complete and training_matches and execution_matches
              and other_arch_matches and memory_changed)
mode = 'invalid' if failures else 'controlled' if controlled else 'observational'
```

为什么必须要求 `memory_changed`？如果两组连记忆层都一样，那就不是"换架构"的对照，只是重复测量。容易写错的是只看 `training_matches`：`test_comparison_conditions` 会保留系列名，分别改数据版本、训练方案、训练词元数、随机种子、每一个执行条件、隐藏宽度，甚至把 `memory_layers` 改回一样，结果都必须是 `observational`；只有 `workload_id` 不同才是 `invalid`。`observational` 的意思是"两组可以各自报告，但不能把差异归因于记忆架构"。

**Step 5：TODO 2——逐档记录是否完整。** 骨架已经把每个长度的记录按 `rows[length]` 建好，并把缺失字段收集到 `missing_fields`。参考答案一行：

```python
row_complete = not missing_fields
```

关键在语义：`failure=None` 表示"执行成功"，这是合法值，但 `failure` 这个键必须存在；`quality_score`、`prefill_ms`、`decode_step_ms`、`peak_memory_mb` 若为 `None`，表示"没测到"，要进 `missing_evidence`。`test_length_coverage` 分别砍掉基线或候选的 4096 档、逐个删除 2048 档的五个字段，缺失项必须出现在 `missing_evidence`；把 `failure` 改成 `'oom'` 必须进 `gate_failures`；重复长度也必须报 `duplicate`。这些区分直接决定后面是"继续测"还是"直接排除"。

**Step 6：TODO 3——所有长度联合判断。** 每个长度有四项检查：`quality_score >= min_quality`、`prefill_ms <= max_prefill_ms`、`decode_step_ms <= max_decode_step_ms`、`peak_memory_mb <= max_peak_memory_mb`。`checks` 已按长度展开，答案是 `all_lengths_pass = all(checks)`。`all` 而不是 `any`：长档失败不能被短档成功掩盖。测试用的门槛是 `min_quality=0.88`、`max_prefill_ms=14.0`、`max_decode_step_ms=2.2`、`max_peak_memory_mb=1500`，fixture 的 4096 档刚好等于这四个边界（0.88、14.0、2.2、1500），因此必须判 `accept`——这说明比较符是闭区间。随后把最后一行依次改成 0.87、14.01、2.21、1501，每次都应变 `reject`。

**Step 7：决策优先级。** `decide_memory_candidate` 的顺序是：有 `gate_failures` 或 `comparison_mode == 'invalid'` → `reject`；有 `missing_evidence` → `tune`；阈值不过 → `reject`；指标全过但比较模式不是 `controlled` → `tune`；最后才 `accept`。这解释了测试里两个容易混的用例：把候选的 `training_conditions` 清空，指标仍然达标，但只能 `tune`；把基线的 `runs` 清空，则因为缺档而 `tune`。

**Step 8：可选 GPU 实验怎么设计、怎么读。** 固定配置是长度阶梯 `seq_lens=[256, 512, 1024]`、`batch_size=1`、`dtype='float16'`、`max_new_tokens=16`、`warmup=2`、`repeats=3`、needle 短语 `MEMORY_ARCH_NEEDLE_77`、协议 `prefill-single-decode/v2`。自变量是模型和长度；因变量有四项：质量（生成结果是否包含 needle，记 0/1）、预填充耗时、固定缓存上的单步解码耗时、峰值显存。`build_memory_arch_prompt` 把答案放在开头、问题放在末尾，中间用 `'The archive contains routine records. '` 填到目标长度；`gap = target_tokens - len(prefix) - len(question)`，`gap <= 0` 直接报错。`summarize_memory_trials` 对非 `None` 的耗时取中位数，峰值取所有窗口最大值，增量显存是 `max(peak - before, 0)` 再取最大，并统计 `completed_decode_trials`。读结果时注意两点：质量生成不计入耗时和显存窗口；单步解码只在固定缓存长度上测一次，不能当成整段生成的平均每词元耗时。notebook 的登记表明确写着"模型清单尚未填写，未采集本版 GPU 结果"，所以这里只讲方法和字段，不引用任何实测数字。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 显式 KV | 每个历史词元都存一份 Key 和 Value，查历史最直接，但随长度线性变厚 |
| 递推状态 | 把读过的内容压进固定形状的状态，长度不改变它的形状 |
| 混合记忆 | 一部分层用 KV、一部分层用递推状态，占用取决于用 KV 的层数 |
| 长度阶梯 | 同一任务按 256/512/1024 等档位逐步加长，看从哪档开始失败 |
| `controlled` | 训练、执行、其他架构条件都匹配，只有记忆层安排改变 |
| `observational` | 条件没配齐，只能分别报告两组表现，不能归因于架构 |
| `invalid` | 模型身份或请求集都对不上，连配对都不成立 |
| 状态账本 | 只按形状估算的字节数，不含权重、激活和临时缓冲 |
| 预填充 / 单步解码 | 一次处理整段输入 / 在已有缓存上再生成一个词元 |
| 增量峰值显存 | 本次窗口峰值减去窗口开始时的占用，排除常驻部分 |
| needle 检索 | 把答案放开头、问题放末尾，检查模型能否找回 |

## 小白常见疑问

**Q1：`estimate_memory_state` 算出的字节数，为什么不等于 `torch.cuda.max_memory_allocated`？**
因为公式只算 KV 或递推状态本身。实际显存还包括模型权重、前向激活、注意力临时张量和 kernel workspace；notebook 也把估算标成 `cpu_state_ledger`，把实测标成 `single_gpu_length_ladder_observational`。两者要分开报告。

**Q2：为什么 `failure=None` 不算缺失，而 `decode_step_ms=None` 算缺失？**
这是两种不同的事实。`failure=None` 是"这一档跑了且没失败"，键存在就说明有结论；`decode_step_ms=None` 是"这一项没测出来"，没有数值就无法判断资源门槛，所以进 `missing_evidence`。`test_length_coverage` 专门测了这两个分支。

**Q3：指标都达标了，为什么还可能是 `tune` 而不是 `accept`？**
因为 `accept` 要求两件事同时成立：所有长度过阈值，且 `comparison_mode == 'controlled'`。如果训练条件没配齐，指标再好也只能说"这个候选在当前任务上能跑"，不能说"这是记忆架构带来的收益"，所以先 `tune`，补匹配条件。

**Q4：为什么要求 `memory_layers` 必须不同？**
`memory_changed` 用来确认这次比较真的换了记忆层安排。`test_comparison_conditions` 里把候选的 `memory_layers` 改回和基线一样，结果必须是 `observational`——没有变化就没有可归因的自变量。

**Q5：单步解码耗时越小，是否就代表整段生成越快？**
不能直接等同。`decode_step_ms` 是在固定缓存长度上测的一次解码，整段生成时缓存会不断变长，每一步的成本不同。notebook 明确说"整段生成会不断增长缓存，因此这里不将它记为平均每词耗时"。

## 动手验证：怎么知道自己做对了
notebook 没有单独的 `run_*` 入口，测试单元按顺序直接调用五个函数，全过才打印 `通过：架构比较条件、双侧长度完整性与逐档阈值判断。`
- `test_architecture_contract`：三种合法配置 `ready` 为真；混合配置改成 3 个 attention 层、4+0、0+4 都必须 `ready=False`。
- `test_state_growth`：纯 KV 从 16 到 32 字节数翻倍；纯递推两档相等；满足 `512 < 4352 < 8192`。
- `test_comparison_conditions`：fixture 默认 `controlled`；改任一训练字段、任一执行字段、清空训练条件、改 `hidden_size`、把 `memory_layers` 改回一样，全部变 `observational`；改 `workload_id` 变 `invalid`。
- `test_length_coverage`：删长档或删任一指标都要出现在 `missing_evidence`；`failure='oom'` 和重复长度要出现在 `gate_failures`；`decode_step_ms=None` 要算缺失。
- `test_all_length_limits`：边界值全部 `accept`；四项分别越界 0.87 / 14.01 / 2.21 / 1501 都 `reject`；条件不全或缺档 `tune`；空的、重复的、含 0 的预期长度列表抛 `ValueError`。

notebook 的测试单元和 Step 5 的 `prepare` 分支都没有保存执行输出，所以以断言和打印结果为准，不要引用不存在的实测数值。
