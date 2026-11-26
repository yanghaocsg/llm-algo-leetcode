# 24. Prefix Cache 匹配与复用 — 小白讲解

> 对应 notebook：[`24_Prefix_Cache_Matching_and_Reuse.ipynb`](./24_Prefix_Cache_Matching_and_Reuse.ipynb)

## 一句话总结
本节用一个不存真实 K/V 的 `PrefixCacheManager`，把「前缀缓存命中」拆成五个可以单测的小机制：词元规范化、完整前缀登记与去重、从请求开头算最长命中、按命中长度拆分前缀/后缀、以及统计命中词元与复用比例。它想说明的核心结论是：**只有曾经完整登记过的前缀对象才能命中，公共词元再多也不算；命中之后的工作量是"整个输入减去命中前缀"，而不是零。**

## 为什么需要它：痛点是什么
多轮对话、Agent 工具调用、批量评测常有一段完全相同的开头：系统提示词、工具说明、few-shot 示例、前几轮历史。如果每个请求都把这段重新跑一遍预填充（Prefill），同一段文字对应的 KV Cache 就被反复算、反复占显存。初学者常见的两个误解正好相反：

- 以为「只要两段文本有相同字符就能复用」——实际上必须从第一个词元开始连续相同，中间某段相同不算。
- 以为「命中缓存后这个请求就不用算了」——实际上只有命中那一段能跳过，第一个不一致词元之后的**后缀**仍要预填充。

生活类比：去银行办业务，前面那位客户已经填过一张「开户资料表」。你办的业务如果和他是同一套表格（从头到尾完全一样的模板），柜员可以直接复印那份已审核的底稿，你只需要补写自己不同的那几栏。但如果你只是中间某一栏碰巧一样，柜员不能把两张表拼起来用；而且复印底稿也不等于你的业务当场办完，剩下的栏位照样要填。

## 核心问题
- 为什么前缀缓存只认「从请求开头连续相同」的部分？中间子串相同为什么不行？
- 为什么只登记了 `[1,2,3]` 时，请求 `[1,2,9]` 明明有两个公共词元，命中长度却是 0？
- 「最长命中」是在哪些候选里选最长？命中长度和可复用 KV 状态是什么关系？
- 命中之后，`split_prompt` 如何保证前缀 + 后缀严格还原原输入？
- `cache_stats` 的三个字段分别代表什么？为什么空输入要单独处理？
- 这套逻辑账本为什么不等于真实的显存节省或延迟收益？

## 解决思路：一步步看懂

### 第 1 步：先分清四个概念
notebook 的 Step 1 用一张表把请求路径拆开：

| 概念 | 含义 | 本次请求怎么处理 |
|---|---|---|
| 请求输入 | 可能带重复系统提示词、工具说明、历史 | 与已登记路径比较 |
| 共享前缀 | 从开头连续相同的词元区间 | 复用对应 KV Cache |
| 未命中后缀 | 第一个不一致词元之后的全部内容 | 继续预填充，产生新状态 |
| 新状态 | 后缀算出的新增 KV Cache | 写回缓存或进入解码 |

这里的关键是「从开头连续」五个字。KV Cache 是按位置顺序累积的：第 $i$ 个词元的 K/V 依赖它前面所有词元。所以只有前 $k$ 个词元完全一致，前 $k$ 个位置的 K/V 才和缓存里的一模一样；如果第 3 个词元就不同了，第 4、5 个词元即使碰巧相同，它们对应的 K/V 也已经建立在不同的上下文上，不能拿来复用。

### 第 2 步：数据结构——只登记「完整对象」
`PrefixCacheManager.__init__` 只维护一个列表：

```python
self.cached_prefixes: List[Tuple[int, ...]] = []
```

注意元素类型是**元组**，不是列表。元组不可变、可哈希，适合当"登记条目"；列表则用来接收调用方传进来的任意 `Sequence[int]`。整个类不存任何 K/V 张量，只做逻辑索引，所以 notebook 反复强调「本题不实现真实 KV Tensor、物理 block 或淘汰策略」。

### 第 3 步：TODO 1 `_normalize`——统一容器并校验类型
要求：保持顺序、统一成列表、拒绝非整数。

```python
normalized = list(tokens)                       # 任何 Sequence 都转成 list，顺序不变
if any(not isinstance(token, int) for token in normalized):
    raise TypeError('tokens 必须是整数序列')
return normalized
```

为什么先转 `list` 再检查？因为入参可能是列表、元组甚至其他序列，先统一容器，后面的切片和比较才有稳定的行为。为什么非法词元要在这里就拦住？如果 `'bad'` 混进缓存索引，`match_prefix` 的比较会变成"整数和字符串比"，要么永远不命中、要么在别的后端里变成难以定位的错误；在入口抛 `TypeError` 能把问题定位到调用方。

容易写错的地方：把校验写在 `return` 之后（永远执行不到）；或者用 `type(token) == int` 而不是 `isinstance`（会拒绝 `int` 的子类）。顺带一提，Python 里 `bool` 是 `int` 的子类，`isinstance(True, int)` 为 `True`，所以 `True/False` 也会通过这个检查；notebook 没有对此做特殊处理。

### 第 4 步：TODO 2 `add_prefix`——登记、非空、去重
```python
prefix = tuple(self._normalize(prefix_tokens))  # 先规范化，再转成不可变元组
if not prefix:                                  # 空元组为假
    raise ValueError('prefix_tokens 不能为空')
if prefix not in self.cached_prefixes:          # 线性查重
    self.cached_prefixes.append(prefix)
```

三个细节：

1. **先 `_normalize` 再 `tuple`**：这样 `[1, 'bad']` 会在类型检查处抛 `TypeError`，而不是被静默转成元组存进去。
2. **空检查放在规范化之后**：`add_prefix([])` 得到空元组，`if not prefix` 成立，抛 `ValueError`。空前缀如果能登记，`match_prefix` 里 `len(cached_prefix)=0` 会让任何请求都"命中 0 个词元"，语义上毫无意义。
3. **用 `not in` 去重**：测试里连续登记 `[1,2,3]`、`[1,2,9]`、`[1,2,3]`，最终 `cached_prefixes == [(1, 2, 3), (1, 2, 9)]`，长度 2 而不是 3。这是线性扫描，条目多时是 $O(n)$，但本节只演示机制。
4. **为什么存元组**：`cached_prefixes` 的类型标注就是 `List[Tuple[int, ...]]`。元组不可变，调用方之后修改原列表不会悄悄改掉缓存内容；测试也直接和元组列表比较，存成列表会因类型不同而不相等。查重时比较的是元组内容，`(1,2,3)` 和 `(1,2,3)` 相等，与它们来自哪个列表对象无关。

### 第 5 步：TODO 3 `match_prefix`——在"完整登记对象"里选最长
参考实现：

```python
prompt = self._normalize(prompt_tokens)
best_len = 0
for cached_prefix in self.cached_prefixes:
    if len(cached_prefix) > len(prompt):        # 比请求还长的对象不可能整体命中
        continue
    is_match = prompt[:len(cached_prefix)] == list(cached_prefix)
    if is_match:
        best_len = max(best_len, len(cached_prefix))
return best_len
```

逐行解释：

- `if len(cached_prefix) > len(prompt): continue` 是剪枝。长度为 5 的缓存对象不可能成为长度为 3 的请求的前缀。
- `prompt[:len(cached_prefix)]` 取请求开头与候选等长的一段，再和 `list(cached_prefix)` 整体比较。注意右边要转成 `list`，因为左边切片是列表。
- `best_len = max(...)` 保证多个候选同时命中时取最长。比如缓存里有 `[1,2]` 和 `[1,2,3]`，请求 `[1,2,3,9]` 两者都命中，应返回 3。
- 初始值 `0` 表示"无命中"，所以空请求、完全不匹配都返回 0。

这里就是本节最反直觉的地方：**它不是求两个序列的最长公共前缀，而是在"已登记对象的集合"里挑一个能整体成为请求前缀的最长对象。** 用 notebook 的例子：

- 注册表只有 `[(1,2,3), (1,2,9)]` 时，请求 `[1,2,0]` 与前两者的公共前缀都是 `[1,2]`，但 `[1,2]` 这个对象没有被登记，所以返回 **0**。
- 调用 `manager.add_prefix([1,2])` 之后，同样的请求返回 **2**。
- 请求 `[0,1,2,3]` 虽然包含 `[1,2,3]` 子串，但不是从开头开始，返回 **0**。

这套"只认完整对象"的模型比真实后端保守：vLLM 的自动前缀缓存按 block 粒度匹配，SGLang 用基数树按词元路径匹配，都能命中更细的公共前缀。notebook 特意在解析里说「真实后端的可复用粒度可能不同」，就是为了避免你把这里的返回值当成生产环境命中率。

把测试里的五条请求放在一张表里，可以看清"命中"到底由什么决定：

| 请求 | 登记表 | 命中长度 | 可复用前缀 | 未命中后缀 | 原因 |
|---|---|---|---|---|---|
| `[1,2,3,9]` | `[(1,2,3),(1,2,9)]` | 3 | `[1,2,3]` | `[9]` | 第一个对象整体匹配开头 |
| `[1,2,9,8]` | 同上 | 3 | `[1,2,9]` | `[8]` | 第二个对象整体匹配开头 |
| `[1,2,0]` | 同上 | 0 | `[]` | `[1,2,0]` | `[1,2]` 只是公共子串，未登记 |
| `[1,2,0]` | 追加 `(1,2)` 后 | 2 | `[1,2]` | `[0]` | 新对象整体匹配开头 |
| `[0,1,2,3]` | 追加 `(1,2)` 后 | 0 | `[]` | `[0,1,2,3]` | 匹配必须从第一个词元开始 |

注意第 1 行和第 2 行的区别：请求 `[1,2,3,9]` 不会命中 `[1,2,9]`，因为第三个词元 `3 ≠ 9`；它也不会"部分命中" `[1,2,9]` 的前两个词元。第 3 行更关键：公共词元有 2 个，但 `[1,2]` 不是一个已登记对象，所以命中 0。这就是"最长公共前缀"和"最长已登记完整对象"的差别。

### 第 6 步：TODO 4 `split_prompt`——用一次切片完成拆分
```python
prompt = self._normalize(prompt_tokens)
hit_len = self.match_prefix(prompt)
reusable_prefix = prompt[:hit_len]   # 命中范围覆盖的开头
suffix = prompt[hit_len:]            # 命中边界之后的全部剩余
return reusable_prefix, suffix, hit_len
```

因为 Python 切片 `[:k]` 和 `[k:]` 天然互补，`reusable_prefix + suffix == prompt` 自动成立（对于 `0 <= hit_len <= len(prompt)` 恒真）。`hit_len` 来自 `match_prefix`，而匹配时已经保证不超过请求长度，所以不会越界。测试用 `[1,2,3,9]` 验证得到 `([1,2,3], [9], 3)`：命中 3 个，后缀只剩 `[9]`。

如果自己写循环去"边匹配边收集"，很容易在零命中或完整命中这两个边界上多切/少切一个词元；直接复用切片是最稳的写法。

### 第 7 步：TODO 5 `cache_stats`——账本三件套
```python
prompt = self._normalize(prompt_tokens)
hit_tokens = self.match_prefix(prompt)
uncached_tokens = len(prompt) - hit_tokens
reuse_ratio = hit_tokens / len(prompt) if prompt else 0.0
return {'hit_tokens': hit_tokens,
        'uncached_tokens': uncached_tokens,
        'reuse_ratio': round(reuse_ratio, 4)}
```

三个字段的含义与约束：

- `hit_tokens`：本次可复用的前缀长度。
- `uncached_tokens`：仍需预填充的后缀长度，满足 `hit_tokens + uncached_tokens == len(prompt)`（词元守恒）。
- `reuse_ratio`：命中数 ÷ 输入数。空输入必须返回 `0.0`，否则 `0/0` 会抛 `ZeroDivisionError`；非空时四舍五入到 4 位小数，让测试里的 `0.75` 这类值可直接比较。

手算测试用例：请求 `[1,2,3,9]` 长度 4，命中 `[1,2,3]` 长度 3，所以 `uncached_tokens = 4 - 3 = 1`，`reuse_ratio = 3 / 4 = 0.75`。空请求长度 0，命中 0，账本为 `{'hit_tokens': 0, 'uncached_tokens': 0, 'reuse_ratio': 0.0}`。

空请求在 `match_prefix` 里也不会报错：`prompt` 为空，登记对象长度都大于 0，`len(cached_prefix) > len(prompt)` 让每个候选都被 `continue` 跳过，`best_len` 保持 0。于是 `split_prompt([])` 得到 `([], [], 0)`，`cache_stats([])` 走 `if prompt` 的空分支。只有 `add_prefix([])` 这种"登记空对象"才抛 `ValueError`；查询空请求是合法的"没有共享前缀"边界。

### 第 8 步：这个账本能说明什么、不能说明什么
notebook 的 Step 3 把"逻辑账本"和"实测收益"分开：

| 账本量 | 含义 | 还应对照什么证据 |
|---|---|---|
| 命中词元数 | 可复用的前缀长度 | 后端 KV 占用、TTFT |
| 未缓存词元数 | 仍需预填充的后缀长度 | 预填充耗时、吞吐 |
| 复用比例 | 命中数 ÷ 输入数 | 匹配工作负载下的命中率分布 |
| 新增状态 | 后缀算出的 KV Cache | 驱逐、迁移、生命周期记录 |

原因是：命中 100% 也不代表首词元成本为 0。执行器可能还缺少"生成下一个词元所需的分数"（比如最后一个位置的 logits），或者 KV 块需要从别的进程搬过来，或者复用的粒度被 block 对齐截断。反过来，账本里的 `hit_tokens` 也不等于省下的显存——缓存本身还占着容量，而且可能被其他请求共享或驱逐。所以 notebook 在 docstring 里写的是「逻辑复用账本，不将词元计数解释为实测性能」；它衡量的是"少算了多少输入词元"，不是"快了多少"。

### 第 9 步：从"逻辑命中"走到真实后端，还差哪几层
本题的 `cached_prefixes` 只是一个存元组的列表，真实前缀缓存要复杂得多。Step 1 已经列出复用的前提：**模型权重、适配器、位置编号、缓存表示相同，且权限允许共享**。逐项拆开看：

- **模型权重不同**：同样是 `[1,2,3]`，在 7B 和 13B 模型里算出的 K/V 完全不同，不能共用。
- **适配器不同**：基座相同但 LoRA 不同，同一段文本的有效 K/V 也不同；多 LoRA 服务必须把 adapter 身份纳入缓存键。
- **位置编号不同**：RoPE 依赖绝对或相对位置；如果同一个前缀在两条请求里起始位置不同，缓存的 K/V 未必能直接搬用。
- **分词器不同**：缓存索引的是 token ID，不是原始字符串。同一句中文在不同分词器下可能切成不同 ID 序列，`match_prefix` 自然不会命中。
- **权限不同**：多租户场景下，即使内容和版本都一样，也要先确认这条请求有权读取那份缓存；缓存键通常还要带租户或权限标签，否则会越权读到别人的 KV。

这五项里任何一项不同，缓存键就不同，这正是缓存键必须带版本与 adapter 身份的原因。工程上常把"模型版本、adapter、tokenizer、前缀 token、位置、权限"组成键或分层索引：先按便宜字段过滤，再比 token 序列。本题只保留 token 序列这一维，其他条件都默认满足，所以四个测试才能用几个整数列表跑通。

真实后端通常还有两层本题没有模拟的机制：

1. **块粒度匹配**。vLLM 的自动前缀缓存按 block 管理 KV，一个 block 往往包含多个 token；即使逻辑上命中了 30 个 token，若块大小是 16，也可能只能复用前 16 或 32 个，剩余部分仍要重算。SGLang 的 RadixAttention 用基数树按 token 路径匹配，粒度比"完整对象列表"细得多。
2. **容量与淘汰**。登记的前缀本身占 KV 空间。容量满时后端会淘汰旧块，之前登记过的前缀可能已经不在显存里；本题的 `cached_prefixes` 永远不会消失，所以它的命中率是"理想上限"，不是生产命中率。

复杂度上，`match_prefix` 对每个登记对象做一次切片比较，最坏是 $O(P \times L)$（$P$ 是登记条目数，$L$ 是最长前缀长度）；`add_prefix` 的 `in` 查重也是 $O(P)$。基数树可以把按路径查找降到与前缀长度相关，但实现和并发控制复杂得多。notebook 选择列表，是为了让五个 TODO 的边界条件都能被单独测出来，而不是为了性能。

举个纯逻辑的算术例子（不是实测）：假设请求有 4096 个 token，登记过的最长完整前缀是 3000 个 token，那么 `suffix` 是 1096 个 token，`reuse_ratio = 3000 / 4096 ≈ 0.7324`。这表示"按 token 计的输入工作少了约 73%"，但不代表端到端延迟也少 73%，更不代表显存少 73%——原因就是上面两层机制。

### 第 10 步：把单条账本汇总成工作负载账本
`cache_stats` 只算一条请求。真实服务要看一批请求的命中情况，最容易写错的是"把每条请求的 `reuse_ratio` 直接求平均"。假设两条请求：

- 请求 A 长度 4，命中 3，`reuse_ratio=0.75`；
- 请求 B 长度 3，命中 0，`reuse_ratio=0.0`。

逐条平均是 $(0.75+0)/2=0.375$，但按词元加权是 $(3+0)/(4+3)=3/7\approx0.4286$。两者不同，因为 A 更长、命中的词元更多。工作负载级指标应写成：

$$
\mathrm{workload\_reuse} = \frac{\sum_i \mathrm{hit\_tokens}_i}{\sum_i \mathrm{len(prompt}_i\mathrm{)}}.
$$

同理，"平均命中长度"也要说明是"每条请求命中长度的平均"还是"总命中词元 ÷ 请求数"；前者受短请求影响大，后者更能反映总工作量，两者在请求长度不均时可能差很多。notebook 没有实现这一步，只给了单条 `cache_stats`；如果你要把它扩成报表，至少同时输出请求数、总输入词元、总命中词元和加权复用比例，并标清每条请求的长度分布，避免只报一个容易误读的百分数。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Prefill（预填充） | 把整段输入词元过一遍模型、算出每层 K/V 的阶段 |
| KV Cache | 缓存每层已处理词元的 Key/Value，避免解码时重算 |
| 前缀缓存 | 把"从开头连续相同"的那段输入对应的 KV 状态登记起来供人复用 |
| 共享前缀 | 从第一个词元起连续相同、且已登记的那一段 |
| 未命中后缀 | 第一个不一致词元之后的全部内容，仍要预填充 |
| 最长前缀匹配 | 在已登记对象中，选能整体成为请求开头的最长那个 |
| 完整对象登记 | 只有整段前缀被登记过才算命中，公共子串不算 |
| 复用比例 | 命中词元数 ÷ 输入词元数 |
| 词元守恒 | `hit_tokens + uncached_tokens == len(prompt)` |
| TTFT | Time To First Token，首词元延迟；账本不能直接预测它 |
| 淘汰（eviction） | 缓存容量不足时把旧条目赶出去；本节未实现 |

## 小白常见疑问

**Q1：请求 `[1,2,9]` 和缓存 `[1,2,3]` 有两个公共词元，为什么 `match_prefix` 返回 0，而不是 2？**
因为当前实现只认"完整登记对象"。`[1,2,3]` 整体不是 `[1,2,9]` 的前缀，所以不命中；`[1,2]` 这个对象又没被登记。测试 `test_longest_prefix_match` 正是用 `manager.match_prefix([1, 2, 0]) == 0`，然后 `manager.add_prefix([1, 2])` 再断言返回 2，来强调"命中粒度由登记对象决定"。真实后端会按 block 或树节点匹配更细的前缀，但本节不模拟。

**Q2：为什么 `match_prefix` 里要写 `list(cached_prefix)`，直接比较不行吗？**
`prompt` 是 `list`，`prompt[:k]` 也是 `list`；`cached_prefix` 是 `tuple`。Python 里 `[1,2] == (1,2)` 的结果是 `False`，类型不同直接判不等。转成 `list` 后才是同类型逐元素比较。这是本 TODO 最容易踩的坑，测试会立刻抓到——所有命中都会变成 0。

**Q3：命中 3 个词元、后缀 1 个词元，是不是说明计算量只剩 1/4？**
不能这么直接换算。账本说的是"还有 1 个词元需要预填充"，但真实耗时还受 block 对齐、kernel 启动、批次里其他请求、以及"取最后一个位置分数"等固定开销影响。notebook Step 3 明确要求把账本和后端计数、TTFT 一起看，而不是把 `reuse_ratio` 当加速比。

**Q4：`cache_stats([])` 为什么单独判断 `if prompt`？**
空列表在布尔上下文里为假。如果不判断，`hit_tokens / len(prompt)` 就是 `0 / 0`，Python 会抛 `ZeroDivisionError`。写成 `hit_tokens / len(prompt) if prompt else 0.0` 后，空输入返回 0.0，符合测试 `manager.cache_stats([]) == {'hit_tokens': 0, 'uncached_tokens': 0, 'reuse_ratio': 0.0}`。

**Q5：两段文字一模一样，但一个用基座模型、一个用 LoRA 适配器，能命中同一个前缀吗？后端又为什么可能只复用一部分？**
不能直接命中。Step 1 要求模型权重、适配器、位置编号、缓存表示都相同；LoRA 会改变每层实际算出的 K/V，token ID 一样也不能共用。即使这些条件都满足，本题按"完整登记对象"匹配，后端却按 block 匹配：一个 block 可能装 16 个 token，命中 30 个只能覆盖一个完整 block（16），剩下 14 个无法单独复用。notebook 在 Step 2 的解析里提醒「真实后端的可复用粒度可能不同」，所以不要把本节的 `hit_tokens` 当成后端计数器。

## 动手验证：怎么知道自己做对了
notebook 的测试入口是 `test_prefix_cache_manager()`，它依次调用四个子测试，全部通过后打印 `✅ PrefixCacheManager 机制测试通过`。TODO 未填时，调用会先抛 `NotImplementedError`。

- `test_prefix_input_contract`：对 `[]` 和 `[1, 'bad']` 分别调用 `add_prefix`，要求抛出 `ValueError` 或 `TypeError`；两者都不抛就 `raise AssertionError('空前缀或非整数 token 应被拒绝')`。
- `test_prefix_registration_and_deduplication`：fixture 依次登记 `[1,2,3]`、`[1,2,9]`、`[1,2,3]`，断言 `manager.cached_prefixes == [(1, 2, 3), (1, 2, 9)]`，即重复项被去掉、顺序保持首次登记顺序。
- `test_longest_prefix_match`：在 fixture 上断言 `[1,2,3,9] → 3`、`[1,2,9,8] → 3`、`[1,2,0] → 0`；再登记 `[1,2]` 后断言 `[1,2,0] → 2`；最后断言 `[0,1,2,3] → 0`（子串不算前缀）。
- `test_prefix_suffix_split_and_stats`：断言 `split_prompt([1,2,3,9]) == ([1,2,3], [9], 3)`；`cache_stats([1,2,3,9]) == {'hit_tokens': 3, 'uncached_tokens': 1, 'reuse_ratio': 0.75}`；`cache_stats([]) == {'hit_tokens': 0, 'uncached_tokens': 0, 'reuse_ratio': 0.0}`。

这个 notebook 没有保存任何执行输出（dump 里没有任何 output），所以不要期待看到打印的账本数值；判断对错以这四个测试的断言为准。跑通后你应看到最后一行是 `✅ PrefixCacheManager 机制测试通过`，没有任何 `AssertionError` 或 `NotImplementedError`。
