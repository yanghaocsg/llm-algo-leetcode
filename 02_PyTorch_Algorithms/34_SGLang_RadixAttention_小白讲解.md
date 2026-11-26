# 34. SGLang RadixAttention 基数注意力 — 小白讲解

> 对应 notebook：[`34_SGLang_RadixAttention.ipynb`](./34_SGLang_RadixAttention.ipynb)

## 一句话总结
本节用 `TreeNode` 和 `SimpleRadixCache` 复现 RadixAttention 的前缀索引：把连续、无分叉的 token 路径压成一条边，新请求到来时沿树找最长公共前缀，只有"完整登记过"的路径终点才能复用。四个 TODO 分别完成边分裂、逐 token 比较、终止节点命中和 prompt 拆分，测试覆盖共享边、反向插入、内部前缀登记和空输入。

## 为什么需要它：痛点是什么
真实推理服务里，多轮对话会反复带上前面的历史，智能体请求常常共享一大段系统提示词。如果每个请求都从头预填充，同一段前缀的 KV Cache 会被重复计算和重复占用，首 token 延迟也跟着变高。

初学者的困惑通常有二：第一，前缀缓存不是"记住一串 token"就够了吗，为什么还要树？第二，`match_prefix` 找到了相同 token，为什么返回值可能是 0？答案分别是：树能把大量前缀的公共部分压成共享路径，查找时不用逐条比对；以及"token 相同"不等于"对应的 KV 状态已经保存"，只有登记过的完整对象才能复用。

生活类比：城市路网。多条公交线路从同一个总站出发，前 5 站完全重合，之后分叉。你不需要为每条线路各画一条前 5 站的重复道路，只要画一条共享主干，在分叉口分成两支。`terminal` 标记就是"这条线路的终点站"：路过某个站不代表可以在这儿下车，只有真正的终点才算一次完整可达。

## 核心问题
- 压缩基数树如何用一条边表示一段连续 token，而不是一个节点一个 token？
- 插入新路径时，遇到部分重叠的边，为什么要分裂？旧边的子节点、终止标记和缓存指针分别怎么处理？
- 最长公共前缀（LCP）长度为什么不能直接当作可复用长度？
- `match_prefix` 为什么只在外层循环走完整条边、且节点是 `terminal` 时才更新答案？
- `split_prompt` 的前缀、后缀和命中长度三者必须满足什么契约？

## 解决思路：一步步看懂

**第 1 步：先看数据结构。** `TreeNode` 有四个字段：`key_tokens` 是这条边覆盖的 token 序列；`children` 是子节点列表；`kv_cache_ptr` 是"模拟指向物理 KV Cache 的指针"，本节只占位、不做真实读写；`terminal` 表示是否有完整请求在这条路径的终点结束。`SimpleRadixCache` 的根节点是 `TreeNode([])`，空 key，不代表任何 token。`_find_child(node, token)` 在所有子边里找"首 token 相同"的那一条，同一节点下不应出现重复首 token。

**第 2 步：TODO 1，边分裂。** `insert` 先拒绝空路径（`ValueError('tokens 不能为空')`），然后从根开始循环：找到首 token 相同的子边，用 `_lcp_len` 算出公共长度 `common`。如果 `common == len(child.key_tokens)`，说明整条子边都被覆盖，直接下沉 `node, offset = child, offset + common`；否则就是"部分重叠"，必须把这条边拆成共享前缀 + 旧后缀：

```python
shared_tokens = child.key_tokens[:common]          # 新旧路径共享的前缀
old_suffix_tokens = child.key_tokens[common:]      # 旧路径多出来的部分
```

随后的装配由骨架完成：新建 `shared` 和 `old_suffix`；`old_suffix` 继承 `child.terminal` 和 `child.kv_cache_ptr`，并把 `child.children` 整体挂到自己下面；`shared` 把 `old_suffix` 收为子节点；最后用 `shared` 替换 `node.children` 里的原 `child`。如果新路径刚好在共享前缀处结束（`offset + common == len(tokens)`），就把 `shared.terminal = True`；否则再挂一个 `TreeNode(tokens[offset + common:], terminal=True)` 作为新分支。

为什么必须保留旧边的终止标记和缓存指针？因为拆分只改变"路径怎么存"，不改变"旧请求已经算完、它的 KV 状态还在"这个事实。如果把 `old_suffix` 建成普通节点、丢掉 `child.terminal`，旧请求会突然变成不可命中；丢掉 `kv_cache_ptr`，真实系统里就找不到它的 KV 块。测试 `test_register_existing_internal_prefix` 正是用一个 `marker = object()` 检查这一点：`[1,2,3]` 的指针在分裂后必须出现在 `[3]` 这个旧后缀节点上。

**第 3 步：TODO 2，逐 token 比较。** `_lcp_len(cached_tokens, prompt_tokens)` 用 `zip` 同时遍历两条序列，参考答案只有一句：

```python
tokens_match = cached_token == prompt_token
```

不匹配就 `break`，最后返回 `match_len`。`zip` 会在任一序列耗尽时停止，所以返回值天然不会超过较短序列的长度；完全一致时就等于较短长度。测试 `_lcp_len([1,2,3], [1,2,4]) == 2` 和 `_lcp_len([7,8], [7,8,9,10]) == 2` 分别覆盖"中途分叉"和"长度不同"。

**第 4 步：TODO 3，只认终止节点。** `match_prefix` 从根出发，每轮找首 token 匹配的子边，算 `common`；如果子边不存在，或者 `common < len(child.key_tokens)`（只匹配了半条边），立即 `break`。只有完整走完一条边，才执行 `offset += common`、`node = child`，然后：

```python
if node.terminal:
    best_match_len = offset
```

注意 `offset` 是"已经完整走过的 token 总数"，不是当前边的长度。为什么半条边不行？因为状态是按节点边界保存的：一条边中间没有缓存对象，走到一半无法读出对应 KV。为什么非终止节点不行？因为那里只是"路过的共享主干"，没有请求在那里结束，也就没有完整状态。测试 `_build_radix_cache()` 插入 `[0,1,2,3]`、`[0,1,2,3,4]`、`[9,9,9]` 后，`match_prefix([0,1,2,3,4,5])` 返回 5（在 `[4]` 的终点停），`match_prefix([9,9,9])` 返回 3，`match_prefix([7,6,5])` 返回 0。

notebook 用集合形式写清了定义：设完整登记对象集合为 $C$，请求为 $P$，可返回的命中长度是

$$H=\max\left(\{0\}\cup\{|c|:c\in C,\ c=P[:|c|]\}\right),$$

其中 $|c|$ 是对象 $c$ 的 token 数，$P[:|c|]$ 是请求开头同样长度的片段。这个定义只看"已登记对象"，不看树里是否存在更长的公共路径。`[1,2,9]` 在只登记 `[1,2,3]` 和 `[1,2,4]` 时，能沿共享边走到 `[1,2]`，但 `[1,2]` 不是任何对象的终点，所以命中是 0；显式 `insert([1,2])` 之后才变成 2。

**第 5 步：TODO 4，拆分 prompt。** `split_prompt` 直接复用命中结果：

```python
hit_len = self.match_prefix(prompt_tokens)
hit_prefix = prompt_tokens[:hit_len]
miss_suffix = prompt_tokens[hit_len:]
return hit_prefix, miss_suffix, hit_len
```

契约是三段拼回去必须等于原 prompt，且 `hit_prefix` 一定来自开头、`miss_suffix` 从第一个未命中的 token 开始。测试要求 `[0,1,2,3,4,5]` 拆成 `([0,1,2,3,4], [5], 5)`；完整命中 `[0,1,2,3,4]` 拆成 `([0,1,2,3,4], [], 5)`；完全未命中 `[7,6,5]` 拆成 `([], [7,6,5], 0)`；空 prompt 拆成 `([], [], 0)`。

**第 6 步：把"索引命中"和"真实可读"分开。** 这一节的 `kv_cache_ptr` 只是占位，`match_prefix` 也明确写了"本函数不检查真实 K/V 驻留"。所以测试通过只证明索引逻辑正确：最长公共路径找对了、完整对象边界找对了、拆分对了。真实系统还要在返回命中后确认对应 KV 块仍然驻留、引用计数安全、没有被换出；notebook 的 Step 3 表格把"公共路径"和"完整对象边界"分成两行，就是在提醒这一点。PagedAttention 按物理块管理 KV，RadixAttention 按 token 前缀组织共享路径，两者是不同层次的机制，可以同时存在。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Radix Tree（基数树） | 把连续无分叉的 token 路径压成一条边的查找树 |
| 压缩边 | 一条边存一整段 token，而不是一个节点一个 token |
| `terminal` | 是否有完整请求在这条路径终点结束，只有它能被复用 |
| LCP（最长公共前缀） | 两条 token 序列从头开始相同的长度 |
| 边分裂 | 新路径只覆盖旧边一部分时，把旧边拆成共享前缀和旧后缀 |
| `kv_cache_ptr` | 指向物理 KV 状态的占位指针，本节不读写真实张量 |
| 命中长度 `best_match_len` | 已登记完整对象覆盖的请求开头长度 |
| 未命中后缀 | 命中边界之后、仍需预填充的部分 |
| `split_prompt` | 按命中长度把 prompt 切成可复用前缀和待计算后缀 |
| 前缀复用 | 命中部分直接读已有 KV，只对后缀做预填充 |
| PagedAttention | 按物理块管理 KV 的另一种机制，和前缀索引不互斥 |

## 小白常见疑问

**Q1：先插 `[0,1,2,3]` 再插 `[0,1,2,3,4]`，为什么没有发生分裂？**
因为第二路径完整覆盖了第一条边：`_lcp_len([0,1,2,3], [0,1,2,3,4])` 返回 4，等于 `len(child.key_tokens)`，代码走 `common == len(child.key_tokens)` 的分支直接下沉。到了 `[0,1,2,3]` 节点后，剩余 token 是 `[4]`，找不到首 token 为 4 的子边，于是新建叶节点 `[4]`。最终根下只有一条 `[0,1,2,3]` 共享边，它自己是 `terminal`，下面挂一个 `[4]` 终止节点，和 `test_insert_and_split` 的断言一致。

**Q2：反向插入 `[4,5,6,7]` 再插 `[4,5]`，为什么 `match_prefix([4,5,8])` 是 2 而不是 0 或 4？**
第二条路径在前 2 个 token 后结束，`common = 2 < 4`，触发分裂：`shared = [4,5]`，`old_suffix = [6,7]`；因为 `offset + common == len(tokens)`，`shared.terminal = True`。所以 `[4,5]` 成了一个完整对象，命中长度是 2；请求的第三个 token 是 8，与 `[6,7]` 分支的首 token 不同，不能再往下走。

**Q3：`match_prefix([1,2,9])` 在登记 `[1,2]` 之前为什么返回 0，而不是 2？**
树里确实有 `[1,2]` 这条共享边，但它是从 `[1,2,3]` 分裂出来的中间节点，`terminal` 为 False。`match_prefix` 只在 `node.terminal` 为真时更新 `best_match_len`，所以沿途走过 `[1,2]` 也不会记账。显式 `insert([1,2])` 会把该节点的 `terminal` 置为 True，之后同样请求才返回 2。这正是"LCP 不等于可复用长度"的最直观例子。

**Q4：为什么 `insert` 遇到半条边不继续往深处找，而是直接分裂返回？**
因为压缩边内部没有节点，也就没有保存状态的位置。新路径和旧边在中间分叉时，必须先把共享前缀提升成一个真实节点，才能把"旧后缀"和"新后缀"分别挂成两个分支。分裂后函数直接 `return`，不再继续处理剩余 token；新后缀已经作为叶节点一次性建好，所以不需要循环。

**Q5：`kv_cache_ptr` 一直是 `None`，这个练习是不是没用到缓存？**
本节练的是"索引"：哪些前缀可以复用、复用到第几个 token。真实 KV 的分配、驻留、引用计数和换出不在这个 CPU 练习里，`kv_cache_ptr` 只是保留接口位置。测试用 `marker = object()` 验证分裂时引用不会丢，但不会解引用它。所以本节通过不等于真实推理服务的前缀缓存已经正确，后者还要看 37 节的容量控制和 69 节的基准。

## 动手验证：怎么知道自己做对了
完成四个 TODO 后运行最后一个测试单元，它会调用 `test_radix_attention()`，汇总六组测试；全部通过打印 `✅ RadixAttention 机制测试通过：LCP、共享边、最长命中和 prompt 拆分均通过。`。如果还有 TODO 未完成，会先打印 `请先完成 TODO 部分的代码！` 再把 `NotImplementedError` 抛出来；其他异常会打印 `❌ RadixAttention 测试失败: ...` 后重抛。dump 中没有保存这一格的运行输出，以断言为准。

- `test_lcp_contract`：`_lcp_len([1,2,3], [1,2,4]) == 2`；`_lcp_len([7,8], [7,8,9,10]) == 2`。
- `test_insert_and_split`：插入三条路径后根下必须只有 2 个子节点；首 token 为 0 的共享边 `key_tokens == [0,1,2,3]`、`terminal is True`、只有一个子节点 `[4]` 且它也是终止节点；反向插入场景 `match_prefix([4,5,8]) == 2`。
- `test_register_existing_internal_prefix`：分裂后 `[3]` 节点的 `kv_cache_ptr is marker`；登记 `[1,2]` 前 `match_prefix([1,2,9]) == 0`，登记后为 2，`match_prefix([1,2,3]) == 3`。
- `test_match_prefix_contract`：`[0,1,2,3,4,5] → 5`、`[9,9,9] → 3`、`[7,6,5] → 0`。
- `test_split_prompt_contract`：三个拆分结果分别是 `([0,1,2,3,4],[5],5)`、`([0,1,2,3,4],[],5)`、`([],[7,6,5],0)`。
- `test_empty_prompt_and_path`：`match_prefix([]) == 0`、`split_prompt([]) == ([],[],0)`、`insert([])` 必须抛 `ValueError`。

想额外自查，可以在 `_build_radix_cache()` 后打印根节点的 `children`：应该看到 `[0,1,2,3]` 和 `[9,9,9]` 两条边，前者带一个 `[4]` 子节点；再打印 `split_prompt([0,1,2,3,4,5])`，确认三段拼接后与输入完全一致。
