# 21. 解码策略 — 小白讲解

> 对应 notebook：[`21_Decoding_Strategies.ipynb`](./21_Decoding_Strategies.ipynb)

## 一句话总结
本节从"模型给出词表 logits"出发，用 `apply_temperature`、`apply_top_k`、`apply_top_p` 三个函数实现温度缩放、Top-K 和 Top-p 候选过滤，再由 `decode_next_token` 完成贪心或采样选择。核心是理解处理顺序——温度 → Top-K → Top-p → 归一化 → 选择——以及每一步改变的是"分布形状"还是"候选集合"。

## 为什么需要它：痛点是什么
模型在每个位置输出的不是"下一个词"，而是整个词表上的未归一化分数（logits）。如果永远取最高分（贪心），生成会稳定但容易重复、缺乏多样性；如果完全按原始概率随机抽，又可能抽到明显不合理的词。温度、Top-K、Top-p 就是在"确定性"和"多样性"之间调的三个旋钮。

初学者常见的困惑是：温度调低是不是就等于贪心？Top-K 和 Top-p 到底谁先谁后？Top-p 的"累计概率达到 0.9"为什么有时保留 3 个词、有时保留 30 个？把这些旋钮的顺序弄反，或者边界多保留/少保留一个 Token，分布就会悄悄变掉。

生活类比：你在餐厅点菜。logits 是菜单上每道菜的推荐分；温度是"你有多信任推荐分"——温度低就只点第一名，温度高就愿意尝试排名靠后的菜；Top-K 是"只看菜单前 K 名"；Top-p 是"按推荐分从高到低夹菜，夹到累计推荐度达到 90% 就停"。最后一步才是真正下单（采样）。

## 核心问题
- 温度 $T$ 为什么用 $z/T$ 而不是 $z\cdot T$？$T<1$ 和 $T>1$ 分别让分布变尖还是变平？
- 温度很低时，采样解码和贪心解码真的一样吗？
- Top-K 的"第 K 大"边界遇到同分怎么办？为什么保留的候选数可能超过 K？
- Top-p 怎样确定最小候选集合？为什么首次越过阈值的 Token 必须保留？
- 三个操作的顺序为什么不能随便换？Top-K 之后再算 Top-p，概率是在哪个集合上归一化的？
- `autoregressive_decode` 为什么每轮只取最后一个位置的 logits？它处理 EOS 吗？

## 解决思路：一步步看懂

**第 1 步：从 logits 到下一个 Token 的四条路径。** Step 1 的表把路径拆开：原始词表分数描述相对偏好；贪心解码取最高分，结果确定，可作基线；采样解码按归一化概率抽取，结果随机；Top-K / Top-p 在采样前缩小候选集合。关键点是：这些操作只改变"本轮如何选 Token"，不修改模型权重，也不回写已有 KV 缓存；但选出的 Token 会进入下一轮上下文，所以会影响后续所有状态。

**第 2 步：温度缩放。** 正温度 $T$ 把分数 $z$ 变成 $z/T$，再过 Softmax：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

$T<1$ 相当于把分数差异放大，Softmax 更尖锐，高分词概率更高；$T>1$ 把差异缩小，分布更平；$T=1$ 保持原分布。因为除以的是同一个正数，所有分数的相对顺序不变，所以 Top-K 的成员不受温度影响，但 Top-p 的成员会受影响——分布变尖后，更少的词就能累计到阈值。

TODO 1 的参考实现：

```python
temperature_is_valid = torch.isfinite(torch.tensor(temperature)).item() and temperature > 0
if not temperature_is_valid:
    raise ValueError("temperature 必须是有限正数")
scaled_logits = logits / temperature
```

`torch.isfinite(...).item()` 同时挡住 `nan` 和 `inf`；`temperature > 0` 挡住 0 和负数。测试 `test_temperature_contract` 用 $T=0.5$ 检查 `scaled[5]-scaled[6]` 是原来的 2 倍，并断言 `argsort` 不变、形状和 dtype 不变；传 0.0 必须抛 `ValueError`。容易写错的是写成 `logits * temperature`，那会让 $T<1$ 变平、$T>1$ 变尖，和约定正好相反。

**第 3 步：Top-K 过滤。** TODO 2 的参考实现：

```python
kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1:]
filtered_logits = torch.where(logits < kth_values,
                              torch.full_like(logits, float("-inf")), logits)
```

`torch.topk(..., top_k).values[..., -1:]` 取第 K 大的值作为阈值，形状保留最后一维以便广播。凡是严格小于阈值的置为 $-\infty$，等于阈值的全部保留——所以同分时候选数可能超过 K。`test_top_k_filtering` 用固定 logits 断言 `top_k=3` 后有限值恰好 3 个；再用 `[[2.0,2.0,2.0,1.0]]`、`top_k=2` 断言有限值是 3 个，而不是 2 个。`top_k=0` 或 `top_k >= vocab` 直接返回原 logits；`None` 也直接返回；负数、布尔值和非整数抛 `ValueError`。

**第 4 步：Top-p 边界。** Top-p（nucleus sampling）先按概率从高到低排序并累计，保留"累计概率首次达到或超过 $p$"的最小集合，边界 Token 本身必须保留。notebook 的例子：概率 `[0.5, 0.3, 0.1, 0.05, 0.05]`，$p=0.85$ 时累计是 `0.5, 0.8, 0.9`，第三个 Token 首次越过 0.85，所以保留前三个，而不是两个或四个。

TODO 3 的参考实现：

```python
shifted_mask = remove_mask[..., :-1].clone()
remove_mask[..., 1:] = shifted_mask
remove_mask[..., 0] = False
sorted_logits[remove_mask] = float("-inf")
restored_logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
```

`remove_mask = cumulative_probs >= top_p` 本来会把"首次达到阈值"的边界 Token 也标成删除；把掩码右移一位（`shifted_mask`）后，删除位置变成边界之后的所有 Token，边界本身保留。`remove_mask[...,0]=False` 保证排名第一的 Token 永远不会被删。最后用 `scatter_` 按 `sorted_indices` 把分数放回原词表位置，得到与输入同形状、同顺序的 `restored_logits`。

`test_top_p_boundary` 在固定 logits 上取 $p=0.8$，断言有限值恰好 3 个，且原词表下标 `[5,6,1]` 上的值保持不变；再用精确概率 `[0.5,0.25,0.125,0.125]`、$p=0.75$ 断言保留前两个——因为累计概率恰好等于 0.75 时必须在边界停止，不能多留一个。传 `0.0` 和 `1.1` 必须抛 `ValueError`；`top_p=1.0` 直接返回原 logits。

**第 5 步：选择与循环。** `decode_next_token` 固定顺序：

```python
logits = apply_temperature(logits, temperature)
logits = apply_top_k(logits, top_k)
logits = apply_top_p(logits, top_p)
probabilities = F.softmax(logits, dim=-1)
if not do_sample:
    return torch.argmax(logits, dim=-1, keepdim=True)
return torch.multinomial(probabilities, num_samples=1, generator=generator)
```

Top-K 把非候选置为 $-\infty$，Top-p 的 Softmax 只在这些剩余候选上归一化，所以后续 `softmax` 的概率质量不会漏到被排除的 Token。贪心路径直接对过滤后的 logits 取 `argmax`；采样路径用 `torch.multinomial`，传入同一个 `generator` 可以复现结果。

`autoregressive_decode` 循环 `max_new_tokens` 次：调用 `logits_fn(tokens)`，如果返回三维 `[batch, seq, vocab]` 就取最后一个位置 `[:, -1, :]`，再调用 `decode_next_token`，最后 `torch.cat` 追加。它按固定步数结束，不处理 EOS；完整请求还需要记录停止原因。

**第 6 步：为什么顺序不能换。** 本节固定"温度 → Top-K → Top-p"：温度保持排序但改变尖锐度，Top-K 砍掉排名外的词，Top-p 在剩余候选上再收一次。若 Top-p 先在整个词表上算边界，保留集合会不同；若温度放在 Top-p 之后，边界又按未缩放分布计算。三种顺序都能跑，但含义不同，测试锁定的是本节约定。

**第 7 步：测试在守哪些边界。** `_fixture_logits()` 给出 `[[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]]`，最高分在下标 5，其次 6、1。`test_selection_and_autoregressive_loop` 断言贪心在 `temperature=1.0, top_k=0, top_p=1.0, do_sample=False` 时返回下标 5；两个 `manual_seed(7)` 的 generator 产生相同结果；`fake_logits` 每步把下标 3 置为 2.0，3 步后生成长度 `2+3=5`，最后三个 Token 都是 3。`test_batch_and_input_contract` 把同一行复制成 batch 2，断言 Top-K/Top-p 保持 `[2,10]` 形状，单条 1D 输入经 `decode_next_token` 返回 `[1,1]`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| logits | 模型对词表每个 Token 给出的未归一化分数 |
| 贪心解码 | 每步取最高分 Token，结果确定 |
| 采样解码 | 按 Softmax 概率随机抽，结果有随机性 |
| 温度 $T$ | 用 $z/T$ 缩放分数；小于 1 更尖，大于 1 更平 |
| Top-K | 只保留分数排名前 K 的候选，同分一起保留 |
| Top-p / nucleus | 保留累计概率首次达到阈值的最小候选集合 |
| 候选掩码 | 把不保留的位置设为 $-\infty$ |
| 重新归一化 | 在保留集合内再做 Softmax，使概率和为 1 |
| `do_sample` | 为 False 时走 argmax，为 True 时走 multinomial |
| `generator` | 控制随机数状态，固定种子可复现采样 |
| 自回归循环 | 每生成一个 Token 就追加到上下文，再算下一个 |
| EOS | 终止词元；本节循环不处理，只按固定步数结束 |

## 小白常见疑问
**Q1：温度调到很低，为什么不等同于贪心？**
只要 $T>0$，Softmax 给每个有限 logit 的概率都严格大于 0，`torch.multinomial` 理论上仍可能抽到非最高分 Token；温度越低这个概率越小，但不等于 0。贪心是 `argmax`，完全跳过随机采样。notebook Step 2 明确写了这一点，所以 `decode_next_token` 用 `do_sample` 区分两条路径，而不是用"温度是否为 0"。

**Q2：Top-K 明明设了 3，为什么可能保留 4 个 Token？**
因为阈值是"第 K 大的值"，所有**等于**阈值的 Token 都保留。`test_top_k_filtering` 用 `[[2.0,2.0,2.0,1.0]]`、`top_k=2` 断言有限值是 3 个：第 2 大是 2.0，三个 2.0 都不小于阈值。这样不会因为排序稳定性随意丢掉同分候选。

**Q3：Top-p 的边界 Token 为什么不能删？**
Top-p 的定义是"累计概率首次达到或超过 $p$ 的最小集合"。如果阈值在第三个 Token 才越过，却把第三个删掉，集合的累计概率就不够 $p$ 了。代码用 `shifted_mask` 把删除掩码右移一位，保留边界；`test_top_p_boundary` 的精确例子 `[0.5,0.25,0.125,0.125]`、$p=0.75$ 断言只保留前两个，说明"恰好等于阈值"也在这里停止。

**Q4：为什么 `sorted_logits[remove_mask] = -inf` 之后还要 `scatter_`？**
排序后张量的顺序是"按概率从高到低"，而调用者需要的是原始词表顺序。`scatter_(-1, sorted_indices, sorted_logits)` 把每个分数放回它原来的下标；不恢复顺序，后续 `argmax` 或采样得到的下标含义就全错了。`test_top_p_boundary` 直接检查原下标 `[5,6,1]` 上的值没变。

**Q5：`autoregressive_decode` 为什么只取 `[:, -1, :]`？**
模型对整段序列每个位置都输出 logits，但生成下一个 Token 只关心最后一个位置。取 `[:, -1, :]` 把 `[batch, seq, vocab]` 压成 `[batch, vocab]`；如果 `logits_fn` 已经返回二维，就跳过这步。测试里的 `fake_logits` 返回三维，正是为了覆盖这条分支。

## 动手验证：怎么知道自己做对了
五个测试依次直接调用，全过打印 `✅ 解码策略测试通过`。
- `test_temperature_contract`：$T=0.5$ 时 `scaled[5]-scaled[6]` 等于原差值乘 2；形状、dtype 不变；`argsort` 完全一致；`T=0.0` 抛 `ValueError`。
- `test_top_k_filtering`：固定 logits 取 `top_k=3` 后有限值恰好 3 个；形状、dtype、device 不变；`[[2,2,2,1]]` 取 `top_k=2` 后有限值 3 个。
- `test_top_p_boundary`：固定 logits 取 `p=0.8` 后有限值 3 个，原下标 `[5,6,1]` 的值保持不变；`log([0.5,0.25,0.125,0.125])` 取 `p=0.75` 后有限掩码是 `[[True,True,False,False]]`；`p=0.0`、`p=1.1` 抛错。
- `test_selection_and_autoregressive_loop`：贪心返回下标 5；两个种子 7 的 generator 输出相同；`fake_logits` 生成形状 `(1,5)`，最后三个 Token 都是 `3`。
- `test_batch_and_input_contract`：batch 2 输入经 Top-K/Top-p 后形状仍为 `[2,10]`；单条 1D 输入经 `decode_next_token` 返回 `(1,1)`。
notebook 没有保存执行输出，以上以测试断言和代码为准；自己跑时看到最后一行 `✅ 解码策略测试通过` 即全部通过。
