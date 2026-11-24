# 38. Prefill Decode Scheduling Beginner | Prefill/Decode 调度 · 小白入门

> 配套 notebook：[38. Prefill/Decode 调度](./38_Prefill_Decode_Scheduling.ipynb) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

正在生成的请求，会被新来的长输入拖慢。这本 notebook 把长输入切成小块，并在每一轮里决定先照顾解码还是先处理输入。

## 为什么需要它

3 个请求正在逐词元生成，这时又来了一个长输入。预填充要把输入里还没算过的部分算一遍，一次占用很多计算和读写。即使还有空闲名额，原有请求的下一个词元也可能要等更久。

先照顾谁，各有代价：

- 先解码。正在生成的请求很顺畅，但新请求可能更晚拿到第一个词元。
- 先预填充。新请求很快开始，但正在生成的请求会停顿一下，输出间隔变长。

一个折中办法是把输入切块。每个预填充请求一轮最多处理一块，每做完一块，调度器都可以回头推进解码。不过分块也增加了提交和调度的次数，每块多大、要不要先给解码留位置，仍会改变等待和吞吐。

这本 notebook 做两件事：把输入按顺序切成块，再给一轮里的解码和预填充分配名额。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 预填充（prefill）/ 解码（decode） | 预填充处理还没算过的输入；解码利用历史 KV 往下生成 |
| 未命中后缀（suffix） | 输入前面已经命中缓存的部分可以复用，剩下还要算的那一段 |
| 分块预填充（Chunked Prefill） | 把未命中后缀切成按顺序排列的几块，一轮只处理其中一部分 |
| 单轮词元预算 | 一轮里送入模型的位置总数上限，即 `max_num_batched_tokens`。解码请求每个占 1 个位置，预填充一块可能占多个 |
| 块上限 | 预填充一块最多多少个词元，即 `max_prefill_chunk_tokens` |
| 阶段配额（phase budget） | 一轮里解码和预填充各分到多少位置，以及先做哪个 |
| 两阶段分离 | 让预填充和解码使用各自独立的执行资源，和“同一实例内分配名额”不是同一种方案 |

## 核心思路

输入是每个预填充请求还剩多少输入词元没算、本轮有哪些解码请求，以及两个数：本轮预算和块上限。

```text
1. 切块：把未命中后缀按顺序切成几块，每块最多 chunk_size 个词元，最后一块可以更短
2. 排顺序：决定这一轮先做解码还是先做预填充
3. 扣预算：解码请求每个占 1 个位置
            预填充每块占“剩余输入量、块上限、剩余预算”三者中最小的位置数
输出：本轮执行哪些解码请求、每个预填充请求处理多大的一块、预算还剩多少
```

三种策略只是顺序不同：`decode_first` 是解码 → 预填充，`prefill_first` 反过来，`balanced` 是“解码（最多 `decode_reserve_tokens` 个）→ 预填充 → 解码”。`balanced` 把解码分成两次：先按预留名额放几个解码请求，处理完输入，再用余量补充其余的解码请求。

下面用 notebook 测试和 Step 2 的数字算一遍：

```python
suffix, chunk_size = [9, 10, 11, 12, 13], 2      # 还没算过的输入，每块最多 2 个词元
chunks = [suffix[start:start + chunk_size] for start in range(0, len(suffix), chunk_size)]
print(chunks)
suffixes = {'p1': 5, 'p2': 2}          # 两条输入还剩这么多词元没算
decode_ids = ['d1', 'd2']              # 两个正在解码的请求
budget, chunk_cap = 4, 2               # 本轮预算 4 个位置，每块最多 2 个

def plan(order, first_decode_limit=None):
    remaining, decodes, chunks = budget, [], []
    for phase in order:
        if phase == 'decode':
            pending = [r for r in decode_ids if r not in decodes]     # 没执行过的解码请求
            limit = remaining if first_decode_limit is None else first_decode_limit
            take = min(len(pending), remaining, limit)                # 这一次解码的名额
            decodes += pending[:take]
            remaining -= take
            first_decode_limit = None
        else:
            for rid, left in suffixes.items():
                size = min(left, chunk_cap, remaining)               # 本块大小
                if size > 0:
                    chunks.append((rid, size))
                    remaining -= size
    return decodes, chunks, remaining

print(plan(('decode', 'prefill')))
print(plan(('prefill', 'decode')))
print(plan(('decode', 'prefill', 'decode'), first_decode_limit=1))
```

```text
[[9, 10], [11, 12], [13]]
(['d1', 'd2'], [('p1', 2)], 0)
([], [('p1', 2), ('p2', 2)], 0)
(['d1'], [('p1', 2), ('p2', 1)], 0)
```

第一行是切块：5 个词元按 2 个一块，最后一块只有 1 个。后三行依次是解码优先、预填充优先、预留 1 个解码名额，每行是“执行了哪些解码、各块多大、剩余预算”。三种安排都用满了 4 个位置，成员却不同。解码优先让两个解码请求都走了，只剩 2 个位置给 `p1`。预填充优先把位置全给了输入，两个解码请求这一轮都没轮到。预留解码时 `d1` 先走，`p1` 和 `p2` 分到 2 个和 1 个位置，最后的第二次解码没有余量可用。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 长输入为什么会打断已有解码，三种请求形态各带来什么压力 | 读 |
| Step 2 | 输入分块怎样给解码留下机会，解码优先、预填充优先、预留解码各付出什么代价 | 读，对照上面的例子 |
| Step 3 | 分块之后争用仍严重时，什么时候才值得评估两阶段分离 | 读 |
| Step 4 · TODO 1 | `chunks` 保持输入顺序，每块至多 `chunk_size` 个词元 | 在 `build_chunked_prefill_plan` 里补 1 行：按 `chunk_size` 依次切片，空输入返回空列表 |
| Step 4 · TODO 2 | `phase_order` 保存各策略的阶段执行次序，`balanced` 需要两次解码机会 | 在 `plan_iteration_budget` 里补 2 行：按策略取 `phase_order`，`first_decode_limit` 只在 `balanced` 下等于 `decode_reserve_tokens` |
| 测试 | 检查分块顺序与尾块、三种策略的成员和预算守恒、非法策略和负长度被拒绝 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实服务上比较整段预填充和分块预填充对解码停顿和短探针首词元延迟的影响 | 需要 `INF-VLLM-011-CU128` 环境，整段和分块两种服务各启动一次；默认是 `RUN_MODE = "dry_run"`，不执行真实后端实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 上的 vLLM 服务，可选，默认是 `RUN_MODE = "dry_run"`。

## 容易踩的坑

- **分块时丢了或打乱了词元**：块要按输入顺序一块接一块，最后一块可以比上限短。测试里 5 个词元按 2 切，得到 `[[9, 10], [11, 12], [13]]`。
- **块大小只看了块上限**：本轮一块的大小是剩余输入量、块上限、剩余预算三者中最小的。预算不够时，这一块要比预先切的上限小。
- **`balanced` 只排了一次解码**：它需要两次解码机会，先按预留名额解码，再处理输入，最后用余量补充解码。第二次解码不能再选已经执行过的请求。
- **把同一个请求排了两次**：一个请求这一轮只能出现在一个阶段，解码请求本轮最多执行一次。同一个请求同时出现在预填充和解码里会直接报错。
- **位置数相同就以为耗时相同**：一轮处理的位置总数要满足预算，但位置数相同不代表耗时相同，历史长度和实际执行内核仍会改变成本。

## 读完后自查

- [ ] 给一串 7 个词元、块上限是 3，能写出切出来的几块。
- [ ] 能写出三种策略的阶段次序，并说出 `balanced` 为什么要两次解码。
- [ ] 预算 4、块上限 2、两个解码请求、剩余输入 5 和 2，能算出三种安排下各自的位置分配。
- [ ] 能说出“一轮里分配名额”和“预填充与解码使用独立资源”为什么不是同一种方案。

## 下一步

- 打开 notebook：[38. Prefill/Decode 调度](./38_Prefill_Decode_Scheduling.ipynb)
- 建议先读：[24. Prefix Cache 匹配与复用 · 小白入门](./24_Prefix_Cache_Matching_and_Reuse_Beginner.md)、[INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界 · 小白入门](./INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds_Beginner.md)、[INF-CONTINUOUS-BATCHING. 连续批处理 · 小白入门](./INF-CONTINUOUS-BATCHING_Continuous_Batching_Beginner.md)
- 做完之后：[INF-SERVING-OVERLAP. 服务重叠与状态交接 · 小白入门](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff_Beginner.md)
