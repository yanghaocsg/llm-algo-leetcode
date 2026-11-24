# 37. KV Cache Lifecycle and Capacity Control Beginner | KV Cache 生命周期与容量控制 · 小白入门

> 配套 notebook：[37. KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

推理时保存的 KV Cache 从分配、追加、共享到释放，每一步都占着有限的块。这本 notebook 教你用引用数和块数记账，并在容量不够时判断哪些缓存可以回收。

## 为什么需要它

分页注意力把 KV Cache 切成了固定大小的块，按需分配，这是 22 号 notebook 的内容。接下来要回答：这些块什么时候被占用，什么时候能还回去。

这件事不简单，因为一份缓存可能同时被多个请求用着：

- 共享：两个请求的开头一样，第二个请求直接引用第一个请求算好的缓存，不用再存一份。
- 结束：其中一个请求结束了，它用过的缓存不能马上回收，另一个请求还在用。
- 共享之后继续写：两个请求开头一样，后面的内容不同。新生成的词元必须写到各自的私有状态里，不能改共享的那份。
- 容量不够：新请求要块，池里却没有空闲的了。只能从没人用的缓存里挑着回收，正在用的不能动。

notebook 里有个例子：一个六词元前缀占两块，两个请求共享时仍只占两块；其中一个请求结束，引用数从 2 降到 1，块数不变。所以判断容量不能只看“缓存有几条”，要同时看有效词元数、块数和引用数。

这本 notebook 在 CPU 上模拟这套账：块数怎么取整、共享怎么记引用、追加什么时候要新块、引用归零后怎么处理。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| KV Cache / 物理块（Block） | KV Cache 是推理时保存下来的 K 和 V；物理块是存放它的固定大小单位，每块放固定数量的词元 |
| 前缀（prefix） | 提示词的开头部分。开头相同的请求可以共用同一份缓存 |
| 引用数（ref count） | 还有多少个请求正在使用这份缓存。大于 0 时不能回收 |
| 可复用（reusable） | 引用数已经降到 0，但缓存还留着，等后面的请求来命中 |
| 驱逐（eviction）/ 回收（reclaim） | 回收是立即释放块；驱逐是在需要容量时，按复用价值挑一份没人用的缓存释放掉 |
| 抢占（preemption） | 让活跃请求暂停，把它的状态换出去，或者丢掉以后重算 |
| 预填充（prefill）/ 解码（decode） | 预填充处理提示词并建立初始缓存；解码处理新位置并追加它的 K/V |

## 核心思路

notebook 里的 `KVCacheLifecycleSim` 类模拟一个总块数固定的缓存池。每份缓存是一条 `CacheRecord`，用前缀字符串作身份：`used_tokens` 是有效词元数，`block_count` 是占用的块数，`ref_count` 是引用数，`hits` 是命中次数，`last_used` 是最近访问时间。空闲块数等于总块数减去所有记录的块数之和。

四个 TODO 对应四个动作：

- 取整：给词元数，算至少需要几块。不足一块也算一块。
- 获取（`acquire`）：前缀已经在池里，只把引用数、命中次数加 1 并更新访问时间；不在，才分配新块。
- 追加（`append`）：先算追加后需要几块，再减去现在占用的块数，只申请差值。
- 释放（`release`）：引用数减 1。减到 0 时，按 `keep_reusable` 决定是留着等复用，还是立即回收。

这本 notebook 没有公式。下面用几行简化代码，把 notebook 里的两个例子走一遍：

```python
block_size = 4

def blocks_for(tokens):                       # 词元数 → 至少需要几块（向上取整）
    return (tokens + block_size - 1) // block_size

records = {}                                  # 前缀 → {"tokens": 有效词元数, "blocks": 块数, "refs": 引用数}

def acquire(prefix, tokens):                  # 新请求要用这个前缀
    if prefix in records:                     # 命中已有前缀：只加引用，不分配新块
        records[prefix]["refs"] += 1
    else:                                     # 没命中：分配块，引用记 1
        records[prefix] = {"tokens": tokens, "blocks": blocks_for(tokens), "refs": 1}

def used_blocks():                            # 当前一共占了多少块
    return sum(r["blocks"] for r in records.values())

acquire("shared", 6)
acquire("shared", 6)                          # 第二个请求共享同一个前缀
print(used_blocks(), records["shared"])
records["shared"]["refs"] -= 1                # 一个请求结束：引用减 1
print(used_blocks(), records["shared"])

acquire("decode", 3)
rec = records["decode"]
print(blocks_for(rec["tokens"] + 1) - rec["blocks"])   # 追加 1 个词元：3 → 4
print(blocks_for(rec["tokens"] + 2) - rec["blocks"])   # 再追加 1 个：4 → 5
```

```text
2 {'tokens': 6, 'blocks': 2, 'refs': 2}
2 {'tokens': 6, 'blocks': 2, 'refs': 1}
0
1
```

前两行是共享：6 个词元占 2 块，第二个请求来了以后引用数变成 2，占用的块数没变；一个请求结束后引用数降回 1，块数还是 2，因为缓存还留着。后两行是追加：3 个词元追加 1 个变成 4 个，刚好填满第一块，不需要新块；再追加 1 个变成 5 个，才多要 1 块。

容量不够时，先检查空闲块，再从引用为零的缓存里按“命中次数少、最近没用过、块数多”的顺序挑着驱逐。还有活动引用的缓存不能动。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | KV Cache 在请求生命周期中如何变化：分配、追加、共享与引用、释放引用、驱逐与回收五个事件 | 读 |
| Step 2 | 引用、尾块与容量账本：六个状态量各自的物理意义，以及影响哪个决策 | 读 |
| Step 3 | 容量不足时如何选择动作：回收、驱逐、抢占、排队或拒绝的适用条件和代价 | 读 |
| Step 4 · TODO 1 | 将 Token 数向上取整为物理块数 | 写 `_blocks_for_tokens`：补 1 行 `block_count` |
| Step 4 · TODO 2 | 命中前缀时增加引用、命中和访问时间 | 写 `acquire`：补 3 行 `record.ref_count`、`record.hits`、`record.last_used` |
| Step 4 · TODO 3 | 只为跨尾块边界的部分申请新增块 | 写 `append`：补 4 行 `added_blocks`、`_ensure_capacity` 的调用、`used_tokens`、`block_count` |
| Step 4 · TODO 4 | 引用归零后转为可复用状态或立即回收 | 写 `release`：补引用数减 1 的 1 行，再按 `keep_reusable` 写分支和日志 |
| 测试 | 检查块数取整、共享引用不重复占块、尾块内追加与跨块追加、释放回收和容量保护 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 启动真实的 vLLM 服务，做共享前缀配对和容量压力矩阵两组实验 | 需要 GPU 服务环境 `INF-VLLM-011-CU128`，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，使用独立的 `INF-VLLM-011-CU128` 服务环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **不足一块的尾巴不算块**：只要词元数大于零，不足一个块的尾部也要占一个块。测试里 5 个词元要占 2 块。
- **命中前缀时重新分配块**：前缀命中只增加引用数、命中次数和访问时间，块数和总物理占用都不变。
- **追加时按目标块数申请**：先算追加后的目标块数，只申请它比当前占用多出来的部分。尾块没满时是 0。
- **在共享前缀上原地追加**：引用数不是 1 的记录不能原地追加，要先建立私有后缀。notebook 的做法是直接报错。
- **回收还有引用的缓存**：引用数不能减成负数，引用数大于零的缓存不能被驱逐。

## 读完后自查

- [ ] 6 个词元、每块 4 个位置，能说出两个请求共享时占几块，其中一个结束后呢。
- [ ] 3 个词元的记录追加 1 个词元要不要新块，再追加 1 个呢。
- [ ] 能说出 `CacheRecord` 里 `used_tokens`、`block_count`、`ref_count` 各表示什么。
- [ ] 容量不足时，能说出哪些缓存可以回收，哪些不能。

## 下一步

- 打开 notebook：[37. KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
- 做完之后可以对照：[24. Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.md)、[34. SGLang 基数注意力](./34_SGLang_RadixAttention.md)、[38. Prefill/Decode 调度](./38_Prefill_Decode_Scheduling.md)
- 做完之后：[24. Prefix Cache 匹配与复用 · 小白入门](./24_Prefix_Cache_Matching_and_Reuse_Beginner.md)
