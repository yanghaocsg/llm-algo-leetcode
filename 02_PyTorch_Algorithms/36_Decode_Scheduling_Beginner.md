# 36. Decode Scheduling Beginner | 解码调度 · 小白入门

> 配套 notebook：[36. 解码调度](./36_Decode_Scheduling.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

多个请求同时等大模型生成时，每一步只能推进其中一个。这本 notebook 写一个小模拟器：决定谁先走、容量不够时暂停谁，以及暂停后怎样接着做。

## 为什么需要它

设想 2 个请求正在逐词元生成，这时又来了一个长输入请求。模拟器每一步只推进一个请求，所以调度器每次都要回答两个问题：这一步让谁走？它要的 KV 容量从哪里来？

如果只按业务给的优先级排，会有两种麻烦：

- 排在前面的请求每次都被照顾，其他请求只能多等一轮，而且等再久也不会往前挪。
- 新请求需要容量，空闲容量却已经被占光。让它一直等，或者暂停一个正在跑的请求，都得有依据。

所以每次选择都要留下记录：谁被选中，谁被暂停，各个请求等了多久。

这本 notebook 把时间简化成“调度步”，写一个小模拟器。你沿着一串事件，就能看清服务顺序怎样变化，以及一个请求被暂停后要保留哪些进度。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 调度步（scheduling step） | notebook 把时间简化成一步一步：一步只推进一个请求，输入处理占一步，之后每步生成一个词元 |
| 处理阶段（phase） | 这个请求在做什么：`prefill` 是处理输入，`decode` 是逐词元生成 |
| 准入状态（status） | 这个请求有没有拿到容量：`waiting`、`running`、`preempted`、`finished` |
| 准入（admission） | 为一个请求预留 KV 容量。没拿到容量的请求只能继续等 |
| 排序键（sort key） | 把请求的几个属性排成一个元组。Python 从左到右逐个比较，键小的先执行 |
| 等待老化（aging） | 老化策略下，每等够 `aging_interval` 步，有效优先级加 1 |
| 抢占（preemption） | 暂停一个已准入的请求，把容量让给更紧急的请求；被暂停的请求叫受害者（victim） |

## 核心思路

每个请求是一条 `RequestState` 记录。每一步做四件事：

```text
1. 找出还没完成的请求，按排序键从小到大排好
2. 从排头开始逐个尝试准入，第一个拿到容量的是 chosen
3. chosen 前进一步，其余请求的 wait_steps 加 1
4. chosen 达到生成上限，就归还容量，状态改为 finished

前进一步：处于预填充阶段时，这一步只把阶段改成解码；之后每步让 generated_len 加 1
```

排序键是一个元组，从左到右依次比较：

```text
排序键 = (完成余量, 阶段, 缓存命中, -有效优先级, 总长度, 请求编号)
阶段        预填充 0，解码 1
缓存命中    命中 0，未命中 1
有效优先级  = 业务优先级 + 累计等待步数 // aging_interval（只有老化策略加后一项）
完成余量    = 截止步 - 当前时钟 - 预计剩余步数（只有期限策略用；其余给一个很大的数）
预计剩余步数 = 还在预填充就加 1，再加上还没生成的词元数
```

前面的字段分不出先后，才轮到后面的。缓存命中（cache hit）表示这个请求的输入状态已经算过，可以复用。优先级取负号，是因为数值越大越早执行，而排序键是小的在前。

下面把排序、老化和抢占各算一遍：

```python
def key(phase, cache_hit, priority, total_len, rid):
    return (0 if phase == 'prefill' else 1, 0 if cache_hit else 1, -priority, total_len, rid)

a = key('prefill', False, 0, 4, 1)      # 预填充、未命中缓存、优先级 0
b = key('decode', True, 9, 4, 2)        # 解码、命中缓存、优先级 9
print(a, b, a < b)                      # 谁排在前面

aging_interval = 4                      # 老化策略：每等 4 步，有效优先级加 1
for wait in [3, 4, 8]:
    print('wait', wait, '-> priority', 1 + wait // aging_interval)

free = 0                                # 空闲块为 0，唯一的一块分给了 low
low = {'priority': 0, 'admitted': True, 'preemptible': True, 'blocks': 1, 'generated': 3}
new = {'priority': 2, 'admitted': False, 'blocks': 1}
ok = low['admitted'] and low['preemptible'] and low['priority'] < new['priority'] and free + low['blocks'] >= new['blocks']
if ok:
    low['admitted'], free = False, free + low['blocks']    # 受害者归还容量，进度不动
    new['admitted'], free = True, free - new['blocks']     # 新请求拿到容量
print(ok, free, low['generated'])
```

```text
(0, 1, 0, 4, 1) (1, 0, -9, 4, 2) True
wait 3 -> priority 1
wait 4 -> priority 2
wait 8 -> priority 3
True 0 3
```

请求 `a` 的键更小，排在 `b` 前面，`b` 的优先级 9 没起作用。中间三行是老化：优先级 1 的请求等了 8 步，有效优先级涨到 3。最后一行是抢占：空闲块是 0，新请求要 1 块，只能暂停优先级更低的 `low`。`low` 归还 1 块，新请求拿走 1 块，空闲块回到 0，`low` 已生成的 3 个词元保持不变。抢占只找一个受害者，它必须已准入、可暂停、有效优先级比新请求低，并且让出后容量足够。找不到就返回 `None`，这个请求继续等。它恢复时不重置进度，只是重新申请相同的容量。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 三种选择规则，以及排序靠前不等于能执行 | 读 |
| Step 2 | 处理阶段和准入状态是两个维度，各种事件下它们怎样变化 | 读 |
| Step 3 | 排序键各字段的先后，老化的作用范围，抢占只找一个受害者 | 读，对照上面的排序键 |
| Step 4 · TODO 1 | 构造请求状态，并加入调度队列 | 在 `enqueue` 里补一个 `RequestState(...)` 构造，传入优先级、容量需求、截止步和可暂停标记 |
| Step 4 · TODO 2 | 比较固定、老化与期限策略 | 在 `_schedule_key` 里补 1 行 `deadline_slack`：期限策略下等于截止步减时钟再减剩余步数，其余策略给一个很大的数 |
| Step 4 · TODO 3 | 执行准入或选择一个抢占对象 | 写 `_ensure_capacity`：已准入的直接放行，容量够就预留，否则找一个受害者并返回它的编号，找不到返回 `None` |
| Step 4 · TODO 4 | 按排序键选择请求，并通过 `_ensure_capacity` 完成准入、抢占或恢复 | 在 `step` 里补一段循环：按排序键排序，逐个尝试准入，第一个获准的是 `chosen` |
| 测试 | 检查入队状态、排序字段的先后、等待步数不改排序、容量守恒、不可抢占保护、恢复后进度不丢和期限记录 | 运行 |
| 参考代码与解析 | 参考答案、每个 TODO 的解释，以及真实对照实验要固定和记录什么 | 卡住再看 |

整本 notebook 只需要 CPU。末尾的“真实 GPU 验证出口”只是说明真实对照实验的做法，由 70 号 notebook 承接。

## 容易踩的坑

- **把两个维度混在一起**：阶段说“在做什么”，准入状态说“有没有拿到容量”。等待中的新请求已经是预填充阶段；被抢占的请求状态变成 `preempted`，阶段不变。
- **抢占时重置了进度**：受害者的阶段和 `generated_len` 都不能动。它恢复时只是重新申请相同的容量。
- **优先级方向弄反**：这里有效优先级越大越早执行，所以排序键里取负号。真实后端可能采用“小数值更优先”，读配置前先确认方向。
- **以为老化能保证公平**：老化只改变同一阶段、同一命中类别内的优先级，持续到达的预填充请求仍可能压住解码请求。默认策略 `fixed_priority` 下，等待步数只是观测，不影响排序。
- **受害者条件写松了**：受害者必须已准入、可暂停、有效优先级更低，并且让出后容量足够。只尝试一个受害者，不合并多个请求的容量。

## 读完后自查

- [ ] 能说出处理阶段和准入状态各回答什么问题，各有哪几个取值。
- [ ] 能按先后顺序写出排序键的字段，并说明优先级为什么要取负号。
- [ ] 一个请求等了 8 步、`aging_interval` 是 4、优先级是 1，能算出老化策略下的有效优先级。
- [ ] 能说出抢占时要检查受害者的哪几个条件，以及它恢复时哪些东西不变。

## 下一步

- 打开 notebook：[36. 解码调度](./36_Decode_Scheduling.ipynb)
- 建议先读：[21. 解码策略 · 小白入门](./21_Decoding_Strategies_Beginner.md)
- 需要补基础：[37. KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb)
- 做完之后：[INF-CONTINUOUS-BATCHING. 连续批处理 · 小白入门](./INF-CONTINUOUS-BATCHING_Continuous_Batching_Beginner.md)
