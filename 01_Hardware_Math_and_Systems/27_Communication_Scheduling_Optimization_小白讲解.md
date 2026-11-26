# 27. 通信调度优化 — 小白讲解

> 对应 notebook：[`27_Communication_Scheduling_Optimization.ipynb`](./27_Communication_Scheduling_Optimization.ipynb)

## 一句话总结
本节用四个纯 CPU 函数把"通信到底拖慢了多少"拆成四层：`ring_allreduce_payload_mib` 把张量大小换成每 rank 的发送量，`communication_time_ms` 把发送量拆成"传输时间 + 启动次数 × 固定开销"，`schedule_gradient_buckets` 给梯度分桶并排出串行时间线，`communication_exposure` 计算通信区间被计算覆盖后还剩多少暴露。核心结论：同样的 192 MiB 总发送量，分 4 个小桶在梯度分批就绪时只暴露 0.9455 ms，合成 1 个大桶反而暴露 3.758 ms；而全部同时就绪时，大桶以 3.758 ms 略胜小桶的 3.782 ms。

## 为什么需要它：痛点是什么
初学者常把"通信量"和"通信时间"当成一回事：既然 All-Reduce 要发 384 MiB，那把带宽一除就完了。但真实的通信时间有两项——**搬数据的钱**和**每发一次消息的固定开销**。同样 256 MiB 拆成 64 次发，传输项不变，启动项却从 0.008 ms 涨到 0.512 ms。第二个误区是"桶越大越好"：大桶少启动几次，但桶内所有梯度都到齐才能发，早到的梯度只能干等。

生活类比：拼车发货。每趟车有固定的出车费（启动开销），油费按货物重量算（传输项）。把 64 个小包裹合成 4 车能省出车费；但如果一车非要等最晚的那件货到齐才走，早到的货就压在仓库里，反而比"小批量先走"更慢。

## 核心问题
- 环形 All-Reduce 的"每 rank 发送量"为什么要乘 $2(P-1)/P$，而不是张量大小本身？
- 同样的总发送量，拆成 1 次、8 次、64 次，时间分别多多少？多出来的部分是什么？
- 梯度分桶的容量从 32 MiB 加到 128 MiB，为什么在梯度分批就绪时反而更慢？
- All-Reduce 的发送量和 All-to-All 的路由负载怎么区分？倾斜路由的 3.25 倍是怎么算的？
- 通信与计算重叠时，"重叠比例 100%"是否等于通信不花时间？

## 解决思路：一步步看懂

### 第 1 步：Q1——先换算发送量，再算传输与启动

环形 All-Reduce 把梯度切成 $P$ 份，在 $P$ 个 rank 之间转两圈（reduce-scatter 一圈、all-gather 一圈）。每个 rank 在每一圈要发送 $(P-1)/P$ 的数据，两圈合计：

$$\text{send per rank} = 2 \times \frac{P-1}{P} \times \text{tensor size}$$

`ring_allreduce_payload_mib(tensor_mib, world_size)` 就是这一行：

```python
return tensor_mib * 2 * (world_size - 1) / world_size
```

4 个 rank、256 MiB 的张量：$256 \times 2 \times 3/4 = 384$ MiB。注意这是**发送量**，接收量相同；它既不是 256 MiB（张量本身），也不是 768 MiB（把收发都算上）。函数校验张量容量非负、`world_size` 是至少 2 的整数——单卡没有集合通信。

`communication_time_ms` 把时间拆成两项：

```python
transfer_ms = payload_mib / 1024 / bandwidth_gib_s * 1000
launch_ms = launches * latency_us / 1000
return {'payload_mib': ..., 'launches': ..., 'transfer_ms': transfer_ms,
        'launch_ms': launch_ms, 'communication_ms': transfer_ms + launch_ms}
```

`payload_mib / 1024` 把 MiB 换成 GiB，除以 `bandwidth_gib_s` 得到秒，再乘 1000 得到毫秒。256 MiB、50 GiB/s 时传输项是 $256/1024/50 \times 1000 = 5.0$ ms；每次 8 μs、8 次是 0.064 ms。notebook 打印 `launches` 为 1、8、64 的三组：5.008、5.064、5.512 ms。

读法：64 次调用只比 1 次多 0.504 ms（$63 \times 8$ μs = 0.504 ms），不是 64 倍。因为传输项是常数 5 ms，启动项是线性增长的小量。这也解释了为什么"通信量一样、时间不同"：差别全在启动次数。函数只要求 `payload_mib >= 0`、带宽为正、延迟非负、`launches` 为正整数；它假设收发可重叠，所以传输项按单向发送量算，不额外乘 2。

### 第 2 步：Q2——分桶：省启动 vs 等就绪

`schedule_gradient_buckets(layers, bucket_cap_mib, bandwidth_gib_s=50, latency_us=8, world_size=4)` 做两件事：按给定层顺序把梯度装进容量不超过 `bucket_cap_mib` 的桶，再给每个桶排一条串行时间线。

装桶规则是"顺序累加，装不下就封桶"：

```python
for layer in layers:
    if pending and pending_mib + layer['size_mib'] > bucket_cap_mib:
        buckets.append((pending, pending_mib)); pending, pending_mib = [], 0.0
    pending.append(layer); pending_mib += layer['size_mib']
```

时间线的三个量：

```python
ready_ms = max(member['ready_ms'] for member in members)   # 桶内最晚就绪
start_ms = max(ready_ms, communication_available_ms)        # 还要等上一个桶发完
duration_ms = communication_time_ms(send_mib, ...)['communication_ms']
end_ms = start_ms + duration_ms
```

`send_mib` 先用 `ring_allreduce_payload_mib(tensor_mib, world_size)` 换成每 rank 发送量，再套 Q1 的公式。最后返回 `bucket_count`、`total_send_mib_per_rank`、`backward_end_ms`（所有层最晚就绪时间）、`communication_end_ms`、`exposed_after_backward_ms`（通信结束减反向结束，小于 0 记 0）和完整 `timeline`。

notebook 的四层梯度是 L4@2 ms、L3@5 ms、L2@9 ms、L1@14 ms，每层 32 MiB。三种桶容量的结果：

| 桶容量 | 桶数 | 每桶张量 | 每 rank 发送 | 单桶耗时 | 通信结束 | 反向结束后暴露 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 MiB | 4 | 32 MiB | 48 MiB | 0.9455 ms | 14.9455 ms | 0.9455 ms |
| 64 MiB | 2 | 64 MiB | 96 MiB | 1.883 ms | 15.883 ms | 1.883 ms |
| 128 MiB | 1 | 128 MiB | 192 MiB | 3.758 ms | 17.758 ms | 3.758 ms |

单桶耗时怎么来的？48 MiB 的传输项是 $48/1024/50 \times 1000 = 0.9375$ ms，加 0.008 ms 启动 = 0.9455 ms；96 MiB 是 1.875 + 0.008 = 1.883 ms；192 MiB 是 3.75 + 0.008 = 3.758 ms。三种配置的 `total_send_mib_per_rank` 都是 192 MiB，因为总张量始终是 $4 \times 32 = 128$ MiB，$128 \times 2 \times 3/4 = 192$。

分批就绪时，4 个小桶构成流水线：第一个桶 2 ms 就绪、2.9455 ms 发完；第二个 5 ms 就绪、5.9455 ms 发完；第三个 9 ms 就绪；第四个 14 ms 就绪、14.9455 ms 发完。前三个桶的通信都藏在反向计算期间，只有最后一个桶在反向结束后还占用 0.9455 ms。桶越大，越多梯度被压进同一个桶等最晚的那层：128 MiB 的单桶要等到 14 ms 才能发，整个 3.758 ms 全部暴露。

但把四层就绪时间都改成 0，结论反过来：小桶串行 4 次，$4 \times 0.9455 = 3.782$ ms；大桶一次 3.758 ms，省了 3 次启动的 0.024 ms。断言就是 `small['communication_end_ms'] == 3.782`、`large['communication_end_ms'] == 3.758`。所以"桶多大"没有统一答案，取决于**就绪时间的分布**：梯度陆续就绪时小桶能提前开工，全部同时就绪时大桶省启动费。

代码有两个前提：按给定层顺序装桶、假设各 rank 的就绪时间和顺序相同；不拆分单层梯度，所以桶容量必须不小于最大单层梯度。校验条件 `0 < layer['size_mib'] <= bucket_cap_mib` 就是这条。

### 第 3 步：Q3——All-Reduce 与 All-to-All 是两种负载

Q3 先把两个原语摆在一起：4 个 rank 对 256 MiB 做环形 All-Reduce，每 rank 发送 384 MiB（Q1 的公式）；All-to-All 的输入则是一张**路由计数矩阵**，行是源 rank、列是目标 rank，每格记录 token 分配次数。

`alltoall_route_ledger(token_matrix, bytes_per_token)` 先校验是至少 2 个 rank 的方阵、每次分配字节数为正整数、计数为非负整数，然后算四组量：

- `remote_sent`：每行去掉对角线后求和，即每个源 rank 发给别人的分配数；
- `remote_received`：每列去掉对角线后求和，即每个目标 rank 从别人那里收到的分配数；
- `received_assignments_including_local`：每列全加，含本地分配，用来区分"网络接收"和"本地专家计算"；
- `max_to_mean_remote_receive_ratio`：最忙 rank 的远程接收量除以平均远程接收量，总远程为 0 时记 `None`。

`remote_send_mib` 用总远程分配数乘 `bytes_per_token` 再除以 $2^{20}$（MiB）。注意这里除的是 $2^{20}$，和 Q1 的 GiB 十进制换算不同，因为字段名写的是 MiB。

平衡路由 `[[16,16,16,16] × 4]`：每行远程 48 次，总计 192 次，乘 4096 字节得 786,432 字节，除以 $2^{20}$ 正好 0.75 MiB。倾斜路由 `[[4,4,52,4] × 4]`：下标 2 的那一行远程是 4+4+4=12（自己那格 52 是本地，不算），另外三行各有 60，总计仍是 192 次，平均每个 rank 远程接收 48 次；目标 rank 2 从另外三个 rank 各收 52 次，共 156 次，$156/48 = 3.25$，断言 `== 3.25`。全零矩阵总远程为 0，比例记 `None`。

这里最容易错的是把对角线算进远程：倾斜矩阵总分配数是 $4 \times 64 = 256$，如果不扣对角线，平均值会算成 64，比例变成 $156/64 = 2.4375$，和断言的 3.25 对不上。另一个容易错的是把"接收"当成"发送"：`remote_sent` 按行统计，`remote_received` 按列统计，只有矩阵对称时两者才相同。

### 第 4 步：Q4——暴露时间 = 通信区间 − 被覆盖的部分

`communication_exposure(comm_interval, independent_compute_intervals)` 接收一个通信区间和一组"独立计算"区间，做三步：

1. **裁剪**：每个计算区间与通信区间求交，`left = max(start, comm_start)`、`right = min(end, comm_end)`，只有 `left < right` 才保留；
2. **合并**：把裁剪后的区间按起点排序，重叠或首尾相接的合成一段。`if not merged or start > merged[-1][1]` 开新段，否则把当前段的右端延长到 `max(...)`。这一步是为了避免两个重叠的计算区间把同一段通信时间算两次；
3. **结算**：`hidden_ms` 是合并后各区间的长度和，`exposed_ms = communication_ms - hidden_ms`，`overlap_ratio = hidden_ms / communication_ms`。

三个用例：`((0,12), [])` 没有计算，隐藏 0、暴露 12、比例 0；`((0,12), [(1,4),(3,8)])` 两个区间并成 `(1,8)`，隐藏 7、暴露 5、比例 $7/12 \approx 0.5833$——如果直接把两段长度相加会得到 $3+5=8$，把重叠的 $(3,4)$ 算了两遍，这就是必须先合并的原因；`((0,12), [(-2,5),(5,20)])` 覆盖整个窗口，隐藏 12、暴露 0、比例 1.0。

notebook 的表格把四个量定义得很清楚：通信总时间是集合通信在设备上的活动区间；已隐藏通信是被独立计算覆盖的部分；暴露通信是没被覆盖、真正加进关键路径的部分；重叠比例是已隐藏除以总时间。最后一句提醒最重要：**重叠比例要和 step 时间一起读**，比例高不等于 step 变快，因为计算本身也可能被通信拖慢；这里假设没有数据依赖、没有资源争用，是理想上界。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 环形 All-Reduce | 数据切成 P 份绕环转两圈，每 rank 只发一部分 |
| 每 rank 发送量 | 单卡在整个集合通信里发出的数据量，不是张量大小 |
| 启动开销（latency） | 每发起一次通信调用要付的固定时间 |
| 梯度桶（bucket） | 把若干层梯度合成一次通信的单位 |
| 就绪时间 | 某层梯度算完、可以开始通信的时刻 |
| 暴露时间 | 通信落在反向计算结束之后、真正拉长 step 的部分 |
| All-to-All | 每个 rank 向所有 rank 发送不同数据的集合通信 |
| 路由计数矩阵 | 行是源、列是目标，格子里是分配次数 |
| 远程比例 | 需要跨设备的那部分分配占比 |
| 重叠（overlap） | 通信与独立计算在时间上并行 |
| 区间并集 | 多个重叠区间合并后的总覆盖，重复部分只算一次 |

## 小白常见疑问

**Q1：为什么 All-Reduce 的发送量要乘 $2(P-1)/P$，直接发整个张量不行吗？**
环形算法把张量切成 P 份，reduce-scatter 阶段每个 rank 发送 $(P-1)/P$ 的张量，all-gather 阶段再发一遍，所以是两倍的 $(P-1)/P$。4 个 rank 就是 $2 \times 3/4 = 1.5$ 倍，256 MiB 变 384 MiB。直接发整个张量是另一种算法（比如递归倍增），计量方式不同，不能混用。

**Q2：64 次调用为什么只多 0.512 ms，而不是 64 倍？**
因为时间 = 传输项 + 启动项，传输项与调用次数无关，始终是 5 ms；只有启动项按次数线性增长，$64 \times 8$ μs = 0.512 ms，比 1 次的 0.008 ms 多 0.504 ms。`communication_time_ms` 返回的 `transfer_ms` 和 `launch_ms` 分开列，就是为了让你看清这一点。

**Q3：桶越大不是越省启动吗，为什么 128 MiB 反而慢？**
在分批就绪的场景里，大桶要等桶内最晚的梯度。128 MiB 的单桶必须等 L1 在 14 ms 就绪才能发，3.758 ms 全部落在反向结束之后；32 MiB 的四个小桶能在 2、5、9、14 ms 分别开跑，前三个被反向计算盖住，只暴露最后一个的 0.9455 ms。只有在四层同时就绪时，大桶才靠少 3 次启动赢 0.024 ms。

**Q4：倾斜路由的 3.25 倍，为什么不是 156 除以 64？**
因为 52 那一格是 rank 2 发给自己的本地分配，不算远程。下标 2 的那一行远程只有 12，另外三行各 60，总计 192 次，平均每个 rank 远程接收 $192/4 = 48$ 次；最忙的 rank 2 收到 $52 \times 3 = 156$ 次，$156/48 = 3.25$。如果把对角线算进去，平均值会变成 64，比例就错了。

**Q5：重叠比例 100% 是不是说明通信免费？**
不是。`communication_exposure((0,12), [(-2,20)])` 返回 `exposed_ms=0`、`overlap_ratio=1.0`，只说明通信区间完全落在计算区间内、不额外拉长这段关键路径。但计算本身可能因为共享带宽、SM 资源而被拖慢，notebook 明说这是"假设没有未解决依赖、资源争用也不改变耗时"的理想覆盖量，实际收益要比较整步完成时间。

## 动手验证：怎么知道自己做对了

这个 notebook 没有保存执行输出，也没有 TODO 空位，以下数字以断言和公式手算为准。

- **Q1**：`ring_allreduce_payload_mib(256, 4) == 384`；`communication_time_ms(256, 50, 8, 8)['transfer_ms'] == 5.0`、`['launch_ms'] == 0.064`。循环会打印 1、8、64 次调用对应的字典，`communication_ms` 分别是 5.008、5.064、5.512。
- **Q2**：三种桶容量的 `total_send_mib_per_rank` 都是 192，`exposed_after_backward_ms` 分别是 0.9455、1.883、3.758（断言用 `==` 精确比较，因为浮点结果恰好落在这些值上）。四层同时就绪时 `small['communication_end_ms'] == 3.782`、`large['communication_end_ms'] == 3.758`。
- **Q3**：`ring_allreduce_payload_mib(256, 4) == 384`；平衡矩阵 `remote_send_mib == 0.75`；倾斜矩阵 `max_to_mean_remote_receive_ratio == 3.25`；全零矩阵该字段为 `None`。
- **Q4**：`communication_exposure((0, 12), [(1, 4), (3, 8)])['hidden_ms'] == 7`；`communication_exposure((0, 12), [(-2, 20)])['exposed_ms'] == 0`。三个用例会打印完整字典，可以直接看 `merged_overlap_intervals` 是不是 `((1, 8),)` 和 `((0, 12),)`。

改参数的建议：把 `latency_us` 从 8 改成 0，观察三种桶配置的暴露时间是否都变成纯传输项（0.9375、1.875、3.75），此时"小桶更早开工"的优势会消失；把 `layers` 的就绪时间全设为 0，再跑一次，就能复现"大桶省启动"的反转。
