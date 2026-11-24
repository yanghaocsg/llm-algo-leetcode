# 42. Activation Offload | 激活卸载 · 小白入门

> 配套 notebook：[42. 激活卸载](../42_Activation_Offload.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

反向传播要用的中间结果太多、显存放不下时，激活卸载把暂时不用的搬到 CPU 内存，等用到再搬回来。这本 notebook 教你在 CPU 上做一次预算：搬哪些、搬多久、值不值得。

## 为什么需要它

训练的前向传播会在 GPU 上产生激活值（activation），反向传播算梯度时要用，所以它们得留在显存里。激活值接近显存预算时，就放不下了。

激活检查点（activation checkpointing）的办法是只留边界、反向时重算，代价是多算一遍。激活卸载（activation offload）换了一种代价：把反向传播暂时用不到的激活值搬到 CPU 内存（也叫主机内存，host memory），等反向传播快要用到时再搬回 GPU。GPU 上的压力降下来了，但搬运数据要花时间，这段时间可能拖慢训练。

比如显存预算只够放 384 MiB 的激活值，现在一共有 736 MiB。至少要腾出一部分空间。腾多少、搬哪些、要搬多久，得先算一算。

这本 notebook 做的就是这样一次 CPU 上的预算模拟：挑出搬走的激活值，算出搬运量和时间，最后给出采用、调优还是拒绝的建议。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 激活卸载（activation offload） | 把暂时不用的激活值从 GPU 搬到 CPU 内存，反向传播要用时再搬回来。用搬运换显存 |
| 冷激活值（cold activation） | 激活值里距离反向传播再次读取还有较长时间的那些。越冷越适合先搬走 |
| D2H / H2D | 数据从 GPU 到 CPU，和从 CPU 回到 GPU。搬出是 D2H，取回是 H2D |
| 预取（prefetch） | 反向传播还没走到这一步时，提前把激活值搬回 GPU |
| 带宽（bandwidth） | 每秒能搬多少字节，公式里记作 `W`。代码把 `bandwidth_gbps` 按 GiB/s 解释。1 GiB 等于 1024 MiB，1 MiB 等于 1024 × 1024 字节 |
| 暴露时间 | 搬运如果能和计算同时进行，只有超出那部分计算时间之外的等待，才真正拖慢训练 |
| 账本 | 把一类量的来龙去脉记清楚。容量账本记字节，时间账本记毫秒 |

## 核心思路

输入是一组激活块，每块有名字、字节数、距离下次使用的层数、是否允许搬运，再加上 GPU 预算和带宽。处理分四步：把允许搬运的块按越冷越先排好；从最冷的开始搬，直到留下的不超过预算；用搬出的字节数除以带宽算时间；最后结合预算和节省比例给出策略。输出是一份摘要，以及 `accept`、`tune` 或 `reject` 之一。notebook 的公式是：

```text
C_release = max(C_total - C_budget, 0)     最低释放量
T_round = 2 * D / W                        往返时间
T_exposed = max(T_round - T_overlap, 0)    暴露时间
```

`C_total` 是激活值的总驻留量，`C_budget` 是 GPU 上留给激活值的预算，`C_release` 是至少要释放多少，`D` 是实际搬运量，`W` 是单向有效带宽，`T_round` 是搬出加搬回的时间，`T_overlap` 是可以和搬运同时进行的计算时间，`T_exposed` 是真正落在关键路径（critical path）上的等待。

`D` 由选中的激活块决定，可能比 `C_release` 大，所以搬出量要用总驻留量减去保留量来算。

下面用 notebook 测试里的四个激活块试一下：embed 256 MiB、mid_a 192 MiB、mid_b 160 MiB、tail 128 MiB。预算 384 MiB，带宽 8 GiB/s：

```python
MiB = 1024 ** 2
chunks = [("embed", 256, 0), ("mid_a", 192, 2), ("mid_b", 160, 3), ("tail", 128, 1)]   # (名字, 大小 MiB, 距下次使用层数)
budget, bandwidth = 384 * MiB, 8            # 预算 384 MiB，带宽 8 GiB/s

total = sum(size for _, size, _ in chunks) * MiB
kept, moved = total, []
for name, size, _ in sorted(chunks, key=lambda c: -c[2]):     # 越冷越先搬
    if kept <= budget:
        break
    kept -= size * MiB
    moved.append(name)

moved_bytes = total - kept                  # 搬出量 = 总驻留量 - 保留量
one_way_ms = moved_bytes / (bandwidth * 1024 ** 3) * 1000     # 字节 / (GiB/s * 2^30) 得到秒，再换成毫秒
round_ms, overlap_ms = 2 * one_way_ms, 60.0                   # 假设有 60 ms 的计算可以和搬运重叠
print("搬出：", moved, "保留", kept / MiB, "MiB，搬出", moved_bytes / MiB, "MiB")
print("单程", round(one_way_ms, 2), "ms，往返", round(round_ms, 2), "ms")
print("节省比例", round(moved_bytes / total, 3), "暴露时间", round(max(round_ms - overlap_ms, 0), 2), "ms")
```

```text
搬出： ['mid_b', 'mid_a'] 保留 384.0 MiB，搬出 352.0 MiB
单程 42.97 ms，往返 85.94 ms
节省比例 0.478 暴露时间 25.94 ms
```

mid_b 最冷，先搬；mid_a 次之。搬完这两个，留下的正好是 384 MiB，没有超预算。单程 42.97 ms，往返 85.94 ms；假设有 60 ms 的计算可以和搬运重叠，暴露时间就只剩 25.94 ms。

和激活检查点放在一起看：

```text
Checkpointing：前向只留边界，反向时重算       代价是多算一遍
Offload：       前向后搬到 CPU，反向时取回     代价是搬运和等待
```

策略判断有固定顺序。没有搬出任何激活值，或者搬完仍然超预算，就是 `reject`。否则看节省比例和单程传输时间：同时达到严格阈值是 `accept`，只达到放宽的阈值是 `tune`，都不够是 `reject`。这些阈值是教学用的，不代表真实的工程决策。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 激活值在前向之后、等待反向、反向需要这三个阶段，卸载和检查点分别怎么处理 | 读 |
| Step 2 | 至少要释放多少、实际搬多少、搬一次要多久，真正拖慢训练的是哪部分时间 | 读，对照上面的公式 |
| Step 3 | 怎样挑选搬走的激活值，怎样从容量和搬运两方面做预算决策 | 读 |
| Step 4 · TODO 1 | 计算实际搬出量 | 在 `summarize_activation_offload` 里补 1 行：`offloaded_bytes` 等于 `total_bytes - kept_bytes` |
| Step 4 · TODO 2 | 计算往返搬运时间 | 在 `estimate_round_trip_transfer_ms` 里补 2 行：先算单程毫秒数 `one_way_ms`，乘 2 得到 `return_ms` |
| Step 4 · TODO 3 | 补全策略判断逻辑 | 在 `recommend_offload_policy` 里补两段判断：`meets_strict` 返回 `accept`，`meets_relaxed` 返回 `tune`，其余返回 `reject` |
| 测试 | 检查候选顺序和容量账本、单位换算和非法带宽、三类策略及阈值边界，以及和激活检查点的对照 | 运行 |
| 参考代码与解析 | 参考答案、每个 TODO 的解释和激活检查点对照的说明 | 卡住再看 |
| 框架一致性核对（可选） | 用 PyTorch `saved_tensors_hooks` 替换保存的张量，核对捕获的字节数，以及输出和梯度是否一致 | 需要 `TRAIN-CPU` 环境，可以先跳过 |
| Step 5（可选） | 在 GPU 上比较激活值留在 GPU 和同步卸载到 CPU 两条路径的梯度、峰值显存、搬运量和训练 step 时间 | 需要单张 CUDA GPU、`TRAIN-CUDA-MECHANISM` 环境和足够的主机内存；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。框架核对需要 `TRAIN-CPU` 环境。Step 5 需要 GPU 和 `TRAIN-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **搬出量算成了最低释放量**：实际是按激活块挑的，可能搬得比最低释放量多。搬出量要用 `total_bytes - kept_bytes`。
- **时间和单位算错**：代码把带宽按 GiB/s 解释，结果要换成毫秒。搬出、搬回各一次，往返时间是单程的 2 倍。
- **预算可行就直接采用**：预算可行不等于值得采用。还要把节省比例和理论传输时间一起看，才给出 `accept`、`tune` 或 `reject`。
- **漏掉了不允许搬运的激活块**：它们只能留在 GPU。把能搬的都搬完还是超预算，这个计划就是不可行的。
- **把理论时间当成真实代价**：估算不包含同步、预取时机和链路波动。notebook 的 GPU 实验里，同步卸载（搬运期间计算只能等）让 step 时间变成基线的 5.87 倍，结论只是 `tune`。

## 读完后自查

- [ ] 能说出激活卸载和激活检查点各自的代价。
- [ ] 总驻留量 736 MiB、预算 384 MiB，能说出例子里搬走了哪些、保留了多少。
- [ ] 搬出 352 MiB、带宽 8 GiB/s，能算出单程和往返时间。
- [ ] 能说出往返时间和暴露时间的区别。

## 下一步

- 打开 notebook：[42. 激活卸载](../42_Activation_Offload.ipynb)
- 建议先读：[09. 激活与损失反向传播 · 小白入门](./09_Activation_and_Loss_Backward.md)、[19. 激活检查点 · 小白入门](./19_Activation_Checkpointing.md)
- 需要补基础：[Part 01 · 03. GPU 物理架构与内存层级](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)
- 做完之后：[43. 统一训练内存账本 · 小白入门](./43_Unified_Training_Memory_Ledger.md)
