# 19. 算子融合导论 — 小白讲解

> 对应 notebook：[`19_Operator_Fusion_Introduction.ipynb`](./19_Operator_Fusion_Introduction.ipynb)

## 一句话总结
本节不写内核，只用三个账本把「融合能省多少」算清楚：`fusion_traffic` 按分组数算启动次数与组间边界（3 个 4 KiB 张量：不融合 3 次启动、2 个边界、16 KiB 逻辑读写；融合偏置与激活后 2 次启动、8 KiB；假想全融合 1 次启动、0 KiB）；`intermediate_traffic` 按中间张量 U 的消费者集合算写回与读取（只有激活融合省 8 KiB，辅助计算留在外面只省 4 KiB，两个都融合省 12 KiB）；`fusion_decision` 再用 `T=[-2,0,2]`、`b=1` 的 ReLU 样例说明融合会改变运算顺序，并区分「样例错」「形状错」「预算不够」「后端不支持」四种失败。

## 为什么需要它：痛点是什么
初学者对「算子融合」的理解常常停在「把几个算子写进一个 kernel 就更快」。真到分析时答不上三个问题：到底省了哪几次读写？中间结果如果还有别的消费者，写回还能不能省？融合改变运算顺序后，数值还对吗？更常见的错误是把「后端声明支持 add 和 relu」当成「已经生成了融合内核」，把配置能力当成执行事实。

生活类比：厨房流水线「切菜 → 炒 → 装盘」。如果中间那盘半成品只有装盘会用，你可以在同一口锅里接着炒，少洗一个盘子、少端一次；但如果旁边还有一道菜要用这盘半成品（辅助消费者），你就必须把它装盘放在台面上，省不掉这一次装盘和取用。融合不是免费的魔法，它取决于「中间结果还有谁要用」。

## 核心问题
- 三个算子串成一条链，中间张量带来多少次启动、多少逻辑读写？
- 融合一部分和全部融合，省下的字节怎么算？
- 中间结果还有别的使用者时，哪些写回和读取真的能省？
- 融合改变运算顺序会怎样？CPU 样例能查出什么、查不出什么？
- 后端声明支持某几个算子，是否等于真的会生成并执行融合内核？

## 解决思路：一步步看懂

**第 1 步：组间边界账本（Q1）。** 导读把计算写成三步：矩阵乘得到 T，加偏置得到 U，激活得到 Y。`fusion_traffic(num_ops, tensor_bytes, fused_groups)` 把「融合」抽象成「把 $n$ 个算子分成几组」，每组一次启动，组与组之间物化一个中间张量：

```python
materialized_boundaries = fused_groups - 1
intermediate_bytes = 2 * materialized_boundaries * tensor_bytes
return {"launches": fused_groups, "materialized_boundaries": materialized_boundaries,
        "intermediate_bytes": intermediate_bytes}
```

为什么每个边界乘 2？因为生产者要完整写一次，消费者要完整读一次；这就是「物化」的代价。参数校验要求 `num_ops`、`fused_groups` 是整数且 $1 \le \text{fused\_groups} \le \text{num\_ops}$，`tensor_bytes` 为正整数，否则抛 `ValueError`。注意 `num_ops` **只用于这条校验**，真正参与计算的是 `fused_groups`——这是一条容易被忽略的细节。

代入导读的例子：T 和 U 各 1024 个 FP32 元素，`tensor_bytes = 1024 * 4 = 4096` B。三组（完全不融合）：2 个边界，$2\times2\times4096 = 16384$ B，即 16 KiB，启动 3 次。两组（偏置与激活融合）：1 个边界，8192 B，启动 2 次。一组（假想全融合）：0 个边界、0 B、启动 1 次。三个断言把这三组数固定下来：

```python
assert unfused == {"launches": 3, "materialized_boundaries": 2, "intermediate_bytes": 16_384}
assert partial == {"launches": 2, "materialized_boundaries": 1, "intermediate_bytes": 8_192}
assert full == {"launches": 1, "materialized_boundaries": 0, "intermediate_bytes": 0}
```

再把张量放大到 64 MiB、算子链拉长到 4 个、分成 2 组，`scaled["intermediate_bytes"] == 128 * 1024**2`——一个边界、写一次读一次。这个例子说明账本随张量大小线性放大，融合的收益也随之放大。

保存的执行输出正是这三行读法：「分离: 启动 3 次，中间逻辑读写 16 KiB」「偏置与激活融合: 2 次，8 KiB」「全部融合假设: 1 次，0 KiB」。最后一行特意叫「假设」，因为把矩阵乘也融进去未必可行，notebook 只算账，不承诺实现。

**第 2 步：使用者集合决定写回（Q2）。** 上一步是粗粒度分组，这一步换成细粒度：固定 U，问「它还有谁要用」。`intermediate_traffic(tensor_bytes, consumers, fused_consumers)` 的规则是：

```python
outside = consumers - fused_consumers
before_write = tensor_bytes
before_read = tensor_bytes * len(consumers)
after_write = tensor_bytes if outside else 0
after_read = tensor_bytes * len(outside)
```

融合前 U 被写一次、被所有消费者各读一次；融合后，只要还有**任何一个**消费者在融合区域外，就必须写出完整 U 给它读，所以写回是「全有或全无」，省不掉；能省的只是那些已经被拉进融合区域的消费者的读取。

以 `u_bytes = 1024 * 4 = 4096` 为例，notebook 的表格给出三种情况：

| U 的使用方式 | 融合前 | 融合后仍需保留 | 减少 |
|---|---|---|---|
| 仅激活使用 | 写 4 KiB、读 4 KiB | 无完整写回/读取 | 8 KiB |
| 激活融合，辅助计算在外 | 写 4 KiB、读 8 KiB | 写 4 KiB、辅助读 4 KiB | 4 KiB |
| 激活与辅助都融合 | 写 4 KiB、读 8 KiB | 无完整写回/读取 | 12 KiB |

代码断言逐项锁住：`only_activation["total_saved_bytes"] == 8192`；`outside_auxiliary["outside_consumers"] == ["aux"]`、`saved_write_bytes == 0`、`saved_read_bytes == 4096`、融合后写和读都还是 4096；`both_inside["total_saved_bytes"] == 12288`；`unchanged` 用 `fused_consumers=set()`，省 0。参数校验也值得注意：`consumers` 必须是非空 `set`，`fused_consumers` 必须是 `set` 且是前者的子集，否则报「融合使用者须属于全部使用者」。

还有一条一致性断言，把两个账本对上：

```python
assert only_activation["total_saved_bytes"] == (
    fusion_traffic(num_ops=2, tensor_bytes=u_bytes, fused_groups=2)["intermediate_bytes"]
    - fusion_traffic(num_ops=2, tensor_bytes=u_bytes, fused_groups=1)["intermediate_bytes"])
```

两个算子两组是 8192 B（一个边界），一组是 0；差额 8192 正好等于「只有激活」省的 8 KiB。这说明两种建模方式在单消费者场景下是同一件事。

保存输出读作：「仅激活: 省写 4 KiB，省读 4 KiB」「辅助计算在外部: 省写 0 KiB，省读 4 KiB」「两个使用者都在内部: 省写 4 KiB，省读 8 KiB」。第二行最容易被误读——它并不是「没省」，而是省了一半：写回保留，读取省了一次。

**第 3 步：CPU 样例与四项声明检查（Q3）。** ReLU 把负值变 0、保留非负值。取 `t_values = [-2, 0, 2]`、`bias = 1`：

- 正确顺序：先加偏置 `[-1, 1, 3]`，再 ReLU 得 `[0, 1, 3]`；
- 错误顺序：先 ReLU 得 `[0, 0, 2]`，再加偏置得 `[1, 1, 3]`。

第一个元素从 0 变成 1，结果不同——**bias 和 ReLU 不可交换**。notebook 用一行代码把错误版本写出来：

```python
"wrong_order": {"output": [max(value, 0) + bias for value in t_values]},
```

`fusion_decision(sample_matches, shape_matches, u_slice_fits, backend_declared, reorder_bytes=0)` 汇总四类检查，返回 `stated_conditions_met`、`failures` 和 `extra_reorder_bytes`。四个布尔任何一个为 `False` 都会进 `failures`；`reorder_bytes` 只要求非负整数，不参与通过与否。`changes` 字典构造了六个场景：

| 场景 | 改了什么 | `failures` | 是否通过 |
|---|---|---|---|
| `matched` | 不改 | `[]` | 是 |
| `wrong_order` | 先 ReLU 后加 bias | `["sample_matches"]` | 否 |
| `shape_miss` | 输出形状 `(1,3)` | `["shape_matches"]` | 否 |
| `budget_miss` | U 小块预算 11 B | `["u_slice_fits"]` | 否 |
| `backend_miss` | 后端只声明 `{"add"}` | `["backend_declared"]` | 否 |
| `layout_copy` | 额外 24 B 布局转换 | `[]` | 是 |

`u_slice_bytes = len(t_values) * 4 = 12`，预算 11 就不够；后端检查是 `{"add", "relu"}.issubset(plan["declared_ops"])`，少一个就失败。最微妙的是 `layout_copy`：`reorder_bytes = 2 * u_slice_bytes = 24`，多了一次 12 B 写和 12 B 读，但 `stated_conditions_met` 仍为 `True`，`extra_reorder_bytes == 24`。布局转换是成本，不是「不能融合」的理由——它应该被记进账，而不是直接判死。

保存输出依次是六行，前五行 `本例检查通过` 只有 `matched` 为 True，`layout_copy` 也为 True 且额外读写 24 B。

**第 4 步：这些检查的边界。** notebook 反复声明它证明了什么、没证明什么。CPU 三个元素的样例只能抓住「顺序写错」这类明显错误，不能证明一般语义正确；真实验收还要满足预定的浮点容差，并保留副作用（输出之外的可见状态改变，例如共享缓存写入顺序）。`u_slice_fits` 只是「给 U 的三个 FP32 元素预留 12 B」这个分块假设，不代表完整内核的共享内存或寄存器够用。后端能力声明也不等于已经生成或执行了融合内核，真实调用与计时要去 Part03·03 的 SwiGLU 看三条路径的对照。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 算子融合 | 把多个算子合成一次执行，减少中间结果的写回与读取 |
| 中间张量（U） | 两个算子之间被物化出来的结果，融合要消灭的就是它 |
| 物化（materialize） | 为中间结果分配存储并实际写进去 |
| 组间边界 | 两个融合组之间必须写出再读入的那一次完整往返 |
| 使用者（consumer） | 读取某个中间张量的后续运算 |
| 数据局部性 | 刚算出的值趁热用掉，不往返内存 |
| 启动开销 | 每多一组算子就要多提交一次执行 |
| 逻辑读写 | 按算法计数的字节，不等于实际显存流量或耗时 |
| 布局转换 | 为满足实现要求而重排或复制数据，本身有读写成本 |
| 副作用 | 输出值之外的可见状态改变，融合时必须保留 |
| 后端能力声明 | 软件栈声称支持哪些运算，不等于已经生成融合内核 |

## 小白常见疑问
**Q1：融合一定更快吗？**
不一定。notebook 只算逻辑读写和启动次数，不算时间。融合可能带来布局转换（`layout_copy` 的 24 B）、寄存器压力、并行度变化；这些在 `fusion_decision` 里只被记录成 `extra_reorder_bytes`，不会被当成失败，但也不代表免费。

**Q2：U 还有外部使用者时，为什么写回完全省不掉？**
因为外部计算要读完整的 U，必须有一份物化副本。`intermediate_traffic` 用 `after_write = tensor_bytes if outside else 0` 表达「全有或全无」：只要 `outside` 非空，就得写完整的 4 KiB；能省的只有已经拉进融合区域的那些消费者的读取。`outside_auxiliary` 的 `saved_read_bytes == 4096` 就是这一条。

**Q3：两个使用者都融合，为什么省的是 12 KiB 而不是 8 KiB？**
融合前 U 写 4 KiB、两个消费者各读 4 KiB，共 12 KiB；融合后完整 U 既不写也不读，12 KiB 全免。`both_inside` 的 `total_saved_bytes == 12288` 就是这个数。对比「只有激活」：写 4 KiB + 读 4 KiB = 8 KiB。

**Q4：先 ReLU 再加 bias 为什么错？**
`[-2, 0, 2]` 先 ReLU 得 `[0, 0, 2]`，加 1 得 `[1, 1, 3]`；正确顺序是 `[-1, 1, 3]` 再 ReLU 得 `[0, 1, 3]`。第一个元素被 ReLU 截断后再加偏置，偏置「救活」了本该为 0 的位置。`results["wrong_order"]["failures"] == ["sample_matches"]` 就是靠这个差异抓到的。

**Q5：`fusion_traffic` 里的 `num_ops` 到底有什么用？**
只用于参数校验（保证 `1 <= fused_groups <= num_ops`），不参与 `intermediate_bytes` 的计算。真正决定读写的是 `fused_groups`：边界数 = 组数 − 1。所以传 `num_ops=4, fused_groups=2` 和传 `num_ops=2, fused_groups=2` 得到相同的字节数。

## 动手验证：怎么知道自己做对了
三个代码单元都保存了执行输出，可以直接对照；断言失败会抛 `AssertionError`。

- Q1 单元断言 `unfused`、`partial`、`full` 三个字典精确相等，`unfused` 与 `partial` 的 `intermediate_bytes` 差 8192；单算子 `num_ops=1, tensor_bytes=4, fused_groups=1` 得到 1 次启动、0 边界、0 字节；放大版 `scaled["intermediate_bytes"] == 128 * 1024**2`。输出三行分别是 3 次/16 KiB、2 次/8 KiB、1 次/0 KiB。
- Q2 单元断言 `only_activation["total_saved_bytes"] == 8192`、`outside_auxiliary["outside_consumers"] == ["aux"]`、`outside_auxiliary["saved_write_bytes"] == 0`、`outside_auxiliary["saved_read_bytes"] == 4096`、`write_bytes_after == read_bytes_after == 4096`、`both_inside["total_saved_bytes"] == 12288`、`unchanged["total_saved_bytes"] == 0`，并用 `fusion_traffic` 的一致性断言交叉验证。输出三行是「省写 4 KiB，省读 4 KiB」「省写 0 KiB，省读 4 KiB」「省写 4 KiB，省读 8 KiB」。
- Q3 单元断言 `u_values == [-1, 1, 3]`、`reference == [0, 1, 3]`；六个场景的 `failures` 与通过状态如上表，`layout_copy["extra_reorder_bytes"] == 24`；输出六行，`layout_copy` 显示通过且额外读写 24 B。

注意没有专门的测试入口，也没有 benchmark；想加深理解可以把 `u_bytes` 改成 8192，观察三种情况的省写省读是否等比例翻倍，再人为把 `fused_consumers` 传成 `{"aux"}` 看 `outside_consumers` 是否变成 `["activation"]`。
