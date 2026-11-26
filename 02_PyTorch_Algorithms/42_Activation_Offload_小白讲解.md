# 42. 激活卸载（Activation Offload） — 小白讲解

> 对应 notebook：[`42_Activation_Offload.ipynb`](./42_Activation_Offload.ipynb)

## 一句话总结
训练前向会留下供反向使用的激活值（activation），本节的 `summarize_activation_offload`、`estimate_round_trip_transfer_ms`、`recommend_offload_policy` 三个 TODO 把"哪些激活值得搬到 CPU、搬多少、搬多久、值不值得"串成一本容量与时间账。结论是：offload 确实能降低 GPU 峰值显存，但同步搬运会进入关键路径；notebook 的 GPU 实测里峰值降了 34.3%，step 时间却涨到 5.87 倍。

## 为什么需要它：痛点是什么
初学显存优化时容易把"显存不够"等同于"模型太大"，于是只想到换小模型或量化。但训练时真正把显存顶爆的，常常是前向产生、反向才用的中间激活。删掉它们反向就算不出来；留着又占地方。
生活类比：书桌（GPU 显存）上摊着一堆做作业要用的草稿纸（激活），后半段解题还要翻回去看。抽屉（CPU/host 内存）容量大但拿取慢。激活卸载就是把暂时不看的草稿先塞进抽屉，用到时再抽出来——桌面清爽了，但每次开抽屉都要花时间，而且如果作业正卡在这一步，等待就暴露出来了。

## 核心问题
- 哪些激活算"冷"，应该优先搬走？排序键是什么？
- 搬出后 GPU 是否回到预算内，`feasible` 到底表示什么？
- 搬运字节数怎么换算成理论往返毫秒数？带宽单位为什么按 GiB/s 算？
- `accept`、`tune`、`reject` 三条策略的阈值边界如何判定？
- offload 与 activation checkpointing 用同一个成本口径怎么比较？
- 真实 GPU 上省下的显存，是否值得多出来的 step 时间？

## 解决思路：一步步看懂

**第 1 步：先建立容量账本（TODO 1）。** 数据类 `ActivationChunkSpec` 记录每个激活块的 `name`、`bytes_`、`reuse_delay_layers`（距下次使用的层数，越大越冷）和 `offloadable`。`summarize_activation_offload` 先把候选按 `(-reuse_delay_layers, -bytes_, name)` 排序：先挑最冷的，最冷里面挑最大的，最后按名字稳定排序。循环里只要 `kept_bytes > gpu_budget_bytes` 就继续搬，搬到回到预算内就 `break`。TODO 1 只差一行：

```python
offloaded_bytes = total_bytes - kept_bytes   # 搬出前后的驻留量之差
```

为什么不能直接写 `total_bytes - gpu_budget_bytes`？因为策略按"整块"选择对象，最后可能搬过头，实际搬出量由被选中块的字节和决定，而不是由预算差决定。`overflow_bytes = max(kept_bytes - gpu_budget_bytes, 0)` 记录仍然超出的部分；`feasible = overflow_bytes == 0` 只说明**不可搬运的对象也没有超预算**，不代表真实 GPU 上一定装得下。

拿测试夹具 `build_activation_fixture` 手算：`embed` 256 MiB/冷度 0、`mid_a` 192/2、`mid_b` 160/3、`tail` 128/1，总计 256+192+160+128 = 736 MiB；预算 `384 * 1024 * 1024`。排序后依次是 mid_b(3)、mid_a(2)、tail(1)、embed(0)。搬 mid_b 后剩 576 MiB，仍超；再搬 mid_a 后剩 384 MiB，正好等于预算，停止。于是 `offloaded_bytes = 352 MiB`、`kept_bytes = 384 MiB`、`offloaded_names == ["mid_b", "mid_a"]`、`kept_names == ["embed", "tail"]`。压力比 `pressure_ratio = 736/384 = 1.9166…` 四舍五入为 `1.917`；节省比 `saved_ratio = 352/736 = 0.478…`，测试要求落在 0.47 到 0.49 之间。

**第 2 步：把字节换成时间（TODO 2）。** 公式是
$$T_{round}=\frac{2D}{W},\qquad T_{exposed}=\max(T_{round}-T_{overlap},0)$$
其中 $D$ 是搬出字节、$W$ 是有效单向带宽、$T_{overlap}$ 是能被计算覆盖的窗口。`estimate_round_trip_transfer_ms` 的骨架已经声明"代码按 GiB/s 解释"带宽，所以 1 GiB = `1024 ** 3` 字节：

```python
one_way_ms = offloaded_bytes / (bandwidth_gbps * (1024 ** 3)) * 1000
return_ms = 2 * one_way_ms          # 搬出一次 + 搬回一次
```

代入 `352 MiB`、`8.0` GiB/s：单程 = 352×2²⁰ ÷ (8×2³⁰) × 1000 = 352/8192×1000 = 42.96875 ms；往返 = 85.9375，`round(..., 2)` 得 **85.94**，正是 `test_round_trip_cost` 的断言。输入为 0 时直接返回 0.0；负字节或带宽 ≤ 0 抛 `ValueError`。注意 summary 里的 `transfer_ms` 是**单程**，往返估算要另调这个函数，两者别混用。

**第 3 步：三档策略（TODO 3）。** 默认阈值 `min_saved_ratio=0.25`、`max_transfer_ms=60.0`。严格档要求预算可行、节省比达标、单程时间不超上限：

```python
meets_strict = (summary.kept_bytes <= summary.gpu_budget_bytes
                and summary.saved_ratio >= min_saved_ratio
                and summary.transfer_ms <= max_transfer_ms)
meets_relaxed = summary.saved_ratio >= min_saved_ratio / 2 and summary.transfer_ms <= max_transfer_ms * 2
```

先 `offloaded_bytes <= 0` 直接 `reject`（没搬东西就没有收益），再拒绝不可行方案，然后 strict→`accept`、relaxed→`tune`、其余→`reject`。测试用同一组 chunks 改带宽验证三档：带宽 8 → 单程 42.97 ms，节省 0.478，`accept`；带宽 4 → 单程 85.94 ms，超过 60 但不超过 120，节省仍达标，`tune`；带宽 1 → 单程 343.75 ms，超过 120，`reject`；预算放大到 1024 MiB 时一块都不用搬，`offloaded_bytes == 0`，`reject`。

**第 4 步：与 checkpointing 用同一把尺子。** `compare_offload_vs_checkpointing` 已完整给出，score = 理论节省字节 ÷ 理论额外时间：
- offload score = `offloaded_bytes / max(transfer_ms, 1e-6)`；
- checkpoint score = `checkpoint_saved_bytes / max(checkpoint_extra_ms, 1e-6)`。

测试里 offload 是 352 MiB / 42.97 ms，checkpoint 若 300 MiB / 80 ms，offload 胜；若 500 MiB / 20 ms，checkpoint 胜。这个 score 只用于排序，不等于真实吞吐，也不是端到端收益。

**第 5 步：框架一致性核对。** 可选单元 `check_activation_offload_framework_conformance` 用 `torch.autograd.graph.saved_tensors_hooks` 包住一次 `tanh(x @ W).square().mean()` 的反向。`pack_hook` 把保存张量 `detach().cpu().clone()` 并记录 `numel × element_size()`，`unpack_hook` 再记录一次。它检查五项：至少捕获一个保存张量、pack/unpack 次数一致、两侧字节多重集一致且总量为正、输出 `allclose(atol=1e-7, rtol=1e-6)`、输入梯度同样 `allclose`。这是"搬运不改语义"的最小证据，不是性能实验。

**第 6 步：GPU 实测怎么读。** Step 5 用 `Qwen/Qwen2.5-0.5B`、batch 2 × 序列 512、BF16，`SavedTensorTransferCounter` 只搬非参数、非叶子的反向保存张量，同步 `non_blocking=False`。notebook 保存的输出显示：GPU 驻留 median 154.01 ms、P95 199.11 ms、allocated 峰值 4222.1 MB；CPU offload median 904.33 ms、P95 1196.40 ms、allocated 2775.6 MB，D2H/H2D 各 2949.9 MB、584 个张量。loss 与全局梯度范数差异均为 0，三道机制 gate 通过；`memory_reduction_mb = 4222.1 − 2775.6 ≈ 1446.5 MB`，比例约 0.343，超过 32 MB 与 1% 的资源门槛；但 `median_time_ratio = 904.33/154.01 ≈ 5.87`，超过 `max_median_time_ratio=2.0`，性能 gate 失败，最终 `decision = "tune"`。读法就是：容量收益成立，但同步搬运的暴露时间太高，下一步要筛选卸载对象、用 pinned memory、异步 copy 和计算重叠。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| activation（激活值） | 前向算出来、反向还要用的中间张量 |
| offload | 把暂时不用的状态搬到 CPU/host，反向需要时再搬回 GPU |
| reuse_delay_layers | 距下次使用的层数，越大越"冷"，越优先搬 |
| D2H / H2D | Device→Host 搬出 / Host→Device 搬回 |
| GiB/s | 带宽单位，1 GiB = 1024³ 字节，和 GB/s 差约 7.4% |
| C_release | 至少需要释放的显存 = max(总驻留 − 预算, 0) |
| T_round | 往返理论时间 = 2D/W |
| T_exposed | 扣掉可重叠窗口后真正进入关键路径的等待 |
| feasible | 不可搬运对象也没超预算，是容量口径而非实测结论 |
| saved_ratio | 搬出字节 / 总驻留字节 |
| saved_tensors_hooks | PyTorch 提供的保存张量 pack/unpack 入口，可用来模拟卸载 |

## 小白常见疑问
**Q1：为什么 `offloaded_bytes` 不直接等于 `total_bytes - gpu_budget_bytes`？**
因为搬出按整块进行，可能一次搬过量。测试里总量 736 MiB、预算 384 MiB，理论差是 352 MiB，恰好等于实际搬出；但如果块大小不能整除，实际值会大于预算差，所以代码用 `total_bytes - kept_bytes` 记录真实选择结果。

**Q2：`transfer_ms` 是单程还是往返？**
summary 里的 `transfer_ms` 是单程（`offloaded_bytes / 带宽`），`estimate_round_trip_transfer_ms` 才乘 2。`recommend_offload_policy` 的 `max_transfer_ms=60.0` 比较的是单程；测试里 8 GiB/s 得 42.97 通过，4 GiB/s 得 85.94 落到 `tune`，就是这个口径。

**Q3：带宽写 8.0，为什么不是 8×10⁹ 而是 8×2³⁰？**
docstring 明确"代码按 GiB/s 解释"。`estimate_round_trip_transfer_ms` 用 `1024 ** 3`，所以 8 表示 8 GiB/s。用十进制 GB 会把时间算小约 7.4%，测试的 85.94 就对不上。

**Q4：`feasible=True` 是不是说明 GPU 一定装得下？**
不是。`feasible` 只检查"搬出后保留量没有超过预算"，并且只针对当前元数据模型；`ActivationChunkSpec` 的注释说它不代表真实 CUDA tensor，也不含 pinned memory、异步拷贝或真实生命周期。真实显存要看 GPU 实验的 `peak_allocated_mb`。

**Q5：GPU 实测省了 34% 显存却判 `tune`，是失败了吗？**
不是失败，是"机制成立但需要调优"。`decision` 的规则是四道 gate 全过才 `accept`；这里 correctness、transfer、resource 都过，只有 `performance_gate`（median 时间比 ≤ 2）失败，所以落到 `tune`。

## 动手验证：怎么知道自己做对了
`run_activation_offload_tests()` 按容量→往返→策略→对照的顺序跑四个测试，全过打印 `✅ CPU 机制测试通过：容量账本、传输时间、策略分支与 checkpoint 对照均符合预期。`；TODO 没填会先捕获 `NotImplementedError` 并打印"请先完成 TODO 部分的代码！"。
- `test_offload_ledger`：`total_bytes == 736 MiB`、`offloaded_bytes == 352 MiB`、`kept_bytes == 384 MiB`、`offloaded_names == ["mid_b", "mid_a"]`、`kept_names == ["embed", "tail"]`、`42.0 < transfer_ms < 44.0`、`pressure_ratio == 1.917`、`0.47 < saved_ratio < 0.49`；空列表的 `pressure_ratio` 与 `saved_ratio` 都是 0.0；不可搬运的 512 MiB 块对 256 MiB 预算，`feasible is False` 且 `overflow_bytes == 256 MiB`；负字节、负冷度、重名三种非法输入都抛 `ValueError`。
- `test_round_trip_cost`：`estimate_round_trip_transfer_ms(352 MiB, 8.0) == 85.94`，输入 0 得 0.0，`(-1, 8.0)` 与 `(1, 0.0)` 抛 `ValueError`。
- `test_offload_policy`：8 GiB/s 得 `accept`、4 GiB/s 得 `tune`、1 GiB/s 得 `reject`、预算 1024 MiB 得 `reject`；负阈值抛 `ValueError`。
- `test_offload_vs_checkpointing`：300 MiB/80 ms 时 `preferred == "offload"`，500 MiB/20 ms 时 `preferred == "checkpointing"`；负值抛 `ValueError`。
- 框架核对函数返回 `status == "passed"` 和五项 `checks` 全为 `True`；notebook 没有保存这个单元的输出，以返回字典为准。
- GPU 单元保存了真实运行输出，可直接对照上面的 154.01 / 904.33 ms、4222.1 / 2775.6 MB、2949.9 MB、584 个张量与 `decision = tune`。若你本地重跑，数值会因硬件不同而变，但判断顺序不变。
