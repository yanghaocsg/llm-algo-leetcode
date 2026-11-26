# 73. 训练性能分析 — 小白讲解

> 对应 notebook：[`73_Training_Performance_Analysis.ipynb`](./73_Training_Performance_Analysis.ipynb)

## 一句话总结
本节把"训练慢 / 显存快满"拆成三类可核对证据：训练状态字节账本、重复计时的稳定性与阶段中位数、以及 FP32 与混合精度的配对 GPU 测量。CPU 部分用三个 TODO 实现基线校验、阶段汇总和下一步实验选择；GPU 部分在 Qwen2.5-0.5B、batch 1、`seq_len=768`、AdamW 下各跑 3 次 FP32 与 AMP，比较步时间、吞吐和峰值显存。登记结果为 `tune`：步时间从 473.353 ms 降到 270.983 ms（改善 42.75%），但峰值显存从 9471.897 MB 微升到 9476.81 MB，时间达标而显存未达标，不能把一项局部收益当成整体成功。

## 为什么需要它：痛点是什么
初学者看到"这一步花了 120 ms、峰值 9 GB"，通常直接问"该改哪里"。但一个总耗时和峰值回答不了三件事：时间花在输入、前向、反向还是更新？显存是参数、梯度、优化器状态还是激活占的？这次测量本身稳不稳？如果重复三次的步时间是 80、120、200 ms，中位数和均值都会误导，先该做的是复测，而不是换优化器。

生活类比：餐厅出餐慢。你不能只看"客人从进门到离开用了 40 分钟"就决定加灶台，要先看排队、备菜、烹饪、结账各占多久，还要确认观察的那几桌是不是遇到同一个厨师、同一批客流。显存也一样：参数、梯度、优化器状态是常驻的"固定库存"，激活和临时工作区是"随订单起伏的备菜"，它们的峰值不一定在同一时刻出现。

## 核心问题
- 参数、梯度、优化器状态各占多少字节？为什么它们相加不等于显存峰值？
- `step_time_ms`、`samples_per_s` 和 `peak_mem_mb` 分别怎么算？计时范围从哪里到哪里？
- 怎样判断一组重复测量"稳定"？`step_time_cv <= 0.10` 是怎么算出来的？
- 阶段占比为什么用"该阶段中位数 / 各阶段中位数之和"，而不是除以总步时间？
- `capacity_ratio` 用峰值中位数而不是最大值，分别适合回答什么问题？
- 为什么 `capacity_ratio >= 0.90` 时先做激活重算/卸载，而不是继续追最高耗时阶段？
- 已登记的 FP32/BF16 对照为什么是 `tune`，而不是 `accept` 或 `reject`？

## 解决思路：一步步看懂

**第 1 步：先把"状态字节"和"显存峰值"分开。** `summarize_training_memory_ledger(model, optimizer)` 是给定实现，只统计三类真实张量：`model.parameters()` 的参数、`parameter.grad` 的梯度、`optimizer.state` 里所有张量的优化器状态。每个张量的字节数是 `numel() * element_size()`。它明确不计算激活、临时 workspace 和 CUDA allocator 的 reserved 内存，所以结果是一个"静态账本"，不是某一时刻的显存占用。

为什么账本加不出峰值？参数和优化器状态常驻，梯度在反向时产生；激活是前向保存、反向使用的中间结果，会随 batch 和序列长度增长，也会在反向中逐段释放；临时工作区随算子分配和释放。峰值取决于"同一时刻还活着的东西"，而不是所有对象的简单求和。notebook 还区分"活跃占用"（当前张量真正使用的内存）和"分配器预留"（还包括暂未使用、留待复用的内存），后者通常不小于前者。

**第 2 步：计时范围要先说清楚。** `measure_train_step(train_step_fn, warmup=2, iters=8, batch_size=1)` 先跑 `warmup` 次不计时，再跑 `iters` 次并计时：

```python
start = time.perf_counter()
for _ in range(iters):
    train_step_fn()
end = time.perf_counter()
elapsed = (end - start) / iters
```

返回 `step_time_ms = elapsed * 1000`、`samples_per_s = batch_size * iters / (end - start)`、`peak_mem_mb = 0.0`。`samples_per_s` 用的是整个计时窗口的总样本数除以总时间，不是"单步时间取倒数再乘 batch"；两者在无抖动时接近，但有固定开销时会有差别。`peak_mem_mb` 固定为 0.0 是刻意的：CPU 上没有可信的 GPU 峰值，不能凭空造一个数，真实峰值由 Step 5 的 GPU 实验单独测量。

notebook 的 Step 2 列出两种计时边界：包含"批次准备与搬到设备"的整步，和"输入已驻留设备"的设备内测量。后者把数据搬运放在计时之前，但仍然计入梯度清理、前向、反向、参数更新和损失读取；损失读回主机会等待 GPU 工作完成，所以这段等待也在窗口内。基线和候选必须用同一个起止点，否则比较没有意义。

**第 3 步：TODO 1，判断样本能不能进入统计。** `validate_training_baseline(report, minimum_samples=3)` 先检查 `report` 是字典、`workload_id` 是非空字符串、`status == 'ok'`、`samples` 是列表且至少 3 条；然后对每条样本要求四个字段齐全：`step_time_ms`、`samples_per_s`、`peak_memory_mb`、`stages_ms`。标量检查用 `_valid_baseline_number`：排除布尔值和非数值，要求有限；`step_time_ms` 和 `samples_per_s` 必须严格为正，`peak_memory_mb` 允许为 0。阶段检查要求 `stages_ms` 是非空字典、阶段名是非空字符串、每个值有限非负，且所有阶段耗时之和严格为正。要补的是最后一行：

```python
sample_is_usable = scalar_values_valid and stages_valid and stage_names_match
```

`stage_names_match` 要求每一轮的阶段名集合都与第一份可用样本一致：如果第一份有 `input/forward/backward/update`，另一份少了 `update` 或多了 `other`，就不能放在一起求中位数。测试里 `{'other': 100}`、`{}`、`None`、`{'': 10}`、`{'forward': True}`、`{'forward': -1}`、`{'forward': nan}`、`{'forward': 1e308, 'backward': 1e308}` 全部必须判为无效；`peak_memory_mb = 0.0` 和阶段值 0.0 则允许，因为"没占显存"和"该阶段没花时间"本身可以是合法观测。

**第 4 步：TODO 2，汇总阶段并找主导阶段。** `build_training_baseline_evidence(report, memory_ledger, capacity_mb)` 先调用上面的校验，任何 issue 都抛 `ValueError`；`capacity_mb` 必须是有限正数；账本三项必须是非负整数。然后对步时间取 `statistics.median`，并用总体标准差算变异系数：

```python
step_median = statistics.median(step_times)
step_cv = statistics.pstdev(step_times) / statistics.fmean(step_times)
stage_medians = {name: statistics.median(sample['stages_ms'][name] for sample in samples)
                 for name in stage_names}
dominant_stage = max(stage_medians, key=stage_medians.get)
```

`stable` 定义为 `step_cv <= 0.10`。用测试的三条样本手算：步时间 100、102、98，均值 100，总体标准差 $\sqrt{((0)^2+(2)^2+(-2)^2)/3}=\sqrt{8/3}\approx1.633$，所以 `step_time_cv ≈ 0.0163`，远小于 0.10。阶段中位数是 `input=10`、`forward=25`、`backward=50`、`update=15`，主导阶段 `backward`，占比 $50/(10+25+50+15)=0.50$。

注意占比的分母是"各阶段中位数之和"，不是 `step_time_median_ms`。在题目给的样本里两者刚好都是 100 ms，因为四个阶段正好拼成一步；但真实 GPU 上阶段活动可能重叠，不能把有重叠的耗时直接相加当墙钟时间。notebook 的解析特意提醒：这个比例适合题目的阶段记录，不能推广成"GPU 关键路径"。

峰值显存取三次的 `statistics.median`，测试里是 9200、9250、9180，中位数 9200；`capacity_ratio = 9200 / 10000 = 0.92`。为什么用中位数而不是最大值？中位数描述"典型占用"，用于判断常规运行是否接近容量；如果要判断"能不能安全跑"，还要另外看最大峰值，不能只看 `capacity_ratio`。

**第 5 步：TODO 3，选择下一项实验。** `classify_training_bottleneck(evidence, capacity_alert_ratio=0.90, hotspot_share=0.40)` 的顺序是：不稳定先复测；稳定后先看容量；容量不高再看阶段占比：

```python
capacity_is_high = evidence['capacity_ratio'] >= capacity_alert_ratio
```

- `stable` 为假：返回 `status='rerun_required'`、`hypothesis='measurement_unstable'`、`next_experiment='repeat_training_baseline'`。
- 容量达到 0.90：返回 `hypothesis='training_capacity_pressure'`、`next_experiment='76_activation_checkpoint_offload'`，证据列 `peak_memory_mb`、`capacity_ratio`、`memory_ledger`。
- 容量不高但主导阶段占比 `>= 0.40`：返回 `hypothesis=f'{stage}_critical_path'`、`next_experiment=f'test_{stage}_candidate'`。
- 都不满足：返回 `needs_more_evidence`，下一步 `collect_trace`。

为什么容量优先？因为显存接近上限时，任何"再快一点"的优化都可能先把训练跑挂；这时应该先比较激活重算、卸载或状态分片，把容量压下来。测试里 `capacity_ratio=0.92` 直接进入 `76_activation_checkpoint_offload`；把容量改成 0.5、占比改成 0.6 后，才变成 `test_backward_candidate`。返回的 `hypothesis` 仍然是"待验证假设"，不是自动识别的真实关键路径；要证明它限制整步速度，还得看事件依赖和完整更新计时。

**第 6 步：用测试里的小模型核对账本。** `test_training_project_template` 构造 `Sequential(Linear(8,16), Tanh(), Linear(16,4))`，参数量是 $(8\times16+16)+(16\times4+4)=144+68=212$。FP32 下参数账本应为 $212\times4=848$ 字节；反向之后梯度也是 848 字节；AdamW 第一次 `step()` 后为每个参数建立一阶矩和二阶矩两个张量，这两个矩本身是 $2\times212\times4=1696$ 字节。如果实现还为每个参数保存 `step` 计数张量，账本会把它也算进去，所以测试只断言 `gradient_bytes > 0`、`optimizer_state_bytes > 0`、`parameter_bytes` 精确相等、`parameter_dtype == 'torch.float32'`，没有写死 1696；你可以用 848 核对参数和梯度，用 1696 作为优化器状态的下界。

同一测试还检查：`measure_train_step(count_step, warmup=2, iters=3)` 应调用 5 次（2 次预热 + 3 次正式），但 `samples_per_s > 0`；用真实 `train_step` 跑 `warmup=0, iters=2` 后计数器必须是 2；`peak_mem_mb` 必须是 0.0；参数必须被更新、梯度必须存在且有限；非法 `warmup=-1` 和 `iters=0` 必须抛 `ValueError`。这些断言把"计时次数""CPU 不虚构显存""优化器真的更新"三件事钉死了。

**第 7 步：GPU 实验的自变量与固定条件。** 5.1 的配置是 `PERF-CUDA-MECHANISM`、单卡至少 8 GiB；`MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'`、本地 `model_cache`；`WORKLOAD='pressure'`，即 `batch_size=1`、`seq_len=768`、`warmup=2`、`iters=5`；`REPEATS=3`、`LEARNING_RATE=1e-5`、`SEED=42`、`REAL_RUN_MODE='paired'`；结果写到 `benchmarks/results/73_real_gpu_training_<RUN_LABEL>.json`。自变量只有计算精度：baseline 是纯 FP32，candidate 用 `torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True)`。`amp_dtype` 先看设备能力：`capability[0] >= 8` 且 `torch.cuda.is_bf16_supported(including_emulation=False)` 为真就用 `bfloat16`，否则用 `float16`。

控制和配对做得很细：`torch.manual_seed(SEED)`、`torch.cuda.manual_seed_all(SEED)`，输入在 CPU 上用固定 `Generator` 生成一次，然后搬到 GPU，两条路径共用同一份 `input_ids` 和 `labels`；每个 repeat 都重新建模型，`model.config.use_cache = False`、`.train()`、AdamW；`train_step` 做 `zero_grad(set_to_none=True)`、前向、`loss.backward()`、`optimizer.step()`，并把 `loss.detach().item()` 读回主机，所以损失读取的同步等待也在计时范围内。每个模式各跑 `REPEATS=3` 个独立模型实例，最后对 `step_time_ms`、`samples_per_s`、`loss`、`peak_mem_mb`、`peak_reserved_mb` 取算术平均，保留每次的 `runs` 供复核。

**第 8 步：结果指标怎么读。** `summarize_training_result(base, tuned)` 统一了方向：`step_time_delta_ms = base - tuned`、`peak_mem_delta_mb = base - tuned`、`throughput_delta = tuned - base`，正值都表示候选更好。`recommend_training_decision` 的门槛默认是 `min_time_delta_ms=10.0`、`min_memory_delta_mb=512.0`：两项都达到才 `accept`；只有一项改善（`time_improved` 或 `memory_improved`）是 `tune`；两项都没改善是 `reject`。GPU 结果还额外记录 `time_improvement_pct`、`memory_improvement_pct` 和 `meaningful_memory_improved = memory_delta >= 512.0`。

读一份真实报告时按顺序看：先看 `schema_version`、`semantic_id`、`failure` 是否为 `None`、`config.repeats >= 3`；再核对 `config` 里的 `model_id`、`workload`、`batch_size`、`seq_len`、`warmup`、`iters`、`seed`、`optimizer`、`dtype` 是否与固定条件逐字段相等；然后看 `baseline` 和 `tuned` 的 `runs` 是否各有 3 条、每条五个指标是否有限（时间、吞吐、峰值显存、reserved 必须为正，loss 非负）；最后看重算的 summary 和决策。

5.5 的 `review_measured_training_report` 会从 `runs` 重新算均值，组均值用 `math.fsum(...) / repeats` 再保留三位小数，允许 `0.0005` 的末位舍入差；summary 的两个小数位允许 `0.005` 的差。方向标记必须是真正的 `bool`，不能用 0/1 冒充。5.6 的 `close` 还会重新读文件、核对 SHA256 没变、重算 summary，再写决策资产；如果同名决策已存在但内容不同，会报错要求换标签。

**第 9 步：已登记结论怎么理解。** 5.6 的登记表写明结论是 `tune`。notebook 的 dump 没有展示具体数字，但 5.6 引用的正式证据 `benchmarks/results/73_real_gpu_training_rtx5070ti_training_baseline_v1.json` 记录了三次均值：FP32 baseline 的 `step_time_ms=473.353`、`samples_per_s=2.113`、`peak_mem_mb=9471.897`；BF16/AMP tuned 是 `270.983`、`3.698`、`9476.81`。算出来 `step_time_delta_ms=202.37`（改善 42.75%）、`throughput_delta=1.58`，但 `peak_mem_delta_mb=-4.91`（峰值反而增加 0.05%），所以 `time_improved=True`、`memory_improved=False`、`meaningful_memory_improved=False`。时间收益远超 10 ms，显存却没有达到 512 MiB 的改善门槛，按 `recommend_training_decision` 只能 `tune`。这正好说明"混合精度主要省计算时间，不一定省峰值显存"：模型以 FP32 加载，AdamW 的参数、梯度和一阶/二阶矩通常还是 FP32，autocast 只覆盖前向的一部分算子。记录里的 `loss` 从 11.477 变成 11.717，是随机词元上的运行损失，不是验证集质量，不能拿来判断模型好坏。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 训练步 step | 完成一次参数更新的完整流程 |
| 状态账本 | 参数、梯度、优化器状态的真实张量字节数 |
| 激活 activation | 前向保存、反向使用的中间结果，随 batch 和长度增长 |
| 临时 workspace | 算子执行时临时分配、用完释放的内存 |
| 活跃占用 / 预留 | 当前张量真正用的内存 / 分配器额外留住备用的内存 |
| `step_time_ms` | 正式迭代的平均单步耗时，不含预热 |
| `samples_per_s` | 计时窗口内处理的样本数除以总时间 |
| 变异系数 CV | 标准差除以均值，衡量重复测量的相对波动 |
| `stable` | `step_time_cv <= 0.10` |
| 阶段中位数 | 每个阶段在多次重复上的中位数 |
| `dominant_stage_share` | 主导阶段中位数除以各阶段中位数之和 |
| `capacity_ratio` | 峰值显存中位数除以设备容量，描述典型占用 |
| `tune` | 有正向证据但未同时达到时间与显存门槛，继续验证 |

## 小白常见疑问

**Q1：为什么 `step_time_cv` 用 `pstdev` 而不是样本标准差 `stdev`？**
因为这里把三次重复当成当前测量窗口的全部观测，不是在估计某个更大总体的标准差，所以用总体标准差，分母是 $n$ 而不是 $n-1$。测试的三条样本 100、102、98：均值 100，`pstdev` 是 $\sqrt{8/3}\approx1.633$，CV 约 0.0163；若用样本标准差会得到约 2.0，CV 约 0.02，虽然都远低于 0.10，但数值不同。代码的阈值判断以 `pstdev` 为准。

**Q2：CPU 上 `peak_mem_mb` 为什么固定是 0.0，不能估一个吗？**
不能。CPU 测量用的是普通 PyTorch 张量，没有 CUDA allocator 的 `max_memory_allocated`，也没有统一的"峰值"口径；估一个数会让人误以为做了 GPU 测量。测试直接断言 `result['peak_mem_mb'] == 0.0`，并配了"CPU 测量不能虚构 GPU 峰值显存"的提示。真实峰值只在 Step 5 的 GPU 路径里用 `torch.cuda.max_memory_allocated` 和 `peak_reserved_mb` 记录。

**Q3：阶段占比为什么不用 `step_time_median_ms` 做分母？**
因为 `stages_ms` 是题目给的阶段记录，真实 GPU 上各阶段活动可能重叠，简单相加不一定等于墙钟步时间；用"各阶段中位数之和"做分母，至少保证四个占比加起来是 1，描述的是"在这份阶段账本里谁占得多"。notebook 的解析明确说这个比例不能直接当成 GPU 关键路径。要判断真实瓶颈，需要 trace 的事件依赖和完整更新计时。

**Q4：`capacity_ratio` 已经 0.92 了，为什么还要看最大峰值？**
`capacity_ratio` 用的是三次峰值的中位数，描述"典型情况"；如果三次里有一次突然冲到容量上限，典型值可能仍然好看，但那次运行已经 OOM 或触发换页。notebook 在 Step 4 的文字里说明：判断能否安全运行时，还要另外查看最大峰值。代码选择中位数是为了不让单次异常主导"典型占用"的判断，不是说最大值不重要。

**Q5：为什么 `recommend_training_decision` 的门槛是 10 ms 和 512 MiB？**
这是本节示例项目的门槛，不是通用标准。10 ms 相对于测试里的 120 ms 基线约 8%，512 MiB 是明显的容量台阶；只有两项都达到才认为"既快又省"，否则即使有改善也只给 `tune`。测试用 baseline 120 ms/8192 MiB、tuned 98 ms/6144 MiB 得到 22 ms 和 2048 MiB，判 `accept`；用 6 ms/256 MiB 判 `tune`；用 -4 ms/0 判 `reject`。真实项目可以改门槛，但必须同时报告门槛值和原始测量。

**Q6：已登记结果是 `tune`，是不是说明这次 GPU 实验失败了？**
不是。`tune` 表示"有基线、可继续"，不是失败。已登记的正式证据里，步时间从 473.353 ms 降到 270.983 ms、吞吐从 2.113 升到 3.698 samples/s，时间收益 202.37 ms；但峰值显存从 9471.897 MB 变成 9476.81 MB，`peak_mem_delta_mb=-4.91`，`meaningful_memory_improved=False`。时间门槛过了，显存门槛没过，所以不是 `accept`。下一步是拿 76 节比较激活重算/卸载，或用 trace 解释剩下的耗时。`reject` 才是"两项都没有改善"；`measure` 表示还缺完整对照。读登记表时要连 `decision_reason` 和 `next_experiment` 一起看，不能只盯一个词。

## 动手验证：怎么知道自己做对了
CPU 测试单元依次调用六个测试，最后打印 `✅ 训练计时、状态计数与基线分析测试通过。`。dump 里保存的输出是 `✅ 训练性能分析项目模板代码通过基础校验。`，与当前测试函数末尾的打印文字不完全一致；以断言为准，不要用这行文字判断版本。参考代码单元还保存了一个小样本输出 `{'step_time_ms': 0.01, 'peak_mem_mb': 0.0}`，只是演示 CPU 计时，不代表 GPU 结果。

- `test_training_project_template`：预热 2 次 + 正式 3 次应调用 5 次；正式 2 次只调用 2 次；返回字段齐全；`peak_mem_mb == 0.0`；参数有梯度、有限、至少一个被更新；账本参数字节精确、梯度与优化器状态字节大于 0、dtype 为 `torch.float32`；非法 `warmup=-1` / `iters=0` 抛 `ValueError`；summary 的 22.0、2048.0、2.0 和三个方向标记正确；`accept / tune / reject` 三档正确；基线证据 `stable`、`dominant_stage == 'backward'`、`capacity_ratio > 0.90`、下一步是 `76_activation_checkpoint_offload`。
- `test_baseline_contract_rejects_incomplete_reports`：空 `workload_id`、`status='failed'`、只有 2 条样本、第一条缺 `stages_ms` 都必须无效，且缺字段时 `issues` 里要有 `sample_0:incomplete`。
- `test_baseline_contract_rejects_invalid_metrics`：`step_time_ms`、`samples_per_s`、`peak_memory_mb` 取 `None`、`True`、`'1'`、`-1`、NaN、Inf（前两者再加 0）都无效；阶段取空字典、`None`、0、空名、布尔、负数、NaN、两个 `1e308` 相加溢出、阶段名不一致都无效；`peak_memory_mb=0` 和阶段 `input=0` 合法；`minimum_samples` 取 0、-1、`True`、1.5 抛 `ValueError`。
- `test_baseline_evidence_rejects_invalid_inputs`：`capacity_mb` 取 0、-1、`True`、NaN、Inf 抛 `ValueError`；账本取 `None`、`{}`、负梯度、布尔参数都抛；样本步时间 NaN 也抛。
- `test_baseline_evidence_summarizes_stability_and_hotspot`：`stable` 为真；`stage_medians_ms['backward'] == 50.0`；主导阶段是 `backward`；`capacity_ratio > 0.90`。
- `test_bottleneck_hypothesis_prioritizes_remeasurement_and_capacity`：`stable=False` 时下一步是 `repeat_training_baseline`；容量 0.92 时是 `76_activation_checkpoint_offload`；容量 0.5、占比 0.6 时是 `test_backward_candidate`。

GPU 侧自查：`prepare` 只检查路径、PyTorch、CUDA 和模型文件，不加载模型；`measure` 应打印环境预检、runtime snapshot 和完整 JSON，结果文件里 `config.repeats` 为 3、`runs` 各有 3 条；`close` 应先通过 5.5 的 `ready_for_decision`，再核对 SHA256 没变，最后写决策文件。如果 `review_measured_training_report` 报 `config_mismatch:*`，先检查 `WORKLOAD`、`SEQ_LEN`、`WARMUP`、`ITERS` 是否被改过；报 `aggregate_mismatch`，说明保存的均值和从 `runs` 重算的结果差了超过末位舍入容差，不能继续登记。
