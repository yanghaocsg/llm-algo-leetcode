# 79. 训练状态并行基准 — 小白讲解

> 对应 notebook：[`79_Distributed_Parallel_Benchmark.ipynb`](./79_Distributed_Parallel_Benchmark.ipynb)

## 一句话总结
同一个模型单卡放不下时，加 GPU 不一定就够——因为普通数据并行会在每个进程里各存一份完整的参数、梯度和优化器状态。本节先用 `training_state_ledger` 按 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 / FSDP 计算“每个 rank 到底常驻多少 MB”，再用 `evaluate_candidate` 和 `select_training_state_strategy` 把显存、吞吐、通信占比和运行失败汇总成筛选与排序规则。最后给出一个默认关闭的双卡 DeepSpeed 实验框架，说明真实测量要看哪些量。

## 为什么需要它：痛点是什么
初学者容易背结论：“ZeRO-3 最省显存，所以选 ZeRO-3”。但分片越深，参数收集和梯度通信越多；显存够了却可能慢到无法接受，或者通信占比高但被反向计算掩盖，单步时间并没有同比变长。另一个困惑是：每张卡明明还剩不少显存，为什么仍 OOM？因为账本只算常驻状态，激活、临时 All-Gather 的参数、梯度桶和通信缓冲都会在运行中叠上去。

生活类比：四人合租。DDP 是每人各买一整套家具；ZeRO-1 只分“储物间”（优化器状态）；ZeRO-2 连“行李”（梯度）也分；ZeRO-3 连“日常家具”（参数）也分，用时再去公共仓库取。越往后房间越空，但每天搬进搬出的次数也越多——省空间和花时间是一对交换。

## 核心问题
- DDP、ZeRO-1、ZeRO-2、ZeRO-3、FSDP 各自分片哪几类训练状态？每 rank 常驻量怎么算？
- 为什么 ZeRO-2 的 1600 MB 会变成每进程 700 MB，而不是简单除以 4 得 400 MB？
- 候选筛选为什么要同时保留“显存、吞吐、通信、运行失败”四类原因，而不是遇到第一个失败就返回？
- 多个候选都合格时，排序键 `(-throughput, communication_ratio, peak_memory_mb)` 表达的是什么优先级？
- 通信占比高是否等于训练一定被通信拖慢？真实多卡实验应该固定什么、测量什么？

## 解决思路：一步步看懂

**Step 1：先建立“每进程保存什么”的账本。** notebook 的表把策略拆成三类状态的组合：

| 策略 | 参数 | 梯度 | 优化器状态 |
|---|---|---|---|
| DDP | 复制 | 复制 | 复制 |
| ZeRO-1 | 复制 | 复制 | 分片 |
| ZeRO-2 | 复制 | 分片 | 分片 |
| ZeRO-3 / FSDP 全分片 | 分片 | 分片 | 分片 |

分片因子直接编码在 `shard_factors` 里：`'ddp': (1,1,1)`、`'zero1': (1,1,world_size)`、`'zero2': (1,world_size,world_size)`、`'zero3': (world_size,world_size,world_size)`、`'fsdp_full_shard'` 与 zero3 相同。每个因子就是“这一类状态被切成几份”，除以因子就得到每 rank 的常驻 MB。

**Step 2：TODO 1——按因子算三类状态。** 参考答案：

```python
parameter_factor, gradient_factor, optimizer_factor = shard_factors[strategy]
parameter_per_rank_mb = parameter_mb / parameter_factor
gradient_per_rank_mb = gradient_mb / gradient_factor
optimizer_per_rank_mb = optimizer_mb / optimizer_factor
resident_state_mb = parameter_per_rank_mb + gradient_per_rank_mb + optimizer_per_rank_mb
```

用测试里的 `parameter_mb=400, gradient_mb=400, optimizer_mb=800, world_size=4` 手算：

- DDP：`400/1 + 400/1 + 800/1 = 1600 MB`，每个 rank 都保留完整一份。
- ZeRO-2：参数不分片 `400/1=400`，梯度 `400/4=100`，优化器状态 `800/4=200`，合计 `700 MB`。
- ZeRO-3：`400/4=100` + `400/4=100` + `800/4=200 = 400 MB`。

所以 ZeRO-2 不是 400 MB：它只分了梯度和优化器状态，参数仍然每个进程一份。`test_training_state_ledger` 断言 `1600 > 700 > 400`，并检查 `world_size=0` 和未知策略 `'pipeline'` 都抛 `ValueError`。常见错误是把三类状态都除以 `world_size`，或者忘了 `world_size=1` 时因子退化为 1。返回值只代表**常驻训练状态**，激活值和临时收集缓冲不在内，notebook 的解析专门强调了这一点。

**Step 3：TODO 2——逐项检查候选，保留全部失败原因。** 参考答案按固定顺序收集：

```python
failures = []
if metrics.get('failure'):
    failures.append('runtime')
if metrics['peak_memory_mb'] > memory_limit_mb:
    failures.append('memory')
if metrics['throughput'] < min_throughput:
    failures.append('throughput')
if metrics['communication_ratio'] > max_communication_ratio:
    failures.append('communication')
```

四个判断互相独立，所以一个候选可以同时因为显存、吞吐、通信失败。`test_candidate_gate` 用 `peak_memory_mb=9000, throughput=110, communication_ratio=0.22` 的零二候选：门槛 `(10000, 100, 0.25)` 时四项全过，`eligible=True`；收紧到 `(8000, 120, 0.20)` 时，`set(gate_failures)` 必须正好是 `{'memory', 'throughput', 'communication'}`。再加一个 `failure='oom'` 的副本，`gate_failures` 必须是 `['runtime']`。注意边界用的是严格不等号：显存正好等于上限、吞吐正好等于下限、通信正好等于上限都算通过。若写成“遇到第一个失败就 `return`”，多失败用例只能看到一个原因，测试会失败。

**Step 4：TODO 3——在合格候选里排序。** 参考答案：

```python
selected = sorted(
    eligible,
    key=lambda item: (-item['throughput'], item['communication_ratio'], item['peak_memory_mb']),
)[0]
```

排序键的含义是三级优先：先比吞吐，`-throughput` 让大的排前面；吞吐相同再比通信占比，小的优先；还相同再比峰值显存，小的优先。测试候选里 `ddp` 虽然吞吐 130 最高，但 `eligible=False`，先被筛掉；`zero2` 吞吐 112、通信 0.20、显存 9000，`zero3` 吞吐 105、通信 0.31、显存 7000，最终选 `zero2`。如果只按显存排，会选到 zero3；如果只按吞吐排，可能选到不可运行的 DDP。没有合格候选时返回 `{'decision': 'reject', 'strategy': None, ...}`，而不是从失败项里勉强挑一个，`test_strategy_selection` 用空列表断言 `decision == 'reject'`。

**Step 5：Step 5 的真机实验框架。** GPU 部分默认关闭：`RUN_GPU_EXPERIMENT = False`。固定工作负载是 `WORLD_SIZE=2`、`MODEL_WIDTH=1024`、`MODEL_LAYERS=8`、`MICRO_BATCH=1`、`GRADIENT_ACCUMULATION=2`、`SEQ_LEN=256`、`DTYPE='bf16'`、`WARMUP=2`、`REPEATS=5`，比较 `ZERO_STRATEGIES=['ddp','zero1','zero2','zero3']`。`RUN_GPU_EXPERIMENT=True` 时，5.4 用 `torchrun --standalone --nproc_per_node=2 tools/run_zero_stage_benchmark.py --strategy ...` 逐个策略跑，结果写到 `benchmarks/results/79_distributed_parallel/zero_<strategy>.json`，再用 `tools/aggregate_distributed_evidence.py` 汇总成 `summary.json`。5.5 用 `tools.distributed_result_schema.validate_record` 校验 schema 必须是 `distributed-benchmark/v1`，并通过 `make_asset_reference("distributed.training_state", SUMMARY_PATH)` 生成供 81 号项目消费的稳定引用。

**Step 6：实验指标怎么读。** 自变量是“训练状态策略”（复制 / 分片深度），因变量是各进程耗时（最慢进程决定单步时间）、显存峰值、吞吐代理 `tokens_per_s_proxy`、通信占比和运行失败。`tokens_per_s_proxy` 按微步计时、分子却包含梯度累积次数，统一到完整更新前应以原始耗时为准。notebook 特别区分**通信活动占比**和**未被计算隐藏的通信等待**：通信可能与反向重叠，占比高不一定同比拖长训练；要判断“最后还等了多久”，需要时间线或明确同步区间。5.6 的读法是：先看运行状态（启动、OOM、通信错误分开），再核对模型、批量、精度、硬件和分片策略是否可比，然后比较最慢进程耗时与最大显存；容量、数值和吞吐都满足目标，才把 `ready_for_gpu` 推进到真正采用。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| rank / world_size | 分布式进程编号 / 进程总数，CPU 题里只当作份额编号 |
| DDP | 每个进程复制完整参数、梯度、优化器状态，靠梯度 All-Reduce 同步 |
| ZeRO-1/2/3 | 依次分片优化器状态 / 再加梯度 / 再加参数 |
| FSDP 全分片 | 参数、梯度、优化器状态都分片，计算前按需收集 |
| 常驻状态（resident） | 训练稳定阶段必须一直占着的状态，不含激活和临时缓冲 |
| All-Reduce | 把各进程贡献归约后分给所有进程 |
| Reduce-Scatter | 归约后只把结果分片留给各进程 |
| All-Gather | 把各进程持有的分片收集成完整张量 |
| 峰值显存 | 运行过程中出现过的最大占用，通常高于常驻量 |
| 通信占比 | 通信活动时间占测量窗口的比例，不等于关键路径等待 |
| 吞吐代理 | 用固定配置换算出的速率，需按计时窗口解释 |

## 小白常见疑问
**Q1：ZeRO-2 三类状态总量 1600 MB，为什么每进程是 700 MB，而不是 400 MB？**
因为 ZeRO-2 只分片梯度和优化器状态，参数仍然每个进程复制一份。手算：参数 `400/1=400`、梯度 `400/4=100`、优化器状态 `800/4=200`，合计 700。只有 ZeRO-3 才把参数也分成 4 份，得到 `100+100+200=400`。

**Q2：既然 ZeRO-3 最省显存，为什么测试不直接选它？**
测试里的 `zero3` 通信占比 0.31，超过门槛 0.25（或排序时输给通信更低的 zero2）。`select_training_state_strategy` 的优先级是“先合格、再吞吐、再通信、再显存”，省显存不是唯一目标。notebook 的 Step 3 也明确说“更深的分片适合解决容量压力，频繁收集参数却可能使训练变慢”。

**Q3：通信占比 0.31 的候选一定比 0.20 的慢吗？**
不一定。占比是“通信活动占窗口的比例”，如果通信与反向计算重叠得好，它未必进入关键路径。notebook 要求把“占比”和“未隐藏的通信等待”分开，后者要靠时间线或同步区间测量。

**Q4：`evaluate_candidate` 为什么用 `deepcopy(metrics)`？**
它要追加 `eligible` 和 `gate_failures`，又不想污染调用方的原始测量记录。`deepcopy` 返回独立副本，测试里才能安全地拿同一个 `metrics` 构造 accepted / rejected / runtime_failed 三种情况。

**Q5：`ready_for_gpu` 是不是“可以直接上生产”？**
不是。它只表示“通过 CPU 示例筛选，值得进入真实多卡复测”。真正的采用还要补完整更新的数值一致性、统一计时后的吞吐和通信等待。

## 动手验证：怎么知道自己做对了
入口是 `run_training_state_tests()`，依次运行三个测试，全过打印 `✅ 状态大小、候选条件与策略选择测试通过。`
- `test_training_state_ledger`：`(400,400,800, world_size=4)` 下 DDP 的 `resident_state_mb == 1600`、ZeRO-2 `== 700`、ZeRO-3 `== 400`，且严格递减；`world_size=0` 与 `strategy='pipeline'` 必须抛 `ValueError`。
- `test_candidate_gate`：门槛 `(10000,100,0.25)` 时 `eligible is True`；门槛 `(8000,120,0.20)` 时 `eligible is False` 且失败集合正好是 `{'memory','throughput','communication'}`；带 `failure='oom'` 时只返回 `['runtime']`。
- `test_strategy_selection`：三个候选里只有 `zero2`、`zero3` 合格，最终 `strategy == 'zero2'`、`decision == 'ready_for_gpu'`；空列表返回 `decision == 'reject'`。

notebook 没有保存执行输出，也没有 `summary.json` 的实测内容（GPU 实验默认关闭），不要引用具体加速比或显存数字。自己开启实验时，先看 `summary.json` 的 `decision` 和 `sources` 数量；`validate_record` 通过只说明字段契约正确，不证明性能结论。想验证账本逻辑，可改 `world_size=8` 重算：ZeRO-2 得 `400+50+100=550 MB`，ZeRO-3 得 `50+50+100=200 MB`。
