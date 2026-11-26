# PT-RL-SYSTEM. 采样、训练与奖励系统 — 小白讲解

> 对应 notebook：[`PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb`](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb)

## 一句话总结
本节先把"每秒 24 组提示、每秒 80 条回答"换成同一个工作单位，说明为什么 80 条/秒在每组 4 条回答时只等于 20 组/秒、队列仍然增长；然后用 `pipeline_capacity`、`transfer_ledger`、`evaluate_topology` 三个函数实现容量换算、跨池传输账本和"先拒绝、再调优、最后接受"的拓扑准入。最后用 verl 单卡实验比较生成引擎"释放 / 保留"两种模式下的阶段耗时、恢复调用和显存驻留。

## 为什么需要它：痛点是什么
后训练系统里，生成器、奖励器、参考策略和学习器可能共享同一张卡，也可能拆到不同资源池。初学者最容易犯的错，是把不同阶段处理的对象都笼统叫"一个样本"：生成器数的是提示组，奖励器数的是单条回答，学习器等的是一批提示组。单位不统一，就会得出"奖励器 80 条/秒比生成器 24 组/秒快"这种错误结论。

生活类比：一条有四个工位的流水线。第一工位每分钟处理 24 个"托盘"，第二工位每分钟处理 80 个"零件"，每个托盘装 4 个零件。第二工位看似更快，实际只能处理 20 个托盘/分钟，比来料少 4 个，托盘会在它前面越堆越多。要比较工位能力，先把单位换成"托盘/分钟"。

## 核心问题
- 生成、奖励、参考策略、学习器各自持有什么状态，输入输出和工作量倍率是什么？
- 共置和分离各有什么收益与代价？为什么不能把各角色单独峰值直接相加？
- 怎样把不同阶段的处理率统一到"每个工作单位"，并判断哪个阶段先积压？
- 跨池传输的"每秒传输秒数"怎么算？共享链路域为什么相加、独立域为什么取最大值？
- 显存超限、版本未同步、恢复不可用和队列不稳定，哪一类应该直接拒绝，哪一类只是调优？

## 解决思路：一步步看懂

**Step 1：统一工作单位。** 一个提示可能生成 $G$ 条候选轨迹，奖励阶段逐条处理，学习器聚合多个提示组才更新一次。表里五个角色分别是：轨迹生成器（策略、KV Cache、采样状态；输出回答、行为对数概率和策略版本）、奖励/验证器（奖励模型或规则版本）、参考策略服务（冻结参考策略，输出参考对数概率）、学习器（策略、优化器、梯度；输出新检查点）、产物协调器（版本、队列、哈希；每次更新触发一次同步）。只有先选定"每个提示组"或"每条轨迹"作为统一单位，后面的速率才有共同分母。

**Step 2：共置 vs 分离。** 共置减少跨池搬运，但生成侧 KV Cache 会和训练侧优化器、梯度、激活抢显存；释放生成状态再更新，可能用"恢复耗时"换"更新阶段容量"。分离可以独立扩缩容，却增加回答、概率、奖励和新权重的传输，还引入队列与版本滞后。关键方法是沿时间记录状态**是否同时驻留**，而不是把各角色峰值直接相加——两个角色错峰使用显存时，峰值远小于两者之和。

**Step 3：TODO 1——容量换算与瓶颈。** 令阶段 $j$ 每秒处理 $q_j$ 个对象，每个工作单位需要 $m_j$ 个该对象，到达率为 $\lambda$：

$$c_j=\frac{q_j}{m_j},\qquad C_{pipe}=\min_j c_j,\qquad s_j=c_j-\lambda$$

`pipeline_capacity` 的参考答案先校验输入，再算：

```python
effective_rates = {name: float(rate) / float(multiplier) ...}
capacity = min(effective_rates.values())
bottlenecks = sorted(name for name, rate in effective_rates.items()
                     if math.isclose(rate, capacity, rel_tol=1e-9, abs_tol=1e-12))
queue_margins = {name: rate - arrival_rate ...}
unstable_stages = sorted(name for name, margin in queue_margins.items() if margin < 0)
```

`math.isclose` 用来处理并列瓶颈：测试里 `rollout=240/3=80`、`reward=160/2=80`、`learner=80/1=80`，三个都等于 80，所以 `bottlenecks == ['learner', 'reward', 'rollout']`（排序后）。到达率 70 时每个余量都是 10，`unstable_stages == []`、`queue_stable is True`。余量恰好为 0（到达 80）也算稳定，因为 `queue_stable` 只表示"给定流量没有超过容量"，随机到达仍需额外缓冲。到达 90、`reward` 容量 80 时余量 −10，`unstable_stages == ['reward']`。非法输入也在这里拦住：阶段集合与倍率集合不一致、速率非有限或非正、到达率为负，都抛 `ValueError`。

**Step 4：TODO 2——跨池传输账本。** 对跨池边 $e$，单对象大小 $z_e$ 字节，每个工作单位有 $m_e$ 个对象，每秒工作单位数为 $\lambda_e$，带宽 $B_e$，可隐藏比例 $o_e$：

$$b_e=z_e m_e\lambda_e,\qquad u_e=\frac{b_e}{B_e},\qquad x_e=u_e(1-o_e)$$

$b_e$ 的单位是字节/秒，$u_e$ 是"每秒需要多少秒传输"，$x_e$ 是其中没有被隐藏的部分。同一链路域 $f$ 的边共享带宽，所以相加；不同域并行，取最大值：

$$X_f=\sum_{e\in f}x_e,\qquad X_{\mathrm{critical}}=\max_f X_f$$

`transfer_ledger` 先看 `placement[source] != placement[target]`，同池边直接令 `traffic=0`，但仍然保留一行记录。跨池边才累加 `total_bytes`，并按 `fabric_id` 聚合。两个校验很容易漏：`overlap_fraction` 必须在 0 到 1 之间；同一个 `fabric_id` 的所有边必须使用相同带宽口径，否则抛 `ValueError`。

用测试 fixture 手算：`learner(B) → rollout(A)` 是 1000×1×1 ÷ 2000 = 0.5 秒/秒，`overlap=0` 所以暴露 0.5；`reward(A) → learner(B)` 是 500×1×1 ÷ 2000 = 0.25，暴露 0.25；两者同在 `pcie-1`，相加得暴露 0.75、原始传输 0.75。`verifier(C) → learner(B)` 是 1000×1×1 ÷ 2000 = 0.5，`overlap=0.5` 所以暴露 0.25，落在独立的 `pcie-2`。`rollout → reward` 同属 A 池，载荷记 0。因此 `cross_pool_bytes_per_s == 2500`（1000+500+1000），`fabric_exposed_seconds_per_s == {'pcie-1': 0.75, 'pcie-2': 0.25}`，`critical_exposed_seconds_per_s == 0.75`，`critical_fabrics == ['pcie-1']`。两条边的暴露恰好看齐时，`critical_fabrics` 会同时列出两个域。

**Step 5：TODO 3——先拒绝，再调优。** `evaluate_topology` 的优先级写在 `decision` 一行：

```python
decision = 'reject' if hard_failures else ('tune' if tuning_pressures else 'accept')
```

硬失败包括：任一链路域的**原始**占用 `raw > 1.0`（即每秒需要超过 1 秒的传输，物理上不可行，记 `link_capacity`）、显存超过上限（`memory_limit`）、`version_sync_ok` 为假、`recovery_ok` 为假。注意这里看的是原始传输，不是曝光后的时间——`test_raw_bandwidth_cannot_be_hidden` 用 `overlap_fraction=1.0` 把暴露抹成 0，但两条边各 0.6 秒/秒让 `fabric_raw_seconds_per_s['shared'] == 1.2`，仍然 `reject` 且原因含 `link_capacity`；把每条边改成 1000 字节、正好 1.0 秒/秒时边界通过，变 `accept`。

调优项包括：存在 `unstable_stages` 或 `queue_stable` 为假（`unstable_queue`），以及某链路域暴露超过预算或 `critical_exposed_seconds_per_s > max_exposed_s_per_s`（`exposed_transfer`，默认预算 0.1 秒/秒）。`test_gate` 里 `unstable` 加 `high_transfer`（`pcie-0` 暴露 0.20）得到 `tune`，原因 `['unstable_queue', 'exposed_transfer']`；而显存 50 超过 40、版本和恢复都为假时得到 `reject`，原因 `['memory_limit', 'version_sync', 'recovery_not_ready']`，不会混入调优项。`reasons` 只保留最高优先级那一组，`unstable_stages`、`overloaded_fabrics`、`over_limit_pools` 这些明细字段另外保留。

**Step 6：verl 单卡实验读什么。** 5.1 固定 `Qwen2.5-0.5B-Instruct`、LoRA rank 16、一次监督更新，只改 `free_cache_engine`（引擎释放/保留）。采集每 0.25 秒尝试采样整卡与整机占用，短阶段漏采时峰值记 `null`，不能当 0。已保存的 `rtx5070ti_verl_mechanism_closure_v2` 显示 13 项检查全为 `true`，包括两组都完成更新与同步、保留实际阶段事件和恢复调用、配置配对、轨迹来源一致、初始参数一致。测量值：释放组整段峰值 10086 MiB、整机峰值 22872.74 MB；初始化 28.38 秒、105 个采样点、峰值 8184 MiB；`sleep_before_initial_sync` 0.35 秒、1 个采样点、峰值 7772 MiB；`initial_sync` 0.91 秒、4 个采样点、峰值 10086 MiB。监督更新阶段释放组 6392 MiB / 1.367 秒，保留组 10489 MiB / 1.345 秒，分别只有 5 / 6 个采样点。读法：释放组整段峰值出现在**初次同步**，不能用它代替更新阶段占用；整卡峰值包含其他应用，单次配对不能宣布稳态加速，显存差也不等于纯 KV Cache 大小。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 工作单位 | 用来统一各阶段计数的口径，如"每个提示组"或"每条轨迹" |
| 阶段倍率 $m_j$ | 一个工作单位在该阶段需要处理多少个对象 |
| 等价处理率 $c_j$ | 阶段原始速率 ÷ 倍率，换算到统一单位后的能力 |
| 流水线容量 $C_{pipe}$ | 最慢阶段的等价处理率 |
| 队列余量 $s_j$ | 阶段容量减到达率，负值表示会积压 |
| 共置 / 分离 | 多角色共享资源池 / 拆到不同资源池 |
| 链路域（fabric） | 共享同一带宽的一组跨池边 |
| 原始传输 $u_e$ | 每秒需要多少秒传输，不考虑隐藏 |
| 暴露占用 $x_e$ | 原始传输里没有被重叠隐藏的部分 |
| `link_capacity` | 原始占用超过 1 秒/秒，物理上不可行的硬失败 |
| 状态驻留 | 某角色在某个时间段是否占着显存/计算资源 |

## 小白常见疑问

**Q1：生成器 24 组/秒、奖励器 80 条/秒，到底谁快？**
换成同一单位再比。每组 4 条回答时，奖励器的等价处理率是 $80/4=20$ 组/秒，低于 24 组/秒的到达率，余量 −4，队列会持续增长。notebook 的 `pipeline_capacity` 正是用 `rate / items_per_work_unit` 做这个换算；测试里 `240/3`、`160/2`、`80/1` 都化成 80。

**Q2：余量为 0 算稳定吗？**
`queue_stable` 只看是否出现负余量，所以 0 算稳定。测试里 `pipeline_capacity({'reward': 80}, 80, {'reward': 1})` 得到 `queue_margins == {'reward': 0.0}` 且 `queue_stable is True`。但这只是理想匀速流量下的边界，随机到达仍需排队缓冲，notebook 也提醒"余量为零表示理想流量刚好平衡，随机到达时还需预留缓冲"。

**Q3：为什么同池边的 `traffic` 记 0，却还要保留这一行？**
因为它仍然要出现在逐边账本 `rows` 里，方便核对"这条边没有跨池"；只是不计入 `cross_pool_bytes_per_s` 和链路域聚合。测试 fixture 的第一条边 `rollout → reward` 同属 A 池，若误算成 100×2×5=1000 字节/秒，`cross_pool_bytes_per_s` 就不是 2500 了。

**Q4：暴露时间已经很小，为什么还会因为 `link_capacity` 被拒绝？**
因为隐藏只是"时间上被计算盖住"，链路带宽仍然被消耗。`test_raw_bandwidth_cannot_be_hidden` 把 `overlap_fraction` 设为 1.0，暴露为 0，但原始占用累计 1.2 秒/秒，超过物理上限 1.0，所以 `evaluate_topology` 返回 `reject`。notebook 在 Step 3 明确说"重叠隐藏等待却仍消耗链路带宽，物理可行性还需检查原始总载荷"。

**Q5：释放组整段峰值 10086 MiB，为什么不能说它比保留组更省显存？**
因为峰值出现的阶段不同。释放组的 10086 MiB 出现在 `initial_sync`，而更新阶段只有 6392 MiB；保留组更新阶段是 10489 MiB，整段峰值也是 10489 MiB。比较时要看同一阶段的 `phases`，不能拿释放组的整段峰值去对保留组的更新峰值。notebook 5.4 节专门把"整段峰值"和"更新阶段占用"分开解释。

## 动手验证：怎么知道自己做对了
测试入口是 `test_raw_bandwidth_cannot_be_hidden()`，随后依次跑 `test_capacity()`、`test_transfer()`、`test_gate()`，全部通过打印 `✅ 后训练系统机制测试通过`。
- `test_capacity`：`pipeline_capacity({'rollout':240,'reward':160,'learner':80}, 70, {'rollout':3,'reward':2,'learner':1})` 的 `effective_rates` 三项都是 80.0，`bottlenecks == ['learner','reward','rollout']`，`queue_stable is True`；到达率 80 时余量 0.0 仍稳定；到达率 90 时 `unstable_stages == ['reward']`；阶段集合与倍率不一致、速率 `nan` 都抛 `ValueError`。
- `test_transfer`：`cross_pool_bytes_per_s == 2500`，`fabric_exposed_seconds_per_s == {'pcie-1': 0.75, 'pcie-2': 0.25}`，`critical_exposed_seconds_per_s == 0.75`，`critical_fabrics == ['pcie-1']`；`overlap_fraction=1.2` 抛 `ValueError`；同一 `fabric_id` 带宽 2000 与 4000 混用抛 `ValueError`；两条边暴露打平时 `critical_fabrics == ['pcie-1','pcie-2']`。
- `test_gate`：稳定 + 低传输（原始 0.1、暴露 0.05）返回 `accept` 且 `reasons == []`；不稳定 + 高传输（原始 0.4、暴露 0.20）返回 `tune`，原因 `['unstable_queue','exposed_transfer']`，`overloaded_fabrics == ['pcie-0']`；显存超限 + 版本/恢复失败返回 `reject`，原因 `['memory_limit','version_sync','recovery_not_ready']`；显存 `nan` 抛 `ValueError`。
- `test_raw_bandwidth_cannot_be_hidden`：两条 600 字节/秒的边累计原始 1.2、暴露 0.0，仍因 `link_capacity` 被拒绝；边界 1.0 秒/秒时 `accept`；删掉 `fabric_raw_seconds_per_s` 字段时 `evaluate_topology` 抛 `ValueError`，不能默认接受。

notebook 保存了 GPU 结果，可对照 5.3 的 `read_summary`：`passed` 为 `true`，`release_update_and_sync`、`retain_restore_matches_mode`、`paired_initial_parameters` 等 13 项检查都应为真；`measurements.release.sampled_gpu_peak_mb` 是 10086.0，`sampled_host_peak_mb` 是 22872.7421875，短阶段没有采样时峰值字段为 `null`。dry_run 模式只打印结果路径，不生成实验结论。
