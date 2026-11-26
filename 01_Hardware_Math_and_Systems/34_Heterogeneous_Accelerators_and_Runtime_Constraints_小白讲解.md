# 34. 异构加速器与运行时约束 — 小白讲解

> 对应 notebook：[`34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb`](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb)

## 一句话总结
本节把「这个模型要多少显存」拆成按阶段记录的账本：预处理只要 0.5 GB、FP32、`tokenize` 和 Python，生成阶段才要 40 GB、BF16、`prefill/decode/attention` 和 vLLM。`check_device_feasibility` 把设备声明分成 matched / mismatch / needs_profile_check 三态，`handoff_ledger` 算出 4096 个 int32 编号转 int64 后主机同时存在 49152 B、16 GB/s 下纯传输 0.002048 ms，`benchmark_admission` 再说明「声明匹配」离「可以开始测量」还差功能冒烟和质量检查。

## 为什么需要它：痛点是什么
新手最常见的错误是拿整模型的一个数去卡所有环节：听说「这个模型要 40 GB」，就认定分词也必须上 GPU，于是把本来能用的 CPU 排除了；或者看到某张卡标着 64 GB、支持 BF16，就认为它一定能跑，却没注意它只有厂商运行时、没有 vLLM。还有一种更隐蔽的错误：把「配置表上写着支持」当成「这条路径真的跑通过」，把声明匹配、实际可执行、性能达标混成一件事。

生活类比：出差分三段——打车去机场、坐飞机、落地打车。你不会用「全程最长一段要 3 小时」去要求每段都预留 3 小时；每段检查的东西也不同：打车看行李放不放得下（容量），飞机看护照和登机牌（运行时），转机还要看行李要不要重新托运（数据交接）。把全程最大需求套给每一段，就会误判。

## 核心问题
- 为什么要把推理拆成 `prepare` 和 `generate`，分别记录峰值、精度、算子、运行时和集合通信？
- 「声明匹配」「明确冲突」「信息待核对」三态怎么区分？为什么缺字段不能直接判失败？
- 4096 个 token 编号从 int32 转 int64，主机和 GPU 各多出多少字节，传输要多久？
- 同一份 40 GB 状态再接收 32 KB 输入，为什么 40 GB 卡放不下、48 GB 卡可以？
- 声明都匹配了，为什么还不能直接测性能？`benchmark_admission` 每一步卡的是什么？

## 解决思路：一步步看懂

**第 1 步：把「一个模型」拆成阶段需求。** `DEVICE_PROFILES` 是设备声明：CPU 128 GB、精度 `{"fp32", "bf16"}`、算子 `{"tokenize", "route", "postprocess"}`、运行时 `{"python"}`、不支持集合通信；GPU 48 GB、精度 `{"fp16", "bf16", "int8"}`、算子 `{"prefill", "decode", "rope", "attention"}`、运行时 `{"torch", "vllm"}`、支持集合通信；NPU 64 GB、精度 `{"bf16", "int8"}`、算子 `{"prefill", "decode", "attention"}`、运行时 `{"vendor-runtime"}`、支持集合通信。

`REQUEST_PLAN` 才是需求，逐阶段检查的是 `stages` 里的两份：`prepare` 是 0.5 GB、fp32、`{tokenize}`、python、不需要集合通信；`generate` 是 40 GB、bf16、`{prefill, decode, attention}`、vllm、不需要集合通信。顶层那份 40 GB 与 `generate` 一致：整模型口径描述的就是生成阶段，套给 `prepare` 会得出「预处理也要 40 GB」的错误结论，换成小内存边缘设备会直接误杀。

`summarize_device_profile` 的计数规则要单独记住：字段缺失或值为 `None` 时计数返回 `None`，字段存在但是空集合时计数返回 `0`。前者是「没写」，后者是「明确写了没有」。它还会把 `operators`、`runtimes` 排序后返回，所以 GPU 算子排成 `attention, decode, prefill, rope`；单元开头的 `assert summarize_device_profile(DEVICE_PROFILES["gpu"])["operator_count"] == 4` 就是在确认这份声明真列了 4 个算子。

**第 2 步：三态判定——缺信息不等于失败。** `check_device_feasibility(profile, plan)` 先校验五个必需字段（`estimated_peak_gb`、`precision`、`required_ops`、`runtime`、`requires_collective`），缺一项抛 `ValueError("阶段需求缺少必要字段")`。它用 `type(peak) not in (int, float)` 而不是 `isinstance`，因为 `True` 是 `int` 的子类；`math.isfinite` 排除 `nan` 和 `inf`；阶段预算允许为 0，但设备显存必须大于 0。

逐项比较时，结果可以是 `True`、`False`、`None`：

```python
memory_ok = None if memory is None else memory >= peak
precision_ok = None if precisions is None else plan["precision"] in precisions
missing_ops = None if operators is None else sorted(set(plan["required_ops"]) - set(operators))
runtime_ok = None if runtimes is None else plan["runtime"] in runtimes
collective_ok = profile.get("collective") if plan["requires_collective"] else True
```

`missing_ops` 存「要求了但声明没有」的算子；`checks["ops"]` 再转成布尔。`collective_ok` 那一行是关键：阶段不需要集合通信时直接记 `True`，不去看设备支不支持——不能因为设备不支持就排除一个根本不需要它的阶段。若阶段真的要集合通信而设备值是 `None`，这一项是未知，整体进 `needs_profile_check`，而不是替设备猜「应该支持」。

结论按优先级给：有 `False` 就是 `mismatch`，没有失败但有 `None` 就是 `needs_profile_check`，全 `True` 才是 `matched`。三个现成例子覆盖三态：`gpu_match` 全部满足；`npu_mismatch` 的 64 GB 和 bf16 都够、算子也够，但 `vllm` 不在 `{"vendor-runtime"}` 里，`failed_fields` 恰好是 `["runtimes"]`；`gpu_pending` 把运行时置 `None`，其余全满足，于是 `declared_requirements_met` 为 `None`、`unknown_fields` 为 `["runtimes"]`。

**第 3 步：按阶段筛设备。** `assign_stages(profiles, plan)` 对每个阶段独立跑一遍检查，把设备分进 `candidate_devices`（True）、`pending_devices`（None）、`rejected_devices`（False），并保留 `checks` 明细。同一台机器在不同阶段结论可以完全相反：

| 阶段 | CPU | GPU | NPU |
|---|---|---|---|
| `prepare` | 匹配 | 拒绝：无 fp32、缺 tokenize、无 python | 拒绝：无 fp32、缺 tokenize、无 python |
| `generate` | 拒绝：缺算子且没有 vllm | 匹配 | 拒绝：运行时只有 `vendor-runtime` |

断言钉死了这个不对称：`"cpu" in STAGE_TRACE["prepare"]["candidate_devices"]`，而 `STAGE_TRACE["generate"]["candidate_devices"] == ["gpu"]`。CPU 在 `generate` 的 `failed_fields` 同时含 `ops` 和 `runtimes`，修的时候要动两个地方。`PENDING_TRACE` 把 GPU 运行时置空后，`generate` 候选是空列表、待核对是 `["gpu_pending"]`——待核对的设备不算候选，所以后面只能返回「先补声明」。

**第 4 步：交接账本。** `handoff_ledger(num_tokens, source_dtype, target_dtype, bandwidth_gb_s)` 只支持 `int32→int32`、`int64→int64`、`int32→int64` 三种组合，其余抛 `ValueError`。`sizes = {"int32": 4, "int64": 8}`，4096 个编号：

| 字段 | 算法 | 数值 |
|---|---|---|
| `source_bytes` | 4096 × 4 | 16384 B |
| `target_bytes` | 4096 × 8 | 32768 B |
| `host_live_bytes` | `source_bytes + conversion_bytes` | 49152 B |
| `transfer_ms` | 32768 ÷ (16×10⁹) × 1000 | 0.002048 ms |

`conversion_bytes` 只在类型变化时等于目标字节数，所以 `int32→int64` 时主机上源编号和转换结果短暂同时存在，合计 49152 B；`int64→int64` 时是 0，只剩 32768 B。传输时间只算目标字节的纯搬运，类型转换、缓冲分配、提交开销、等待就绪都没算；文档字符串写明「传输窗口不包含转换、分配和提交」，所以 0.002048 ms 是下界，不是真实耗时。

**第 5 步：同时驻留。** `placement_memory_check(resident_bytes, incoming_bytes, capacity_bytes)` 把「已经活着的状态」和「新到的输入」相加：

```python
peak_bytes = resident_bytes + incoming_bytes
"shortfall_bytes": max(peak_bytes - capacity_bytes, 0)
```

`40_000_000_000 + 32768` 对 `40_000_000_000`，差 32768 B，`fits` 为 `False`；48 GB 卡则 `fits` 为 `True`。`shortfall_bytes` 用 `max(..., 0)` 截断，放得下时永远是 0；判断用 `<=`，容量恰好等于峰值也算放得下。这个 32 KiB 的缺口看起来荒谬，却是「预算卡在边界」时最真实的失败方式：日志只报一个 allocation 失败，很难联想到 token 编号的位宽。

**第 6 步：准入流程。** `benchmark_admission(stage_trace, required_stages, quality_gate_passed, runtime_smoke_passed)` 是状态机，顺序不能乱：

1. 校验结果只能是 `bool` 或 `None`，`required_stages` 必须非空；
2. 缺阶段 → `blocked`，`missing` 列出缺的名字（空 `stage_trace` 时是 `["generate", "prepare"]`）；
3. 某阶段既无候选也无待核对 → `blocked`；
4. 质量检查 `False` → `blocked`；
5. 功能冒烟 `False` → `blocked`；
6. 还有阶段只有待核对 → `needs_profile_check`；
7. 功能冒烟、质量检查依次为 `None` → `needs_runtime_check`、`needs_quality_check`；
8. 全通过 → `benchmark_ready`。

两个细节值得注意。第一，已知失败优先于未知：`quality_gate_passed=False` 直接挡住，不会因为还有字段没填就走「待核对」。第二，两个失败同时出现时，代码先判质量，所以原因字符串是「质量检查未通过」。`given_results` 的五个场景跑遍五种状态：`needs_runtime_check`、`blocked`（功能未通过）、`needs_quality_check`、`blocked`（质量未通过）、`benchmark_ready`。而 `benchmark_admission(PENDING_TRACE, ...)` 即使功能和质量都填 `True`，也只能得到 `needs_profile_check`——声明没补齐，后面免谈。

「配置表匹配」只是第一关，后面还有真实加载、编号类型与顺序、缓冲寿命、输出正确性、固定评测集质量，最后才轮到性能数字。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 阶段需求（stage plan） | 一个处理阶段单独记录的内存、精度、算子、运行时要求 |
| 能力声明（device profile） | 设备声称自己有多少内存、支持哪些精度/算子/运行时 |
| `matched` / `mismatch` / `needs_profile_check` | 声明全满足 / 有明确冲突 / 有字段没填所以未知 |
| 集合通信（collective） | 多设备一起完成的通信操作；单机阶段可能根本不需要 |
| 交接（handoff） | 数据从一个阶段/设备交给下一个阶段/设备的过程 |
| 转换缓冲 | 类型转换产生的新副本，会和源数据短暂同时存在 |
| 纯传输时间 | 只算字节 ÷ 带宽，不含转换、分配、提交、等待 |
| 同时驻留 | 明确同时活着的对象字节数相加，再和容量比 |
| 冒烟测试（runtime smoke） | 最小加载 + 一次真实请求，确认整条路径能走通 |
| 质量门（quality gate） | 在固定评测集上确认输出质量达标 |
| 十进制 GB | 1 GB = 10⁹ B，和 GiB（2³⁰ B）不是一回事 |

## 小白常见疑问
**Q1：NPU 有 64 GB，比 40 GB 需求还大，为什么被判 `mismatch`？**
容量只是五项检查之一。NPU 的运行时集合是 `{"vendor-runtime"}`，阶段要求 `vllm`，`runtime_ok` 为 `False`，所以 `failed_fields == ["runtimes"]`。这正是本节反复强调的：放得下 ≠ 跑得起来。

**Q2：既然 `runtimes=None` 只是没填，为什么不保守一点直接判失败？**
判失败会错误排除可能可用的设备，判成功会让未验证的路径混进候选。`gpu_pending` 的结果是 `None`，被放进 `pending_devices` 而不是 `candidate_devices`，`benchmark_admission` 返回 `needs_profile_check`——先补信息，再决定。

**Q3：0.002048 ms 是不是说明这次交接可以忽略？**
不能。这个数只含 32768 B ÷ 16 GB/s 的纯搬运，类型转换、缓冲分配、提交开销、消费者等待都没算；带宽 16 GB/s 本身也是示例假设。真实开销要用执行轨迹测，不能拿下界当结论。

**Q4：同样 4096 个编号，为什么 `int64→int64` 是 32768 B，而 `int32→int64` 是 49152 B？**
`int64→int64` 不需要转换，`conversion_bytes` 为 0，只有源缓冲 32768 B；`int32→int64` 要新开一块 32768 B 的转换结果，源缓冲 16384 B 还在，两者相加 49152 B。`assert unchanged["host_live_bytes"] == 32768` 固定了这个区别。

## 动手验证：怎么知道自己做对了
这个 notebook 没有 `run_*_tests` 之类的测试入口，也没有保存任何执行输出；所有检查都是 cell 末尾的 `assert`，任何一条不成立都会抛 `AssertionError`，以代码和断言为准。

- Q1 单元断言 `summarize_device_profile(DEVICE_PROFILES["gpu"])["operator_count"] == 4`，并核对 `REQUEST_PLAN` 两个阶段的峰值分别是 0.5 和 40。
- Q2 单元断言 `gpu_match["declared_requirements_met"] is True`、`npu_mismatch["failed_fields"] == ["runtimes"]`、`gpu_pending["declared_requirements_met"] is None`、`gpu_pending["unknown_fields"] == ["runtimes"]`。最后一段循环按「声明状态 / 待核对 / 冲突」三列打印三份报告。
- Q3 单元断言 `"cpu" in STAGE_TRACE["prepare"]["candidate_devices"]`、`STAGE_TRACE["generate"]["candidate_devices"] == ["gpu"]`、`PENDING_TRACE["generate"]["candidate_devices"] == []`、`PENDING_TRACE["generate"]["pending_devices"] == ["gpu_pending"]`；交接断言 `source_bytes == 16384`、`target_bytes == 32768`、`host_live_bytes == 49152`，用 `math.isclose(handoff["transfer_ms"], 0.002048)` 卡浮点；驻留断言 40 GB 卡 `shortfall_bytes == 32768` 且 `fits` 为假，48 GB 卡为真，容量恰好 `40_000_032_768` 时也为真。
- Q4 单元用五个场景断言 `needs_runtime_check`、`blocked`（原因「真实功能检查未通过」）、`needs_quality_check`、`blocked`、`benchmark_ready`；再断言 `PENDING_TRACE` 只能到 `needs_profile_check`，空 `stage_trace` 是 `blocked` 且 `missing == sorted(required_stages)`。

改参数时最直观的是把 `DEVICE_PROFILES["gpu"]["memory_gb"]` 调到 39，观察 `failed_fields` 变成 `["memory_gb"]`。所有结论以你自己跑出的字典为准，本节没有可引用的已保存数值。
