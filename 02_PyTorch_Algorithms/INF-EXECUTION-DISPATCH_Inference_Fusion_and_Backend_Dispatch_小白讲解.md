# INF-EXECUTION-DISPATCH. 推理融合与后端分派 — 小白讲解

> 对应 notebook：[`INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb`](./INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb)

## 一句话总结
本节从一串有序算子中找出连续、互不重叠的融合候选，再按后端能力字典逐项检查模式、数据类型、设备、序列长度和动态形状，选出第一个兼容后端，全部失败就回退 eager 并保留拒绝原因。可选的 GPU 实验用 `torch.compile` 对同一段归一化 + SwiGLU + 残差计算在 `1 / 32 / 256 / 1024` 个 token 下对照 eager，用输出一致性、图数量、图中断、median/P95 和首次调用成本判断哪些形状值得采用。

## 为什么需要它：痛点是什么
初学者容易把「算子融合能省一次读写」直接推成「融合一定更快」，也容易把「用了 `torch.compile`」当成「一定加速」。真实情况是：融合和后端都有适用条件，输入宽度、精度、设备或形状一变，原来的编译结果可能失效，甚至要重新编译；某条路径在 token=1 时快 30%，到 token=1024 时可能反而更慢。

生活类比：开车去机场。导航（分派器）先看目的地、车型、限行和路况（能力门槛），有高速就走高速（编译/融合路径），但高速封路、车型不符或时间不对，就得走普通道路（eager）。不能因为「高速通常更快」就无视条件硬闯；也不能只看一次行程，就断定以后每次都该走高速。

## 核心问题
- 计算图、eager、专用内核、`torch.compile` 和 CUDA Graph 各自处理什么，别混在哪？
- 哪些相邻算子组合值得作为融合候选？各自的限制是什么？
- 后端能力门槛有哪五类，每类失败后应该做什么？
- 为什么模式匹配要「长模式优先」，命中后游标怎么移动？
- 编译路径的收益用什么指标判断，为什么 median 和 P95 要一起看？
- 首次编译成本怎样摊薄，什么情况下「快」也不值得采用？

## 解决思路：一步步看懂

**Step 1：从图到实际执行。** 计算图用节点表示算子、边表示张量依赖。图编译先收集运算，再识别可以一起处理的模式，后端把支持的模式编译成执行代码。逐算子即时执行叫 eager；显式调用专用内核则直接进入对应实现。notebook 特别区分：**模型图**捕获的是算子与张量关系，**CUDA Graph** 捕获的是一组 GPU 工作的提交序列，供后续重放，两者对象不同。后端选择同时依赖「计算含义」和「输入条件」：能执行就记录路径名，不能执行就换后端或 eager，并记录原因。

**Step 2：哪些模式值得融合。** notebook 给了四类候选：QKV 投影 + RoPE、残差相加 + RMSNorm、线性映射 + SiLU + 逐元素乘法（SwiGLU 中间激活）、Logits 处理 + 采样。以残差相加 + RMSNorm 为例：分开执行要把相加结果写回显存，再读进来做归一化；融合后可以在同一内核里继续处理，省掉中间张量落地和一次内核启动。代价是寄存器压力、编译约束和调试成本都会增加，所以「识别出模式」只是开始，还要比较中间张量读写、实际内核与完整调用时间。

**Step 3：能力门槛决定命中与回退。** 后端声明一组支持条件，分派时把当前输入逐项比较：模式、数据类型/设备、形状、动态形状、执行证据，任一项不满足都可能触发重新编译、换后端或回退。分桶（bucketing）是把相近输入长度归入少量受支持的形状档位，以复用编译结果；补齐和掩码也有额外成本。图捕获结果说明编译路径是否完整，性能分析记录则用来确认「实际跑的算子」和「以为跑的路径」一致。

**Step 4：三个 TODO 与参考写法。**

TODO 1 在 `identify_fusion_candidates` 里只补一行排序：

```python
ordered_patterns = sorted(FUSION_PATTERNS.items(), key=lambda item: len(item[0]), reverse=True)
```

`FUSION_PATTERNS` 把算子元组映射到模式名，例如 `('residual_add', 'rms_norm') -> 'residual_rmsnorm'`、`('linear_gate', 'silu', 'multiply') -> 'swiglu'`。为什么按长度降序？如果在同一个起点既有长模式又有短模式，短模式先匹配会「吃掉」节点，长模式就再也匹配不上；先试长的，才能让更具体的融合优先。后面的 `while` 用 `index = matched['end']` 跳过已消费节点，保证候选不重叠；没匹配就 `index += 1`。

测试用的算子序列是：

```python
['embedding', 'linear_qkv', 'rope', 'attention',
 'residual_add', 'rms_norm', 'linear_gate', 'silu', 'multiply']
```

从索引 1 匹配 `linear_qkv + rope`（`start=1, end=3`），索引 3 的 `attention` 无匹配后前移，索引 4 匹配 `residual_add + rms_norm`，索引 6 匹配三元的 `swiglu`。断言模式列表恰好是 `['qkv_rope', 'residual_rmsnorm', 'swiglu']`，最后一项的 `ops` 是 `['linear_gate', 'silu', 'multiply']`。

TODO 2 在 `evaluate_backend_capability` 里写五个布尔量：

```python
pattern_supported = candidate.get('pattern') in backend.get('patterns', [])
dtype_supported = workload.get('dtype') in backend.get('dtypes', [])
device_supported = workload.get('device') in backend.get('devices', [])
sequence_supported = int(workload.get('seq_len', 0)) <= int(backend.get('max_seq_len', 0))
dynamic_shape_supported = (
    not bool(workload.get('dynamic_shape', False))
    or bool(backend.get('supports_dynamic_shape', False))
)
```

注意最后一行：**静态形状无条件通过**，只有 `dynamic_shape=True` 时才要求后端显式支持。五个变量分别对应 `checks` 里的五个键：`unsupported_pattern`、`unsupported_dtype`、`unsupported_device`、`sequence_too_long`、`dynamic_shape_not_supported`。这些键名描述的是「失败原因」，变量本身却是「是否支持」，读代码时别被命名绕晕：`reasons = [reason for reason, passed in checks.items() if not passed]`。还有一个坑：`max_seq_len` 缺省为 0，任何正的 `seq_len` 都会判为超长，所以后端字典必须写全。

TODO 3 在 `dispatch_candidate` 里选第一个通过的报告：

```python
selected = next((report for report in capability_reports if report['supported']), None)
```

`backends` 列表的顺序就是优先级；`next(..., None)` 表示全都不支持时返回 `None`。选中有 `execution_path = selected['backend']`、`decision = 'optimized'`；否则回退 `execution_path = 'eager'`、`decision = 'fallback'`。无论选没选中，都把未通过的后端及原因收进 `rejected_backends`，方便事后审计「为什么没走优化路径」。

**Step 5：GPU 实验怎么设计和怎么读。** 默认 `RUN_MODE = 'dry_run'`，只打印配置；预检通过后改 `'real_gpu'`。固定条件是 FP16、隐藏维 1024、堆叠 `GPU_BLOCK_DEPTH = 4` 个 `ResidualSwiGLUBlock`，token 数取 `[1, 32, 256, 1024]`，预热 10 次、正式 30 次，`atol = rtol = 2e-2`，门槛是 `GPU_MIN_SPEEDUP = 1.05` 且 `GPU_MAX_P95_RATIO = 1.05`（约等于允许中位数至少快 5%、P95 最多慢 5%）。

`ResidualSwiGLUBlock` 的前向是：

```python
normalized = self.norm(x)
return x + torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized)
```

注意它和 Step 2 举的「相加后归一化」是**两个不同的融合例子**：这里是先 RMSNorm，再做门控投影，最后加回输入。`compiled = torch.compile(model, fullgraph=True, dynamic=False)` 要求整段捕获成一张图，`dynamic=False` 则按具体形状特化，因此每个 token 数都可能是独立的编译与首次调用成本。

计时的 `measure` 用 `torch.cuda.Event` 包住重复调用，先预热再 `reset_peak_memory_stats`，返回 `median_ms`、`p95_ms`（最近秩分位数）和 `peak_allocated_bytes`。`profile_operator_keys` 用 `torch.profiler` 取设备时间最长的前 10 个算子，并排除名字里含 `dynamo`、`compile`、`cache lookup` 的编译控制事件。`torch._dynamo.explain(model)(x)` 给出 `graph_count` 和 `graph_break_count`。

判定逻辑是：所有形状 `output_close`、至少一张图且无图中断、至少一个形状通过收益门槛、且没有运行失败，才给 `accept_for_selected_shapes`，否则 `tune_or_fallback`。

登记的 `rtx5070ti_dispatch_v3` 可以这样读：四档输出一致，均为单图且无图中断，编译路径出现 Triton 融合算子；token `1` 和 `32` 的加速比分别是 `1.12×`、`1.33×`，通过门槛；`256` 只有 `1.03×`；`1024` 的 P95 回退约 `30%`。所以只把 `1/32` 作为稳态候选，`256/1024` 保留 eager。首次调用约 `454–658 ms`，按每次节省的时间粗估，`1/32` 分别约需 `3.1 万`、`1.8 万` 次调用才能摊薄；这个数字只是量级估算，真实采用还要看复用次数。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 计算图 | 节点是算子、边是张量依赖的结构 |
| eager | 按顺序逐个算子立即执行 |
| 后端（backend） | 接收一段计算并真正执行它的实现 |
| 分派（dispatch） | 按当前输入为计算选择执行路径 |
| 融合模式 | 可以合并进同一内核的连续算子序列 |
| 能力门槛 | 后端对模式、精度、设备、形状等的前置条件 |
| 回退（fallback） | 优化路径不适用时退回 eager 等路径 |
| 分桶（bucketing） | 把相近形状归入少数档位以复用编译结果 |
| CUDA Graph | 捕获一组 GPU 工作的提交序列供重放，不是模型图 |
| `fullgraph=True` | 要求编译整张图，不允许图中断 |
| median / P95 | 中位延迟 / 95% 分位延迟，后者反映尾部抖动 |
| 首次调用成本 | 包含编译与等待的一次性开销，需单独摊薄 |

## 小白常见疑问

**Q1：`FUSION_PATTERNS` 是普通字典，为什么不能直接按插入顺序匹配？**
插入顺序是人为的，和模式长短无关。如果短模式排在前面，它在某个起点先命中，就会消费掉节点，长模式即使更具体也匹配不到。`sorted(..., key=len, reverse=True)` 把「更具体的融合优先」变成显式规则，测试里的长短模式竞争就是检查这一点。

**Q2：`checks` 里的 `unsupported_dtype` 为 True，到底表示支持还是不支持？**
表示**支持**。五个变量都是「是否通过」的布尔量，但键名写的是「失败原因」。`supported = not reasons`，所以当 `dtype_supported = False` 时，`reasons` 里才会出现 `unsupported_dtype`。读返回值时看 `supported` 和 `reasons`，不要只看键名。

**Q3：`dynamic_shape=False` 时，为什么后端不声明 `supports_dynamic_shape` 也能通过？**
因为静态形状不需要这项能力。参考写法是 `not dynamic_shape or supports_dynamic_shape`：前面为 True 就短路通过。只有工作负载确实是动态形状时，才要求后端显式支持。测试里把 `dynamic_shape` 改成 `True`，才断言 `dynamic_shape_not_supported` 出现在原因里。

**Q4：`256` 的加速比 `1.03×` 和 `1024` 的 P95 回退，为什么不能「平均一下」说编译整体有效？**
因为采用决策是**按形状**做的。`1.03×` 低于 `1.05` 门槛；`1024` 的 P95 回退约 `30%`，远超 `1.05` 的允许比例。把四档平均会掩盖「某些形状变慢」的事实。notebook 只把 `1/32` 列为候选，其余保留 eager。

**Q5：`fullgraph=True, dynamic=False` 对首次调用成本意味着什么？**
`fullgraph=True` 要求不出现图中断；`dynamic=False` 按具体形状特化，所以每个 token 数可能各自触发一次编译。首次调用 `454–658 ms` 就是这类一次性成本，必须和稳态的 median/P95 分开看，再按实际复用次数决定值不值得。

## 动手验证：怎么知道自己做对了
测试单元依次调用 `test_candidate_identification`、`test_backend_capability_gate`、`test_dispatch_and_fallback`，再由 `test_inference_execution_dispatch` 汇总；全部通过打印 `✅ 推理融合与后端分派测试通过：候选、能力门槛和回退均通过。`

- `test_candidate_identification`：用上面 9 个算子的序列，断言模式列表严格等于 `['qkv_rope', 'residual_rmsnorm', 'swiglu']`，第一项 `start == 1 and end == 3`，最后一项 `ops == ['linear_gate', 'silu', 'multiply']`。这同时验证了连续匹配、最长优先和不重复消费。
- `test_backend_capability_gate`：候选 `{'pattern': 'swiglu'}`、工作负载 FP16/CUDA/2048/静态、后端支持 `swiglu`、`float16`、`cuda`、`max_seq_len=4096` 时必须 `supported is True`；把 dtype 改成 `float32`，必须 `supported is False` 且原因含 `unsupported_dtype`；把 `dynamic_shape` 改成 `True`，原因里必须含 `dynamic_shape_not_supported`。
- `test_dispatch_and_fallback`：`backend-a` 只支持 `swiglu`，`backend-b` 支持 `qkv_rope`，候选 `qkv_rope` 必须选中 `backend-b`，且 `rejected_backends` 含 `backend-a`；只给 `backend-a` 时必须 `execution_path == 'eager'` 且 `decision == 'fallback'`。
- 汇总函数会把 `NotImplementedError` 原样抛出，把其它 `NameError` / `AttributeError` / `TypeError` / `ValueError` / `AssertionError` 包装成 `NotImplementedError('请先完成 TODO 代码或检查字段名！')`——看到这句话先检查 TODO 和字典键名。

dump 里没有保存任何 CPU 测试输出，也没有执行 GPU 单元；5.4 的 `1.12× / 1.33× / 1.03×`、P95 回退 `30%` 和首次调用 `454–658 ms` 都来自登记的 `rtx5070ti_dispatch_v3` 结果。自己复跑时，先看 `failure` 和 `output_close`，再看 `graph_count` / `graph_break_count`，最后才按形状比较 `speedup`、`p95_ratio`、`first_call_ms` 和峰值显存。
