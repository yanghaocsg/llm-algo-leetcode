# INF-AFD. Attention–FFN 解耦 — 小白讲解

> 对应 notebook：[`INF-AFD_Attention_FFN_Disaggregation.ipynb`](./INF-AFD_Attention_FFN_Disaggregation.ipynb)

## 一句话总结
AFD（Attention–FFN Disaggregation）把 Attention 和 FFN/专家计算放到不同执行组，让两侧可以按各自压力独立扩缩容。本节用 `estimate_afd_payload_bytes`、`validate_afd_topology`、`estimate_afd_time_ms`、`allocate_afd_ranks` 四个 TODO，把跨组载荷、拓扑合法性、可隐藏等待和 rank 分配算清楚。核心结论：拆分能解耦资源比例，但每层都新增双向激活传输和依赖；只有独立扩容的收益能覆盖暴露时间，拆分才值得保留。notebook 目前没有登记真实 AFD 对照。

## 为什么需要它：痛点是什么
MoE 推理有两个不同来源的压力：长上下文让 Attention 侧的历史 KV 缓存越来越重；大量词元同时路由到少数专家，会让 FFN 侧排队。如果两类计算始终绑在同一组 GPU 上，扩容时只能一起加卡——想缓解专家热点，却顺带为 Attention 买了用不上的资源。
生活类比：一家餐厅把"点单/传菜"（Attention）和"后厨做菜"（FFN）放在同一个开放厨房。客流大时两边都喊忙，但老板只能整间店一起扩租。AFD 相当于把后厨独立出去：点单台忙就加点单员，后厨忙就加厨师；代价是每道菜都要经过一次传菜窗口（连接器），传菜慢了，独立扩容也白搭。

## 核心问题
- Attention 组、FFN/专家组和连接器各自处理什么、压力来自哪里？
- 一层拆分后，双向激活和可选路由元数据一共多少字节？
- 什么样的 rank 拓扑和异步执行组合才允许进入真实实验？
- 微批次流水能隐藏哪部分等待，哪些依赖仍然存在？
- 按压力比例分配 rank 时，为什么两侧都至少留 1 个？
- 用什么条件判断拆分"能跑"且"值得保留"？

## 解决思路：一步步看懂

**第 1 步：角色与数据流。** `AFDTopology` 记录 `world_size`、`attention_ranks`、`ffn_ranks`，默认 `mode='sync'`、`stage='decode'`、`graph_enabled=False`。设计上，Attention 组维护请求、KV 缓存和生成顺序；FFN/专家组处理路由后的词元和专家参数；连接器传激活、路由元数据和 FFN 输出。路由器放在哪一侧取决于实现，本题在"放在 Attention 侧"时额外计路由字节。

**第 2 步：双向载荷（TODO 1）。** 设词元数 $N$、隐藏宽度 $H$、元素字节 $B$、拆分层数 $L$，一层把 $N\times H$ 的隐藏表示送到 FFN 组，专家算完再把同样形状的输出送回，所以双向激活是
$$Q_{\mathrm{act}}=2\times N\times H\times B\times L$$
参考答案先算单向 `one_way_activation = tokens * hidden_size * dtype_bytes * split_layers`，再乘 2。若 `router_on_attention=True`，还要为每个词元、每个 top-k 专家传专家编号和权重：
```python
routing_bytes = tokens * top_k * (4 + dtype_bytes) * split_layers if router_on_attention else 0
```
其中专家编号按 4 字节（int32）估。测试 `estimate_afd_payload_bytes(8, 16, 2, 2)`：单向 $8\times16\times2\times2=512$ 字节，双向 `activation_bytes == 1024`，无路由时 `routing_bytes == 0`；加上 `top_k=2, router_on_attention=True` 后路由为 $8\times2\times(4+2)\times2=192$，总量 1216，严格大于基础版。参数校验要求 tokens/hidden/split_layers/dtype_bytes 为正、`top_k>=0`。

**第 3 步：拓扑能力门（TODO 2）。** 参考答案算出四个布尔量：
```python
roles_positive = topology.attention_ranks > 0 and topology.ffn_ranks > 0
ranks_match = topology.attention_ranks + topology.ffn_ranks == topology.world_size
mode_supported = topology.mode in {'sync', 'async'}
async_contract_supported = topology.stage == 'prefill' and not topology.graph_enabled
```
再分别把问题追加到 `failures`：角色非正、rank 总数不等于 `world_size`、未知 mode、以及 async 但阶段/图执行条件不满足。`test_topology_gate` 的四条用例就是四条边界：`AFDTopology(8,4,4)` 返回空列表（通过）；`(8,4,3)` 总数 7≠8；`mode='async', stage='decode'` 因 stage 不是 `prefill` 失败；`mode='async', stage='prefill', graph_enabled=True` 因图执行开启失败。注意最后一条约束只对 `async` 生效，`sync` 不受 stage/graph 限制，这是本题给定的简化条件，真实插件仍要按版本能力表核对。

**第 4 步：暴露时间（TODO 3）。** 串行时间 `serial_ms = attention + transfer_to_ffn + ffn + transfer_to_attention`。能被其他微批次覆盖的只有非 Attention 段：
```python
overlapable_ms = transfer_to_ffn_ms + ffn_ms + transfer_to_attention_ms
effective_overlap_ms = min(overlap_budget_ms, overlapable_ms)
exposed_ms = serial_ms - effective_overlap_ms
```
因为覆盖量不超过非 Attention 段，`exposed_ms` 不会小于 `attention_ms`——本请求自己的 Attention 依赖仍在。`test_exposed_time_bounds` 用 `(4, 1, 6, 1)`：串行 12 ms；预算 0 得暴露 12；预算 5 得 7；预算 99 被可重叠段 $1+6+1=8$ 截住，暴露 4。这三个数正好验证"零覆盖、部分覆盖、过大预算"三种情况。

**第 5 步：rank 分配（TODO 4）。** 先按压力比例算：
```python
raw_attention = round(world_size * attention_pressure / (attention_pressure + ffn_pressure))
attention_ranks = min(world_size - 1, max(1, raw_attention))
ffn_ranks = world_size - attention_ranks
```
`world_size=8` 且压力 1:1 时 raw=4，得到 4/4；压力 3:1 时 raw=6，得到 6/2，Attention 侧增加且两侧都 ≥1。裁剪 `[1, world_size-1]` 是为了避免比例极端时把另一侧压到 0 个进程；`ffn_ranks` 用总数相减保证守恒。测试只验证 `attention_ranks + ffn_ranks == 8`、重压时 Attention 更多、两侧至少 1。真实配置还要看角色排队和设备利用率，按比例分只是初始方案。

**第 6 步：实验设计与结果怎么读。** Step 5 要求同一设备类型与总数下比较普通 MoE 服务和 AFD 服务，模型权重、精度、请求集一致，先核对输出误差再比性能。指标是 TTFT、TPOT、吞吐、跨组传输和排队等待。可现实是 notebook 的 `AFD_MODEL_ID=''` 会直接拦住真实运行：5.2 开头就 `if not AFD_MODEL_ID: raise ValueError`；`AFD_BASELINE_URL`/`AFD_CANDIDATE_URL` 只是默认端口，`AFD_NUM_PROMPTS=16`、`AFD_MAX_TOKENS=64`、`AFD_CONCURRENCY=4`，还需要固定兼容的 AFD 插件。5.4 的表格五项状态全是"待验证"，`decision` 一栏写"未形成实测决策"。所以本节只能确认 CPU 侧的载荷、拓扑和时间公式，不能声称拆分已经带来加速。

**第 7 步：微批次流水到底隐藏什么。** 同一个微批次仍然依次经历 Attention→发送→FFN→返回，流水只是让 Attention 组在等 FFN 结果时去处理**另一个**微批次。可隐藏的是本组空闲，不是本请求的依赖链；所以 `exposed_ms` 要按单个请求或单个窗口算，别把"整体窗口缩短"误读成"这个请求的 Attention 也变快了"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| AFD | 把 Attention 与 FFN/专家计算分到不同执行组的部署方式 |
| rank | 通信组里的进程编号；若干 rank 组成一个角色组 |
| Attention 组 | 管注意力、KV 缓存、请求状态和生成顺序 |
| FFN/专家组 | 处理路由后的词元与专家权重 |
| 连接器 | 在两组之间传激活、路由元数据和结果 |
| 双向激活 | 一层里 A→F 的隐藏表示 + F→A 的 FFN 输出 |
| 路由元数据 | 每个词元选中的专家编号与权重 |
| 微批次流水 | 用另一个微批次填满本组等待时间 |
| 串行时间 | Attention + 去程 + FFN + 回程的总和 |
| 可重叠段 | 非 Attention 的三段，最多被覆盖这么多 |
| 暴露时间 | 串行时间减有效覆盖，仍包含 Attention 依赖 |
| 同总设备预算 | 两种部署用同样多的设备做公平比较 |

## 小白常见疑问
**Q1：为什么激活要算两遍？**
因为一层拆分后，隐藏表示要从 Attention 组送到 FFN 组（A→F），专家算完的输出还要送回 Attention 组做残差连接（F→A）。测试里 `activation_bytes == 2 * 8 * 16 * 2 * 2` 就是显式乘 2。

**Q2：路由字节为什么是 `top_k * (4 + dtype_bytes)`？**
每个词元会选 `top_k` 个专家，每个选择要传一个专家编号（按 4 字节 int32 估）和一个路由权重（按模型 dtype 字节数）。乘上词元数和拆分层数就是总量。这是本题的估算假设，真实连接器的序列化格式可能不同。

**Q3：`overlap_budget_ms=99` 时，`exposed_ms` 为什么是 4 而不是 0？**
有效覆盖被 `overlapable_ms = 1+6+1 = 8` 截住，只能覆盖 8，串行 12 减 8 得 4，正好等于 Attention 自己的 4 ms。本请求的 Attention 依赖不能被"别的微批次"消掉。

**Q4：为什么 async 只在 `stage='prefill'` 且 `graph_enabled=False` 时通过？**
这是 notebook 给定的简化契约，用来练习组合检查，不是通用硬件规则。`test_topology_gate` 里 async+decode 和 async+prefill+graph 都失败，只有 `sync` 不受这条限制。真实后端要按固定插件版本的支持表验证。

**Q5：按 3:1 压力分到 6 个 Attention rank，能保证更快吗？**
不能。它只是按比例给出的初始拓扑；如果专家热点没被分散，连接器等待和专家排队仍可能占主导。5.3 要求先固定插件、模型和总设备预算，再比较输出、传输、TTFT/TPOT 和吞吐，不能只看 rank 数量。

## 动手验证：怎么知道自己做对了
`test_afd_mechanisms()` 依次跑载荷、拓扑、暴露时间、rank 分配四个测试，全过打印 `✅ AFD 机制测试通过：载荷、拓扑、暴露时间与角色分配均符合预期。`；它把 `NameError`/`UnboundLocalError` 转成 `NotImplementedError('请根据变量级提示完成 TODO。')`。
- `test_payload_accounting`：`activation_bytes == 2*8*16*2*2 == 1024`，无路由时 `routing_bytes == 0`，开启 `top_k=2` 后 `total_bytes` 严格大于基础版。
- `test_topology_gate`：`AFDTopology(8,4,4)` 的失败列表为空；`(8,4,3)` 非空；`async+decode` 非空；`async+prefill+graph_enabled=True` 非空。
- `test_exposed_time_bounds`：`(4,1,6,1,0)` 暴露 12；预算 5 暴露 7；预算 99 暴露 4。
- `test_rank_allocation`：8 rank、1:1 得 4/4 且和为 8；3:1 时 Attention rank 更多；两侧最小值 ≥1。
- Step 5 没有真实运行结果：`AFD_MODEL_ID` 为空时真实单元会抛 `ValueError`，5.3 读取两份报告只会得到 `not_run` 或缺字段的 `invalid`。以四个测试的断言为准，不要把 5.4 的"待验证"当成实测结论。
