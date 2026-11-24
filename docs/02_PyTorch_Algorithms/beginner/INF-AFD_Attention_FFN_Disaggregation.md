# INF-AFD. Attention FFN Disaggregation | Attention–FFN 解耦 · 小白入门

> 配套 notebook：[INF-AFD. Attention–FFN 解耦](../INF-AFD_Attention_FFN_Disaggregation.md) ｜ 难度：Hard ｜ 环境：CPU-first，GPU/backend optional

## 一句话说清

模型每一层都有注意力和前馈两部分，它们吃资源的方式不一样。这本 notebook 把这两部分拆到不同的进程组上，教你怎么算拆开后多出来的传输量、等待时间和进程分配。

## 为什么需要它

在混合专家（MoE）模型的一层里，注意力和专家按固定顺序排在一起，但压力来源不一样：

- 注意力要保存每个请求的状态和 KV Cache。并发越多、上下文越长，它越吃力。
- 专家那边要看路由过来的词元有多少、专家负载偏不偏、算力够不够。

把两者绑在同一组进程上，一个吃紧就得整体一起扩容，另一种资源可能白放着。拆开的代价是：每一层都要在两边来回传数据，还要保持请求身份和同步关系。传输和专家计算的时间如果直接暴露在等待路径上，请求反而更慢。

这本 notebook 在 CPU 上算清三件事：每个拆分的层来回传多少字节，扣掉能重叠的部分后还要等多久，两类角色各分多少进程。真实收益要到多卡环境里验证。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 混合专家（MoE）层 | 把前馈网络换成多个“专家”的层。路由器给每个词元挑少数几个专家来算，个数是 `top_k` |
| Attention–FFN 解耦（AFD） | 把每一层的注意力部分和前馈部分放到不同的 worker 组上运行，让两者可以各自扩容。A 指注意力（Attention）这一侧，F 指前馈和专家（FFN）这一侧 |
| rank / 全局进程数（world size） | rank 是参与计算的一个进程，可以理解为一个干活的工人；`world_size` 是 rank 的总数，一组 rank 也叫 worker 组 |
| 隐藏状态（hidden state） | 注意力算完后，要送去前馈的那串数。每个词元有 `hidden_size` 个数 |
| 载荷（payload） | 每个被拆分的层在 A、F 之间来回传的字节数 |
| 拓扑（topology） | 两类角色各占多少个 rank，同步还是异步。题目里是一条 `AFDTopology` 记录 |
| 暴露时间（exposed time） | 传输和专家计算的时间里，没被重叠盖住、仍然要等的部分 |

## 核心思路

输入是一组数字：词元数、隐藏维度、拆分层数、每个数占几字节，以及注意力、传输、专家计算各要多久。

拆开以后，每个被拆分的层多出两趟传输：A 算完注意力，把隐藏状态送到 F；F 算完专家，把 FFN 输出送回 A，A 再继续往下算。路由器（给词元挑专家的那部分）如果在 A 侧，送过去时还要带上每个词元选中的 `top_k` 个专家编号和权重。三本账这样记：

```text
载荷   单向激活 = 词元数 × 隐藏维度 × 每个数的字节数 × 拆分层数
       激活总量 = 2 × 单向激活（去一趟，回一趟）
       路由元数据 = 词元数 × top_k × (4 + 每个数的字节数) × 拆分层数，只在路由器位于 A 侧时才有
时间   串行时间 = 注意力 + 送到 F + 专家计算 + 送回 A
       可重叠时间 = 送到 F + 专家计算 + 送回 A
       有效重叠 = min(重叠预算, 可重叠时间)
       暴露时间 = 串行时间 - 有效重叠
拓扑   两类角色都至少 1 个 rank，加起来等于 world_size；mode 只能是 sync 或 async
```

下面用 notebook 测试里的数字算一遍：

```python
tokens, hidden, layers, dtype_bytes = 8, 16, 2, 2     # 2 表示每个数占 2 字节
one_way = tokens * hidden * dtype_bytes * layers      # 单向：Attention 之后送到 FFN 的 hidden state
routing = tokens * 2 * (4 + dtype_bytes) * layers     # Router 在 Attention 侧：top-2 的专家编号（int32）和权重
print(one_way, 2 * one_way, routing, 2 * one_way + routing)

def ledger(a, to_ffn, ffn, to_a, budget):
    serial = a + to_ffn + ffn + to_a                  # 完全串行的时间
    effective = min(budget, to_ffn + ffn + to_a)      # 有效重叠：不超过预算，也不超过可重叠区间
    return serial, effective, serial - effective

print(ledger(4, 1, 6, 1, 0))
print(ledger(4, 1, 6, 1, 5))
print(ledger(4, 1, 6, 1, 99))
```

```text
512 1024 192 1216
(12, 0, 12)
(12, 5, 7)
(12, 8, 4)
```

前四个数是载荷。单向 512 字节，来回 1024 字节。路由器在 A 侧、每个词元选 2 个专家时，再多 192 字节的路由元数据。后三行是“串行时间、有效重叠、暴露时间”。重叠预算是 0 时全部暴露，共 12 毫秒。预算是 5 时藏起 5，剩 7。预算给到 99，也只能藏住可重叠的 8，剩 4。这 4 毫秒是注意力自身的依赖时间，再多的重叠也消不掉。

Attention–FFN 解耦（AFD）不减少模型计算，它改变的是资源怎么用、等待怎么藏。拓扑这本账由 `validate_afd_topology` 逐条检查，失败的原因全部返回。题目还约定：异步模式只在 `prefill` 阶段、并且没有启用 graph（题目里的字段是 `graph_enabled`）时才算满足条件。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 为什么拆分 Attention 与 FFN，三类角色各自承受什么压力 | 读 |
| Step 2 | 每个拆分层要来回传什么，同步、异步、不拆分三条路径各有什么优缺点 | 读 |
| Step 3 | 一个 AFD 候选要通过哪六道 Gate（门槛）：模型、拓扑、connector、执行模式、精度、性能 | 读 |
| Step 4 · TODO 1 | 计算跨角色载荷 | 在 `estimate_afd_payload_bytes` 里补 2 行：`activation_bytes` 是单向的两倍，`routing_bytes` 只在路由器位于 A 侧时非零 |
| Step 4 · TODO 2 | 检查 Attention / FFN 拓扑能力 | 在 `validate_afd_topology` 里补 4 个布尔值：两侧 rank 都为正、rank 总数等于 `world_size`、`mode` 受支持、异步条件满足 |
| Step 4 · TODO 3 | 计算暴露时间 | 在 `estimate_afd_time_ms` 里补 2 行：`effective_overlap_ms` 和 `exposed_ms` |
| Step 4 · TODO 4 | 按压力分配角色 rank | 在 `allocate_afd_ranks` 里补 2 行：按压力比例算 `attention_ranks` 并限制在 1 到 `world_size - 1` 之间，`ffn_ranks` 取剩余 |
| 测试 | 检查载荷分项与总量、可行拓扑与逐项失败原因、暴露时间的上下界、不同压力比下的 rank 守恒 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 在多 GPU 上比较普通 MoE 和 AFD 的同步、异步两种模式 | 需要 `INF-VLLM-011-CU128` 环境、兼容的 AFD plugin（插件）和两个独立角色组各自的服务地址（endpoint）；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要多张 GPU 和 AFD plugin，可选，默认是 `RUN_MODE = 'dry_run'`。

## 容易踩的坑

- **激活只算了一趟**：A 到 F 送隐藏状态，F 到 A 送回 FFN 输出，激活是单向的两倍。测试里 `(8, 16, 2, 2)` 的激活是 `1024` 字节。
- **重叠超过了可重叠区间**：有效重叠取重叠预算和可重叠时间中较小的那个。预算再大，注意力自身的依赖时间也消不掉，测试里暴露时间最少是 `4`。
- **rank 总数不守恒**：两类角色都至少要 1 个 rank，加起来必须等于 `world_size`。`allocate_afd_ranks` 里注意力的份额要限制在 1 到 `world_size - 1` 之间。
- **看到 MoE 就默认启用 AFD**：它首先是一个部署候选。六道 Gate（门槛）要一起过，其中 connector 指连接 A、F 两侧的通信部件。精度不过就拒绝，不进入性能排序。
- **把账本当成结论**：载荷估算用于比较候选，不替代 connector 的真实序列化字节数。rank 计划也不是性能结论，只是进入 backend（执行后端）实验的候选。

## 读完后自查

- [ ] 给词元数、隐藏维度、拆分层数和每个数的字节数，能算出双向激活的字节数。
- [ ] 串行 12 毫秒、可重叠 8 毫秒，能说出重叠预算是 5 和 99 时的暴露时间。
- [ ] 能说出一个可行的拓扑要满足哪四个条件。
- [ ] 能说出 AFD 不减少什么，它靠什么改变等待时间。

## 下一步

- 打开 notebook：[INF-AFD. Attention–FFN 解耦](../INF-AFD_Attention_FFN_Disaggregation.md)
- 建议先读：[06. MoE 路由器 · 小白入门](./06_MoE_Router.md)、[46. MoE 专家并行 · 小白入门](./46_MoE_Expert_Parallel.md)、[INF-SERVING-OVERLAP. 服务重叠与状态交接 · 小白入门](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md)
- 还可以看：[39. 异构 PD 与服务分层](../39_Hetero_PD_and_Serving_Tiers.md)、[48. 通信热点与缓解策略](../48_Communication_Hotspots_and_Mitigation.md)
