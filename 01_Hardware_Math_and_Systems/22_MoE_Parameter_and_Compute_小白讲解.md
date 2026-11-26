# 22. MoE 模型参数量计算 — 小白讲解

> 对应 notebook：[`22_MoE_Parameter_and_Compute.ipynb`](./22_MoE_Parameter_and_Compute.ipynb)

## 一句话总结
本节用一组纯 CPU 的公式函数，把混合专家模型（MoE）的四笔账拆开算：`dense_ffn_params` / `moe_expert_params` / `active_expert_params` 算总参数与活跃参数，`route_tokens` + `load_balance_stats` 算 top-k 路由后的专家负载与容量溢出，`estimate_expert_parallel_traffic` 算专家并行的跨设备发送量，`moe_follow_up_evidence` 把账本转成待测清单。以一个 4096 维、8 专家、top-2 的单层为例：总专家参数 1,073,741,824（约 1.07B），单 token 只激活 268,435,456（约 0.27B），但整层远程通信仍有约 469.8 MB。

## 为什么需要它：痛点是什么
看到"8 个专家、top-2"就以为推理要付 8 份算力，或者反过来以为总参数也只有 2 份——这是两种相反的误读。真实情况是：8 个专家的权重一张卡装不下，要按专家切到多张卡；每个 token 虽然只算 2 个专家，却要先打分、再把 token 发到专家所在的卡、算完再收回来。**总参数、活跃计算、显存驻留、通信量是四本不同的账**，任何一本算错都会得出错误结论。

生活类比：餐厅有 8 个档口，每份订单只去 2 个档口取菜（活跃计算小），但厨房要备齐 8 个档口的食材和厨师（总参数大），传菜员还要在档口之间来回跑（通信）。若订单都挤向同一个档口，它就会排队溢出（负载不均），其他档口却闲着。

## 核心问题
- 为什么总参数按专家数翻倍，单 token 的计算量却只按 top-k 算？
- 一个专家的参数量为什么是 `2 × d × d_ff`，对应的 FLOPs 为什么再乘 2？
- top-k 路由后，专家负载与容量溢出怎么统计？为什么用"分配次数"而不是"token 数"？
- 专家并行的通信量为什么与专家总数无关，只与设备数、top-k 和隐藏维有关？
- 为什么参数与计算账本只能说明"容量候选"，不能直接决定采用 MoE？

## 解决思路：一步步看懂

### 第 1 步：Q1——专家参数、活跃参数与 FLOPs

notebook 先给出四个公式函数，全部只做整数乘法：

```python
def dense_ffn_params(d, d_ff):
    return 2 * d * d_ff

def moe_expert_params(d, d_ff, num_experts):
    return num_experts * dense_ffn_params(d, d_ff)

def active_expert_params(d, d_ff, active_experts):
    return active_experts * dense_ffn_params(d, d_ff)
```

`2 * d * d_ff` 的 2 来自"升维 + 降维"两个矩阵：一个 $d \times d_{ff}$ 把隐藏维升到中间维，一个 $d_{ff} \times d$ 再降回来，每个矩阵有 $d \times d_{ff}$ 个权重。所以一个专家的参数量是 $2 d d_{ff}$，不是 $d d_{ff}$。notebook 明确说这是**两矩阵专家**的教学配置；真实的 SwiGLU 专家有 gate、up、down 三个矩阵，应改用 `3 × d × d_ff`，真实模型还要按每层结构逐层累计。

代入 Q1 的配置 $d=4096$、$d_{ff}=16384$、8 个专家、top-2：

- 单个专家：$2 \times 4096 \times 16384 = 134{,}217{,}728$（约 134.2M）。
- 8 个专家：$8 \times 134{,}217{,}728 = 1{,}073{,}741{,}824$（约 1.07B）。
- 单 token 激活 2 个：$2 \times 134{,}217{,}728 = 268{,}435{,}456$（约 268.4M）。

比例关系很直观：总参数是活跃参数的 $8/2 = 4$ 倍。这就是 MoE 的核心卖点——**容量按专家数增长，计算只按 top-k 增长**。

FLOPs 由 `active_expert_ffn_flops` 给出：

```python
def active_expert_ffn_flops(d, d_ff, active_experts, flops_per_param=2):
    if active_experts <= 0 or flops_per_param <= 0:
        raise ValueError('active_experts 和 flops_per_param 必须为正数')
    return active_expert_params(d, d_ff, active_experts) * flops_per_param
```

乘 2 的理由是：每个权重参与一次"乘 + 加"，按一次乘加等于 2 FLOPs 计。于是 $268{,}435{,}456 \times 2 = 536{,}870{,}912$ FLOPs/token。这个数**只含专家 FFN 的矩阵乘**，不含 Attention、Router、激活函数、路由搬运和通信——notebook 的 docstring 把边界写得很清楚，读的时候不要把它当成整层前向的全部计算。

`moe_layer_params` 再把单层拼起来：

```python
attention_params = 4 * d * d
router_params = d * num_experts if include_router else 0
return attention_params + moe_expert_params(...) + router_params
```

- 注意力按 MHA 的四个投影 $4d^2$ 粗算：$4 \times 4096^2 = 67{,}108{,}864$。
- 路由器是 $d \times E$ 的线性层：$4096 \times 8 = 32{,}768$；专家合计 $1{,}073{,}741{,}824$。

三者相加得 $1{,}140{,}883{,}456$。路由器只占约 0.003%，专家占了 94%。`include_router=False` 可以去掉这 32,768，单独观察专家占比。

### 第 2 步：Q2——top-k 路由、容量与溢出

`route_tokens(router_scores, top_k)` 接收形状 `[num_tokens, num_experts]` 的分数矩阵，先用 `torch.topk(..., k=top_k, dim=-1).indices` 取每行最大的 k 个专家，再用 `torch.bincount(selected.reshape(-1), minlength=num_experts)` 统计每个专家被选了多少次。返回 `selected_experts`、`token_counts`、`top_k`。两个校验：必须是二维张量，`1 <= top_k <= num_experts`。

notebook 给的分数矩阵是：

```python
router_scores = torch.tensor([
    [4.0, 3.0, 0.0, -1.0],
    [4.0, 3.0, 0.0, -1.0],
    [4.0, 0.0, 3.0, -1.0],
    [4.0, 3.0, 0.0, -1.0],
])
```

前两行和第 4 行选专家 0、1，第 3 行选专家 0、2。于是计数为 `[4, 3, 1, 0]`，合计 8 次分配；断言 `selected_experts.shape == (4, 2)` 和 `token_counts == [4, 3, 1, 0]`。

第一个关键点：**负载统计的是"token—专家分配次数"，不是互不重复的 token 数**。4 个 token、top-2 产生 8 次分配，所以计数加起来是 8 而不是 4。若把它当成"每个专家分到几个 token"，后面所有比例都会错。

容量由 `expert_capacity` 算：

```python
return math.ceil(num_tokens * top_k / num_experts * capacity_factor)
```

即总分配次数平均分给每个专家，再乘容量因子并向上取整。本例 `math.ceil(4 * 2 / 4 * 1.0) = 2`。向上取整保证容量是整数；`capacity_factor` 大于 1 相当于给每个专家多留缓冲，代价是显存和计算增加。函数校验 `num_tokens >= 0`、`num_experts > 0`、`1 <= top_k <= num_experts`、`capacity_factor > 0`。

溢出由 `overflow_tokens` 算：

```python
return sum(max(0, count - capacity) for count in token_counts)
```

负载 `[4, 3, 1, 0]`、容量 2 时，专家 0 超出 2 次、专家 1 超出 1 次、专家 2 和 3 不超，合计 3 次。`load_balance_stats` 把它包装成字典：`total_assignments=8`、`capacity_per_expert=2`、`max_assignments=4`、`min_assignments=0`、`overflow_assignments=3`、`overflow_assignment_rate=3/8`。断言覆盖总数、容量、溢出数和溢出率。

为什么强调"分配次数"？因为同一个 token 的两次分配可能一次成功、一次被丢。若只看"有多少独立 token 完全没拿到任何专家输出"，本例是 0（每个 token 至少有一个专家没超容量），但按分配算溢出率是 37.5%。notebook 特意提醒这两个指标不同。真实实现可以丢弃、重分配、延后，或用负载均衡损失调整路由，本节只统计选择与溢出。

### 第 3 步：Q3——把专家分配换算成跨设备通信

专家并行（EP）把不同专家放在不同设备上。一个 token 被路由到远程专家时，输入要发过去（dispatch），结果要发回来（combine），所以每份远程分配产生**去程 + 回程**两次载荷。`estimate_expert_parallel_traffic` 的公式是：

```python
assignments = batch * seq_len * top_k
logical_bytes = assignments * hidden_dim * bytes_per_element * dispatch_and_gather_factor
return int(logical_bytes * remote_fraction)
```

逐项解释：`batch * seq_len * top_k` 是整层的分配次数；乘 `hidden_dim * bytes_per_element` 得到一份 token 向量的字节数；`dispatch_and_gather_factor` 默认 `2.0`，就是去程加回程；最后乘 `remote_fraction` 只保留需要跨设备的比例。默认 `bytes_per_element=2`，对应 BF16/FP16。

`remote_fraction` 的默认值基于一个假设：专家等量放在各设备、路由与源设备无关，则任一分配落到"非本机"的概率是 $1 - 1/N$，8 台设备为 $7/8 = 0.875$，单设备为 0。此时代码要求 `num_experts % num_devices == 0`，否则必须显式给 `remote_fraction`。

代入默认配置 $batch=8$、$seq=2048$、$hidden=4096$、8 专家、8 设备、top-2：

- 分配次数：$8 \times 2048 \times 2 = 32{,}768$。
- 全部分配的逻辑载荷：$32{,}768 \times 4096 \times 2 \times 2 = 536{,}870{,}912$ 字节 ≈ 536.9 MB。
- 远程部分：$536{,}870{,}912 \times 7/8 = 469{,}762{,}048$ 字节 ≈ 469.8 MB ≈ 0.470 GB。

按 notebook 的 `bytes_to_mb`（除以 $10^6$）和 `bytes_to_gb`（除以 $10^9$），打印出来就是 `469.8 MB / layer` 和 `0.470 GB / layer`。注意这是**一整层、单向统计的远程发送量**，已经含分发与回传两次；真实链路上的耗时还受路由偏斜和具体通信实现影响。

第二个关键点：**专家总数不进入流量公式**。一份分配只去一个专家，无论它是 8 个里的还是 16 个里的，单次载荷都是 `hidden_dim × bytes`；专家数只影响整除条件和摆放。测试里专家数 8 和 16 的结果都是 768 字节。

`test_traffic_estimation` 可以手算：1 个 batch、4 个 token、32 维、2 字节、top-2、4 台设备时，分配数 8，逻辑载荷 $8 \times 32 \times 2 \times 2 = 1024$ 字节，远程比例 $3/4$，得 768；单设备时远程比例为 0；top-1 时分配数减半得 384；显式给 `remote_fraction=0.5` 时得 512。

### 第 4 步：Q4——账本只是候选，不是结论

`moe_follow_up_evidence(capacity_need, communication_ready, load_balance_ok, simplicity_priority)` 接收四个布尔值，返回待测清单。前三个 `quality`、`throughput`、`tail_latency` 是任何 MoE 评估都躲不掉的；`capacity_need` 为真追加 `capacity_gain`，通信没准备好追加 `all_to_all_time`、`scaling_efficiency`，负载没把握追加 `expert_load_distribution`、`overflow_rate`，在意实现简单追加 `scheduling_complexity`、`fallback_path`。返回值里 `required_measurements` 用 `sorted(set(...))` 去重排序，`evidence_level` 固定为 `'theoretical_ledger'`，`ledger_signal` 在 `capacity_need` 为真时是 `'capacity_candidate'`，否则是 `'no_capacity_pressure'`。四个参数必须是 `bool`，否则抛 `TypeError`。

三个测试用例分别是 `(True, True, True, False)`、`(True, False, True, False)`、`(False, True, True, True)`，断言 `capacity_gain`、`all_to_all_time`、`fallback_path` 分别出现在对应清单里。函数不返回"用不用 MoE"，只返回"还缺哪些证据"——参数账本能告诉你"容量够不够"，但质量、吞吐、通信、尾延迟必须实测。Q4 的信号表也是这个意思：总参数高而活跃参数低，仍需验证质量、吞吐和显存驻留；专家跨设备，需验证 All-to-All 通信量、暴露时间和扩展效率。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| MoE | 用多个专家 FFN 替换单个 FFN，每个 token 只走其中几个 |
| 专家（expert） | 一个独立的前馈子网络，有自己的一整套权重 |
| 路由器 / Gate | 给每个 token 对每个专家打分，决定选谁 |
| top-k 路由 | 每个 token 只选分数最高的 k 个专家 |
| 总参数 / 活跃参数 | 全部专家的参数 / 单个 token 实际用到的参数 |
| 分配次数 | token 数 × top-k，负载统计的基本单位 |
| 容量（capacity） | 每个专家最多接收多少分配，超出就溢出 |
| capacity factor | 容量相对于平均负载的放大倍数 |
| 专家并行（EP） | 把不同专家放到不同设备上 |
| dispatch / combine | 把 token 发给专家 / 把专家输出送回原位 |
| All-to-All | 每个 rank 向所有 rank 发送不同数据的集合通信 |

## 小白常见疑问

**Q1：8 个专家、top-2，为什么总参数是单个专家的 8 倍，计算却只有 2 倍？**
因为参数要**全部保存**（8 个专家各一整套权重），计算只走**被选中的** 2 个专家。所以总参数是单个专家的 8 倍，活跃参数是 2 倍，两者之比是 4。notebook 的 1.07B 对 268.4M 正是这个 4 倍关系。

**Q2：FLOPs 为什么等于参数量乘 2？**
因为每个权重参与一次乘法和一次加法，按一次乘加 = 2 FLOPs 计。`active_expert_ffn_flops` 的 `flops_per_param=2` 就是它，而且函数只算专家 FFN，不含 Attention 和 Router。

**Q3：为什么通信量公式里没有 `num_experts`？专家越多不是要发得越多吗？**
一次分配只把一份 token 向量发给**一个**专家，专家总数不改变单次载荷。专家变多只会改变"这些专家分布在哪、远程比例是多少"，所以公式用 `remote_fraction` 而不是 `num_experts`。测试里 8 专家和 16 专家在同样 4 台设备上都得 768 字节。

**Q4：溢出 3 次，是不是有 3 个 token 没被处理？**
不是。溢出统计的是"分配次数"，同一个 token 的两次分配可能只丢一次。本例 4 个 token 每个至少有一个专家没超容量，所以完全失去输出的 token 是 0 个，但溢出率仍是 $3/8$。两个指标回答不同问题，不能混用。

**Q5：容量因子调大不就不会溢出了？**
会减少溢出，但每个专家要预留更多槽位，显存和计算都增加，真实负载还可能依旧倾斜。notebook 只把 `capacity_factor` 当输入，不评估丢弃、重分配等策略。

## 动手验证：怎么知道自己做对了

这个 notebook 没有保存执行输出，也没有 TODO 空位，以下数字以代码断言和公式手算为准。

- `test_moe_parameter_formulas`：$d=4096$、$d_{ff}=16384$、8 专家、top-2 时，`dense == 134_217_728`、`total_expert == 1_073_741_824`、`active_params == 268_435_456`、`active_flops == 536_870_912`、`layer_params == 1_140_883_456`，全过才打印 `✅ MoE 参数量公式测试通过`。
- Q2 单元：`selected_experts.shape == (4, 2)`、`token_counts == [4, 3, 1, 0]`；`capacity == 2`、`total_assignments == 8`、`overflow_assignments == 3`、`overflow_assignment_rate == 3 / 8`。
- `test_traffic_estimation`：分别断言 768、768、0、384、512 字节，打印 `专家并行发送载荷公式核对通过`。第二个 768 是把专家数从 8 换成 16，用来证明专家数不影响流量。
- Q4 单元：断言 `capacity_gain`、`all_to_all_time`、`fallback_path` 分别在对应清单里，打印 `✅ MoE 待测证据清单检查通过`。

改参数时一次只动一个：`top_k` 从 2 改 1，活跃参数和流量都减半；`num_devices` 从 8 改 1，`remote_fraction` 变 0、流量归零；`capacity_factor` 改成 2，容量从 2 变 4、溢出从 3 变 0。这三组分别对应"稀疏度""本地性""缓冲"三个旋钮。
