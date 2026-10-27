# Task1 打卡 · 注意力计算与执行分派

> 202610 推理优化 | Issue #188 | 截止 10/15 周四 12:00

---

## 打卡信息

- **微信群昵称**：（填写）
- **环境**：本地 NVIDIA / 云 GPU / 无 GPU / 国产卡 / Mac / 其他
- **系统**：Linux / WSL2 / Windows / 其他
- **Python**：（填写）
- **AI 占比 vs 人工占比**：（如 50:50 / 70:30）
- **任务项**：4.1 最小打卡（+ 4.2 / + 4.3 可选）
- **当前卡点**：无

---

## 4.1 最小打卡

### 4.1.1 MHA 与 GQA 的 KV Cache 区别

**核心区别：KV 头数不同。**

| 架构 | Query 头数 | KV 头数 | 每个 Query 头是否有独立 KV |
|------|-----------|---------|--------------------------|
| **MHA**（Multi-Head Attention） | 32 | 32 | 是，一一对应 |
| **GQA**（Grouped-Query Attention） | 32 | 8 | 否，每 4 个 Query 头共享 1 组 KV |

GQA 介于 MHA（32 KV 头）和 MLA/多查询注意力 MQA（1 KV 头）之间：多组 Query 头共享同一组 K/V，减少 KV 头数但保留一定的表达多样性。

### 显存计算

**已知条件**：batch=1，seq_len=4096，query_heads=32，head_dim=128，FP16（2 bytes/元素）

单层 KV Cache 显存公式：

```
KV显存 = batch × seq_len × kv_heads × head_dim × 2(bytes) × 2(K和V)
```

**MHA（32 个 KV 头）**：

```
K = 1 × 4096 × 32 × 128 × 2 = 33,554,432 B = 32 MiB
V = 1 × 4096 × 32 × 128 × 2 = 33,554,432 B = 32 MiB
K + V = 64 MiB / 层
```

**GQA（8 个 KV 头）**：

```
K = 1 × 4096 × 8 × 128 × 2 = 8,388,608 B = 8 MiB
V = 1 × 4096 × 8 × 128 × 2 = 8,388,608 B = 8 MiB
K + V = 16 MiB / 层
```

**对比**：

| 架构 | K (MiB) | V (MiB) | K+V (MiB) | 相对 MHA |
|------|---------|---------|-----------|----------|
| MHA (32 KV heads) | 32 | 32 | 64 | 100% |
| GQA (8 KV heads) | 8 | 8 | 16 | 25% |

GQA 单层节省 **48 MiB**，降至 MHA 的 **1/4**。若模型有 32 层，则 MHA 需 2 GiB，GQA 仅需 512 MiB。

### GQA 为什么降低解码阶段访存开销

解码阶段每生成一个 token，注意力需要读取**全部历史 KV Cache**：

1. **KV 数据量减少**：GQA 的 KV 头数是 MHA 的 1/4，从 HBM 读取的 KV 字节数直接减少 75%。
2. **解码是访存-bound**：解码时每次只处理 1 个新 token，计算量小（O(seq_len × head_dim)），瓶颈在从 HBM 搬运 KV。KV 越小，搬运越快，TPOT 越低。
3. **显存容量释放**：KV Cache 占比降低后，相同显存可服务更多并发请求（更大 batch），提升吞吐。
4. **代价**：多组 Query 共享 KV 可能轻微影响表达能力，但实践中 GQA（如 8 KV 头）在质量损失极小的情况下显著降低解码成本。

---

### 4.1.2 FlashAttention 为什么能降低显存占用

#### 普通 Attention 的问题

标准 Attention 计算：`Attention(Q, K, V) = softmax(QK^T / √d) · V`

普通实现需要**物化完整的 N×N 注意力矩阵** S = QK^T 和 P = softmax(S)：

```
Q (N×d) → S = QK^T (N×N) → P = softmax(S) (N×N) → O = P·V (N×d)
              ↑ 中间结果 N×N，需写回 HBM
                              ↑ 再从 HBM 读回
```

- **显存**：N×N 的 S 和 P 各需一份，序列长时显存爆炸（N=4096 时单头 4096×4096×2B = 32 MiB，多头叠加）
- **访存**：S 写回 HBM → P 读 HBM → O 写 HBM，多次 HBM 往返

#### FlashAttention 的三个核心机制

| 机制 | 解决的问题 | 做法 |
|------|-----------|------|
| **分块计算（Tiling）** | 避免物化完整 N×N 矩阵 | 将 Q/K/V 按块切分，逐块计算局部注意力，块内结果在 SRAM 中累积，不写回 HBM |
| **在线 Softmax（Online Softmax）** | 分块后如何正确计算全局 softmax | 用 `m_i = max(m_{i-1}, max(row_i))` 和 `l_i = l_{i-1}·e^{m_{i-1}-m_i} + sum(exp(row_i - m_i))` 递推，无需全局最大值即可归一化 |
| **算子融合（Kernel Fusion）** | 减少 HBM 读写次数 | 将 QK^T、softmax、P·V 融合在一个 CUDA kernel 内完成，中间结果全部留在 SRAM/寄存器，不落地 HBM |

#### 显存对比

| 实现 | 中间结果存储 | 显存复杂度 |
|------|-------------|-----------|
| 普通 Attention | 完整 N×N 的 S 和 P 矩阵写回 HBM | O(N²) |
| FlashAttention | 仅保留块内局部结果在 SRAM，输出 O(N×d) | O(N)（输出）+ O(block²)（SRAM 内） |

#### FlashAttention 是否改变数学结果和计算复杂度

- **数学结果**：**不改变**。FlashAttention 是精确算法（exact），通过在线 softmax 递推保证数值上与标准 Attention 等价（存在浮点舍入差异，但非近似）。
- **计算复杂度（FLOPs）**：**不改变**。仍为 O(N²d)，分块只是改变了计算顺序和数据搬运方式，总乘加运算量相同。
- **实际加速来源**：减少 HBM 访存（IO-aware），而非减少计算量。在访存-bound 的场景（长序列、小 head_dim）下收益显著。

---

## 4.2 学有余力增项 1

### 4.2.1 如何设计公平的 Attention 性能对比实验

#### 需要固定的条件

| 类别 | 必须固定 | 原因 |
|------|---------|------|
| **输入** | Q/K/V 形状（batch, heads, seq_len, head_dim）、dtype、设备 | 不同形状/精度直接影响耗时 |
| **模型/算子参数** | causal mask、dropout、scale 因子 | 影响计算路径和 kernel 选择 |
| **硬件环境** | GPU 型号、驱动/CUDA 版本、PyTorch 版本 | 不同硬件后端能力不同 |
| **运行状态** | 预热轮数、同步方式、重复次数 | 排除冷启动和 JIT 编译开销 |
| **测量方式** | 计时起点/终点、显存采集时机 | 保证两种实现测量口径一致 |

#### 预热与同步

```python
import torch

# 1. 预热：先跑若干轮，让 kernel 完成 JIT 编译和缓存
for _ in range(warmup_iters):
    out = attention_func(q, k, v)
torch.cuda.synchronize()  # 等待预热完成

# 2. 正式测量：每轮前后同步
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(measure_iters):
    out = attention_func(q, k, v)
torch.cuda.synchronize()
t1 = time.perf_counter()

avg_ms = (t1 - t0) / measure_iters * 1000
```

- **预热**：FlashAttention/SDPA 首次调用可能触发 kernel 编译或 autotune，必须排除
- **`torch.cuda.synchronize()`**：CUDA 操作异步，不同步会测到提交时间而非执行时间
- **多次重复取均值**：减少系统噪声，建议 ≥20 次

#### 耗时与峰值显存采集

```python
# 耗时：CUDA Event（比 CPU timer 更精确）
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
out = attention_func(q, k, v)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)

# 峰值显存：重置后测量
torch.cuda.reset_peak_memory_stats()
out = attention_func(q, k, v)
torch.cuda.synchronize()
peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MiB
```

#### 对照组设计

- **同一输入张量**：两种实现共用完全相同的 Q/K/V（同一块内存或 clone）
- **输出正确性校验**：`torch.allclose(flash_out, standard_out, atol=1e-2)` 确认数值一致
- **变量唯一**：只改变注意力实现，其他条件全部相同

---

### 4.2.2 Attention 算子加速为什么不一定带来同等幅度的请求加速

请求端到端延迟 = Prefill + Decode × 输出长度，Attention 只是其中一部分。

| 原因 | 说明 |
|------|------|
| **Attention 不是唯一瓶颈** | 前馈网络（FFN）、归一化、残差、KV Cache 搬运都占时间。Attention 加速 2×，若它只占总时间 30%，端到端仅提升 ~18%（Amdahl 定律） |
| **Prefill vs Decode 差异** | FlashAttention 在长序列 Prefill（计算-bound→访存-bound 过渡）收益大；Decode 时 seq_len 增量为 1，注意力计算量小，瓶颈可能在 KV 读取和 FFN，算子加速收益有限 |
| **调度与批次开销** | Continuous batching 下，请求延迟受排队、批次组织、调度策略影响，单算子快不代表请求快 |
| **Kernel launch 开销** | 小序列/小 batch 时，kernel launch 固定开销占比大，融合带来的访存节省被 launch 开销稀释 |
| **内存分配与碎片** | 中间张量分配、CUDA caching allocator 行为可能引入额外开销 |
| **尾部延迟** | P99 受慢请求、调度抖动影响，平均延迟下降不代表尾延迟同比例下降 |

**结论**：算子级加速是必要非充分条件，必须在固定工作负载下测量端到端 TTFT/TPOT/吞吐，才能判断真实收益。

---

## 4.3 学有余力增项 2

### 4.3.1 调用 SDPA 是否意味着实际使用了 FlashAttention？

**不一定。**

`torch.nn.functional.scaled_dot_product_attention`（SDPA）是一个**调度接口**，内部根据输入条件和后端可用性自动选择实现：

```
SDPA 调用
  ├── FlashAttention（最优先，需 CUDA + 支持的 dtype/形状）
  ├── Memory-Efficient Attention（次优先，cutlass 实现）
  ├── Math / PyTorch 原生实现（兜底，纯 PyTorch，无融合）
  └── CUDNN Attention（特定条件）
```

即使代码写了 `F.scaled_dot_product_attention`，如果：
- 运行在 CPU 上 → 只能用 math 实现
- dtype 不支持（如 FP64）→ 回退 math
- 序列长度/head_dim 超出 FlashAttention kernel 支持范围 → 回退
- 没有安装 FlashAttention 或 CUDA 版本不匹配 → 回退

所以"调用了 SDPA" ≠ "用了 FlashAttention"。

### 4.3.2 如何证明程序实际执行了预期的注意力后端

#### 方法 1：显式指定后端并检查

```python
import torch.nn.functional as F

# 强制只允许 FlashAttention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v)
# 如果 FlashAttention 不可用，会直接报错而非静默回退
```

#### 方法 2：查看后端选择日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# PyTorch SDPA 在 debug 模式下会打印选择了哪个 backend
```

或设置环境变量：
```bash
PYTORCH_ENABLE_SDP_DEBUG=1 python your_script.py
```

#### 方法 3：Profiler 采集 kernel 记录

```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    out = F.scaled_dot_product_attention(q, k, v)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

在 profiler 输出中查找：
- **FlashAttention**：kernel 名含 `flash_fwd` / `flash_attn`
- **Memory-Efficient**：kernel 名含 `efficient_attention` / `cutlass`
- **Math 回退**：出现大量独立 kernel（`aten::mm`、`aten::softmax`、`aten::bmm` 分开调用）

#### 方法 4：显存对比

FlashAttention 的峰值显存应显著低于 math 实现（无 N×N 中间矩阵）。如果两种方式显存接近，说明可能没命中 FlashAttention。

---

### 4.3.3 性能没有明显提升时的诊断框架

如果代码能跑但性能没提升，按以下三种情况排查：

| 情况 | 判断证据 | 排查方向 |
|------|---------|---------|
| **① 后端不可用** | SDPA debug 日志显示回退 math；强制 `enable_flash=True` 报错；Profiler 中无 `flash_fwd` kernel | 检查 CUDA 版本、GPU 算力（需 ≥8.0）、dtype（需 FP16/BF16）、PyTorch 版本是否内置 FlashAttention |
| **② 执行路径未命中** | 后端可用但实际走了其他路径；Profiler 显示 `efficient_attention` 而非 `flash_fwd`；或 attention 被其他模块包裹导致形状变化 | 检查 causal mask、attn_mask、dropout 参数；检查 Q/K/V 是否为 contiguous；检查 head_dim 是否为 8 的倍数且 ≤256；检查是否被 `torch.compile` 改写 |
| **③ 已命中但收益有限** | Profiler 确认 `flash_fwd` 执行，但端到端耗时变化小 | 检查 Amdahl 瓶颈（Attention 占总时间比例）；检查序列长度（短序列 launch 开销主导）；检查 batch 大小；检查是否被 FFN/其他层瓶颈掩盖；检查显存是否已不是瓶颈 |

#### 诊断步骤清单

1. **确认后端**：`sdp_kernel(enable_flash=True, enable_math=False)` 强制测试，看是否报错
2. **看日志**：开 `PYTORCH_ENABLE_SDP_DEBUG=1`，确认选择的 backend
3. **Profiler**：抓 CUDA kernel，确认 `flash_fwd` 是否出现及耗时占比
4. **显存**：对比峰值显存，FlashAttention 应明显更低
5. **瓶颈分析**：用 profiler 看 Attention 在总前向中的时间占比，判断是否是瓶颈
6. **形状扫描**：在不同 seq_len/batch 下测试，找到收益显著的区间

---

## 参考与致谢

- 教程地址：https://github.com/datawhalechina/llm-algo-leetcode
- Datawhale 社区：https://www.datawhale.cn/
- 专题正文：`topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md`
- 核心 Notebook：`02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`、`20_FlashAttention_Sim.ipynb`、`INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb`
