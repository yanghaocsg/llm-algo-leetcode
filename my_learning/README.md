# 推理优化 202609 Task0-6 学习实现

> 课程：LeetCode 大模型算法与系统教程（推理优化方向）
> 仓库：https://github.com/datawhalechina/llm-algo-leetcode
> 专题入口：`topic_discussion/inference_optimization/intro.md`

---

## 目录结构

```
my_learning/
├── 学习规则总结.md              # Datawhale 学习者规则 v2 总结
├── README.md                    # 本文件（总览）
├── task0/
│   ├── request_path_and_metrics.py   # 请求结构与指标
│   └── benchmark_results.json         # 运行结果
├── task1/
│   ├── prefill_and_attention_kernel.py # Prefill 与 Attention Kernel
│   └── attention_comparison.json       # 运行结果
├── task2/
│   └── speculative_decoding.py         # 投机式生成与 Decode 加速
├── task3/
│   └── kv_cache_lifecycle.py           # KV Cache 状态与生命周期
├── task4/
│   └── serving_scheduling.py           # 多请求调度、异构 PD 与 Serving
├── task5/
│   └── quantized_inference.py          # 量化部署与成本
└── task6/
    └── distributed_inference.py        # 分布式推理、部署比较与决策
```

---

## Task 概览

| Task | 主题 | 核心内容 | 关键指标/概念 | 对应正文 |
|------|------|----------|---------------|----------|
| **Task0** | 请求结构与指标 | 请求生命周期、TTFT/TPOT 测量、Workload 分类、瓶颈判断框架 | TTFT, TPOT, throughput, P50/P95/P99, peak memory | `01_request_path_and_metrics.md` |
| **Task1** | Prefill 与 Attention Kernel | 标准 Attention vs FlashAttention、访存模型、Chunked Prefill、Roofline 分析 | 中间矩阵, HBM 读写, 算术强度, SDPA dispatch | `02_prefill_and_attention_kernel.md` |
| **Task2** | 投机式生成与 Decode 加速 | 贪心/采样解码、经典投机解码、多 Token 投机、接受率与有效推进量 | acceptance rate, effective tokens/step, 回退成本 | `03_decoding_strategies.md` |
| **Task3** | KV Cache 状态与生命周期 | KV Cache 增长模型、PagedAttention、Prefix Cache/RadixAttention、容量治理 | Block 管理, 前缀复用, 驱逐/重算, MHA/GQA/MLA | `04_kv_cache_lifecycle_and_reuse.md` |
| **Task4** | 多请求调度与 Serving | Continuous Batching、Decode 调度、KV Cache 调度、PD 分离、异构路由 | 批次效率, 排队时间, KV handoff, 接纳控制 | `07_serving_scheduling_and_pd.md` |
| **Task5** | 量化部署与成本 | W8A16、GPTQ、AWQ、FP8、KV Cache 量化、质量回归框架、成本分析 | 压缩比, 量化误差, 质量回归, backend/kernel 路径 | `05_quantized_inference_and_deployment.md` |
| **Task6** | 分布式推理与决策 | DP/TP/PP/EP 并行、通信观测、扩展效率、端到端 Benchmark、accept/tune/reject | 扩展效率, 通信时间, 决策框架, 成本/token | `08_distributed_inference_and_parallelism.md` + `06_benchmark_and_decision.md` |

---

## 运行方式

### 环境要求
- Python 3.10+
- PyTorch (CPU 版本即可运行机制模拟)
- 无需 GPU（所有实现均为 CPU 可运行的机制模拟和分析）

### 安装依赖
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 运行单个 Task
```bash
python task0/request_path_and_metrics.py
python task1/prefill_and_attention_kernel.py
# ... 以此类推
```

### 运行全部 Task
```bash
for i in 0 1 2 3 4 5 6; do
    python task${i}/*.py
done
```

---

## 学习路线图

```
Task0: 请求结构与指标
  │  建立请求生命周期和指标口径 (TTFT/TPOT/吞吐/P99/显存)
  ▼
Task1: Prefill 与 Attention Kernel
  │  长 Prompt 为什么推高 TTFT? FlashAttention 改变了什么?
  ▼
Task2: 投机式生成与 Decode 加速
  │  每轮如何提高有效推进量? draft/target 验证机制
  ▼
Task3: KV Cache 状态与生命周期
  │  Cache 如何建立/追加/分页/复用/驱逐? 容量边界在哪?
  ▼
Task4: 多请求调度与 Serving
  │  多请求如何共享资源? PD 分离和异构路由如何决策?
  ▼
Task5: 量化部署与成本
  │  权重/激活/KV 量化如何改变显存、速度、质量?
  ▼
Task6: 分布式推理与决策
     单实例不够时如何扩展? 如何形成可复查的部署决策?
```

---

## 瓶颈判断速查表

| 观察到的现象 | 优先怀疑的瓶颈 | 进入 Task |
|:---|:---|:---|
| 长 prompt 使 TTFT 明显升高 | prefill-bound / Attention 访存 | Task1 |
| TPOT 高、生成阶段占比大 | decode-bound / KV Cache / 调度 | Task2 / Task3 |
| peak memory 接近预算、batch 上不去 | memory-bound / Cache 容量 | Task3 / Task5 |
| 多请求时 P99 明显升高 | 排队 / batch 组织 / 调度 | Task4 |
| 模型装不下 / 显存下降但速度没改善 | 权重量化 / kernel 路径 | Task5 |
| 单卡吞吐不足 / 多卡扩展异常 | 通信 / 并行策略 / 负载不均 | Task6 |
| 没有明显单点瓶颈 | 需要端到端比较 | Task6 |

---

## 注意事项

1. **本实现为机制模拟**：所有代码在 CPU 上运行，用于理解核心概念和验证逻辑。真实性能数据需要 GPU + vLLM/SGLang backend 环境。
2. **教程原始 Notebook**：仓库 `02_PyTorch_Algorithms/` 目录下有更详细的 Notebook 版本，建议结合学习。
3. **打卡要求**：按照 Datawhale 学习规则，每个 Task 完成后需在 GitHub issue 打卡，并在群内接龙附截图。
4. **环境预检**：运行真实推理项目前，先查看 `docs/guide.md` 中的环境边界。

---

*生成时间：2026-10-08*
