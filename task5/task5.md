# Task5 打卡 · 低精度执行与单卡部署准入

> 202610 推理优化 | Issue #230 | 截止 10/27 周二 12:00

---

## 打卡信息

- **微信群昵称**：（填写）
- **环境**：本地 NVIDIA / 云 GPU / 无 GPU / 国产卡 / Mac / 其他
- **系统**：Windows
- **Python**：（填写）
- **AI 占比 vs 人工占比**：（填写，如 70:30）
- **任务项**：4.1 + 4.2 + 4.3
- **当前卡点**：无

> 说明：本机未做 GPU 实测，下面的数字均为按教程口径手算/推导的结果；实测部分给出可复现步骤（见文末「复现与本地材料」），不伪造测量值。

---

## 4.1 最小打卡

### 4.1.1 误差机制题（异常值、裁剪与饱和）

已知：某层权重 99% 落在 `[-2, 2]`，混入一个异常值 `100`；对称 INT8，`scale = max|x| / 127`。

**(a) 不裁剪时，`x = 1.0` 的舍入误差边界从多少恶化到多少？**

| 条件 | 校准范围 | scale = range / 127 | 舍入误差边界 ±scale/2 |
|:---|:---|:---|:---|
| 干净数据 `[-2, 2]` | 2 | 2 / 127 ≈ 0.015748 | ±0.007874 |
| 混入异常值 `100` | 100 | 100 / 127 ≈ 0.787402 | ±0.393701 |

结论：**从 ±0.0079 恶化到 ±0.394，约 50 倍**。格式没变、位宽没变，只是网格被一个点拉粗，普通值可用的有效码点从约 127 级掉到约 2.5 级。

**(b) 裁剪到 0.99 分位（约 `[-2, 2]`）后，`x = 100` 量化-反量化变成多少？属于哪类误差？**

- `scale = 2 / 127 ≈ 0.015748`
- `q = clamp(round(100 / 0.015748), -127, 127) = clamp(6350) = 127`
- 反量化 `x̂ = 127 × 0.015748 ≈ 2.0`

即 `100 → 2.0`，误差约 **98**（相对误差 98%）。这属于**裁剪误差（饱和误差 / saturation error）**：数值超出候选范围后被钳位，误差不可逆，且只集中在被裁掉的那约 1% 尾部样本上。

**(c) 每个裁剪候选必须同时记录哪两个指标？为什么只看 MSE 不够？**

必须同时记录：

1. **恢复误差（MSE，配最大绝对误差）** —— 衡量大部分数值的舍入精度；
2. **饱和率（saturation rate）** —— 衡量有多少数值被裁掉。

只看 MSE 不够的原因：饱和样本占比很小，但单个误差极大，放在均值里会被稀释；更糟的是可能出现「裁剪后 MSE 反而下降」的情况，而真实后果是少量 token/logits 被整体压平。教程因此规定选择顺序是**先用饱和率约束淘汰候选，再在可行集合里取验证 MSE 最低者**，且缩放因子只能来自校准集、误差只能在独立验证集上算，避免自证。

---

### 4.1.2 表示与粒度题（4096×4096 的字节账本）

已知：Linear 权重 4096×4096，FP16；对称 INT8 权重量化，scale 按 FP16（2 B）存。

元素数 = 4096 × 4096 = **16,777,216**；FP16 原始 = 33,554,432 B = **32 MiB**；INT8 权重 = 16,777,216 B = **16 MiB**。

**(a) 三种粒度的 scale 数量与总账本**

| 粒度 | 共享范围 | scale 个数 | scale 字节 | 权重 + 元数据 | 约合 |
|:---|:---|:---|:---|:---|:---|
| 逐张量 | 整个张量 | 1 | 2 B | 16,777,218 B | **16.00 MiB**（16.78 MB） |
| 逐通道 | 每个输出通道（4096 行） | 4,096 | 8,192 B = 8 KiB | 16,785,408 B | **16.01 MiB**（16.79 MB） |
| 逐组（g=128） | 每行 32 组 | 4096 × 32 = 131,072 | 262,144 B = 256 KiB | 17,039,360 B | **16.25 MiB**（17.04 MB） |

压缩比对照 FP16 的 32 MiB：约 2.07× / 2.06× / 2.03×。粒度越细，误差越低但元数据与内核索引成本上升，这是容量与精度的直接交换。

**(b) 某个输出通道混入异常值，谁受影响最大/最小？**

- **逐张量受影响最大**：一个异常值把全局 scale 抬到 ~100/127，全部 1677 万个权重一起掉回粗网格，整层精度被一个点绑架。
- **逐组受影响最小**：只有包含异常值的那一组（128 个权重）scale 被拉大，其余 13 万个组不受影响。
- **逐通道居中且方向不同**：异常值只污染它所在的那一个输出通道（4096 个权重），其他 4095 个通道完全正常；但如果这个通道整体敏感，损失会集中在一整条输出特征上。

所以「异常值局部化」的能力排序是：逐张量 < 逐通道 < 逐组。

**(c) W8A16 显存降了 50%，为什么首 token 延迟可能不变？**

“A16”意味着激活仍是 FP16，**被量化节省的只是权重驻留字节**，而反量化发生在计算之前：

- eager 路径：先把 INT8 权重显式恢复成 FP16 落地，再调用通用 GEMM；
- fused 路径：内核边读低比特权重边反量化，省掉落地的中间权重。

无论哪条，每层前向依然产生 FP16 精度的乘加，且额外多出反量化指令/地址计算。而 TTFT 由 **prefill 的计算量（FLOPs）与激活/KV 规模**决定，prefill 通常是 compute-bound，权重带宽不是主瓶颈；batch 越大权重读取还被摊薄。再加上激活、KV Cache、attention 仍为 FP16，显存账本里那部分没动。因此「显存下降 50%」与「TTFT 不变」并不矛盾——量化省的是容量和权重带宽，收益应该体现在小 batch decode 的带宽-bound 场景或并发容量上，而不是 prefill 延迟。

---

### 4.1.3 部署判断题（AutoDL 上加载 GPTQ 产物）

现象：加载成功、生成正常、峰值显存降 55%。

**(a) 按五层验证表，目前拿到了哪几层证据？缺哪一层关键证据？**

| 验证层 | 当前状态 |
|:---|:---|
| 模型产物 | 部分（文件存在且能被加载；哈希/manifest 未核对） |
| 加载器与执行后端 | 有（成功加载 + 非空生成，但缺 backend 版本记录） |
| 计算内核路径 | **缺**——没有任何证据证明跑的是低精度内核 |
| 服务工作负载 | 只有峰值显存一项，缺 TTFT/TPOT/吞吐/P99 的配对结果 |
| 质量门槛 | **缺**——没有独立质量集的 accept / tune / reject 结论 |

最关键的缺口是**计算内核路径**：显存下降可以由「权重以 INT8 存放」直接解释，不需要真的命中低比特 GEMM。只凭显存就下结论，等于把「格式兼容」当成「执行正确」。

**(b) 用什么证据确认实际执行的是低精度内核，而不是静默回退到 FP16？**

至少两种具体证据：

1. **Profiler / Nsight 的 kernel 级证据**：`torch.profiler`（CUDA activity）或 Nsight Systems/Compute 抓时间线，确认出现 int8/4bit dequant、quantized GEMM、FP8 kernel（如 marlin/gptq 相关 kernel 名），且主导耗时不是 `aten::mm` + FP16 GEMM 的常规组合；同时看 kernel 输入张量的 dtype。
2. **后端启动日志与模型状态回显**：vLLM / TensorRT-LLM 等会打印 quantization method 与所选 kernel backend（如 `gptq_marlin`、`awq_marlin`、`fp8`），再对照 `model.config.quantization_config` 与实际权重 dtype（`next(model.parameters()).dtype` 为 int8/uint8），两者一致才算命中。

补充佐证：同 workload 下显存带宽利用率与 GPU 时间构成、以及相对 FP16 基线的吞吐差异，都能帮助排除「加载了量化文件却按 FP16 执行」。

**(c) 内核证据也有了，但 TTFT 和吞吐都没改善，下一步查什么？**

对应现象-判断表的「显存下降但速度没改善 → 带宽或计算内核没有受益」。按顺序查：

1. **口径与配对**：TTFT 是否混入了排队/首包网络开销，是否与 FP16 基线同模型、同 backend、同输入输出长度、同并发；
2. **workload 形状**：是否是长 prompt、prefill 主导（compute-bound，权重带宽无关），decode 占比太低；
3. **并发与 batching**：batch 太小，kernel launch 与反量化开销无法摊薄，权重读取也未被复用；
4. **KV Cache 是否量化**：长上下文/高并发下容量仍被 KV 占满，量化权重带来的容量没有被转成并发；
5. **反量化是否重复发生**：eager 路径每步都重新恢复 FP16 权重并落地临时张量，可能吃掉全部收益；
6. **非 GPU 瓶颈**：调度器、tokenizer、采样、CPU 侧同步是否成为端到端瓶颈。

若以上都排除，结论应是「该 workload 上量化只买到显存/容量，不买延迟」，据此改选部署目标（提升并发而非追求单请求变快），而不是继续调量化参数。

---

## 4.2 学有余力增项 1（浮点基线、能力门槛、执行路径）

**(a) 既然要量化，为什么不直接 INT8 对比 FP16，还要先花时间跑浮点基线？**

基线的作用是把「表示变化」变成唯一变量。没有 FP16（同模型 revision、同 backend 与版本、同 workload、同测量口径）的 TTFT/吞吐/显存数字时：

- 量化后的数字**失去参照**，无法区分收益/损失来自量化、驱动升级、形状变化还是 batch 变化；
- 无法发现**静默回退**：回退到 FP16 时程序照样生成正常，数字看着「合理」，但其实是 FP16 的成绩；
- 无法计算质量-成本权衡（掉多少质量换多少容量），也无法在多个候选（W8A16/GPTQ/AWQ/FP8）之间做同口径排序；
- 结果不可复现、不可交接，换个人/换台机器就解释不了差异。

**(b) 融合候选 `dequant+matmul`，后端版本过旧不支持该内核，会怎样？为什么比直接报错更危险？**

分派器会逐项检查能力门槛（pattern、dtype、device、shape、dynamic shape），该项不通过就**跳过这个后端**，尝试下一个兼容路径；全部不兼容时**显式回退 eager**，并把拒绝原因写进决策记录（`decision: fallback`）。

危险在于：它不抛异常、不中断服务，输出数值还是对的。于是团队很容易把「回退后的 FP16 成绩」当成「量化后的成绩」，或者反过来把「量化没效果」归咎于算法；性能数字、容量规划、成本模型全部建立在错误前提上，而且没有任何告警。正确做法是把 `execution_path`、`decision`、拒绝原因作为指标打进日志并对 fallback 设告警，让「安静的成功」变成「可见的失败」。

**(c) 本地 4050 上用 bitsandbytes 4bit 跑 Qwen3.5-4B，属于哪条执行路径？判断依据？**

属于 **eager dequantization**：bnb 先用自带 kernel 把 4bit 权重恢复成 `bnb_4bit_compute_dtype`（通常是 FP16/BF16），再走通用 GEMM；4bit 只改了权重的存储与读取，矩阵乘仍是浮点。

判断依据：

- 4050 是 Ada（sm89），没有 int4 tensor core，也没有原生 4bit GEMM 指令；
- 配置侧看到 `load_in_4bit=True` / `bnb_4bit_quant_type` / `bnb_4bit_compute_dtype`，未指定任何 fused 量化内核；
- profiler 中应看到 dequant 类 kernel 与 FP16 GEMM 成对出现，而不是单一 4bit GEMM；
- 现象上显存明显下降，但小 batch 速度提升有限甚至变慢——这正是 eager 反量化的典型特征。

---

## 4.3 学有余力增项 2（产物准入清单、配对复测、质量门槛）

**(a) 产物侧：除文件存在，还要固定哪四项信息？**

1. **文件哈希**：权重 / config / tokenizer 的 sha256，保证「测的就是要上线的」；
2. **量化对象**：被量化的是权重、激活、KV Cache 还是混合位宽，哪些层/模块被量化；
3. **粒度与位宽**：format（GPTQ/AWQ/GGUF/FP8）、bits、group_size、对称/非对称、zero point 规则；
4. **校准版本**：校准数据集 + revision + 样本数 + 校准脚本/配置版本。

配套还要记：base model id 与 revision、backend 与版本、独立质量集指标与阈值、`evidence_level`。缺任何一项都不能把「文件存在」升级为「可部署」。

**(b) 单卡配对复测：量化前后必须在哪些维度严格配对？（至少五个）**

1. **artifact 血缘**：同一 base model revision、同一量化 format/bits/group_size；
2. **backend 与版本**：同推理框架、同版本、同启动参数（含 kernel backend 选择）；
3. **硬件**：同 GPU 型号、驱动、CUDA、是否 MIG/多卡，同功耗与温度状态；
4. **workload**：同 prompt 长度分布、`max_output_tokens`、并发/batch、重复次数与 warmup；
5. **tokenizer 与采样参数**：同 tokenizer、同 temperature/top_p/stop 条件；
6. **测量口径**：同计时方式与同步方式，显存采集点一致，queue wait 是否计入统一；
7. **质量集**：同 held-out 质量集与阈值，避免用调参数据证明自己。

**(c) 质量门槛：为什么不能只测一组短文本，而要覆盖长上下文、结构化输出、工具调用？**

这是**统计偏移**问题：校准/评测分布与真实服务分布不一致时，误差在不同输入区间的表现完全不同。

- **短文本**只覆盖了浅层、短 KV 的场景，几乎不触发长上下文里的误差累积；
- **长上下文**：KV Cache 量化误差随序列长度累积，attention 的长程依赖对数值扰动更敏感，短样本看不到；
- **结构化输出（JSON/SQL）**：对 logits 边界极度敏感，微小偏移就可能产生非法格式，解析失败率远高于「文本看起来差不多」；
- **工具调用**：函数名与参数必须是精确 token，任何偏移都直接变成调用失败，属于硬错误而非软降级。

只测短文本会系统性低估误差，得出虚假的 accept；正确做法是按服务真实关心的输入构造分桶质量集，再给出 accept / tune / reject。

---

## 复现与本地材料

- 本地练习脚本：`my_learning/task5/quantized_inference.py`（W8A16 对称/非对称、简化 GPTQ/AWQ、FP8 E4M3/E5M2、KV Cache 量化、三层质量回归框架、部署成本账本）。
- 跑通后可得到权重误差与压缩比对照；当前 `my_learning/task5/quantization_analysis.json` 只记录了占位说明，未包含 GPU 实测数据。
- 若要做单卡实测，按以下顺序补齐证据：FP16 基线（66）→ 产物 manifest 与独立质量集（82）→ backend 加载与 profiler kernel 证据（67 / INF-EXECUTION-DISPATCH）→ 单卡配对复测（83）。命令示例：

```bash
python my_learning/task5/quantized_inference.py
# 实测路线另行记录：baseline -> artifact gate -> kernel evidence -> paired benchmark
```

---

## 参考与致谢

- 教程地址：https://github.com/datawhalechina/llm-algo-leetcode
- Datawhale 社区：https://www.datawhale.cn/
- 专题正文：`topic_discussion/inference_optimization/05_quantized_inference_and_deployment.md`
- 核心 Notebook：`02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb`、`52_Quantization_Calibration_and_Error.ipynb`、`INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb`
- 扩展阅读：`66_Inference_Performance_Comparison.ipynb`、`67_Quantized_Inference_and_Deployment.ipynb`、`82_Quantization_Artifact_Evaluation_Project.ipynb`、`83_Quantized_Serving_Profile_Benchmark.ipynb`
