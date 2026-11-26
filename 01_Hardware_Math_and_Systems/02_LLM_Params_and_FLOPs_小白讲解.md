# 02. 大模型参数量与算力推导 — 小白讲解

> 对应 notebook：[`02_LLM_Params_and_FLOPs.ipynb`](./02_LLM_Params_and_FLOPs.ipynb)

## 一句话总结
这节把"模型配置 → 参数量 → 计算量 → 训练时间 → 算力利用率"串成一条估算链：先用层数、隐藏维度、词表数出一个 Transformer 有多少参数，再用 2PT（一次前向）和 6PT（一次完整训练）估算要多少 FLOPs，最后用 MFU 看硬件实际用上了几成。全程只是在 CPU 上做乘除法。

## 为什么需要它：痛点是什么
"7B"只是模型名字里的规模标签。被人问"这个配置为什么是 7B""用 1T 数据训一遍要多少张卡跑几天""你的训练算力用到几成"时，只会引用别人的数字，自己算不出来，也判断不了数字合不合理。
类比：装修报价。先按户型数出要买多少材料（参数量），再按面积估工时（FLOPs），最后看工人真正干活的时间占在岗时间几成（MFU）。在岗 8 小时不等于干了 8 小时活，"卡开了多久"也不等于"算了多少"。

## 核心问题
- 给定隐藏维度、词表和层数，怎么估一个 Decoder-only Transformer 的参数量？
- 处理 $T$ 个 token，一次前向要多少 FLOPs？为什么是 2PT？
- 为什么完整训练是前向的 3 倍，也就是 6PT？
- 给定 GPU 峰值、卡数和效率，怎么粗估训练时间？
- MFU 是什么？为什么很难到 100%，低了又能说明什么？

## 解决思路：一步步看懂
1. **数参数（Q1）。** 设隐藏维度 $d$（模型内部每个向量的长度）、词表大小 $V$（模型认识多少种 token，即词块）、层数 $L$。结构取简化的 Decoder-only Transformer，用多头注意力（MHA），每层两个只含缩放向量的归一化层；省略偏置和末尾归一化层，词嵌入与输出头默认不共享。参数分四处：
   - 词嵌入（把 token 编号变成向量）和输出头（把向量变回词表上的分数）各 $Vd$，共享时只减少一份 $Vd$，不是整个模型减半；
   - 注意力的 Q、K、V、输出投影各一个 $d \times d$ 矩阵，每层 $4d^2$（GQA 会改变 K、V 投影的大小，要按实际头数另算）；
   - 前馈网络（FFN）：设中间维度为 $m$，标准 FFN 有升维、降维两个矩阵共 $2dm$，SwiGLU 多一个门控投影共 $3dm$；
   - 每层两个归一化层的缩放向量共 $2d$。

   标准 FFN 取 $m=4d$、或 SwiGLU 取 $m \approx 8d/3$ 时，前馈都约为 $8d^2$，于是省略归一化项得到常用近似 $P \approx 2Vd + 12Ld^2$（每层 $4d^2 + 8d^2 = 12d^2$，$P$ 是总参数量）。手算小例子 $V=10, d=6, L=1$、标准 FFN $m=24$：词嵌入和输出头 $2 \times 10 \times 6 = 120$，注意力 $4 \times 6^2 = 144$，前馈 $2 \times 6 \times 24 = 288$，归一化 $2 \times 6 = 12$，合计 564；SwiGLU 取 $m=16$ 时前馈是 $3 \times 6 \times 16 = 288$，总数不变。`calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size=None, tie_embeddings=False, ffn_type='swiglu')` 会计入 $2Ld$ 个归一化参数；不给 `intermediate_size` 时 dense 取 $4d$、SwiGLU 把 $8d/3$ 四舍五入为整数。7B 级配置 $V=32000, d=4096, L=32$、SwiGLU 中间维度 11008（这是显式配置，不替换成未取整的 $8d/3$）：$262{,}144{,}000 + 2{,}147{,}483{,}648 + 4{,}328{,}521{,}728 + 262{,}144 = 6{,}738{,}411{,}520$，约 6.74B。
2. **前向：2PT（Q2）。** FLOPs 是浮点运算次数。矩阵乘法 $Y = W \times X$ 里，每个权重元素对每个 token 要做一次乘法加一次加法，叫 MAC（Multiply-Accumulate，乘加），所以"1 个参数处理 1 个 token 约 2 次 FLOPs"。于是 $C_{forward} \approx 2 \times P \times T$：$P$ 是参数量，$T$ 是 token 数。例：7B 模型处理 1T（$10^{12}$）token，$2 \times 7 \times 10^9 \times 10^{12} = 1.4 \times 10^{22}$。它忽略了 Attention 矩阵乘积、归一化、激活函数、padding 和其他算子，这些的占比随序列长度、模型结构和 batch 改变，不能固定写成 99%。`compare_forward_token_scales` 固定参数量、比较不同 token 数，看的只是 FLOPs 对 token 数的线性关系，不是实测耗时。
3. **训练：6PT（Q3）。** 训练 = 前向算损失 + 反向算梯度。反向要做两件事：算激活值（网络中间结果）的梯度，把误差继续往后传，约 $2PT$；算权重的梯度，用来更新参数，也约 $2PT$。所以反向约是前向的 2 倍，$C_{train} \approx 2PT + 4PT = 6 \times P \times T$：

   ```python
   def training_flops_breakdown(num_params_b, num_tokens):
       forward = calculate_forward_flops(num_params_b, num_tokens)          # 2PT
       activation_backward = 2 * num_params_b * 1_000_000_000 * num_tokens  # 2PT
       weight_backward = 2 * num_params_b * 1_000_000_000 * num_tokens      # 2PT
       return {'forward': forward, 'activation_backward': activation_backward,
               'weight_backward': weight_backward, 'total': forward + activation_backward + weight_backward}
   ```

   实战估算：LLaMA-7B 训 1T token，$6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22}$ FLOPs；按每张 A100 的 BF16 峰值 312 TFLOPs、35% 的效率、1000 张卡，有效算力是 $1000 \times 312 \times 0.35$ TFLOPs，相除约 4.5 天。`estimate_training_time(num_params_b, num_tokens, gpu_tflops, num_gpus, efficiency=0.35)` 做的就是这个除法，返回小时数。注意 6PT 只是全参数训练和微调的基线；LoRA、QLoRA 在这里只作概念对照，它们的收益不能只从 FLOPs 公式判断。
4. **MFU（Q4）。** MFU（Model FLOPs Utilization，模型 FLOPs 利用率）= 模型 FLOPs ÷ 实际秒数 ÷ 硬件峰值 FLOP/s。三项必须覆盖同一范围：工作量算的是整个集群，分母就用所有参与设备的峰值之和；算的是单卡，就用单卡峰值。它很难到 100%，因为 GPU 很多时间在等：等数据从内存搬过来（Memory-bound，显存墙），或等别的卡把 All-Reduce（多卡同步数据的通信操作）传过来（Communication，通信瓶颈）。`calculate_mfu(model_flops, elapsed_seconds, peak_tflops)` 里，单卡 $10^{15}$ FLOPs、10 秒、峰值 312 的 MFU 约为 $1/3.12 \approx 0.32$，换成 8 卡（工作量和峰值都乘 8、耗时不变）MFU 不变。`compute_time_fraction(compute_ms, memory_wait_ms, comm_wait_ms)` 把互不重叠的计算、等访存、等通信三段时间相加，返回计算时间占比 `compute_time_fraction`、主要等待来源 `dominant_stall`（`memory`、`communication`、`mixed` 或 `none`）和等待占比 `stall_ratio`；按代码手算，三组用例 (100, 30, 20)、(100, 60, 40)、(100, 10, 5) 的计算占比约为 0.667、0.5、0.87。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 隐藏维度 $d$ | 模型内部每个向量的长度 |
| 词表大小 $V$ / token | 模型认识多少种词块 / 一次处理的一个词块 |
| 词嵌入 / 输出头 | token 编号变向量 / 向量变回词表分数，各 $Vd$ 个参数 |
| MHA（多头注意力） | Q、K、V、输出四个 $d \times d$ 投影，每层 $4d^2$ |
| FFN / SwiGLU | 前馈网络；标准两个矩阵 $2dm$，SwiGLU 三个矩阵 $3dm$ |
| 中间维度 $m$ | 前馈先升维到的宽度，常取 $4d$ 或约 $8d/3$ |
| `tie_embeddings` | 词嵌入和输出头共用一份参数，只省一份 $Vd$ |
| FLOPs / MAC | 浮点运算次数 / 一次乘法加一次加法 |
| 2PT / 6PT | 一次前向 / 一次训练（前向 + 反向）的计算量估算 |
| MFU | 模型 FLOPs ÷ 实际秒数 ÷ 硬件峰值，硬件"用上了几成" |
| Memory-bound / All-Reduce | 等数据搬运的显存瓶颈 / 多卡同步数据的通信操作 |

## 小白常见疑问
**Q1：词嵌入和输出头共享后，参数量会减半吗？**
不会。两者各 $Vd$，共享只省一份 $Vd$。小例子里不共享是 564，`tie_embeddings=True` 是 504，只少了 60。

**Q2：SwiGLU 多了一个矩阵，前馈参数为什么没有变多？**
矩阵多一个，中间维度却从 $4d$ 降到约 $8d/3$：$2 \times 4d = 8d$，$3 \times 8d/3 = 8d$，再乘 $d$ 都是 $8d^2$。

**Q3：能不能说"2PT 和 6PT 已经很准，其他算子只占 1%"？**
不能。它们是 dense Transformer 的数量级估算，被忽略的 Attention 矩阵乘积、归一化、激活等算子占比会随序列长度、模型结构和 batch 改变，notebook 明确说不能固定写成 99%。

**Q4：MFU 低，能不能直接断定是通信或 I/O 拖后腿？**
不能。MFU 没有脱离模型、精度、硬件、并行方式和统计口径的通用合格线，低了只说明要继续检查，得和时间线、访存、通信、输入流水线的记录一起看。反过来，计算时间占比高也不能单独证明 MFU 高。

**Q5：估出的 4.5 天是实际训练时间吗？**
不是。它随峰值和效率的假设变化，也不检查显存够不够，只用来建立数量级直觉。

## 动手验证：怎么知道自己做对了
`test_calculate_transformer_params` 断言：`(10, 6, 1)` 用 dense FFN 得到 `120 + 144 + 288 + 12 == 564`，SwiGLU 中间维度取 16 时结果相同，减去近似式 $2Vd + 12Ld^2$ 后正好剩 12（归一化项），`tie_embeddings=True` 得 504；`(1000, 64, 2, 256)` 在共享时得 195328、不共享得 259328、共享且 dense 得 162560；`(32000, 4096, 32, 11008)` 得 6_738_411_520，通过后打印"✅ 参数量函数测试通过！"。`test_calculate_flops` 断言 7B 模型处理 1T token 的前向 FLOPs 是 14_000_000_000_000_000_000_000（即 1.4e22）。
Q3、Q4 没有独立测试函数，断言写在单元里：拆分后的 `total` 等于 `calculate_training_flops(7, 1e12)`，且等于前向的 3 倍；`estimate_training_time(7, 1e12, 312, 1000, 0.35)` 换算成天约为 4.451566951566952（误差小于 1e-9）；单卡与 8 卡 MFU 相等且约为 1/3.12；`compute_time_fraction(100, 0, 0)` 和 `(0, 0, 0)` 的 `dominant_stall` 都是 `'none'`，`(100, 30, 20)` 的计算占比等于 0.667。实验单元格会打印：7B 模型在 1000、8000、32000 token 下的前向 FLOPs（应随 token 数线性增长）、训练 FLOPs 拆分、四个场景（1 张、8 张、1000 张 A100 以及"8 卡假设峰值 500 TFLOP/s"）的小时数和天数。notebook 没有 benchmark，dump 里也没有保存输出，数字以你自己运行后打印的为准。
