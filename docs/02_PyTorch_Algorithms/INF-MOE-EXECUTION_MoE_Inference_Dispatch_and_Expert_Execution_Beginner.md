# INF-MOE-EXECUTION. MoE Inference Dispatch and Expert Execution Beginner | MoE 推理中的词元分组与专家执行 · 小白入门

> 配套 notebook：[INF-MOE-EXECUTION. MoE 推理中的词元分组与专家执行](./INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

混合专家模型里，每个词元只会交给少数几个“专家”网络处理。这本 notebook 教你把发给同一个专家的词元排在一起算，再按原顺序和各自的权重把结果加回去。

## 为什么需要它

大模型里有一层并排放着很多个专家，每个专家是一个前馈网络。词元进来时，路由器只挑出得分最高的 K 个专家处理它，这叫 Top-K 路由，输出再按路由权重加起来。

路由结果定了以后，怎么执行有两种做法：

- 逐词元调用：一个词元找一次专家，每次只处理一行。解码时每轮词元少，有的专家只处理一两行。
- 按专家批量调用：把发给同一个专家的行排到一起，让这个专家一次处理一批。

两种做法的输出相同，矩阵大小和任务数量却不同。第二种要多做两步：排队之前，记下每一行来自哪个词元、用哪一路权重；算完以后，按原位置把同一个词元的多份贡献加回去。这两步做错了，输出形状照样对，数值却错了。

这本 notebook 在 CPU 上走一遍这条流程：按专家分组、加权合并，再数每个专家分到多少行。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 混合专家（MoE） | 一层里并排放着很多个专家，每个词元只交给其中少数几个处理 |
| 专家 | 一个前馈网络，可以理解成这一层里可供选择的一条计算路径 |
| Top-K 路由 | 路由器给每个词元选出得分最高的 K 个专家；路由权重是这几个专家的输出各占多少 |
| 分配 | 一次“某个词元交给某个专家”的记录。`T` 个词元各选 `K` 个专家，就有 `T * K` 次分配 |
| 预填充（prefill）/ 解码（decode） | 预填充一次处理整段输入，词元多；解码每轮只处理新生成的词元，词元少 |
| 填充（padding） | 为了凑满固定的行数补上去的无效行，不对应任何词元 |
| 负载偏斜 | 最忙专家的行数和全体专家平均行数的比值，每个专家分到的行数都一样时等于 1 |

## 核心思路

“按专家批量执行”分三步：

- 分组：输入是词元的特征 `x`、选中的专家编号 `expert_ids` 和路由权重 `route_weights`。把所有分配按专家编号排序，输出排好序的输入、每行的原词元位置和权重，以及每个专家的行数 `counts` 和区间边界 `offsets`。靠记下的原词元位置就能恢复原顺序，所以叫“可逆”。
- 专家计算：每个专家处理自己的那几行，测试里用“乘一个系数”代替真正的专家。
- 合并：每行输出乘上自己的路由权重，再按原词元位置累加，输出的行数和原来的词元数一样。

合并的依据是 notebook Step 2 的公式：

```text
y_t = sum over j = 0 .. K-1 of  g[t, j] * F[e[t, j]](x_t)
```

`t` 是词元编号，`j` 是第几路，`e[t, j]` 是这一路交给哪个专家，`F[e]` 是专家 `e` 的计算，`g[t, j]` 是这一路的路由权重，`y_t` 是第 `t` 个词元最后的输出。

Step 3 的两个负载式子里，`c_e` 是专家 `e` 分到的行数，`E` 是专家个数，`b` 是假定的行块宽：

```text
padded_e   = b * ceil(c_e / b)         # 向上取整到块宽，空专家为 0
有效行比例 = sum_e c_e / sum_e padded_e
偏斜 I     = max_e c_e / (T * K / E)    # 最忙专家的行数 ÷ 全体专家的平均行数
```

下面用 notebook 测试里的路由跑一遍：5 个词元、4 个专家，每个词元选 2 个。

```python
import torch

ids = torch.tensor([[2, 0], [0, 2], [2, 1], [1, 0], [0, 1]])   # 5 个词元，各选 2 个专家
flat_experts = ids.reshape(-1)                                  # 10 次分配的专家编号
flat_tokens = torch.arange(5).repeat_interleave(2)              # 每次分配来自哪个词元

order = torch.argsort(flat_experts, stable=True)                # 按专家编号排序，同组保持原顺序
print(flat_experts[order].tolist())
print(flat_tokens[order].tolist())
counts = torch.bincount(flat_experts, minlength=4)               # 每个专家分到几行
print(counts.tolist(), torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).tolist())

padded = ((counts + 16 - 1) // 16) * 16                         # 行块宽 16：非空专家向上取整
print(padded.tolist())
print(counts.sum() / padded.sum(), counts.max() / (counts.sum() / 4))

out, w = torch.tensor([2.0, 6.0]), torch.tensor([0.75, 0.25])   # 两个专家对同一个词元的输出
print((out * w).sum(), (out * w.flip(0)).sum())                 # 权重对应正确、权重错换
```

```text
[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
[0, 1, 3, 4, 2, 3, 4, 0, 1, 2]
[4, 3, 3, 0] [0, 4, 7, 10, 10]
[16, 16, 16, 0]
tensor(0.2083) tensor(1.6000)
tensor(3.) tensor(5.)
```

排序以后，专家 0 的 4 行来自词元 0、1、3、4；专家 3 一行也没有，区间边界都是 10。每个专家的区间包含起点、不包含终点。行块宽取 16 时，三个非空专家各补到 16 行，有效行比例约 0.21，偏斜是 1.6。最后一行是合并：权重对应正确得 3，错换得 5。

notebook 特意提醒：偏斜和填充是两项独立的代价。“激活参数”指本轮所选专家涉及的参数，不是前向传播产生的激活张量；全部专家的权重如果常驻设备，占的容量不会随本轮激活数量同比下降。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从固定路由到专家批次与原序输出：逐词元调用和按专家批量调用的差别 | 读 |
| Step 2 | 排序改变执行顺序，权重与词元身份保持对应 | 读，对照上面的公式 |
| Step 3 | 词元规模与专家偏斜怎样改变执行成本，以及“激活参数”和全部权重驻留的区别 | 读，对照上面两个负载式子 |
| Step 4 · TODO 1 | 按专家建立可逆分组 | 写 `pack_expert_assignments`：补 4 行 `order`、`token_ids`、`counts`、`offsets` |
| Step 4 · TODO 2 | 恢复词元顺序并合并贡献 | 写 `combine_expert_outputs`：补 2 行 `weighted` 和 `output`，按位置累加 |
| Step 4 · TODO 3 | 量化填充与专家负载 | 写 `expert_load_metrics`：补 3 行 `padded`、`utilization`、`imbalance` |
| 测试 | 用独立的逐词元循环核对结果，并覆盖空专家、空输入、顺序打乱和非法路由 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上用同一组 SwiGLU 专家参数，对照逐词元执行与按专家批量执行 | 需要 GPU 环境 `INF-CUDA-MECHANISM`，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU，使用 `INF-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **排序时只搬了输入**：要用同一个排列同时搬动输入、原词元位置和权重。只恢复位置、没恢复权重的对应关系，形状仍然正确，数值已经变了。
- **合并时覆盖写入**：一个词元有多行专家输出，要按位置累加。覆盖写回会丢掉其他专家的贡献。
- **边界情况没处理**：填充只对非空专家向上取整，空专家保持零；一项分配都没有时，有效行比例和偏斜都约定为 0。
- **把偏斜当成延迟预测**：偏斜只是负载描述。偏斜变大有时还减少活跃专家数，不能仅凭它就断言单卡更慢。

## 读完后自查

- [ ] 能说出 5 个词元各选 2 个专家时，一共有多少次分配。
- [ ] 两个专家的输出是 2 和 6、权重是 0.75 和 0.25，能算出合并后的输出。
- [ ] 能说出 `counts` 和 `offsets` 各表示什么，空专家的区间长什么样。
- [ ] 给一组 `counts` 和行块宽 `b`，能算出填充后的行数、有效行比例和负载偏斜。

## 下一步

- 打开 notebook：[INF-MOE-EXECUTION. MoE 推理中的词元分组与专家执行](./INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md)
- 建议先读：[06. MoE 路由器 · 小白入门](./06_MoE_Router_Beginner.md)、[02. SwiGLU 激活 · 小白入门](./02_SwiGLU_Activation_Beginner.md)
- 需要补基础：[MoE 参数与计算](../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md)
- 做完之后可以对照：[46. MoE 专家并行](./46_MoE_Expert_Parallel.md)、[INF-EXECUTION-DISPATCH. 推理融合与后端分派](./INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)、[80. MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
