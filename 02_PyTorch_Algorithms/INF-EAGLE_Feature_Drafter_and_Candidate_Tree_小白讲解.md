# INF-EAGLE. 特征草稿器与候选树 — 小白讲解

> 对应 notebook：[`INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb`](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb)

## 一句话总结
EAGLE 类投机解码让轻量草稿器复用目标模型的隐藏表示，一次提出多个候选词元，再交给目标模型验证。本节用 `normalize_fusion_weights`、`fuse_layer_features`、`score_candidate_tree` 三个 TODO 练习两件事：多层表示怎样加权融合成草稿输入，候选树怎样沿父路径累加 log-score 并找出最高分叶路径。核心结论：草稿分数只用于排序，实际提交多少由目标模型的连续验证决定；本节只做 CPU 机制练习和资产检查，没有真实 EAGLE 基准结果。

## 为什么需要它：痛点是什么
普通投机解码要另训一个小模型做草稿，草稿器看不到目标模型内部已经算出的隐藏表示，只能从词元序列重新理解上下文。EAGLE 的想法是"别浪费目标模型已经算过的东西"：把浅、中、深几层的表示拿来，直接预测下一个词元。但候选不止一条，多个分支怎么组织、分数怎么累计、哪条路径值得送验，是另一个容易糊涂的点。
生活类比：导航软件已经知道整条路的实时路况（目标模型隐藏表示），EAGLE 让副驾驶（草稿器）据此先猜接下来几个路口怎么走，并画出几条备选路线（候选树）。副驾驶给路线打分只是排序；真正能不能走，要由交管（目标模型验证）逐段确认，第一段不通过就只保留前面已确认的部分。

## 核心问题
- 目标模型和特征草稿器各自负责什么？草稿器复用了哪些信息？
- 多层隐藏表示怎样变成下一步草稿输入？为什么还要一个学习投影？
- 候选树的边分数、节点累计分数和路径概率是什么关系？
- 为什么"草稿器认为最高分"不等于"目标模型会接受"？
- 权重归一化和逐维融合分别要拒绝哪些非法输入？
- 本节为什么只做到资产检查，不能给出 EAGLE 加速结论？

## 解决思路：一步步看懂

**第 1 步：分工与表示来源。** 目标模型先处理上下文并产生各层隐藏表示；EAGLE-3 读取浅、中、深三层，把三个同宽向量拼接，再通过学习得到的全连接投影压回草稿器需要的宽度；当前词元的嵌入也参与草稿计算。后续位置还没有新的目标表示时，草稿器就用上一步的草稿状态继续扩展，所以**沿一条路径仍有先后依赖**：第 3 个草稿词元依赖第 2 个草稿状态，不能把一条路径内部当成完全并行。

**第 2 步：权重归一化（TODO 1）。** `normalize_fusion_weights` 要求权重非空且非负，然后除以总和：

```python
if not weights or any(weight < 0 for weight in weights):
    raise ValueError('weights 必须非空且非负')
total = float(sum(weights))
if total <= 0:
    raise ValueError('权重总和必须为正')
return [float(weight) / total for weight in weights]
```

为什么先检查负权重再检查总和？空列表的 `sum` 是 0，`[0.0, 0.0]` 总和也是 0，两者都应拒绝；而 `[1.0, -1.0]` 总和为 0、单项还有负数，先拒绝负数语义更清楚。测试断言 `[1.0, 3.0] → [0.25, 0.75]`，并对 `[]`、`[0.0, 0.0]`、`[1.0, -1.0]` 三种输入要求抛 `ValueError`。归一化后权重和为 1，表示各层在融合中的相对贡献。

**第 3 步：逐维融合（TODO 2）。** `fuse_layer_features` 先检查层数非零、层数与权重数相同、每层隐藏维一致且非零，再对每个维度做跨层加权和：
$$\mathrm{fused}[j]=\sum_{\ell=1}^{L} \tilde w_\ell \cdot f_\ell[j]$$
其中 $\tilde w_\ell$ 是归一化权重，$f_\ell[j]$ 是第 $\ell$ 层第 $j$ 个分量。参考答案：

```python
return [
    sum(normalized[layer] * float(layer_features[layer][dimension])
        for layer in range(len(layer_features)))
    for dimension in range(hidden_size)
]
```

测试 `fuse_layer_features([[2.0, 4.0], [6.0, 0.0]], [1.0, 3.0])`：归一化权重 `[0.25, 0.75]`；第 0 维 `0.25×2 + 0.75×6 = 0.5 + 4.5 = 5.0`；第 1 维 `0.25×4 + 0.75×0 = 1.0`，结果 `[5.0, 1.0]`。两层宽度不同（`[[1.0], [1.0, 2.0]]`）必须失败，否则逐维加权没有定义。

**第 4 步：候选树累计分数（TODO 3）。** 每个节点是 `{'parent', 'token', 'log_score'}`。边分数用 log-score；若它表示条件概率的对数，沿路径相加就对应概率相乘。节点累计分数 = 父路径分数 + 本边分数：
$$S_i=S_{\mathrm{parent}(i)}+\log p_i,\qquad S_{\mathrm{root}}=0+\log p_{\mathrm{root}}$$
参考答案严格检查 `parent`：必须是 `-1` 或更早的节点（`parent >= index` 或 `parent < -1` 都拒绝），然后：

```python
parent_score = 0.0 if parent == -1 else scores[parent]
scores.append(parent_score + float(node['log_score']))
```

算完所有分数后，收集"从未当过别人父节点"的索引作为叶子，取 `max(leaves, key=lambda i: scores[i])`，再沿 `parent` 回溯、反转，得到根到叶的词元路径。测试的四个节点是：根 token 10 分数 −0.1；节点 1（父 0）token 11 分数 −0.2；节点 2（父 0）token 12 分数 −1.0；节点 3（父 1）token 13 分数 −0.05。累计为 `[-0.1, -0.30000000000000004, -1.1, -0.35000000000000003]`；叶子里节点 3 的 −0.35 最高，回溯得到 `[10, 11, 13]`。注意浮点尾数：−0.1 + −0.2 在二进制下不等于精确的 −0.3，测试直接比较这些浮点值，写代码时不要自己四舍五入。

**第 5 步：分数与验收是两件事。** Step 3 的例子说得很清楚：一条路径边分数 −0.1、−0.2、−0.05，累计 −0.35，即使排名最高，目标模型在第三个词元拒绝时也只能保留此前接受的前缀并执行修正，后续草稿状态随被丢弃的后缀一起撤销。`score_candidate_tree` 的返回值只是**候选排序**，不是接受长度，也不是最终输出。

**第 6 步：真实实验准备到什么程度。** Step 5 默认 `RUN_MODE='asset_check'`，用 `speculative_runtime.build_asset_gate_result` 检查五个资产（`target_model_id`、`target_model_revision`、`eagle_model_id`、`eagle_model_revision`、`eagle_backend`）和两个依赖包（`vllm`、`speculators`）。notebook 保存的输出显示：五个字段全部为空、两个包都不可用，`asset_gate.ready=False`，`baseline=None`、`candidates=[]`，`evidence_level='not_ready'`，`decision='defer_to_multi_gpu'`，失败类型 `assets_not_ready`。推荐组合是 `Qwen/Qwen3-8B` 配 `RedHatAI/Qwen3-8B-speculator.eagle3`，但本机参考显存 `LOCAL_GPU_MEMORY_GB=11.48`；8B 参数按 BF16 每参数 2 字节约 16 GB，单是目标权重就放不下，所以 notebook 明确"当前 12 GB 单卡不执行替代性探针"。默认 `TREE_WIDTH=4`、`TREE_DEPTH=4`，改树宽/树深或配对模型时要递增 `WORKLOAD_VERSION`。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 投机解码 | 草稿器先猜几个词元，目标模型一次验证多个 |
| 隐藏表示 | 模型中间层为每个词元算出的向量 |
| 特征草稿器 | 复用目标表示、与目标模型配对的轻量预测器 |
| 多层融合 | 把浅/中/深几层表示拼起来，再投影回草稿器宽度 |
| 学习投影 | 可训练的全连接层，决定各层分量怎么组合 |
| 候选树 | 多条草稿分支按父子关系组成的树 |
| log-score | 条件概率的对数；沿路径相加对应概率相乘 |
| 节点累计分数 | 父路径分数 + 当前边分数，用于排序 |
| 叶节点 | 没有子节点的候选末端，最高分叶决定送验路径 |
| 已接受前缀 | 目标模型从路径起点连续通过的部分 |
| 首个拒绝位置 | 验证第一次失败的位置，之后要修正 |
| 资产门 | 模型、版本、后端字段和依赖包是否齐全的检查 |

## 小白常见疑问
**Q1：为什么融合要自己归一化权重，而不是直接用原始权重？**
加权和里权重的绝对大小会整体缩放输出，不便于解释"每层占多少贡献"。归一化到和为 1 后，`[1,3]` 与 `[0.25,0.75]` 表达同样的相对比例，测试也用它来检查实现。注意这是本题的教学简化；真实 EAGLE-3 用的是拼接加学习投影，不是这组手工权重。

**Q2：节点的 `parent` 为什么必须"更早"？**
累计分数要求父节点先算完。如果 `parent >= index`，父节点还没算，`scores[parent]` 会越界或取到错误值；`parent < -1` 则是非法编号。`test_candidate_tree_contract` 用根节点的 `parent=0`（指向自己）和 `parent=-2` 验证两种拒绝。

**Q3：最高分叶路径就是最终输出吗？**
不是。它只是草稿器推荐送验的路径。目标模型逐词元验证，第三个词元被拒时只能提交前两个已接受词元，再做修正；被丢弃后缀的草稿状态也要撤销。分数高只代表草稿器"觉得有希望"。

**Q4：为什么 `fuse_layer_features` 里又调了一次 `normalize_fusion_weights`？**
因为融合函数对外接收原始权重，必须自己保证归一化，不能假设调用方已经处理过。这样单独调用 `fuse_layer_features` 也安全；重复归一化同一组权重不会改变结果（除非有浮点误差）。

**Q5：notebook 里有 EAGLE 的加速数据吗？**
没有。5.2 只保存资产检查记录；5.4 写得很明确：五个配对字段为空、`vllm` 和 `speculators` 不可用、`baseline` 为 `None`、`candidates` 为空，`decision` 是 `defer_to_multi_gpu`。任何"加速多少倍"的说法都不来自本 notebook。

## 动手验证：怎么知道自己做对了
notebook 直接按顺序调用四个测试，全过打印 `✅ EAGLE 表示融合与候选树测试通过`。TODO 未完成会抛 `NotImplementedError`。
- `test_normalize_fusion_weights`：`[1.0, 3.0]` 必须等于 `[0.25, 0.75]`；`[]`、`[0.0, 0.0]`、`[1.0, -1.0]` 都必须抛 `ValueError`。
- `test_fuse_layer_features`：两层 `[[2.0, 4.0], [6.0, 0.0]]`、权重 `[1.0, 3.0]` 融合为 `[5.0, 1.0]`，逐元素误差小于 `1e-9`；`[[1.0], [1.0, 2.0]]` 因隐藏维不一致抛 `ValueError`。
- `test_score_candidate_tree`：四个节点的分数必须是 `[-0.1, -0.30000000000000004, -1.1, -0.35000000000000003]`，路径必须是 `[10, 11, 13]`。
- `test_candidate_tree_contract`：空树、`parent=0` 出现在索引 0、`parent=-2` 三种非法拓扑都抛 `ValueError`。
- Step 5 保存了资产检查输出：`missing_packages == ['vllm', 'speculators']`、`missing_assets` 为五个字段、`ready is False`、`decision == 'defer_to_multi_gpu'`。5.3 的 `read_mechanism_result` 读取同一文件，字段应与 5.2 一致；它只证明"资产没准备好"，不证明任何性能结论。
