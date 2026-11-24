# INF-EAGLE. Feature Drafter and Candidate Tree | 特征草稿器与候选树 · 小白入门

> 配套 notebook：[INF-EAGLE. 特征草稿器与候选树](../INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

小模型帮忙猜词时，会重复算一遍大模型已经算过的东西。这本 notebook 讲一种叫 EAGLE 的做法：直接用大模型的中间结果来猜，把候选排成一棵树，再让大模型检查。

## 为什么需要它

投机解码让一个独立的小模型先猜词。但小模型拿到的是文字，它要把上下文从头再编码一遍，而这件事大模型刚刚才做过，等于同样的活干了两遍。

EAGLE 类方法的想法是：别让小模型从头算，直接把大模型算出来的中间结果拿给它用。大模型的层会产生一串数字，叫隐藏表示，里面带着这一层对上下文的理解。小模型（这里叫草稿器）拿这些表示和当前词元当输入，用更低的成本往后猜几个候选词元。

这本 notebook 拆开讲三件事：

- 多层表示怎么合成一份草稿输入。不同层的表示含义不同，数值大小也不同。直接相加的话，数值大的那一层会盖住别的层，所以要先把权重归一化。
- 候选怎么组织。草稿器每一步可以给出不止一个候选，按“接在谁后面”排起来，就是一棵候选树。每个节点的分数是从根一路加过来的。
- 猜出来的候选怎么用。候选只是提议，仍要目标模型验证。一条路径上某个位置被拒绝，它后面的部分就不能再提交。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 隐藏表示（hidden representation） | 大模型某一层算出来的一串数字。本节里每一层是一个等长的列表，如 `[2.0, 4.0]` |
| 草稿器（drafter） | 负责猜候选的轻量模块。它不取代目标模型，只用当前词元和隐藏表示往后猜 |
| 验证器（verifier） | 检查候选的目标模型。候选通过它的验证才能提交到输出 |
| 融合权重 | 每层表示在合成时占的比重。必须非负、总和为正，融合前归一化成和为 1 |
| 候选树 | 用父节点关系表示多个可能的后续词元：一个节点是一个词元，它的子节点是接在它后面的候选 |
| 路径分数、`log_score` | 每条边自带一个分数，字段名是 `log_score`。从根一路加到某个节点，就是它的累计分数，用来给候选排序 |
| 自回归依赖 | 草稿器猜下一个候选时，要用到自己前面已经猜出的词元和状态，不能把各位置当成彼此独立 |

## 核心思路

题目里有两块小计算，都用普通的 Python 列表完成，不运行模型。

第一块是融合。输入是每层一个等长的列表 `layer_features`，以及每层一个权重 `weights`。先把权重归一化，再对每一个维度分别算跨层的加权和。下面是测试里的数字：

```text
权重 [1.0, 3.0] → 各除以总和 4.0 → 融合系数 [0.25, 0.75]
第 d 维的融合结果 = 0.25 × 第 1 层的第 d 维 + 0.75 × 第 2 层的第 d 维
```

第二块是候选树。输入是一个节点列表，按父节点在前的顺序排列。每个节点有三个字段：`parent` 是父节点的下标（根节点是 -1），`token` 是这个节点代表的词元，`log_score` 是这条边自带的分数。

```text
节点的累计分数 = 父节点的累计分数 + 自己的 log_score     （根节点从 0 加起）
```

算完所有节点的分数后，骨架代码在叶节点里挑分数最高的，再沿着 `parent` 走回根，把经过的词元倒过来，得到一条候选路径。

```python
weights = [1.0, 3.0]                                  # 两层表示的权重
total = sum(weights)
normalized = [w / total for w in weights]             # 归一化后的融合系数
layers = [[2.0, 4.0], [6.0, 0.0]]                      # 两层的隐藏表示，各 2 维
fused = [sum(n * layer[d] for n, layer in zip(normalized, layers)) for d in range(2)]
print(normalized, fused)

nodes = [
    {'parent': -1, 'token': 10, 'log_score': -0.1},   # 根节点
    {'parent': 0, 'token': 11, 'log_score': -0.2},    # 接在根后面的候选
    {'parent': 0, 'token': 12, 'log_score': -1.0},    # 接在根后面的另一个候选
    {'parent': 1, 'token': 13, 'log_score': -0.05},   # 接在词元 11 后面的候选
]
scores = []
for index, node in enumerate(nodes):
    parent = node['parent']
    parent_score = 0.0 if parent == -1 else scores[parent]   # 根从 0 开始，其余继承父节点的累计分数
    scores.append(parent_score + node['log_score'])
print(scores)

leaves = [i for i in range(len(nodes)) if i not in {n['parent'] for n in nodes}]
best = max(leaves, key=lambda i: scores[i])
path = []
while best != -1:
    path.append(nodes[best]['token'])
    best = nodes[best]['parent']
print(list(reversed(path)))
```

```text
[0.25, 0.75] [5.0, 1.0]
[-0.1, -0.30000000000000004, -1.1, -0.35000000000000003]
[10, 11, 13]
```

第一行是权重归一化和逐维融合的结果：`0.25 × 2 + 0.75 × 6 = 5.0`，`0.25 × 4 + 0.75 × 0 = 1.0`。第二行是 4 个节点的累计分数，小数末尾多出来的数字是浮点加法的误差，notebook 的测试里也原样写着这组数。节点 1 的分数是根节点的 -0.1 加上自己的 -0.2；节点 3 继承节点 1 的 -0.3，再加自己的 -0.05。叶节点是 2 和 3，节点 3 的分数更高，所以最佳路径是 `10 → 11 → 13`。

路径分数只用来决定先验证谁，它不是接受结论。目标模型验证一条路径时，一旦某个位置被拒绝，它后面的后缀就不能再提交，能提交的只有从起点开始连续通过的那一段。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | EAGLE 草稿器在投机解码链里的位置：它提出候选，目标模型仍然决定输出 | 读 |
| Step 2 | 多层隐藏表示怎样融合成草稿输入，候选为什么要逐步扩展 | 读 |
| Step 3 | 候选树怎样累计分数，验证后为什么只能提交连续通过的路径 | 读，对照小例子的后两行 |
| Step 4 · TODO 1 | 归一化多层表示权重 | 在 `normalize_fusion_weights` 里写 2 行：`total` 是权重之和，`normalized` 是每项除以 `total` |
| Step 4 · TODO 2 | 融合多层隐藏表示 | 在 `fuse_layer_features` 里写 1 个列表推导式：`fused` 的每一维是各层特征乘融合系数后相加 |
| Step 4 · TODO 3 | 累计候选树路径分数 | 在 `score_candidate_tree` 的循环里写 1 行：`parent_score`，根节点取 `0.0`，否则取 `scores[parent]` |
| 测试 | 检查非法权重、逐维融合结果和维度不一致、候选树的分支累计与最高分叶路径 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上用配对的目标模型和 EAGLE 草稿器，对照只用目标模型 | 需要 `INF-CUDA-MECHANISM` 环境、明确配对的模型和足够显存；默认是 `RUN_MODE = 'asset_check'`，只做资产检查，资产齐全后才执行真实 EAGLE 基准测试 |

Step 1–4 只需要 CPU，用的是 Python 列表，不运行模型。Step 5 是 GPU 实验，默认是 `RUN_MODE = 'asset_check'`：只检查目标模型、EAGLE 草稿器和执行后端是否齐全，条件未满足时保存失败记录，不生成性能结论，资产齐全后才执行真实 EAGLE 基准测试。notebook 记录的当前结果是实验条件尚未满足，这不表示 EAGLE 无效。

## 容易踩的坑

- **把各层表示直接相加**：某一层可能只是数值尺度大，就把草稿输入盖住了。要先固定层集合和形状，用非负、归一化后和为 1 的权重来融合。
- **权重没有校验**：权重必须非空、非负，而且总和为正。`[]`、`[0.0, 0.0]`、`[1.0, -1.0]` 都要报 `ValueError`。
- **各层隐藏维度不一致**：层数必须和权重数相同，每层的隐藏维度必须一致且非零，否则没法逐维相加。
- **把路径分数当成接受结论**：候选排序只确定验证顺序，不能替代目标模型的接受判定。某个位置被拒绝后，它后面的后缀不能继续提交。
- **父节点编号不合法**：节点按父节点在前的顺序排列，`parent` 必须是 -1 或更早节点的下标。指向自己、后面的节点或 -2 都要报错。

## 读完后自查

- [ ] 给权重 `[1.0, 3.0]`，能写出归一化后的融合系数。
- [ ] 给两层特征 `[[2.0, 4.0], [6.0, 0.0]]` 和上面的权重，能算出融合结果。
- [ ] 给一个小候选树，能算出每个节点的累计分数，并找出最高分的叶节点路径。
- [ ] 能解释为什么路径分数高的候选不一定会被全部接受。

## 下一步

- 打开 notebook：[INF-EAGLE. 特征草稿器与候选树](../INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb)
- 建议先读：[23. Speculative Decoding｜投机解码 · 小白入门](./23_Speculative_Decoding.md)、[35. Native Multi-Token Prediction｜原生多 Token 预测 · 小白入门](./35_Multi_Token_Decoding.md)
- 做完之后：[INF-PARALLEL-DRAFTING. DFlash 与 DSpark 并行草稿 · 小白入门](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md)
- 相关 notebook：[68. 投机解码机制选型基准](../68_Speculative_Decoding_Benchmark.ipynb)
