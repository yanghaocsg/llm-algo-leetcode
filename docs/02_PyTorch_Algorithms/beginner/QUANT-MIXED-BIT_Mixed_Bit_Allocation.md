# QUANT-MIXED-BIT. Mixed Bit Allocation | 混合位宽分配 · 小白入门

> 配套 notebook：[QUANT-MIXED-BIT. 混合位宽分配](../QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) ｜ 难度：Hard ｜ 环境：CPU-first，GPU 可选

## 一句话说清

混合位宽给不同的层安排不同的量化位宽：怕压的层留高精度，耐压的层用低位。这本 notebook 教你在容量预算内，决定哪些层值得多花字节去升级。

## 为什么需要它

最省事的做法是所有层用同一个位宽，比如全部压到 4 位。这等于把所有层看成同样怕压，但实际不是。题目测试里有两条记录：同样降到 4 位，`attention` 层的误差是 0.40，`mlp` 层只有 0.18。

想把 `attention` 留在高精度，又不能给所有层都留，因为总容量有预算。于是要回答：多出来的字节花在哪一层最值？做这个决定之前，有四样东西必须先备齐：

- 每层在各个位宽下的实测误差。它们必须用同一套条件测出来，否则没法比。
- 真实的容量。除了权重，还要算上缩放因子这类元数据。
- 执行后端支持哪些位宽。后端跑不了的位宽，写了也没用。
- 保护层规则。嵌入层、输出头这类层有最低位宽要求，不能被预算压过去。

这本 notebook 实现一个简单的分法：每层先选最省字节的合法位宽，再在预算内一层一层地升级。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 位宽（bit width） | 每个权重用几位保存。位数越少越省空间，误差通常越大 |
| 候选（candidate） | 某一层在某个位宽下的一条记录：位宽、权重字节、实测误差 |
| 敏感度（sensitivity） | 一层被量化后误差有多大。越敏感的层越值得保留高精度 |
| 容量预算 | 所有层的权重字节加起来不能超过的上限 |
| 保护层 | 有最低位宽要求的层，比如嵌入层、输出头 |
| 执行后端 | 真正运行量化模型的加载器和计算内核，它不支持的位宽不能用 |

## 核心思路

输入是每层的实测误差表、执行后端支持的位宽、容量预算和保护规则，输出是每层的位宽、总字节和总误差。过程分三步，连同题目里的算式一起列在这里：

```text
① 建候选    bytes = (numel * bits + 7) // 8
            只留后端支持、又有实测误差的位宽；numel 是这一层的参数量，结果向上取整到整字节
② 定基线    每层先选满足保护规则、字节最少的候选；合计已经超过预算就直接报错
③ 逐步升级  extra_bytes = 候选的字节 - 当前选中的字节      error_reduction = 当前选中的误差 - 候选的误差
            优先级 = error_reduction / extra_bytes；两者都为正才比较，每轮挑优先级最高的执行，直到没有
```

下面用题目测试里的两层走一遍。这是一种启发式做法，不保证全局最优，但每次升级都有可以追溯的理由：

```python
specs = {"attention": (1000, {4: 0.40, 8: 0.03, 16: 0.0}),    # 参数量，各档位宽的实测误差
         "mlp":       (2000, {4: 0.18, 8: 0.08, 16: 0.0})}
options = {name: [{"bits": b, "bytes": (n * b + 7) // 8, "error": e} for b, e in errors.items()]
           for name, (n, errors) in specs.items()}             # 每层的候选：位宽、字节、误差

def allocate(budget, protected={}):
    policy = {name: min(options[name], key=lambda o: o["bytes"]) for name in specs}   # 先选最省字节的
    for name, minimum in protected.items():                     # 保护层：位宽不能低于下限
        policy[name] = min((o for o in options[name] if o["bits"] >= minimum), key=lambda o: o["bytes"])
    total = sum(o["bytes"] for o in policy.values())
    if total > budget:
        raise ValueError("budget cannot satisfy the minimum feasible policy")
    while True:
        ups = [(cur["error"] - c["error"], c["bytes"] - cur["bytes"], name, c)
               for name, cur in policy.items() for c in options[name]
               if c["bytes"] > cur["bytes"] and cur["error"] > c["error"]
               and total + c["bytes"] - cur["bytes"] <= budget]
        if not ups:
            break
        gain, extra, name, cand = max(ups, key=lambda u: (u[0] / u[1], u[0]))   # 单位字节收益最大的一次升级
        policy[name], total = cand, total + extra
    return total, round(sum(o["error"] for o in policy.values()), 4), {n: o["bits"] for n, o in policy.items()}

print(options["attention"])
print(round((0.40 - 0.03) / 500, 7), round((0.18 - 0.08) / 1000, 7))   # 从 4 位升到 8 位：每多花 1 字节换来多少误差下降
print(allocate(2000))
print(allocate(3000, protected={"attention": 8}))
try:
    allocate(100)
except ValueError as exc:
    print(exc)
```

```text
[{'bits': 4, 'bytes': 500, 'error': 0.4}, {'bits': 8, 'bytes': 1000, 'error': 0.03}, {'bits': 16, 'bytes': 2000, 'error': 0.0}]
0.00074 0.0001
(2000, 0.21, {'attention': 8, 'mlp': 4})
(3000, 0.11, {'attention': 8, 'mlp': 8})
budget cannot satisfy the minimum feasible policy
```

- 第一行：`attention` 层的三个候选，4、8、16 位分别占 500、1000、2000 字节。
- 第二行：从 4 位升到 8 位，`attention` 每多花 1 字节换来 0.00074 的误差下降，`mlp` 只有 0.0001。
- 第三行：预算 2000 字节。两层都用 4 位是 1500 字节，把 `attention` 升到 8 位正好 2000 字节；升级 `mlp` 要 2500 字节，超预算。结果是 `attention` 8 位、`mlp` 4 位，总误差 0.21。
- 第四行：预算 3000 字节，并要求 `attention` 不低于 8 位。基线是 2000 字节，剩下的 1000 字节把 `mlp` 升到 8 位，总误差 0.11。
- 第五行：预算只有 100 字节，连最低可行的方案都放不下，直接报错。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 做混合位宽决策需要哪四类共同输入，缺了会怎样失败 | 读 |
| Step 2 | 三种逐层敏感度的测量方式各自的优点、局限和适用阶段 | 读 |
| Step 3 | “先保底，再按单位字节的误差下降升级”这个启发式，以及它的三个阶段 | 读，对照上面的小例子 |
| Step 4 · TODO 1 | 过滤不可执行的位宽并计算每个候选的权重字节 | 在 `build_options` 里写 1 个列表推导式：只留后端支持且有实测误差的位宽，字节数向上取整 |
| Step 4 · TODO 2 | 为每层选择满足保护规则的最小容量候选 | 在 `select_initial_option` 里写 4 行：筛出位宽不低于下限的候选，没有就报错，再取字节最少的 |
| Step 4 · TODO 3 | 在预算内选择位宽升级 | 在 `allocate` 里写 1 个 `max` 表达式：按“误差下降除以额外字节，再按误差下降”取最大 |
| 测试 | 检查后端不支持的位宽是否被过滤、保护层位宽、预算是否超限、只够一次升级时先升级谁、预算不可行时是否报错 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在真实模型上按 Block 测敏感度，比较统一 INT4、统一 INT8 和三档保护预算的混合位宽 | 需要 GPU 和 `Q-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU。Step 5 需要 GPU 和 `Q-CUDA-MECHANISM` 环境，默认是 `RUN_MODE = "dry_run"`，不执行真实 GPU 实验。

## 容易踩的坑

- **候选里混进了后端不支持的位宽**：只保留 `supported_bits` 里有的位宽。策略写在纸上可以，加载器或计算内核跑不了就没有意义。一层一个候选都不剩时直接报错。
- **各层误差不是同一套条件测出来的**：校准数据、其余层的状态和评测方法必须固定，否则“某层更敏感”可能只是实验条件不同。
- **字节只算了打包的权重**：算真实容量时，缩放因子这类元数据也要算进去。本题的教学实现只按参数量和位宽算权重字节。
- **保护层被预算压低了**：先满足保护下限，再选容量最小的候选。没有合法候选时直接失败，不能悄悄换成低位宽。
- **升级接受了不划算的候选**：只接受额外字节为正、误差下降也为正、并且升级后不超预算的候选。

## 读完后自查

- [ ] 能说出一个位宽候选至少要包含哪几项信息。
- [ ] 给两层的候选表和一个预算，能手算出基线，以及第一次升级选哪一层。
- [ ] 能说出预算放不下最低可行方案时应该怎么办。
- [ ] 能说出这种做法为什么不保证全局最优。

## 下一步

- 打开 notebook：[QUANT-MIXED-BIT. 混合位宽分配](../QUANT-MIXED-BIT_Mixed_Bit_Allocation.md)
- 建议先读：[52. 量化校准范围与裁剪 · 小白入门](./52_Quantization_Calibration_and_Error.md)、[40. GPTQ 与 AWQ 权重量化 · 小白入门](./40_GPTQ_and_AWQ_Weight_Quantization.md)、[53. 激活量化与 SmoothQuant · 小白入门](./53_Activation_Quantization_and_SmoothQuant.md)
- 做完之后：[41. FP8 运行时低精度路径 · 小白入门](./41_FP8_Runtime_Low_Precision_Paths.md)
