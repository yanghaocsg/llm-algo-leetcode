# INF-EXECUTION-DISPATCH. Inference Fusion and Backend Dispatch | 推理融合与后端分派 · 小白入门

> 配套 notebook：[INF-EXECUTION-DISPATCH. 推理融合与后端分派](../INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md) ｜ 难度：Medium ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

模型里相邻的几个计算步骤可以合并成一个来执行，但不是每次都行。这本 notebook 教你找出能合并的片段，再判断负责执行的一方接不接受。

## 为什么需要它

模型的前向计算是一串步骤。比如“先做残差相加，再做 RMSNorm”，按部就班的做法是：先把相加结果写回显存，再读出来做归一化，每一步各启动一次内核。步骤一多，“写出去、读回来”和“启动”的次数也跟着多。融合（fusion）的想法是把这几步合成一个内核，相加结果不用写出去，直接接着做归一化。

但“看起来能合并”和“这次真的合并了”是两回事：

- 计算图里出现一段可合并的算子序列，它只是候选。输入形状、数据类型、设备和后端能力共同决定实际路径。
- “后端已安装”和“当前候选可执行”也是两件事。任何一项条件不满足，都可能触发重新编译、换后端或者退回逐个执行。
- 融合不是白拿的好处。它可能减少内核启动和中间张量落地，也可能增加寄存器压力、编译约束和调试成本。

这本 notebook 用几个字典模拟这套判断：从算子序列里识别候选，逐项检查后端能力，再选择执行路径；全部不通过时退回逐个执行，并把原因记下来。它的前置阅读是 20. FlashAttention 模拟和 Part 01 的算子融合导论。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 算子（operator） | 模型计算里的一个步骤，比如残差相加、RMSNorm、SiLU。代码里是一个字符串，如 `'rms_norm'` |
| 计算图（graph） | 用节点表示算子、用边表示张量依赖的一张图 |
| 即时执行（eager） | 每遇到一个算子就立刻执行一个，不做合并。所有后端都不接受时就退回这里 |
| 内核（kernel） | 真正在硬件上跑一段计算的程序。融合之后，一个内核完成原来几个算子的工作 |
| 后端（backend） | 负责接收并执行这段计算的实现。本节用一个字典描述它支持什么 |
| 模式（pattern） | 一段固定的算子序列，比如 `('residual_add', 'rms_norm')`。模式表里一共 4 个 |
| 回退（fallback） | 所有后端都不接受候选时，退回即时执行，并保留每个后端的拒绝原因 |

## 核心思路

分派分三步。输入是有序算子列表、工作负载（workload）字典和后端列表：

```text
算子序列 → ① 识别融合候选 → ② 对每个后端逐项检查五类门槛 → ③ 取第一个通过的后端
                                                    全部不通过 → 回退 eager，并记录拒绝原因
```

第 ① 步是扫描。在每个位置依次尝试各个模式，看从当前位置开始的一段是否和某个模式完全一致。匹配上的这一段叫融合候选（candidate），记下 `start` 和 `end`，游标跳到候选末尾继续；没匹配上就右移一位。模式要按长度从长到短排，否则短模式如果恰好是长模式的开头，会先把节点消费掉，长模式就再也对不上了。

下面位置 1 开始是三个算子 `linear_gate`、`silu`、`multiply`。为了看清顺序的影响，这里多假设了一个只有一个算子的短模式（notebook 的模式表里没有它）：

```python
patterns = {('linear_gate',): 'short', ('linear_gate', 'silu', 'multiply'): 'swiglu'}
ops = ['embedding', 'linear_gate', 'silu', 'multiply']
by_len = lambda item: len(item[0])      # 按模式长度排序用的关键字

def scan(ordered_patterns):
    found, index = [], 0
    while index < len(ops):
        matched = None
        for pattern, name in ordered_patterns:
            end = index + len(pattern)
            if tuple(ops[index:end]) == pattern:
                matched = (name, index, end)
                break
        if matched is None:
            index += 1          # 没匹配上：右移一位
        else:
            found.append(matched)
            index = end         # 命中：游标跳到候选末尾
    return found

print(scan(sorted(patterns.items(), key=by_len)))
print(scan(sorted(patterns.items(), key=by_len, reverse=True)))
```

```text
[('short', 1, 2)]
[('swiglu', 1, 4)]
```

每个元组是（模式名，起点，终点），终点不含在内。位置 0 的 `embedding` 不匹配任何模式，直接跳过。短模式先试时，它在位置 1 就匹配上，游标跳到 2，后面的 `silu`、`multiply` 谁也对不上，只得到一个只覆盖 1 个算子的候选。长模式先试时，三个算子合成一个候选，覆盖位置 1、2、3。

第 ② 步是能力检查。下面前两行是测试里的一组取值，后面是五类门槛：各用一个变量判断，哪一项没过就记一个原因。只要记下一个原因，这个后端就不支持该候选。

```text
workload = {'dtype': 'float16', 'device': 'cuda', 'seq_len': 2048, 'dynamic_shape': False}
backend  = {'name': 'compiled', 'patterns': ['swiglu'], 'dtypes': ['float16'], 'devices': ['cuda'],
            'max_seq_len': 4096, 'supports_dynamic_shape': False}

候选模式  在 backend['patterns'] 里       否则记 unsupported_pattern
dtype    在 backend['dtypes'] 里          否则记 unsupported_dtype
device   在 backend['devices'] 里         否则记 unsupported_device
seq_len  不超过 backend['max_seq_len']    否则记 sequence_too_long
动态形状   为 False，或后端显式支持         否则记 dynamic_shape_not_supported
```

第 ③ 步是选择。后端列表的顺序就是优先级：取第一个 `supported` 为真的报告。一个都没有时，`execution_path` 记为 `eager`，`decision` 记为 `fallback`，并保留每个后端的拒绝原因。函数返回的是按能力字典推导出来的选择，不是实际的编译或内核命中记录；真实路径要到 Step 5 在 GPU 上核对。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从模型图到实际执行路径，候选为什么要过后端的能力门槛 | 读 |
| Step 2 | 哪些模式值得形成融合候选，融合的收益和代价 | 读 |
| Step 3 | 模式、数据类型、设备、形状、动态形状五类门槛怎样决定命中与回退 | 读，对照上面的五类门槛 |
| Step 4 · TODO 1 | 让长模式先参与匹配，避免短模式提前消费节点 | 在 `identify_fusion_candidates` 里写 1 行：把 `FUSION_PATTERNS.items()` 按模式长度降序排成 `ordered_patterns` |
| Step 4 · TODO 2 | 每个变量只判断一种后端约束 | 在 `evaluate_backend_capability` 里写 5 行：`pattern_supported` 到 `dynamic_shape_supported`，每个只查一项 |
| Step 4 · TODO 3 | 按列表优先级选择第一个 supported 报告 | 在 `dispatch_candidate` 里写 1 行：从 `capability_reports` 里取第一个 `supported` 的报告，没有就是 `None` |
| 测试 | 检查长短模式竞争与不重叠消费、每类单项能力失败的原因、后端优先级与即时执行回退 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5（可选） | 在 GPU 上改变 Token 数，对同一段 Residual + RMSNorm + SwiGLU 计算做即时执行和 `torch.compile` 的配对测量 | 需要 `INF-CUDA-MECHANISM` 环境；默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验 |

Step 1–4 只需要 CPU，测试也不依赖 GPU 或具体编译后端。Step 5 需要 GPU，默认是 `RUN_MODE = 'dry_run'`，不执行真实 GPU 实验。

## 容易踩的坑

- **短模式先匹配**：短模式如果是长模式的开头，会先消费掉节点。要把模式按长度降序排列，再从长到短依次尝试。
- **一个节点被两个候选重复使用**：候选之间不能重叠。命中后游标要移到候选末尾，而不是只前进一位。
- **一个变量里混着几种判断**：五类门槛各用一个变量判断。比如 `dtype` 不支持就只记 `unsupported_dtype`，失败时才能定位。
- **把动态形状一律判为不支持**：静态形状始终通过；动态形状只有在后端没有显式支持时才判失败。
- **全部被拒时丢掉了原因**：要选第一个通过的后端。全部失败时必须回退即时执行，并把每个后端的拒绝原因留在 `rejected_backends` 里。

## 读完后自查

- [ ] 能说出计算图里出现一段可融合的模式时，为什么它只是候选。
- [ ] 给一段算子序列和模式表，能手写出每个候选的 `start` 和 `end`。
- [ ] 能说出五类能力门槛各查 `workload` 和 `backend` 的哪个字段。
- [ ] 能解释后端全部拒绝时函数返回什么，为什么还要保留拒绝原因。

## 下一步

- 打开 notebook：[INF-EXECUTION-DISPATCH. 推理融合与后端分派](../INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)
- 做完之后：[21. Decoding Strategies｜解码策略 · 小白入门](./21_Decoding_Strategies.md)
