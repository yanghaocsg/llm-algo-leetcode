# 32. 编译阶段与目标计划 — 小白讲解

> 对应 notebook：[`32_TVM_MLIR_Deep_Practice.ipynb`](./32_TVM_MLIR_Deep_Practice.ipynb)

## 一句话总结
本节不调用 TVM 或 MLIR，而是用普通 Python 字典模拟一条编译流水线：把形状为 `(M, 256, 64)` 的 FP16 矩阵乘，依次做形状绑定、算子合法化、分块调度和计划生成，得到 `(128, 256, 64)`、分块 `(8, 128, 64)`、4 个 Warp 的 `python_plan`。第二问在同一矩阵上筛选 5 个分块候选，第三问证明目标缺少 `mma_fp16` 时计划退回 `(8, 64, 64)`，第四问用阶段快照核对每一步留下的内容是否真的自洽。

## 为什么需要它：痛点是什么
初学者容易以为"形状对、精度对，编译器就该能跑"。真实的编译链上，任何一环信息不足，后面的环节都无法判断：`M` 还是符号 `"M"` 时，没法检查 `128 % 8` 这样的整除；目标只声明支持 FP16 时，BF16 输入在合法化阶段就会被拒；分块 `(64, 256, 64)` 的输入暂存要 40960 B，超过 32768 B 预算就不可行。更隐蔽的是"假成功"：某一步被人手写了一个 `schedule_valid: True`，但里面的分块根本不是这个矩阵能用的——最终计划看起来完整，实际不成立。所以这一节的重点不是"生成代码"，而是**每个阶段留下什么、下一步能检查什么**。

把这些失败按顺序串起来，就是一条很具体的排查链：形状没绑定 → 特化报错；精度不支持 → 合法化报错；分块不整除或暂存超限 → 调度报错；计划里的分块被改过 → 最终核对报错。每个错误消息只说明"这一步缺什么"或"这一步对不上"，不会替你回头补信息——这正是本节要练的定位能力：看到报错，先判断它属于哪一段，再检查上一段留下了什么。

生活类比：装修报批。户型图（形状）没定，施工队排不了工期；材料（精度）不在物业允许清单里，方案会被打回；吊顶方案（分块）超出承重预算（共享内存），再快也不能用；最后拿到的施工许可证（`python_plan`）只证明"这套纸面方案自洽"，不等于房子已经盖好（生成机器码并真正执行）。

## 核心问题
- 一条矩阵乘的编译计划要经过哪几个阶段？每个阶段检查什么、留下什么？
- 为什么符号维度 `M` 必须先绑定成正整数？不绑定会在哪里报错？
- 输入暂存 `2·B_K·(B_M+B_N)` 是怎么算的？为什么 40960 B 被拒？
- 目标缺少矩阵指令能力时，同一个矩阵的计划会怎样变化？
- 怎样证明每个阶段的记录与同一份输入一致，而不是只看一个"已完成"标记？

## 解决思路：一步步看懂

整条流水线可以先用一张表看全貌。每一行都是"前面留下什么、这一步在此基础上新增或检查什么"：

| 阶段 | 本例保留的内容 | 这一步新增或检查什么 |
| --- | --- | --- |
| 输入表示 | 矩阵乘、FP16、形状 `(M, 256, 64)` | `M` 仍是待绑定的符号 |
| 形状绑定 | 矩阵乘、FP16、形状 `(128, 256, 64)` | 以 `M=128` 得到完整正整数维度 |
| 算子合法化 | 相同精度与形状 | 把操作改成本例目标提供的矩阵乘 |
| 分块调度 | 已合法化的矩阵 | 加入 `(8, 128, 64)` 与 4 个 Warp，核对整除规则 |
| 目标计划 | 相同操作、精度、形状与调度 | 返回可继续核对的 Python 计划 |

这张表的读法是"从上往下，信息只增不减"：后一阶段必须能看到前一阶段留下的形状、精度和标记，否则它没有判断依据。下面逐段展开。

**第 1 步：用字典表示 IR，先绑定符号维度。** 本节没有引入任何编译框架，`source_ir = {"op": "matmul", "dtype": "fp16", "shape": ("M", 256, 64)}` 就是一个"最小中间表示（IR，Intermediate Representation）"：只保留算子、精度和形状三个决策字段。注意 `(M, 256, 64)` 的顺序是**问题维度 (M, N, K)**，不是张量的三个轴：左输入是 M×K，右输入是 K×N，输出是 M×N，这里 M 仍是一个符号。

`specialize_shape(graph, bindings)` 做形状特化：

```python
shape = tuple(bindings.get(dim, dim) if type(dim) is str else dim
              for dim in graph["shape"])
if any(type(dim) is not int or dim <= 0 for dim in shape):
    raise ValueError("全部M、N、K维度须绑定为正整数")
return {**graph, "shape": shape, "shape_specialized": True}
```

三个细节值得注意。第一，`type(dim) is str` 用来识别"哪些维还是符号"；第二，`bindings.get(dim, dim)` 对没给绑定的符号原样保留，于是紧接着的类型检查会把它抓出来——传 `bindings={}` 时形状仍是 `("M", 256, 64)`，直接抛 `ValueError`；第三，返回值用 `{**graph, ...}` 生成新字典，所以本问最后能断言 `source_ir` 没有被改动。函数开头还检查了 `bindings` 必须是字典、`graph["shape"]` 必须是长度 3 的元组或列表。

**第 2 步：算子合法化——把源算子换成目标真正提供的算子。** `legalize_ops` 先要求 `shape_specialized is True`，没有这个标记就说明跳过了上一步，直接报错"先完成形状绑定"。随后它又调用了一次 `specialize_shape(graph=graph)`（不传绑定），这不是重复劳动，而是**重新确认形状里的每一维确实都是正整数**——如果有人绕过第一步伪造了标记，这里会失败。接着检查 `op == "matmul"` 且 `dtype == "fp16"`，通过后返回 `op: "example.matmul"`、`legalized: True`。

为什么要把算子改名？因为源 IR 里的 `matmul` 是"数学上要做矩阵乘"，而目标后端可能只提供一个叫 `example.matmul` 的实现，合法化就是把"想做什么"翻译成"目标能做什么"。真实编译器在这一步还会插入布局转换、拆分不支持的算子、处理边界情况；本例只保留"改名 + 检查"这个骨架。反例 `bound | {"dtype": "bf16"}` 会抛 `ValueError`，因为本例目标只声明支持 FP16。

**第 3 步：分块调度——数据切多大、用多少线程束。** `select_schedule` 先要求 `legalized is True`、算子名是 `example.matmul`、精度是 FP16，然后检查分块：

```python
schedule = {"tiles_mnk": (8, 128, 64), "warps": 4} if schedule is None else dict(schedule)
...
if any(dim % tile != 0 for dim, tile in zip(shape, tiles)):
    raise ValueError("形状须满足本例M、N、K整除规则")
```

`tiles_mnk = (B_M, B_N, B_K)` 同样是 (M, N, K) 顺序：`B_M=8` 是输出行块，`B_N=128` 是输出列块，`B_K=64` 是归约维块。三个数都必须是正整数，`warps` 必须在 1 到 8 之间；`(128, 256, 64)` 对 `(8, 128, 64)` 的整除分别是 `128%8=0`、`256%128=0`、`64%64=0`，全部通过。反例把 K 改成 65：`65%64=1`，直接报错。为什么要求整除？因为本例的计划假设**没有尾块掩码**——每个块都完整覆盖数据；真实内核当然可以用 `mask` 处理尾巴，但那是另一套计划，本节不覆盖。

`warps=4` 记录的是"这个分块用 4 个线程束"。notebook 明确区分：**分块大小描述处理多少数据，Warp 数描述线程组织**；一个 Warp 按 NVIDIA 的约定是 32 个线程。这两件事不能混为一谈。

顺带把"程序实例"（Program，沿用 Triton 的说法，指执行一块计算的一次实例）也算清楚。输出是 128×256，按 `(8, 128, 64)` 分块，行方向有 `128/8=16` 块、列方向有 `256/128=2` 块，一共 32 个实例，每个实例覆盖 `8×128=1024` 个输出；归约维 `64/64=1`，每个实例只需一轮。换成 `(8, 64, 64)`，则是 `16×4=64` 个实例、每个覆盖 512 个输出。实例数只是账目，不是耗时模型；候选表里的 `latency_us` 是**给定值**，本节不用实例数去推算它。

**第 4 步：形成最终计划，并重新核对。** `build_backend_plan` 要求 `shape_specialized`、`legalized`、`schedule_valid` 三个标记全为 `True`，然后**用图里记录的分块再跑一遍 `select_schedule`**，并比较重算结果与图里的 `schedule` 是否相等：

```python
checked = select_schedule(graph=graph, schedule=graph["schedule"])
if graph.get("schedule") != checked["schedule"]:
    raise ValueError("计划中的调度与本例规则不一致")
return {**graph, "backend": "example_target", "representation": "python_plan"}
```

这一步是整节的"诚信检查"：前面的布尔标记可能是手写的，只有重算并比对内容，才能确认调度真的属于这个矩阵。比较的是整个 `schedule` 字典；`select_schedule` 会把分块规范成元组再返回，所以如果记录里用列表存分块，类型不同就会被判为不一致。返回的 `representation` 是 `python_plan`，字面意思就是"一份 Python 计划"，不是 CUDA、LLVM IR 或机器码；`build_backend_plan` 不生成任何设备代码。

Q1 的验证里，除了五个正向断言，还列了四个反例：`bindings={}`、`dtype="bf16"`、形状 `(128, 256, 65)`、调度 `(7, 64, 64)`，每一个都必须抛 `ValueError`。反例是用一个循环跑的：

```python
for step, inputs in invalid_steps:
    try:
        step(**inputs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{step.__name__}应拒绝未满足前提的输入")
```

注意 `else` 分支：只有 try 块**没有**抛异常时才会执行。也就是说，如果某个步骤悄悄接受了非法输入，这里会主动失败。这种"正向能过、反向必须拒"的写法，是后面几问的共同套路——只验证成功路径，无法发现检查被写松了。

**第 5 步：Q2——分块的输入暂存怎么算。** 一个分块实例每轮装载 `B_M×B_K` 的左输入块和 `B_K×B_N` 的右输入块。假设两个输入块各在共享内存里保存一份，元素是 FP16、每元素 2 字节，那么暂存字节数为：

$$\text{shared\_bytes} = 2 \times B_K \times (B_M + B_N)$$

括号里的 $B_M+B_N$ 是"左块一行的元素数 + 右块一行的元素数"，乘 $B_K$ 得到两块的总元素数，再乘 2 字节。代入三组：

- `(8, 64, 64)`：$2 \times 64 \times (8+64) = 2 \times 64 \times 72 = 9216$ B；
- `(8, 128, 64)`：$2 \times 64 \times (8+128) = 2 \times 64 \times 136 = 17408$ B；
- `(64, 256, 64)`：$2 \times 64 \times (64+256) = 2 \times 64 \times 320 = 40960$ B。

预算 32768 B。前两个能放下，第三个放不下。notebook 特别说明这个公式**只算输入暂存**，累加器等资源要另行检查；也不要把它当成整个内核的显存占用。

为什么共享内存会成为一个"预算"？因为它是每个线程块（block）在芯片上分到的一小块快速存储，容量有限，而且和全局显存不是一回事——全局显存还剩很多，也不代表某个块能申请到超过上限的共享内存。分块之所以要把输入搬进共享内存，是为了让块内多个线程复用同一份数据，少读几次全局内存；但块越大，这份"暂存"越大，超限时内核根本无法按这个配置启动。这就是"性能更好的分块"必须先过资源门槛的原因。

**第 6 步：Q2——五个候选，四道门槛，按耗时选。** `select_tuning_candidate` 的流程是：先校验每个候选的字段（三维正整数分块、正整数 Warp 数、`required_capabilities` 必须是 `set`、`latency_us` 必须是有限正数），再按固定顺序收集拒绝原因：

```python
if any(dim % tile != 0 for dim, tile in zip(problem_shape, tiles)):
    reasons.append("tile_divisibility")
if shared_bytes > shared_memory_limit:
    reasons.append("shared_memory")
if row["warps"] > max_warps:
    reasons.append("warps")
if (target_capabilities is not None
        and not row["required_capabilities"].issubset(target_capabilities)):
    reasons.append("target_capabilities")
```

最后 `selected = min(feasible, key=lambda row: row["latency_us"], default=None)`。五个候选的结局如下：

| 候选 | 分块 | Warps | 构造耗时 | 结局 |
| --- | --- | ---: | ---: | --- |
| `regular` | (8, 64, 64) | 4 | 80 μs | 可行，9216 B |
| `matrix` | (8, 128, 64) | 4 | 54.7 μs | 可行，17408 B，要求 `mma_fp16` |
| `over_shared` | (64, 256, 64) | 4 | 40 μs | 拒绝：`shared_memory`（40960 > 32768） |
| `not_divisible` | (7, 64, 64) | 4 | 30 μs | 拒绝：`tile_divisibility`（128 % 7 = 2） |
| `over_warps` | (8, 64, 64) | 9 | 20 μs | 拒绝：`warps`（9 > 8） |

读表的关键：**最快的三个候选全部不可行**（20、30、40 μs 都被拒），所以只能在 80 和 54.7 里选，结果 `(8, 128, 64)`。这正是"先筛可行、再比性能"的意义；如果按耗时排序直接取第一个，会拿到一个跑不起来的计划。Q2 还验证了三个边界：预算恰好 9216 时 `regular` 仍可选（用的是 `>` 不是 `>=`），预算 9215 时返回 `None`，候选列表为空时也返回 `None`。

**第 7 步：Q3——目标能力如何改变计划。** `target_capabilities` 是一个集合，函数用 `issubset` 判断候选声明的能力是否都被目标支持。当目标只有 `{"shared_memory", "block_barrier"}` 时，`matrix` 候选还额外要求 `mma_fp16`，于是被拒，理由里出现 `target_capabilities`；此时可行集合只剩 `regular`，选中 `(8, 64, 64)`、80 μs。把 `mma_fp16` 加进目标能力后，`matrix` 重新进入比较集合，以 54.7 μs 胜出。

两次结果的差异**只来自目标能力和候选集合**，矩阵问题 `(128, 256, 64)` 和 FP16 输入完全没变。代码把两次选中的分块分别接回 Q1 的同一个 `legal` 结果，再走 `select_schedule` 和 `build_backend_plan`，就是在证明这一点。

这里用 `issubset` 而不是 `==` 也有讲究：`regular` 只声明需要 `shared_memory` 和 `block_barrier`，当目标额外拥有 `mma_fp16` 时，它仍然应该可用。能力检查问的是"候选要的，目标是不是都有"，不是"两边清单是否一模一样"。notebook 也提醒：这些能力是"具名候选的给定前提"，不代表所有 FP16 矩阵乘都必须走矩阵指令，真实情况要查设备和内核支持。

**第 8 步：Q4——用阶段快照定位首个内容错误。** `validate_pipeline(source, records)` 的输入是源 IR 和按顺序排列的 `(stage, graph)` 记录，阶段名固定为 `["shape_specialize", "legalize_ops", "scheduling", "backend_plan"]`。它做四层检查：

1. 记录条数不等于 4 → `incomplete_pipeline`；
2. 当前记录名与预期阶段名不一致 → `stage_order`；
3. 按 `source` 和上一条记录重算本阶段应有的结果，过程中捕获 `KeyError`、`TypeError`、`ValueError` → `invalid_content`；
4. 重算结果与记录内容不完全相等 → `content_mismatch`。

最后返回 `(True, {"representation": ..., "shape": ..., "schedule": ...})`。几种篡改分别命中不同分支：

| 篡改方式 | 失败原因 | 为什么是这个原因 |
| --- | --- | --- |
| 合法化记录的形状改成 `(64, 256, 64)` | `content_mismatch` | 输入 `bound` 本身合法，重算得到 `(128, 256, 64)`，只是快照写错了 |
| 合法化记录的算子改成 `wrong.matmul` | `content_mismatch` | 重算的 `expected` 仍是 `example.matmul` |
| 调度记录改成 `(7, 64, 64)` | `invalid_content` | `select_schedule` 重算时 `128 % 7 = 2`，直接抛 `ValueError` |
| 最终计划的调度换成 `(8, 64, 64)` | `content_mismatch` | 上一阶段的记录决定重算结果只能是 `(8, 128, 64)` |
| 少一条记录 | `incomplete_pipeline` | 长度检查先失败 |
| 前两条记录对调 | `stage_order` | 第一条记录名就不是 `shape_specialize` |

这里有个容易混淆的点：同样是"调度不对"，为什么有的是 `invalid_content`、有的是 `content_mismatch`？区别在于**错误发生在哪一层**。记录里的调度非法（7 除不尽 128），重算时函数自己就抛异常，属于内容无效；记录里的调度本身合法（8, 64, 64），只是与上一阶段记录推出的结果不一致，属于快照对不上。这比"检查一个 `done=True` 标记"强的地方在于：它**不相信标记，只相信内容**。

**为什么计划里还要记 `backend` 和 `representation`？** `backend` 记录这份计划是为哪个目标生成的（本例是 `example_target`），`representation` 记录它是什么形态（本例是 `python_plan`）。两个字段看起来只是标签，作用却是防止"把计划当成产物"：同一个阶段快照如果接到另一个后端，`backend` 会不同；如果将来真的生成了设备代码，`representation` 就不该再是 `python_plan`。它们让"计划"和"代码"在同一套字典里有明确的身份区分。

还有一处容易忽略的实现细节：`select_schedule` 里的 `dict(schedule)` 和 `select_tuning_candidate` 里的 `{**row, ...}` 都在**复制**输入，而不是原地修改。Q3 会把同一批 `rows` 候选传给两次筛选（受限目标和扩展目标），如果函数改写了候选字典，第二次比较就会看到被污染的数据。复制保证了"筛选"是只读操作，这也是后面能用同一份 `source_ir` 做多次断言的前提。

**把四问连起来看：这份计划在什么意义上成立。** 它成立的范围是"本例的规则"：形状必须能整除分块、精度必须是 FP16、目标能力必须覆盖候选声明、每一步的字典必须能被下一层重算出来。它没有覆盖的东西同样明确：没有内存布局（行优先还是分块布局）、没有指令选择（用几条 MMA、怎么排流水）、没有运行时调度（多少个 block、怎么分给 SM）、也没有任何实测数字。notebook 结尾说"真实实现进入算子优化，运行时兼容性继续在硬件适配组核对"，就是在划这条边界。所以读这份计划时，应该把它当成一张"决策清单"：能告诉你计划在哪一步因为什么失败，不能告诉你最终跑多快。

## 关键概念速查表
| 术语 | 大白话解释 |
| --- | --- |
| IR（中间表示） | 编译过程中保存"要算什么、用什么精度、什么形状"的数据结构，本例是一个字典 |
| Pass（编译遍） | 对 IR 做一次检查或变换的步骤，如形状特化、合法化、调度 |
| 形状特化 / 绑定 | 把符号维度 `"M"` 换成正整数，本例把 `"M"` 绑成 128 |
| 算子合法化 | 把源算子改写成目标后端真正提供的算子，本例是 `matmul → example.matmul` |
| Lowering（逐层降低） | 把抽象运算一步步换成更具体的实现，本节只走到 Python 计划 |
| 分块（tile） | 把大矩阵切成小块处理，`(B_M, B_N, B_K)` 分别是输出行块、输出列块、归约块 |
| Warp（线程束） | NVIDIA GPU 上 32 个线程组成的执行单位，本例用 `warps` 记录一个分块用几个 |
| 共享内存暂存 | 一个分块每轮把两个输入块放进快速内存，字节数 `2·B_K·(B_M+B_N)` |
| 目标能力（capability） | 目标声明支持的特性集合，如 `shared_memory`、`mma_fp16` |
| 构造耗时 | 候选分块的给定代价（微秒），本例用来代替真实基准测量 |
| `python_plan` | 最终计划的表示形式：只是 Python 数据结构，不是设备代码 |
| 尾块掩码 | 形状不能整除分块时用来屏蔽越界位置的机制，本例的计划假设不需要 |

## 小白常见疑问

**Q1：为什么整节都没有 `import tvm` 或 `import mlir`？TVM 和 MLIR 是什么关系？**
因为环境是 CPU-first，本节要讲的是"阶段之间的约束关系"，不是工具链 API。真实的 TVM／MLIR 要安装各自的工具链，并按各自的接口写 pass 或 schedule；本例把决策字段压缩进字典，保证不装任何东西也能复现检查逻辑。也正因如此，输出是 `representation: "python_plan"`，不能当作代码生成结果，更不能拿它测性能。至于两个名字为什么并列：notebook 开头特意说明，TVM 是机器学习编译器框架，MLIR 是组织多层中间表示与变换的编译基础设施，**两者是不同项目**；并列是因为它们共享"多层 IR + 逐层 lowering + 按目标选择实现"的思路，本节借用这套思路组织检查步骤，但既没有装 TVM，也没有用 MLIR 的 pass 管理。

**Q2：`(64, 256, 64)` 只要 40 μs，为什么不选它？**
它被 `shared_memory` 拒绝：暂存 40960 B 超过 32768 B 预算。分块越大，数据复用越好、构造耗时越低，但需要的共享内存也越大；预算是硬门槛，不是可以折算的分数。如果把预算调到 40960，它就会以 40 μs 胜出——同一个候选合不合规，取决于给定预算。

**Q3：Warp 数为什么限定在 1 到 8？**
这是 `select_schedule` 里写死的构造规则（`1 <= warps <= 8`），用来制造 `over_warps` 这个反例，并不是所有 GPU 的通用上限。真实约束取决于架构、每块线程数、寄存器和占用率。

**Q4：形状 `(128, 256, 65)` 只差 1，为什么不行？**
`65 % 64 = 1`，不满足"无尾块掩码"的整除规则。真实内核可以加掩码处理尾块，但那需要另一份计划和额外的边界判断，本例没有覆盖。

**Q5：`build_backend_plan` 跑通，是不是说明能生成 GPU 代码了？**
不是。它只重算并核对调度，返回的是 `python_plan`。真正生成代码要交给 TVM 或 MLIR 工具链；即便生成了，能不能跑、跑多快还取决于硬件、驱动和运行时。

## 动手验证：怎么知道自己做对了
本节没有 `test_*` 函数，也没有 pytest 收集，断言直接写在四个代码单元里。用法就是从上到下依次运行：每个单元先算、再 assert、最后 print，某个 assert 失败会立刻抛 `AssertionError` 并停在那一行，四个单元都跑完不报错就算通过。想定位问题，可以先把该单元的 print 打开，看输出的形状、分块和拒绝理由是否与预期一致。Q1 断言：`bound["shape"] == (128, 256, 64)`、`legal["op"] == "example.matmul"`、`scheduled["schedule"] == {"tiles_mnk": (8, 128, 64), "warps": 4}`、`plan["representation"] == "python_plan"`，并确认四个非法步骤都抛 `ValueError`。Q2 断言可行候选的暂存字节序列是 `[9216, 17408]`、选中 `(8, 128, 64)`、三个拒绝理由依次是 `["shared_memory"]`、`["tile_divisibility"]`、`["warps"]`，以及 9216／9215／空列表三个边界。Q3 断言受限目标是 `(8, 64, 64)`、扩展后是 `(8, 128, 64)`，且 `source_ir` 不变。Q4 断言两种目标生成的记录都能通过，四种篡改、缺记录、乱序全部失败。

notebook 保存了 4 个单元的输出，可以直接对照：Q1 打印 `问题维度(M,N,K)： (128, 256, 64)`、`目标算子： example.matmul ；分块计划： {'tiles_mnk': (8, 128, 64), 'warps': 4}`、`返回表示： python_plan`；Q2 打印两行暂存 `9216 B` 和 `17408 B`，以及 `按构造耗时入选： (8, 128, 64)`；Q3 打印 `{'tiles_mnk': (8, 64, 64), 'warps': 4}` 和 `{'tiles_mnk': (8, 128, 64), 'warps': 4}`；Q4 打印 `阶段内容核对通过： {'representation': 'python_plan', 'shape': (128, 256, 64), 'schedule': {'tiles_mnk': (8, 128, 64), 'warps': 4}}`。如果你手改预算或候选，打印结果应以你自己的断言为准。一个值得亲手做的练习：把 `shared_memory_limit` 改成 9215，预测 `report["selected"]` 会变成 `None`，再跑一遍验证——能提前说清为什么，说明筛选顺序和边界条件都读懂了。
