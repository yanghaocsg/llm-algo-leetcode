# 18. Triton Block 模型 — 小白讲解

> 对应 notebook：[`18_Triton_Block_Model.ipynb`](./18_Triton_Block_Model.ipynb)

## 一句话总结
本节用三个纯 CPU 函数把 Triton 的"块模型"算清楚：`program_coverage` 算一维张量需要多少个 Program、尾块有多少有效元素、多少槽被掩码；`tile_layout` 算二维分块的 Program 网格、覆盖槽数、填充量和有效比例；`program_offsets` 生成某个 Program 的偏移、布尔掩码和有效索引，并验证所有 Program 恰好覆盖输入一次。以 100 个元素按 32 个槽分块为例：需要 4 个 Program，最后一个只有 4 个有效元素、其余 28 个槽被 Mask 掉；100×130 的矩阵按 32×32 分块则覆盖 20,480 个槽，其中 7,480 个是填充。

## 为什么需要它：痛点是什么
初学 Triton 最容易把三个概念混成一个：**Program**（处理一块数据的执行实例）、**Tile/Block**（它覆盖的逻辑数据块）、**线程**（真正在 SM 上跑的硬件执行单元）。于是会以为"`BLOCK_SIZE=1024` 就是启动 1024 个线程""`grid=(4,)` 就是 4 个 SM""尾块多出来的元素会自动消失"。这些理解会导致两个具体错误：一是把逻辑槽数当成线程数去推寄存器或占用率，二是忘了给尾块加 Mask，让越界访问踩坏内存。

生活类比：把 100 份文件装进 32 格一层的文件柜。你需要 4 层柜子（Program），最后一层只放 4 份、剩下 28 格空着（Mask）。"一层 32 格"是**逻辑容量**，不是"32 个管理员"；真正搬文件的人（线程）可能只有几个，每人负责好几格。

## 核心问题
- 100 个元素、每块 32 个槽，为什么需要 4 个 Program 而不是 3.125 个？尾块的有效元素和掩码怎么算？
- `BLOCK_SIZE=1024` 与 `num_warps=4` 同时出现时，1024 和 128 分别是什么？为什么不能说"1024 个线程的 CUDA Block"？
- 100×130 的矩阵按 32×32 分块，为什么覆盖 20,480 个槽而不是 13,000 个？7,480 个填充从哪来？
- 块形状从 16×16 换到 64×64，填充和 Program 数怎么变？"块更大"为什么不等于"更快"？
- 怎样验证所有 Program 的有效索引既不遗漏也不重复？

## 解决思路：一步步看懂

### 第 1 步：Q1——一维覆盖：向上取整与尾块

`program_coverage(num_elements, block_size)` 只做四则运算：

```python
programs = (num_elements + block_size - 1) // block_size
covered_slots = programs * block_size
last_program_valid = num_elements - (programs - 1) * block_size
return {
    "programs": programs, "covered_slots": covered_slots,
    "last_program_valid": last_program_valid,
    "masked_slots": covered_slots - num_elements,
}
```

第一行是"向上取整除法"的标准写法：先加 `block_size - 1` 再整除。100 个元素、32 个槽时，$(100+31)//32 = 131//32 = 4$。为什么不能直接 $100/32 = 3.125$ 取 3？因为 3 个 Program 只能覆盖 96 个元素，剩下 4 个没人处理。向上取整保证覆盖不少于输入。

第二行把 Program 数乘回块大小：$4 \times 32 = 128$ 个逻辑槽。第三行算尾块的有效元素：前 3 个 Program 各覆盖 32 个，共 96 个，最后一块只剩 $100-96=4$ 个。第四行算被掩码的槽：$128-100=28$。断言就是这四个数：

```python
assert coverage == {
    "programs": 4, "covered_slots": 128, "last_program_valid": 4, "masked_slots": 28,
}
```

四个边界用例把"整除"和"差一个"讲透了，`block_size` 固定 32：

| 元素数 | Program 数 | 尾块有效 | 掩码槽 | 说明 |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 31 | 一个元素也要占满一块 |
| 32 | 1 | 32 | 0 | 正好整除，无掩码 |
| 33 | 2 | 1 | 31 | 多一个就得多开一块 |
| 64 | 2 | 32 | 0 | 整除，尾块也满 |

`(33, 2, 1, 31)` 是最容易错的：33 个元素不是"一块半"，而是两块，第二块只有 1 个有效、31 个被屏蔽。`(32, 1, 32, 0)` 则说明整除时 `masked_slots` 是 0，尾块没有浪费。

函数校验两个参数都是正整数（`type(value) is not int or value <= 0`），所以传 0 或浮点会直接抛 `ValueError("输入元素数与块大小须为正整数")`。

Q1 还澄清了一个概念：**一个 Program 处理 1024 个元素、`num_warps=4`，在本节假设的普通 NVIDIA 单 CTA 路径下是 4 个 Warp、128 个线程**（每个 Warp 32 条 lane）。1024 是逻辑元素槽数，128 是线程数，不能把这个逻辑块叫"1024 线程的 CUDA Block"。`grid` 指定的是 Program 实例数，不是 SM 数量；`BLOCK_SIZE` 改数据覆盖，`num_warps` 改执行配置，两者还会影响寄存器、共享内存和指令。本节只验证 Program 数量、尾块有效元素和掩码槽，物理布局与资源要等生成代码和真实内核。

### 第 2 步：Q2——二维分块：网格、覆盖与填充

`tile_layout(rows, cols, block_rows, block_cols)` 把一维的向上取整用到两个方向：

```python
row_programs = (rows + block_rows - 1) // block_rows
col_programs = (cols + block_cols - 1) // block_cols
covered_elements = row_programs * col_programs * block_rows * block_cols
```

注意 `covered_elements` 不是 `rows * cols`，而是"网格数 × 每块面积"。`useful_elements = rows * cols`，`padding_elements = covered - useful`，`useful_ratio = useful / covered`。

以 100×130 的矩阵、32×32 的块为例：

- 行方向：$(100+31)//32 = 4$；列方向：$(130+31)//32 = 5$。网格是 $(4,5)$，共 20 个 Program。
- 覆盖槽数：$4 \times 5 \times 32 \times 32 = 20{,}480$。
- 有效元素：$100 \times 130 = 13{,}000$。
- 填充：$20{,}480 - 13{,}000 = 7{,}480$。
- 有效比例：$13{,}000 / 20{,}480 \approx 0.6348$。

断言把五个字段全部固定：`program_grid == (4, 5)`、`useful_elements == 13_000`、`covered_elements == 20_480`、`padding_elements == 7480`、`useful_ratio == 13_000 / 20_480`。

换成更小和更大的块，结论会变：

| 块形状 | 网格 | 覆盖槽数 | 填充槽 | 有效比例 |
|---|---|---:|---:|---:|
| 16×16 | (7, 9) | 16,128 | 3,128 | 0.806 |
| 32×32 | (4, 5) | 20,480 | 7,480 | 0.635 |
| 64×64 | (2, 3) | 24,576 | 11,576 | 0.529 |

16×16 的网格是 $(100+15)//16=7$、$(130+15)//16=9$，$7 \times 9 \times 256 = 16{,}128$；64×64 是 $(100+63)//64=2$、$(130+63)//64=3$，$2 \times 3 \times 4096 = 24{,}576$，填充 $24{,}576-13{,}000=11{,}576$。断言 `reports[(16, 16)]["covered_elements"] == 16_128` 和 `reports[(64, 64)]["padding_elements"] == 11_576`。

这组数字说明一个反直觉的事实：**块越大，Program 越少，但填充越多**。64×64 只开 6 个 Program，却浪费了 11,576 个槽（占总覆盖的 47%）；16×16 开 63 个 Program，只浪费 3,128 个。但不代表 16×16 更好——Program 多了启动和调度开销，块小了块内复用也少。notebook 不给出"最优块"，只算几何代价，并提醒合并访问、片上复用和资源占用要看具体布局。

两条边界断言也值得注意：`tile_layout(rows=128, cols=128, block_rows=32, block_cols=32)["useful_ratio"] == 1`，整除时没有任何填充；`tile_layout(rows=1, cols=1, block_rows=16, block_cols=16)["padding_elements"] == 255`，一个元素也要占满 16×16 的块，255 个槽全是填充。

函数要求四个参数都是正整数。注释里还有一句重要的话：任意正整数块形状都能参与这个**几何**计算，但不保证该形状能直接编译成 Triton 核函数——真实编译器对块形状、布局和维度有约束，几何上算得通不等于能生成代码。

### 第 3 步：Q3——偏移、Mask 与"恰好覆盖一次"

`program_offsets(program_id, block_size, num_elements)` 先调用 `program_coverage` 拿到 `programs`，再校验 `program_id` 是整数且落在 $[0, \text{programs})$：

```python
offsets = [program_id * block_size + slot for slot in range(block_size)]
mask = [0 <= offset < num_elements for offset in offsets]
valid_offsets = [offset for offset, valid in zip(offsets, mask) if valid]
```

`offsets` 是当前块内每个槽的全局下标；`mask` 是同样长度的布尔列表，下标越界为 `False`；`valid_offsets` 只保留 `True` 的那些。返回值还有 `valid_count` 和 `safe`（所有有效下标都在范围内，按构造必然为真，但仍显式检查一遍）。

尾块是最有代表性的：`program_offsets(program_id=3, block_size=32, num_elements=100)` 的偏移是 $96$ 到 $127$，前 4 个（96、97、98、99）有效，后 28 个越界。断言：

```python
assert tail["offsets"] == list(range(96, 128))
assert tail["mask"] == [True] * 4 + [False] * 28
assert tail["valid_offsets"] == [96, 97, 98, 99]
assert tail["valid_count"] == 4 and tail["safe"]
```

"恰好覆盖一次"是更重要的验证：notebook 对元素数 1、31、32、33、64、100 逐个遍历所有 Program，把所有 `valid_offsets` 拼起来，断言等于 `list(range(size))`：

```python
for size in (1, 31, 32, 33, 64, 100):
    coverage = program_coverage(num_elements=size, block_size=32)
    all_valid = [
        offset
        for program_id in range(coverage["programs"])
        for offset in program_offsets(program_id=program_id, block_size=32,
                                      num_elements=size)["valid_offsets"]
    ]
    assert all_valid == list(range(size))
```

这个断言同时排除了三种错误：**遗漏**（少了一个元素，列表会短）、**重复**（同一个下标出现两次，列表会不等）、**越界**（出现大于等于 `size` 的值，列表会不等）。它比单独检查每个尾块更严格，因为它是全局视角。

还有一条 `program_offsets(program_id=1, block_size=32, num_elements=64)["valid_count"] == 32`，验证整除时中间块也是满的、没有多余掩码。

真实 Triton 里，Mask 不是 Python 布尔列表，而是用 `tl.arange(0, BLOCK_SIZE)` 生成偏移、和边界比较得到布尔张量，再把它传给 `tl.load(..., mask=..., other=0.0)` 和 `tl.store(..., mask=...)`。CPU 计算器只验证逻辑索引；实际实现还要选编译器支持的块形状、把掩码同时用于加载和存储，并核对数值结果与资源用量。只给 load 加 mask、忘了给 store 加，是尾块写坏内存的常见原因。

### 第 4 步：块模型真正关心的是"数据怎么分"，不是"线程怎么排"

把三问串起来，可以看到一条清晰的分层：

1. **数据空间**：`program_coverage` 和 `tile_layout` 只回答"要开多少个 Program、覆盖多少槽、浪费多少槽"。这是纯几何。
2. **索引与边界**：`program_offsets` 回答"每个 Program 负责哪些下标、哪些要屏蔽"。这是逻辑正确性。
3. **执行映射**：一个 Program 内部怎么分给 Warp、每个线程处理几个元素、用多少寄存器/共享内存，由编译器根据 `num_warps`、块形状和布局决定。CPU 计算器完全不碰这一层。

这也是为什么 notebook 反复强调"元素槽不等于线程""逻辑块更大不等于执行更快"。块模型给你的是**可计算的覆盖账本**，不是性能模型。真正的性能还要看合并访问（coalescing）、片上复用、占用率和指令数。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| Program | 处理一块逻辑数据的执行实例，由 Program ID 区分 |
| Program ID | 当前实例在网格里的编号，决定它负责哪一块 |
| Tile / Block | 一个 Program 覆盖的逻辑数据块，如 32 个槽或 32×32 |
| BLOCK_SIZE | 一维块的大小，即每个 Program 的逻辑槽数 |
| grid | 启动的 Program 实例数，不是 SM 数 |
| Mask | 标记哪些槽越界、不该读写的布尔条件 |
| 尾块 | 最后一个不满的块，有效元素可能远少于块大小 |
| 填充（padding） | 二维分块时为了凑齐矩形而多覆盖的槽 |
| 有效比例 | 有效元素除以覆盖槽数，衡量分块的浪费程度 |
| num_warps | 每个 Program 用几个 Warp，属于执行配置 |
| Warp / CTA | 32 条 lane 的执行组 / CUDA 线程块 |
| 合并访问 | 相邻线程读相邻地址，能一次取回更多数据 |

## 小白常见疑问

**Q1：100 个元素、每块 32 个，为什么不能开 3.125 个 Program？**
Program 是整数个执行实例，没有"半个"。3 个只能覆盖 96 个元素，剩下 4 个必须由第 4 个 Program 处理；它只有 4 个有效槽，另外 28 个用 Mask 屏蔽。`programs = (100 + 31) // 32 = 4` 就是向上取整。

**Q2：`BLOCK_SIZE=1024`、`num_warps=4`，到底是 1024 个线程还是 128 个？**
128 个线程。4 个 Warp × 32 lane = 128；1024 是逻辑元素槽数。编译器把 1024 个元素分摊给 128 个线程，每个线程处理 8 个。把 1024 叫"线程数"会高估 8 倍并行度，也会把寄存器估算带偏。

**Q3：尾块那 28 个槽既然不用，为什么还要覆盖？**
因为块大小是编译期常量，所有 Program 用同一套代码和同一块形状。不能给最后一个 Program 单独换一个 4 槽的块（那会多一份特化）。统一按 32 槽算、用 Mask 屏蔽多余部分，代码最简单，代价是 28 个槽的掩码开销。

**Q4：块越大，Program 越少，不是应该更快吗？**
不一定。64×64 只开 6 个 Program，但填充 11,576 个槽，有效比例只有 0.529；16×16 开 63 个 Program，填充只有 3,128 个。块大还可能增加寄存器/共享内存压力、降低占用率。notebook 只算几何覆盖，不给性能结论；哪个快要在真实内核上测。

**Q5：`program_offsets` 返回的 `mask` 和 Triton 里的 mask 是一回事吗？**
逻辑上是一回事，形态不同。这里是 Python 布尔列表，便于在 CPU 上验证；真实 Triton 用 `tl.arange` 和比较生成布尔张量，传给 `tl.load`/`tl.store` 的 `mask` 参数。两者都必须同时用于加载和存储，否则尾块仍可能越界写。

## 动手验证：怎么知道自己做对了

这个 notebook 没有保存执行输出，也没有 TODO 空位，以下结论以断言为准。

- **Q1**：`program_coverage(100, 32)` 必须等于 `{"programs": 4, "covered_slots": 128, "last_program_valid": 4, "masked_slots": 28}`；四个边界 `(1, 32, 31)`、`(32, 32, 0)`、`(33, 1, 31)`、`(64, 32, 0)` 分别对应 `(programs, last_program_valid, masked_slots)`。打印一行"4 个 Program，尾块有效 4 个，总屏蔽槽 28 个"。
- **Q2**：32×32 的五个字段全部断言；`reports[(16, 16)]["covered_elements"] == 16_128`；`reports[(64, 64)]["padding_elements"] == 11_576`；128×128 对 32×32 的 `useful_ratio == 1`；1×1 对 16×16 的 `padding_elements == 255`。打印三种块形状的网格、覆盖和屏蔽槽。
- **Q3**：尾块 `offsets == range(96, 128)`、`mask` 是 4 个 `True` 加 28 个 `False`、`valid_offsets == [96, 97, 98, 99]`；`valid_count == 4`、`safe` 为真；六个元素数的全局覆盖都等于 `range(size)`；`program_offsets(1, 32, 64)["valid_count"] == 32`。打印尾块有效索引和屏蔽槽数。

改参数做对照：把 `block_size` 从 32 改成 16，看 100 个元素的 Program 数是否变成 7、尾块有效是否变成 4、掩码是否变成 12；把矩阵换成 128×128、块保持 32×32，看 `useful_ratio` 是否变成 1；把 `num_elements` 改成 96，看尾块是否恰好满、`masked_slots` 是否为 0。这三组分别验证"向上取整""整除无填充""尾块边界"。
