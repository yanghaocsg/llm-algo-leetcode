# 01 Triton 向量加法 — 小白讲解

> 对应 notebook：[`01_Triton_Vector_Addition.ipynb`](./01_Triton_Vector_Addition.ipynb)
> NPU 版：[`01_Triton_Vector_Addition_NPU.ipynb`](./01_Triton_Vector_Addition_NPU.ipynb)

## 一句话总结

用最简单的"两个向量相加"，学会 GPU 编程的第一课：怎么把 N 个元素切给一个个并行程序，以及最后不够一整块时怎么用掩码防止越界。

## 为什么需要它：痛点是什么

在 PyTorch 里写 `x + y` 一行就搞定向量加法，但它是别人写好的黑盒。轮到自己写 GPU 内核（kernel，提交到 GPU 执行的计算函数）时，马上会遇到两个 PyTorch 帮你藏起来的问题：

- **谁来干哪一段？** GPU 上有成千上万个可以同时干活的"工人"——Program（同一内核的一份并行执行实例，可以理解成一个工人线程块）。10000 个元素怎么分给它们，才能既不重复、也不遗漏？
- **最后一段不够整块怎么办？** 工人一次固定搬 1024 个元素（块宽 BLOCK_SIZE），如果总长是 1025，最后一个工人只有 1 个真活儿，剩下 1023 个位置是"空气"——直接读写的内存可能根本不属于你，程序就崩了。

这就像搬家：10000 件物品分给若干工人，每人搬一箱（1024 件）。你得给箱子编号、告诉每人"你负责第几箱"，还得叮嘱最后那箱没装满的人："空位别乱塞，也别把别人家的东西搬走。"

## 核心问题

- 全局元素位置怎么由 Program 编号和块内偏移拼出来？
- 尾块（不足一整块的部分）怎么用 mask（掩码，一组布尔条件，控制每个位置是否允许读写）保护读和写？
- 为什么读和写要分别加保护，不能互相替代？

## 解决思路：一步步看懂

1. **切块分工**：块宽记为 B，长度为 N 的向量需要启动 `⌈N/B⌉` 个 Program（向上取整，保证尾部不漏）。这次启动的 Program 集合叫 grid（网格）。比如 N=10、B=4，三个 Program 分别负责位置 0–3、4–7、8–11。
2. **算出全局位置**：Program 编号 p，块内位置 r，全局位置就是 `i = p*B + r`。相邻 Program 起点相差 B，区间互不重叠——这保证了"不重复写"；每个有效位置都能唯一写成这种形式——这保证了"不遗漏"。
3. **造掩码**：`i < N` 的位置才有效。上面例子里第三个 Program 生成 8、9、10、11，有效条件是真、真、假、假。
4. **读写都戴掩码**：

```python
pid = tl.program_id(axis=0)                    # 我是几号工人
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 我负责的全局位置
mask = offsets < n_elements                    # 只有真元素位置才放行
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)     # 读输入，越界位置补 0
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
z = x + y
tl.store(z_ptr + offsets, z, mask=mask)        # 写回也要同一张 mask
```

注意 `other=0.0` 只是把无效读取填成零，让计算不报错；但算出来的值仍然不许写回——"算了"不等于"能写"。只保护写不保护读，仍可能越界读输入；反过来也一样，两者不能互相替代。

Python 启动函数（wrapper）`triton_add(x, y)` 负责检查输入、分配输出、按 `block_size = 1024` 启动 grid，本题不需要改它。

## 关键概念速查表

| 术语 | 大白话解释 |
|---|---|
| kernel（内核） | 提交到 GPU 执行的计算函数 |
| Program | 同一内核的一份并行执行实例，像一个工人 |
| grid | 本次启动的全部 Program 集合，`triton.cdiv(N, B)` 算出需要几个 |
| BLOCK_SIZE | 每个 Program 处理的逻辑块宽（数据规模），不是物理线程数 |
| offsets | 相对指针起点、以元素为单位的偏移 |
| mask（掩码） | 每个位置是否允许读写的布尔条件，如 `offsets < n_elements` |
| `tl.load` / `tl.store` | 带掩码的读 / 写，`other` 指定无效位置的填充值 |
| `@triton.jit` | 把 Python 函数编译成 GPU 代码的装饰器 |
| `tl.constexpr` | 标出编译期已知的参数，本题的块宽就是 |
| 尾块（tail） | 长度不是块宽整数倍时，最后那个不满的块 |

## 小白常见疑问

**Q1：为什么不能直接写一个 for 循环逐元素加？**
for 循环是串行的，一个核一步步算完 100 万个元素；GPU 的意义就是让几千个 Program 同时各算一段。Triton 写的是"一个 Program 干一段"的逻辑，循环被并行替代了。

**Q2：为什么边界必须用严格小于 `<`，不能用 `<=`？**
合法位置是 0 到 `n_elements - 1`。用 `<=` 会把位置 N 也放行，而它是越界的——差一个符号就写出去了。

**Q3：无效位置补零（other=0.0），会不会把结果污染成 0？**
不会。补零只发生在 mask 为假的位置，这些位置的计算结果不会被写回；有效输出的值只由对应的真实输入决定。

**Q4：向上取整和 mask 能互相替代吗？**
不能。向下取整会直接丢掉尾部元素；不取整只加 mask 则可能多启动空 Program。两者配合才是"不遗漏 + 不越界"。

## 动手验证：怎么知道自己做对了

- **正确性**：`test_vector_add` 在长度 1、100、1023、1024、1025、98432（覆盖单元素、不满一块、恰好一块、多块带尾块）和 FP16/FP32 两种类型下，用 `torch.testing.assert_close` 与 PyTorch 的 `x + y` 逐元素对照；同时断言输出的 shape/dtype/device 正确、输入张量没被改动、非法输入（维度不对、类型不支持等）会被拒绝。
- **性能**：`benchmark_vector_add` 用 `triton.testing.do_bench` 预热后重复测量 PyTorch 与 Triton 的完整调用（含输出分配，不含输入搬运和首次编译），输出 `speedup_vs_torch`。

## 与 NPU 版的区别

NPU 版把设备从 `cuda` 换成 `npu`（导入 `torch_npu`），kernel 里的 pid / offsets / mask 这套 Block 模型逻辑与 CUDA 版完全一致，`tl.load`/`tl.store` 的 mask 语义也相同。开头多了一段"昇腾差异提醒"：`num_warps` 在昇腾 Triton 里只是编译提示，不再对应固定 32 线程的 warp；grid 仍代表 Program 实例数，不等于 NPU 核数。核心算法与测试用例（尺寸、dtype）完全相同。

