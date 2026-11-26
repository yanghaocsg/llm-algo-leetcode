# 19. Triton kernel 接入 PyTorch（autograd 与 torch.compile）— 小白讲解

> 对应 notebook：[`19_Triton_PyTorch_Op_Integration.ipynb`](./19_Triton_PyTorch_Op_Integration.ipynb)

## 一句话总结
解决"kernel 单测算对了，怎么接进训练流程"的工程问题：用 `torch.library.triton_op` 把 Triton kernel 注册成可被 autograd 和 torch.compile 理解的算子，先做一个缩放算子讲清"保存什么、梯度怎么传"，再把已有的 Softmax 前反向封装成完整算子。

## 为什么需要它：痛点是什么
会写 Triton kernel 只完成了一半：训练时 PyTorch 的自动求导（autograd）要能对它求梯度，torch.compile 要能把它整图捕获。裸调用 kernel 两者都做不到——框架根本不知道这个函数的前向保存了什么、反向该怎么算。这就像把一台自制设备接进工厂流水线：设备本身能转（kernel 算得对）还不够，还得提供标准的进料口和出料口（前向/反向边界、可追踪的调用方式），否则整条线（训练图）到它这就断了。

## 核心问题
- 前向要为反向保存哪些值？保存谁（输入、系数还是输出）由什么决定？
- 上游梯度（grad_output）怎样进入反向 kernel？切片、广播出来的梯度怎么办？
- `custom_op` 和 `triton_op` 两种封装边界有什么区别？

## 解决思路：一步步看懂
1. **前向的两项职责**：算出输出，同时保存反向需要的状态。保存什么由导数公式决定：缩放 $y_i=\alpha x_i$ 的输入梯度是 $\alpha g_i$，只需保存同一次前向的系数 α（例：α=0.5、g=(-2,0,3)，输入梯度 (-1,0,1.5)）；Softmax 的输入梯度依赖本次输出概率 y，所以 `save_for_backward` 保存的是 y 而不是原始 logits。
2. **Softmax 反向公式**：$\frac{\partial L}{\partial x_i}=y_i\left(g_i-\sum_j y_jg_j\right)$。文字说：先算整行点积 $d=\sum_j y_jg_j$，再用每列概率乘（上游梯度减 d）。例：y=(0.25,0.75)、g=(-1,2)，d=1.25，输入梯度 (-0.5625, 0.5625)。漏掉行点积、或把保存的 logits 当概率用，结果都会变。
3. **算子边界**：`custom_op` 提供"不透明"边界（编译器看不见内部）；`triton_op` 让输出分配和 Triton 调用可被追踪（traceable），配合 `wrap_triton` 启动内核，torch.compile 才能整图调用：
```python
@triton_op("part03_lesson19::scale", mutates_args={})
def triton_scale(x: torch.Tensor, alpha: float) -> torch.Tensor:
    ...
    wrap_triton(scale_kernel)[grid](x, output, x.numel(), alpha, BLOCK=block)
```
4. **布局适配**：上游梯度可能来自切片（列步长大于 1）或广播（expand 出现零步长），而反向内核按连续行寻址，所以封装层先把概率与梯度各自 `contiguous()` 再启动 `row_softmax_backward_kernel`。已经连续的张量不会复制，只有非连续或广播视图才真正整理一份。
5. **五个机制点**对应 TODO 1–5：kernel 里 `y = alpha * x`；`setup_context` 保存 inputs 里的 α；`backward` 返回 `grad_output * ctx.alpha`（Python 标量系数的梯度返回 None）；Softmax 的 `softmax_setup_context` 保存当前输出 y；`softmax_backward` 从 saved_tensors 取 y，与实际梯度一起交给封装好的 `triton_row_softmax_grad`。
6. **版本兼容**：`triton_op` 与 `wrap_triton` 由较新的 PyTorch 提供（本文按 2.11 的接口书写）。导入单元先尝试这两个接口；旧版本导入失败时，自动改用 `custom_op` 建立不透明边界，并登记与首个张量输入同形的 Fake 实现，让 `torch.compile` 仍能整图追踪。两条路径的内核、autograd 注册和 TODO 完全一样，测试报告的 `backend` 字段记录本次实际走的是哪条。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| autograd | PyTorch 的自动求导机制，反向沿计算图传梯度 |
| 上游梯度 grad_output | 损失函数传回来的 ∂L/∂y，反向的输入 |
| save_for_backward | 前向为反向暂存张量的标准接口 |
| setup_context | 注册算子里"前向结束时保存状态"的钩子 |
| triton_op | torch.library 提供的可追踪 Triton 算子封装 |
| wrap_triton | 让 Triton kernel 调用可被编译器追踪的包装器 |
| custom_op | 不透明算子边界，编译器不分析内部；旧版 PyTorch 用它兼容 triton_op |
| Fake 实现 | 追踪时只返回同形元数据的替身实现，不真正启动内核 |
| 别名（alias）| 共享底层存储的张量；算子须声明是否修改输入 |
| 零步长 / 大步长 | 广播和切片视图的典型布局，反向前要连续化 |

## 小白常见疑问
- **Q：为什么保存输出概率 y 而不保存输入 logits？** 反向公式里的 $y_i$、$y_j$ 都是概率；保存 logits 还要在反向里重算一次 softmax，且直接"把 logits 当概率用"是错的。
- **Q：梯度为什么必须先 contiguous()？** 反向内核假定每行元素在存储里挨着；切片视图列步长是 2、广播视图步长是 0，直接按连续寻址会读错元素。
- **Q：标量系数 alpha 的梯度为什么返回 None？** 它是 Python 标量不是张量，PyTorch 不对标量输入求梯度；`backward` 只需返回各张量输入的梯度。
- **Q：为什么旧版 PyTorch 也能跑？** 旧版没有 `triton_op`/`wrap_triton`，导入单元改用 `custom_op` 加 Fake 实现；这相当于把整段前反向当成一个不透明算子交给编译器，教学用的内核、保存规则和梯度核对都不受影响。

## 动手验证：怎么知道自己做对了
缩放：五档长度 × 四种系数（含零、负、单位系数）与参考对照，正负零系数下的梯度分别核对；另用 CPU 上下文对象检查 `setup_context` 保存的就是本轮系数、`backward` 的第二项返回 None。Softmax：用 CPU 上下文对象核对 `save_for_backward` 保存的就是当前输出；六档普通形状与两档整图（torch.compile）形状对照 PyTorch 自动求导；三种上游梯度——真实切片、expand 常量广播、全零——分别验证布局适配（常量梯度应得到接近零的输入梯度）；反向封装还会拒绝形状或类型不匹配的上游梯度。上下文对象核对用 CPU 张量完成，设备内核与整图编译由 GPU 用例验证。
