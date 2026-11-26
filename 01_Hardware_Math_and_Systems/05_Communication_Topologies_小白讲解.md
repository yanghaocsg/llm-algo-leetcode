# 05. 通信拓扑与分布式基石 — 小白讲解

> 对应 notebook：[`05_Communication_Topologies.ipynb`](./05_Communication_Topologies.ipynb)

## 一句话总结
大模型从单卡扩到多卡，DP、TP、PP 切的东西不同，通信也就不同。本节用四个计算器建立分类关系：`three_d_parallel` 说明三种并行相乘得到 worker 数（8、16、32），并各自对应梯度同步、激活收集/规约、阶段激活传输；`collective_shape` 在 world size 4 下推出 All-Reduce 形状不变、All-Gather 从 `(8,1024)` 变 `(32,1024)`、Reduce-Scatter 从 `(32,1024)` 变 `(8,1024)`；`path_transfer_lower_bound` 算出 256 MiB 在 100 GiB/s 机内链路上要 2.5 ms，跨节点被 12.5 GiB/s 卡到 20 ms；`check_group_placement` 按规则检查 TP 机内、EP 被放到跨节点、DP 两者皆可这三种实际放置是否合规。

## 为什么需要它：痛点是什么
初学者常把"多卡"理解成"同样的活分给更多人，肯定更快"，于是忽略了每加一种切分就多一种数据交换。数据并行要同步梯度，张量并行每层都要交换激活，流水线并行要在阶段之间传激活并等待。通信量和通信频率不同，对链路的要求也完全不同：把高频的 TP 通信放到跨节点慢链路上，算得再快也会被等回来。

另一个困惑是集合通信的形状。All-Reduce、All-Gather、Reduce-Scatter 的名字很像，每个 rank 的输入输出到底怎么变，初学时容易混。还有单位问题：把 Gb/s 当成 GiB/s 直接代入，算出的时间会差约 8.59 倍。

生活类比：餐厅后厨。DP 像开几家分店，每家做不同客人点的整套菜，晚上统一核对库存（梯度同步）；TP 像把一道菜拆成切配和掌勺，两个人不停递半成品（高频、低延迟）；PP 像流水线，前一道工序做完把盘子递给下一站（点到点、有等待）。分店之间一天对一次账就行，切配和掌勺却几乎每秒钟都在沟通，所以它们该待在同一间厨房里。

## 核心问题
- DP、TP、PP 分别切分什么对象？各自主要交换什么数据、常用什么通信方式？
- 三种并行组合成 3D 并行时，worker 数为什么是乘积而不是和？
- 在 world size 4 下，All-Reduce、All-Gather、Reduce-Scatter 每个 rank 的输入输出形状怎样变？
- 同一份 256 MiB 消息，机内 100 GiB/s 和跨节点 12.5 GiB/s 的传输下界差多少？
- 怎样检查一个并行组是否落在允许的放置域？"位置类别"为什么不等于带宽或时延等级？

## 解决思路：一步步看懂

### 第 1 步：Q1 `three_d_parallel`——先分清三种切分
三种并行的定义和通信关注点：

| 并行方式 | 切分对象 | 主要交换的数据 | 常见通信方式 |
|:---|:---|:---|:---|
| DP（数据并行） | 不同数据批次 | 梯度或参数更新结果 | 常见 All-Reduce |
| TP（张量并行） | 层内张量或激活 | 局部激活、局部结果 | All-Gather、Reduce-Scatter 或 All-Reduce |
| PP（流水线并行） | 不同层或阶段 | 相邻阶段的激活 | 点到点传输，伴随流水线等待 |

DP 里每张卡有完整模型、吃不同 batch，反向算完要把各卡梯度合成一致，所以是梯度同步；TP 把一层里的大矩阵切到多卡，前向/反向中间要拼回或规约，通信频率最高；PP 把层分段，只相邻阶段传激活，但会有流水线气泡（等待）。

用一个前向/反向的例子把三者串起来：DP 的第 $i$ 张卡用第 $i$ 份数据独立跑完整前向，反向得到本地梯度，最后一步做 All-Reduce 把梯度求平均；以常见的列并行线性层为例，TP 把权重按列切开，每张卡算出部分输出，前向用 All-Gather 把部分结果拼成完整激活，反向可能需要 Reduce-Scatter 把梯度按列归还；PP 的第 $k$ 段算完后，把激活用点到点发送给第 $k+1$ 段，自己等下一批数据，反向再按相反方向传梯度。三者的通信对象、频率和同步点都不同，所以不能用一个"通信量"概括。

函数把三个维度组合起来：

```python
return {
    'dp_groups': dp, 'tp_shards': tp, 'pp_stages': pp,
    'effective_workers': dp * tp * pp,
    'partition_axis': {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'},
    'communication_focus': {
        'dp': 'gradient synchronization',
        'tp': 'activation gather / reduce-scatter',
        'pp': 'stage activation transfer',
    },
}
```

`effective_workers` 是乘积，因为三个维度正交：8 路数据并行、每组 2 路张量并行、2 个流水线阶段，一共 $8\times2\times2=16$ 个 worker。notebook 的三个构造是 `(8,1,1)` → 8、`(4,2,2)` → 16、`(2,4,4)` → 32。输入校验要求三者都是正整数。

为什么是乘积而不是相加？每个 worker 可以用一个三元组 $(\text{dp\_rank},\text{tp\_rank},\text{pp\_rank})$ 唯一标识：dp_rank 取 0 到 7，tp_rank 取 0 或 1，pp_rank 取 0 或 1，组合起来正好 $8\times2\times2=16$ 种。你可以把它想成一个三维网格，每个格点是一张卡；数据并行是沿一个轴复制模型，张量并行是沿另一个轴切一层，流水线是沿第三个轴切层。三个轴都独立，所以总数相乘。

注意"3D 并行"不是说三个维度都要大于 1，`(8,1,1)` 也是合法的 3D 配置，只是 TP 和 PP 退化成 1。它的目标不是"切得越多越好"，而是让模型大小、单卡算力和拓扑带宽匹配：TP 通信最频繁，应尽量留在机内；DP 通信可以跨节点；PP 介于两者之间。`(2,4,4)` 的 32 个 worker 里，每个 TP 组 4 卡，每个 PP 阶段有 $dp\times tp=2\times4=8$ 个 worker，通信模式会和 `(8,1,1)` 完全不同，不能只比 worker 数。

`partition_axis` 和 `communication_focus` 两个字典是把"切什么"和"传什么"绑在一起的索引：看到 `'dp': 'data'` 就知道它切数据，看到 `'dp': 'gradient synchronization'` 就知道它主要传梯度。这三个字符串也是后面选择集合通信原语的依据：DP 常用 All-Reduce，TP 可能用 All-Gather/Reduce-Scatter，PP 用点到点。

### 第 2 步：Q2 `collective_shape`——每个 rank 的输入输出怎么变
集合通信要求一组进程一起参加。进程编号叫 rank，参与的进程数叫 world size。notebook 固定 4 个 rank，并强调代码里的形状始终指"每个 rank 实际传给接口的输入"，不在不同原语之间偷换含义。

| 原语 | 每 rank 输入 | 每 rank 输出 | 数据变化 |
|:---|:---:|:---:|:---|
| All-Reduce | `(8, 1024)` | `(8, 1024)` | 逐元素归约，每个 rank 拿到相同的完整结果 |
| All-Gather | `(8, 1024)` | `(32, 1024)` | 沿第 0 维拼接 4 个 rank 的分片 |
| Reduce-Scatter | `(32, 1024)` | `(8, 1024)` | 先归约完整输入，再沿第 0 维等分成 4 份 |

代码：

```python
if kind == 'allreduce':
    output_shape = shape
elif kind == 'allgather':
    output_shape = (shape[0] * world_size, *shape[1:])
elif kind == 'reducescatter':
    if shape[0] % world_size != 0:
        raise ValueError('Reduce-Scatter的第0维须能被进程数整除')
    output_shape = (shape[0] // world_size, *shape[1:])
```

All-Reduce 是"人人有完整输入、人人拿完整结果"，形状不变。All-Gather 是"人人有一片、拼起来给每个人"，第 0 维从 8 变成 $8\times4=32$。Reduce-Scatter 是"人人有完整输入、先求和再各拿一片"，第 0 维从 32 变成 $32/4=8$。所有切分/拼接都沿第 0 维，其它维度不变。

元素数量可以手算：输入 `(8,1024)` 有 8192 个元素，All-Gather 后每个 rank 拿到 32768 个元素；Reduce-Scatter 的输入是 32768 个，归约后每个 rank 拿 8192 个。notebook 没有给 dtype，所以只谈形状和元素数，不换算字节。Reduce-Scatter 还要求第 0 维能被 world size 整除，否则无法等分；未知原语名也会抛 `ValueError`。

把"每个 rank"换成"所有 rank 合计"，能看清复制和切分的区别：

- All-Gather：每个 rank 输入 8192，输出 32768；4 个 rank 合计输出 $4\times32768=131072$，是输入总量的 4 倍。数据被复制了 4 份，每个人都拿到完整结果。
- All-Reduce：每个 rank 输入 8192，输出 8192；合计输出 32768，和输入总量相同。归约后大家拿到相同的完整结果，没有额外复制。
- Reduce-Scatter：每个 rank 输入 32768，输出 8192；合计输入 131072，合计输出 32768，后者是前者的四分之一。完整输入被归约并切成 4 份，每人只保留一份。

这个"合计"视角在估算显存和带宽时很有用：All-Gather 之后每个 rank 都持有完整张量，内存占用会按 world size 放大；All-Reduce 的合计输出与输入总量相同；Reduce-Scatter 的合计输出是输入总量的 $1/\text{world\_size}$，每人只保留一片。通信过程中仍可能产生临时缓冲。notebook 只给形状，不给字节，是因为字节还要乘 dtype；换成 FP16，8192 个元素就是 16 KiB，32768 个就是 64 KiB，你可以按自己的 dtype 继续算。

还要记住"每 rank 输入"这个约定。notebook 说代码里的形状始终指每个 rank 实际传给接口的输入，不在不同原语间改变含义。如果你把 `(8,1024)` 当成"全局张量"，就会误以为 All-Gather 后是 `(8,4096)`；实际上全局是由 4 个 rank 的输入拼起来的，每人才拿 8 行，拼成 32 行。同理，Reduce-Scatter 的 `(32,1024)` 是每个 rank 都持有的完整输入，不是全局总量；归约后每人拿 8 行。把"每 rank 视角"和"全局视角"分清，形状就不会推错。

### 第 3 步：Q3 `path_transfer_lower_bound`——传输下界由最窄链路决定
函数统一用 MiB 表示载荷、GiB/s 表示单向链路带宽，取所有链路中带宽最小的一条作为瓶颈：

```python
bottleneck = min(links, key=lambda link: link['bandwidth_gib_s'])
lower_bound_ms = payload_mib / 1024 / bottleneck['bandwidth_gib_s'] * 1000
```

`payload_mib / 1024` 把 MiB 换成 GiB，再除以 GiB/s 得到秒，乘 1000 换成毫秒。

256 MiB 等于 0.25 GiB。机内 100 GiB/s：$0.25/100\times1000=2.5$ ms。跨节点路径有两跳：GPU 到网卡 50 GiB/s、节点间网络 12.5 GiB/s，瓶颈是 12.5，时间 $0.25/12.5\times1000=20$ ms。断言正是 `intra_node['lower_bound_ms'] == 2.5`、`cross_node['lower_bound_ms'] == 20.0`、`cross_node['bottleneck_link'] == 'network'`。

这个 20 ms 是"理想持续吞吐下界"：假设数据能像流水一样连续传输，只受最窄链路限制，没有计入各跳启动、转发延迟、协议开销和共享路径拥塞。真实跨节点时间通常更高，所以它适合做"至少不会低于多少"的判断，不适合当精确预测。

也正因为只算带宽，这个下界对大消息更有意义；消息很小时，固定延迟（每跳的启动、协议握手、软件栈）会占主导，带宽再宽也快不过延迟。所以看到"20 ms"不要直接推断小控制消息也要 20 ms；要按消息大小分别看：大消息接近带宽下界，小消息看延迟。notebook 的 `payload_mib=256` 是一个偏大的教学载荷，正是为了让带宽项显眼。

单位换算也要小心。规格常写 Gb/s（小写 b 是 bit），而函数要 GiB/s（大写 B 是 Byte）。1 GiB/s 约等于 8.59 Gb/s，因为 1 GiB = $2^{30}$ 字节、1 Gb = $10^9$ bit。把 100 Gb/s 直接当 100 GiB/s 代入，会把时间算小约 8.59 倍。notebook 明确说"二者不能直接代入同一公式"。

把 2.5 ms 的推导完整写一遍：256 MiB $=256/1024=0.25$ GiB；链路 100 GiB/s；$0.25/100=0.0025$ 秒 $=2.5$ 毫秒。如果你用十进制换算，100 GiB/s 约等于 $100\times2^{30}/10^9\approx107.37$ GB/s；256 MiB 是 $256\times2^{20}=268{,}435{,}456$ B；$268{,}435{,}456/107.37\times10^9\approx0.0025$ 秒，仍然是 2.5 ms。两种算法一致，前提是单位配套：MiB 配 GiB/s，MB 配 GB/s，不能混。

反过来，如果规格写的是 100 Gb/s，先换成 GiB/s：$100/8.59\approx11.64$ GiB/s，256 MiB 需要 $0.25/11.64\times1000\approx21.5$ ms，而不是 2.5 ms。这个差距足以改变"跨节点是否可接受"的判断，所以看到带宽数字第一件事是确认大小写 b 和 B。

### 第 4 步：Q4 `check_group_placement`——规则检查和性能判断分开
最后一步不是算时间，而是检查并行组有没有落在任务允许的放置域。函数只认两个域：`intra_node`（机内）和 `cross_node`（跨节点）。

```python
allowed = set(group['allowed_domains'])
if not allowed or not allowed <= domains or group['actual_domain'] not in domains:
    raise ValueError('使用明确的机内或跨节点放置域')
...
'placement_allowed': group['actual_domain'] in allowed
```

notebook 的三个组：

| 组 | 允许域 | 实际域 | 是否符合 |
|:---|:---|:---|:---:|
| TP group | `intra_node` | `intra_node` | 是 |
| EP group | `intra_node` | `cross_node` | 否 |
| DP group | `intra_node`、`cross_node` | `cross_node` | 是 |

所以 `[row['placement_allowed'] for row in reports] == [True, False, True]`。这里 EP（专家并行）在 Q4 的规则表里出现，是因为它常需要跨节点交换 token（All-to-All），规则是否允许、负载是否承受得住要单独检查；但 notebook 的这个构造把 EP 组限制在机内，实际却放到了跨节点，于是判为不符合。

最重要的提醒是：`placement_allowed` 只是"符不符合这份任务的规则"，不是性能结论。位置类别不代表带宽或时延等级，跨节点路径也可能比机内某条拥塞链路更快；要比较成本，得带着消息大小和调用频率进入第 26 节，用实际链路数据判断。高频通信（如 TP）更需要关注带宽、时延和共享流量。

函数里的校验也值得注意：`allowed_domains` 不能为空，且必须是 `{'intra_node','cross_node'}` 的子集；`actual_domain` 必须是这两个之一。这防止你把 `'same_rack'`、`'nvlink'` 这种更细的类别直接塞进来，却不定义它的语义。所有组都通过后，报告里的 `allowed_domains` 用 `sorted` 输出，是为了让结果稳定、方便比较，不改变集合含义。

规则本身是任务给定的，不是拓扑的固有属性。同一个 DP 组，在一个任务里可能只允许机内，在另一个任务里允许跨节点；`placement_allowed=False` 表示当前配置违反了这份规则，应该停下来调整，而不是"先跑起来再说"。反过来说，`True` 也不代表性能达标，只代表没有越界。把规则检查和成本比较分成两步，才不会把"合规"误当成"最优"。

这四问的模型都有明确边界，读结果时别越界使用：

- `three_d_parallel` 只做分类和计数，不估算每种并行的通信量、气泡比例或扩展效率。
- `collective_shape` 只给形状，不给字节、时延、带宽或算法步骤；All-Reduce 具体用 ring 还是 tree、走几步，要到第 20 节。
- `path_transfer_lower_bound` 只取最窄链路的稳态下界，忽略启动、转发、协议、拥塞和共享流量；真实时间通常更高。
- `check_group_placement` 只检查任务规则，不测链路质量；跨节点也可能比机内快，必须按实际数据判断。

所以本节适合回答"这种切分大概要什么通信、消息形状怎么变、链路下界大概多少、放置有没有越界"，不适合回答"最终加速比是多少"。后者要固定模型、并行配置、消息大小和调用频率，进入第 26、27 节做实测或更细的调度模型。

## 关键概念速查表
| 术语 | 大白话解释 |
|:---|:---|
| DP（数据并行） | 每张卡跑完整模型、吃不同数据，最后同步梯度 |
| TP（张量并行） | 把一层内的大张量切到多卡，合作算同一层 |
| PP（流水线并行） | 把不同层分到不同阶段，像流水线一样传激活 |
| 3D 并行 | DP、TP、PP 三个维度组合，worker 数是三者乘积 |
| rank / world size | 进程编号 / 参与通信的进程总数 |
| All-Reduce | 所有 rank 归约，人人都拿到完整结果 |
| All-Gather | 收集各 rank 的分片，拼成完整结果发给每个人 |
| Reduce-Scatter | 先对完整输入归约，再把结果等分给各 rank |
| 传输下界 | 只按最窄链路带宽算出的最短时间，不含启动和拥塞 |
| MiB / GiB/s | 二进制字节单位 / 每秒 GiB 的单向带宽 |
| 放置域 | 并行组被允许放在机内还是跨节点 |
| EP（专家并行） | MoE 中不同专家分到不同设备，常需 All-to-All 换 token |

## 小白常见疑问

**Q1：为什么 DP 常用 All-Reduce，而不是 All-Gather？**
DP 的每张卡都有完整模型，反向得到的是同形状的梯度；它要把各卡梯度求和/平均，并让每张卡都拿到同样的结果继续更新，这正是 All-Reduce（形状不变、人人有完整结果）。All-Gather 是把不同分片拼起来，适合 TP 里把切开的激活拼回完整张量。也可以把 All-Reduce 理解成"先 Reduce-Scatter 归约并切片，再 All-Gather 把切片收集起来"，这样三个原语的形状关系就串起来了。

**Q2：All-Gather 的输入是 `(8,1024)`，为什么输出不是 `(8,4096)`？**
因为拼接沿第 0 维，不是最后一维。4 个 rank 各贡献 8 行，拼起来是 $8\times4=32$ 行，列数 1024 不变，所以是 `(32,1024)`。如果沿最后一维拼，才会变成 `(8,4096)`，但本节的约定是第 0 维。

**Q3：Reduce-Scatter 的输入为什么反而比输出大？**
它先把每个 rank 的完整输入 `(32,1024)` 逐元素归约，再把归约结果沿第 0 维等分成 4 份，每份 `(8,1024)`。可以把 All-Reduce 理解成"Reduce-Scatter + All-Gather"：先规约并切片，再把切片收集起来。

**Q4：跨节点 20 ms 是不是等于 2.5 ms 加别的延迟？**
不是。20 ms 是按瓶颈链路 12.5 GiB/s 算的稳态下界，不是各跳时间相加。GPU 到网卡那一跳 50 GiB/s 比 12.5 宽，数据可以在它上面流水，真正卡住的是网络。启动、转发和拥塞都没计入，所以真实值只会更高。也正因为如此，20 ms 不能反过来证明"跨节点一定不可接受"；要看消息大小和调用频率，小消息可能主要被启动延迟支配，大消息才接近这个带宽下界。

**Q5：TP 组允许机内、实际也在机内，就说明性能一定好吗？**
不能。`placement_allowed=True` 只说明没违反放置规则。机内链路也可能拥塞，TP 的消息大小、调用频率、是否与其他组共享带宽都会影响实际成本。notebook 反复强调，通过规则检查后还要比较同步成本，不能把"位置对"当成"性能好"。尤其 TP 每层都要交换激活，调用频率高，固定启动开销会被放大；这就是为什么要把消息大小和频率一起带进第 26 节，而不是只看单次传输时间。

**Q6：跨节点一定比机内慢吗？**
不一定。Q4 的规则表专门列了"跨节点路径反而更快"这一行，提醒按实际链路数据判断，不要套用"机内一定强于跨节点"的排序。拓扑是树形还是环状、是否共享、是否拥塞，都会改变结果。机内也可能走不同代际的 PCIe/NVLink，跨节点也可能有高带宽互联；`intra_node`/`cross_node` 只是放置类别，不是带宽等级。真正比较时要把链路型号、消息大小、并发流量都带进来。

## 动手验证：怎么知道自己做对了
本节没有 TODO 空位，四个函数都是完整参考实现，验证靠单元里的 `assert`。notebook 没有保存执行输出，运行后每个单元打印字典或摘要；以断言和返回值为准。这些断言只检查机制算术和规则判断，不证明真实链路的吞吐或时延，也不替代 NCCL 的实际测试。

`three_d_parallel` 的断言：
- `cases[0]['partition_axis'] == {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'}`；
- `'gradient' in cases[1]['communication_focus']['dp']`（`(4,2,2)` 这一组）；
- `'activation' in cases[2]['communication_focus']['pp']`（`(2,4,4)` 这一组）；
- 三个 case 的 `effective_workers` 分别是 8、16、32，并打印 `3D parallelism = DP × TP × PP`。

`collective_shape` 的断言：
- `collective_shape('allreduce', 4, (8, 1024))['output_per_rank'] == (8, 1024)`；
- `collective_shape('allgather', 4, (8, 1024))['output_per_rank'] == (32, 1024)`；
- `collective_shape('reducescatter', 4, (32, 1024))['output_per_rank'] == (8, 1024)`。

`path_transfer_lower_bound` 的断言：
- 机内单链路 100 GiB/s、256 MiB：`lower_bound_ms == 2.5`；
- 跨节点 50 + 12.5 GiB/s、256 MiB：`lower_bound_ms == 20.0`；
- `bottleneck_link == 'network'`。

`check_group_placement` 的断言：
- 三个组的 `placement_allowed` 列表精确等于 `[True, False, True]`；
- 每个组的报告会打印 `name`、排序后的 `allowed_domains`、`actual_domain` 和布尔结果。

这些断言全过，并打印出 8/16/32 个 worker、三种集合通信的形状、2.5 ms 与 20 ms 的传输下界、TP/EP/DP 的放置检查结果，就说明你已经能把"切分对象 → 通信原语 → 瓶颈链路 → 放置规则"这条链走通了。

读输出时可以按四条线核对：

- **worker 线**：三个 case 的 `effective_workers` 应是 8、16、32；如果第三组得到 10（$2+4+4$），说明把乘积写成了加法。
- **形状线**：All-Reduce 输出 `(8,1024)`；All-Gather 输出 `(32,1024)`，不是 `(8,4096)`；Reduce-Scatter 输出 `(8,1024)`，不是 `(32,1024)`。沿第 0 维还是最后一维，一测就露馅。
- **下界线**：机内 2.5 ms、跨节点 20.0 ms，瓶颈链路是 `network`；如果跨节点算出 2.5 ms，说明取了最大带宽而不是最小带宽。
- **放置线**：三个组的布尔结果应是 `[True, False, True]`；如果 EP 组变成 True，说明没有用 `actual_domain in allowed`，而是只看 `actual_domain` 合法就放行。

真正选并行策略时，还要把这里的下界和规则检查带到第 20、26、27 节，用实际消息大小、调用频率和链路数据继续比较。本节给的是"机制账"，不是"性能承诺"。
