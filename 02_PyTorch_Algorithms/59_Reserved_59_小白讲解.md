# 59. 通用预留（Reserved 59） — 小白讲解

> 对应 notebook：[`59_Reserved_59.ipynb`](./59_Reserved_59.ipynb)

## 一句话总结
这一页不是正式知识点，而是课程目录里的一个**结构化占位位**：先把编号 `59` 的用途、上下游衔接和未来扩展接口固定下来。骨架提供 `describe_reserved_slot`、`map_slot_dependencies`、`reserve_future_topic` 三个函数和 `test_reserved_template` 测试，三个 TODO 分别补“职责描述、依赖关系、可替换主题”。当前骨架能跑通结构测试，但返回内容还是空的，不能当作正式主题成果。

## 为什么需要它：痛点是什么
课程目录越写越长，总会出现“这个方向以后要补，但现在还没排期”的位置。如果直接留白，后续同学可能把相邻主题塞进同一个编号，造成重复；如果随便写一个半成品，又会被误认为已经交付。预留页要解决的就是这种“先占位、又不误导”的管理问题。

生活类比：小区规划图里先画出“社区服务中心（预留）”的地块。它现在没有家具、没有营业时间，但边界、出入口和与周边道路的接法先定好；将来无论是做快递站还是活动室，都能按同一块地接入，不用重新改管线。预留页的价值在“接口清楚”，不在“现在有内容”。

## 核心问题
- 编号 `59` 将来承接的是概念补位、项目补位，还是评估补位？
- 它和前面的偏好数据、在线 DPO，以及后面的指令微调项目、架构探索是什么关系？
- 骨架里的三个 TODO 各自要返回什么结构？哪些字段是“先占位”的？
- 当前的 `test_reserved_template` 能验证什么、不能验证什么？
- 将来真正替换主题时，哪些文件位置和契约不能动？

## 解决思路：一步步看懂

**第 1 步：先定位这一页的角色。** notebook 的标题是“59. Reserved 59 | 通用预留”，标签里写着 `预留`、`generic reserve`、`项目衔接`。它明确说“当前内容是结构化占位，不是最终主题实现”。所以读这一页时，不要寻找算法公式或性能数字，而要看它怎样把一个空位描述得足够清楚，让后来者能无歧义地接手。

**第 2 步：看上下游怎样收束到 59。** notebook 的“前置阅读”列了四节：`50_Preference_Data_and_Evaluation.ipynb`（偏好数据与评估）、`51_Online_DPO.ipynb`（在线 DPO 变体）、`62_Instruction_Fine_Tuning_Project.ipynb`（指令微调项目）、`61_Model_Architecture_Exploration.ipynb`（架构验证）。图示把关系画成：

```text
upstream  -> current route coverage
         │
         ▼
59 Reserved  future extension slot
         ▲
         │
downstream -> project / evaluation continuation
```

可以这样理解：`50`/`51` 提供偏好数据与在线对齐的背景，属于上游输入；`61`/`62` 是项目与架构方向的承接，属于下游可能衔接的位置。`59` 本身不是内容终点，而是一个缓冲位，避免相邻章节的主题互相挤占。

**第 3 步（TODO 1）：`describe_reserved_slot(context)` 描述职责。** 骨架返回：

```python
return {
    'slot_name': '59',
    'future_role': None,
    'context_keys': list(context.keys()) if isinstance(context, dict) else [],
}
```

三个字段的分工是：`slot_name` 固定编号；`future_role` 说明将来承接哪类内容，骨架故意留成 `None`；`context_keys` 只在传入字典时列出键名，用来记录这个占位位所处的上下文，不做内容判断。
参考答案把 `future_role` 填成 `'通用预留_extension'`。这个值不是算法结论，而是管理标签：告诉后来者“这里准备承接通用方向的扩展”。容易写错成把 `context` 整个塞进返回值，或者对非字典输入直接调用 `.keys()` 导致 `AttributeError`；骨架用 `isinstance(context, dict)` 做了保护。

**第 4 步（TODO 2）：`map_slot_dependencies(links)` 记录衔接关系。** 骨架返回 `{'upstream': [], 'downstream': []}`，两个列表都是空的。参考答案：

```python
return {
    'upstream': [item for item in links if item == 'upstream'],
    'downstream': [item for item in links if item == 'downstream'],
}
```

它只从传入的 `links` 里挑出名为 `'upstream'` 和 `'downstream'` 的项，分别放进两个列表。这里没有拓扑排序，也没有自动推导依赖；它的职责只是“把两类关系分开记录”，保证返回结构里两个字段永远存在。测试里传 `['upstream', 'downstream']`，就是检查两个键都能出现。容易写错成把 `links` 原样返回，或者漏掉其中一个键，测试的 `'upstream' in deps and 'downstream' in deps` 会失败。

**第 5 步（TODO 3）：`reserve_future_topic(topic_name)` 预留主题。** 骨架和参考答案一致：

```python
return {
    'topic_name': topic_name,
    'ready_for_content': False,
}
```

关键是 `ready_for_content` 必须为 `False`。它表达的是“这个主题名可以先登记，但内容还没准备好，不能对外宣称已交付”。测试 `test_reserved_template` 最后一条断言就是 `reserved['ready_for_content'] is False`。

**第 6 步：测试到底验证了什么。** `test_reserved_template` 构造 `context = {'route': 'generic reserve', 'status': 'reserved'}`，然后依次检查：`summary['slot_name'] == '59'`；`deps` 同时含 `'upstream'` 和 `'downstream'` 两个键；`reserved['ready_for_content'] is False`。通过时打印 `测试通过：预留页模板结构正常。`，异常时打印 `测试未通过：{exc}`。
要注意：这个测试是**结构测试**，不是语义测试。骨架里 `future_role=None`、依赖列表为空，也能满足上述断言。换句话说，当前代码可以直接跑出“测试通过”，但这只证明占位模板的字段没丢，不证明职责已经写清楚。真正替换主题时，应该把 `future_role` 写成具体方向，并在 `map_slot_dependencies` 里返回实际的上游/下游章节名，而不是停在空列表。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 预留页（Reserved） | 目录里先占住编号、暂不写正式内容的页面 |
| 结构化占位 | 用固定字段描述“这里将来放什么”，而不是留白 |
| `slot_name` | 预留位的编号，这里是 `'59'` |
| `future_role` | 将来承接的内容类型，骨架先留 `None` |
| `context_keys` | 传入上下文里有哪些键，只记录不判断 |
| upstream / downstream | 上游输入与下游承接，分开记录避免混淆 |
| `ready_for_content` | 是否已可交付内容，预留阶段必须是 `False` |
| 结构测试 | 只检查字段和取值形态，不检查业务语义 |

## 小白常见疑问
**Q1：既然骨架不改也能让测试通过，那三个 TODO 还有意义吗？**
有。测试只守住“编号、字段、未就绪状态”这三条底线，防止后来者把占位页改成空文件或改错编号；TODO 的意义是把 `future_role` 和依赖列表从“空壳”补成“可交接的说明”。当前能跑的是结构，不能跑的是正式主题。

**Q2：`describe_reserved_slot` 为什么对非字典的 `context` 不报错？**
因为骨架用 `list(context.keys()) if isinstance(context, dict) else []` 做了类型分支。预留页的输入可能来自不同调用方，这行代码保证即使传列表或 `None`，函数也能返回结构完整的字典，而不是在 `.keys()` 上崩掉。

**Q3：`map_slot_dependencies` 会检查循环依赖吗？**
不会。它只做名称筛选，不做图算法。循环依赖、未知引用这些检查属于正式章节的职责（例如组合策略类 notebook 里的顺序校验）；预留页刻意保持最小，避免提前写死一套将来可能不用的规则。

**Q4：这一页和 `61`、`62` 是什么关系？**
`59` 是通用预留位，`61` 是架构验证、`62` 是指令微调项目。图示把项目/评估延续画在 `59` 的下游，意思是将来 `59` 若被填成某个通用主题，应当能和这些项目页衔接，而不是另起一套互不相干的目录。具体接哪一条，要等主题确定后再写进 `map_slot_dependencies`。

## 动手验证：怎么知道自己做对了
运行最后一个代码单元即可，它直接调用 `test_reserved_template()`。预期看到 `测试通过：预留页模板结构正常。`；如果某个字段被改坏，会打印 `测试未通过：...` 并给出具体断言信息。
可以自己加三项检查：把 `describe_reserved_slot({'a': 1})` 的 `context_keys` 改为断言 `['a']`；把 `map_slot_dependencies(['upstream', 'x', 'downstream'])` 的结果断言为 `{'upstream': ['upstream'], 'downstream': ['downstream']}`；把 `reserve_future_topic('t')['ready_for_content']` 保持为 `False`。
notebook 没有保存执行输出，也没有 GPU 或训练单元格；它能在 CPU 上直接运行，且不依赖任何第三方库。**不能**用它验证某个算法是否正确，因为它本来就没有算法实现。
