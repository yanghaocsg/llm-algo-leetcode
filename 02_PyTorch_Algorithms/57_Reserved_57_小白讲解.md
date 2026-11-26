# 57. 通用预留（Reserved 57） — 小白讲解

> 对应 notebook：[`57_Reserved_57.ipynb`](./57_Reserved_57.ipynb)

## 一句话总结
这一页不是正式知识点，而是课程目录里的结构化占位位：先把编号 `57` 的用途、上下游衔接和未来扩展接口固定下来。骨架提供 `describe_reserved_slot`、`map_slot_dependencies`、`reserve_future_topic` 三个函数，测试 `test_reserved_template` 只检查编号、字段和“未就绪”状态。当前骨架已经能跑通结构测试，但返回内容仍是空壳，不能当作正式主题成果。

## 为什么需要它：痛点是什么
课程目录越写越长，总会遇到“这个方向以后要补、但现在还没排期”的位置。直接留白，后来者可能把相邻主题塞进同一编号，造成重复；随便写个半成品，又会被误认为已经交付。预留页要解决的是“先占位、又不误导”的管理问题：编号和接口先定，内容后补。
生活类比：小区规划图里先画出“社区服务中心（预留）”的地块。现在没有家具、没有营业时间，但边界、出入口和与周边道路的接法先定好；将来无论做快递站还是活动室，都能按同一块地接入，不用重新改管线。预留页的价值在“接口清楚”，不在“现在有内容”。

## 核心问题
- 编号 `57` 将来承接的是概念补位、项目补位，还是评估补位？
- 它和前面的偏好数据、在线 DPO，以及后面的指令微调项目、架构验证是什么关系？
- 骨架里的三个 TODO 各自要返回什么结构？哪些字段是“先占位”的？
- `test_reserved_template` 能验证什么、不能验证什么？
- 当前能跑什么、不能跑什么？将来替换主题时哪些契约不能动？

## 解决思路：一步步看懂

**第 1 步：先定位这一页的角色。** notebook 标题是“57. Reserved 57 | 通用预留”，标签是 `预留`、`generic reserve`、`项目衔接`，正文明确写“当前内容是结构化占位，不是最终主题实现”。所以读这一页时，不要寻找算法公式或性能数字，而要看它怎样把一个空位描述得足够清楚，让后来者能无歧义地接手。Step 1 的三个问题——“是概念补位、项目补位，还是评估补位”“与前后章节什么关系”“未来扩展面是什么”——就是接手前必须先回答的三件事。

**第 2 步：看上下游怎样收束到 57。** “前置阅读”列了四节：`50_Preference_Data_and_Evaluation.ipynb`（偏好数据与评估）、`51_Online_DPO.ipynb`（在线 DPO 变体）、`62_Instruction_Fine_Tuning_Project.ipynb`（指令微调项目）、`61_Model_Architecture_Exploration.ipynb`（架构验证）。图示是：

```text
upstream  -> current route coverage
         │
         ▼
57 Reserved  future extension slot
         ▲
         │
downstream -> project / evaluation continuation
```

可以这样理解：`50`/`51` 提供偏好数据与在线对齐的背景，属于上游输入；`61`/`62` 是项目与架构方向的承接，属于下游可能衔接的位置。`57` 本身不是内容终点，而是 `通用预留` 的过渡和缓冲位，避免相邻章节的主题互相挤占。

**第 3 步（TODO 1）：`describe_reserved_slot(context)` 描述职责。** 骨架返回：

```python
return {
    'slot_name': '57',
    'future_role': None,
    'context_keys': list(context.keys()) if isinstance(context, dict) else [],
}
```

`slot_name` 固定编号；`future_role` 说明将来承接哪类内容，骨架故意留成 `None`；`context_keys` 只在传入字典时列出键名，用来记录占位位所处的上下文，不做内容判断。参考答案把 `future_role` 填成 `'通用预留_extension'`。这个值不是算法结论，而是管理标签：告诉后来者“这里准备承接通用方向的扩展”。容易写错成把 `context` 整个塞进返回值，或者对非字典输入直接调用 `.keys()` 导致 `AttributeError`；骨架用 `isinstance(context, dict)` 做了保护。

**第 4 步（TODO 2）：`map_slot_dependencies(links)` 记录衔接关系。** 骨架返回 `{'upstream': [], 'downstream': []}`，两个列表都是空的。参考答案：

```python
return {
    'upstream': [item for item in links if item == 'upstream'],
    'downstream': [item for item in links if item == 'downstream'],
}
```

它只从传入的 `links` 里挑出名为 `'upstream'` 和 `'downstream'` 的项，分别放进两个列表；没有拓扑排序，也没有自动推导依赖。职责只是“把两类关系分开记录”，保证返回结构里两个字段永远存在。测试传 `['upstream', 'downstream']`，检查两个键都能出现。容易写错成把 `links` 原样返回，或者漏掉其中一个键，测试的 `'upstream' in deps and 'downstream' in deps` 会失败。

**第 5 步（TODO 3）：`reserve_future_topic(topic_name)` 预留主题。** 骨架和参考答案一致：

```python
return {
    'topic_name': topic_name,
    'ready_for_content': False,
}
```

关键是 `ready_for_content` 必须为 `False`，表达“主题名可以先登记，但内容还没准备好，不能对外宣称已交付”。测试最后一条断言就是 `reserved['ready_for_content'] is False`。

**第 6 步：测试到底验证了什么。** `test_reserved_template` 构造 `context = {'route': 'generic reserve', 'status': 'reserved'}`，依次检查：`summary['slot_name'] == '57'`；`deps` 同时含 `'upstream'` 和 `'downstream'` 两个键；`reserved['ready_for_content'] is False`。通过时打印 `测试通过：预留页模板结构正常。`，异常时打印 `测试未通过：{exc}`。
要注意两点。第一，这是**结构测试**，不是语义测试：骨架里 `future_role=None`、依赖列表为空，也能满足上述断言，所以“测试通过”只证明占位模板的字段没丢，不证明职责已经写清楚。第二，测试用 `try/except Exception` 包住全部断言，失败时只打印而不抛出，所以看到“测试未通过”时仍要继续检查，不能只看程序有没有崩。真正替换主题时，应把 `future_role` 写成具体方向，并在 `map_slot_dependencies` 里返回实际的上游/下游章节名，而不是停在空列表。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 预留页（Reserved） | 目录里先占住编号、暂不写正式内容的页面 |
| 结构化占位 | 用固定字段描述“这里将来放什么”，而不是留白 |
| `slot_name` | 预留位编号，这里是 `'57'` |
| `future_role` | 将来承接的内容类型，骨架先留 `None` |
| `context_keys` | 传入上下文里有哪些键，只记录不判断 |
| upstream / downstream | 上游输入与下游承接，分开记录避免混淆 |
| `ready_for_content` | 是否已可交付内容，预留阶段必须是 `False` |
| 结构测试 | 只检查字段和取值形态，不检查业务语义 |

## 小白常见疑问
**Q1：既然骨架不改也能让测试通过，三个 TODO 还有意义吗？**
有。测试只守住“编号、字段、未就绪状态”三条底线，防止后来者把占位页改成空文件或改错编号；TODO 的意义是把 `future_role` 和依赖列表从“空壳”补成“可交接的说明”。当前能跑的是结构，不能跑的是正式主题内容。

**Q2：`describe_reserved_slot` 为什么对非字典的 `context` 不报错？**
因为骨架用 `list(context.keys()) if isinstance(context, dict) else []` 做了类型分支。预留页的输入可能来自不同调用方，这行代码保证即使传列表或 `None`，函数也能返回结构完整的字典，而不是在 `.keys()` 上崩掉。

**Q3：`map_slot_dependencies` 会检查循环依赖吗？**
不会。它只做名称筛选，不做图算法。循环依赖、未知引用这些检查属于正式章节的职责；预留页刻意保持最小，避免提前写死一套将来可能不用的规则。

**Q4：这一页和 `61`、`62` 是什么关系？**
`57` 是通用预留位，`61` 是架构验证、`62` 是指令微调项目。图示把项目/评估延续画在 `57` 的下游，意思是将来 `57` 若被填成某个通用主题，应当能和这些项目页衔接，而不是另起一套互不相干的目录。具体接哪一条，要等主题确定后再写进 `map_slot_dependencies`。

**Q5：当前这一页能跑训练或 GPU 吗？**
不能，也不需要。整个 notebook 只有标准库 `typing` 的导入和纯 Python 函数，没有模型、数据、训练或 GPU 单元；它能在 CPU 上直接运行，且不依赖第三方库。它验证的是目录管理契约，不是算法正确性。

## 动手验证：怎么知道自己做对了
运行最后一个代码单元，它直接调用 `test_reserved_template()`。预期看到 `测试通过：预留页模板结构正常。`；如果某个字段被改坏，会打印 `测试未通过：...` 并给出具体异常信息。可以自己加三项检查：把 `describe_reserved_slot({'a': 1})` 的 `context_keys` 断言为 `['a']`；把 `map_slot_dependencies(['upstream', 'x', 'downstream'])` 断言为 `{'upstream': ['upstream'], 'downstream': ['downstream']}`；把 `reserve_future_topic('t')['ready_for_content']` 保持为 `False`。
notebook 没有保存执行输出，也没有 GPU 或训练单元格，因此以代码与断言为准，不要编造运行结果。**不能**用它验证某个算法是否正确，因为它本来就没有算法实现；它的交付物是“编号、上下游接口、可替换主题骨架”这三样管理契约。
