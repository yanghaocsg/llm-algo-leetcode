# 58. 通用预留 — 小白讲解

> 对应 notebook：[`58_Reserved_58.ipynb`](./58_Reserved_58.ipynb)

## 一句话总结
`58_Reserved_58.ipynb` 不是一节正式课程，而是一个**结构化占位页**：它先把编号 `58` 的用途、上下游关系和最小交付协议固定下来，留出 `通用预留`（generic reserve）这个可替换入口，等后续确定主题后再填内容。页面里的三个 TODO 只要求写出"这个位置将来承接哪类内容""和前后章节怎么衔接""未来主题的骨架是什么"，并用 `test_reserved_template` 做一次模板结构检查；它不训练模型、不加载数据，也没有可评测的指标。

## 为什么需要它：痛点是什么
课程目录里经常需要预留编号：有时是某个方向还没定题，有时是等上下游章节完成后再补项目或评估。如果没有一个明确的占位页，容易出现三种混乱：编号被别的主题占用、后来补的内容和相邻章节重复、读者点进来只看到空白，不知道这里将来会有什么。

生活类比：装修时先在墙上留一个配电箱位置，标好"这里以后装总闸"，并把进线、出线、相邻插座的位置写在图纸上。预留页就是这张图纸：它不假装自己已经装好电器，但让后来的人知道"这里能接什么、不能接什么、接的时候要满足哪些条件"。

## 核心问题
- `58` 这个编号当前是什么状态？它是正式课程还是占位？
- TODO 1、TODO 2、TODO 3 分别要求补什么？骨架已经提供了哪些字段？
- `test_reserved_template` 检查了什么、没检查什么？为什么骨架原样也能通过？
- 这个预留位和 50、51、61、62 是什么关系？
- 当前能运行什么、不能运行什么？什么时候才算"可以替换成正式内容"？

## 解决思路：一步步看懂

**第 1 步：先认清它只是占位。** 页面开头写明"当前内容是结构化占位，不是最终主题实现"，标签是 `预留`、`generic reserve`、`项目衔接`，运行环境是 CPU-first。正文没有 Step 5，没有数据、模型、GPU 实验，也没有保存任何执行输出。它的交付物不是知识点，而是一份"未来扩展的接口约定"。

**第 2 步：TODO 1——描述预留位职责。** 骨架函数是 `describe_reserved_slot(context)`，接收一个上下文字典，返回 `slot_name`、`future_role`、`context_keys` 三个字段。骨架里 `future_role` 还是 `None`，`context_keys` 用 `list(context.keys()) if isinstance(context, dict) else []` 取键名。参考答案把 `future_role` 填成 `'通用预留_extension'`。这一步要回答的核心问题是：这个编号以后承接的是**概念补位、项目补位还是评估补位**？先把用途写死，才能避免和相邻章节重复。

**第 3 步：TODO 2——说明衔接关系。** `map_slot_dependencies(links)` 接收一个链接/方向列表，返回 `{'upstream': [...], 'downstream': [...]}`。骨架先返回两个空列表；参考答案按元素是否等于 `'upstream'` 或 `'downstream'` 过滤：

```python
return {
    'upstream': [item for item in links if item == 'upstream'],
    'downstream': [item for item in links if item == 'downstream'],
}
```

这一步的产物是"依赖顺序"：上游是谁、下游是谁。页面把 50（偏好数据与评估）、51（在线 DPO）、62（指令微调项目）、61（架构验证）列为前置阅读；按图示，58 位于"上游内容覆盖"与"下游项目/评估承接"之间。预留页的作用是把这条路线收束到一个明确入口，而不是另起一条重复的支线。

**第 4 步：TODO 3——预置未来扩展骨架。** `reserve_future_topic(topic_name)` 返回 `topic_name` 和 `ready_for_content`。骨架和参考答案都返回 `ready_for_content: False`。这个字段是"状态开关"：只要还是预留，就必须是 `False`，避免下游误以为这里已经有可交付内容。将来真正写内容时，应该新开正式章节或把本页替换掉，而不是悄悄把 `False` 改成 `True` 却什么都不补。

**第 5 步：测试到底检查了什么。** `test_reserved_template` 做三件事：

```python
context = {'route': 'generic reserve', 'status': 'reserved'}
summary = describe_reserved_slot(context)
assert summary['slot_name'] == '58', '预留位编号不正确！'
deps = map_slot_dependencies(['upstream', 'downstream'])
assert 'upstream' in deps and 'downstream' in deps, '衔接关系字段缺失！'
reserved = reserve_future_topic('future_topic')
assert reserved['ready_for_content'] is False, '预留状态不应为可交付内容！'
```

它只验证"编号是 58""两个键存在""状态是 False"这三条结构约束，**不检查** `future_role` 是否有意义、`upstream/downstream` 列表是否真的填了章节、主题说明是否完整。因此骨架原样运行也能打印 `测试通过：预留页模板结构正常。`——这不代表内容写完了，只代表模板没坏。反过来说，如果未来有人把 `ready_for_content` 改成 `True`，或者把 `slot_name` 改错，这个测试会立刻报出来。

还要注意：整个测试包在 `try/except Exception` 里，失败时只打印 `测试未通过：{exc}`，不会向外抛异常。所以不能只看"单元跑完没报错"来判断通过，必须看打印的是"测试通过"还是"测试未通过"。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 预留页（Reserved） | 占住编号、写清用途和接口，但还没有正式内容的页面 |
| 结构化占位 | 用固定字段描述"将来放什么"，不是空白页 |
| `generic reserve` | 本页的路线标签，表示通用预留方向 |
| `future_role` | 这个编号将来承担的角色说明 |
| `ready_for_content` | 是否已经可以交付正式内容；预留阶段必须是 `False` |
| 上下游衔接 | 这个位置从哪些章节接进来、又交给哪些章节 |
| 模板冒烟测试 | 只检查结构字段是否存在、状态是否正确，不检查内容质量 |

## 小白常见疑问

**Q1：这一页现在能跑出什么结果？**
只能跑 `test_reserved_template` 这一个模板检查，打印"测试通过"或"测试未通过"。它不加载模型、不读数据、不训练、不需要 GPU，也没有损失、准确率或显存之类的指标。页面没有保存执行输出，所以不要引用任何"运行结果"；以你本地打印的文本为准。

**Q2：骨架里 TODO 都没填，为什么测试还能过？**
因为测试只断言 `slot_name == '58'`、`deps` 里有 `upstream`/`downstream` 两个键、`ready_for_content is False`。骨架的 `describe_reserved_slot` 已经返回 `'58'`，`map_slot_dependencies` 已经返回这两个键，`reserve_future_topic` 已经返回 `False`。这三条都能满足，所以结构测试通过。TODO 1 的 `future_role` 和 TODO 2 的实际过滤逻辑没有被断言，属于"测试没覆盖到的内容"，需要人工检查。

**Q3：那这个测试是不是没用？**
有用，但它的定位是"防回归"而不是"验收内容"。它能防止编号被改错、字段被删、预留状态被误改成可交付；它不能证明主题选得对、上下游列得全。真正替换成正式内容时，应该新增针对该主题的测试，而不是继续依赖这个模板测试。

**Q4：为什么预留页还要写上下游？直接空着不行吗？**
空着会让后来的人重新做一遍"这个编号该放哪"的决策，也容易和相邻章节撞题。notebook 的 Step 1 明确要求先回答"承接哪类内容"，再固定"与前后章节的关系"，最后才决定最终主题。先把接口写清楚，替换成本才低。

## 动手验证：怎么知道自己做对了

运行 notebook 里的测试单元，观察打印文本。通过时输出 `测试通过：预留页模板结构正常。`；失败时输出 `测试未通过：...`，但因为异常被 `try/except` 吞掉，进程不会以非零状态退出，所以要把打印内容当作判据。

具体断言有三条：`summary['slot_name'] == '58'`（编号不能错）、`'upstream' in deps and 'downstream' in deps`（两个方向字段不能缺）、`reserved['ready_for_content'] is False`（预留状态不能变成可交付）。如果把 `future_role` 填成 `'通用预留_extension'`、把 `map_slot_dependencies` 改成真正过滤 `links`，测试仍然通过——这正说明它只保结构、不保语义。判断这一节是否"完成"，不能只看测试，还要检查三件事：`future_role` 是否说清了承接类型、`upstream/downstream` 是否列出了真实章节、以及是否明确标注"当前不可交付、替换时应新开正式内容"。
