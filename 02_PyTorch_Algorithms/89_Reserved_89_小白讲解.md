# 89. 通用预留 — 小白讲解

> 对应 notebook：[`89_Reserved_89.ipynb`](./89_Reserved_89.ipynb)

## 一句话总结
`89_Reserved_89.ipynb` 不是一节正式课程，而是课程目录里的**结构化占位页**：它先把编号 `89`、上下游关系（接 84/85/86 的后训练对齐项目，往后接项目或评估方向）和最小交付协议固定下来，留下 `describe_reserved_slot`、`map_slot_dependencies`、`reserve_future_topic` 三个待填函数。当前骨架只保证"结构可运行、编号不冲突"，不提供任何模型、数据或实验结果。

## 为什么需要它：痛点是什么
一个持续扩展的教程仓库最怕两件事：编号被后来的主题抢走，导致旧链接和目录错位；以及"这里以后要补"只写在某个人脑子里，下一位作者不知道该接在哪、该交付什么。预留页就是给未来内容留一个**有门牌号、有接口、有验收线**的空位，而不是留一段空白。

生活类比：商场装修时先租下并围挡一个铺位，挂上"预留"牌子，标好左边挨着谁、右边挨着谁、消防通道在哪。围挡里现在不卖货，但水电接口和门牌已经就位，将来谁来开店都不用重新改编号。这个 notebook 里的三个函数就是"水电接口"，`test_reserved_template` 就是"验收线"。

## 核心问题
- 编号 `89` 将来要承接哪一类内容：概念补位、项目补位，还是评估补位？
- 它和上游 84/85/86、下游的项目/评估路线是什么依赖顺序？
- 一个"预留页"最小要交付什么，才能既不算空页、又不假装已经有内容？
- 当前骨架能跑什么、不能跑什么？测试通过是否等于内容完成？
- 将来把占位替换成正式主题时，哪些字段必须改、哪些约束必须保留？

## 解决思路：一步步看懂
### 第 1 步：Step 1 先定义"未来职责"
notebook 的 Step 1 只做一件事：回答这个编号后续承接的是**概念补位、项目补位还是评估补位**。它给了三条原则：先固定编号用途，避免和相邻章节重复；明确与前后章节的关系；先把未来扩展面写清楚，再决定最终主题。这一步没有代码，产出是文字约定——`89` 被定义成"通用预留"的过渡和缓冲位，不是内容终点。

### 第 2 步：图解上下游
cell 4 用一段 ASCII 图说明位置：

```text
upstream  -> current route coverage
         │
         ▼
89 Reserved  future extension slot
         ▲
         │
downstream -> project / evaluation continuation
```

翻译成课程语境：上游是已经完成的 84 DPO 偏好项目、85 GRPO 分组对齐项目、86 在线后训练生命周期与采用决策；`89` 自己不承接具体算法，只做"后续扩展的统一入口"；下游是把这里替换成正式主题后，继续接项目或评估。相关阅读里的 `2_10.md`（项目验证与工程收口）提供交付规范，告诉后来者"什么算做完"。

### 第 3 步：三个 TODO 与它们的返回值
cell 6 是骨架，三个函数各有一个 TODO：

1. **TODO 1 `describe_reserved_slot(context)`**：描述预留位职责。骨架返回
   ```python
   {'slot_name': '89', 'future_role': None,
    'context_keys': list(context.keys()) if isinstance(context, dict) else []}
   ```
   参考实现把 `future_role` 填成 `'通用预留_extension'`。`context_keys` 只是把调用方传进来的上下文字段名摊平，方便未来主题读取 route、status 等信息。
2. **TODO 2 `map_slot_dependencies(links)`**：说明衔接关系。骨架返回 `{'upstream': [], 'downstream': []}`；参考实现用列表推导把 `links` 里等于 `'upstream'` 或 `'downstream'` 的项分别放进两个桶：
   ```python
   'upstream': [item for item in links if item == 'upstream'],
   'downstream': [item for item in links if item == 'downstream'],
   ```
   它不解析复杂的依赖图，只保证"上下游两个字段一定存在且能承接条目"，这是最小契约。
3. **TODO 3 `reserve_future_topic(topic_name)`**：预置未来主题。返回 `{'topic_name': topic_name, 'ready_for_content': False}`。`False` 是核心：它明确宣告"这里还没有可交付内容"，防止下游误把占位当成品。

三个 TODO 的共同点是**只固定结构，不写死内容**。`future_role` 可以换、`links` 可以增、`topic_name` 可以改，但键名和"未就绪"状态必须保留。

### 第 4 步：测试到底在验什么
`test_reserved_template()` 构造 `context={'route': 'project reserve', 'status': 'reserved'}`，然后：

```python
summary = describe_reserved_slot(context)
assert summary['slot_name'] == '89', '预留位编号不正确！'
deps = map_slot_dependencies(['upstream', 'downstream'])
assert 'upstream' in deps and 'downstream' in deps, '衔接关系字段缺失！'
reserved = reserve_future_topic('future_topic')
assert reserved['ready_for_content'] is False, '预留状态不应为可交付内容！'
print('测试通过：预留页模板结构正常。')
```

它只检查三件事：编号是 `'89'`、依赖字典有 `upstream` 和 `downstream` 两个键、预留状态为 `False`。注意：**即使三个 TODO 都不填，当前骨架也能通过**——因为骨架返回的 `slot_name` 已经是 `'89'`、两个键已经存在、`ready_for_content` 已经是 `False`。这不是测试写错了，而是预留页的验收标准本来就只是"结构在位"，不是"内容完成"。测试用 `try/except` 包住，失败时打印 `测试未通过：{exc}`，不会让整本 notebook 中断。

### 第 5 步：当前能跑什么、不能跑什么
**能跑**：`from typing import Dict, List` 导入；三个函数调用；`test_reserved_template()` 在 CPU 上瞬时完成并打印"测试通过：预留页模板结构正常。"；参考代码单元可对照看 `future_role` 和依赖过滤的写法。

**不能跑/不存在**：没有模型加载、没有数据集、没有 GPU 配置、没有 benchmark 数字、没有 `benchmarks/results/` 产物，也没有可复现的实验指标。不要从这一页推断任何性能或质量结论。它的 `STOP HERE` 只是提醒读者先自己看骨架，再往下看参考代码，不是实验暂停点。

### 第 6 步：将来怎么替换
后续作者接手时，最小改动是：把 `future_role` 换成真实主题说明；把 `links` 换成真实的上下游章节；把 `reserve_future_topic` 的 `ready_for_content` 从 `False` 改成 `True` 之前，先补上正式内容、测试和结果记录。编号 `89`、上下游字段名、`slot_name` 这三样应保持不变，否则会破坏目录和测试契约。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 预留页（Reserved） | 占住编号、定义接口、暂不提供正式内容的页面 |
| 占位说明 | 明确写着"当前是结构化占位，不是最终实现"的声明 |
| 最小交付协议 | 一个页面至少要有编号、职责、上下游和未就绪标记 |
| `slot_name` | 预留位编号，这里是 `'89'` |
| `future_role` | 将来要承接的内容类型，参考值为 `'通用预留_extension'` |
| `context_keys` | 调用方上下文里有哪些字段名 |
| `map_slot_dependencies` | 把上下游条目分桶的函数 |
| `ready_for_content` | 是否已经可以交付正式内容，占位阶段必须是 `False` |
| 上游/下游 | 接在 84/85/86 之后，往项目或评估方向延伸 |

## 小白常见疑问
**Q1：三个 TODO 都没填，测试却打印"测试通过"，是不是测试失效了？**
不是失效，是验收标准不同。`test_reserved_template` 只断言结构：`slot_name == '89'`、`deps` 里有 `upstream`/`downstream`、`ready_for_content is False`，而骨架的默认返回值已经满足这三点。它防的是"编号写错、字段被删、占位被误标成可交付"，不负责判断主题内容好不好。真要验收内容，得等这个编号被替换成正式主题后另写测试。

**Q2：`map_slot_dependencies` 的参考实现只是把 `links` 过滤一遍，这算"依赖关系"吗？**
在这一页的语境里算。预留页只需要一个**可承接**的结构：上游、下游两个桶存在，条目能放进去。真正的依赖图、执行顺序和结果引用，要等正式主题确定后由项目页实现；在占位阶段写复杂解析反而是过度设计。

**Q3：为什么 `ready_for_content` 必须是 `False`，不能先写 `True` 占着？**
因为它是对下游的承诺。写成 `True` 会让目录、自动检查或读者以为这里已经有可运行内容，点进来却发现是空壳；`False` 则明确表示"这是预留位"。参考实现也保持 `False`，测试还专门断言它不能是别的值。

**Q4：这一页和 84/85/86 是什么关系？为什么不直接删掉？**
84/85/86 是已经交付的后训练对齐项目，`89` 是它们之后的**扩展缓冲位**，不重复它们的内容，也不替代它们。删掉会留下编号空洞，未来新增主题时容易和现有编号冲突；保留一个结构化的 `89`，等于提前把门牌和接口装好，后续替换成本最低。

## 动手验证：怎么知道自己做对了
运行 `test_reserved_template()`，如果打印 `测试通过：预留页模板结构正常。`，说明三个结构断言都过了；如果打印 `测试未通过：...`，冒号后面就是失败原因（例如编号不对、字段缺失、`ready_for_content` 不为 `False`）。参考代码单元与应用代码单元的函数签名一致，可以逐行对照 `future_role` 和上下游过滤的差异。notebook 没有保存任何执行输出，也没有 benchmark 或结果 JSON，所以"做对"的标准就是这三条断言，而不是某个数字。
