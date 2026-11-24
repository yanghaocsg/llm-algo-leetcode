# 64. SFT Data Quality Project | SFT 数据质量项目 · 小白入门

> 配套 notebook：[64. SFT 数据质量项目](../64_SFT_Data_Quality_Project.ipynb) ｜ 难度：Hard ｜ 环境：CPU 主线

## 一句话说清

一批训练数据能读进来，不等于能拿去训练。这本 notebook 教你按固定的门槛判断一批数据能不能用、清洗后的版本能不能替换旧版本，以及哪个版本可以交付。

## 为什么需要它

监督微调要给模型看很多“提示词加回答”的示范。一批示范能成功加载，不等于可以用来训练。具体会出这几种事：

- 有的样本回答是空的。拼接、截断之后，参与损失计算的只剩一个结束标记，模型学不到怎么回答。
- 同一条记录出现了很多次。它不算错误，但会让这条样本的权重变大。
- 训练集和评测集里有同一个任务。模型训练时见过这道题，评测成绩就不再独立。
- 样本太长被截断，回答所剩无几，甚至一个词元都不剩，没有可学的目标。

这些风险要一起看，靠平均长度或者一个质量总分判断不出来。修数据的过程也要留痕：清洗前后各是什么样，用了哪个分词器、多长的窗口，后面的实验才能找到同一批数据。

这本 notebook 要做的是三件事：根据一份审计结果判断这批数据能不能训练，判断清洗后的版本能不能替换旧版本，再从几个候选版本里选出可以交付的那一个。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 数据准入（data admission） | 判断一批数据能不能拿来训练的检查流程 |
| 监督微调（supervised fine-tuning，SFT） | 用“提示词加回答”的示范继续训练模型 |
| 训练快照（training snapshot） | 通过检查后固定下来的样本集合，后面的实验都用这一份 |
| 数据清单（manifest） | 记录这批数据的来源、分词器、最大长度、切分成员和它们的哈希 |
| 阻断项 / 可修复项 | 阻断项出现就不能训练，比如空回答；可修复项超限要先清洗再重审，比如重复率 |
| 切分泄漏（split leakage） | 训练集和评测集里出现了同一个任务，评测成绩就不再独立 |
| `accept` / `tune` / `reject` | 准入的三种状态：接受、待修复、拒绝 |

## 核心思路

输入是一份审计结果（各项计数和比例）、一组门槛，以及清洗前后的两个版本。处理分三步：判断这批数据的准入状态；判断清洗后的版本能不能替换旧版本；从多个候选版本里挑出可以交付的那一个。输出是状态 `accept`、`tune` 或 `reject`，以及全部失败原因。

失败分两类，判断规则如下：

```text
阻断项：出现任何一个，状态就是 reject
    空指令、空回答、零监督词元、训练与评测重合，这四项计数 > 0
    评测样本数 < 最低评测样本数
可修复项：没有阻断项，但有任意一项，状态是 tune
    重复率 > 最大重复率
    截断率 > 最大截断率
两类都没有失败项，状态是 accept
```

重复率是重复样本所占的比例，截断率是被最大长度切掉的样本所占的比例。两类风险同时出现时，状态取 `reject`，但两类原因都要保留：状态决定能不能继续，全部失败项决定先修什么。

清洗后的版本要替换旧版本，几条要同时满足：来源、处理规则、分词器版本和最大长度相同；旧问题全部解决，新版本没有剩余风险；评测样本数不减少；保留的样本比例不低于下限；旧版本还不是 `accept`，新版本变成了 `accept`。挑选可交付的版本时，候选要同时满足三条：状态是 `accept`、清单完整、没有失败项。

下面是最小例子，审计结果和门槛取自 notebook 的测试：

```python
limits = {'minimum_evaluation_count': 2, 'maximum_duplicate_ratio': 0.02, 'maximum_truncation_ratio': 0.10}
clean = {'empty_instruction_count': 0, 'empty_response_count': 0, 'zero_supervision_count': 0,
         'split_leakage_count': 0, 'evaluation_count': 16, 'duplicate_ratio': 0.01, 'truncation_ratio': 0.05}

def admit(audit):                                # 返回：状态、阻断项、可修复项
    hard = {'empty_instruction': audit['empty_instruction_count'] > 0,
            'empty_response': audit['empty_response_count'] > 0,
            'zero_supervision': audit['zero_supervision_count'] > 0,
            'split_leakage': audit['split_leakage_count'] > 0,
            'missing_evaluation': audit['evaluation_count'] < limits['minimum_evaluation_count']}
    soft = {'duplicate_ratio': audit['duplicate_ratio'] > limits['maximum_duplicate_ratio'],
            'truncation_ratio': audit['truncation_ratio'] > limits['maximum_truncation_ratio']}
    hard = [k for k, failed in hard.items() if failed]
    soft = [k for k, failed in soft.items() if failed]
    return ('reject' if hard else 'tune' if soft else 'accept'), hard, soft

print(admit(clean))
print(admit({**clean, 'duplicate_ratio': 0.02}))             # 重复率正好等于门槛
print(admit({**clean, 'duplicate_ratio': 0.03}))
print(admit({**clean, 'duplicate_ratio': 0.03, 'zero_supervision_count': 1}))
print(92 / 100)                                              # 清洗后的保留率
```

```text
('accept', [], [])
('accept', [], [])
('tune', [], ['duplicate_ratio'])
('reject', ['zero_supervision'], ['duplicate_ratio'])
0.92
```

前两行是没有问题的情况：重复率正好等于门槛 0.02 也算通过，只有超过才算失败。第三行只有重复率超限，状态是 `tune`。第四行在重复率超限的同时出现了零监督词元，状态取 `reject`，两类原因都保留了下来。最后一行是保留率：清洗前 100 条样本，清洗后 92 条，保留率是 0.92。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 从候选数据到可训练快照：记录来源、监督审计、清洗与切分、判断能否训练四个阶段 | 读 |
| Step 2 | 固定分词与截断方式，统计重复和监督数量：六项审计指标各自算什么、对准入有什么影响 | 读 |
| Step 3 | 判断数据能否训练、清洗版本能否替换：拒绝、待修复、接受三种状态，以及替换要满足的条件 | 读，对照上面的规则 |
| Step 4 · TODO 1 | 判断数据准入状态 | 写 `decide_data_admission`：1 行，有阻断项是 `reject`，否则有可修复项是 `tune`，否则 `accept` |
| Step 4 · TODO 2 | 判断清洗版本能否替换 | 写 `check_cleaned_snapshot_transition`：3 行，算出 `identity_compatible`、`risks_repaired`、`eligible` |
| Step 4 · TODO 3 | 筛选可用于训练的数据版本 | 写 `select_training_snapshot`：1 行，算出 `handoff_allowed`，三个条件同时成立才为真 |
| 测试 | 检查准入优先级、清洗后的风险解除、来源是否可比、能否交接，以及非法事实和门槛等号这类边界值 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解释 | 卡住再看 |
| Step 5 | 真实数据准入实验：固定数据来源、分词器、切分规则和最大长度，审计真实数据并生成数据准入清单 | 需要 `PT-SFT-CUDA` 环境的数据与分词器依赖；默认是 `RUN_STAGE = "prepare"`，只做检查，不采集真实数据 |

Step 4 和测试只需要 CPU。Step 5 的采集阶段需要 `PT-SFT-CUDA` 环境的数据与分词器依赖，不加载模型权重、不调用 GPU；默认是 `RUN_STAGE = "prepare"`，只做检查，不采集真实数据。

## 容易踩的坑

- **两类风险只留了一类**：阻断项和可修复项同时出现时，状态是 `reject`，但两类原因都要保留。状态决定能不能继续，全部失败项决定先修什么。
- **门槛等号当成了失败**：重复率、截断率正好等于上限时算通过，只有大于才失败。评测样本数正好等于下限也算通过。
- **样本多就能掩盖风险**：挑选可交付版本时，`accept`、清单完整、没有失败项三条缺一不可。样本更多的候选不能抵消失败项或清单缺失。
- **换了分词器还说是同一批数据**：换了分词器或最大长度，长度与监督计数就不能沿用，要重新审计；来源或配置不同的新版本也不能替换旧版本。
- **清洗后样本变多了**：清洗只能删除样本。新版本的样本数大于旧版本时直接抛出 `ValueError`，要扩充数据应该另建版本。

## 读完后自查

- [ ] 能说出哪些问题是阻断项，哪些是可修复项，分别对应什么状态。
- [ ] 重复率正好等于上限时，能判断算不算失败。
- [ ] 能说出清洗后的版本要替换旧版本需要同时满足哪几个条件。
- [ ] 能说出一个候选版本要同时满足哪三条才能交付训练。

## 下一步

- 打开 notebook：[64. SFT 数据质量项目](../64_SFT_Data_Quality_Project.ipynb)
- 建议先读：[13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md)、[30. 长上下文微调机制 · 小白入门](./30_Long_Context_Fine_Tuning.md)、[32. SFT 数据契约与泄漏审计 · 小白入门](./32_Data_Engineering_for_SFT.md)
- 做完之后：[62. SFT 训练交付项目 · 小白入门](./62_Instruction_Fine_Tuning_Project.md)
- 相关 notebook：[63. LoRA 变体选型项目](../63_LoRA_Variants_Benchmark.ipynb)、[65. QLoRA 选型项目](../65_QLoRA_Selection_Project.ipynb)
