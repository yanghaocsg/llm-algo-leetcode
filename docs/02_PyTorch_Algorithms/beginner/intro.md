# Part 02 小白入门 | 逐本入门导读

Part 02 的每本 notebook 都配了一篇「小白入门」页：先用人话把概念讲清楚，再打开 notebook 动手。入门页面向会一点 Python 和 PyTorch、但没学过这个主题的读者。

## 每篇入门页的结构

| 部分 | 作用 |
|:---|:---|
| 一句话说清 | 用一两句话说明这本 notebook 在解决什么问题 |
| 为什么需要它 | 没有它会遇到什么麻烦，用一个具体场景说明 |
| 先认识几个词 | 解释 notebook 里直接使用、但没有解释的术语 |
| 核心思路 | 机制的输入、步骤和输出，配一个能运行或能手算的小例子 |
| 这本 notebook 里你要做什么 | 逐个 Step 和 TODO 说明读什么、写什么、需要什么环境 |
| 容易踩的坑 | 做题时最常见的错误和正确做法 |
| 读完后自查 | 几个自测问题 |
| 下一步 | notebook 本身、建议先读的页面、做完之后的下一篇 |

建议的用法：先读入门页的“一句话说清”“为什么需要它”和“核心思路”，跑一遍里面的小例子；然后打开 notebook 读 Step 1–3，做 Step 4 的 TODO；卡住时回看“容易踩的坑”。入门页是铺垫，不能替代 notebook 的讲解、代码和测试。

## 怎么选起点

- 完全没接触过大模型内部结构：从 2.1 的 00 开始，按顺序读到 05。
- 只关心怎样微调：先读 2.4 的 33，判断是否需要更新参数，再依次读数据、损失和训练。
- 只关心推理服务：先读 2.6 的 21，再进入 2.7。
- 显存不够或训练太慢：进入 2.3。

## 2.1 Decoder 组件与注意力

先把解码器的几个零件认清楚：词元怎样变成向量，归一化、门控前馈、位置编码和注意力各自解决什么问题。建议按顺序读。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [00. PyTorch 热身 · 小白入门](./00_PyTorch_Warmup.md) | [notebook](../00_PyTorch_Warmup.md) | Easy | CPU | 这本 notebook 练三个 PyTorch 基本功：把一组数字整理成模型要的形状、按编号查表取向量、手算梯度并传回给参数。后面的模型组件都要用到它们。 |
| [01. RMSNorm 教程 · 小白入门](./01_RMSNorm_Tutorial.md) | [notebook](../01_RMSNorm_Tutorial.md) | Easy | CPU | RMSNorm 把每个词元内部的一串数字调回到统一的大小，让深层网络的计算保持稳定。它比 LayerNorm 少一步“减均值”，所以计算更少。 |
| [02. SwiGLU 激活 · 小白入门](./02_SwiGLU_Activation.md) | [notebook](../02_SwiGLU_Activation.md) | Easy | CPU | SwiGLU 是大模型里一段小网络的写法：把每个词元的向量分成两路，一路当闸门去乘另一路，让每个特征有自己的通过强度。这本 notebook 教你算它的宽度，并动手写出这段计算。 |
| [03. 旋转位置编码教程 · 小白入门](./03_RoPE_Tutorial.md) | [notebook](../03_RoPE_Tutorial.md) | Medium | CPU | RoPE 是让大模型知道“谁在前、谁在后”的办法：按每个词元的位置，把它的查询向量和键向量转动一个角度，注意力比较它们时就能感觉到两个词元相距多远。 |
| [04. 多头注意力 · 小白入门](./04_Attention_MHA_GQA.md) | [notebook](../04_Attention_MHA_GQA.md) | Medium | CPU | 注意力让每个词元按“相关程度”汇总其他词元的信息。这本 notebook 实现多头注意力，并让多个头共用同一份历史记录（GQA、MQA），减少生成时要保存的内容。 |

## 2.2 模型架构

把零件组装成完整的模型块，再看 MoE、稀疏注意力和混合记忆这些结构分支。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [05. LLaMA3 Block 教程 · 小白入门](./05_LLaMA3_Block_Tutorial.md) | [notebook](../05_LLaMA3_Block_Tutorial.md) | Medium | CPU | 解码器块是大模型里重复堆叠的一层：注意力先让词元之间交换信息，前馈网络再在每个词元内部加工，两步的结果都加回原来的向量。这本 notebook 练的是按顺序把它们组装起来。 |
| [06. MoE 路由器 · 小白入门](./06_MoE_Router.md) | [notebook](../06_MoE_Router.md) | Medium | CPU | MoE 把大模型里的前馈网络拆成几个“专家”，每个词元只交给其中最合适的少数几个处理。这本 notebook 写做选择的路由器，以及专家装不下时怎么接纳。 |
| [07. MoE 负载均衡损失 · 小白入门](./07_MoE_Load_Balancing_Loss.md) | [notebook](../07_MoE_Load_Balancing_Loss.md) | Hard | CPU | MoE 的路由器可能“偏科”：把大部分词元都送给少数几个专家。这本 notebook 算一个衡量“偏得多厉害”的损失，并用各专家的忙闲程度去调整下一轮的选择。 |
| [08. 模型配置与参数归属 · 小白入门](./08_Architecture_Tricks.md) | [notebook](../08_Architecture_Tricks.md) | Easy | CPU | 模型的配置文件写着各种结构选择，它们到底怎样改变模型？这本 notebook 动手复现两个小例子：输入和输出两端是否共用同一份参数，以及归一化层的缩放初值。 |
| [44. 局部、滑动窗口与稀疏注意力 · 小白入门](./44_Local_Sliding_and_Sparse_Attention.md) | [notebook](../44_Local_Sliding_and_Sparse_Attention.md) | Medium | CPU | 标准注意力里，每个位置都能读取全部历史。这本 notebook 讲怎样规定“谁能看谁”：只看最近的几个位置，或者隔一段开一条捷径，再数清楚一共少看了多少。 |
| [45. 线性注意力与递推状态 · 小白入门](./45_Linear_Attention_and_Recurrent_State.md) | [notebook](../45_Linear_Attention_and_Recurrent_State.md) | Hard | CPU | 标准注意力要保留历史上每个词元的键和值。线性注意力改成只保留一份固定大小的累计状态，每来一个词元就更新它，再从状态里读出当前的输出。 |
| [ARCH-MLA. MLA 与紧凑 KV 表示 · 小白入门](./ARCH-MLA_Multihead_Latent_Attention.md) | [notebook](../ARCH-MLA_Multihead_Latent_Attention.md) | Hard | CPU-first + 可选单卡 GPU | 模型每生成一小段文本，都要把前面的信息缓存下来，而且越写缓存越大。这本 notebook 用同一组配置给三种缓存方式算账：每种占多少字节，谁最省。 |
| [ARCH-HYBRID-MEMORY. Attention、SSM 与混合记忆架构 · 小白入门](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md) | [notebook](../ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md) | Hard | CPU | 有的层靠注意力逐条回看历史，有的层只维护一份固定大小的状态。这本 notebook 讲怎样安排这两种层各放在模型的哪一层，再算清状态占多少、够不够回看历史。 |

## 2.3 反向传播与训练控制

沿一次参数更新展开：梯度怎样传回来，微批次怎样合成一次更新，显存被什么占用、怎样省下来。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [09. 激活与损失反向传播 · 小白入门](./09_Activation_and_Loss_Backward.md) | [notebook](../09_Activation_and_Loss_Backward.md) | Medium | CPU | 训练时，误差从损失函数出发，一层一层往回传。这本 notebook 手写其中两段规则：激活函数 ReLU 怎么决定梯度能不能通过，交叉熵这个损失函数怎么把误差送回模型的输出分数。 |
| [10. Attention 反向传播与自定义 Autograd · 小白入门](./10_Attention_Backward_and_Custom_Autograd.md) | [notebook](../10_Attention_Backward_and_Custom_Autograd.md) | Medium | CPU | 注意力算出输出之后，训练还需要把梯度沿原路传回它的三个输入。这本 notebook 把这条回传路径拆成四小段，让你逐段手写，再拼成一个完整的反向传播。 |
| [11. 梯度累积 · 小白入门](./11_Gradient_Accumulation.md) | [notebook](../11_Gradient_Accumulation.md) | Medium | CPU 主线，GPU 可选 | 训练时显存装不下一整批数据，梯度累积就把这批数据分成几小批依次计算，把每小批算出的梯度攒起来，攒够再更新一次参数。这本 notebook 教你让分批算出的结果和整批一次算的一样。 |
| [TRAIN-MIXED-PRECISION-BACKWARD. 混合精度反向传播与梯度安全 · 小白入门](./TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.md) | [notebook](../TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.md) | Hard | CPU 主线，GPU 可选 | 用低精度数字训练时，太小的梯度会被当成 0。这本 notebook 教你先把损失放大再算梯度，更新前再除回来；遇到溢出，就跳过这次更新。 |
| [12. 优化器更新与学习率调度 · 小白入门](./12_Optimizer_Updates_and_Learning_Rate_Scheduling.md) | [notebook](../12_Optimizer_Updates_and_Learning_Rate_Scheduling.md) | Medium | CPU | 学习率决定每次更新参数时走多大一步。这本 notebook 教你按训练进度调整它：先从小到大热身，中间保持不变，最后逐步减小，并且只在参数真正更新后才往前走一步。 |
| [55. 训练更新动力学 · 小白入门](./55_Training_Update_Dynamics.md) | [notebook](../55_Training_Update_Dynamics.md) | Hard | CPU 主线，GPU 可选 | 一次训练更新到底把参数挪了多远，要由梯度大小、裁剪和学习率一起决定。这本 notebook 教你算出这几个量，并据此判断一次更新是数值失效、没有信号、被大幅压缩，还是可以继续观察。 |
| [19. 激活检查点 · 小白入门](./19_Activation_Checkpointing.md) | [notebook](../19_Activation_Checkpointing.md) | Hard | CPU 主线，GPU 可选 | 训练时为了算梯度，要把前向传播的中间结果都留在显存里。激活检查点只留下每一段的输入，等反向传播用到时再把这一段重新算一遍，用多算一次换更少的显存占用。 |
| [42. 激活卸载 · 小白入门](./42_Activation_Offload.md) | [notebook](../42_Activation_Offload.md) | Hard | CPU 主线，GPU 可选 | 反向传播要用的中间结果太多、显存放不下时，激活卸载把暂时不用的搬到 CPU 内存，等用到再搬回来。这本 notebook 教你在 CPU 上做一次预算：搬哪些、搬多久、值不值得。 |
| [43. 统一训练内存账本 · 小白入门](./43_Unified_Training_Memory_Ledger.md) | [notebook](../43_Unified_Training_Memory_Ledger.md) | Medium | CPU | 训练时显存的峰值，取决于某个阶段有哪些东西同时留在 GPU 上。这本 notebook 教你给每种状态记下多大、在哪个设备、从哪个阶段到哪个阶段，再按阶段算出峰值出现在哪、由谁造成。 |
| [56. 持续预训练与能力保留 · 小白入门](./56_Continual_Pretraining_and_Retention.md) | [notebook](../56_Continual_Pretraining_and_Retention.md) | Hard | CPU 主线，GPU 可选 | 持续预训练是拿一个新领域的文本接着训练已有模型。这本 notebook 教你同时检查两件事：模型在新领域上有没有变好，原有能力有没有变差，然后决定继续、调整还是回退。 |

## 2.4 监督微调与参数适配

从判断要不要微调开始，依次过数据、监督目标、参数更新和保存。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [33. 后训练准入与阶段选择 · 小白入门](./33_Fine_Tuning_Readiness.md) | [notebook](../33_Fine_Tuning_Readiness.md) | Medium | CPU | 动手训练模型之前，要先判断该不该训练、该用哪种训练。这本 notebook 给出一套检查顺序：先确认失败是稳定出现的，再试不改参数的办法，最后按手头的反馈选出下一训练阶段。 |
| [32. SFT 数据契约与泄漏审计 · 小白入门](./32_Data_Engineering_for_SFT.md) | [notebook](../32_Data_Engineering_for_SFT.md) | Medium | CPU | 准备用示范回答训练模型之前，这本 notebook 教你先给这批数据做体检：每条记录的字段齐不齐，训练集里有没有重复，有没有混进本来留作评测的题目。 |
| [PT-SYNTHETIC-DATA. 合成指令生成与验证准入 · 小白入门](./PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.md) | [notebook](../PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.md) | Medium | CPU | 让模型自己写出训练数据之后，这本 notebook 教你给每条新生成的样本做筛查：记清它是怎么产生的，和已有题目像不像，回答有没有通过验证，最后分成接纳、复核、拒绝三类。 |
| [13. SFT 监督边界与损失 · 小白入门](./13_SFT_Training_Loop.md) | [notebook](../13_SFT_Training_Loop.md) | Medium | CPU | 监督微调是用“提示加标准回答”的数据继续训练模型。这本 notebook 处理一件容易做错的事：只让回答部分参与学习，提示和填充部分不算。 |
| [14. 低秩适配教程 · 小白入门](./14_LoRA_Tutorial.md) | [notebook](../14_LoRA_Tutorial.md) | Medium | CPU 主线，GPU 可选 | 微调大模型时不必更新全部参数。LoRA 把原来的权重冻结住，只在旁边加两个小矩阵来学改动；训练完还能把改动并回原权重。 |
| [15. SFT 有效更新与训练闭环 · 小白入门](./15_End_to_End_Fine_Tuning_Experiment.md) | [notebook](../15_End_to_End_Fine_Tuning_Experiment.md) | Medium | CPU 主线，GPU 可选 | 训练时把一批数据拆成几小份依次算梯度，攒够了再更新一次参数，这叫一次有效更新。这本 notebook 教你把评估、参数更新、学习率和训练记录都对齐到同一个计数上。 |
| [30. 长上下文微调机制 · 小白入门](./30_Long_Context_Fine_Tuning.md) | [notebook](../30_Long_Context_Fine_Tuning.md) | Medium | CPU 主线，GPU 可选 | 训练样本有长有短，超过模型能处理的长度时，直接截断会切掉要学的回答。这本 notebook 教你先按保留下多少回答来决定截断还是拒绝，再把几条短样本安全地拼进同一个窗口。 |
| [31. LoRA 变体缩放与预算 · 小白入门](./31_LoRA_Variants_Theory.md) | [notebook](../31_LoRA_Variants_Theory.md) | Medium | CPU 主线，GPU 可选 | 微调大模型时可以只训练一小块新增参数，这叫 LoRA，它有好几种改法。这本 notebook 教你算清每种改法训练多少参数、更新被放大多少倍，以及参数上限内能用多大的秩。 |

## 2.5 偏好优化与对齐

从偏好数据和评测开始，按手头的反馈信号选择 DPO、KTO、PPO、GRPO 或 RLOO。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [50. 偏好数据与对齐评测 · 小白入门](./50_Preference_Data_and_Evaluation.md) | [notebook](../50_Preference_Data_and_Evaluation.md) | Medium | CPU | 训练出一个新模型后，怎样判断它真的比原来好？这本 notebook 先检查“两个回答里选一个”的训练数据能不能用，再用覆盖率、胜率等数字决定新模型是通过、复核还是拒绝。 |
| [PT-REWARD-SIGNAL. 奖励、价值与优势信号 · 小白入门](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) | [notebook](../PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) | Hard | CPU 主线，GPU 可选 | 模型答完一道题只得到一个总分。这本 notebook 教你把总分变成每个词元都能用的训练信号，再比较两种“比基线好多少”的算法：PPO 和 GRPO。 |
| [17. 直接偏好优化损失 · 小白入门](./17_DPO_Loss_Tutorial.md) | [notebook](../17_DPO_Loss_Tutorial.md) | Hard | CPU 主线，GPU 可选 | 你已经准备好“同一个问题、两个回答、哪个更好”的数据，DPO 直接拿它训练模型，让模型更偏向好的回答。这本 notebook 带你把这个比较写成一个可以求梯度的损失。 |
| [PT-KTO. 二元反馈对齐 · 小白入门](./PT-KTO_Binary_Feedback_Alignment.md) | [notebook](../PT-KTO_Binary_Feedback_Alignment.md) | Hard | CPU | DPO 要求同一条数据的两个回答配成对，KTO 不用：每条回答只要一个“好”或“不好”的标记就能训练。这本 notebook 教你把这个标记变成损失，并分点赞、点踩两类做好记录。 |
| [PT-REWARD-MODEL. 奖励模型训练、校准与过度优化 · 小白入门](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md) | [notebook](../PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md) | Hard | CPU 主线，GPU 可选 | 奖励模型给每个回答打一个分数，后面的训练靠这个分数判断好坏。这本 notebook 教你用“两个回答哪个更好”的数据训练它，再检查排序对不对、打分有多自信、分数高是否质量真的好。 |
| [16. RLHF / PPO 资源与状态 · 小白入门](./16_RLHF_PPO_Memory.md) | [notebook](../16_RLHF_PPO_Memory.md) | Hard | CPU | PPO 是一种让模型根据奖励小幅调整生成倾向的训练方法。这本 notebook 先讲清一轮 PPO 要用到哪几个模型，再带你实现其中最核心的一步：把每次更新限制在安全范围内。 |
| [18. 组相对策略优化与可验证奖励 · 小白入门](./18_GRPO_Loss_Tutorial.md) | [notebook](../18_GRPO_Loss_Tutorial.md) | Medium-Hard | CPU 主线，GPU 可选 | 让模型对同一个问题写出几个回答并分别打分，组相对策略优化根据每个回答比同组平均分好多少，来决定提高还是降低它出现的概率，并且限制每次改动的幅度。 |
| [PT-RLOO. 留一基线策略优化 · 小白入门](./PT-RLOO_Leave_One_Out_Policy_Optimization.md) | [notebook](../PT-RLOO_Leave_One_Out_Policy_Optimization.md) | Hard | CPU | 给同一个问题让模型写出几个回答并分别打分，留一基线策略优化在评价其中一个回答时，把其他回答的平均分当作参照，再克制地更新模型。 |
| [PT-ROLLOUT-LIFECYCLE. 轨迹与策略新鲜度 · 小白入门](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | [notebook](../PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | Hard | CPU 主线，GPU 可选 | 在线训练时模型一边生成数据、一边更新自己。这本 notebook 给每条生成记录记下“由哪个版本生成、处理到哪一步”，只放行还合用的数据，等所有进程确认后再切换到新模型。 |
| [PT-AGENTIC-RL. 环境轨迹与奖励归因 · 小白入门](./PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md) | [notebook](../PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md) | Hard | CPU | 模型调用工具、分几步完成任务时，成功是哪一步的功劳要算清楚。这本 notebook 用一个两步的小环境走通这条多步记录，再判断该继续用现成的成功案例训练，还是可以开始在线探索。 |
| [PT-RL-SYSTEM. 采样、训练与奖励系统 · 小白入门](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) | [notebook](../PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) | Hard | CPU 主线，GPU 可选 | 在线训练要经过生成、打分、更新几个环节，一个环节快不代表整体快。这本 notebook 把各环节的处理能力换算成同一种单位，找出瓶颈，再判断一套部署方案能不能用。 |
| [51. 在线 DPO · 小白入门](./51_Online_DPO.md) | [notebook](../51_Online_DPO.md) | Hard | CPU | 在线 DPO 一边收集“哪个回答更好”的反馈，一边一轮一轮地更新模型。这本 notebook 规定每轮用哪些反馈、怎样让每轮的样本清单可以复现，以及更新后何时采用、何时退回旧模型。 |

## 2.6 推理执行与生成加速

先看一次注意力怎样执行，再看解码策略，以及怎样用草稿模型减少目标模型的调用次数。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [20. FlashAttention 模拟 · 小白入门](./20_FlashAttention_Sim.md) | [notebook](../20_FlashAttention_Sim.md) | Hard | CPU 主线，GPU 可选 | 注意力计算会先生成一张很大的中间分数表，序列越长它越大。这本 notebook 教你把这张表切成小块逐块计算，并且得到和完整计算一样的结果。 |
| [INF-EXECUTION-DISPATCH. 推理融合与后端分派 · 小白入门](./INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md) | [notebook](../INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md) | Medium | CPU 主线，GPU 可选 | 模型里相邻的几个计算步骤可以合并成一个来执行，但不是每次都行。这本 notebook 教你找出能合并的片段，再判断负责执行的一方接不接受。 |
| [21. 解码策略 · 小白入门](./21_Decoding_Strategies.md) | [notebook](../21_Decoding_Strategies.md) | Medium | CPU | 模型生成文字时，每一步都给词表里每个词打分，再挑一个作为下一个词。这本 notebook 教你在挑之前怎样调分数、怎样缩小候选范围，让输出更确定或者更多样。 |
| [23. 投机解码 · 小白入门](./23_Speculative_Decoding.md) | [notebook](../23_Speculative_Decoding.md) | Hard | CPU 主线，GPU 可选 | 大模型生成文字，通常每跑一遍只能吐出一个词。这本 notebook 教一种可能提速的办法：先让小模型连续猜几个词，再让大模型一次检查，能接受的留下，不能接受的就地改对。 |
| [35. 原生多 Token 预测 · 小白入门](./35_Multi_Token_Decoding.md) | [notebook](../35_Multi_Token_Decoding.md) | Hard | CPU + GPU | 大模型生成时通常一次只出一个词。这本 notebook 讲一种把“猜词模块”直接做进大模型里的办法：内部先给出几个候选词，再逐个核对，对得上几个就用几个。 |
| [INF-EAGLE. 特征草稿器与候选树 · 小白入门](./INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md) | [notebook](../INF-EAGLE_Feature_Drafter_and_Candidate_Tree.md) | Hard | CPU | 小模型帮忙猜词时，会重复算一遍大模型已经算过的东西。这本 notebook 讲一种叫 EAGLE 的做法：直接用大模型的中间结果来猜，把候选排成一棵树，再让大模型检查。 |
| [INF-PARALLEL-DRAFTING. 并行草稿与置信度调度 · 小白入门](./INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) | [notebook](../INF-PARALLEL-DRAFTING_DFlash_and_DSpark.md) | Hard | CPU | 让小模型一个接一个地猜后面的词，猜得越多等得越久。这本 notebook 讲一种一次把几个位置一起猜出来的办法，再用前一个词轻轻修正后一个，并把没把握的候选拦下来。 |
| [INF-PROPOSAL. 动态候选生成与验证预算 · 小白入门](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) | [notebook](../INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) | Hard | CPU 主线，GPU 可选 | 投机解码先让一个模型快速猜出几个词元，再让负责输出的模型一次性检查。这本 notebook 教你按最近猜中的情况调整每轮猜多少个，并把省下的时间和多花的成本算清楚。 |
| [INF-MOE-EXECUTION. MoE 推理中的词元分组与专家执行 · 小白入门](./INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md) | [notebook](../INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md) | Hard | CPU 主线，GPU 可选 | 混合专家模型里，每个词元只会交给少数几个“专家”网络处理。这本 notebook 教你把发给同一个专家的词元排在一起算，再按原顺序和各自的权重把结果加回去。 |

## 2.7 KV Cache 与 Serving 系统

KV Cache 怎样存、怎样复用，多请求并发时每轮接纳谁、怎样调度。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [INF-REQUEST-LIFECYCLE. 推理请求生命周期与核心指标 · 小白入门](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md) | [notebook](../INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md) | Medium | CPU | 你给大模型发一条请求，从提交到回答结束要经过排队、准备、出第一个词、继续输出几步。这本 notebook 教你把每一步的时间点记下来，再算出各个延迟指标。 |
| [22. vLLM 分页注意力 · 小白入门](./22_vLLM_PagedAttention.md) | [notebook](../22_vLLM_PagedAttention.md) | Hard | CPU | 大模型生成时要为每个请求保存一份越来越长的缓存。这本 notebook 教你把缓存切成固定大小的小块，用到多少分多少，并用一张表记录每个请求的缓存放在哪些块里。 |
| [37. KV Cache 生命周期与容量控制 · 小白入门](./37_KV_Cache_Lifecycle_and_Capacity_Control.md) | [notebook](../37_KV_Cache_Lifecycle_and_Capacity_Control.md) | Hard | CPU 主线，GPU 可选 | 推理时保存的 KV Cache 从分配、追加、共享到释放，每一步都占着有限的块。这本 notebook 教你用引用数和块数记账，并在容量不够时判断哪些缓存可以回收。 |
| [24. Prefix Cache 匹配与复用 · 小白入门](./24_Prefix_Cache_Matching_and_Reuse.md) | [notebook](../24_Prefix_Cache_Matching_and_Reuse.md) | Hard | CPU | 很多请求的开头是一样的，比如同一段系统提示词。前缀缓存把这段开头算过的结果留下来，后面的请求直接复用。这本 notebook 教你判断一个请求的开头能复用多少。 |
| [34. SGLang 基数注意力 · 小白入门](./34_SGLang_RadixAttention.md) | [notebook](../34_SGLang_RadixAttention.md) | Hard | CPU | 很多请求的开头相同，重复算这些开头很浪费。这本 notebook 教你用一棵树把登记过的开头串起来，新请求沿着树找最长的一段直接复用，剩下的再算。 |
| [INF-ITERATION-BUDGET. Serving 迭代成本与批次容量上界 · 小白入门](./INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) | [notebook](../INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) | Hard | CPU 主线，GPU 可选 | 大模型在线服务是一轮一轮处理请求的。这本 notebook 教你算清一轮要花多久、一批最多能放几个请求，以及批次加到多大就不值得再加。 |
| [36. 解码调度 · 小白入门](./36_Decode_Scheduling.md) | [notebook](../36_Decode_Scheduling.md) | Hard | CPU | 多个请求同时等大模型生成时，每一步只能推进其中一个。这本 notebook 写一个小模拟器：决定谁先走、容量不够时暂停谁，以及暂停后怎样接着做。 |
| [INF-CONTINUOUS-BATCHING. 连续批处理 · 小白入门](./INF-CONTINUOUS-BATCHING_Continuous_Batching.md) | [notebook](../INF-CONTINUOUS-BATCHING_Continuous_Batching.md) | Hard | CPU 主线，GPU 可选 | 一批请求一起生成，短的先结束就会留下空位。连续批处理在每轮结束时清走已结束的请求、补进新请求。这本 notebook 教你实现一次这样的补位。 |
| [38. Prefill/Decode 调度 · 小白入门](./38_Prefill_Decode_Scheduling.md) | [notebook](../38_Prefill_Decode_Scheduling.md) | Medium | CPU 主线，GPU 可选 | 正在生成的请求，会被新来的长输入拖慢。这本 notebook 把长输入切成小块，并在每一轮里决定先照顾解码还是先处理输入。 |
| [INF-SERVING-OVERLAP. 服务重叠与状态交接 · 小白入门](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md) | [notebook](../INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.md) | Hard | CPU | 把数据搬运和计算同时进行可以省时间，但能省多少要看计算能盖住多少搬运。这本 notebook 教你算真正省下的时间，并判断交接过来的状态能不能安全使用。 |
| [39. 异构 PD 与服务分层 · 小白入门](./39_Hetero_PD_and_Serving_Tiers.md) | [notebook](../39_Hetero_PD_and_Serving_Tiers.md) | Hard | CPU | 把预填充和解码放到不同的机器上跑，还要决定请求去哪、中间状态是搬过去还是重算一遍。这本 notebook 把这些判断写成一条决策链。 |
| [INF-AFD. Attention–FFN 解耦 · 小白入门](./INF-AFD_Attention_FFN_Disaggregation.md) | [notebook](../INF-AFD_Attention_FFN_Disaggregation.md) | Hard | CPU-first，GPU/backend optional | 模型每一层都有注意力和前馈两部分，它们吃资源的方式不一样。这本 notebook 把这两部分拆到不同的进程组上，教你怎么算拆开后多出来的传输量、等待时间和进程分配。 |

## 2.8 量化与低比特适配

量化怎样表示、范围怎样选，以及权重、激活、KV Cache 各自怎样压缩。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [25. W8A16 量化 · 小白入门](./25_Quantization_W8A16.md) | [notebook](../25_Quantization_W8A16.md) | Medium | CPU 主线，GPU 可选 | 量化是把模型权重从小数换成“小整数加一个缩放因子”来保存。这本 notebook 带你手写一个用 8 位整数存权重的线性层，并看清它省在哪里、不省在哪里。 |
| [52. 量化校准范围与裁剪 · 小白入门](./52_Quantization_Calibration_and_Error.md) | [notebook](../52_Quantization_Calibration_and_Error.md) | Medium | CPU | 量化前要先定一个范围：范围太大，普通值被挤得很粗糙；范围太小，超出的值会被截断。这本 notebook 教你用数据挑出一个两头都照顾到的范围。 |
| [26. QLoRA 与 4-bit 量化 · 小白入门](./26_QLoRA_and_4bit_Quantization.md) | [notebook](../26_QLoRA_and_4bit_Quantization.md) | Hard | CPU 主线，GPU 可选 | QLoRA 把模型原有的权重压成 4 位保存、不再改动，只在旁边加一小份可训练的参数来学新任务。这本 notebook 带你走一遍它的前向计算，并分清哪些东西在训练、哪些只是在占显存。 |
| [QUANT-QAT. 量化感知训练 · 小白入门](./QUANT-QAT_Quantization_Aware_Training.md) | [notebook](../QUANT-QAT_Quantization_Aware_Training.md) | Hard | CPU-first，GPU 可选 | 量化感知训练是在训练时就让模型提前见到量化带来的误差，再用一个近似的梯度照常更新权重。这本 notebook 带你写出其中的范围统计、误差模拟和梯度传递。 |
| [40. GPTQ 与 AWQ 权重量化 · 小白入门](./40_GPTQ_and_AWQ_Weight_Quantization.md) | [notebook](../40_GPTQ_and_AWQ_Weight_Quantization.md) | Hard | CPU 主线，GPU 可选 | GPTQ 和 AWQ 是两种把权重压到 4 位的方法。它们都会先看一批真实输入在这一层长什么样，再判断哪些权重的误差更要紧。这本 notebook 在 CPU 上写出它们的简化版。 |
| [53. 激活量化与 SmoothQuant · 小白入门](./53_Activation_Quantization_and_SmoothQuant.md) | [notebook](../53_Activation_Quantization_and_SmoothQuant.md) | Hard | CPU 主线，GPU 可选 | SmoothQuant 在量化前把激活里特别大的通道调小，把对应的权重调大，这一层的输出保持不变。这本 notebook 带你实现这个变换，并挑选调整的力度。 |
| [QUANT-MIXED-BIT. 混合位宽分配 · 小白入门](./QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) | [notebook](../QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) | Hard | CPU-first，GPU 可选 | 混合位宽给不同的层安排不同的量化位宽：怕压的层留高精度，耐压的层用低位。这本 notebook 教你在容量预算内，决定哪些层值得多花字节去升级。 |
| [41. FP8 运行时低精度路径 · 小白入门](./41_FP8_Runtime_Low_Precision_Paths.md) | [notebook](../41_FP8_Runtime_Low_Precision_Paths.md) | Hard | CPU 主线，GPU 可选 | FP8 是只用 8 个二进制位保存小数的格式。这本 notebook 教你把一组数缩放进 FP8 的表示范围，并判断一次计算是否真的用上了 FP8。 |
| [54. KV Cache 量化策略 · 小白入门](./54_KV_Cache_Quantization_Strategies.md) | [notebook](../54_KV_Cache_Quantization_Strategies.md) | Hard | CPU 主线，GPU 可选 | 模型生成时要为每个词元保存两份状态，也就是 KV Cache，对话越长占显存越多。这本 notebook 教你用更少的位保存它们：键和值各用一种量化方式，最新的词元先保持高精度。 |

## 2.9 通信与并行

把模型或数据切到多张卡上：切什么、切完怎样通信、怎样选组合。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [27. ZeRO 优化器模拟 · 小白入门](./27_ZeRO_Optimizer_Sim.md) | [notebook](../27_ZeRO_Optimizer_Sim.md) | Hard | CPU | 数据并行训练时，每张 GPU 都保存一份完整的训练状态，很浪费显存。ZeRO 把这些状态切开分给各张卡，这本 notebook 在 CPU 上模拟这个过程。 |
| [28. Pipeline 并行微批次 · 小白入门](./28_Pipeline_Parallelism_MicroBatch.md) | [notebook](../28_Pipeline_Parallelism_MicroBatch.md) | Hard | CPU | 模型太大放不进一张 GPU 时，可以把层切给几张卡接力计算。接力时前面的卡在工作、后面的卡在等，这本 notebook 教你把一批数据切成小份来减少这种等待。 |
| [29. Tensor 并行模拟 · 小白入门](./29_Tensor_Parallelism_Sim.md) | [notebook](../29_Tensor_Parallelism_Sim.md) | Hard | CPU | 一层太大、一张 GPU 扛不住时，可以把它的权重矩阵切成几块，分给几张卡一起算。这本 notebook 教你两种切法，以及怎样让相邻两层接得上、少通信。 |
| [PAR-CONTEXT. 上下文与序列并行 · 小白入门](./PAR-CONTEXT_Context_and_Sequence_Parallelism.md) | [notebook](../PAR-CONTEXT_Context_and_Sequence_Parallelism.md) | Hard | CPU | 注意力要每个词元都看到全部词元。上下文并行把长序列切成几段分给几张卡，这本 notebook 教你切开以后怎样算出和不切一样的结果，以及要为此搬多少数据。 |
| [46. MoE 专家并行 · 小白入门](./46_MoE_Expert_Parallel.md) | [notebook](../46_MoE_Expert_Parallel.md) | Hard | CPU | 混合专家模型里每个词元只经过少数几个专家，而专家分散在不同的 GPU 上。这本 notebook 教你算清楚：多少词元挤不进专家，多少词元要跨卡传输。 |
| [47. NCCL 通信剖析 · 小白入门](./47_Communication_Profiling_with_NCCL.md) | [notebook](../47_Communication_Profiling_with_NCCL.md) | Hard | CPU | 多张显卡一起训练时要互相传数据，等数据的时间会拖慢每一步。这本 notebook 教你从一条时间线里算出：哪类通信没有被计算盖住，真正在拖时间。 |
| [PAR-REPLICA. Serving 副本与请求路由 · 小白入门](./PAR-REPLICA_Serving_Replica_Routing.md) | [notebook](../PAR-REPLICA_Serving_Replica_Routing.md) | Hard | CPU | 线上服务常常同时开着好几份一样的模型，每个请求都要决定送去哪一份。这本 notebook 教你三种分法，以及怎样在“排队长短”和“是否已有缓存”之间取舍。 |
| [48. 通信热点与缓解策略 · 小白入门](./48_Communication_Hotspots_and_Mitigation.md) | [notebook](../48_Communication_Hotspots_and_Mitigation.md) | Medium | CPU | 多卡程序慢在通信上时，原因可能不止一种。这本 notebook 教你把通信记录按类型汇总，找出最该处理的那一类，再按症状挑一个缓解动作，并决定下一步是接受还是继续调。 |
| [49. 并行策略选型 · 小白入门](./49_Parallelism_Strategy_Selection.md) | [notebook](../49_Parallelism_Strategy_Selection.md) | Medium | CPU | 模型装不下或者跑不快时，有好几种把活分给多张卡的办法，各自解决不同的麻烦。这本 notebook 教你从观测到的信号出发挑出最合适的一种，并列出下一步要复测什么。 |

## 2.10 项目验证与工程收口

项目页把前面各组的方法放到固定的工作负载下验证，产出可比较、可复现的结果。按项目链排列，每条链内部有先后依赖，链与链之间可以独立阅读。

### 大模型架构演进

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [60. Decoder Block 稳定性基准 · 小白入门](./60_Decoder_Block_Stability_Benchmark.md) | [notebook](../60_Decoder_Block_Stability_Benchmark.md) | Hard | CPU-first；GPU 可选 | Decoder Block 是大模型里重复堆叠的基本单元，里面的归一化、激活函数可以有不同的选法。这本 notebook 教你怎样公平地比较这些选法：一次只改一个地方，再检查结果。 |
| [78. Attention 访问模式基准 · 小白入门](./78_Attention_Access_Pattern_Benchmark.md) | [notebook](../78_Attention_Access_Pattern_Benchmark.md) | — |  | 注意力默认让每个词元读遍前面所有词元，序列越长计算越多。这本 notebook 把“只读哪些历史”写成一张可见表，再数清每种规则省了多少计算、丢了多少远处的信息。 |
| [71. MLA 与 KV Cache 结构基准 · 小白入门](./71_MLA_KV_Cache_Architecture_Benchmark.md) | [notebook](../71_MLA_KV_Cache_Architecture_Benchmark.md) | Hard | CPU-first；GPU/backend 可选 | MLA 是一种把注意力的历史记录压小了再保存的办法。这本 notebook 教你算清它和常见做法 GQA 各要存多少状态，再判断一个 MLA 模型有没有资格进入下一步评测。 |
| [77. 长序列记忆架构基准 · 小白入门](./77_Long_Sequence_Memory_Architecture_Benchmark.md) | [notebook](../77_Long_Sequence_Memory_Architecture_Benchmark.md) | Hard | CPU-first；GPU 可选 | 处理长文本时，模型把读过的内容存成什么，决定了它记不记得住、要多占多少显存。这本 notebook 用一张状态账本和一个小测试，比较三种保存历史的办法。 |
| [61. 模型架构审计 · 小白入门](./61_Model_Architecture_Exploration.md) | [notebook](../61_Model_Architecture_Exploration.md) | Hard | CPU-first；真实 checkpoint 配置审计可选 | 看到一份模型配置，怎样判断它和对照模型差在哪里、这个差别改变了什么、还缺哪些证据？这本 notebook 教你把一次结构改动写成一张审计卡，并给出下一步。 |

### 后训练优化

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [64. SFT 数据质量项目 · 小白入门](./64_SFT_Data_Quality_Project.md) | [notebook](../64_SFT_Data_Quality_Project.md) | Hard | CPU | 一批训练数据能读进来，不等于能拿去训练。这本 notebook 教你按固定的门槛判断一批数据能不能用、清洗后的版本能不能替换旧版本，以及哪个版本可以交付。 |
| [62. SFT 训练交付项目 · 小白入门](./62_Instruction_Fine_Tuning_Project.md) | [notebook](../62_Instruction_Fine_Tuning_Project.md) | Hard | CPU + 单卡 GPU | 模型微调训练完，不能只凭“损失降了”就把结果交出去。这本 notebook 带你写三道检查：能不能开始训练、中断后能不能接着练、训练产物能不能交付。 |
| [63. LoRA 变体选型项目 · 小白入门](./63_LoRA_Variants_Benchmark.md) | [notebook](../63_LoRA_Variants_Benchmark.md) | Hard | CPU 主线，GPU 可选 | LoRA 微调的秩怎么选？这本 notebook 教你一套选法：固定其他条件，逐项检查质量和资源，用相同随机种子比较损失差，再选代价更低的配置。 |
| [65. QLoRA 选型项目 · 小白入门](./65_QLoRA_Selection_Project.md) | [notebook](../65_QLoRA_Selection_Project.md) | Hard | CPU + 单卡 GPU | QLoRA 把 LoRA 微调里不训练的权重压成 4 位来存。这本 notebook 教你判断它能否替代普通 LoRA：先确认其他条件相同，再看证据够不够。 |
| [84. DPO 偏好优化项目 · 小白入门](./84_DPO_Preference_Project.md) | [notebook](../84_DPO_Preference_Project.md) | Hard | CPU + 单卡 GPU | DPO 训练让模型更偏向人认为更好的回答。这本 notebook 教你判断它的结果能不能交付：数据和评测要隔开，质量、资源和产物要同时过关。 |
| [85. GRPO 组内奖励优化项目 · 小白入门](./85_GRPO_Groupwise_Alignment_Project.md) | [notebook](../85_GRPO_Groupwise_Alignment_Project.md) | Hard | CPU + 单卡 GPU | GRPO 让模型对同一道能自动判分的题生成一组回答，再按组内谁好谁坏来更新。这本 notebook 教你判断这样的训练结果能不能交付。 |
| [86. 在线后训练生命周期与恢复 · 小白入门](./86_DPO_Online_Benchmark.md) | [notebook](../86_DPO_Online_Benchmark.md) | Hard | CPU + 真实产物 | 84 或 85 节训练出的候选，交给下一环评测前还要过一道检查：是否来自当前基线、证据是否还有效、出了问题能否恢复。这本 notebook 就是这道检查。 |

### 推理优化

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [66. 单卡推理测量基线 · 小白入门](./66_Inference_Performance_Comparison.md) | [notebook](../66_Inference_Performance_Comparison.md) | Hard | CPU 主线，GPU 可选 | 测大模型的推理速度，一组请求跑完有输出，还不算一份能拿来比较的基线。这本 notebook 教你逐条核对请求记录和汇总，判断一份测量结果能不能引用。 |
| [68. 投机解码机制选型基准 · 小白入门](./68_Speculative_Decoding_Benchmark.md) | [notebook](../68_Speculative_Decoding_Benchmark.md) | Hard | CPU 主线，GPU 可选 | 投机解码先猜出几个后续词元，再交给目标模型验证，希望生成得更快。这本 notebook 教你判断一种猜法能不能启动、是否真的更快，以及能不能采用。 |
| [69. 前缀缓存基准 · 小白入门](./69_Prefix_Caching_Benchmark.md) | [notebook](../69_Prefix_Caching_Benchmark.md) | Hard | CPU 主线，GPU 可选 | 前缀缓存把已经算过的开头部分存起来，后面的请求开头相同时直接复用。这本 notebook 教你比较开关缓存的效果，并判断该不该采用。 |
| [70. 推理服务调度基准 · 小白入门](./70_Serving_Scheduler_Benchmark.md) | [notebook](../70_Serving_Scheduler_Benchmark.md) | Hard | CPU 主线，GPU 可选 | 推理服务里请求要排队，排队规则决定谁先执行。这本 notebook 用一个小队列模型比较几种排队规则，并判断新的规则值不值得采用。 |
| [72. 单卡推理部署与 Backend 对比项目 · 小白入门](./72_Multi_Model_Deployment_Comparison.md) | [notebook](../72_Multi_Model_Deployment_Comparison.md) | Hard | CPU 完成候选审计与决策；NVIDIA 单 GPU 完成可选 backend benchmark | 把一个模型部署到一张显卡上，要同时确认模型身份、显存够不够、各项指标达标。这本 notebook 教你把这些检查串起来，给出采用、调优或拒绝的结论。 |

### 性能优化

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [73. 训练性能分析 · 小白入门](./73_Training_Performance_Analysis.md) | [notebook](../73_Training_Performance_Analysis.md) | Hard | CPU 可完成代码与模板验证；有 GPU 时可补充真实运行数据 | 训练太慢或者显存不够时，先别急着改。这本 notebook 教你把一次训练的耗时和显存记成一份基线，再从里面读出瓶颈可能在哪里。 |
| [76. Activation / Checkpoint / Offload 对比项目 · 小白入门](./76_Activation_Checkpoint_Offload_Benchmark.md) | [notebook](../76_Activation_Checkpoint_Offload_Benchmark.md) | Hard | CPU 可完成正确性验证；GPU 用于策略 benchmark | 训练时激活值要占显存，存不下时可以少存、用到时重算，也可以先搬到内存里。这本 notebook 教你在不同预算下比较这些做法，并检查结论稳不稳。 |
| [75. AdamW 执行路径基准 · 小白入门](./75_Optimizer_Execution_Path_Benchmark.md) | [notebook](../75_Optimizer_Execution_Path_Benchmark.md) | Hard | CPU 验证比较规则；单 GPU 测量执行路径 | 训练时每一步都要更新一次参数，这个更新有好几种执行写法，快慢不同。这本 notebook 教你判断某种写法值不值得换：整步是否真的变快，显存和参数结果有没有变差。 |
| [74. profiling 驱动的候选准入与交接 · 小白入门](./74_Profiling_Driven_End_to_End_Optimization.md) | [notebook](../74_Profiling_Driven_End_to_End_Optimization.md) | Hard | CPU 可完成报告收口；GPU 用于真实 trace 与候选复测 | 优化训练的手段测出了一堆数据，各自的实验条件不同，结论不能混着用。这本 notebook 教你给每份结论标明适用范围，并且只用完整训练的结果做项目总决策。 |

### 通信与并行

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [79. 训练状态并行基准 · 小白入门](./79_Distributed_Parallel_Benchmark.md) | [notebook](../79_Distributed_Parallel_Benchmark.md) | Hard | CPU | 用多张卡训练时，参数、梯度和优化器状态可以每张卡都存一份，也可以切开分着存。这本 notebook 教你算每种存法每张卡占多少，再用显存、吞吐和通信三个门槛选出一种。 |
| [PAR-MODEL-INTERNAL. 模型内部并行基准 · 小白入门](./PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md) | [notebook](../PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md) | Hard | CPU | 模型大到一张卡放不下时，可以按层、按矩阵、按序列三个方向切开，分给多张卡。这本 notebook 教你检查一种切法能不能排到卡上，再用显存、空转、通信和吞吐挑出一种组合。 |
| [80. MoE 专家并行基准 · 小白入门](./80_MoE_Expert_Parallel_Benchmark.md) | [notebook](../80_MoE_Expert_Parallel_Benchmark.md) | Hard | CPU | 混合专家模型里，每个词元只交给少数几个专家处理，专家分散在不同卡上时词元要跨卡传送。这本 notebook 教你汇总几种传送方式的测量结果，再按收益、容量和稳定性决定采不采用。 |
| [81. 分布式推理逻辑验证 · 小白入门](./81_Distributed_Inference_Project.md) | [notebook](../81_Distributed_Inference_Project.md) | Hard | CPU | 大模型对外提供推理服务时，可以多放几个副本，可以把模型切到多张卡，也可以把预填充和解码分开。这本 notebook 先在 CPU 上估算每种做法的时间，再判断值不值得搬到多卡上实测。 |

### 量化与压缩

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [82. 量化产物评估项目 · 小白入门](./82_Quantization_Artifact_Evaluation_Project.md) | [notebook](../82_Quantization_Artifact_Evaluation_Project.md) | Hard | CPU 练习，NVIDIA GPU 实验 | 量化后的权重文件能保存下来，不等于能拿去部署。这本 notebook 教你检查一份量化后的模型文件包：文件齐不齐、质量记录可不可信、能不能真正加载，再决定放行还是拒绝。 |
| [67. 量化推理与部署 · 小白入门](./67_Quantized_Inference_and_Deployment.md) | [notebook](../67_Quantized_Inference_and_Deployment.md) | Hard | CPU 主线，GPU 可选 | 量化后的模型能不能上线，要和它自己的浮点版本在同样条件下比一比。这本 notebook 教你把两次运行配成一对，算清快了多少、省了多少、质量掉了多少，再得出部署结论。 |
| [83. 量化服务 Profile 基准 · 小白入门](./83_Quantized_Serving_Profile_Benchmark.md) | [notebook](../83_Quantized_Serving_Profile_Benchmark.md) | Hard | CPU 主线，GPU 可选 | 一个量化模型在固定请求下跑赢了，换个请求压力未必还行。这本 notebook 教你给三种服务场景各定一个目标：先筛掉不达标的配置，再按最关心的指标挑出最好的。 |

## 专题页（未归入 2.1–2.9 组页）

下面这些页面由各自的专题负责，目前没有列入 2.1–2.9 的组页：性能分析与性能优化专题的 8 本，以及后训练优化专题的蒸馏 2 本。

| 入门页 | notebook | 难度 | 环境 | 一句话说清 |
|:---|:---|:---|:---|:---|
| [PERF-TRACE. 执行轨迹依赖与关键路径 · 小白入门](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.md) | [notebook](../PERF-TRACE_Trace_Dependencies_and_Critical_Path.md) | Medium | CPU 主线，GPU 可选 | 程序慢，往往不是算得慢，而是有地方在干等。这本 notebook 教你从一份“每件事几点开始、几点结束”的记录里，找出哪几段执行和等待连成了拖长总时间的那条链。 |
| [PERF-ROOFLINE. 算术强度与资源上界 · 小白入门](./PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md) | [notebook](../PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md) | Medium | CPU 主线，GPU 可选 | 一次矩阵乘法慢，可能是搬数据慢，也可能是算得慢。这本 notebook 教你先算“每搬 1 字节做多少次运算”，再判断这次运算卡在哪一种限制上。 |
| [PERF-REGRESSION. 配对基准与性能回归阈值 · 小白入门](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md) | [notebook](../PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md) | Medium | CPU 主线，GPU 可选 | 判断一版改动有没有让程序变慢，不能各跑一次就比时间。这本 notebook 把新旧两版在同一轮的运行配成一对，从每对延迟比例估出范围，再按事先定好的“允许慢多少”给出结论。 |
| [PERF-MEMORY-TIMELINE. 显存驻留与峰值重叠 · 小白入门](./PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.md) | [notebook](../PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.md) | Medium | CPU 主线，GPU 可选 | 训练时显存不够，常常是几块内存在同一时刻叠在了一起。这本 notebook 教你用“创建 / 释放”记录算出每个时刻占着多少字节，找出最高的那一刻是哪几块叠出来的。 |
| [PERF-IO. 模型产物、状态传输与输出回写的 I/O 关键路径 · 小白入门](./PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md) | [notebook](../PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md) | Hard | CPU 主线，GPU 可选 | 读写数据花的时间，并不都会拖慢程序，有的能和计算同时进行。这本 notebook 教你算三件事：一次读写要多久，其中多少真的被等到，写回队列满了会怎样拖住生产。 |
| [PERF-INPUT. 数据加载、装箱与输入搬运性能 · 小白入门](./PERF-INPUT_Training_Input_Pipeline_Performance.md) | [notebook](../PERF-INPUT_Training_Input_Pipeline_Performance.md) | Hard | CPU 主线，GPU 可选 | GPU 常在等数据：一批数据要先读取、整理，再从 CPU 搬到 GPU。这本 notebook 教你把短样本装进同一个盒子少补空位，再算清提前准备、边搬边算各能省多少等待。 |
| [PERF-CANDIDATE. 从资源约束到候选改动与新增代价 · 小白入门](./PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md) | [notebook](../PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md) | Hard | CPU 主线，GPU 可选 | 给模型提速的一个改动，常常在省下一种资源的同时，在别处增加开销、拉低质量。这本 notebook 教你在实测之前把这几笔账分开记、分开比，再决定丢弃、调整还是去测。 |
| [PERF-COMPOSITION. 策略相互作用与瓶颈转移 · 小白入门](./PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md) | [notebook](../PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md) | Hard | CPU 主线，GPU 可选 | 两个各自有效的提速办法，合在一起不一定更好：可能互相冲突，也可能抢同一种资源。这本 notebook 教你先检查能不能组合，再实测比较，决定保留组合还是退回单个办法。 |
| [DISTILL-OPD. 在策略蒸馏与 GKD · 小白入门](./DISTILL-OPD_On_Policy_Distillation_and_GKD.md) | [notebook](../DISTILL-OPD_On_Policy_Distillation_and_GKD.md) | Hard | CPU | 蒸馏是让大模型教小模型。这本 notebook 处理一种做法：让小模型自己写回答，再由大模型逐个位置给出反馈，并且只在回答部分计算训练信号。 |
| [88. 蒸馏学生交付项目 · 小白入门](./88_Distilled_Student_Delivery_Project.md) | [notebook](../88_Distilled_Student_Delivery_Project.md) | Hard | CPU 完成合同与决策练习；PT-DISTILL-CUDA 执行真实蒸馏实验 | 蒸馏训练后损失下降，不等于学生模型可以交付。这本 notebook 教你先固定三组对比实验的条件，再用质量、模型文件、大小和延迟的证据，决定接受、调整、补测还是拒绝。 |

编号 57、58、59、87、89 是预留占位页，没有配套入门页。
