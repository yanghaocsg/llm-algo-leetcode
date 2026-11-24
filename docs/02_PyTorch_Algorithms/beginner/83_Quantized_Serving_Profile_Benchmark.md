# 83. Quantized Serving Profile Benchmark | 量化服务 Profile 基准 · 小白入门

> 配套 notebook：[83. 量化服务 Profile 基准](../83_Quantized_Serving_Profile_Benchmark.md) ｜ 难度：Hard ｜ 环境：CPU 主线，GPU 可选

## 一句话说清

一个量化模型在固定请求下跑赢了，换个请求压力未必还行。这本 notebook 教你给三种服务场景各定一个目标：先筛掉不达标的配置，再按最关心的指标挑出最好的。

## 为什么需要它

67 节的结论只对一个固定的请求负载成立。真实服务的压力不止一种：交互式聊天一次来不了几个请求，但要快；离线任务可以堆很多并发，只求总吞吐。同一个量化模型，换个压力可能就不达标了。

比较不同压力下的结果，容易出四种错：

- 把不同场景放在一起排名。低并发的配置延迟低，但吞吐上不去；高并发的配置吞吐高，首词元延迟却被拉长。放在一起比，说不清谁好。
- 目标不带方向。首词元延迟不能超过多少，是上限；吞吐不能低于多少，是下限。不写清方向，检查就会判反。
- 缺测被单类成功盖住。三类场景只跑完一类，不能当成整体通过。
- 拿 CPU 记录当证据。证据要来自真实后端的重复测量，CPU 或配置记录不能得到 `accept`。

看 notebook 测试里的延迟优先场景：请求数 8，并发 1，每个请求最多生成 64 个词元，重复 3 次。主指标是首词元延迟的 P95（`ttft_p95_ms`），上限 500.0；失败率的上限是 0.0。一条记录的首词元延迟是 450.0，另一条是 560.0，第二条直接出局。

这本 notebook 固定 67 节接受的量化部署资产，在同一后端上分别跑三类服务 Profile。每个 Profile 内部先查可比性和完整性，再按服务目标过滤，最后按主指标选一个。通过的场景生成稳定引用，交给 72 结合其他候选复测。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 服务 Profile | 一套请求压力加一套服务目标的组合，可以理解成一种“服务场景”。本 notebook 有延迟优先、均衡、吞吐优先三类 |
| 工作负载（workload） | 请求压力的具体数字：请求数、并发、生成上限、重复次数。并发指同时发多少个请求 |
| 首词元延迟（TTFT） | 从提交请求到收到第一个词元的时间，交互式请求最关心 |
| 单词元时间（TPOT） | 解码时平均每个后续词元的耗时 |
| P95 | 把一批测量值从小到大排，95% 的值不超过它。用来看尾延迟，也就是最慢的那部分请求有多慢 |
| SLO（服务等级目标） | 可接受的指标上限或下限。`direction` 为 `max` 表示不能超过 `limit`，为 `min` 表示不能低于 `limit` |
| 主指标 | 每个 Profile 最关心的那一个指标，候选按它排序 |

## 核心思路

输入是 67 节的部署记录、一个 Profile 的定义，以及同一 Profile 下的多条测量记录。处理分三步：

```text
lineage_ready = accepted and artifact_ready and gate_lineage_ready

workload_ready = 有 workload ID，且请求数、并发、生成上限、重复次数都是正整数
slo_ready      = SLO 非空，且每项规则的方向是 max 或 min、limit 是非负有限数
primary_ready  = 主指标出现在这个 Profile 的 SLO 里

一条记录可比   = status 是 ok，artifact、Profile、workload 与预期一致，evidence_level 是 real_project_benchmark
满足 SLO       = 每个指标：max 型 value <= limit，min 型 value >= limit
候选更好       = 主指标按方向更优；主指标持平时，SLO 余量更大
```

前三个 TODO 对应三组判断。`accepted` 是 67 节的结论为 `accept`，`artifact_ready` 是产物有编号和路径，`gate_lineage_ready` 是带着 82 节准入记录的引用和哈希。方向决定比较方式：主指标是 `max` 型（比如延迟）时取更小的值，是 `min` 型（比如吞吐）时取更大的值。余量是每项指标离门槛还有多远，按门槛大小归一化后相加。

用测试里的延迟优先 Profile 试一下：

```python
slo = {'ttft_p95_ms': {'direction': 'max', 'limit': 500.0},
       'failure_rate': {'direction': 'max', 'limit': 0.0}}

def meets_slo(metrics):
    for metric, rule in slo.items():
        value = metrics[metric]
        if rule['direction'] == 'max' and value > rule['limit']:
            return False
        if rule['direction'] == 'min' and value < rule['limit']:
            return False
    return True

records = [
    {'name': 'a', 'metrics': {'ttft_p95_ms': 450.0, 'failure_rate': 0.0}},
    {'name': 'b', 'metrics': {'ttft_p95_ms': 560.0, 'failure_rate': 0.0}},
    {'name': 'c', 'metrics': {'ttft_p95_ms': 420.0, 'failure_rate': 0.0}},
]
feasible = [r for r in records if meets_slo(r['metrics'])]
print([r['name'] for r in feasible])                                    # 先按 SLO 过滤
print(min(feasible, key=lambda r: r['metrics']['ttft_p95_ms'])['name'])  # 再按主指标选，延迟取更小值
```

```text
['a', 'c']
c
```

`b` 的首词元延迟 560.0 超过上限 500.0，被过滤掉。剩下两条都达标，主指标是延迟，取更小值，所以选 `c`。过滤和 SLO 检查由骨架里的 `select_serving_candidate` 完成，TODO 3 只写最后的选择规则。

notebook 对结论留了余地。三类压力都要完成，每类在自己的负载和 SLO 里判断。至少一类达标时的 `accept` 只适用于达标场景，不表示三类都达标，也不比较不同场景的指标排名。重复实验的 P95 取各轮 P95 的平均值，不能理解成合并所有请求后的 P95。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 沿用已接受的部署血缘：产物血缘、执行路径、环境身份、基准入口四样要固定 | 读 |
| Step 2 | 三类 Profile 各自的请求压力、主指标和约束指标 | 读，对照“先认识几个词”里的服务 Profile |
| Step 3 | 在一个 Profile 内部按什么顺序检查 SLO、选择配置 | 读，对照上面的三组判断 |
| Step 4 · TODO 1 | 已提取的接受状态、产物身份和 82 血缘 | 在 `validate_deployment_lineage` 里写 1 行：把三个条件用 `and` 连起来 |
| Step 4 · TODO 2 | workload、主指标和方向化 SLO | 在 `validate_serving_profile` 里写 3 行：`workload_ready`、`slo_ready`、`primary_ready`，前两个可以直接用骨架给出的 `_positive_workload` 和 `_valid_slo_rules` |
| Step 4 · TODO 3 | 已完成可比性和 SLO 过滤的记录 | 在 `choose_primary_record` 的循环里写 3 行：`primary_is_better`、`primary_is_tied`、`candidate_is_better` |
| 测试 | 删掉血缘后不能开始比较；非标准方向名和非法规模参数不能静默通过；延迟取更小值、吞吐取更大值，持平时取余量更大的 | 运行 |
| 参考代码与解析 | 参考答案和每个 TODO 的解析 | 卡住再看 |
| Step 5（可选） | 同一量化产物下跑延迟、均衡、吞吐三类请求压力，逐场景判断是否达标 | 需要 `INF-VLLM-011-CU128` 环境、NVIDIA GPU 和已准备好的量化产物。默认是 `RUN_STAGE = 'prepare'`，只读预检；`prepare` 和 `close` 只检查磁盘证据，不启动后端 |

Step 1–4 只需要 CPU（`BASE-CPU` 环境）。Step 5 需要 `INF-VLLM-011-CU128` 环境、NVIDIA GPU 和已准备好的量化产物，默认停在只读预检的 `prepare`，不启动后端。

## 容易踩的坑

- **血缘缺一仍开始比较**：接受状态、产物身份、82 血缘三个条件缺一不可。测试里把 `lineage` 清空，结果必须是不通过。
- **SLO 方向名随便写**：方向只能是 `max` 或 `min`。写成 `smaller` 这样的词，契约不通过，不会静默放过。规模参数必须是正整数，测试里 `repeats` 取 `True`、`0`、`-1`、`1.5` 都不通过。
- **主指标不在 SLO 里**：主指标必须出现在当前 Profile 的 SLO 中，否则它没有门槛可比。
- **跨 Profile 排名**：只拿主指标在同一 Profile 内比较，请求集、并发、生成上限和重复次数要保持一致。不同场景的指标不做排名。
- **单类达标当成全部达标**：三类压力都要完成。至少一类达标时的 `accept` 只适用于达标场景；场景间后端、硬件或精度身份变了，是 `reject`。

## 读完后自查

- [ ] 能说出三类 Profile 的名字、请求压力特点和主指标。
- [ ] 给一个 `max` 型和一个 `min` 型的 SLO，能判断一条记录是否达标。
- [ ] 能说出主指标持平时怎么选，以及 `max` 型主指标为什么取更小值。
- [ ] 能解释为什么 `accept` 不表示三类场景都达标。

## 下一步

- 打开 notebook：[83. 量化服务 Profile 基准](../83_Quantized_Serving_Profile_Benchmark.md)
- 建议先读：[67. 量化推理与部署 · 小白入门](./67_Quantized_Inference_and_Deployment.md)、[36. 解码调度 · 小白入门](./36_Decode_Scheduling.md)
