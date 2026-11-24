# 39. Hetero PD and Serving Tiers Beginner | 异构 PD 与服务分层 · 小白入门

> 配套 notebook：[39. 异构 PD 与服务分层](./39_Hetero_PD_and_Serving_Tiers.ipynb) ｜ 难度：Hard ｜ 环境：CPU-first

## 一句话说清

把预填充和解码放到不同的机器上跑，还要决定请求去哪、中间状态是搬过去还是重算一遍。这本 notebook 把这些判断写成一条决策链。

## 为什么需要它

前面的 notebook 已经把预填充和解码分到两个逻辑池。当两个池其实不一样，比如一个算力强、一个显存大且带宽稳，光分开还不够，还要多回答三个问题：

- 这个请求该去哪个池？输入很长的更吃算力，要生成很多的更吃显存和读带宽。
- 中间的 KV 状态怎么办？搬过去要时间。链路慢的话，还不如在目标池把输入重新算一遍。
- 目标池出问题怎么办？显存快满、机器不健康、KV 格式对不上，都要有退路。

看第二个问题的一个例子：100 MB 的状态，在 200 Gbit/s 的链路上 4 毫秒就能搬完，而重算要 12 毫秒，当然选传输。换成 25 Gbit/s，要搬 32 毫秒，比重算还慢。

这本 notebook 把这些判断串成一条决策链：先给请求分类，再选交接方式，再检查资源池能力，最后定路由、回退还是拒绝。

## 先认识几个词

| 词 | 人话解释 |
|:---|:---|
| 异构 PD（heterogeneous PD） | 预填充和解码跑在不同的资源池上，而这些池的算力、显存、带宽或后端能力并不相同 |
| 资源池（pool） | 一组能跑某个阶段的资源。题目里用一张字典描述它：支持哪些阶段、显存多少、健不健康 |
| 资源画像 | 请求或资源池的特征表。请求看输入长度和生成长度，资源池看计算能力、显存、链路和后端支持 |
| KV 交接（handoff） | 预填充算出的 KV 状态交给解码一侧。有三种做法：传输过去、在目标池重算、留在原来的池 |
| 重算（recompute） | 不接收现成的 KV，而是在目标池把输入重新预填充一遍 |
| 执行兼容 / KV 兼容 | 资源池能不能跑这个请求；在此基础上，能不能直接接收现成的 KV |
| 回退池（fallback pool） | 主池不能用时，按顺序尝试的候选资源池 |

## 核心思路

输入是三样：请求的特征、资源池的特征、当前的服务压力。决策链分四步：

```text
1. 分类：看请求的输入长度和生成长度，和两个阈值比较
       输入长、生成不长  → prefill
       生成长、输入不长  → decode
       其余              → shared
2. 选交接方式：传输时间（毫秒）= 状态大小（MB）× 8 ÷ 链路（Gbit/s）
       传输时间 ≤ 重算时间 → transfer，否则 → recompute
3. 查资源池能力，分两层：
       执行兼容：健康、支持这个阶段、模型版本一致、显存够
       KV 兼容：执行兼容，并且 KV 的数据类型和布局也一致
4. 定路由：先看主池，不能执行就按顺序找回退池
       能执行但接收不了 KV → 改为 recompute
       一个能执行的都没有 → reject
```

下面把前两步算一遍，数字来自 notebook 的测试：

```python
def classify(prompt, decode, long_prompt=2048, long_decode=256):
    if prompt > long_prompt and decode <= long_decode:
        return 'prefill'
    if decode > long_decode and prompt <= long_prompt:
        return 'decode'
    return 'shared'

print(classify(3000, 64), classify(128, 512), classify(800, 128))

def handoff(link_gbps, mb, recompute_ms):
    transfer_ms = mb * 8 / link_gbps          # MB 乘 8 换成兆比特，再除以链路，得到毫秒
    return transfer_ms, 'transfer' if transfer_ms <= recompute_ms else 'recompute'

print(handoff(200, 100, 12))                  # 快链路
print(handoff(25, 100, 12))                   # 慢链路
```

```text
prefill decode shared
(4.0, 'transfer')
(32.0, 'recompute')
```

分类看的是“长输入”和“长生成”哪个占主导。两个都长或者都不长时，归为 `shared`，三类结果互斥。

换算里乘 8，是把 MB 换成兆比特。每秒 1 吉比特等于每毫秒 1 兆比特，所以兆比特数除以链路的 Gbit/s 数，得到的直接就是毫秒。

第 3、4 步用测试里的请求和资源池走一遍：

```text
请求   stage=decode  model_revision=rev-a  kv_dtype=float16  kv_layout=paged-v1  需要 8 GB
资源池 decode-a  支持 decode  rev-a  float16  paged-v1  可用 16 GB  健康

资源池不改                  → can_execute=True   can_accept_kv=True
kv_layout 改成 contiguous   → can_execute=True   can_accept_kv=False，原因里有 kv_layout_mismatch
healthy 改成 False          → can_execute=False
```

第二行就是“KV 不兼容不等于不能执行”：资源池还能跑这个请求，只是要改为重算。只有主池不能执行时，才按顺序去找回退池。

## 这本 notebook 里你要做什么

| 部分 | 讲什么 | 你要做什么 |
|:---|:---|:---|
| Step 1 | 请求压力怎样匹配异构资源，请求和资源池各要记录哪些特征 | 读 |
| Step 2 | KV 状态该传输、重算还是留在原池，三种方式各自适合什么条件 | 读，对照上面的换算 |
| Step 3 | 服务分层与回退怎样保护服务等级目标，执行兼容和 KV 兼容分两层 | 读 |
| Step 4 · TODO 1 | 根据输入与生成长度识别主要资源压力 | 在 `classify_resource_need` 里补两个条件判断，返回 `prefill`、`decode` 或 `shared` |
| Step 4 · TODO 2 | 将状态大小和链路带宽换算为传输时间 | 在 `choose_handoff_mode` 里补 2 行：`transfer_ms` 和 `mode` |
| Step 4 · TODO 3 | 分离执行兼容与 KV 兼容 | 写 `evaluate_pool_capability`：先收集执行不兼容的原因，再收集 KV 不兼容的原因，返回 `can_execute` 和 `can_accept_kv` |
| Step 4 · TODO 4 | 在主池、回退池与拒绝之间重规划 | 写 `plan_route_with_fallback`：按顺序试主池和回退池，不能执行就跳过，能执行但收不了 KV 就改为 `recompute`，都不行返回 `reject` |
| 测试 | 检查三类请求和阈值边界、快慢链路、KV 布局不匹配、不健康的资源池、KV 重算、换池和无池可用时拒绝 | 运行 |
| 参考代码与解析 | 参考答案、每个 TODO 的机制和不变量，以及真实对照实验要比较什么 | 卡住再看 |

整本 notebook 只需要 CPU。Step 4 里的传输时间和重算时间是教学用的成本估算。末尾的“真实 GPU / backend 验证出口”只说明真实对照实验的做法，由 70 和 81 号 notebook 承接。

## 容易踩的坑

- **三类分错了**：只有“输入长且生成不长”才是 `prefill`，“生成长且输入不长”才是 `decode`，其余都是 `shared`。测试里 800 和 128 归为 `shared`。
- **单位没统一就比较**：状态大小是 MB，链路是 Gbit/s。先乘 8 换成兆比特，再除以链路，得到毫秒。测试里 100 MB 在 200 Gbit/s 上是 `4.0` 毫秒。
- **把 KV 不兼容当成不能执行**：KV 的数据类型或布局对不上，资源池仍然能跑这个请求，只是要改为重算。测试里布局不匹配时 `can_execute` 仍是 `True`。
- **路由到不健康的资源池**：主池不能执行就按顺序试回退池，不能执行的候选一律跳过，并把原因记下来。
- **没有可用的池却不拒绝**：一个能执行的候选都没有时，必须显式返回 `reject`，不能悄悄选一个。

## 读完后自查

- [ ] 给一个请求的输入长度和生成长度，能判断它归 `prefill`、`decode` 还是 `shared`。
- [ ] 能算出 100 MB 的状态在 200 Gbit/s 链路上的传输时间，并说出它和重算比该选哪个。
- [ ] 能说出执行兼容要检查哪四项，KV 兼容再检查哪两项。
- [ ] 主池能执行但接收不了 KV 时，能说出路由结果里的 `execution_mode` 和 `decision`。

## 下一步

- 打开 notebook：[39. 异构 PD 与服务分层](./39_Hetero_PD_and_Serving_Tiers.ipynb)
- 建议先读：[38. Prefill/Decode 调度 · 小白入门](./38_Prefill_Decode_Scheduling_Beginner.md)、[INF-SERVING-OVERLAP. 服务重叠与状态交接 · 小白入门](./INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff_Beginner.md)
- 做完之后：[INF-AFD. Attention–FFN 解耦 · 小白入门](./INF-AFD_Attention_FFN_Disaggregation_Beginner.md)
