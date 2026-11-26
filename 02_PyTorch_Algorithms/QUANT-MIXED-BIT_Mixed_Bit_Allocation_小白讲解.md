# QUANT-MIXED-BIT. 混合位宽分配 — 小白讲解

> 对应 notebook：[`QUANT-MIXED-BIT_Mixed_Bit_Allocation.ipynb`](./QUANT-MIXED-BIT_Mixed_Bit_Allocation.ipynb)

## 一句话总结
`MixedBitAllocator` 把"每一层用 4 / 8 / 16 位各会带来多少误差、各占多少字节"整理成候选表，再在总字节预算内，按"每多花一字节能换回多少误差下降"逐次升级最值得升级的层。CPU 部分用三个 TODO 实现候选过滤、保护层基线和预算内贪心；GPU 部分把同一思路用到 Qwen2.5-0.5B 的 24 个 Decoder Block 上，比较统一 INT4、统一 INT8 与三档混合位宽的质量—容量权衡。

## 为什么需要它：痛点是什么
初学者做量化时容易默认"整网统一位宽"：要么全 INT8，要么全 INT4。但真实模型里各层对精度的敏感度差别很大——embedding 之后的第一层、最后一层、某些注意力投影，降精度后输出可能明显变差；另一些层降到 4 位几乎无感。统一 INT4 为了照顾少数敏感层，把不敏感的层也一起压得太狠；统一 INT8 又白白浪费了本可以省下的容量。

生活类比：搬家时只有一个固定大小的行李箱，每件衣服（层）都可以"折叠成小包（4 位）"或"原样放（8 位）"。聪明的做法不是全部硬塞，而是先量一下每件衣服压皱后有多心疼（误差），再看多占一格能少多少心疼，优先把预算花在"多占一格最划算"的那件上。

## 核心问题
- 一层的位宽候选是怎么构造的？为什么必须同时满足"后端支持"和"有实测误差"？
- 权重字节数怎么由参数量和位宽算出来？为什么是向上取整？
- "保护层"是什么？为什么要在分配前先给某些层设最低位宽？
- 总预算只够升级一次时，应该先升哪一层？"误差下降 ÷ 额外字节"怎么比较？
- GPU 实验里，怎样判断混合位宽真的比统一 INT4 更值得？NLL 比、状态字节、质量恢复率、Pareto 各是什么？

## 解决思路：一步步看懂
### 第 1 步：把优化问题写清楚
notebook 的目标函数是

$$\min_{b_l\in\mathcal{B}_l}\sum_l E_l(b_l)\quad\text{s.t.}\quad \sum_l C_l(b_l)\le B,\qquad b_l\ge b_l^{\min}.$$

其中 $l$ 是层编号，$b_l$ 是第 $l$ 层选中的位宽，$\mathcal{B}_l$ 是该层所有可执行位宽（例如 $\{4,8,16\}$），$E_l(b_l)$ 是这一层在该位宽下的误差代理（实测 MSE 或 NLL 增量），$C_l(b_l)$ 是这一层占的字节数，$B$ 是总预算，$b_l^{\min}$ 是保护规则给出的最低位宽。这个式子说的是：**在总字节不超预算、且每层不低于保护下限的前提下，让各层误差之和尽量小。** notebook 明确说这是"以各层误差相加作近似"，层与层之间其实会相互影响，所以贪心结果还要回到完整模型复验。

### 第 2 步：TODO 1——构造可执行候选
`build_options` 拿到一层的 `numel` 和 `error_by_bits`，要把每个"后端支持且有误差记录"的位宽变成字典：

```python
options = [
    {"layer": layer_name, "bits": bits,
     "bytes": (numel * bits + 7) // 8,   # 参数量 × 位宽 ÷ 8，向上取整
     "error": error}
    for bits, error in error_by_bits.items()
    if bits in self.supported_bits
]
if not options:
    raise ValueError(f"{layer_name} has no supported bit-width candidate")
return sorted(options, key=lambda option: option["bytes"])
```

三个要点：
1. **过滤**：`if bits in self.supported_bits`。`__init__` 里 `supported_bits` 被 `tuple(sorted(set(...)))` 去重排序，方便判断"这个位宽能不能执行"；返回前再按 `bytes` 从小到大排序，保证调用方拿到的是容量递增的候选。
2. **字节**：`(numel * bits + 7) // 8` 是"向上取整到字节"的标准写法，和 `_blocks_for_tokens` 里的向上取整是同一个套路。`+7` 再整除 8，保证 1 个 4 位参数也占 1 字节而不是 0。
3. **空集合报错**：如果这层一个合法候选都没有，说明配置或后端有问题，必须显式失败，不能静默返回空列表。

用测试里的 `make_mixed_bit_specs()` 手算：

| 层 | `numel` | 4 位字节 | 8 位字节 | 16 位字节 | 4 位误差 | 8 位误差 | 16 位误差 |
|---|---|---|---|---|---|---|---|
| attention | 1000 | 500 | 1000 | 2000 | 0.40 | 0.03 | 0.0 |
| mlp | 2000 | 1000 | 2000 | 4000 | 0.18 | 0.08 | 0.0 |

`numel * bits` 是总位数，除以 8 得字节。例如 attention 4 位：1000 × 4 = 4000 位，4000 ÷ 8 = 500 字节。

### 第 3 步：TODO 2——选保护层的基线
`select_initial_option(options, minimum_bits)` 做两件事：先过滤出 `bits >= minimum_bits` 的候选，再选字节最少的那一个；空集合抛 `ValueError`。

```python
eligible = [option for option in options if option["bits"] >= minimum_bits]
if not eligible:
    raise ValueError(f"no candidate satisfies minimum_bits={minimum_bits}")
initial = min(eligible, key=lambda option: option["bytes"])
```

为什么要单独拆成"过滤"和"取最小"两步？因为保护规则是**硬约束**，不能靠后面的贪心去补救。如果先把层压到 4 位再指望升级回来，预算可能根本不够；而且贪心只升不降，一旦基线违规就再也回不去。`protected_min_bits` 默认是空字典，`allocate` 用 `protected_min_bits.get(name, 0)` 取值，所以没有保护的层下限就是 0，任何候选都合法。

### 第 4 步：TODO 3——预算内的贪心升级
`allocate` 先给每层选最小可行候选作为基线，检查 `total_bytes <= budget_bytes`，否则说明连最低方案都放不下，直接 `ValueError("budget cannot satisfy the minimum feasible policy")`。然后每一轮收集所有"额外字节 > 0、误差下降 > 0、且放得进剩余预算"的升级，挑单位收益最高的一个：

```python
best_upgrade = max(
    upgrades,
    key=lambda upgrade: (
        upgrade["error_reduction"] / upgrade["extra_bytes"],  # 主排序：每字节换回的误差
        upgrade["error_reduction"],                            # 并列时看总收益
    ),
)
policy[best_upgrade["layer"]] = best_upgrade["candidate"]
total_bytes += best_upgrade["extra_bytes"]
```

循环直到没有任何可行升级。返回的字典包含 `policy`（每层最终位宽）、`total_bytes`、`total_error` 和 `details`（每层完整的候选字典）。

**为什么按"误差下降 ÷ 额外字节"而不是"误差下降"排序？** 因为预算有限，同样的误差收益如果花更少字节，就还能留钱做下一次升级。notebook 的 Step 3 举的例子是：一层多 500 字节减 0.37 误差，另一层多 1000 字节只减 0.10，前者单位收益 0.00074，后者 0.0001，显然先升前者。**这个贪心不保证全局最优**：层间误差会相互影响，而且位宽是离散的，可能"先升 A 再升 B"比"只升 C"差，所以 Step 3 的表里专门有"组合复验"一栏。

拿测试手算一遍：

- **`test_upgrade_prefers_higher_error_reduction_density`**：`supported_bits=(4, 8)`，预算 2000。基线是 attention 4 位（500）+ mlp 4 位（1000）= 1500，余 500。attention 升 8 位要 +500、误差降 0.37（密度 7.4×10⁻⁴）；mlp 升 8 位要 +1000，超预算。于是只升 attention，结果 `{"attention": 8, "mlp": 4}`。
- **`test_protected_layer_uses_minimum_bits`**：预算 3000，`protected_min_bits={"attention": 8}`。基线变成 attention 8 位（1000）+ mlp 4 位（1000）= 2000。两个升级都花 1000：attention→16 降 0.03（密度 3×10⁻⁵），mlp→8 降 0.10（密度 1×10⁻⁴），先升 mlp，总字节到 3000；attention→16 再也放不下。最终 attention 仍为 8，断言 `>= 8` 通过。
- **`test_budget_is_never_exceeded`**：默认支持 (4,8,16)，预算 2000，最后总字节正好 2000，断言 `<= 2000`。
- **`test_impossible_budget_is_rejected`**：预算 100 < 基线 1500，抛 `ValueError`。

### 第 5 步：框架一致性核对
`check_mixed_bit_framework_conformance` 把本地策略翻译成 TorchAO 的 `FqnToConfig`（按完整模块名 FQN 指定每层配置）：4 位对应 `Int4WeightOnlyConfig(group_size=32)`，8 位对应 `Int8WeightOnlyConfig()`。然后建一个 `nn.ModuleDict`，里面是 `attention` 和 `mlp` 两个 `nn.Linear(64, 64, bias=False)`，用 `quantize_(model, executable_mapping, filter_fn=None)` 只把 attention 转成 INT8、mlp 保持浮点。检查项包括：策略与本地分配器一致、两个 FQN 都登记、配置类型匹配、attention 权重类型改变、mlp 权重类型不变、前向输出形状 `(2, 64)` 且全部有限。注意这里 CPU 只跑 INT8，INT4 路径要上 GPU 另测。

### 第 6 步：GPU 实验怎么设计和读结果
实验对象是 `Qwen/Qwen2.5-0.5B-Instruct`，数据是 WikiText-2（`wikitext-2-raw-v1`，固定 revision `f7762941...`）。自变量是"保护多少比例的 Block 为 INT8"，取 `PROTECTED_FRACTIONS = (0.125, 0.25, 0.50)`，24 个 Block 换算成 3 / 6 / 12 个。因变量有四类：

- **质量**：`mean_nll`（token 加权），换算成 `nll_ratio_vs_bf16`，门槛 `MAX_NLL_RATIO = 1.05`。
- **容量**：`state_bytes(model)` 实测状态字节，换算成 `capacity_reduction_vs_bf16 = 1 - 候选字节 / BF16 字节`。
- **计划精度**：`effective_average_bits`，按每层权重元素数加权，不是简单按层数平均。
- **恢复效率**：`quality_recovery_fraction = (NLL_int4 - NLL_候选) / (NLL_int4 - NLL_int8)`，即"从统一 INT4 往 INT8 的方向补回了多少差距"；还有 `nll_recovery_per_extra_mib`。

敏感度扫描的做法是：以统一 INT8 为起点，每次只把一个 Block 的 7 个线性层降成 INT4，在 16 条校准文本上测 NLL 增量，按增量从大到小排序。保存的输出里最敏感的三个 Block 是 21（+0.03594）、3（+0.03486）、2（+0.02674），所以保护 3 个时优先保护它们。最终候选矩阵在 64 条验证文本上比较，`selected_candidate` 是 `mixed_int4_protect_3_of_24_blocks`，`decision` 为 `accept_mixed_bit_candidate`。

结果怎么读：
- 保护 3/24 的计划平均位宽 4.36，状态字节比 BF16 少约 39.1%，NLL 比 1.0494，刚好压在 1.05 门槛内——**质量余量很小**。
- 保护 6/24 时 NLL 比降到 1.0352，余量更大；两档的质量恢复率分别约 31.3% 和 49.4%。
- 但 5.4 的表也写明：三档混合候选的评测吞吐都低于 BF16，峰值显存反而更高。所以这次结论只是"质量—容量"维度上的 Pareto 选择，不是端到端加速的证明；吞吐和独立资源窗口要专门复测。
- 验证集参与了候选选择，最终还要用没参与选择的测试样本复验。

## 关键概念速查表
| 术语 | 大白话解释 |
|---|---|
| 混合位宽（mixed-bit） | 不同层用不同精度，而不是整网一个位宽 |
| 敏感度 | 某一层降精度后，输出或任务指标变差多少 |
| MSE / NLL | 均方误差 / 负对数似然，都是越低越好，但看的对象不同 |
| 候选（option） | 某层在某个位宽下的 `{bits, bytes, error}` 三元组 |
| 保护层（`protected_min_bits`） | 强制不低于某位宽的层，防止被总预算压穿 |
| 容量预算 `B` | 所有层权重字节之和的上限 |
| 单位字节收益 | `误差下降 / 额外字节`，贪心升级的排序依据 |
| 理想权重字节 | 只按 `参数量 × 位宽 / 8` 算，不含 scale、zero-point、padding |
| `state_bytes` | 转换后模型真实状态占的字节，含量化元数据 |
| 有效平均位宽 | 按权重元素数加权的平均位宽，不是层数平均 |
| NLL 比 | 候选 NLL ÷ BF16 NLL，越接近 1 越好 |
| Pareto 最优 | 没有别的候选同时"更省字节且 NLL 更低" |

## 小白常见疑问
**Q1：`(numel * bits + 7) // 8` 为什么加 7，直接 `// 8` 不行吗？**
不行。`//` 是向下取整，1000 个 4 位参数是 4000 位 = 500 字节刚好整除；但如果换成 1001 个 4 位参数，4004 位 = 500.5 字节，直接 `// 8` 得 500，会少算半个字节。加 7 再整除等价于 `ceil(x / 8)`，保证"不足一字节也占一字节"，和真实存储一致。

**Q2：保护层为什么不能先压到 4 位，等后面有预算再升回来？**
因为 `allocate` 的搜索方向是单向的：基线取"满足下限的最小容量"，之后只做 `extra_bytes > 0` 的升级，没有降级动作。如果基线违规，后面要么预算不够、要么根本不会回头，策略就失效了。所以保护规则必须在基线阶段就生效。

**Q3：为什么 `test_upgrade_prefers_higher_error_reduction_density` 里 attention 的误差下降 0.37 比 mlp 的 0.10 大，但两者不是都很大吗？为什么只升 attention？**
不只是"更大"，还要看花了多少字节。attention 用 500 字节换 0.37，密度 7.4×10⁻⁴；mlp 用 1000 字节换 0.10，密度 1×10⁻⁴，前者划算 7 倍多。更直接的原因是预算只剩 500（基线 1500、总预算 2000），mlp 的 +1000 根本放不进去，连候选都不会进入 `upgrades`。

**Q4：GPU 结果里保护 3/24 的平均位宽为什么是 4.36，而不是 (3×8 + 21×4)/24 = 4.5？**
因为 `effective_average_bits` 是按每层权重元素数加权的，不是按 Block 数平均。被保护的 3 个 Block 的线性层参数量占比与"3/24"并不相等，所以加权结果 4.36 会偏离 4.5。notebook 同时记录 `planned_weight_numel_by_bits` 直方图，就是为了让这个加权过程可追溯。

**Q5：NLL 比 1.0494 已经低于 1.05，为什么决策还需要复验？**
因为 `MAX_NLL_RATIO = 1.05` 只是准入门槛，不是"安全"。1.0494 距离门槛只有 0.0006 的余量，换一批样本很可能越界；而且这次验证集参与了候选选择，存在选择偏差。5.4 明确要求"候选确定后再用未参与选择的测试样本复验"。

## 动手验证：怎么知道自己做对了
`run_mixed_bit_tests()` 依次跑五个测试，全过才打印 `✅ 混合位宽机制测试通过：候选、保护规则、预算和敏感层升级均已验证。`。TODO 没填时 `build_options` 会先抛 `NotImplementedError`。
- `test_options_respect_backend_support`：`supported_bits=(4, 8)` 时，attention 的候选位宽序列必须恰好是 `[4, 8]`，16 位被排除。
- `test_protected_layer_uses_minimum_bits`：预算 3000、保护 attention ≥ 8，最终 `policy["attention"] >= 8`。
- `test_budget_is_never_exceeded`：预算 2000，`total_bytes <= 2000`。
- `test_upgrade_prefers_higher_error_reduction_density`：只支持 4/8、预算 2000 时，`policy == {"attention": 8, "mlp": 4}`。
- `test_impossible_budget_is_rejected`：预算 100 必须抛 `ValueError`。

框架核对额外断言：`allocation["policy"] == {"attention": 8, "mlp": 4}`、`list(framework_policy.fqn_to_config) == ["attention", "mlp"]`、attention 配置类型是 `Int8WeightOnlyConfig`、mlp 是 `Int4WeightOnlyConfig`、attention 权重类型变化而 mlp 不变、前向输出形状 `(2, 64)` 且 `torch.isfinite(output).all()` 为真。

notebook 里 Step 5 保存过执行输出：`failure` 为 `None`，`evidence_level` 为 `matched_mixed_bit_quality_capacity_pareto_matrix`，`selected_candidate` 为 `mixed_int4_protect_3_of_24_blocks`。注意这是特定机器、特定提交（`7ae5576...`）上的历史记录，你自己复跑时以新生成的 JSON 为准，不要把它当成固定答案。
