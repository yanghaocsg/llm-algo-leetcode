"""
Task6 · 分布式推理、部署比较与决策 (Distributed Inference and Benchmark Decision)
======================================================================================
本脚本实现推理优化专题 Task6 的核心内容：
1. 并行策略: DP / TP / PP / EP 的适用条件与通信代价
2. 通信观测: NCCL 集合通信模拟与热点分析
3. 扩展效率: 线性扩展分析与瓶颈定位
4. MoE 专家并行: token dispatch / All-to-All / 负载均衡
5. 端到端 Benchmark 框架: 固定 workload, baseline/candidate 对照
6. 决策框架: accept / tune / reject
7. 成本分析: 资源成本、单位产出成本、容量收益

参考: topic_discussion/inference_optimization/08_distributed_inference_and_parallelism.md
      topic_discussion/inference_optimization/06_benchmark_and_decision.md
      02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb
      02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb
      02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb
"""

import math
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================
# 1. 数据结构
# ============================================================

class ParallelStrategy(Enum):
    DP = "Data Parallel"
    TP = "Tensor Parallel"
    PP = "Pipeline Parallel"
    EP = "Expert Parallel"


@dataclass
class ParallelConfig:
    """并行配置"""
    strategy: ParallelStrategy
    world_size: int
    model_params_b: float = 70.0
    hidden_dim: int = 8192
    num_layers: int = 80
    num_heads: int = 64
    # 硬件
    gpu_memory_gb: float = 80.0
    interconnect_bandwidth_gbps: float = 600.0  # NVLink
    # 性能
    peak_tflops: float = 312.0  # A100 BF16 Tensor Core


@dataclass
class CommunicationResult:
    """通信结果"""
    operation: str
    size_bytes: float
    bandwidth_gbps: float
    time_ms: float
    exposed_ms: float  # 暴露在关键路径的时间 (未被计算重叠)
    overlap_ratio: float


@dataclass
class BenchmarkRecord:
    """Benchmark 记录"""
    name: str
    model: str
    backend: str
    dtype: str
    prompt_tokens: int
    generated_tokens: int
    batch_size: int
    concurrency: int
    cache_policy: str
    # 性能
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    throughput_tps: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    peak_memory_gb: float = 0.0
    # 质量
    quality_score: float = 1.0
    # 决策
    decision: str = "pending"  # accept / tune / reject


# ============================================================
# 2. 并行策略分析
# ============================================================

class ParallelStrategyAnalyzer:
    """并行策略分析器"""

    def __init__(self, config: ParallelConfig):
        self.config = config

    def analyze(self) -> Dict:
        """分析当前并行策略"""
        c = self.config
        result = {
            "strategy": c.strategy.value,
            "world_size": c.world_size,
            "model_params_b": c.model_params_b,
        }

        if c.strategy == ParallelStrategy.DP:
            result.update(self._analyze_dp())
        elif c.strategy == ParallelStrategy.TP:
            result.update(self._analyze_tp())
        elif c.strategy == ParallelStrategy.PP:
            result.update(self._analyze_pp())
        elif c.strategy == ParallelStrategy.EP:
            result.update(self._analyze_ep())

        return result

    def _analyze_dp(self) -> Dict:
        """数据并行分析"""
        c = self.config
        # DP: 每个 GPU 保存完整模型副本
        weight_per_gpu_gb = c.model_params_b * 2 / 1  # FP16
        # 推理时 DP 主要是请求分发，不需要梯度同步
        # 通信: 推理时几乎无通信 (独立处理请求)
        comm_per_step_bytes = 0  # 推理 DP 无集合通信
        comm_time_ms = 0

        return {
            "weight_per_gpu_gb": weight_per_gpu_gb,
            "can_fit": weight_per_gpu_gb * 1.2 < c.gpu_memory_gb,
            "comm_per_step_bytes": comm_per_step_bytes,
            "comm_time_ms": comm_time_ms,
            "main_comm": "无 (推理时独立处理请求)",
            "scaling": "接近线性 (无通信瓶颈)",
            "suitable_for": "单卡能装下模型，需要高吞吐/多实例",
            "limitation": "单卡必须装下完整模型，不解决单卡容量问题",
        }

    def _analyze_tp(self) -> Dict:
        """张量并行分析"""
        c = self.config
        # TP: 层内权重切分到多个 GPU
        weight_per_gpu_gb = c.model_params_b * 2 / c.world_size
        # 通信: 每层 All-Reduce (或 All-Gather + Reduce-Scatter)
        # All-Reduce 数据量 ~ 2 * batch * seq_len * hidden_dim
        batch, seq_len = 1, 2048
        allreduce_bytes = 2 * batch * seq_len * c.hidden_dim * 2  # FP16, 每层
        total_allreduce_bytes = allreduce_bytes * c.num_layers
        comm_time_ms = (total_allreduce_bytes * 8 / (c.interconnect_bandwidth_gbps * 1e9)) * 1000

        return {
            "weight_per_gpu_gb": weight_per_gpu_gb,
            "can_fit": weight_per_gpu_gb * 1.5 < c.gpu_memory_gb,
            "comm_per_step_bytes": total_allreduce_bytes,
            "comm_time_ms": round(comm_time_ms, 2),
            "main_comm": "All-Reduce (每层)",
            "scaling": "受限于互联带宽，通常 2-8 卡效率较好",
            "suitable_for": "单卡装不下模型，低延迟推理",
            "limitation": "高频 All-Reduce，对互联带宽敏感，不适合跨节点",
        }

    def _analyze_pp(self) -> Dict:
        """流水线并行分析"""
        c = self.config
        # PP: 模型层切分到多个 GPU (stage)
        layers_per_stage = c.num_layers // c.world_size
        weight_per_gpu_gb = c.model_params_b * 2 / c.world_size
        # 通信: stage 间激活传递 (每步一次)
        batch, seq_len = 1, 2048
        activation_bytes = batch * seq_len * c.hidden_dim * 2  # FP16
        comm_time_ms = (activation_bytes * 8 / (c.interconnect_bandwidth_gbps * 1e9)) * 1000
        # Pipeline 气泡
        bubble_ratio = (c.world_size - 1) / c.world_size if batch == 1 else 0

        return {
            "weight_per_gpu_gb": weight_per_gpu_gb,
            "can_fit": weight_per_gpu_gb * 1.5 < c.gpu_memory_gb,
            "layers_per_stage": layers_per_stage,
            "comm_per_step_bytes": activation_bytes,
            "comm_time_ms": round(comm_time_ms, 2),
            "bubble_ratio": bubble_ratio,
            "main_comm": "stage 间激活传递 (Point-to-Point)",
            "scaling": "受 pipeline 气泡限制，需要大 batch 填充",
            "suitable_for": "超大模型，TP 不够用时的补充",
            "limitation": "pipeline 气泡，单请求延迟高，需要 micro-batch",
        }

    def _analyze_ep(self) -> Dict:
        """专家并行分析 (MoE)"""
        c = self.config
        # EP: MoE 专家切分到多个 GPU
        num_experts = 64  # 典型值
        experts_per_gpu = num_experts // c.world_size
        weight_per_gpu_gb = c.model_params_b * 2 / c.world_size  # 简化
        # 通信: token dispatch / All-to-All
        batch, seq_len, top_k = 1, 2048, 2
        # All-to-All: 每个 token 发送到目标专家 GPU
        alltoall_bytes = batch * seq_len * c.hidden_dim * 2 * 2  # forward + backward
        comm_time_ms = (alltoall_bytes * 8 / (c.interconnect_bandwidth_gbps * 1e9)) * 1000

        return {
            "weight_per_gpu_gb": weight_per_gpu_gb,
            "can_fit": weight_per_gpu_gb * 1.5 < c.gpu_memory_gb,
            "num_experts": num_experts,
            "experts_per_gpu": experts_per_gpu,
            "comm_per_step_bytes": alltoall_bytes,
            "comm_time_ms": round(comm_time_ms, 2),
            "main_comm": "All-to-All (token dispatch)",
            "scaling": "受专家负载均衡和 All-to-All 限制",
            "suitable_for": "MoE 模型 (DeepSeek/Mixtral 等)",
            "limitation": "专家负载不均、All-to-All 通信热点",
        }


def run_parallel_strategy_comparison():
    """运行并行策略对比"""
    print(f"\n{'='*80}")
    print("  并行策略对比 (70B 模型, A100 80GB, NVLink 600GB/s)")
    print(f"{'='*80}")

    strategies = [
        (ParallelStrategy.DP, 1),
        (ParallelStrategy.DP, 4),
        (ParallelStrategy.TP, 2),
        (ParallelStrategy.TP, 4),
        (ParallelStrategy.TP, 8),
        (ParallelStrategy.PP, 4),
        (ParallelStrategy.PP, 8),
        (ParallelStrategy.EP, 4),
        (ParallelStrategy.EP, 8),
    ]

    print(f"\n  {'策略':<10} {'GPU数':<6} {'每卡权重(GB)':<14} {'能装入':<8} "
          f"{'通信(ms)':<10} {'主要通信':<25} {'适合场景':<20}")
    print(f"  {'-'*93}")

    for strategy, world_size in strategies:
        config = ParallelConfig(strategy=strategy, world_size=world_size)
        analyzer = ParallelStrategyAnalyzer(config)
        result = analyzer.analyze()
        fit = "✓" if result["can_fit"] else "✗"
        comm = result.get("comm_time_ms", 0)
        print(f"  {strategy.value[:8]:<10} {world_size:<6} "
              f"{result['weight_per_gpu_gb']:<14.1f} {fit:<8} "
              f"{comm:<10.2f} {result['main_comm']:<25} {result['suitable_for']:<20}")

    print(f"\n  选择顺序: 先确认能否装下 → 再确认通信链路 → 估计同步频率与 payload")
    print(f"  → 比较扩展后的吞吐、P95/P99、峰值显存和成本")


# ============================================================
# 3. 通信观测与热点缓解
# ============================================================

class CommunicationProfiler:
    """通信分析器 (模拟 NCCL)"""

    def __init__(self, bandwidth_gbps: float = 600.0, latency_us: float = 5.0):
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us

    def all_reduce_time(self, size_bytes: float, world_size: int) -> float:
        """All-Reduce 时间 (ms): ring algorithm, ~2*(N-1)/N * size"""
        effective_bytes = 2 * (world_size - 1) / world_size * size_bytes
        time_us = self.latency_us + (effective_bytes * 8 / (self.bandwidth_gbps * 1e9)) * 1e6
        return time_us / 1000

    def all_gather_time(self, size_bytes: float, world_size: int) -> float:
        """All-Gather 时间 (ms)"""
        effective_bytes = (world_size - 1) * size_bytes
        time_us = self.latency_us + (effective_bytes * 8 / (self.bandwidth_gbps * 1e9)) * 1e6
        return time_us / 1000

    def all_to_all_time(self, size_bytes: float, world_size: int) -> float:
        """All-to-All 时间 (ms)"""
        effective_bytes = (world_size - 1) / world_size * size_bytes
        time_us = self.latency_us + (effective_bytes * 8 / (self.bandwidth_gbps * 1e9)) * 1e6
        return time_us / 1000

    def reduce_scatter_time(self, size_bytes: float, world_size: int) -> float:
        """Reduce-Scatter 时间 (ms)"""
        effective_bytes = (world_size - 1) / world_size * size_bytes
        time_us = self.latency_us + (effective_bytes * 8 / (self.bandwidth_gbps * 1e9)) * 1e6
        return time_us / 1000


def run_communication_analysis():
    """运行通信分析"""
    print(f"\n{'='*80}")
    print("  通信观测与热点分析 (NCCL, NVLink 600GB/s)")
    print(f"{'='*80}")

    profiler = CommunicationProfiler(bandwidth_gbps=600.0)

    # 不同消息大小的集合通信时间
    sizes_mb = [0.1, 1, 10, 50, 100, 500]
    world_size = 8

    print(f"\n  消息大小 vs 通信时间 (world_size={world_size}):")
    print(f"  {'大小(MB)':<10} {'AllReduce(ms)':<14} {'AllGather(ms)':<14} "
          f"{'All2All(ms)':<14} {'ReduceScatter(ms)':<18}")
    print(f"  {'-'*70}")
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024**2
        ar = profiler.all_reduce_time(size_bytes, world_size)
        ag = profiler.all_gather_time(size_bytes, world_size)
        a2a = profiler.all_to_all_time(size_bytes, world_size)
        rs = profiler.reduce_scatter_time(size_bytes, world_size)
        print(f"  {size_mb:<10.1f} {ar:<14.3f} {ag:<14.3f} {a2a:<14.3f} {rs:<18.3f}")

    # 通信热点缓解
    print(f"\n  通信热点缓解策略:")
    mitigations = [
        ("通信-计算重叠", "将通信与计算 pipeline 化，隐藏通信时间", "overlap_ratio, idle_gap"),
        ("梯度压缩", "降低通信数据量 (如 1-bit Adam)", "通信字节数, 精度影响"),
        ("拓扑感知调度", "按 NVLink/PCIe 拓扑分配 rank", "通信时间, 跨节点流量"),
        ("批次拆分 (DBO)", "将大 batch 拆分为小 batch 交错通信", "暴露通信时间, 吞吐"),
        ("异步通信", "使用非阻塞通信，计算不等待", "exposed_comm_ms, 正确性"),
    ]
    print(f"  {'策略':<18} {'原理':<35} {'关键证据':<20}")
    print(f"  {'-'*73}")
    for name, principle, evidence in mitigations:
        print(f"  {name:<18} {principle:<35} {evidence:<20}")


# ============================================================
# 4. 扩展效率分析
# ============================================================

def run_scaling_efficiency_analysis():
    """运行扩展效率分析"""
    print(f"\n{'='*80}")
    print("  扩展效率分析")
    print(f"{'='*80}")

    # 模拟不同 GPU 数量下的扩展效率
    gpu_counts = [1, 2, 4, 8, 16, 32]

    # 三种场景
    scenarios = {
        "DP (理想, 无通信)": lambda n: 1.0,  # 完美线性
        "TP (通信受限)": lambda n: 1.0 / (1 + 0.05 * (n - 1)),  # 通信开销随 GPU 数增长
        "PP (气泡+通信)": lambda n: (1 - (n - 1) / n * 0.3) / (1 + 0.03 * (n - 1)),
    }

    print(f"\n  {'GPU数':<8} {'DP效率':<12} {'DP吞吐':<12} {'TP效率':<12} {'TP吞吐':<12} "
          f"{'PP效率':<12} {'PP吞吐':<12}")
    print(f"  {'-'*80}")

    baseline_throughput = 100  # tokens/s
    for n in gpu_counts:
        row = f"  {n:<8}"
        for name, eff_func in scenarios.items():
            eff = eff_func(n)
            throughput = baseline_throughput * n * eff
            row += f" {eff:<12.1%} {throughput:<12.1f}"
        print(row)

    print(f"\n  关键洞察:")
    print(f"  1. DP 在无通信时接近线性扩展，但单卡必须装下模型")
    print(f"  2. TP 受 All-Reduce 通信限制，通常 2-8 卡效率较好")
    print(f"  3. PP 有 pipeline 气泡，需要大 batch/micro-batch 填充")
    print(f"  4. 超过 8 卡后，通信和同步开销通常使扩展效率显著下降")
    print(f"  5. 实际扩展效率必须用真实 workload 测量，不能只看理论值")


# ============================================================
# 5. 端到端 Benchmark 框架
# ============================================================

class EndToEndBenchmark:
    """端到端 Benchmark 框架"""

    def __init__(self):
        self.records: List[BenchmarkRecord] = []

    def add_record(self, record: BenchmarkRecord):
        self.records.append(record)

    def compare(self, baseline_name: str, candidate_name: str) -> Dict:
        """对比 baseline 和 candidate"""
        baseline = next((r for r in self.records if r.name == baseline_name), None)
        candidate = next((r for r in self.records if r.name == candidate_name), None)

        if not baseline or not candidate:
            return {"error": "Record not found"}

        comparison = {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "ttft_change_pct": (candidate.ttft_ms - baseline.ttft_ms) / baseline.ttft_ms * 100,
            "tpot_change_pct": (candidate.tpot_ms - baseline.tpot_ms) / baseline.tpot_ms * 100,
            "throughput_change_pct": (candidate.throughput_tps - baseline.throughput_tps) / baseline.throughput_tps * 100,
            "p99_change_pct": (candidate.p99_ms - baseline.p99_ms) / baseline.p99_ms * 100,
            "memory_change_pct": (candidate.peak_memory_gb - baseline.peak_memory_gb) / baseline.peak_memory_gb * 100,
            "quality_change": candidate.quality_score - baseline.quality_score,
        }
        return comparison

    def make_decision(self, record: BenchmarkRecord, constraints: Dict) -> str:
        """
        决策框架: accept / tune / reject
        - accept: 满足所有约束
        - tune: 方向有效但证据或配置不足
        - reject: 收益不足、代价过高或质量不达标
        """
        issues = []
        if record.ttft_ms > constraints.get("max_ttft_ms", float('inf')):
            issues.append(f"TTFT {record.ttft_ms:.1f} > {constraints['max_ttft_ms']}")
        if record.p99_ms > constraints.get("max_p99_ms", float('inf')):
            issues.append(f"P99 {record.p99_ms:.1f} > {constraints['max_p99_ms']}")
        if record.quality_score < constraints.get("min_quality", 0.95):
            issues.append(f"质量 {record.quality_score:.2%} < {constraints['min_quality']:.0%}")
        if record.peak_memory_gb > constraints.get("max_memory_gb", float('inf')):
            issues.append(f"显存 {record.peak_memory_gb:.1f}GB > {constraints['max_memory_gb']}GB")

        if not issues:
            return "accept"
        elif len(issues) <= 2 and record.throughput_tps > constraints.get("min_throughput", 0):
            return f"tune ({'; '.join(issues)})"
        else:
            return f"reject ({'; '.join(issues)})"


def run_benchmark_demo():
    """运行 Benchmark 演示"""
    print(f"\n{'='*80}")
    print("  端到端 Benchmark 与决策演示")
    print(f"{'='*80}")

    benchmark = EndToEndBenchmark()

    # 模拟多种配置的 benchmark 结果
    configs = [
        ("FP16 单卡 (baseline)", "fp16", 1, 50, 30, 500, 80, 1.0),
        ("W8A16 单卡", "int8", 1, 45, 28, 550, 45, 0.99),
        ("W4 GPTQ 单卡", "int4", 1, 48, 26, 580, 25, 0.96),
        ("TP=2 FP16", "fp16", 2, 55, 35, 900, 42, 1.0),
        ("TP=4 FP16", "fp16", 4, 60, 42, 1500, 22, 1.0),
        ("vLLM + PagedAttn", "fp16", 1, 40, 25, 800, 70, 1.0),
        ("vLLM + PrefixCache", "fp16", 1, 25, 25, 850, 65, 1.0),
    ]

    for name, dtype, gpus, ttft, tpot, tps, mem_gb, quality in configs:
        record = BenchmarkRecord(
            name=name, model="sim-70b", backend="sim", dtype=dtype,
            prompt_tokens=512, generated_tokens=256, batch_size=4, concurrency=8,
            cache_policy="standard",
            ttft_ms=ttft, tpot_ms=tpot, e2e_ms=ttft + tpot * 256,
            throughput_tps=tps, p50_ms=tpot * 0.9, p95_ms=tpot * 1.3, p99_ms=tpot * 1.8,
            peak_memory_gb=mem_gb, quality_score=quality,
        )
        benchmark.add_record(record)

    # 打印对比表
    print(f"\n  {'配置':<25} {'TTFT':<8} {'TPOT':<8} {'吞吐':<8} {'P99':<8} "
          f"{'显存':<8} {'质量':<8} {'决策':<10}")
    print(f"  {'-'*83}")

    constraints = {
        "max_ttft_ms": 100,
        "max_p99_ms": 80,
        "min_quality": 0.95,
        "max_memory_gb": 80,
        "min_throughput": 400,
    }

    for record in benchmark.records:
        decision = benchmark.make_decision(record, constraints)
        print(f"  {record.name:<25} {record.ttft_ms:<8.0f} {record.tpot_ms:<8.0f} "
              f"{record.throughput_tps:<8.0f} {record.p99_ms:<8.0f} "
              f"{record.peak_memory_gb:<8.0f} {record.quality_score:<8.0%} {decision:<10}")

    # 对比分析
    print(f"\n  关键对比 (vs FP16 单卡 baseline):")
    comparisons = [
        ("FP16 单卡 (baseline)", "W8A16 单卡"),
        ("FP16 单卡 (baseline)", "TP=4 FP16"),
        ("FP16 单卡 (baseline)", "vLLM + PrefixCache"),
    ]
    for base, cand in comparisons:
        comp = benchmark.compare(base, cand)
        print(f"  {cand}:")
        print(f"    TTFT {comp['ttft_change_pct']:+.1f}%, TPOT {comp['tpot_change_pct']:+.1f}%, "
              f"吞吐 {comp['throughput_change_pct']:+.1f}%, 显存 {comp['memory_change_pct']:+.1f}%")

    print(f"\n  决策框架说明:")
    print(f"  - accept: 满足所有约束 (TTFT/P99/质量/显存/吞吐)")
    print(f"  - tune: 方向有效但有 1-2 个约束不满足，需调整配置")
    print(f"  - reject: 多个约束不满足或质量不达标，不值得采用")


# ============================================================
# 6. 成本分析
# ============================================================

def run_cost_analysis():
    """运行部署成本分析"""
    print(f"\n{'='*80}")
    print("  部署成本分析")
    print(f"{'='*80}")

    gpu_hourly_cost = 2.0  # A100 80GB on-demand
    daily_requests = 1_000_000
    avg_tokens_per_request = 500

    configs = [
        ("FP16 单卡", 1, 500, 80, 1.0),
        ("W8A16 单卡", 1, 550, 45, 0.99),
        ("TP=4 FP16", 4, 1500, 22, 1.0),
        ("vLLM 优化单卡", 1, 850, 65, 1.0),
    ]

    print(f"\n  假设: ${gpu_hourly_cost}/GPU/h, {daily_requests:,} 请求/天, "
          f"{avg_tokens_per_request} tokens/请求")
    print(f"\n  {'配置':<20} {'GPU数':<6} {'吞吐(t/s)':<10} {'日成本':<12} "
          f"{'$/1M tokens':<14} {'质量':<8} {'性价比':<10}")
    print(f"  {'-'*80}")

    for name, gpus, tps, mem_gb, quality in configs:
        daily_cost = gpus * gpu_hourly_cost * 24
        daily_tokens = tps * 86400
        cost_per_million = daily_cost / (daily_tokens / 1e6)
        # 性价比 = 质量 * 吞吐 / (GPU数 * 单位成本)
        value = quality * tps / (gpus * gpu_hourly_cost)
        print(f"  {name:<20} {gpus:<6} {tps:<10} ${daily_cost:<11.0f} "
              f"${cost_per_million:<13.4f} {quality:<8.0%} {value:<10.1f}")

    print(f"\n  成本字段:")
    print(f"  - 资源成本 = GPU数量 × 运行时间 × 单卡小时成本")
    print(f"  - 单位产出成本 = 资源成本 ÷ 有效输出 tokens")
    print(f"  - 失败代价 = 超时、拒绝、重试和质量不达标请求的比例")
    print(f"  - 容量收益 = 同一 SLA 下可接纳的并发或 tokens/s")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 80)
    print("  Task6 · 分布式推理、部署比较与决策")
    print("=" * 80)

    # 1. 为什么需要分布式推理
    print(f"\n{'='*80}")
    print("  1. 分布式推理触发信号")
    print(f"{'='*80}")
    triggers = [
        ("单卡装不下模型或 KV 状态", "模型、Cache 或 batch 超出容量", "峰值显存、Cache 使用量、模型状态账本"),
        ("单卡吞吐不足", "计算、访存或请求容量受限", "kernel/阶段时间、吞吐、队列等待"),
        ("增加 GPU 后收益很小", "通信暴露、同步或负载不均", "通信时间、暴露时间、空转和扩展效率"),
        ("MoE workload 扩展异常", "token dispatch、All-to-All 或专家不均衡", "expert load、通信字节数、热点专家"),
    ]
    print(f"  {'触发信号':<30} {'可能原因':<30} {'先补什么证据':<30}")
    print(f"  {'-'*90}")
    for signal, cause, evidence in triggers:
        print(f"  {signal:<30} {cause:<30} {evidence:<30}")

    # 2. 并行策略对比
    run_parallel_strategy_comparison()

    # 3. 通信分析
    run_communication_analysis()

    # 4. 扩展效率
    run_scaling_efficiency_analysis()

    # 5. 端到端 Benchmark
    run_benchmark_demo()

    # 6. 成本分析
    run_cost_analysis()

    # 7. 完整证据链
    print(f"\n{'='*80}")
    print("  7. 分布式项目最小证据链")
    print(f"{'='*80}")
    evidence_chain = [
        ("单实例 baseline", "建立对照基准", "TTFT、TPOT、吞吐、P99、显存"),
        ("GPU 数量扫描", "确定扩展曲线", "1/2/4/8 卡的吞吐和延迟"),
        ("并行策略对比", "DP/TP/PP/EP 选择", "通信时间、扩展效率、显存"),
        ("通信观测", "定位通信热点", "通信时间、暴露时间、overlap_ratio"),
        ("MoE 专家分析", "EP 专用分析", "expert load、All-to-All、空转"),
        ("质量验证", "确保分布式不影响质量", "输出一致性、任务准确率"),
        ("成本决策", "形成最终部署决策", "accept/tune/reject、成本/token"),
    ]
    print(f"  {'阶段':<18} {'目的':<25} {'最小证据':<35}")
    print(f"  {'-'*78}")
    for stage, purpose, evidence in evidence_chain:
        print(f"  {stage:<18} {purpose:<25} {evidence:<35}")

    # 8. 瓶颈判断框架
    print(f"\n{'='*80}")
    print("  8. 瓶颈判断框架")
    print(f"{'='*80}")
    framework = [
        ("单卡装不下", "容量或模型切分问题", "先看显存账本，再评估 TP/PP"),
        ("多卡吞吐没有线性提升", "通信暴露或负载不均", "查看通信时间、空转和 overlap"),
        ("MoE 扩展明显变慢", "All-to-All 或专家热点", "进入 EP 与负载均衡分析"),
        ("吞吐提升但 P99 变差", "通信或排队进入关键路径", "回到 Serving 和统一 benchmark"),
        ("跨节点扩展效率骤降", "互联带宽不足 (PCIe/以太网 vs NVLink)", "评估 TP 范围限制，改用 DP/PP"),
    ]
    for phenomenon, bottleneck, next_step in framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 判断: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task6\\distributed_benchmark.json"
    output = {"note": "Distributed inference and benchmark decision results"}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")

    print(f"\n{'='*80}")
    print("  Task6 完成!")
    print("  关键概念: 并行策略 → 通信观测 → 扩展效率 → Benchmark框架 → 决策 → 成本分析")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
