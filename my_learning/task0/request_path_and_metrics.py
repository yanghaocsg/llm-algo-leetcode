"""
Task0 · 请求结构与指标 (Request Path and Metrics)
=====================================================
本脚本实现推理优化专题 Task0 的核心内容：
1. 模拟一次请求经过 Prefill → KV Cache → Decode 的完整生命周期
2. 测量 TTFT (Time To First Token)、TPOT (Time Per Output Token)
3. 统计 throughput、P50/P95/P99 延迟、peak memory
4. 区分不同 workload 类型（交互式短请求、长上下文、批处理、突发并发）
5. 建立可复现的 benchmark 框架

参考: topic_discussion/inference_optimization/01_request_path_and_metrics.md
"""

import time
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


# ============================================================
# 1. 数据结构定义
# ============================================================

@dataclass
class RequestConfig:
    """请求配置：固定 workload 的关键参数"""
    prompt_tokens: int = 128          # 输入 prompt 长度
    generated_tokens: int = 64        # 输出生成 token 数
    batch_size: int = 1               # 批次大小
    concurrency: int = 1              # 并发请求数
    model_name: str = "sim-llm-1.3b" # 模型标识
    dtype: str = "fp16"               # 计算数据类型
    cache_policy: str = "standard"    # 缓存策略


@dataclass
class RequestTimeline:
    """单次请求的时间线记录"""
    request_id: int
    config: RequestConfig
    # 阶段时间 (秒)
    queue_wait_ms: float = 0.0        # 排队等待
    prefill_ms: float = 0.0           # Prefill 阶段
    kv_cache_write_ms: float = 0.0    # KV Cache 写入
    decode_steps_ms: List[float] = field(default_factory=list)  # 每个 Decode step
    # 派生指标
    ttft_ms: float = 0.0              # Time To First Token
    tpot_ms: float = 0.0              # Time Per Output Token
    e2e_latency_ms: float = 0.0       # 端到端延迟
    throughput_tps: float = 0.0       # 吞吐 (tokens/sec)
    peak_memory_mb: float = 0.0       # 峰值显存 (模拟)
    # 状态
    status: str = "pending"            # pending / running / completed / failed


@dataclass
class BenchmarkResult:
    """Benchmark 汇总结果"""
    config: RequestConfig
    num_requests: int
    # 中心趋势
    ttft_mean_ms: float = 0.0
    ttft_median_ms: float = 0.0
    tpot_mean_ms: float = 0.0
    tpot_median_ms: float = 0.0
    e2e_mean_ms: float = 0.0
    # 尾部延迟
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    e2e_p99_ms: float = 0.0
    # 系统指标
    total_throughput_tps: float = 0.0
    avg_peak_memory_mb: float = 0.0
    # 原始记录
    timelines: List[Dict] = field(default_factory=list)


# ============================================================
# 2. 模拟推理引擎
# ============================================================

class SimulatedInferenceEngine:
    """
    模拟推理引擎：在 CPU 上模拟 Prefill/Decode 的计算行为，
    用于验证指标测量逻辑和请求生命周期管理。
    真实性能数据需要 GPU + backend 环境。
    """

    def __init__(self, hidden_dim: int = 2048, num_layers: int = 24,
                 num_heads: int = 32, kv_heads: int = 8):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = hidden_dim // num_heads

        # 模拟模型权重 (仅用于计算量估算，不实际训练)
        self._init_simulated_weights()

    def _init_simulated_weights(self):
        """初始化模拟权重，用于计算 FLOPs 估算"""
        # 每层参数量估算 (Llama 风格)
        # attention: 4 * hidden^2, MLP: ~8/3 * hidden^2 (SwiGLU)
        self.params_per_layer = 4 * self.hidden_dim**2 + (8/3) * self.hidden_dim**2
        self.total_params = self.num_layers * self.params_per_layer
        # KV Cache 每 token 字节数 (GQA: 2 * kv_heads * head_dim * dtype_bytes)
        self.kv_bytes_per_token = 2 * self.kv_heads * self.head_dim * 2  # fp16

    def estimate_prefill_flops(self, seq_len: int, batch: int = 1) -> float:
        """估算 Prefill 阶段 FLOPs: ~2 * params * seq_len"""
        return 2 * self.total_params * seq_len * batch

    def estimate_decode_flops(self, seq_len: int, batch: int = 1) -> float:
        """估算 Decode 单步 FLOPs: ~2 * params * batch"""
        return 2 * self.total_params * batch

    def estimate_kv_cache_memory(self, seq_len: int, batch: int = 1) -> float:
        """估算 KV Cache 显存 (MB)"""
        return (self.kv_bytes_per_token * seq_len * batch * self.num_layers) / (1024**2)

    def simulate_prefill(self, seq_len: int, batch: int = 1) -> Tuple[float, float]:
        """
        模拟 Prefill 执行，返回 (耗时ms, 峰值显存MB)
        使用实际矩阵乘法来产生真实计算负载
        GQA: Q 投影到 num_heads*head_dim, KV 投影到 kv_heads*head_dim
        """
        torch.manual_seed(42)
        kv_dim = self.kv_heads * self.head_dim
        x = torch.randn(batch, seq_len, self.hidden_dim)
        w_q = torch.randn(self.hidden_dim, self.num_heads * self.head_dim)
        w_kv = torch.randn(self.hidden_dim, 2 * kv_dim)

        start = time.perf_counter()
        # Q projection and KV projection (GQA)
        q = x @ w_q
        kv = x @ w_kv
        k, v = kv.chunk(2, dim=-1)
        # Reshape for attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        # GQA: expand kv heads
        if self.kv_heads != self.num_heads:
            repeat = self.num_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        elapsed = (time.perf_counter() - start) * 1000

        peak_mem = self.estimate_kv_cache_memory(seq_len, batch)
        return elapsed, peak_mem

    def simulate_decode_step(self, seq_len: int, batch: int = 1) -> float:
        """
        模拟单个 Decode step，返回耗时ms
        Decode 每步只处理 1 个新 token，但读取全部历史 KV
        """
        torch.manual_seed(int(time.time() * 1000) % (2**31))
        kv_dim = self.kv_heads * self.head_dim
        x = torch.randn(batch, 1, self.hidden_dim)
        w_q = torch.randn(self.hidden_dim, self.num_heads * self.head_dim)
        w_kv = torch.randn(self.hidden_dim, 2 * kv_dim)
        # 模拟历史 KV (随 seq_len 增长)
        k_cache = torch.randn(batch, self.kv_heads, seq_len, self.head_dim)
        v_cache = torch.randn(batch, self.kv_heads, seq_len, self.head_dim)

        start = time.perf_counter()
        q = x @ w_q
        kv = x @ w_kv
        k_new, v_new = kv.chunk(2, dim=-1)
        q = q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(batch, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(batch, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        # Append to cache
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
        # GQA expand
        if self.kv_heads != self.num_heads:
            repeat = self.num_heads // self.kv_heads
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k_full.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v_full
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed


# ============================================================
# 3. 请求执行器
# ============================================================

class RequestExecutor:
    """请求执行器：管理请求生命周期并测量指标"""

    def __init__(self, engine: SimulatedInferenceEngine):
        self.engine = engine
        self.request_counter = 0

    def execute(self, config: RequestConfig, warmup: bool = False) -> RequestTimeline:
        """执行单次请求并记录完整时间线"""
        self.request_counter += 1
        timeline = RequestTimeline(request_id=self.request_counter, config=config)
        timeline.status = "running"

        # 1. 排队等待 (模拟)
        if config.concurrency > 1:
            timeline.queue_wait_ms = (config.concurrency - 1) * 0.5  # 模拟排队
            time.sleep(timeline.queue_wait_ms / 1000)

        # 2. Prefill 阶段
        prefill_time, peak_mem = self.engine.simulate_prefill(
            config.prompt_tokens, config.batch_size
        )
        timeline.prefill_ms = prefill_time
        timeline.peak_memory_mb = peak_mem

        # 3. KV Cache 写入 (包含在 prefill 中，单独记录)
        timeline.kv_cache_write_ms = prefill_time * 0.15  # 估算 15% 用于 cache 写入

        # 4. TTFT = 排队 + Prefill (首 token 在 Prefill 后立即可用)
        timeline.ttft_ms = timeline.queue_wait_ms + timeline.prefill_ms

        # 5. Decode 阶段：逐 token 生成
        current_seq_len = config.prompt_tokens
        for step in range(config.generated_tokens):
            step_time = self.engine.simulate_decode_step(
                current_seq_len, config.batch_size
            )
            timeline.decode_steps_ms.append(step_time)
            current_seq_len += 1

        # 6. 派生指标
        if timeline.decode_steps_ms:
            timeline.tpot_ms = statistics.mean(timeline.decode_steps_ms)
        timeline.e2e_latency_ms = (
            timeline.queue_wait_ms + timeline.prefill_ms + sum(timeline.decode_steps_ms)
        )
        total_tokens = config.prompt_tokens + config.generated_tokens
        timeline.throughput_tps = total_tokens / (timeline.e2e_latency_ms / 1000)
        timeline.status = "completed"
        return timeline


# ============================================================
# 4. Benchmark 运行器
# ============================================================

class BenchmarkRunner:
    """Benchmark 运行器：固定 workload，多次运行并汇总统计"""

    def __init__(self, engine: SimulatedInferenceEngine):
        self.executor = RequestExecutor(engine)

    def run(self, config: RequestConfig, num_requests: int = 10,
            warmup_requests: int = 2) -> BenchmarkResult:
        """
        运行 benchmark:
        1. 先 warmup (不纳入统计)
        2. 正式测量 num_requests 次
        3. 汇总中心趋势和尾部延迟
        """
        # Warmup
        for _ in range(warmup_requests):
            self.executor.execute(config, warmup=True)

        # 正式测量
        timelines = []
        for _ in range(num_requests):
            tl = self.executor.execute(config)
            timelines.append(tl)

        return self._aggregate(config, timelines)

    def _aggregate(self, config: RequestConfig,
                   timelines: List[RequestTimeline]) -> BenchmarkResult:
        """汇总统计结果"""
        result = BenchmarkResult(config=config, num_requests=len(timelines))

        ttft_list = [t.ttft_ms for t in timelines]
        tpot_list = [t.tpot_ms for t in timelines]
        e2e_list = [t.e2e_latency_ms for t in timelines]
        mem_list = [t.peak_memory_mb for t in timelines]

        # 中心趋势
        result.ttft_mean_ms = statistics.mean(ttft_list)
        result.ttft_median_ms = statistics.median(ttft_list)
        result.tpot_mean_ms = statistics.mean(tpot_list)
        result.tpot_median_ms = statistics.median(tpot_list)
        result.e2e_mean_ms = statistics.mean(e2e_list)

        # 百分位
        result.ttft_p50_ms = self._percentile(ttft_list, 50)
        result.ttft_p95_ms = self._percentile(ttft_list, 95)
        result.ttft_p99_ms = self._percentile(ttft_list, 99)
        result.e2e_p50_ms = self._percentile(e2e_list, 50)
        result.e2e_p95_ms = self._percentile(e2e_list, 95)
        result.e2e_p99_ms = self._percentile(e2e_list, 99)

        # 系统指标
        total_tokens = sum(
            t.config.prompt_tokens + t.config.generated_tokens for t in timelines
        )
        total_time_s = sum(t.e2e_latency_ms for t in timelines) / 1000
        result.total_throughput_tps = total_tokens / total_time_s if total_time_s > 0 else 0
        result.avg_peak_memory_mb = statistics.mean(mem_list)

        # 原始记录 (精简)
        result.timelines = [
            {
                "request_id": t.request_id,
                "ttft_ms": round(t.ttft_ms, 3),
                "tpot_ms": round(t.tpot_ms, 3),
                "e2e_ms": round(t.e2e_latency_ms, 3),
                "throughput_tps": round(t.throughput_tps, 2),
                "peak_memory_mb": round(t.peak_memory_mb, 2),
            }
            for t in timelines
        ]
        return result

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """计算百分位"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ============================================================
# 5. Workload 场景对比
# ============================================================

def run_workload_comparison(engine: SimulatedInferenceEngine) -> Dict[str, BenchmarkResult]:
    """
    运行四种典型 workload 对比：
    1. 交互式短请求: 短 prompt + 短输出, 关注 TTFT/P99
    2. 长上下文: 长 prompt, 关注 TTFT/peak memory
    3. 批处理: 大 batch, 关注 throughput
    4. 突发并发: 高并发, 关注排队/P99
    """
    runner = BenchmarkRunner(engine)
    results = {}

    workloads = {
        "interactive_short": RequestConfig(
            prompt_tokens=32, generated_tokens=32, batch_size=1, concurrency=1,
            dtype="fp16", cache_policy="standard"
        ),
        "long_context": RequestConfig(
            prompt_tokens=1024, generated_tokens=128, batch_size=1, concurrency=1,
            dtype="fp16", cache_policy="standard"
        ),
        "batch_processing": RequestConfig(
            prompt_tokens=256, generated_tokens=128, batch_size=8, concurrency=1,
            dtype="fp16", cache_policy="standard"
        ),
        "burst_concurrency": RequestConfig(
            prompt_tokens=128, generated_tokens=64, batch_size=1, concurrency=16,
            dtype="fp16", cache_policy="standard"
        ),
    }

    for name, config in workloads.items():
        print(f"  Running workload: {name} ...")
        results[name] = runner.run(config, num_requests=5, warmup_requests=1)

    return results


# ============================================================
# 6. 结果展示与保存
# ============================================================

def print_benchmark_summary(result: BenchmarkResult, title: str = ""):
    """打印 benchmark 摘要"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    cfg = result.config
    print(f"  配置: prompt={cfg.prompt_tokens}, gen={cfg.generated_tokens}, "
          f"batch={cfg.batch_size}, concurrency={cfg.concurrency}, dtype={cfg.dtype}")
    print(f"  请求数: {result.num_requests}")
    print(f"  --- 中心趋势 ---")
    print(f"  TTFT:   mean={result.ttft_mean_ms:.2f}ms, median={result.ttft_median_ms:.2f}ms")
    print(f"  TPOT:   mean={result.tpot_mean_ms:.2f}ms, median={result.tpot_median_ms:.2f}ms")
    print(f"  E2E:    mean={result.e2e_mean_ms:.2f}ms")
    print(f"  --- 尾部延迟 ---")
    print(f"  TTFT:   P50={result.ttft_p50_ms:.2f}ms, P95={result.ttft_p95_ms:.2f}ms, "
          f"P99={result.ttft_p99_ms:.2f}ms")
    print(f"  E2E:    P50={result.e2e_p50_ms:.2f}ms, P95={result.e2e_p95_ms:.2f}ms, "
          f"P99={result.e2e_p99_ms:.2f}ms")
    print(f"  --- 系统指标 ---")
    print(f"  吞吐:   {result.total_throughput_tps:.2f} tokens/s")
    print(f"  峰值显存: {result.avg_peak_memory_mb:.2f} MB (KV Cache 估算)")


def print_workload_comparison(results: Dict[str, BenchmarkResult]):
    """打印 workload 对比表格"""
    print(f"\n{'='*80}")
    print(f"  Workload 对比 (不同请求形态的瓶颈差异)")
    print(f"{'='*80}")
    print(f"  {'Workload':<20} {'TTFT(ms)':<12} {'TPOT(ms)':<12} "
          f"{'吞吐(t/s)':<12} {'P99(ms)':<12} {'显存(MB)':<10}")
    print(f"  {'-'*78}")
    for name, r in results.items():
        print(f"  {name:<20} {r.ttft_mean_ms:<12.2f} {r.tpot_mean_ms:<12.2f} "
              f"{r.total_throughput_tps:<12.2f} {r.e2e_p99_ms:<12.2f} "
              f"{r.avg_peak_memory_mb:<10.2f}")


def save_results(results: Dict[str, BenchmarkResult], output_path: str):
    """保存结果为 JSON"""
    output = {}
    for name, r in results.items():
        output[name] = {
            "config": asdict(r.config),
            "num_requests": r.num_requests,
            "ttft_mean_ms": round(r.ttft_mean_ms, 3),
            "ttft_p50_ms": round(r.ttft_p50_ms, 3),
            "ttft_p95_ms": round(r.ttft_p95_ms, 3),
            "ttft_p99_ms": round(r.ttft_p99_ms, 3),
            "tpot_mean_ms": round(r.tpot_mean_ms, 3),
            "e2e_mean_ms": round(r.e2e_mean_ms, 3),
            "e2e_p99_ms": round(r.e2e_p99_ms, 3),
            "total_throughput_tps": round(r.total_throughput_tps, 2),
            "avg_peak_memory_mb": round(r.avg_peak_memory_mb, 2),
        }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_path}")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Task0 · 请求结构与指标 (Request Path and Metrics)")
    print("=" * 60)

    # 初始化引擎
    engine = SimulatedInferenceEngine(
        hidden_dim=2048, num_layers=24, num_heads=32, kv_heads=8
    )
    print(f"\n  模拟引擎: hidden={engine.hidden_dim}, layers={engine.num_layers}, "
          f"heads={engine.num_heads}, kv_heads={engine.kv_heads}")
    print(f"  估算参数量: {engine.total_params / 1e9:.2f}B")
    print(f"  KV Cache/token: {engine.kv_bytes_per_token} bytes (fp16, GQA)")

    # 单次请求演示
    print(f"\n{'='*60}")
    print("  1. 单次请求生命周期演示")
    print(f"{'='*60}")
    config = RequestConfig(prompt_tokens=256, generated_tokens=32, batch_size=1)
    executor = RequestExecutor(engine)
    timeline = executor.execute(config)
    print(f"  请求 #{timeline.request_id}: prompt={config.prompt_tokens}, "
          f"gen={config.generated_tokens}")
    print(f"  排队等待: {timeline.queue_wait_ms:.3f}ms")
    print(f"  Prefill:  {timeline.prefill_ms:.3f}ms")
    print(f"  KV写入:   {timeline.kv_cache_write_ms:.3f}ms")
    print(f"  Decode步数: {len(timeline.decode_steps_ms)}, "
          f"平均 {statistics.mean(timeline.decode_steps_ms):.3f}ms/step")
    print(f"  --- 关键指标 ---")
    print(f"  TTFT:  {timeline.ttft_ms:.3f}ms (排队+Prefill)")
    print(f"  TPOT:  {timeline.tpot_ms:.3f}ms (Decode平均)")
    print(f"  E2E:   {timeline.e2e_latency_ms:.3f}ms")
    print(f"  吞吐:  {timeline.throughput_tps:.2f} tokens/s")
    print(f"  峰值显存: {timeline.peak_memory_mb:.2f} MB (KV Cache)")

    # 标准 Benchmark
    print(f"\n{'='*60}")
    print("  2. 标准 Benchmark (固定 workload, 多次运行)")
    print(f"{'='*60}")
    runner = BenchmarkRunner(engine)
    std_config = RequestConfig(prompt_tokens=128, generated_tokens=64, batch_size=1)
    std_result = runner.run(std_config, num_requests=10, warmup_requests=2)
    print_benchmark_summary(std_result, "标准配置 Benchmark 结果")

    # Workload 对比
    print(f"\n{'='*60}")
    print("  3. 四种典型 Workload 对比")
    print(f"{'='*60}")
    workload_results = run_workload_comparison(engine)
    print_workload_comparison(workload_results)

    # 瓶颈判断框架演示
    print(f"\n{'='*60}")
    print("  4. 瓶颈判断框架 (根据现象选择下一步)")
    print(f"{'='*60}")
    bottleneck_framework = [
        ("长 prompt 使 TTFT 明显升高", "prefill-bound / Attention 访存", "→ Task1: FlashAttention"),
        ("TPOT 高、生成阶段占比大", "decode-bound / KV Cache / 调度", "→ Task2/Task3: 解码策略/KV Cache"),
        ("peak memory 接近预算、batch 上不去", "memory-bound / Cache 容量", "→ Task3/Task5: KV Cache/量化"),
        ("多请求时 P99 明显升高", "排队 / batch 组织 / 调度", "→ Task4: Serving 调度"),
        ("没有明显单点瓶颈", "需要端到端比较", "→ Task6: Benchmark 与决策"),
    ]
    for phenomenon, bottleneck, next_step in bottleneck_framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 瓶颈: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    all_results = {"standard": std_result, **workload_results}
    save_results(all_results, "D:\\llm-algo-leetcode\\my_learning\\task0\\benchmark_results.json")

    print(f"\n{'='*60}")
    print("  Task0 完成!")
    print("  关键概念: 请求生命周期 → TTFT/TPOT → Workload 分类 → 瓶颈判断")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
