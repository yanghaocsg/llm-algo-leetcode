"""
Task1 · Prefill 与 Attention Kernel (Prefill and Attention Kernel)
======================================================================
本脚本实现推理优化专题 Task1 的核心内容：
1. 标准 Attention (显式 NxN 中间矩阵) vs FlashAttention (分块+在线Softmax)
2. 访存行为对比：HBM 读写量、中间工作集大小
3. Roofline 分析：计算受限 vs 访存受限
4. Chunked Prefill：长 Prompt 分块处理
5. SDPA dispatch 路径说明
6. Prefill vs Decode 工作负载差异

参考: topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md
      02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb
"""

import time
import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class AttentionConfig:
    """Attention 配置"""
    seq_len: int = 512
    batch_size: int = 1
    num_heads: int = 8
    head_dim: int = 64
    dtype: str = "fp32"
    block_size: int = 64  # FlashAttention 分块大小


@dataclass
class AttentionResult:
    """Attention 执行结果"""
    method: str
    config: AttentionConfig
    time_ms: float = 0.0
    # 访存统计 (估算)
    hbm_reads_bytes: float = 0.0
    hbm_writes_bytes: float = 0.0
    intermediate_memory_bytes: float = 0.0
    # 计算统计
    flops: float = 0.0
    arithmetic_intensity: float = 0.0  # FLOPs/byte
    # 输出验证
    max_abs_error: float = 0.0
    output_shape: Tuple = ()


# ============================================================
# 2. 标准 Attention (显式中间矩阵)
# ============================================================

def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """
    标准 Attention: 显式计算 NxN attention 矩阵
    访存模式: Q,K,V 读入 → S=QK^T 写出 → P=softmax(S) 写出 → O=PV 写出
    中间矩阵 S 和 P 都是 NxN，随序列长度平方增长
    """
    stats = {"method": "standard"}
    batch, num_heads, seq_len, head_dim = q.shape

    start = time.perf_counter()
    # Step 1: Q @ K^T → S (NxN)
    scale = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 2: Softmax → P (NxN)
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)

    # Step 3: P @ V → O
    output = torch.matmul(attn_weights, v)  # (B, H, N, D)
    elapsed = (time.perf_counter() - start) * 1000

    # 访存估算
    elem_bytes = q.element_size()
    # 读取: Q(N*D) + K(N*D) + V(N*D)
    hbm_reads = 3 * seq_len * head_dim * elem_bytes * num_heads * batch
    # 写入: S(N*N) + P(N*N) + O(N*D)
    hbm_writes = (2 * seq_len * seq_len + seq_len * head_dim) * elem_bytes * num_heads * batch
    # 中间工作集: S + P (峰值)
    intermediate_mem = 2 * seq_len * seq_len * elem_bytes * num_heads * batch
    # FLOPs: QK^T = 2*N*N*D, PV = 2*N*N*D
    flops = 4 * seq_len * seq_len * head_dim * num_heads * batch

    stats.update({
        "time_ms": elapsed,
        "hbm_reads_bytes": hbm_reads,
        "hbm_writes_bytes": hbm_writes,
        "intermediate_memory_bytes": intermediate_mem,
        "flops": flops,
        "arithmetic_intensity": flops / (hbm_reads + hbm_writes) if (hbm_reads + hbm_writes) > 0 else 0,
    })
    return output, stats


# ============================================================
# 3. FlashAttention 模拟 (分块 + 在线 Softmax)
# ============================================================

def flash_attention_sim(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        block_size: int = 64,
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """
    FlashAttention 模拟: 分块计算 + 在线 Softmax，不物化 NxN 中间矩阵
    核心思想:
    1. 外部循环遍历 Q 的 block (Br)
    2. 内部循环遍历 K/V 的 block (Bc)
    3. 每个 block 内计算 S_ij = Q_i @ K_j^T
    4. 在线更新 m_i (row max) 和 l_i (row sum)，rescale 之前的 O_i
    5. 最终 O = diag(l)^-1 @ O_tilde

    访存模式: Q 分块读入 SRAM，K/V 流式读取，O 分块写回
    中间状态只有 block 大小的 S_ij 和 O_i，不随 N^2 增长
    """
    stats = {"method": "flash_attention_sim"}
    batch, num_heads, seq_len, head_dim = q.shape
    elem_bytes = q.element_size()

    # 输出和在线统计量
    output = torch.zeros_like(q)
    m_i = torch.full((batch, num_heads, seq_len, 1), float('-inf'))
    l_i = torch.zeros((batch, num_heads, seq_len, 1))

    start = time.perf_counter()
    scale = head_dim ** -0.5

    # 外部循环: Q 的 block
    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, Br, D)

        # 初始化当前 block 的统计量
        m_block = torch.full((batch, num_heads, q_end - q_start, 1), float('-inf'))
        l_block = torch.zeros((batch, num_heads, q_end - q_start, 1))
        o_block = torch.zeros_like(q_block)

        # 内部循环: K/V 的 block
        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)
            k_block = k[:, :, kv_start:kv_end, :]  # (B, H, Bc, D)
            v_block = v[:, :, kv_start:kv_end, :]  # (B, H, Bc, D)

            # 计算当前 block 的 attention scores
            s_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # (B, H, Br, Bc)

            if mask is not None:
                mask_block = mask[:, :, q_start:q_end, kv_start:kv_end]
                s_block = s_block.masked_fill(mask_block == 0, float('-inf'))

            # 在线 Softmax: 更新 m 和 l
            m_new = torch.maximum(m_block, s_block.max(dim=-1, keepdim=True).values)
            # rescale 之前的累积
            alpha = torch.exp(m_block - m_new)
            p_block = torch.exp(s_block - m_new)
            l_new = alpha * l_block + p_block.sum(dim=-1, keepdim=True)

            # 更新输出: rescale 旧 O + 新 P @ V
            o_block = alpha * o_block + torch.matmul(p_block, v_block)

            m_block = m_new
            l_block = l_new

        # 最终归一化
        o_block = o_block / l_block
        output[:, :, q_start:q_end, :] = o_block
        m_i[:, :, q_start:q_end, :] = m_block
        l_i[:, :, q_start:q_end, :] = l_block

    elapsed = (time.perf_counter() - start) * 1000

    # 访存估算 (FlashAttention 的关键优势)
    # 读取: Q 读一次 (N*D), K/V 每个 Q block 都读一次 (N*D * N/Br)
    # 但 K/V 可以流式从 HBM 读取，不需要全部驻留
    num_q_blocks = math.ceil(seq_len / block_size)
    hbm_reads = (1 + num_q_blocks) * seq_len * head_dim * elem_bytes * num_heads * batch
    # 写入: O 写一次 (N*D), 没有 NxN 中间矩阵写回
    hbm_writes = seq_len * head_dim * elem_bytes * num_heads * batch
    # 中间工作集: 只有 block 大小的 S (Br*Bc) 和 O (Br*D)
    intermediate_mem = (block_size * block_size + block_size * head_dim) * elem_bytes * num_heads * batch
    # FLOPs 与标准 Attention 相同 (计算语义不变)
    flops = 4 * seq_len * seq_len * head_dim * num_heads * batch

    stats.update({
        "time_ms": elapsed,
        "hbm_reads_bytes": hbm_reads,
        "hbm_writes_bytes": hbm_writes,
        "intermediate_memory_bytes": intermediate_mem,
        "flops": flops,
        "arithmetic_intensity": flops / (hbm_reads + hbm_writes) if (hbm_reads + hbm_writes) > 0 else 0,
        "block_size": block_size,
        "num_q_blocks": num_q_blocks,
    })
    return output, stats


# ============================================================
# 4. PyTorch SDPA (Scaled Dot-Product Attention)
# ============================================================

def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    is_causal: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    PyTorch SDPA: 统一 Attention API，运行时自动选择 backend
    dispatch 优先级: FlashAttention > Memory Efficient > Math
    """
    stats = {"method": "sdpa"}
    start = time.perf_counter()
    output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    elapsed = (time.perf_counter() - start) * 1000

    batch, num_heads, seq_len, head_dim = q.shape
    elem_bytes = q.element_size()
    flops = 4 * seq_len * seq_len * head_dim * num_heads * batch

    stats.update({
        "time_ms": elapsed,
        "flops": flops,
        "dispatch_note": "SDPA 自动选择: FlashAttn > MemEfficient > Math (需满足条件)",
    })
    return output, stats


# ============================================================
# 5. Chunked Prefill 模拟
# ============================================================

@dataclass
class ChunkedPrefillResult:
    """Chunked Prefill 结果"""
    total_prefill_time_ms: float = 0.0
    num_chunks: int = 0
    chunk_times_ms: List[float] = field(default_factory=list)
    decode_blocked_ms: float = 0.0  # Decode 被阻塞的时间
    avg_chunk_size: int = 0


def simulate_chunked_prefill(seq_len: int, chunk_size: int,
                              engine_compute_time_per_token: float = 0.05,
                              decode_time_per_step: float = 0.5) -> ChunkedPrefillResult:
    """
    模拟 Chunked Prefill:
    - 完整 Prefill: 一次处理全部 prompt，期间 Decode 请求被阻塞
    - Chunked Prefill: 将长 prompt 分成多个 chunk，每个 chunk 之间让出执行机会给 Decode

    关键指标: Decode 被阻塞的最大时间 (影响 TPOT/P99)
    """
    result = ChunkedPrefillResult()
    num_chunks = math.ceil(seq_len / chunk_size)
    result.num_chunks = num_chunks
    result.avg_chunk_size = seq_len / num_chunks

    # 模拟时间线: 每个 chunk 执行后，让出一个 Decode step
    timeline = []
    for i in range(num_chunks):
        current_chunk_size = min(chunk_size, seq_len - i * chunk_size)
        chunk_time = current_chunk_size * engine_compute_time_per_token
        result.chunk_times_ms.append(chunk_time)
        result.total_prefill_time_ms += chunk_time
        timeline.append(("prefill_chunk", chunk_time))

        # chunk 之间让出 Decode 机会 (最后一个 chunk 后不需要)
        if i < num_chunks - 1:
            decode_gap = decode_time_per_step
            timeline.append(("decode_yield", decode_gap))

    # 计算 Decode 被阻塞的最长连续时间
    # 完整 Prefill: 全部 prefill 时间都阻塞 Decode
    # Chunked: 每个 chunk 时间内阻塞，但 chunk 之间有间隙
    max_blocked = max(result.chunk_times_ms) if result.chunk_times_ms else 0
    result.decode_blocked_ms = max_blocked

    return result


# ============================================================
# 6. Roofline 分析
# ============================================================

def roofline_analysis(flops: float, hbm_bytes: float,
                       peak_compute_tflops: float = 10.0,
                       peak_bandwidth_gb_s: float = 500.0) -> Dict:
    """
    Roofline 模型分析:
    - 算术强度 = FLOPs / Bytes
    - 计算上限 = peak_compute
    - 带宽上限 = peak_bandwidth * arithmetic_intensity
    - 实际性能 = min(计算上限, 带宽上限)
    - 计算受限: arithmetic_intensity > 平衡点
    - 访存受限: arithmetic_intensity < 平衡点
    """
    arithmetic_intensity = flops / hbm_bytes if hbm_bytes > 0 else 0
    # 平衡点: peak_compute / peak_bandwidth (FLOPs/byte)
    ridge_point = (peak_compute_tflops * 1e12) / (peak_bandwidth_gb_s * 1e9)

    if arithmetic_intensity >= ridge_point:
        bound_type = "compute-bound (计算受限)"
        achievable_flops = peak_compute_tflops * 1e12
    else:
        bound_type = "memory-bound (访存受限)"
        achievable_flops = peak_bandwidth_gb_s * 1e9 * arithmetic_intensity

    return {
        "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 4),
        "ridge_point_flops_per_byte": round(ridge_point, 4),
        "bound_type": bound_type,
        "peak_compute_tflops": peak_compute_tflops,
        "peak_bandwidth_gb_s": peak_bandwidth_gb_s,
        "achievable_performance_tflops": round(achievable_flops / 1e12, 4),
        "theoretical_time_ms": round(flops / achievable_flops * 1000, 3) if achievable_flops > 0 else 0,
    }


# ============================================================
# 7. 对比实验
# ============================================================

def generate_test_tensors(config: AttentionConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成测试用 Q/K/V 张量"""
    torch.manual_seed(42)
    shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(shape, dtype=torch.float32)
    v = torch.randn(shape, dtype=torch.float32)
    return q, k, v


def run_attention_comparison(seq_lens: List[int] = [128, 256, 512, 1024]) -> List[Dict]:
    """运行不同序列长度下的 Attention 方法对比"""
    results = []
    print(f"\n  {'SeqLen':<8} {'方法':<20} {'时间(ms)':<12} {'中间显存(MB)':<14} "
          f"{'HBM读(MB)':<12} {'HBM写(MB)':<12} {'算术强度':<10}")
    print(f"  {'-'*88}")

    for seq_len in seq_lens:
        config = AttentionConfig(seq_len=seq_len, num_heads=4, head_dim=64, block_size=64)
        q, k, v = generate_test_tensors(config)

        # 标准 Attention
        out_std, stats_std = standard_attention(q, k, v)
        print(f"  {seq_len:<8} {'standard':<20} {stats_std['time_ms']:<12.3f} "
              f"{stats_std['intermediate_memory_bytes']/1024**2:<14.3f} "
              f"{stats_std['hbm_reads_bytes']/1024**2:<12.3f} "
              f"{stats_std['hbm_writes_bytes']/1024**2:<12.3f} "
              f"{stats_std['arithmetic_intensity']:<10.4f}")

        # FlashAttention 模拟
        out_flash, stats_flash = flash_attention_sim(q, k, v, block_size=config.block_size)
        max_error = (out_std - out_flash).abs().max().item()
        print(f"  {'':<8} {'flash_attention':<20} {stats_flash['time_ms']:<12.3f} "
              f"{stats_flash['intermediate_memory_bytes']/1024**2:<14.3f} "
              f"{stats_flash['hbm_reads_bytes']/1024**2:<12.3f} "
              f"{stats_flash['hbm_writes_bytes']/1024**2:<12.3f} "
              f"{stats_flash['arithmetic_intensity']:<10.4f}")

        # SDPA
        out_sdpa, stats_sdpa = sdpa_attention(q, k, v)
        print(f"  {'':<8} {'sdpa':<20} {stats_sdpa['time_ms']:<12.3f} "
              f"{'N/A':<14} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

        print(f"  {'':<8} Flash vs Std max error: {max_error:.6f}")
        print()

        results.append({
            "seq_len": seq_len,
            "standard": {k: v for k, v in stats_std.items() if isinstance(v, (int, float, str))},
            "flash_attention": {k: v for k, v in stats_flash.items() if isinstance(v, (int, float, str))},
            "sdpa": {k: v for k, v in stats_sdpa.items() if isinstance(v, (int, float, str))},
            "max_error_flash_vs_std": max_error,
        })

    return results


def run_chunked_prefill_demo():
    """运行 Chunked Prefill 演示"""
    print(f"\n{'='*60}")
    print("  Chunked Prefill 演示")
    print(f"{'='*60}")
    print(f"  长 Prompt 一次完整 Prefill vs 分块 Prefill")
    print(f"  {'配置':<25} {'Chunk数':<8} {'总Prefill(ms)':<14} {'Decode最大阻塞(ms)':<16}")
    print(f"  {'-'*63}")

    seq_len = 2048
    # 完整 Prefill (chunk_size = seq_len)
    full = simulate_chunked_prefill(seq_len, chunk_size=seq_len)
    print(f"  {'完整 Prefill (1 chunk)':<25} {full.num_chunks:<8} "
          f"{full.total_prefill_time_ms:<14.2f} {full.decode_blocked_ms:<16.2f}")

    # 不同 chunk size
    for chunk_size in [512, 256, 128]:
        result = simulate_chunked_prefill(seq_len, chunk_size=chunk_size)
        print(f"  {'Chunked (' + str(chunk_size) + '/chunk)':<25} {result.num_chunks:<8} "
              f"{result.total_prefill_time_ms:<14.2f} {result.decode_blocked_ms:<16.2f}")

    print(f"\n  关键洞察: Chunked Prefill 以略增总 Prefill 时间为代价，")
    print(f"  显著降低 Decode 请求被阻塞的最大连续时间，改善 P99/TPOT 抖动。")


def run_roofline_demo():
    """运行 Roofline 分析演示"""
    print(f"\n{'='*60}")
    print("  Roofline 分析演示 (假设 A100: 19.5 TFLOPS FP32, 1555 GB/s)")
    print(f"{'='*60}")

    configs = [
        ("标准 Attention N=512", 4*512*512*64*4, 3*512*64*4 + 2*512*512*4 + 512*64*4),
        ("FlashAttn N=512", 4*512*512*64*4, (1+8)*512*64*4 + 512*64*4),
        ("标准 Attention N=2048", 4*2048*2048*64*4, 3*2048*64*4 + 2*2048*2048*4 + 2048*64*4),
        ("FlashAttn N=2048", 4*2048*2048*64*4, (1+32)*2048*64*4 + 2048*64*4),
    ]

    print(f"  {'场景':<25} {'算术强度':<12} {'瓶颈类型':<25} {'理论性能(TFLOPS)':<18}")
    print(f"  {'-'*80}")
    for name, flops, hbm_bytes in configs:
        analysis = roofline_analysis(flops, hbm_bytes, peak_compute_tflops=19.5, peak_bandwidth_gb_s=1555)
        print(f"  {name:<25} {analysis['arithmetic_intensity_flops_per_byte']:<12.4f} "
              f"{analysis['bound_type']:<25} {analysis['achievable_performance_tflops']:<18.4f}")

    print(f"\n  关键洞察: 序列越长，标准 Attention 的 NxN 中间矩阵写回越多，")
    print(f"  算术强度越低，越容易成为访存受限；FlashAttention 通过消除 NxN 写回提升算术强度。")


# ============================================================
# 8. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Task1 · Prefill 与 Attention Kernel")
    print("=" * 60)

    # 1. Attention 数据流说明
    print(f"\n{'='*60}")
    print("  1. Attention 数据流 (从 Prompt 到 KV 状态)")
    print(f"{'='*60}")
    data_flow = [
        ("Q/K/V projection", "为整段 Prompt 生成 Q、K、V", "矩阵计算、权重读取", "Prefill time, TTFT"),
        ("Score & Softmax", "计算 token 两两关系并归一化", "中间工作集、HBM 读写", "Attention time, peak memory"),
        ("Value aggregation", "按权重聚合 V 并输出", "矩阵计算、片上复用", "Prefill time, TTFT"),
        ("KV Cache 写入", "保存后续 Decode 需要的 K/V", "Cache 容量、写入带宽", "peak memory, 并发容量"),
    ]
    for stage, what, cost, metrics in data_flow:
        print(f"  {stage}:")
        print(f"    发生: {what}")
        print(f"    成本: {cost}")
        print(f"    影响: {metrics}\n")

    # 2. Attention 方法对比
    print(f"{'='*60}")
    print("  2. 标准 Attention vs FlashAttention vs SDPA 对比")
    print(f"{'='*60}")
    comparison_results = run_attention_comparison([128, 256, 512])

    # 3. 访存优势量化
    print(f"{'='*60}")
    print("  3. FlashAttention 访存优势量化")
    print(f"{'='*60}")
    for r in comparison_results:
        n = r["seq_len"]
        std_intermediate = r["standard"]["intermediate_memory_bytes"] / 1024**2
        flash_intermediate = r["flash_attention"]["intermediate_memory_bytes"] / 1024**2
        std_writes = r["standard"]["hbm_writes_bytes"] / 1024**2
        flash_writes = r["flash_attention"]["hbm_writes_bytes"] / 1024**2
        print(f"  N={n}: 中间显存 {std_intermediate:.2f}MB → {flash_intermediate:.4f}MB "
              f"(减少 {std_intermediate/flash_intermediate:.0f}x)")
        print(f"         HBM写回 {std_writes:.2f}MB → {flash_writes:.2f}MB "
              f"(减少 {std_writes/flash_writes:.1f}x)")

    # 4. Chunked Prefill
    run_chunked_prefill_demo()

    # 5. Roofline 分析
    run_roofline_demo()

    # 6. Prefill vs Decode 工作负载差异
    print(f"\n{'='*60}")
    print("  4. Prefill vs Decode 工作负载差异")
    print(f"{'='*60}")
    print(f"  {'维度':<15} {'Prefill':<30} {'Decode':<30}")
    print(f"  {'-'*75}")
    comparisons = [
        ("输入形态", "一次处理完整/分块 Prompt", "每轮追加 1 个 token, 读取历史 KV"),
        ("典型瓶颈", "Attention 工作集、矩阵计算", "KV 读取、批次组织、请求调度"),
        ("优先观察", "TTFT、Prefill time、峰值显存", "TPOT、P99、Cache 访问"),
        ("优化方向", "FlashAttention、Chunked Prefill", "Continuous Batching、投机解码"),
    ]
    for dim, prefill, decode in comparisons:
        print(f"  {dim:<15} {prefill:<30} {decode:<30}")

    # 7. 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task1\\attention_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果已保存: {output_path}")

    print(f"\n{'='*60}")
    print("  Task1 完成!")
    print("  关键概念: Attention数据流 → 标准vsFlash → 访存模型 → ChunkedPrefill → Roofline")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
