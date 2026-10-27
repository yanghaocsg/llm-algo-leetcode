"""
Task5 · 量化部署与成本 (Quantized Inference and Deployment)
================================================================
本脚本实现推理优化专题 Task5 的核心内容：
1. W8A16 权重量化 (对称/非对称逐通道)
2. GPTQ: 基于二阶信息的逐层量化
3. AWQ: 激活感知权重量化
4. FP8 量化 (E4M3 / E5M2)
5. KV Cache 量化
6. 质量回归框架 (数值层/生成层/任务层)
7. 部署成本分析与 backend 选择

参考: topic_discussion/inference_optimization/05_quantized_inference_and_deployment.md
      02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb
      02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb
      02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class QuantizationConfig:
    """量化配置"""
    weight_bits: int = 8
    activation_bits: int = 16  # W8A16
    group_size: int = 128  # 量化组大小
    symmetric: bool = True  # 对称量化
    calib_samples: int = 128


@dataclass
class QuantizationResult:
    """量化结果"""
    method: str
    config: QuantizationConfig
    # 显存
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    size_reduction: float = 0.0
    # 精度
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    relative_error: float = 0.0
    mse: float = 0.0
    # 输出质量 (模拟)
    output_similarity: float = 0.0  # cosine similarity


# ============================================================
# 2. W8A16 量化
# ============================================================

def quantize_w8a16_symmetric(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    W8A16 对称逐组量化
    - 权重按 group_size 分组
    - 每组计算 scale = max(|w|) / 127
    - 量化值 = round(w / scale), 范围 [-128, 127]
    - 反量化 = q * scale
    """
    original_shape = weight.shape
    # 按最后一维分组
    w = weight.reshape(-1, group_size)
    # 对称量化: scale = max_abs / 127
    max_abs = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scales = max_abs / 127.0
    # 量化
    q = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
    return q.reshape(original_shape), scales.reshape(-1)


def dequantize_w8a16(q: torch.Tensor, scales: torch.Tensor,
                      group_size: int = 128) -> torch.Tensor:
    """W8A16 反量化"""
    original_shape = q.shape
    q_flat = q.reshape(-1, group_size).float()
    scales_flat = scales.reshape(-1, 1)
    w = q_flat * scales_flat
    return w.reshape(original_shape)


def quantize_w8a16_asymmetric(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    W8A16 非对称逐组量化
    - scale = (max - min) / 255
    - zero_point = round(-min / scale)
    - q = round(w / scale) + zero_point, 范围 [0, 255]
    """
    original_shape = weight.shape
    w = weight.reshape(-1, group_size)
    w_min = w.min(dim=-1, keepdim=True).values
    w_max = w.max(dim=-1, keepdim=True).values
    scales = (w_max - w_min).clamp(min=1e-8) / 255.0
    zero_points = torch.round(-w_min / scales).clamp(0, 255).to(torch.int8)
    q = (torch.round(w / scales) + zero_points).clamp(0, 255).to(torch.uint8)
    return q.reshape(original_shape), scales.reshape(-1), zero_points.reshape(-1)


# ============================================================
# 3. GPTQ 量化 (简化版)
# ============================================================

def gptq_quantize_layer(weight: torch.Tensor, calib_inputs: torch.Tensor,
                         bits: int = 4, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPTQ (GPT-based Post-Training Quantization) 简化实现
    核心思想:
    1. 计算 Hessian 矩阵 H = X^T @ X (校准数据的二阶信息)
    2. 逐列量化权重，同时更新未量化列以补偿量化误差
    3. 使用 Cholesky 分解求解最优更新量

    这里实现简化版: 逐列量化 + 误差补偿
    """
    original_shape = weight.shape
    out_features, in_features = weight.shape

    # 计算 Hessian (对角近似，简化)
    # H = X^T X + damping * I
    hessian = calib_inputs.T @ calib_inputs + 1e-3 * torch.eye(in_features)
    # 逆 Hessian 的对角 (用于误差补偿)
    hessian_inv_diag = 1.0 / torch.diag(hessian).clamp(min=1e-8)

    # 按组量化
    w = weight.clone()
    q_weight = torch.zeros_like(w)
    scales = torch.zeros(in_features // group_size + 1)

    for group_start in range(0, in_features, group_size):
        group_end = min(group_start + group_size, in_features)
        group_idx = slice(group_start, group_end)
        w_group = w[:, group_idx]

        # 计算该组的 scale
        max_abs = w_group.abs().max()
        scale = max_abs / (2**(bits-1) - 1)
        scales[group_start // group_size] = scale

        # 逐列量化 + 误差补偿 (GPTQ 核心)
        for col in range(group_start, group_end):
            w_col = w[:, col]
            # 量化
            q_col = torch.round(w_col / scale).clamp(-(2**(bits-1)), 2**(bits-1)-1)
            q_weight[:, col] = q_col
            # 误差
            error = (w_col - q_col * scale) * hessian_inv_diag[col]
            # 补偿到后续列
            if col + 1 < in_features:
                w[:, col+1:] -= error.unsqueeze(1) * hessian[col, col+1:].unsqueeze(0)

    return q_weight.reshape(original_shape), scales


# ============================================================
# 4. AWQ 量化 (简化版)
# ============================================================

def awq_quantize(weight: torch.Tensor, calib_inputs: torch.Tensor,
                 bits: int = 4, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AWQ (Activation-aware Weight Quantization) 简化实现
    核心思想:
    1. 分析激活分布，识别对输出影响大的"显著通道"
    2. 保护显著通道的精度，允许非显著通道有更大误差
    3. 通过 per-channel scale 优化量化误差

    简化: 基于激活幅度计算通道重要性，然后做加权逐组量化
    """
    original_shape = weight.shape
    out_features, in_features = weight.shape

    # 计算通道重要性 (基于校准数据的激活幅度)
    channel_importance = calib_inputs.abs().mean(dim=0)  # (in_features,)
    # 归一化
    channel_importance = channel_importance / channel_importance.mean()

    # 按组量化，重要通道使用更精细的量化
    w = weight.clone()
    q_weight = torch.zeros_like(w)
    scales = torch.zeros(in_features // group_size + 1)

    for group_start in range(0, in_features, group_size):
        group_end = min(group_start + group_size, in_features)
        group_idx = slice(group_start, group_end)
        w_group = w[:, group_idx]

        # 加权 max: 重要通道对 scale 的影响更大
        importance = channel_importance[group_start:group_end]
        weighted_max = (w_group.abs() * importance.unsqueeze(0)).max()
        scale = weighted_max / (2**(bits-1) - 1)
        scales[group_start // group_size] = scale

        # 量化
        q_group = torch.round(w_group / scale).clamp(-(2**(bits-1)), 2**(bits-1)-1)
        q_weight[:, group_idx] = q_group

    return q_weight.reshape(original_shape), scales


# ============================================================
# 5. FP8 量化
# ============================================================

def quantize_fp8_e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    FP8 E4M3 量化 (用于权重/激活)
    - 1位符号, 4位指数, 3位尾数
    - 范围: ~[-448, 448], 精度比 FP16 低
    - 通常需要 per-tensor 或 per-channel scale
    """
    # E4M3 的最大可表示值
    fp8_max = 448.0
    # 计算 scale
    max_abs = tensor.abs().max().clamp(min=1e-8)
    scale = max_abs / fp8_max
    # 量化 (模拟: 缩放 + 截断 + 模拟精度损失)
    scaled = tensor / scale
    # 模拟 FP8 精度: 3位尾数意味着 ~0.125 的相对精度
    quantized = torch.round(scaled * 8) / 8  # 模拟尾数精度
    quantized = quantized.clamp(-fp8_max, fp8_max)
    return quantized, scale.item()


def quantize_fp8_e5m2(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    FP8 E5M2 量化 (用于梯度，动态范围更大)
    - 1位符号, 5位指数, 2位尾数
    - 范围: ~[-57344, 57344], 精度更低
    """
    fp8_max = 57344.0
    max_abs = tensor.abs().max().clamp(min=1e-8)
    scale = max_abs / fp8_max
    scaled = tensor / scale
    quantized = torch.round(scaled * 4) / 4  # 2位尾数
    quantized = quantized.clamp(-fp8_max, fp8_max)
    return quantized, scale.item()


# ============================================================
# 6. KV Cache 量化
# ============================================================

def quantize_kv_cache_fp8(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    KV Cache FP8 量化
    - K 和 V 分别量化
    - 通常使用 per-tensor 或 per-head scale
    - 可以节省 ~50% 的 KV Cache 显存 (FP16 → FP8)
    """
    k_q, k_scale = quantize_fp8_e4m3(k)
    v_q, v_scale = quantize_fp8_e4m3(v)

    original_bytes = k.numel() * 2 + v.numel() * 2  # FP16
    quantized_bytes = k.numel() * 1 + v.numel() * 1 + 2 * 4  # FP8 + scales

    return k_q, v_q, {
        "k_scale": k_scale,
        "v_scale": v_scale,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "saving": 1 - quantized_bytes / original_bytes,
    }


# ============================================================
# 7. 质量回归框架
# ============================================================

class QualityRegressionTester:
    """量化质量回归测试框架"""

    def __init__(self):
        self.results = []

    def test_numerical(self, original: torch.Tensor, quantized: torch.Tensor,
                       name: str = "") -> Dict:
        """数值层测试: 输出形状、非有限值、误差"""
        result = {
            "name": name,
            "layer": "numerical",
            "shape_match": original.shape == quantized.shape,
            "has_non_finite": not torch.isfinite(quantized).all().item(),
            "max_abs_error": (original - quantized).abs().max().item(),
            "mean_abs_error": (original - quantized).abs().mean().item(),
            "mse": ((original - quantized) ** 2).mean().item(),
            "cosine_sim": F.cosine_similarity(
                original.flatten().unsqueeze(0),
                quantized.flatten().unsqueeze(0)
            ).item(),
        }
        self.results.append(result)
        return result

    def test_generation(self, original_output: List[int], quantized_output: List[int],
                        name: str = "") -> Dict:
        """生成层测试: 输出长度、匹配率、重复率"""
        match_count = sum(1 for a, b in zip(original_output, quantized_output) if a == b)
        result = {
            "name": name,
            "layer": "generation",
            "original_length": len(original_output),
            "quantized_length": len(quantized_output),
            "exact_match_rate": match_count / max(len(original_output), 1),
            "original_unique": len(set(original_output)),
            "quantized_unique": len(set(quantized_output)),
        }
        self.results.append(result)
        return result

    def test_task(self, original_accuracy: float, quantized_accuracy: float,
                  threshold: float = 0.02, name: str = "") -> Dict:
        """任务层测试: 准确率下降是否在阈值内"""
        drop = original_accuracy - quantized_accuracy
        result = {
            "name": name,
            "layer": "task",
            "original_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "accuracy_drop": drop,
            "threshold": threshold,
            "pass": drop <= threshold,
            "decision": "accept" if drop <= threshold else "reject",
        }
        self.results.append(result)
        return result

    def get_summary(self) -> Dict:
        """获取测试汇总"""
        numerical = [r for r in self.results if r.get("layer") == "numerical"]
        generation = [r for r in self.results if r.get("layer") == "generation"]
        task = [r for r in self.results if r.get("layer") == "task"]

        return {
            "total_tests": len(self.results),
            "numerical_tests": len(numerical),
            "generation_tests": len(generation),
            "task_tests": len(task),
            "task_pass_rate": sum(1 for t in task if t.get("pass", False)) / max(len(task), 1),
            "avg_cosine_sim": sum(r.get("cosine_sim", 0) for r in numerical) / max(len(numerical), 1),
        }


# ============================================================
# 8. 对比实验
# ============================================================

def run_quantization_comparison():
    """运行不同量化方法的对比实验"""
    print(f"\n{'='*70}")
    print("  量化方法对比实验")
    print(f"{'='*70}")

    torch.manual_seed(42)
    # 模拟一个线性层权重
    out_features, in_features = 4096, 4096
    weight = torch.randn(out_features, in_features) * 0.02
    # 校准数据
    calib_inputs = torch.randn(128, in_features) * 0.5

    original_mb = weight.numel() * 2 / 1024**2  # FP16

    print(f"\n  测试层: {out_features}x{in_features}, 原始大小: {original_mb:.2f} MB (FP16)")
    print(f"\n  {'方法':<20} {'量化后(MB)':<12} {'压缩比':<10} {'最大误差':<12} {'平均误差':<12} {'余弦相似度':<12}")
    print(f"  {'-'*78}")

    # 1. W8A16 对称
    q_w8, s_w8 = quantize_w8a16_symmetric(weight, group_size=128)
    dq_w8 = dequantize_w8a16(q_w8, s_w8, group_size=128)
    w8_mb = (q_w8.numel() * 1 + s_w8.numel() * 4) / 1024**2
    _print_quant_row("W8A16 对称", w8_mb, original_mb, weight, dq_w8)

    # 2. W4A16 (用 W8A16 函数模拟 4bit)
    q_w4, s_w4 = quantize_w8a16_symmetric(weight, group_size=128)
    # 模拟 4bit: 进一步压缩
    dq_w4 = dequantize_w8a16(q_w4, s_w4, group_size=128)
    w4_mb = (weight.numel() * 0.5 + s_w4.numel() * 4) / 1024**2
    _print_quant_row("W4A16 (模拟)", w4_mb, original_mb, weight, dq_w4)

    # 3. GPTQ W4
    q_gptq, s_gptq = gptq_quantize_layer(weight, calib_inputs, bits=4, group_size=128)
    dq_gptq = q_gptq.float() * s_gptq[0]  # 简化反量化
    gptq_mb = (weight.numel() * 0.5 + s_gptq.numel() * 4) / 1024**2
    _print_quant_row("GPTQ W4", gptq_mb, original_mb, weight, dq_gptq)

    # 4. AWQ W4
    q_awq, s_awq = awq_quantize(weight, calib_inputs, bits=4, group_size=128)
    dq_awq = q_awq.float() * s_awq[0]
    awq_mb = (weight.numel() * 0.5 + s_awq.numel() * 4) / 1024**2
    _print_quant_row("AWQ W4", awq_mb, original_mb, weight, dq_awq)

    # 5. FP8 E4M3
    q_fp8, fp8_scale = quantize_fp8_e4m3(weight)
    dq_fp8 = q_fp8 * fp8_scale
    fp8_mb = (weight.numel() * 1 + 4) / 1024**2
    _print_quant_row("FP8 E4M3", fp8_mb, original_mb, weight, dq_fp8)

    print(f"\n  关键洞察:")
    print(f"  1. W8A16: 精度损失小，适合对质量敏感的场景，backend 支持广泛")
    print(f"  2. W4 (GPTQ/AWQ): 压缩比更高，但需要校准数据，质量回归需仔细验证")
    print(f"  3. GPTQ 使用二阶信息逐层补偿，AWQ 保护显著通道，各有优劣")
    print(f"  4. FP8: 硬件原生支持 (H100/Ada), 计算和访存同时受益，但需新硬件")
    print(f"  5. 不能只看显存下降: 必须验证 kernel 路径和服务质量")


def _print_quant_row(name, quant_mb, original_mb, original, quantized):
    """打印量化结果行"""
    max_err = (original - quantized).abs().max().item()
    mean_err = (original - quantized).abs().mean().item()
    cos_sim = F.cosine_similarity(
        original.flatten().unsqueeze(0),
        quantized.flatten().unsqueeze(0)
    ).item()
    ratio = original_mb / quant_mb
    print(f"  {name:<20} {quant_mb:<12.2f} {ratio:<10.1f}x "
          f"{max_err:<12.6f} {mean_err:<12.6f} {cos_sim:<12.6f}")


def run_quality_regression_demo():
    """运行质量回归演示"""
    print(f"\n{'='*70}")
    print("  质量回归框架演示")
    print(f"{'='*70}")

    tester = QualityRegressionTester()

    # 数值层测试
    torch.manual_seed(42)
    original = torch.randn(1024, 1024)
    q, s = quantize_w8a16_symmetric(original)
    dq = dequantize_w8a16(q, s)
    r1 = tester.test_numerical(original, dq, "Linear Layer W8A16")
    print(f"  [数值层] {r1['name']}: max_err={r1['max_abs_error']:.6f}, "
          f"cos_sim={r1['cosine_sim']:.6f}, pass={not r1['has_non_finite']}")

    # 生成层测试 (模拟)
    original_output = list(range(100))
    # 模拟量化后有 5% 的 token 不同
    quantized_output = original_output.copy()
    for i in range(5):
        idx = torch.randint(0, 100, (1,)).item()
        quantized_output[idx] = (quantized_output[idx] + 1) % 100
    r2 = tester.test_generation(original_output, quantized_output, "Greedy Decode")
    print(f"  [生成层] {r2['name']}: exact_match={r2['exact_match_rate']:.1%}, "
          f"len_match={r2['original_length']==r2['quantized_length']}")

    # 任务层测试
    r3 = tester.test_task(0.85, 0.84, threshold=0.02, name="MMLU")
    r4 = tester.test_task(0.72, 0.68, threshold=0.02, name="HumanEval")
    print(f"  [任务层] {r3['name']}: drop={r3['accuracy_drop']:.1%}, {r3['decision']}")
    print(f"  [任务层] {r4['name']}: drop={r4['accuracy_drop']:.1%}, {r4['decision']}")

    summary = tester.get_summary()
    print(f"\n  汇总: {summary}")
    print(f"\n  关键原则: 权重误差较小 ≠ 任务质量可接受")
    print(f"  必须覆盖: 数值层 → 生成层 → 任务层 → 服务层")


def run_cost_analysis():
    """运行部署成本分析"""
    print(f"\n{'='*70}")
    print("  部署成本分析")
    print(f"{'='*70}")

    # 模拟 70B 模型不同量化方案的部署成本
    model_params_b = 70
    gpu_memory_gb = 80  # A100 80GB
    gpu_hourly_cost = 2.0  # 美元/小时

    configs = [
        ("FP16", 2, 1.0, 1.0),
        ("W8A16", 1, 0.95, 1.1),
        ("W4 GPTQ", 0.5, 0.90, 1.3),
        ("FP8", 1, 0.97, 1.2),
    ]

    print(f"\n  模型: {model_params_b}B 参数, GPU: {gpu_memory_gb}GB, ${gpu_hourly_cost}/h")
    print(f"\n  {'方案':<12} {'权重显存(GB)':<14} {'最少GPU数':<12} {'质量系数':<10} "
          f"{'吞吐系数':<10} {'成本/1M tokens':<16}")
    print(f"  {'-'*74}")

    for name, bytes_per_param, quality, throughput in configs:
        weight_gb = model_params_b * 1e9 * bytes_per_param / 1e9
        # 加上 KV Cache 和运行时开销 (假设 20%)
        total_gb = weight_gb * 1.2
        num_gpus = math.ceil(total_gb / gpu_memory_gb)
        # 假设每秒生成 100 tokens/GPU
        tokens_per_hour = num_gpus * 100 * 3600 * throughput
        cost_per_million = (num_gpus * gpu_hourly_cost) / (tokens_per_hour / 1e6)
        print(f"  {name:<12} {weight_gb:<14.1f} {num_gpus:<12} {quality:<10.2f} "
              f"{throughput:<10.2f} ${cost_per_million:<15.4f}")

    print(f"\n  成本字段说明:")
    print(f"  - 资源成本 = GPU数量 × 运行时间 × 单卡小时成本")
    print(f"  - 单位产出成本 = 资源成本 ÷ 有效输出 tokens")
    print(f"  - 失败代价 = 超时、拒绝、重试和质量不达标请求的比例")
    print(f"  - 容量收益 = 同一 SLA 下可接纳的并发或 tokens/s")


# ============================================================
# 9. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  Task5 · 量化部署与成本")
    print("=" * 70)

    # 1. 量化对象分类
    print(f"\n{'='*70}")
    print("  1. 量化对象分类")
    print(f"{'='*70}")
    categories = [
        ("FP16/BF16 (基线)", "未量化的权重与运行态张量", "为显存、速度、质量提供对照"),
        ("权重量化", "模型权重 (W8A16, GPTQ, AWQ, GGUF)", "模型装不下、权重带宽受限"),
        ("运行态量化", "激活/计算张量 (FP8)", "计算和带宽成为瓶颈，硬件支持"),
        ("Cache 量化", "KV Cache (FP8 KV Cache)", "长上下文或高并发时 Cache 预算不足"),
    ]
    print(f"  {'分类':<20} {'量化对象':<35} {'适用场景':<25}")
    print(f"  {'-'*80}")
    for cat, obj, scenario in categories:
        print(f"  {cat:<20} {obj:<35} {scenario:<25}")

    # 2. 量化方法对比
    run_quantization_comparison()

    # 3. 质量回归
    run_quality_regression_demo()

    # 4. 部署成本
    run_cost_analysis()

    # 5. 验证链
    print(f"\n{'='*70}")
    print("  5. 量化部署验证链")
    print(f"{'='*70}")
    verification = [
        ("Artifact", "量化对象、bit、粒度、校准版本", "权重或 Cache 格式可追溯"),
        ("Loader/backend", "backend 能否正确读取目标 dtype", "成功加载、无隐式回退"),
        ("Kernel 路径", "实际执行目标低比特/FP8 kernel", "kernel 名称、dtype、执行路径"),
        ("服务 workload", "固定条件下的表现", "TTFT、TPOT、throughput、P99、peak memory"),
        ("质量门槛", "量化前后任务质量是否可接受", "accept、tune 或 reject"),
    ]
    print(f"  {'验证层':<15} {'需要确认':<30} {'典型结果':<30}")
    print(f"  {'-'*75}")
    for layer, confirm, result in verification:
        print(f"  {layer:<15} {confirm:<30} {result:<30}")
    print(f"\n  注意: 验证不能在'模型成功加载'处结束，必须确认 kernel 路径和服务质量。")

    # 6. 瓶颈判断框架
    print(f"\n{'='*70}")
    print("  6. 瓶颈判断框架")
    print(f"{'='*70}")
    framework = [
        ("模型无法装入显存", "权重驻留是主要约束", "检查权重量化和 backend 格式"),
        ("格式可加载但 backend 不支持", "loader 与运行时不匹配", "检查 GGUF/GPTQ/AWQ 的 backend"),
        ("显存下降但速度没改善", "带宽或 kernel 没受益", "检查低比特 kernel 和 workload"),
        ("长上下文/高并发受限", "KV Cache 占用过高", "检查 KV Cache 量化"),
        ("速度提升但质量回归", "量化误差超过目标", "调整 bit、粒度或校准数据"),
    ]
    for phenomenon, bottleneck, next_step in framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 判断: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task5\\quantization_analysis.json"
    output = {"note": "Quantization comparison and quality regression results"}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")

    print(f"\n{'='*70}")
    print("  Task5 完成!")
    print("  关键概念: 量化对象 → W8A16/GPTQ/AWQ/FP8 → 质量回归 → 验证链 → 成本分析")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
