"""
Task2 · 投机式生成与 Decode 加速 (Speculative Decoding and Decode Acceleration)
====================================================================================
本脚本实现推理优化专题 Task2 的核心内容：
1. 贪心解码基线 (Greedy Decoding)
2. 随机采样 (Temperature / Top-K / Top-P)
3. 经典投机解码 (Draft/Target 验证)
4. 多 Token 投机式推进 (Multi-Token Speculation)
5. 接受率、有效推进量与回退成本分析
6. 质量-速度权衡框架

参考: topic_discussion/inference_optimization/03_decoding_strategies.md
      02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb
      02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb
      02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb
"""

import time
import math
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class DecodeConfig:
    """解码配置"""
    vocab_size: int = 32000
    hidden_dim: int = 1024
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0  # 0 表示不使用
    top_p: float = 1.0
    seed: int = 42


@dataclass
class DecodeResult:
    """单次解码结果"""
    method: str
    generated_tokens: List[int]
    num_steps: int
    total_time_ms: float
    tpot_ms: float
    # 投机解码特有
    acceptance_rate: float = 0.0
    effective_tokens_per_step: float = 0.0
    num_rejections: int = 0
    draft_model_calls: int = 0
    target_model_calls: int = 0


# ============================================================
# 2. 模拟语言模型 (用于解码实验)
# ============================================================

class SimulatedLM:
    """
    模拟语言模型: 给定输入序列，返回下一个 token 的 logits 分布
    使用简单的 n-gram + 随机模式模拟真实模型的输出分布
    """

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 1024,
                 model_type: str = "target", compute_time_ms: float = 5.0):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.compute_time_ms = compute_time_ms
        self.call_count = 0
        # 模拟一个简单的"语言模型"：某些 token 后倾向于跟随特定 token
        torch.manual_seed(hash(model_type) % (2**31))
        self.transition_bias = torch.randn(vocab_size, vocab_size) * 0.1

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 返回 logits (batch, vocab_size)
        模拟计算延迟
        """
        self.call_count += 1
        time.sleep(self.compute_time_ms / 1000)

        batch_size = input_ids.shape[0]
        # 基于最后一个 token 的偏置 + 随机噪声生成 logits
        last_token = input_ids[:, -1]
        base_logits = self.transition_bias[last_token]  # (batch, vocab)
        # 添加上下文相关的噪声 (模拟模型对上下文的感知)
        context_len = input_ids.shape[1]
        noise_scale = 1.0 / (1 + math.log2(context_len + 1))
        noise = torch.randn_like(base_logits) * noise_scale
        logits = base_logits + noise
        return logits


# ============================================================
# 3. 基础解码策略
# ============================================================

def greedy_decode(model: SimulatedLM, input_ids: torch.Tensor,
                  max_new_tokens: int, eos_token_id: int = 2) -> DecodeResult:
    """
    贪心解码: 每步选择 logits 最大的 token
    确定性基线，适合作为性能对照
    """
    result = DecodeResult(method="greedy", generated_tokens=[], num_steps=0,
                          total_time_ms=0.0, tpot_ms=0.0)
    current = input_ids.clone()
    start = time.perf_counter()

    for step in range(max_new_tokens):
        logits = model(current)
        next_token = logits.argmax(dim=-1)  # (batch,)
        result.generated_tokens.append(next_token[0].item())
        current = torch.cat([current, next_token.unsqueeze(-1)], dim=-1)
        result.num_steps += 1
        if next_token[0].item() == eos_token_id:
            break

    result.total_time_ms = (time.perf_counter() - start) * 1000
    result.tpot_ms = result.total_time_ms / result.num_steps if result.num_steps > 0 else 0
    return result


def sample_decode(model: SimulatedLM, input_ids: torch.Tensor,
                  config: DecodeConfig, eos_token_id: int = 2) -> DecodeResult:
    """
    随机采样: 支持 temperature、top-k、top-p
    """
    result = DecodeResult(method=f"sample(T={config.temperature},k={config.top_k},p={config.top_p})",
                          generated_tokens=[], num_steps=0, total_time_ms=0.0, tpot_ms=0.0)
    current = input_ids.clone()
    torch.manual_seed(config.seed)
    start = time.perf_counter()

    for step in range(config.max_new_tokens):
        logits = model(current)

        # Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-K filtering
        if config.top_k > 0:
            top_k_vals, _ = torch.topk(logits, config.top_k, dim=-1)
            min_val = top_k_vals[:, -1:]
            logits = logits.masked_fill(logits < min_val, float('-inf'))

        # Top-P (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cum_probs > config.top_p
            # Shift right to keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        result.generated_tokens.append(next_token[0].item())
        current = torch.cat([current, next_token.unsqueeze(-1)], dim=-1)
        result.num_steps += 1
        if next_token[0].item() == eos_token_id:
            break

    result.total_time_ms = (time.perf_counter() - start) * 1000
    result.tpot_ms = result.total_time_ms / result.num_steps if result.num_steps > 0 else 0
    return result


# ============================================================
# 4. 经典投机解码 (Speculative Decoding)
# ============================================================

def speculative_decode(draft_model: SimulatedLM, target_model: SimulatedLM,
                       input_ids: torch.Tensor, max_new_tokens: int,
                       speculation_length: int = 4,
                       eos_token_id: int = 2) -> DecodeResult:
    """
    经典投机解码 (Leviathan et al., 2023):
    1. Draft 模型自回归生成 K 个候选 token
    2. Target 模型并行验证这 K 个 token (一次 forward)
    3. 从左到右检查: 接受匹配的 token，第一个不匹配处用 target 分布采样修正
    4. 如果全部 K 个都接受，额外从 target 分布采样 1 个 bonus token

    关键指标:
    - acceptance_rate: 平均每个候选被接受的比例
    - effective_tokens_per_step: 每次 target forward 实际产出的 token 数
    - 速度提升 = effective_tokens_per_step / (1 + draft_cost_ratio)
    """
    result = DecodeResult(
        method=f"speculative(K={speculation_length})",
        generated_tokens=[], num_steps=0, total_time_ms=0.0, tpot_ms=0.0,
        acceptance_rate=0.0, effective_tokens_per_step=0.0,
        num_rejections=0, draft_model_calls=0, target_model_calls=0
    )
    current = input_ids.clone()
    total_accepted = 0
    total_drafted = 0
    start = time.perf_counter()

    while len(result.generated_tokens) < max_new_tokens:
        # Step 1: Draft 模型生成 K 个候选
        draft_tokens = []
        draft_current = current.clone()
        for _ in range(speculation_length):
            draft_logits = draft_model(draft_current)
            draft_token = draft_logits.argmax(dim=-1)
            draft_tokens.append(draft_token[0].item())
            draft_current = torch.cat([draft_current, draft_token.unsqueeze(-1)], dim=-1)
            result.draft_model_calls += 1
            if draft_token[0].item() == eos_token_id:
                break

        # Step 2: Target 模型验证
        # 真实模型中一次 forward 处理 K+1 个位置 (causal attention)
        # 这里简化模型只能返回最后一个位置的 logits，因此用 K+1 次顺序调用模拟
        target_preds = []
        verify_current = current.clone()
        for _ in range(len(draft_tokens) + 1):
            t_logits = target_model(verify_current)  # (batch, vocab)
            target_preds.append(t_logits[0].argmax().item())
            result.target_model_calls += 1
            # 用 draft token 继续 (模拟并行验证中的因果掩码)
            if len(target_preds) <= len(draft_tokens):
                next_t = torch.tensor([[draft_tokens[len(target_preds) - 1]]], dtype=torch.long)
                verify_current = torch.cat([verify_current, next_t], dim=-1)

        # Step 3: 从左到右验证
        accepted_count = 0
        for i in range(len(draft_tokens)):
            target_pred = target_preds[i]
            draft_pred = draft_tokens[i]

            if target_pred == draft_pred:
                # 接受
                result.generated_tokens.append(draft_pred)
                accepted_count += 1
                total_accepted += 1
                if draft_pred == eos_token_id:
                    break
            else:
                # 拒绝: 用 target 分布采样修正 (这里用贪心)
                result.generated_tokens.append(target_pred)
                result.num_rejections += 1
                break

        total_drafted += len(draft_tokens)

        # 如果全部接受，额外 bonus token
        if accepted_count == len(draft_tokens) and accepted_count > 0:
            bonus_token = target_preds[-1]
            result.generated_tokens.append(bonus_token)
            total_accepted += 1
            if bonus_token == eos_token_id:
                break

        # 更新 current
        new_tokens = result.generated_tokens[-(accepted_count + (1 if accepted_count == len(draft_tokens) else 1)):]
        current = torch.cat([current, torch.tensor([new_tokens], dtype=torch.long)], dim=-1)
        result.num_steps += 1

        if len(result.generated_tokens) >= max_new_tokens:
            break

    result.total_time_ms = (time.perf_counter() - start) * 1000
    result.acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    result.effective_tokens_per_step = len(result.generated_tokens) / result.num_steps if result.num_steps > 0 else 0
    result.tpot_ms = result.total_time_ms / len(result.generated_tokens) if result.generated_tokens else 0
    return result


# ============================================================
# 5. 多 Token 投机式推进 (Multi-Token Speculation)
# ============================================================

def multi_token_speculative_decode(draft_model: SimulatedLM, target_model: SimulatedLM,
                                    input_ids: torch.Tensor, max_new_tokens: int,
                                    num_candidates: int = 4,
                                    eos_token_id: int = 2) -> DecodeResult:
    """
    多 Token 投机式推进:
    与经典投机解码不同，多 token 方法在单个位置提出多个候选，
    或使用多预测头同时预测未来多个 token。
    这里模拟"树状验证"：draft 提出多条候选路径，target 选择最优路径。

    有效推进量更高，但验证成本也更高。
    """
    result = DecodeResult(
        method=f"multi_token(candidates={num_candidates})",
        generated_tokens=[], num_steps=0, total_time_ms=0.0, tpot_ms=0.0,
        acceptance_rate=0.0, effective_tokens_per_step=0.0,
        num_rejections=0, draft_model_calls=0, target_model_calls=0
    )
    current = input_ids.clone()
    total_accepted = 0
    total_proposed = 0
    start = time.perf_counter()

    while len(result.generated_tokens) < max_new_tokens:
        # Draft: 为每个位置提出 num_candidates 个候选 (模拟多预测头)
        candidates_per_position = []
        draft_current = current.clone()

        for pos in range(num_candidates):
            draft_logits = draft_model(draft_current)
            # 取 top-k 候选
            top_candidates = draft_logits[0].topk(num_candidates).indices.tolist()
            candidates_per_position.append(top_candidates)
            result.draft_model_calls += 1
            # 用第一个候选继续 (模拟自回归草稿)
            next_token = torch.tensor([[top_candidates[0]]], dtype=torch.long)
            draft_current = torch.cat([draft_current, next_token], dim=-1)

        # Target: 验证 (顺序调用模拟树状验证)
        target_preds = []
        verify_current = current.clone()
        for pos in range(num_candidates):
            t_logits = target_model(verify_current)
            target_preds.append(t_logits[0].argmax().item())
            result.target_model_calls += 1
            # 用已接受的 token 继续
            next_t = torch.tensor([[target_preds[-1]]], dtype=torch.long)
            verify_current = torch.cat([verify_current, next_t], dim=-1)

        # 从候选中选择与 target 最匹配的路径
        accepted_in_step = 0
        for pos in range(num_candidates):
            if len(result.generated_tokens) >= max_new_tokens:
                break
            target_pred = target_preds[pos]
            # 检查 target 预测是否在候选中
            if target_pred in candidates_per_position[pos]:
                result.generated_tokens.append(target_pred)
                accepted_in_step += 1
                total_accepted += 1
                if target_pred == eos_token_id:
                    break
            else:
                result.generated_tokens.append(target_pred)
                result.num_rejections += 1
                break

        total_proposed += num_candidates
        result.num_steps += 1

        # 更新 current
        new_count = accepted_in_step + (1 if accepted_in_step < num_candidates else 0)
        new_tokens = result.generated_tokens[-new_count:]
        current = torch.cat([current, torch.tensor([new_tokens], dtype=torch.long)], dim=-1)

    result.total_time_ms = (time.perf_counter() - start) * 1000
    result.acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0
    result.effective_tokens_per_step = len(result.generated_tokens) / result.num_steps if result.num_steps > 0 else 0
    result.tpot_ms = result.total_time_ms / len(result.generated_tokens) if result.generated_tokens else 0
    return result


# ============================================================
# 6. 对比实验与分析
# ============================================================

def run_decode_comparison() -> List[DecodeResult]:
    """运行所有解码策略的对比实验"""
    print(f"\n{'='*70}")
    print("  解码策略对比实验")
    print(f"{'='*70}")

    vocab_size = 1000  # 缩小词表加速实验
    hidden_dim = 512
    max_new_tokens = 32

    # 初始化模型
    target_model = SimulatedLM(vocab_size, hidden_dim, "target", compute_time_ms=3.0)
    draft_model = SimulatedLM(vocab_size, hidden_dim, "draft", compute_time_ms=1.0)

    # 输入
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 16))

    results = []

    # 1. 贪心解码
    print("  运行: 贪心解码 ...")
    r_greedy = greedy_decode(target_model, input_ids, max_new_tokens)
    results.append(r_greedy)

    # 2. 采样 (temperature=0.7)
    print("  运行: 采样 (T=0.7) ...")
    config_sample = DecodeConfig(vocab_size=vocab_size, max_new_tokens=max_new_tokens,
                                  temperature=0.7, top_k=50, top_p=0.9)
    r_sample = sample_decode(target_model, input_ids, config_sample)
    results.append(r_sample)

    # 3. 投机解码 K=2
    print("  运行: 投机解码 K=2 ...")
    r_spec2 = speculative_decode(draft_model, target_model, input_ids, max_new_tokens,
                                  speculation_length=2)
    results.append(r_spec2)

    # 4. 投机解码 K=4
    print("  运行: 投机解码 K=4 ...")
    r_spec4 = speculative_decode(draft_model, target_model, input_ids, max_new_tokens,
                                  speculation_length=4)
    results.append(r_spec4)

    # 5. 多 Token 投机
    print("  运行: 多 Token 投机 (candidates=4) ...")
    r_multi = multi_token_speculative_decode(draft_model, target_model, input_ids,
                                              max_new_tokens, num_candidates=4)
    results.append(r_multi)

    # 打印对比表
    print(f"\n  {'方法':<30} {'Token数':<8} {'步数':<6} {'TPOT(ms)':<10} "
          f"{'接受率':<8} {'有效推进':<10} {'回退数':<8}")
    print(f"  {'-'*80}")
    for r in results:
        acc = f"{r.acceptance_rate:.2f}" if r.acceptance_rate > 0 else "N/A"
        eff = f"{r.effective_tokens_per_step:.2f}" if r.effective_tokens_per_step > 0 else "1.00"
        rej = str(r.num_rejections) if r.num_rejections > 0 else "0"
        print(f"  {r.method:<30} {len(r.generated_tokens):<8} {r.num_steps:<6} "
              f"{r.tpot_ms:<10.2f} {acc:<8} {eff:<10} {rej:<8}")

    return results


def analyze_speculative_tradeoff():
    """分析投机解码的速度-质量权衡"""
    print(f"\n{'='*70}")
    print("  投机解码: 接受率与速度提升的理论关系")
    print(f"{'='*70}")
    print(f"  假设: target 单次 forward = T_t, draft 单次 forward = T_d (T_d < T_t)")
    print(f"  经典投机 K 个候选: 每轮成本 = K*T_d + T_t, 产出 = K*p + 1 (p=接受率)")
    print(f"  速度提升比 = (K*p + 1) * T_t / (K*T_d + T_t)\n")

    print(f"  {'K':<4} {'接受率p':<8} {'Draft/Target':<12} {'每轮产出':<10} {'每轮成本':<10} {'速度提升':<10}")
    print(f"  {'-'*54}")
    for k in [2, 4, 6, 8]:
        for p in [0.5, 0.7, 0.9]:
            draft_ratio = 0.3  # draft 耗时是 target 的 30%
            output = k * p + 1
            cost = k * draft_ratio + 1
            speedup = output / cost
            print(f"  {k:<4} {p:<8.1f} {draft_ratio:<12.1f} {output:<10.2f} "
                  f"{cost:<10.2f} {speedup:<10.2f}x")

    print(f"\n  关键洞察:")
    print(f"  1. 接受率 p 越高，速度提升越显著")
    print(f"  2. K 增大在高接受率时收益更大，但低接受率时可能反而变慢")
    print(f"  3. Draft 模型越轻量 (draft_ratio 越小)，投机解码收益越大")
    print(f"  4. 实际部署需测量真实接受率，不能只看理论上限")


def run_quality_check():
    """质量检查: 不同解码策略的输出多样性"""
    print(f"\n{'='*70}")
    print("  解码策略质量检查: 输出多样性与重复率")
    print(f"{'='*70}")

    vocab_size = 500
    hidden_dim = 256
    model = SimulatedLM(vocab_size, hidden_dim, "target", compute_time_ms=1.0)
    input_ids = torch.randint(0, vocab_size, (1, 8))

    configs = [
        ("贪心", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=1.0)),
        ("T=1.0", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=1.0)),
        ("T=0.7", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=0.7)),
        ("T=0.3", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=0.3)),
        ("top-p=0.9", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=1.0, top_p=0.9)),
        ("top-k=10", DecodeConfig(vocab_size=vocab_size, max_new_tokens=20, temperature=1.0, top_k=10)),
    ]

    print(f"  {'策略':<15} {'唯一token数':<12} {'重复率':<10} {'前5token':<30}")
    print(f"  {'-'*67}")
    for name, config in configs:
        if name == "贪心":
            result = greedy_decode(model, input_ids, config.max_new_tokens)
        else:
            result = sample_decode(model, input_ids, config)
        tokens = result.generated_tokens
        unique = len(set(tokens))
        repeat_rate = 1 - unique / len(tokens) if tokens else 0
        first5 = str(tokens[:5])
        print(f"  {name:<15} {unique:<12} {repeat_rate:<10.2%} {first5:<30}")

    print(f"\n  关键洞察: temperature 越低、top-k/top-p 越小，输出越确定但多样性越低")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  Task2 · 投机式生成与 Decode 加速")
    print("=" * 70)

    # 1. Decode 基本流程
    print(f"\n{'='*70}")
    print("  1. Decode 基本流程")
    print(f"{'='*70}")
    decode_steps = [
        ("读取状态", "本轮使用了哪些历史 KV?", "Cache 长度、TPOT、读写量"),
        ("计算 logits", "模型前向传播得到下一个 token 分布", "计算时间、KV 读取量"),
        ("选择 token", "贪心/采样/draft 提议后验证", "temperature、top-k/p、acceptance rate"),
        ("检查停止", "何时遇到 EOS，何时拒绝候选?", "generated tokens、停止原因、回退次数"),
        ("追加状态", "将新 token 的 K/V 写入 Cache", "Cache 增长、扩容次数"),
    ]
    for step, question, evidence in decode_steps:
        print(f"  {step}: {question}")
        print(f"    证据: {evidence}")

    # 2. 解码策略对比
    results = run_decode_comparison()

    # 3. 投机解码理论分析
    analyze_speculative_tradeoff()

    # 4. 质量检查
    run_quality_check()

    # 5. 瓶颈判断框架
    print(f"\n{'='*70}")
    print("  5. 瓶颈判断框架")
    print(f"{'='*70}")
    framework = [
        ("每轮生成耗时高", "Decode 计算、KV Cache 访问或 sampling", "检查基础 Decode 和 KV Cache (Task3)"),
        ("循环次数过多", "生成长度或单轮产出受限", "检查 speculative / multi-token 解码"),
        ("acceptance rate 低", "draft model 或 proposal 不匹配", "调整 draft、长度或放弃策略"),
        ("多请求吞吐低、P99 高", "请求组织或调度问题", "进入 Task4: Serving 调度"),
        ("输出质量下降", "采样参数或投机验证语义问题", "固定质量门槛后再比较速度"),
    ]
    for phenomenon, bottleneck, next_step in framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 瓶颈: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task2\\decode_comparison.json"
    output = []
    for r in results:
        d = asdict(r)
        d["generated_tokens"] = d["generated_tokens"][:10]  # 只保存前10个
        output.append(d)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")

    print(f"\n{'='*70}")
    print("  Task2 完成!")
    print("  关键概念: Decode流程 → 贪心/采样 → 投机解码 → 多Token推进 → 质量-速度权衡")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
