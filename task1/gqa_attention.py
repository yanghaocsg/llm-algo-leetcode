"""
202610 Task1 · MHA / GQA 注意力实现
对应 Notebook: 02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb
完成 4 个 TODO：多头拆分、KV Cache 拼接、缩放点积注意力、输出 reshape
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将形状 [B, H_kv, S, D] 的 KV 头复制 n_rep 次，
    输出为 [B, H_kv * n_rep, S, D]，以匹配 Query 头的数量。
    当 n_rep == 1 时（即 MHA），直接返回原张量。
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim 和 num_heads 必须为正整数")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # GQA/MQA 要求 Query head 可以平均分配给 KV head。
        if self.num_kv_heads <= 0 or num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
            )

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads

        # Q 始终按 num_heads 投影；K/V 按 num_kv_heads 投影以支持 MHA、GQA 与 MQA。
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache: Tuple[torch.Tensor, torch.Tensor] = None
    ):
        """
        前向传播。
        Args:
            x: 输入张量，形状 [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码，形状应为 [batch, 1, 1, seq_len]（因果掩码）
                        或 [batch, 1, seq_len, seq_len]；可见位置通常为 0，
                        屏蔽位置为 -inf，并广播到 scores。
            kv_cache: 缓存的 (K, V) 张量，形状为
                      ([batch, num_kv_heads, cached_seq_len, head_dim], ...)。

        Returns:
            输出张量 [batch, seq_len, hidden_dim]，更新后的 KV Cache
        """
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # ==========================================
        # TODO 1：把投影结果拆成多头格式。
        # ==========================================
        # [B, S, H*D] → [B, H, S, D]，把 head 维提到 seq 前面便于批量矩阵乘。
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ==========================================
        # TODO 2：若给出历史 Cache，在序列维把它放在当前 K/V 前。
        # ==========================================
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # 只拼接未扩展的 KV，避免把 GQA 的缓存放大到 MHA 大小。
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)

        new_kv_cache = (xk, xv)

        # 仅为当前 Attention 计算临时扩展 KV，不把扩展结果写回 Cache。
        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)

        # ==========================================
        # TODO 3：计算缩放分数，并将 mask 后的 scores 归一化后聚合 V。
        # ==========================================
        # scores = [B, H, S_query, S_key]；probs 在最后一维归一化。
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        # ==========================================
        # TODO 4：将 [B, H, S, D] 转回 [B, S, H×D]，再交给输出投影。
        # ==========================================
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(output), new_kv_cache


# ============================================================
# 测试
# ============================================================

def test_mha_forward_shape():
    """MHA (num_kv_heads == num_heads) 前向输出形状正确。"""
    torch.manual_seed(42)
    B, S, H, D = 2, 16, 4, 8
    hidden_dim = H * D  # 32
    mha = GroupedQueryAttention(hidden_dim, num_heads=H)
    x = torch.randn(B, S, hidden_dim)
    out, cache = mha(x)
    assert out.shape == (B, S, hidden_dim), f"Expected {(B, S, hidden_dim)}, got {out.shape}"
    assert cache[0].shape == (B, H, S, D)
    assert cache[1].shape == (B, H, S, D)
    print("✅ test_mha_forward_shape passed")


def test_gqa_forward_shape():
    """GQA (num_kv_heads < num_heads) 前向输出形状正确，KV cache 头数较少。"""
    torch.manual_seed(42)
    B, S, H, H_kv, D = 2, 16, 8, 2, 8
    hidden_dim = H * D  # 64
    gqa = GroupedQueryAttention(hidden_dim, num_heads=H, num_kv_heads=H_kv)
    x = torch.randn(B, S, hidden_dim)
    out, cache = gqa(x)
    assert out.shape == (B, S, hidden_dim)
    # KV cache 只有 H_kv 个头，不是 H 个
    assert cache[0].shape == (B, H_kv, S, D), f"KV cache K shape: {cache[0].shape}"
    assert cache[1].shape == (B, H_kv, S, D)
    print("✅ test_gqa_forward_shape passed")


def test_gqa_kv_cache_reuse():
    """GQA 解码时复用 KV cache，第二次前向 KV 序列长度累加。"""
    torch.manual_seed(42)
    B, S1, S2, H, H_kv, D = 1, 8, 1, 8, 2, 8
    hidden_dim = H * D
    gqa = GroupedQueryAttention(hidden_dim, num_heads=H, num_kv_heads=H_kv)

    # Prefill: 处理 8 个 token
    x1 = torch.randn(B, S1, hidden_dim)
    _, cache1 = gqa(x1)
    assert cache1[0].shape[2] == S1

    # Decode: 处理 1 个新 token，带上历史 cache
    x2 = torch.randn(B, S2, hidden_dim)
    _, cache2 = gqa(x2, kv_cache=cache1)
    assert cache2[0].shape[2] == S1 + S2, f"Expected KV seq len {S1 + S2}, got {cache2[0].shape[2]}"
    print("✅ test_gqa_kv_cache_reuse passed")


def test_causal_mask():
    """因果掩码下，每个位置只能看到自己及之前的 token。"""
    torch.manual_seed(42)
    B, S, H, D = 1, 4, 2, 8
    hidden_dim = H * D
    mha = GroupedQueryAttention(hidden_dim, num_heads=H)

    x = torch.randn(B, S, hidden_dim)
    # 因果上三角掩码: [B, 1, S, S]，屏蔽位置为 -inf
    mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1).unsqueeze(0).unsqueeze(0)
    out_masked, _ = mha(x, attention_mask=mask)
    out_full, _ = mha(x)

    # 有 mask 和无 mask 结果应不同
    assert not torch.allclose(out_masked, out_full, atol=1e-5)
    print("✅ test_causal_mask passed")


def test_mha_equals_gqa_when_kv_heads_equal():
    """当 num_kv_heads == num_heads 时，GQA 退化为 MHA，结果一致。"""
    torch.manual_seed(42)
    B, S, H, D = 1, 8, 4, 8
    hidden_dim = H * D

    mha = GroupedQueryAttention(hidden_dim, num_heads=H, num_kv_heads=H)
    # 用相同权重初始化
    gqa = GroupedQueryAttention(hidden_dim, num_heads=H, num_kv_heads=H)
    gqa.load_state_dict(mha.state_dict())

    x = torch.randn(B, S, hidden_dim)
    out_mha, _ = mha(x)
    out_gqa, _ = gqa(x)
    assert torch.allclose(out_mha, out_gqa, atol=1e-6)
    print("✅ test_mha_equals_gqa_when_kv_heads_equal passed")


def test_kv_cache_memory_saving():
    """验证 GQA 的 KV cache 显存确实小于 MHA。"""
    B, S, H, H_kv, D = 1, 4096, 32, 8, 128
    # MHA K+V 元素数
    mha_kv_elements = 2 * B * S * H * D
    gqa_kv_elements = 2 * B * S * H_kv * D
    mha_mib = mha_kv_elements * 2 / 1024 / 1024
    gqa_mib = gqa_kv_elements * 2 / 1024 / 1024
    print(f"  MHA K+V (FP16): {mha_mib:.0f} MiB")
    print(f"  GQA K+V (FP16): {gqa_mib:.0f} MiB")
    print(f"  节省比例: {(1 - gqa_mib/mha_mib)*100:.0f}%")
    assert gqa_mib < mha_mib
    assert abs(gqa_mib - mha_mib * H_kv / H) < 1e-6
    print("✅ test_kv_cache_memory_saving passed")


if __name__ == '__main__':
    test_mha_forward_shape()
    test_gqa_forward_shape()
    test_gqa_kv_cache_reuse()
    test_causal_mask()
    test_mha_equals_gqa_when_kv_heads_equal()
    test_kv_cache_memory_saving()
    print("\n✅ 202610 Task1 MHA/GQA 注意力实现全部测试通过 (6/6)")
