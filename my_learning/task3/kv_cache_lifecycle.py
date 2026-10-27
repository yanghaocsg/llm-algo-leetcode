"""
Task3 · KV Cache 状态与生命周期 (KV Cache Lifecycle and Reuse)
====================================================================
本脚本实现推理优化专题 Task3 的核心内容：
1. KV Cache 增长模型与显存估算
2. PagedAttention: 分页 Block 管理与碎片分析
3. Prefix Cache: 前缀匹配与复用 (RadixAttention 树)
4. Cache 容量治理: 保留/驱逐/重算/压缩策略
5. MHA vs GQA vs MLA: 不同架构的 Cache 表示成本对比
6. Cache 正确性条件检查

参考: topic_discussion/inference_optimization/04_kv_cache_lifecycle_and_reuse.md
      02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb
      02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb
      02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.ipynb
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import OrderedDict
import torch


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class CacheConfig:
    """KV Cache 配置"""
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    kv_heads: int = 8  # GQA
    head_dim: int = 128
    dtype_bytes: int = 2  # fp16
    block_size: int = 16  # PagedAttention block 大小
    max_num_blocks: int = 1024  # 最大物理 block 数


@dataclass
class CacheMemoryReport:
    """Cache 显存报告"""
    config: CacheConfig
    seq_len: int
    batch_size: int
    # 不同架构
    mha_bytes: float = 0.0
    gqa_bytes: float = 0.0
    mla_bytes: float = 0.0  # MLA 压缩表示
    # PagedAttention
    num_logical_blocks: int = 0
    num_physical_blocks: int = 0
    internal_fragmentation_bytes: float = 0.0
    block_utilization: float = 0.0


# ============================================================
# 2. KV Cache 显存模型
# ============================================================

def calculate_kv_cache_memory(config: CacheConfig, seq_len: int,
                               batch_size: int = 1) -> CacheMemoryReport:
    """
    计算不同 Attention 架构下的 KV Cache 显存占用
    公式: KV_Cache = 2 (K和V) * num_layers * kv_heads * head_dim * seq_len * batch * dtype_bytes
    """
    report = CacheMemoryReport(config=config, seq_len=seq_len, batch_size=batch_size)

    # MHA: kv_heads = num_heads
    mha_kv_heads = config.num_heads
    report.mha_bytes = (2 * config.num_layers * mha_kv_heads * config.head_dim
                         * seq_len * batch_size * config.dtype_bytes)

    # GQA: kv_heads < num_heads
    report.gqa_bytes = (2 * config.num_layers * config.kv_heads * config.head_dim
                         * seq_len * batch_size * config.dtype_bytes)

    # MLA (Multi-head Latent Attention): 压缩 KV 表示
    # DeepSeek V2/V3 风格: 压缩到 latent_dim (通常 ~512), 解码时重构
    mla_latent_dim = 512  # 典型值
    report.mla_bytes = (2 * config.num_layers * mla_latent_dim
                         * seq_len * batch_size * config.dtype_bytes)

    # PagedAttention: 按 block 分配
    tokens_per_block = config.block_size
    report.num_logical_blocks = math.ceil(seq_len / tokens_per_block) * batch_size
    # 物理 block 数 = 逻辑 block 数 (无共享时)
    report.num_physical_blocks = report.num_logical_blocks
    # 内部碎片: 最后一个 block 可能不满
    tokens_in_last_block = seq_len % tokens_per_block
    if tokens_in_last_block > 0:
        wasted_tokens = tokens_per_block - tokens_in_last_block
        report.internal_fragmentation_bytes = (
            wasted_tokens * 2 * config.kv_heads * config.head_dim
            * config.num_layers * config.dtype_bytes * batch_size
        )
    total_capacity = report.num_physical_blocks * tokens_per_block
    report.block_utilization = seq_len / total_capacity if total_capacity > 0 else 1.0

    return report


def print_memory_comparison(config: CacheConfig, seq_lens: List[int]):
    """打印不同架构和序列长度的显存对比"""
    print(f"\n  {'SeqLen':<8} {'MHA(MB)':<12} {'GQA(MB)':<12} {'MLA(MB)':<12} "
          f"{'GQA/MHA':<10} {'MLA/MHA':<10}")
    print(f"  {'-'*64}")
    for seq_len in seq_lens:
        report = calculate_kv_cache_memory(config, seq_len)
        ratio_gqa = report.gqa_bytes / report.mha_bytes
        ratio_mla = report.mla_bytes / report.mha_bytes
        print(f"  {seq_len:<8} {report.mha_bytes/1024**2:<12.2f} "
              f"{report.gqa_bytes/1024**2:<12.2f} {report.mla_bytes/1024**2:<12.2f} "
              f"{ratio_gqa:<10.2%} {ratio_mla:<10.2%}")


# ============================================================
# 3. PagedAttention: Block 管理器
# ============================================================

class BlockManager:
    """
    PagedAttention Block 管理器 (vLLM 风格)
    - 物理内存被划分为固定大小的 block
    - 每个请求的逻辑 block 映射到物理 block
    - 支持 Copy-on-Write (前缀共享)
    - 跟踪引用计数和碎片
    """

    def __init__(self, num_physical_blocks: int = 256, block_size: int = 16):
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        # 物理 block 池: block_id -> ref_count
        self.free_blocks = list(range(num_physical_blocks))
        self.ref_counts = [0] * num_physical_blocks
        # 请求的逻辑->物理映射: req_id -> [block_id, ...]
        self.block_table: Dict[int, List[int]] = {}
        # 统计
        self.total_allocations = 0
        self.total_frees = 0
        self.copy_on_writes = 0

    def allocate(self, req_id: int, num_tokens: int) -> List[int]:
        """为请求分配 block，返回逻辑 block 列表"""
        num_blocks = math.ceil(num_tokens / self.block_size)
        if num_blocks > len(self.free_blocks):
            raise MemoryError(f"Not enough free blocks: need {num_blocks}, have {len(self.free_blocks)}")

        physical_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            self.ref_counts[block_id] = 1
            physical_blocks.append(block_id)
            self.total_allocations += 1

        self.block_table[req_id] = physical_blocks
        return physical_blocks

    def allocate_with_prefix_share(self, req_id: int, num_tokens: int,
                                     prefix_req_id: int,
                                     prefix_shared_tokens: int) -> List[int]:
        """
        带前缀共享的分配 (Copy-on-Write)
        前缀部分共享已有请求的物理 block (ref_count++)
        非前缀部分分配新 block
        """
        if prefix_req_id not in self.block_table:
            return self.allocate(req_id, num_tokens)

        prefix_blocks = self.block_table[prefix_req_id]
        shared_blocks_count = math.ceil(prefix_shared_tokens / self.block_size)
        shared_blocks_count = min(shared_blocks_count, len(prefix_blocks))

        # 共享前缀 block (增加引用计数)
        physical_blocks = []
        for i in range(shared_blocks_count):
            block_id = prefix_blocks[i]
            self.ref_counts[block_id] += 1
            physical_blocks.append(block_id)

        # 分配新 block 给非前缀部分
        remaining_tokens = num_tokens - prefix_shared_tokens
        if remaining_tokens > 0:
            remaining_blocks = math.ceil(remaining_tokens / self.block_size)
            for _ in range(remaining_blocks):
                if not self.free_blocks:
                    raise MemoryError("Out of memory during prefix share allocation")
                block_id = self.free_blocks.pop(0)
                self.ref_counts[block_id] = 1
                physical_blocks.append(block_id)
                self.total_allocations += 1

        self.block_table[req_id] = physical_blocks
        return physical_blocks

    def copy_on_write(self, req_id: int, block_idx: int) -> int:
        """
        Copy-on-Write: 当共享 block 需要修改时，复制到新 block
        """
        if req_id not in self.block_table:
            raise KeyError(f"Request {req_id} not found")

        old_block_id = self.block_table[req_id][block_idx]
        if self.ref_counts[old_block_id] > 1:
            # 需要复制
            if not self.free_blocks:
                raise MemoryError("Out of memory for CoW")
            new_block_id = self.free_blocks.pop(0)
            self.ref_counts[new_block_id] = 1
            self.ref_counts[old_block_id] -= 1
            self.block_table[req_id][block_idx] = new_block_id
            self.copy_on_writes += 1
            return new_block_id
        return old_block_id

    def free(self, req_id: int):
        """释放请求的所有 block (引用计数减一)"""
        if req_id not in self.block_table:
            return
        for block_id in self.block_table[req_id]:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)
                self.total_frees += 1
        del self.block_table[req_id]

    def get_stats(self) -> Dict:
        """获取管理器统计信息"""
        used_blocks = self.num_physical_blocks - len(self.free_blocks)
        shared_blocks = sum(1 for r in self.ref_counts if r > 1)
        return {
            "total_physical_blocks": self.num_physical_blocks,
            "free_blocks": len(self.free_blocks),
            "used_blocks": used_blocks,
            "utilization": used_blocks / self.num_physical_blocks,
            "shared_blocks (ref>1)": shared_blocks,
            "active_requests": len(self.block_table),
            "total_allocations": self.total_allocations,
            "total_frees": self.total_frees,
            "copy_on_writes": self.copy_on_writes,
        }


def run_paged_attention_demo():
    """运行 PagedAttention 演示"""
    print(f"\n{'='*60}")
    print("  PagedAttention: Block 管理与前缀共享演示")
    print(f"{'='*60}")

    manager = BlockManager(num_physical_blocks=64, block_size=16)

    # 请求1: 独立分配
    print("\n  [Step 1] 请求A 分配 64 tokens (4 blocks)")
    manager.allocate("A", 64)
    print(f"    {manager.get_stats()}")

    # 请求2: 共享请求1的前缀 (32 tokens = 2 blocks)
    print("\n  [Step 2] 请求B 分配 64 tokens, 共享请求A的前32 tokens")
    manager.allocate_with_prefix_share("B", 64, "A", prefix_shared_tokens=32)
    print(f"    {manager.get_stats()}")

    # 请求3: 共享请求1的更长前缀
    print("\n  [Step 3] 请求C 分配 48 tokens, 共享请求A的前48 tokens")
    manager.allocate_with_prefix_share("C", 48, "A", prefix_shared_tokens=48)
    print(f"    {manager.get_stats()}")

    # CoW 演示
    print("\n  [Step 4] 请求B 修改共享 block (触发 Copy-on-Write)")
    manager.copy_on_write("B", 0)
    print(f"    {manager.get_stats()}")

    # 释放请求A
    print("\n  [Step 5] 释放请求A (共享 block 不会立即释放)")
    manager.free("A")
    print(f"    {manager.get_stats()}")

    # 释放所有
    print("\n  [Step 6] 释放所有请求")
    manager.free("B")
    manager.free("C")
    print(f"    {manager.get_stats()}")

    print(f"\n  关键洞察: 前缀共享通过引用计数实现，CoW 保证修改隔离；")
    print(f"  物理 block 只有在 ref_count 归零时才真正释放。")


# ============================================================
# 4. Prefix Cache / RadixAttention: 前缀树
# ============================================================

class RadixCacheNode:
    """RadixAttention 前缀树节点"""
    def __init__(self, token_ids: List[int] = None):
        self.token_ids = token_ids or []  # 该节点代表的 token 序列
        self.children: Dict[int, 'RadixCacheNode'] = {}  # 第一个 token -> 子节点
        self.cache_ref = None  # 关联的 KV Cache 引用 (block 列表)
        self.last_access_time = 0
        self.ref_count = 0


class RadixAttentionCache:
    """
    SGLang RadixAttention: 基于前缀树的 KV Cache 复用
    - 所有请求的 token 序列组织成前缀树
    - 新请求匹配最长前缀，复用对应的 KV Cache
    - 只计算未命中的后缀部分
    """

    def __init__(self):
        self.root = RadixCacheNode()
        self.hit_count = 0
        self.miss_count = 0
        self.total_tokens = 0
        self.reused_tokens = 0
        self.access_counter = 0

    def _match_prefix(self, token_ids: List[int]) -> Tuple[RadixCacheNode, int, int]:
        """
        匹配最长前缀
        返回: (匹配到的节点, 匹配的 token 数, 在节点内的偏移)
        """
        node = self.root
        matched = 0
        i = 0

        while i < len(token_ids):
            first_token = token_ids[i]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            # 检查 child 的 token 序列是否匹配
            match_len = 0
            for j, tid in enumerate(child.token_ids):
                if i + j >= len(token_ids) or token_ids[i + j] != tid:
                    break
                match_len += 1

            if match_len == len(child.token_ids):
                # 完全匹配，继续向下
                node = child
                matched += match_len
                i += match_len
            elif match_len > 0:
                # 部分匹配，需要分裂节点
                return node, matched, match_len
            else:
                break

        return node, matched, 0

    def insert(self, token_ids: List[int], cache_ref=None) -> int:
        """
        插入 token 序列到前缀树，返回命中的 token 数
        """
        self.access_counter += 1
        self.total_tokens += len(token_ids)

        if not token_ids:
            return 0

        node, matched, partial_match = self._match_prefix(token_ids)

        if partial_match > 0 and partial_match < len(node.children[token_ids[matched]].token_ids):
            # 需要分裂节点
            child = node.children[token_ids[matched]]
            existing_tokens = child.token_ids
            shared = existing_tokens[:partial_match]
            remaining = existing_tokens[partial_match:]

            # 创建新的中间节点
            new_node = RadixCacheNode(shared)
            new_node.ref_count = child.ref_count
            new_node.cache_ref = child.cache_ref  # 共享 cache

            # 原节点变为子节点
            child.token_ids = remaining
            new_node.children[remaining[0]] = child
            del node.children[token_ids[matched]]
            node.children[shared[0]] = new_node
            node = new_node
            matched += partial_match

        # 插入剩余 token
        remaining_tokens = token_ids[matched:]
        if remaining_tokens:
            # 检查是否已有匹配的子节点
            if remaining_tokens[0] in node.children:
                # 不应该发生 (前面已经匹配过了)，但做防御
                pass
            else:
                new_node = RadixCacheNode(remaining_tokens)
                new_node.cache_ref = cache_ref
                new_node.last_access_time = self.access_counter
                node.children[remaining_tokens[0]] = new_node

        if matched > 0:
            self.hit_count += 1
            self.reused_tokens += matched
        else:
            self.miss_count += 1

        return matched

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total_requests = self.hit_count + self.miss_count
        return {
            "total_requests": total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total_requests if total_requests > 0 else 0,
            "total_tokens": self.total_tokens,
            "reused_tokens": self.reused_tokens,
            "token_reuse_rate": self.reused_tokens / self.total_tokens if self.total_tokens > 0 else 0,
        }


def run_prefix_cache_demo():
    """运行 Prefix Cache 演示"""
    print(f"\n{'='*60}")
    print("  Prefix Cache / RadixAttention: 前缀复用演示")
    print(f"{'='*60}")

    cache = RadixAttentionCache()

    # 模拟共享 system prompt 的请求
    system_prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # 共享的 system prompt

    requests = [
        ("请求1: system + question A", system_prompt + [10, 11, 12]),
        ("请求2: system + question B", system_prompt + [20, 21, 22, 23]),
        ("请求3: system + question A + followup", system_prompt + [10, 11, 12, 13, 14]),
        ("请求4: 完全不同的请求", [100, 101, 102]),
        ("请求5: system + question C", system_prompt + [30, 31]),
    ]

    print(f"\n  {'请求':<35} {'Token数':<8} {'命中数':<8} {'复用率':<10}")
    print(f"  {'-'*61}")
    for name, tokens in requests:
        hit = cache.insert(tokens)
        reuse_rate = hit / len(tokens)
        print(f"  {name:<35} {len(tokens):<8} {hit:<8} {reuse_rate:<10.1%}")

    stats = cache.get_stats()
    print(f"\n  汇总: {stats}")
    print(f"\n  关键洞察: 共享 system prompt 的请求可以复用前缀 KV Cache，")
    print(f"  显著减少重复 Prefill 计算；RadixAttention 用前缀树高效管理共享。")


# ============================================================
# 5. Cache 容量治理策略
# ============================================================

class CacheEvictionSimulator:
    """Cache 容量治理策略模拟器"""

    def __init__(self, capacity_blocks: int = 100, block_size: int = 16):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self.used_blocks = 0
        self.lru_cache = OrderedDict()  # req_id -> num_blocks
        self.evictions = 0
        self.recomputations = 0
        self.total_requests = 0

    def admit(self, req_id: str, num_tokens: int) -> str:
        """
        接纳请求，返回策略动作
        - 'admit': 直接接纳
        - 'evict_and_admit': 驱逐旧请求后接纳
        - 'reject': 拒绝 (容量不足)
        """
        self.total_requests += 1
        needed_blocks = math.ceil(num_tokens / self.block_size)

        if self.used_blocks + needed_blocks <= self.capacity:
            self._admit(req_id, needed_blocks)
            return "admit"

        # 尝试 LRU 驱逐
        freed = 0
        while self.lru_cache and self.used_blocks + needed_blocks > self.capacity:
            old_req, old_blocks = self.lru_cache.popitem(last=False)
            self.used_blocks -= old_blocks
            freed += old_blocks
            self.evictions += 1
            # 被驱逐的请求如果还需要，就要重算
            self.recomputations += 1

        if self.used_blocks + needed_blocks <= self.capacity:
            self._admit(req_id, needed_blocks)
            return f"evict_and_admit (freed {freed} blocks)"
        else:
            return "reject (insufficient capacity)"

    def _admit(self, req_id: str, num_blocks: int):
        self.lru_cache[req_id] = num_blocks
        self.lru_cache.move_to_end(req_id)
        self.used_blocks += num_blocks

    def access(self, req_id: str):
        """访问请求 (更新 LRU 顺序)"""
        if req_id in self.lru_cache:
            self.lru_cache.move_to_end(req_id)

    def get_stats(self) -> Dict:
        return {
            "capacity_blocks": self.capacity,
            "used_blocks": self.used_blocks,
            "utilization": self.used_blocks / self.capacity,
            "total_requests": self.total_requests,
            "evictions": self.evictions,
            "recomputations": self.recomputations,
            "active_requests": len(self.lru_cache),
        }


def run_eviction_demo():
    """运行 Cache 驱逐策略演示"""
    print(f"\n{'='*60}")
    print("  Cache 容量治理: LRU 驱逐策略演示")
    print(f"{'='*60}")

    sim = CacheEvictionSimulator(capacity_blocks=10, block_size=16)

    events = [
        ("admit", "A", 64),   # 4 blocks
        ("admit", "B", 32),   # 2 blocks
        ("admit", "C", 48),   # 3 blocks → total 9
        ("access", "A", 0),
        ("admit", "D", 32),   # 2 blocks → need 11 > 10, evict LRU (B)
        ("admit", "E", 64),   # 4 blocks → evict
    ]

    print(f"\n  {'事件':<20} {'请求':<6} {'Tokens':<8} {'动作':<30} {'已用Block':<10}")
    print(f"  {'-'*74}")
    for action, req_id, tokens in events:
        if action == "admit":
            result = sim.admit(req_id, tokens)
        else:
            sim.access(req_id)
            result = "access (LRU update)"
        print(f"  {action:<20} {req_id:<6} {tokens:<8} {result:<30} {sim.used_blocks:<10}")

    print(f"\n  最终状态: {sim.get_stats()}")
    print(f"\n  容量治理策略对比:")
    strategies = [
        ("保留 (Retain)", "命中后可直接继续 Decode", "并发容量下降、OOM 风险上升"),
        ("驱逐 (Evict)", "释放显存、提高可接纳并发", "再次请求需要重算 Prefill"),
        ("重算 (Recompute)", "降低驻留状态需求", "增加 Prefill 计算时间"),
        ("压缩/量化", "降低每 token Cache 字节数", "质量和 kernel 支持风险"),
    ]
    print(f"  {'策略':<15} {'直接收益':<25} {'代价/风险':<30}")
    print(f"  {'-'*70}")
    for name, benefit, cost in strategies:
        print(f"  {name:<15} {benefit:<25} {cost:<30}")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Task3 · KV Cache 状态与生命周期")
    print("=" * 60)

    config = CacheConfig()

    # 1. KV Cache 显存模型
    print(f"\n{'='*60}")
    print("  1. KV Cache 显存模型 (MHA vs GQA vs MLA)")
    print(f"{'='*60}")
    print(f"  配置: layers={config.num_layers}, hidden={config.hidden_dim}, "
          f"heads={config.num_heads}, kv_heads={config.kv_heads}, head_dim={config.head_dim}")
    print_memory_comparison(config, [128, 512, 1024, 4096, 8192])

    # 单请求详细报告
    print(f"\n  单请求详细报告 (seq_len=4096, batch=1):")
    report = calculate_kv_cache_memory(config, 4096)
    print(f"    MHA:  {report.mha_bytes/1024**3:.3f} GB")
    print(f"    GQA:  {report.gqa_bytes/1024**3:.3f} GB (节省 {(1-report.gqa_bytes/report.mha_bytes):.1%})")
    print(f"    MLA:  {report.mla_bytes/1024**3:.3f} GB (节省 {(1-report.mla_bytes/report.mha_bytes):.1%})")
    print(f"    PagedAttention: {report.num_logical_blocks} logical blocks, "
          f"利用率 {report.block_utilization:.1%}, "
          f"内部碎片 {report.internal_fragmentation_bytes/1024:.1f} KB")

    # 2. PagedAttention
    run_paged_attention_demo()

    # 3. Prefix Cache / RadixAttention
    run_prefix_cache_demo()

    # 4. Cache 容量治理
    run_eviction_demo()

    # 5. Cache 正确性条件
    print(f"\n{'='*60}")
    print("  5. Cache 复用正确性条件")
    print(f"{'='*60}")
    conditions = [
        ("模型与 tokenizer", "model revision、词表和特殊 token 必须一致", "不一致则重新建立 Cache"),
        ("位置与上下文", "position offset、RoPE 配置、上下文边界", "校验位置后再命中"),
        ("适配器与参数", "LoRA adapter、量化配置、采样相关状态", "按配置隔离 Cache"),
        ("租户与权限", "不同用户或权限域的前缀状态", "禁止跨域共享"),
        ("失效与回收", "版本变化、驱逐、异常中断", "清理引用并记录失效原因"),
    ]
    print(f"  {'检查项':<15} {'需要保持一致':<35} {'失败处理':<20}")
    print(f"  {'-'*70}")
    for item, requirement, failure in conditions:
        print(f"  {item:<15} {requirement:<35} {failure:<20}")

    # 6. 瓶颈判断框架
    print(f"\n{'='*60}")
    print("  6. 瓶颈判断框架")
    print(f"{'='*60}")
    framework = [
        ("cache 随上下文/batch 快速增长", "容量边界", "估算预算，看压缩或限制 (Task5)"),
        ("重复前缀明显但 TTFT 没下降", "复用没生效", "检查 Prefix/Radix Cache 配置"),
        ("并发上升后显存碎片和 P99 变差", "分配或分页问题", "检查 PagedAttention block 大小和调度"),
        ("Cache 仍可用但吞吐低", "请求组织问题", "进入 Task4: Serving 调度"),
        ("驱逐后 TTFT 明显升高", "重算成本过高", "调整驱逐策略或增加 Cache 预算"),
    ]
    for phenomenon, bottleneck, next_step in framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 判断: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task3\\kv_cache_analysis.json"
    output = {
        "memory_comparison": {
            str(n): {
                "mha_mb": calculate_kv_cache_memory(config, n).mha_bytes / 1024**2,
                "gqa_mb": calculate_kv_cache_memory(config, n).gqa_bytes / 1024**2,
                "mla_mb": calculate_kv_cache_memory(config, n).mla_bytes / 1024**2,
            } for n in [128, 512, 1024, 4096, 8192]
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")

    print(f"\n{'='*60}")
    print("  Task3 完成!")
    print("  关键概念: Cache增长模型 → PagedAttention → PrefixCache → 容量治理 → 正确性条件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
