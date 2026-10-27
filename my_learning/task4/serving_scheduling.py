"""
Task4 · 多请求调度、异构 PD 与 Serving (Serving Scheduling and PD Disaggregation)
======================================================================================
本脚本实现推理优化专题 Task4 的核心内容：
1. Continuous Batching: 动态批次组织
2. Decode 调度: FCFS / 优先级 / 公平性
3. KV Cache 调度: 接纳控制与暂停/恢复
4. Prefill/Decode 分离 (PD Disaggregation)
5. Chunked Prefill 与调度集成
6. 异构 PD 路由: KV 传输 vs 重算 vs 同池
7. I/O 证据契约与执行关键路径分析

参考: topic_discussion/inference_optimization/07_serving_scheduling_and_pd.md
      02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb
      02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb
      02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb
      02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.ipynb
"""

import math
import json
import heapq
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque, OrderedDict
import enum


# ============================================================
# 1. 数据结构
# ============================================================

class RequestState(enum.Enum):
    WAITING = "waiting"           # 排队等待 Prefill
    PREFILLING = "prefilling"     # 正在 Prefill
    DECODING = "decoding"         # 正在 Decode
    PAUSED = "paused"             # 暂停 (Cache 资源不足)
    COMPLETED = "completed"       # 完成
    REJECTED = "rejected"         # 被拒绝


@dataclass
class Request:
    """Serving 请求"""
    req_id: int
    prompt_tokens: int
    max_new_tokens: int
    arrival_time: float = 0.0
    priority: int = 0  # 0=normal, 1=high
    # 状态
    state: RequestState = RequestState.WAITING
    generated_tokens: int = 0
    # 时间记录
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0
    first_token_time: float = 0.0
    completion_time: float = 0.0
    # 资源
    cache_blocks_used: int = 0
    # 调度统计
    decode_steps: int = 0
    total_decode_time_ms: float = 0.0
    queue_wait_ms: float = 0.0


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_batch_size: int = 16
    max_num_seqs: int = 32
    max_model_len: int = 4096
    # Cache
    num_gpu_blocks: int = 512
    block_size: int = 16
    # Prefill
    max_num_batched_tokens: int = 512  # Chunked Prefill 上限
    # PD
    pd_disaggregated: bool = False
    prefill_pool_size: int = 1
    decode_pool_size: int = 1
    # 调度策略
    scheduling_policy: str = "fcfs"  # fcfs / priority / lottery


@dataclass
class SchedulerStats:
    """调度器统计"""
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    # 延迟
    ttft_list: List[float] = field(default_factory=list)
    tpot_list: List[float] = field(default_factory=list)
    e2e_list: List[float] = field(default_factory=list)
    queue_wait_list: List[float] = field(default_factory=list)
    # 资源
    avg_cache_utilization: float = 0.0
    max_cache_utilization: float = 0.0
    # 调度
    prefill_batches: int = 0
    decode_batches: int = 0
    chunked_prefill_count: int = 0
    pauses: int = 0


# ============================================================
# 2. Continuous Batching 调度器
# ============================================================

class ContinuousBatchingScheduler:
    """
    Continuous Batching (连续批处理) 调度器 (vLLM 风格)
    - 每个 iteration 选择一批请求执行
    - Prefill 和 Decode 可以在同一 batch (或分离)
    - 请求完成后立即从 batch 移除，新请求立即加入
    - 支持 Chunked Prefill: 长 prompt 分块处理
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.completed: List[Request] = []
        self.rejected: List[Request] = []
        # Cache 管理
        self.free_blocks = config.num_gpu_blocks
        self.stats = SchedulerStats()
        self.current_time = 0.0
        self.req_counter = 0

    def _get_cache_blocks_needed(self, req: Request) -> int:
        """计算请求需要的 cache block 数"""
        total_tokens = req.prompt_tokens + req.generated_tokens + req.max_new_tokens
        return math.ceil(total_tokens / self.config.block_size)

    def _can_allocate(self, req: Request) -> bool:
        """检查是否有足够的 cache block"""
        needed = self._get_cache_blocks_needed(req)
        return needed <= self.free_blocks

    def _allocate(self, req: Request):
        """分配 cache blocks"""
        needed = self._get_cache_blocks_needed(req)
        self.free_blocks -= needed
        req.cache_blocks_used = needed

    def _free(self, req: Request):
        """释放 cache blocks"""
        self.free_blocks += req.cache_blocks_used
        req.cache_blocks_used = 0

    def submit(self, prompt_tokens: int, max_new_tokens: int,
               priority: int = 0) -> Request:
        """提交新请求"""
        self.req_counter += 1
        req = Request(
            req_id=self.req_counter,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            arrival_time=self.current_time,
            priority=priority,
        )

        # 接纳控制
        if len(self.running) + len(self.waiting) >= self.config.max_num_seqs:
            req.state = RequestState.REJECTED
            self.rejected.append(req)
            self.stats.rejected_requests += 1
            return req

        if not self._can_allocate(req):
            # 尝试暂停低优先级请求
            paused = self._try_pause_low_priority(req)
            if not paused:
                req.state = RequestState.REJECTED
                self.rejected.append(req)
                self.stats.rejected_requests += 1
                return req

        self.waiting.append(req)
        self.stats.total_requests += 1
        return req

    def _try_pause_low_priority(self, new_req: Request) -> bool:
        """尝试暂停低优先级请求以释放资源"""
        for req in self.running:
            if req.priority < new_req.priority and req.state == RequestState.DECODING:
                req.state = RequestState.PAUSED
                self._free(req)
                self.stats.pauses += 1
                return True
        return False

    def _select_prefill_batch(self) -> List[Request]:
        """选择下一批 Prefill 请求 (支持 Chunked Prefill)"""
        batch = []
        total_tokens = 0

        # 按策略排序 waiting 队列
        waiting_list = list(self.waiting)
        if self.config.scheduling_policy == "priority":
            waiting_list.sort(key=lambda r: (-r.priority, r.arrival_time))

        for req in waiting_list:
            if len(batch) >= self.config.max_batch_size:
                break
            # Chunked Prefill: 限制单批总 token 数
            prompt_tokens = min(req.prompt_tokens,
                                self.config.max_num_batched_tokens - total_tokens)
            if prompt_tokens <= 0:
                break
            total_tokens += prompt_tokens
            batch.append(req)

        return batch

    def _select_decode_batch(self) -> List[Request]:
        """选择下一批 Decode 请求"""
        decoding = [r for r in self.running if r.state == RequestState.DECODING]
        if self.config.scheduling_policy == "priority":
            decoding.sort(key=lambda r: (-r.priority, r.req_id))
        return decoding[:self.config.max_batch_size]

    def step(self) -> Dict:
        """
        执行一个调度迭代
        返回本迭代的执行信息
        """
        self.current_time += 1  # 模拟时间推进
        step_info = {"prefill_requests": [], "decode_requests": [], "completed": []}

        # 1. 处理 Prefill
        prefill_batch = self._select_prefill_batch()
        for req in prefill_batch:
            if req in self.waiting:
                self.waiting.remove(req)
            req.state = RequestState.PREFILLING
            req.prefill_start_time = self.current_time
            req.queue_wait_ms = (self.current_time - req.arrival_time) * 10  # 模拟

            # Chunked Prefill: 检查是否需要分块
            if req.prompt_tokens > self.config.max_num_batched_tokens:
                self.stats.chunked_prefill_count += 1

            # 模拟 Prefill 完成
            req.prefill_end_time = self.current_time
            req.first_token_time = self.current_time
            req.state = RequestState.DECODING
            self._allocate(req)
            self.running.append(req)
            step_info["prefill_requests"].append(req.req_id)

        self.stats.prefill_batches += 1 if prefill_batch else 0

        # 2. 处理 Decode
        decode_batch = self._select_decode_batch()
        for req in decode_batch:
            req.decode_steps += 1
            req.generated_tokens += 1
            req.total_decode_time_ms += 5  # 模拟每步 5ms

            if req.generated_tokens >= req.max_new_tokens:
                req.state = RequestState.COMPLETED
                req.completion_time = self.current_time
                self._free(req)
                self.running.remove(req)
                self.completed.append(req)
                self.stats.completed_requests += 1
                step_info["completed"].append(req.req_id)

                # 记录指标
                ttft = (req.first_token_time - req.arrival_time) * 10
                tpot = req.total_decode_time_ms / req.decode_steps if req.decode_steps > 0 else 0
                e2e = (req.completion_time - req.arrival_time) * 10
                self.stats.ttft_list.append(ttft)
                self.stats.tpot_list.append(tpot)
                self.stats.e2e_list.append(e2e)
                self.stats.queue_wait_list.append(req.queue_wait_ms)

        self.stats.decode_batches += 1 if decode_batch else 0

        # 3. 更新 cache 利用率
        used = self.config.num_gpu_blocks - self.free_blocks
        util = used / self.config.num_gpu_blocks
        self.stats.avg_cache_utilization += util
        self.stats.max_cache_utilization = max(self.stats.max_cache_utilization, util)

        return step_info

    def run_until_complete(self, max_steps: int = 1000) -> SchedulerStats:
        """运行直到所有请求完成或达到最大步数"""
        for _ in range(max_steps):
            if not self.waiting and not self.running:
                break
            self.step()

        # 计算平均 cache 利用率
        total_steps = self.stats.prefill_batches + self.stats.decode_batches
        if total_steps > 0:
            self.stats.avg_cache_utilization /= total_steps
        return self.stats


# ============================================================
# 3. PD 分离与异构路由模拟
# ============================================================

class PDDisaggregationSimulator:
    """
    Prefill/Decode 分离模拟器
    - Prefill 池和 Decode 池独立资源
    - KV Cache 交接: 传输 vs 重算 vs 同池
    - 异构路由决策
    """

    def __init__(self, prefill_gpus: int = 1, decode_gpus: int = 2,
                 network_bandwidth_gbps: float = 100.0):
        self.prefill_gpus = prefill_gpus
        self.decode_gpus = decode_gpus
        self.network_bandwidth_gbps = network_bandwidth_gbps
        # 统计
        self.handoff_count = 0
        self.recompute_count = 0
        self.total_handoff_bytes = 0
        self.total_handoff_ms = 0
        self.total_recompute_ms = 0

    def estimate_kv_transfer_time(self, seq_len: int, kv_heads: int = 8,
                                    head_dim: int = 128, num_layers: int = 32,
                                    dtype_bytes: int = 2) -> float:
        """估算 KV Cache 传输时间 (ms)"""
        kv_bytes = 2 * num_layers * kv_heads * head_dim * seq_len * dtype_bytes
        transfer_bits = kv_bytes * 8
        transfer_seconds = transfer_bits / (self.network_bandwidth_gbps * 1e9)
        return transfer_seconds * 1000

    def estimate_recompute_time(self, seq_len: int, model_flops_per_token: float = 1e9) -> float:
        """估算重算 Prefill 时间 (ms)"""
        total_flops = 2 * model_flops_per_token * seq_len
        # 假设 Prefill 池算力 10 TFLOPS
        compute_seconds = total_flops / (10e12)
        return compute_seconds * 1000

    def route_decision(self, seq_len: int, prefill_pool_busy: bool = False,
                       decode_pool_busy: bool = False) -> Dict:
        """
        异构 PD 路由决策
        比较 KV 传输 vs 重算 vs 保持同池
        """
        transfer_ms = self.estimate_kv_transfer_time(seq_len)
        recompute_ms = self.estimate_recompute_time(seq_len)

        decision = {
            "seq_len": seq_len,
            "kv_transfer_ms": round(transfer_ms, 2),
            "recompute_ms": round(recompute_ms, 2),
            "transfer_bytes": 2 * 32 * 8 * 128 * seq_len * 2,
        }

        # 决策逻辑
        if transfer_ms < recompute_ms and not decode_pool_busy:
            decision["action"] = "handoff (KV 传输)"
            decision["estimated_cost_ms"] = transfer_ms
            self.handoff_count += 1
            self.total_handoff_ms += transfer_ms
            self.total_handoff_bytes += decision["transfer_bytes"]
        elif prefill_pool_busy and not decode_pool_busy:
            decision["action"] = "handoff (强制，Prefill池繁忙)"
            decision["estimated_cost_ms"] = transfer_ms
            self.handoff_count += 1
            self.total_handoff_ms += transfer_ms
        else:
            decision["action"] = "recompute (重算 Prefill)"
            decision["estimated_cost_ms"] = recompute_ms
            self.recompute_count += 1
            self.total_recompute_ms += recompute_ms

        return decision

    def get_stats(self) -> Dict:
        return {
            "prefill_gpus": self.prefill_gpus,
            "decode_gpus": self.decode_gpus,
            "network_bandwidth_gbps": self.network_bandwidth_gbps,
            "handoff_count": self.handoff_count,
            "recompute_count": self.recompute_count,
            "total_handoff_ms": round(self.total_handoff_ms, 2),
            "total_recompute_ms": round(self.total_recompute_ms, 2),
            "total_handoff_gb": round(self.total_handoff_bytes / 1e9, 4),
        }


# ============================================================
# 4. 对比实验
# ============================================================

def run_scheduling_comparison():
    """运行不同调度策略的对比实验"""
    print(f"\n{'='*70}")
    print("  调度策略对比: 基础批处理 vs Continuous Batching vs PD分离")
    print(f"{'='*70}")

    # 模拟工作负载: 混合长短请求
    workload = [
        (64, 32), (128, 64), (256, 32), (512, 128),
        (32, 16), (1024, 64), (128, 32), (256, 128),
        (64, 64), (512, 32), (128, 128), (256, 64),
    ]

    configs = {
        "基础批处理 (max_batch=4)": SchedulerConfig(
            max_batch_size=4, max_num_batched_tokens=8192, num_gpu_blocks=256
        ),
        "Continuous Batching (max_batch=16)": SchedulerConfig(
            max_batch_size=16, max_num_batched_tokens=512, num_gpu_blocks=256
        ),
        "Chunked Prefill (max_tokens=256)": SchedulerConfig(
            max_batch_size=16, max_num_batched_tokens=256, num_gpu_blocks=256
        ),
    }

    print(f"\n  {'策略':<35} {'完成':<6} {'拒绝':<6} {'平均TTFT':<10} {'平均TPOT':<10} "
          f"{'平均E2E':<10} {'Cache利用':<10}")
    print(f"  {'-'*93}")

    for name, config in configs.items():
        scheduler = ContinuousBatchingScheduler(config)
        for prompt, gen in workload:
            scheduler.submit(prompt, gen)
        stats = scheduler.run_until_complete(max_steps=500)

        avg_ttft = sum(stats.ttft_list) / len(stats.ttft_list) if stats.ttft_list else 0
        avg_tpot = sum(stats.tpot_list) / len(stats.tpot_list) if stats.tpot_list else 0
        avg_e2e = sum(stats.e2e_list) / len(stats.e2e_list) if stats.e2e_list else 0

        print(f"  {name:<35} {stats.completed_requests:<6} {stats.rejected_requests:<6} "
              f"{avg_ttft:<10.1f} {avg_tpot:<10.1f} {avg_e2e:<10.1f} "
              f"{stats.avg_cache_utilization:<10.1%}")

    print(f"\n  关键洞察:")
    print(f"  1. Continuous Batching 通过动态组装批次提高吞吐和 GPU 利用率")
    print(f"  2. Chunked Prefill 限制单批 Prefill token 数，减少长 Prompt 阻塞 Decode")
    print(f"  3. 更大的 max_batch_size 通常提高吞吐，但可能增加 P99 延迟")


def run_pd_routing_demo():
    """运行 PD 分离与异构路由演示"""
    print(f"\n{'='*70}")
    print("  PD 分离与异构路由: KV 传输 vs 重算决策")
    print(f"{'='*70}")

    sim = PDDisaggregationSimulator(
        prefill_gpus=1, decode_gpus=2, network_bandwidth_gbps=100.0
    )

    print(f"\n  配置: Prefill池={sim.prefill_gpus}GPU, Decode池={sim.decode_gpus}GPU, "
          f"网络={sim.network_bandwidth_gbps}Gbps")
    print(f"\n  {'SeqLen':<8} {'KV传输(ms)':<12} {'重算(ms)':<12} {'传输数据(MB)':<14} {'决策':<25}")
    print(f"  {'-'*71}")

    for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
        decision = sim.route_decision(seq_len)
        print(f"  {seq_len:<8} {decision['kv_transfer_ms']:<12.2f} "
              f"{decision['recompute_ms']:<12.2f} {decision['transfer_bytes']/1024**2:<14.2f} "
              f"{decision['action']:<25}")

    print(f"\n  汇总: {sim.get_stats()}")
    print(f"\n  关键洞察:")
    print(f"  1. 短序列: KV 传输通常更快 (数据量小)")
    print(f"  2. 长序列: 重算可能更优 (KV 数据量大，传输耗时超过重算)")
    print(f"  3. 网络带宽是关键瓶颈: 更高带宽 (如 NVLink/InfiniBand) 使传输更优")
    print(f"  4. Prefill 池繁忙时，即使重算更慢也可能选择传输以平衡负载")


def run_io_evidence_demo():
    """运行 I/O 证据契约演示"""
    print(f"\n{'='*70}")
    print("  I/O 证据契约: 执行关键路径分析")
    print(f"{'='*70}")

    # 模拟一次请求的完整时间线
    timeline = [
        ("请求进入", "queue_wait_ms", 2.5, "反映接纳、排队与输入准备"),
        ("输入传输", "input_transfer_ms", 0.8, "CPU→GPU 数据搬运"),
        ("Prefill 执行", "prefill_compute_ms", 15.0, "Attention + MLP 计算"),
        ("KV 写入", "kv_write_ms", 2.0, "Prefill 结果写入 KV Cache"),
        ("首 token 返回", "first_token_ms", 0.5, "TTFT 终点"),
        ("Decode step 1", "decode_1_ms", 3.0, "读取 KV + 计算"),
        ("Decode step 2", "decode_2_ms", 3.2, "读取 KV + 计算"),
        ("...", "...", 0, "更多 Decode steps"),
        ("输出发送", "stream_output_ms", 1.0, "GPU→CPU 输出传输"),
    ]

    print(f"\n  {'阶段':<15} {'字段':<20} {'时间(ms)':<10} {'说明':<30}")
    print(f"  {'-'*75}")
    total = 0
    for stage, field_name, t, desc in timeline:
        print(f"  {stage:<15} {field_name:<20} {t:<10.1f} {desc:<30}")
        total += t if isinstance(t, (int, float)) else 0

    print(f"  {'-'*75}")
    print(f"  {'总计 (不含... )':<35} {total:<10.1f}")

    print(f"\n  关键原则:")
    print(f"  1. 排队时间上升 ≠ GPU 变慢 (可能是调度/接纳问题)")
    print(f"  2. 模型加载/冷启动不能和热请求 TTFT 混合平均")
    print(f"  3. KV handoff 与 recompute 不能重复计入")
    print(f"  4. 通信暴露时间应和总 elapsed 分开记录")
    print(f"  5. 只有通信/搬运从关键路径移开，吞吐或尾延迟才可能改善")


# ============================================================
# 5. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  Task4 · 多请求调度、异构 PD 与 Serving")
    print("=" * 70)

    # 1. 调度四层模型
    print(f"\n{'='*70}")
    print("  1. 调度问题的四个层次")
    print(f"{'='*70}")
    layers = [
        ("单步 Decode 调度", "当前 Decode step 给哪些活跃请求分配 token 计算", "36"),
        ("Cache 资源调度", "请求能否继续占用 Cache，何时暂停、驱逐或恢复", "37"),
        ("批次与队列调度", "哪些 Prefill/Decode 请求组成下一批", "38, 70"),
        ("服务池调度", "请求进入哪个实例、Prefill 池或 Decode 池", "38, 70"),
        ("异构 PD 路由", "交接 KV、重算 Prefill，还是保持同池", "39, Task6"),
    ]
    print(f"  {'层次':<20} {'调度者决定什么':<45} {'入口':<10}")
    print(f"  {'-'*75}")
    for layer, decision, entry in layers:
        print(f"  {layer:<20} {decision:<45} {entry:<10}")

    # 2. 调度策略对比
    run_scheduling_comparison()

    # 3. PD 分离与异构路由
    run_pd_routing_demo()

    # 4. I/O 证据契约
    run_io_evidence_demo()

    # 5. 接纳控制
    print(f"\n{'='*70}")
    print("  5. 接纳控制策略")
    print(f"{'='*70}")
    admission = [
        ("排队", "当前资源短时不足，但预计可以恢复", "queue time, P99, 队列长度"),
        ("限流/拒绝", "队列或 Cache 超过安全水位", "rejection rate, 超时率, P99"),
        ("降级", "质量目标允许降低资源开销", "输出长度, dtype, 质量指标"),
        ("转移/扩容", "单实例或单池持续饱和", "利用率, 迁移/通信时间, 吞吐"),
    ]
    print(f"  {'动作':<12} {'触发条件':<30} {'需要观察':<25}")
    print(f"  {'-'*67}")
    for action, condition, observe in admission:
        print(f"  {action:<12} {condition:<30} {observe:<25}")

    # 6. 瓶颈判断框架
    print(f"\n{'='*70}")
    print("  6. 瓶颈判断框架")
    print(f"{'='*70}")
    framework = [
        ("长 Prompt 到达后 Decode 请求明显抖动", "Prefill 抢占或批次组织不合理", "检查 Chunked Prefill 和 Continuous Batching"),
        ("TTFT 可接受但 TPOT/P99 变差", "Decode 资源被挤占或排队积累", "检查 Decode 调度和资源配额"),
        ("单实例无法同时满足两类请求", "Prefill 与 Decode 资源需求不同", "评估 Chunked Prefill 与 PD 分离"),
        ("PD 吞吐改善但 P99 变差", "交接或池间失衡暴露在关键路径", "比较 KV 传输、重算与同池回退"),
        ("GPU 利用率低但排队时间高", "调度粒度、批次或跨实例通信不合理", "进入 profiling 和 serving benchmark"),
    ]
    for phenomenon, bottleneck, next_step in framework:
        print(f"  现象: {phenomenon}")
        print(f"    → 判断: {bottleneck}")
        print(f"    → 下一步: {next_step}\n")

    # 保存结果
    output_path = "D:\\llm-algo-leetcode\\my_learning\\task4\\scheduling_analysis.json"
    output = {"note": "Scheduling simulation results", "pd_routing": "see console output"}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")

    print(f"\n{'='*70}")
    print("  Task4 完成!")
    print("  关键概念: 四层调度 → ContinuousBatching → PD分离 → 异构路由 → I/O证据契约")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
