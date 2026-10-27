"""
202610 Task0 · 请求、工作负载与指标 — 推理请求生命周期与核心指标
======================================================================
对应 Notebook: 02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb

本文件完成 notebook 中的 4 个 TODO：
  TODO 1: 推进合法请求状态
  TODO 2: 计算相邻词元间隔 (TPOT)
  TODO 3: 按阶段占比归因请求
  TODO 4: 固定工作负载统计分母

运行方式:
  python task0_request_lifecycle.py
"""

from statistics import mean

# ============================================================
# 状态机定义
# ============================================================

TERMINAL_EVENTS = {
    'complete': 'completed',
    'cancel': 'cancelled',
    'timeout': 'timed_out',
    'fail': 'failed',
}

TRANSITIONS = {
    'not_arrived': {'arrive': 'queued'},
    'queued': {'admit': 'preprocessing'},
    'preprocessing': {'input_ready': 'prefill'},
    'prefill': {'first_token': 'decode'},
    'decode': {'token': 'decode', 'complete': 'completed'},
}

TERMINAL_STATES = {'completed', 'cancelled', 'timed_out', 'failed'}


# ============================================================
# TODO 1: 回放事件并返回请求状态
# ============================================================

def replay_request_events(events):
    """回放事件并返回请求状态、关键时间戳与输出词元时间。

    状态迁移规则:
      not_arrived --arrive--> queued --admit--> preprocessing
      --input_ready--> prefill --first_token--> decode
      --token/complete--> decode/completed

    不变量:
      1. 事件时间必须单调非递减
      2. 终止状态后不能继续追加事件
      3. 请求到达前不能终止
      4. 轨迹必须以终止事件结束
    """
    if not events:
        raise ValueError('events 不能为空')

    state = 'not_arrived'
    previous_time = None
    timestamps = {}
    token_times = []
    failure_stage = None

    for item in events:
        event = item['event']
        timestamp = float(item['timestamp'])

        # 不变量 1: 事件时间必须单调
        if previous_time is not None and timestamp < previous_time:
            raise ValueError('事件时间必须单调')

        # 不变量 2: 终止状态后不能继续追加事件
        if state in TERMINAL_STATES:
            raise ValueError('终止状态后不能继续追加事件')

        if event in {'cancel', 'timeout', 'fail'}:
            # 不变量 3: 请求到达前不能终止
            if state == 'not_arrived':
                raise ValueError('请求到达前不能终止')
            failure_stage = state
            state = TERMINAL_EVENTS[event]
            timestamps['terminal'] = timestamp
        else:
            # ==========================================
            # TODO 1：推进合法请求状态
            # ==========================================
            # 当前状态接收当前事件后的唯一下一状态；非法组合保持为 None。
            next_state = TRANSITIONS.get(state, {}).get(event)

            if next_state is None:
                raise ValueError(f'状态 {state} 不接受事件 {event}')
            state = next_state

            timestamp_key = {
                'arrive': 'arrival',
                'admit': 'admitted',
                'input_ready': 'input_ready',
                'first_token': 'first_token',
                'complete': 'terminal',
            }.get(event)
            if timestamp_key is not None:
                timestamps[timestamp_key] = timestamp
            if event in {'first_token', 'token'}:
                token_times.append(timestamp)

        previous_time = timestamp

    # 不变量 4: 轨迹必须以终止事件结束
    if state not in TERMINAL_STATES:
        raise ValueError('轨迹缺少终止事件')

    return {
        'status': state,
        'failure_stage': failure_stage,
        'timestamps': timestamps,
        'token_times': token_times,
    }


# ============================================================
# TODO 2: 从状态报告计算延迟指标
# ============================================================

def compute_request_metrics(report):
    """从状态报告计算当前轨迹已经具备的延迟指标。

    指标说明:
      queue_wait:     排队等待时间 (arrival -> admitted)
      preprocessing:  输入准备时间 (admitted -> input_ready)
      prefill:        预填充时间 (input_ready -> first_token)
      ttft:           首词元延迟 (arrival -> first_token) = queue + preproc + prefill
      tpot:           每输出词元时间 (相邻 token 间隔均值)
      decode_time:    解码阶段总时长 (first_token -> terminal)
      token_span:     首个到末个输出 token 的时间跨度
      e2e:            端到端延迟 (arrival -> terminal)
      streamed_tokens: 已输出 token 数量
    """
    timestamps = report['timestamps']
    token_times = report['token_times']

    def duration(start, end):
        return timestamps[end] - timestamps[start] if start in timestamps and end in timestamps else None

    # ==========================================
    # TODO 2：计算相邻词元间隔
    # ==========================================
    # 相邻输出词元的时间差列表；单位与输入时间戳一致。
    token_gaps = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]
    # 间隔均值；没有有效间隔时保持 None。
    tpot = mean(token_gaps) if token_gaps else None

    return {
        'queue_wait': duration('arrival', 'admitted'),
        'preprocessing': duration('admitted', 'input_ready'),
        'prefill': duration('input_ready', 'first_token'),
        'ttft': duration('arrival', 'first_token'),
        'tpot': tpot,
        'decode_time': duration('first_token', 'terminal'),
        'token_span': token_times[-1] - token_times[0] if token_times else None,
        'e2e': duration('arrival', 'terminal'),
        'streamed_tokens': len(token_times),
    }


# ============================================================
# TODO 3: 按阶段占比归因请求
# ============================================================

def classify_request_outcome(report, metrics, dominance_threshold=0.45):
    """返回异常终止阶段，或完成请求的阶段占比与瓶颈标签。

    归因规则:
      - 异常终止 (cancel/timeout/fail): 返回终止时所处阶段
      - 正常完成: 计算各阶段占可解释总时长的比例
        - 最大阶段占比 >= dominance_threshold: 标记该阶段为主导瓶颈
        - 最大阶段占比 < dominance_threshold: 标记为 'mixed' (混合瓶颈)
    """
    # 异常终止: 直接返回失败阶段
    if report['status'] != 'completed':
        return {
            'outcome': report['status'],
            'stage': report['failure_stage'],
            'dominant_share': None,
        }

    if not (0.0 < dominance_threshold <= 1.0):
        raise ValueError('dominance_threshold 必须位于 (0, 1]')

    phase_durations = {
        'queue': metrics['queue_wait'],
        'preprocessing': metrics['preprocessing'],
        'prefill': metrics['prefill'],
        'decode': metrics['decode_time'],
    }
    # 只统计有数据的阶段 (部分轨迹可能缺少某些阶段)
    available = {name: value for name, value in phase_durations.items() if value is not None}

    # ==========================================
    # TODO 3：按阶段占比归因请求
    # ==========================================
    # 可解释阶段的总时长；必须为正。
    explained_ms = sum(available.values())
    if explained_ms <= 0:
        raise ValueError('可解释阶段总时长必须为正')
    # 阶段名到占比的映射；占比总和应为 1。
    phase_shares = {name: value / explained_ms for name, value in available.items()}
    # 占比最大的阶段名。
    dominant_stage = max(phase_shares, key=phase_shares.get)
    # 主导阶段对应的占比，用于和阈值比较。
    dominant_share = phase_shares[dominant_stage]

    return {
        'outcome': 'completed',
        'stage': dominant_stage if dominant_share >= dominance_threshold else 'mixed',
        'dominant_share': dominant_share,
    }


# ============================================================
# TODO 4: 汇总工作负载并显式返回统计分母
# ============================================================

def summarize_workload(request_results):
    """汇总 (report, metrics) 列表并显式返回统计分母。

    统计分母规则:
      - total_requests: 所有请求数 (包括失败/取消)
      - completed_requests: 正常完成的请求数
      - success_rate: completed / total
      - ttft_sample_count: 有有效 TTFT 的请求数 (非 None)
      - mean_ttft: 有效 TTFT 的均值 (不用 0 代替缺失值)
    """
    total_requests = len(request_results)

    # ==========================================
    # TODO 4：固定工作负载统计分母
    # ==========================================
    # 正常完成的请求数。
    completed_count = sum(1 for report, _ in request_results if report['status'] == 'completed')
    # 所有非空 TTFT；不得用 0 代替缺失值。
    ttft_values = [metrics['ttft'] for _, metrics in request_results if metrics['ttft'] is not None]

    return {
        'total_requests': total_requests,
        'completed_requests': completed_count,
        'success_rate': completed_count / total_requests if total_requests else None,
        'ttft_sample_count': len(ttft_values),
        'mean_ttft': mean(ttft_values) if ttft_values else None,
    }


# ============================================================
# 测试用例 (与 notebook 中完全一致)
# ============================================================

def _completed_events():
    """构造阶段耗时接近、没有明显主导瓶颈的完整请求轨迹。"""
    return [
        {'event': 'arrive', 'timestamp': 0.0},
        {'event': 'admit', 'timestamp': 2.0},
        {'event': 'input_ready', 'timestamp': 3.0},
        {'event': 'first_token', 'timestamp': 7.0},
        {'event': 'token', 'timestamp': 9.0},
        {'event': 'token', 'timestamp': 11.0},
        {'event': 'complete', 'timestamp': 12.0},
    ]


def test_completed_trace_and_metrics():
    """完整轨迹应恢复状态、指标和 TTFT 阶段可加关系。"""
    report = replay_request_events(_completed_events())
    metrics = compute_request_metrics(report)
    assert report['status'] == 'completed'
    assert report['token_times'] == [7.0, 9.0, 11.0]
    assert metrics == {
        'queue_wait': 2.0, 'preprocessing': 1.0, 'prefill': 4.0,
        'ttft': 7.0, 'tpot': 2.0, 'decode_time': 5.0,
        'token_span': 4.0, 'e2e': 12.0, 'streamed_tokens': 3,
    }
    assert metrics['ttft'] == metrics['queue_wait'] + metrics['preprocessing'] + metrics['prefill']
    assert metrics['e2e'] == metrics['ttft'] + metrics['decode_time']


def test_stage_share_attribution():
    """最大阶段占比不足阈值时应标记混合瓶颈。"""
    report = replay_request_events(_completed_events())
    outcome = classify_request_outcome(report, compute_request_metrics(report))
    assert outcome['outcome'] == 'completed'
    assert outcome['stage'] == 'mixed'
    assert abs(outcome['dominant_share'] - 5.0 / 12.0) < 1e-9

    prefill_heavy = _completed_events()
    prefill_heavy[3] = {'event': 'first_token', 'timestamp': 9.0}
    prefill_heavy[4] = {'event': 'token', 'timestamp': 10.0}
    prefill_heavy[5] = {'event': 'token', 'timestamp': 11.0}
    heavy_report = replay_request_events(prefill_heavy)
    heavy_outcome = classify_request_outcome(heavy_report, compute_request_metrics(heavy_report))
    assert heavy_outcome['stage'] == 'prefill'


def test_partial_trace_preserves_missing_metrics():
    """首词元前取消时，TTFT 保持为空但 E2E 仍可计算。"""
    events = _completed_events()[:2] + [{'event': 'cancel', 'timestamp': 5.0}]
    report = replay_request_events(events)
    metrics = compute_request_metrics(report)
    outcome = classify_request_outcome(report, metrics)
    assert metrics['queue_wait'] == 2.0
    assert metrics['ttft'] is None and metrics['e2e'] == 5.0
    assert outcome == {'outcome': 'cancelled', 'stage': 'preprocessing', 'dominant_share': None}


def test_failed_decode_keeps_streamed_tokens():
    """解码中超时仍应保留已经输出的词元与失败阶段。"""
    events = _completed_events()[:5] + [{'event': 'timeout', 'timestamp': 10.0}]
    report = replay_request_events(events)
    metrics = compute_request_metrics(report)
    outcome = classify_request_outcome(report, metrics)
    assert metrics['streamed_tokens'] == 2
    assert outcome['outcome'] == 'timed_out' and outcome['stage'] == 'decode'


def test_workload_denominators():
    """成功率与 TTFT 均值必须报告各自的统计分母。"""
    completed_report = replay_request_events(_completed_events())
    cancelled_report = replay_request_events(
        _completed_events()[:2] + [{'event': 'cancel', 'timestamp': 5.0}]
    )
    results = [
        (completed_report, compute_request_metrics(completed_report)),
        (cancelled_report, compute_request_metrics(cancelled_report)),
    ]
    summary = summarize_workload(results)
    assert summary['total_requests'] == 2
    assert summary['completed_requests'] == 1
    assert summary['success_rate'] == 0.5
    assert summary['ttft_sample_count'] == 1 and summary['mean_ttft'] == 7.0


def test_invalid_transitions():
    """非法起点、逆序时间和终止后事件都应被拒绝。"""
    cases = [
        [{'event': 'admit', 'timestamp': 0.0}],
        [{'event': 'cancel', 'timestamp': 0.0}],
        [{'event': 'arrive', 'timestamp': 1.0}, {'event': 'admit', 'timestamp': 0.5}],
        _completed_events() + [{'event': 'token', 'timestamp': 13.0}],
    ]
    for events in cases:
        try:
            replay_request_events(events)
            raise AssertionError('非法事件序列应被拒绝')
        except ValueError:
            pass


# ============================================================
# 主入口: 运行所有测试
# ============================================================

if __name__ == '__main__':
    test_completed_trace_and_metrics()
    test_stage_share_attribution()
    test_partial_trace_preserves_missing_metrics()
    test_failed_decode_keeps_streamed_tokens()
    test_workload_denominators()
    test_invalid_transitions()
    print('✅ 202610 Task0 请求生命周期机制测试全部通过 (6/6)')

    # 演示: 用真实事件序列计算指标
    print('\n--- 演示: 完整请求轨迹指标 ---')
    events = _completed_events()
    report = replay_request_events(events)
    metrics = compute_request_metrics(report)
    outcome = classify_request_outcome(report, metrics)
    print(f'  状态: {report["status"]}')
    print(f'  TTFT = queue({metrics["queue_wait"]}) + preproc({metrics["preprocessing"]}) + prefill({metrics["prefill"]}) = {metrics["ttft"]}')
    print(f'  TPOT = {metrics["tpot"]} (相邻 token 间隔均值)')
    print(f'  E2E = {metrics["e2e"]}, 输出 token 数 = {metrics["streamed_tokens"]}')
    print(f'  瓶颈归因: {outcome["stage"]} (主导占比 {outcome["dominant_share"]:.1%})')
