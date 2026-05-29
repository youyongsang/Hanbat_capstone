import time

# 채널 전환 핑퐁(Ping-pong) 방지를 위한 전역 상태
_rrm_state = {
    'last_switch_time': 0,
    'current_channel': None
}

def optimize_channel(predicted_label, current_channel, available_channels, metrics=None, current_time=None):
    """
    RRM (Radio Resource Management) 기반 채널 최적화 알고리즘.
    
    Args:
        predicted_label (int): 분류 모델 결과 (0: 정상, 1: 경고, 2: 혼잡, 3: 심각)
        current_channel (int): 현재 운용 중인 채널
        available_channels (list): 가용 채널 목록
        metrics (dict, optional): 실측 채널 상태 (RSSI, Noise, Utilization 등)
        current_time (float, optional): Hysteresis 계산용 현재 시간
        
    Returns:
        tuple: (next_channel, action)
    """
    global _rrm_state
    
    if not available_channels:
        return current_channel, 'keep'
        
    if current_time is None:
        current_time = time.time()
        
    # [Policy Parameters]
    HYSTERESIS_SEC = 30      # 비교 실험용 최소 유지 시간 (잦은 전환 방지)
    SWITCH_COST_DB = 10      # 채널 전환에 따른 트래픽 단절 패널티 (10dB 수준)
    LABEL_PENALTY = {
        0: 0,    # 정상: 현재 채널 유지
        1: 0,    # 경고: 모니터링만 수행
        2: 15,   # 혼잡: 현재 채널 품질 점수에 패널티 부여
        3: 30,   # 심각 혼잡: 강한 패널티로 긴급 전환 유도
    }
    
    # 1. Hysteresis Check (핑퐁 현상 방지)
    # 첫 평가 시점에는 아직 실제 전환이 발생한 적이 없으므로,
    # 초기 last_switch_time=0 때문에 전환 검토가 막히지 않도록 한다.
    if _rrm_state.get('current_channel') is None:
        _rrm_state['current_channel'] = current_channel
        _rrm_state['last_switch_time'] = current_time - HYSTERESIS_SEC
        time_since_switch = HYSTERESIS_SEC
    else:
        time_since_switch = current_time - _rrm_state['last_switch_time']
    if predicted_label < 3 and time_since_switch < HYSTERESIS_SEC:
        return current_channel, 'keep'

    # 2. Channel Quality Scoring (채널 품질 평가)
    def get_channel_score(ch):
        if not metrics or ch not in metrics:
            # 실측 데이터 부재 시 기본 거리 기반 가중치 부여
            is_5g = 100 if ch >= 36 else 0
            return is_5g + abs(ch - current_channel)
            
        m = metrics[ch]
        # SNR(Signal-to-Noise Ratio) 및 Utilization(점유율) 기반 스코어링
        snr = m.get('rssi', -70) - m.get('noise', -95)
        utilization_penalty = m.get('utilization', 0.5) * 50
        
        score = snr - utilization_penalty
        if ch == current_channel:
            score -= LABEL_PENALTY.get(predicted_label, 0)
        
        # Switch Cost 반영: 현재 채널이 아니면 전환 패널티 부여
        if ch != current_channel:
            score -= SWITCH_COST_DB
            
        return score

    next_channel = current_channel
    action = 'keep'

    # 3. Decision Logic
    if predicted_label == 0:
        pass
        
    elif predicted_label == 1:
        action = 'monitor'
        
    elif predicted_label == 2:
        candidates = [ch for ch in available_channels if ch != current_channel]
        if candidates:
            best_candidate = max(candidates, key=get_channel_score)
            # 전환 패널티를 극복할 만큼 품질이 좋을 때만 전환 (Switch Cost Validation)
            if get_channel_score(best_candidate) > get_channel_score(current_channel):
                next_channel = best_candidate
                action = 'switch'
                
    elif predicted_label == 3:
        candidates = [ch for ch in available_channels if ch != current_channel]
        if candidates:
            # 5GHz(36번 이상) 대역 우선 탐색
            candidates_5g = [ch for ch in candidates if ch >= 36]
            target_list = candidates_5g if candidates_5g else candidates
            next_channel = max(target_list, key=get_channel_score)
            action = 'emergency'

    # 4. Update State
    if next_channel != current_channel:
        _rrm_state['last_switch_time'] = current_time
        _rrm_state['current_channel'] = next_channel

    return next_channel, action


if __name__ == "__main__":
    # [테스트] 실측 기반(RRM) 채널 전환 시뮬레이션
    current_ch = 1
    avail_chs = [1, 6, 11, 36]
    
    # 가상의 실측 메트릭 (RSSI, Noise Floor, Utilization)
    simulated_metrics = {
        1:  {'rssi': -65, 'noise': -90, 'utilization': 0.8}, # 현재 채널 (혼잡)
        6:  {'rssi': -70, 'noise': -92, 'utilization': 0.6}, # 약간 혼잡
        11: {'rssi': -55, 'noise': -95, 'utilization': 0.2}, # 매우 깨끗함
        36: {'rssi': -60, 'noise': -98, 'utilization': 0.1}  # 5GHz 대역
    }
    
    print("--- RRM 엔진 시뮬레이션 ---")
    # 혼잡 발생 시 (Label 2)
    next_ch, act = optimize_channel(2, current_ch, avail_chs, metrics=simulated_metrics)
    print(f"Label 2 (혼잡): Ch.{current_ch} -> Ch.{next_ch} (Action: {act})")
    
    # Hysteresis(핑퐁 방지) 테스트: 방금 바꿨는데 또 혼잡(Label 2)이 뜬 경우
    next_ch_hys, act_hys = optimize_channel(2, next_ch, avail_chs, metrics=simulated_metrics)
    print(f"Label 2 연속 발생 (Hysteresis 동작): Ch.{next_ch} -> Ch.{next_ch_hys} (Action: {act_hys})")
    
    # 심각(Label 3) 발생 시 (Hysteresis 무시)
    next_ch_em, act_em = optimize_channel(3, next_ch, avail_chs, metrics=simulated_metrics)
    print(f"Label 3 (심각): Ch.{next_ch} -> Ch.{next_ch_em} (Action: {act_em})")
