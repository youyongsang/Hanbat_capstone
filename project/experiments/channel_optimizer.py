# project/experiments/channel_optimizer.py

def optimize_channel(predicted_label, current_channel, available_channels=[1, 6, 11]):
    """
    Args:
        predicted_label (int): 모델이 예측한 혼잡도 단계 (0 ~ 3)
        current_channel (int): 현재 AP가 사용 중인 무선 채널 번호
        available_channels (list): 사용 가능한 전체 독립 채널 리스트
    Returns:
        next_channel (int): 전환 제어된 다음 채널 번호
        action (str): 수행된 제어 액션 ('keep' / 'monitor' / 'switch' / 'emergency')
    """
    # 0: 정상 -> 현재 채널 유지
    if predicted_label == 0:
        return current_channel, 'keep'
        
    # 1: 혼잡 경고 -> 인접 채널 스캔 및 모니터링 강화 (채널은 유지)
    elif predicted_label == 1:
        return current_channel, 'monitor'
        
    # 2: 혼잡 -> 회피 자율 스위칭 실행 (가장 여유 있는 다음 채널로 이동)
    elif predicted_label == 2:
        other_channels = [ch for ch in available_channels if ch != current_channel]
        # 실무적으로는 스캔 데이터 기반 최적 채널을 고르나, 프로토타입에서는 순차 전환 모사
        next_channel = other_channels[0] if other_channels else current_channel
        return next_channel, 'switch'
        
    # 3: 심각 혼잡 -> 즉각적인 강제 채널 회피 및 비상 대책 실행
    elif predicted_label == 3:
        other_channels = [ch for ch in available_channels if ch != current_channel]
        next_channel = other_channels[-1] if other_channels else current_channel
        return next_channel, 'emergency'
        
    return current_channel, 'keep'
