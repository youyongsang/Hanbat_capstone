# project/experiments/channel_optimizer.py

def optimize_channel(predicted_label: int, current_channel: int, available_channels: list, metrics: dict = None, current_time: float = None):
    """
    김호중 담당: Stage 2 가이드라인 규칙 기반 채널 전환 로직 구현
    Args:
        predicted_label: 분류 결과 (0~3)
        current_channel: 현재 채널 번호
        available_channels: 사용 가능한 채널 목록
    Returns:
        next_channel: 전환할 채널 번호 (변경 없으면 current_channel 반환)
        action: 'keep' / 'monitor' / 'switch' / 'emergency'
    """
    # [라벨 0: 정상] -> 채널 유지
    if predicted_label == 0:
        return current_channel, 'keep'
        
    # [라벨 1: 혼잡 경고] -> 인접 채널 모니터링, 전환 준비
    elif predicted_label == 1:
        return current_channel, 'monitor'
        
    # [라벨 2: 혼잡] -> 덜 혼잡한 다른 채널로 전환
    elif predicted_label == 2:
        other_channels = [ch for ch in available_channels if ch != current_channel]
        next_channel = other_channels[0] if other_channels else current_channel
        return next_channel, 'switch'
        
    # [라벨 3: 심각 혼잡] -> 즉시 5GHz 대역(예: 36번) 또는 다른 채널로 비상 전환
    elif predicted_label == 3:
        if 36 in available_channels and current_channel != 36:
            return 36, 'emergency'
        else:
            other_channels = [ch for ch in available_channels if ch != current_channel]
            next_channel = other_channels[0] if other_channels else current_channel
            return next_channel, 'emergency'
            
    return current_channel, 'keep'
