"""Shared AP measurement feature contract."""

from __future__ import annotations

from typing import Tuple


AP_FEATURE_COLUMNS: Tuple[str, ...] = (
    "throughput_mbps",
    "channel_occupancy_percent",
    "tx_retry_ratio",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
    "sta_tx_bitrate_min",
    "sta_tx_bitrate_mean",
)

# 2026-08-27 심야: sta_tx_bitrate_min / _mean 추가 (6 -> 8 feature).
#   iw station dump의 station별 "tx bitrate"(PHY rate). rate control이
#   간섭·경합에 물러나면 여기부터 떨어지는데 채널 전체 occupancy엔 안 보임.
#   min = 가장 굶는 station(capture effect / victim 프록시). 재학습 결과
#   occ 60~72%에서 label 2 vs 3이 나머지 6 feature로는 완전히 구별 불가였음
#   (ap_v2_redesign_threshold_comparison.txt). 이 feature 있는 데이터는
#   metrics_v2_pi_redesign.csv엔 없음 — 새로 수집해야 함.
#
# 2026-08-27 혼잡 라벨 재설계 (docs/yongsang/congestion_label_redesign.md):
#   - latency_ms / jitter_ms 를 모델 입력에서 제거. jitter/loss는 victim
#     프로브 실측, latency는 ping RTT — 라벨 축이자 배포 시점엔 없는 측정.
#     모델은 "채널 상태만 보고 victim QoS를 예측"해야 하므로 정답(프로브·
#     ping)을 입력으로 주지 않는다.
#   - tx_retries_per_s / tx_failed_per_s -> tx_retry_ratio 하나로.
#     retry_ratio = (retries + failed) / (retries + failed + tx_packets).
#     WLAN 헬스 표준 문턱(10/15/25/40%)에 매핑. 비율이라 폴링 주기 무관.
#   - occupancy / retry는 배포 시 AP 텔레메트리로 available하므로 라벨 축이면서
#     모델 입력으로도 유지.
#
# 이전(2026-08-27 오전): tx_retries_delta/tx_failed_delta -> _per_s.
# 그 이전 metrics_v2.csv 는 이 재설계에 완전 relabel 불가(프로브·tx_packets
# 없음) — 레거시/사전학습용으로만.
