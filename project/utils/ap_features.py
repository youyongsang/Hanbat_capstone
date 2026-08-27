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
)

# 2026-08-28 진단: sta_tx_bitrate_min/_mean 를 feature 후보로 추가했다가 뺌.
#   `collect_metrics.py`는 CSV에 계속 기록(정보용)하지만 모델 입력엔 안 씀.
#   diag_25 런(146행)에서: min 은 트래픽 없는 유휴 station 하나가 MCS 0(6.5
#   Mbit/s)에 물려 모든 행을 지배 → 상수. mean 은 throughput을 그대로 따라가
#   (모델이 이미 가진 신호) occ 55~75% 어려운 구간에선 L3가 오히려 높음.
#   그 구간 L3는 latency 0 + loss 14% = failure=max — 채널 쪽엔 진짜 지문이
#   없음. 결론: 이 feature로는 6-feature 모델의 한계(occ 55~75% L2 vs L3)를
#   못 메움. 개선하려면 "이번 폴링에 패킷 보낸 station만" 같은 재정의 필요.
#   상세: project/results/yongsang/ap_v2_redesign_threshold_comparison.txt
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
