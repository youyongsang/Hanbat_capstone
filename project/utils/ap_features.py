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
    "sta_tx_bitrate_mean",
)

# 2026-08-29: sta_tx_bitrate_mean 승격 (6 -> 7 feature).
#   가설은 "혼잡할수록 속도가 떨어진다(rate collapse)"였으나 실측(2115행)은
#   정반대: 혼잡할수록 값이 오히려 올라간다(부하 테스트라 혼잡 구간에서
#   기기가 실제로 데이터를 계속 밀어넣기 때문 - 한가할 때는 관리 프레임만
#   드문드문 잡혀 기본 저속으로 기록됨). occ 60~72%(다른 6개 feature가
#   label 2/3 사이에 완전히 동일해지는 구간)에서 label 2/3 간 Cohen's d=0.52로
#   유의미하게 갈라짐. 랜덤 시드 5개로 검증(그 전까지 이 파이프라인엔 시드
#   고정이 없었음) - exit-loss 가중치와 무관하게 7-feature가 6-feature보다
#   항상 Label3 F1이 높음(+5~11pt). 상세: .work-log/current.md 2026-08-29.
#   sta_tx_bitrate_min은 여전히 정보용(모델 입력 아님) - min은 위 검증에서
#   mean만큼 뚜렷한 신호를 보이지 않았음.
#
# 2026-08-28 (archived): 승격 전 상태 — 정보용 CSV 컬럼, 1차 정의(전체
#   station min)는 diag_25 런에서 유휴 station이 MCS 0(6.5)에 물려 상수 →
#   무신호. 2차 정의(이번 폴링에 실제 송신한 station만)로 바꿈.
#   상세: project/results/yongsang/ap_v2_redesign_threshold_comparison.txt
#
# 2026-08-27 혼잡 라벨 재설계 (docs/yongsang/congestion_label_redesign.{md,html}):
#   초기 9개 = throughput_mbps, channel_occupancy_percent, latency_ms, jitter_ms,
#             tx_retries_delta, tx_failed_delta, rssi_dbm, rssi_delta_db,
#             rssi_moving_avg_dbm. 재설계로 9 -> 6 (두 변경이 겹쳐 -3):
#   - (-2) latency_ms / jitter_ms 를 모델 입력에서 제거. jitter/loss는 victim
#     프로브 실측, latency는 ping RTT — 라벨 축이자 배포 시점엔 없는 측정.
#     모델은 "채널 상태만 보고 victim QoS를 예측"해야 하므로 정답(프로브·
#     ping)을 입력으로 주지 않는다.
#   - (-1) tx_retries_delta / tx_failed_delta (-> per_s) 2개를 tx_retry_ratio
#     하나로 통합.
#     retry_ratio = (retries + failed) / (retries + failed + tx_packets).
#     WLAN 헬스 표준 문턱(10/15/25/40%)에 매핑. 비율이라 폴링 주기 무관.
#   - occupancy / retry는 배포 시 AP 텔레메트리로 available하므로 라벨 축이면서
#     모델 입력으로도 유지.  => 9 - 2 - 1 = 6.
#
# 이전(2026-08-27 오전): tx_retries_delta/tx_failed_delta -> _per_s.
# 그 이전 metrics_v2.csv 는 이 재설계에 완전 relabel 불가(프로브·tx_packets
# 없음) — 레거시/사전학습용으로만.
