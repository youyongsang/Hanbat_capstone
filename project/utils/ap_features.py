"""Shared AP measurement feature contract."""

from __future__ import annotations

from typing import Tuple


AP_FEATURE_COLUMNS: Tuple[str, ...] = (
    "throughput_mbps",
    "channel_occupancy_percent",
    "latency_ms",
    "jitter_ms",
    "tx_retries_per_s",
    "tx_failed_per_s",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
)

# 2026-08-27: tx_retries_delta / tx_failed_delta -> _per_s 로 이름·의미 변경.
# 델타값은 폴링 주기에 비례해서 흔들렸다(4초 폴링 = 1초 폴링 x4). 이제
# 초당 재전송률로 정규화한다. 기존 metrics_v2.csv는 remeasure_metrics_v2.py로
# 마이그레이션됨(폴링 간격은 timestamp 차이로 역산, 근사치).
