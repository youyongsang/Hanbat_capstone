"""Shared AP measurement feature contract."""

from __future__ import annotations

from typing import Tuple


AP_FEATURE_COLUMNS: Tuple[str, ...] = (
    "throughput_mbps",
    "channel_occupancy_percent",
    "latency_ms",
    "jitter_ms",
    "tx_retries_delta",
    "tx_failed_delta",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
)
