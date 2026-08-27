"""Recompute every derived column in metrics_v2.csv straight from the raw
measurement columns.

Unlike relabel_metrics_v2.py (which only recombines the four stored *_score
columns with new weights), this recomputes the sub-scores themselves from the
raw features. Use it when a score *formula* changes, not just its weight.

2026-08-27 one-time migration it was written for:
    tx_retries_delta / tx_failed_delta  ->  tx_retries_per_s / tx_failed_per_s

    The retry/failed deltas were "since last poll" counts, so their magnitude
    scaled with the polling interval (a 4s poll delta is ~4x a 1s poll delta
    for the same congestion). Moving the collector to the Pi dropped the
    interval to ~1s and deflated the retry signal, pinning everything at
    label 1. Fix: normalize to a per-second rate, exactly like throughput.

    Historical rows have no recorded interval, so it is back-computed from the
    gap between consecutive timestamps within each scenario. Timestamps are at
    1s resolution, so for the usual ~4s historical polls this is +/-25%; for
    rows right after an SSH poll stall (gap up to 156s) dividing by the real
    gap actually *corrects* a previously inflated value. The first row of each
    scenario has no predecessor -> interval 0, rates 0 (prepare drops it as a
    delta cold-start anyway).

After running this, re-run prepare_ap_metrics_dataset.py and retrain.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with calculate_scores() in collect_metrics.py.
THROUGHPUT_MAX_MBPS = 150.0
RETRY_FAILED_MAX_PER_SEC = 6250.0
JITTER_MAX_MS = 300.0
WEIGHTS = (
    ("throughput_score", 0.20),
    ("occupancy_score", 0.45),
    ("retry_failed_score", 0.20),
    ("jitter_score", 0.15),
)
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# Column order of the migrated file (matches CSV_COLUMNS in collect_metrics.py).
OUT_COLUMNS = [
    "timestamp",
    "throughput_mbps",
    "channel_occupancy_percent",
    "channel_occupancy_method",
    "latency_ms",
    "jitter_ms",
    "packet_loss_udp_percent",
    "poll_interval_s",
    "tx_retries_per_s",
    "tx_failed_per_s",
    "rssi_dbm",
    "connected_clients",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
    "scenario",
    "throughput_score",
    "occupancy_score",
    "retry_failed_score",
    "jitter_score",
    "congestion_score",
    "label",
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def label_for(score: float) -> int:
    if score < 0.25:
        return 0
    if score < 0.50:
        return 1
    if score < 0.75:
        return 2
    return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute metrics_v2.csv derived columns from raw features."
    )
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "scripts" / "metrics_v2.csv"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy before overwriting.",
    )
    return parser.parse_args()


def poll_interval_for_row(row: dict, prev_row: dict | None) -> float:
    """Seconds since the previous poll in the same scenario.

    Uses the recorded poll_interval_s if the collector wrote one, otherwise
    back-computes it from consecutive timestamps.
    """
    recorded = row.get("poll_interval_s", "")
    if recorded not in (None, ""):
        try:
            value = float(recorded)
            if value > 0:
                return value
        except ValueError:
            pass

    if prev_row is None or prev_row["scenario"] != row["scenario"]:
        return 0.0

    try:
        t0 = datetime.strptime(prev_row["timestamp"], TS_FORMAT)
        t1 = datetime.strptime(row["timestamp"], TS_FORMAT)
    except ValueError:
        return 0.0

    diff = (t1 - t0).total_seconds()
    if diff <= 0:
        # Same-second poll (never happens in the ~4s historical data, but the
        # Pi at ~1s would hit it). 1s floor slightly underestimates the rate.
        return 1.0
    return diff


def raw_delta(row: dict, per_s_col: str, delta_col: str, interval: float) -> float:
    """The retransmission count for this interval, whichever form is present."""
    if delta_col in row and row[delta_col] not in (None, ""):
        return float(row[delta_col])
    if per_s_col in row and row[per_s_col] not in (None, ""):
        return float(row[per_s_col]) * interval
    return 0.0


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"no rows in {args.input}")

    old_labels: Counter[int] = Counter(int(r["label"]) for r in rows)
    old_retry_col = (
        "tx_retries_delta" if "tx_retries_delta" in rows[0] else "tx_retries_per_s"
    )

    prev_row: dict | None = None
    for row in rows:
        interval = poll_interval_for_row(row, prev_row)

        retries = raw_delta(row, "tx_retries_per_s", "tx_retries_delta", interval)
        failed = raw_delta(row, "tx_failed_per_s", "tx_failed_delta", interval)

        if interval > 0:
            retries_per_s = round(retries / interval, 2)
            failed_per_s = round(failed / interval, 2)
        else:
            retries_per_s = 0.0
            failed_per_s = 0.0

        throughput_score = clamp01(float(row["throughput_mbps"]) / THROUGHPUT_MAX_MBPS)
        occupancy_score = clamp01(float(row["channel_occupancy_percent"]) / 100.0)
        retry_failed_score = clamp01(
            (retries_per_s + failed_per_s) / RETRY_FAILED_MAX_PER_SEC
        )
        jitter_score = clamp01(float(row["jitter_ms"]) / JITTER_MAX_MS)

        scores = {
            "throughput_score": round(throughput_score, 4),
            "occupancy_score": round(occupancy_score, 4),
            "retry_failed_score": round(retry_failed_score, 4),
            "jitter_score": round(jitter_score, 4),
        }
        congestion = round(
            clamp01(sum(weight * scores[key] for key, weight in WEIGHTS)), 4
        )

        row["poll_interval_s"] = round(interval, 3)
        row["tx_retries_per_s"] = retries_per_s
        row["tx_failed_per_s"] = failed_per_s
        row.update(scores)
        row["congestion_score"] = congestion
        row["label"] = label_for(congestion)
        row.pop("tx_retries_delta", None)
        row.pop("tx_failed_delta", None)

        prev_row = row

    if not args.no_backup:
        backup = args.input.with_suffix(args.input.suffix + ".bak")
        shutil.copy2(args.input, backup)
        print(f"backup: {backup}")

    with open(args.input, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    new_labels: Counter[int] = Counter(int(r["label"]) for r in rows)
    print(f"remeasured: {args.input}  ({len(rows)} rows)")
    print(f"  retry source column: {old_retry_col}")
    print(f"  label   old: {dict(sorted(old_labels.items()))}")
    print(f"  label   new: {dict(sorted(new_labels.items()))}")


if __name__ == "__main__":
    main()
