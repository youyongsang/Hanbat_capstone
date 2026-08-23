"""Recompute congestion_score/label in metrics_v2.csv from stored sub-scores.

Use this after changing the score weights in collect_metrics.py's
calculate_scores() (the four *_score columns already hold the raw
per-feature normalizations, so relabeling only needs to recombine them
with the new weights - no need to re-collect from the AP).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with calculate_scores() in collect_metrics.py.
WEIGHTS = {
    "throughput_score": 0.20,
    "occupancy_score": 0.45,
    "retry_failed_score": 0.20,
    "jitter_score": 0.15,
}


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
    parser = argparse.ArgumentParser(description="Relabel metrics_v2.csv with current score weights.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "scripts" / "metrics_v2.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    old_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    new_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for row in rows:
        old_counts[int(row["label"])] += 1

        score = sum(WEIGHTS[key] * float(row[key]) for key in WEIGHTS)
        score = round(clamp01(score), 4)
        new_label = label_for(score)

        row["congestion_score"] = f"{score}"
        row["label"] = str(new_label)
        new_counts[new_label] += 1

    with open(args.input, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relabeled: {args.input}")
    print(f"Old label distribution: {old_counts}")
    print(f"New label distribution: {new_counts}")


if __name__ == "__main__":
    main()
