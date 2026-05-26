"""Generate label-aware dummy traffic data for baseline LSTM training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["rps", "channel_occupancy", "packet_loss", "latency"]
LABEL_COUNTS = {0: 600, 1: 200, 2: 150, 3: 50}
LABEL_RANGES = {
    0: ((0, 400), (0, 40), (0, 2), (0, 50)),
    1: ((400, 650), (40, 65), (2, 8), (50, 150)),
    2: ((650, 850), (65, 85), (8, 20), (150, 300)),
    3: ((850, 1000), (85, 100), (20, 30), (300, 500)),
}
SCALER_MAX = np.array([1000.0, 100.0, 30.0, 500.0], dtype=np.float32)


def generate_sample(label: int, rng: np.random.Generator) -> np.ndarray:
    ranges = LABEL_RANGES[label]
    center = np.array([rng.uniform(low, high) for low, high in ranges], dtype=np.float32)
    span = np.array([high - low for low, high in ranges], dtype=np.float32)
    trend = rng.normal(0, span * 0.015, size=(10, 4)).astype(np.float32)
    jitter = rng.normal(0, span * 0.035, size=(10, 4)).astype(np.float32)
    sample = center + np.cumsum(trend, axis=0) + jitter
    lows = np.array([low for low, _ in ranges], dtype=np.float32)
    highs = np.array([high for _, high in ranges], dtype=np.float32)
    return np.clip(sample, lows, highs) / SCALER_MAX


def build_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = []
    labels = []
    for label, count in LABEL_COUNTS.items():
        for _ in range(count):
            samples.append(generate_sample(label, rng))
            labels.append(label)

    indices = rng.permutation(len(samples))
    return np.array(samples, dtype=np.float32)[indices], np.array(labels, dtype=np.int64)[indices]


def stratified_split(
    samples: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    split_indices = {"train": [], "val": [], "test": []}

    for label in sorted(LABEL_COUNTS):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.70)
        val_end = train_end + round(len(indices) * 0.15)
        split_indices["train"].extend(indices[:train_end])
        split_indices["val"].extend(indices[train_end:val_end])
        split_indices["test"].extend(indices[val_end:])

    splits = {}
    for split_name, indices in split_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        splits[split_name] = (samples[indices], labels[indices])
    return splits


def write_csv(path: Path, samples: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "timestep", *FEATURE_COLUMNS, "label"])
        for sample_id, (sample, label) in enumerate(zip(samples, labels)):
            for timestep, row in enumerate(sample):
                writer.writerow([sample_id, timestep, *[round(float(value), 6) for value in row], int(label)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy LSTM dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dummy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, labels = build_dataset(args.seed)
    splits = stratified_split(samples, labels, args.seed)
    for split_name, (split_samples, split_labels) in splits.items():
        write_csv(args.output_dir / f"{split_name}.csv", split_samples, split_labels)
        counts = {label: int((split_labels == label).sum()) for label in sorted(LABEL_COUNTS)}
        print(f"{split_name}: shape={split_samples.shape}, labels={counts}")


if __name__ == "__main__":
    main()
