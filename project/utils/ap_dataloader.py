"""Data loader for AP measurement feature windows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.ap_features import AP_FEATURE_COLUMNS, WINDOW_SIZE


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_ap_csv_windows(
    path: str | Path,
    window_size: int = WINDOW_SIZE,
    feature_columns: Sequence[str] = AP_FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-windowed AP measurement CSV as model tensors."""

    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    if not {"sample_id", "timestep"}.issubset(rows[0].keys()):
        raise ValueError(
            f"{path} must be pre-windowed with sample_id and timestep columns. "
            "Run project/scripts/prepare_ap_metrics_dataset.py first."
        )

    grouped_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["sample_id"])].append(row)

    samples = []
    labels = []
    for sample_id in sorted(grouped_rows):
        sample_rows = sorted(grouped_rows[sample_id], key=lambda row: int(row["timestep"]))
        if len(sample_rows) != window_size:
            raise ValueError(f"sample_id={sample_id} has {len(sample_rows)} timesteps, expected {window_size}")
        samples.append([[float(row[column]) for column in feature_columns] for row in sample_rows])
        labels.append(int(sample_rows[-1]["label"]))

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_ap_dataloader(
    data_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    window_size: int = WINDOW_SIZE,
) -> DataLoader:
    samples, labels = load_ap_csv_windows(data_path, window_size=window_size)
    dataset = TensorDataset(torch.from_numpy(samples), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def validate_ap_dataset(path: str | Path, window_size: int = WINDOW_SIZE) -> dict[str, object]:
    samples, labels = load_ap_csv_windows(path, window_size=window_size)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    feature_min = samples.min(axis=(0, 1))
    feature_max = samples.max(axis=(0, 1))
    return {
        "path": str(path),
        "samples": int(samples.shape[0]),
        "shape": list(samples.shape),
        "label_counts": {str(label): int(count) for label, count in zip(unique_labels, label_counts)},
        "feature_min": {name: float(value) for name, value in zip(AP_FEATURE_COLUMNS, feature_min)},
        "feature_max": {name: float(value) for name, value in zip(AP_FEATURE_COLUMNS, feature_max)},
    }
