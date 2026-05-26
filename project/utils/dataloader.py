"""CSV loading and preprocessing utilities for traffic windows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = ["rps", "channel_occupancy", "packet_loss", "latency"]
SCALER_PARAMS = {
    "rps": {"min": 0.0, "max": 1000.0},
    "channel_occupancy": {"min": 0.0, "max": 100.0},
    "packet_loss": {"min": 0.0, "max": 30.0},
    "latency": {"min": 0.0, "max": 500.0},
}


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Apply fixed Min-Max normalization used by the simulator."""

    mins = np.array([SCALER_PARAMS[name]["min"] for name in FEATURE_COLUMNS], dtype=np.float32)
    maxs = np.array([SCALER_PARAMS[name]["max"] for name in FEATURE_COLUMNS], dtype=np.float32)
    return (features - mins) / (maxs - mins)


def make_sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 10,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat time-series rows into LSTM windows."""

    windows = []
    window_labels = []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        windows.append(features[start:end])
        window_labels.append(labels[end - 1])
    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_csv_windows(path: str | Path, window_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Load either pre-windowed CSV or flat timestamp CSV as model tensors."""

    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    if {"sample_id", "timestep"}.issubset(rows[0].keys()):
        return _load_prewindowed_rows(rows, window_size=window_size)

    features = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    if features.max() > 1.0:
        features = normalize_features(features)
    return make_sliding_windows(features, labels, window_size=window_size)


def _load_prewindowed_rows(
    rows: list[dict[str, str]],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["sample_id"])].append(row)

    samples = []
    labels = []
    for sample_id in sorted(grouped_rows):
        rows = sorted(grouped_rows[sample_id], key=lambda row: int(row["timestep"]))
        if len(rows) != window_size:
            raise ValueError(f"sample_id={sample_id} has {len(rows)} timesteps, expected {window_size}")
        samples.append([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows])
        labels.append(int(rows[-1]["label"]))

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64)


def validate_csv_dataset(path: str | Path, window_size: int = 10) -> dict[str, object]:
    """Validate CSV shape, labels, missing values, and feature range."""

    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    missing_values = {
        column: sum(row.get(column, "") == "" for row in rows)
        for column in rows[0].keys()
    }
    samples, labels = load_csv_windows(path, window_size=window_size)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    feature_min = samples.min(axis=(0, 1))
    feature_max = samples.max(axis=(0, 1))

    return {
        "path": str(path),
        "columns": list(rows[0].keys()),
        "rows": len(rows),
        "samples": int(samples.shape[0]),
        "shape": list(samples.shape),
        "label_counts": {str(label): int(count) for label, count in zip(unique_labels, label_counts)},
        "missing_values": missing_values,
        "feature_min": {name: float(value) for name, value in zip(FEATURE_COLUMNS, feature_min)},
        "feature_max": {name: float(value) for name, value in zip(FEATURE_COLUMNS, feature_max)},
    }


def get_dataloader(
    data_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    window_size: int = 10,
) -> DataLoader:
    samples, labels = load_csv_windows(data_path, window_size=window_size)
    dataset = TensorDataset(torch.from_numpy(samples), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
