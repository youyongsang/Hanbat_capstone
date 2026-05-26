"""CSV dataset loading utilities."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = ["rps", "channel_occupancy", "packet_loss", "latency"]


def load_csv_windows(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    grouped_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            grouped_rows[int(row["sample_id"])].append(row)

    samples = []
    labels = []
    for sample_id in sorted(grouped_rows):
        rows = sorted(grouped_rows[sample_id], key=lambda item: int(item["timestep"]))
        if len(rows) != 10:
            raise ValueError(f"sample_id={sample_id} has {len(rows)} timesteps, expected 10")
        samples.append([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows])
        labels.append(int(rows[-1]["label"]))

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_dataloader(path: str | Path, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    samples, labels = load_csv_windows(path)
    dataset = TensorDataset(torch.from_numpy(samples), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
