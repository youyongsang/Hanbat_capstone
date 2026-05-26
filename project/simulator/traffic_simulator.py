"""Factory traffic simulator for congestion-detection datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


FEATURE_NAMES = ["rps", "channel_occupancy", "packet_loss", "latency"]
SCALER_PARAMS = {
    "rps": {"min": 0.0, "max": 1000.0},
    "channel_occupancy": {"min": 0.0, "max": 100.0},
    "packet_loss": {"min": 0.0, "max": 30.0},
    "latency": {"min": 0.0, "max": 500.0},
}
TARGET_LABEL_COUNTS = {0: 600, 1: 200, 2: 150, 3: 50}


def assign_label(channel_occupancy: float) -> int:
    """Assign congestion label from channel occupancy."""
    if channel_occupancy < 40:
        return 0
    if channel_occupancy < 65:
        return 1
    if channel_occupancy < 85:
        return 2
    return 3


def _ramp(start: float, end: float, count: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    values = np.linspace(start, end, count)
    return values + rng.normal(0, noise, count)


def _clip_features(data: np.ndarray) -> np.ndarray:
    bounds = np.array([[0, 1000], [0, 100], [0, 30], [0, 500]], dtype=float)
    return np.clip(data, bounds[:, 0], bounds[:, 1])


def _series_from_segments(segments: list[np.ndarray]) -> np.ndarray:
    data = np.vstack(segments)
    labels = np.array([assign_label(row[1]) for row in data], dtype=int)
    return np.column_stack([_clip_features(data), labels])


def simulate_startup_surge(rng: np.random.Generator) -> np.ndarray:
    """Scenario 1: shift start surge around 08:00."""
    idle = np.column_stack(
        [
            rng.uniform(50, 100, 22),
            rng.uniform(10, 20, 22),
            rng.uniform(0, 1, 22),
            rng.uniform(10, 30, 22),
        ]
    )
    warmup = np.column_stack(
        [
            _ramp(95, 390, 24, rng, 18),
            _ramp(20, 55, 24, rng, 3),
            _ramp(1, 7, 24, rng, 0.8),
            _ramp(30, 100, 24, rng, 9),
        ]
    )
    peak = np.column_stack(
        [
            rng.uniform(450, 700, 24),
            rng.uniform(66, 78, 24),
            rng.uniform(6, 15, 24),
            rng.uniform(90, 200, 24),
        ]
    )
    return _series_from_segments([idle, warmup, peak])


def simulate_emergency_ramp(rng: np.random.Generator) -> np.ndarray:
    """Scenario 2: unplanned production spike around 14:15."""
    normal = np.column_stack(
        [
            rng.uniform(300, 500, 24),
            rng.uniform(30, 39, 24),
            rng.uniform(1, 3, 24),
            rng.uniform(30, 80, 24),
        ]
    )
    warning = np.column_stack(
        [
            _ramp(500, 720, 16, rng, 28),
            _ramp(42, 63, 16, rng, 2),
            _ramp(3, 10, 16, rng, 0.9),
            _ramp(80, 170, 16, rng, 12),
        ]
    )
    severe = np.column_stack(
        [
            _ramp(760, 1000, 30, rng, 25),
            _ramp(86, 99, 30, rng, 2),
            _ramp(20, 30, 30, rng, 1.1),
            _ramp(300, 500, 30, rng, 20),
        ]
    )
    return _series_from_segments([normal, warning, severe])


def simulate_lunch_restart(rng: np.random.Generator) -> np.ndarray:
    """Scenario 3: gradual restart after lunch."""
    lunch = np.column_stack(
        [
            rng.uniform(10, 30, 24),
            rng.uniform(5, 10, 24),
            rng.uniform(0, 0.2, 24),
            rng.uniform(5, 15, 24),
        ]
    )
    step1 = np.column_stack(
        [
            _ramp(30, 120, 18, rng, 10),
            _ramp(10, 25, 18, rng, 1.5),
            _ramp(0.2, 2, 18, rng, 0.3),
            _ramp(15, 45, 18, rng, 5),
        ]
    )
    step2 = np.column_stack(
        [
            _ramp(120, 330, 18, rng, 18),
            _ramp(25, 55, 18, rng, 2),
            _ramp(2, 8, 18, rng, 0.6),
            _ramp(45, 130, 18, rng, 8),
        ]
    )
    step3 = np.column_stack(
        [
            _ramp(330, 550, 20, rng, 20),
            _ramp(55, 75, 20, rng, 2),
            _ramp(8, 15, 20, rng, 0.8),
            _ramp(130, 230, 20, rng, 12),
        ]
    )
    return _series_from_segments([lunch, step1, step2, step3])


def simulate_imbalanced_ap_load(rng: np.random.Generator) -> np.ndarray:
    """Scenario 4: maintenance shifts load toward a specific AP."""
    warning = np.column_stack(
        [
            _ramp(300, 470, 20, rng, 15),
            _ramp(40, 62, 20, rng, 2),
            _ramp(2, 7, 20, rng, 0.5),
            _ramp(50, 135, 20, rng, 8),
        ]
    )
    congestion = np.column_stack(
        [
            _ramp(470, 700, 26, rng, 18),
            _ramp(66, 82, 26, rng, 2),
            _ramp(7, 18, 26, rng, 0.8),
            _ramp(135, 250, 26, rng, 12),
        ]
    )
    severe_tail = np.column_stack(
        [
            _ramp(700, 850, 12, rng, 18),
            _ramp(85, 92, 12, rng, 1.5),
            _ramp(18, 24, 12, rng, 0.8),
            _ramp(250, 340, 12, rng, 14),
        ]
    )
    return _series_from_segments([warning, congestion, severe_tail])


SCENARIO_FUNCTIONS = [
    ("startup_surge", simulate_startup_surge),
    ("emergency_ramp", simulate_emergency_ramp),
    ("lunch_restart", simulate_lunch_restart),
    ("imbalanced_ap_load", simulate_imbalanced_ap_load),
]


def make_sliding_windows(series: np.ndarray, window_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    features = series[:, :4]
    labels = series[:, 4].astype(int)
    windows = []
    window_labels = []
    for start in range(0, len(series) - window_size + 1):
        end = start + window_size
        windows.append(features[start:end])
        window_labels.append(labels[end - 1])
    return np.array(windows, dtype=float), np.array(window_labels, dtype=int)


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    mins = np.array([SCALER_PARAMS[name]["min"] for name in FEATURE_NAMES])
    maxs = np.array([SCALER_PARAMS[name]["max"] for name in FEATURE_NAMES])
    return (windows - mins) / (maxs - mins)


def generate_candidate_windows(seed: int = 42, cycles: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_windows = []
    all_labels = []
    all_scenarios = []

    for _ in range(cycles):
        for scenario_name, scenario_fn in SCENARIO_FUNCTIONS:
            series = scenario_fn(rng)
            windows, labels = make_sliding_windows(series)
            all_windows.append(windows)
            all_labels.append(labels)
            all_scenarios.extend([scenario_name] * len(labels))

    return (
        np.vstack(all_windows),
        np.concatenate(all_labels),
        np.array(all_scenarios, dtype=object),
    )


def select_target_distribution(
    windows: np.ndarray,
    labels: np.ndarray,
    scenarios: np.ndarray,
    seed: int = 42,
    target_counts: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target_counts = target_counts or TARGET_LABEL_COUNTS
    selected_indices = []

    for label, count in target_counts.items():
        indices = np.flatnonzero(labels == label)
        if len(indices) < count:
            raise ValueError(f"Not enough label {label} samples: need {count}, got {len(indices)}")
        selected_indices.extend(rng.choice(indices, size=count, replace=False))

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    return windows[selected_indices], labels[selected_indices], scenarios[selected_indices]


def stratified_split(
    windows: np.ndarray,
    labels: np.ndarray,
    scenarios: np.ndarray,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    split_indices = {"train": [], "val": [], "test": []}

    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        total = len(indices)
        train_end = int(total * 0.70)
        val_end = train_end + round(total * 0.15)
        split_indices["train"].extend(indices[:train_end])
        split_indices["val"].extend(indices[train_end:val_end])
        split_indices["test"].extend(indices[val_end:])

    result = {}
    for split_name, indices in split_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        result[split_name] = (windows[indices], labels[indices], scenarios[indices])
    return result


def write_split_csv(path: Path, windows: np.ndarray, labels: np.ndarray, scenarios: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "timestep",
                "rps",
                "channel_occupancy",
                "packet_loss",
                "latency",
                "label",
                "scenario",
            ]
        )
        for sample_id, (window, label, scenario) in enumerate(zip(windows, labels, scenarios)):
            for timestep, row in enumerate(window):
                writer.writerow([sample_id, timestep, *[round(value, 6) for value in row], label, scenario])


def save_dataset(output_dir: Path, seed: int = 42) -> dict[str, object]:
    windows, labels, scenarios = generate_candidate_windows(seed=seed)
    windows, labels, scenarios = select_target_distribution(windows, labels, scenarios, seed=seed)
    windows = normalize_windows(windows)
    splits = stratified_split(windows, labels, scenarios, seed=seed)

    for split_name, (split_windows, split_labels, split_scenarios) in splits.items():
        write_split_csv(output_dir / f"{split_name}.csv", split_windows, split_labels, split_scenarios)

    scaler_path = output_dir / "scaler_params.json"
    scaler_path.write_text(json.dumps(SCALER_PARAMS, indent=2), encoding="utf-8")

    summary = {
        "shape": {"timesteps": 10, "features": 4},
        "feature_names": FEATURE_NAMES,
        "scaler_params": SCALER_PARAMS,
        "splits": {},
    }
    for split_name, (split_windows, split_labels, split_scenarios) in splits.items():
        summary["splits"][split_name] = {
            "samples": int(len(split_windows)),
            "window_shape": list(split_windows.shape),
            "label_counts": {str(k): int(v) for k, v in sorted(Counter(split_labels).items())},
            "scenario_counts": {str(k): int(v) for k, v in sorted(Counter(split_scenarios).items())},
        }

    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
