"""Run ONNX Early Exit inference and save Raspberry Pi latency results.

This script is designed to run both from the repository root and from a
Raspberry Pi deployment bundle directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


FEATURE_COLUMNS = ["rps", "channel_occupancy", "packet_loss", "latency"]


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path(root: Path) -> Path:
    local_model = Path(__file__).resolve().parent / "early_exit_fixed.onnx"
    if local_model.exists():
        return local_model
    return root / "project" / "checkpoints" / "early_exit_fixed.onnx"


def default_data_path(root: Path) -> Path:
    local_data = Path(__file__).resolve().parent / "test.csv"
    if local_data.exists():
        return local_data
    return root / "project" / "data" / "real" / "test.csv"


def default_output_path(root: Path) -> Path:
    local_dir = Path(__file__).resolve().parent
    if (local_dir / "early_exit_fixed.onnx").exists():
        return local_dir / "pi_inference_results.csv"
    return root / "project" / "results" / "hojung" / "pi_inference_results.csv"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_samples(csv_path: Path, max_samples: int | None) -> tuple[np.ndarray, list[dict]]:
    df = pd.read_csv(csv_path)
    required = {"sample_id", "timestep", *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    samples: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for sample_id, group in df.groupby("sample_id", sort=True):
        group = group.sort_values("timestep")
        if len(group) != 10:
            continue

        features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        samples.append(features)
        first = group.iloc[0]
        last = group.iloc[-1]
        meta_rows.append(
            {
                "sample_id": int(sample_id),
                "true_label": int(last["label"]) if "label" in group.columns else "",
                "scenario": str(first["scenario"]) if "scenario" in group.columns else "",
            }
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError(f"No valid 10-timestep samples found in {csv_path}")

    return np.stack(samples).astype(np.float32), meta_rows


def choose_exit(outputs: list[np.ndarray], threshold: float) -> tuple[int, int, float]:
    for exit_idx, logits in enumerate(outputs, start=1):
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        elif logits.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape at exit {exit_idx}: {logits.shape}")

        probs = softmax(logits)
        predicted_label = int(np.argmax(probs[0]))
        confidence = float(np.max(probs[0]))
        if confidence >= threshold or exit_idx == len(outputs):
            return exit_idx, predicted_label, confidence
    raise RuntimeError("Early Exit selection failed")


def summarize(latencies: list[float], exit_points: list[int]) -> dict[str, float]:
    summary = {
        "sample_count": len(latencies),
        "avg_inference_ms": statistics.mean(latencies),
        "min_inference_ms": min(latencies),
        "max_inference_ms": max(latencies),
        "p50_inference_ms": statistics.median(latencies),
        "p95_inference_ms": float(np.percentile(latencies, 95)),
    }
    for exit_id in (1, 2, 3):
        summary[f"exit{exit_id}_count"] = exit_points.count(exit_id)
        summary[f"exit{exit_id}_rate"] = exit_points.count(exit_id) / len(exit_points)
    return summary


def write_summary(path: Path, summary: dict[str, float], model_path: Path, data_path: Path) -> None:
    txt_path = path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("Raspberry Pi ONNX inference summary\n")
        fh.write(f"model: {model_path}\n")
        fh.write(f"data: {data_path}\n")
        for key, value in summary.items():
            if isinstance(value, float):
                fh.write(f"{key}: {value:.6f}\n")
            else:
                fh.write(f"{key}: {value}\n")


def main() -> None:
    root = resolve_repo_root()
    parser = argparse.ArgumentParser(description="Measure ONNX Early Exit latency on Raspberry Pi")
    parser.add_argument(
        "--model",
        default=str(default_model_path(root)),
        help="Path to early_exit_fixed.onnx",
    )
    parser.add_argument(
        "--data",
        default=str(default_data_path(root)),
        help="Path to windowed test CSV",
    )
    parser.add_argument(
        "--output",
        default=str(default_output_path(root)),
        help="CSV path for measured latency results",
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for pseudo Early Exit")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum samples to measure")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup inference count excluded from timing")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {data_path}")

    samples, meta_rows = load_samples(data_path, args.max_samples)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    warmup_sample = samples[:1]
    for _ in range(args.warmup):
        session.run(output_names, {input_name: warmup_sample})

    result_rows: list[dict] = []
    latencies: list[float] = []
    exit_points: list[int] = []

    for sample, meta in zip(samples, meta_rows):
        batch = sample.reshape(1, 10, 4).astype(np.float32)
        start = time.perf_counter()
        outputs = session.run(output_names, {input_name: batch})
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        exit_point, predicted_label, confidence = choose_exit(outputs, args.threshold)
        latencies.append(elapsed_ms)
        exit_points.append(exit_point)

        result_rows.append(
            {
                **meta,
                "predicted_label": predicted_label,
                "exit_point": exit_point,
                "confidence": round(confidence, 6),
                "inference_ms": round(elapsed_ms, 6),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = summarize(latencies, exit_points)
    write_summary(output_path, summary, model_path, data_path)

    print("Raspberry Pi ONNX inference measurement complete")
    print(f"model: {model_path}")
    print(f"data: {data_path}")
    print(f"output_csv: {output_path}")
    print(f"output_txt: {output_path.with_suffix('.txt')}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"avg_inference_ms: {summary['avg_inference_ms']:.6f}")
    print(f"min_inference_ms: {summary['min_inference_ms']:.6f}")
    print(f"max_inference_ms: {summary['max_inference_ms']:.6f}")
    print(
        "exit_rate: "
        f"exit1={summary['exit1_rate']:.2%}, "
        f"exit2={summary['exit2_rate']:.2%}, "
        f"exit3={summary['exit3_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
