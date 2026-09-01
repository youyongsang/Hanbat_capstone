"""Run ONNX Early Exit inference and save Raspberry Pi latency results.

AP-measurement (6-feature, ap_metrics_v2_redesign2) variant of
project/deploy/raspberry_pi/inference_pi.py (1st-semester, 4-feature). This
copy is self-contained (no repo imports) so it works from a bundle directory
copied onto the Pi.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


FEATURE_COLUMNS = [
    "throughput_mbps",
    "channel_occupancy_percent",
    "tx_retry_ratio",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
]
WINDOW_SIZE = 12  # 2026-09-01: 10 -> 12 (matches ap_features.WINDOW_SIZE + [1,12,7] ONNX)
DEFAULT_THETA_1 = 0.3
DEFAULT_THETA_2 = 0.6
DEFAULT_DYNAMIC_MIN_THRESHOLD = 0.22
DEFAULT_DYNAMIC_RECENT_STEPS = 5
DEFAULT_DYNAMIC_SPIKE_THRESHOLD = 0.25


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def entropy_from_logits(logits: np.ndarray) -> float:
    probs = softmax(logits)
    return float(-(probs * np.log(probs + 1e-8)).sum(axis=-1)[0])


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
        if len(group) != WINDOW_SIZE:
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
        raise ValueError(f"No valid {WINDOW_SIZE}-timestep samples found in {csv_path}")

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


def compute_dynamic_threshold(
    sample: np.ndarray,
    base_theta_1: float,
    base_theta_2: float,
    min_threshold: float,
    recent_steps: int,
    spike_threshold: float,
) -> tuple[float, float]:
    occupancy = sample[-recent_steps:, 1].astype(np.float32)
    if len(occupancy) < 2:
        return base_theta_1, base_theta_2

    delta = abs(float(occupancy[-1] - occupancy[-2]))
    if delta > spike_threshold:
        theta_1 = base_theta_1
        theta_2 = base_theta_2
    else:
        theta_1 = base_theta_1 * 1.25
        theta_2 = base_theta_2 * 1.25

    return max(theta_1, min_threshold), max(theta_2, min_threshold * 2)


def normalize_logits(logits: np.ndarray, exit_idx: int) -> np.ndarray:
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    elif logits.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape at exit {exit_idx}: {logits.shape}")
    return logits


def predict_from_logits(logits: np.ndarray) -> tuple[int, float]:
    probs = softmax(logits)
    return int(np.argmax(probs[0])), float(np.max(probs[0]))


def run_full_inference(
    session: ort.InferenceSession,
    output_names: list[str],
    input_name: str,
    batch: np.ndarray,
    threshold: float,
) -> tuple[int, int, float, float]:
    outputs = session.run(output_names, {input_name: batch})
    exit_point, predicted_label, confidence = choose_exit(outputs, threshold)
    entropy = entropy_from_logits(normalize_logits(outputs[exit_point - 1], exit_point))
    return exit_point, predicted_label, confidence, entropy


def run_staged_inference(
    sessions: list[ort.InferenceSession],
    sample: np.ndarray,
    batch: np.ndarray,
    dynamic_theta: bool,
    theta_1: float,
    theta_2: float,
    dynamic_min_threshold: float,
    dynamic_recent_steps: int,
    dynamic_spike_threshold: float,
) -> tuple[int, int, float, float]:
    if dynamic_theta:
        theta_1, theta_2 = compute_dynamic_threshold(
            sample,
            theta_1,
            theta_2,
            dynamic_min_threshold,
            dynamic_recent_steps,
            dynamic_spike_threshold,
        )

    stage1 = sessions[0]
    hidden1, logits1 = stage1.run(None, {stage1.get_inputs()[0].name: batch})
    logits1 = normalize_logits(logits1, 1)
    entropy1 = entropy_from_logits(logits1)
    if entropy1 < theta_1:
        predicted_label, confidence = predict_from_logits(logits1)
        return 1, predicted_label, confidence, entropy1

    stage2 = sessions[1]
    hidden2, logits2 = stage2.run(None, {stage2.get_inputs()[0].name: hidden1})
    logits2 = normalize_logits(logits2, 2)
    entropy2 = entropy_from_logits(logits2)
    if entropy2 < theta_2:
        predicted_label, confidence = predict_from_logits(logits2)
        return 2, predicted_label, confidence, entropy2

    stage3 = sessions[2]
    (logits3,) = stage3.run(None, {stage3.get_inputs()[0].name: hidden2})
    logits3 = normalize_logits(logits3, 3)
    predicted_label, confidence = predict_from_logits(logits3)
    return 3, predicted_label, confidence, entropy_from_logits(logits3)


def summarize(latencies: list[float], exit_points: list[int]) -> dict[str, float]:
    summary = {
        "sample_count": len(latencies),
        "avg_inference_ms": statistics.mean(latencies),
        "std_inference_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
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
        fh.write("Raspberry Pi ONNX inference summary (ap_metrics_v2_redesign2)\n")
        fh.write(f"model: {model_path}\n")
        fh.write(f"data: {data_path}\n")
        for key, value in summary.items():
            if isinstance(value, float):
                fh.write(f"{key}: {value:.6f}\n")
            else:
                fh.write(f"{key}: {value}\n")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Measure ONNX Early Exit latency on Raspberry Pi (AP redesign model)")
    parser.add_argument("--model", default=str(here / "ap_early_exit_fixed.onnx"))
    parser.add_argument("--mode", choices=["full", "staged"], default="staged")
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--stage2", type=Path)
    parser.add_argument("--stage3", type=Path)
    parser.add_argument("--dynamic-theta", action="store_true")
    parser.add_argument("--theta-1", type=float, default=DEFAULT_THETA_1)
    parser.add_argument("--theta-2", type=float, default=DEFAULT_THETA_2)
    parser.add_argument("--dynamic-min-threshold", type=float, default=DEFAULT_DYNAMIC_MIN_THRESHOLD)
    parser.add_argument("--dynamic-recent-steps", type=int, default=DEFAULT_DYNAMIC_RECENT_STEPS)
    parser.add_argument("--dynamic-spike-threshold", type=float, default=DEFAULT_DYNAMIC_SPIKE_THRESHOLD)
    parser.add_argument("--data", default=str(here / "test.csv"))
    parser.add_argument("--output", default=str(here / "pi_inference_results.csv"))
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = "ap_early_exit_dynamic" if args.dynamic_theta else "ap_early_exit_fixed"
    if args.mode == "staged":
        if not args.stage1:
            args.stage1 = here / f"{prefix}_stage1.onnx"
        if not args.stage2:
            args.stage2 = here / f"{prefix}_stage2.onnx"
        if not args.stage3:
            args.stage3 = here / f"{prefix}_stage3.onnx"
        stage_paths = [Path(args.stage1), Path(args.stage2), Path(args.stage3)]
        for stage_path in stage_paths:
            if not stage_path.exists():
                raise FileNotFoundError(f"Stage ONNX model not found: {stage_path}")
    elif not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {data_path}")

    samples, meta_rows = load_samples(data_path, args.max_samples)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = None
    input_name = ""
    output_names: list[str] = []
    stage_sessions: list[ort.InferenceSession] = []
    if args.mode == "staged":
        stage_sessions = [
            ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
            for path in stage_paths
        ]
    else:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

    warmup_sample = samples[:1]
    for _ in range(args.warmup):
        if args.mode == "staged":
            run_staged_inference(
                stage_sessions,
                warmup_sample[0],
                warmup_sample,
                args.dynamic_theta,
                args.theta_1,
                args.theta_2,
                args.dynamic_min_threshold,
                args.dynamic_recent_steps,
                args.dynamic_spike_threshold,
            )
        else:
            if session is None:
                raise RuntimeError("Full ONNX session is not initialized")
            session.run(output_names, {input_name: warmup_sample})

    result_rows: list[dict] = []
    avg_latencies: list[float] = []
    exit_points: list[int] = []

    for sample, meta in zip(samples, meta_rows):
        batch = sample.reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS)).astype(np.float32)
        sample_latencies: list[float] = []
        exit_point = 0
        predicted_label = 0
        confidence = 0.0
        entropy = 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            if args.mode == "staged":
                exit_point, predicted_label, confidence, entropy = run_staged_inference(
                    stage_sessions,
                    sample,
                    batch,
                    args.dynamic_theta,
                    args.theta_1,
                    args.theta_2,
                    args.dynamic_min_threshold,
                    args.dynamic_recent_steps,
                    args.dynamic_spike_threshold,
                )
            else:
                if session is None:
                    raise RuntimeError("Full ONNX session is not initialized")
                exit_point, predicted_label, confidence, entropy = run_full_inference(
                    session,
                    output_names,
                    input_name,
                    batch,
                    args.threshold,
                )
            sample_latencies.append((time.perf_counter() - start) * 1000.0)

        avg_latency = statistics.mean(sample_latencies)
        avg_latencies.append(avg_latency)
        exit_points.append(exit_point)

        result_rows.append(
            {
                **meta,
                "predicted_label": predicted_label,
                "exit_point": exit_point,
                "confidence": round(confidence, 6),
                "entropy": round(entropy, 6),
                "inference_ms": round(avg_latency, 6),
                "inference_std_ms": round(
                    statistics.pstdev(sample_latencies) if len(sample_latencies) > 1 else 0.0,
                    6,
                ),
                "inference_min_ms": round(min(sample_latencies), 6),
                "inference_max_ms": round(max(sample_latencies), 6),
                "inference_p50_ms": round(statistics.median(sample_latencies), 6),
                "inference_p95_ms": round(float(np.percentile(sample_latencies, 95)), 6),
                "measurement_repeats": args.repeats,
                "inference_mode": args.mode,
                "dynamic_theta": int(args.dynamic_theta),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = summarize(avg_latencies, exit_points)
    summary["measurement_repeats"] = args.repeats
    summary["inference_mode"] = args.mode
    summary["dynamic_theta"] = int(args.dynamic_theta)
    write_summary(output_path, summary, model_path if args.mode == "full" else stage_paths[0], data_path)

    print("Raspberry Pi ONNX inference measurement complete (ap_metrics_v2_redesign2)")
    print(f"mode: {args.mode}")
    if args.mode == "staged":
        print("stage_models: " + ", ".join(str(path) for path in stage_paths))
    else:
        print(f"model: {model_path}")
    print(f"data: {data_path}")
    print(f"output_csv: {output_path}")
    print(f"output_txt: {output_path.with_suffix('.txt')}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"measurement_repeats: {summary['measurement_repeats']}")
    print(f"avg_inference_ms: {summary['avg_inference_ms']:.6f}")
    print(f"std_inference_ms: {summary['std_inference_ms']:.6f}")
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
