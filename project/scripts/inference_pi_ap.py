"""Run ONNX inference (AP 9-feature) and save Raspberry Pi latency results.

This script is designed to run both from the repository root and from a
Raspberry Pi deployment bundle directory. In staged mode, each exit point is
a separate ONNX session, so later LSTM layers are skipped after an early
stop. Baseline mode runs a single non-branching ONNX graph.

Feature columns and order must match utils/ap_features.py exactly. They are
hardcoded here (not imported) so this file can be copied alone into a
Raspberry Pi deployment bundle without the rest of the repo.
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
    "latency_ms",
    "jitter_ms",
    "tx_retries_delta",
    "tx_failed_delta",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
]
DEFAULT_THETA_1 = 0.3
DEFAULT_THETA_2 = 0.6
DEFAULT_DYNAMIC_MIN_THRESHOLD = 0.22
DEFAULT_DYNAMIC_RECENT_STEPS = 5
DEFAULT_DYNAMIC_SPIKE_THRESHOLD = 0.25


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path(root: Path) -> Path:
    local_model = Path(__file__).resolve().parent / "ap_early_exit_fixed.onnx"
    if local_model.exists():
        return local_model
    return root / "project" / "checkpoints" / "ap_cleaned_strict" / "ap_early_exit_fixed.onnx"


def default_stage_path(root: Path, stage: int, prefix: str) -> Path:
    local_model = Path(__file__).resolve().parent / f"{prefix}_stage{stage}.onnx"
    if local_model.exists():
        return local_model
    return root / "project" / "checkpoints" / "ap_cleaned_strict" / f"{prefix}_stage{stage}.onnx"


def default_data_path(root: Path) -> Path:
    local_data = Path(__file__).resolve().parent / "test.csv"
    if local_data.exists():
        return local_data
    return root / "project" / "data" / "ap_metrics_cleaned_strict" / "test.csv"


def default_output_path(root: Path) -> Path:
    local_dir = Path(__file__).resolve().parent
    if (local_dir / "ap_early_exit_fixed.onnx").exists():
        return local_dir / "pi_ap_inference_results.csv"
    return root / "project" / "results" / "yongsang" / "pi_ap_inference_results.csv"


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


def compute_dynamic_threshold(
    sample: np.ndarray,
    base_theta_1: float,
    base_theta_2: float,
    min_threshold: float,
    recent_steps: int,
    spike_threshold: float,
) -> tuple[float, float]:
    # feature index 1 is channel_occupancy_percent (normalized 0~1), same as
    # the 1st-semester rps/channel_occupancy/packet_loss/latency layout.
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


def run_baseline_inference(
    session: ort.InferenceSession,
    output_names: list[str],
    input_name: str,
    batch: np.ndarray,
) -> tuple[int, int, float, float]:
    """Baseline LSTM always runs the full 3-layer graph; report it as exit 3."""

    (logits,) = session.run(output_names, {input_name: batch})
    logits = normalize_logits(logits, 3)
    predicted_label, confidence = predict_from_logits(logits)
    return 3, predicted_label, confidence, entropy_from_logits(logits)


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


def run_staged_confidence_inference(
    sessions: list[ort.InferenceSession],
    batch: np.ndarray,
    threshold: float,
) -> tuple[int, int, float, float]:
    """SDN-style staged inference: stop on max softmax confidence, not entropy."""

    stage1 = sessions[0]
    hidden1, logits1 = stage1.run(None, {stage1.get_inputs()[0].name: batch})
    logits1 = normalize_logits(logits1, 1)
    predicted_label, confidence = predict_from_logits(logits1)
    if confidence >= threshold:
        return 1, predicted_label, confidence, entropy_from_logits(logits1)

    stage2 = sessions[1]
    hidden2, logits2 = stage2.run(None, {stage2.get_inputs()[0].name: hidden1})
    logits2 = normalize_logits(logits2, 2)
    predicted_label, confidence = predict_from_logits(logits2)
    if confidence >= threshold:
        return 2, predicted_label, confidence, entropy_from_logits(logits2)

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
        fh.write("Raspberry Pi ONNX inference summary (AP 9-feature)\n")
        fh.write(f"model: {model_path}\n")
        fh.write(f"data: {data_path}\n")
        for key, value in summary.items():
            if isinstance(value, float):
                fh.write(f"{key}: {value:.6f}\n")
            else:
                fh.write(f"{key}: {value}\n")


def main() -> None:
    root = resolve_repo_root()
    parser = argparse.ArgumentParser(description="Measure ONNX inference latency on Raspberry Pi (AP 9-feature)")
    parser.add_argument("--model", default=str(default_model_path(root)), help="Path to a full-graph ONNX model")
    parser.add_argument(
        "--mode",
        choices=["full", "staged", "staged-confidence", "baseline"],
        default="full",
        help=(
            "full: single confidence-threshold ONNX graph. "
            "staged: stage1/2/3 graphs with entropy thresholds (Proposed Fixed/Dynamic). "
            "staged-confidence: stage1/2/3 graphs with confidence threshold (SDN-style). "
            "baseline: single non-branching Baseline LSTM graph."
        ),
    )
    parser.add_argument("--stage-prefix", default="ap_early_exit_fixed", help="Filename prefix for stage ONNX files")
    parser.add_argument("--stage1", type=Path, help="Path to stage1 ONNX model")
    parser.add_argument("--stage2", type=Path, help="Path to stage2 ONNX model")
    parser.add_argument("--stage3", type=Path, help="Path to stage3 ONNX model")
    parser.add_argument("--dynamic-theta", action="store_true", help="Use dynamic entropy thresholds in staged mode")
    parser.add_argument("--theta-1", type=float, default=DEFAULT_THETA_1, help="Exit 1 entropy threshold")
    parser.add_argument("--theta-2", type=float, default=DEFAULT_THETA_2, help="Exit 2 entropy threshold")
    parser.add_argument(
        "--dynamic-min-threshold", type=float, default=DEFAULT_DYNAMIC_MIN_THRESHOLD, help="Minimum dynamic theta value"
    )
    parser.add_argument(
        "--dynamic-recent-steps", type=int, default=DEFAULT_DYNAMIC_RECENT_STEPS, help="Recent occupancy steps used by dynamic theta"
    )
    parser.add_argument(
        "--dynamic-spike-threshold",
        type=float,
        default=DEFAULT_DYNAMIC_SPIKE_THRESHOLD,
        help="Occupancy delta threshold used by dynamic theta",
    )
    parser.add_argument("--data", default=str(default_data_path(root)), help="Path to windowed test CSV")
    parser.add_argument("--output", default=str(default_output_path(root)), help="CSV path for measured latency results")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for confidence-based early exit")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum samples to measure")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup inference count excluded from timing")
    parser.add_argument("--repeats", type=int, default=5, help="Repeated measurements per sample")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staged_modes = {"staged", "staged-confidence"}
    stage_sessions: list[ort.InferenceSession] = []
    session: ort.InferenceSession | None = None
    input_name = ""
    output_names: list[str] = []
    stage_paths: list[Path] = []

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    if args.mode in staged_modes:
        if not args.stage1:
            args.stage1 = default_stage_path(root, 1, args.stage_prefix)
        if not args.stage2:
            args.stage2 = default_stage_path(root, 2, args.stage_prefix)
        if not args.stage3:
            args.stage3 = default_stage_path(root, 3, args.stage_prefix)
        stage_paths = [Path(args.stage1), Path(args.stage2), Path(args.stage3)]
        for stage_path in stage_paths:
            if not stage_path.exists():
                raise FileNotFoundError(f"Stage ONNX model not found: {stage_path}")
        stage_sessions = [
            ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
            for path in stage_paths
        ]
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {data_path}")
        session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

    if not data_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {data_path}")

    samples, meta_rows = load_samples(data_path, args.max_samples)

    def run_once(sample: np.ndarray, batch: np.ndarray) -> tuple[int, int, float, float]:
        if args.mode == "staged":
            return run_staged_inference(
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
        if args.mode == "staged-confidence":
            return run_staged_confidence_inference(stage_sessions, batch, args.threshold)
        if args.mode == "baseline":
            assert session is not None
            return run_baseline_inference(session, output_names, input_name, batch)
        assert session is not None
        return run_full_inference(session, output_names, input_name, batch, args.threshold)

    warmup_sample = samples[:1]
    for _ in range(args.warmup):
        run_once(warmup_sample[0], warmup_sample)

    result_rows: list[dict] = []
    avg_latencies: list[float] = []
    exit_points: list[int] = []

    for sample, meta in zip(samples, meta_rows):
        batch = sample.reshape(1, 10, len(FEATURE_COLUMNS)).astype(np.float32)
        sample_latencies: list[float] = []
        exit_point = 0
        predicted_label = 0
        confidence = 0.0
        entropy = 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            exit_point, predicted_label, confidence, entropy = run_once(sample, batch)
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
                    statistics.pstdev(sample_latencies) if len(sample_latencies) > 1 else 0.0, 6
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
    write_summary(output_path, summary, model_path if args.mode not in staged_modes else stage_paths[0], data_path)

    print("Raspberry Pi ONNX inference measurement complete (AP 9-feature)")
    print(f"mode: {args.mode}")
    if args.mode in staged_modes:
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
