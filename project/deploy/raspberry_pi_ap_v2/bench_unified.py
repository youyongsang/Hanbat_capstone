"""Benchmark the single-graph (unified, ONNX If-node) Early Exit model on Pi.

Same methodology as inference_pi_ap.py (5 repeats/sample, 10 warmup, perf_counter),
but calls one ort.InferenceSession.run() per sample instead of up to 3 staged calls.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from collections import Counter

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
    "sta_tx_bitrate_mean",
]
WINDOW_SIZE = 12  # 2026-09-01: 10 -> 12 (matches ap_features.WINDOW_SIZE + [1,12,7] ONNX)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(here / "ap_early_exit_fixed_unified.onnx"))
    p.add_argument("--data", default=str(here / "test.csv"))
    p.add_argument("--max-samples", type=int, default=310)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    samples = []
    for sample_id, g in df.groupby("sample_id", sort=True):
        g = g.sort_values("timestep")
        if len(g) != WINDOW_SIZE:
            continue
        samples.append(g[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        if len(samples) >= args.max_samples:
            break

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    warm = samples[0][None, :, :]
    for _ in range(args.warmup):
        sess.run(None, {input_name: warm})

    latencies = []
    exits = []
    for s in samples:
        x = s[None, :, :]
        per_sample = []
        exit_point = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            _, exit_point = sess.run(None, {input_name: x})
            per_sample.append((time.perf_counter() - start) * 1000.0)
        latencies.append(statistics.mean(per_sample))
        exits.append(int(exit_point))

    counts = Counter(exits)
    total = len(exits)
    print(f"model: {args.model}")
    print(f"sample_count: {total}")
    print(f"avg_inference_ms: {statistics.mean(latencies):.6f}")
    print(f"std_inference_ms: {statistics.pstdev(latencies):.6f}")
    print(f"min_inference_ms: {min(latencies):.6f}")
    print(f"max_inference_ms: {max(latencies):.6f}")
    print(f"p50_inference_ms: {statistics.median(latencies):.6f}")
    print(f"p95_inference_ms: {float(np.percentile(latencies, 95)):.6f}")
    for e in (1, 2, 3):
        print(f"exit{e}_rate: {counts.get(e, 0) / total:.2%}  (n={counts.get(e, 0)})")

    # per-exit latency breakdown
    by_exit: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for lat, ex in zip(latencies, exits):
        by_exit[ex].append(lat)
    for e in (1, 2, 3):
        if by_exit[e]:
            print(f"exit{e}_avg_ms: {statistics.mean(by_exit[e]):.6f}")


if __name__ == "__main__":
    main()
