"""Build a run-level-split windowed dataset from a multi-session relabeled pool (2026-09-22).

Why: the canonical dataset (prepare_ap_metrics_dataset.py) splits *windows* at random, which leaks
neighbouring windows across train/val/test and only covers the 8/28 + 9/1 sessions. This script builds
train/val from many sessions and splits by whole *runs*: each session's last run becomes validation,
everything else is training. There is no held-out test session here -- the deployment model is meant to
use every session, and its expected performance is quoted from leave-one-session-out (loso.py), not from
this val split. `test.csv` is therefore a copy of `val.csv`, kept only so the ONNX export / Pi benchmark
tooling (which reads test.csv for input shapes and equality checks) keeps working.

Scaler: the canonical min-max scaler (ap_metrics_v2_redesign2/scaler_params.json) is reused, with
clipping to [0, 1], so the live-inference scaler on the Pi does not change.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.ap_features import AP_FEATURE_COLUMNS, WINDOW_SIZE  # noqa: E402
import prepare_ap_metrics_dataset as P  # noqa: E402

DEFAULT_SCALER = PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2" / "scaler_params.json"


def session_of(scenario: str) -> str:
    match = re.search(r"_(g\d)$", scenario)
    if match:
        return match.group(1)
    return "0828" if scenario.startswith("step_run") else "0901"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="relabeled pool CSV (scenario, label, timestamp, feature columns)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="scenario names to drop (contaminated runs)")
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out_dir} is not empty; pass --overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = P.load_raw(args.pool)
    raw = raw[~raw["scenario"].isin(args.exclude)].copy()
    df = P.drop_scenario_cold_start(raw)
    df, _ = P.fix_occupancy_outliers(df)
    df = df.ffill().bfill().fillna(0)

    scaler = json.loads(args.scaler.read_text(encoding="utf-8"))
    for column in AP_FEATURE_COLUMNS:
        lo, hi = scaler[column]["min"], scaler[column]["max"]
        df[column] = 0.0 if np.isclose(hi, lo) else ((df[column] - lo) / (hi - lo)).clip(0.0, 1.0)

    df["session"] = df["scenario"].map(session_of)
    starts = pd.to_datetime(df["timestamp"]).groupby(df["scenario"]).min()
    val_runs = []
    for session, group in df.groupby("session"):
        runs = starts[group["scenario"].unique()].sort_values()
        val_runs.append(runs.index[-1])
    val_mask = df["scenario"].isin(val_runs)

    windowed_train = P.apply_sliding_window(df[~val_mask], AP_FEATURE_COLUMNS, WINDOW_SIZE, 1)
    windowed_val = P.apply_sliding_window(df[val_mask], AP_FEATURE_COLUMNS, WINDOW_SIZE, 1)
    windowed_train.to_csv(args.out_dir / "train.csv", index=False)
    windowed_val.to_csv(args.out_dir / "val.csv", index=False)
    windowed_val.to_csv(args.out_dir / "test.csv", index=False)  # placeholder, see module docstring
    (args.out_dir / "scaler_params.json").write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")

    def dist(w: pd.DataFrame) -> dict[str, int]:
        first = w.groupby("sample_id").label.first()
        return {str(k): int(v) for k, v in first.value_counts().sort_index().items()}

    summary = {
        "note": "run-level split, multi-session pool; test.csv is a copy of val.csv (no held-out session)",
        "pool": str(args.pool),
        "excluded_scenarios": args.exclude,
        "sessions": sorted(df["session"].unique().tolist()),
        "val_runs": sorted(val_runs),
        "window_size": WINDOW_SIZE,
        "features": list(AP_FEATURE_COLUMNS),
        "train_windows": int(windowed_train["sample_id"].nunique()),
        "val_windows": int(windowed_val["sample_id"].nunique()),
        "train_label_distribution": dist(windowed_train),
        "val_label_distribution": dist(windowed_val),
        "scaler": str(args.scaler),
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
