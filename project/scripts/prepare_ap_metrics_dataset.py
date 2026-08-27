"""Prepare AP measurement metrics for the AP Early Exit LSTM.

Input:
    metrics_v2_relabel.csv from the AP measurement collector/relabel step.

Output:
    train.csv, val.csv, test.csv in pre-windowed project format, but with
    AP measurement feature columns instead of the first-semester 4-feature
    simulator columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402


WINDOW_SIZE = 10
STRIDE = 1
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_NAMES = {0: "정상", 1: "경고", 2: "혼잡", 3: "심각"}


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AP measurement feature dataset.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="metrics_v2_relabel.csv path")
    parser.add_argument("--out-dir", "-o", type=Path, default=PROJECT_ROOT / "data" / "ap_metrics")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--keep-first-sample",
        action="store_true",
        help="Do not drop the first row of each scenario. Defaults to dropping it because delta features are cold-started.",
    )
    parser.add_argument(
        "--no-occupancy-outlier-fix",
        action="store_true",
        help="Disable conservative smoothing for delta occupancy spikes at 0 or 100.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV files.")
    return parser.parse_args()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in AP_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"AP feature columns missing from {path}: {missing}")
    for required in ("scenario", "label"):
        if required not in df.columns:
            raise KeyError(f"required column missing from {path}: {required}")
    return df


def drop_scenario_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    groups = [group.iloc[1:] for _, group in df.groupby("scenario", sort=False)]
    if not groups:
        return df.copy()
    return pd.concat(groups, ignore_index=True)


def fix_occupancy_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Smooth obvious survey-delta spikes without changing labels.

    The collector sometimes reports delta occupancy as 0 or 100 while the
    adjacent throughput/retry pattern does not support a real state jump. For
    modeling, replace only those edge values with a scenario-local rolling
    median so the LSTM learns traffic behavior instead of driver counter noise.
    """

    fixed = df.copy()
    column = "channel_occupancy_percent"
    method = fixed.get("channel_occupancy_method")
    if method is None:
        return fixed, 0

    artifact_mask = method.eq("delta") & fixed[column].isin([0.0, 100.0])
    if not artifact_mask.any():
        return fixed, 0

    replacement = (
        fixed.groupby("scenario")[column]
        .transform(lambda series: series.mask(series.isin([0.0, 100.0])).rolling(5, min_periods=1).median())
        .ffill()
        .bfill()
    )
    fixed.loc[artifact_mask, column] = replacement.loc[artifact_mask]
    return fixed, int(artifact_mask.sum())


def minmax_normalize(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    normalized = df.copy()
    scaler: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        col_min = float(normalized[column].min())
        col_max = float(normalized[column].max())
        scaler[column] = {"min": col_min, "max": col_max}
        if np.isclose(col_max, col_min):
            normalized[column] = 0.0
        else:
            normalized[column] = ((normalized[column] - col_min) / (col_max - col_min)).clip(0.0, 1.0)
    return normalized, scaler


def apply_sliding_window(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    rows = []
    sample_id = 0
    for scenario, group in df.groupby("scenario", sort=False):
        group = group.reset_index(drop=True)
        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            window = group.iloc[start:end]
            label = int(window.iloc[-1]["label"])
            for timestep in range(window_size):
                row = {
                    "sample_id": sample_id,
                    "timestep": timestep,
                }
                for column in feature_columns:
                    row[column] = float(window.iloc[timestep][column])
                row["label"] = label
                row["scenario"] = scenario
                rows.append(row)
            sample_id += 1
    return pd.DataFrame(rows)


def stratified_split(windowed: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = windowed.groupby("sample_id")["label"].last().reset_index()
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    for _, label_meta in meta.groupby("label", sort=True):
        ids = label_meta["sample_id"].to_numpy().copy()
        rng.shuffle(ids)
        total = len(ids)
        train_count = int(round(total * TRAIN_RATIO))
        val_count = int(round(total * VAL_RATIO))
        if total >= 3:
            train_count = max(1, min(train_count, total - 2))
            val_count = max(1, min(val_count, total - train_count - 1))
        else:
            train_count = max(1, total - 1)
            val_count = 0

        train_ids.extend(int(i) for i in ids[:train_count])
        val_ids.extend(int(i) for i in ids[train_count : train_count + val_count])
        test_ids.extend(int(i) for i in ids[train_count + val_count :])

    def select_and_reindex(ids: list[int]) -> pd.DataFrame:
        split = windowed[windowed["sample_id"].isin(ids)].copy()
        id_map = {old: new for new, old in enumerate(split["sample_id"].unique())}
        split["sample_id"] = split["sample_id"].map(id_map)
        return split.reset_index(drop=True)

    return select_and_reindex(train_ids), select_and_reindex(val_ids), select_and_reindex(test_ids)


def sample_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df.drop_duplicates("sample_id")["label"].value_counts().sort_index()
    return {f"{label}:{LABEL_NAMES.get(int(label), 'unknown')}": int(count) for label, count in counts.items()}


def write_outputs(
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: dict[str, dict[str, float]],
    source: dict[str, object],
    window_size: int,
    outlier_fixes: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    summary = {
        "source": source,
        "features": list(AP_FEATURE_COLUMNS),
        "window_size": window_size,
        "label_names": LABEL_NAMES,
        "scaler_params": scaler,
        "preprocessing": {
            "dropped_first_row_per_scenario": source["dropped_first_row_per_scenario"],
            "occupancy_outlier_fixes": outlier_fixes,
            "model_excluded_columns": [
                "timestamp",
                "scenario",
                "channel_occupancy_method",
                "packet_loss_udp_percent",
                "poll_interval_s",
                "connected_clients",
                "throughput_score",
                "occupancy_score",
                "retry_failed_score",
                "jitter_score",
                "congestion_score",
            ],
        },
        "splits": {
            name: {
                "rows": len(split),
                "samples": len(split) // window_size,
                "shape": f"({len(split) // window_size}, {window_size}, {len(AP_FEATURE_COLUMNS)})",
                "label_distribution": sample_label_distribution(split),
            }
            for name, split in (("train", train_df), ("val", val_df), ("test", test_df))
        },
    }

    (out_dir / "scaler_params.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "AP Metrics Dataset Conversion Report",
        f"Input: {source['input_file']}",
        f"Raw rows: {source['raw_rows']}",
        f"Prepared rows before windowing: {source['prepared_rows']}",
        f"Window size: {window_size}",
        f"Feature count: {len(AP_FEATURE_COLUMNS)}",
        f"Occupancy outlier fixes: {outlier_fixes}",
        "",
    ]
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        report_lines.append(f"[{split_name}] {len(split_df) // window_size} samples / {len(split_df)} rows")
        for label_name, count in sample_label_distribution(split_df).items():
            report_lines.append(f"  {label_name}: {count}")
    (out_dir / "conversion_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.glob("*.csv")) and not args.overwrite:
        raise SystemExit(f"{display_path(args.out_dir)} already has CSV files. Pass --overwrite to replace them.")

    raw = load_raw(args.input)
    prepared = raw.copy()
    dropped_first = not args.keep_first_sample
    if dropped_first:
        prepared = drop_scenario_cold_start(prepared)

    outlier_fixes = 0
    if not args.no_occupancy_outlier_fix:
        prepared, outlier_fixes = fix_occupancy_outliers(prepared)

    prepared = prepared.ffill().bfill().fillna(0)
    normalized, scaler = minmax_normalize(prepared, AP_FEATURE_COLUMNS)
    windowed = apply_sliding_window(
        normalized,
        AP_FEATURE_COLUMNS,
        window_size=args.window_size,
        stride=args.stride,
    )
    if windowed.empty:
        raise SystemExit("No windows were generated. Check input length and window size.")

    train_df, val_df, test_df = stratified_split(windowed, seed=args.seed)
    source = {
        "input_file": str(args.input),
        "raw_rows": int(len(raw)),
        "prepared_rows": int(len(prepared)),
        "dropped_first_row_per_scenario": dropped_first,
        "stride": args.stride,
    }
    write_outputs(
        args.out_dir,
        train_df,
        val_df,
        test_df,
        scaler,
        source,
        args.window_size,
        outlier_fixes,
    )

    print(f"AP metrics dataset saved: {display_path(args.out_dir)}")
    print(f"Features: {', '.join(AP_FEATURE_COLUMNS)}")
    print(f"Occupancy outlier fixes: {outlier_fixes}")


if __name__ == "__main__":
    main()
