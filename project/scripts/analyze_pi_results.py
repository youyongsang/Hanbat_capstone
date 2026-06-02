"""Analyze Raspberry Pi ONNX inference result files.

The input CSV is produced by project/scripts/inference_pi.py. This script
summarizes accuracy, latency, exit distribution, and scenario-level behavior.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "hojung" / "pi_inference_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hojung"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Raspberry Pi inference result CSV.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="pi_inference_results.csv path")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--name",
        default="pi_inference_analysis",
        help="Output file basename without extension.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"sample_id", "true_label", "predicted_label", "exit_point", "confidence", "inference_ms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def rate(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), 4)


def percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def latency_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "samples": int(len(df)),
        "accuracy": rate((df["true_label"] == df["predicted_label"]).mean()),
        "avg_inference_ms": round(float(df["inference_ms"].mean()), 6),
        "min_inference_ms": round(float(df["inference_ms"].min()), 6),
        "max_inference_ms": round(float(df["inference_ms"].max()), 6),
        "p50_inference_ms": round(float(df["inference_ms"].median()), 6),
        "p95_inference_ms": round(float(np.percentile(df["inference_ms"], 95)), 6),
        "avg_confidence": round(float(df["confidence"].mean()), 6),
    }


def summarize_by_exit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exit_point, group in df.groupby("exit_point", sort=True):
        rows.append(
            {
                "exit_point": int(exit_point),
                "samples": int(len(group)),
                "rate": rate(len(group) / len(df)),
                "accuracy": rate((group["true_label"] == group["predicted_label"]).mean()),
                "avg_inference_ms": round(float(group["inference_ms"].mean()), 6),
                "p95_inference_ms": round(float(np.percentile(group["inference_ms"], 95)), 6),
                "avg_confidence": round(float(group["confidence"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario" not in df.columns:
        return pd.DataFrame()

    rows = []
    for scenario, group in df.groupby("scenario", sort=True):
        row = {
            "scenario": str(scenario),
            "samples": int(len(group)),
            "accuracy": rate((group["true_label"] == group["predicted_label"]).mean()),
            "avg_inference_ms": round(float(group["inference_ms"].mean()), 6),
            "p95_inference_ms": round(float(np.percentile(group["inference_ms"], 95)), 6),
        }
        for exit_point in (1, 2, 3):
            row[f"exit{exit_point}_rate"] = rate((group["exit_point"] == exit_point).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_label(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("true_label", sort=True):
        rows.append(
            {
                "true_label": int(label),
                "samples": int(len(group)),
                "accuracy": rate((group["true_label"] == group["predicted_label"]).mean()),
                "avg_inference_ms": round(float(group["inference_ms"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    input_path: Path,
    overall: dict[str, float],
    exit_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> None:
    lines = [
        "# Raspberry Pi Inference Analysis",
        "",
        f"- input: `{input_path}`",
        f"- samples: {overall['samples']}",
        f"- accuracy: {percent(overall['accuracy'])}",
        f"- avg inference: {overall['avg_inference_ms']:.6f} ms",
        f"- p95 inference: {overall['p95_inference_ms']:.6f} ms",
        f"- avg confidence: {overall['avg_confidence']:.6f}",
        "",
        "## Exit Summary",
        "",
        "| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in exit_df.to_dict("records"):
        lines.append(
            "| {exit_point} | {samples} | {rate} | {accuracy} | {avg:.6f} | {p95:.6f} | {conf:.6f} |".format(
                exit_point=row["exit_point"],
                samples=row["samples"],
                rate=percent(row["rate"]),
                accuracy=percent(row["accuracy"]),
                avg=row["avg_inference_ms"],
                p95=row["p95_inference_ms"],
                conf=row["avg_confidence"],
            )
        )

    if not scenario_df.empty:
        lines.extend(
            [
                "",
                "## Scenario Summary",
                "",
                "| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in scenario_df.to_dict("records"):
            lines.append(
                "| {scenario} | {samples} | {accuracy} | {avg:.6f} | {p95:.6f} | {e1} | {e2} | {e3} |".format(
                    scenario=row["scenario"],
                    samples=row["samples"],
                    accuracy=percent(row["accuracy"]),
                    avg=row["avg_inference_ms"],
                    p95=row["p95_inference_ms"],
                    e1=percent(row["exit1_rate"]),
                    e2=percent(row["exit2_rate"]),
                    e3=percent(row["exit3_rate"]),
                )
            )

    lines.extend(
        [
            "",
            "## Label Summary",
            "",
            "| True label | Samples | Accuracy | Avg ms |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in label_df.to_dict("records"):
        lines.append(
            "| {label} | {samples} | {accuracy} | {avg:.6f} |".format(
                label=row["true_label"],
                samples=row["samples"],
                accuracy=percent(row["accuracy"]),
                avg=row["avg_inference_ms"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text(path: Path, overall: dict[str, float], exit_df: pd.DataFrame) -> None:
    lines = [
        "Raspberry Pi inference analysis",
        f"samples: {overall['samples']}",
        f"accuracy: {percent(overall['accuracy'])}",
        f"avg_inference_ms: {overall['avg_inference_ms']:.6f}",
        f"p50_inference_ms: {overall['p50_inference_ms']:.6f}",
        f"p95_inference_ms: {overall['p95_inference_ms']:.6f}",
        f"min_inference_ms: {overall['min_inference_ms']:.6f}",
        f"max_inference_ms: {overall['max_inference_ms']:.6f}",
        f"avg_confidence: {overall['avg_confidence']:.6f}",
    ]
    for row in exit_df.to_dict("records"):
        lines.append(
            "exit{exit_point}: samples={samples}, rate={rate}, accuracy={accuracy}, avg_ms={avg:.6f}".format(
                exit_point=row["exit_point"],
                samples=row["samples"],
                rate=percent(row["rate"]),
                accuracy=percent(row["accuracy"]),
                avg=row["avg_inference_ms"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    require_columns(df, args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall = latency_summary(df)
    exit_df = summarize_by_exit(df)
    scenario_df = summarize_by_scenario(df)
    label_df = summarize_by_label(df)

    base = args.output_dir / args.name
    write_csv(base.with_name(f"{args.name}_overall.csv"), [overall])
    exit_df.to_csv(base.with_name(f"{args.name}_by_exit.csv"), index=False)
    if not scenario_df.empty:
        scenario_df.to_csv(base.with_name(f"{args.name}_by_scenario.csv"), index=False)
    label_df.to_csv(base.with_name(f"{args.name}_by_label.csv"), index=False)
    write_text(base.with_suffix(".txt"), overall, exit_df)
    write_markdown(base.with_suffix(".md"), args.input, overall, exit_df, scenario_df, label_df)

    print("Pi inference analysis complete")
    print(f"input: {args.input}")
    print(f"overall: {base.with_name(f'{args.name}_overall.csv')}")
    print(f"by_exit: {base.with_name(f'{args.name}_by_exit.csv')}")
    if not scenario_df.empty:
        print(f"by_scenario: {base.with_name(f'{args.name}_by_scenario.csv')}")
    print(f"by_label: {base.with_name(f'{args.name}_by_label.csv')}")
    print(f"txt: {base.with_suffix('.txt')}")
    print(f"markdown: {base.with_suffix('.md')}")
    print(f"accuracy: {percent(overall['accuracy'])}")
    print(f"avg_inference_ms: {overall['avg_inference_ms']:.6f}")


if __name__ == "__main__":
    main()
