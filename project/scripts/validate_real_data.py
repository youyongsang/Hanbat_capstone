"""Validate Stage 1 real traffic CSV files and DataLoader output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloader import get_dataloader, validate_csv_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real traffic dataset.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "yena" / "yena_stage2_validation_report.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = ["Yena Stage 2 Data Validation Report"]

    for split in ("train", "val", "test"):
        csv_path = args.data_dir / f"{split}.csv"
        summary = validate_csv_dataset(csv_path, window_size=args.window_size)
        loader = get_dataloader(
            csv_path,
            batch_size=args.batch_size,
            shuffle=False,
            window_size=args.window_size,
        )
        x_batch, y_batch = next(iter(loader))

        lines.extend(
            [
                "",
                f"[{split}]",
                f"Path: {summary['path']}",
                f"Columns: {summary['columns']}",
                f"Rows: {summary['rows']}",
                f"Samples: {summary['samples']}",
                f"Window Shape: {summary['shape']}",
                f"Label Counts: {summary['label_counts']}",
                f"Missing Values: {summary['missing_values']}",
                f"Feature Min: {summary['feature_min']}",
                f"Feature Max: {summary['feature_max']}",
                f"DataLoader X Batch Shape: {tuple(x_batch.shape)}",
                f"DataLoader y Batch Shape: {tuple(y_batch.shape)}",
            ]
        )

    report = "\n".join(lines)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
