"""Generate the real traffic dataset for stage 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.traffic_simulator import save_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate factory traffic CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real",
        help="Directory for train.csv, val.csv, test.csv, and scaler_params.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = save_dataset(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
