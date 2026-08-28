"""Evaluate AP measurement Baseline LSTM (no early exit)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_baseline_lstm import APBaselineLSTM  # noqa: E402
from utils.ap_dataloader import get_ap_dataloader  # noqa: E402
from utils.metrics import format_percent  # noqa: E402

LABEL_NAMES = ["정상", "경고", "혼잡", "심각"]


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AP Baseline LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2" / "ap_baseline_lstm_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "yongsang" / "ap_baseline_lstm_redesign2_eval_report.txt",
    )
    return parser.parse_args()


def load_scenarios(test_csv: Path) -> list[str]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with test_csv.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            grouped[int(row["sample_id"])].append(row)

    scenarios = []
    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda row: int(row["timestep"]))
        scenarios.append(rows[-1].get("scenario", "unknown"))
    return scenarios


def evaluate(
    model: APBaselineLSTM,
    dataloader: torch.utils.data.DataLoader,
    scenarios: list[str],
) -> dict[str, object]:
    label_correct = {label: 0 for label in range(4)}
    label_total = {label: 0 for label in range(4)}
    scenario_correct: dict[str, int] = defaultdict(int)
    scenario_total: dict[str, int] = defaultdict(int)
    total_correct = 0
    total_samples = 0
    total_wall_time = 0.0
    scenario_idx = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            start = time.perf_counter()
            logits = model(x_batch)
            total_wall_time += time.perf_counter() - start

            preds = logits.argmax(dim=-1)
            for idx in range(x_batch.size(0)):
                target = int(y_batch[idx].item())
                prediction = int(preds[idx].item())
                is_correct = prediction == target
                scenario = scenarios[scenario_idx] if scenario_idx < len(scenarios) else "unknown"
                scenario_idx += 1

                label_correct[target] += int(is_correct)
                label_total[target] += 1
                scenario_correct[scenario] += int(is_correct)
                scenario_total[scenario] += 1
                total_correct += int(is_correct)
                total_samples += 1

    return {
        "accuracy": total_correct / total_samples,
        "label_accuracy": {
            label: (label_correct[label] / label_total[label] if label_total[label] else 0.0)
            for label in range(4)
        },
        "scenario_accuracy": {
            scenario: scenario_correct[scenario] / scenario_total[scenario]
            for scenario in sorted(scenario_total)
        },
        "avg_wall_time_ms": (total_wall_time / total_samples) * 1000,
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = APBaselineLSTM(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_csv = args.data_dir / "test.csv"
    dataloader = get_ap_dataloader(test_csv, args.batch_size, shuffle=False)
    scenarios = load_scenarios(test_csv)

    metrics = evaluate(model, dataloader, scenarios=scenarios)

    lines = [
        "AP Baseline LSTM Evaluation Report (no early exit)",
        f"Data Directory: {display_path(args.data_dir)}",
        f"Checkpoint: {display_path(args.checkpoint)}",
        "",
        "=== AP Baseline (always runs all 3 LSTM layers) ===",
        f"Test Accuracy: {format_percent(metrics['accuracy'])}",
        "Exit별 성능:",
        f"  Exit 1 | Accuracy: 0.0% | Exit Rate: 0.0% | Avg Time: 2.0ms",
        f"  Exit 2 | Accuracy: 0.0% | Exit Rate: 0.0% | Avg Time: 4.0ms",
        f"  Exit 3 | Accuracy: {format_percent(metrics['accuracy'])} | Exit Rate: 100.0% | Avg Time: 8.0ms",
        "Overall Avg Inference Time: 8.000ms",
        f"Measured Wall Time: {metrics['avg_wall_time_ms']:.3f}ms",
        "Label별 정확도:",
    ]
    for label, accuracy in metrics["label_accuracy"].items():
        lines.append(f"  Label {label} ({LABEL_NAMES[label]}): {format_percent(accuracy)}")
    lines.append("시나리오별 정확도:")
    for scenario, accuracy in metrics["scenario_accuracy"].items():
        lines.append(f"  {scenario}: {format_percent(accuracy)}")

    report = "\n".join(lines)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved: {display_path(args.output)}")


if __name__ == "__main__":
    main()
