"""Evaluate fixed and dynamic Early Exit LSTM inference modes."""

from __future__ import annotations

import argparse
import csv
import time
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_exit_lstm import EarlyExitLSTM  # noqa: E402
from utils.dataloader import get_dataloader  # noqa: E402
from utils.metrics import ExitStats, format_percent  # noqa: E402


LABEL_NAMES = ["정상", "혼잡 경고", "혼잡", "심각"]
SIMULATED_EXIT_TIME_MS = {1: 2.0, 2: 4.0, 3: 8.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Early Exit LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "early_exit_lstm_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "yongsang" / "early_exit_eval_report.txt",
        help="Text report path for evaluation results.",
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


def evaluate_mode(
    model: EarlyExitLSTM,
    dataloader: torch.utils.data.DataLoader,
    dynamic: bool,
    scenarios: list[str],
) -> dict[str, object]:
    exit_stats = {1: ExitStats(), 2: ExitStats(), 3: ExitStats()}
    label_correct = {label: 0 for label in range(4)}
    label_total = {label: 0 for label in range(4)}
    scenario_correct: dict[str, int] = defaultdict(int)
    scenario_total: dict[str, int] = defaultdict(int)
    total_correct = 0
    total_samples = 0
    total_wall_time = 0.0
    simulated_time = 0.0
    scenario_idx = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            start = time.perf_counter()
            decisions = model.infer_batch(x_batch, dynamic=dynamic)
            total_wall_time += time.perf_counter() - start

            for idx, decision in enumerate(decisions):
                target = int(y_batch[idx].item())
                prediction = int(decision.logits.argmax(dim=-1).item())
                is_correct = prediction == target
                scenario = scenarios[scenario_idx] if scenario_idx < len(scenarios) else "unknown"
                scenario_idx += 1

                exit_stats[decision.exit_point].add(is_correct)
                label_correct[target] += int(is_correct)
                label_total[target] += 1
                scenario_correct[scenario] += int(is_correct)
                scenario_total[scenario] += 1
                total_correct += int(is_correct)
                total_samples += 1
                simulated_time += SIMULATED_EXIT_TIME_MS[decision.exit_point]

    return {
        "accuracy": total_correct / total_samples,
        "exit_stats": exit_stats,
        "label_accuracy": {
            label: (label_correct[label] / label_total[label] if label_total[label] else 0.0)
            for label in range(4)
        },
        "scenario_accuracy": {
            scenario: scenario_correct[scenario] / scenario_total[scenario]
            for scenario in sorted(scenario_total)
        },
        "avg_wall_time_ms": (total_wall_time / total_samples) * 1000,
        "avg_simulated_time_ms": simulated_time / total_samples,
    }


def append_report(lines: list[str], title: str, metrics: dict[str, object]) -> None:
    lines.extend(["", title, f"Test Accuracy: {format_percent(metrics['accuracy'])}", "Exit별 성능:"])
    exit_stats = metrics["exit_stats"]
    total_samples = sum(stats.total for stats in exit_stats.values())
    for exit_point in (1, 2, 3):
        stats = exit_stats[exit_point]
        exit_rate = stats.total / total_samples if total_samples else 0.0
        lines.append(
            f"  Exit {exit_point} | Accuracy: {format_percent(stats.accuracy)} | "
            f"Exit Rate: {format_percent(exit_rate)} | Avg Time: {SIMULATED_EXIT_TIME_MS[exit_point]:.1f}ms"
        )
    lines.append(f"Overall Avg Inference Time: {metrics['avg_simulated_time_ms']:.3f}ms")
    lines.append(f"Measured Wall Time: {metrics['avg_wall_time_ms']:.3f}ms")
    lines.append("Label별 정확도:")
    for label, accuracy in metrics["label_accuracy"].items():
        lines.append(f"  Label {label} ({LABEL_NAMES[label]}): {format_percent(accuracy)}")
    lines.append("시나리오별 정확도:")
    for scenario, accuracy in metrics["scenario_accuracy"].items():
        lines.append(f"  {scenario}: {format_percent(accuracy)}")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = EarlyExitLSTM(
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        theta_1=float(checkpoint.get("theta_1", 0.3)),
        theta_2=float(checkpoint.get("theta_2", 0.6)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_csv = args.data_dir / "test.csv"
    dataloader = get_dataloader(test_csv, args.batch_size, shuffle=False)
    scenarios = load_scenarios(test_csv)
    fixed_metrics = evaluate_mode(model, dataloader, dynamic=False, scenarios=scenarios)
    dynamic_metrics = evaluate_mode(model, dataloader, dynamic=True, scenarios=scenarios)

    lines = [
        "Early Exit LSTM Stage 2 Evaluation Report",
        f"Data Directory: {args.data_dir}",
        f"Checkpoint: {args.checkpoint}",
    ]
    append_report(lines, "=== Baseline 3: Early Exit + fixed theta ===", fixed_metrics)
    append_report(lines, "=== Proposed Model 4: Early Exit + dynamic theta ===", dynamic_metrics)
    lines.extend(
        [
            "",
            "동적 theta 효과:",
            (
                f"  Accuracy Change: "
                f"{format_percent(dynamic_metrics['accuracy'] - fixed_metrics['accuracy'])}"
            ),
            (
                f"  Avg Inference Time Change: "
                f"{dynamic_metrics['avg_simulated_time_ms'] - fixed_metrics['avg_simulated_time_ms']:.3f}ms"
            ),
        ]
    )

    report = "\n".join(lines)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
