"""Evaluate Early Exit LSTM checkpoint with exit rates."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_exit_lstm import EarlyExitLSTM  # noqa: E402
from utils.dataloader import get_dataloader  # noqa: E402
from utils.metrics import ExitStats, format_percent  # noqa: E402


LABEL_NAMES = ["정상", "혼잡 경고", "혼잡", "심각"]


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
        default=PROJECT_ROOT / "results" / "early_exit_eval_report.txt",
        help="Text report path for evaluation results.",
    )
    return parser.parse_args()


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

    dataloader = get_dataloader(args.data_dir / "test.csv", args.batch_size, shuffle=False)
    exit_stats = {1: ExitStats(), 2: ExitStats(), 3: ExitStats()}
    label_correct = {label: 0 for label in range(4)}
    label_total = {label: 0 for label in range(4)}
    total_correct = 0
    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            start = time.perf_counter()
            decisions = model.infer_batch(x_batch)
            total_time += time.perf_counter() - start

            for idx, decision in enumerate(decisions):
                target = int(y_batch[idx].item())
                prediction = int(decision.logits.argmax(dim=-1).item())
                is_correct = prediction == target
                exit_stats[decision.exit_point].add(is_correct)
                label_correct[target] += int(is_correct)
                label_total[target] += 1
                total_correct += int(is_correct)
                total_samples += 1

    lines = [
        "Early Exit LSTM Evaluation Report",
        f"Data Directory: {args.data_dir}",
        f"Checkpoint: {args.checkpoint}",
        f"Test Accuracy: {format_percent(total_correct / total_samples)}",
    ]
    for exit_point in (1, 2, 3):
        stats = exit_stats[exit_point]
        exit_rate = stats.total / total_samples
        lines.append(
            f"Exit {exit_point} Accuracy: {format_percent(stats.accuracy)} | "
            f"Exit Rate: {format_percent(exit_rate)}"
        )
    for label in range(4):
        accuracy = label_correct[label] / label_total[label] if label_total[label] else 0.0
        lines.append(f"Label {label} ({LABEL_NAMES[label]}) Accuracy: {format_percent(accuracy)}")
    lines.append(f"Average Inference Time: {(total_time / total_samples) * 1000:.3f}ms")

    report = "\n".join(lines)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
