"""Evaluate standard Baseline LSTM inference performance.

Matches the infrastructure, scenario tracking, and log reporting formats
defined in Yongsang's early exit evaluation script for a fair ablation study.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_lstm import BaselineLSTM
from utils.dataloader import get_dataloader
from utils.metrics import format_percent

LABEL_NAMES = ["정상", "혼잡 경고", "혼잡", "심각"]
BASELINE_SIMULATED_TIME_MS = 8.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Baseline LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_lstm_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_eval_report.txt",
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

def evaluate_baseline(
    model: BaselineLSTM,
    dataloader: torch.utils.data.DataLoader,
    scenarios: list[str],
    device: torch.device,
) -> dict[str, object]:
    label_correct = {label: 0 for label in range(4)}
    label_total = {label: 0 for range(4)}
    scenario_correct: dict[str, int] = defaultdict(int)
    scenario_total: dict[str, int] = defaultdict(int)
    total_correct = 0
    total_samples = 0
    total_wall_time = 0.0
    scenario_idx = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            
            start = time.perf_counter()
            outputs = model(x_batch)
            total_wall_time += time.perf_counter() - start

            predictions = outputs.argmax(dim=-1).cpu()

            for idx in range(y_batch.size(0)):
                target = int(y_batch[idx].item())
                prediction = int(predictions[idx].item())
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
        "avg_simulated_time_ms": BASELINE_SIMULATED_TIME_MS,
    }

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} 가중치 파일이 없습니다. train.py를 먼저 실행해 주세요.")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BaselineLSTM(hidden_size=int(checkpoint.get("hidden_size", 128))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_csv = args.data_dir / "test.csv"
    dataloader = get_dataloader(test_csv, args.batch_size, shuffle=False)
    scenarios = load_scenarios(test_csv)
    
    metrics = evaluate_baseline(model, dataloader, scenarios, device)

    lines = [
        "Baseline LSTM Stage 1 Evaluation Report (For Ablation Sync)",
        f"Data Directory: {args.data_dir}",
        f"Checkpoint: {args.checkpoint}",
        "",
        "=== Baseline 2: Standard 3-Layer LSTM ===",
        f"Test Accuracy: {format_percent(metrics['accuracy'])}",
        "Exit별 성능:",
        f"  Exit 1 | Accuracy: N/A | Exit Rate: 0.0% | Avg Time: 2.0ms",
        f"  Exit 2 | Accuracy: N/A | Exit Rate: 0.0% | Avg Time: 4.0ms",
        f"  Exit 3 | Accuracy: {format_percent(metrics['accuracy'])} | Exit Rate: 100.0% | Avg Time: 8.0ms",
        f"Overall Avg Inference Time: {metrics['avg_simulated_time_ms']:.3f}ms",
        f"Measured Wall Time: {metrics['avg_wall_time_ms']:.3f}ms",
        "Label별 정확도:",
    ]
    
    for label, acc in metrics["label_accuracy"].items():
        lines.append(f"  Label {label} ({LABEL_NAMES[label]}): {format_percent(acc)}")
        
    lines.append("시나리오별 정확도:")
    for scenario, acc in metrics["scenario_accuracy"].items():
        lines.append(f"  {scenario}: {format_percent(acc)}")

    report = "\n".join(lines)
    print(report)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved: {args.output}")

if __name__ == "__main__":
    main()
