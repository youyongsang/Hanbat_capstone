"""Fill Yongsang scenario-analysis template with Early Exit predictions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_exit_lstm import EarlyExitLSTM, compute_dynamic_threshold  # noqa: E402
from utils.dataloader import FEATURE_COLUMNS  # noqa: E402


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill scenario analysis predictions.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "real" / "test_with_scenario.csv")
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "results" / "yongsang" / "scenario_analysis_template.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "yongsang" / "scenario_analysis_filled.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "yongsang" / "scenario_analysis_summary.csv",
    )
    parser.add_argument(
        "--scenario-output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "scenario_analysis",
        help="Directory for scenario_{id}_analysis.csv files.",
    )
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "early_exit_lstm_best.pth")
    parser.add_argument("--model-info", type=Path, default=PROJECT_ROOT / "checkpoints" / "model_info.json")
    parser.add_argument("--fixed-final", type=Path, default=PROJECT_ROOT / "checkpoints" / "early_exit_fixed_final.pth")
    parser.add_argument("--dynamic-final", type=Path, default=PROJECT_ROOT / "checkpoints" / "early_exit_dynamic_final.pth")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_samples(data_path: Path) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(data_path):
        grouped[int(row["sample_id"])].append(row)

    samples = {}
    for sample_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["timestep"]))
        features = [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]
        samples[sample_id] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return samples


def load_model(checkpoint_path: Path) -> EarlyExitLSTM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EarlyExitLSTM(
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        theta_1=float(checkpoint.get("theta_1", 0.3)),
        theta_2=float(checkpoint.get("theta_2", 0.6)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows = []
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    groups["all"] = rows
    for row in rows:
        groups[str(row["scenario_id"])].append(row)

    for scenario_id, group in sorted(groups.items(), key=lambda item: item[0]):
        total = len(group)
        fixed_correct = sum(int(row["fixed_pred"]) == int(row["true_label"]) for row in group)
        dynamic_correct = sum(int(row["dynamic_pred"]) == int(row["true_label"]) for row in group)
        fixed_exit_counts = {exit_point: sum(int(row["fixed_exit_point"]) == exit_point for row in group) for exit_point in (1, 2, 3)}
        dynamic_exit_counts = {exit_point: sum(int(row["dynamic_exit_point"]) == exit_point for row in group) for exit_point in (1, 2, 3)}
        scenario_name = "all" if scenario_id == "all" else str(group[0]["scenario"])

        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario": scenario_name,
                "samples": total,
                "fixed_accuracy": round(accuracy(fixed_correct, total), 4),
                "dynamic_accuracy": round(accuracy(dynamic_correct, total), 4),
                "fixed_exit1_rate": round(accuracy(fixed_exit_counts[1], total), 4),
                "fixed_exit2_rate": round(accuracy(fixed_exit_counts[2], total), 4),
                "fixed_exit3_rate": round(accuracy(fixed_exit_counts[3], total), 4),
                "dynamic_exit1_rate": round(accuracy(dynamic_exit_counts[1], total), 4),
                "dynamic_exit2_rate": round(accuracy(dynamic_exit_counts[2], total), 4),
                "dynamic_exit3_rate": round(accuracy(dynamic_exit_counts[3], total), 4),
            }
        )
    return summary_rows


def write_scenario_files(
    output_dir: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["scenario_id"])].append(row)

    for scenario_id, group in sorted(grouped.items()):
        output_path = output_dir / f"scenario_{scenario_id}_analysis.csv"
        write_csv(output_path, group, fieldnames)


def update_model_info(path: Path, summary_rows: list[dict[str, object]], model: EarlyExitLSTM) -> None:
    info = json.loads(path.read_text(encoding="utf-8"))
    all_row = next(row for row in summary_rows if row["scenario_id"] == "all")
    info["fixed_theta_1"] = model.theta_1
    info["fixed_theta_2"] = model.theta_2
    info["dynamic_base_theta_1"] = model.theta_1
    info["dynamic_base_theta_2"] = model.theta_2
    info["dynamic_high_variance"] = 0.22
    info["dynamic_mid_variance"] = 0.12
    info["dynamic_min_threshold"] = 0.22
    info["dynamic_recent_steps"] = 5
    info["dynamic_spike_threshold"] = 0.25
    info["test_accuracy_fixed"] = all_row["fixed_accuracy"]
    info["test_accuracy_dynamic"] = all_row["dynamic_accuracy"]
    info["_note"] = "Stage 3 scenario analysis completed. Accuracy values are from test_with_scenario.csv."
    info["checkpoints"] = {
        "best": "early_exit_lstm_best.pth",
        "fixed": "early_exit_fixed_final.pth",
        "dynamic": "early_exit_dynamic_final.pth",
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model = load_model(args.checkpoint)
    samples = load_samples(args.data)
    template_rows = read_csv(args.template)
    filled_rows = []

    with torch.no_grad():
        for row in template_rows:
            sample_id = int(row["sample_id"])
            sample = samples[sample_id]
            fixed_decision = model.infer_batch_stepwise(sample, dynamic=False)[0]
            dynamic_decision = model.infer_batch_stepwise(sample, dynamic=True)[0]
            dynamic_theta_1, _ = compute_dynamic_threshold(
                sample[0, :, 1],
                base_theta_1=model.theta_1,
                base_theta_2=model.theta_2,
            )

            row = dict(row)
            row["fixed_pred"] = int(fixed_decision.logits.argmax(dim=-1).item())
            row["fixed_exit_point"] = fixed_decision.exit_point
            row["fixed_theta_1"] = model.theta_1
            row["dynamic_pred"] = int(dynamic_decision.logits.argmax(dim=-1).item())
            row["dynamic_exit_point"] = dynamic_decision.exit_point
            row["dynamic_theta_1"] = round(dynamic_theta_1, 6)
            filled_rows.append(row)

    write_csv(args.output, filled_rows, list(filled_rows[0].keys()))
    write_scenario_files(args.scenario_output_dir, filled_rows, list(filled_rows[0].keys()))
    summary_rows = summarize(filled_rows)
    write_csv(args.summary, summary_rows, list(summary_rows[0].keys()))

    shutil.copy2(args.checkpoint, args.fixed_final)
    shutil.copy2(args.checkpoint, args.dynamic_final)
    update_model_info(args.model_info, summary_rows, model)

    print(f"Filled scenario analysis: {display_path(args.output)}")
    print(f"Per-scenario analysis files: {display_path(args.scenario_output_dir)}")
    print(f"Scenario summary: {display_path(args.summary)}")
    print(f"Fixed final checkpoint: {display_path(args.fixed_final)}")
    print(f"Dynamic final checkpoint: {display_path(args.dynamic_final)}")
    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['scenario_id']} {row['scenario']}: "
            f"fixed={row['fixed_accuracy']*100:.1f}% "
            f"dynamic={row['dynamic_accuracy']*100:.1f}% "
            f"dyn_exit={row['dynamic_exit1_rate']*100:.1f}/"
            f"{row['dynamic_exit2_rate']*100:.1f}/"
            f"{row['dynamic_exit3_rate']*100:.1f}"
        )


if __name__ == "__main__":
    main()
