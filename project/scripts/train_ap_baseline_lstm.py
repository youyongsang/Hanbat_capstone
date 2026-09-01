"""Train AP measurement Baseline LSTM (no early exit) with 7-feature windows.

Trained with the same class-weighted loss and balanced-accuracy checkpoint
selection as train_ap_early_exit.py (default --class-weight-power 0.0 since
the 2026-08-30 re-sweep) so the Baseline/SDN/Early-Exit comparison isolates
architecture, not training regime. Pass --class-weight-power 1 for the old
full inverse-frequency weighting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_baseline_lstm import APBaselineLSTM  # noqa: E402
from utils.ap_dataloader import get_ap_dataloader  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402
from utils.metrics import format_percent  # noqa: E402


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def compute_class_weights(labels: torch.Tensor, num_classes: int = 4, power: float = 0.0) -> torch.Tensor:
    """Power-softened inverse-frequency class weights. See train_ap_early_exit.py
    for the rationale (power=0 disables weighting entirely)."""
    if power == 0:
        return torch.ones(num_classes)
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    return (labels.numel() / (num_classes * counts)) ** power


def run_epoch(
    model: APBaselineLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
    num_classes: int = 4,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((preds == y_batch).sum().item())
        total_samples += batch_size

        for c in range(num_classes):
            mask = y_batch == c
            class_total[c] += int(mask.sum().item())
            class_correct[c] += int((preds[mask] == c).sum().item())

    per_class_recall = [
        class_correct[c] / class_total[c] for c in range(num_classes) if class_total[c] > 0
    ]
    balanced_acc = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0.0

    return total_loss / total_samples, total_correct / total_samples, balanced_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AP Baseline LSTM (no early exit).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--class-weight-power", type=float, default=0.0,
                        help="inverse-frequency class-weight exponent; 0 = no weighting (default since 2026-08-30 re-sweep), 1 = old full weighting.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (unset by default; see train_ap_early_exit.py --seed help for why this matters).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_ap_dataloader(args.data_dir / "train.csv", args.batch_size, shuffle=True)
    val_loader = get_ap_dataloader(args.data_dir / "val.csv", args.batch_size, shuffle=False)

    train_labels = train_loader.dataset.tensors[1]
    class_weights = compute_class_weights(train_labels, power=args.class_weight_power).to(device)
    print(f"Class weights (inverse frequency ^{args.class_weight_power}): {class_weights.tolist()}")

    model = APBaselineLSTM(hidden_size=args.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "ap_baseline_lstm_best.pth"
    best_val_balanced_acc = -1.0
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_loss, val_acc, val_balanced_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.3f} | Train Acc: {format_percent(train_acc)} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {format_percent(val_acc)} | "
            f"Val Balanced Acc: {format_percent(val_balanced_acc)}"
        )

        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "hidden_size": args.hidden_size,
                "input_size": len(AP_FEATURE_COLUMNS),
                "feature_columns": list(AP_FEATURE_COLUMNS),
                "val_accuracy": best_val_acc,
                "val_balanced_accuracy": best_val_balanced_acc,
                "class_weight_power": args.class_weight_power,
            }
            torch.save(checkpoint, checkpoint_path)

    print(f"Best AP baseline model saved: {display_path(checkpoint_path)}")
    print(f"Feature count: {len(AP_FEATURE_COLUMNS)}")
    print(f"Best Val Accuracy: {format_percent(best_val_acc)}")
    print(f"Best Val Balanced Accuracy: {format_percent(best_val_balanced_acc)}")


if __name__ == "__main__":
    main()
