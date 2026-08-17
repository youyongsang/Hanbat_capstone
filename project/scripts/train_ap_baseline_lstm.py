"""Train AP measurement Baseline LSTM with 9-feature windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_lstm import BaselineLSTM  # noqa: E402
from utils.ap_dataloader import get_ap_dataloader  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402
from utils.metrics import format_percent  # noqa: E402


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def run_epoch(
    model: BaselineLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AP Baseline LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "ap_metrics_cleaned_strict")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "ap_cleaned_strict")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_ap_dataloader(args.data_dir / "train.csv", args.batch_size, shuffle=True)
    val_loader = get_ap_dataloader(args.data_dir / "val.csv", args.batch_size, shuffle=False)

    model = BaselineLSTM(
        input_size=len(AP_FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_classes=4,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "ap_baseline_lstm_best.pth"
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.3f} | Train Acc: {format_percent(train_acc)} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {format_percent(val_acc)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "hidden_size": args.hidden_size,
                "input_size": len(AP_FEATURE_COLUMNS),
                "feature_columns": list(AP_FEATURE_COLUMNS),
                "val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, checkpoint_path)

    print(f"Best AP baseline model saved: {display_path(checkpoint_path)}")
    print(f"Feature count: {len(AP_FEATURE_COLUMNS)}")
    print(f"Best Val Accuracy: {format_percent(best_val_acc)}")


if __name__ == "__main__":
    main()
