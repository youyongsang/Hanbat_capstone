"""Train the baseline LSTM classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_lstm import BaselineLSTM  # noqa: E402
from utils.dataloader import get_dataloader  # noqa: E402
from utils.metrics import accuracy  # noqa: E402


def run_epoch(
    model: BaselineLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        if is_train:
            optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "dummy")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader = get_dataloader(args.data_dir / "train.csv", batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.data_dir / "val.csv", batch_size=args.batch_size, shuffle=False)

    model = BaselineLSTM(hidden_size=args.hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "baseline_lstm_best.pth"
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.1f}% | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": args.hidden_size,
                    "val_accuracy": best_val_acc,
                },
                checkpoint_path,
            )

    print(f"Best model saved: {checkpoint_path}")
    print(f"Best Val Accuracy: {best_val_acc * 100:.1f}%")


if __name__ == "__main__":
    main()
