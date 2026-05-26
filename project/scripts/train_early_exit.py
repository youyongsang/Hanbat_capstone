"""Train Early Exit LSTM with multi-exit loss."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_exit_lstm import EarlyExitLSTM, multi_exit_loss  # noqa: E402
from utils.dataloader import get_dataloader  # noqa: E402
from utils.metrics import format_percent  # noqa: E402


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: EarlyExitLSTM,
    dataloader: torch.utils.data.DataLoader,
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

        exit_logits = model(x_batch)
        loss = multi_exit_loss(exit_logits, y_batch)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((exit_logits[-1].argmax(dim=1) == y_batch).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Early Exit LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--theta-1", type=float, default=0.3)
    parser.add_argument("--theta-2", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader = get_dataloader(args.data_dir / "train.csv", args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.data_dir / "val.csv", args.batch_size, shuffle=False)

    model = EarlyExitLSTM(
        hidden_size=args.hidden_size,
        theta_1=args.theta_1,
        theta_2=args.theta_2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "early_exit_lstm_best.pth"
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.3f} | Train Acc: {format_percent(train_acc)} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {format_percent(val_acc)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": args.hidden_size,
                    "theta_1": args.theta_1,
                    "theta_2": args.theta_2,
                    "val_accuracy": best_val_acc,
                },
                checkpoint_path,
            )

    print(f"Best model saved: {checkpoint_path}")
    print(f"Best Val Accuracy: {format_percent(best_val_acc)}")


if __name__ == "__main__":
    main()
