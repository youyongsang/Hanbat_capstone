"""Evaluate the baseline LSTM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_lstm import BaselineLSTM  # noqa: E402
from utils.dataloader import get_dataloader  # noqa: E402
from utils.metrics import confusion_matrix, precision_recall_from_confusion  # noqa: E402


LABEL_NAMES = ["정상", "혼잡 경고", "혼잡", "심각"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "dummy")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_lstm_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = BaselineLSTM(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = get_dataloader(args.data_dir / "test.csv", batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    predictions = []
    targets = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(y_batch.cpu().numpy().tolist())

    y_pred = np.array(predictions)
    y_true = np.array(targets)
    test_acc = float((y_pred == y_true).mean())
    matrix = confusion_matrix(y_pred, y_true)

    print(f"Test Loss: {total_loss / total_samples:.3f}")
    print(f"Test Accuracy: {test_acc * 100:.1f}%")
    for label, (precision, recall) in enumerate(precision_recall_from_confusion(matrix)):
        print(
            f"Label {label} ({LABEL_NAMES[label]}): "
            f"Precision: {precision:.2f}  Recall: {recall:.2f}"
        )
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
