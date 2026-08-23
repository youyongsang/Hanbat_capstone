"""Train AP measurement Early Exit LSTM with 9-feature windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from models.early_exit_lstm import multi_exit_loss  # noqa: E402
from utils.ap_dataloader import get_ap_dataloader  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402
from utils.metrics import format_percent  # noqa: E402


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 4,
    power: float = 0.7,
) -> torch.Tensor:
    """Power-softened inverse-frequency class weights for imbalanced labels.

    Without this, plain cross-entropy on a skewed label distribution (e.g.
    AP strict live-collection congestion labels, where label 1/2 heavily
    outnumber label 3) lets the model collapse to always predicting the
    majority class(es) and never output the rare ones at all.

    Plain inverse frequency (N / (K * count_c)), i.e. power=1.0, overcorrects
    on this data: with only 9-14 label-3 (severe) train examples, the raw
    weight lands around 20x, aggressive enough that label 2 (congested) gets
    routinely over-predicted as label 3 (label 2 recall dropped to ~19% in an
    earlier run). power=0.5 (sqrt) fixed that (label 2 recall 79%) but swung
    too far the other way (label 3 recall 0%). power=0.7 is a midpoint
    between the two, tuned by observing that tradeoff on this dataset.
    """

    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = (labels.numel() / (num_classes * counts)) ** power
    return weights


def run_epoch(
    model: APEarlyExitLSTM,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
    class_weights: torch.Tensor | None = None,
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

        exit_logits = model(x_batch)
        loss = multi_exit_loss(exit_logits, y_batch, class_weights=class_weights)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = exit_logits[-1].argmax(dim=1)
        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((preds == y_batch).sum().item())
        total_samples += batch_size

        for c in range(num_classes):
            mask = y_batch == c
            class_total[c] += int(mask.sum().item())
            class_correct[c] += int((preds[mask] == c).sum().item())

    # Balanced accuracy: mean of per-class recall, ignoring classes absent
    # from this split. Plain accuracy alone rewards checkpoints that just
    # collapse to the majority class(es) on an imbalanced split.
    per_class_recall = [
        class_correct[c] / class_total[c] for c in range(num_classes) if class_total[c] > 0
    ]
    balanced_acc = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0.0

    return total_loss / total_samples, total_correct / total_samples, balanced_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AP Early Exit LSTM.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "ap_metrics")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_ap_dataloader(args.data_dir / "train.csv", args.batch_size, shuffle=True)
    val_loader = get_ap_dataloader(args.data_dir / "val.csv", args.batch_size, shuffle=False)

    train_labels = train_loader.dataset.tensors[1]
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"Class weights (inverse frequency ^0.7): {class_weights.tolist()}")

    model = APEarlyExitLSTM(
        hidden_size=args.hidden_size,
        theta_1=args.theta_1,
        theta_2=args.theta_2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "ap_early_exit_lstm_best.pth"
    fixed_checkpoint_path = args.checkpoint_dir / "ap_early_exit_fixed.pth"
    dynamic_checkpoint_path = args.checkpoint_dir / "ap_early_exit_dynamic.pth"
    best_val_balanced_acc = -1.0
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, optimizer, device=device, class_weights=class_weights
        )
        with torch.no_grad():
            val_loss, val_acc, val_balanced_acc = run_epoch(model, val_loader, device=device)

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
                "theta_1": args.theta_1,
                "theta_2": args.theta_2,
                "input_size": len(AP_FEATURE_COLUMNS),
                "feature_columns": list(AP_FEATURE_COLUMNS),
                "val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save({**checkpoint, "dynamic_threshold": False}, fixed_checkpoint_path)
            torch.save({**checkpoint, "dynamic_threshold": True}, dynamic_checkpoint_path)

    print(f"Best AP model saved: {display_path(checkpoint_path)}")
    print(f"Fixed-threshold AP checkpoint saved: {display_path(fixed_checkpoint_path)}")
    print(f"Dynamic-threshold AP checkpoint saved: {display_path(dynamic_checkpoint_path)}")
    print(f"Feature count: {len(AP_FEATURE_COLUMNS)}")
    print(f"Best Val Accuracy: {format_percent(best_val_acc)}")
    print(f"Best Val Balanced Accuracy: {format_percent(best_val_balanced_acc)}")


if __name__ == "__main__":
    main()
