"""Train AP measurement Early Exit LSTM with 9-feature windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from models.early_exit_lstm import DEFAULT_LOSS_WEIGHTS, multi_exit_loss  # noqa: E402
from utils.ap_dataloader import get_ap_dataloader  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402
from utils.metrics import format_percent  # noqa: E402


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 4,
    power: float = 0.0,
) -> torch.Tensor:
    """Power-softened inverse-frequency class weights for imbalanced labels.

    power=0 means no class weighting at all (plain cross-entropy); power=1
    is full inverse-frequency weighting; values in between soften it.

    HISTORY: power=1.0 was the default from 2026-08-23 to 2026-08-30. That
    was fixed on a much smaller/older dataset (4-feature, ~23 label-3 train
    examples) where a sharp cliff existed: power<=0.85 gave label-3 recall
    exactly 0%, only power=1.0 got the model to predict label 3 at all.
    After the label redesign + 6->7-feature promotion (train label-3 count
    grew to 141) that cliff is gone. A 2026-08-30 re-sweep (0.0/0.1/0.15/
    0.2/0.3/0.5/0.7/0.85/1.0, 3 seeds each) found power=0.0 best on BOTH
    overall accuracy (91.3%+-0.5% vs 87.0%+-1.1% at 1.0) and label-3 F1
    (69.8% vs 63.2%) with no tradeoff -- power=1.0 was over-protecting
    label 1/3 and leaving label 2 to bleed errors both ways. Default is
    now 0.0. See .work-log/current.md checkpoints 4-5 (2026-08-30).
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
    exit_loss_weights: tuple[float, float, float] = DEFAULT_LOSS_WEIGHTS,
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
        loss = multi_exit_loss(
            exit_logits, y_batch, weights=exit_loss_weights, class_weights=class_weights
        )

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
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.0,
        help=(
            "inverse-frequency class-weight exponent. 0 = no weighting "
            "(default since 2026-08-30 re-sweep; best on both accuracy and "
            "label-3 F1 for the 7-feature dataset). 1 = full inverse "
            "frequency (old default)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for model init and dataloader shuffling. Unset by default "
            "(matches prior behavior), but a 2026-08-29 multi-seed sweep found "
            "run-to-run variance (e.g. test Label 3 F1 swinging 52-63% across "
            "seeds on ap_metrics_v2_redesign2) large enough to make single-run "
            "A/B comparisons between hyperparameter choices unreliable. Pass a "
            "seed for reproducible runs, and compare configs across several "
            "seeds rather than trusting one run each."
        ),
    )
    parser.add_argument(
        "--exit-loss-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_LOSS_WEIGHTS),
        metavar=("W1", "W2", "W3"),
        help=(
            "Per-exit loss weights (exit1, exit2, exit3). Default matches the "
            "uniform-ish EE policy (0.3/0.3/0.4). Pass SDN-style 0.15 0.30 0.55 "
            "to weight the deepest exit more heavily, matching the SDN paper's "
            "loss schedule."
        ),
    )
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
    exit_loss_weights = tuple(args.exit_loss_weights)
    print(f"Exit loss weights: {exit_loss_weights}")

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
            model,
            train_loader,
            optimizer,
            device=device,
            class_weights=class_weights,
            exit_loss_weights=exit_loss_weights,
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
                "exit_loss_weights": list(exit_loss_weights),
                "class_weight_power": args.class_weight_power,
                "seed": args.seed,
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
