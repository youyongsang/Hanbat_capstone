"""Export ap_metrics_v2_redesign2 Early Exit LSTM as a single ONNX graph.

The staged export (export_onnx_ap.py) splits the model into 3 separate ONNX
sessions so a Raspberry Pi runner can skip deeper LSTM layers after an early
exit. Measured on-device (2026-08-28, project/results/yongsang/
ap_v2_redesign2_pi_latency_comparison.txt), that 3-session approach was
SLOWER on average than a single full-graph baseline — the fixed per-session-
call overhead (thread sync, tensor marshalling) outweighs the compute saved
by skipping a small (hidden_size=128) LSTM layer.

This script instead `torch.jit.script`s the early-exit control flow (an
`if entropy < theta: return` per stage) so `torch.onnx.export` emits ONNX
`If` nodes. The whole decision lives inside ONE ONNX graph, so Raspberry Pi
inference is a single `InferenceSession.run()` call regardless of which exit
fires — no repeated session-call tax, while the graph itself still skips the
deeper LSTM/classifier ops when the `If` condition is false.

Only supports batch_size=1 (real-time single-window inference) — the control
flow branches on a single sample's entropy, which does not generalize to
per-sample early exits within a batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2"
INPUT_SIZE = len(AP_FEATURE_COLUMNS)

# Dynamic-threshold constants, mirrored from models/early_exit_lstm.py
# compute_dynamic_threshold (occupancy is feature index 1).
DYNAMIC_MIN_THRESHOLD = 0.22
DYNAMIC_RECENT_STEPS = 5
DYNAMIC_SPIKE_THRESHOLD = 0.25


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class UnifiedEarlyExitFixed(nn.Module):
    """Single-graph Early Exit with fixed entropy thresholds."""

    def __init__(self, model: APEarlyExitLSTM, theta_1: float, theta_2: float) -> None:
        super().__init__()
        self.lstm1 = model.lstm1
        self.lstm2 = model.lstm2
        self.lstm3 = model.lstm3
        self.classifier1 = model.exit_classifier1
        self.classifier2 = model.exit_classifier2
        self.classifier3 = model.exit_classifier3
        self.theta_1 = theta_1
        self.theta_2 = theta_2

    def _entropy(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out1, _ = self.lstm1(x)
        logits1 = self.classifier1(out1[:, -1, :])
        e1 = self._entropy(logits1)[0]

        if float(e1.item()) < self.theta_1:
            return logits1, torch.tensor(1, dtype=torch.int64)

        out2, _ = self.lstm2(out1)
        logits2 = self.classifier2(out2[:, -1, :])
        e2 = self._entropy(logits2)[0]

        if float(e2.item()) < self.theta_2:
            return logits2, torch.tensor(2, dtype=torch.int64)

        out3, _ = self.lstm3(out2)
        logits3 = self.classifier3(out3[:, -1, :])
        return logits3, torch.tensor(3, dtype=torch.int64)


class UnifiedEarlyExitDynamic(nn.Module):
    """Single-graph Early Exit with occupancy-trend-adjusted thresholds.

    Mirrors models.early_exit_lstm.compute_dynamic_threshold: if the most
    recent occupancy step changed by more than DYNAMIC_SPIKE_THRESHOLD, use
    the base thresholds as-is; otherwise relax them (x1.25, floored) so a
    calm channel exits earlier.
    """

    def __init__(self, model: APEarlyExitLSTM, theta_1: float, theta_2: float) -> None:
        super().__init__()
        self.lstm1 = model.lstm1
        self.lstm2 = model.lstm2
        self.lstm3 = model.lstm3
        self.classifier1 = model.exit_classifier1
        self.classifier2 = model.exit_classifier2
        self.classifier3 = model.exit_classifier3
        self.base_theta_1 = theta_1
        self.base_theta_2 = theta_2
        self.recent_steps = DYNAMIC_RECENT_STEPS
        self.spike_threshold = DYNAMIC_SPIKE_THRESHOLD
        self.min_threshold = DYNAMIC_MIN_THRESHOLD

    def _entropy(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        occupancy = x[0, -self.recent_steps:, 1]
        delta = float(torch.abs(occupancy[-1] - occupancy[-2]).item())
        if delta > self.spike_threshold:
            theta_1 = self.base_theta_1
            theta_2 = self.base_theta_2
        else:
            theta_1 = max(self.base_theta_1 * 1.25, self.min_threshold)
            theta_2 = max(self.base_theta_2 * 1.25, self.min_threshold * 2)

        out1, _ = self.lstm1(x)
        logits1 = self.classifier1(out1[:, -1, :])
        e1 = self._entropy(logits1)[0]

        if float(e1.item()) < theta_1:
            return logits1, torch.tensor(1, dtype=torch.int64)

        out2, _ = self.lstm2(out1)
        logits2 = self.classifier2(out2[:, -1, :])
        e2 = self._entropy(logits2)[0]

        if float(e2.item()) < theta_2:
            return logits2, torch.tensor(2, dtype=torch.int64)

        out3, _ = self.lstm3(out2)
        logits3 = self.classifier3(out3[:, -1, :])
        return logits3, torch.tensor(3, dtype=torch.int64)


def load_ee_model(checkpoint_path: Path) -> tuple[APEarlyExitLSTM, float, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {display_path(checkpoint_path)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = APEarlyExitLSTM(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    theta_1 = float(checkpoint.get("theta_1", 0.3))
    theta_2 = float(checkpoint.get("theta_2", 0.6))
    return model, theta_1, theta_2


def export_unified(checkpoint_path: Path, onnx_path: Path, dynamic: bool) -> None:
    model, theta_1, theta_2 = load_ee_model(checkpoint_path)
    wrapper_cls = UnifiedEarlyExitDynamic if dynamic else UnifiedEarlyExitFixed
    wrapped = wrapper_cls(model, theta_1, theta_2).eval()

    scripted = torch.jit.script(wrapped)
    dummy_input = torch.randn(1, 10, INPUT_SIZE, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        scripted,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits", "exit_point"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Unified ONNX export complete: {display_path(onnx_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AP redesign Early Exit LSTM as a single ONNX graph (If nodes).")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_unified(
        args.checkpoint_dir / "ap_early_exit_fixed.pth",
        args.checkpoint_dir / "ap_early_exit_fixed_unified.onnx",
        dynamic=False,
    )
    export_unified(
        args.checkpoint_dir / "ap_early_exit_dynamic.pth",
        args.checkpoint_dir / "ap_early_exit_dynamic_unified.onnx",
        dynamic=True,
    )


if __name__ == "__main__":
    main()
