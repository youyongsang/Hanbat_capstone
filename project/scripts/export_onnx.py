"""Export the fixed Early Exit LSTM checkpoint to a self-contained ONNX file."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.early_exit_lstm import EarlyExitLSTM


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "early_exit_fixed.pth"
ONNX_PATH = PROJECT_ROOT / "checkpoints" / "early_exit_fixed.onnx"


class EarlyExitOnnxWrapper(nn.Module):
    """Return the three classifier heads as separate ONNX graph outputs."""

    def __init__(self, model: EarlyExitLSTM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        exit_logits = self.model(x)
        return exit_logits[0], exit_logits[1], exit_logits[2]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {display_path(CHECKPOINT_PATH)}")

    model = EarlyExitLSTM(input_size=4, hidden_size=128, num_classes=4)
    model.load_state_dict(load_checkpoint(CHECKPOINT_PATH))
    model.eval()

    wrapped = EarlyExitOnnxWrapper(model).eval()
    dummy_input = torch.randn(1, 10, 4, dtype=torch.float32)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["exit1", "exit2", "exit3"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "exit1": {0: "batch_size"},
            "exit2": {0: "batch_size"},
            "exit3": {0: "batch_size"},
        },
        opset_version=16,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )

    print(f"ONNX export complete: {display_path(ONNX_PATH)}")


if __name__ == "__main__":
    main()
