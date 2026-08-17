"""Export the AP measurement (9-feature) SDN-style LSTM checkpoint to ONNX.

The full model exports all three exit heads in one graph. Stage exports
split the model into sequential ONNX graphs so Raspberry Pi inference can
skip deeper LSTM layers after an early confidence-based stop.
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

from models.ap_sdn_lstm import APSDNLSTM  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ap_cleaned_strict"
INPUT_SIZE = len(AP_FEATURE_COLUMNS)


class SDNOnnxWrapper(nn.Module):
    """Return the three classifier heads as separate ONNX graph outputs."""

    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        exit_logits = self.model(x)
        return exit_logits[0], exit_logits[1], exit_logits[2]


class Stage1Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm1
        self.dropout = model.dropout
        self.classifier = model.exit_classifier1

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden_seq, _ = self.lstm(x)
        logits = self.classifier(self.dropout(hidden_seq[:, -1, :]))
        return hidden_seq, logits


class Stage2Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm2
        self.dropout = model.dropout
        self.classifier = model.exit_classifier2

    def forward(self, hidden_seq: Tensor) -> tuple[Tensor, Tensor]:
        hidden_seq2, _ = self.lstm(hidden_seq)
        logits = self.classifier(self.dropout(hidden_seq2[:, -1, :]))
        return hidden_seq2, logits


class Stage3Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm3
        self.dropout = model.dropout
        self.classifier = model.exit_classifier3

    def forward(self, hidden_seq: Tensor) -> Tensor:
        hidden_seq3, _ = self.lstm(hidden_seq)
        return self.classifier(self.dropout(hidden_seq3[:, -1, :]))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_model(checkpoint_path: Path) -> APSDNLSTM:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {display_path(checkpoint_path)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = APSDNLSTM(
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        confidence_threshold=float(checkpoint.get("confidence_threshold", 0.85)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_graph(
    wrapped: nn.Module,
    dummy_input: Tensor,
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )


def export_one(checkpoint_path: Path, onnx_path: Path, name: str) -> None:
    model = load_model(checkpoint_path)
    wrapped = SDNOnnxWrapper(model).eval()
    dummy_input = torch.randn(1, 10, INPUT_SIZE, dtype=torch.float32)

    export_graph(
        wrapped,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["exit1", "exit2", "exit3"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "exit1": {0: "batch_size"},
            "exit2": {0: "batch_size"},
            "exit3": {0: "batch_size"},
        },
    )
    print(f"ONNX export complete ({name}): {display_path(onnx_path)}")


def export_staged(checkpoint_path: Path, output_prefix: Path, name: str) -> None:
    model = load_model(checkpoint_path)
    stages = [
        (
            Stage1Wrapper(model).eval(),
            torch.randn(1, 10, INPUT_SIZE, dtype=torch.float32),
            output_prefix.with_name(f"{output_prefix.name}_stage1.onnx"),
            ["input"],
            ["hidden1", "exit1"],
            {"input": {0: "batch_size"}, "hidden1": {0: "batch_size"}, "exit1": {0: "batch_size"}},
        ),
        (
            Stage2Wrapper(model).eval(),
            torch.randn(1, 10, 128, dtype=torch.float32),
            output_prefix.with_name(f"{output_prefix.name}_stage2.onnx"),
            ["hidden1"],
            ["hidden2", "exit2"],
            {"hidden1": {0: "batch_size"}, "hidden2": {0: "batch_size"}, "exit2": {0: "batch_size"}},
        ),
        (
            Stage3Wrapper(model).eval(),
            torch.randn(1, 10, 128, dtype=torch.float32),
            output_prefix.with_name(f"{output_prefix.name}_stage3.onnx"),
            ["hidden2"],
            ["exit3"],
            {"hidden2": {0: "batch_size"}, "exit3": {0: "batch_size"}},
        ),
    ]
    for wrapped, dummy_input, onnx_path, input_names, output_names, dynamic_axes in stages:
        export_graph(wrapped, dummy_input, onnx_path, input_names, output_names, dynamic_axes)
        print(f"Stage ONNX export complete ({name}): {display_path(onnx_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AP SDN-style LSTM checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_DIR / "ap_sdn_lstm_best.pth"
    )
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR / "ap_sdn_fixed.onnx")
    parser.add_argument(
        "--staged",
        action="store_true",
        default=True,
        help="Also export stage1/stage2/stage3 ONNX files for real Early Exit deployment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_one(args.checkpoint, args.output, args.checkpoint.stem)
    if args.staged:
        export_staged(args.checkpoint, args.output.with_suffix(""), args.checkpoint.stem)


if __name__ == "__main__":
    main()
