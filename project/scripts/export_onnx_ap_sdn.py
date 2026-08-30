"""Export the AP SDN-style Early Exit LSTM checkpoint to staged ONNX graphs.

Mirrors export_onnx_ap.py's staged split (stage1/2/3, one LSTM+classifier
each) so export_onnx_ap_sdn_unified_int8.py can quantize each flat stage
independently (LSTM quantizes correctly there) before reassembling into a
single confidence-threshold If-node graph, following the same fix used for
the Early Exit model (see docs/yongsang/onnx_early_exit_redesign.md).
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

INPUT_SIZE = len(AP_FEATURE_COLUMNS)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class Stage1Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm1
        self.classifier = model.exit_classifier1  # SDNInternalClassifier: pools internally

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden_seq, _ = self.lstm(x)
        logits = self.classifier(hidden_seq)
        return hidden_seq, logits


class Stage2Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm2
        self.classifier = model.exit_classifier2  # SDNInternalClassifier: pools internally

    def forward(self, hidden_seq: Tensor) -> tuple[Tensor, Tensor]:
        hidden_seq2, _ = self.lstm(hidden_seq)
        logits = self.classifier(hidden_seq2)
        return hidden_seq2, logits


class Stage3Wrapper(nn.Module):
    def __init__(self, model: APSDNLSTM) -> None:
        super().__init__()
        self.lstm = model.lstm3
        self.dropout = model.dropout
        self.classifier = model.exit_classifier3  # base network head: last timestep -> linear

    def forward(self, hidden_seq: Tensor) -> Tensor:
        hidden_seq3, _ = self.lstm(hidden_seq)
        return self.classifier(self.dropout(hidden_seq3[:, -1, :]))


def export_graph(wrapped, dummy_input, onnx_path, input_names, output_names, dynamic_axes) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy_input, onnx_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=16,
        do_constant_folding=True, dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AP SDN-style LSTM to staged ONNX graphs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_dir / "ap_sdn_lstm_best.pth", map_location="cpu")
    model = APSDNLSTM(
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        confidence_threshold=float(checkpoint.get("confidence_threshold", 0.85)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prefix = args.checkpoint_dir / "ap_sdn"
    stages = [
        (Stage1Wrapper(model).eval(), torch.randn(1, 10, INPUT_SIZE), f"{prefix}_stage1.onnx",
         ["input"], ["hidden1", "exit1"],
         {"input": {0: "batch_size"}, "hidden1": {0: "batch_size"}, "exit1": {0: "batch_size"}}),
        (Stage2Wrapper(model).eval(), torch.randn(1, 10, 128), f"{prefix}_stage2.onnx",
         ["hidden1"], ["hidden2", "exit2"],
         {"hidden1": {0: "batch_size"}, "hidden2": {0: "batch_size"}, "exit2": {0: "batch_size"}}),
        (Stage3Wrapper(model).eval(), torch.randn(1, 10, 128), f"{prefix}_stage3.onnx",
         ["hidden2"], ["exit3"],
         {"hidden2": {0: "batch_size"}, "exit3": {0: "batch_size"}}),
    ]
    for wrapped, dummy, path, in_names, out_names, dyn_axes in stages:
        export_graph(wrapped, dummy, Path(path), in_names, out_names, dyn_axes)
        print(f"Stage ONNX export complete: {display_path(Path(path))}")

    print(f"confidence_threshold: {model.confidence_threshold}")


if __name__ == "__main__":
    main()
