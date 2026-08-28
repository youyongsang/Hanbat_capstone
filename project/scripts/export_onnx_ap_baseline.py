"""Export the AP Baseline LSTM (no early exit) checkpoint to ONNX, fp32 and INT8.

Single flat graph (no control flow), so onnxruntime's dynamic quantizer
converts the LSTM layers correctly without the If-subgraph pitfall that
affected the unified Early Exit export (see
docs/yongsang/onnx_early_exit_redesign.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_baseline_lstm import APBaselineLSTM  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402

INPUT_SIZE = len(AP_FEATURE_COLUMNS)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AP Baseline LSTM to ONNX (fp32 + INT8).")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_dir / "ap_baseline_lstm_best.pth"
    fp32_path = args.checkpoint_dir / "ap_baseline_lstm.onnx"
    int8_path = args.checkpoint_dir / "ap_baseline_lstm_int8.onnx"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = APBaselineLSTM(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 10, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=16,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"ONNX export complete: {display_path(fp32_path)}")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "LSTM"],
    )
    print(f"INT8 quantized: {display_path(int8_path)}")


if __name__ == "__main__":
    main()
