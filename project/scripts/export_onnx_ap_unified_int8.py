"""INT8 dynamic-quantize the unified (single-graph, ONNX If node) Early Exit models.

Run export_onnx_ap_unified.py first. This script only quantizes the already
exported *_unified.onnx graphs -- it does not touch the staged exports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def quantize(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Not found: {display_path(src)} -- run export_onnx_ap_unified.py first")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "LSTM"],
    )
    print(f"INT8 quantized: {display_path(dst)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8-quantize the unified AP Early Exit ONNX graphs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize(
        args.checkpoint_dir / "ap_early_exit_fixed_unified.onnx",
        args.checkpoint_dir / "ap_early_exit_fixed_unified_int8.onnx",
    )
    quantize(
        args.checkpoint_dir / "ap_early_exit_dynamic_unified.onnx",
        args.checkpoint_dir / "ap_early_exit_dynamic_unified_int8.onnx",
    )


if __name__ == "__main__":
    main()
