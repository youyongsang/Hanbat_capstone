"""Quantize the exported Early Exit ONNX model with ONNX Runtime.

This script expects project/scripts/export_onnx.py to be run first. It does
not manually assemble an ONNX graph; it preserves the exported LSTM graph and
creates a separate INT8 model file for deployment comparison.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "checkpoints" / "early_exit_fixed.onnx"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "early_exit_fixed_int8.onnx"
QUANT_TEMP_DIR = PROJECT_ROOT / ".tmp" / "onnx_quant"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create INT8 ONNX model from exported FP32 ONNX.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="FP32 ONNX model path")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="INT8 ONNX output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {display_path(args.input)}. "
            "Run `python project\\scripts\\export_onnx.py` first."
        )

    QUANT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(QUANT_TEMP_DIR)
    os.environ["TEMP"] = str(QUANT_TEMP_DIR)
    tempfile.tempdir = str(QUANT_TEMP_DIR)

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnxruntime.quantization.base_quantizer as base_quantizer
        import onnxruntime.quantization.onnx_quantizer as onnx_quantizer
        import onnxruntime.quantization.quant_utils as quant_utils
        import onnxruntime.quantization.quantize as quantize_module
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime quantization tools are required. "
            "Install with `pip install onnx onnxruntime`."
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def infer_shapes_without_external_data(model):
        return onnx.shape_inference.infer_shapes(model)

    quant_utils.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    quantize_module.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    base_quantizer.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    onnx_quantizer.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    quantize_dynamic.__globals__["save_and_reload_model_with_shape_infer"] = infer_shapes_without_external_data

    model = onnx.load(str(args.input))
    quantize_dynamic(
        model_input=model,
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )

    input_size = args.input.stat().st_size / (1024 * 1024)
    output_size = args.output.stat().st_size / (1024 * 1024)
    print("INT8 ONNX quantization complete")
    print(f"input: {display_path(args.input)} ({input_size:.4f} MB)")
    print(f"output: {display_path(args.output)} ({output_size:.4f} MB)")
    print("Next: compare FP32 and INT8 with inference_pi.py before using Pi results in the report.")


if __name__ == "__main__":
    main()
