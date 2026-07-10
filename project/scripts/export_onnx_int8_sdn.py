"""Quantize the exported SDN ONNX models with ONNX Runtime.

This script expects project/scripts/export_onnx_sdn.py to be run first.
Creates separate INT8 model files for baseline comparison on Raspberry Pi.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_INPUT = PROJECT_ROOT / "checkpoints" / "sdn_fixed.onnx"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "sdn_fixed_int8.onnx"
QUANT_TEMP_DIR = PROJECT_ROOT / ".tmp" / "onnx_quant_sdn"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create INT8 ONNX model from exported SDN FP32 ONNX.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="FP32 SDN ONNX path")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="INT8 SDN ONNX output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    # SDN 통합 그래프 및 Stage 분할 그래프 양자화 태스크 정의
    tasks = [
        {
            "name": "SDN Fixed Confidence",
            "input": args.input,
            "output": args.output,
        },
        {
            "name": "SDN Stage 1",
            "input": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage1.onnx",
            "output": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage1_int8.onnx",
        },
        {
            "name": "SDN Stage 2",
            "input": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage2.onnx",
            "output": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage2_int8.onnx",
        },
        {
            "name": "SDN Stage 3",
            "input": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage3.onnx",
            "output": PROJECT_ROOT / "checkpoints" / "sdn_fixed_stage3_int8.onnx",
        },
    ]

    for task in tasks:
        input_path = task["input"]
        output_path = task["output"]

        if not input_path.exists():
            print(f"[warning] {task['name']} source file is missing: {display_path(input_path)}")
            continue

        model = onnx.load(str(input_path))
        quantize_dynamic(
            model_input=model,
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )

        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n[{task['name']}] INT8 ONNX quantization complete")
        print(f"   input: {display_path(input_path)} ({input_size:.4f} MB)")
        print(f"   output: {display_path(output_path)} ({output_size:.4f} MB)")

    print("\nSDN Baseline ONNX Quantization Phase Complete.")


if __name__ == "__main__":
    main()