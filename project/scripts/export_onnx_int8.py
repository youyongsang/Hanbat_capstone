"""Quantize the exported Early Exit ONNX models (Fixed & Dynamic) with ONNX Runtime.

This script expects project/scripts/export_onnx.py to be run first. It does
not manually assemble an ONNX graph; it preserves the exported LSTM graph and
creates separate INT8 model files for deployment comparison.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

# [기존 원본 양식 유지] 기본 고정(Fixed) 모델 경로
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

    tasks = [
        {
            "name": "Fixed Theta",
            "input": args.input,
            "output": args.output,
        },
        {
            "name": "Dynamic Theta",
            "input": PROJECT_ROOT / "checkpoints" / "early_exit_dynamic.onnx",
            "output": PROJECT_ROOT / "checkpoints" / "early_exit_dynamic_int8.onnx",
        },
    ]
    for prefix, label in (("early_exit_fixed", "Fixed Theta"), ("early_exit_dynamic", "Dynamic Theta")):
        for stage in (1, 2, 3):
            tasks.append(
                {
                    "name": f"{label} Stage {stage}",
                    "input": PROJECT_ROOT / "checkpoints" / f"{prefix}_stage{stage}.onnx",
                    "output": PROJECT_ROOT / "checkpoints" / f"{prefix}_stage{stage}_int8.onnx",
                }
            )

    # 반복문을 돌면서 두 모델 다 경량화(INT8) 수행
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

    print("\nNext: compare FP32 and INT8 with inference_pi.py before using Pi results in the report.")


if __name__ == "__main__":
    main()
