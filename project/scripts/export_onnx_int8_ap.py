"""Quantize the exported AP measurement (9-feature) ONNX models with ONNX Runtime.

Expects export_onnx_ap.py and export_onnx_ap_sdn.py to be run first. It does
not manually assemble an ONNX graph; it preserves the exported LSTM graph and
creates separate INT8 model files for deployment comparison.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ap_cleaned_strict"
QUANT_TEMP_DIR = PROJECT_ROOT / ".tmp" / "onnx_quant_ap"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_tasks() -> list[dict[str, object]]:
    tasks = [
        {"name": "Baseline", "input": CHECKPOINT_DIR / "ap_baseline.onnx", "output": CHECKPOINT_DIR / "ap_baseline_int8.onnx"},
    ]
    for prefix, label in (
        ("ap_early_exit_fixed", "Fixed Theta"),
        ("ap_early_exit_dynamic", "Dynamic Theta"),
        ("ap_sdn_fixed", "SDN-style"),
    ):
        tasks.append(
            {
                "name": label,
                "input": CHECKPOINT_DIR / f"{prefix}.onnx",
                "output": CHECKPOINT_DIR / f"{prefix}_int8.onnx",
            }
        )
        for stage in (1, 2, 3):
            tasks.append(
                {
                    "name": f"{label} Stage {stage}",
                    "input": CHECKPOINT_DIR / f"{prefix}_stage{stage}.onnx",
                    "output": CHECKPOINT_DIR / f"{prefix}_stage{stage}_int8.onnx",
                }
            )
    return tasks


def main() -> None:
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

    def infer_shapes_without_external_data(model):
        return onnx.shape_inference.infer_shapes(model)

    quant_utils.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    quantize_module.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    base_quantizer.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    onnx_quantizer.save_and_reload_model_with_shape_infer = infer_shapes_without_external_data
    quantize_dynamic.__globals__["save_and_reload_model_with_shape_infer"] = infer_shapes_without_external_data

    for task in build_tasks():
        input_path: Path = task["input"]
        output_path: Path = task["output"]

        if not input_path.exists():
            print(f"[warning] {task['name']} source file is missing: {display_path(input_path)}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
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

    print("\nNext: compare FP32 and INT8 with inference_pi_ap.py before using Pi results in the report.")


if __name__ == "__main__":
    main()
