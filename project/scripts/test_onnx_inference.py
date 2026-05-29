"""ONNX Runtime shape smoke test for the Stage 4 Early Exit model.

This script verifies that the exported ONNX graph accepts a variable batch
input and returns three classifier heads with the expected (batch, classes)
shape. It is intentionally separate from inference_pi.py, which measures the
deployment-time inference and pseudo early-exit decision pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "project" / "checkpoints" / "early_exit_fixed.onnx"
EXPECTED_CLASSES = 4


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX file not found: {display_path(ONNX_PATH)}")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(2, 10, 4).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})

    if len(outputs) != 3:
        raise RuntimeError(f"Expected 3 ONNX outputs, got {len(outputs)}")

    expected_shape = (dummy.shape[0], EXPECTED_CLASSES)
    for idx, output in enumerate(outputs, start=1):
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"exit{idx} shape mismatch: expected {expected_shape}, got {tuple(output.shape)}"
            )
        print(f"exit{idx}: shape={tuple(output.shape)}, dtype={output.dtype}")

    print(f"ONNX inference smoke test passed: {display_path(ONNX_PATH)}")


if __name__ == "__main__":
    main()
