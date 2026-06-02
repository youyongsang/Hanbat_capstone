"""Create a Raspberry Pi deployment bundle for Stage 5."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT_DIR / "project" / "deploy" / "raspberry_pi"


FILES = [
    (ROOT_DIR / "project" / "checkpoints" / "early_exit_fixed.onnx", BUNDLE_DIR / "early_exit_fixed.onnx"),
    (ROOT_DIR / "project" / "checkpoints" / "early_exit_fixed_int8.onnx", BUNDLE_DIR / "early_exit_fixed_int8.onnx"),
    (ROOT_DIR / "project" / "data" / "real" / "test.csv", BUNDLE_DIR / "test.csv"),
    (ROOT_DIR / "project" / "scripts" / "inference_pi.py", BUNDLE_DIR / "inference_pi.py"),
]


README = """# Raspberry Pi Stage 5 Bundle

Copy this folder to Raspberry Pi and run the commands below from inside the folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_fp32_results.csv --max-samples 351
python inference_pi.py --model early_exit_fixed_int8.onnx --data test.csv --output pi_int8_results.csv --max-samples 351
```

Expected outputs:

- `pi_fp32_results.csv`
- `pi_fp32_results.txt`
- `pi_int8_results.csv`
- `pi_int8_results.txt`

After the run, copy both result files back to:

`project/results/hojung/`
"""


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for src, dst in FILES:
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, dst)

    (BUNDLE_DIR / "README.md").write_text(README, encoding="utf-8")

    print(f"Raspberry Pi bundle created: {BUNDLE_DIR}")
    for _, dst in FILES:
        print(f"- {dst.name}")
    print("- README.md")


if __name__ == "__main__":
    main()
