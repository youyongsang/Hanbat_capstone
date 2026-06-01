# Raspberry Pi Stage 5 Bundle

Copy this folder to Raspberry Pi and run the commands below from inside the folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_inference_results.csv --max-samples 100
```

Expected outputs:

- `pi_inference_results.csv`
- `pi_inference_results.txt`

After the run, copy both result files back to:

`project/results/hojung/`
