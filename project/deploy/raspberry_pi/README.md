# Raspberry Pi Stage 5 Bundle

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
