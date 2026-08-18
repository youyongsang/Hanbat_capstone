# Raspberry Pi Stage 5 Bundle

Copy this folder to Raspberry Pi and run the commands below from inside the folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas
python inference_pi.py --mode staged --stage1 early_exit_fixed_stage1.onnx --stage2 early_exit_fixed_stage2.onnx --stage3 early_exit_fixed_stage3.onnx --data test.csv --output pi_fixed_staged_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi.py --mode staged --stage1 early_exit_fixed_stage1_int8.onnx --stage2 early_exit_fixed_stage2_int8.onnx --stage3 early_exit_fixed_stage3_int8.onnx --data test.csv --output pi_fixed_staged_int8_results.csv --max-samples 351 --repeats 5
python inference_pi.py --mode staged --dynamic-theta --stage1 early_exit_dynamic_stage1.onnx --stage2 early_exit_dynamic_stage2.onnx --stage3 early_exit_dynamic_stage3.onnx --data test.csv --output pi_dynamic_staged_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi.py --mode staged --dynamic-theta --stage1 early_exit_dynamic_stage1_int8.onnx --stage2 early_exit_dynamic_stage2_int8.onnx --stage3 early_exit_dynamic_stage3_int8.onnx --data test.csv --output pi_dynamic_staged_int8_results.csv --max-samples 351 --repeats 5
python analyze_pi_results.py --input pi_fixed_staged_fp32_results.csv --output-dir . --name pi_fixed_staged_fp32_analysis
python analyze_pi_results.py --input pi_fixed_staged_int8_results.csv --output-dir . --name pi_fixed_staged_int8_analysis
python analyze_pi_results.py --input pi_dynamic_staged_fp32_results.csv --output-dir . --name pi_dynamic_staged_fp32_analysis
python analyze_pi_results.py --input pi_dynamic_staged_int8_results.csv --output-dir . --name pi_dynamic_staged_int8_analysis
```

The older single-model ONNX files are also included for compatibility, but the
staged commands above are the intended Raspberry Pi Early Exit deployment test.

Expected outputs:

- `pi_fixed_staged_fp32_results.csv`
- `pi_fixed_staged_fp32_analysis.txt`
- `pi_fixed_staged_int8_results.csv`
- `pi_fixed_staged_int8_analysis.txt`
- `pi_dynamic_staged_fp32_results.csv`
- `pi_dynamic_staged_fp32_analysis.txt`
- `pi_dynamic_staged_int8_results.csv`
- `pi_dynamic_staged_int8_analysis.txt`

After the run, copy the generated result and analysis files back to:

`project/results/hojung/`
