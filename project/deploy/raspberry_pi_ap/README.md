# Raspberry Pi AP Strict (9-feature) Bundle

Copy this folder to Raspberry Pi and run the commands below from inside the folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas

# Baseline (always full-depth, no early exit)
python inference_pi_ap.py --mode baseline --model ap_baseline.onnx --data test.csv --output pi_ap_baseline_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi_ap.py --mode baseline --model ap_baseline_int8.onnx --data test.csv --output pi_ap_baseline_int8_results.csv --max-samples 351 --repeats 5

# SDN-style (confidence-threshold staged exit)
python inference_pi_ap.py --mode staged-confidence --stage-prefix ap_sdn_fixed --data test.csv --output pi_ap_sdn_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi_ap.py --mode staged-confidence --stage1 ap_sdn_fixed_stage1_int8.onnx --stage2 ap_sdn_fixed_stage2_int8.onnx --stage3 ap_sdn_fixed_stage3_int8.onnx --data test.csv --output pi_ap_sdn_int8_results.csv --max-samples 351 --repeats 5

# Proposed Fixed theta (entropy-threshold staged exit)
python inference_pi_ap.py --mode staged --stage-prefix ap_early_exit_fixed --data test.csv --output pi_ap_fixed_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi_ap.py --mode staged --stage1 ap_early_exit_fixed_stage1_int8.onnx --stage2 ap_early_exit_fixed_stage2_int8.onnx --stage3 ap_early_exit_fixed_stage3_int8.onnx --data test.csv --output pi_ap_fixed_int8_results.csv --max-samples 351 --repeats 5

# Proposed Dynamic theta (entropy-threshold staged exit, dynamic theta from recent occupancy)
python inference_pi_ap.py --mode staged --dynamic-theta --stage-prefix ap_early_exit_dynamic --data test.csv --output pi_ap_dynamic_fp32_results.csv --max-samples 351 --repeats 5
python inference_pi_ap.py --mode staged --dynamic-theta --stage1 ap_early_exit_dynamic_stage1_int8.onnx --stage2 ap_early_exit_dynamic_stage2_int8.onnx --stage3 ap_early_exit_dynamic_stage3_int8.onnx --data test.csv --output pi_ap_dynamic_int8_results.csv --max-samples 351 --repeats 5

# Analysis (accuracy / latency / exit distribution / scenario breakdown)
python analyze_pi_results.py --input pi_ap_baseline_fp32_results.csv --output-dir . --name pi_ap_baseline_fp32_analysis
python analyze_pi_results.py --input pi_ap_baseline_int8_results.csv --output-dir . --name pi_ap_baseline_int8_analysis
python analyze_pi_results.py --input pi_ap_sdn_fp32_results.csv --output-dir . --name pi_ap_sdn_fp32_analysis
python analyze_pi_results.py --input pi_ap_sdn_int8_results.csv --output-dir . --name pi_ap_sdn_int8_analysis
python analyze_pi_results.py --input pi_ap_fixed_fp32_results.csv --output-dir . --name pi_ap_fixed_fp32_analysis
python analyze_pi_results.py --input pi_ap_fixed_int8_results.csv --output-dir . --name pi_ap_fixed_int8_analysis
python analyze_pi_results.py --input pi_ap_dynamic_fp32_results.csv --output-dir . --name pi_ap_dynamic_fp32_analysis
python analyze_pi_results.py --input pi_ap_dynamic_int8_results.csv --output-dir . --name pi_ap_dynamic_int8_analysis
```

INT8 staged runs pass `--stage1/--stage2/--stage3` explicitly because
`export_onnx_int8_ap.py` names quantized stage files `..._stageN_int8.onnx`
(suffix after the stage number), which does not match the `--stage-prefix`
pattern (`{prefix}_stageN.onnx`) used for FP32 runs.

After the run, copy the generated `pi_ap_*_results.csv` and `pi_ap_*_analysis.*`
files back to `project/results/yongsang/` (or `project/results/hojung/`) in the repo.

This bundle is the AP strict (9-feature) counterpart of `project/deploy/raspberry_pi/`,
which is for the 1st-semester 4-feature models. Do not mix `test.csv` files between
the two bundles — column counts differ (4 vs 9) and results will be silently wrong.
