# Raspberry Pi Inference Analysis

- input: `C:\Users\PC\Documents\카카오톡 받은 파일\pi_inference_results.csv`
- samples: 100
- accuracy: 19.0%
- avg inference: 0.056563 ms
- p95 inference: 0.065774 ms
- avg confidence: 0.667468

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | 41.0% | 26.8% | 0.058359 | 0.069871 | 0.994950 |
| 3 | 59 | 59.0% | 13.6% | 0.055315 | 0.060684 | 0.439896 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 29 | 24.1% | 0.058093 | 0.064530 | 48.3% | 0.0% | 51.7% |
| imbalanced_ap_load | 31 | 19.4% | 0.056912 | 0.068000 | 48.4% | 0.0% | 51.6% |
| lunch_restart | 20 | 20.0% | 0.054773 | 0.059202 | 45.0% | 0.0% | 55.0% |
| startup_surge | 20 | 10.0% | 0.055593 | 0.059939 | 15.0% | 0.0% | 85.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 46.7% | 0.056244 |
| 1 | 46 | 19.6% | 0.057562 |
| 2 | 35 | 8.6% | 0.055685 |
| 3 | 4 | 0.0% | 0.053949 |
