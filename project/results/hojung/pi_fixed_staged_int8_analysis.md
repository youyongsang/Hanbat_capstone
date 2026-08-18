# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fixed_staged_int8_results.csv`
- samples: 351
- measurement repeats: 5
- accuracy: 97.2%
- avg inference: 1.297376 ms
- p95 inference: 2.721184 ms
- avg sample std: 0.312038 ms
- avg confidence: 0.921296

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 3.7% | 100.0% | 0.584285 | 0.616712 | 0.954046 |
| 2 | 305 | 86.9% | 100.0% | 1.251606 | 2.689886 | 0.943601 |
| 3 | 33 | 9.4% | 69.7% | 2.001325 | 3.333209 | 0.702244 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 95.2% | 1.456651 | 2.783052 | 1.2% | 86.8% | 12.0% |
| imbalanced_ap_load | 80 | 97.5% | 1.220302 | 1.732040 | 1.2% | 90.0% | 8.8% |
| lunch_restart | 84 | 98.8% | 1.199785 | 1.736942 | 10.7% | 79.8% | 9.5% |
| startup_surge | 104 | 97.1% | 1.308376 | 2.731600 | 1.9% | 90.4% | 7.7% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 100.0% | 1.142159 |
| 1 | 162 | 98.2% | 1.256004 |
| 2 | 116 | 94.0% | 1.394429 |
| 3 | 17 | 100.0% | 1.540689 |
