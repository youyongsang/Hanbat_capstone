# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fixed_staged_fp32_results.csv`
- samples: 351
- measurement repeats: 1
- accuracy: 95.7%
- avg inference: 1.481112 ms
- p95 inference: 1.547379 ms
- avg sample std: 0.000000 ms
- avg confidence: 0.919691

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 41.3% | 100.0% | 1.483503 | 1.555134 | 0.917690 |
| 2 | 157 | 44.7% | 100.0% | 1.481779 | 1.550907 | 0.960373 |
| 3 | 49 | 14.0% | 69.4% | 1.471900 | 1.522989 | 0.795268 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 92.8% | 1.491718 | 1.543511 | 38.6% | 47.0% | 14.5% |
| imbalanced_ap_load | 80 | 97.5% | 1.484229 | 1.547212 | 41.2% | 42.5% | 16.2% |
| lunch_restart | 84 | 96.4% | 1.473904 | 1.527718 | 40.5% | 47.6% | 11.9% |
| startup_surge | 104 | 96.2% | 1.476072 | 1.555202 | 44.2% | 42.3% | 13.5% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 98.2% | 1.492832 |
| 1 | 162 | 97.5% | 1.480563 |
| 2 | 116 | 91.4% | 1.476793 |
| 3 | 17 | 100.0% | 1.477213 |
