# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_int8_results.csv`
- samples: 351
- accuracy: 95.7%
- avg inference: 0.918881 ms
- p95 inference: 1.020873 ms
- avg confidence: 0.920378

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 41.3% | 100.0% | 0.922785 | 1.034894 | 0.918240 |
| 2 | 158 | 45.0% | 100.0% | 0.919095 | 0.997853 | 0.960097 |
| 3 | 48 | 13.7% | 68.8% | 0.906378 | 0.970416 | 0.796091 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 94.0% | 0.916641 | 1.017293 | 39.8% | 45.8% | 14.5% |
| imbalanced_ap_load | 80 | 97.5% | 0.918470 | 0.996153 | 41.2% | 41.2% | 17.5% |
| lunch_restart | 84 | 96.4% | 0.916831 | 1.014480 | 39.3% | 51.2% | 9.5% |
| startup_surge | 104 | 95.2% | 0.922639 | 1.023825 | 44.2% | 42.3% | 13.5% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 96.4% | 0.895640 |
| 1 | 162 | 97.5% | 0.923806 |
| 2 | 116 | 92.2% | 0.923585 |
| 3 | 17 | 100.0% | 0.916400 |
