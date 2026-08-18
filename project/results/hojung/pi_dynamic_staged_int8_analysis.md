# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_dynamic_staged_int8_results.csv`
- samples: 351
- measurement repeats: 1
- accuracy: 92.0%
- avg inference: 0.880486 ms
- p95 inference: 0.958239 ms
- avg sample std: 0.000000 ms
- avg confidence: 0.789436

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 348 | 99.2% | 92.2% | 0.880617 | 0.958786 | 0.790783 |
| 2 | 3 | 0.9% | 66.7% | 0.865231 | 0.883134 | 0.633140 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 94.0% | 0.883862 | 0.961879 | 100.0% | 0.0% | 0.0% |
| imbalanced_ap_load | 80 | 90.0% | 0.880039 | 0.965228 | 98.8% | 1.2% | 0.0% |
| lunch_restart | 84 | 90.5% | 0.879118 | 0.923158 | 98.8% | 1.2% | 0.0% |
| startup_surge | 104 | 93.3% | 0.879240 | 0.932796 | 99.0% | 1.0% | 0.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 85.7% | 0.881171 |
| 1 | 162 | 98.2% | 0.882352 |
| 2 | 116 | 94.8% | 0.876606 |
| 3 | 17 | 35.3% | 0.886924 |
