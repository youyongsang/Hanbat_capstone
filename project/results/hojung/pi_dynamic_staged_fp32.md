# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_dynamic_staged_fp32_results.csv`
- samples: 351
- measurement repeats: 1
- accuracy: 92.0%
- avg inference: 1.463609 ms
- p95 inference: 1.547658 ms
- avg sample std: 0.000000 ms
- avg confidence: 0.788983

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 349 | 99.4% | 92.3% | 1.463551 | 1.547708 | 0.790358 |
| 2 | 2 | 0.6% | 50.0% | 1.473742 | 1.479392 | 0.549065 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 94.0% | 1.463172 | 1.538770 | 100.0% | 0.0% | 0.0% |
| imbalanced_ap_load | 80 | 88.8% | 1.452839 | 1.507416 | 100.0% | 0.0% | 0.0% |
| lunch_restart | 84 | 91.7% | 1.470223 | 1.578955 | 97.6% | 2.4% | 0.0% |
| startup_surge | 104 | 93.3% | 1.466899 | 1.551748 | 100.0% | 0.0% | 0.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 83.9% | 1.457335 |
| 1 | 162 | 98.2% | 1.468549 |
| 2 | 116 | 95.7% | 1.461297 |
| 3 | 17 | 35.3% | 1.452965 |
