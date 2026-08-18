# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_dynamic_staged_fp32_results.csv`
- samples: 351
- measurement repeats: 5
- accuracy: 97.4%
- avg inference: 1.989039 ms
- p95 inference: 3.304772 ms
- avg sample std: 0.607081 ms
- avg confidence: 0.916990

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 5.1% | 100.0% | 0.771918 | 0.802749 | 0.941237 |
| 2 | 312 | 88.9% | 98.1% | 1.972177 | 3.287283 | 0.931502 |
| 3 | 21 | 6.0% | 85.7% | 3.282805 | 4.297955 | 0.680607 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 95.2% | 2.070102 | 3.326531 | 4.8% | 87.9% | 7.2% |
| imbalanced_ap_load | 80 | 97.5% | 1.966394 | 3.287114 | 3.8% | 92.5% | 3.8% |
| lunch_restart | 84 | 98.8% | 1.959569 | 3.293941 | 10.7% | 82.1% | 7.1% |
| startup_surge | 104 | 98.1% | 1.965564 | 3.276420 | 1.9% | 92.3% | 5.8% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 100.0% | 1.715235 |
| 1 | 162 | 98.2% | 2.001447 |
| 2 | 116 | 94.8% | 2.070780 |
| 3 | 17 | 100.0% | 2.214965 |
