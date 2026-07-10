# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fixed_staged_int8_results.csv`
- samples: 351
- measurement repeats: 1
- accuracy: 95.7%
- avg inference: 0.860520 ms
- p95 inference: 0.913369 ms
- avg sample std: 0.000000 ms
- avg confidence: 0.920378

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 41.3% | 100.0% | 0.859695 | 0.912241 | 0.918240 |
| 2 | 158 | 45.0% | 100.0% | 0.862440 | 0.915775 | 0.960097 |
| 3 | 48 | 13.7% | 68.8% | 0.856693 | 0.903145 | 0.796091 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 94.0% | 0.866394 | 0.922351 | 39.8% | 45.8% | 14.5% |
| imbalanced_ap_load | 80 | 97.5% | 0.859589 | 0.906874 | 41.2% | 41.2% | 17.5% |
| lunch_restart | 84 | 96.4% | 0.858741 | 0.910286 | 39.3% | 51.2% | 9.5% |
| startup_surge | 104 | 95.2% | 0.857986 | 0.907809 | 44.2% | 42.3% | 13.5% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 96.4% | 0.862649 |
| 1 | 162 | 97.5% | 0.858972 |
| 2 | 116 | 92.2% | 0.861622 |
| 3 | 17 | 100.0% | 0.860745 |
