# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fp32_results.csv`
- samples: 351
- accuracy: 95.7%
- avg inference: 1.512202 ms
- p95 inference: 1.580113 ms
- avg confidence: 0.919691

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 41.3% | 100.0% | 1.497081 | 1.560952 | 0.917690 |
| 2 | 157 | 44.7% | 100.0% | 1.513439 | 1.583997 | 0.960373 |
| 3 | 49 | 14.0% | 69.4% | 1.552986 | 1.595793 | 0.795268 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 92.8% | 1.537863 | 1.581456 | 38.6% | 47.0% | 14.5% |
| imbalanced_ap_load | 80 | 97.5% | 1.516794 | 1.591568 | 41.2% | 42.5% | 16.2% |
| lunch_restart | 84 | 96.4% | 1.502248 | 1.565737 | 40.5% | 47.6% | 11.9% |
| startup_surge | 104 | 96.2% | 1.496230 | 1.577191 | 44.2% | 42.3% | 13.5% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 98.2% | 1.495003 |
| 1 | 162 | 97.5% | 1.510853 |
| 2 | 116 | 91.4% | 1.523324 |
| 3 | 17 | 100.0% | 1.505834 |
