# Raspberry Pi Inference Analysis

- input: `pi_ap_baseline_int8_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 92.7%
- avg inference: 1.132252 ms
- p95 inference: 1.144162 ms
- avg sample std: 0.074834 ms
- avg confidence: 0.902716

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 82 | 100.0% | 92.7% | 1.132252 | 1.144162 | 0.902716 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 1.135158 | 1.153310 | 0.0% | 0.0% | 100.0% |
| low_load | 18 | 100.0% | 1.131198 | 1.142616 | 0.0% | 0.0% | 100.0% |
| medium_load | 13 | 84.6% | 1.131341 | 1.138745 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.134114 | 1.160569 | 0.0% | 0.0% | 100.0% |
| stress_load | 19 | 79.0% | 1.129803 | 1.138043 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.134114 |
| 1 | 30 | 96.7% | 1.130979 |
| 2 | 22 | 77.3% | 1.135063 |
| 3 | 15 | 100.0% | 1.128811 |
