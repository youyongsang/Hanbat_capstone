# Raspberry Pi Inference Analysis

- input: `pi_ap_fixed_int8_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 1.470089 ms
- p95 inference: 2.020222 ms
- avg sample std: 0.272842 ms
- avg confidence: 0.774453

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 14.6% | 100.0% | 0.621084 | 0.917340 | 0.952676 |
| 2 | 28 | 34.2% | 100.0% | 1.127207 | 1.168424 | 0.899869 |
| 3 | 42 | 51.2% | 83.3% | 1.941249 | 3.346599 | 0.639921 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 1.605226 | 2.102909 | 0.0% | 41.2% | 58.8% |
| low_load | 18 | 100.0% | 0.813585 | 1.170761 | 61.1% | 38.9% | 0.0% |
| medium_load | 13 | 84.6% | 1.897696 | 2.505872 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.085456 | 1.161379 | 6.7% | 93.3% | 0.0% |
| stress_load | 19 | 73.7% | 1.982211 | 3.367159 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.085456 |
| 1 | 30 | 96.7% | 1.254422 |
| 2 | 22 | 77.3% | 1.644882 |
| 3 | 15 | 93.3% | 2.029689 |
