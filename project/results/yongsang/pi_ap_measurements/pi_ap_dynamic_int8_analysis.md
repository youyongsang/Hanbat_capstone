# Raspberry Pi Inference Analysis

- input: `pi_ap_dynamic_int8_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 1.185654 ms
- p95 inference: 1.950874 ms
- avg sample std: 0.280928 ms
- avg confidence: 0.773703

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31 | 37.8% | 100.0% | 0.545239 | 0.554836 | 0.930351 |
| 2 | 32 | 39.0% | 93.8% | 1.310421 | 1.938795 | 0.758166 |
| 3 | 19 | 23.2% | 73.7% | 2.020407 | 3.429187 | 0.544286 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 1.292593 | 1.937786 | 0.0% | 100.0% | 0.0% |
| low_load | 18 | 100.0% | 0.625938 | 1.227996 | 88.9% | 11.1% | 0.0% |
| medium_load | 13 | 84.6% | 1.337506 | 1.928424 | 0.0% | 100.0% | 0.0% |
| normal_idle | 15 | 100.0% | 0.547156 | 0.560347 | 100.0% | 0.0% | 0.0% |
| stress_load | 19 | 73.7% | 2.020407 | 3.429187 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 0.547156 |
| 1 | 30 | 96.7% | 0.911796 |
| 2 | 22 | 77.3% | 1.461135 |
| 3 | 15 | 93.3% | 1.967828 |
