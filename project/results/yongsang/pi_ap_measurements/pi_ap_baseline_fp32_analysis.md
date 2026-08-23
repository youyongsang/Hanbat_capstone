# Raspberry Pi Inference Analysis

- input: `pi_ap_baseline_fp32_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 92.7%
- avg inference: 1.837473 ms
- p95 inference: 1.844310 ms
- avg sample std: 0.046348 ms
- avg confidence: 0.902792

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 82 | 100.0% | 92.7% | 1.837473 | 1.844310 | 0.902792 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 1.831377 | 1.843191 | 0.0% | 0.0% | 100.0% |
| low_load | 18 | 100.0% | 1.831449 | 1.843268 | 0.0% | 0.0% | 100.0% |
| medium_load | 13 | 84.6% | 1.864836 | 2.019908 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.834554 | 1.844223 | 0.0% | 0.0% | 100.0% |
| stress_load | 19 | 79.0% | 1.832219 | 1.840447 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.834554 |
| 1 | 30 | 96.7% | 1.846177 |
| 2 | 22 | 77.3% | 1.830471 |
| 3 | 15 | 100.0% | 1.833257 |
