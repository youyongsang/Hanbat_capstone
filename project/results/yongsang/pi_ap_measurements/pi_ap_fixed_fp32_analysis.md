# Raspberry Pi Inference Analysis

- input: `pi_ap_fixed_fp32_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 2.286111 ms
- p95 inference: 4.223325 ms
- avg sample std: 0.530205 ms
- avg confidence: 0.773949

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 15.8% | 100.0% | 0.874092 | 1.453056 | 0.951050 |
| 2 | 27 | 32.9% | 100.0% | 1.962721 | 3.058378 | 0.898156 |
| 3 | 42 | 51.2% | 83.3% | 2.931058 | 4.262858 | 0.639285 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 2.643594 | 3.521053 | 0.0% | 41.2% | 58.8% |
| low_load | 18 | 100.0% | 1.194320 | 1.828621 | 66.7% | 33.3% | 0.0% |
| medium_load | 13 | 84.6% | 2.880823 | 3.602971 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.780410 | 2.463201 | 6.7% | 93.3% | 0.0% |
| stress_load | 19 | 73.7% | 2.992915 | 4.261929 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.780410 |
| 1 | 30 | 96.7% | 1.873549 |
| 2 | 22 | 77.3% | 2.697013 |
| 3 | 15 | 93.3% | 3.014279 |
