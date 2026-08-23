# Raspberry Pi Inference Analysis

- input: `pi_ap_sdn_int8_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 1.496654 ms
- p95 inference: 1.906077 ms
- avg sample std: 0.231640 ms
- avg confidence: 0.838566

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 12.2% | 100.0% | 0.558009 | 0.588028 | 0.873876 |
| 2 | 25 | 30.5% | 100.0% | 1.218454 | 1.764898 | 0.917238 |
| 3 | 47 | 57.3% | 85.1% | 1.844345 | 1.892922 | 0.789207 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 94.1% | 1.984062 | 3.348707 | 0.0% | 11.8% | 88.2% |
| low_load | 18 | 100.0% | 0.805086 | 1.129209 | 55.6% | 44.4% | 0.0% |
| medium_load | 13 | 84.6% | 1.788206 | 1.864194 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.174874 | 1.390727 | 0.0% | 100.0% | 0.0% |
| stress_load | 19 | 79.0% | 1.770277 | 1.797241 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.174874 |
| 1 | 30 | 96.7% | 1.199801 |
| 2 | 22 | 72.7% | 1.934837 |
| 3 | 15 | 100.0% | 1.769474 |
