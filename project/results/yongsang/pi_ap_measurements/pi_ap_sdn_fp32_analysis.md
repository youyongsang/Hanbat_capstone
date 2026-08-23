# Raspberry Pi Inference Analysis

- input: `pi_ap_sdn_fp32_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 2.443507 ms
- p95 inference: 4.312932 ms
- avg sample std: 0.641967 ms
- avg confidence: 0.838980

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 12.2% | 100.0% | 0.703192 | 0.758018 | 0.873420 |
| 2 | 25 | 30.5% | 100.0% | 1.818944 | 2.454762 | 0.917221 |
| 3 | 47 | 57.3% | 85.1% | 3.146002 | 4.316339 | 0.790035 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 94.1% | 2.982164 | 4.309285 | 0.0% | 11.8% | 88.2% |
| low_load | 18 | 100.0% | 1.179944 | 1.821743 | 55.6% | 44.4% | 0.0% |
| medium_load | 13 | 84.6% | 2.987814 | 4.320308 | 0.0% | 0.0% | 100.0% |
| normal_idle | 15 | 100.0% | 1.847127 | 2.463448 | 0.0% | 100.0% | 0.0% |
| stress_load | 19 | 79.0% | 3.257018 | 4.389019 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 1.847127 |
| 1 | 30 | 96.7% | 1.905394 |
| 2 | 22 | 72.7% | 3.074903 |
| 3 | 15 | 100.0% | 3.190067 |
