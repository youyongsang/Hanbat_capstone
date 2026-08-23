# Raspberry Pi Inference Analysis

- input: `pi_ap_dynamic_fp32_results.csv`
- samples: 82
- measurement repeats: 5
- accuracy: 91.5%
- avg inference: 1.699441 ms
- p95 inference: 2.830281 ms
- avg sample std: 0.318772 ms
- avg confidence: 0.773336

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31 | 37.8% | 100.0% | 0.693559 | 0.807533 | 0.930472 |
| 2 | 32 | 39.0% | 93.8% | 1.922579 | 2.506308 | 0.757783 |
| 3 | 19 | 23.2% | 73.7% | 2.964806 | 4.388334 | 0.543151 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_load | 17 | 100.0% | 1.954325 | 2.512700 | 0.0% | 100.0% | 0.0% |
| low_load | 18 | 100.0% | 0.825898 | 1.697915 | 88.9% | 11.1% | 0.0% |
| medium_load | 13 | 84.6% | 1.905515 | 2.501520 | 0.0% | 100.0% | 0.0% |
| normal_idle | 15 | 100.0% | 0.677430 | 0.696803 | 100.0% | 0.0% | 0.0% |
| stress_load | 19 | 73.7% | 2.964806 | 4.388334 | 0.0% | 0.0% | 100.0% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 15 | 100.0% | 0.677430 |
| 1 | 30 | 96.7% | 1.275194 |
| 2 | 22 | 77.3% | 2.083684 |
| 3 | 15 | 93.3% | 3.006388 |
