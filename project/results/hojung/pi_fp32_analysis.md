# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fp32_results.csv`
- samples: 351
- accuracy: 95.7%
- avg inference: 1.529763 ms
- p95 inference: 1.561634 ms
- avg confidence: 0.919691

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145 | 41.3% | 100.0% | 1.518631 | 1.555024 | 0.917690 |
| 2 | 157 | 44.7% | 100.0% | 1.549901 | 1.561606 | 0.960373 |
| 3 | 49 | 14.0% | 69.4% | 1.498184 | 1.560910 | 0.795268 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 92.8% | 1.496946 | 1.555602 | 38.6% | 47.0% | 14.5% |
| imbalanced_ap_load | 80 | 97.5% | 1.550599 | 1.574950 | 41.2% | 42.5% | 16.2% |
| lunch_restart | 84 | 96.4% | 1.553271 | 1.562001 | 40.5% | 47.6% | 11.9% |
| startup_surge | 104 | 96.2% | 1.520940 | 1.541648 | 44.2% | 42.3% | 13.5% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 98.2% | 1.492753 |
| 1 | 162 | 97.5% | 1.511599 |
| 2 | 116 | 91.4% | 1.578597 |
| 3 | 17 | 100.0% | 1.491562 |
