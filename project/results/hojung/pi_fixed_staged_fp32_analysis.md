# Raspberry Pi Inference Analysis

- input: `project\results\hojung\pi_fixed_staged_fp32_results.csv`
- samples: 351
- measurement repeats: 5
- accuracy: 97.4%
- avg inference: 2.088706 ms
- p95 inference: 3.311013 ms
- avg sample std: 0.791140 ms
- avg confidence: 0.921462

## Exit Summary

| Exit | Samples | Rate | Accuracy | Avg ms | P95 ms | Avg confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 4.6% | 100.0% | 0.909849 | 2.254166 | 0.946785 |
| 2 | 300 | 85.5% | 100.0% | 2.034934 | 3.274193 | 0.945170 |
| 3 | 35 | 10.0% | 74.3% | 3.088507 | 4.222277 | 0.706670 |

## Scenario Summary

| Scenario | Samples | Accuracy | Avg ms | P95 ms | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergency_ramp | 83 | 95.2% | 2.158795 | 3.313501 | 3.6% | 85.5% | 10.8% |
| imbalanced_ap_load | 80 | 97.5% | 2.183544 | 4.116218 | 2.5% | 87.5% | 10.0% |
| lunch_restart | 84 | 98.8% | 1.979598 | 3.258118 | 10.7% | 78.6% | 10.7% |
| startup_surge | 104 | 98.1% | 2.047941 | 3.265266 | 1.9% | 89.4% | 8.6% |

## Label Summary

| True label | Samples | Accuracy | Avg ms |
|---:|---:|---:|---:|
| 0 | 56 | 100.0% | 1.854820 |
| 1 | 162 | 98.2% | 2.126017 |
| 2 | 116 | 94.8% | 2.157168 |
| 3 | 17 | 100.0% | 2.036445 |
