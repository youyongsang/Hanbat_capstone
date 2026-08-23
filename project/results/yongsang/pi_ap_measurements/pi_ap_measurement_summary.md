# Raspberry Pi AP Strict (9-feature) 실측 결과

측정일: 2026-08-23
Pi: capstone@CapsTone (192.168.45.31), Python 3.13.5, onnxruntime 1.29.0
데이터: `project/deploy/raspberry_pi_ap/test.csv` (82 samples), repeats=5
번들: `project/deploy/raspberry_pi_ap/` (staged ONNX 실행 방식 — layer skip 실제 반영)

| 모델 | 정확도 | 평균(ms) | p50(ms) | p95(ms) | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LSTM FP32 | 92.7% | 1.837 | 1.832 | 1.844 | - | - | 100% |
| Baseline LSTM INT8 | 92.7% | 1.132 | 1.132 | 1.144 | - | - | 100% |
| SDN-style FP32 | 91.5% | 2.444 | 2.690 | 4.313 | 12.20% | 30.49% | 57.32% |
| SDN-style INT8 | 91.5% | 1.497 | 1.759 | 1.906 | 12.20% | 30.49% | 57.32% |
| Proposed Fixed theta FP32 | 91.5% | 2.286 | 2.683 | 4.223 | 15.85% | 32.93% | 51.22% |
| Proposed Fixed theta INT8 | 91.5% | 1.470 | 1.731 | 2.020 | 14.63% | 34.15% | 51.22% |
| Proposed Dynamic theta FP32 | 91.5% | 1.699 | 1.727 | 2.830 | 37.80% | 39.02% | 23.17% |
| Proposed Dynamic theta INT8 | 91.5% | 1.186 | 1.158 | 1.951 | 37.80% | 39.02% | 23.17% |

## 해석

- 정확도는 PC 평가 결과와 완전히 일치 (Baseline 92.7%, 나머지 91.5%) — 동일 checkpoint/test.csv/9-feature 파이프라인 확인됨.
- Pi + staged ONNX 환경에서는 PC와 달리 Early Exit 속도 이득이 뚜렷하게 관찰됨:
  - Proposed Dynamic FP32(1.699ms)는 Baseline FP32(1.837ms)보다 평균 7.5% 빠름.
  - Proposed Dynamic INT8(1.186ms)는 Baseline INT8(1.132ms)보다 근소하게 느림 — INT8 자체가 이미 매우 가벼워 staged 오버헤드(다중 세션 로드/분기 처리)가 layer-skip 이득을 상쇄.
  - Dynamic theta는 Fixed theta 대비 Exit1 비율이 크게 높음(37.8% vs 15.9%)으로 초기 종료가 많아 평균 지연이 더 낮음(FP32 기준 1.699ms vs 2.286ms).
  - SDN-style은 Exit3 비율이 가장 높아(57.3%) Proposed 대비 평균 지연이 더 큼.
- Baseline은 staged 구조가 아니라 항상 Exit3(full-depth)이므로 Exit 비율이 의미 없음(100% 고정).

## 원본 파일

- `pi_ap_*_results.csv`: 샘플별 원시 측정 결과
- `pi_ap_*_analysis.txt` / `.md`: 텍스트 리포트
- `pi_ap_*_analysis_overall.csv`: 정확도/지연 통계(평균/최소/최대/p50/p95)
- `pi_ap_*_analysis_by_label.csv`: 라벨별 정확도
- `pi_ap_*_analysis_by_exit.csv`: exit stage별 분포/정확도/지연
- `pi_ap_*_analysis_by_scenario.csv`: 시나리오별 분석
