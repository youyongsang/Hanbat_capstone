# Claude용 프로젝트 맥락 정리

이 문서는 Claude 또는 외부 AI 도구에 현재 프로젝트 맥락을 전달하기 위한 요약 파일이다. 기존 1학기 실험, 방학 중 작업, 현재 AP 실측 strict 데이터 기준 진행 상태를 구분해서 설명한다.

## 프로젝트 개요

본 프로젝트는 산업 무선망 또는 AP 기반 무선 환경에서 트래픽 혼잡 상태를 시계열 feature로 판단하는 LSTM 기반 분류 시스템이다. 최종 목표는 Raspberry Pi 같은 엣지 장비에서 실시간 혼잡 수준을 추론하고, Early Exit 및 ONNX/INT8 경량화를 통해 추론 지연을 줄이는 것이다.

혼잡 라벨은 4단계이다.

| Label | 의미 |
|---|---|
| 0 | 정상 |
| 1 | 경고 |
| 2 | 혼잡 |
| 3 | 심각 |

## 1학기 실험 요약

1학기에는 `project/data/real` 기준의 4-feature 데이터셋을 사용했다. 당시 입력 feature는 RPS, channel occupancy, packet loss, latency 계열이었다.

주요 모델 비교는 다음 4개 방식이었다.

| 모델 | 역할 |
|---|---|
| Threshold 규칙 | 채널 점유율 기준 규칙 baseline |
| Baseline LSTM Full | Early Exit 없는 3-layer LSTM |
| Early Exit Fixed theta | 고정 threshold 기반 Early Exit |
| Early Exit Dynamic theta | 최근 채널 점유율 변화 기반 동적 threshold Early Exit |

1학기 대표 PC 결과는 다음과 같다.

| 모델 | 정확도 | 평균 추론 시간 | Label 2 정확도 | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|
| Threshold 규칙 | 42.2% | 0.0043ms | 37.9% | - | - | - |
| LSTM Full | 97.2% | 0.5915ms | 94.8% | - | - | - |
| EE Fixed | 95.7% | 0.5098ms | 91.4% | 20.5% | 71.8% | 7.7% |
| EE Dynamic | 96.3% | 0.5203ms | 94.0% | 25.6% | 69.5% | 4.8% |

용상 Early Exit 단독 리포트 기준으로는 다음 결과가 있었다.

| 모델 | 정확도 | 구조상 평균 지연 | PC wall-time |
|---|---:|---:|---:|
| EE Fixed | 95.7% | 3.897ms | 0.416ms |
| EE Dynamic | 96.3% | 3.681ms | 0.399ms |

해석:

- LSTM Full이 정확도는 가장 높았다.
- Dynamic theta는 Fixed보다 Exit3 비율을 줄이고 구조상 평균 지연을 낮췄다.
- 다만 PC Python 환경에서는 분기 처리와 threshold 계산 오버헤드 때문에 wall-time 우위가 항상 명확하지 않았다.

## 1학기 SDN-style 논문 baseline

호중 브랜치에는 SDN/Shallow-Deep Networks 논문 정책을 기반으로 한 별도 `SDNLSTM` 구현이 존재한다.

관련 파일:

```text
project/models/sdn_lstm.py
project/scripts/train_sdn.py
project/scripts/evaluate_sdn.py
project/scripts/export_onnx_sdn.py
project/scripts/export_onnx_int8_sdn.py
project/checkpoints/sdn_lstm_best.pth
project/results/yongsang/sdn_eval_report.txt
project/results/hojung/sdn_baseline.csv
```

1학기 SDN-style LSTM 결과:

| 모델 | 정확도 | 구조상 평균 | PC wall-time | Exit1 | Exit2 | Exit3 |
|---|---:|---:|---:|---:|---:|---:|
| SDN-style LSTM | 94.9% | 3.766ms | 0.707ms | 44.7% | 38.7% | 16.5% |

주의:

- 이것은 SDN 원 논문 전체를 그대로 재현한 것이 아니라, SDN의 핵심 정책인 internal classifier + confidence threshold early exit를 우리 LSTM 문제에 맞게 재구현한 baseline이다.
- threshold는 confidence 기준 `0.85`였다.

## 방학 중 변경 방향

방학 중에는 교수님 피드백에 따라 실험 방향이 다음처럼 바뀌었다.

1. 기존 가공 데이터 또는 1학기 데이터만 쓰지 않고, 실제 AP 장비에서 CSV를 수집한다.
2. AP 실측 데이터에 맞는 feature를 새로 구성한다.
3. 라벨을 채널 점유율 하나만 보는 방식에서 벗어나, throughput, occupancy, retry, jitter 등을 종합한 congestion score 기반으로 만든다.
4. 최종 실험은 PC 결과만으로 끝내지 않고 Raspberry Pi + ONNX + INT8 환경에서 실측한다.
5. 비교 대상은 Baseline LSTM, SDN-style baseline, Proposed EE Fixed, Proposed EE Dynamic으로 정리한다.

## 현재 AP strict 데이터 기준

현재 최종 기준으로 잡은 AP 실측 데이터셋은 다음 경로에 있다.

```text
project/data/ap_metrics_cleaned_strict/
├── raw/
│   ├── metrics_cleaned_strict.csv
│   └── metrics_cleaned_strict_report.txt
├── train.csv
├── val.csv
├── test.csv
├── scaler_params.json
├── dataset_summary.json
└── conversion_report.txt
```

원본 기준 CSV는 `metrics_cleaned_normal_idle_outliers_removed.csv`와 동일한 내용이다. `normal_idle` 구간의 비정상 throughput/retry outlier를 제거했다.

라벨 분포:

| Label | 의미 | 개수 |
|---|---|---:|
| 0 | 정상 | 116 |
| 1 | 경고 | 213 |
| 2 | 혼잡 | 152 |
| 3 | 심각 | 107 |

## 현재 AP 입력 feature

AP strict 모델 입력은 9개 feature이다.

```text
throughput_mbps
channel_occupancy_percent
latency_ms
jitter_ms
tx_retries_delta
tx_failed_delta
rssi_dbm
rssi_delta_db
rssi_moving_avg_dbm
```

### 4-feature에서 9-feature로 늘린 이유

1학기 4-feature 실험은 RPS, channel occupancy, packet loss, latency처럼 비교적 단순한 지표 조합을 사용했다. 하지만 AP 실측 환경에서는 기존 feature만으로 혼잡 상태를 안정적으로 설명하기 어려웠다.

주요 이유는 다음과 같다.

1. `channel_occupancy_percent`가 실제 부하를 줘도 40~46% 근처에서 크게 변하지 않는 구간이 있었다.
2. `packet_loss_udp_percent`는 이번 AP 측정에서 대부분 0 또는 N/A에 가까워 구분력이 낮았다.
3. 실측 AP에서는 혼잡이 단일 지표가 아니라 throughput 변화, 채널 점유율, jitter, retry/failed, RSSI 변화가 함께 나타나는 현상에 가깝다.
4. 따라서 모델이 채널 점유율 하나에만 의존하지 않도록, 트래픽 양, 무선 채널 상태, 전송 실패/재시도, 신호 세기 변화를 함께 입력해야 했다.

9개 feature는 아래 의미를 나눠서 반영한다.

| feature 그룹 | 포함 feature | 반영 의도 |
|---|---|---|
| 트래픽 부하 | `throughput_mbps` | 실제 전송량 증가 여부 |
| 채널 사용 상태 | `channel_occupancy_percent` | 무선 채널 busy 비율 |
| 지연 품질 | `latency_ms`, `jitter_ms` | 응답 지연 및 변동성 |
| 전송 안정성 | `tx_retries_delta`, `tx_failed_delta` | 재전송/실패 증가 여부 |
| 무선 신호 상태 | `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm` | 신호 세기와 시간적 변화 |

즉 9-feature 전환은 입력을 무작정 늘린 것이 아니라, AP 실측에서 단일 지표로 혼잡을 설명하기 어렵다는 문제를 보완하기 위한 변경이다.

아래 컬럼은 모델 입력에 넣지 않는다.

| 컬럼 | 용도 | 제외 이유 |
|---|---|---|
| `timestamp` | 측정 시각 | 시간 자체가 혼잡 원인이 아니며 과적합 위험 |
| `scenario` | 측정 상황명 | 상황명을 보고 맞히는 문제가 될 수 있음 |
| `channel_occupancy_method` | 점유율 계산 방식 | 측정 메타데이터 |
| `packet_loss_udp_percent` | UDP 손실률 | 현재 데이터에서 신뢰 가능한 변동 부족 |
| `connected_clients` | 연결 클라이언트 수 | 이번 데이터에서는 구분력이 제한적 |
| `congestion_score` | 라벨 생성용 종합 점수 | 정답 생성 중간값이므로 입력 시 데이터 누수 |
| `label` | 정답 라벨 | 모델이 맞혀야 하는 정답 |

정규화는 train split의 min-max 기준이다. `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 써야 한다.

## 현재 AP 코드 및 결과

AP strict 기준으로 새로 추가된 핵심 파일:

```text
project/README_AP_STRICT.md
project/utils/ap_features.py
project/utils/ap_dataloader.py
project/models/ap_early_exit_lstm.py
project/models/ap_sdn_lstm.py
project/scripts/prepare_ap_metrics_dataset.py
project/scripts/train_ap_baseline_lstm.py
project/scripts/train_ap_early_exit.py
project/scripts/evaluate_ap_early_exit.py
project/scripts/train_ap_sdn.py
project/scripts/evaluate_ap_sdn.py
project/scripts/generate_ap_comparison.py
```

AP strict checkpoint:

```text
project/checkpoints/ap_cleaned_strict/ap_baseline_lstm_best.pth
project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth
project/checkpoints/ap_cleaned_strict/ap_early_exit_lstm_best.pth
project/checkpoints/ap_cleaned_strict/ap_early_exit_fixed.pth
project/checkpoints/ap_cleaned_strict/ap_early_exit_dynamic.pth
```

AP strict 결과:

```text
project/results/yongsang/ap_baseline_lstm_cleaned_strict_eval_report.txt
project/results/yongsang/ap_sdn_cleaned_strict_eval_report.txt
project/results/yongsang/ap_early_exit_cleaned_strict_eval_report.txt
project/results/yongsang/ap_model_comparison_cleaned_strict.csv
project/results/yongsang/ap_model_comparison_cleaned_strict.txt
```

현재 PC 비교 결과:

| 모델 | 정확도 | Label 2 | Label 3 | Exit1 | Exit2 | Exit3 | PC 실측 | 구조상 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline LSTM Full | 92.7% | 77.3% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0667ms | 8.000ms |
| SDN-style Early Exit (trained) | 91.5% | 72.7% | 100.0% | 12.2% | 30.5% | 57.3% | 0.4956ms | 6.049ms |
| Proposed Early Exit Fixed theta | 91.5% | 77.3% | 93.3% | 15.9% | 32.9% | 51.2% | 0.5143ms | 5.732ms |
| Proposed Early Exit Dynamic theta | 91.5% | 77.3% | 93.3% | 37.8% | 39.0% | 23.2% | 0.4251ms | 4.171ms |

주의:

- `SDN-style Early Exit (trained)`는 더 이상 임시 confidence-only 스크립트가 아니라, `project/models/ap_sdn_lstm.py` + `project/scripts/train_ap_sdn.py`로 AP 9-feature 데이터에서 별도 학습한 결과이다(체크포인트: `project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth`).
- Early Exit 백본을 재사용하지 않고 SDN 논문의 loss 가중치(0.15/0.30/0.55)로 독립 학습했으므로 방어 가능한 비교다.
- 전체 정확도는 Proposed와 동일한 91.5%이지만 Label 2(혼잡) 정확도가 72.7%로 Proposed(77.3%)보다 4.6%p 낮다. 이 격차가 제안 모델의 실질적 차별점이다.

## Fixed/Dynamic 실험 방식

Fixed와 Dynamic은 별도의 완전히 다른 모델을 학습하는 것이 아니다.

1학기와 현재 AP strict 실험 모두 다음 방식이다.

1. 하나의 Early Exit LSTM backbone을 학습한다.
2. Fixed 평가에서는 고정 threshold를 사용한다.
3. Dynamic 평가에서는 최근 channel occupancy 또는 traffic 변화량을 보고 threshold를 조정한다.

따라서 `ap_early_exit_fixed.pth`와 `ap_early_exit_dynamic.pth`는 서로 다른 학습 결과라기보다, 같은 backbone 가중치에 fixed/dynamic 설정 메타를 분리해 저장한 파일로 보면 된다. 실제 학습된 best 가중치는 `ap_early_exit_lstm_best.pth`이다.

## 자주 헷갈리는 점

### 기존 compare_baselines.py를 AP strict에 그대로 쓰면 안 되는 이유

`project/experiments/compare_baselines.py`는 1학기 4-feature `project/data/real` 기준 코드이다. AP strict 데이터는 9-feature이고 `utils/ap_dataloader.py`, `utils/ap_features.py`를 사용해야 한다.

### 정확도가 58% 근처로 나올 때 의심할 점

1. `project/data/real/test.csv` 같은 1학기 test.csv를 잘못 사용했는지 확인한다.
2. raw CSV를 그대로 넣지 않았는지 확인한다. AP 모델은 window size 10으로 변환된 `test.csv`를 사용한다.
3. feature 순서가 9개 정의와 동일한지 확인한다.
4. `label`, `congestion_score`, `scenario`, `timestamp`가 입력 feature에 섞이지 않았는지 확인한다.
5. 4-feature용 checkpoint를 9-feature AP 데이터에 사용하지 않았는지 확인한다.
6. `scaler_params.json` 기준이 다른지 확인한다.

## AP strict 재현 명령어

데이터 변환:

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\data\ap_metrics_cleaned_strict\raw\metrics_cleaned_strict.csv --out-dir project\data\ap_metrics_cleaned_strict --overwrite --no-occupancy-outlier-fix
```

Baseline LSTM 학습:

```powershell
python project\scripts\train_ap_baseline_lstm.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint-dir project\checkpoints\ap_cleaned_strict --epochs 20 --batch-size 32
```

Early Exit LSTM 학습:

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint-dir project\checkpoints\ap_cleaned_strict --epochs 20 --batch-size 32
```

Early Exit 평가:

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint project\checkpoints\ap_cleaned_strict\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_early_exit_cleaned_strict_eval_report.txt
```

SDN-style LSTM 학습:

```powershell
python project\scripts\train_ap_sdn.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint-dir project\checkpoints\ap_cleaned_strict --epochs 20 --batch-size 32
```

SDN-style LSTM 평가:

```powershell
python project\scripts\evaluate_ap_sdn.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint project\checkpoints\ap_cleaned_strict\ap_sdn_lstm_best.pth --output project\results\yongsang\ap_sdn_cleaned_strict_eval_report.txt
```

비교표 생성:

```powershell
python project\scripts\generate_ap_comparison.py
```

## Raspberry Pi 실측 전 남은 작업

AP strict 최종 실험을 위해 다음 작업이 남아 있다.

1. ~~AP용 SDN LSTM checkpoint 재학습~~ — 완료 (`ap_sdn_lstm_best.pth`)
2. Baseline LSTM ONNX 변환
3. SDN-style staged ONNX 변환
4. Proposed Fixed/Dynamic staged ONNX 변환
5. FP32 ONNX와 INT8 ONNX 생성
6. `project/deploy/raspberry_pi/` 배포 번들 갱신
7. Pi에서 동일 `test.csv` 기준으로 평균, p50, p95, Exit 비율 재측정

Early Exit의 실제 속도 이득을 보려면 반드시 staged ONNX 방식이 필요하다.

```text
stage1.onnx 실행
조건 만족 시 종료
조건 미충족 시 stage2.onnx 실행
조건 만족 시 종료
조건 미충족 시 stage3.onnx 실행
```

하나의 ONNX 모델을 끝까지 실행한 뒤 exit point만 기록하면 실제 layer skip 효과를 측정할 수 없다.

## 현재 결론

현재 상태는 다음처럼 정리할 수 있다.

- AP 실측 strict 데이터 기준 4단계 혼잡 분류는 동작한다.
- Baseline LSTM 정확도는 92.7%이고, Proposed EE Dynamic은 91.5%로 정확도 손실은 1.2%p 수준이다.
- Label 2 혼잡 정확도는 77.3%로, 1학기 데이터보다 낮다. 이는 실제 AP 데이터가 더 어렵고 라벨 경계가 덜 깔끔하기 때문이다.
- PC Python 환경에서는 Early Exit 속도 우위를 강하게 주장하기 어렵다.
- 최종 주장은 Raspberry Pi + staged ONNX + INT8 실측 결과로 확정해야 한다.

## Claude가 추가로 참고해야 할 파일

Claude가 이 프로젝트를 처음 보는 경우, 아래 순서대로 파일을 확인하면 현재 맥락을 가장 빨리 이해할 수 있다.

### 1. 전체 프로젝트 개요

```text
README.md
```

역할:

- 프로젝트의 원래 목표와 1학기 전체 파이프라인을 설명한다.
- `project/data/real` 기반 4-feature 실험, ONNX/INT8, Raspberry Pi 배포 흐름을 담고 있다.
- 단, 최신 AP strict 실험은 별도 문서인 `project/README_AP_STRICT.md`를 우선 참고해야 한다.

### 2. 현재 AP strict 실험 기준 문서

```text
project/README_AP_STRICT.md
```

역할:

- 현재 기준 데이터셋인 `project/data/ap_metrics_cleaned_strict`를 설명한다.
- 9개 입력 feature, 제외 컬럼, 데이터 변환, 학습, 평가, 비교표 생성 명령어를 정리한다.
- 호중이 또는 다른 팀원이 AP strict 데이터로 실험을 재현하려면 이 문서를 먼저 봐야 한다.

### 3. 1학기 결과와 해석

```text
docs/yongsang/result_text_analysis.md
project/results/hojung/comparison_summary.txt
project/results/hojung/comparison_summary.csv
project/results/yongsang/early_exit_stage2_comparison_report.txt
```

역할:

- 1학기 4-feature 데이터 기준 PC 결과를 확인할 수 있다.
- Baseline LSTM, Early Exit Fixed, Early Exit Dynamic의 정확도, 추론 시간, Exit 비율이 정리되어 있다.
- `docs/yongsang/result_text_analysis.md`는 결과를 어떻게 해석해야 하는지 설명한다.

### 4. SDN-style 논문 baseline 확인

1학기 4-feature 버전은 호중 브랜치 또는 원격 `origin/hojung` 기준으로 아래 파일을 확인한다.

```text
project/models/sdn_lstm.py
project/scripts/train_sdn.py
project/scripts/evaluate_sdn.py
project/results/yongsang/sdn_eval_report.txt
project/results/hojung/sdn_baseline.csv
```

AP strict 9-feature 버전은 `yongsang` 브랜치에 이미 포팅 및 재학습이 완료되어 있다.

```text
project/models/ap_sdn_lstm.py
project/scripts/train_ap_sdn.py
project/scripts/evaluate_ap_sdn.py
project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth
project/results/yongsang/ap_sdn_cleaned_strict_eval_report.txt
```

역할:

- SDN/Shallow-Deep Networks(ICML 2019) 논문 정책을 LSTM에 맞게 재구현한 baseline이다.
- confidence threshold(0.85) 기반 early exit를 사용하며, SDN 고유의 loss 가중치(0.15/0.30/0.55)로 별도 학습한다.
- AP strict용 `APSDNLSTM`은 `APEarlyExitLSTM`과 동일한 패턴으로 `SDNLSTM`을 상속해 `input_size`만 9로 바꾼 구조다. Early Exit 백본을 재사용하지 않으므로 방어 가능한 비교다.

### 5. AP strict 데이터와 현재 결과 파일

```text
project/data/ap_metrics_cleaned_strict/dataset_summary.json
project/data/ap_metrics_cleaned_strict/conversion_report.txt
project/data/ap_metrics_cleaned_strict/scaler_params.json
project/results/yongsang/ap_baseline_lstm_cleaned_strict_eval_report.txt
project/results/yongsang/ap_sdn_cleaned_strict_eval_report.txt
project/results/yongsang/ap_early_exit_cleaned_strict_eval_report.txt
project/results/yongsang/ap_model_comparison_cleaned_strict.txt
project/results/yongsang/ap_model_comparison_cleaned_strict.csv
```

역할:

- 현재 AP strict 데이터의 split, feature scaling, label 분포, 모델 평가 결과를 확인할 수 있다.
- `ap_model_comparison_cleaned_strict.*`는 Baseline / SDN-style(trained) / Proposed Fixed / Proposed Dynamic 4개 비교표를 담고 있다.

### 6. AP strict 코드 흐름

```text
project/utils/ap_features.py
project/utils/ap_dataloader.py
project/models/ap_early_exit_lstm.py
project/models/ap_sdn_lstm.py
project/scripts/prepare_ap_metrics_dataset.py
project/scripts/train_ap_baseline_lstm.py
project/scripts/train_ap_early_exit.py
project/scripts/evaluate_ap_early_exit.py
project/scripts/train_ap_sdn.py
project/scripts/evaluate_ap_sdn.py
project/scripts/generate_ap_comparison.py
```

역할:

- AP strict 9-feature 데이터가 모델에 들어가는 실제 코드 흐름이다.
- 1학기 코드와 달리 `utils/ap_features.py`의 feature 순서가 중요하다.
- 58% 근처의 낮은 정확도가 나오면 이 파일들의 feature 순서, scaler, checkpoint 경로를 먼저 점검해야 한다.

### 7. Raspberry Pi 및 ONNX 과거 실험

```text
docs/hochung/stage5_pi_execution_order.md
docs/hochung/stage5_work_log.md
project/deploy/raspberry_pi/README.md
project/results/hojung/pi_fp32_analysis.txt
project/results/hojung/pi_int8_analysis.txt
project/results/hojung/pi_fixed_staged_fp32_analysis.txt
project/results/hojung/pi_fixed_staged_int8_analysis.txt
project/results/hojung/pi_dynamic_staged_fp32_analysis.txt
project/results/hojung/pi_dynamic_staged_int8_analysis.txt
```

역할:

- 1학기 또는 기존 데이터 기준 Raspberry Pi 실측 흐름과 결과를 확인할 수 있다.
- FP32 ONNX, INT8 ONNX, staged ONNX 실행 구조가 들어 있다.
- AP strict 최종 실험에서도 이 흐름을 재사용하되, 데이터와 checkpoint는 AP strict 기준으로 다시 만들어야 한다.

### 8. 방학 중 작업 계획과 활동 기록

```text
docs/vacation_activity_overview.md
docs/yongsang/guideline_yongsang_vacation_stage1.md
docs/yongsang/guideline_yongsang_vacation_stage3.md
docs/yongsang/guideline_yongsang_vacation_stage4.md
docs/hochung/guideline_hochung_vacation_stage3.md
docs/hochung/guideline_hochung_vacation_stage4.md
docs/hochung/guideline_hochung_vacation_stage5.md
```

역할:

- 방학 중 역할 분담과 작업 방향을 확인할 수 있다.
- 교수님 피드백에 따라 논문 baseline, AP 실측, ONNX/INT8, Raspberry Pi 실험으로 방향이 이동한 흐름을 이해하는 데 도움이 된다.

### 9. Claude에게 중요한 해석 기준

Claude가 답변할 때는 아래 기준을 지켜야 한다.

1. 1학기 결과와 AP strict 결과를 섞어서 말하지 않는다.
2. 1학기 `project/data/real`은 4-feature이고, AP strict는 9-feature이다.
3. 기존 `compare_baselines.py`는 AP strict 데이터에 그대로 쓰면 안 된다.
4. AP strict의 `SDN-style Early Exit (trained)` 결과는 더 이상 임시 비교가 아니라, `ap_sdn_lstm.py`/`train_ap_sdn.py`로 AP 9-feature 데이터에서 별도 학습한 최종 checkpoint(`ap_sdn_lstm_best.pth`) 기준이다.
5. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책을 바꿔 평가하는 구조이다.
6. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다.
7. 최종 속도 주장은 Raspberry Pi + staged ONNX + INT8 실측으로 확정해야 한다.
