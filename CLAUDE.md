# Claude용 프로젝트 맥락 정리

이 문서는 Claude 또는 외부 AI 도구에 현재 프로젝트 맥락을 전달하기 위한 요약 파일이다. 기존 1학기 실험, 방학 중 작업, 현재 AP 실측 strict 데이터(1차) 기준 진행 상태, 그리고 2026-08-23부터 시작된 실제 구매 장비 실측 재설계(2차, `ap_metrics_v2`) 진행 상태를 구분해서 설명한다. 1차와 2차의 정확한 구분은 아래 "데이터 계보" 섹션을 반드시 먼저 읽는다.

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

## 데이터 계보: 1차(ap_cleaned_strict, archived) vs 2차(ap_metrics_v2, 진행 중)

이 프로젝트에는 "AP 실측"이라는 이름을 쓰는 서로 다른 두 데이터 라인이 있다. 섞어서 말하면 안 된다.

**1차 = `ap_cleaned_strict`(588행).** 아래 "현재 AP strict 데이터 기준" ~ "Raspberry Pi 실측 전 남은 작업" 섹션이 전부 이 라인 기준이다. Baseline/SDN/Proposed Fixed/Dynamic 4개 모델 학습, ONNX/INT8 변환, Raspberry Pi 배포 번들까지 전부 완료된 상태다. **2026-08-23 확인: 이 데이터는 실제로 팀이 구매한 AP 장비의 실측이 아니라 인터넷에 공개돼 있던 데이터를 가공해서 만든 것**이었다. 즉 "방학 중 변경 방향"에서 말한 "실제 AP 장비에서 CSV 수집"이라는 목표는 1차 단계에서는 완전히 실현되지 않았다(이게 이 데이터셋의 feature 스케일이 실측 라이브 데이터와 크게 다른 이유이기도 하다 — `latency_ms` 원본 0.047~0.163 vs 실측 2~841ms 등). 그럼에도 9-feature 파이프라인, congestion score 라벨링, Early Exit/SDN 비교 구조 전체를 미리 검증하는 역할을 했으므로 그대로 archived 상태로 남겨둔다. **더 이상 재라벨링하거나 재학습하지 않는다** — 필요하면 아래 2차 라인에서 이어간다.

**2차 = `ap_metrics_v2`.** 2026-08-23부터 팀이 실제로 구매한 GL.iNet Opal(GL-SFT1200) AP에서 `project/scripts/collect_metrics.py`로 직접 라이브 수집한 진짜 실측 데이터다. "방학 중 변경 방향"의 취지(실제 AP 장비 실측)를 온전히 만족하는 라인이며, 현재 진행 중인 공식 재설계 라인이다. 자세한 내용은 아래 "2차 AP 실측 재설계(ap_metrics_v2)" 섹션과 `project/README_AP_V2.md`를 참고한다(1차의 `project/README_AP_STRICT.md`에 대응하는 문서).

두 라인은 **congestion_score 계산 가중치 자체가 다르므로 label 정의가 다르다.** 정확도나 label 분포를 같은 표에 놓고 직접 비교하지 않는다.

## 현재 AP strict 데이터 기준 (1차, archived)

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

ONNX 변환 (Baseline + Proposed Fixed/Dynamic, full + staged):

```powershell
python project\scripts\export_onnx_ap.py
```

ONNX 변환 (SDN-style, full + staged):

```powershell
python project\scripts\export_onnx_ap_sdn.py
```

INT8 양자화 (위 두 스크립트 실행 후):

```powershell
python project\scripts\export_onnx_int8_ap.py
```

Pi 배포 번들 생성:

```powershell
python project\scripts\prepare_pi_bundle_ap.py
```

## Raspberry Pi 실측 전 남은 작업

AP strict 최종 실험을 위해 다음 작업이 남아 있다.

1. ~~AP용 SDN LSTM checkpoint 재학습~~ — 완료 (`ap_sdn_lstm_best.pth`)
2. ~~Baseline LSTM ONNX 변환~~ — 완료 (`ap_baseline.onnx`)
3. ~~SDN-style staged ONNX 변환~~ — 완료 (`ap_sdn_fixed*.onnx`)
4. ~~Proposed Fixed/Dynamic staged ONNX 변환~~ — 완료 (`ap_early_exit_fixed/dynamic*.onnx`)
5. ~~FP32 ONNX와 INT8 ONNX 생성~~ — 완료 (`export_onnx_int8_ap.py`)
6. ~~`project/deploy/raspberry_pi_ap/` 배포 번들 갱신~~ — 완료 (1학기 4-feature용 `raspberry_pi/`와는 별도 폴더)
7. Pi에서 동일 `test.csv` 기준으로 평균, p50, p95, Exit 비율 재측정 — **아직 미완료. 다음 단계.**

Early Exit의 실제 속도 이득을 보려면 반드시 staged ONNX 방식이 필요하다.

`docs/hochung/Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx`에 이미 호중이 올린 Pi 실측 결과가 있지만, 그 표의 SDN 행은 오늘 재학습한 `ap_sdn_fixed.onnx`가 아니라 예전 confidence-only 재사용 모델 기준이라 최종본으로 쓰면 안 된다. `project/deploy/raspberry_pi_ap/`로 다시 측정해야 한다.

```text
stage1.onnx 실행
조건 만족 시 종료
조건 미충족 시 stage2.onnx 실행
조건 만족 시 종료
조건 미충족 시 stage3.onnx 실행
```

하나의 ONNX 모델을 끝까지 실행한 뒤 exit point만 기록하면 실제 layer skip 효과를 측정할 수 없다.

## 2차 AP 실측 재설계 (ap_metrics_v2)

2026-08-23부터 시작된 두 번째 데이터 라인이다. 1차(`ap_cleaned_strict`)와 달리 팀이 실제로 구매한 GL.iNet Opal(GL-SFT1200) AP에서 `project/scripts/collect_metrics.py`로 직접 라이브 수집하며, `metrics_v2.csv`는 계속 자라는 누적 원본이다.

### 파일 위치

```text
project/scripts/metrics_v2.csv          원본 raw 수집 CSV (누적, 계속 자람)
project/scripts/collect_metrics.py      라이브 수집 스크립트 (congestion_score 계산도 여기서)
project/scripts/relabel_metrics_v2.py   가중치가 바뀔 때 raw CSV를 재수집 없이 재라벨링
project/scripts/prepare_ap_metrics_dataset.py  windowed train/val/test 변환 (1차와 공용 스크립트)
project/data/ap_metrics_v2/             windowed train/val/test, scaler, dataset_summary.json
project/checkpoints/ap_v2/              Early Exit LSTM 체크포인트
project/results/yongsang/ap_v2_eval_report.txt              최신 평가 리포트
project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt  1차 체크포인트/스케일러로 2차 데이터를 잘못 평가했을 때의 진단 기록(역사적 자료, 최종 결과 아님)
```

### congestion_score 계산식 (1차와 다름)

```text
congestion_score = 0.20 * throughput_score + 0.45 * occupancy_score + 0.20 * retry_failed_score + 0.15 * jitter_score
```

1차 `ap_cleaned_strict`는 `0.35 * throughput + 0.35 * occupancy + 0.20 * retry + 0.10 * jitter`를 그대로 쓴다 — 서로 다른 라벨 기준이므로 섞어서 비교하지 않는다. label 경계 자체는 동일: `<0.25`→0, `0.25~0.50`→1, `0.50~0.75`→2, `≥0.75`→3.

가중치를 바꾼 이유: 실측 stress_load 구간에서 label 2와 3의 sub-score 평균을 비교해보니 `throughput_score`(0.665→0.707)는 거의 차이가 없었던 반면 `occupancy_score`(0.449→0.898)와 `jitter_score`(0.512→0.802)는 뚜렷한 차이를 보였다. throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 기여가 거의 없었으므로, occupancy·jitter 비중을 높이고 throughput 비중을 낮췄다. 이 재조정을 이미 모아둔 raw 데이터에 재적용(`relabel_metrics_v2.py`)한 것만으로 label 3이 21개→33개로 늘었다 — AP를 다시 부하 테스트하지 않고도 얻은 개선이다.

### class weight power = 1.0

`train_ap_early_exit.py --class-weight-power`(기본값 1.0, `compute_class_weights`)로 조절한다. 이 데이터셋에서는 완만한 트레이드오프가 아니라 절벽형이었다:

| power | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
| 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
| **1.0** | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

0.7/0.85는 label 3 recall이 계속 0%이고, power=1.0(순수 역빈도)에서만 label 3이 잡히기 시작하며 그 대신 label 2 recall이 하락한다. "심각(label 3) 미탐지가 혼잡을 심각으로 과잉 경고하는 것보다 더 치명적"이라는 프로젝트 판단으로 **power=1.0을 기본값으로 채택**했다.

### 현재 라벨 분포 (2026-08-24 새벽 기준, `metrics_v2.csv` 1514행, windowed)

| Label | train | val | test |
|---|---:|---:|---:|
| 0 정상 | 118 | 25 | 26 |
| 1 경고 | 515 | 110 | 111 |
| 2 혼잡 | 363 | 78 | 78 |
| 3 심각 | 28 | 6 | 6 |

### 현재 평가 결과 (power=1.0, `ap_v2_eval_report.txt`)

| | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| Fixed theta | 67.0% | 88.5% | 86.5% | 32.1% | 66.7% |
| Dynamic theta | 67.0% | 88.5% | 86.5% | 32.1% | 66.7% |

test label 3이 5~6개 수준이라 recall이 실행마다 크게 흔들린다(40.0% → 20.0% → 66.7% 순으로 관측). 표본이 두 자릿수 중반 이상으로 늘 때까지는 추세로만 참고할 것.

### 알려진 한계

- Label 3(심각) 샘플이 여전히 얇다(test 5개) — recall이 실행마다 20~40%로 크게 흔들린다. 통계적으로 안정적이라 보기 어렵다.
- Label 2/3 트레이드오프가 절벽형이라 class weight power 중간값 튜닝이 잘 안 먹힌다.
- AP(Opal) 장비가 다중 station(2대 이상) 동시 부하에서는 재현성 있게 크래시한다. 폰 1대 단일 스트림은 훨씬 안정적(100Mbps로 5분 완주 사례 있음)이지만 완전히 안전하진 않다 — 휴식 없이 반복 사용하면 단일 station도 누적 피로로 크래시할 수 있다(같은 밤 세 번째 연속 시도가 2분 41초 만에 실패). 원인 미확정, 자세한 내용은 `.work-log/current.md` 참고.
- 아직 ONNX/INT8/Raspberry Pi 배포 파이프라인이 없다. 1차의 `export_onnx_ap.py`/`prepare_pi_bundle_ap.py`를 새 경로로 재사용할지 별도 스크립트를 만들지 미정.
- 1차 `ap_cleaned_strict`를 새 가중치로 재라벨링할지는 아직 팀 미결정 상태다.

### 재현 명령어

데이터 변환:

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2.csv --out-dir project\data\ap_metrics_v2 --overwrite
```

가중치만 바뀌었을 때 raw 데이터 재라벨링(AP 재수집 불필요, 그 다음 위 변환 명령을 다시 돌려야 windowed 데이터에 반영됨):

```powershell
python project\scripts\relabel_metrics_v2.py
```

Early Exit LSTM 학습:

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint-dir project\checkpoints\ap_v2 --epochs 50 --batch-size 32 --class-weight-power 1.0
```

평가:

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint project\checkpoints\ap_v2\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_v2_eval_report.txt
```

## 현재 결론

**1차(`ap_cleaned_strict`, archived) 기준:**

- AP 실측 strict 데이터 기준 4단계 혼잡 분류는 동작한다.
- Baseline LSTM 정확도는 92.7%이고, Proposed EE Dynamic은 91.5%로 정확도 손실은 1.2%p 수준이다.
- Label 2 혼잡 정확도는 77.3%로, 1학기 데이터보다 낮다. 이는 실제 AP 데이터가 더 어렵고 라벨 경계가 덜 깔끔하기 때문이다.
- PC Python 환경에서는 Early Exit 속도 우위를 강하게 주장하기 어려웠으나, Raspberry Pi + staged ONNX 실측에서는 Proposed Dynamic FP32가 Baseline FP32보다 7.5% 빠르다는 실측 우위를 확인했다(`project/results/yongsang/pi_ap_measurements/`).
- 이 라인은 완료 상태이며 더 이상 갱신하지 않는다.

**2차(`ap_metrics_v2`, 진행 중) 기준:**

- 실제 구매 장비 실측이라는 목표를 처음으로 만족하는 라인이지만, 아직 label 3 데이터가 얇고 ONNX/Pi 배포 파이프라인이 없어 1차만큼 완성되지 않았다.
- AP 하드웨어 안정성 문제로 데이터 추가 수집이 막혀 있는 상태다(다음 세션 최우선 과제). 원인 분석은 `docs/yongsang/ap_crash_analysis.md` — "몇 대가 붙었는가"보다 "누가 송신하는가"(노트북 송신은 안정적, 폰 송신은 거의 즉시 크래시)가 핵심 변수로 보인다.
- 최종적으로 이 라인이 1차를 대체할지, 두 라인을 report에서 어떻게 병기할지는 아직 팀 논의가 필요하다.

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

### 9. 2차 AP 실측 재설계(ap_metrics_v2) 파일

```text
project/README_AP_V2.md
docs/yongsang/ap_crash_analysis.md
project/scripts/metrics_v2.csv
project/scripts/collect_metrics.py
project/scripts/relabel_metrics_v2.py
project/data/ap_metrics_v2/dataset_summary.json
project/data/ap_metrics_v2/conversion_report.txt
project/checkpoints/ap_v2/ap_early_exit_lstm_best.pth
project/results/yongsang/ap_v2_eval_report.txt
.work-log/current.md
```

역할:

- `project/README_AP_V2.md`는 1차의 `project/README_AP_STRICT.md`에 대응하는 2차 전용 문서다. 기준 데이터, congestion_score 계산식, 재라벨링/변환/학습/평가 명령어, 1차와의 차이, 알려진 한계를 정리한다. 2차 라인을 재현하려면 이 문서를 먼저 본다.
- `docs/yongsang/ap_crash_analysis.md`는 AP(Opal) 반복 크래시 원인을 분석한 문서다. "다중 station"이 아니라 "누가 송신하는가"(노트북 송신은 안정적, 폰 송신은 거의 즉시 크래시)가 핵심 변수라는 결론과 근거 데이터, 다음 검증 방향을 정리한다.
- 1차와 별개로 진행 중인, 팀이 실제로 구매한 AP 장비 실측 기반 재설계 라인이다. 자세한 내용은 위 "2차 AP 실측 재설계(ap_metrics_v2)" 섹션도 참고한다.
- `.work-log/current.md`에 AP 하드웨어 크래시, congestion_score 재조정, class weight power 실험 등 이 라인의 최신 진행 상황이 세션별로 기록되어 있다 — 이 문서(CLAUDE.md)보다 더 최신 세부사항은 여기서 확인한다.

### 10. Claude에게 중요한 해석 기준

Claude가 답변할 때는 아래 기준을 지켜야 한다.

1. 1학기 결과와 AP strict(1차) 결과를 섞어서 말하지 않는다.
2. 1학기 `project/data/real`은 4-feature이고, AP strict(1차)와 `ap_metrics_v2`(2차)는 9-feature이다.
3. 기존 `compare_baselines.py`는 AP strict 데이터에 그대로 쓰면 안 된다.
4. AP strict의 `SDN-style Early Exit (trained)` 결과는 더 이상 임시 비교가 아니라, `ap_sdn_lstm.py`/`train_ap_sdn.py`로 AP 9-feature 데이터에서 별도 학습한 최종 checkpoint(`ap_sdn_lstm_best.pth`) 기준이다.
5. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책을 바꿔 평가하는 구조이다.
6. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다.
7. 최종 속도 주장은 Raspberry Pi + staged ONNX + INT8 실측으로 확정해야 한다.
8. **1차(`ap_cleaned_strict`)와 2차(`ap_metrics_v2`)는 congestion_score 가중치 자체가 달라 label 정의가 다르다.** 두 라인의 정확도/label 분포를 같은 표에서 직접 비교하지 않는다. "AP 실측 데이터"라고만 말하면 어느 라인인지 모호하니 항상 1차/2차(또는 데이터셋 경로)를 명시한다.
9. 1차는 실제 구매 장비 실측이 아니라 인터넷 공개 데이터 기반이었다(2026-08-23 확인). "실제 AP 장비 실측"이라는 표현은 2차(`ap_metrics_v2`)에만 쓴다.
