# AP 실측 strict 데이터 모델 파이프라인

이 문서는 AP 장비에서 측정한 최종 정제 CSV를 기준으로 9개 feature 모델을 학습, 평가, 비교하는 절차를 정리한다. 기존 1학기 `project/data/real` 기반 4-feature 실험과 구분하기 위해, AP 실측 데이터는 별도 경로와 별도 스크립트를 사용한다.

## 기준 데이터

최종 기준 데이터셋은 아래 경로에 있다.

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

원본 기준 CSV는 `metrics_cleaned_normal_idle_outliers_removed.csv`와 같은 내용이다. `normal_idle` 구간의 비정상 throughput/retry outlier를 제거한 뒤 사용한다.

라벨 분포는 다음과 같다.

| Label | 의미 | 개수 |
|---|---|---:|
| 0 | 정상 | 116 |
| 1 | 경고 | 213 |
| 2 | 혼잡 | 152 |
| 3 | 심각 | 107 |

## 입력 feature

모델 입력은 아래 9개 feature만 사용한다. 순서가 바뀌면 checkpoint 결과가 달라질 수 있으므로 반드시 이 순서를 유지한다.

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

### 4개 feature에서 9개 feature로 늘린 이유

기존 1학기 실험은 RPS, channel occupancy, packet loss, latency 중심의 4개 feature를 사용했다. 하지만 AP 실측 데이터에서는 단일 지표만으로 혼잡을 안정적으로 설명하기 어려웠다.

- 채널 점유율은 부하를 줘도 특정 구간에서 크게 변하지 않았다.
- 패킷 손실률은 대부분 0 또는 N/A에 가까워 구분력이 낮았다.
- 실제 AP 환경의 혼잡은 throughput, occupancy, jitter, retry/failed, RSSI 변화가 함께 나타나는 복합 현상이다.

따라서 AP strict 실험에서는 트래픽 부하, 채널 상태, 지연 품질, 전송 안정성, 신호 세기 변화를 함께 반영하도록 9개 feature로 확장했다.

| feature 그룹 | 포함 feature | 의미 |
|---|---|---|
| 트래픽 부하 | `throughput_mbps` | 실제 전송량 |
| 채널 사용 상태 | `channel_occupancy_percent` | 채널 busy 비율 |
| 지연 품질 | `latency_ms`, `jitter_ms` | 지연과 지연 변동 |
| 전송 안정성 | `tx_retries_delta`, `tx_failed_delta` | 재전송 및 실패 증가 |
| 무선 신호 상태 | `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm` | 신호 세기 및 변화 |

아래 컬럼은 모델 입력에 넣지 않는다.

| 컬럼 | 용도 | 입력 제외 이유 |
|---|---|---|
| `timestamp` | 측정 시각 | 시간값 자체가 혼잡 원인이 아니며 과적합 위험이 있음 |
| `scenario` | 측정 상황명 | 상황 이름을 보고 맞히는 문제가 될 수 있음 |
| `channel_occupancy_method` | 점유율 계산 방식 | 측정 메타데이터 |
| `packet_loss_udp_percent` | UDP 손실률 | 현재 데이터에서 신뢰 가능한 변동이 부족함 |
| `connected_clients` | 연결 클라이언트 수 | 이번 데이터에서는 구분력이 제한적이라 제외 |
| `congestion_score` | 라벨 생성용 종합 점수 | 정답 생성에 사용한 중간값이므로 입력하면 누수 |
| `label` | 정답 라벨 | 모델이 맞혀야 하는 정답 |

정규화는 train split의 feature별 min-max 기준으로 수행한다. 생성된 `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 적용해야 한다.

## 데이터 변환

최종 raw CSV에서 window size 10의 train/val/test CSV를 생성한다.

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\data\ap_metrics_cleaned_strict\raw\metrics_cleaned_strict.csv --out-dir project\data\ap_metrics_cleaned_strict --overwrite --no-occupancy-outlier-fix
```

생성 결과는 `dataset_summary.json`과 `conversion_report.txt`에서 확인한다.

## 모델 학습

Baseline LSTM을 학습한다.

```powershell
python project\scripts\train_ap_baseline_lstm.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint-dir project\checkpoints\ap_cleaned_strict --epochs 20 --batch-size 32
```

Early Exit LSTM을 학습한다.

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint-dir project\checkpoints\ap_cleaned_strict --epochs 20 --batch-size 32
```

생성 checkpoint는 아래와 같다.

```text
project/checkpoints/ap_cleaned_strict/
├── ap_baseline_lstm_best.pth
├── ap_early_exit_lstm_best.pth
├── ap_early_exit_fixed.pth
└── ap_early_exit_dynamic.pth
```

`Fixed`와 `Dynamic`은 같은 Early Exit backbone을 사용하고, 평가 시 threshold 정책만 다르게 적용한다.

## 모델 평가

Baseline LSTM 평가는 `ap_baseline_lstm_best.pth` 기준으로 수행한다. 현재 결과 리포트는 아래 파일에 저장되어 있다.

```text
project/results/yongsang/ap_baseline_lstm_cleaned_strict_eval_report.txt
```

Early Exit Fixed/Dynamic 평가는 다음 명령으로 수행한다.

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_cleaned_strict --checkpoint project\checkpoints\ap_cleaned_strict\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_early_exit_cleaned_strict_eval_report.txt
```

## 비교표 생성

아래 명령은 AP strict 데이터 기준 비교표를 생성한다.

```powershell
python project\scripts\generate_ap_comparison.py
```

출력 파일은 아래와 같다.

```text
project/results/yongsang/ap_model_comparison_cleaned_strict.csv
project/results/yongsang/ap_model_comparison_cleaned_strict.txt
```

현재 비교표는 다음 4개 항목을 포함한다.

| 모델 | 역할 |
|---|---|
| Baseline LSTM Full | Early Exit 없는 전체 LSTM 기준선 |
| SDN-style Early Exit (trained) | SDN/Shallow-Deep Networks(ICML 2019) loss 가중치(0.15/0.30/0.55)로 AP 9-feature 기준 별도 학습한 confidence-only 조기 종료 baseline |
| Proposed Early Exit Fixed theta | 우리 AP Early Exit 구조에서 동적 threshold를 제거한 ablation |
| Proposed Early Exit Dynamic theta | 최근 트래픽 변화 기반 동적 threshold를 적용한 제안 방식 |

현재 PC 기준 결과는 다음과 같다.

| 모델 | 정확도 | Label 2 | Label 3 | Exit1 | Exit2 | Exit3 | PC 실측 | 구조상 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline LSTM Full | 92.7% | 77.3% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0667ms | 8.000ms |
| SDN-style Early Exit (trained) | 91.5% | 72.7% | 100.0% | 12.2% | 30.5% | 57.3% | 0.4956ms | 6.049ms |
| Proposed Early Exit Fixed theta | 91.5% | 77.3% | 93.3% | 15.9% | 32.9% | 51.2% | 0.5143ms | 5.732ms |
| Proposed Early Exit Dynamic theta | 91.5% | 77.3% | 93.3% | 37.8% | 39.0% | 23.2% | 0.4251ms | 4.171ms |

SDN-style은 Early Exit 백본을 재사용하지 않고 `project/scripts/train_ap_sdn.py`로 AP 9-feature 데이터에서 별도 학습한 결과이다(체크포인트: `project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth`, 평가 리포트: `project/results/yongsang/ap_sdn_cleaned_strict_eval_report.txt`). 전체 정확도는 Proposed와 동일한 91.5%이지만 Label 2(혼잡) 정확도는 72.7%로 Proposed(77.3%)보다 4.6%p 낮다. 이는 제안 모델이 SDN 대비 갖는 실질적 차별점으로 볼 수 있다.

PC 실측 시간은 Python stepwise 분기 오버헤드가 포함되어 Early Exit에 불리하게 측정될 수 있다. 최종 속도 주장은 Raspberry Pi + staged ONNX 실측으로 확정해야 한다.

## 기존 1학기 실험과의 차이

| 구분 | 1학기 실험 | AP strict 실험 |
|---|---|---|
| 데이터 경로 | `project/data/real` | `project/data/ap_metrics_cleaned_strict` |
| 입력 feature 수 | 4개 | 9개 |
| 주요 feature | RPS, channel occupancy, packet loss, latency | throughput, occupancy, latency, jitter, retry/failed, RSSI |
| checkpoint | `baseline_lstm_best.pth`, `early_exit_lstm_best.pth` | `ap_baseline_lstm_best.pth`, `ap_early_exit_lstm_best.pth` |
| dataloader | `utils/dataloader.py` | `utils/ap_dataloader.py` |
| feature 정의 | 기존 4-feature 고정 | `utils/ap_features.py` |

기존 `project/experiments/compare_baselines.py`는 1학기 4-feature 데이터용이다. AP strict 9-feature 데이터에는 그대로 사용하면 안 된다.

## Raspberry Pi 실측 전 남은 작업

AP strict 기준 최종 Pi 실험을 위해서는 아래 산출물을 다시 만들어야 한다.

1. ~~AP용 SDN LSTM checkpoint 재학습~~ — 완료 (`ap_sdn_lstm_best.pth`, `train_ap_sdn.py`/`evaluate_ap_sdn.py`)
2. ~~Baseline LSTM ONNX 변환~~ — 완료 (`export_onnx_ap.py` → `ap_baseline.onnx`)
3. ~~SDN-style staged ONNX 변환~~ — 완료 (`export_onnx_ap_sdn.py` → `ap_sdn_fixed*.onnx`, `ap_sdn_fixed_stage{1,2,3}.onnx`)
4. ~~Proposed Fixed/Dynamic staged ONNX 변환~~ — 완료 (`export_onnx_ap.py` → `ap_early_exit_fixed/dynamic*.onnx`, `_stage{1,2,3}.onnx`)
5. ~~FP32 ONNX와 INT8 ONNX 양자화 파일 생성~~ — 완료 (`export_onnx_int8_ap.py`, Baseline/SDN/Fixed/Dynamic × full+stage1/2/3 전부)
6. ~~`project/deploy/raspberry_pi_ap/` 배포 번들 갱신~~ — 완료 (`prepare_pi_bundle_ap.py`; 1학기용 `raspberry_pi/`와는 별도 폴더, 4-feature test.csv와 섞지 말 것)
7. Pi에서 동일 `test.csv` 기준으로 평균, p50, p95, Exit 비율 재측정 — **아직 미완료. 다음 단계.**

라즈베리파이 실험에서는 반드시 staged ONNX 방식으로 실행해야 실제 layer skip 효과를 볼 수 있다.

Pi에서 실행할 절차는 `project/deploy/raspberry_pi_ap/README.md`를 그대로 따르면 된다 (`python3 -m venv .venv` → `pip install onnxruntime numpy pandas` → 모델별 `inference_pi_ap.py` 실행 → `analyze_pi_results.py`로 분석). `docs/hochung/Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx`의 기존 결과는 오늘 재학습한 `ap_sdn_fixed.onnx`(SDN 백본)를 반영하지 않은 값이므로, 이 번들로 다시 측정해야 최종 결과로 쓸 수 있다.
