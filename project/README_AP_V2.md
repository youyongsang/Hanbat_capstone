# AP 실측 2차 재설계 데이터 모델 파이프라인 (ap_metrics_v2)

이 문서는 팀이 실제로 구매한 GL.iNet Opal(GL-SFT1200) AP에서 `project/scripts/collect_metrics.py`로 직접 라이브 수집한 데이터를 기준으로 9개 feature 모델을 학습, 평가하는 절차를 정리한다.

`project/README_AP_STRICT.md`(1차, `ap_cleaned_strict`)와는 별개의 라인이다. 1차는 실제로는 인터넷에 공개돼 있던 데이터를 가공해서 만든 것이었음이 2026-08-23에 확인됐고, 이 2차 라인이 "실제 AP 장비 실측"이라는 목표를 처음으로 만족한다. 1차는 archived 상태로 그대로 두고 더 이상 갱신하지 않는다. 두 라인은 congestion_score 계산 가중치가 달라 label 정의 자체가 다르므로, 정확도나 label 분포를 같은 표에서 직접 비교하지 않는다.

## 기준 데이터

원본 raw 데이터와 변환된 windowed 데이터는 아래 경로에 있다.

```text
project/scripts/metrics_v2.csv          원본 raw 수집 CSV (누적, 계속 자람)
project/data/ap_metrics_v2/
├── train.csv
├── val.csv
├── test.csv
├── scaler_params.json
├── dataset_summary.json
└── conversion_report.txt
```

`metrics_v2.csv`는 `project/scripts/collect_metrics.py <scenario>`로 라이브 수집할 때마다 한 행씩 append되는 누적 파일이다(1차의 `raw/metrics_cleaned_strict.csv`처럼 고정된 스냅샷이 아니다). 새로 수집하거나 congestion_score 가중치를 바꾼 뒤에는 반드시 아래 "데이터 변환" 명령을 다시 돌려서 `ap_metrics_v2/`를 최신 상태로 맞춰야 한다.

라벨 분포(2026-08-23 기준, `metrics_v2.csv` 1265행 → windowed 샘플):

| Label | 의미 | train | val | test |
|---|---|---:|---:|---:|
| 0 | 정상 | 100 | 21 | 22 |
| 1 | 경고 | 452 | 97 | 97 |
| 2 | 혼잡 | 275 | 59 | 59 |
| 3 | 심각 | 23 | 5 | 5 |

Label 3(심각)이 여전히 얇다. AP 하드웨어가 다중 station 부하에서 반복 크래시하는 문제가 있어 추가 수집이 제한적인 상태다 — 자세한 내용과 다음 시도 방향은 `.work-log/current.md`를 참고한다.

## 입력 feature

1차와 동일하게 아래 9개 feature만 사용한다. 순서가 바뀌면 checkpoint 결과가 달라지므로 반드시 이 순서를 유지한다.

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

아래 컬럼은 모델 입력에 넣지 않는다(1차와 동일한 이유).

| 컬럼 | 용도 | 입력 제외 이유 |
|---|---|---|
| `timestamp` | 측정 시각 | 시간값 자체가 혼잡 원인이 아니며 과적합 위험이 있음 |
| `scenario` | 측정 상황명 | 상황 이름을 보고 맞히는 문제가 될 수 있음 |
| `channel_occupancy_method` | 점유율 계산 방식 | 측정 메타데이터 |
| `packet_loss_udp_percent` | UDP 손실률 | 데이터에서 신뢰 가능한 변동이 부족함 |
| `connected_clients` | 연결 클라이언트 수 | 구분력이 제한적이라 제외 |
| `throughput_score`, `occupancy_score`, `retry_failed_score`, `jitter_score`, `congestion_score` | 라벨 생성용 중간 점수 | 정답 생성에 쓴 값이므로 입력하면 데이터 누수 |
| `label` | 정답 라벨 | 모델이 맞혀야 하는 정답 |

정규화는 train split의 feature별 min-max 기준으로 수행한다. 생성된 `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 적용해야 한다.

## congestion_score 계산식 (1차와 다름)

`project/scripts/collect_metrics.py`의 `calculate_scores()`에서 계산한다.

```text
congestion_score = 0.20 * throughput_score + 0.45 * occupancy_score + 0.20 * retry_failed_score + 0.10 * jitter_score
```

sub-score는 각각 아래 상한으로 0~1 clamp한 값이다.

| sub-score | 계산 | 상한 |
|---|---|---:|
| `throughput_score` | `throughput_mbps / THROUGHPUT_MAX_MBPS` | 150 Mbps |
| `occupancy_score` | `channel_occupancy_percent / 100.0` | 100% |
| `retry_failed_score` | `(tx_retries_delta + tx_failed_delta) / RETRY_FAILED_MAX` | 25,000 |
| `jitter_score` | `jitter_ms / JITTER_MAX_MS` | 300 ms |

label 경계는 1차와 동일: `<0.25`→0, `0.25~0.50`→1, `0.50~0.75`→2, `≥0.75`→3.

**1차(`ap_cleaned_strict`)는 `0.35 * throughput + 0.35 * occupancy + 0.20 * retry + 0.10 * jitter`를 그대로 쓴다.** 가중치를 바꾼 이유: 실측 stress_load 구간에서 label 2와 3의 sub-score 평균을 비교해보니 `throughput_score`(0.665→0.707)는 거의 차이가 없었던 반면 `occupancy_score`(0.449→0.898)와 `jitter_score`(0.512→0.802)는 뚜렷한 차이를 보였다. throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 기여가 거의 없었으므로, occupancy·jitter 비중을 높이고 throughput 비중을 낮췄다.

가중치를 다시 조정하고 싶으면 `calculate_scores()`의 가중치를 바꾼 뒤, 아래 재라벨링 명령으로 **AP를 다시 부하 테스트하지 않고도** 이미 모아둔 raw 데이터에 즉시 반영할 수 있다.

```powershell
python project\scripts\relabel_metrics_v2.py
```

이 스크립트는 `metrics_v2.csv`에 이미 저장된 4개 sub-score 컬럼을 새 가중치로 재조합해서 `congestion_score`/`label`만 갱신한다(원시 feature 값은 건드리지 않음). 실행 후 콘솔에 재라벨링 전/후 label 분포가 출력된다.

## 데이터 변환

raw CSV에서 window size 10의 train/val/test CSV를 생성한다. 새로 수집했거나 재라벨링한 뒤에는 반드시 다시 실행한다.

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2.csv --out-dir project\data\ap_metrics_v2 --overwrite
```

생성 결과는 `dataset_summary.json`과 `conversion_report.txt`에서 확인한다.

## 모델 학습

Early Exit LSTM을 학습한다.

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint-dir project\checkpoints\ap_v2 --epochs 50 --batch-size 32 --class-weight-power 1.0
```

생성 checkpoint는 아래와 같다.

```text
project/checkpoints/ap_v2/
├── ap_early_exit_lstm_best.pth
├── ap_early_exit_fixed.pth
└── ap_early_exit_dynamic.pth
```

`Fixed`와 `Dynamic`은 같은 Early Exit backbone을 사용하고, 평가 시 threshold 정책만 다르게 적용한다(1차와 동일한 방식).

### class-weight-power

`--class-weight-power`(기본값 `1.0`)로 클래스 불균형 보정 강도를 조절한다. `compute_class_weights()`가 `(N / (K * count_c)) ** power`로 가중치를 계산한다. 이 데이터셋에서는 완만한 트레이드오프가 아니라 절벽형이었다.

| power | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
| 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
| **1.0** | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

0.7/0.85는 label 3 recall이 계속 0%이고, `power=1.0`(순수 역빈도)에서만 label 3이 잡히기 시작하며 그 대신 label 2 recall이 하락한다. "심각(label 3) 미탐지가 혼잡을 심각으로 과잉 경고하는 것보다 더 치명적"이라는 프로젝트 판단으로 `power=1.0`을 기본값으로 채택했다. 데이터가 더 모이면(특히 label 3 샘플) 재실험이 필요할 수 있다.

## 모델 평가

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint project\checkpoints\ap_v2\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_v2_eval_report.txt
```

현재 결과(`power=1.0`):

| | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| Fixed theta | 66.1% | 95.5% | 75.3% | 42.4% | 40.0% |
| Dynamic theta | 66.1% | 95.5% | 75.3% | 42.4% | 40.0% |

참고: `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt`는 1차 체크포인트/스케일러로 2차 데이터를 잘못 평가했을 때(정확도 39.3%)의 진단 기록이다. 역사적 자료일 뿐 최종 결과가 아니다.

## 1차(`ap_cleaned_strict`)와의 차이

| 구분 | 1차 (`ap_cleaned_strict`) | 2차 (`ap_metrics_v2`) |
|---|---|---|
| 데이터 출처 | 인터넷 공개 데이터 가공 (2026-08-23 확인) | 팀이 구매한 GL.iNet Opal AP 실측 |
| 원본 raw | `raw/metrics_cleaned_strict.csv` (588행, 고정) | `project/scripts/metrics_v2.csv` (1265행, 계속 누적) |
| congestion_score 가중치 | throughput 35% / occupancy 35% / retry 20% / jitter 10% | throughput 20% / occupancy 45% / retry 20% / jitter 15% |
| class weight | 미사용(plain cross-entropy) | `--class-weight-power` (기본 1.0) |
| checkpoint 경로 | `project/checkpoints/ap_cleaned_strict/` | `project/checkpoints/ap_v2/` |
| ONNX/INT8/Pi 배포 | 완료 | 아직 없음 |
| 상태 | 완료, archived, 더 이상 갱신 안 함 | 진행 중인 공식 재설계 라인 |

입력 feature 정의(9개), 제외 컬럼, dataloader(`utils/ap_dataloader.py`), feature 정의 파일(`utils/ap_features.py`), window size(10), Fixed/Dynamic 방식은 1차와 동일하게 공유한다.

## 알려진 한계 / 남은 작업

- **Label 3(심각) 데이터가 얇다** — test 5개뿐이라 recall 40%가 통계적으로 안정적이라 보기 어렵다.
- **Label 2/3 트레이드오프가 절벽형** — class weight power 중간값 튜닝이 잘 안 먹힌다. label 3 표본이 더 늘어야 완만한 튜닝이 가능할 것으로 보인다.
- **AP(Opal) 하드웨어 안정성 문제** — 다중 station 동시 부하에서 반복 크래시한다. 부하를 낮춰도(40M→25M), station 수를 줄여도(3대→검증됐던 2대 조합) 오히려 더 빨리 크래시하는 패턴이 관찰됐다(90초 완주 → 40초 → 18초 → 22초). 반복된 크래시-재부팅 사이클 자체가 AP를 불안정하게 만들었을 가능성이 있다. 원인 미확정. 다음 시도 전 AP를 충분히 쉬게 하고, 폰 1대 단일 스트림 또는 채널폭 제한(5GHz 20MHz, 2.4GHz 전환) 등 station 수를 늘리지 않는 방식부터 안정성을 재확인해야 한다. 자세한 내용은 `.work-log/current.md`.
- **ONNX/INT8/Raspberry Pi 배포 파이프라인 없음** — 1차의 `export_onnx_ap.py`/`export_onnx_int8_ap.py`/`prepare_pi_bundle_ap.py`를 새 경로(`ap_metrics_v2`, `ap_v2`)로 재사용할지, 별도 스크립트를 만들지 아직 미정.
- **SDN-style / Baseline 비교 없음** — 1차처럼 Baseline LSTM, SDN-style Early Exit과의 비교표가 아직 이 라인에 없다. `train_ap_baseline_lstm.py`, `train_ap_sdn.py`를 `--data-dir project\data\ap_metrics_v2 --checkpoint-dir project\checkpoints\ap_v2`로 재사용 가능한지는 검증 필요.
- **1차를 새 가중치로 재라벨링할지 미결정** — 팀 논의 필요 항목. 재라벨링하면 1차의 완료된 학습/ONNX/Pi 배포 전체를 다시 돌려야 하므로 신중히 결정한다.
