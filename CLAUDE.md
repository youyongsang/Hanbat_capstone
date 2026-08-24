# Claude용 프로젝트 맥락 정리 (capstoneDesign2 브랜치)

이 문서는 Claude 또는 외부 AI 도구에 현재 프로젝트 맥락을 전달하기 위한 요약 파일이다. 이 브랜치(`capstoneDesign2`)는 `yongsang` 브랜치에서 분기해 코드를 정리한 브랜치다.

- **제외**: 1학기 4-feature 학습/평가/ONNX 파이프라인 코드와 1차 실측(`ap_cleaned_strict`, 588행 — 실제로는 인터넷 공개 데이터 기반이었음) 전체 파이프라인. 그 코드가 필요하면 `yongsang` 브랜치를 참고한다.
- **포함**: 2학기 실측 데이터 라인(`ap_metrics_v2`, 이 문서의 핵심 대상), docs 문서 전체(팀원별 가이드라인·work log 포함, 코드는 없어도 기록은 다 남겨둠), 그리고 1학기 Raspberry Pi 실측 결과(`project/results/hojung/`, `project/results/final_figures/`, `project/deploy/raspberry_pi/`)는 `origin/hojung`(원격, 로컬 `hojung` 브랜치보다 최신)에서 가져와 유지한다 — staged ONNX 기준 Baseline/Fixed/Dynamic Early Exit의 Pi 실측 지연 비교 자료다. 이건 1학기 4-feature 모델 기준이며 `ap_metrics_v2`(9-feature)와는 무관하니 섞어서 비교하지 않는다.

## 프로젝트 개요

본 프로젝트는 AP(무선 공유기) 기반 무선 환경에서 트래픽 혼잡 상태를 시계열 feature로 판단하는 LSTM 기반 분류 시스템이다. 최종 목표는 Raspberry Pi 같은 엣지 장비에서 실시간 혼잡 수준을 추론하고, Early Exit 및 ONNX/INT8 경량화를 통해 추론 지연을 줄이는 것이다.

혼잡 라벨은 4단계이다.

| Label | 의미 |
|---|---|
| 0 | 정상 |
| 1 | 경고 |
| 2 | 혼잡 |
| 3 | 심각 |

## 배경 (1학기 → 1차 → 2차)

- **1학기**: `project/data/real` 기준 4-feature(RPS, channel occupancy, packet loss, latency) 실험. Threshold 규칙, Baseline LSTM, Early Exit Fixed/Dynamic, SDN-style baseline을 비교했다. 이 브랜치에는 코드가 없다 — `yongsang` 브랜치 참고.
- **1차(`ap_cleaned_strict`, archived)**: 방학 중 "실제 AP 장비 실측"을 목표로 9-feature 파이프라인을 구축했으나, 2026-08-23 확인 결과 실제로는 인터넷 공개 데이터를 가공한 것이었다(팀이 구매한 AP 장비 실측이 아니었음). 9-feature 파이프라인, congestion score 라벨링, Early Exit/SDN 비교 구조를 미리 검증하는 역할은 했다. 재라벨링/재학습하지 않기로 팀이 결정했고(2026-08-24), 관련 코드·데이터·체크포인트·결과는 `yongsang` 브랜치에만 남아 있다.
- **2차(`ap_metrics_v2`, 이 브랜치의 기준선)**: 2026-08-23부터 팀이 실제로 구매한 GL.iNet Opal(GL-SFT1200) AP에서 `project/scripts/collect_metrics.py`로 직접 라이브 수집한 진짜 실측 데이터. "실제 AP 장비 실측"이라는 목표를 처음으로 만족하는 라인이며, 이 브랜치의 공식 진행 라인이다.

## 2차 AP 실측 (ap_metrics_v2)

### 파일 위치

```text
project/scripts/metrics_v2.csv          원본 raw 수집 CSV (누적, 계속 자람)
project/scripts/collect_metrics.py      라이브 수집 스크립트 (congestion_score 계산도 여기서)
project/scripts/relabel_metrics_v2.py   가중치가 바뀔 때 raw CSV를 재수집 없이 재라벨링
project/scripts/prepare_ap_metrics_dataset.py  windowed train/val/test 변환
project/scripts/train_ap_early_exit.py
project/scripts/evaluate_ap_early_exit.py
project/data/ap_metrics_v2/             windowed train/val/test, scaler, dataset_summary.json
project/checkpoints/ap_v2/              Early Exit LSTM 체크포인트
project/results/yongsang/ap_v2_eval_report.txt              최신 평가 리포트
project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt  1차 체크포인트/스케일러로 2차 데이터를 잘못 평가했을 때의 진단 기록(역사적 자료)
project/models/ap_early_exit_lstm.py    APEarlyExitLSTM (EarlyExitLSTM 상속, input_size=9)
project/models/early_exit_lstm.py       EarlyExitLSTM 베이스, multi_exit_loss
project/utils/ap_features.py            9개 feature 컬럼 정의 (AP_FEATURE_COLUMNS)
project/utils/ap_dataloader.py          windowed CSV → DataLoader
project/README_AP_V2.md                 2차 전용 상세 문서(기준 데이터, 명령어, 한계)
docs/yongsang/ap_crash_analysis.md      AP(Opal) 반복 크래시 원인 분석
docs/yongsang/congestion_label_criteria.md  congestion_score 계산식·라벨 경계 정리
.work-log/current.md                    세션별 최신 진행 상황(이 문서보다 최신)
```

### 입력 feature (9개)

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

`timestamp`, `scenario`, `channel_occupancy_method`, `packet_loss_udp_percent`, `connected_clients`, `congestion_score`, `label`은 모델 입력에서 제외한다(각각 시간 정보, 메타데이터, 낮은 변별력, 라벨 생성 중간값, 정답이기 때문). 정규화는 train split의 min-max 기준이며 `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 써야 한다.

### congestion_score 계산식

```text
congestion_score = 0.20 * throughput_score + 0.45 * occupancy_score + 0.20 * retry_failed_score + 0.15 * jitter_score
```

label 경계: `<0.25`→0, `0.25~0.50`→1, `0.50~0.75`→2, `≥0.75`→3.

가중치를 바꾼 이유: 실측 stress_load 구간에서 label 2와 3의 sub-score 평균을 비교해보니 `throughput_score`(0.665→0.707)는 거의 차이가 없었던 반면 `occupancy_score`(0.449→0.898)와 `jitter_score`(0.512→0.802)는 뚜렷한 차이를 보였다. throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 기여가 거의 없었으므로, occupancy·jitter 비중을 높이고 throughput 비중을 낮췄다.

### class weight power = 1.0

`train_ap_early_exit.py --class-weight-power`(기본값 1.0, `compute_class_weights`)로 조절한다. 이 데이터셋에서는 완만한 트레이드오프가 아니라 절벽형이었다:

| power | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
| 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
| **1.0** | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

0.7/0.85는 label 3 recall이 계속 0%이고, power=1.0(순수 역빈도)에서만 label 3이 잡히기 시작하며 그 대신 label 2 recall이 하락한다. "심각(label 3) 미탐지가 혼잡을 심각으로 과잉 경고하는 것보다 더 치명적"이라는 프로젝트 판단으로 **power=1.0을 기본값으로 채택**했다.

### 최신 라벨 분포 및 평가 결과

정확한 최신 수치는 `.work-log/current.md`를 확인한다(세션마다 갱신됨). 2026-08-24 밤 세션 기준 `metrics_v2.csv` 2510행(data rows 2509), 최근 평가는 전체 정확도 84.3%(fixed)/83.7%(dynamic), Label 3 recall 12.5~25.0%(test 8개 중 1~2개) — 직전 세션(57.1%)에서 크게 떨어짐, 표본 8개 수준의 전형적인 노이즈로 추정됨(전체 정확도는 오히려 최고치).

### 알려진 한계

- Label 3(심각) 샘플이 여전히 얇다 — recall이 실행마다 크게 흔들린다.
- Label 2/3 트레이드오프가 절벽형이라 class weight power 중간값 튜닝이 잘 안 먹힌다.
- AP(Opal) 장비가 특정 조건에서 반복적으로 크래시한다. 2026-08-24 저녁 세션 기준 **"다중 station(2대 이상 동시 부하)"이 핵심 변수**라는 쪽으로 무게가 실려 있다 — S26/191 각각 단독으로는 40~150Mbps까지 전부 크래시 없이 완주했지만, 191+S26 동시 부하 조합에서는 부하 크기와 무관하게 불안정성이 나타났다. 재부팅 후 스위핑 결과 **191=60M/S26=60M(2분 내외)이 안정성·label 3 생성력 둘 다에서 스위트스팟**이었고, 80M/80M이나 10분 이상 장시간 콤보는 오히려 더 불안정했다(완전 크래시는 아니지만 SSH가 최대 110초까지 끊기는 증상). "191 개별 하드웨어 문제"였다는 이전 가설은 191의 다단계 연속 생존으로 반증됨. 상세·최신 상황은 `docs/yongsang/ap_crash_analysis.md`와 `.work-log/current.md` 참고.
- ONNX/INT8/Raspberry Pi 배포 파이프라인이 아직 이 라인에는 없다. 1차의 `export_onnx_ap.py`/`prepare_pi_bundle_ap.py`(`yongsang` 브랜치에 있음)를 새 경로로 재사용할지 별도 스크립트를 만들지 미정.

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

## 자주 헷갈리는 점

### torch DLL 로딩 실패

이 노트북(들)의 anaconda base 환경에서 `import torch`가 `OSError: [WinError 1114]`로 실패하는 문제가 있다. 별도 conda 환경(`capstone`)에 torch(CPU)+pandas+numpy를 설치해서 써야 한다.

### 정확도가 이상하게 낮게 나올 때 의심할 점

1. raw CSV(`metrics_v2.csv`)를 그대로 넣지 않았는지 확인한다. 모델은 window size 10으로 변환된 `test.csv`를 사용한다.
2. feature 순서가 `ap_features.py`의 9개 정의와 동일한지 확인한다.
3. `label`, `congestion_score`, `scenario`, `timestamp`가 입력 feature에 섞이지 않았는지 확인한다.
4. `scaler_params.json` 기준이 다른지 확인한다(재라벨링만 하고 재변환을 안 하면 라벨과 스케일러가 어긋난다).

## Claude가 추가로 참고해야 할 파일

1. `project/README_AP_V2.md` — 2차 전용 기준 문서. 데이터, congestion_score 계산식, 명령어, 알려진 한계를 정리한다.
2. `docs/yongsang/ap_crash_analysis.md` — AP(Opal) 반복 크래시 원인 분석. "몇 대가 붙었는가"가 아니라 "누가 송신하는가"가 핵심 변수라는 가설과 근거, 다음 검증 방향.
3. `docs/yongsang/congestion_label_criteria.md` — congestion_score 계산식과 라벨 경계 정리(1차/2차 비교 포함, 역사적 맥락용).
4. `.work-log/current.md` — 세션별 최신 진행 상황. 이 문서(CLAUDE.md)보다 최신 세부사항은 여기서 확인한다.
5. `project/utils/ap_features.py`, `project/utils/ap_dataloader.py`, `project/models/ap_early_exit_lstm.py` — 실제 코드 흐름. 58% 근처의 낮은 정확도가 나오면 이 파일들의 feature 순서, scaler, checkpoint 경로를 먼저 점검한다.

## Claude에게 중요한 해석 기준

1. 이 브랜치에는 1학기 4-feature 코드와 1차(`ap_cleaned_strict`) 코드가 없다. 그 자료를 언급해야 하면 `yongsang` 브랜치를 확인하라고 안내한다.
2. "AP 실측 데이터"는 이 브랜치에서는 항상 `ap_metrics_v2`를 가리킨다.
3. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다 — 최종 속도 주장은 Raspberry Pi + staged ONNX 실측으로 확정해야 하지만, 이 라인에는 아직 그 파이프라인이 없다.
4. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책을 바꿔 평가하는 구조이다.
