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
project/scripts/relabel_metrics_v2.py   가중치가 바뀔 때 sub-score 재조합만 (재수집 불필요)
project/scripts/remeasure_metrics_v2.py sub-score 공식이 바뀔 때 raw feature에서 전부 재계산 (2026-08-27 retry per-s 마이그레이션에 사용)
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
tx_retries_per_s
tx_failed_per_s
rssi_dbm
rssi_delta_db
rssi_moving_avg_dbm
```

**2026-08-27 변경**: `tx_retries_delta`/`tx_failed_delta` → **`tx_retries_per_s`/`tx_failed_per_s`**. 델타값(지난 폴링 이후 재전송 수)이 폴링 주기에 그대로 비례해서(4초 폴링 = 1초 폴링 × 4), 파이 유선 수집으로 폴링을 ~1초로 당기자 retry 신호가 1/4로 눌려 label 2/3가 안 나오는 문제가 드러났다. 이제 `delta / poll_interval_s`로 정규화한다(throughput과 동일 방식). 기존 `metrics_v2.csv`는 `remeasure_metrics_v2.py`로 마이그레이션됨(폴링 간격은 timestamp 차이로 역산, 근사치 — 상세는 아래 "알려진 한계").

`timestamp`, `scenario`, `channel_occupancy_method`, `packet_loss_udp_percent`, `poll_interval_s`, `connected_clients`, `congestion_score`, `label`은 모델 입력에서 제외한다(각각 시간 정보, 메타데이터, 낮은 변별력, 폴링 타이밍, 라벨 생성 중간값, 정답이기 때문). 정규화는 train split의 min-max 기준이며 `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 써야 한다.

### congestion_score 계산식

```text
congestion_score = 0.20 * throughput_score + 0.45 * occupancy_score + 0.20 * retry_failed_score + 0.15 * jitter_score

retry_failed_score = clamp01( (tx_retries_per_s + tx_failed_per_s) / RETRY_FAILED_MAX_PER_SEC )   # RETRY_FAILED_MAX_PER_SEC = 6250 (= 옛 25000 / 옛 폴링간격 ~4초)
```

label 경계: `<0.25`→0, `0.25~0.50`→1, `0.50~0.75`→2, `≥0.75`→3.

가중치(0.20/0.45/0.20/0.15)를 바꾼 이유: 실측 stress_load 구간에서 label 2와 3의 sub-score 평균을 비교해보니 `throughput_score`(0.665→0.707)는 거의 차이가 없었던 반면 `occupancy_score`(0.449→0.898)와 `jitter_score`(0.512→0.802)는 뚜렷한 차이를 보였다. throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 기여가 거의 없었으므로, occupancy·jitter 비중을 높이고 throughput 비중을 낮췄다.

### class weight power = 1.0

`train_ap_early_exit.py --class-weight-power`(기본값 1.0, `compute_class_weights`)로 조절한다. 이 데이터셋에서는 완만한 트레이드오프가 아니라 절벽형이었다:

| power | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
| 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
| **1.0** | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

0.7/0.85는 label 3 recall이 계속 0%이고, power=1.0(순수 역빈도)에서만 label 3이 잡히기 시작하며 그 대신 label 2 recall이 하락한다. "심각(label 3) 미탐지가 혼잡을 심각으로 과잉 경고하는 것보다 더 치명적"이라는 프로젝트 판단으로 **power=1.0을 기본값으로 채택**했다.

### 최신 라벨 분포 및 평가 결과

정확한 최신 수치는 `.work-log/current.md`를 확인한다(세션마다 갱신됨).

- **최신 raw 데이터(2026-08-27 저녁, `tx_retries_per_s` 마이그레이션 후)**: `metrics_v2.csv` data rows 5574, 라벨 분포 0:3208 / 1:1328 / 2:982 / **3:56**. 마이그레이션으로 raw label 3이 85→56으로 줄었음(폴링 스톨로 부풀려진 ~29개 제거). 8/26 저녁~밤 수집분은 이미 포함됨.
- **최신 모델 평가(2026-08-27 저녁 재학습, 5574행, best val balanced acc 80.7%)**: windowed test 764샘플(test label 3 8개)로 전체 정확도 **91.2%(fixed/dynamic 동일, 역대 최고)**, Label 0 97%대 / Label 1 88% / Label 2 79% / **Label 3 recall 12.5%(1/8) — 급락**. label 3 급락은 마이그레이션이 test 표본을 11→8로 줄여서 1개 차이가 12.5%p인 소표본 노이즈. 전체 정확도가 최고인 건 retry feature가 폴링 노이즈를 안 타게 된 효과. **다음: 파이 유선 수집으로 label 3 다시 쌓기.**

### 알려진 한계

- Label 3(심각) 샘플이 여전히 얇다 — recall이 실행마다 크게 흔들린다.
- Label 2/3 트레이드오프가 절벽형이라 class weight power 중간값 튜닝이 잘 안 먹힌다.
- AP(Opal) 장비가 특정 조건에서 반복적으로 크래시한다. **2026-08-27 세션 기준 최신 가설: "몇 대가 붙었는가"보다 "각 폰이 AP 와이파이에 제대로·대칭적으로 붙어있는가"가 핵심 변수** — S26/191 각각 단독으로는 40~150Mbps까지 전부 크래시 없이 완주했고, 191+S26 콤보도 **시작 전 두 폰 와이파이를 재연결하면** 120초·5~7분 모두 크래시 없이 완주했다. 신호가 강한 S26이 채널을 독점하고 신호가 약한 191이 굶는 비대칭(Wi-Fi capture effect)이 반복 관측됨(191 RSSI 열세 추정). 부하 스위트스팟은 **191=60M/S26=60M(2~7분)**이 기준선이고 8/26 밤에 **75M/75M 대칭**이 label 3 생성 후보로 추가됨(재현성 불완전, 3회 합산 label 3 5개). **상한은 확실: 80M/80M은 10분 시도에서 완전 크래시(물리 재부팅), 60/60도 500초 넘기면 완전 크래시** — 안전 구간 대략 300~420초. "191 개별 하드웨어 문제"였다는 이전 가설은 191의 다단계 연속 생존으로 반증됨. 상세·최신 상황은 `docs/yongsang/ap_crash_analysis.md`와 `.work-log/current.md` 참고.
- 실시간 폴링 자기참조 지연: `collect_metrics.py`가 매 루프마다 새 SSH를 띄우던 걸 2026-08-27 세션에 지속 SSH 세션(`APPoller` 클래스)으로 전환했다 — 문법 검사만 됨, **실기기 미검증**. 관리 트래픽이 여전히 같은 무선 채널을 탄다는 근본 구조는 그대로라, 유선 관리채널 분리(collector를 라즈베리 파이로, Opal LAN 포트를 별도 서브넷으로)를 다음 세션 실행 대상으로 설계 확정함.
- **(2026-08-28 갱신) ONNX/Raspberry Pi 배포 파이프라인 생김, INT8까지 완료** — `ap_metrics_v2_redesign2`(본수집 2차, label 3 202개) 기준으로 실제 Pi 실측까지 완료했다. staged(세션 3개 분리) 첫 시도는 baseline보다 느렸는데(세션 호출 오버헤드), `torch.jit.script`+ONNX `If` 노드로 단일 그래프 재설계(`export_onnx_ap_unified.py`) 후 baseline 대비 **-40%**. 이 unified 그래프를 그대로 양자화하면 ONNX 양자화 도구가 `If` 서브그래프 안의 LSTM을 건너뛰어(도구 한계) 속도 이득이 없었는데, staged(flat) 그래프로 먼저 양자화한 뒤 손수 재조립(`export_onnx_ap_unified_int8_v2.py`)하는 방식으로 우회해 **최종 baseline 대비 -67%**까지 확인했다(1학기 4-feature 자료로 두 현상 모두 교차검증). 상세: `docs/yongsang/onnx_early_exit_redesign.md`, `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`.

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
6. `docs/yongsang/onnx_early_exit_redesign.md` — ONNX Early Exit 배포를 staged(세션 3개)에서 단일 그래프(If 노드)로, 다시 INT8(staged로 양자화 후 재조립)까지 재설계한 기록. Pi latency 주장을 쓸 때는 이 문서의 최종 결론(unified INT8, baseline 대비 -67%)을 따른다 — staged나 fp32-only 수치를 최종 결과로 인용하지 않는다.

## Claude에게 중요한 해석 기준

1. 이 브랜치에는 1학기 4-feature 코드와 1차(`ap_cleaned_strict`) 코드가 없다. 그 자료를 언급해야 하면 `yongsang` 브랜치를 확인하라고 안내한다.
2. "AP 실측 데이터"는 이 브랜치에서는 항상 `ap_metrics_v2`를 가리킨다.
3. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다 — 최종 속도 주장은 Raspberry Pi + staged ONNX 실측으로 확정해야 하지만, 이 라인에는 아직 그 파이프라인이 없다.
4. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책을 바꿔 평가하는 구조이다.
