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

라벨 분포(2026-08-27 저녁 재학습 기준, `metrics_v2.csv` 5574 data rows → windowed 샘플):

| Label | 의미 | train | val | test |
|---|---|---:|---:|---:|
| 0 | 정상 | 2053 | 440 | 440 |
| 1 | 경고 | 850 | 182 | 183 |
| 2 | 혼잡 | 620 | 133 | 133 |
| 3 | 심각 | 37 | 8 | 8 |

raw 라벨 분포: 0:3208 / 1:1328 / 2:982 / **3:56**. **2026-08-27 `tx_retries_per_s` 마이그레이션으로 raw label 3이 85→56으로 줄었다** — 없어진 ~29개는 폴링 스톨로 `tx_retries_delta`가 부풀려져서 문턱을 넘었던 것들(정직한 수치). test label 3도 11→8. 상세는 아래 "핵심 검증 질문"·"알려진 한계".

Label 3(심각)이 여전히 얇다. AP 하드웨어가 부하 종류에 따라 반복 크래시하는 문제가 있어 추가 수집이 제한적인 상태다 — 크래시 원인 분석은 `docs/yongsang/ap_crash_analysis.md`(및 아티팩트), 다음 시도 방향은 `.work-log/current.md`를 참고한다.

## 입력 feature

1차와 동일하게 아래 9개 feature만 사용한다. 순서가 바뀌면 checkpoint 결과가 달라지므로 반드시 이 순서를 유지한다.

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

> **2026-08-27**: `tx_retries_delta`/`tx_failed_delta` → **`tx_retries_per_s`/`tx_failed_per_s`**. 델타값(지난 폴링 이후 재전송 수)이 폴링 주기에 비례해서(4초 폴링 = 1초 폴링 × 4) 흔들렸다. 파이 유선 수집으로 폴링을 ~1초로 당기자 retry 신호가 1/4로 눌려 label 2/3가 안 나오는 문제가 드러남 → `delta / poll_interval_s` (초당 재전송률)로 정규화. 기존 데이터는 `remeasure_metrics_v2.py`로 마이그레이션(폴링 간격은 timestamp 차이로 역산, 근사치).

아래 컬럼은 모델 입력에 넣지 않는다(1차와 동일한 이유).

| 컬럼 | 용도 | 입력 제외 이유 |
|---|---|---|
| `timestamp` | 측정 시각 | 시간값 자체가 혼잡 원인이 아니며 과적합 위험이 있음 |
| `scenario` | 측정 상황명 | 상황 이름을 보고 맞히는 문제가 될 수 있음 |
| `channel_occupancy_method` | 점유율 계산 방식 | 측정 메타데이터 |
| `packet_loss_udp_percent` | UDP 손실률 | 데이터에서 신뢰 가능한 변동이 부족함 |
| `poll_interval_s` | 폴링 간격(초) | retry/failed 정규화용 메타데이터. 2026-08-27 추가 |
| `connected_clients` | 연결 클라이언트 수 | 구분력이 제한적이라 제외 |
| `throughput_score`, `occupancy_score`, `retry_failed_score`, `jitter_score`, `congestion_score` | 라벨 생성용 중간 점수 | 정답 생성에 쓴 값이므로 입력하면 데이터 누수 |
| `label` | 정답 라벨 | 모델이 맞혀야 하는 정답 |

정규화는 train split의 feature별 min-max 기준으로 수행한다. 생성된 `scaler_params.json`을 val/test 및 실시간 추론에도 동일하게 적용해야 한다.

## congestion_score 계산식 (1차와 다름)

`project/scripts/collect_metrics.py`의 `calculate_scores()`에서 계산한다.

```text
congestion_score = 0.20 * throughput_score + 0.45 * occupancy_score + 0.20 * retry_failed_score + 0.15 * jitter_score
```

sub-score는 각각 아래 상한으로 0~1 clamp한 값이다.

| sub-score | 계산 | 상한 |
|---|---|---:|
| `throughput_score` | `throughput_mbps / THROUGHPUT_MAX_MBPS` | 150 Mbps |
| `occupancy_score` | `channel_occupancy_percent / 100.0` | 100% |
| `retry_failed_score` | `(tx_retries_per_s + tx_failed_per_s) / RETRY_FAILED_MAX_PER_SEC` | 6,250 /초 |
| `jitter_score` | `jitter_ms / JITTER_MAX_MS` | 300 ms |

`RETRY_FAILED_MAX_PER_SEC = 6250` = 옛 `RETRY_FAILED_MAX`(25,000) ÷ 옛 폴링 간격(~4초). 즉 폴링이 4초였을 때와 같은 캘리브레이션이되, 이제 폴링 주기가 바뀌어도 값이 안 흔들린다. 새 데이터가 쌓이면 재보정할 것.

label 경계는 1차와 동일: `<0.25`→0, `0.25~0.50`→1, `0.50~0.75`→2, `≥0.75`→3.

**1차(`ap_cleaned_strict`)는 `0.35 * throughput + 0.35 * occupancy + 0.20 * retry + 0.10 * jitter`를 그대로 쓴다.** 가중치를 바꾼 이유: 실측 stress_load 구간에서 label 2와 3의 sub-score 평균을 비교해보니 `throughput_score`(0.665→0.707)는 거의 차이가 없었던 반면 `occupancy_score`(0.449→0.898)와 `jitter_score`(0.512→0.802)는 뚜렷한 차이를 보였다. throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 기여가 거의 없었으므로, occupancy·jitter 비중을 높이고 throughput 비중을 낮췄다.

라벨링 로직을 바꾼 뒤에는 **AP를 다시 부하 테스트하지 않고도** 이미 모아둔 raw 데이터에 즉시 반영할 수 있다. 두 스크립트가 있다:

```powershell
python project\scripts\relabel_metrics_v2.py      # 가중치만 바뀐 경우
python project\scripts\remeasure_metrics_v2.py    # sub-score 공식 자체가 바뀐 경우
```

- `relabel_metrics_v2.py` — `metrics_v2.csv`에 저장된 4개 sub-score 컬럼을 새 가중치로 재조합해서 `congestion_score`/`label`만 갱신(원시 feature 불변).
- `remeasure_metrics_v2.py` — raw feature에서 4개 sub-score를 **처음부터 다시 계산**한다(공식·상한이 바뀌었을 때). 2026-08-27 `tx_retries_delta`→`tx_retries_per_s` 마이그레이션에 쓴 스크립트. `.bak` 백업 후 in-place로 덮어씀.

둘 다 실행 후 콘솔에 전/후 label 분포를 출력한다.

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

현재 결과(`power=1.0`, 2026-08-27 저녁 `tx_retries_per_s` 마이그레이션 후 5574행으로 재학습, best val balanced acc 80.7%):

| | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| Fixed theta | **91.2%** | 97.7% | 88.0% | 78.9% | **12.5% (1/8)** |
| Dynamic theta | **91.2%** | 97.3% | 88.5% | 79.7% | **12.5% (1/8)** |

- **전체 정확도는 역대 최고**(직전 최고 89.6%). retry feature가 폴링 주기 노이즈를 안 타게 되면서 전반적으로 더 잘 보정된 것으로 보인다.
- **Label 3 recall은 54.5%(6/11) → 12.5%(1/8)로 급락**. 마이그레이션이 폴링 아티팩트 label 3을 제거해서 test 표본이 11→8로 줄었고, 8개 중 1개 차이가 12.5%p라 소표본 노이즈가 극심하다. **숫자 자체보다 "이제 깨끗한 파이프라인으로 label 3을 다시 쌓아야 한다"가 요점이다** — 파이 유선 수집(폴링 1.1초, 5분이면 ~270행)으로 60/60·75/75를 반복.
- Exit 분포(fixed θ): Exit1 46.5%(정확도 98.6%) / Exit2 36.9%(91.1%) / Exit3 16.6%(70.9%).

참고: `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt`는 1차 체크포인트/스케일러로 2차 데이터를 잘못 평가했을 때(정확도 39.3%)의 진단 기록이다. 역사적 자료일 뿐 최종 결과가 아니다.

## 1차(`ap_cleaned_strict`)와의 차이

| 구분 | 1차 (`ap_cleaned_strict`) | 2차 (`ap_metrics_v2`) |
|---|---|---|
| 데이터 출처 | 인터넷 공개 데이터 가공 (2026-08-23 확인) | 팀이 구매한 GL.iNet Opal AP 실측 |
| 원본 raw | `raw/metrics_cleaned_strict.csv` (588행, 고정) | `project/scripts/metrics_v2.csv` (5490 data rows, 계속 누적) |
| congestion_score 가중치 | throughput 35% / occupancy 35% / retry 20% / jitter 10% | throughput 20% / occupancy 45% / retry 20% / jitter 15% |
| class weight | 미사용(plain cross-entropy) | `--class-weight-power` (기본 1.0) |
| checkpoint 경로 | `project/checkpoints/ap_cleaned_strict/` | `project/checkpoints/ap_v2/` |
| ONNX/INT8/Pi 배포 | 완료 | 아직 없음 |
| 상태 | 완료, archived, 더 이상 갱신 안 함 | 진행 중인 공식 재설계 라인 |

입력 feature 정의(9개), 제외 컬럼, dataloader(`utils/ap_dataloader.py`), feature 정의 파일(`utils/ap_features.py`), window size(10), Fixed/Dynamic 방식은 1차와 동일하게 공유한다.

## 핵심 검증 질문 (2026-08-27 정리)

AP 장비 성능 자체는 상수로 두고, 이 프로젝트가 실제로 답해야 하는 질문은:

> **공장 같은 환경에서 "단순히 기기가 많아서"가 아니라 다른 원인(경합·간섭·재전송·RF 열화)으로 혼잡이 생겼을 때 — 즉 채널 점유율만으로는 안 잡히는 혼잡 — 조기종료 LSTM이 그 혼잡 수준을 얼마나 정확히 잡아내는가.**

지금 상태의 구멍:

- **label 3(심각) 샘플이 거의 전부 `channel_occupancy_percent`가 100%에 포화된 순간이다** (8/26 밤 신규 3개도 모두 그랬다). 그러면 모델이 "occupancy≈100 → label 3"이라는 지름길만 학습했을 가능성이 크고, 공장에서 흔한 "occupancy는 50~70%인데 retry가 폭증한" 상황(우리가 잡아냈다고 주장하고 싶은 바로 그 케이스)에서 label 3을 놓칠 수 있다 — 아직 검증이 안 됐다.
- `congestion_score`도 occupancy 45%라 ground truth 자체가 occupancy-heavy다. retry 20% + jitter 15% = 35%로 non-occupancy 혼잡도 라벨에 반영되지만, **그런 샘플이 데이터에 적다.**

> **2026-08-27 후속**: 소패킷 실측으로 이게 산술적 문제임이 확인됐다 — occupancy 66%면 나머지 축이 다 맥스여도 congestion_score가 ~0.71에서 막혀 label 3 불가. **가중치를 안 바꾸면 "occupancy 아닌 label 3"은 물리적으로 안 나옴.** 해결책으로 라벨 정의를 표준 문턱 + victim 프로브 + `max` 조합으로 재설계하는 설계가 확정됨: **`docs/yongsang/congestion_label_redesign.md`**. 아래 "지금 할 수 있는 진단"은 재설계 전 현행 데이터 분석용으로만 유효.

그래서 부하 생성 방법 실험(`docs/yongsang/ap_crash_analysis.md` "부하 생성 방법 대안")의 목적은 "AP를 더 세게 굴리기"가 아니라 **occupancy가 아니라 retry/jitter가 주도하는 label 2/3 샘플을 확보**하는 것이다.

데이터 안 늘리고 지금 할 수 있는 진단 (재설계 전 현행 데이터용):

1. test label 3의 정답/오답을 `channel_occupancy_percent` 구간별로 쪼개기 — 전부 90%+ 인가?
2. 각 test 샘플의 `congestion_score`를 어느 sub-score가 주도했는지로 분류 → 주도 요인별 recall
3. `channel_occupancy_percent`를 입력 feature에서 뺀 ablation 학습 — "occupancy 외 신호"를 실제로 쓰는지 측정

## 알려진 한계 / 남은 작업

- **Label 3(심각) 데이터가 여전히 얇다 — 오히려 줄었다.** 2026-08-27 `tx_retries_per_s` 마이그레이션으로 raw label 3 85→56, windowed test 11→8. recall이 실행마다 크게 흔들린다(역대 12.5%~66.7%). 파이 유선 수집(폴링 1.1초)으로 60/60·75/75를 반복해서 깨끗한 파이프라인 기준으로 다시 쌓는 게 최우선.
- **Label 2/3 트레이드오프가 절벽형** — class weight power 중간값 튜닝이 잘 안 먹힌다. label 3 표본이 더 늘어야 완만한 튜닝이 가능할 것으로 보인다.
- **AP(Opal) 하드웨어 안정성 문제** — 상세 원인 분석은 `docs/yongsang/ap_crash_analysis.md`. **2026-08-27 세션 기준 최신 결론: "몇 대가 붙었는가"보다 "각 폰이 AP 와이파이에 제대로·대칭적으로 붙어있는가"가 핵심 변수.** S26/191 두 폰 모두 단독으로는 40~150Mbps까지 전 구간 크래시 없이 완주했고, 191+S26 콤보도 **시작 전 두 폰 와이파이를 재연결하면** 120초·5~7분 모두 크래시 없이 완주했다. 신호가 강한 S26이 채널을 독점하고 신호가 약한 191이 굶는 비대칭(Wi-Fi capture effect)이 반복 관측됨(191 RSSI 열세 추정) — 191이 실질적으로 빠지면 콤보가 사실상 단독 부하가 되어 label 3이 안 나온다. 부하 스위트스팟은 **191=60M/S26=60M(2~7분)**이 기준선이고, 8/26 밤에 **정확히 대칭인 75M/75M**이 label 3 생성 후보로 추가됨(3회 합산 label 3 5개, 재현성 불완전, 단 3회 모두 크래시 없이 완주). **상한은 확실: 80M/80M은 10분 시도에서 완전 크래시(물리 재부팅 필요), 60/60도 500초를 넘기면 완전 크래시** — 안전 구간 대략 300~420초. "191 폰 개별 하드웨어 문제"라는 이전 결론은 191의 다단계(40/70/100/120/150M) 연속 생존으로 반증됨. 자세한 내용은 `.work-log/current.md`와 `docs/yongsang/ap_crash_analysis.md`.
- **실시간 폴링 자기참조 지연(2026-08-27 코드 조치, 미검증)** — `collect_metrics.py`가 매 루프마다 새 SSH를 띄우던 걸 지속 SSH 세션(`APPoller`)으로 전환했다. 문법 검사만 됐고 실기기 검증은 다음 AP 세션 과제. 관리 트래픽이 여전히 같은 무선 채널을 탄다는 근본 구조는 그대로라, 유선 관리채널 분리(collector를 라즈베리 파이로, Opal LAN 포트를 별도 서브넷으로)를 다음 세션 실행 대상으로 설계 확정함.
- **ONNX/INT8/Raspberry Pi 배포 파이프라인 없음** — 1차의 `export_onnx_ap.py`/`export_onnx_int8_ap.py`/`prepare_pi_bundle_ap.py`를 새 경로(`ap_metrics_v2`, `ap_v2`)로 재사용할지, 별도 스크립트를 만들지 아직 미정.
- **SDN-style / Baseline 비교 없음** — 1차처럼 Baseline LSTM, SDN-style Early Exit과의 비교표가 아직 이 라인에 없다. `train_ap_baseline_lstm.py`, `train_ap_sdn.py`를 `--data-dir project\data\ap_metrics_v2 --checkpoint-dir project\checkpoints\ap_v2`로 재사용 가능한지는 검증 필요.
- **1차를 새 가중치로 재라벨링할지 미결정** — 팀 논의 필요 항목. 재라벨링하면 1차의 완료된 학습/ONNX/Pi 배포 전체를 다시 돌려야 하므로 신중히 결정한다.
