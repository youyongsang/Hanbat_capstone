# Claude용 프로젝트 맥락 정리 (capstoneDesign2 브랜치)

이 문서는 Claude 또는 외부 AI 도구에 현재 프로젝트 맥락을 전달하기 위한 요약 파일이다. 이 브랜치(`capstoneDesign2`)는 `yongsang` 브랜치에서 분기해 코드를 정리한 브랜치다.

- **제외**: 1학기 4-feature 학습/평가/ONNX 파이프라인 코드와 1차 실측(`ap_cleaned_strict`, 588행 — 실제로는 인터넷 공개 데이터 기반이었음) 전체 파이프라인. 그 코드가 필요하면 `yongsang` 브랜치를 참고한다.
- **포함**: 2차 실측 데이터 라인(`ap_metrics_v2` → `redesign` → `redesign2`, 이 문서의 핵심 대상), docs 문서 전체(팀원별 가이드라인·work log 포함, 코드는 없어도 기록은 다 남겨둠), 그리고 1학기 Raspberry Pi 실측 결과(`project/results/hojung/`, `project/results/final_figures/`, `project/deploy/raspberry_pi/`)는 `origin/hojung`에서 가져와 유지한다 — staged ONNX 기준 Baseline/Fixed/Dynamic Early Exit의 Pi 실측 지연 비교 자료다. 이건 1학기 4-feature(시뮬레이션) 모델 기준이며 2차 실측 라인과는 무관하니 섞어서 비교하지 않는다.

> **이 문서보다 최신인 것**: `.work-log/current.md`(세션마다 갱신, 최신 수치·다음 할 일). CLAUDE.md의 수치가 work-log와 어긋나면 work-log가 맞다. 형제 문서 `project/README_AP_V2.md`와 `docs/yongsang/congestion_label_criteria.md`는 **아직 9-feature·weighted-sum 시절 기준이라 stale하다** — 라벨 정의는 `docs/yongsang/congestion_label_redesign.md`, feature 목록은 `project/utils/ap_features.py`가 authoritative.

## 프로젝트 개요

본 프로젝트는 AP(무선 공유기) 기반 무선 환경에서 트래픽 혼잡 상태를 시계열 feature로 판단하는 LSTM 기반 분류 시스템이다. 최종 목표는 Raspberry Pi 같은 엣지 장비에서 실시간 혼잡 수준을 추론하고, Early Exit 및 ONNX/INT8 경량화를 통해 추론 지연을 줄이는 것이다.

혼잡 라벨은 4단계이다.

| Label | 의미 |
|---|---|
| 0 | 정상 |
| 1 | 경고 |
| 2 | 혼잡 |
| 3 | 심각 |

**발표자료(`docs/캡스톤디자인I_최종발표.pptx`) 슬라이드 8 "정량적 목표"**:
- 목표1 = Raspberry Pi 환경에서 **혼잡 분류 정확도 95% 이상** — 2026-08-30 class-weight-power=0.0 승격으로 Baseline 92.3% / Early Exit 90.6%(fp32)까지 옴, **아직 미달 (진짜 남은 숙제, 목표까지 test 310창 중 8.4개)**.
- 목표2 = **추론 지연 < 1ms** — 달성 (2026-08-30 Pi INT8 재측정: Baseline 0.74ms / Early Exit 0.54ms / SDN 0.53ms, 전부 <1ms).
- SDN 비교는 원래 정량 목표가 아니다. 핵심 기여 주장은 "간섭 감지에 Early Exit LSTM 구조를 최초 적용"이고, "혼잡 판단 → 채널 전환 필요 여부 + 전환 명령 후보 생성"까지가 최종 목표 문장(슬라이드 7).

## 배경 (1학기 → 1차 → 2차)

- **1학기**: `project/data/real` 기준 4-feature(RPS, channel occupancy, packet loss, latency) 실험. **시뮬레이터 생성 데이터**(`traffic_simulator.py`, 공장 운영 패턴 4시나리오). 라벨은 **채널 점유율 단일 문턱**(40/65/85)으로 자동 부여 → Threshold baseline과 순환 구조. Threshold / Baseline LSTM / Early Exit Fixed·Dynamic / SDN-style을 비교. 이 브랜치에는 코드가 없다 — `yongsang` 브랜치 참고. 정리본: `docs/capstone1_summary.html`.
- **1차(`ap_cleaned_strict`, archived)**: 방학 중 "실제 AP 장비 실측"을 목표로 9-feature 파이프라인을 구축했으나, 2026-08-23 확인 결과 실제로는 인터넷 공개 데이터를 가공한 것이었다. 9-feature 파이프라인, congestion score 라벨링, Early Exit/SDN 비교 구조를 미리 검증하는 역할은 했다. 재라벨링/재학습 안 하기로 결정(2026-08-24), 코드·데이터는 `yongsang` 브랜치에만.
- **2차(이 브랜치의 공식 진행 라인)**: 2026-08-23부터 팀이 실제로 구매한 GL.iNet Opal(GL-SFT1200) AP에서 `project/scripts/collect_metrics.py`로 직접 라이브 수집한 진짜 실측 데이터. "실제 AP 장비 실측" 목표를 처음으로 만족. 아래 세 단계로 진화했다.

### 2차 데이터 라인의 3단계

| 단계 | 데이터셋 이름 | feature | 라벨 정의 | 상태 |
|---|---|---|---|---|
| 초기 | `ap_metrics_v2` | 9개 | weighted-sum congestion_score (occ 0.45) | archived — 라벨이 occupancy 문턱과 사실상 동일한 순환논리로 폐기 |
| 재설계 | `ap_metrics_v2_redesign` | 6개 | `max(표준 앵커)` + victim 프로브 (2026-08-27) | 캘리브레이션용 중간 단계 |
| **현행** | **`ap_metrics_v2_redesign2`** | **7개** | 위와 동일 (`sta_tx_bitrate_mean` feature만 2026-08-29 추가) | **본수집, 활성 라인** |

"AP 실측 데이터"·"2차"는 지금 시점에서 항상 **`ap_metrics_v2_redesign2` + 7-feature**를 가리킨다.

## 2차 AP 실측 (현행: ap_metrics_v2_redesign2)

### 파일 위치

```text
project/scripts/collect_metrics.py                     라이브 수집 (victim 프로브 ProbeRunner + 지속 SSH APPoller + congestion_score 계산)
project/scripts/metrics_v2_pi_redesign2_relabeled.csv  현행 raw CSV (2115행, raw label 3 202개) ← canonical
project/scripts/metrics_v2.csv                         레거시 raw (5574행, 프로브·tx_packets 없음 + retry 3× 버그) — 사전학습/ablation용으로만
project/scripts/remeasure_redesign.py                  raw feature에서 sub-score·라벨 전부 재계산 (재설계 공식 적용)
project/scripts/prepare_ap_metrics_dataset.py          windowed train/val/test 변환
project/scripts/train_ap_early_exit.py                 Early Exit LSTM 학습 (--seed, --class-weight-power, --exit-loss-weights)
project/scripts/train_ap_baseline_lstm.py              Baseline(EE 없음) LSTM
project/scripts/train_ap_sdn.py                        SDN-style LSTM (EarlyExitLSTM과 백본 동일, 임계값 정책만 다름)
project/scripts/evaluate_ap_early_exit.py              평가
project/scripts/forecast_eval_redesign.py              조기경보(k폴링 뒤 라벨) 재평가
project/scripts/generate_ap_comparison.py              Baseline/SDN/Proposed 비교표 생성
project/data/ap_metrics_v2_redesign2/                  현행 windowed train/val/test, scaler_params.json, dataset_summary.json
project/checkpoints/ap_v2_redesign2/                   현행 Early Exit LSTM 체크포인트 (배포 기준)
project/deploy/raspberry_pi_ap_v2/                     Pi 배포 번들 (ONNX staged/unified/int8_v2 + bench 스크립트)
project/results/yongsang/ap_v2_redesign2_eval_report.txt          현행 평가 리포트
project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt Pi 실측 지연 비교 (Baseline/SDN/Proposed)
project/results/yongsang/ap_v2_redesign2_forecast_eval.txt        조기경보 프레이밍 결과
project/results/yongsang/ap_model_comparison_redesign2.{txt,csv}  아키텍처 비교표
project/models/ap_early_exit_lstm.py    APEarlyExitLSTM (EarlyExitLSTM 상속, input_size = len(AP_FEATURE_COLUMNS) = 7)
project/models/early_exit_lstm.py       EarlyExitLSTM 베이스, multi_exit_loss(weights 파라미터)
project/utils/ap_features.py            AP_FEATURE_COLUMNS = 7개 feature (authoritative)
project/utils/ap_dataloader.py          windowed CSV → DataLoader
docs/yongsang/congestion_label_redesign.md  현행 라벨 정의 (max 앵커 + victim 프로브) ← 라벨 관련은 이 문서가 authoritative
docs/yongsang/ap_crash_analysis.md      AP(Opal) 반복 크래시 원인 분석
docs/yongsang/onnx_early_exit_redesign.md  ONNX Early Exit 배포 재설계 (staged → unified If 노드 → INT8 재조립)
.work-log/current.md                    세션별 최신 진행 상황 (이 문서보다 최신)
```

아카이브(덮어쓰지 않고 보존): `project/data/ap_metrics_v2_redesign2_6feat_archived_20260829/`, `project/checkpoints/ap_v2_redesign2_6feat_archived_20260829/`, `project/checkpoints/ap_v2_redesign2/archived_uniform_ee_weights_20260829/`, `project/results/yongsang/*_6feat_archived_20260829.*` 등.

### 입력 feature (7개) — `project/utils/ap_features.py`

```text
throughput_mbps
channel_occupancy_percent
tx_retry_ratio                 # (tx_retries + tx_failed) / (retries + failed + tx_packets), 비율이라 폴링 주기 무관
rssi_dbm
rssi_delta_db
rssi_moving_avg_dbm
sta_tx_bitrate_mean            # 이번 폴링에 실제 송신한 station들의 tx bitrate 평균
```

**변천**: 1학기 4개 → 초기 `ap_metrics_v2` 9개 → 라벨 재설계로 6개(2026-08-27) → 7개(2026-08-29).

- **6→7 (`sta_tx_bitrate_mean` 추가, 2026-08-29)**: occ 60~72% 구간(나머지 6개 feature 평균이 label 2/3 사이에 완전히 동일해지는 구간)에서 label 2 vs 3이 Cohen's d=0.52로 갈라짐. 5개 랜덤 시드 검증에서 exit-loss 가중치와 무관하게 7-feature가 6-feature보다 Label3 F1 +5~11pt. (가설 "혼잡할수록 bitrate 하락"은 실측과 반대 — 부하 테스트라 혼잡 구간에서 오히려 오름. 신호 방향이 반대일 뿐 변별력은 유효.)
- **9→6 (라벨 재설계, 2026-08-27)**: `latency_ms`·`jitter_ms`를 모델 입력에서 **제거** — 이들이 라벨을 만드는 축이자(아래 참조) 배포 시점엔 없는 측정(victim 프로브 필요)이라, 모델에 주면 정답 leakage. 모델은 "채널 상태만 보고 victim QoS를 예측"해야 함. `tx_retries_per_s`+`tx_failed_per_s` → `tx_retry_ratio` 하나로 통합.

모델 입력 제외 컬럼(`dataset_summary.json`의 `model_excluded_columns`): `timestamp`, `scenario`, `poll_interval_s`, `channel_occupancy_method`, `packet_loss_udp_percent`, `connected_clients`, `latency_ms`, `probe_jitter_ms`, `probe_loss_pct`, `probe_ok`, `tx_retx_delta`, `tx_packets_delta`, sub-score 6개, `congestion_score`. 정규화는 train split의 min-max 기준이며 `scaler_params.json`을 val/test·실시간 추론에 동일하게 써야 한다.

### congestion_score = 정답 라벨 판별 기준 (재설계, 2026-08-27~)

상세·근거는 `docs/yongsang/congestion_label_redesign.md`. 요약:

```text
congestion_score = max(occupancy_score, jitter_score, loss_score, latency_score)

label = 0 if score < 0.25 | 1 if < 0.50 | 2 if < 0.75 | 3 if ≥ 0.75   (경계는 1차·초기와 동일)
```

- 각 축을 4개 앵커(경고 0.25 / 혼잡 0.5 / 심각 0.75 / 완전 1.0)에 piecewise-linear 매핑, [0,1] clamp. 앵커 값은 **국제 표준**에서: occupancy는 Cisco/Aruba WLAN 설계 가이드(40/55/75/90%), jitter·loss·latency는 ITU-T Y.1541 / G.114 / RFC 4594 / Cisco Enterprise QoS.
- `jitter_score`·`loss_score`는 **victim 프로브**(부하와 별개로 흘리는 300kbps 경량 UDP 스트림, `iperf3 -u -b 300k`)가 그 스트림 자체에서 겪은 IPDV/손실을 측정한 값. "이 혼잡 속에서 VoIP 통화 한 통이 얼마나 깨지나". `latency_score`는 ping RTT/2 (편도 추정).
- **조합이 `max`인 이유**: 표준은 "각 축이 언제 나쁜가"는 주지만 "어떻게 합치나"는 안 준다. `max`면 가중치 논쟁이 원천 봉쇄됨 — `label 3 = 최소 한 축이 표준 심각 문턱 돌파`.
- **"실패 = max"**: 채널이 실제로 바쁜 상태(`throughput ≥ 3Mbps or occupancy ≥ 40%`)에서 victim 경로가 완전히 죽으면(ping 무응답 / 프로브 stale) 해당 축을 1.0으로 본다 — 안 그러면 occupancy 단독으로 조용히 되돌아감. 유선 SSH로 AP 텔레메트리를 방금 정상 파싱했으므로 AP는 살아있고 무선 채널만 막힌 것.
- **`throughput_score`·`retry_score`는 라벨 축 아님** (label 2/3 변별력 없음 / 이 2.4GHz AP는 idle에도 retry_ratio med 18%라 QoS 피해의 독립 증거가 아님) — 계산은 하되 `max`에서 제외, 모델 입력 feature로는 유지.

**핵심 논리 — LSTM이 왜 필요한가**: `max(표준 앵커)`는 그 자체로 규칙 기반 분류기다. 그런데 배포 시스템엔 victim 프로브가 없다(협조 싱크 + 지속 스트림 필요). 모델은 **AP-side 텔레메트리(점유율·retry·RSSI·bitrate)만 보고 "지금 victim QoS가 깨지고 있는가"를 예측**해야 한다 — 이건 규칙으로 못 한다. 거기에 시계열 추세 + Early Exit 효율.

### 학습 설정

- 아키텍처: hidden_size 128, num_layers 3, dropout 0.2, num_classes 4. (김호중 baseline과 동일하게 맞춤. 세션 내내 고정값 — **한 번도 스윕 안 함**.)
- Optimizer Adam, lr 0.001, batch 32, epochs 50. 체크포인트 선택 기준 = **val balanced accuracy** (raw accuracy 아님 — 다수 클래스 찍는 에폭 선택 방지).
- **multi-exit loss 가중치 = 균등 0.3/0.3/0.4** (`--exit-loss-weights`). SDN 스타일 0.15/0.30/0.55로 바꾸면 exit3 심각 탐지에 유리하다는 가설이 있었으나, 다중 시드(5개) 검증에서 차이가 표준편차보다 작아 **노이즈로 판명 → 균등 유지**. (SDN-style 백본과 EE 백본이 의도적으로 동일해서, 같은 loss 가중치+시드면 사실상 같은 네트워크가 되는 문제도 있음.)
- **class weight power = 0.0** (`--class-weight-power`, `compute_class_weights`) — 클래스 가중치를 아예 안 씀(plain CE). **2026-08-30 재스윕(0.0/0.1/0.15/0.2/0.3/0.5/0.7/0.85/1.0, 3시드씩)에서 power=0.0이 정확도(91.3%±0.5% vs power=1.0의 87.0%±1.1%)·Label3 F1(69.8% vs 63.2%) 둘 다 최고, 트레이드오프 없음**이라 기본값을 1.0→0.0으로 변경. 옛 1.0은 2026-08-23에 4-feature·train label3=23개 시절 "power≤0.85면 label3 recall 절벽" 때문에 정한 값인데, 7-feature·train label3=141개가 되면서 그 절벽이 사라짐(옛 결정을 재검증 안 해 계속 손해보고 있었음). 상세: work-log 4~5차 체크포인트(2026-08-30).
- `train_*.py` 전부에 `--seed` 옵션 있음(기본 `None` = 기존 동작). **하이퍼파라미터 A/B 비교는 반드시 여러 시드로** — 단일 실행 비교가 노이즈였던 전례가 여러 번 있었다.

### 최신 평가 결과 (2026-08-30, 7-feature `ap_metrics_v2_redesign2`, **class-weight-power=0.0 승격**)

정확한 최신 수치는 `.work-log/current.md` 7차 체크포인트. windowed test 310 샘플 (label 분포 0:95 / 1:67 / 2:117 / 3:31). fp32 eval:

- **Baseline (EE 없음)**: 전체 **92.3%**. Label 0 97.9% / 1 97.0% / 2 94.0% / 3 58.1%. (3시드 중 val 최고 = seed0.) **95% 목표에 가장 근접 — 목표까지 test 310창 중 8.4개.**
- **SDN-style**: 전체 90.6%. Label 3 recall 48.4%.
- **Early Exit Fixed θ** (5시드 중 val 최고 = seed4): 전체 **90.6%**. Label 0 97.9% / 1 97.0% / 2 90.6% / 3 54.8%. Label3 F1 68.0%. Exit 종료율 30.0 / 50.6 / 19.4%.
- **Early Exit Dynamic θ**: 전체 91.0%. Label 3 recall 54.8%, F1 69.4%.
- **EE 5시드 평균(fixed)**: acc 90.7%±0.7%, L3 F1 64.5%±5.5%, L3 recall 51.0%±6.6% — **EE 시드 분산이 큼**(seed1은 L3 recall 38.7%, seed3은 test 91.9%인데 val 최저라 선택 안 됨). 배포 체크포인트는 정직한 selector(val balanced acc) 준수.
- **INT8 v2 (EE seed4)**: fixed 90.3% / F1 65.3%, dynamic 90.6% / F1 66.7%. unified fp32는 PyTorch와 310/310 일치, INT8은 308/310·309/310(양자화 노이즈).
- **power=1.0 → 0.0 효과**: 전 모델 정확도 +1.6~3.6pt. 대신 EE/SDN label3 recall은 내려감(power=1.0이 label3 과보호하던 것 — 4차 confusion matrix 분석대로). EE label3 F1은 precision 상승으로 오히려 65.5→68.0.
- **서사**: Baseline(EE 없음)이 정확도 1위 → Proposed(Early Exit)의 가치는 정확도가 아니라 **속도·효율**(목표2 <1ms) + "간섭 감지에 EE 최초 적용". 95% 목표는 **Baseline 92.3%를 기준선**으로.
- **Pi INT8 재측정 완료** (2026-08-30, power=0.0, `capstone@192.168.8.109`, test 310창): Baseline **0.739ms** / SDN 0.534 / Proposed Fixed **0.540** / Dynamic 0.555. 전부 목표2(<1ms) 달성. power=1.0 대비 EE Fixed 0.641→0.540(-16%), **이제 EE가 Baseline보다 -27% 빠름**(exit3 도달률 52%→19%). SDN≈Proposed Fixed 속도·정확도 동률(Proposed가 Label3 우위).
- **Confusion matrix 분해 (power=1.0 시절)**: label 2 오답이 label 3보다 개수가 많았음. power=0.0에서 label2가 상당히 정상화됨(94.0%). label2→1 오답은 occ 55~57% 앵커 경계 측정 노이즈, label2→3 오답은 occ 60~72% 정보 부족 구간.
- **조기경보(forecasting) 프레이밍** (`forecast_eval_redesign.py`): k=3폴링(≈3~6s) 앞 escalation(현재 not-severe → k 뒤 severe) recall 61.5%, occupancy 규칙은 구조적으로 0/13. "점 분류 95%"와는 다른 지표라 발표 목표1을 직접 만족하진 않지만 대안 서사로 유효.

### 알려진 한계

- **Label 3(심각) 표본이 여전히 얇다** (test 31개) — recall이 실행마다 흔들린다. AP가 부하 종류에 따라 반복 크래시해서 추가 수집이 제한적.
- **occupancy 55~57% 경계 노이즈**: label 1/2가 물리적으로 거의 같은 채널 상태에서 뒤집힘. 라벨링 쪽 rolling median 스무딩(window 3·5)을 전체 파이프라인으로 검증했으나 — 같은 스무딩 값이 다른 앵커 경계(75% 등)에도 쓰여 다른 데서 새 오답이 생겨 **순효과 없음(두더지 잡기) → 라벨링 스무딩 방향 폐기**. 모델 입력 쪽 스무딩은 미시도.
- **AP(Opal) 반복 크래시**: "몇 대 붙었나"보다 "각 폰이 대칭적으로 붙었나"가 핵심 변수 — 신호 강한 S26이 채널 독점하고 약한 191이 굶는 Wi-Fi capture effect. 부하 스위트스팟 191=60M/S26=60M, 안전 구간 대략 300~420초 (60/60도 500초 넘기면 완전 크래시, 80/80은 10분 시도에서 물리 재부팅). 상세: `docs/yongsang/ap_crash_analysis.md`.
- **유선 관리채널**: `collect_metrics.py`가 지속 SSH(`APPoller`)로 전환됐고 victim 프로브(`ProbeRunner`)가 추가됨. collector를 라즈베리 파이로 옮겨 관리 트래픽을 무선 채널에서 분리하는 구조 확정.
- **ONNX/Pi 배포**: staged(세션 3개) → `torch.jit.script`+ONNX `If` 노드 단일 그래프(`export_onnx_ap_unified.py`) → INT8은 staged(flat)로 먼저 양자화 후 손수 재조립(`export_onnx_ap_unified_int8_v2.py`, `If` 서브그래프 안 LSTM을 양자화 도구가 건너뛰는 한계 우회). baseline 대비 대략 -60% 내외 (1학기 4-feature 자료로 두 현상 교차검증). Pi latency 주장 시 `docs/yongsang/onnx_early_exit_redesign.md`의 결론을 따르고 staged/fp32-only 수치를 최종 결과로 인용하지 않는다.
- 하이퍼파라미터(lr/epochs/batch) 스윕 미실시. hidden_size·dropout은 2026-08-30에 스윕(각 2시드) — 현재 기본값(128/0.2)이 이미 최적, 추가 이득 없음. class-weight-power는 2026-08-30 재스윕 완료(→ 0.0).

### 재현 명령어

라벨 재계산(공식/앵커가 바뀌었을 때, raw feature에서 전부):

```powershell
python project\scripts\remeasure_redesign.py
```

windowed 데이터 변환:

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2_pi_redesign2_relabeled.csv --out-dir project\data\ap_metrics_v2_redesign2 --overwrite
```

Early Exit LSTM 학습(균등 exit 가중치, 시드 고정):

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint-dir project\checkpoints\ap_v2_redesign2 --epochs 50 --batch-size 32 --seed 0
# (--class-weight-power 기본값 0.0, --exit-loss-weights 기본값 0.3 0.3 0.4)
```

평가:

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint project\checkpoints\ap_v2_redesign2\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_v2_redesign2_eval_report.txt
```

## 자주 헷갈리는 점

### torch DLL 로딩 실패

이 노트북(들)의 anaconda base 환경에서 `import torch`가 `OSError: [WinError 1114]`로 실패한다. 별도 conda 환경(`capstone`)에 torch(CPU)+pandas+numpy를 설치해서 써야 한다.

### 정확도가 이상하게 낮게 나올 때 의심할 점

1. raw CSV를 그대로 넣지 않았는지 확인 — 모델은 window size 10으로 변환된 `test.csv`를 쓴다.
2. feature 개수/순서가 `ap_features.py`의 **7개** 정의와 동일한지 확인. (ONNX 재조립 스크립트가 `[1,10,6]`으로 하드코딩돼 있던 버그 전례 — Pi에서 "Got 7 Expected 6" 에러.)
3. `label`, `congestion_score`, sub-score, `probe_*`, `latency_ms`, `jitter_ms`가 입력 feature에 섞이지 않았는지 확인.
4. `scaler_params.json` 기준이 다른지 확인 (재라벨링만 하고 재변환을 안 하면 라벨과 스케일러가 어긋난다).
5. 데이터 디렉터리가 `ap_metrics_v2_redesign2`(7-feat)인지, 아카이브된 `_6feat_archived_*`를 잘못 가리키는 건 아닌지 확인.

## Claude가 추가로 참고해야 할 파일

1. `.work-log/current.md` — 세션별 최신 진행 상황. **이 문서보다 항상 최신.** 최신 수치·다음 할 일은 여기.
2. `docs/yongsang/congestion_label_redesign.md` — 현행 라벨 정의(max 앵커 + victim 프로브)와 그 근거. 라벨 관련 질문은 여기가 authoritative (`congestion_label_criteria.md`는 구 정의).
3. `project/utils/ap_features.py` — 현행 7개 feature 정의 + 변천 주석.
4. `docs/yongsang/ap_crash_analysis.md` — AP 반복 크래시 원인 분석.
5. `docs/yongsang/onnx_early_exit_redesign.md` — ONNX Early Exit 배포 재설계(staged → unified If 노드 → INT8 재조립). Pi latency 주장은 이 문서 결론을 따른다.
6. `docs/capstone1_summary.html` — 1학기(4-feature 시뮬레이션) 활동·지표 정리.
7. `project/models/ap_early_exit_lstm.py`, `project/utils/ap_dataloader.py` — 실제 코드 흐름.
8. `project/README_AP_V2.md`, `docs/yongsang/congestion_label_criteria.md` — **stale (9-feature·weighted-sum 시절).** 역사적 맥락용으로만.

## Claude에게 중요한 해석 기준

1. 이 브랜치에는 1학기 4-feature 코드와 1차(`ap_cleaned_strict`) 코드가 없다. 그 자료는 `yongsang` 브랜치.
2. "AP 실측 데이터"·"2차"는 지금 항상 **`ap_metrics_v2_redesign2` + 7-feature + max-앵커 라벨**을 가리킨다. 초기 `ap_metrics_v2`(9-feature, weighted-sum)는 archived.
3. **라벨 입력 ≠ 모델 입력.** 라벨은 victim 프로브(jitter/loss)·ping(latency)에 의존하고, 모델은 그걸 못 본다. 모델의 일은 채널 상태만으로 victim QoS를 예측하는 것.
4. 하이퍼파라미터/가중치 A/B 비교 수치는 **다중 시드로 검증됐는지** 먼저 확인한다. 단일 실행 비교가 노이즈였던 전례가 여러 번 있었다(SDN 가중치 승격 → 철회 등).
5. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다 — 최종 속도 주장은 Raspberry Pi INT8 실측 기준(현재 0.6~0.8ms, 목표2 <1ms 달성).
6. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책(고정 θ vs 변동률 기반 θ)만 바꿔 평가하는 구조이다.
7. 남은 핵심 숙제는 **Pi 정확도 95%**(2026-08-30 power=0.0 승격 후 Baseline 92.3% / EE 90.6% fp32, Pi 재측정 미실시). Baseline이 정확도 1위 — Proposed의 가치 주장은 속도·효율.
