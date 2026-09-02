# Claude용 프로젝트 맥락 정리 (capstoneDesign2 브랜치)

이 문서는 Claude 또는 외부 AI 도구에 현재 프로젝트 맥락을 전달하기 위한 요약 파일이다. 이 브랜치(`capstoneDesign2`)는 `yongsang` 브랜치에서 분기해 코드를 정리한 브랜치다.

- **제외**: 1학기 4-feature 학습/평가/ONNX 파이프라인 코드와 1차 실측(`ap_cleaned_strict`, 588행 — 실제로는 인터넷 공개 데이터 기반이었음) 전체 파이프라인. 그 코드가 필요하면 `yongsang` 브랜치를 참고한다.
- **포함**: 2차 실측 데이터 라인(`ap_metrics_v2` → `redesign` → `redesign2`, 이 문서의 핵심 대상), docs 문서 전체(팀원별 가이드라인·work log 포함, 코드는 없어도 기록은 다 남겨둠), 그리고 1학기 Raspberry Pi 실측 결과(`project/results/hojung/`, `project/results/final_figures/`, `project/deploy/raspberry_pi/`)는 `origin/hojung`에서 가져와 유지한다 — staged ONNX 기준 Baseline/Fixed/Dynamic Early Exit의 Pi 실측 지연 비교 자료다. 이건 1학기 4-feature(시뮬레이션) 모델 기준이며 2차 실측 라인과는 무관하니 섞어서 비교하지 않는다.

> **이 문서보다 최신인 것**: `.work-log/current.md`(세션마다 갱신, 최신 수치·다음 할 일). CLAUDE.md의 수치가 work-log와 어긋나면 work-log가 맞다. 형제 문서 `project/README_AP_V2.md`와 `docs/yongsang/congestion_label_criteria.{md,html}`는 **가중합 시절 기준이라 stale하다** — 라벨 정의는 `docs/yongsang/congestion_label_redesign.{md,html}`, feature 목록은 `project/utils/ap_features.py`(상세는 `docs/yongsang/model_features.{md,html}`)가 authoritative.

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
- 목표1 = Raspberry Pi 환경에서 **혼잡 분류 정확도 95% 이상** — 발표 시스템 = Early Exit(Proposed) 기준. 2026-09-02(18차): window 12 + 데이터 2551 + **라벨 지속성 게이트 k=2/m=2**(17차의 임시 k=3/m=2를 k·m 스윕으로 교체) → **EE 배포 Fixed 92.3% / Dynamic 92.9%, 5시드 평균 EE Fixed 91.7±0.7 · Dynamic 92.3±0.7**. 단일 배포 최고는 SDN seed0 **92.3%**(T=0.72), Baseline seed2 89.9%(약한 draw). **17차의 SDN 94.8%보다 낮아졌지만 이는 k2m2의 손해가 아니라 이번 draw가 5시드 평균에 더 가까운 정직한 값이기 때문**(17차 94.8%는 SDN 5시드 평균 93.3±0.9보다 +1.6σ 튄 값이었음) — 상한선: Baseline 5시드 92.7±1.5, SDN 91.9±1.0. **k2m2의 실질 성과는 L3(심각) recall·F1이 전 모델 6~10pt/1.5~3.0pt 개선된 것** (예: EE Fixed L3 recall 75.2→85.7%, F1 82.9→85.9%) — 좁은 의미의 95% 문턱과는 별개로 실제 심각 탐지력은 뚜렷이 나아짐. 남은 오답 구간은 k2m2로 재확인 필요(k3m2 시절엔 지속형 3↔2 + occ 72~73 경계로 분석됨).
- 목표2 = **추론 지연 < 1ms** — 달성. **2026-09-02 Pi 재측정 (window 12, 2551+k2m2게이트, test 365창)**: Baseline 0.864 / EE Fixed 0.625 / EE Dynamic 0.632 / SDN 0.575ms — 전부 avg <1ms. EE Fixed가 Baseline −28%. per-exit는 EE가 SDN보다 전 stage 가벼움(0.327/0.644/0.953 vs 0.332/0.646/0.962) — SDN 평균이 낮은 건 T=0.72가 exit1을 front-load한 것.
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
project/scripts/live_congestion.py                     실시간 혼잡 감지 라이브 추론 루프 (APPoller 재사용 → 7-feature → window 12 → scaler → ONNX INT8 → 라벨. victim 프로브 없음. 라벨 히스테리시스로 스파이크 억제). repo·Pi 번들 양쪽 동작. window 12 + 게이트 ONNX Pi scp 완료 (2026-09-02)
project/demo/demo_server.py + demo.html                 데모 웹 대시보드 — 버튼으로 계단형 부하(두 폰 iperf3) 걸고 모델 실시간 혼잡 예측을 SSE로 표시. live_congestion 로직 재사용. `python project/demo/demo_server.py` → localhost:8000. 상세: project/demo/README.md
project/scripts/metrics_v2_pi_redesign2_relabeled.csv  현행 raw CSV (2551행, k=2/m=2 게이트 후 raw label 3 284개) ← canonical. 2026-09-01 소패킷 부하로 +436행 (metrics_collect_smallpkt_*_20260901.csv), 2026-09-02 라벨 지속성 게이트 적용·18차에 k=3/m=2→k=2/m=2로 교체 (pre-gate: *_nogate_archived_20260902.csv, k3m2 시절: *_k3m2_archived_20260902.csv)
project/scripts/metrics_v2.csv                         레거시 raw (5574행, 프로브·tx_packets 없음 + retry 3× 버그) — 사전학습/ablation용으로만
project/scripts/remeasure_redesign.py                  raw feature에서 sub-score·라벨 전부 재계산 (재설계 공식 적용)
project/scripts/prepare_ap_metrics_dataset.py          windowed train/val/test 변환
project/scripts/train_ap_early_exit.py                 Early Exit LSTM 학습 (--seed, --class-weight-power, --exit-loss-weights)
project/scripts/train_ap_baseline_lstm.py              Baseline(EE 없음) LSTM
project/scripts/train_ap_sdn.py                        SDN 비교모델 (Kaya et al. ICML 2019 — pooling IC + 램프 depth-weighted loss + val 캘리브레이션 T. base 백본·하이퍼파라미터는 EE와 동일=통제변수)
project/models/sdn_lstm.py                             SDNLSTM: SDNInternalClassifier(maxpool⊗avgpool→Linear), sdn_loss_coeffs(램프), calibrate_confidence_threshold(val 스윕)
project/scripts/evaluate_ap_early_exit.py              평가
project/scripts/forecast_eval_redesign.py              조기경보(k폴링 뒤 라벨) 재평가
project/scripts/generate_ap_comparison.py              Baseline/SDN/Proposed 비교표 생성
project/data/ap_metrics_v2_redesign2/                  현행 windowed train/val/test, scaler_params.json, dataset_summary.json
project/checkpoints/ap_v2_redesign2/                   현행 Early Exit LSTM 체크포인트 (배포 기준)
project/deploy/raspberry_pi_ap_v2/                     Pi 배포 번들 (ONNX staged/unified/int8_v2 + bench 스크립트)
project/results/yongsang/ap_v2_redesign2_eval_report.txt          현행 평가 리포트
project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt Pi 실측 지연 비교 (Baseline/SDN/Proposed)
project/results/yongsang/ap_v2_redesign2_forecast_eval.txt        조기경보 프레이밍 결과 (k3m2 시절 마지막 실행 — k2m2로 재실행 안 함, canonical 경로에 없음. `*_k3m2_archived_20260902.txt` 참고)
project/results/yongsang/ap_model_comparison_redesign2.{txt,csv}  아키텍처 비교표
project/models/ap_early_exit_lstm.py    APEarlyExitLSTM (EarlyExitLSTM 상속, input_size = len(AP_FEATURE_COLUMNS) = 7)
project/models/early_exit_lstm.py       EarlyExitLSTM 베이스, multi_exit_loss(weights 파라미터)
project/utils/ap_features.py            AP_FEATURE_COLUMNS = 7개 feature (authoritative)
project/utils/ap_dataloader.py          windowed CSV → DataLoader
docs/yongsang/model_features.{md,html}  7개 feature 레퍼런스 (계산·스무딩·스케일러·라벨 축 vs 모델 입력·변천)
docs/yongsang/congestion_label_redesign.{md,html}  현행 라벨 정의 (max 앵커 + victim 프로브) ← 라벨 관련은 이 문서가 authoritative. html=근거 요약, md=전체 로그
docs/yongsang/ap_crash_analysis.{md,html}      AP(Opal) 반복 크래시 원인 분석
docs/yongsang/onnx_early_exit_redesign.{md,html}  ONNX Early Exit 배포 재설계 (staged → unified If 노드 → INT8 재조립)
docs/yongsang/sdn_comparison.html      SDN 비교모델 한 장 요약 (SDN이 뭔지·통제 변수·3축 차이·5시드/배포/Pi 결과). 정본은 sdn_lstm.py docstring + work-log 8~9차
docs/yongsang/system_overview.html     세 그림 한 장 (데이터 계보 / congestion_score 라벨 파이프라인 / 수집 토폴로지). 온보딩용 개요, 정본은 각 절 하단 링크
docs/yongsang/model_results.html       비교 결과 그래프 (정확도 vs 지연 산점도 / 정확도·F1 막대 / Pi 지연 막대 / exit 분포). 수치 정본은 ap_model_comparison_redesign2.{txt,csv} + ap_v2_redesign2_pi_latency_comparison.txt
.work-log/current.md                    세션별 최신 진행 상황 (이 문서보다 최신)
```

아카이브(덮어쓰지 않고 보존): `project/data/ap_metrics_v2_redesign2_6feat_archived_20260829/`, `project/checkpoints/ap_v2_redesign2_6feat_archived_20260829/`, `project/checkpoints/ap_v2_redesign2/archived_uniform_ee_weights_20260829/`, `project/results/yongsang/*_6feat_archived_20260829.*` 등.

### 입력 feature (7개) — `project/utils/ap_features.py`

```text
throughput_mbps
channel_occupancy_percent
tx_retry_ratio                 # (tx_retries + tx_failed) / (retries + failed + tx_packets), 비율이라 폴링 주기 무관
rssi_dbm                       # RSSI = 수신 신호 세기(dBm, 음수·0에 가까울수록 강함). 이번 폴링 station 평균
rssi_delta_db                  # 직전 폴링 대비 RSSI 변화 (악화 추세 신호)
rssi_moving_avg_dbm            # RSSI 최근 5폴링 이동평균 (노이즈 걷어낸 기저선). ↑ 3개 합쳐 "RSSI 3종", 서로 상관 높음
sta_tx_bitrate_mean            # 이번 폴링에 실제 송신한 station들의 tx bitrate 평균
```

**변천**: 1학기 4개 → 초기 `ap_metrics_v2` 9개 → 라벨 재설계로 6개(2026-08-27) → 7개(2026-08-29).

초기 9개 = `throughput_mbps`, `channel_occupancy_percent`, `latency_ms`, `jitter_ms`, `tx_retries_delta`, `tx_failed_delta`, `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm`.

- **9→6 (라벨 재설계, 2026-08-27)** — 두 변경이 겹쳐 −3:
  - **−2**: `latency_ms`·`jitter_ms`를 모델 입력에서 **제거**. 이들이 라벨을 만드는 축이자(아래 참조) 배포 시점엔 없는 측정(victim 프로브 필요)이라, 모델에 주면 정답 leakage. 모델은 "채널 상태만 보고 victim QoS를 예측"해야 함.
  - **−1**: `tx_retries_delta` + `tx_failed_delta`(→ per_s) 2개를 `tx_retry_ratio` **하나로 통합**.
  - 9 − 2 − 1 = **6**.
- **6→7 (`sta_tx_bitrate_mean` 추가, 2026-08-29)**: occ 60~72% 구간(나머지 6개 feature 평균이 label 2/3 사이에 완전히 동일해지는 구간)에서 label 2 vs 3이 Cohen's d=0.52로 갈라짐. 5개 랜덤 시드 검증에서 exit-loss 가중치와 무관하게 7-feature가 6-feature보다 Label3 F1 +5~11pt. (가설 "혼잡할수록 bitrate 하락"은 실측과 반대 — 부하 테스트라 혼잡 구간에서 오히려 오름. 신호 방향이 반대일 뿐 변별력은 유효.)

모델 입력 제외 컬럼(`dataset_summary.json`의 `model_excluded_columns`): `timestamp`, `scenario`, `poll_interval_s`, `channel_occupancy_method`, `packet_loss_udp_percent`, `connected_clients`, `latency_ms`, `probe_jitter_ms`, `probe_loss_pct`, `probe_ok`, `tx_retx_delta`, `tx_packets_delta`, sub-score 6개, `congestion_score`. 정규화는 train split의 min-max 기준이며 `scaler_params.json`을 val/test·실시간 추론에 동일하게 써야 한다.

### congestion_score = 정답 라벨 판별 기준 (재설계, 2026-08-27~)

상세·근거는 `docs/yongsang/congestion_label_redesign.{md,html}`. 요약:

```text
congestion_score = max(occupancy_score, jitter_score, loss_score, latency_score)

label = 0 if score < 0.25 | 1 if < 0.50 | 2 if < 0.75 | 3 if ≥ 0.75   (경계는 1차·초기와 동일)
```

- 각 축을 4개 앵커(경고 0.25 / 혼잡 0.5 / 심각 0.75 / 완전 1.0)에 piecewise-linear 매핑, [0,1] clamp. **심각(0.75) 앵커는 표준이 직접 지지** (원문 대조 2026-08-31): jitter 50ms = ITU-T Y.1541 Class 0/1 IPDV, latency 150/400ms = ITU-T G.114 편도 티어 경계, occupancy 75% = Aruba WLAN 가이드, loss 5% = Cisco Enterprise QoS. 경고·혼잡 앵커는 Cisco voice 값(jitter 30ms, loss 1%, latency 150ms) 또는 그 아래 보간. **RFC 4594·G.113은 앵커 수치 근거 아님** (정성 등급 / E-model 계수) — 상세는 `congestion_label_redesign.{md,html}` §3.
- `jitter_score`·`loss_score`는 **victim 프로브**(부하와 별개로 흘리는 300kbps 경량 UDP 스트림, `iperf3 -u -b 300k`)가 그 스트림 자체에서 겪은 IPDV/손실을 측정한 값. "이 혼잡 속에서 VoIP 통화 한 통이 얼마나 깨지나". `latency_score`는 ping RTT/2 (편도 추정).
- **조합이 `max`인 이유**: 표준은 "각 축이 언제 나쁜가"는 주지만 "어떻게 합치나"는 안 준다. `max`면 가중치 논쟁이 원천 봉쇄됨 — `label 3 = 최소 한 축이 표준 심각 문턱 돌파`.
- **"실패 = max"**: 채널이 실제로 바쁜 상태(`throughput ≥ 3Mbps or occupancy ≥ 40%`)에서 victim 경로가 완전히 죽으면(ping 무응답 / 프로브 stale) 해당 축을 1.0으로 본다 — 안 그러면 occupancy 단독으로 조용히 되돌아감. 유선 SSH로 AP 텔레메트리를 방금 정상 파싱했으므로 AP는 살아있고 무선 채널만 막힌 것.
- **지속성 게이트 (2026-09-02, `--persistence-gate` 기본 ON, `--gate-k`/`--gate-m`)**: occupancy 심각(≥0.75)이 아닌데 non-occ 축(jitter/loss/latency)으로 label 3이 붙은 행은, 그 축이 **최근 k폴링 중 m폴링 이상 ≥0.75 유지**될 때만 label 3. 아니면 그 축을 0.749로 캡 → 재라벨(대개 혼잡 2). 단발 ping timeout / 1폴링 loss 스파이크 = 지터이지 QoS 붕괴 아님. 17차에 k=3/m=2로 처음 도입(raw label 3 319→285, 5시드 정확도 전 모델 +1.5~2.4pt) → **18차에 6개 k·m 조합 x 5시드 스윕으로 k=2/m=2가 최적임을 확인, 채택**(raw 319→284, L3 recall·F1이 k3m2보다 크게 개선). 상세: `congestion_label_redesign.{md,html}` §4.
- **`throughput_score`·`retry_score`는 라벨 축 아님** (label 2/3 변별력 없음 / 이 2.4GHz AP는 idle에도 retry_ratio med 18%라 QoS 피해의 독립 증거가 아님) — 계산은 하되 `max`에서 제외, 모델 입력 feature로는 유지.

**핵심 논리 — LSTM이 왜 필요한가**: `max(표준 앵커)`는 그 자체로 규칙 기반 분류기다. 그런데 배포 시스템엔 victim 프로브가 없다(협조 싱크 + 지속 스트림 필요). 모델은 **AP-side 텔레메트리(점유율·retry·RSSI·bitrate)만 보고 "지금 victim QoS가 깨지고 있는가"를 예측**해야 한다 — 이건 규칙으로 못 한다. 거기에 시계열 추세 + Early Exit 효율.

### 학습 설정

- 아키텍처: hidden_size 128, num_layers 3, dropout 0.2, num_classes 4. (김호중 baseline과 동일하게 맞춤. hidden/dropout은 2026-08-30 스윕 — 128/0.2 최적.)
- **window size 12** (`utils/ap_features.py` `WINDOW_SIZE` 단일 소스, `prepare_ap_metrics_dataset.py`·`ap_dataloader.py`·ONNX export 6개·Pi 번들이 전부 import). 2026-09-01 스윕에서 10→12 승격 (10/12/13/14/15/20 다중 시드, 12가 최적 — acc·Label3 F1 분산 개선). 15차 참조.
- Optimizer Adam, lr 0.001, batch 32, epochs 50. 체크포인트 선택 기준 = **val balanced accuracy** (raw accuracy 아님 — 다수 클래스 찍는 에폭 선택 방지). 2026-09-02: `train_ap_*.py` 3종이 `val_balanced_accuracy`를 체크포인트에 저장하므로 다시드 배포 선택은 그 키로.
- **multi-exit loss 가중치 = 균등 0.3/0.3/0.4** (`--exit-loss-weights`). SDN 스타일 0.15/0.30/0.55로 바꾸면 exit3 심각 탐지에 유리하다는 가설이 있었으나, 다중 시드(5개) 검증에서 차이가 표준편차보다 작아 **노이즈로 판명 → 균등 유지**. (별도로, SDN 비교모델은 2026-08-30에 논문 충실 재구현되어 백본만 공유하고 IC·loss·threshold가 실제로 다름 — 아래 참조.)
- **class weight power = 0.0** (`--class-weight-power`, `compute_class_weights`) — 클래스 가중치를 아예 안 씀(plain CE). **2026-08-30 재스윕(0.0/0.1/0.15/0.2/0.3/0.5/0.7/0.85/1.0, 3시드씩)에서 power=0.0이 정확도(91.3%±0.5% vs power=1.0의 87.0%±1.1%)·Label3 F1(69.8% vs 63.2%) 둘 다 최고, 트레이드오프 없음**이라 기본값을 1.0→0.0으로 변경. 옛 1.0은 2026-08-23에 4-feature·train label3=23개 시절 "power≤0.85면 label3 recall 절벽" 때문에 정한 값인데, 7-feature·train label3=141개가 되면서 그 절벽이 사라짐(옛 결정을 재검증 안 해 계속 손해보고 있었음). 상세: work-log 4~5차 체크포인트(2026-08-30).
- `train_*.py` 전부에 `--seed` 옵션 있음(기본 `None` = 기존 동작). **하이퍼파라미터 A/B 비교는 반드시 여러 시드로** — 단일 실행 비교가 노이즈였던 전례가 여러 번 있었다.

### 최신 평가 결과 (2026-09-02, 7-feature `ap_metrics_v2_redesign2`, **window 12 + 데이터 2551 + 라벨 지속성 게이트 k=2/m=2**)

정확한 최신 수치는 `.work-log/current.md` 18차/9차 체크포인트. windowed test **365** 샘플 (label 분포 0:109 / 1:82 / 2:132 / 3:42).

- **k·m 게이트 스윕 (18차)**: 17차가 채택한 k=3/m=2는 임시였다 — 6개 config(nogate·k2m2·k3m2·k3m3·k5m2·k5m3) x 5시드 스윕 결과 **k=2/m=2가 최적**(강등 35개로 nogate 대비 거의 최소인데 L3 recall·F1은 6개 중 최고). raw label 3 319→284.
- **5시드 특성화 (k2m2 게이트, test 365, 각 모델 시드 0~4 평균 — 배포보다 이걸 기준 수치로 인용)**:
  - Baseline: acc **92.7%±1.5** / L3 recall 87.1%±1.9 / L3 F1 84.3%±3.5  (k3m2: 92.9±1.2 / 78.1±1.0 / 81.3±3.1)
  - SDN (Kaya et al. 2019, val 캘리브레이션 T): acc **91.9%±1.0** / L3 recall 87.6%±1.8 / L3 F1 **86.4%±1.3**  (k3m2: 93.3±0.9 / 78.6±2.6 / 84.2±1.5)
  - Early Exit Fixed θ: acc **91.7%±0.7** / L3 recall 85.7%±3.7 / L3 F1 85.9%±2.5  (k3m2: 92.1±0.6 / 75.2±2.9 / 82.9±2.0)
  - Early Exit Dynamic θ: acc **92.3%±0.7** / L3 recall 83.8%±3.2 / L3 F1 **86.7%±1.8**  (k3m2: 92.6±0.7 / 78.1±2.8 / 85.2±1.8)
  - → **정확도는 ±1σ 안에서 보합~소폭 하락, L3 recall·F1은 전 모델 명확히 개선**(recall +6~10pt, F1 +1.5~3.0pt, 전부 시드 표준편차보다 큰 폭). k2m2가 학습 가능한 심각 클래스를 k3m2보다 더 일관되게 남긴 것으로 해석. EE Dynamic의 L3 F1(86.7)이 프로젝트 역대 최고.
- **배포 단일 체크포인트 (val balanced acc 최고, fp32 eval — "정직한 pick" 원칙 유지, cherry-pick 안 함)**:
  - **EE Fixed θ seed2** (val bal 90.1%): **92.3%**, L3 P/R/F1 84/88/86. Exit 33/40/27%. **Dynamic θ: 92.9%**, L3 P/R/F1 86/88/87, Exit 34/54/12%. ← 발표 평가 대상.
  - Baseline seed2 (val bal 91.0%): **89.9%** — **약한 test draw**(seed0이 val bal 91.0 근사·test 92.6%로 더 높았지만 val-bal 규칙상 선택 안 함). L3 P/R/F1 69/90/78.
  - SDN seed0 (val bal 90.6%, T=0.72): **92.3%**, L3 P/R/F1 83/90/86. Exit 37/48/15%.
  - **k3m2 시절(SDN 94.8%, Baseline 93.7%, EE 93.2%)보다 전부 낮아 보이지만, k3m2 수치들은 5시드 평균보다 훨씬 위로 튄 운 좋은 draw였다** — 이번 draw는 5시드 평균에 더 가까운 정직한 값. 목표1(95%) 좁은 의미로는 단일 배포 최고치가 낮아졌지만, L3 탐지력은 전 모델 개선.
- **라벨 지속성 게이트**: label 3 & occupancy_score<0.75 & non-occ 축(jitter/loss/latency) 주도 → 그 축이 최근 k폴링 중 m폴링 이상 심각(≥0.75) 유지될 때만 label 3, 아니면 혼잡으로 강등(k=2, m=2 — "최근 2폴링 둘 다 심각"). 근거: label 3 = victim QoS 붕괴, 패킷 1폴링 드랍은 붕괴 아님. 상세: `docs/yongsang/congestion_label_redesign.{md,html}` §4, k·m 스윕 근거는 §4 후속.
- **power=1.0 → 0.0 효과** (2026-08-30): 5시드 평균 정확도 +2~4pt. label2 정상화.
- **서사**: 세 모델 정확도 여전히 동급 → Proposed(Early Exit)의 가치 주장은 정확도가 아니라 **속도·효율**(목표2 <1ms, Baseline 대비 −28%) + "간섭 감지에 EE 최초 적용" + L3(심각) 탐지력이 게이트 튜닝으로 전 모델 개선됐다는 점.
- **ONNX 재수출 완료 (2026-09-02, k2m2)**: EE unified fp32 = PyTorch 365/365, INT8 v2 fixed 364/365(92.1%)·dynamic 365/365(92.9%). Baseline INT8 365/365(89.9%). SDN INT8 364/365(92.1%, T=0.72). Pi 번들 sync + scp 완료.
- **Pi 지연 재측정 (2026-09-02, window 12, k2m2, test 365)**: Baseline 0.864 / SDN 0.575 / EE Fixed **0.625** / Dynamic 0.632ms — 전부 avg <1ms(목표2). EE Fixed가 Baseline −28%. 상세: `ap_v2_redesign2_pi_latency_comparison.txt` 9차.
- **속도 원리**: SDN pooling IC(ReduceMax+Mean per exit)가 Proposed의 last-timestep linear head보다 무거움 — **per-exit 지연은 EE가 SDN보다 전 stage 가벼움**(0.327/0.644/0.953 vs 0.332/0.646/0.962ms). SDN 평균(0.575)이 EE(0.625)보다 낮은 건 T=0.72가 exit1 비중을 37.5%로 front-load한 threshold 정책 artifact이지 구조 우위 아님.
- **SDN 비교모델 = "기존 조기종료 방법 vs 우리 방법" 통제 비교**: base 3층 LSTM·하이퍼파라미터는 Proposed와 완전 동일, SDN이 규정하는 3축만 논문(Kaya et al. ICML 2019)대로 다름 — (1) pooling IC, (2) 커리큘럼 램프 depth-weighted loss, (3) val 캘리브레이션 confidence T. 결과: 정확도 동급(5시드 EE Fixed 91.7 / Dynamic 92.3 / SDN 91.9), Label3 F1도 동급(SDN 86.4 / EE Dynamic 86.7 / EE Fixed 85.9). 주장은 "SDN을 이겼다"가 아니라 "동급 정확도 + 더 가벼운 head + 트래픽 적응형 임계값".
- **남은 오답의 정체**: k2m2 게이트 후 재확인 필요(k3m2 시절엔 지속형 label 2↔3(occ 57~69) + occ 72~73 경계로 분석됨) — occ 60~73대 관측 한계는 여전할 것으로 추정.
- **조기경보(forecasting) 프레이밍**: k3m2 데이터 기준(`ap_v2_redesign2_forecast_eval_k3m2_archived_20260902.txt`)이며 k2m2로 재실행 안 함(우선순위 낮음) — k≥3폴링에서 LSTM이 반응형 baseline보다 severe F1 우위였으나 escalation recall이 게이트로 떨어지는 패턴을 보였음. k2m2 재확인 필요.

### 알려진 한계

- **Label 3(심각)**: 16차 데이터 수집(raw 202→318) + 17차 게이트(k3m2, 319→285) + 18차 게이트 재튜닝(k2m2, 319→284). **5시드 recall 55.5%(초기)→85.7%(EE Fixed, k2m2) / 83.8%(Dynamic), F1 69.9→85.9/86.7%** — 얇음/흔들림 문제 크게 완화. jitter 축 심각은 이 하드웨어(2폰+iperf3)로 불가.
- **k2m2 게이트 후 남은 벽**: 재확인 필요(위 "남은 오답의 정체" 참고). k3m2 시절엔 지속형 2↔3(occ 57~69) + occ 72~73 경계로 분석됐고, k2m2도 비슷한 구간일 것으로 추정되나 오답 분해를 다시 안 함. **단일 배포 정확도는 k2m2로 오히려 낮아졌다(운 나쁜 draw)** — 95% 완주는 더 나은 AP·telemetry(per-station airtime·MCS) 또는 forecast 재프레이밍, 혹은 시드를 더 돌려 좋은 draw 재추첨.
- **AP(Opal) 반복 크래시**: "몇 대 붙었나"보다 "각 폰이 대칭적으로 붙었나"가 핵심 변수 — 신호 강한 S26이 채널 독점하고 약한 191이 굶는 Wi-Fi capture effect. 부하 스위트스팟 191=60M/S26=60M, 안전 구간 대략 300~420초 (60/60도 500초 넘기면 완전 크래시, 80/80은 10분 시도에서 물리 재부팅). 상세: `docs/yongsang/ap_crash_analysis.{md,html}`.
- **유선 관리채널**: `collect_metrics.py`가 지속 SSH(`APPoller`)로 전환됐고 victim 프로브(`ProbeRunner`)가 추가됨. collector를 라즈베리 파이로 옮겨 관리 트래픽을 무선 채널에서 분리하는 구조 확정.
- **ONNX/Pi 배포**: staged(세션 3개) → `torch.jit.script`+ONNX `If` 노드 단일 그래프(`export_onnx_ap_unified.py`) → INT8은 staged(flat)로 먼저 양자화 후 손수 재조립(`export_onnx_ap_unified_int8_v2.py`, `If` 서브그래프 안 LSTM을 양자화 도구가 건너뛰는 한계 우회). baseline 대비 대략 -60% 내외 (1학기 4-feature 자료로 두 현상 교차검증). Pi latency 주장 시 `docs/yongsang/onnx_early_exit_redesign.{md,html}`의 결론을 따르고 staged/fp32-only 수치를 최종 결과로 인용하지 않는다.
- hidden_size·dropout 스윕 완료(2026-08-30, 128/0.2 최적). class-weight-power 재스윕 완료(2026-08-30 → 0.0). **window·lr·batch·모델입력 EMA 스윕 완료(2026-09-01 — window 10→12만 승격)**. **라벨 지속성 게이트 k·m 스윕 완료(2026-09-02, 18차) → k=2/m=2 채택** (17차 임시값 k=3/m=2에서 교체). epochs 스윕은 미실시.

### 재현 명령어

라벨 재계산(공식/앵커가 바뀌었을 때, raw feature에서 전부 — 지속성 게이트 기본 ON):

```powershell
python project\scripts\remeasure_redesign.py -i project\scripts\metrics_v2_pi_redesign2_relabeled_nogate_archived_20260902.csv -o project\scripts\metrics_v2_pi_redesign2_relabeled.csv
# 게이트 끄려면 --no-persistence-gate, k/m 조절은 --gate-k --gate-m
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

1. raw CSV를 그대로 넣지 않았는지 확인 — 모델은 window size **12**로 변환된 `test.csv`를 쓴다. ONNX(`[1,12,7]`)·Pi 번들·Pi 지연 재측정 전부 완료 (2026-09-02, 2551+게이트).
2. feature 개수/순서가 `ap_features.py`의 **7개** 정의와 동일한지 확인. (ONNX 재조립 스크립트가 `[1,10,6]`으로 하드코딩돼 있던 버그 전례 — Pi에서 "Got 7 Expected 6" 에러.)
3. `label`, `congestion_score`, sub-score, `probe_*`, `latency_ms`, `jitter_ms`가 입력 feature에 섞이지 않았는지 확인.
4. `scaler_params.json` 기준이 다른지 확인 (재라벨링만 하고 재변환을 안 하면 라벨과 스케일러가 어긋난다).
5. 데이터 디렉터리가 `ap_metrics_v2_redesign2`(7-feat)인지, 아카이브된 `_6feat_archived_*`를 잘못 가리키는 건 아닌지 확인.

## Claude가 추가로 참고해야 할 파일

0. `docs/README.{md,html}` — **문서 안내 (질문 → 문서 매핑).** "왜 이런 라벨을 정했나" 같은 질문에 어느 문서를 볼지. 팀원 온보딩용.
1. `.work-log/current.md` — 세션별 최신 진행 상황. **이 문서보다 항상 최신.** 최신 수치·다음 할 일은 여기.
2. `docs/yongsang/congestion_label_redesign.{md,html}` — 현행 라벨 정의(max 앵커 + victim 프로브)와 그 근거(ITU-T Y.1541/G.114 · Cisco Enterprise QoS · Aruba WLAN 가이드, §3에 표준 원문 대조 결과). 라벨 관련 질문은 여기가 authoritative. html=정의·근거 요약, md=세션 로그까지 전체 (`congestion_label_criteria.{md,html}`는 구 정의, archived).
3. `project/utils/ap_features.py` — 현행 7개 feature 정의 + 변천 주석 (정본). 각 feature 상세(계산·스무딩·스케일러·왜 라벨 축 아닌지)는 `docs/yongsang/model_features.{md,html}`.
4. `docs/yongsang/ap_crash_analysis.{md,html}` — AP 반복 크래시 원인 분석.
5. `docs/yongsang/onnx_early_exit_redesign.{md,html}` — ONNX Early Exit 배포 재설계(staged → unified If 노드 → INT8 재조립). Pi latency 주장은 이 문서 결론을 따른다.
6. `docs/capstone1_summary.html` — 1학기(4-feature 시뮬레이션) 활동·지표 정리. `docs/yongsang/capstone2_vacation_summary.html` — 2학기 방학(2026-08-21~30) 개발 흐름 정리(데이터 실측·라벨 재설계 2회·power=0.0 재검증·SDN 논문 재구현·라이브 데모·되돌린 결정 4건).
7. `project/models/ap_early_exit_lstm.py`, `project/utils/ap_dataloader.py` — 실제 코드 흐름.
8. `project/README_AP_V2.md`(redirect 스텁으로 축약됨), `docs/yongsang/congestion_label_criteria.{md,html}` — **stale (가중합 4 sub-score 시절).** 역사적 맥락용으로만.

## Claude에게 중요한 해석 기준

1. 이 브랜치에는 1학기 4-feature 코드와 1차(`ap_cleaned_strict`) 코드가 없다. 그 자료는 `yongsang` 브랜치.
2. "AP 실측 데이터"·"2차"는 지금 항상 **`ap_metrics_v2_redesign2` + 7-feature + max-앵커 라벨**을 가리킨다. 초기 `ap_metrics_v2`(9-feature, weighted-sum)는 archived.
3. **라벨 입력 ≠ 모델 입력.** 라벨은 victim 프로브(jitter/loss)·ping(latency)에 의존하고, 모델은 그걸 못 본다. 모델의 일은 채널 상태만으로 victim QoS를 예측하는 것.
4. 하이퍼파라미터/가중치 A/B 비교 수치는 **다중 시드로 검증됐는지** 먼저 확인한다. 단일 실행 비교가 노이즈였던 전례가 여러 번 있었다(SDN 가중치 승격 → 철회 등).
5. PC wall-time만으로 Early Exit 속도 우위를 주장하면 위험하다 — 최종 속도 주장은 Raspberry Pi INT8 실측 기준(현재 0.6~0.8ms, 목표2 <1ms 달성).
6. Fixed/Dynamic은 별도 backbone을 새로 학습하는 것이 아니라, 같은 Early Exit backbone에서 threshold 정책(고정 θ vs 변동률 기반 θ)만 바꿔 평가하는 구조이다. **SDN은 다르다** — 2026-08-30 논문 충실 재구현으로 pooling IC head + 램프 loss + 캘리브레이션 T를 실제로 별도 학습(base 백본만 공유). "SDN-style이 EE와 사실상 같은 네트워크"라는 옛 서술은 이 재구현 이전 기준이며 stale.
7. 남은 핵심 숙제는 **Pi 정확도 95%**(18차 k2m2 게이트 후 5시드 평균 Baseline 92.7% / SDN 91.9% / EE Fixed 91.7% / EE Dynamic 92.3% — 사실상 동급, 단일 배포 최고 SDN 92.3%). k2m2로 L3(심각) recall·F1은 전 모델 크게 개선(예: EE Fixed recall 75.2→85.7%)됐지만 단일 배포 정확도는 17차의 운 좋은 draw(SDN 94.8%)보다 낮아짐 — 5시드 평균이 진짜 대표값이라는 원칙을 잊지 말 것. Proposed의 가치 주장은 정확도가 아니라 속도·효율(Pi에서 Baseline 대비 -28%).
