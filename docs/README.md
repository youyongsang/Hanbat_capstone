# 문서 안내 (`capstoneDesign2` 브랜치)

> 문서가 많다. **뭘 알고 싶은지**에 따라 어느 문서를 보면 되는지만 정리한다.
> 1학기 코드·문서는 `yongsang` 브랜치. 이 브랜치는 2차 실측 라인(`ap_metrics_v2_redesign2`, 7-feature).
>
> 브라우저로 보기 좋은 버전: [`docs/README.html`](README.html) (같은 내용).

---

## 처음 오면 이 5개

| # | 문서 | 무엇 |
|---|---|---|
| 1 | **`CLAUDE.md`** (레포 루트) | 프로젝트 전체 맥락, 데이터 계보(1학기→1차→2차), 현재 수치, 자주 헷갈리는 점. **항상 여기부터.** |
| 2 | **`.work-log/current.md`** | 세션별 최신 진행·다음 할 일. **CLAUDE.md보다 최신** (수치가 어긋나면 이게 맞음). |
| 3 | **`docs/yongsang/system_overview.html`** | 세 그림 한 장 — 데이터 계보 / 라벨 파이프라인 / 수집 토폴로지. 말로 된 걸 그림으로 보고 싶을 때. 브라우저로. |
| 4 | **`docs/yongsang/capstone2_vacation_summary.html`** | 방학(8/21~30) 개발 흐름 한 장 요약 — 6단계 타임라인 + 되돌린 결정 4건. 브라우저로. |
| 5 | **`docs/yongsang/congestion_label_redesign.{md,html}`** | 라벨 정의의 **정본** (표준 앵커 + victim 프로브). 아래 "왜" 표의 절반이 이 문서. HTML = 정의·근거 요약, MD = 전체 로그. |

---

## "왜 이렇게 정했나" — 질문 → 문서

| 알고 싶은 것 | 어디를 보면 되나 |
|---|---|
| **왜 이런 라벨(정상/경고/혼잡/심각)을 정했나** | `congestion_label_redesign.{md,html}` — §1 "왜 재설계하나"(순환논리), §3 "표준 문턱 앵커"(ITU-T Y.1541·G.114 · Cisco Enterprise QoS · Aruba WLAN 가이드, 원문 대조 결과 포함), §4 "조합 = max" |
| **congestion_score는 어떻게 계산하나** | `congestion_label_redesign.{md,html}` §3~4 + `collect_metrics.py`의 `calculate_scores()`/`ANCHORS` (코드 정본) |
| **7개 feature 각각이 뭔가 / 계산·스무딩·스케일러** | **`docs/yongsang/model_features.{md,html}`** — feature 레퍼런스 (정본은 `project/utils/ap_features.py`) |
| **왜 latency·jitter를 모델 입력에서 뺐나** | `model_features.{md,html}` §3 + `congestion_label_redesign.{md,html}` §5 — 정답 leakage + 배포 시점엔 없는 측정 |
| **왜 feature가 7개인가 (9→6→7), `sta_tx_bitrate_mean`은 왜** | `model_features.md` §2(7번)·§5 + `.work-log/current.md` "2026-08-29 2차" |
| **왜 `class-weight-power=0.0`인가** | 아티팩트 "Class-Weight-Power Zero" (`.work-log/current.md` 4~5차 체크포인트에 링크·표) |
| **왜 window size 12인가** (10→12) | `.work-log/current.md` 15차 — window/lr/batch/EMA 스윕에서 12만 이득 (EE 정확도 +1.2pt, Label3 F1 분산 절반). `docs/yongsang/capstone2_vacation_summary.html` §07 |
| **AP가 왜 자꾸 크래시하나 / 부하는 얼마까지** | `docs/yongsang/ap_crash_analysis.{md,html}` — 신호 비대칭(capture effect)이 핵심 |
| **ONNX를 왜 이렇게 배포했나** (staged→unified If→INT8 재조립) | `docs/yongsang/onnx_early_exit_redesign.{md,html}` |
| **SDN 비교 모델은 뭔가 / 우리 모델과 뭐가 다른가** | **`docs/yongsang/sdn_comparison.html`** — 한 장 요약(SDN이 뭔지·통제 변수·3축 차이·결과). 정본은 `project/models/sdn_lstm.py` docstring (Kaya et al. 2019 재구현) + `.work-log/current.md` 8~9차 + `project/results/yongsang/ap_model_comparison_redesign2.{txt,csv}` |
| **정확도·지연 결과 그래프로** | **`docs/yongsang/model_results.html`** (인터랙티브) — 정확도 vs 지연 산점도 / 정확도·F1 막대 / Pi 지연 막대 / exit 분포. **발표 슬라이드용 이미지는 `docs/yongsang/figures/`** (SVG·PNG). 수치 정본은 `ap_model_comparison_redesign2.{txt,csv}` + `ap_v2_redesign2_pi_latency_comparison.txt` |
| **Fixed θ vs Dynamic θ 차이** | `CLAUDE.md` "해석 기준" 6번 |
| **데모를 어떻게 돌리나** | `project/demo/README.md` |
| **데모를 (팀이) 어떻게 만드나** | `project/demo/API.md` — API·SSE 스키마·모델 계약·확장 항목 |
| **발표 정량 목표·지표** | `CLAUDE.md` 상단 "정량적 목표" + `docs/캡스톤디자인I_최종발표.pptx` |
| **처음부터 재현하는 명령어** | `docs/terminal_command_guide.md` + `CLAUDE.md` "재현 명령어" |
| **환경 셋업** (conda `capstone`, torch DLL 등) | `docs/yongsang/environment.md` + `CLAUDE.md` "자주 헷갈리는 점" |

---

## 결과 파일 (`project/results/yongsang/`)

| 파일 | 무엇 |
|---|---|
| `ap_v2_redesign2_eval_report.txt` | 최신 Early Exit 평가 (fixed/dynamic θ) |
| `ap_baseline_lstm_redesign2_eval_report.txt` · `ap_sdn_redesign2_eval_report.txt` | Baseline · SDN 평가 |
| `ap_model_comparison_redesign2.{txt,csv}` | Baseline / SDN / Proposed 비교표. **그래프 버전 = `docs/yongsang/model_results.html`** |
| `ap_v2_redesign2_pi_latency_comparison.txt` | Raspberry Pi INT8 지연 실측 (3~5차 + 이전 차수 보존). 그래프 버전 위 동일 |
| `ap_v2_redesign2_threshold_comparison.txt` | 학습 모델 vs occupancy 단일 문턱 (심각 탐지) |
| `ap_v2_redesign2_forecast_eval.txt` | 조기경보(k폴링 뒤 라벨) 프레이밍 |
| `ap_v2_redesign2_pi_bench_power0_20260830.txt` | Pi INT8 벤치 raw 로그 |
| `ap_v2_redesign2_live_*_20260830.txt` · `..._demo_full_run_...` | 라이브 혼잡 감지 부하 테스트 실측 |
| `*_archived_*` / `*_6feat_*` / `*_power1_*` | 옛 버전. **안 봐도 됨** (덮어쓰지 않고 보존한 것) |

---

## 참고만 — stale / 역사용 (안 읽어도 됨)

| 문서 | 상태 |
|---|---|
| `congestion_label_criteria.{md,html}` | 옛 라벨 정의(가중합 4 sub-score 시절). `congestion_label_redesign.{md,html}`가 대체. 문서 전체 stale — 상단 배너 참고. |
| `README_AP_V2.md` | 9-feature·가중합 시절. stale (내용은 redirect 스텁으로 축약됨). |
| `dummy_data_spec.md` | 1학기, 시뮬레이터 데이터 전 임시 스펙. |
| `docs/yongsang/result_text_analysis.md` | `yongsang` 브랜치 결과 분석. 이 브랜치와 무관. |
| `docs/hochung/*` · `docs/yena/*` · `docs/yongsang/guideline_*` · `stage*_work_log.md` | 1학기·방학 팀원별 계획서·단계별 work log. 맥락용으로 보존. |
| `docs/vacation_activity_overview.md` | 팀 전체 방학 활동 개요 (yongsang 시점은 위 #3 요약이 더 최신). |
| `.work-log/progress.md` | 초기 일자별 메모. `current.md`가 현행. |

---

## HTML로 읽기 좋은 것

브라우저로 열면 표·차트가 렌더된다:
`system_overview.html` · `capstone2_vacation_summary.html` · `congestion_label_redesign.html` · `model_features.html` · `sdn_comparison.html` · `model_results.html` · `onnx_early_exit_redesign.html` · `ap_crash_analysis.html` · `congestion_label_criteria.html`(archived) · `docs/capstone1_summary.html`(1학기)

발행 아티팩트 (링크는 `.work-log/current.md`): "Class-Weight-Power Zero", "AP 혼잡 분류 모델 비교", "방학 개발 흐름".
