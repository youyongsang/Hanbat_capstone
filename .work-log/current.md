# Capstone-Design 현재 상태
최종 업데이트: 2026-09-01 (Claude Code) — **window size 스윕 → window 10→12 승격 (15차)**: 목표1(정확도 95%) 공략. window/lr/batch/EMA-입력스무딩 다중 시드 스윕에서 **window 12가 최적**. 발표 시스템 = Early Exit이므로 **EE 기준: 배포 Fixed 90.6→91.9% / Dynamic 91.0→92.2%** (5시드 평균 91.9±0.5 / 92.0±0.2). 상한선 참고: Baseline 91.6→93.5%, SDN 90.3→91.6%. 5시드 평균은 Baseline 92.0±1.3 / SDN 91.2±0.7 / EE 91.9±0.5 — **여전히 동급이나 Label3 F1 전 모델 +3~10pt, 분산 절반 이하**(SDN std 8.1→2.6). canonical 데이터셋·`prepare`·`ap_dataloader` 기본값 10→12, w10 아카이브(`*_w10_archived_20260901`). **남은 것: ONNX 재수출([1,10,7]→[1,12,7])·Pi 지연 재측정(Pi 오프라인, 현 지연 수치 STALE)·비교 그래프/문서(sdn_comparison·model_results·figures) 갱신·런타임 deque(live/demo/collect) 10→12.** 아래 "15차" 참조. 그 전 14차 — **문서 대청소·표준 인용 검증·새 문서 2종**: 코드·수치 변경 0, 전부 문서. ① 신규 `docs/yongsang/model_features.{md,html}` (7-feature 레퍼런스 — 계산·스무딩·스케일러·라벨축 vs 모델입력·변천) + 신규 `docs/yongsang/congestion_label_redesign.html` (브라우저용 라벨 정의). ② **표준 앵커 인용 원문 대조** — jitter 50ms=ITU-T Y.1541 IPDV·latency 150/400=G.114 확인, **RFC 4594·G.113 인용은 부적합으로 제거**(정성 등급/E-model 계수). ③ 9→6 feature 산수 정정(−2 −1), "RSSI 3종" 표기 통일 + RSSI 설명 추가, `congestion_label_criteria` 전체 archived 재프레이밍. ④ 문서 흐름 스캔(README.html부터) — 죽은 참조 정리(`README_AP_V2.md "핵심 검증 질문"` 6곳·`API.md §4 표` SDN), 모든 `.md` 참조 → `.{md,html}`. ⑤ `README.html` 파일명 → 클릭 링크(`.html` 상대경로, 나머지 GitHub blob). 커밋 `e17a50e`~`69c9f78`(19개) 푸시. **Pi 데모 실기기 검증은 여전히 대기(Pi·AP 오프라인).** 그 전 13차: 데모 서버 Pi 이식(dual-import + 전면 인자화). 그 전 8/30: class-weight-power=0.0 승격 + SDN 논문 재구현 + 라이브 추론 + 데모 웹 대시보드. **5시드 평균 정확도: Baseline 92.0%±0.7 / SDN(논문) 90.4%±1.4 / EE 90.7%±0.7 — 정확도 동급.** 갈리는 축: label3 안정성, 속도(EE 0.540 vs SDN 0.572ms Pi INT8). 아래 "10차"부터 확인.

## ⭐ 다음 세션 시작 지점 — Pi 온라인 되면 데모 서버 실기기 검증 (내일)

**전제**: 코드·번들·문서 다 됨(13차). Pi(`capstone@192.168.8.109`)·AP(`root@192.168.8.1`)가 이 세션 내내 오프라인이라 실기기 테스트만 남음. Opal 망에 붙은 뒤 진행.

0. **연결 확인**: `ssh root@192.168.8.1 uptime` (AP 살아있나 + 안 크래시 상태인가), `ssh capstone@192.168.8.109 "hostname -I; which iperf3; python3 -c 'import numpy,onnxruntime'"` (Pi deps). iperf3 없으면 `sudo apt install -y iperf3`.
1. **번들 배포**: `scp -r project/deploy/raspberry_pi_ap_v2 capstone@192.168.8.109:~/demo` (기존 `~/ap_pi_v2/`와 별개 폴더로. onnx·scaler·collect_metrics 이미 번들에 포함).
2. **Pi→폰 SSH 셋업** (지금은 노트북 키만 폰에 등록됨): Pi에서 `cat ~/.ssh/id_ed25519.pub` 없으면 `ssh-keygen -t ed25519 -N ""` → 그 pub키를 두 폰 Termux `~/.ssh/authorized_keys`에 추가 → `ssh <user@폰IP> echo ok` 양쪽 확인. 폰 IP·user(`u0_aXXX`)는 AP `cat /tmp/dhcp.leases` 또는 폰 Termux `whoami`/`ip addr`로 확인.
3. **실행**: Pi에서
   ```bash
   cd ~/demo
   python3 demo_server.py --iperf-target 192.168.8.109 --s21 <user@폰1IP> --s26 <user@폰2IP>
   ```
   (`--iperf-target`이 Pi 자신 IP라 `iperf3 -s` 자동 기동. 부하 경로 = 폰→AP→Pi.)
4. **브라우저**: 노트북에서 `http://192.168.8.109:8000/` 열기.
5. **신호 대칭 확인 먼저** (`/signal` 또는 UI 하단) — 두 폰 12dBm 이내. 비대칭이면 폰 위치 조정 (S21 약신호 → capture effect → 저부하 크래시).
6. **부하 시나리오**: 버튼 10M→20M→30M→정지. 확인할 것:
   - occ 60~76% 구간에서 **심각(3)** 뜨는가 (occupancy 단독 문턱 75%로 못 잡는 구간).
   - 정지 후 정상(0) 복귀.
   - AP 크래시 없이 완주 (30M×2=60M, 대칭 강신호면 안전. up 시간·SSID 유지 확인).
   - 원시 패널 vs 안정화(CONFIRM=5) 패널 둘 다 정상 갱신.
7. **결과 저장**: 콘솔 로그 → `project/results/yongsang/ap_v2_redesign2_demo_pi_run_YYYYMMDD.txt`. work-log 13차에 "Pi 실기기 검증 완료" 추가.
8. **(선택) Pi latency 재확인**: 데모 추론이 Pi에서 도는 김에 `time` 몇 폴링 재보면 목표2(<1ms) 라이브 재확인 가능. 필수 아님 — test 310창 벤치가 이미 있음.

**주의**: AP 크래시 시 물리 재부팅. 30M×2가 500초 넘거나 신호 비대칭이면 위험 (`docs/yongsang/ap_crash_analysis.{md,html}`). 이상하면 즉시 정지 버튼.

### 그 밖에 다음 세션에 마무리할 것 (문서 관련, 급하지 않음)
- **`ap_cleaned_strict` "인터넷 공개 데이터" 문구** — 14차에서 재검토 결과 근거 약함(증거: `metrics_cleaned.csv`가 호중 카톡 수신본, 생성 코드 미커밋, latency 0.05ms·RSSI −20dBm대로 물리적 비현실적). 다만 팀이 8/23~24에 "실측 아님 → archived"로 판단한 건 유효. **호중에게 `metrics_cleaned.csv` 원본 출처 직접 확인** 후 `CLAUDE.md`·`capstone2_vacation_summary.html`의 "인터넷 공개 데이터 가공" 문구를 "출처 불명·값 비현실적 → 실측 불인정"으로 톤 조정. (1학기 `data/real`은 확실히 Kaggle 6G Network Slicing — 별개 라인.)
- **`README.html` 링크 실제 브라우저 클릭 확인** — 14차는 Claude-in-Chrome 확장 미연결이라 Node+DOM셰임 실행으로만 검증(콘솔 에러 없음, 링크 52개 정상 생성). 실제 `file://`로 열어 `.html` 상대링크·GitHub blob 링크 클릭 동작 확인 필요.

## 15차 (2026-09-01) — window size 스윕 → **window 10 → 12 승격** (전 모델 재학습·배포)

**동기**: 목표1(Pi 정확도 95%) 공략. CLAUDE.md가 "한 번도 스윕 안 함"으로 남긴 window size, "미시도"로 남긴 모델 입력 스무딩을 **다중 시드로** 검증(단일 실행 A/B가 노이즈였던 전례 준수).

### 스윕 (in-process harness, 3시드 → 유망 config 5시드)
| 축 | 값 | 결과 (EE Fixed θ, 5시드 test 평균) |
|---|---|---|
| **window** | 10(기준)/12/13/14/15/20 | **12 최적.** w10 acc 91.0±0.9·L3F1 66.2±5.0 → **w12 91.9±0.5·69.9±2.9**. w15 L3F1 71.4로 최고지만 acc 90.7(이득 없음)·L2 하락. w20 88.4로 악화. |
| lr | 0.0005/0.001(기준)/0.002 | 0.0005 악화(88.9), 0.002 acc 91.3이나 L3 F1 62.5±9.7 불안정 |
| batch | 16/32(기준)/64 | 16 소폭↑(91.3±0.3), 64 악화(89.5) |
| **모델입력 EMA** | α 0.3/0.5/0.7 | **순효과 없음** (88~91%). 라벨링 스무딩(10차 이전 폐기)에 이어 **스무딩 방향 2연속 폐기** |

- **window sweet spot이 12.** 10→12로 시계열 맥락(≈2폴링=2~4초)이 늘어 occ 60~72% 정보부족 구간·심각 판정이 안정화. 20은 과길어 오히려 악화.
- 핵심 이득은 **희소클래스(Label 3) 안정성** — 전 모델 F1 분산이 절반 이하로. lr/batch/EMA는 이득 미미~노이즈.

### 배포 (w12, 전 모델 5시드 재학습 → val balanced acc 최고 선택)
- `prepare_ap_metrics_dataset.py` `WINDOW_SIZE` 10→12, `ap_dataloader.py` 기본값(3곳) 10→12.
- 아카이브: `project/data/ap_metrics_v2_redesign2_w10_archived_20260901/`, `project/checkpoints/ap_v2_redesign2_w10_archived_20260901/`, `project/results/yongsang/*_w10_archived_20260901.*`.
- w12 windowed: train 1429 / val 305 / **test 309** (label 0:94 / 1:67 / 2:117 / 3:31).

| 모델 | 배포 seed | 배포 test acc (w10→**w12**) | 5시드 평균 acc | 5시드 L3 F1 (w10→w12) |
|---|---|---|---|---|
| Baseline | seed 4 | 91.6 → **93.5%** | 92.0 → 92.0 ±1.3 | 66.1±2.3 → **68.7 ±3.1** |
| SDN (논문, T=0.71) | seed 2 | 90.3 → **91.6%** | 90.4 → **91.2 ±0.7** | 60.4±8.1 → **70.3 ±2.6** |
| EE Fixed θ | seed 1 | 90.6 → **91.9%** | 90.7 → **91.9 ±0.5** | 64.5±5.5 → **69.9 ±2.9** |
| EE Dynamic θ | seed 1 | 91.0 → **92.2%** | — → 92.0 ±0.2 | — → 69.6 ±3.6 |

- **세 모델 5시드 평균 92.0 / 91.2 / 91.9 — 여전히 동급.** window 12의 가치는 **Label 3 F1 분산 붕괴** (SDN std 8.1→2.6이 극적 — per-model 캘리브레이션 T가 더 긴 window에서 안정).
- **목표1(95%) = EE 기준**: EE 배포 Fixed 91.9 / Dynamic 92.2% (5시드 91.9±0.5 / 92.0±0.2). w10 90.7 대비 **+1.2~1.3pt**, **여전히 미달 (~3pt)**. Baseline 93.5%는 상한선 참고이지 발표 평가 대상 아님.
- EE 배포 seed 1은 val balanced 최고인 정직한 pick(seed 0이 test 92.6로 더 높지만 선택 안 함).
- 배포 EE 단일 체크포인트 L3는 51.6%로 낮음 — seed 1의 test draw. 5시드 평균 L3 recall은 55.5%(w10 52.9 대비 개선).

### 남은 것 (window 변경 ripple — 대부분 하드웨어 대기 / 후속 세션)
- **ONNX 재수출** — `export_onnx_ap_{unified_int8_v2,sdn_unified_int8,baseline,unified}.py` 등이 입력 shape `[1,10,7]` 하드코딩 → `[1,12,7]`. staged/unified/INT8 재양자화. 배포 checkpoint dir의 옛 w10 ONNX는 삭제됨(archive에 보존).
- **Pi latency 재측정** — Pi(`capstone@192.168.8.109`) 오프라인. `ap_v2_redesign2_pi_latency_comparison.txt`·`ap_model_comparison_redesign2` 지연 수치 전부 **w10 STALE**. window +2 = LSTM step 2회 추가 → 소폭 증가 예상, <1ms 유지 전망.
- **비교 문서/그래프 갱신** — `ap_model_comparison_redesign2.{txt,csv}` 재생성함(acc/exit는 w12, Pi/5시드 문자열도 갱신). `docs/yongsang/sdn_comparison.html` §02·§04, `docs/yongsang/model_results.html` 차트·표, `docs/yongsang/figures/*` (build_figures.js `MODELS` 배열) 미갱신.
- **런타임 window deque** — `live_congestion.py`·`demo_server.py`·`collect_metrics.py`의 window 길이 10→12 (repo + `deploy/raspberry_pi_ap_v2/` 번들 양쪽). ONNX 재수출과 함께.
- **문서 잔여** — `CLAUDE.md`·`model_features.{md,html}`의 "window 10"·"[1,10,7]"·"test 310창" 일부.

### 재현
```powershell
# 스윕 harness / 배포 스크립트는 세션 스크래치(.../scratchpad/{exp.py,sweep.sh,deploy.sh}). 배포 요지:
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2_pi_redesign2_relabeled.csv --out-dir project\data\ap_metrics_v2_redesign2 --window-size 12 --overwrite
python project\scripts\train_ap_early_exit.py   --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint-dir project\checkpoints\ap_v2_redesign2 --epochs 50 --batch-size 32 --seed 1
python project\scripts\train_ap_baseline_lstm.py --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint-dir project\checkpoints\ap_v2_redesign2 --epochs 50 --batch-size 32 --seed 4
python project\scripts\train_ap_sdn.py          --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint-dir project\checkpoints\ap_v2_redesign2 --epochs 50 --batch-size 32 --seed 2
```

## 14차 (2026-08-31) — 문서 대청소 · 표준 인용 검증 · 새 문서 2종

**동기**: 사용자가 문서를 훑으며 발견한 오류·모호함을 연쇄적으로 정리. 코드·데이터·수치 변경 없음, 전부 문서.

### 새 문서 2종
- **`docs/yongsang/model_features.{md,html}`** — 현행 7-feature 레퍼런스 (`congestion_label_criteria`가 4-가중합 시절만 다뤄서 7-feature용 문서가 없었음). §1 feature 표(원천·스무딩·train-split scaler min/max), §2 feature별 상세 + RSSI 설명, §3 **라벨 축 vs 모델 입력**(§3.1 y vs X 개념 — "라벨 축=학습용/모델입력=추론용"이 아니라 7개는 학습·추론 공통, 라벨 축은 양쪽 다 모델에 안 들어감 / §3.2 모델 입력 7개 전부 나열 + 제외 컬럼), §4 scaler, §5 변천(9→6→7). 정본은 `ap_features.py`.
- **`docs/yongsang/congestion_label_redesign.html`** — 그 authoritative `.md`의 브라우저 판(옛 라벨 HTML은 archived `congestion_label_criteria.html`뿐이었음). §1~5 정의·근거 + §3 표준 앵커 표(출처 포함) + §4 max + §6 캘리브레이션(collapsed `<details>`). `.md`는 세션 로그·열린 항목까지 전체본.

### 표준 앵커 인용 원문 대조 (2026-08-31)
ITU-T Y.1541·G.114·G.113·RFC 4594·Cisco QoS·Aruba 문서를 실제로 확인:
- **유지(확인됨)**: jitter 심각 50ms = Y.1541 Class 0/1 IPDV ≤ 50ms · latency 150/400ms = G.114 편도 티어 경계 · occupancy ~50/75% = Aruba WLAN 가이드 · Cisco Enterprise QoS voice(편도 150 / jitter 30 / loss 1%).
- **정정(제거)**: RFC 4594는 텔레포니를 "jitter Very Low" **정성 등급**으로만 규정(§2.3, 수치는 Y.1541로 위임) — "RFC 4594 ~30ms" 잘못, 30ms는 Cisco 값. G.113 App.I는 **코덕별 Ie/Bpl(E-model 계수) 표**라 손실 문턱 아님. Y.1541 IPLR ≤ 0.1%는 우리 loss 스케일(0.5~10%)보다 **엄격** — 이 스케일은 Cisco 실무 기준.
- 반영: `congestion_label_redesign.{md,html}` §3, `collect_metrics.py`(repo+Pi번들) `ANCHORS` 주석, `CLAUDE.md`, `demo_api_spec.{md,html}`.
- redesign §3에 **"실측값이 앵커 대비 어디쯤"** 표 추가(idle/60·60/소패킷 캘리브레이션 각 축이 어느 밴드) — occupancy가 주력, jitter는 이 셋업에서 거의 안 뜸(peak 19ms < 경고 20ms), loss/latency는 peak만 심각.
- §4 "심각 = ping RTT ≥ 150ms" → **"편도 ≥ 150ms = RTT ≥ 300ms"** 정정 (앵커가 편도, `calculate_scores`가 RTT/2).

### 문서 정합성 정리
- **9→6 feature 산수** — "latency·jitter 2개 뺐다"만 적혀 7로 읽힘. 실제 `−2`(latency/jitter, leakage) `−1`(tx_retries+tx_failed → tx_retry_ratio) `= 6`. `CLAUDE.md`·`capstone2_vacation_summary.html`·`ap_features.py`·`redesign.md §5`(헤더 "6개"→"7개").
- **"RSSI 3종"** 표기 통일(`rssi×3`/`RSSI×3` → `RSSI 3종`), 각 문서 첫 등장 시 풀어씀. `model_features`에 RSSI 정의(수신 신호 세기, dBm, 0에 가까울수록 강함, 약하면 저MCS→airtime↑) 추가.
- **`congestion_label_criteria.{md,html}`** — "2차 진행 중" 배지·현재형 서술이 남아 있었음(2차는 재설계됨). 문서 **전체가 가중합 시절 기록**임을 상단에 명시, "2차 ap_metrics_v2 · 진행 중" → "2차 초기 (가중합) · archived".
- **`capstone2_vacation_summary.html`** 결과 표 — "모델 (5시드 평균)" 헤더인데 Pi INT8 열은 배포 단일 체크포인트. 열별 스코프 분리 + 배포 체크포인트 test 정확도(Baseline 91.6 등)가 5시드 평균보다 약간 낮다는 캡션.

### 문서 흐름 스캔 (README.html → 전체)
- 참조 무결성: 40여 개 파일 전부 존재, 섹션 번호 실제 헤더와 일치.
- **죽은 참조 정리**: `project/demo/API.md §4 표`(SDN 비교라고 가리켰으나 API.md에 SDN 내용 0) → `sdn_lstm.py` docstring + `CLAUDE.md` + `ap_model_comparison_redesign2.{txt,csv}` + work-log 9차. `project/README_AP_V2.md "핵심 검증 질문"` **6곳**(스텁이 돼서 그 섹션·정의 없음: `ap_crash_analysis`·`demo_api_spec`·`redesign.md`·`congestion_label_criteria` 각 .md/.html) → `congestion_label_redesign §1·§3` + `ap_v2_redesign2_threshold_comparison.txt`. `pi_latency_comparison.txt` 설명 "1~5차"→"3~5차+보존".
- **`.md` 참조 전부 `.{md,html}`로 통일** — HTML 형제 있는 6개 문서(`congestion_label_redesign`·`model_features`·`ap_crash_analysis`·`onnx_early_exit_redesign`·`congestion_label_criteria`·`demo_api_spec`)를 가리키는 모든 bare `.md`(문서 본문 + 소스 파일 헤더/주석). `collect_metrics.py`·`demo_server.py` repo↔Pi번들 바이트 동일 유지.

### `README.html` 링크화
- "문서 안내"가 네비게이션 목적인데 파일명이 전부 plain `<code>`였음. 로드 후 스크립트가 각 `<code>` → 링크: `.html`은 이 폴더 기준 상대경로(`yongsang/*.html`), `.md`/`.py`/`.txt`/`.csv`/`.pptx`는 `capstoneDesign2` GitHub blob(새 탭 + ↗), `x.{md,html}`는 `.html` 쪽, 글롭·bare 확장자·이미 `<a>` 안인 것은 스킵. `try/catch`로 실패해도 plain 유지.
- 검증: `node --check` + DOM셰임 실행 — 예외 없음, `<code>` 75개 중 링크 52 / plain 23(섹션참조·글롭 등 의도대로).

### `ap_cleaned_strict` 출처 재검토 (문서 변경 없음, 논의만)
사용자 질문 "인터넷 공개 데이터 맞나". git 증거로 재구성: raw는 `metrics_cleaned_strict_report.txt`에 `C:\...\카카오톡 받은 파일\metrics_cleaned.csv`(호중 수신본), 5시나리오 정확히 120행씩, 생성 코드 미커밋. 값이 물리적으로 비현실적(latency 0.047~0.163ms, RSSI −17~−30dBm, retry_delta max 23 vs 팀 실측 20만). 형식은 오히려 `collect_metrics.py` 계열 산출물(`channel_occupancy_method=instantaneous_fallback` 등)이라 "인터넷 데이터"라는 특정은 근거 약함. → 위 "다음 세션" 항목으로.

### 커밋
`e17a50e`~`69c9f78` 19개, `git push origin capstoneDesign2` 완료 (`cd68b94..69c9f78`).

## 13차 (2026-08-31) — 데모 서버 Pi 이식 (엣지 추론)

사용자: "demo_server.py도 Pi에서 돌게 이식해줘 그래야 실험이 맞지" (추론이 노트북이 아니라 실제 엣지 장비에서 돌아야 latency·엣지 서사가 성립).

### `project/demo/demo_server.py` 재작성
- **dual-import**: `try: from scripts.collect_metrics ... except ImportError: from collect_metrics ...` (`live_congestion.py`와 동일). `sys.path`에 스크립트 옆 디렉터리 추가 → 저장소/번들 같은 파일로 동작.
- **전면 인자화** (하드코딩 IP·별칭 전부 제거):
  - `--host --port` (기본 `0.0.0.0:8000`, env `DEMO_HOST`/`DEMO_PORT`)
  - `--iperf-target` (폰이 iperf3 쏠 대상 — 노트북이면 노트북 wifi IP, Pi면 Pi 자신 IP)
  - `--s21 --s26` (폰 SSH 대상 — 노트북은 `~/.ssh/config` 별칭 `s21`/`s26`, Pi는 `user@192.168.8.x`)
  - `--s21-port --s26-port --model --scaler --confirm --no-iperf-server`
  - `MODEL`/`SCALER` 기본값이 저장소 경로 없으면 스크립트 옆(번들)으로 fallback (`_resolve`)
- **iperf3 -s 자동 기동 조건**: `--iperf-target`이 로컬 IP(`hostname -I`로 확인)일 때만. 아니면 대상 호스트에서 직접 띄우라고 안내.
- `PHONES` dict로 폰 SSH 대상·포트 관리 (`PHONE_PORTS` 상수 대체).

### Pi 번들 갱신 (`project/deploy/raspberry_pi_ap_v2/`)
- `demo_server.py` + `demo.html` 추가 (이미 있던 `collect_metrics.py`, `scaler_params.json`, `*_int8_v2.onnx`와 함께 자립 실행 가능).

### 문서
- `project/demo/API.md` §6 재작성 — Pi 실행 명령·인자·env·번들 목록, Pi→폰 SSH 키 등록 절차.
- `project/demo/README.md` — "실행 (라즈베리 파이 — 엣지 추론)" 섹션 추가, 사전조건 표를 인자 기준으로.

### 검증
- 노트북 스모크 테스트 2경로 통과: (a) `project/demo/demo_server.py` (b) 번들 `raspberry_pi_ap_v2/demo_server.py` — 둘 다 모델 로드, HTTP bind, `/health`·`/` 응답 OK, dual-import 정상.
- **Pi 실기기 테스트 미완**: 이 세션 내내 Pi(`192.168.8.109`)·AP(`192.168.8.1`) 둘 다 connection timeout (이 PC가 Opal 망에 없음). Pi 온라인 시: 번들 scp → Pi 공개키를 두 폰 `authorized_keys`에 등록 → `python3 demo_server.py --iperf-target <PiIP> --s21 <user@폰> --s26 <user@폰>` 실행 후 부하 시나리오 재현.

## 11차 (2026-08-30 후속4) — 최소 라이브 추론 스크립트

**동기**: 모델·ONNX·비교는 다 됐지만 "실제 AP 데이터를 보고 현 상태가 혼잡인지 실시간 감지"하는 연속 루프가 없었음.

### `project/scripts/live_congestion.py` (신규)
- `collect_metrics.py`의 `APPoller`(지속 SSH `iw` 폴링) + parser 헬퍼(`parse_ap_cycle`, `summarize_stations`, `calculate_channel_occupancy`, `calculate_station_deltas`)를 **그대로 import** — feature 계산이 학습 시점과 동일하게 맞음.
- victim 프로브(`ProbeRunner`) 없음 — 라벨링 전용이라 추론엔 불필요.
- 루프: 폴링 → 7-feature 계산(occ 3폴링 median, retry/rssi 5폴링 rolling) → window deque(10) → min-max 정규화(`scaler_params.json`) → ONNX(`ap_early_exit_fixed_unified_int8_v2.onnx`) → softmax → 라벨.
- **"스파이크 아닌 현 상태" 4겹 안정화**: ① LSTM window 10(~10~20초 이력) ② occ 3폴링 median ③ retry/rssi 5폴링 rolling ④ 라벨 히스테리시스(`--confirm N`, 원시 예측 N회 연속 일치해야 확정 라벨 변경).
- 출력: `HH:MM:SS 혼잡: 정상(0) [원시 정상(0) p=1.00] P(정/경/혼/심)=[...] exit1 clients=3`

### 실측 검증 (idle AP)
laptop·**Pi 둘 다** 실행 확인. Pi(`capstone@192.168.8.109`)에서 `~/ap_pi_v2/live_congestion.py` → Pi가 직접 AP에 SSH(키 이미 됨) → window 채운 뒤 **정상(0), p≈1.00, exit1로 일관**. idle AP 안정 판정 + early-exit 동작 확인.

### Pi 이식 완료 (커밋 6672fde)
- `live_congestion.py`를 repo·번들 양쪽에서 동작하게: collect_metrics dual import, `AP_FEATURE_COLUMNS` 인라인, model/scaler 기본값이 스크립트 옆 파일로 fallback.
- 번들에 복사: `project/deploy/raspberry_pi_ap_v2/{collect_metrics.py, live_congestion.py, scaler_params.json}` (onnx는 이미 있음). Pi `~/ap_pi_v2/`에 scp.
- **Pi→AP SSH는 이미 됨** (별도 키 셋업 불필요 — 확인함).

### 부하 테스트 완료 — 라이브 혼잡 감지 실측 성공 (`ap_v2_redesign2_live_detection_20260830.txt`)
1차 시도(`ramp_load_remote.sh step`, 10/20/30/40M×60s)는 **~120초에 AP 크래시** — S21(191)이 STALE(약신호)라 capture effect. AP 전원 재부팅.
2차: S21 신호 회복(-30dBm, S26 -23dBm 대칭) 후 **두 폰 동시 iperf3 UDP -b 8M -l 1400 × 40s (합계 ~16M)** — AP 크래시 없이 완주.

Pi live_congestion.py --raw 실측 라벨 전이:

| 시각 | 확정 라벨 | occ | 상황 |
|---|---|---:|---|
| 18:50:54 | 경고(1) | 50% | 부하 시작 ~2초 후 |
| 18:50:58 | **심각(3)** | 67% | 채널 급악화 |
| 18:51:06 | **심각(3)** | 63%, retry 43% | |
| 18:51:22 | 경고(1) | 55% | 부하 감소 |
| 18:51:46 | 정상(0) | 37%, thr 0M | 부하 종료 후 복귀 |

라벨 분포: 정상 38 / 경고 48 / 혼잡 31 / **심각 21**.

**검증됨**:
- 실제 AP 텔레메트리로 **현 상태(스파이크 아님) 혼잡 라이브 감지** — 부하 ON→경고/혼잡/심각, OFF→정상 복귀.
- **심각(3)을 occ 63~73%에서 탐지** — occupancy 단독 문턱(75%)으론 못 잡는 구간을 학습 모델이 잡음. 발표 핵심 포인트가 라이브로 증명됨.
- early-exit 동작(idle exit1, 혼잡/심각 exit2/3), 히스테리시스가 경계(occ 50~73%) 원시 예측 튐을 확정 라벨로 안정화.

### 심각 지속 구간 재현 성공 (`ap_v2_redesign2_live_severe_sustained_20260830.txt`)
2단계 부하 (각 폰 iperf3 UDP -l 1400):
- phase 1: -b 13M × 55s → occ 33~59%, 대부분 경고(1). **채널이 건강해 13M로는 심각 안 됨** (앞선 8M 심각은 S21 신호 marginal했던 탓). 신호 좋아지니 같은 throughput에 occ가 더 낮음(MCS↑ = airtime↓).
- phase 2: -b 20M × 70s → occ 60~90%, retry 30~45%, throughput 47~131M → **심각(3) 지속**.

**결과**: 확정 라벨 심각(3)이 **92 폴링 연속 = ~46~49초** 유지 (18:57:17~18:58:06). p(심각) 대부분 0.90~0.99. 부하 종료 직후 정상(0) 복귀 (window lag로 occ 하락 후 몇 폴링 심각 유지 — 정상 동작). AP 크래시 없음(up 21분, 20M×2=40M는 대칭 강신호에서 안전). 라벨 분포(전체): 정상 131 / 경고 125 / 혼잡 33 / 심각 101.

### 데모 웹 대시보드 완성 (`project/demo/`)
사용자 요청: "버튼으로 계단형 부하 걸고 모델이 실시간으로 혼잡 예측하는 걸 보여주는 간단한 웹사이트".

- **`project/demo/demo_server.py`** (stdlib http.server + numpy/onnxruntime): `live_congestion.py` 로직 재사용(APPoller + 7-feature + ONNX) + SSE 스트림(`/events`) + `POST /load {rate:10M|20M|30M|off}`(두 폰 iperf3 제어) + `GET /signal`(폰 신호 대칭 확인). `iperf3 -s` 서버 자동 기동. 30M 상한.
- **`project/demo/demo.html`** (서버가 서빙 = 동일 출처, CSP 무관): 큰 혼잡 레벨 카드(색상 코딩) + 4클래스 확률 바 + 7-feature 실시간 스트립 + `10/20/30M`·정지 버튼 + 폰 신호세기(비대칭 경고) + 90초 canvas 차트(레벨 + occ).
- 전 엔드포인트 실측 테스트 통과: `/health` `{ready:true}`, `/signal` `{symmetric:true}`, `/events` SSE 정상, `POST /load` → 두 폰 iperf3 제어.
- `project/demo/README.md` (빠른 실행), **`project/demo/API.md` (팀 구현용 확정 스펙)**.
- **UI 정직성 수정** (사용자 지적: "confirm이 모델 겉모습을 좋게 만들지 않냐"): `demo.html`을 좌우 2패널로 — **모델 출력(원시 argmax, 정확도 평가 기준)** + **안정화 상태(N폴링 debounce, 표시/제어용 후처리)**. 차트도 원시(점선)/확정(실선) 둘 다. 안정화 패널에 "모델·학습·평가엔 없음" 명시. → `CONFIRM=5`로 (30M 심각 지속 ~22초→~58초, 원시 패널은 그대로 노출).

### `project/demo/API.md` — 팀 구현용 명세서 (신규)
"팀이 데모를 만들어야 하는데 확실히 보고 만들 수 있는 설명서 필요" 요청. 레퍼런스 구현(`demo_server.py`+`demo.html`)의 **계약을 고정**:
1. 무엇을 하는가 + 아키텍처 다이어그램
2. **HTTP API 전체** — `GET / /health /events`(SSE 2형태) `POST /load` `GET /signal`, 요청/응답 스키마·예시
3. **모델/추론 계약** (feature 순서·window 10·scaler·ONNX I/O·`collect_metrics` 헬퍼 재사용·`label` vs `raw_label`) — "바꾸지 말 것"
4. 부하/AP 크래시 규칙 (30M 상한, 신호 대칭)
5. 실행/사전조건 + Pi 이식 메모
6. 레퍼런스 검증 상태 (2026-08-30 결과 표)
7. 팀이 만들/확장할 것 (백엔드 프레임워크, 설정 외부화, 대시보드 강화, 계단 자동 진행, 안전장치, Pi 배포, **밴드 스티어링 훅**, 인증)

`docs/yongsang/demo_api_spec.md`(원래 구상한 3-면 큰 그림)는 상단에 API.md 포인터 추가 — "실제 만든 최소 버전의 확정 스펙은 API.md, 이 문서는 확장 방향 참고".

### 데모 전체 시나리오 실측 (`ap_v2_redesign2_demo_full_run_20260830.txt`)
버튼 10M→20M→30M→정지 (폰 신호 대칭 S21/S26 둘 다 -23dBm):
- 10M → 경고/혼잡, 20M → 혼잡/심각(occ 63~69%), **30M → 심각 지속 ~22초** (occ 76%, retry 54%), 정지 → 정상 복귀
- 라벨 분포: 정상 29 / 경고 73 / 혼잡 68 / **심각 42**
- 30M에서 채널 포화: 폰은 30M씩 밀지만 delivered throughput 12~50M로 떨어지고 retry 54%까지 → 혼잡의 전형 시그니처, 모델이 심각으로 잡음.
- **스파이크 저항 확인**: throughput 3174M 스파이크(SSH 타이밍 아티팩트)에도 확정 라벨 혼잡 유지 (3폴링 occ median + window 10 흡수).
- AP 크래시 없음 (30M×2=60M ~70초, up 40분).
- occ 50~77% 경계에서 원시 예측 튐 (label 1/2/3 경계 노이즈) — `demo_server.py`는 `CONFIRM=5`로 설정(30M 심각 ~58초 연속). 원시 패널은 그대로 노출.
- CONFIRM=5 재실행 확인: 확정 라벨 변경 16회 vs 원시 22회, 30M 심각 ~58초 연속. AP 크래시 없음(up 2h).

### 방학 개발 흐름 정리 (`docs/yongsang/capstone2_vacation_summary.html`)
"이번 방학 진행 흐름 html" 요청. `docs/capstone1_summary.html`(1학기)의 2학기 판.
- 6개 phase 타임라인 (데이터 실측 → 라벨 재설계 → 본수집·ONNX → 7-feature·시드 → power=0.0·SDN 재구현 → 라이브·데모), 8/21~8/30 커밋 히스토리 기반.
- 결과 스냅샷(정확도 90~92% / 지연 0.54ms / 라이브 검증), 발표 목표 대비(목표1 미달·목표2 달성).
- **되돌린 결정 4건** 별도 섹션 (SDN 가중치 승격→철회, occupancy 스무딩→폐기, power 절벽 가정→소멸, 1차 데이터→포기).
- 아티팩트: "방학 개발 흐름" https://claude.ai/code/artifact/d3948502-2ebc-43b9-b1f4-237acf4c1b3b (repo에도 standalone HTML 저장, CLAUDE.md 참고문서 목록에 추가).

### 남은 것
- Pi 정확도 95% (현재 90~92%) — label 2 경계 노이즈 / label 3 관측성 한계.
- 데모 대시보드 팀 구현 (`project/demo/API.md` 기준).
- 밴드 스티어링 (발표 슬라이드7 최종 목표).

## 12차 (2026-08-31) — 문서 정리 + 네비게이션 문서

사용자: "문서가 너무 많다. 필요없는 것 정리하고, 팀원이 보고 '왜 이런 라벨 정했는지 이 문서 보면 되겠네' 할 수 있는 안내 문서 만들어줘."

### 신규: `docs/README.md` — 문서 안내 (질문 → 문서)
- "처음 오면 이 4개" (CLAUDE.md / work-log / capstone2_vacation_summary.html / congestion_label_redesign.md)
- **"왜 이렇게 정했나" 질문별 표** — 라벨 정의·congestion_score·latency/jitter 제외·7-feature·class-weight-power·AP 크래시·ONNX 배포·SDN·데모·발표 목표·재현·환경 → 각각 어느 문서/코드
- 결과 파일 목록 (`project/results/yongsang/`), stale/역사용 목록, HTML 목록
- CLAUDE.md 참고파일 목록 맨 앞(0번)에 추가.

### 정리
- **`README.md`(레포 루트) 전면 재작성** — 최악의 stale였음(9-feature·가중합·"ONNX/Pi 배포 아직 없음"·`--class-weight-power 1.0`·`metrics_v2.csv`). GitHub 첫 화면이라 우선. 현재 상태 표 + `docs/README.md` 포인터 + 정확한 실행 명령.
- **`project/README_AP_V2.md` → redirect 스텁** — 9-feature·가중합 시절 내용, 여러 문서가 참조 중이라 삭제 대신 축약(현행 문서 표 + git 히스토리 안내).
- `image.png`(루트 방치 스크린샷), `__pycache__` 삭제.
- **팀원별 기록(`docs/hochung/`, `docs/yena/`, `guideline_*`, `stage*_work_log`)은 보존** — CLAUDE.md "기록은 다 남겨둠" 정책. `docs/README.md`에 "맥락용, 안 읽어도 됨"으로 분류.

## 10차 (2026-08-30 후속3) — 발표자료 비교표 정리 + 문서 stale 감사·수정

### 발표자료용 비교표
- occupancy 문턱 vs 학습 모델 비교를 현행 배포 체크포인트로 재측정(`ap_v2_redesign2_threshold_comparison.txt` 갱신, 옛 8/28 6-feature 버전 대체): 학습 모델 L3 F1 65~69% ≫ 최선의 문턱(occ≥70%) F1 44%. 심각 창의 74%가 occ<75%(프로브 축으로 심각) — 문턱은 정의상 0%, 학습 모델은 39~48% 잡음.
- 조기경보(forecasting): k=3 escalation recall 61.5% / k=5 45.5% (occupancy 규칙은 구조상 0%).
- 아티팩트 "AP 혼잡 분류 모델 비교" 게시 (https://claude.ai/code/artifact/fc9c86b1-37e3-40e1-b0c8-aee9cca59f8f) — 정량목표 현황 / 아키텍처 비교표(5시드+배포) / 심각 탐지 vs 문턱 / 조기경보 / 발표 3줄 요약.

### 문서 stale 감사 (사용자 요청: "과거 기록으로 남아 최신화 안 된 것")
전 HTML/MD를 7-feature·power=0.0·SDN 재구현·Baseline seed3 기준으로 훑어 수정:

| 문서 | stale 내용 | 수정 |
|---|---|---|
| `congestion_label_criteria.{html,md}` §6 | "절벽형 → power=1.0 채택" (4-feature·label3=23개 시절) | archived callout — 8/30 재스윕 결과(power=0.0 최고, 트레이드오프 없음) |
| `congestion_label_redesign.md` | "sta_tx_bitrate_* 뺌", "6개 feature" | 8/29 다중시드 재검증으로 `sta_tx_bitrate_mean` 승격됨을 반영, feature set 7개 |
| `demo_api_spec.html` | .md는 8/29 갱신됐는데 html 방치 — `/meta`·`/stream`이 9-feature·`trained_on_rows 5574`·옛 가중합 공식 | max(anchor) + 7-feature로 동기화 (infer_ms도 0.87→0.55) |
| `onnx_early_exit_redesign.{html,md}` | 상단 callout이 SDN 재구현·seed3·Pi 재측정 전 시점 | 최신 갱신 (Baseline 0.746 / SDN 0.572 / EE 0.540, −28%) |
| `capstone1_summary.html` | 2학기 forward-ref 한 줄 "9-feature / 0.747·0.595ms" | "7-feature / 0.746·0.540ms" |

- **건드리지 않은 것(의도적)**: `onnx_early_exit_redesign.html` 본문의 `0.641ms·−67%` 바 차트 등 — 각 문서가 "그 시점 기록으로 보존" 정책 명시, 상단 callout이 현재 수치 제공. `ap_crash_analysis.html`·1학기 guideline은 영향 없음.
- 커밋 `d79930e`(threshold 갱신)·`d914348`(문서 stale 수정).

## 9차 (2026-08-30 후속2) — SDN을 논문 충실 비교모델로 재구현

### 배경
"SDN이 Proposed와 백본 100% 동일 → 비교 모델로 성립 안 됨"을 사용자와 확인. SDN은 "기존 조기종료 LSTM 방법 vs 우리 방법 — 뭐가 다른가"의 통제 비교점이어야 하므로, **base 백본·하이퍼파라미터는 완전히 고정하고 SDN이 실제로 규정하는 3축만 논문대로**:

Kaya et al. (Shallow-Deep Networks, ICML 2019) 공식 코드 확인:
- **IC 구조**: `alpha·maxpool + (1-alpha)·avgpool → Linear` (공식은 spatial pooling, 여기선 timestep pooling). Proposed는 마지막 timestep→Linear.
- **loss 가중**: `cur_coeffs = min(max_coeffs, 0.01 + epoch·max_coeffs/epochs)`, `max_coeffs=[0.15,0.30,...]` IC마다 +0.15, **최종 분류기는 항상 1.0**. Proposed는 고정 균등 0.3/0.3/0.4.
- **threshold**: `early_exit_experiments.py`가 val에서 스윕. → `calibrate_confidence_threshold`: "full-network 정확도 −1% 이내 최소 비용 exit" 운영점. Proposed는 entropy θ 고정/변동.

### 구현 (`models/sdn_lstm.py` 전면 재작성)
- `SDNInternalClassifier` (pooling head), `sdn_loss_coeffs`(램프), `sdn_multi_exit_loss`(최종=1.0), `calibrate_confidence_threshold`(grid 0.50~0.99)
- exit3 = base 네트워크 원래 헤드(마지막 timestep→Linear, BaselineLSTM과 동일) — "기존 네트워크에 IC 추가, 최종 헤드 유지"
- `train_ap_sdn.py`: epoch별 램프 계수, 학습 후 val 캘리브레이션해서 T를 체크포인트에 저장
- `export_onnx_ap_sdn.py`: staged 래퍼가 IC에 full seq 전달(pooling은 IC 내부)

### 5시드 결과 (power=0.0)
| | acc | L3 F1 | L3 recall | 캘리브레이션 T |
|---|---:|---:|---:|---|
| SDN 논문 충실 | 90.4%±1.4 | **60.4%±8.1** | 45.8%±8.8 | 0.71~0.86 (시드별 요동) |
| (옛 SDN-style, archived) | 90.4%±0.7 | 65.3%±2.5 | 52.3%±3.2 | 0.85 고정 |
| Proposed EE Fixed | 90.7%±0.7 | 64.5%±5.5 | 51.0%±6.6 | — |

- **정확도는 동급, label3에서 논문 SDN이 뚜렷이 덜 안정**(F1 편차 2배) — per-model 캘리브레이션이 작은 val(308창)에서 취약.
- 배포 = seed3 (val-best, T=0.80): test 90.3%, L3 F1 65.3%, exit 36/33/31%
- ONNX 재수출: staged + unified int8 (PyTorch 대비 pred 308/310·exit 307/310, INT8 정확도 91.0%)
- **Pi INT8 재측정: 0.572 / 0.574ms** (옛 SDN-style 0.534보다 +0.04 — pooling IC의 ReduceMax/Mean 오버헤드). Proposed(0.540)보다 -6% 느림.
- 아카이빙: 옛 SDN-style 체크포인트 → `archived_sdn_style_20260830/`, 리포트 → `*_style_archived_20260830.txt`

### 서사
"SDN을 이겼다"가 아니라 **"기존 조기종료 방법(SDN)과 동급 정확도를 내면서, exit head가 더 가볍고(→ 더 빠름), 희소 클래스에서 더 안정적이며(→ per-model 캘리브레이션 불필요), 트래픽 적응형 임계값(Dynamic θ)을 쓸 수 있다"** — 우리 설계 선택의 효과를 보여주는 통제 비교.

## 8차 (2026-08-30 후속) — Baseline·SDN 5시드 특성화, Baseline 배포 seed 교체

7차에서 Proposed EE만 5시드로 특성화했던 걸 Baseline·SDN도 동일하게(power=0.0, 시드 0~4). `char_baseline_sdn_5seed.py`.

| 모델 | 5시드 test 정확도 | Label3 F1 | Label3 recall | val-best seed (val bal) |
|---|---:|---:|---:|---|
| Baseline   | 92.0% ±0.7 | 66.1% ±2.3 | 54.2% ±3.8 | **seed3 (89.0%)** ← 기존 배포 seed0(87.8%)에서 교체 |
| SDN-style  | 90.4% ±0.7 | 65.3% ±2.5 | 52.3% ±3.2 | seed0 (88.2%) — 배포 그대로 |
| EE Fixed θ | 90.7% ±0.7 | 64.5% ±5.5 | 51.0% ±6.6 | seed4 — 배포 그대로 |

### 핵심
- **세 모델 정확도가 5시드 평균으로 사실상 동급** (92.0 / 90.4 / 90.7%, 전부 ±0.7 — 차이가 std의 2배 수준). Label3 F1도 64~66%로 동률. → "Baseline이 정확도 1위"라는 7차 서술은 **"셋이 비슷한데 Baseline이 약간 앞"으로 완화**. 배포 단일 체크포인트 숫자(Baseline seed3 91.6% 등)는 test draw 편차가 있으니 **5시드 평균을 기준 수치로 인용**.
- **Baseline 배포 교체**: seed0(val bal 87.8%, test 92.3%)은 val-best가 아니었음 — 운 좋은 test draw. 정직한 selector(val balanced acc)로는 seed3(89.0%, test 91.6%, L3 F1 65.5%). 교체 후:
  - eval report 재생성 (91.6%), 기존 seed0 report → `*_seed0_archived_20260830.txt`, 체크포인트 → `checkpoints/ap_v2_redesign2/archived_baseline_seed0_20260830/`
  - `export_onnx_ap_baseline.py` 재실행, deploy 번들 sync, Pi `~/ap_pi_v2/` scp
  - **Pi INT8 재측정: 0.746 / 0.752ms** (seed0의 0.739와 오차범위 — Baseline은 weight 값 무관 항상 3층)
- SDN·EE는 val-best가 배포된 그대로라 재수출·재측정 불필요.
- 문서 갱신: `ap_v2_redesign2_pi_latency_comparison.txt`(4차 섹션), `ap_v2_redesign2_pi_bench_power0_20260830.txt`(후속 섹션), `generate_ap_comparison.py`(하드코딩 + 5시드 평균) + 비교표 재생성, `CLAUDE.md`, 아티팩트.

## 7차 (2026-08-30) — power=0.0 전체 승격 (재학습·평가·ONNX·Pi 재측정·문서 전부 완료)

5차에서 power=0.0 확정 → item 3(전체 승격) 실행.

### 완료
1. **코드 기본값 변경**: `train_ap_early_exit.py`·`train_ap_baseline_lstm.py`·`train_ap_sdn.py` 셋 다 `--class-weight-power` 기본값 1.0 → 0.0. `compute_class_weights` docstring의 stale한 "power≤0.85 절벽" 근거를 2026-08-30 재스윕 결과로 정정(보존). `--seed`/`--class-weight-power` help 문자열 갱신.
2. **아카이빙**: `checkpoints/ap_v2_redesign2/` → `checkpoints/ap_v2_redesign2_power1_archived_20260830/`(전체 복사). 결과 리포트 8종 → `project/results/yongsang/*_power1_archived_20260830.*`.
3. **재학습** (power=0.0, 균등 exit weights 0.3/0.3/0.4, hidden 128, dropout 0.2, epochs 50):
   - Baseline: 시드 0/1/2 (val bal 87.8/87.7/87.1) → **seed0 선택**
   - SDN: 시드 0/1/2 (val bal 88.2/86.7/87.1) → **seed0 선택**
   - Proposed EE: 시드 0~4 (val bal 86.5/86.8/86.5/86.4/86.8) → val 최고 동률(seed1·4) 중 fixed L3 F1 tiebreak로 **seed4 선택** (seed1이 L3 recall 38.7%로 유독 나빴음 — 4차에서 예고된 EE 시드 불안정성 재확인, 그래서 EE만 5시드로 늘림)
4. **평가** (fp32, canonical):

   | 모델 | 정확도 (p1 → **p0**) | Label3 recall | Label3 F1 |
   |---|---|---:|---:|
   | **Baseline (EE 없음)** | 88.7% → **92.3%** | 58.1% (유지) | — |
   | SDN-style | 89.0% → **90.6%** | 58.1% → 48.4% | — |
   | Proposed EE fixed θ (seed4) | 88.1% → **90.6%** | 61.3% → 54.8% | 65.5% → **68.0%** |
   | Proposed EE dynamic θ (seed4) | 87.7% → **91.0%** | 61.3% → 54.8% | — → 69.4% |

   - Proposed EE 5시드 평균(fixed): acc 90.7%±0.7%, L3 F1 64.5%±5.5%, L3 recall 51.0%±6.6%. (seed3이 test는 91.9%/F1 70.6%로 최고지만 val bal 최저라 선택 안 됨 — 정직한 selector 준수.)
   - **핵심**: power=0.0가 전 모델 정확도를 +1.6~3.6pt 올림. **Baseline이 92.3%로 95% 목표에 가장 근접.** 대신 EE/SDN의 label3 recall은 내려감(power=1.0이 label3 과보호하던 것 — 4차 confusion matrix 분석대로). EE label3 F1은 65.5→68.0으로 오히려 소폭 상승(precision이 크게 오름).
   - Label별(EE seed4 fixed): L0 97.9% / L1 97.0% / L2 90.6% / L3 54.8%. Exit 종료율 30.0/50.6/19.4%.

### 완료 (계속)
5. **ONNX 재수출**: baseline/sdn/ee staged+unified+int8_v2 전부. unified fp32 = PyTorch와 310/310 일치(fixed·dynamic). INT8 v2 = fixed 308/310·dynamic 309/310(양자화 노이즈). INT8 직접 정확도: fixed 90.3%/F1 65.3%, dynamic 90.6%/F1 66.7%.
6. **문서 갱신**: `ap_model_comparison_redesign2.{txt,csv}` 재생성(+`generate_ap_comparison.py` 하드코딩 문자열 power=1.0→0.0, Pi 수치는 "재측정 pending"으로), `ap_v2_redesign2_pi_latency_comparison.txt` 상단 STALE 배너, `CLAUDE.md`(최신 평가 결과 섹션 전면 갱신 + 목표1 줄 + 해석기준 7번), `docs/yongsang/onnx_early_exit_redesign.md`(2026-08-30 콜아웃).

### 커밋·푸시 완료
- 커밋 `a3f9bde` "AP redesign2: promote 7-feature + class-weight-power=0.0" (253 files) — 8/29 7-feature 승격 + 8/30 power=0.0 승격 + `collect_metrics.py` survey 패치를 한 커밋에(마지막 커밋 `87c90ef` 이후 두 세션 분량이 섞여 있어 체크포인트 분리 불가). `image.png`(루트 방치 스크린샷)만 제외.
- `git push origin capstoneDesign2` 완료: `87c90ef..a3f9bde`.

### Pi INT8 재측정 완료 (2026-08-30, power=0.0)
`capstone@192.168.8.109`, ort 1.26.0 CPU, test 310창, 5회 반복/샘플. 새 ONNX를 `~/ap_pi_v2/`로 scp 후 `bench_baseline.py`/`bench_unified.py`.

| 모델 | Pi INT8 avg(ms) | p50 | p95 | exit 1/2/3 |
|---|---:|---:|---:|---|
| Baseline | **0.739** | 0.738 | 0.748 | -/-/100% |
| SDN-style | **0.534** | 0.535 | 0.821 | 33.9/36.5/29.7% |
| Proposed Fixed θ | **0.540** | 0.563 | 0.858 | 30.3/50.3/19.4% |
| Proposed Dynamic θ | **0.555** | 0.613 | 0.913 | 31.9/57.7/10.3% |

- power=1.0 대비: Baseline 0.756→0.739(≈동일), SDN 0.636→0.534, **Proposed Fixed 0.641→0.540(-16%)**, Dynamic 0.645→0.555(-14%). EE가 빨라진 건 exit3 도달률 52%→19%(Fixed)/10%(Dynamic) 때문 — power=1.0의 label3 과보호가 없어지며 뒤 exit로 미루는 경향 감소. Pi 실측 exit 분포가 fp32 eval과 정확히 일치.
- **4개 모델 전부 목표2(<1ms) 달성.** 이번엔 **Proposed(EE)가 Baseline보다 명확히 빠름**(0.54 vs 0.74ms, -27%) — power=1.0 라운드에선 비슷했는데 격차 벌어짐. SDN(0.534)≈Proposed Fixed(0.540) 속도 동률, 정확도도 둘 다 90.6%, Proposed가 Label3 recall/F1 우위.
- 문서 갱신: `ap_v2_redesign2_pi_latency_comparison.txt`(3차 섹션 추가, 2차는 보존), `generate_ap_comparison.py`(Pi 수치 하드코딩 갱신) + 비교표 재생성.

### 남음
- Baseline/SDN도 5시드로 완전 특성화 (지금 3시드) — 여유되면.
- HTML 아티팩트("Early-Exit Reweighting") 갱신 여부 판단 — 이번 변화가 크므로 별도 아티팩트가 나을 수도.

### 참고 — 이번 결과의 서사적 함의
Baseline(EE 없음)이 정확도 1위가 됨. Proposed(Early Exit)의 가치 주장은 **정확도가 아니라 속도/효율**(발표 목표2 <1ms, Pi INT8 0.6ms대) + "간섭 감지에 EE 최초 적용" — 이건 원래 발표자료 프레이밍과 일치. 95% 목표 관점에서는 **Baseline 92.3%를 기준선으로 삼는 게 정직** (목표까지 test 310창 중 8.4개).

## 진행 중 (2026-08-30, 6차) — survey rx/tx airtime + 외부 점유(간섭) feature 탐색

### 배경
label3 오답 분해(위 3차·이번 세션): label3의 75%가 **victim 프로브 loss 축**으로 결정되는데 모델은 그 축을 못 봄. 기존 28개 raw 컬럼에서 파생 가능한 후보(bitrate_min, retx_per_s, clients, occ/client 등) 전부 loss_score와 상관 ≤ 0.31로 약함 — 지금 데이터엔 victim 손실 대리 신호가 없음. 1학기 모델은 loss·latency를 입력으로 직접 줬기 때문에 label3 recall 100%였던 것(+ 시뮬레이터 데이터).

Opal(GL-SFT1200)은 SSH로 `iw dev wlan0 station dump` + `iw dev wlan0 survey dump` 두 명령만 제공. survey 출력에 `channel receive time` / `channel transmit time` / `noise`가 들어있는데 `collect_metrics.py`가 active/busy만 읽고 이 3줄을 버리고 있었음. **busy − (rx + tx) = 우리 AP는 아무것도 안 했는데 채널이 바빴던 시간 = 동일 채널 다른 AP·간섭(co-channel interference)** → occupancy는 중간인데 이게 크면 victim이 contention으로 굶는 상황의 대리 신호가 될 수 있음.

### 코드 변경 (`project/scripts/collect_metrics.py`) — 완료, 재수집 대기
- `parse_channel_occupancy`: `channel receive/transmit time`·`noise` 추가 파싱, 5-tuple 반환. mt76가 이 줄들을 안 뱉으면 None → CSV 빈칸, occupancy 계산엔 영향 없음(방어적).
- `survey_counter_percent` 헬퍼 추가: 누적 카운터(ms)를 active time 대비 %로, occupancy와 동일하게 delta 우선.
- 신규 CSV 컬럼 4개 (`channel_occupancy_method` 뒤): `channel_rx_time_percent`, `channel_tx_time_percent`, `channel_ext_busy_percent`(= busy−rx−tx), `noise_dbm`. **전부 탐색용 — 라벨 max 축 아님, 모델 feature 아님.** 유의미하면 그때 승격.
- `CSV_FILE`을 `COLLECT_CSV_FILE` 환경변수로 override 가능하게 (탐색 수집을 canonical CSV와 분리).
- 단위 테스트 통과 (rx/tx/noise 파싱, delta·first·reset·none-safe 경로).
- 다운스트림 영향 없음 확인: `remeasure_redesign.py`는 `df.copy()` 후 score 컬럼만 덮어써서 새 컬럼 통과, `prepare_ap_metrics_dataset.py`는 명시적 feature 리스트라 새 컬럼 무시.

### 다음 (사용자 실행)
1. **필드 존재 확인**: `ssh root@192.168.8.1 "iw dev wlan0 survey dump"` — `channel receive time` / `channel transmit time` / `noise` 줄이 나오는지. (안 나오면 이 방향 폐기.)
2. **탐색 수집**: step 프로파일(10/20/30/40M × 60초 = 240초) 몇 회. `COLLECT_CSV_FILE=metrics_v2_surveyext.csv python collect_metrics.py step_ext_runN`. victim 프로브(iperf3 -u -b 300k, 노트북 5203) + ping 서버 살아있어야 loss/latency 축이 라벨에 반영됨.
3. **분석**: label2 vs label3 (occ 45–68% 혼동 구간)에서 `channel_ext_busy_percent`·`noise_dbm`의 Cohen's d, loss_score와의 상관. bitrate_mean(d≈0.43)·loss_score 상관(≤0.31) 넘으면 8번째(또는 그 이상) feature로 승격 검토.

## 체크포인트 (2026-08-30, 5차) — power 0.1/0.15 갭 스윕, 절충점 없음 → power=0.0 확정

4차에서 비어있던 그리드 값(0.1, 0.15) 3시드씩 채움 (`class_weight_power_sweep_010_015.py`, EXIT_WEIGHTS=(0.3,0.3,0.4), hidden 128, dropout 0.2, epochs 50):

| power | 정확도 | Label3 F1 | Label3 recall | Label3 precision |
|---|---:|---:|---:|---:|
| **0.0** | **91.3%±0.5%** | **69.8%±2.3%** | ~51.6% | 높음 |
| 0.1 | 91.0%±0.9% | 68.4%±2.3% | 54.8%±2.6% | 91.1%±2.4% |
| 0.15 | 90.8%±0.2% | 67.6%±1.7% | 53.8%±1.5% | 90.9%±2.5% |
| 0.2 | 90.1%±1.4% | 66.2%±3.1% | | |
| 0.3 | 89.7%±0.5% | 64.3%±2.6% | | |
| 1.0(현재 기본값) | 87.0%±1.1% | 63.2%±0.5% | | |

**결론**: 0.1/0.15는 정확도·F1 둘 다 0.0과 0.2 사이에 단조롭게 놓임 — label3 recall이 살아나는 절충점 없음(0.1의 recall 54.8%가 0.0보다 근소 높지만 그 차이는 std 안, 정확도·F1은 여전히 0.0이 최고). **power=0.0으로 확정.** 사용자 예측("낮을수록 좋았으니 상관없을 것") 맞음. 이 축의 재검증 끝 — 다음은 전체 승격(item 3).

### 다음 세션 첫 번째 할 일 — 전체 승격 (item 3), power=0.0 기준

8/29 `sta_tx_bitrate_mean` 승격 때와 동일한 순서:
1. `train_ap_early_exit.py` + `train_ap_baseline_lstm.py` + `train_ap_sdn.py` 셋 다 `--class-weight-power` 기본값 1.0 → 0.0 변경. `train_ap_early_exit.py`의 `compute_class_weights` docstring에 남아있는 "power≤0.85면 label3 recall 절벽" 근거는 stale(4-feature·train label3=23개 시절) — 주석으로 정정하며 보존.
2. 기존 체크포인트/결과 아카이빙 (`*_power1_archived_20260830.*` 식). Pi(`~/ap_pi_v2/`)도.
3. Baseline/SDN/Proposed 각각 시드 0/1/2 재학습, val balanced acc 최고를 배포 체크포인트로.
4. ONNX 재수출 (feature 개수 안 바뀌니 8/29 hardcode 버그 재발 없음): `export_onnx_ap_baseline.py`, `export_onnx_ap_sdn*.py`, `export_onnx_ap*.py` + `export_onnx_ap_unified.py` + `export_onnx_ap_unified_int8_v2.py`.
5. Pi 재측정 (INT8).
6. 문서 갱신: `ap_v2_redesign2_eval_report.txt`, `ap_v2_redesign2_pi_latency_comparison.txt`, `ap_model_comparison_redesign2.{txt,csv}`, `CLAUDE.md`(class weight power 1.0 → 0.0), `docs/yongsang/onnx_early_exit_redesign.md`.

## ⭐ (4차 기준) 시작 지점 — class-weight-power 승격 마무리부터

**지금 상태**: `train_ap_early_exit.py`의 `--class-weight-power` 기본값은 아직 **1.0**(코드 미변경, 순수 실험만 한 상태). 실험 결과 **power=0.0(가중치 없음)이 정확도·F1 둘 다 최고**라는 게 3시드로 확인됐지만, 아직 어떤 코드/체크포인트/문서도 안 바꿨다 — 전부 스크래치패드 실험.

**다음 세션 첫 번째 할 일 (순서 확정)**:
1. **power=0.1, 0.15 추가 스윕(3시드씩)** — label3 recall을 살리면서 label2 개선폭을 최대한 유지하는 절충점이 있는지 확인. 지금까지 그리드(0.0/0.2/0.3/0.5/0.7/0.85/1.0)에 0.1·0.15만 비어있음. 재현 코드: `class_weight_power_sweep.py`의 `POWERS` 리스트를 `[0.1, 0.15]`로 바꿔서 그대로 재사용(스크립트 구조는 위 "체크포인트 (2026-08-30, 4차)" 참고, 스크래치패드에 없으면 그 섹션 세팅대로 재작성).
2. 절충점(또는 여전히 0.0이 최선)이 정해지면 최종 값 확정.
3. 확정되면: item 2(sta_tx_bitrate_mean) 때와 동일한 순서로 전체 승격 진행 — ①`train_ap_early_exit.py`(및 `train_ap_baseline_lstm.py`/`train_ap_sdn.py`, 지금 셋 다 `--class-weight-power` 기본값 1.0) 기본값 변경 ②기존 체크포인트/결과 아카이빙(`*_power1_archived_20260830.*` 식) ③Baseline/SDN/Proposed 재학습(멀티시드, val 최고 선택) ④ONNX 재수출(`export_onnx_ap_unified_int8_v2.py` 등 — 이번엔 feature 개수 안 바뀌니 8/29에 고친 hardcode 버그는 재발 안 함) ⑤Pi 재측정 ⑥비교표·문서 갱신

## 체크포인트 (2026-08-30, 4차) — class-weight-power 재검증, 87.0%→91.9%까지 개선(미승격)

### 배경
8/29 밤 confusion matrix 분해에서 "label2(혼잡)가 label3보다 오답이 많다"는 걸 확인한 뒤, 다른 개선 방안 후보 중 **class-weight-power 재검증**(8/23에 4-feature·label3 23개짜리 옛날 데이터로 고정된 값, 그 뒤 라벨 재설계·7-feature 승격을 다 겪고도 재검증 안 함)을 사용자가 선택.

### 1차 스윕 (0.5/0.7/0.85/1.0, 3시드씩, 균등 exit 가중치 고정)
| power | 정확도 | Label3 F1 |
|---|---:|---:|
| 0.5 | 89.6%±0.8% | 67.1%±3.3% |
| 0.7 | 88.3%±1.7% | 61.2%±3.2% |
| 0.85 | 88.0%±1.0% | 62.4%±2.4% |
| 1.0(현재 기본값) | 87.0%±1.1% | 63.2%±0.5% |

power를 낮출수록 개선되는 추세 확인 → 사용자 요청으로 더 낮은 값 추가 스윕.

### 2차 스윕 (0.0/0.2/0.3, 3시드씩)
| power | 정확도 | Label3 F1 |
|---|---:|---:|
| **0.0** | **91.3%±0.5%** | **69.8%±2.3%** |
| 0.2 | 90.1%±1.4% | 66.2%±3.1% |
| 0.3 | 89.7%±0.5% | 64.3%±2.6% |

**power=0.0(클래스 가중치 완전히 없음)이 정확도·F1 둘 다 최고** — 0 밑으로는 못 내려가므로(방향이 뒤집힘) 이 축의 경계 최적값. 87.0%→91.3%(+4.3pt), F1 63.2%→69.8%(+6.6pt) — 트레이드오프 없는 순수 개선.

**왜 이렇게 됐나**: 8/23 결정 당시(4-feature, train label3=23개)는 "power≤0.85면 label3 recall이 0%로 떨어지는 절벽"이 있었음(`train_ap_early_exit.py`의 `compute_class_weights` docstring에 그 근거가 남아있음, 이제 stale). 지금(7-feature, train label3=141개)은 그 절벽이 사라짐 — 데이터가 늘면서 가중치 없이도 label3를 자연스럽게 학습 가능해졌는데, 옛 결정을 한 번도 재검증 안 해서 계속 손해보고 있었음.

### power=0.0 confusion matrix — label2가 사실상 해결됨, label3는 트레이드오프
seed=0 (val balanced acc 85.4%, test acc 91.9%) 기준:
```
        pred0  pred1  pred2  pred3
true0:    93     1      1      0    (n=95)
true1:     0    65      2      0    (n=67)
true2:     0     5    111      1    (n=117)   ← recall 82.9%→94.9%, 양방향 오답 다 줄음
true3:     0     5     10     16    (n=31)    ← recall 61.3%→51.6%, 오히려 하락
```
95% 목표까지 **9.5개만 남음**(8/29 밤엔 21.5개). label2→1(12→5)·label2→3(8→1) **양방향 동시 개선** — occupancy 스무딩 실험(8/29)이 한쪽 고치면 한쪽이 나빠지는 두더지 잡기였던 것과 대조적. 원인 추정: power=1.0이 label1(1.15배)·label3(2.55배)를 과보호하느라 label2(0.66배)가 상대적으로 방치돼 양방향으로 새던 것 — 가중치를 없애니 label2가 정상화됨. 대신 label3는 보호막이 없어져 recall이 내려감(오답 5→1, 10→2로 사실상 전부 label3 자체 오답).

### hidden_size / dropout 스윕(power=0.0 고정, 2시드씩) — 추가 이득 없음, 현재 기본값이 이미 최적
| hidden_size | 정확도 | F1 |
|---|---:|---:|
| 64 | 91.8% | 67.2% |
| **128(현재)** | **91.5%** | **68.6%** |
| 256 | 89.5%(악화) | 67.2% |

| dropout | 정확도 | F1 |
|---|---:|---:|
| 0.1 | 89.5%(악화) | 63.1%(악화) |
| **0.2(현재)** | **91.5%** | **68.6%** |
| 0.3 | 90.3% | 66.7% |
| 0.4 | 90.5% | 64.6%(악화) |

hidden_size=256은 작은 데이터셋에 과적합 추정(악화), dropout은 0.2가 이미 최적. **이 두 축은 더 건드릴 필요 없음** — 이번 세션에서 실질적 레버는 class-weight-power 하나였음.

### 실험 산출물 (전부 스크래치패드, 코드/데이터/체크포인트 변경 없음)
`C:\Users\dkssu\AppData\Local\Temp\claude\C--dev-Hanbat-capstone\826c2c75-c14f-4e8d-b7ad-c172f7aa404b\scratchpad\`의 `class_weight_power_sweep*.py`, `power0_confusion_and_hparam.py` 및 대응 `*_out.txt` — 세션이 끝나면 사라질 수 있으므로, 다음 세션에서 재현하려면 위 표의 정확한 세팅(EXIT_WEIGHTS=(0.3,0.3,0.4), hidden=128, dropout=0.2, epochs=50, batch=32, lr=0.001)으로 `train_ap_early_exit.py --class-weight-power 0.0`을 시드 여러 개로 돌리면 동일 재현 가능(스크립트 자체는 이미 `--class-weight-power` 인자를 지원하므로 스크래치패드 스크립트 없이도 재현 가능).

### occupancy 라벨링 스무딩(어제 이어서 검증 완료, 폐기 확정)
어제 밤 오답 37건 전체 재분해 + window=3 전체 파이프라인 검증까지 마침 — 정확도 순효과 없음(88.1%→88.3%), Label3 F1 오히려 악화(65.5%→57.1%). 상세는 아래 "체크포인트 (2026-08-29, 3차)" 참고. **이 방향은 폐기, class-weight-power가 훨씬 나은 대안으로 확인됨**.

## 체크포인트 (2026-08-29, 3차) — 발표자료로 실제 목표 재확인, confusion matrix 분해, occupancy 스무딩 실험(보류)

### 발표자료(`docs/캡스톤디자인I_최종발표.pptx`) 확인 — SDN 비교는 원래 목표가 아니었음
사용자가 "1학기엔 왜 정확도가 높았나" 질문 → `project/results/hojung/baseline_eval_report.txt` 확인 결과 1학기 데이터(`project/data/real`)는 시나리오명이 `emergency_ramp`/`startup_surge` 등 **시뮬레이터 생성 데이터**였음(label3 recall 100%로 확인) — 실측이 아니라 쉬운 문제였을 뿐. 이 대화가 "그럼 스마트공장이면 jitter/loss를 입력으로 그냥 줘도 되지 않냐"는 질문으로 이어져, Cisco Catalyst 9800처럼 엔터프라이즈급 AP는 실제로 WMM Traffic Stream Metrics로 클라이언트별 latency/loss를 네이티브 제공한다는 것도 웹 검색으로 확인(반면 이 프로젝트가 쓴 GL.iNet Opal 같은 저가 AP는 안 됨 — 그래서 victim 프로브를 직접 만든 것).

이 논의 중 사용자 요청으로 `docs/캡스톤디자인I_최종발표.pptx`(pptx, 텍스트가 대부분 이미지로 붙어있어 PowerPoint COM 자동화로 슬라이드를 PNG로 export해서 육안 확인) 검토 → **중요 발견**:
- **슬라이드 8 "정량적 목표"**: 목표1 = **혼잡 분류 정확도 95% 이상**(Raspberry Pi 환경), 목표2 = **추론 지연 <1ms**. SDN 비교는 목표에 없음. 목표2(<1ms)는 이미 달성(0.6~0.8ms). **목표1(95%)이 진짜 남은 숙제** — 지금 전체 정확도 87~91%.
- **슬라이드 6 "주요 시스템 개요"**: 핵심 주장은 "간섭 감지에 Early Exit LSTM 구조를 최초 적용" — SDN을 이기는 게 목표가 아니라 최초 적용 자체가 기여.
- **슬라이드 7 "프로젝트 최종 목표"**: "혼잡 판단 결과를 기반으로 **채널 전환 필요 여부와 전환 명령 후보를 생성한다**"가 최종 목표 문장에 명시돼 있음 — 이건 지금 `demo_api_spec.md` §9에 "미착수(팀 결정 필요)"로 남겨둔 **밴드 스티어링**과 동일. work-log에선 "선택적 확장"처럼 다뤘지만 **원래 최종 목표에 포함된 항목**이었음 — item 4(밴드 스티어링) 우선순위 재검토 필요.
- 사용자가 "전체 정확도 95%부터"로 우선순위 결정.

### Confusion matrix 분해 — label3보다 label2가 더 큰 걸림돌
현재 배포 체크포인트(7-feature, 균등 가중치) test 310창 confusion matrix:

```
         pred0  pred1  pred2  pred3
true0:     92     2      1      0    (n=95)
true1:      0    65      2      0    (n=67)
true2:      0    12     97      8    (n=117)
true3:      0     6      6     19    (n=31)
```

95% 정확도 = 294.5개 정답 필요, 지금 273개(88.1%) → **21.5개만 더 맞히면 목표 도달**. label0/1은 이미 97% 근처. **label2(117개 중 20개 오답)가 label3(31개 중 12개 오답)보다 오답 개수 자체가 많음** — 지금까지 label3 F1에만 집중했는데 95% 목표엔 label2 개선이 더 효율적.

### label2 오답 원인 분해 — 두 가지 다른 문제
`scaler_params.json`으로 역스케일링해서 raw occupancy 확인:
- **label2→1 (12개 중 9개)**: occupancy가 정확히 **55.2~56.7%에 몰림** — label1/2 anchor 경계(정확히 55%)에 걸친 순수 측정 잡음. 물리적으로 거의 같은 채널 상태인데 라벨만 뒤집힘. **데이터를 더 모아도 안 고쳐지는 유형**.
- **label2→3 (8개)**: occ 63~73%로 넓게 분산, bitrate도 뒤섞임 — 기존에 진단한 "occ 60~72% 정보 부족 구간"과 동일한 유형. 데이터/대리 feature 추가가 부분적으로 도움될 수 있음.

### occupancy 라벨링 스무딩 실험 — 보류(위험 판단)
label2→1 노이즈성 오답을 고치기 위해 `remeasure_redesign.py`에 `channel_occupancy_percent`의 scenario별 rolling median(window=5, `fix_occupancy_outliers`와 동일 스타일) 스무딩을 라벨링 계산에만 적용(모델 입력 feature는 원본 유지)하는 실험을 시도.

- **결과**: raw 2115행 중 **251개(12%) 라벨이 바뀜** — 목표였던 "12개 경계 노이즈"보다 훨씬 큰 변화. label1 -24 / label2 +37 순변화 — step 프로파일(60초마다 실제 부하 전환) 특성상 window=5 스무딩이 순수 잡음뿐 아니라 **진짜 전환 타이밍까지 몇 초 밀어버릴 위험**이 있음. forecasting(조기경보, k=3 3~6초 타이밍에 의존)에도 영향 줄 수 있어 사용자가 "위험해 보임"으로 보류 결정.
- **되돌림**: `remeasure_redesign.py` 코드 변경은 `git checkout`으로 원복. 정식 데이터(`metrics_v2_pi_redesign2_relabeled.csv`)는 스크래치패드에만 출력했으므로 애초에 안 건드림 — 실험 흔적 없음.
- **다음에 재시도한다면**: window=3처럼 더 좁은 창으로 순수 잡음만 줄이는지 재확인 필요. 또는 라벨링이 아니라 **모델 입력 쪽**(occupancy feature 자체에 rolling 평균 추가)으로 접근을 바꿔서, 라벨은 그대로 두고 모델이 노이즈에 더 강해지도록 유도하는 방향도 검토 가치 있음.

### occupancy 라벨링 스무딩 window=3 전체 파이프라인 검증 — 순효과 없음, 방향 폐기 (같은 세션 후속)

먼저 오답 37건 전체를 다시 분해(label3 자체 오답 12건, label0/1 오답 5건 포함)해서 확인: 오답의 89%(33/37)가 exit3(가장 깊은 레이어)에 몰려있고 오답 평균 entropy(0.740)가 정답(0.422)보다 뚜렷이 높음 — 모델의 자기 보정 자체는 잘 됨. label3 오답 12건은 occupancy가 전부 50~63%(심각 앵커 75%에 한참 못 미침) — 프로브 축(jitter/loss)만으로 결정된 정보 부족 케이스 100%. 그리고 label1→2 오답 2건도 정확히 occ=53.3%에서 발생 — 53~57% 경계 노이즈가 label2→1뿐 아니라 **양방향**(label1→2도)이라는 걸 추가 확인.

사용자 요청으로 window=3 스무딩을 전체 파이프라인(relabel→convert→retrain 3시드→평가)까지 실제로 돌려서 검증:
- `project/scripts/metrics_v2_pi_redesign2_relabeled_occsmooth_w3.csv`, `project/data/ap_metrics_v2_redesign2_occsmooth_w3/` 신규 생성(둘 다 보존 — 부정적 결과의 증거로 남김, 기존 파일과 충돌 없음)
- raw 2115행 중 161개(7.6%) 라벨 변경(window=5의 251개보다 적지만 여전히 넓음)
- 3시드 학습(seed2, val balanced acc 83.6% 최고) → confusion matrix 비교:

| | 원본 라벨 | window=3 스무딩 |
|---|---|---|
| 정확도 | 88.1% | 88.3%(거의 그대로) |
| label2→1 오답 | 12 | **6(절반으로 줄음 — 가설 확인)** |
| label2→3 오답 | 8 | **14(오히려 늘어남)** |
| label0 recall | 96.8% | 100%(개선) |
| label1 오답(신규 유출) | 0 | 5(신규 발생) |
| Label3 F1 | 65.5% | **57.1%(악화)** |

**결론**: 가설(55% 경계 노이즈)은 맞았고 그 오답은 실제로 절반 줄었지만, 같은 스무딩된 occupancy 값이 다른 anchor 경계(75% 등)에도 동시에 쓰여서 **다른 곳에서 새 오답이 생겨 순효과가 없음**(전형적인 두더지 잡기) — window 크기를 더 줄여도 이 구조적 트레이드오프는 안 없어질 가능성이 높음. **occupancy 라벨링 스무딩 방향은 폐기**. 코드(`remeasure_redesign.py`)는 실험 후 매번 `git checkout`으로 원복, 실험 산출물만 보존.

### 다음 세션 우선순위 — 다른 방안 검토 필요
occupancy 라벨링 스무딩(입력/출력 양쪽 시도) 폐기 이후 재검토한 대안들(미착수, 우선순위 미정):
1. **모델 입력 쪽 스무딩** — 라벨은 원본 유지, occupancy feature 자체에 rolling 평균만 추가해서 모델이 노이즈에 덜 민감하게. 라벨을 안 건드리므로 forecasting 등 다른 결과에 영향 없음 — 위 실험보다 안전한 변형.
2. **class-weight-power 재검증** — 8/23에 4-feature 시절 데이터로 power=1.0을 확정한 뒤 한 번도 재검증 안 함(6→7-feature, 라벨 재설계 전부 그 이후). 지금 데이터로 0.7~1.0 재스윕 가치 있음.
3. **하이퍼파라미터 튜닝** — hidden_size=128/dropout=0.2/lr=0.001/epochs=50을 세션 내내 고정값으로만 씀, 한 번도 스윕 안 함.
4. **label2→3 구간(occ 60~72%) 대리 feature 추가 탐색** — `iw station dump`의 다른 미사용 필드(MCS 분포, tx 큐 등).
5. **시나리오별 오답 편차 재조사** — step_run1(18.8%)·step_run6(16.4%)·step_run5(13.4%) 오답률이 step_run2(4.2%)보다 훨씬 높음 — 특정 세션에 다른 잡음원이 있었는지 확인 가치.
6. **조기경보(forecasting) 프레이밍** — 이미 검증됨(k=3 61.5%), "점 분류 정확도 95%"와는 다른 지표라 발표자료 목표1을 직접 만족하진 않지만 대안 서사로 유효.

문서 최상단 "내일 할 일 (2026-08-30)" 참고 — 이 체크포인트의 발견을 반영해 그쪽에서 재정리함.

## 체크포인트 (2026-08-29, 2차) — sta_tx_bitrate_mean 7-feature 승격, item 1(SDN 가중치) 정정, 학습 파이프라인에 --seed 추가

### 배경: item 1 재검토 — SDN 가중치 승격이 노이즈였을 가능성 확인
work-log의 "내일 할 일" 2번(`sta_tx_bitrate_min/mean` escalation 지문 재검토)을 진행하는 과정에서, 학습 파이프라인 어디에도 `torch.manual_seed`가 없다는 걸 발견. 같은 설정(6-feature, SDN 가중치)을 시드만 바꿔 재실행하니 test 정확도 91.3%→80.0%, Label3 F1 66.7%→51.1%로 요동침 — 아래 "체크포인트 (2026-08-29, 1차)"에서 SDN 가중치를 승격한 근거(단일 실행 비교)가 노이즈였을 가능성이 높다는 뜻. 사용자 확인 후 **item 1은 일단 보류**(균등 가중치로 유지도 승격 확정도 아님), item 2부터 제대로 마무리하기로 함.

### sta_tx_bitrate_mean 검증 — 다중 시드로 실제 신호 확인
가설(혼잡할수록 station tx bitrate가 떨어진다, "rate collapse")은 실측과 정반대: 혼잡할수록 오히려 값이 올라감(부하 테스트 특성상 혼잡 구간에서 기기가 실제로 데이터를 계속 밀어넣기 때문 — 한가할 때는 관리 프레임만 드문드문 잡혀 저속 기본값으로 기록됨). occ 60~72%(다른 6개 feature가 label 2/3 사이에 완전히 동일해지는, 8/27 진단된 구간)에서 label 2/3 간 Cohen's d=0.52로 유의미하게 갈라짐.

5개 시드 x 4개 설정(6/7-feature x 균등/SDN 가중치) 다중 시드 스윕 결과(Label3 F1, mean±std):

| | 균등 가중치 | SDN 가중치 |
|---|---:|---:|
| 6-feature | 56.0%±3.3% | 52.0%±5.8% |
| 7-feature(+bitrate) | 60.9%±3.0% | 63.1%±3.6% |

**결론이 깔끔하게 갈림**: feature 축(6→7)은 가중치와 무관하게 항상 이기는 확실한 신호. exit 가중치 축(균등→SDN)은 7-feature 안에서도 차이(2.2pt)가 표준편차보다 작아 여전히 노이즈와 구분 안 됨 — item 1의 승격 근거가 이걸로도 재확인 약화됨.

### 학습 파이프라인에 --seed 옵션 추가 (영구 수정)
`train_ap_early_exit.py`/`train_ap_sdn.py`/`train_ap_baseline_lstm.py` 전부에 `--seed` CLI 옵션 추가(`torch.manual_seed`+`np.random.seed`, 체크포인트 메타데이터에도 기록). 기본값은 `None`(기존 동작 유지)이지만, 앞으로 하이퍼파라미터 A/B 비교는 여러 시드로 해야 한다는 경고를 `--seed` help 문자열에 남김.

### 전체 파이프라인 승격 — Baseline/SDN-style/Proposed 전부 7-feature로 재작업
사용자 확인("SDN에는 왜 넣어?"라는 질문에 "아키텍처 비교의 통제 변수로서 입력 feature도 맞춰야 공정하다"고 답변, 이해 확인 후 진행). 순서:

1. `ap_features.py`: `AP_FEATURE_COLUMNS`에 `sta_tx_bitrate_mean` 추가(6→7). 승격 배경을 주석에 기록, 기존 "정보용" 코멘트는 "(archived)"로 보존.
2. **아카이빙**(전부 보존, 덮어쓰지 않음): `data/ap_metrics_v2_redesign2/` → `data/ap_metrics_v2_redesign2_6feat_archived_20260829/`(전체 복사), `checkpoints/ap_v2_redesign2/` → `checkpoints/ap_v2_redesign2_6feat_archived_20260829/`(전체 복사, item 1의 중첩 archived 폴더 포함), 결과 리포트 6종 → `*_6feat_archived_20260829.*`, Pi(`~/ap_pi_v2/`) → `archived_6feat_20260829/`.
3. `prepare_ap_metrics_dataset.py`로 `ap_metrics_v2_redesign2` 재변환(7-feature) — split은 이전과 완전히 동일(train 1437/val 308/test 310, test label3=31 그대로, seed=42 결정론적 확인됨).
4. Baseline·SDN·Proposed(EE, **균등 가중치로 되돌림** — 위 item1 재검토 참고) 각각 시드 0/1/2로 재학습, val balanced acc 최고를 배포 체크포인트로 선택:
   - Baseline: seed1 선택(val 86.2%)
   - SDN: seed2 선택(val 85.5%)
   - Proposed(EE, 균등 가중치): seed2 선택(val 86.3%)
   - **주의**: SDN과 EE에 우연히 같은 가중치(0.15/0.30/0.55)를 썼던 중간 시도에서 두 모델이 완전히 동일한 학습 곡선을 보임 — `SDNLSTM`은 `EarlyExitLSTM`과 백본이 의도적으로 동일하게 설계돼 있어서, 같은 loss 가중치+같은 시드를 쓰면 사실상 같은 네트워크가 됨(추론 시 엔트로피 vs confidence 임계값 정책만 다름). 그래서 최종적으로 Proposed는 균등 가중치를 씀 — item 1 이슈와 별개로, 이 조합 자체가 비교를 무의미하게 만들기 때문.
5. ONNX 재수출: `export_onnx_ap_baseline.py`, `export_onnx_ap_sdn.py`+`export_onnx_ap_sdn_unified_int8.py`, `export_onnx_ap.py`+`export_onnx_ap_unified.py`+`export_onnx_ap_unified_int8_v2.py`. **버그 발견+수정**: `export_onnx_ap_unified_int8_v2.py`와 `export_onnx_ap_sdn_unified_int8.py` 둘 다 재조립 시 그래프 입력 shape에 feature 개수를 `[1, 10, 6]`으로 하드코딩하고 있었음(7-feature로 바꿔도 옛 6 그대로 남아있어서 Pi에서 "Got 7 Expected 6" 에러로 발각) — `AP_FEATURE_COLUMNS` 기반 `INPUT_SIZE`로 고쳐서 재실행.
6. 검증: unified fp32 PyTorch 대비 fixed/dynamic 각각 310/310 100% 일치. INT8 v2는 fixed 306/310·dynamic 305/310 일치(양자화 노이즈 수준), INT8 직접 정확도 88.1%(fixed/dynamic 동일), Label3 F1 64.4%.
7. Pi 배포 스크립트(`bench_baseline.py`, `bench_unified.py`)도 feature 리스트가 6개로 하드코딩돼 있어서 같은 문제 재발 — 7개로 수정 후 재배포.
8. Pi 재측정(같은 세션, INT8):

   | 모델 | Pi INT8(ms) | 정확도 | Label3 recall | Label3 precision | Label3 F1 |
   |---|---:|---:|---:|---:|---:|
   | Baseline | 0.756 | 88.7% | 58.1% | 69.2% | 63.2% |
   | **SDN-style** | **0.636** | 89.0% | 58.1% | **81.8%** | **67.9%** |
   | Proposed Fixed(균등) | 0.641 | 88.1% | **61.3%** | 70.4% | 65.5% |
   | Proposed Dynamic(균등) | 0.645 | 87.7% | **61.3%** | 67.9% | 64.4% |

**정직한 결론**: item 1 때와 달리 이번 라운드는 "Proposed가 전면 우위"라고 말할 수 없다 — SDN-style이 속도와 F1에서 근소 우위, Proposed는 recall이 근소 우위. 세 모델 다 3시드 중 val 최고를 뽑은 단일 배포 체크포인트 비교라, SDN/Baseline도 Proposed처럼 5시드로 완전히 특성화하면 이 근소한 차이가 노이즈인지 실제인지 더 명확해질 것(향후 과제). **7-feature 승격 자체(다중 시드 평균 기준)는 확실한 이득**이었다는 게 이번 세션의 핵심 성과.

문서 갱신: `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`(전면 재작성), `ap_model_comparison_redesign2.txt/.csv`(재생성 + 하드코딩 문자열도 스크립트 소스에서 갱신), `docs/yongsang/onnx_early_exit_redesign.md`(2차 정정 콜아웃 추가), `project/utils/ap_features.py`(주석).

아티팩트("Early-Exit Reweighting")도 정정 배너 추가해 재게시 완료 — https://claude.ai/code/artifact/4e32bd34-9e28-4d43-9c1a-80ba2bb06ed1

## 체크포인트 (2026-08-29, 1차) — EE exit-loss 가중치를 SDN 스타일로 바꿔 재학습 → 가설 검증됨, 새 최고 F1 (아래 "2차"에서 부분 정정됨 — SDN 가중치 승격은 보류, 7-feature 승격은 별개로 확정)

8/28 밤 "왜 SDN이 이겼나" 재조사에서 나온 미검증 가설(SDN의 뒤쪽 exit 강조 loss 가중치 0.15/0.30/0.55가 exit3 심각 탐지에 유리했을 것) 검증. `train_ap_early_exit.py`에 `--exit-loss-weights W1 W2 W3` CLI 옵션 추가(기존 하드코딩 0.3/0.3/0.4 → 인자화, `models/early_exit_lstm.py`의 `multi_exit_loss` weights 파라미터 그대로 사용, 코드 변경 없음). `ap_metrics_v2_redesign2` 데이터로 `--exit-loss-weights 0.15 0.30 0.55`, `--class-weight-power 1.0`(기존과 동일, 통제 변수 유지)로 재학습 → `project/checkpoints/ap_v2_redesign2_sdnw/`(기존 `ap_v2_redesign2/` 체크포인트는 보존, 덮어쓰지 않음).

**결과 — 가설 확인, 게다가 SDN 아키텍처 자체보다도 좋음**:

| 모델 | 정확도 | Label3 recall | Label3 precision | Label3 F1 |
|---|---:|---:|---:|---:|
| EE 균등 가중치(0.3/0.3/0.4) Fixed | 88.4% | 51.6% | 66.7% | 58.2% |
| EE 균등 가중치 Dynamic | 89.0% | 51.6% | 76.2% | 61.5% |
| SDN-style 아키텍처(별도 백본) | 86.8% | 64.5% | 46.5% | 54.1% |
| **EE + SDN 가중치(0.15/0.30/0.55) Fixed** | **91.3%** | **58.1%** | **78.3%** | **66.7%** |
| EE + SDN 가중치 Dynamic | 90.6% | 58.1% | 75.0% | 65.5% |

- val balanced acc도 87.0%(기존 85.4%보다 개선)
- true label-3(31창) exit별 분해: 균등 가중치는 exit3까지 넘어가면 정답률 6/21(29%)였는데, SDN 가중치는 exit3 도달률 자체가 줄고(21→14/31) exit2에서 더 많이(13/16, 81%) 정답을 맞힘 — 가설대로 뒤쪽 exit 학습이 강화됨
- **결론**: "SDN이 이겼다"는 재조사에서 이미 정정됐었지만(F1 근소열세), 이번 재학습으로 EE가 SDN의 장점(exit3 심각 탐지력)만 가중치 조정으로 흡수해 **정확도·recall·precision·F1 전부 SDN-style 아키텍처를 앞섬** — 별도 아키텍처(SDN) 없이도 손실 가중치만 바꿔서 달성. 아키텍처 변경 없이 얻은 개선이라 배포 리스크도 낮음
- 파일: `project/scripts/train_ap_early_exit.py`(CLI 옵션 추가), `project/checkpoints/ap_v2_redesign2_sdnw/`, `project/results/yongsang/ap_v2_redesign2_sdnw_eval_report.txt`

### 새 기본값으로 승격 완료 (같은 세션, 후속) — 사용자 확인 후 진행

사용자 확인("1번으로 하지만 왜 이런 식으로 모델 변경을 했는지는 html 아티팩트로 남겨둬야 돼")에 따라 SDN 가중치 체크포인트를 `ap_v2_redesign2`의 기본 EE 체크포인트로 승격. **기존 균등 가중치 체크포인트/ONNX/결과는 전부 보존**(덮어쓰지 않음, 아래 참고).

- **아카이빙**: 옛 EE `.pth`/`.onnx`(staged, unified, int8 v2 전부, 23개 파일) → `project/checkpoints/ap_v2_redesign2/archived_uniform_ee_weights_20260829/`. 옛 eval report/비교표/Pi latency 문서 → `*_uniform_weights_archived_20260829.*`. Pi(`~/ap_pi_v2/`) 쪽도 동일하게 `archived_uniform_ee_weights_20260829/`에 보존.
- **교체**: `ap_v2_redesign2_sdnw/`의 `.pth` 3개를 `ap_v2_redesign2/`로 복사(파일명 그대로 유지) → `evaluate_ap_early_exit.py`로 canonical eval report 재생성(91.3%/90.6%, Label3 58.1% 확인).
- **ONNX 재생성**: `export_onnx_ap.py`(staged) → `export_onnx_ap_unified.py`(fp32 unified, PyTorch 대비 310/310 100% 일치 검증) → `export_onnx_ap_unified_int8_v2.py`(INT8 v2, PyTorch 대비 fixed 305/310·dynamic 304/310 일치 — 옛 체크포인트의 309/310보다 미스매치가 조금 늘었지만 int8 자체 정확도는 fp32 대비 ~1pp 하락 수준, 여전히 전부 SDN-style 아키텍처보다 우위).
- **Pi 재실측**: 같은 세션에서 Baseline/SDN-style/Proposed(신규) 4개 모델 전부 재측정(모델 안 바뀐 Baseline/SDN도 같은 세션 기준으로 재확인):

  | 모델 | Pi INT8(ms) | 정확도 | Label3 F1 |
  |---|---:|---:|---:|
  | Baseline(EE 없음) | 0.747 | 89.4% | - |
  | SDN-style | 0.615 | 86.8% | 54.1% |
  | **Proposed Fixed θ(신규)** | **0.595** | **91.3%(fp32 eval)** | **66.7%(fp32)/65.5%(int8)** |
  | **Proposed Dynamic θ(신규)** | **0.591** | **90.6%(fp32 eval)** | **65.5%(fp32)/63.2%(int8)** |

  **Proposed가 이제 속도까지 Baseline/SDN-style을 앞섬**(이전 체크포인트는 0.641/0.679ms로 SDN 0.600ms보다 느렸었음) — exit-loss 가중치만 바꿨는데 정확도·F1·속도 전부 개선된 것은 예상 밖의 보너스(속도 개선은 exit1/2로 더 일찍 빠지는 분포 변화 때문으로 추정, 인과 확정은 안 함).
- **문서 갱신**: `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`(신규 수치로 갱신, 옛 파일 archived), `project/results/yongsang/ap_model_comparison_redesign2.txt/.csv`(`generate_ap_comparison.py` 재실행 + 하드코딩된 Pi 수치 문자열도 스크립트 소스에서 갱신), `docs/yongsang/onnx_early_exit_redesign.md`(상단에 "2026-08-29 업데이트" 콜아웃 추가, 본문 수치는 그 시점 기록으로 보존 — 방법론적 결론은 그대로 유효함을 명시).
- **HTML 아티팩트**: 이 모델 변경의 배경(가설)·실험·결과를 정리해 게시(사용자 요청) — "Early-Exit Reweighting", https://claude.ai/code/artifact/4e32bd34-9e28-4d43-9c1a-80ba2bb06ed1

## 내일 할 일 (2026-08-30 아침 기준, 8/29 밤 3차 체크포인트에서 작성) — item 1은 8/30 낮 4차 체크포인트에서 진행됨, 최상단 "⭐ 다음 세션 시작 지점" 참고

우선순위순. 상세 배경은 각 항목 옆 참고 섹션에. **8/29 밤 발표자료 확인으로 우선순위가 바뀜** — SDN 비교나 밴드 스티어링을 "선택적 확장"으로 다루던 이전 판단이 정정됨, "체크포인트 (2026-08-29, 3차)" 참고.

1. ~~전체 정확도 95% 달성~~ — **진행 중, 최상단 "⭐ 다음 세션 시작 지점"과 "체크포인트 (2026-08-30, 4차)" 참고**. class-weight-power 재검증으로 87.0%→91.9%까지 개선(아직 승격 전, 코드/체크포인트 미변경). ~~occupancy 라벨링 스무딩(window=5·3)~~은 순효과 없어 폐기 확정. hidden_size/dropout 스윕도 완료, 추가 이득 없음.
2. **밴드 스티어링(채널 전환 후보 생성)** — 발표자료 슬라이드7 "프로젝트 최종 목표"에 "채널 전환 필요 여부와 전환 명령 후보를 생성한다"가 명시돼 있음을 확인 — **선택적 확장이 아니라 원래 최종 목표에 포함된 항목**이었음. `demo_api_spec.md` §9에 설계는 이미 있음(미착수). 착수 여부·시점 판단 필요
3. **데모 대시보드 구현 착수** — 스펙은 최신화 완료(`docs/yongsang/demo_api_spec.md`, 7-feature 기준). 백엔드+대시보드+에이전트+파이 서버 4개 컴포넌트 신규 구현 필요
4. item 1(EE exit-loss 가중치, 균등 vs SDN) — 여전히 미해결. 균등 유지 중이나 "균등이 낫다"도 증명 안 됨. 필요하면 SDN-style·Baseline까지 5시드로 완전히 특성화해서 재검토(지금은 Proposed만 5시드 특성화함)
5. (급하지 않음) `ramp_load.sh`에 `termux-wake-lock` 추가

<details>
<summary>이전 버전 (8/29 낮, 2차 체크포인트 직후 — 3차에서 위와 같이 재정리됨)</summary>

1. ~~EE의 exit별 loss 가중치를 SDN 스타일로 바꿔 재학습~~ — **1차는 노이즈로 판명, 정정됨**(위 "체크포인트 (2026-08-29, 2차)" 참고). 다중 시드로 재검증한 결과 SDN 가중치가 균등보다 낫다는 근거 없음 — **여전히 미해결**: 균등 가중치를 유지 중이나 "균등이 SDN보다 낫다"는 것도 증명 안 됨. 필요하면 SDN-style·Baseline까지 5시드로 완전히 특성화해서 재검토
2. ~~`sta_tx_bitrate_min/mean`이 escalation 지문을 보이는지 재검토~~ — **완료, 7번째 feature로 승격**(위 "체크포인트 (2026-08-29, 2차)" 참고). Baseline/SDN/Proposed 전부 7-feature로 재학습·ONNX 재수출·Pi 재측정까지 완료
3. ~~데모 대시보드 착수 여부 결정~~ — **스펙만 최신화, 구현은 보류**(사용자 판단). `docs/yongsang/demo_api_spec.md`가 8/27 작성 당시의 9-feature 가중합 스키마 그대로였음(그 사이 6→7-feature, 가중합→max/anchor 방식으로 두 번 바뀜) — FeatureVector·SubScores·congestion_formula·`/meta` 예시를 전부 현재(`ap_metrics_v2_redesign2`, 7-feature) 기준으로 갱신. "신규 구현 부담은 ONNX export 하나"라는 전제도 이제 틀림(그 작업은 이미 끝났고 남은 건 백엔드+대시보드+에이전트+파이 서버 4개 컴포넌트 자체, 다음 세션 착수 대상)
4. (팀 결정 필요) **밴드 스티어링 시스템으로 주제 확장할지** — 상세는 "향후 시스템 구상" 섹션(8/27 논의). 이번 세션 성과(occupancy 문턱 대비 우위, 조기경보, Pi 지연 -67%)로 방어력은 충분히 쌓였으니, 확장이냐 지금 라인 심화냐 판단할 시점
5. (급하지 않음) `ramp_load.sh`에 `termux-wake-lock` 추가 — 이번엔 폰 배터리 최적화 제외로 화면 꺼짐 문제 자체는 우회됨

</details>

## 세션 마무리 요약 (2026-08-28 밤, yongsang)

하루 종일 이어진 긴 세션. 순서대로: **① 본수집** — 191/S26 폰 SSH 원격제어 구축 → `ramp_load_remote.sh` 실기기 검증 → step 프로파일 반복 수집(AP 크래시 2회·프로브 서버 누락 사고 극복) → `metrics_v2_pi_redesign2.csv` 2115행, **label 3 202개** 목표 달성. **② 재학습/평가** — `ap_metrics_v2_redesign2` 재변환·재학습(val balanced acc 85.4%), occupancy 문턱 대비 우위 재확인(F1 58.2% vs 44.0%), forecasting 조기경보 재현(k=3 61.5%/k=5 45.5%, 표본 커져도 안정). **③ ONNX/Pi 배포 재설계** — staged(세션 3개)가 baseline보다 느린 문제 발견 → 단일 그래프(If 노드)로 재설계해 baseline 대비 -40% → INT8 순진하게 적용하면 이득 없음(오판) → 사용자가 1학기 자료로 재확인 질문 → 원인 재진단(양자화 도구가 If 서브그래프의 LSTM을 건너뜀) → staged로 먼저 양자화 후 재조립하는 방식으로 **최종 baseline 대비 -67%**(0.64~0.68ms) 확보. **④ Baseline/SDN-style 비교** — 이 브랜치에 처음 구현, 공정 비교(class-weight 통일) 후 Pi 배포까지 — SDN이 속도·recall 우위처럼 보였으나 precision까지 보면 F1은 막상막하(EE 근소 우위)임을 재조사로 확인.

**핵심 교훈**: 이번 세션 세 번의 "성급한 결론"(ONNX int8 1차, SDN이 이겼다는 1차 판단, docs 최신화 누락)이 전부 사용자의 재확인 질문으로 바로잡혔음 — 결과를 정직하게 기록하고 원인을 끝까지 추적하는 패턴이 잘 작동했다.

## 체크포인트 (2026-08-28 밤) — 본수집 완료, label 3 202개

191/S26 폰 SSH 원격제어 셋업 완성 → `ramp_load_remote.sh` 실기기 검증 → 초기 두 런은 노트북 iperf3 프로브 서버(5203) 누락으로 무효(archived) → 5203 기동 후 재시작, step 프로파일 10회 반복(AP 크래시 2회·폰 개별 죽음 여러 번 겪으며 진행) 끝에 label 3 202개 확보하고 마무리. 상세는 바로 아래 "본수집 완료" 섹션과 "다음 세션 최우선" 참고.

## 체크포인트 (2026-08-28 오후, 2회차) — congestion_label_criteria.html을 .md 수정에 맞춰 동기화

`docs/yongsang/congestion_label_criteria.md`에 반영했던 3건의 "(archived — 2026-08-28 확인)" 주석(§03 weighted-sum 공식이 더 이상 코드에 없음 / retry가 congestion_score에서 완전히 빠짐 / label 3 표본 수치가 2026-08-23 스냅샷)을 `.html` 쌍둥이 파일에도 동일하게 반영 — 헤더 아래 요약 callout 1개, §03 lede 한 줄, retry 정규화 callout 옆에 archived callout 1개, footer 한 줄 추가. 기존 `.callout` 스타일 재사용(새 디자인 시스템 도입 안 함). 게시된 아티팩트("혼잡 라벨 분류 기준", 2026-08-23 이후 미갱신 상태였음) 재게시는 자동 승인 분류기에 막혀 **보류** — 사용자 승인 필요.

## 체크포인트 (2026-08-28 오후, 1회차) — 상태 검증 + 문서 오류 수정

### 상태 검증 (와이파이 일시 끊김 → 재연결 후)
아티팩트 실시간 watch 연결이 끊겼다가(재연결 시도 소진) 와이파이 복구 후 확인 요청 받음. 점검 결과 **끊김으로 인한 데이터 손실 없음**:
- `.work-log/current.md`, `~/.ssh/config`(s21/s26 항목), `project/scripts/ramp_load.sh`, `ramp_load_remote.sh` 전부 로컬 파일이라 무관하게 그대로 있었음
- 아티팩트(폰 SSH 런북, AP 혼잡 탐지 콘솔) 둘 다 내용 그대로 정상 게시 상태 확인 (끊김은 실시간 알림 채널에만 영향, 콘텐츠 자체는 서버에 이미 반영돼 있었음)
- 두 아티팩트 watch 재설정 완료

### `docs/yongsang/congestion_label_criteria.md` 오류 3건 발견 → 수정 완료
사용자 요청으로 감사(audit) 진행, 코드(`collect_metrics.py`)와 대조해서 확인:
1. **존재하지 않는 코드 참조**: 문서가 "sub-score는 `calculate_scores()`에서 계산"이라고 했지만, 그 함수는 이미 재설계(max/anchor 방식)로 완전히 다시 쓰였음. `RETRY_FAILED_MAX_PER_SEC`·`JITTER_MAX_MS` 상수는 코드에서 **아예 삭제됨**(grep으로 확인, 검색 결과 0건) — "superseded" 배너만으로는 부족했고 구체적 참조가 틀려 있었음.
2. **retry 축 변화 축소 서술**: 원문은 "정규화 방식만 바뀜"으로 읽혔지만 실제로는 재설계에서 retry를 congestion_score(라벨 축)에서 **완전히 제외**함(지금은 `tx_retry_ratio`로 모델 입력 feature용으로만 유지, `max()`엔 안 들어감).
3. **날짜 없는 스냅샷 수치**: "label 3 표본 test 5개"가 2026-08-23 시점 값인데 날짜 표시가 없어 현재 상태처럼 보임 — 실제 최신(재설계 기준)은 test label 3 38개.
→ 세 곳 모두 "(archived — 2026-08-28 확인)" 인라인 주석으로 수정, 최신 수치가 있는 문서(`congestion_label_redesign.md`, `.work-log/current.md`, `ap_v2_redesign_threshold_comparison.txt`)로 포인터 추가. 원본 수치·서술은 "그 시점 기록"으로 보존(삭제 안 함).

### 다음 세션 최우선 (그대로 유지, 위 "진행 중" 섹션 참고)
191폰 SSH 셋업 → 두 폰 IP 재확인 → 연결 검증 → ramp_load 실기기 테스트 → 본수집. 상세는 바로 아래 "진행 중 (2026-08-28 오후 — 램프형 부하 스크립트 + 폰 SSH 원격제어 셋업)" 섹션.

## 진행 중 (2026-08-28 오후 — 램프형 부하 스크립트 + 폰 SSH 원격제어 셋업)

### 배경
어젯밤 계획(위 "내일 할 일" 참고)대로 escalation 창 확보용 램프형 부하 수집을 준비하다가, "폰 원격제어를 어차피 나중에(데모용) 해야 하니 지금 하자"는 사용자 결정으로 SSH 기반 원격제어를 먼저 구축하는 쪽으로 방향 전환. `docs/yongsang/demo_api_spec.md` §4.B에 이미 설계돼 있던 "SSH exec (폰 쪽 코드 없음)" 컨벤션(`Host s21`=191/5201, `Host s26`=S26/5202, Termux sshd 8022)을 그대로 구현.

### 신규 스크립트
- **`project/scripts/ramp_load.sh`**: 폰(Termux) 로컬 실행용. `profile step`(계단식 10M→20M→30M→40M, 60초씩, 총 240초) / `profile knee`(무릎근처 22M 고정 240초) 두 프로파일. Ctrl-C 시 trap으로 현재 iperf3만 즉시 정리. **실기기 미검증**(문법 미확인 수준, 아직 한 번도 실행 안 해봄).
- **`project/scripts/ramp_load_remote.sh`**: 노트북에서 두 폰에 `ramp_load.sh`를 scp로 배포하고 SSH로 동시 실행/동시 정지(Ctrl-C 하나로 양쪽 다 pkill)하는 오케스트레이터. `ramp_load.sh`를 대체하지 않고 그 위에서 원격 실행만 대신함.

### 노트북 쪽 SSH 셋업 (완료)
- 폰 전용 키페어 `~/.ssh/id_ed25519_phones` 생성(패스프레이즈 없음, AP 전용 `id_rsa_ap`와 분리)
- `~/.ssh/config`에 `Host s21`(HostName 192.168.8.191, Port 8022) / `Host s26`(HostName 192.168.8.103, Port 8022) 추가. **레포 밖 개인 설정이라 git에는 안 잡힘** — 새 환경에서 이어가려면 이 섹션 참고해서 재생성 필요
- 공개키(`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINM51EQYNdMlGZFCDpSI2+6R9YZWL0b61+zwIBeSfvAS phone-control@ap-testbed`)는 각 폰 `~/.ssh/authorized_keys`에 등록 필요

### 폰 쪽 셋업 — 191/S26 둘 다 완료
- **S26**: `pkg install openssh` → `sshd` 기동 → `mkdir ~/.ssh && chmod 700` → 공개키 `authorized_keys`에 등록·`chmod 600` → `whoami` = `u0_a579` (노트북 config에 반영 완료)
- S26 IP 확인은 **보류**: 세팅 시점에 폰이 AP 와이파이가 아니라 LTE 데이터 상태였음. Termux `ip addr show wlan0`은 최신 안드로이드의 netlink 권한 제약으로 `Permission denied` (Wi-Fi 연결 상태와 무관한 Termux 자체 제약). AP 와이파이에 실제로 붙었을 때 안드로이드 설정 UI(Wi-Fi → 네트워크 상세)로 확인하는 게 더 간단 — 그때 `192.168.8.103`과 다르면 config 갱신 필요
- **191**: 같은 순서로 완료. `whoami` = `u0_a29` (노트북 config `Host s21` User 반영 완료). `ssh s21 echo ok` 검증 성공(`192.168.8.191:8022`, known_hosts 등록됨) — **191 SSH 연결 확인 끝**

`ssh s21 echo ok` / `ssh s26 echo ok` **둘 다 검증 완료** — 노트북→폰 SSH 원격제어 셋업 끝.

### `ramp_load_remote.sh` 실기기 테스트 (knee 프로파일, timeout 65s) — 성공, S26에 iperf3 사전 설치 필요했음
- 노트북 iperf3 서버 5201/5202 기동 → 두 폰 ping 확인(0% 손실) → `timeout -s INT 65 bash ramp_load_remote.sh knee` 실행
- **1차 시도 실패**: S26 Termux에 iperf3 자체가 설치 안 되어 있어서 `iperf3: command not found`로 즉시 중단. `ssh s26 "pkg install -y iperf3"`로 설치(3.21, 83KB) 후 해결
- **2차 시도 성공**: 두 폰 다 정상 동작, 22M 부하로 ~68~69초(timeout 65s + 정리 오버헤드) 실행 후 SIGINT로 정상 종료. 원격 배포(`scp`)·동시 실행·Ctrl-C 시 양쪽 `pkill` 정리까지 스크립트 설계대로 작동 확인
- **관찰**: 22M×2(합계 44M)에서 loss가 시간이 갈수록 커짐 — 초반 10~20초는 1~10%대, 후반엔 191 평균 ~35%(피크 71%) / S26 평균 ~35~37%(피크 79%). knee 프로파일이 실제로 채널을 압박하고 있다는 신호(escalation 창 확보 목적엔 긍정적)
- 테스트 후 AP 정상(ping 0% 손실, 1~15ms) — 크래시 없음. 두 폰 다 잔여 iperf3/ramp_load 프로세스 없이 정리됨(직접 확인). 노트북 iperf3 서버도 테스트 직후 종료함(`taskkill`)
- 스크립트 출력 관련 사소한 이슈: `sed "s/^/[호스트] /"` 프리픽스가 SSH 세션 버퍼링 때문에 실시간으로 안 뜨고 몰려서 나오는 경우가 있음(기능엔 문제없음, 로그 가독성만 영향)

### 본수집 1차 세션 시작 (`metrics_v2_pi_redesign2.csv`, step 프로파일)
파이(`capstone@192.168.8.109`, 사용자가 직접 SSH 접속)에서 `python3 collect_metrics.py step_run1` 기동 확인 후, 노트북에서 iperf3 서버(5201/5202) 띄우고 `ramp_load_remote.sh step` 실행.

- **1차 시도(19:33)**: 191(s21)은 4단계(10/20/30/40M×60s) 전부 클린하게 완주. **S26(s26)은 10M 스텝 중 14초쯤부터 데이터가 0바이트로 끊김**(56~76초 잠깐 재개 후 재차 끊김, TCP/UDP 세션 자체는 안 죽음) — 원격 SSH 실행 중 폰 화면이 꺼지면서 백그라운드 스로틀링된 것으로 진단(사용자 확인: "화면이 한번 꺼지긴 했었어"). 근본 원인: `ramp_load.sh`에 `termux-wake-lock` 등 절전 방지 코드가 아예 없었음(코드 확인 완료, 아직 미수정 — 대신 폰 쪽 설정으로 우회)
- **조치**: 사용자가 191/S26 둘 다 Termux 배터리 최적화 제외 + 화면 유지 설정 후 재시도
- **2차 시도(19:39~19:43, 성공)**: 두 폰 다 240초 전 구간 클린하게 완주, 중단 없음. **로드가 오르면서 loss가 뚜렷하게 escalation**: 10M 단계 0% → 20M 단계 61~70% → 30M 단계 93~94% → 40M 단계 96~97%(양쪽 폰 비슷한 패턴). AP는 종료 후 정상(ping 0% 손실, 1~9ms) — 크래시 없음
- **의의**: 이게 정확히 원했던 escalation 창 패턴(occupancy/loss가 부하에 따라 단계적으로 오름) — `metrics_v2_pi_redesign2.csv`의 `step_run1` 시나리오에 label 0→1→2→3 전이가 담겼을 가능성 높음. 파이 콘솔에서 라벨 확인 필요(아직 미확인 — 파이는 사용자가 직접 보고 있음)

### 치명적 실수 발견 + 정정: 노트북 iperf3 프로브 서버(5203) 안 켜서 두 런(904~994행) 전부 프로브 축 없이 라벨링됨
2차 step 런 후 사용자가 파이 콘솔의 "프로브 실행/실패 : 35 / 35"(100% 실패)를 캡처해서 질문 → 원인 추적.

- **원인**: `collect_metrics.py`의 victim 프로브는 노트북 `192.168.8.226:5203`으로 붙는데(`PROBE_TARGET`/`PROBE_PORT`), Claude가 iperf3 서버를 **5201/5202만 띄우고 5203을 빼먹음**. 이번 세션 전체(1차+2차 step 런) 동안 프로브가 단 한 번도 성공 못 함
- **영향**: `probe_ever_ok`가 계속 False라 `probe_hard_fail` 오버라이드(loss=1.0)가 발동 안 하고, `anchor_score(None,...)`가 0.0을 반환 — 즉 `congestion_score = max(occupancy, jitter=0, loss=0, latency)`로, **이 재설계의 핵심(occupancy 문턱은 못 잡는데 jitter/loss로 잡히는 심각)이 이번 세션 데이터엔 전혀 반영 안 됨**. 라벨이 틀렸다기보다 과소평가 방향(프로브가 있었으면 더 심각으로 잡혔을 케이스를 못 잡음). 이미 지난 시간대라 재계산으로 못 살림(그 순간 실제 jitter/loss를 측정 못 했으므로)
- **조치**: 5203 서버 기동 확인(`netstat`으로 LISTENING 확인). 기존 `metrics_v2_pi_redesign2.csv`(994행)는 `metrics_v2_pi_redesign2_probefail_archived_20260828_1958.csv`로 백업(삭제 안 함) 후, 원본 파일은 헤더만 남기고 초기화. 사용자 확인: "지금까지 쌓인건 지우자"
- **교훈**: 앞으로 iperf3 서버 기동 시 반드시 **5201(191 부하) / 5202(S26 부하) / 5203(victim 프로브)** 세 개 다 확인할 것. 콘솔의 "프로브 실행/실패" 카운터를 런 시작 직후 한 번 확인하는 습관 필요(0/0이면 아직 프로브 스레드가 못 돌았다는 뜻이라 이상 없음, 실패율이 100%면 서버 미기동 의심)

### 4차 step 런 도중 AP 크래시(자가 재부팅) — 3차까지 프로브 정상 데이터 254행 확보한 상태
3차 런 성공 후(프로브 정상, 254행: label 0:43/1:88/2:90/**3:33**) 4차 런 시작. 약 150초 지점(20M~30M 단계 근처)에서 사용자가 "크래시 났어" 보고, 노트북 와이파이 자체도 순간 끊김(`ping 192.168.8.226` 대상 못 찾음) 관측.

- **확인 결과**: `ssh 192.168.8.1 uptime` → **7분**(load average 1.47) — AP가 실제로 크래시 후 자가 재부팅한 것으로 확인. 두 폰은 ping 정상 복귀(0% 손실), `ramp_load_remote.sh` 백그라운드 태스크는 SSH 세션이 끊긴 채 멈춰있어서 `TaskStop`으로 강제 종료
- **폰 잔여 프로세스 정리**: `pkill -f iperf3` 등을 ssh로 보낼 때 **명령어 문자열 자체에 "iperf3"가 포함**돼서 pkill이 자기 자신(그 SSH 세션의 셸)까지 죽여버리는 부작용 발견 — 확인 메시지 없이 연결이 끊겨서 처음엔 실패로 보였지만, 재확인(`pgrep`)해보니 실제로는 정리가 잘 됐음. `ramp_load_remote.sh`의 cleanup trap에도 같은 패턴이 있으나 결과적으로 문제없이 동작(참고 기록만)
- **이번 세션 프로브-정상 데이터 요약**: `metrics_v2_pi_redesign2.csv` 254행(3차까지) — label 0:43/1:88/2:90/**3:33**. 목표(label 3 최소 200개)까지 아직 한참 남음
- **다음 재개 시 주의**: AP가 막 재부팅됐으니 충분히 쿨다운(수 분) 후 재시도 권장. 4번째 런처럼 30M/40M 단계에서 크래시 위험이 있다는 기존 안전 상한(45M 금지, 300~420s 상한)이 이번에도 재확인됨 — step 프로파일 자체가 40M까지 올라가는 구조라 크래시 위험을 안고 가는 것, 필요하면 knee(고정 22M)로 바꿔서 안정성 우선하는 것도 고려

### 본수집 완료 — label 3 목표(200개) 달성, `metrics_v2_pi_redesign2.csv` 2115행
프로브 정상화(위 참고) 이후 step 프로파일을 총 6회(`step_run1`~`step_run6`, 사용자가 재시작할 때마다 이름이 늘어남 — 시나리오명은 모델 입력에서 제외되니 무해) 반복. AP 크래시(자가 재부팅) 2회, 191/S26 개별 폰 죽음 3~4회 겪었지만 그때마다 AP uptime으로 실제 크래시 여부 확인 후 정리하고 재개하는 패턴으로 진행.

- **최종 라벨 분포**: 0:689 / 1:449 / 2:775 / **3:202** (총 2115행)
- **시나리오별**: step_run1 275 / step_run2 345 / step_run3 151 / step_run4 455 / step_run5 599 / step_run6 290
- **세션 중 확인된 크래시 패턴**: step 프로파일이 30~40M 단계로 올라갈 때 AP가 크래시하는 경우가 반복 관측(uptime 리셋으로 확인, 2회) — 기존 안전 상한(45M 금지, 300~420s)과 별개로 **step 프로파일 자체가 크래시를 유발하는 빈도가 꽤 높다는 것**이 이번 세션에서 재확인됨. 그럼에도 AP는 매번 자가복구했고 파이 유선 관리채널 덕에 데이터 손실은 없었음(폴링이 크래시 직전까지 계속 잡힘 — 오히려 label 3이 크래시 직전 구간에 몰려서 나옴)
- **정리 완료**: 파이 수집기 종료(대화 중 자연 종료됨, 확인함), 노트북 iperf3 서버 3개(5201/5202/5203) 전부 kill, 두 폰 잔여 프로세스 없음

### 재라벨링 → 재변환 → 재학습 → 평가 → occupancy 문턱 비교 완료 (2026-08-28 밤)
- `remeasure_redesign.py`: **0 labels changed**(프로브가 라이브 수집 내내 정상 작동했다는 검증). occ>35% 부하행 1423개 중 label 3 201개, 그중 **occ<75%가 128개(64%)** — 이전(15/38=39%)보다 "occupancy로는 원리상 못 잡는 심각" 비중이 훨씬 높은 데이터
- `prepare_ap_metrics_dataset.py` → `project/data/ap_metrics_v2_redesign2`(기존 `ap_metrics_v2_redesign`와 별도 디렉토리, 병합 안 함): train 1437 / val 308 / test 310(label 3 31개)
- `train_ap_early_exit.py --class-weight-power 1.0` → `project/checkpoints/ap_v2_redesign2/`, best val balanced acc **85.4%**(이전 82.6%보다 개선)
- 평가(`ap_v2_redesign2_eval_report.txt`): fixed θ test 88.4% (Label0 97.9%/L1 89.6%/L2 89.7%/**L3 51.6%**), dynamic θ 89.0%(L3 동일 51.6%)
- **L3 recall이 이전(76.3%)보다 낮아진 이유 확인**: 이번 test set은 label 3 중 occ<75%(어려운 케이스) 비중이 훨씬 높아서(위 참고) — 데이터가 더 어려워진 것이지 모델 퇴보가 아님
- **occupancy 문턱 비교**(`ap_v2_redesign2_threshold_comparison.txt`, scratchpad 분석 스크립트로 생성):

  | 방법 | recall | precision | F1 |
  |---|---:|---:|---:|
  | LSTM | 51.6% | 66.7% | **58.2%** |
  | occ≥65% | 48.4% | 34.1% | 40.0% |
  | occ≥70% | 35.5% | 57.9% | 44.0% |
  | occ≥75% | 25.8% | 100% | 41.0% |
  | occ≥80% | 22.6% | 100% | 36.8% |

  → **LSTM이 F1에서 모든 occupancy 문턱을 여전히 이김**(58.2% vs 최고 44.0%). 참 label3&occ<75%(23창, 이전 15창에서 증가) 중 LSTM 8/23(35%) 탐지, occ≥75 문턱은 정의상 0/23 — LSTM 고유 가치 재확인

### 다음 세션 최우선
- **이전 세션 데이터(1614행, 8/27)와 병합은 안 하기로 결정** — 별개 수집 세션이라 굳이 섞을 이유 없음(사용자 판단, 2026-08-28). `ap_metrics_v2_redesign`(1614행, 이전)과 `ap_metrics_v2_redesign2`(2115행, 이번)는 각자 별도 데이터셋/체크포인트로 유지
- **더 많은 데이터가 만능은 아님(사용자 질문에 대한 결론)**: occ 60~72% 구간의 label 2/3은 6-feature 평균이 사실상 동일해서(8/27 진단) 데이터量을 늘려도 못 뚫는 상한이 있음. recall 추정치 노이즈는 줄겠지만 근본 해결책은 아님 — 아래 1번(forecasting)이 더 유망한 방향

1. `sta_tx_bitrate_min/mean`이 이번 대량 데이터에서 escalation 지문을 보이는지 재검토(8/28 새벽 세션엔 diag_25 146행으로만 봐서 결론 보류 상태였음)
2. `ramp_load.sh`에 `termux-wake-lock` 추가는 여전히 미반영 상태(이번엔 배터리 최적화 제외로 화면 꺼짐 문제 자체는 해결됨, 급하지 않음)
3. (검토) 밴드 스티어링 시스템으로 주제 확장할지 팀 결정 — 상세는 `.work-log/current.md` "향후 시스템 구상" 섹션(8/27 논의)
4. **EE의 exit별 loss 가중치를 SDN 스타일(0.15/0.30/0.55, 뒤쪽 exit에 더 강하게)로 바꿔 재학습해보기** — "왜 SDN이 이겼나" 재조사에서 나온 가설(SDN이 exit3 심각 탐지를 더 잘 학습함)의 검증되지 않은 다음 단계. 지금 EE는 균등 가중치(0.3/0.3/0.4)
5. 데모 대시보드 구축 (설계는 돼있음, `docs/yongsang/demo_api_spec.md`) — 아직 미착수
6. ~~SDN-style이 우리 모델을 이긴 이유 후속 조사~~ — **완료, 결론 정정됨**(아래 참고)
7. ~~Baseline/SDN-style 비교표~~ — **완료**(아래 참고)

### Baseline/SDN-style 비교표 + Pi 배포 완료 — 1차 판단(SDN이 더 낫다) 재조사로 정정됨 (같은 세션, 후속)
1학기엔 없었던 `ap_metrics_v2_redesign2` 기준 Baseline LSTM/SDN-style Early Exit을 이 브랜치에 새로 구현(`models/baseline_lstm.py`, `models/sdn_lstm.py`, `models/ap_baseline_lstm.py`, `models/ap_sdn_lstm.py` — yongsang 브랜치 원본을 `git show`로 참고해 6-feature용 재작성). **1학기와 달리 Baseline/SDN도 Proposed와 동일하게 class-weight-power=1.0으로 학습**(아키텍처만 통제 변수로 비교하기 위함). 학습(`train_ap_baseline_lstm.py`, `train_ap_sdn.py`) → 평가(`evaluate_ap_baseline_lstm.py`, `evaluate_ap_sdn.py`) → 비교표(`generate_ap_comparison.py`) → 사용자 요청으로 Baseline/SDN도 Pi ONNX 배포(`export_onnx_ap_baseline.py`, `export_onnx_ap_sdn.py` + `export_onnx_ap_sdn_unified_int8.py` — SDN도 Early Exit과 같은 "staged 양자화 후 If 노드 재조립" 기법 재사용, 종료조건만 confidence>=threshold로 교체).

**최종 Pi 실측(INT8, 5회 반복 평균)**:

| 모델 | Pi INT8(ms) | 정확도 | Label 3 |
|---|---:|---:|---:|
| Baseline(EE 없음) | 0.765 | 89.4% | 48.4% |
| **SDN-style** | **0.600** | 86.8% | **64.5%** |
| Proposed Fixed θ | 0.641 | 88.4% | 51.6% |
| Proposed Dynamic θ | 0.679 | 89.0% | 51.6% |

**1차 결론(속도·recall만 봄)**: SDN-style이 Pi 실측에서 더 빠르고(0.600ms) Label 3 recall도 더 높다(64.5%) — 이대로 두면 "SDN이 이겼다"로 오독될 수 있어 재조사함.

### "왜 SDN이 이겼나" 재조사 — threshold 문제 아님, recall/precision 트레이드오프였음
- **threshold sweep(val로 튜닝)**: SDN(T=0.80~0.99)·EE(theta=0.15~0.4/0.3~0.7) 둘 다 배포값이 이미 val 기준 plateau 안 — 튜닝 부족이 원인 아님
- **precision까지 본 재평가**: SDN recall 64.5%/precision 46.5%/**F1 54.1%** vs EE recall 51.6%/precision 66.7%/**F1 58.2%** — **F1은 오히려 EE가 근소 우위**. SDN은 recall을 챙기려고 precision(과잉탐지)을 크게 희생한 것 — recall 숫자만 보고 "SDN이 이겼다"고 한 건 성급했음
- **true label-3(31창) exit별 분해**: EE는 exit2로 가면 100% 정답(10/10)인데 exit3까지 넘어가는 비중이 SDN보다 큼(21 vs 16)and exit3 심각 탐지력이 낮음(6/21=29%) — SDN의 loss 가중치(0.15/0.30/0.55, 뒤쪽 exit에 더 강함)가 exit3 심각 학습에 유리했을 가능성. EE도 SDN 스타일 가중치로 재학습해보는 건 검토만 함(미실행, 후속 과제)
- **정정된 최종 결론**: 속도는 SDN이 확실히 앞섬(0.600ms vs 0.641ms). 정확도는 recall만 보면 SDN 우위처럼 보이지만 precision까지 고려한 F1은 막상막하(EE 근소 우위) — "SDN이 전면적으로 낫다"는 부정확하고 "SDN은 recall-precision 트레이드오프를 다르게 잡았고 속도가 더 빠르다"가 정확함. 상세: `project/results/yongsang/ap_model_comparison_redesign2.txt`
4. ~~Early Exit 세션 구조 재설계~~ — **완료**(같은 세션에 이어서 진행, 아래 "단일 그래프 재설계 성공" 섹션 참고). baseline 대비 40% 빠른 통합 그래프 확보
5. ~~INT8 양자화~~ — **완료, 1차 결론 정정됨**(같은 세션에 이어서 진행, 아래 참고). 최종적으로 unified fp32보다 추가 43~46% 빠른 int8 확보

### INT8 양자화 1차 시도 — "정확도 유지, 속도 이득 없음" (오판, 아래서 정정됨)
신규 `project/scripts/export_onnx_ap_unified_int8.py`(`onnxruntime.quantization.quantize_dynamic`, LSTM 포함 동적 양자화)로 unified 그래프(fixed/dynamic)를 INT8화. 정확도는 fp32와 동일했지만(0 mismatch) 속도는 fp32 unified와 사실상 동일(fixed 1.204ms/dynamic 1.193ms) — "이 모델 크기에서는 양자화 이득이 없다"고 1차 결론 냄.

### 사용자 재확인 질문("1학기 땐 실제로 양자화 효과 있었는데?") → 원인 재진단 → 해결
사용자가 1학기 INT8 결과(baseline -40%, staged -38%, 실제로 빨랐음)를 근거로 위 결론에 의문 제기. 재조사 결과 **1차 결론이 틀렸음**을 발견:

- **진짜 원인**: ONNX 그래프를 재귀적으로 순회해서 op 목록을 세어보니, unified(If 노드 포함) 그래프를 양자화하면 **LSTM 3개가 전부 그대로 float로 남아있고** 제일 작은 classifier1의 Gemm 하나만 int8로 바뀌어 있었음. `staged` 그래프(제어 흐름 없는 flat 그래프) 하나만 따로 양자화해보니 LSTM이 정상적으로 `DynamicQuantizeLSTM`으로 변환됨 — **ONNX 양자화 도구가 If 서브그래프 안의 LSTM은 건너뛴다**는 도구상의 한계가 원인이었음(모델 크기 문제 아니었음)
- **해결**: 신규 `project/scripts/export_onnx_ap_unified_int8_v2.py` — ① staged 3개 그래프를 각각 독립적으로 양자화(LSTM 정상 변환 확인) ② 양자화된 조각들을 `onnx.helper`로 손수 재조립(entropy/threshold/If 배선을 fp32 unified와 동일하게 재현, dynamic θ의 occupancy 기반 임계값 조정 로직도 그래프 안에 재현)
- **검증**: PyTorch 참조 대비 test 310창 중 309개 정확히 일치(fixed/dynamic 각각 1개만 경계값 오차), 정확도 fp32와 동일
- **Pi 재측정 — 최종 결과**:

  | 방법 | avg(ms) | vs baseline | vs unified fp32 |
  |---|---:|---:|---:|
  | Baseline | 1.966 | — | |
  | 통합 fp32 Fixed θ | 1.183 | -40% | |
  | 통합 int8(1차, LSTM 미양자화) | 1.204 | -39% | +2%(이득 없음) |
  | **통합 int8 v2 Fixed θ(진짜 양자화)** | **0.641** | **-67%** | **-46%** |
  | **통합 int8 v2 Dynamic θ(진짜 양자화)** | **0.679** | **-65%** | **-43%** |

- **최종 결론**: 1학기 자료가 보여준 int8 이득이 이번에도 재현 가능했음 — 처음엔 그래프 구조(If 노드) 때문에 막혀 있었을 뿐. **최종 배포 구성은 "단일 그래프 + INT8(stage별 양자화 후 재조립)"**로 확정. 상세: `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`

### Raspberry Pi 실기기 지연 측정 완료 (2026-08-28 밤) — Early Exit이 이번엔 baseline보다 느림
- **신규 `project/deploy/raspberry_pi_ap_v2/`**: 1학기 `project/deploy/raspberry_pi/inference_pi.py`(4-feature)를 이 브랜치의 6-feature 모델용으로 재작성(`inference_pi_ap.py`, repo import 없이 독립 실행). ONNX 8개 + `ap_metrics_v2_redesign2/test.csv`를 파이(`~/ap_pi_v2/`)에 배포
- 측정: baseline(항상 전체 그래프 1회), fixed θ(staged), dynamic θ(staged) — 각 test 310창, 샘플당 5회 반복 평균

  | 방법 | avg(ms) | exit1/2/3 비율 |
  |---|---:|---|
  | Baseline(전체 그래프 1회) | **1.966** | 0/0/100% |
  | Fixed θ (staged) | 2.337 | 29.7/11.6/58.7% |
  | Dynamic θ (staged) | 2.189 | 30.0/22.3/47.7% |

- **핵심 발견**: Early Exit이 이번 Pi 실측에서 **평균적으로 baseline보다 느림**. exit별로 쪼개보면 exit1(29.7%)은 baseline 대비 -51%(0.97ms)로 확실히 빠른데, exit3(58.7%)는 baseline 대비 **+57%**(3.08ms)로 오히려 느림 — staged 방식이 별도 ONNX 세션을 최대 3번 순차 호출하는 구조라 세션 호출당 고정 오버헤드가 붙는데, 이 모델(hidden_size=128)은 작아서 LSTM 연산량보다 오버헤드가 더 큼. 이번 test set은 라벨이 어려워서(L3 recall 51.6%) exit3 비율이 높아 가중평균이 나빠지는 조건이 갖춰짐
- **결론**: 지금 구조(세션 3개 분리)는 "레이어를 실제로 skip할 수 있다"는 개념 검증에는 유효하지만, 이 배포 구성 그대로는 latency 우위를 주장할 수 없음. 상세: `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`
- **1학기 결과(`project/results/hojung/`, 4-feature, 다른 모델 크기)와 직접 비교 금지** — CLAUDE.md 지침대로 별개 라인(수치 자체를 나란히 놓고 우열 비교하지 않는다는 뜻, 아래 구조적 패턴 교차검증과는 별개)

### 1학기 자료와 교차검증 — "staged Early Exit이 baseline보다 느림" 현상이 1학기에도 이미 있었음
사용자 질문("1학기 비교 자료 있어?")으로 `project/results/hojung/`(1학기 4-feature, origin/hojung에서 유지 중인 실제 Pi 실측)을 다시 확인.

- `pi_fp32_analysis.txt`(baseline, 전체 그래프 1회): avg **1.530ms**, exit1/2/3 지연이 1.52/1.55/1.50ms로 거의 동일(=애초에 매번 전체 계산하고 사후에 exit 라벨만 붙인 것)
- `pi_fixed_staged_fp32_analysis.txt`(staged): avg **2.089ms**(+37%), exit1 0.91ms / exit2 2.03ms / exit3 3.09ms
- `pi_dynamic_staged_fp32_analysis.txt`(staged): avg **1.989ms**(+30%)
- **오늘(6-feature) 결과와 방향이 정확히 일치** — baseline 1.966ms vs staged fixed 2.337ms(+19%)/dynamic 2.189ms(+11%). 모델 세대·feature 수와 무관하게 **이 배포 구조(ONNX 세션 3개 분리) 자체가 이 Pi+ONNX Runtime 조합에서 구조적으로 느리다**는 근거가 두 독립적인 실험에서 재현됨
- **1학기 `comparison_summary.txt`(PC 기준으로 보이는 별도 파일)는 반대 결론**("LSTM Full vs Early Exit Dynamic: -0.16ms", Early Exit이 근소 우위) — PC 타이밍만 보고 "Early Exit이 이긴다"고 판단했을 가능성. 진짜 Pi 실측끼리(`pi_fp32_analysis.txt` vs `pi_*_staged_fp32_analysis.txt`) 직접 대조한 기록은 이번에 처음 확인함
- **시사점**: 논문/보고서에서 "Early Exit이 지연을 줄인다"고 쓰려면 PC 타이밍이 아니라 반드시 실제 Pi 실측 기준이어야 하고, 지금 구조로는 그 주장이 성립 안 함 — 위 "ONNX If 연산자로 세션 통합" 재설계가 이 주장을 살리기 위한 전제조건

### 단일 그래프(ONNX If 노드) 재설계 성공 — Early Exit이 baseline 대비 40% 빠름 (같은 세션, 후속)
사용자 질문("시간 측면에선 baseline을 못 이기는 거 아냐? 모델이 작을수록 그 경향이 더할 것 같은데")에서 출발 — 정확한 직관이었음: 문제는 LSTM 재계산이 아니라 staged 구조가 세션을 최대 3번 호출하는 고정 오버헤드였고, 모델이 작을수록(hidden_size=128) 그 오버헤드 비중이 커짐.

- **신규 `project/scripts/export_onnx_ap_unified.py`**: `torch.jit.script`로 entropy 기반 if/else 분기를 캡처 → `torch.onnx.export`가 ONNX `If` 노드로 내보냄. 세션 3개 대신 **세션 1개, 그래프 내부에서 조건부 실행**. batch_size=1 전제(실시간 단일 윈도우 추론용)
- **정확도 검증**: PyTorch staged reference 대비 test 310창 전체에서 예측·exit_point **100% 일치**(fixed·dynamic 둘 다, 미스매치 0)
- **Pi 재측정**:

  | 방법 | avg(ms) | vs baseline |
  |---|---:|---:|
  | Baseline(전체 그래프 1회) | 1.966 | — |
  | Staged Fixed(세션 3개) | 2.337 | +19% |
  | Staged Dynamic(세션 3개) | 2.189 | +11% |
  | **통합 Fixed(세션 1개, If)** | **1.183** | **-40%** |
  | **통합 Dynamic(세션 1개, If)** | **1.190** | **-39%** |

- exit3(전체 계산)조차 baseline보다 빠름(1.621ms vs 1.966ms) — baseline은 3개 분류기 출력을 매번 다 계산하는데 통합 그래프는 필요한 만큼만 계산하기 때문. 재실행해도 1.183ms/1.187ms로 재현됨
- **결론 갱신**: "Early Exit이 이 모델 크기에서 latency를 못 줄인다"는 이전 결론은 staged(세션 3개) 구조에 국한된 얘기였음. 원인(세션 호출 오버헤드)을 없애자 Early Exit의 latency 이득이 실측으로 확인됨 — **논문/보고서엔 이 통합 그래프 결과를 "Proposed"로 쓸 것**, staged는 "왜 단순 분리가 아니라 단일 그래프가 필요했는가"의 동기/반례로 남김
- 파일: `project/scripts/export_onnx_ap_unified.py`, `project/checkpoints/ap_v2_redesign2/ap_early_exit_{fixed,dynamic}_unified.onnx`, `project/deploy/raspberry_pi_ap_v2/bench_unified.py`, 상세 `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`(후속 섹션에 추가)
- **INT8 양자화는 여전히 미실시** — 다음 후보(통합 그래프 기준으로 다시 검증 필요)

### 문서화 + CLAUDE.md 갱신 (같은 세션, 마무리)
- 신규 `docs/yongsang/onnx_early_exit_redesign.md` + `.html`(아티팩트로도 게시: "ONNX Early Exit 재설계") — staged→unified 재설계 전체 스토리(문제 발견, 원인 분석, 해결, 검증, 1학기 교차검증, 결론)를 정리
- `CLAUDE.md` "알려진 한계"의 "ONNX/INT8/Raspberry Pi 배포 파이프라인이 아직 이 라인에는 없다" 항목이 이제 거짓이라 갱신, "Claude가 추가로 참고해야 할 파일" 목록에 새 문서 추가

### ONNX export 완료 (2026-08-28 밤) — `ap_metrics_v2_redesign2` 6-feature Early Exit
- **신규 `project/scripts/export_onnx_ap.py`**: 1학기 버전(`yongsang` 브랜치, 9-feature)을 이 브랜치의 6-feature 모델에 맞게 재작성(`git show yongsang:...`로 원본 참고, `models/baseline_lstm.py`가 이 브랜치엔 없어서 baseline export는 뺌 — Early Exit Fixed/Dynamic만)
- `pip install onnx onnxruntime`(capstone 환경에 없어서 설치)
- `project/checkpoints/ap_v2_redesign2/`에 8개 파일 생성: `ap_early_exit_{fixed,dynamic}.onnx`(전체 3-exit 그래프) + 각각의 `_stage{1,2,3}.onnx`(파이에서 단계별 실행 후 조기종료용으로 분리한 그래프)
- **검증 완료**: PyTorch 출력과 ONNX 출력 오차 최대 2e-6(부동소수점 오차 수준), staged(1→2→3 순차 실행) 파이프라인도 full-graph와 완전히 동일한 결과
- **다음**: 아직 PC(노트북)에서의 export/검증만 끝난 상태 — Raspberry Pi 실기기 배포·지연 측정은 미착수(위 "다음 세션 최우선" 4번)

### forecasting 재평가 완료 (2026-08-28 밤) — escalation 조기경보 결과가 큰 표본에서도 재현됨
`forecast_eval_redesign.py`의 CSV 경로를 `metrics_v2_pi_redesign2_relabeled.csv`(2115행)로 갱신 후 재실행. 결과: `project/results/yongsang/ap_v2_redesign2_forecast_eval.txt`

| k | escalation 표본(이전→이번) | LSTM 탐지(이전→이번) |
|---|---|---|
| k=3 (~3~6s) | 12→13창 | 67%→61.5% |
| k=5 (~5~10s) | 14→22창 | 36%→45.5% |

- 표본이 커졌는데도(특히 k=5는 14→22창) 탐지율이 같은 자릿수로 재현 — "LSTM = 심각 전이 3~6초 조기경보기" 서사가 노이즈가 아니라 견고한 패턴이라는 근거 강화
- occupancy 문턱/persistence는 escalation subset에서 구조적으로 0%(정의상 "아직 심각 아님"만 보고 판단) — LSTM만 추세(occ/retry 상승, RSSI 불안정화)로 선제 탐지
- 전체 정확도는 이번에도 persistence가 LSTM을 이김(k=3: 69.9% vs 61.5%) — 지난 결론과 일치, LSTM의 가치는 일반 분류가 아니라 조기경보

## 완료 (2026-08-27 심야 — 재설계 스키마 재학습 + occupancy 문턱 비교)

### 파이프라인 완주: 수집 → remeasure → 변환 → 학습 → 평가
- **신규 `project/scripts/remeasure_redesign.py`**: `metrics_v2_pi_redesign.csv` 전체를 현재 `calculate_scores`(RTT/2 + failure=max)로 재계산. failure=max 게이트(`probe_ever_ok` per-scenario + `channel_active`)도 replay. `--drop load_45`로 dud 제외.
- `metrics_v2_pi_redesign_relabeled.csv` **1614행** {0:837, 1:184, 2:335, 3:258}. (load_25b 139행 추가 수집됨, load_20 크래시 후 재개해서 총 8시나리오)
- `project/data/ap_metrics_v2_redesign/` (6 feature, window 10): train 1067 / val 229 / **test 228 (label 3 = 38창)**. 스케일러 자체.
- `project/checkpoints/ap_v2_redesign/` — `--class-weight-power 1.0`, best val balanced acc 82.6%.
- `project/results/yongsang/ap_v2_redesign_eval_report.txt`, `ap_v2_redesign_threshold_comparison.txt`.

### 평가 결과 — 역대 최고이자 가장 안정적
- **fixed θ: test 85.5%** | Label0 95.5% / Label1 70.4% / Label2 78.4% / **Label3 76.3%**
- dynamic θ: 84.6%, Label3 81.6%(label2 70.6%로)
- **class collapse 없음** — 4클래스 다 70~95%. 역대 label 3 recall이 0/12.5/40/54.5%로 8~11창에서 요동쳤는데, 이번엔 38창에서 76%.

### 핵심: 심각(label 3) 탐지 — LSTM vs occupancy 단일 문턱

| 방법 | recall | precision | F1 |
|---|---:|---:|---:|
| **LSTM** | 76.3% | **85.3%** | **80.6%** |
| occ ≥ 65% | 81.6% | 56.4% | 66.7% |
| occ ≥ 70% | 78.9% | 71.4% | 75.0% |
| occ ≥ 75% | 60.5% | 100% | 75.4% |
| occ ≥ 80% | 55.3% | 100% | 71.2% |

- occupancy 문턱은 트레이드오프에 갇힘: 낮추면 recall↑ FP↑(label 2를 심각이라 함), 높이면 FP 0이지만 심각의 40~45% 놓침. **LSTM은 recall 76% + precision 85% 동시** — retry/RSSI/throughput/추세로 "occ 70% + victim 정상"과 "occ 70% + victim 붕괴"를 구분.
- **참 label 3 & occ<75% (15창, occupancy 문턱이 원리상 못 잡음)**: LSTM 6/15 정탐(대부분 occ 73% 경계), 9창은 label 1~2로 과소, **정상이라 한 건 0건**. occ≥75 문턱은 0/15.

### 한계 진단 + tx_bitrate 시도 (2026-08-28) — 실패, 6-feature 모델이 최종
- **정확한 문제**: occ 60~72%에서 label 2 vs 3 창의 6 feature 평균이 **완전히 동일**(occ 66/65, thr 66/66, retry 0.30/0.30, rssi -35/-34). 라벨 차이는 프로브 축(jitter/loss/latency)에서만 나오는데 프로브는 입력 아님 → 그 구간 일부는 **원리상 학습 불가**. 나머지는 학습 가능(LSTM은 throughput 붕괴한 심각은 잡음, 높은 throughput+moderate occ 심각은 label 2로 과소).
- **retry_ratio가 프록시 실패**: 이 RF-험한 2.4GHz는 부하만 걸리면 retry ~0.30 고정, L2/L3 구분 못 함.
- **`sta_tx_bitrate` feature 추가 시도 → 뺌** (커밋 `dbfe148` → `f634777` 되돌림 → `f7f1373` 재정의): `iw station dump`의 station별 tx bitrate. 1차 정의(전체 station min)는 diag_25 런에서 유휴 station(MCS 0, 6.5Mbit/s)이 매 행 지배 → 상수. `f7f1373`에서 **"이번 폴링에 실제 송신한 station만"으로 재정의**(유휴 배제). **collect_metrics.py는 두 컬럼을 CSV에 계속 기록(정보용), 모델 입력엔 미사용** — 다음 램프형 수집에서 escalation 창에 rate-collapse 지문 있는지 재검토 후 승격 여부 결정. 실기기 미검증(문법만).
- **6-feature 모델 결론**: `ap_metrics_v2_redesign` test 85.5% / L3 recall 76% / F1 0.81 vs occupancy 문턱. 점 분류로는 이게 최종.

### forecasting 평가 (커밋 `6331831`, `project/scripts/forecast_eval_redesign.py`)
윈도우 target을 k폴링 뒤 라벨로 두고 재학습. 폴링 ~1~2s → k=3은 ~3~6s 앞.
- **전체 정확도로는 LSTM이 persistence("5초 뒤도 같음")를 못 이김** (k=3: 69.6% vs 80%).
- **escalation 창(지금 심각 아닌데 k폴링 뒤 심각): k=3에서 LSTM 8/12=67%, occupancy 문턱·persistence는 0%(원리상).** occ·retry 상승 + RSSI 불안정화 추세로 선 넘기 전 감지. 대가는 precision 43%. 유용 지평선 짧음(k=5는 36%).
- **재포지셔닝**: LSTM = "더 나은 분류기"가 아니라 **"심각 전이 3~6초 조기경보기"**. band steering이 QoS 붕괴 전에 조치하려면 이게 정확히 필요한 것. 단 escalation 표본 12~14창으로 작음 → 데이터 더 필요.

---

## 내일 할 일 (2026-08-28)

### 0. 시작 전
- AP 밤새 7번 크래시 + load average ~1.5 → **AP 전원 뽑았다 꽂아 신선한 상태로 시작**
- 노트북 `GL-SFT1200-a08` 고정 확인 (`netsh wlan set profileparameter name="SK_0600_5G" connectionmode=manual` 이미 적용, 재부팅 후 로밍하면 다시)
- 노트북에 iperf3 서버 3개: `5201`/`5202` 부하, `5203` 프로브
- 파이 collector는 `~/ap_collect/collect_metrics.py` (f7f1373 반영됨, `CSV_FILE=metrics_v2_pi_redesign.csv`)
- **이전 relabeled 데이터는 그대로** (`metrics_v2_pi_redesign_relabeled.csv` 1614행). 새 수집분을 새 파일 `metrics_v2_pi_redesign2.csv`에 모으고, 나중에 합치거나 별도 학습

### 1. 램프형 부하 수집 (핵심 — escalation 창 확보 목적)
blast-UDP는 순식간에 포화로 튀어 1→2→3 전이가 안 생김. **서서히 오르는 부하**가 필요.
- **계단식**: 폰 A/B를 `10M(60s) → 20M(60s) → 30M(60s) → 40M(60s)` 순차 실행 (또는 두 폰 시차 30~60s). 각 시나리오 240s.
- **또는 무릎 근처**: 합계 ~40~50M(폰당 20~25M)로 240s — 상태가 2/3 경계를 자연스럽게 오르내림
- **5~6런, occ 50~75% 집중, escalation 창 40~60개 목표**
- 안전 상한: 15M/240s·25M/180s·35M/120s. 45M 금지. 런 사이 AP 60~90s 쿨다운. 종료 즉시 collector kill
- 콘솔에 `Sta tx rate min/avg`가 무부하 0 / 부하 시 실제값 뜨는지 확인 (f7f1373 검증)

### 2. 재평가
- `remeasure_redesign.py`로 새 파일 relabel (스코어링은 그대로)
- `forecast_eval_redesign.py` 재실행 → escalation recall이 60~70% 유지되는지 확인
  - 유지 → "조기경보기" 서사 확정 (논문 결과)
  - persistence 수준으로 무너짐 → 정직하게 "분류는 문턱과 동급, 기여 = 라벨 재설계 + Early Exit"
- `sta_tx_bitrate_min` (active 정의)이 escalation 창에서 rate-collapse 보이는지 → 보이면 7번째 feature로 승격 후 재학습

### 3. (2번이 잘 되면) 6-feature 모델 재학습
- 합친 데이터로 `prepare_ap_metrics_dataset.py` → `train_ap_early_exit.py --class-weight-power 1.0` → `evaluate_ap_early_exit.py`
- threshold_comparison 분석 스크립트 재실행 (occ<75 L3 recall)

### 나중 (내일 아닐 수도)
- ONNX export (데모 전제)
- `congestion_label_redesign.md` HTML
- 밴드 스티어링 확장 (팀 결정)



## 완료된 작업 (2026-08-27 심야 — 재설계 스키마 본격 수집)

### 라벨 채점 수정 2건 (커밋 `13c22a4`, `2ee7eea`)
1. **failure = max** (`13c22a4`): 채널 포화로 victim 프로브·ping이 완전히 죽으면 축을 0이 아니라 **1.0**으로. `channel_active`(throughput≥3 or occ≥40) 게이트 + 프로브 `ever_ok` 게이트로 idle 오탐/서버 미기동 배제. 이 코드 경로 = 유선 AP 텔레메트리 정상 = AP 다운 아니라 채널 포화. 45/45 런 재처리: occ<75% label 3이 11→22.
2. **latency RTT/2** (`2ee7eea`): `ANCHORS["latency"]`(30/60/150/400)는 G.114 **편도** 값인데 ping은 RTT라, RTT 150ms(≈편도 75ms, 경미)를 "심각"으로 채점 → label 3이 부하행의 79% 과다 발생. `calculate_scores`가 `latency_ms/2`를 편도 추정치로 넣도록. 재처리: 부하행 label 3 111→85, label 2 28→54.
   - 사용자 지적("심각이 너무 잘 뜨는 거 아니야?")에서 나온 수정. occupancy/loss/jitter 앵커는 표준값이라 안 건드림.

### 재설계 스키마 데이터 수집 (`metrics_v2_pi_redesign.csv`, 파이 유선)
- 이전 캘리브레이션 데이터는 `metrics_v2_pi_redesign_calib_0827.csv`로 아카이브(파이). 새 파일로 시작.
- 노트북 Opal 고정(`SK_0600_5G` 프로필 수동 연결로) + 서버 3개(5201/5202 부하, 5203 프로브). 파이가 collector.
- **유효 1390행** (dud `load_45` 제외 — 폰이 부하 안 실어서 occ>35 9행뿐):

  | 시나리오 | 행 | L0 | L1 | L2 | L3 |
  |---|---|---|---|---|---|
  | idle | 75 | 75 | — | — | — |
  | load_15 (120s) | 175 | 88 | 29 | 50 | 8 |
  | load_15b (240s) | 238 | 55 | 75 | 79 | 29 |
  | load_15c (300s) | 231 | 27 | 67 | 86 | 51 |
  | load_25 (120s) | 169 | 88 | 4 | 39 | 38 |
  | load_35 (120s) | 261 | 197 | 1 | 15 | 48 |
  | load_45b (120s) | 241 | 171 | 4 | 18 | 48 |
  | **합계** | **1390** | 701 | 180 | 287 | 222 |

- **L3 222개 중 occ<75%가 108개** — occupancy 문턱이 못 잡는 심각. 재설계 목표 달성 근거.
- 부하 세기 → 라벨 그라데이션 명확: load_15은 L1/L2 위주, load_35/45는 L3 위주.
- **주의: CSV에 저장된 `label` 컬럼은 스코어링 혼재** (latency RTT/2 수정 전후, failure=max 전후). 학습 전 `remeasure_redesign.py`(미작성)로 전체 재계산 필요. 위 표 수치는 재계산 후 기준.

### AP 크래시 재확인 — 45M은 매번, 15M도 5분이면 크래시
- **45M×2: 2/2 크래시** (120초 안팎). load_45b는 크래시 전 70행 확보하긴 함. **45M 이상 부하는 폐기 결정** — 리스크 대비 이득 없음(L3는 25/35에서 충분).
- **15M×2: 4분(240s) 안정, 5분(300s) 크래시.** load_15c 5분 런 끝에 크래시(데이터는 231행 다 건짐, 폴링 갭 없음).
- **20M×2: 240초 시도 → ~2분경 크래시** (2026-08-27 심야 마지막 런). AP 100% 손실 + 노트북이 SK_0600으로 로밍(수동 설정했는데도). load_20 데이터는 파이 SD에 일부 남았을 것 — 다음 세션 회수.
- 크래시 후 대부분 자가 복구됨(1~2분 뒤 ping 정상). 물리 재부팅 필요는 이번 세션엔 없었음.
- **결론: 240초/15M가 안전 상한. 25M은 180초, 35M은 120초까지.**

### 다음 세션 (재개 지점)
1. 파이 접속 회복 → `metrics_v2_pi_redesign.csv`에서 load_20 잔여 행 확인, 필요시 1~2런 더 (load_25b 180s, load_35b 120s)로 ~1800행
2. **`remeasure_redesign.py` 작성** — `metrics_v2_pi_redesign.csv` 전체를 현재 `calculate_scores`(RTT/2 + failure=max)로 sub-score·label 재계산. dud `load_45` 시나리오 제외. (로직은 이번 세션 scratchpad 분석 스크립트에 있음: `probe_ever_ok` per-scenario 추적 + `channel_active` 게이트)
3. `prepare_ap_metrics_dataset.py` → windowed 6-feature 변환
4. `train_ap_early_exit.py --class-weight-power 1.0` → 평가, **occ<75% L3 recall** 집중 확인 (occupancy 문턱 대비 우위 지표)

## 완료된 작업 (2026-08-27 저녁 세션 — 집, 파이+AP 실기기)

### AP 상태 확인 + 발견: 지금까지 모든 데이터가 2.4GHz였음
- 세션 시작 시 AP 크래시 상태(80/80 10분 크래시에서 자가재부팅 안 됨) → 물리 재부팅 후 정상. 이후 세션 중 400초 소패킷 런 끝에서 한 번 라디오 순간 리셋(SSID는 유지, 자가재부팅) 있었으나 완전 크래시 아님
- **SSID `GL-SFT1200-a08` = radio0 = 2.4GHz(채널 1, HT40)**. 폰 3대 다 여기 붙어 있음. 5GHz(`GL-SFT1200-a08-5G`, 채널 40, VHT80)는 아무도 안 씀. **즉 `ap_metrics_v2`의 모든 데이터가 2.4GHz다.** (앞선 문서 논의에서 "5GHz 유지"라고 적은 건 착오 — 실제로는 계속 2.4GHz였음. 관련 문서는 이번에 정정 못 함, 다음에 손봐야 함)
- AP wireless/network 설정 백업: `uci export` 결과를 scratchpad에 저장. LAN 포트는 vlan1(포트 1,2), WAN은 포트 0. 파이가 LAN 포트에 유선 연결되어 `192.168.8.109`

### 파이 유선 수집 세팅 완료 — 유선 관리채널 실동작 확인
- **파이(`CapsTone`, `192.168.8.109` eth0 유선 / `192.168.45.31` wlan0 집공유기)**. onnxruntime 1.26 + numpy 이미 설치돼 있음
- 파이→AP SSH: 노트북의 `~/.ssh/id_rsa_ap` 키를 파이로 복사 + 파이 `~/.ssh/config`에 Host 블록(HostKeyAlgorithms/PubkeyAcceptedAlgorithms +ssh-rsa). 파이에서 `ssh ap "..."` 무패스워드 동작 확인 (eth0 유선 경로, 무선 채널 안 탐)
- `collect_metrics.py`를 파이 `~/ap_collect/`에 복사 (stdlib만 써서 그대로 돌아감. `CSV_FILE`만 `metrics_v2_pi.csv`로 바꿈, 파이 데이터는 latency 베이스라인이 달라서 메인 CSV와 분리)
- **주의**: `packet_loss_udp_percent`는 파이에서 N/A (로컬 iperf3 JSON 없음) — 모델 입력 아니라 무방

### 폴링 속도: 4초 → 3.3초 → 1.1초
| | 노트북 방식 | 파이 + APPoller | 파이 + `ping -c 2` |
|---|---|---|---|
| 행당 간격 | ~4초 | ~3.3초 | **~1.1초** |
| SSH 폴링 | 매번 새 접속 0.5~1초+, 혼잡 시 최대 156초 스톨 | 지속 세션 캐시에서 즉시(~0) | 동일 |
| latency 측정(`ping`) | `-c 4` ~3초 (병목) | `-c 4` ~3초 (병목) | `-c 2` ~1초 |

- `collect_metrics.py`의 Linux 핑을 `-c 4` → `-c 2`로 변경 (jitter는 여전히 mdev, 2샘플이면 약한 추정). Windows 핑(`-n 4`)은 그대로 둠 — 파이가 앞으로 collector라 Linux 경로만 손봄
- **폴링 안정성 검증 성공**: 파이 유선 수집으로 소패킷 부하 런(280초 목표, 실제로는 종료 지연으로 42분·2414행 수집) 동안 **평균 1.14초, 5초 넘는 갭 0건**. 노트북 방식의 156초 스톨 같은 게 전혀 없음. 유선 관리채널 + APPoller 효과 실측 확인
- 이 런 데이터(`metrics_v2_pi_run1.csv`)는 학습에 안 씀 — latency 베이스라인 mismatch(파이 eth0→AP→폰 홉 추가로 idle에도 70~125ms) + old 스키마로 수집됨(poll_interval_s 없음, 1초 해상도 timestamp라 역산 불가)

### `tx_retries_delta` → `tx_retries_per_s` — 폴링 주기 의존성 버그 수정 (정석)
폴링을 ~1초로 당기니 label이 1에 갇히는 게 관측됨. 원인: `tx_retries_delta`/`tx_failed_delta`는 "지난 폴링 이후 재전송 수" = 델타값인데 **시간으로 안 나눠져 있어서** 폴링 주기에 그대로 비례(4초 폴링 delta ≈ 1초 폴링 delta × 4). `RETRY_FAILED_MAX=25000`이 4초 폴링 기준이라 1초 폴링에선 `retry_failed_score`가 1/4로 눌림. (throughput은 이미 `/elapsed`로 나눠져 있어서 무관)

- **`collect_metrics.py`**: ① CSV에 `poll_interval_s` 컬럼 추가 ② `tx_retries_delta`/`tx_failed_delta` → **`tx_retries_per_s`/`tx_failed_per_s`** (= delta ÷ poll_interval_s) ③ `RETRY_FAILED_MAX` → `RETRY_FAILED_MAX_PER_SEC = 6250` (= 25000 ÷ 4초, 새 데이터로 재보정 예정) ④ `calculate_scores` 시그니처·콘솔 출력 갱신
- **`utils/ap_features.py`**: 9개 feature 중 5·6번 이름 변경
- **신규 `remeasure_metrics_v2.py`**: raw feature에서 4개 sub-score를 처음부터 재계산(공식이 바뀔 때용). `relabel_metrics_v2.py`(가중치만 재조합)의 상위 버전
- **`prepare_ap_metrics_dataset.py`**: `model_excluded_columns`에 `poll_interval_s` 추가 (feature 선택은 `AP_FEATURE_COLUMNS` 명시 참조라 자동 반영됨)

### 기존 데이터 in-place 마이그레이션 (데이터 안 버림)
- **정크 정리**: `combo_smallpkt_l250_s21_s26`(노트북 수집)이 종료 지연으로 77분·1137행 수집됨(실부하는 앞 ~6분뿐). 16:55:00 이후 runaway idle 1053행 삭제 → 84행만 남김(label 1×10/2×71/3×3). `metrics_v2.csv` 5490→5574행 상태로 정리
- `remeasure_metrics_v2.py` 실행: 폴링 간격은 timestamp 차이로 역산(1초 해상도, ~4초 폴링은 ±25%, 스톨 뒤 행은 실제 간격으로 나눠서 오히려 부풀려진 값 교정). raw feature에서 sub-score·congestion_score·label 전부 재계산
- **label 분포: {0:3195, 1:1087, 2:1207, 3:85} → {0:3208, 1:1328, 2:982, 3:56}** — raw label 3이 85→56으로 줄었음. 없어진 ~29개는 폴링 스톨로 retry_delta가 부풀려져서 문턱을 넘었던 것들 (정직한 수치). CSV 컬럼: `tx_retries_delta`/`tx_failed_delta` 제거, `poll_interval_s`/`tx_retries_per_s`/`tx_failed_per_s` 추가 (`.bak` 백업됨, scratchpad에 `metrics_v2_premigration_5574.csv`도)

### 재변환 + 재학습 + 평가
- `prepare_ap_metrics_dataset.py` 재변환: train 3560 / val 763 / **test 764**, label 3 **train 37 / val 8 / test 8** (8/26 새벽 재학습 때 test 11 → 8로 감소, 마이그레이션이 아티팩트 label 3 제거한 결과)
- `train_ap_early_exit.py --class-weight-power 1.0` 재학습 (best val balanced acc **80.7%**, epoch 50)
- 평가 (`ap_v2_eval_report.txt`):

  | | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
  |---|---:|---:|---:|---:|---:|
  | Fixed θ | **91.2%** | 97.7% | 88.0% | 78.9% | **12.5% (1/8)** |
  | Dynamic θ | **91.2%** | 97.3% | 88.5% | 79.7% | **12.5% (1/8)** |

- **전체 정확도는 역대 최고**(직전 최고 89.6%, 8/26 새벽 87.2%). retry feature가 폴링 노이즈 안 타게 되면서 전반적으로 더 잘 보정된 것으로 보임
- **Label 3 recall은 54.5%(6/11) → 12.5%(1/8)로 급락** — 마이그레이션이 폴링 아티팩트 label 3을 제거해서 test 표본이 11→8로 줄었고, 8개 중 1개 차이가 12.5%p라 소표본 노이즈가 극심. 숫자 자체보다 "이제 깨끗한 파이프라인으로 label 3을 다시 쌓아야 한다"가 요점
- Exit 분포(fixed θ): Exit1 46.5%(98.6%) / Exit2 36.9%(91.1%) / Exit3 16.6%(70.9%)

### 소패킷 실측 관찰 — "핵심 검증 질문"이 실측 수치로 확인됨
- 소패킷 25M/25M(`-l 250`)은 **occupancy 55~69%에서 retry 폭증**(retry_failed_score 1.0)한, 정확히 "retry 주도" 혼잡을 만들어냄 → **label 2 대량 확보**
- 근데 label 3에 도달한 순간은 **여전히 전부 occupancy=100%**. 수치상 폰이 throughput을 못 올리는 한(≈0.3) retry·jitter 다 맥스여도 occupancy 66%면 congestion_score가 ~0.71에서 막힘 → **가중치를 안 바꾸면 "occupancy 아닌 label 3"은 물리적으로 안 나옴** (occ 0.45 비중). 팀 결정 대기 (occ 비중↓ / 문턱↓ / "occupancy-only 분류기 대비 우위"로 서사 전환)
- **새 관찰**: 소패킷 고PPS는 bps가 아니라 프레임 수로 AP를 압박하는 별개 스트레스 벡터 — 25/25(60/60보다 저부하)인데도 400초에서 라디오 리셋. 다음엔 250~300초로

### 혼잡 라벨 재설계 — 코드 구현 + 캘리브레이션 완료 (프로브·표준 문턱·max)
- **코드 구현 완료** (커밋 `ec109d0`, `013df75`): `collect_metrics.py`에 `ProbeRunner`(백그라운드 300kbps UDP victim 프로브, `iperf3 -c 192.168.8.226 -p 5203`), `anchor_score()`(표준 문턱 4-앵커 piecewise-linear), `calculate_scores`가 `max(occupancy, jitter, loss, latency)` — retry는 계산하되 max에서 제외. `ap_features.py` 9→6 feature(−2: `latency_ms`/`jitter_ms` 제거, −1: `tx_retries_delta`+`tx_failed_delta` → `tx_retry_ratio` 통합. `connected_clients`는 원래 모델 입력이 아니라 후보였다가 기각). 9−2−1=6. `prepare_ap_metrics_dataset.py` 제외 컬럼 갱신. 새 스키마라 `prepare_csv()` 가드가 옛 CSV append 거부 → **새 파일 `metrics_v2_pi_redesign.csv`로 수집**
- **idle 캘리브레이션 완료** (커밋 `3e0780a`): 파이 무부하 수집, v6 앵커에서 idle 77/77이 label 0, congestion_score max 0.17. retry는 idle에서도 retry_ratio 18~36%라 (RF 험한 2.4GHz 채널) — **retry를 라벨 축에서 뺌** (사용자 결정 "retry 빼고 가자"), 모델 feature로는 유지
- **부하 캘리브레이션 완료** (커밋 `ee6c776` 60/60, `8f93c2d` 소패킷):
  - 60/60 런: 폰이 60M씩 못 실어 채널은 ~36초만 혼잡했지만, 그 구간에서 **occupancy 60~73%(포화 아님)인데 label 3** — latency 140~291ms / loss 7.4% 주도
  - 소패킷 25/25 런(`-l 250`, 180초, 완전 크래시 없음): occ med 77 / max 90, label 3 48행 중 41행이 occupancy 주도 · 9행 latency · 4행 loss
  - **핵심 검증 결과**: 재설계 라벨은 occupancy 포화가 아닌 상태(60~73%)에서도 latency/loss로 label 3을 만든다 → occupancy-only 문턱(≥75%)이면 놓치는 행이 실측으로 존재 → "LSTM이 occupancy 문턱을 이긴다"의 경험적 근거 확보
- 상세: `docs/yongsang/congestion_label_redesign.md` (idle/60-60/소패킷 캘리브레이션 표 포함)

### 다음 세션 최우선
- [ ] **본격 다중 시나리오 수집 (재설계 스키마)** — 캘리브레이션은 끝. 이제 학습셋용으로 `metrics_v2_pi_redesign.csv`에 **누적**(매번 rm 하지 말 것) 수집: 노트북에 iperf3 서버 3개(5201/5202 부하, 5203 프로브) 띄우고 폰 60/60·75/75·소패킷·idle을 시나리오별로 3~5분씩. 종료 즉시 프로세스 kill
- [ ] **재변환 → 재학습 → 평가** — `prepare_ap_metrics_dataset.py`로 windowed 변환(6 feature) → `train_ap_early_exit.py --class-weight-power 1.0` → 평가. **occupancy 60~73% label 3 행의 recall**을 집중 확인 (occupancy 문턱 대비 우위 지표)
- [ ] `congestion_label_redesign.md` HTML 버전 (사용자: "html는 후술로 넘어가고" — 보류됨)
- [ ] **retry 앵커(10/15/25/40%) 검토** — 라벨 축에선 뺐지만 정보 컬럼으로 남아있음. idle retry_ratio가 18~36%라 앵커가 idle에서도 warn을 넘음 — 정보성 컬럼이라 무해하나 문서에 명시
- [ ] 파이 수집 데이터의 latency 베이스라인이 노트북과 다름 — 파이 데이터를 메인 학습셋에 섞을지, 파이 전용으로 재학습할지 결정 (실배포는 파이가 collector니 파이 측정값이 "진짜")
- [ ] ONNX export (데모 파이프라인 전제)
- [ ] (검토) 밴드 스티어링 시스템으로 주제 확장할지 팀 결정 — 상세는 위 "향후 시스템 구상"
- [ ] (이월) 유선 관리채널은 파이 eth0로 사실상 완성 — AP 서브넷 분리(방법 2)는 불필요 확인됨

### 실시간 폴링 지연 문제(8/26 밤 세션 설계 메모) — 지속 SSH 세션으로 전환하는 코드 조치, 실기기 미검증
8/26 밤 세션에서 "설계 메모, 코드 변경 없음"으로 남겨뒀던 문제를 이어서 코드로 구현. 근본 원인은 `collect_metrics.py`가 매 루프(~0.5~1초)마다 새 SSH 프로세스를 띄워 TCP+SSH 핸드셰이크를 반복하고, 이 관리 트래픽이 측정 대상과 같은 무선 채널을 타서 혼잡할수록 SSH도 같이 느려지는 자기참조적 구조였던 것.

- `get_ap_metrics()`(루프마다 새 SSH 접속)를 `APPoller` 클래스로 교체 — 원격에서 `iw station/survey dump`를 무한루프로 도는 셸 명령을 SSH 세션 **1개**로 띄우고 백그라운드 스레드가 표준출력을 스트리밍으로 읽어 사이클 단위 캐시. 메인 루프는 로컬 캐시에서 새 사이클을 기다리기만 해서(네트워크 호출 없음) 더 이상 느려진 SSH 응답에 동기적으로 블록되지 않음. 연결 끊기면 자동 재연결(횟수는 `poller.reconnects`로 추적, 종료 시 출력)
- CSV 스키마·congestion_score 계산·라벨링 로직은 전혀 건드리지 않음(파싱 함수 재사용, 1행=1회 실제 AP 샘플이라는 의미론도 유지)
- `python -m py_compile`로 문법만 확인됨 — **이 세션엔 AP에 물리적으로 붙어있지 않아 실기기 검증은 못 함**. 상세는 `docs/yongsang/ap_crash_analysis.md`의 "추가 조치 (2026-08-27)" 섹션

### 유선 관리채널 분리 아키텍처 논의 — 라즈베리 파이를 collector로, AP 네트워크 자체를 서브넷 분리하는 방향으로 결정(코드/설정 변경 없음, 다음 세션 실행 대상)

`APPoller`(지속 SSH)로도 못 없애는 남은 문제를 이어서 논의: 관리 스트림 자체가 여전히 같은 무선 채널을 타고 있어서, 채널이 가장 혼잡한 순간(정확히 감지하고 싶은 순간)에 그 스트림도 같이 지연/왜곡될 수 있다는 점, 그리고 측정 행위 자체가 혼잡도를 미세하게 더 올리는 자기참조 문제. 근본 해결책으로 "관리 트래픽(SSH)만 유선으로, 측정 대상 트래픽(iperf3)은 계속 무선으로" 분리하는 방향을 논의해서 다음과 같이 정리됨:

- **노트북은 이더넷 포트가 없어서 유선 분리 대상에서 제외** — 대신 라즈베리 파이(이더넷+WiFi 내장, AP 옆에 물리적으로 배치하기도 쉬움)를 collector 역할로 씀. 파이는 모니터 없이 완전 headless로 운용(Raspberry Pi Imager로 SD카드 굽는 단계에서 WiFi/SSH 미리 설정 가능, 노트북에서 SSH로 원격 제어만 하면 됨). 노트북↔파이 제어 SSH는 간헐적 트래픽이라(스크립트를 tmux/nohup으로 띄워두면 측정 도중엔 연결을 끊어놔도 됨) 연속 폴링 스트림과 달리 무해함
- **구조**: 폰(191/S26)→AP는 무선(부하 발생, 안 건드림) / AP→파이(iperf3 수신)는 **계속 무선 유지 필수** / 파이→AP(SSH 관리)만 신규로 유선. iperf3 수신까지 유선으로 옮기면 안 됨 — `ap_crash_analysis.md`에 이미 기록된 "노트북→파이 유선 150Mbps 시도, occupancy 100%까지 찍혀도 congestion_score는 0.646에서 막힘(label 3 미달)" 사례가 정확히 이 실수를 하면 무슨 일이 생기는지 보여주는 반례
- **분리 방법 두 가지 논의**:
  1. 라우팅 규칙(`ip route add 192.168.8.1/32 dev eth0`)으로 AP 관리 IP 하나만 유선 강제 — 간단하지만 eth0/wlan0이 같은 서브넷을 동시에 쓰게 되어 ARP 쪽에서 꼬일 가능성 있음(리눅스 sysctl로 완화 가능하지만 다소 지저분)
  2. **Opal(OpenWrt) 자체의 `/etc/config/network`를 고쳐서 LAN(이더넷) 포트를 별도 서브넷으로 분리** — 파이의 eth0/wlan0이 애초에 다른 서브넷에 있게 되어 라우팅 모호함 자체가 안 생김(더 깔끔). 다만 이미 크래시 이력 있는 장비의 네트워크 설정을 건드리는 리스크가 있어 기존 설정 백업 후 진행 필요
- **결정**: 사용자가 **방법 2(AP 서브넷 분리)부터 시작**하기로 결정. 이번 세션은 설계 논의만, 실기기 작업은 다음 세션(파이+AP 둘 다 물리적으로 있어야 함)

### 다음 세션 최우선
- [ ] **`APPoller` 실기기 검증** — 지속 SSH 연결이 정상적으로 station/survey를 스트리밍하는지, 콤보 부하 중 폴링 지연 스파이크가 실제로 줄어드는지, AP가 죽었을 때 재연결 루프가 폭주하지 않는지 확인. 문제 있으면 되돌릴 수 있도록 검증 전엔 기존 방식(루프당 새 SSH)으로 수집된 데이터와 섞어 비교하지 말 것
- [ ] **유선 관리채널 분리, 방법 2(AP 서브넷 분리)부터 시도** — 라즈베리 파이 준비(모델 확인: 이더넷 포트 유무, WiFi로 60~100Mbps대 수신 가능한지), Opal `/etc/config/network` 백업 후 LAN 포트를 별도 서브넷으로 분리, 파이 eth0(관리)/wlan0(iperf3 수신) 라우팅이 자연스럽게 나뉘는지 확인. 안 되면 방법 1(static host route)로 폴백
- [ ] (이전 세션에서 이월) AP 물리 재부팅 상태 확인, 75/75 반복, 재라벨링/재변환/재학습 반영 등 — 아래 8/26 밤 세션 항목 계속 유효
- [ ] **[데이터 안 늘리고 가능] occupancy 의존도 진단** — 현재 test label 3 정답/오답을 `channel_occupancy_percent` 구간별로 쪼개기(전부 90%+인가?), congestion_score 주도 sub-score별 recall, occupancy feature 뺀 ablation 학습. 목적: 모델이 "occupancy≈100→label 3" 지름길만 쓰는지 확인. 상세는 `project/README_AP_V2.md` "핵심 검증 질문"
- [ ] **`-b ??M` 대안 부하 방법 시도** — 아래 "부하 생성 방법 대안 — 실험 즉시 참고용" 치트시트 순서대로. 5GHz/80MHz 유지, 소패킷 UDP(PPS↑) → 동일 채널 간섭원(RF↓). 크래시 천장(80/80·500초) 때문에 부하를 더 키우는 대신 airtime·재전송을 올리는 방향. 2.4GHz 전환은 배포 regime과 안 맞아 안 함. **목적 = occupancy가 아니라 retry/jitter가 주도하는 label 2/3 샘플 확보**

### 부하 생성 방법 대안 — 실험 즉시 참고용 (2026-08-27 논의, 미실험)

**왜 하나 (핵심 검증 질문)**: "기기 많아서"가 아니라 경합·간섭·재전송으로 생긴 혼잡 — 즉 **occupancy만으로는 안 잡히는 혼잡** — 을 조기종료 LSTM이 얼마나 정확히 잡는가가 이 프로젝트의 진짜 질문. 그런데 지금 label 3 샘플이 거의 다 occupancy=100% 순간이라 모델이 "occupancy≈100→label 3" 지름길만 학습했을 수 있음. 그래서 필요한 건 **occupancy가 아니라 retry/jitter가 주도하는 label 2/3 샘플**. (상세: `project/README_AP_V2.md` "핵심 검증 질문")

**목표(부하 방법)**: 위 샘플을 만드는 건 **높은 PPS·airtime + 열악한 RF 환경**이다("낮은 throughput"이 목적 아님). lab은 RF가 깨끗해서 재전송이 안 생김 — 공장은 금속·다중경로·밀집 AP로 낮은 부하에서도 retry가 터짐. **5GHz/80MHz는 그대로 두고** (a) 프레임을 잘게 쪼개 PPS↑ (b) 동일 채널 간섭원으로 RF↓. 2.4GHz 전환은 배포 regime과 안 맞아 **안 함**. Opal 소비자 AP 한계는 감안(더 나은 AP가 정공법). 상세는 `docs/yongsang/ap_crash_analysis.md` "부하 생성 방법 대안" 섹션.

**시작 전 (매번)**
- [ ] 두 폰 AP 와이파이 재연결 + `ping -n 5 192.168.8.191` / `...103` 확인 (capture effect 방지)
- [ ] `ssh 192.168.8.1 "uptime"` 로 AP 살아있는지 + 재부팅 직후인지 확인
- [ ] iperf3 서버 폰별 기동: `iperf3 -s -p 5201`(191) / `iperf3 -s -p 5202`(S26)

**1단계 — 소패킷 UDP (5GHz/80MHz 그대로, AP 설정 무변경, 먼저 이것부터)**
```
# 각 폰에서 (또는 원격 실행), 대칭으로:
iperf3 -u -c 192.168.8.226 -p 5201 -l 250 -b 25M -t 300
iperf3 -u -c 192.168.8.226 -p 5202 -l 250 -b 25M -t 300
```
- `-l`은 200~400에서 조정. 서버 출력의 실제 pps/throughput 보고 폰 CPU 병목이면 `-l` 키우기
- 시나리오명: `combo_smallpkt_l250_s21_s26` (기존 60/60·75/75와 안 섞기)
- 관찰 포인트: 낮은 합계 throughput인데 occupancy 100% 찍히는지, retry/jitter 동반되는지
- 근거: 소패킷 = 산업 주기 트래픽(PROFINET/EtherNet-IP/OPC-UA/MQTT/측위/PTT)의 지배적 패턴 → 대용량 blast보다 오히려 현실적

**2단계 — 동일 채널 간섭원 (1단계에 겹치기, 실제 다중 AP 공장 모사)**
- 측정 안 쓰는 3번째 기기(여분 폰/노트북)로 같은 채널에서 유튜브 4K 재생 또는 별도 `iperf3`
- 여유되면 두 번째 AP를 같은 채널에서 방송 = co-channel interference 그 자체
- 이 기기는 congestion_score feature에 안 들어감(`connected_clients`는 제외 컬럼) → 라벨 오염 없음
- 시나리오명: `combo_smallpkt_cochannel_s21_s26`

**3단계 (여유 시) — RSSI 약화 / 낮은 MCS**
- 폰을 AP에서 멀리 두거나 금속판 뒤에 배치 → 매 프레임 on-air 시간↑ → retry↑ (regime 크게 안 바꿈)

**종료 시**
- [ ] "끝" 즉시 iperf3 서버/수집기 프로세스 종료 (유휴 방치 금지)
- [ ] 간섭원 기기도 정지
- [ ] `prepare_ap_metrics_dataset.py` 재변환 전에 새 시나리오가 기존 데이터와 섞여도 되는지 팀 확인(분포 shift — scaler 재적합됨)

### 향후 데모 구상 — 부하 제어 + 실시간 혼잡도 대시보드 (2026-08-27 논의, 미착수)

**목적**: 노트북 웹 화면의 버튼을 누르면 클라이언트 폰에 부하 명령이 가고, 파이가 AP를 실측 + ONNX Early Exit LSTM 추론해서, 노트북 웹 화면에 혼잡 레벨(0~3)이 실시간으로 변동하는 걸 눈으로 보는 데모.

**구성 (4-노드)**
- **브라우저(노트북)** — 대시보드: 부하 제어판(폰별 rate/패킷크기/시간, 간섭원 on/off) + 실시간 표시(혼잡 게이지 · 9 feature 스파크라인 · congestion_score · Exit 단계/추론 지연)
- **노트북 백엔드(FastAPI/Flask)** — 대시보드 서빙 + 부하 명령 중계 + `iperf3 -s`(5201/5202) 싱크 + 파이 스트림을 브라우저로 중계(단일 origin)
- **폰 191 / S26 (Termux)** — 부하 에이전트. 노트북이 SSH exec 또는 HTTP 에이전트(`POST /load`)로 `iperf3 -c` 기동. `termux-wake-lock` + 배터리 최적화 예외 필수
- **라즈베리 파이** — `APPoller`(AP 유선 SSH 폴링) → 롤링 윈도우(10) → ONNX 추론 → SSE/WebSocket로 초당 1회 push. **추론은 파이에서**(엣지 서사)

**신규 작업 = `ap_metrics_v2` 모델 ONNX export 하나** (1차 `export_onnx_ap.py` 재활용). 나머지는 조립.

**전제조건 / 제약**
- 유선 관리채널 필수 — 무선 폴링이면 혼잡 최고조(데모 하이라이트)에 스트림이 멈춤(과거 156초 스톨·완전 크래시)
- 부하 프리셋을 안전 범위로 하드코딩 — ≤75/75, ≤420초, 80/80 금지, 소패킷 위주. "전체 정지" 버튼 상시 + 쿨다운. AP 크래시 = 데모 중 물리 재부팅
- 게이지는 윈도우 10샘플 채우는 ~10초 지연 후 반응("warming up" 표시)
- 폰 IP 재접속 시 변동 → hostname 또는 pull 모델
- 재학습/재라벨링 시 ONNX와 scaler 동기 유지

**API 명세**: `docs/yongsang/demo_api_spec.md` (+ `.html`/아티팩트) — 브라우저↔백엔드(REST+SSE), 백엔드↔폰(부하 에이전트), 파이↔백엔드(추론 스트림), 공유 스키마, 안전 제약.

### 향후 시스템 구상 — 혼잡 감지 기반 밴드 스티어링 (2026-08-27 논의, 미착수)

**개념**: LSTM이 2.4GHz 혼잡을 판단하면 파이가 AP에 명령해서 클라이언트를 5GHz로 전환 → 혼잡 해소. 분류기(센서) → 판단(혼잡) → 액추에이터(밴드 전환) → **측정 가능한 결과**(전환 후 지연/손실 감소)의 닫힌 루프. "단순 분류" 대비 훨씬 방어하기 좋은 주제.

**신규성 프레이밍(중요)**: 밴드 스티어링 자체는 소비자/기업 AP 표준 기능이지만 대부분 **정적 휴리스틱**(신호 세기, 클라이언트 수). 차별점 = **학습 기반 + 반응형 + 조기 감지** — occupancy 문턱 방식보다 나은 타이밍에 전환한다는 걸 downstream 지표로 증명. 앞서 계속 신경 쓰던 "occupancy-only 분류기 대비 LSTM 우위"가 여기서 구체적 수치로 나옴.

**구현 경로**
- 파이 → AP 명령 경로: **이번 세션에 유선 SSH로 이미 완성**
- 클라이언트를 5GHz로: OpenWrt `dawn` 패키지(802.11k/v/r) 또는 수동 `hostapd_cli bss_tm_req <mac> ...`(802.11v BTM). S26 지원, 191 확인 필요. 안 되면 2.4 SSID deauth + TX power 낮추기로 재접속 유도
- 파이가 모델·정책·액추에이션 다 함(최저 지연, 단일 결정점). 백엔드는 관찰 + 수동 오버라이드

**전제 / 제약**
- 양 밴드 **같은 SSID**로 방송해야 seamless 로밍 (지금은 `-a08` / `-a08-5G` 따로 → 통합 필요)
- Opal 5GHz 채널 40 → 집 공유기 `SK_0600_5G`와 충돌, 36/149로 이동
- 이상적으로는 **두 밴드 다 혼잡 데이터** 필요(5GHz도 언제 나쁜지 알아야). 데모 수준이면 "5GHz는 항상 여유" 가정 가능
- 스티어링 플랩 방지 — 쿨다운 + 히스테리시스 + 분당 최대 전환 수
- Opal 불안정성이 발목 잡을 수 있음

**실험 설계(캡스톤용)**: 같은 부하에서 3-way 비교 — ① 스티어링 없음(baseline) ② occupancy 문턱 스티어링(static) ③ LSTM label≥2/3 스티어링(proposed). victim flow의 throughput/loss/latency 회복 속도, 오탐(불필요 전환) 비교.

**현재 방침(2026-08-27)**: **일단 2.4GHz 데이터 수집에 집중.** 스티어링은 데이터·모델·데모 파이프라인이 갖춰진 뒤. 밴드는 2.4GHz 유지(기존 5574행 그대로 살림), 5GHz 전환 논의는 보류. API 명세엔 `POST /api/steer` / `POST /steer` 를 미리 넣어둠(`docs/yongsang/demo_api_spec.md` §9).

## 완료된 작업 (2026-08-26 밤 세션, Claude Code와 진행, 저녁 세션 이어서)

### 191 단독 60M/120초 재확인 — 이번엔 크래시 없이 완주, 191 개별 문제였을 가능성에 무게
저녁 세션 종료 시점 "다음 세션 최우선"이었던 "191 단독 안정성 재확인"을 이어서 바로 진행. AP는 세션 시작 시점 기준 uptime 5분(직전에 재부팅된 상태)이었고, 시작 전 ping 확인 결과 AP는 정상(0% 손실, 1~7ms)이었지만 191은 0% 손실이어도 지연이 25~1078ms로 크게 흔들려 와이파이 상태가 불안정해 보였음.

- **s21_60m_solo_test(191 단독, 60Mbps, 목표 120초)**: 이번엔 **크래시 없이 깨끗하게 완주**. 실부하가 19:55:04~19:57:02(약 118초, 목표에 거의 정확히 부합)로 잡혔고 `connected_clients`가 그 구간 내내 2(191+노트북)로 안정적 유지, AP도 끝까지 정상. 51행, label 0×22/1×25/2×4, **label 3은 0개**(congestion 최고 0.7314로 문턱 0.75에 근접했으나 미달)
- **해석**: 저녁 세션에서 콤보(27초)·단독(19초) 두 번 다 191이 짧게 죽었던 것과 달리 이번엔 정상 완주 — "191 개별 문제(앱/와이파이)"였을 가능성에 다시 무게가 실림. 다만 이번엔 label 3을 못 만들었으므로 데이터 기여도는 낮고, 순수하게 안정성 재확인 목적에 부합하는 실행이었음

시나리오별 신규 수집: `s21_60m_solo_test` 51(label3 0). `metrics_v2.csv` 4079행 → **4130행**(4129 data rows), raw label 3 개수는 73개로 변화 없음.

### 콤보(191+S26) 60/60/120초 재시도(rep2) — S26 와이파이 재연결 후 크래시 없이 완주
191 단독 재확인 직후 바로 콤보로 복귀. 시작 전 확인 결과 S26(103)이 AP 와이파이에 안 붙어있어(ping 무응답) 사용자가 재연결 후 진행.

- **combo_s21_60m_s26_60m_120s_rep2**: 크래시 없이 완주, AP uptime도 재부팅 없이 계속 유지(시작 13분→종료 19분). `connected_clients=3`(191+S26+노트북) 전 구간 유지. 실부하 20:02:08~20:04:04(약 116초, 목표 120초에 근접). 34행, label 0×15/1×6/2×13, **label 3 0개**(congestion 최고 0.7446 — 문턱 0.75에 매우 근접했으나 미달)
- **해석**: 191 단독뿐 아니라 콤보에서도 이번엔 안정적으로 완주 — 저녁 세션 내내 반복되던 불안정성이 "191/S26 와이파이 재연결 여부"에 크게 좌우된다는 정황이 강해짐. 다중 station 자체보다 개별 폰의 와이파이 연결 상태 점검이 선행 조건으로 봐야 할 듯

시나리오별 신규 수집: `combo_s21_60m_s26_60m_120s_rep2` 34(label3 0). `metrics_v2.csv` 4130행 → **4164행**(4163 data rows), raw label 3 개수는 73개로 변화 없음.

### 데이터 반영 안 함(재라벨링/재변환/재학습 미실시)
두 신규 수집분 다 label 3 0개라 재학습 우선순위 낮음, 이번 세션에서 돌리지 않음.

### 콤보(191+S26) 60/60/5분 재시도(rep12) — 크래시 없이 완주, label 3 3개 신규 확보
120초 콤보(rep2)가 문턱 근접(0.7446)까지 갔던 걸 보고 검증된 스위트스팟인 5분(300초)으로 연장해서 바로 재시도.

- **combo_s21_60m_s26_60m_rep12**: 크래시 없이 완주, AP uptime도 재부팅 없이 계속 유지(21분→53분). 실부하는 20:06:42~20:11:40(약 5분, 목표에 정확히 부합) — 이후 사용자가 "다 했다"고 알릴 때까지 수집기를 못 끄고 약 27분 더 유휴 상태로 돌아감(8/25 새벽 세션의 "끝났다 보고 후 방치" 패턴 재발, 데이터 품질엔 문제없음). 총 412행(대부분 유휴), label 0×370/1×13/2×26/**3×3**
- **label 3 상세**: congestion 0.7826/0.8870/0.7659, 셋 다 `channel_occupancy_percent=100.0`(delta 방식) 포화 순간에 retry/jitter가 같이 튀며 확보됨(retry_failed 최대 10,989)
- **해석**: 191/S26 둘 다 세션 시작 시 와이파이 재연결을 거친 뒤로는 120초·5분 콤보 모두 크래시 없이 안정적으로 완주됨 — 저녁 세션의 반복 불안정성이 다중 station 자체보다는 개별 폰 와이파이 연결 상태에 크게 좌우된다는 가설이 더 강해짐

시나리오별 신규 수집: `combo_s21_60m_s26_60m_rep12` 412(label3 3). `metrics_v2.csv` 4164행 → **4576행**(4575 data rows), raw label 3 개수 73개 → **76개**

### 데이터 반영 대기 (재라벨링/재변환/재학습 아직 미실시)
label 3 신규 3개 확보했으니 다음 조치로 재라벨링→재변환→재학습 반영 권장(이번 세션에서는 아직 실행 안 함).

### 7분(420초) 콤보 시도(rep13) — 191 iperf3가 TCP 스톨, 정크로 판정 후 삭제
5분 성공(rep12) 직후 7분으로 연장 시도. 그런데 수집 중 throughput이 시종일관 0에 가까워서 사용자가 의문 제기, 191 폰 화면(스크린샷)을 직접 확인.

- **스크린샷 확인 결과**: 191의 iperf3 클라이언트가 **300초 내내 연결 자체는 유지했지만(재연결 없음) TCP 전송이 거의 완전히 스톨**됨 — 초당 `0.00 Bytes`가 대부분이고 총 전송량 300초간 14.6MB(평균 409Kbits/sec, 목표 60Mbps 대비 1% 미만), Retr(재전송) 258회. AP 크래시나 와이파이 단절이 아니라 **TCP 스트림 자체가 죽어있었던 패턴** — 안드로이드가 앱을 백그라운드로 인식해 네트워크 소켓을 스로틀링했을 가능성이 유력한 새 가설로 추가됨(기존 "191 개별 문제" 가설에 구체적 메커니즘 후보 하나 추가)
- **중요 정정**: 사용자 확인 결과 **7분(420초) 목표 테스트는 실제로 실행된 적이 없음** — 스크린샷은 300초짜리 별도/이전 실행이었던 것으로 보이고, `collect_metrics.py`로 수집한 `combo_s21_60m_s26_60m_rep13`(31분, 427행)은 콤보 부하가 사실상 전혀 안 걸린 정크 데이터였음(throughput 전 구간 최대 1.91Mbps, label 0이 424/427)
- **처리**: `--help` 정크 삭제 사례와 동일한 방식으로 `metrics_v2.csv`에서 `scenario==combo_s21_60m_s26_60m_rep13` 427행 전체 삭제(백업 후 awk 필터, 사용자 승인 받고 진행). `metrics_v2.csv` 5003행(rep12 반영 후) → **4576행**(rep12까지의 상태로 복귀, raw label 3은 76개 그대로 유지)

### 191 단독 20초 재검증 + 7분 콤보 재시도(rep13_v2) — 191 문제는 "백그라운드 스로틀링"이 아니라 "S26과의 채널 경쟁(capture effect)"였을 가능성
rep13 정크 삭제 후, 사용자가 191의 Wi-Fi 절전모드/앱 배터리 제한을 점검했다고 확인. 재시도 전 짧게 191 단독 20초로 먼저 검증.

- **s21_60m_solo_quickcheck_20s**: 191 단독으로는 21:22:30~21:22:47(약 17초) 구간에서 20~38Mbps까지 정상적으로 올라감 — 지난번 같은 완전 스톨(0.4Mbps) 재현 안 됨. 설정 조정이 효과 있어 보여서 바로 7분 콤보 진행 결정
- **combo_s21_60m_s26_60m_rep13_v2(목표 420초)**: 시작하자마자 사용자가 "191은 거의 0으로 뜬다"고 재보고 — 그런데 수집기 쪽 집계 throughput은 오히려 정상적으로 잡힘(21:24:50~21:31:48 약 7분간 20~93Mbps, 최고 93.39Mbps). **즉 완전 스톨이 아니라 S26이 채널을 거의 독점하고 191이 굶는 비대칭 현상**으로 재해석됨 — 191이 단독일 땐 정상 작동하다가 S26과 동시 경쟁이 붙는 순간부터 죽는 패턴이 이번에도 재현되어, "191 개별 문제"보다는 **Wi-Fi capture effect(신호가 강한 S26이 채널 점유, 신호가 약한 191이 배제됨)**가 더 유력한 설명으로 부상. 191은 그동안 ping 지연이 S26보다 계속 크고 불안정했음(RSSI 열세로 추정)
- **결과**: 크래시 없이 7분 완주. 90행, label 0×19/1×30/2×41, **label 3 0개**(congestion 최고 0.7008, 문턱 0.75 근접했으나 미달). 다만 이 데이터는 "191+S26 대칭 60/60 부하"가 아니라 "S26 위주 비대칭 부하"에 가까움 — 실측 자체는 유효하지만 시나리오명이 내포하는 대칭성과는 다소 어긋남

시나리오별 신규 수집: `s21_60m_solo_quickcheck_20s`(행 수 미집계, 소규모) / `combo_s21_60m_s26_60m_rep13_v2` 90(label3 0). `metrics_v2.csv` 4576행 → **4685행**(4684 data rows)

### 콤보 부하 상향 시도(75/85, 120초) — 잘못된 시나리오명 정리 + 실제 부하는 191=75M/S26=85M(의도치 않은 비대칭)
7분 rep13_v2 이후 부하를 60/60보다 올려서(75/75 예정) 시도. 진행 중 목표 시간을 300초→120초로 변경했는데, 이미 유휴 상태로 17~20행이 옛 이름(`combo_s21_75m_s26_75m_5min`)에 찍혀있어 삭제(전부 유휴, `--help`/`rep13` 정크와 동일 성격) 후 `combo_s21_75m_s26_75m_120s`로 수집기 재시작.

- 사용자가 S26 쪽 명령에서 실수로 75가 아니라 85를 입력 — 실제 부하는 **191=75M / S26=85M(비대칭)**이었음. 실부하 21:37:07~21:38:32(약 85초, 목표 120초에 못 미침, 한쪽이 일찍 끝난 것으로 추정). 22행, label 0×13/1×3/**2×6**, label 3 0개(occupancy는 46~58%에 그쳐 미포화, 대신 retry_failed_score가 여러 번 1.0 포화 — 재전송은 폭증했지만 채널 점유율 자체는 안 오른 패턴)
- **시나리오명 정정**: `s26_100m_test_2` 사례와 동일한 방식으로 `combo_s21_75m_s26_75m_120s` → **`combo_s21_75m_s26_85m_120s`**로 일괄 치환(22행, 모델 입력에는 영향 없음, 표기 정확성 목적)
- **해석**: 부하를 60/60보다 올려도(75/85) 재전송만 폭증하고 occupancy/congestion은 오히려 60/60 콤보 때보다 낮았음 — 기존 "60/60이 스위트스팟, 그 이상은 나빠짐" 결론과 일치하는 추가 근거

시나리오별: `combo_s21_75m_s26_75m_5min`(전부 유휴, 20행 삭제) / `combo_s21_75m_s26_85m_120s` 22(label3 0, 시나리오명 정정 완료). `metrics_v2.csv` 4685행 → **4707행**(4706 data rows), raw label 3 76개로 변화 없음

### 콤보 75/75 대칭 재시도(120초) — label 3 3개 확보, 새로운 스위트스팟 후보
75/85 비대칭 시도 직후, 오타 없이 정확히 대칭(75/75)으로 재시도.

- **combo_s21_75m_s26_75m_120s_v2**: 크래시 없이 완주, AP uptime 계속 유지. 실부하 21:41:39~21:43:31(약 112초, 목표 120초에 근접), 최고 결합 처리량 **127.72Mbps**. 22행, label 0×5/1×3/2×11/**3×3**(congestion 0.8353/0.7765/0.7745)
- **해석**: 75/85 비대칭(같은 세션 직전 시도)에선 label 3이 0개였는데, 정확히 대칭(75/75)으로 맞추자마자 22행이라는 적은 표본에도 label 3이 3개나 나옴 — **"60/60이 스위트스팟"이라는 기존 결론에 반례**. 대칭성이 절대 부하 크기보다 더 중요한 변수일 가능성이 새로 제기됨. 다만 표본이 1회(22행)뿐이라 재현성 확인 필요

시나리오별 신규 수집: `combo_s21_75m_s26_75m_120s_v2` 22(label3 3). `metrics_v2.csv` 4707행 → **4729행**(4728 data rows), raw label 3 76개 → **79개**

### 75/75 재현성 검증(rep2) — label 3 재현 안 됨, 소표본 변동성으로 판단
75/75_v2(label 3 3개)의 재현성을 확인하러 바로 재시도. 시작 전 191이 AP 와이파이에서 완전히 끊겨있어(연결 자체 없음, ping "대상 호스트에 연결할 수 없음") 사용자가 재연결 후 진행.

- **combo_s21_75m_s26_75m_120s_rep2**: 크래시 없이 완주, AP uptime 계속 유지. 실부하 22:21:07~22:23:05(약 118초, 목표 120초에 근접), 최고 결합 처리량 114.73Mbps. 23행, label 0×11/**2×12**, **label 3 0개**(congestion 최고 0.746 — 문턱 0.75 바로 아래)
- **해석**: 직전 75/75_v2(label 3 3/22, 13.6%)와 이번(0/23)이 크게 엇갈림 — "75/75가 새 스위트스팟"이라는 결론은 아직 성급함, 22~23행 수준의 표본에서 흔한 소표본 변동성일 가능성이 큼. 두 번 다 congestion 최고치가 0.746~0.89 범위로 문턱(0.75) 근처에서 왔다갔다 하는 걸 보면 75/75 자체는 문턱 근처를 계속 건드리는 유효한 구간인 것은 맞아 보임 — 반복 횟수를 더 늘려야 결론 가능

시나리오별 신규 수집: `combo_s21_75m_s26_75m_120s_rep2` 23(label3 0). `metrics_v2.csv` 4729행 → **4752행**(4751 data rows), raw label 3 79개로 변화 없음

### 75/75 3차 시도(7분) — label 3 2개 추가, 종료 후 53분 유휴 방치
rep2(0개) 다음으로 지속시간을 7분(420초)으로 늘려서 재시도.

- **combo_s21_75m_s26_75m_rep3_7min**: 크래시 없이 완주, AP uptime 계속 유지. 실부하 22:54:05~23:01:00(약 415초, 목표 420초에 거의 정확히 부합). label 0×649/1×44/2×21/**3×2**(congestion 0.7543 이상)
- **주의**: 부하 종료 후 수집기를 즉시 못 꺼서 **약 53분간 유휴 상태로 계속 수집됨**(전체 범위 22:53:48~23:53:23, 총 716행 중 649행이 유휴 label 0) — 데이터 품질엔 무해하지만 8/25 새벽 세션 이후 반복되는 "종료 신호 지연" 패턴이 다시 나타남
- **75/75 누적 결과**: rep1(3/22, 13.6%) → rep2(0/23, 0%) → rep3(2/~68 실부하 행, 약 3%) — 3회 합산 **5개**로 60/60보다는 나은 편이나 rep2처럼 완전히 0으로 나오는 경우도 있어 재현성이 완벽하지는 않음. 그래도 3회 모두 크래시 없이 안정적으로 완주됐다는 점은 고무적

시나리오별 신규 수집: `combo_s21_75m_s26_75m_rep3_7min` 716(label3 2). `metrics_v2.csv` 4752행 → **5468행**(5467 data rows), raw label 3 79개 → **81개**

### 80/80 10분 시도 — 완전 크래시(물리 재부팅 필요), 오늘 세션 마지막 시도
75/75 3연속(rep1~rep3) 이후 마지막으로 부하를 80/80, 지속시간을 10분(600초)까지 올려서 시도. 과거 기록상 80/80·장시간 둘 다 불안정성이 컸던 조합이라 미리 위험성 안내 후 진행.

- **combo_s21_80m_s26_80m_10min**: 실부하 00:46:43~00:50:39(약 4분 만에 중단, 목표 10분 중 40%만 진행) 시점에서 **AP가 완전히 응답 불능 상태로 크래시**(ping 100% 손실, SSH 불가). 점진적 저하 없이 갑자기 끊김 — retry는 10k~40k대로 이미 높았지만 진행 중이었고 특별한 폭주 신호 없이 바로 죽음. 23행, label 0×2/1×3/**2×18**, label 3 0개. **AP 물리 재부팅 필요**(이 세션 종료 시점까지 미완료)
- **해석**: "부하를 올릴수록(80/80)/오래 끌수록(10분) 불안정성이 커진다"는 기존 결론과 일치하는 크래시 사례 추가. 75/75(3회 모두 크래시 없음)와 80/80(1회 크래시)을 비교하면 75가 안전 상한에 가깝고 80은 이미 위험 구간일 가능성

시나리오별 신규 수집: `combo_s21_80m_s26_80m_10min` 23(label3 0, 크래시로 중단). `metrics_v2.csv` 5468행 → **5491행**(5490 data rows), raw label 3 81개로 변화 없음

### 실시간 추론 시 폴링 지연 문제 논의 (설계 메모, 코드 변경 없음)
세션 마무리 즈음 "나중에 실측 실시간 판별할 때 지금처럼 SSH 폴링이 느리면 실시간이 아니게 되는 거 아니냐"는 질문이 나옴.

- **문제의 핵심**: `collect_metrics.py`가 매 루프마다 AP에 SSH로 새로 접속해서 `iw` station/survey 정보를 가져오는데, 이 SSH 관리 트래픽이 AP가 서비스 중인 **같은 무선 채널을 그대로 타고 간다** — 그래서 혼잡할수록(측정하려는 대상 그 자체) SSH 응답도 같이 느려지는 자기참조적 구조. 이게 지금까지 관측된 "폴링 지연 스파이크(최대 156초)"의 근본 원인일 가능성이 높음
- **논의된 방향(결정 아님)**: (1) 통계 수집 로직을 AP 자체에서 로컬로 실행(OpenWrt 위에서 `iw`/`/proc`를 직접 읽어 로컬로 노출) — 네트워크 왕복 자체가 없어짐, 다만 OpenWrt 리소스 제약으로 Python 대신 가벼운 셸/C로 재작성 필요할 수 있음. (2) 관리 트래픽을 무선이 아닌 유선 경로로 분리 — 측정 채널과 결과 전달 채널을 물리적으로 분리
- **상태**: 아직 방향만 논의, 코드/설계 변경 없음. 다음에 ONNX/Pi 배포 파이프라인을 붙일 때(README_AP_V2.md의 "알려진 한계" 항목) 반드시 같이 고려해야 할 문제

### 다음 세션 최우선
- [ ] **AP 물리 재부팅 필요** — 80/80 10분 시도에서 완전 크래시(ping 100% 손실)된 채로 이 세션 종료
- [ ] **실시간 추론용 폴링 아키텍처 재설계 필요** — 현재 SSH 폴링 방식은 혼잡 상황에서 자기 자신이 느려지는 구조라 실시간 배포에 부적합. AP 온보드 로컬 수집 vs 유선 관리채널 분리 중 방향 결정 필요(ONNX/Pi 배포 파이프라인 작업 시작 전에 결정할 것)
- [ ] 75/75가 안전 상한에 가깝고 80/80은 이미 크래시 위험 구간으로 보임 — 다음에도 부하를 올릴 땐 75까지만 유지하고 80 이상은 지양할 것
- [ ] "다 했다" 보고 후 수집기 종료를 더 철저히 할 것 — 이번엔 53분이나 유휴 방치됨(과거 최고 기록인 8/25 새벽 세션의 1시간+에는 못 미치지만 여전히 반복되는 문제)
- [ ] 75/75 콤보 결과가 60/60보다 대체로 낫지만 재현성이 완벽하진 않음(3회 중 1회는 0개) — 표본을 더 쌓아서 60/60 대비 통계적으로 유의한 차이인지 검증 계속할 것
- [ ] 세션 중간에 191이 AP 와이파이에서 완전히 끊기는 사례가 또 발생함(콤보 도중이 아니라 대기 중에) — 매 시도 전 ping 확인 습관을 계속 유지할 것
- [ ] **"191 개별 문제" 가설을 "Wi-Fi capture effect(신호 강한 S26이 채널 독점)"로 갱신 검토** — `docs/yongsang/ap_crash_analysis.md`에 반영 필요
- [ ] **재라벨링(`relabel_metrics_v2.py`) → 재변환(`prepare_ap_metrics_dataset.py`) → 재학습(`train_ap_early_exit.py --class-weight-power 1.0`) → 평가 반영할 것** — raw label 3이 73→81개로 늘었음(이번 세션 순증 8개: rep12 3개 + 75/75_v2 3개 + rep3_7min 2개, rep2는 0개)
- [ ] `combo_s21_60m_s26_60m_120s_rep1`, `combo_s21_60m_s26_60m_rep13_v2`(191 굶는 비대칭) 등 시나리오명이 실제와 다소 어긋나는 케이스들 나중에 한 번에 정리 검토

## 완료된 작업 (2026-08-26 저녁 세션, Claude Code와 진행)

### AP 재부팅 후 60/60 콤보 재시도 3회 — 이번엔 AP보다 191 폰이 더 자주 죽음, label 3은 0개
새벽 세션 rep10(500초 시도)에서 완전 크래시된 채로 종료됐던 AP를 세션 시작 시 물리 재부팅(uptime 8분으로 확인). 이번 세션부터 iperf3 서버(포트 5201=191, 5202=S26)를 Claude Code가 직접 기동/종료하는 방식은 그대로 유지.

- **rep11(목표 300초 60/60 콤보)**: 재부팅 직후 첫 시도인데도 **다시 완전 크래시**. retry_delta가 87,033/90,567까지 폭주한 직후 AP가 100% ping 손실로 완전히 끊김(SSH 타임아웃). 검증됐다던 300~420초 스위트스팟 안(대략 100초 안팎)에서 무너져서, 직전 세션의 "500초라서 무너졌다"는 결론과 배치됨 — **지속시간 자체보다 "재부팅 후 몇 번째 시도인가"(누적 피로)가 더 중요한 변수일 가능성**이 다시 제기됨(8/23 저녁 세션 가설과 같은 방향). 78행, label 3 0개. **AP 물리 재부팅 필요 → 재부팅 완료**(uptime 25분으로 재확인)
- **120s_rep1(목표 120초 60/60 콤보)**: 재부팅 후 재시도. 실부하는 17:38:40에 시작됐는데 `connected_clients`가 3→2로 떨어진 시점이 17:39:07 — **실부하 시작 27초 만에 191만 이탈**. 이번엔 **AP 자체는 끝까지 정상**(SSH/ping 정상, 재부팅 불필요) — AP 크래시가 아니라 191 폰(iperf3 클라이언트 또는 와이파이) 쪽 문제였음. 결과적으로 "콤보 27초 + S26 단독 90초"가 섞인 데이터가 됨(시나리오명은 정정 안 하고 그대로 유지, 원할 때 나중에 `s26_100m_test_2` 사례처럼 정정 가능). 18행, label 3 0개
- **s21_60m_test_120s_v2(191 단독 120초 재시도)**: S26 없이 191 혼자만 시도했는데도 **실부하가 17:49:59~17:50:18 약 19초만 잡히고 이후 유휴로 복귀**. AP는 이번에도 끝까지 완전 정상(uptime 38분 연속). **191이 콤보든 단독이든 상관없이 이번 세션 내내 짧게 죽는 패턴이 반복**돼서, "다중 station이 핵심 변수"라는 최근 가설과 별개로 **191 개별 문제(앱/와이파이 안정성)가 다시 유력해짐** — 세션 종료 시점까지 재확인 못 함, 다음 세션에서 191 와이파이 재연결/앱 재시작 후 우선 확인 필요. 25행, label 3 0개

세 시도 다 label 3 0개 — 오늘은 순수하게 크래시/불안정성 재현 및 원인 분리에 그침, 데이터 품질 기여는 낮음.

### 정크 데이터 발견 및 정리: `--help` 시나리오 35행 삭제
세션 초반 Claude가 `collect_metrics.py`의 사용법을 확인하려고 `python collect_metrics.py --help`를 실행했는데, 스크립트가 `--help`를 옵션으로 인식하지 않고 `sys.argv[1]`을 시나리오명으로 그대로 받아버려서 실제로 수집이 시작돼버림(AP는 이미 재부팅 후라 정상 응답 중이었고 폰은 아직 부하 전이라 전부 유휴 데이터). Claude가 뒤늦게 발견해서 프로세스를 죽였지만 이미 35행이 `metrics_v2.csv`에 `scenario="--help"`로 찍혀 있었음. 데이터 자체는 진짜 유휴 실측이라 문제없지만(`normal_idle`과 성격 동일) 시나리오명이 무의미해서 사용자 판단으로 **삭제**(`awk`로 `scenario!="--help"`인 행만 남기고 원본 교체, 수집기가 그 시점에도 계속 실행 중이었지만 파일 교체 후에도 정상적으로 이어서 씀을 확인함).

시나리오별 신규 수집: `combo_s21_60m_s26_60m_rep11` 78(label3 0) / `combo_s21_60m_s26_60m_120s_rep1` 18(label3 0, 콤보 27초만 유효) / `s21_60m_test_120s_v2` 25(label3 0, 실부하 19초만) — 실측 신규 121행, `--help` 정크 35행은 수집 후 삭제. `metrics_v2.csv` 3982행(새벽 세션 종료 시점) → **4079행**(4078 data rows). **raw label 3 개수는 73개로 변화 없음**(오늘은 label 3 신규 확보 없음).

### 데이터 반영 안 함(재라벨링/재변환/재학습 미실시)
오늘 신규 수집분이 전부 label 3 0개라 재학습 우선순위가 낮다고 판단, 재라벨링/재변환/재학습은 이번 세션에서 돌리지 않음. 다음 세션에서 신규 label 3 데이터가 좀 더 쌓인 뒤 한 번에 반영 권장.

### 다음 세션 최우선
- [ ] **191 폰 상태 점검 우선** — 오늘 콤보(27초)와 단독(19초) 두 번 다 191이 짧게 죽음, AP는 두 번 다 완전 정상이었음. 와이파이 재연결/iperf3 앱 재시작 후 191 단독으로 먼저 안정성 재확인할 것(바로 콤보로 가지 말 것)
- [ ] AP는 세션 종료 시점 기준 정상(재부팅 이력: 이번 세션 중 1회, rep11 크래시 직후). 재부팅 상태 유지되는지 다음 세션 시작 시 ping/SSH로 먼저 확인
- [ ] rep11 크래시는 300초 이내(추정 100초 안팎)에 발생 — "재부팅 후 반복 시도 자체가 누적 피로를 유발한다"는 8/23 가설이 다시 힘을 얻는 중, 앞으로 재부팅 직후 1~2회 시도로 제한하고 그 이후엔 쉬는 텀을 두는 방식도 고려할 것
- [ ] `combo_s21_60m_s26_60m_120s_rep1` 시나리오명이 부정확(실제 콤보는 27초뿐, 나머지 90초는 S26 단독) — 나중에 정리할 때 `s26_100m_test_2` 사례처럼 정정할지 결정 필요
- [ ] test label 3 표본이 여전히 11개 수준에 머물러 있음(오늘 신규 확보 없음) — 두 자릿수 중반 목표를 위해 계속 반복 필요, 단 191 안정성 확인이 선행돼야 함

## 완료된 작업 (2026-08-26 새벽 세션, Claude Code와 진행)

### 60/60 콤보 5회 반복(rep6~rep10) — AP 재부팅 필요한 완전 크래시 재현, label 3 recall 54.5%로 최고 경신
계속 191+S26 콤보, 목적지는 이 노트북(무선, `192.168.8.226`). 이번 세션부터 iperf3 서버 2개(포트 5201/5202, 폰별 1개씩 필요 — 동시 접속은 서버 인스턴스가 분리되어야 함)를 Claude Code가 직접 기동/종료하는 방식으로 바꿈.

- **rep6(목표 300초)**: Claude가 자동종료 타이머를 "수집기 실행 시점" 기준으로 걸었는데, 실제 폰 부하는 그로부터 3.5분 뒤에나 시작돼서 **타이머가 실제 부하 시작 후 약 100~140초 만에 조기 발동, 수집기·iperf3 서버를 오발 종료**시킴 — AP 자체는 크래시 아니었음(SSH 정상, uptime 연속 확인). 69행, label 3 3개 확보. **교훈: 자동종료 타이머는 실제 부하 시작 시점 기준으로 걸거나, 아예 쓰지 말고 사용자가 "다 됐어"라고 알릴 때 즉시 종료하는 쪽이 안전함** — 이후 rep7부터는 타이머 방식을 버리고 수동 종료로 전환
- **rep7(재시도, 목표 300초)**: 수동 종료 방식으로 정상 진행. 29행, label 3 3개
- **rep8(목표 300초)**: 완주, 5분 29초. 52행, label 3 1개 — rep6/7보다 적어 회당 label 3 개수가 여전히 크게 흔들리는 패턴 재확인
- **rep9(목표 300초)**: 완주, 6분. 도중 폰 쪽에서 "unable to read from stream socket: Try again" 에러가 떴지만 이전에 기록된 패턴대로 **폰 클라이언트가 스트림 종료 시점 근처에서 뱉는 무해한 에러**였음 — connected_clients=3이 57행 내내 유지됐고 AP도 SSH 정상(uptime 연속). 57행, **label 3 4개(오늘 최다)**
- **rep10(500초로 연장 시도)**: 이번엔 **진짜 완전 크래시** — throughput이 84.86Mbps에서 급격히 붕괴, retry_delta 10,935→33,288→39,060으로 폭주, latency/jitter 0으로 죽은 직후 SSH 완전 타임아웃(ping 100% loss). **8행에서 중단, label 3 0개. AP 물리 재부팅 필요**(이 세션 종료 시점까지 미완료). **500초는 검증된 60/60 스위트스팟(300~420초) 범위를 벗어나서 예상보다 빨리 무너진 것으로 추정** — 다음 시도는 300~420초로 복귀 권장

시나리오별 신규 수집: `combo_s21_60m_s26_60m_rep6` 69(label3 3) / `rep7` 29(label3 3) / `rep8` 52(label3 1) / `rep9` 57(label3 4) / `rep10` 8(label3 0, 크래시로 중단) — 합계 215행 신규. `metrics_v2.csv` 3766행(8/25 새벽 기준) → **3981행**(3980 data rows), label 3 62개 → **73개**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py`: 가중치 변경 없었으므로 재라벨링 전/후 분포 동일 확인(0:1997, 1:935, 2:976, 3:73)
- `prepare_ap_metrics_dataset.py` 재변환: train 2543 / val 544 / test 547, label 3 train 49 / val 10 / **test 11**(9→11, 계속 증가 중)
- `power=1.0`으로 재학습(best epoch 24, val balanced acc 73.4%) → 평가: **전체 정확도 87.2%(fixed)/87.4%(dynamic)**(직전 최고 89.6%보다는 소폭 하락), Label 0 97.8% / Label 1 76.3~77.1% / Label 2 78.5~80.0% / **Label 3 54.5%(6/11) — 역대 최고 recall**
- 체크포인트: `project/checkpoints/ap_v2/`, 리포트: `project/results/yongsang/ap_v2_eval_report.txt`
- 오늘 수집·평가 결과를 요약한 대시보드 아티팩트 생성(세션별 정확도/recall 추이, 오늘 반복별 레이블 구성, fixed/dynamic 레이블별 정확도, 원본 분포)

### 다음 세션 최우선
- [ ] **AP 물리 재부팅 필요** — rep10(500초 콤보)에서 SSH 완전 타임아웃으로 크래시된 채로 이 세션 종료. 재부팅 후 곧바로 500초 이상 시도하지 말고 검증된 300~420초 구간으로 복귀할 것
- [ ] 자동종료 타이머를 걸 때는 반드시 "실제 부하 시작 시점" 기준으로 걸 것(수집기 시작 시점 기준으로 걸면 오발 종료 위험) — 아니면 타이머 없이 사용자 신호로 즉시 종료하는 쪽이 안전
- [ ] test label 3이 11개로 늘었지만 두 자릿수 중반(15~20개) 목표까지는 아직 — 300~420초 60/60 반복을 계속할 것
- [ ] 폰 쪽 "unable to read from stream socket" 에러는 스트림 종료 시점 근처의 무해한 증상으로 재확인됨(connected_clients/AP SSH로 매번 검증하는 습관 계속 유지)

## 완료된 작업 (2026-08-25 새벽 세션, Claude Code와 진행, 8/24 밤 세션 이어서)

### 60/60 반복 + 80/80 재시도 — "60/60을 5~7분 정도로 돌리는 게 최고 효율", 단일 실행 label 3 5개 확보
1시간 휴식(AP 자연 재부팅됨, uptime 28분에서 재시작) 후 재개. 계속 191+S26 콤보, 목적지는 이 노트북(무선).

- **60/60(120초) 3회차**: S26 쪽에 "unable to read from stream socket" 에러가 떴지만 실제로는 connected_clients=3이 처음부터 끝까지 유지되며 정상 진행됨(에러는 종료 시점 근처에 뜬 것으로 추정) — **label 3 2개**(congestion 0.9495 포함, 당시 역대 최고), retry 최대 46,590(역대 최고)
- **60/60(120초) 4회차**: 정상 진행(connected_clients=3, ~107초), label 3 0개(최고 congestion 0.726)
- **80/80(120초) 재시도**: 완주, label 3 0개, 폴링 지연 최대 41초(1차 시도의 110초보다 양호) — **60/60이 80/80보다 낫다는 패턴이 다시 확인됨**(오늘 60/60 4회 중 3회 label3 확보, 80/80은 2회 다 0개)
- **60/60(목표 300초) 5회차 — 우발적으로 35분간 수집**: iperf3 자체는 5분 근처(23:52:16~23:59:46, 약 7.5분)에 끝났지만 수집기를 안 끄고 계속 켜둬서 그 뒤로도 유휴 상태로 데이터가 쌓임(총 35분, 489행). **실질 부하 구간(7.5분) 안에서 label 3이 5개**나 나옴 — 단일 실행 기준 역대 최다, congestion 최고 **0.9632**(역대 최고 경신)
- **60/60(목표 420초, "7분") 6회차**: 도중 폰 하나가 크래시("크러시 떴네")했지만 **AP 자체는 안 죽었음**(SSH 정상, uptime 연속 유지) — connected_clients가 2로 떨어진 것으로 봐서 폰 쪽 iperf3 크래시였던 것으로 판단. 이후 수집기를 못 끈 채로 **1시간 이상(00:28~01:43) 유휴 상태로 계속 수집**됨(668행, 대부분 label 0, label 3 0개)
- **작업 실수 기록**: 이번 세션에서 두 번이나 "테스트 끝났다"는 보고 후 수집기/iperf3 서버를 즉시 안 끄고 방치해서 유휴 시간이 몇십 분씩 raw CSV에 섞여 들어갔다(5회차 35분, 6회차 1시간+). 데이터 자체는 정상 라벨링되어 문제는 없지만, 다음부터는 "끝났어" 보고 즉시 프로세스 종료를 더 철저히 할 것
- **60/60을 5~7분 정도로 돌리는 게 지금까지 가장 효율이 좋았다**(2분 버전보다도 절대 개수가 많음) — 너무 짧으면(2분) 기회가 적고, 너무 길면(10분+) 폴링 지연·크래시 리스크만 커진다는 게 다시 확인됨

시나리오별 신규 수집: `combo_s21_60m_s26_60m_rep3` 32 / `combo_s21_60m_s26_60m_rep4` 43 / `combo_s21_80m_s26_80m_rep2` 25 / `combo_s21_60m_s26_60m_rep5` 489 / `combo_s21_60m_s26_60m_7min` 668 — 합계 1257행 신규(대부분 유휴). `metrics_v2.csv` 2510행 → **3767행**(3766 data rows), label 3 55개 → **62개**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py`: 가중치 변경 없었으므로 재라벨링 전/후 분포 동일 확인(0:1916, 1:900, 2:888, 3:62)
- `prepare_ap_metrics_dataset.py` 재변환: train 2426 / val 520 / test 521, label 3 train 41 / val 9 / **test 9**(드디어 한 자릿수 후반대 진입)
- `power=1.0`으로 재학습(best val balanced acc 71.1%) → 평가: **전체 정확도 89.6%(fixed/dynamic 동일)**, Label 0 99.2% / Label 1 87.3% / Label 2 75.6% / **Label 3 33.3%**(9개 중 3개) — 지금까지 최고 전체 정확도
- 체크포인트: `project/checkpoints/ap_v2/`, 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

### 다음 세션 참고
- [ ] **"끝났어" 보고 즉시 iperf3 서버/수집기 프로세스 종료를 철저히 할 것** — 이번 세션에서 두 번이나 방치돼서 유휴 시간이 raw CSV에 몇십 분씩 섞임(데이터 품질엔 문제없지만 비효율)
- [ ] 60/60을 5~7분 길이로 반복하는 게 최고 효율 — 이 길이로 계속 반복해서 label 3 표본을 늘릴 것
- [ ] test label 3이 9개로 늘어남 — 두 자릿수 중반(15~20개) 목표에 가까워지는 중
- [ ] 폰 쪽 iperf3가 종종 "unable to send/read" 류 에러로 끊기는 사례가 반복됨(AP 크래시와는 무관) — 실제 데이터가 잘 잡혔는지 매번 connected_clients/timestamp로 확인하는 습관 필요

## 완료된 작업 (2026-08-24 밤 세션, Claude Code와 진행, 저녁 세션 이어서)

### 콤보 추가 스위핑 — 60/60 재확인, 비대칭 조합(80/60) 약함, 100/100도 타이밍만 맞으면 label 3 가능
저녁 세션 커밋 이후 이어서 진행. 목적지는 계속 이 노트북(무선), AP는 저녁 세션 중 재부팅한 뒤로 계속 무재부팅 상태(세션 끝 uptime 1시간 28분).

- **60M/60M 10분 반복(2회차)**: 완주, label 3 3개/72행(4.2%) — 1회차 10분(2.6%)보다는 낫지만 2분 버전(9.8%)에는 못 미침. 폴링 지연 최대 **156초**(지금까지 최고 스파이크), 그래도 자연 복구
- **191=80M/S26=60M(비대칭, 120초)**: 완주, **label 3 0개**, retry_delta가 전 구간 0에 가까움 — 비대칭 부하는 대칭(60/60)보다 채널 경합이 약하게 발생한다는 패턴 추가 확인
- **191=100M/S26=100M 1차 시도**: 191쪽 iperf3가 "unable to send control message" 오류로 미접속 → S26 혼자만 100M을 보낸 **사실상 단독 데이터**였음이 확인됨(connected_clients 계속 2). 재라벨링 시 시나리오명을 `combo_s21_100m_s26_100m` → **`s26_100m_test_2`로 정정**(모델 입력 feature/label에는 영향 없음, 표기 정확성 목적)
- **191=100M/S26=100M 2차 시도**: 191 폰의 와이파이 자체가 그 시점에 불안정(핑 손실 66%)했던 게 원인으로 확인됨 — 와이파이 재연결 후 재시도. 이번엔 접속은 됐으나 두 폰이 실제로 겹친 구간이 22초뿐이었고(앞뒤 2.5분+는 거의 무신호) 폴링 지연 51초 스파이크도 있었음 → **label 3 0개**(최고 congestion 0.66)
- **191=100M/S26=100M 3차 시도**: 이번엔 깨끗하게 동시 시작됨 → 완주, **label 3 1개**(retry 19,179 / jitter 769ms — 오늘 최고 / congestion 0.764), 폴링 지연 최대 94초, 크래시 없음
- **결론**: 60/60 스위트스팟은 재확인됐고(짧게 반복이 장시간보다 효율적이라는 것도 재확인), 100/100도 **타이밍만 맞으면**(두 폰이 실제로 동시에 부하를 걸어야) label 3을 만들 수 있음이 확인됨 — 다만 콤보 실험은 폰 쪽 연결 안정성(와이파이 재연결, 시작 타이밍 어긋남)이 결과 재현성에 큰 영향을 준다는 게 이번에 뚜렷해짐

시나리오별 신규 수집: `combo_s21_60m_s26_60m_rep2` 72 / `combo_s21_80m_s26_60m` 24 / `s26_100m_test_2`(정정 전 `combo_s21_100m_s26_100m`) 29 / `combo_s21_100m_s26_100m_v2` 53 / `combo_s21_100m_s26_100m_v3` 35 — 합계 213행 신규. `metrics_v2.csv` 2288행(저녁 세션 커밋) → **2510행**(2509 data rows), label 3 51개 → **55개**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py`: 가중치 변경 없었으므로 재라벨링 전/후 분포 동일 확인(0:821, 1:875, 2:758, 3:55)
- `prepare_ap_metrics_dataset.py` 재변환(시나리오명 정정 반영), `power=1.0`으로 재학습(best val balanced acc 74.9%) → 평가: **전체 정확도 84.3%(fixed)/83.7%(dynamic)**, Label 0 97.0% / Label 1 84.4% / Label 2 74.8~77.6% / **Label 3 12.5~25.0%**(8개 중 1~2개)
- **주의**: Label 3 recall이 이번엔 크게 떨어졌다(직전 저녁 세션 57.1%→12.5~25%). test label 3이 8개뿐이라 1~2개 차이로 recall이 30%p 이상 흔들리는 전형적인 소표본 노이즈 — 전체 정확도(84%대)는 오히려 최고치를 찍었으니 숫자를 개별적으로 해석하지 말 것
- 체크포인트: `project/checkpoints/ap_v2/`, 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

### 다음 세션 참고
- [ ] 콤보 스위트스팟(60M/60M, 2분 내외)을 계속 반복해서 label 3 표본을 두 자릿수 중반 이상으로 늘릴 것 — 지금까지 누적 효율이 가장 좋음
- [ ] 콤보 실험 시 **두 폰이 실제로 동시에 시작했는지 확인**(와이파이 상태, 시작 타이밍)이 결과 재현성에 핵심적임 — 폰 와이파이가 불안정하면 콤보를 걸어도 실질적으로 단독 부하가 되어 label 3이 안 나옴
- [ ] Label 3 test 표본이 8개로 늘었지만 recall 변동성은 오히려 더 커짐 — 표본이 두 자릿수 중반(15~20개)이 될 때까지는 recall 숫자 자체보다 "얼마나 흔들리는지"를 계속 추적할 것

## 완료된 작업 (2026-08-24 저녁 세션, Claude Code와 진행)

## 완료된 작업 (2026-08-24 저녁 세션, Claude Code와 진행)

### AP 재부팅 후 191+S26 콤보 부하 스위핑 — "60/60이 스위트스팟" 발견
오후 세션 끝에 콤보(191+S26 동시)에서 크래시가 나서 AP를 물리 재부팅한 뒤 재개. 목적지는 계속 이 노트북(무선). 여러 부하 조합을 시도해서 안정성과 label 3 생성력을 동시에 비교.

- **40M/60M(90초)**: 완주, 크래시 없음, **label 3 0개**(최고 congestion 0.717). 다만 폴링 지연 70초 스파이크 한 번 — 낮은 부하인데도 지연은 오히려 컸음
- **60M/60M(120초)**: 완주, **label 3 4개**(41행 중, 9.8%) — 오늘 콤보 중 최고 성과. occupancy 100%/80%/77%/73% 네 순간 모두 잡힘, connected_clients=3(191+S26+노트북) 동시 경합. 폴링 지연 최대 13초로 양호
- **80M/80M(150초)**: 완주(크래시는 아님)했지만 **label 3 0개**, retry는 최고치(66,870)까지 튀었는데 occupancy(56~67%)와 안 맞물림. **폴링 지연 110초 스파이크**(SSH 완전 끊김 구간 포함, 최대 12.5초 연결 자체 실패) — 지금까지 관측된 것 중 가장 심한 비-크래시 증상. 물리 재부팅 없이 자연 복구됨
- **60M/60M을 10분(600초)으로 연장**: 완주, 크래시 없음, label 3 2개(76행 중, 2.6%) — **비율로는 120초 버전(9.8%)보다 오히려 낮음**. 중반부(대략 시작 후 3~7분)에 폴링 지연이 9~16초대로 계속 이어지다 32초·77초·94초 스파이크까지 나옴, 초반·후반은 4초로 정상
- **핵심 발견**: 부하를 올릴수록(80/80) 좋아지는 게 아니라 **60/60 근처가 안정성·label 3 생성 둘 다의 스위트스팟**이고 그 위로 올리면 둘 다 나빠짐. 또한 **지속시간을 늘리는 것도 효율이 안 좋음** — 짧게(2분) 여러 번 반복하는 쪽이 길게(10분) 한 번 도는 것보다 label 3 비율이 높았고 폴링 지연도 덜 심했음(장시간 노출이 크래시 리스크만 누적시키는 것으로 보임)
- **크래시는 없었지만 "SSH 완전 끊김 후 자연 복구"라는 새로운 중간 심각도 증상을 두 번 관측**(80/80에서 12.5초 연결 실패, 이전 콤보 60/70에서는 완전 크래시까지 갔었음) — 완전 크래시와 정상 폴링 사이에 스펙트럼이 있는 것으로 보임

시나리오별 신규 수집: `combo_s21_40m_s26_60m` 28 / `combo_s21_60m_s26_60m` 41 / `combo_s21_80m_s26_80m` 32 / `combo_s21_60m_s26_60m_long` 76 — 합계 177행 신규. `metrics_v2.csv` 2111행(직전 커밋) → **2288행**(2287 data rows), label 3 45개 → **51개**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py`: 가중치 변경 없었으므로 재라벨링 전/후 분포 동일 확인(0:690, 1:848, 2:698, 3:51)
- `prepare_ap_metrics_dataset.py` 재변환: train 1468 / val 315 / test 314, label 3 train 35 / val 8 / **test 7**
- `power=1.0`으로 재학습(best val balanced acc 73.1%) → 평가: **전체 정확도 82.5%(fixed)/83.1%(dynamic)**, Label 0 96.6% / Label 1 78.5~79.3% / Label 2 76.8~77.8% / **Label 3 57.1%**(7개 중 4개) — 오후 세션(80.5%, label3 28.6%)보다 전체 정확도·label 3 recall 둘 다 개선. 다만 표본이 여전히 7개 수준이라 세션마다 흔들릴 수 있음
- 체크포인트: `project/checkpoints/ap_v2/`, 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

### 다음 세션 참고
- [ ] 콤보 스위트스팟(60M/60M, 2분 내외) 반복해서 label 3 표본을 더 늘릴 것 — 이 조합이 지금까지 효율이 가장 좋았음
- [ ] 장시간(10분+) 콤보는 지양 — 폴링 지연만 누적되고 label 3 비율은 오히려 떨어짐
- [ ] "SSH 완전 끊김 후 자연 복구" 증상(완전 크래시와 정상 사이의 중간 단계)을 더 체계적으로 추적할 필요 — 몇 초 이상 끊기면 물리 재부팅 없이 복구 가능한지의 경계값이 아직 불명확

## 완료된 작업 (2026-08-24 오후 세션, Claude Code와 진행)

### S26/191 단계별 부하 테스트 — "폰이 송신하면 크래시"가 아니라 "다중 station이 핵심 변수"로 재확정
아침 세션 결론("191 폰 개별 하드웨어 문제였을 가능성")을 검증하기 위해 S26과 191 각각 단독으로 전송률을 단계적으로 올리며 안정성·label 3 생성력을 비교. 목적지는 항상 이 노트북(`192.168.8.226`, 무선), AP는 세션 시작 시 재부팅 직후 상태(uptime 5분)에서 출발.

- **S26 단독**: 70M(90초)·100M(180초) 둘 다 완주, 크래시 없음. 그러나 **label 3은 0개** — channel occupancy는 100%까지 포착됐지만 retry/jitter가 같은 순간에 같이 안 터져서 congestion score가 최대 0.692에 막힘(문턱 0.75)
- **191(옛 S21) 단독**: 40M→70M→100M→120M→150M **다섯 단계 전부 완주, 크래시 없음**. 40~120M 네 단계 모두 label 3을 1개씩 만들어냄(150M만 예외). retry_delta 최대치는 S26(25,422)과 191(23,487)이 비슷한 수준이라 "191이 유독 노이즈가 많다"기보다는 **191의 occupancy 포화와 retry/jitter 폭주가 같은 순간에 겹치는 빈도가 S26보다 높았음**이 label 3 생성력 차이의 실제 원인으로 보임
- **결정적 재해석**: 191이 이번엔 5단계 내내 크래시 없이 버텼다는 것 자체가, 지난 세션의 "191 = 크래시 유발" 가설과 배치됨. 반면 **다중 station 조합(191+S26 동시)은 두 번 시도해서 1승 1패**: 191=60M/S26=100M(90초) 조합은 완주(label 3 1개 포함), 곧이어 191=70M/S26=100M 조합은 **SSH 자체가 완전히 타임아웃 나는 진짜 크래시**로 종료(물리 재부팅 필요, 아직 미실시). → 오늘 하루치 데이터만 보면 **"몇 대가 붙어있는가"(다중 station)가 크래시의 핵심 변수라는 8/24 새벽 가설 쪽으로 다시 무게가 실림**. "191 개별 문제"는 오늘 5단계 생존으로 반증에 가까움. 단, 콤보가 1승 1패라 표본이 너무 적어 확정은 이름
- **크래시 양상**: 점진적 저하가 아니라 throughput이 급격히 0 근처로 붕괴한 뒤 SSH 자체가 타임아웃(완전 크래시, 이전의 "40Mbps 구간 30초 지연 후 자연 복구"보다 심각한 유형)
- **작업 버그 수정**: 수집 스크립트를 저장소 루트에서 실행해서 `metrics_v2.csv`가 루트에 잘못 생성된 적이 있었음(S26 70M 테스트 78행) — `project/scripts/metrics_v2.csv`로 병합하고 스트레이 파일 삭제로 정리. 이후 항상 `project/scripts` 안에서 실행하도록 함
- **`collect_metrics.py`의 `SERVER_IP`를 103(S26)에서 191로 변경**함(지연시간 측정 대상). 다음 세션에서 S26을 다시 쓰려면 103으로 원복 필요

시나리오별 신규 수집: `s26_70m_test` 78 / `s26_100m_test` 44 / `s21_40m_test` 37 / `s21_70m_test` 29 / `s21_100m_test` 30 / `s21_120m_test` 49 / `s21_150m_test` 42 / `combo_s21_60m_s26_100m` 67 / `combo_s21_70m_s26_100m` 24(크래시로 조기 종료) — 합계 약 400행 신규. `metrics_v2.csv` 1711행 → **2111행**(2110 data rows), label 3 40개 → **45개**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py`: 가중치 변경 없었으므로 재라벨링 전/후 분포 동일 확인(0:587, 1:832, 2:646, 3:45)
- `prepare_ap_metrics_dataset.py` 재변환: train 1372 / val 295 / test 293, label 3 train 31 / val 7 / **test 7**(이전 5~6개에서 소폭 증가)
- `power=1.0`으로 재학습(best epoch 43~47, val balanced acc 75.3%) → 평가: **전체 정확도 80.5%**, Label 0 93.4% / Label 1 83.9% / Label 2 69.6% / **Label 3 28.6%**(7개 중 2개). 이전 세션(67.0%, label3 66.7%)보다 전체 정확도는 크게 올랐지만 label 3 recall은 표본이 늘면서 다시 낮아짐 — 여전히 표본 7개 수준이라 세션마다 크게 흔들릴 수 있음, 숫자 자체보다 추세로만 참고
- 체크포인트: `project/checkpoints/ap_v2/`, 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

### 다음 세션 최우선
- [ ] **AP 물리 재부팅 필요** — `combo_s21_70m_s26_100m`에서 SSH 완전 타임아웃으로 크래시된 채로 세션 종료함
- [ ] 콤보(다중 station) 승패가 1승 1패라 표본 부족 — 재부팅 후 몇 분 쉬고 같은 조합(191=60~70M/S26=100M)으로 반복해서 크래시 재현성 확인할 것
- [ ] label 3 test 표본이 7개로 늘었지만 여전히 두 자릿수 이전 — 191 단독 40~120M 반복으로 계속 보충
- [ ] `docs/yongsang/ap_crash_analysis.md`의 "191 개별 하드웨어 문제" 결론을 오늘 결과로 갱신 필요(다중 station 가설로 재선회)

## 완료된 작업 (2026-08-24 아침 세션, Claude Code와 진행)

### AP 크래시 원인 재검증: 새 폰(S26)으로 교체 → 크래시 재현 안 됨, "191 폰 특정 문제" 쪽에 무게 실림
- 새벽 세션 `docs/yongsang/ap_crash_analysis.md`의 "다음 검증 방향 1"(다른 폰으로 교체해서 재현 여부 확인)을 실행. 팀이 새로 산 폰 S26(`192.168.8.103`, hostname `yongsang-ui-S26`)을 Opal에 연결하고 `collect_metrics.py`의 `SERVER_IP`를 103으로 갱신
- 파이(`192.168.8.109`)가 이번엔 유선으로 안 잡혀서(50% ping loss), 대신 이 노트북(`192.168.8.226`, 무선)을 iperf3 서버로 세움 — station 2개(노트북+S26) 구성, S26이 송신자
- **20Mbps, 120초 완주** — 크래시 없음, station 연결 유지
- **40Mbps, 90초 완주** — 도중 AP가 SSH 폴링에 약 30초간 응답 지연(`ap=DEAD` 관측)됐지만, station 연결은 끊기지 않았고 자연 복구됨(재부팅 불필요). 예전의 "SSID 자체가 증발해서 물리적 재부팅 필요" 패턴과는 다른, 더 경미한 증상
- **결론**: 이전 정정 가설("폰이 능동적으로 송신하면 크래시")이 S26에서는 재현되지 않음 — station 수·목적지·전송률(최대 40Mbps)이 이전 191 폰의 즉시 크래시 조건과 비슷한데도 안정적이었음. **"191 폰 개별 하드웨어/드라이버 문제였을 가능성"이 "폰 송신 자체가 위험"이라는 가설보다 유력해짐**. 단, 40Mbps 구간의 SSH 응답 지연은 완전히 무결하다고 보긴 어려워서 70~100Mbps 이상 고부하에서도 S26이 계속 안정적인지는 아직 미검증
- 수집: `metrics_v2.csv` 1514행 → **1711행**(+197), 시나리오 `s26_sender_test` 196행(label 0×133+/1×26+/2×10+ 정도, **label 3은 0개** — 채널 점유율은 100%까지 여러 번 찍혔지만 retry/jitter가 아직 부족해서 문턱(0.75)을 못 넘음)
- **주의**: `SERVER_IP`가 103으로 바뀌었으므로, 이후 세션에서 191(옛 S21) 폰을 다시 쓰려면 원복 필요. 지금은 103(S26)이 라이브 상태의 기본값

## 완료된 작업 (2026-08-24 새벽 세션, Claude Code와 진행)

### AP 크래시 원인 재조사 — "다중 station"이 아니라 "누가 송신하는가"가 핵심 변수였음
- 재연결 루프 이론에 이어 station 개수 이론도 데이터로 재검증. 라즈베리파이(`capstone@192.168.8.109`, Opal LAN 포트에 유선 연결, `eth0`)를 iperf3 유선 서버로 써서 무선 홉을 1개로 줄인 구조로 실험
- **노트북(무선, 유일한 station)→파이(유선) 스트림**: 40M(2분)→100M(3분)→150M(3분) 전부 완주, AP 크래시 없음. 총 106행 확보(대부분 label 1/2, label 3은 0개)
- **결정적 발견**: channel_occupancy가 100%까지 여러 번 찍혔는데도 congestion_score 최대치가 0.646에서 막혀서 label 3(문턱 0.75) 자체가 안 나옴 — 유선 목적지 구조는 재전송/충돌이 거의 없는 "너무 깨끗한" 경로라 retry_failed_score/jitter_score가 낮게 유지되기 때문으로 추정. **즉 이 구조는 안전하지만 label 3을 만들 수 없음**
- **폰(무선)→파이(유선) 스트림**(노트북 대신 폰이 송신자, station 수는 여전히 1개)을 시도 → **거의 즉시 크래시**(수집 행 0개)
- **정정된 결론**: "동시 station 2개 이상"이 아니라 **"폰이 능동적으로 송신하는가"가 크래시의 진짜 변수로 보임**. 노트북이 송신자면 목적지(폰이든 유선 파이든)·전송률(150Mbps까지)과 무관하게 안정적이었고, 폰이 송신자면 목적지·전송률과 무관하게 거의 즉시 크래시함. 오늘 밤 전체 크래시 사례를 이 기준으로 재검토하면 전부 일치함
- 안전성(노트북 송신)과 label 3 생성 능력(무선 경합 필요, 현재는 폰 송신만 가능)이 트레이드오프 관계라는 게 새로운 핵심 문제로 부상
- 상세 분석은 `docs/yongsang/ap_crash_analysis.md`(+ HTML 아티팩트)에 별도 문서화함

### 데이터 반영 (밤 세션 이어서)
- 오늘 새벽 세션 전체 수집 반영: 재라벨링(검증, 변화 없음) + `ap_metrics_v2` 재변환(train 1024/val 219/test 221, label 3 28/6/6) + `power=1.0` 재학습
- 평가 결과: 전체 정확도 67.0%, Label 2 32.1%(하락), **Label 3 66.7%**(4/6, 큰 폭 상승) — 단, test label 3이 5~6개 수준이라 실행마다 recall이 크게 흔들림(직전 3번의 실행에서 40.0%→20.0%→66.7%). 표본이 늘 때까지는 추세로만 참고
- `metrics_v2.csv` 1338행→**1514행**, raw label 3 38→**40개**

### AP 재시도: "다중 station이 크래시 원인"이라는 가설 재검증, 단일 폰으로 label 3 추가 확보
- AP를 몇 시간 쉬게 한 뒤 재시도. **재연결 반복(2~5초마다 iperf3 재접속)이 크래시 원인이라는 가설을 먼저 재검토** — 오늘 저녁 run5(2대, 100Mbps, 재연결 루프 이미 제거된 상태)도 ~22초 만에 크래시했던 기록을 다시 보니, 재연결 루프 유무와 무관하게 **"동시 station 2개 이상"이 실제 변수였을 가능성이 더 크다는 결론으로 정정**
- 폰 1대(`192.168.8.191`)만 단일 연속 iperf3 스트림으로 테스트: 70Mbps 2분 완주(크래시 없음, label 0/1/2만) → 100Mbps 5분 완주(크래시 없음, **label 3 4개 신규 확보**, 그중 2개는 `channel_occupancy_percent=100.0` 완전 포화) → **같은 세션에서 3번째로 20분 시도했다가 2분 41초 만에 크래시** (label 3 1개는 그 전에 추가 확보)
- **새로운 관찰**: 단일 station이 다중 station보다 훨씬 안정적인 건 맞지만 완전히 안전하진 않음 — 휴식 없이 연속으로 여러 번 부하를 주면 단일 station이라도 누적 피로(추정: 열, 원인 미확정)로 결국 크래시함. "재부팅 직후 첫 시도가 제일 잘 버틴다"는 패턴이 오늘 밤 내내 반복 관찰됨(3대 첫 시도 90초 완주, 단일 폰 첫/둘째 시도 각각 2분·5분 완주 후 셋째 시도부터 급격히 나빠짐)
- **다음 시도 권장**: 폰 1대 단일 스트림으로 시작하되, **세션 사이에 AP를 몇 분씩 쉬게 하는 휴식 시간을 의도적으로 넣을 것**. 연속으로 몰아붙이지 말 것.
- 밤 세션 총 수집: label 0 5개 / label 1 22개 / label 2 45개 / **label 3 5개** 추가. `metrics_v2.csv` 1265행 → **1338행**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py` 실행 — 신규 수집분도 이미 최신 가중치로 라벨링돼 있어서 재라벨링 전/후 분포 동일함을 확인(정상 동작 검증)
- `prepare_ap_metrics_dataset.py`로 재변환: train 843→**903**, val 181→**194**, test 179→**191**. label 3은 train 23→**27**, val 5→**6**, test 5→5(동일)
- `power=1.0`(기본값)으로 재학습 → 평가 결과: 전체 정확도 65~66%→**73.3%**, **Label 2 recall 36~42%→70.8%로 크게 개선**, 그런데 **Label 3 recall 40%→20%로 하락**(5개 중 1개만 정답)
- **해석**: test label 3이 여전히 5개뿐이라 1개 차이가 20%p를 좌우함 — 통계적 노이즈일 가능성이 높고, Label 2/3 트레이드오프가 실행마다 흔들리는 걸 보면 표본이 더 늘어야 진짜 경향을 볼 수 있음. 지금 숫자 하나하나에 너무 의미 부여하지 말 것.
- `project/README_AP_V2.md`, `CLAUDE.md` 2차 섹션의 라벨 분포/평가 결과 표를 최신 수치로 갱신함

### congestion_score 가중치 재조정 — AP 재부팅 없이 label 3을 21개→33개로 늘림
- AP가 반복 크래시로 더 이상 데이터를 못 모으는 상황에서, 이미 모아둔 `metrics_v2.csv`(1265행)를 분석해보니 **원래 가중치(throughput 35% / occupancy 35% / retry 20% / jitter 10%)가 실제 변별력과 안 맞았음**을 발견
  - stress_load 구간에서 label 2 vs label 3의 sub-score 평균 비교: `throughput_score` 0.665→0.707(거의 차이 없음), `occupancy_score` 0.449→0.898(거의 2배), `jitter_score` 0.512→0.802(큰 차이), `retry_score` 0.724→0.834(약한 차이)
  - 즉 throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 거의 기여를 못 하고 있었음
- `collect_metrics.py`의 `calculate_scores()` 가중치를 **throughput 20% / occupancy 45% / retry 20% / jitter 15%**로 재조정 (occupancy·jitter 비중 상향, throughput 비중 하향)
- 새 가중치를 기존 원시 데이터(1265행)에 그대로 재적용하는 `project/scripts/relabel_metrics_v2.py` 신설 — **AP 재수집 없이** raw 데이터 재라벨링만으로 label 3을 21개→33개로 늘림 (label 0: 166→149, label 1: 516→670, label 2: 562→413)
- `ap_metrics_v2` 재변환: label 3 샘플이 train 14→23, val 3→5, test 3→5로 증가

### class weight power 재실험 — power=1.0으로 확정 (트레이드오프가 완만하지 않고 절벽형)
- `train_ap_early_exit.py`에 `--class-weight-power` CLI 인자 추가(기존엔 하드코딩)
- 재라벨링된 데이터로 power 0.7 / 0.85 / 1.0 비교:

  | power | 전체 정확도 | Label 0 | Label 1 | Label 2 | **Label 3** |
  |---|---:|---:|---:|---:|---:|
  | 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
  | 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
  | 1.0 | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

- **0.7/0.85는 label 3을 아예 0%로 계속 놓침. 1.0(순수 역빈도)에서만 label 3이 잡히기 시작**하며, 그 대신 label 2 정확도가 크게 하락(88%→36~42%) — 완만한 트레이드오프가 아니라 거의 전부/전무에 가까운 절벽
- 팀 판단으로 **power=1.0을 기본값으로 확정**: 이 프로젝트 목적상 심각 혼잡을 놓치는 것(false negative)이 혼잡을 심각으로 과잉 경고하는 것(false positive)보다 더 치명적이라는 이유
- 최종 체크포인트: `project/checkpoints/ap_v2/`(power=1.0), 평가 리포트: `project/results/yongsang/ap_v2_eval_report.txt`
- **주의**: 이 재조정으로 `ap_metrics_v2`의 label 정의(congestion_score 가중치)가 production `ap_cleaned_strict`(588행, 옛 가중치 0.35/0.35/0.20/0.10)와 달라짐. 두 데이터셋을 같은 기준으로 비교하면 안 됨 — `ap_cleaned_strict`를 새 가중치로 재라벨링할지는 아직 미결정(팀 논의 필요)

## 완료된 작업 (2026-08-23 저녁 세션, Claude Code와 진행)

### AP(Opal GL-SFT1200)가 다중 station 부하에서 반복적으로 크래시됨 — 중요 하드웨어 이슈 발견
- 폰 3대(103/191/221) 동시 부하로 label 3 데이터를 더 모으려고 시도. 오늘 Claude Code가 노트북에서 iperf3 부하를 직접 걸고 `collect_metrics.py`를 동시 실행하는 방식으로 자동화 시도
- **1차(3대, 40Mbps 각각, 90초)는 완주 성공** — label 3 1개 포함 6개 샘플 확보(채널 100% 근처 포화, retry 24508/25836, congestion_score 0.756)
- 이후 **총 4번 더 시도했으나 전부 AP 크래시로 조기 중단**: 3대/40Mbps/10분(~40초에 폰 2대 연결 끊김), 3대/25Mbps(~18초), **심지어 예전에 검증됐던 2대/100Mbps 조합조차 ~22초 만에 크래시**(죽기 직전 1.5~1.7 Gbps라는 물리적으로 불가능한 카운터 스파이크 관측 — WiFi 드라이버/라디오 재시작 추정)
- 크래시마다 AP의 WiFi SSID(`GL-SFT1200-a08`) 자체가 사라지고 노트북이 다른 아는 네트워크(`192.168.45.x`)로 자동 전환됨 → 매번 AP 전원을 물리적으로 껐다 켜야 복구됨 (총 4회 재부팅)
- **핵심 관찰: 부하를 낮춰도(40M→25M), station 수를 줄여도(3대→검증된 2대 조합) 크래시가 오히려 더 빨리 발생함(90초 완주 → 40초 → 18초 → 22초)** — 이는 부하 세팅 문제가 아니라 **반복된 크래시-재부팅 사이클 자체가 AP를 점점 더 불안정하게 만들었을 가능성**을 시사함 (열, 메모리 누수, 펌웨어 상태 꼬임 등 추정, 원인 미확정)
- 사용자 판단으로 오늘은 여기서 중단, AP를 충분히 쉬게 한 뒤 재시도하기로 결정
- **폰 `192.168.8.103`이 세션 중간에 오프라인 상태**가 되어(iperf3 서버 응답 없음, ping도 unreachable) `collect_metrics.py`의 하드코딩된 `SERVER_IP`를 `192.168.8.191`로 임시 변경함 (지연시간 측정 대상 폰 교체, 커밋됨)
- **총 성과**: 크래시 반복에도 불구하고 실측 데이터 12행 추가(label 0×4, 1×1, 2×6, **3×1**) → `metrics_v2.csv` 1254행→**1266행**. 다음 세션에서 `prepare_ap_metrics_dataset.py`로 재변환 + 재학습 필요 (아직 안 함, 증가폭이 작아서 우선순위는 낮음)

## 완료된 작업 (2026-08-23 오후 세션, Claude Code와 진행)

### Pi를 TV(HDMI)에 연결, SSH 키 인증 설정 완료 — 새벽 세션의 "최우선" 항목 해결
- TV에 HDMI로 연결해서 부팅 화면 확인 → 정상 부팅되어 `capstone@CapsTone:~ $` 프롬프트까지 뜸 (SD카드 접촉 불량 의심은 해소된 것으로 보임)
- Pi를 와이파이(`192.168.45.x` 대역, 노트북과 같은 네트워크)에 새로 연결. `hostname -I`로 IP 확인 → `192.168.45.31`
- 이 노트북의 `~/.ssh/id_ed25519` 공개키를 Pi의 `~/.ssh/authorized_keys`에 등록해서 비밀번호 없이 `ssh capstone@192.168.45.31` 접속 가능해짐 (AP용 `id_rsa_ap`와는 별개 키)

### `project/deploy/raspberry_pi_ap/` 번들로 Pi 실측 완료 (단, 구버전 588행 `ap_cleaned_strict` 데이터 기준)
- 번들 전체를 scp로 Pi에 전송, `.venv`에 onnxruntime 1.29.0 설치, README의 8개 조합(Baseline/SDN/Fixed/Dynamic × FP32/INT8) 전부 실행 완료
- 정확도는 PC 평가와 완전히 일치(Baseline 92.7%, 나머지 91.5%) — 동일 checkpoint/scaler 확인됨
- **핵심 발견**: PC에서는 안 보이던 Early Exit 속도 우위가 Pi + staged ONNX 실측에서는 실제로 나타남. Proposed Dynamic FP32(1.699ms 평균)가 Baseline FP32(1.837ms)보다 7.5% 빠름. Dynamic theta의 Exit1 비율(37.8%)이 Fixed(15.9%)보다 훨씬 높은 게 원인
- 결과 저장: `project/results/yongsang/pi_ap_measurements/` (원시 CSV 56개 + `pi_ap_measurement_summary.md` 요약표)
- **주의**: 이 실측은 `ap_cleaned_strict`(588행) 파이프라인 기준이고, 아래 `ap_metrics_v2`(1253행, 실제 최신 데이터)과는 별개 — 아직 새 데이터 기준 ONNX/Pi 번들은 없음

### `ap_metrics_v2` 데이터셋이 낡은 스냅샷 기준이었던 버그 발견 및 수정
- 어젯밤 커밋(`bef79fd`, "1254행으로 확장")의 커밋 메시지는 "재생성 완료(train 707/val 151/test 152)"라고 했지만, 실제 `conversion_report.txt`를 까보니 **`raw_rows: 1060`** — 즉 `metrics_v2.csv`가 1253행까지 다 채워지기 *전* 스냅샷으로 변환 스크립트를 돌리고, 그 뒤 193행을 추가 수집한 걸 같은 커밋에 묶어 올린 것으로 보임
- `prepare_ap_metrics_dataset.py --input project/scripts/metrics_v2.csv --out-dir project/data/ap_metrics_v2 --overwrite`로 최신 1253행 전체 기준 재변환 → **train 843 / val 181 / test 179** (raw_rows 1253으로 일치 확인). label 3(심각)도 train 14/val 3/test 3으로 소폭 증가(이전 9/2/2)

### 클래스 가중치 완화 실험 (새벽 세션에 남겨둔 "다음 할 일" 항목 해결)
- `train_ap_early_exit.py`의 `compute_class_weights`가 순수 역빈도(`N/(K*count)`, power=1.0)만 지원해서 label 3에 ~20배 가중치가 붙어 label 2를 3으로 오버슈팅하던 문제(label 2 recall 19%)를 재현할 것으로 예상됨
- `power` 파라미터를 추가해서 `(N/(K*count))**power` 형태로 일반화. power=0.5(sqrt)로 재학습 → label 2 recall 19%→79%로 크게 개선됐지만 label 3 recall이 0%로 떨어짐(트레이드오프가 반대로 과함)
- **power=0.7로 재조정**(현재 값) → val balanced acc 62.8%(0.5일 때 58.3%보다 나음), test 전체 정확도 74.3%, label 0=77.3%/label 1=84.9%/label 2=66.7%/**label 3=0%**
- label 3는 test 샘플이 3개뿐이라 가중치를 아무리 조정해도 한계가 있음 — 알고리즘이 아니라 데이터 부족 문제. 다음 수집에서 label 3(채널 100% 포화) 샘플을 더 확보해야 근본 해결됨
- 재학습된 체크포인트: `project/checkpoints/ap_v2/`(검증용, production `ap_cleaned_strict`와는 별개 그대로 유지)
- 평가 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

## 완료된 작업 (2026-08-23 새벽 세션, Claude Code와 진행)

### Pi SSH 로그인 미스터리 해결: 계정명은 "capstone", "CapsTone" 정체도 파악
- 라즈베리파이가 Opal LAN 포트에 유선 연결되어 있다는데 station dump/DHCP/ARP 어디에도 안 잡히는 문제를 오래 추적함
- SD카드를 노트북에 꽂아 boot 파티션(`bootfs`, FAT32)의 cloud-init 설정을 직접 확인 → **계정명이 `pi`가 아니라 `capstone`**이었음이 확인됨 (지난 세션의 추측이 맞았음). `hostname: CapsTone`으로 설정되어 있는 것도 확인
- **중요한 재해석**: 어젯밤(8/22) 극한 부하 테스트에서 "새 공기계"로 추가했던 `192.168.8.109`(hostname "CapsTone")가 사실 새 폰이 아니라 **이 라즈베리파이 자체**였음을 MAC 벤더 조회(`d8:3a:dd:48:55:97` → Raspberry Pi Trading Ltd, API로 검증)로 확인. 즉 어젯밤엔 파이가 이더넷으로 정상 연결되어 9분 넘게 iperf3 트래픽까지 잘 주고받았음
- 오늘은 케이블/SD카드를 여러 번 재연결해도 Opal 쪽에서도, SD카드에 설정된 백업 Wi-Fi(`SK_0600_5G`, 집 공유기)에서도 전혀 안 잡힘. 중간에 초록 활동 LED가 "깜빡이다 꺼짐" 현상 관찰 → SD카드 접촉 불량으로 인한 boot 실패 패턴으로 추정
- SD카드의 boot 파티션(FAT32)은 노트북에서 읽히지만, 이건 OS가 있는 루트 파티션(ext4, 노트북에서 못 읽음)이 멀쩡하다는 뜻은 아님 — boot는 되지만 그 다음 단계에서 막히는 것과 완전히 다른 문제
- **결론: 잠정 보류.** TV에 직접 연결해서 화면으로 부팅 진행 상황을 봐야 확실해짐 (오늘은 시간 관계상 보류, 다음 세션 과제)
- 참고: `network-config`(netplan) 내용 — `eth0: dhcp4 true, optional true` / `wlan0: SK_0600_5G에 자동연결, optional true`. `user-data`(cloud-init) — 계정 `capstone`, `enable_ssh: true`, `ssh_pwauth: true`, `avahi-daemon` 설치됨. **비밀번호는 해시(yescrypt)라 SD카드에서 평문 확인 불가** — 호중에게 "capstone 계정" 비밀번호로 다시 확인 필요

### 데이터 수집 계속 확대 (834행 → 1254행) — 진짜 다중 station 경합으로 label 3 대폭 확보
- 어제 세션 마지막에 발견한 문제("노트북이 발신 허브라 진짜 다중 station 경합이 아니었다")를 해결: 폰을 2대(`192.168.8.103`, 나중엔 새 폰 `192.168.8.191`) 동시에 사용해서 각각 독립적으로 100~120Mbps UDP 부하를 걸어 진짜 2-station 경합 재현
- `high_load` 단일기기로 10분 추가 수집(244행), 2폰 동시 부하로 `stress_load`에 2회 추가 수집(총 523행까지), `medium_load`도 폰 하나로 보충(187행) → 최종 **1254행**
- 라벨 재계산 결과 **label 3(심각) 4개 → 13개**로 대폭 증가 (채널 100% 포화 순간이 훨씬 자주 잡힘). val/test에도 각각 2개씩 포함되어 처음으로 통계적으로 유의미한 평가가 가능해짐
- 재학습 후 confusion matrix 확인: **모델이 이제 4개 클래스를 전부 예측함**(전엔 label 2/3을 아예 안 찍었음). label 3 정확도 0%→50%(2개 중 1개), val balanced accuracy 54.4%→69.4%
- **새로 드러난 부작용**: label 3에 준 클래스 가중치가 너무 세서(train 9개뿐이라 가중치 ~20배) label 2(혼잡)를 label 3으로 오버슈팅하는 경향 생김(label 2 정확도 19%로 하락, 58개 중 33개를 3으로 오분류). 다음에 가중치를 완화(역빈도 대신 제곱근 역빈도 등)하면 다듬을 수 있을 것으로 보임
- 폰 iperf3 서버가 화면 꺼짐으로 중간에 두 번 죽음(`termux-wake-lock` 안 걸어둔 폰들) → 재시작 대기 중 세션 일시 중단, 여기서 기록 저장

## 다음 할 일 (2026-08-24 새벽 세션 기준 갱신)
- [x] `project/deploy/raspberry_pi_ap/` 번들로 8개 조합 Pi 실측 완료 (단, 구버전 588행 데이터 기준 — 위 섹션 참고)
- [x] congestion_score 가중치 재조정(occupancy/jitter 상향, throughput 하향) + class weight power=1.0 확정
- [x] AP 재시도(폰 1대 단일 스트림) — 100Mbps로 5분 완주해서 label 3 4개 확보
- [x] 라즈베리파이 유선 서버 구조 시도 — 노트북 송신은 150Mbps까지 안정적이나 label 3이 구조적으로 안 나옴, 폰 송신은 거의 즉시 크래시. **"폰이 송신하면 크래시"라는 정정된 결론** 도출 (`docs/yongsang/ap_crash_analysis.md`)
- [x] 새벽 세션 전체 수집분 반영 — `ap_metrics_v2` 재변환(train 1024/val 219/test 221, label 3 28/6/6) + 재학습. 전체 정확도 67.0%, Label 3 66.7%(4/6)
- [x] 다른 폰(S26)으로 교체해서 "폰 송신 = 즉시 크래시" 재현 여부 확인 — **재현 안 됨**(20/40Mbps 완주). 191 폰 특정 문제 쪽에 무게 실림. 상세는 위 "아침 세션" 항목
- [x] **완료(2026-08-24 오후 세션)**: S26/191 각각 단독 70~150Mbps 단계별 램프업 완료 — 둘 다 크래시 없음, 191이 label 3 생성력 더 강함. 다중 station(191+S26) 조합에서만 크래시 발생. 상세는 위 "오후 세션" 섹션 참고. 아래 하위 항목들(세팅/램프업/피로 패턴)은 이 완료 항목으로 대체됨
- [ ] ~~S26으로 전송률을 70~100Mbps까지 단계적으로 올리면서 (a) 그 구간에서도 안정적인지, (b) label 3(채널 포화 + retry/jitter 동반)을 안전하게 만들 수 있는지 확인. 40Mbps에서 관측된 30초 SSH 응답 지연이 더 심해지는지도 주시할 것~~
  - **세팅**: 송신자는 S26 고정(191은 은퇴), **목적지는 반드시 무선**(유선 파이 금지 — congestion_score가 0.646에서 막혀 label3이 구조적으로 안 나옴 + 애초에 이 프로젝트가 겨냥하는 게 산업 "무선"망이라 유선 데이터는 목적에도 안 맞음). AP는 재부팅 직후 상태에서 시작
  - **램프업**: 70Mbps로 짧게(1~2분) 통과 확인 → 문제 없으면 바로 100Mbps로. 70은 그 자체가 목표가 아니라 "이상 신호 조기 감지용" 관문
  - **피로 패턴 동시 확인**: 시도 사이에 의도적으로 몇 분씩 휴식 넣고 반복 — "191 개별 문제" vs "폰은 다 결국 누적 피로로 죽는다"를 같은 데이터로 갈라볼 것
  - **과거 100Mbps 시도 이력(참고)**: 재부팅 후 1~2번째 시도는 100Mbps에서 거의 항상 완주했고 그때마다 label3도 나왔음(8/22 밤 폰2대 560초 완주+3개 신규, 8/23~24 밤 191 단독 5분 완주+4개 신규). 반면 크래시-재부팅이 누적된 뒤 재시도하거나(8/23 저녁 같은 조합이 22초 만에 크래시), 연속 3번째 시도이거나(191 20분 시도 2분41초 크래시), 폰이 유선 파이로 송신한 경우(거의 즉시 크래시)는 실패함 — "재부팅 후 이른 시도 + 무선 목적지"가 핵심 성공 조건으로 보임
- [ ] label 3 test가 아직 5~6개 수준이라 recall이 세션마다 크게 흔들림(40%→20%→66.7%) — 두 자릿수 중반 이상으로 늘리는 게 목표
- [x] **`ap_cleaned_strict`(production, 588행) 재라벨링 여부 팀 결정 완료(2026-08-24)**: 재라벨링하지 않음. 1차는 1학기 모델링 검증용이자 인터넷 공개 데이터 기반이라는 한계가 있어 archived로 고정하고, 앞으로의 실측 기반 라벨링은 2차(`ap_metrics_v2`)에서만 진행. `ap_metrics_v2`과 `ap_cleaned_strict`는 여전히 서로 다른 라벨 기준(가중치)이므로 두 데이터셋을 같은 기준으로 비교하지 않는다는 원칙은 유지
- [ ] `192.168.8.103` 폰이 오프라인됐던 원인 확인 (Termux 강제종료 추정) — 배터리 최적화 예외 설정 확인 권장, 복귀하면 `collect_metrics.py`의 `SERVER_IP`를 다시 `103`으로 되돌릴지 `191` 유지할지는 팀 편의대로
- [ ] `ap_metrics_v2`(1514행) 기준 ONNX export + Pi 배포 번들은 아직 없음 — `ap_cleaned_strict`용 `export_onnx_ap.py`/`prepare_pi_bundle_ap.py`를 새 데이터 경로로 재사용할지, 별도 스크립트를 만들지 결정 필요
- [ ] 호중에게 이 밤/새벽 크래시 원인 분석(`docs/yongsang/ap_crash_analysis.md`) 공유 — 펌웨어 업데이트 여부나 다른 AP 확보 가능성 문의
- [ ] label 1/2 경계(congestion_score 0.50 부근) 재설계 논의 — threshold 재조정 또는 feature 추가 필요할 수 있음
- [ ] 스케일러 불일치 발견 사항을 예나·팀에 공유 (원본 `ap_cleaned_strict`의 latency_ms/rssi_dbm 측정 방식 재검토 필요)
- [ ] 장기적으로 이 실측 방식으로 모델을 새로 학습/파인튜닝해서 `ap_cleaned_strict`를 대체할지, 팀 논의 필요
- [ ] (여유 시) AP strict용 실시간 추론 파이프라인 설계 착수 — 현재 어느 브랜치에도 코드 없음


## 프로젝트 개요
산업 무선망(AP) 트래픽 혼잡을 Early Exit LSTM으로 실시간 분류하고, Raspberry Pi + ONNX/INT8로 엣지 배포하는 캡스톤 프로젝트. 방학 중 교수 피드백에 따라 1학기 4-feature 시뮬레이터 기반에서 실제 GL.iNet AP 실측 9-feature(`ap_metrics_cleaned_strict`) 기반으로 피벗함. 팀: 유용상(모델 설계), 장예나(데이터), 김호중(경량화·배포).

## 완료된 작업 (2026-08-21 밤 ~ 2026-08-22 새벽 세션)

### AP(Opal) 네트워크 문제 원인 파악 및 해결
- 처음엔 AP WiFi SSID(`GL-SFT1200-a08`)에는 연결됐지만, `192.168.8.1` 관리 페이지/SSH가 전혀 안 열리는 문제 발생. IP를 확인해보니 노트북이 Opal 자신이 아니라 **집 공유기(다른 물리 기기)가 나눠준 IP(`192.168.75.x`)**를 받고 있었음
- 원인: Opal의 LAN 포트에 집 인터넷 랜선을 꽂아서 Opal이 브릿지/익스텐더처럼 동작 → WiFi 클라이언트가 Opal이 아닌 상위 공유기 서브넷을 그대로 받음
- 호중에게 확인 결과, 원래 성공했던 구성은 **"집 공유기(LAN) → Opal의 WAN 포트"**로 연결하는 것이었음 (LAN이 아니라 WAN에 꽂아야 Opal이 독립적으로 `192.168.8.0/24`를 WiFi로 뿌리면서 인터넷도 별도로 받음)
- 랜선을 Opal의 WAN 포트로 옮겨 연결 → 노트북이 `192.168.8.226`을 받고, **인터넷도 되고 `192.168.8.1` 관리 페이지도 열림** (Claude Code 세션도 안 끊기고 유지됨)
- 중간에 Opal 본체의 "MODE" 슬라이드 스위치를 만져봤으나 효과 없었음(아마 라우터/AP 모드 전환이 아니라 펌웨어 부팅 슬롯 선택 스위치로 추정) — 원위치로 되돌려둠

### SSH 인증 정리
- **AP(root@192.168.8.1)**: 호중이 알려준 비밀번호로 로그인 성공. 단, `ssh-rsa` 알고리즘을 명시적으로 허용해야 함(`-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa`) — 오래된 dropbear라 최신 SSH 클라이언트 기본값과 안 맞음
- 이 노트북에 RSA 키(`~/.ssh/id_rsa_ap`, ed25519는 이 dropbear가 거부해서 RSA로 재생성)를 만들어 `ssh-copy-id`로 AP에 등록 → `~/.ssh/config`에 `Host 192.168.8.1` 항목 추가해서 **비밀번호 없이 자동 인증**되도록 설정 완료 (`collect_metrics.py`가 내부적으로 `BatchMode=yes`를 쓰기 때문에 키 인증이 필수였음)
- **Pi(pi@192.168.8.109)**: 호중이 알려준 비밀번호가 안 먹힘(3회 시도 실패). 호중 본인은 "그 비번 맞다"고 하는데 우리 쪽에서 안 됨 — **계정 이름이 `pi`가 아닐 가능성**이 유력. 호중에게 실제 로그인 명령어 전체(아이디 포함)를 그대로 복사해서 보내달라고 요청한 상태, **아직 미해결**
- 라즈베리파이 SD카드 재굽기(Imager 고급설정으로 새 계정 지정)도 대안으로 검토했으나, 프로젝트 파일은 이미 git에 다 커밋되어 있어 손실 위험이 크지 않다는 결론 → 호중 답 기다리는 쪽으로 결정, 아직 안 구움

### `collect_metrics.py` 버그 수정
- **Windows ping 파싱 버그 (수정 완료)**: 기존 코드가 Linux `ping -c/-W` 문법과 `rtt min/avg/max/mdev` 출력 형식만 파싱해서, Windows(특히 한글 로캘)에서 실행하면 latency/jitter/packet_loss가 에러 없이 조용히 전부 0으로 찍히는 버그였음. `platform.system()`으로 분기해서 Windows에서는 `ping -n/-w` 사용 + `TTL=`이 포함된 응답 줄에서 로케일 무관하게 `(숫자)ms` 패턴을 추출하도록 수정 (`project/scripts/collect_metrics.py`)
- **station 재연결 스파이크 버그 (미수정, 팀 공유 필요)**: `connected_clients`가 1→2로 바뀌는 순간(폰 WiFi 절전모드 등으로 station 목록에서 빠졌다 다시 나타날 때), 그 station의 누적 rx/tx bytes·재전송 카운터 전체가 "순간 증가분"으로 잘못 계산되어 throughput이 수백~수천 Mbps로 튀는 현상 확인. 여러 station의 바이트를 그냥 합산하는 `calculate_station_throughput` 로직의 구조적 문제 — 코드 수정은 안 했고, 데이터 후처리 시 `connected_clients` 전환 시점 행을 걸러내는 방식으로 대응 필요

### 실측 데이터 수집 완료
- 이 노트북(WiFi Client, `iperf3` winget으로 설치) + 폰(Termux, `iperf3 -s` 서버, IP `192.168.8.103`) 구성으로 5개 시나리오 라이브 수집
- `normal_idle`(26행) / `low_load`(19행, 20Mbps) / `medium_load`(17행, 50Mbps) / `high_load`(17행, 100Mbps) / `stress_load`(12행, 150Mbps×4병렬) → 총 91행, `project/scripts/metrics_v2.csv` (**아직 git 미커밋**)
- 부하를 올릴수록 UDP 손실률이 뚜렷하게 증가(1.2% → 20.0% → 28.4% → 68.1%)해서 시나리오 설계가 의도대로 작동함을 확인
- 첫 `low_load` 수집 때 프로세스를 늦게 종료해서 157초 중 대부분이 부하 없는 상태로 잘못 라벨링된 데이터 오염 발견 → 삭제 후 깨끗하게(68초 정확히 맞춰서) 재수집함

### 모델 파이프라인 end-to-end 검증 + 중요 발견
- 이 노트북 anaconda base 환경은 기존에 알려진 것과 동일한 torch DLL 로딩 실패 문제 있음 → **새 conda 환경 `capstone` 생성**(`C:\Users\dkssu\anaconda3\envs\capstone`)에 torch(CPU)+pandas+numpy 설치해서 해결
- `prepare_ap_metrics_dataset.py --input project/scripts/metrics_v2.csv --out-dir project/data/ap_metrics_v2`로 윈도우 변환 (41 샘플: train 28/val 7/test 6) — **기존 `ap_metrics_cleaned_strict` 폴더는 건드리지 않고 별도 폴더로 생성**
- **중요 발견**: 원본 학습 스케일러(`ap_metrics_cleaned_strict/scaler_params.json`)와 우리 실측 데이터의 실제 범위가 완전히 다름
  - `latency_ms`: 원본 0.047~0.163 vs 실측 2~841 (원본이 ms 단위가 맞는지 의심스러움)
  - `tx_retries_delta`: 원본 최대 23 vs 실측 최대 20만대
  - `rssi_dbm`: 원본 -30~-17(매우 근접 측정) vs 실측 -67~-53.5
  - → 1학기/AP strict 원본 학습 데이터의 측정 방식 자체에 단위 버그가 있거나, 완전히 다른 물리적 실험 조건(매우 가까운 거리)에서 수집됐을 가능성. **예나·팀에 공유 필요**
- `evaluate_ap_early_exit.py`로 `ap_early_exit_lstm_best.pth` 평가 (자체 스케일러 사용, `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt`에 저장): 전체 정확도 50%(단, test 샘플 6개뿐이라 통계적으로 거의 무의미), Label 0/1(정상/경고)은 100% 정확했지만 **Label 2/3(혼잡/심각)은 0%** — 사전학습된 모델이 이 새로운 측정 환경에 일반화되지 않음을 시사

## 완료된 작업 (2026-08-22 밤 세션, Claude Code와 진행)

### station 재연결 스파이크 버그 실제 수정 (이전 세션엔 "미수정"으로 남아있던 것)
- 원인: `parse_station_info`가 연결된 모든 station의 누적 rx/tx bytes·재시도 카운터를 그냥 합산해서 반환 → 어떤 station이 station dump에서 잠깐 빠졌다 재등장하면 그 station의 전체 누적값이 "한 폴링 주기 증가분"으로 잘못 계산되어 throughput이 수천 Mbps로 튀는 버그였음
- 수정: station을 MAC 주소별로 개별 추적하도록 변경(`parse_station_info`가 dict 반환), `calculate_station_deltas` 신설 — 직전 폴링에 없던(방금 나타난) station은 이번 폴링 델타를 0으로 스킵. 시뮬레이션 테스트 + 실측(약 45분 연속 수집) 양쪽에서 재현 안 됨 확인 (`project/scripts/collect_metrics.py`)
- 커밋: `126c782`

### 데이터 재수집 (67행 → 636행 → 834행, 3단계)
1. 버그 수정 스크립트로 5개 시나리오 짧게(각 60~90초) 재수집 → 67행, 커밋 `d328115`
2. "샘플이 너무 적다"는 판단 하에 5개 시나리오를 각 9~10분씩 재수집 → 636행. `connected_clients`가 9분 내내 안정적으로 유지되고 스파이크 없음을 재검증
3. 2대 동시 부하(아래 항목)로 stress_load에 197행 추가 → 최종 **834행**

### congestion_score 계산식 재보정 (`JITTER_MAX_MS`, `RETRY_FAILED_MAX`)
- 기존 `JITTER_MAX_MS=1.0`, `RETRY_FAILED_MAX=100.0`은 시뮬레이터 데이터 기준값이라 실측 AP 데이터(jitter 수백ms, retry 수천~수만)에서 `jitter_score`/`retry_failed_score`가 거의 항상 1.0으로 clamp됨 → label 2(혼잡)로 66% 쏠리는 문제 확인
- 실측 분포 p90 근처로 재보정(`JITTER_MAX_MS=300.0`, `RETRY_FAILED_MAX=25000.0`) → saturation 문제는 해결됐으나, congestion_score 자체가 0.25~0.55 구간에 몰려있어 label 3(≥0.75) 문턱을 못 넘는 문제가 새로 드러남 → **채널 100% 포화 같은 진짜 극단 조건이 있어야 label 3이 실제로 나온다**는 결론

### 2대 동시 부하로 진짜 다중 station 혼잡 재현 시도
- 처음엔 노트북 하나로 iperf3 `-P 4`(다중 논리 스트림)를 시도했으나 폰 서버 프로세스가 병목이 되어 오히려 약해짐 → **물리적으로 다른 기기**가 필요하다는 결론
- 공기계(Termux+iperf3, `192.168.8.235`) 추가 확보 → 노트북에서 두 폰으로 각각 150Mbps UDP 동시 발사 → **AP(Opal)가 58초만에 크래시, WiFi SSID 자체가 완전히 사라짐**(재부팅 필요)
- 재부팅 후 100M×2로 재시도 성공 — 560초 전부 완료, AP 생존, `channel_occupancy_percent=100.0`(완전 포화) 순간을 포착해 **진짜 label 3(심각) 샘플 3개 신규 확보**(기존 1개 포함 총 4개)
- 이후 새 공기계(`192.168.8.109`, "CapsTone")로 교체 진행. **중요 인사이트**: 지금까지 구성은 "노트북 1대가 발신 허브로 두 폰에 동시 전송"이라 실제로는 노트북의 단일 업링크가 병목일 수 있음 — 진짜 독립적인 다중 station 경합을 만들려면 폰↔폰 직접 전송이나 노트북을 수신측(iperf3 -s)으로 추가하는 식으로 트래픽 발신원 자체를 분산해야 함 (다음 세션 과제)

### 모델 학습 파이프라인의 근본 버그 발견 및 수정: class imbalance로 인한 완전한 클래스 붕괴
- 체크포인트 불일치(스케일러 다른 모델로 새 데이터 평가) 때문에 정확도가 낮다는 가설을 직접 검증: 새 데이터로 처음부터 재학습 → 39.3% → 79.8%로 확인, 가설 맞음
- 그런데 재학습해도 label 2(혼잡)가 여전히 0%로 나와서 confusion matrix를 직접 뽑아봄 → **모델이 label 2/3을 단 한 번도 예측하지 않는 완전한 class collapse** 확인 (`actual 2: [0, 30, 0, 0]`)
- 원인 1: `multi_exit_loss`(`project/models/early_exit_lstm.py`)가 클래스 비율을 전혀 반영 안 하는 순수 `F.cross_entropy` → `class_weights` 파라미터 추가(옵션, 기본값 None이라 다른 호출부(`train_early_exit.py`, `train_ap_sdn.py`)는 영향 없음)
- 원인 2 (더 결정적): `train_ap_early_exit.py`의 체크포인트 저장 기준이 raw val accuracy였음 — val set이 label 0+1로 74% 쏠려있어서, class-weighted loss로 학습해도 "다수 클래스만 찍어서 raw acc가 우연히 높은 에폭"이 선택되고 있었음 → **balanced accuracy(클래스별 recall 평균) 기준으로 체크포인트 선택하도록 변경**
- 결과: label 2 정확도 0% → 60.0%로 개선 (전체 정확도는 69.5%→57.6%로 하락했지만, 이는 "다수 클래스 찍기로 만든 가짜 높은 점수"가 없어진 것이라 더 정직한 수치). label 3은 train 2개/test 1개뿐이라 가중치를 줘도 여전히 학습 불가 — 데이터 자체가 부족한 문제라 알고리즘으로 해결 안 됨
- 검증용 체크포인트: `project/checkpoints/ap_v2/` (프로덕션 `ap_cleaned_strict` 체크포인트는 안 건드림)

## 주요 파일
- `project/scripts/collect_metrics.py` — AP 라이브 측정 스크립트. station 재연결 스파이크 버그 수정 완료, congestion_score 임계값(`JITTER_MAX_MS`, `RETRY_FAILED_MAX`) 재보정 완료
- `project/scripts/metrics_v2.csv` — 실측 데이터 누적본 (834행, 5개 시나리오, 2대 동시 부하 포함)
- `project/data/ap_metrics_v2/` — 새 실측 데이터 기반 windowed train/val/test (자체 스케일러, `ap_metrics_cleaned_strict`와 별개, train 548/val 118/test 118)
- `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt` — 기존(스케일러 다른) 체크포인트로 새 데이터 평가한 리포트
- `project/results/yongsang/ap_v2_eval_report.txt` — 새 데이터로 처음부터 학습한 체크포인트 평가 리포트
- `project/checkpoints/ap_v2/` — 새 데이터 전용 검증용 체크포인트 (class-weighted + balanced-accuracy 선택 적용)
- `project/models/early_exit_lstm.py` — `multi_exit_loss`에 옵션 `class_weights` 파라미터 추가
- `project/scripts/train_ap_early_exit.py` — inverse-frequency 클래스 가중치 계산(`compute_class_weights`) + balanced accuracy 기준 체크포인트 선택 추가
- `~/.ssh/config`, `~/.ssh/id_rsa_ap*` — 이 노트북 로컬 SSH 키/설정 (git에는 없음, 이 기기에서만 유효). AP(`root@192.168.8.1`) 비밀번호 없이 접속 가능
- `C:\Users\dkssu\anaconda3\envs\capstone` — 이 노트북에서 torch(CPU)+pandas+numpy 설치된 conda 환경 (base는 DLL 로딩 실패)
- `project/README_AP_STRICT.md` — AP strict 파이프라인 전체 기준 문서
- `docs/hochung/ap_traffic_measurement_guide.md` — AP 라이브 트래픽 측정 방법 정리 문서
- `project/deploy/raspberry_pi_ap/README.md` — Pi ONNX 8개 조합 재측정 명령어 (아직 미착수, Pi 로그인 대기 중)

## 특이사항 / 결정 사항
- **AP(Opal)가 과도한 부하에서 완전히 크래시될 수 있음**: 150Mbps×2(합계 300M) 동시 부하에서 58초 만에 WiFi 자체가 완전히 죽음(SSID 방송 중단), 물리적 재부팅 필요했음. 100M×2(합계 200M)는 9분 넘게 안정적으로 버팀 — 이 AP로 극한 테스트할 땐 200M대에서 시작해서 조심스럽게 올릴 것
- **iperf3 UDP `-b` 타겟은 실제 전달량과 다름**: 단일 스트림이든 2개 스트림이든, 이 AP의 실제 물리 채널 용량은 대략 35~50Mbps대에서 포화되는 것으로 보임(타겟을 100M로 걸든 150M로 걸든 실제 전달량은 비슷). 부하를 더 세게 걸고 싶으면 타겟 숫자보다 "몇 대가 동시에 붙어있는지"가 더 중요함
- **폰 iperf3 서버 화면 꺼짐 대응**: `termux-wake-lock` 먼저 실행해두고 `iperf3 -s` 띄우는 걸 권장 (화면 꺼지면 서버 죽을 위험)
- **Opal 포트 구분 중요**: 집 인터넷은 반드시 Opal의 **WAN 포트**에 꽂아야 함. LAN 포트에 꽂으면 Opal이 브릿지 모드처럼 동작해서 자기 관리 IP(`192.168.8.1`)가 WiFi 클라이언트에서 안 열림
- **AP 비번 ≠ Pi 비번, Pi 계정명은 `capstone`**: 호중이 알려준 비밀번호는 `root@192.168.8.1`(AP)엔 맞지만 Pi엔 안 맞음. SD카드 cloud-init 설정 확인 결과 **Pi 계정명은 `pi`가 아니라 `capstone`**(hostname `CapsTone`) — 로그인 시도할 땐 `capstone@<Pi IP>`로 해야 함
- **주의**: 어젯밤(8/22) "새 공기계"로 착각하고 극한 부하 테스트에 썼던 `192.168.8.109`(hostname "CapsTone")는 사실 폰이 아니라 **라즈베리파이 그 자체**였음(MAC `d8:3a:dd:48:55:97` → Raspberry Pi Trading Ltd 벤더 조회로 확인). 그 세션에선 파이가 이더넷으로 정상 동작했었음 — 즉 하드웨어 자체는 멀쩡했던 적이 있으므로, 지금 안 잡히는 건 케이블/SD카드 접촉 문제일 가능성이 하드웨어 고장보다 높음
- AP 모델은 GL.iNet Opal(GL-SFT1200), 관리 IP `192.168.8.1`(WAN 포트로 인터넷 연결 시 정상적으로 이 IP로 접속 가능)
- AP dropbear SSH가 오래돼서 `ssh-rsa`/RSA 키만 지원함 (ed25519 거부됨, `HostKeyAlgorithms`/`PubkeyAcceptedAlgorithms` 옵션 필수)
- AP strict `test.csv`(`ap_metrics_cleaned_strict`)는 82개 샘플, 이번에 새로 만든 `ap_metrics_v2`은 41개 샘플 — 서로 다른 데이터셋이니 항상 경로 확인할 것
- `torch`가 시스템 기본 python(anaconda base)에서 DLL 로딩 실패하는 문제는 노트북이 바뀌어도(이전 DESKTOP-5A9LEGQ, 이번 DESKTOP-29GLQJF) 계속 재현됨 — 항상 별도 conda 환경에 torch 설치할 것
- Fixed/Dynamic은 같은 체크포인트(`ap_early_exit_lstm_best.pth`)에서 threshold 정책만 다르게 평가하는 것이고, SDN은 반드시 독립 학습해야 공정 비교가 됨
