# 혼잡 라벨 재설계 — 표준 문턱 + victim 프로브 (2026-08-27)

`ap_metrics_v2`의 `congestion_score`/`label` 정의를 근거 기반으로 다시 세운다. `congestion_label_criteria.md`의 옛(weighted-sum) 정의를 대체한다.

> 브라우저로 보기 좋은 버전: [`congestion_label_redesign.html`](congestion_label_redesign.html) — 정의·근거 중심 요약(§1~5 + 캘리브레이션). 이 markdown이 세션 로그·열린 항목까지 담은 전체본.

> **현재 상태 (2026-08-31)**: 이 설계는 **전부 구현·본수집·배포 완료**. 데이터 `ap_metrics_v2_redesign2`(2115행, feature는 이후 7개로), 세 모델 학습·ONNX·Pi 실측까지. §3 표준 앵커는 원문 대조 완료(§3 표). §6~8의 "구현 순서/열린 항목/남은 것"은 **2026-08-27 시점 기록**이며 대부분 완료됐다 — 최신 진행은 `.work-log/current.md`.

**상태 (2026-08-27 밤, 원문 보존)**: `collect_metrics.py`·`ap_features.py`·`prepare_ap_metrics_dataset.py` 구현 완료. Pi 캘리브레이션: idle(v6: 77/77 label 0) + 60/60·소패킷·45/45 부하 — **occupancy 포화 아닌 occ 60~72%에서 latency/loss 주도 label 3 확인**(45/45 런 22행). "실패=max" 채점 추가(§4) — victim 경로가 죽으면 occupancy 단독으로 안 되돌아감. **남은 것: 여러 시나리오 재수집(새 파일, 누적) → 재변환 → 재학습 → 평가.** 기존 5574행은 레거시(프로브·tx_packets 없음 + retry 3× 버그).

## 0. 한 줄 요약

라벨 = **victim 프로브(경량 UDP 스트림)의 실측 QoS + AP 채널 상태**를 각각 국제 표준 문턱에 매핑한 뒤 **`max`로 합친 값**. 모델 입력에서는 프로브 파생값과 `jitter_ms`/`latency_ms`를 빼서, 모델이 "재지 못하는 victim QoS를 채널 상태만으로 예측"하게 만든다.

## 1. 왜 재설계하나

### 현재 문제 (실측으로 확인됨, 2026-08-27)

현재: `congestion_score = 0.20·throughput + 0.45·occupancy + 0.20·retry + 0.15·jitter`, `label`은 0.25/0.5/0.75로 자름.

- **label 3(심각)이 사실상 `channel_occupancy_percent ≈ 100%`에서만 나온다.** 소패킷 부하로 occupancy 55~69%에서 retry를 맥스로 터뜨려도 congestion_score가 ~0.71에서 막혀 label 2에 갇힘. 폰이 throughput을 못 올리는 한 occupancy 66%면 나머지 축이 다 맥스여도 산술적으로 label 3 불가.
- → label 3 학습 데이터가 전부 occ=100 순간 → 모델이 "occ≈100 → label 3" 지름길만 학습
- → **ground truth 라벨 자체가 occupancy 문턱과 거의 같음** → "occupancy 문턱 대비 LSTM 우위"를 증명할 수 없음 (비교 대상이 정답과 동일한 순환논리)

### 가중치가 그렇게 된 이유 = 순환논리

현재 가중치(occ 0.45)는 2026-08-23에 **우리 데이터의 label 2 vs 3 sub-score 평균 차이**로 정했다 (occ 0.449→0.898로 차이 최대 → 가중치 최대). "우리 데이터에서 잘 갈리는 축에 큰 가중치" → 그 축이 라벨을 지배하게 됨. 외부 근거가 아니라 자기 데이터 fit.

### 재설계 원칙

1. **축별 문턱은 외부 표준에서** (ITU-T, Cisco/WLAN 가이드). 자기 데이터 fit 금지.
2. **조합은 `max`** — 표준은 "각 축이 언제 나쁜가"는 주지만 "어떻게 합치나"는 안 준다. `max`면 가중치 논쟁이 원천 봉쇄됨. `label 3 = 최소 한 축이 표준 심각 문턱 돌파`.
3. **라벨 입력 ≠ 모델 입력.** 라벨은 victim QoS(프로브) 실측에 의존. 모델은 프로브를 못 봄 → "채널 상태로 victim QoS 예측"이 모델의 일.

## 2. victim 프로브

### 무엇 / 왜

QoS에 민감한 **작고 일정한 스트림 하나**를 배경 부하와 별개로 흘리고, **그 스트림 자체의 jitter/loss를 측정**한다. "이 혼잡 속에서 VoIP 통화 한 통이 얼마나 깨지나"를 재는 것. 합성 점수가 아니라 실제 앱 피해.

부하(폰이 만드는 25M+)와는 **완전히 별개**다. 프로브는 300kbps라 아무것도 못 막는다 — 센서지 부하가 아니다.

### 어떻게 (새 기기 불필요)

```
── 부하 (혼잡 유발, 안 바뀜) ──
  폰 191 ──iperf3 -u -b 25M──┐
  폰 S26 ──iperf3 -u -b 25M──┤──> AP ──무선──> 노트북(부하 싱크)
                              │
── 프로브 (QoS 피해 측정, 신규) ──
  파이 ──iperf3 -u -b 300k -l 200──> AP ──무선 downlink──> 노트북(프로브 싱크, 포트 하나 더)
        지속 스트림. 파이가 JSON 스트림에서 구간별 jitter/loss 읽음 (APPoller 패턴)
```

- 파이→AP는 유선이지만 **AP→노트북 홉이 무선**이라 프로브가 폰들의 혼잡을 실제로 겪는다.
- 300kbps·200B ≈ 187pps → VoIP급.
- `collect_metrics.py`에 프로브 리더 스레드 추가. 새 컬럼 `probe_jitter_ms`, `probe_loss_pct`.
- `latency_ms`(ping RTT)는 유지 — iperf3 UDP는 RTT를 못 줌.

## 3. sub-score 표준 문턱

각 축을 4개 앵커(경고→0.25 / 혼잡→0.5 / 심각→0.75 / 완전→1.0)로 piecewise-linear 매핑, [0,1] clamp. **구현됨**: `collect_metrics.py`의 `ANCHORS` 딕셔너리 + `anchor_score()`.

**라벨 축 (4개)** — `congestion_score = max(이 4개)`:

| 축 | 경고 (0.25) | 혼잡 (0.5) | 심각 (0.75) | 완전 (1.0) | 출처 (표준 원문 대조 2026-08-31) |
|---|---:|---:|---:|---:|---|
| `occupancy_score` (채널 airtime %) | 40% | 55% | 75% | 90% | **심각=75, 경고≈50**: Aruba WLAN 설계 가이드(~50% good threshold, >75% 문제). 40/55/90은 그 근처 보간 — 4-티어 정식 표준은 아님 |
| `jitter_score` (프로브 IPDV) | 20ms | 30ms | 50ms | 100ms | **심각=50ms**: ITU-T Y.1541 Class 0/1 (IPDV ≤ 50ms). **경고/혼잡 20/30ms**: Cisco Enterprise QoS (voice jitter ≤ 30ms). (RFC 4594는 텔레포니를 "jitter Very Low" 정성 등급으로만 규정 §2.3, 수치는 Y.1541로 위임 — 30ms는 Cisco 값) |
| `loss_score` (프로브 패킷 손실) | 0.5% | 1% | 5% | 10% | Cisco Enterprise QoS (voice loss ≤ 1%, > 5% 사용 불가). ITU-T Y.1541 Class 0/1 IPLR ≤ 0.1%는 이보다 **엄격** — 이 스케일은 Cisco 실무 기준. (G.113 App.I는 E-model용 코덕별 Ie/Bpl 표라 손실 문턱을 규정하지 않음 — 인용에서 제외) |
| `latency_score` (ping **편도 추정** = RTT/2) | 30ms | 60ms | 150ms | 400ms | **심각=150ms, 완전=400ms**: ITU-T G.114 (편도 전송시간 <150 transparent / 150–400 acceptable-with-awareness / >400 unacceptable). 30/60ms는 그 아래 보간. **앵커는 편도 규격이고 ping은 RTT를 재므로 `calculate_scores`가 `latency_ms/2`를 넣는다** (2026-08-27 밤: RTT 생값 → 편도 앵커라 RTT 150ms(≈편도 75ms)가 "심각"으로 채점, label 3 79% 과다 → 수정. 부하행 label 3 111→85, label 2 28→54). idle RTT med ~2ms |

**라벨 축 아님 (정보용 컬럼 + 모델 입력)**:

| 축 | 값 | 왜 라벨 축이 아닌가 |
|---|---|---|
| `retry_score` (재전송 비율, 앵커 10/15/25/40%) | `(tx_retries + tx_failed) / (retries + failed + tx_packets)`, 최근 5폴링 rolling | **v4 베이스라인 (2026-08-27)**: 이 2.4GHz AP는 RF가 열악해서 **idle에도 retry_ratio가 med 18% / p90 36%**인데(자는 폰·S26 백그라운드 버스트) victim 프로브는 완벽(jitter med 1ms, loss 0%). retry는 jitter/loss를 유발하는 *원인*이지 QoS 피해의 독립 증거가 아니다. 재전송이 victim을 해치면 프로브가 잡는다. → 라벨에서 제외, `tx_retry_ratio`는 **모델 입력 feature로 유지**(모델이 "재전송 많다 → victim 곧 깨질 것" 추론) |
| `throughput_score` (상한 150Mbps) | `throughput_mbps / 150` | label 2/3 변별력 없음 (0.665→0.707). 채널이 빠르게 도는 것 자체는 혼잡 아님. 모델 입력으론 유지 |

### 실측값이 앵커 대비 어디쯤 (2026-08-27 캘리브레이션)

`정상 < 0.25 · 경고 < 0.50 · 혼잡 < 0.75 · 심각 ≥ 0.75`. 60/60 = 대표 부하.

| 축 | 경고 | 혼잡 | **심각** | 완전 | idle | 60/60 부하 (med → peak) | 소패킷 부하 (med → peak) |
|---|--:|--:|--:|--:|---|---|---|
| **occupancy** (%) | 40 | 55 | **75** | 90 | med 17 · `정상` | med 63 `혼잡` → max 87 `심각` | med 77 `심각` → max 90 `완전` |
| **probe jitter** (ms) | 20 | 30 | **50** | 100 | med 1 / max 3.5 · `정상` | med 6 → max 19 · `정상` (앵커 안 건드림) | med 10 → max 11 · `정상` |
| **probe loss** (%) | 0.5 | 1 | **5** | 10 | 0 · `정상` | med 0.27 `정상` → max 7.4 `심각` | p90 4.3 `혼잡` → max 8.2 `심각` |
| **latency** 편도=RTT/2 (ms) | 30 | 60 | **150** | 400 | RTT 3 → OW ~1.5 · `정상` | RTT med 53→OW 27 `정상` · p90 186→OW 93 `혼잡` · max 291→OW 146 `혼잡`(심각 직전) | RTT med 146→OW 73 `혼잡` → max 247→OW 124 `혼잡` |

읽는 법:
- **occupancy가 주력 축** — idle 17% 바닥, 부하에서 혼잡~심각.
- **jitter 축은 이 셋업에서 거의 안 뜬다** — peak(19ms)도 경고(20ms) 문턱 못 넘음(문서에 명시된 한계).
- **loss·latency는 peak에서만 심각/심각직전** — med는 거의 정상, 부하 셀 때 순간 튐.
- **latency는 ping RTT 측정값이고 채점은 ÷2 한 편도(OW)** — 위 표 OW가 실제 채점값.
- **"실패=max" 오버라이드** (§4): ping 무응답 / 프로브 stale → 그 축 곧장 1.0. 45/45 런 프로브 생존율 26/75라 심각의 다수가 이 경로.

정보용 축(라벨 아님): `retry` idle 18~36% / 부하 30~45% (RF 험한 2.4GHz라 idle에도 혼잡~심각대 → 라벨 축에서 뺀 이유), `throughput` 60/60 ≈ 0.4.

### 결정 (2026-08-27)
- `latency` — 라벨 축 **유지**(지연은 QoS 핵심, Y.1541/G.1010이 지연·지터를 별개 축으로 다룸). 모델 입력에선 제외.
- `retry` — **라벨 축에서 제외** (위 v4 근거). 모델 입력으론 유지. `tx_failed`는 retry 비율에 합침(별도 축 없음).
- `connected_clients` — 모델 입력에 **추가 안 함** (우리 데이터 2~3으로 변별력 없음).

### 구현 후 캘리브레이션 (2026-08-27, 프로브 켜고 idle 5회 반복)

| 축 | idle 실측 → 조치 |
|---|---|
| jitter / loss (프로브) | jitter med 1ms/max 3.5ms, loss 0% → 그대로. Y.1541 앵커에 여유 많음 |
| occupancy | med 17%, 가끔 100% 스파이크(`instantaneous_fallback` 경로가 survey 카운터 리셋 시 뱉음) → **3-폴링 median** (`occ_history`), 앵커 그대로 |
| latency | **v1~v3**: 폰(191) ping이 절전 때문에 31~295ms로 요동 → **노트북 대상으로 변경**(`SERVER_IP = 192.168.8.226`). 노트북은 Windows 방화벽이 ICMP 차단 → **inbound ICMP 허용 규칙 추가**. **v6**: 가끔 146~298ms 단일 폴링 스파이크 → **3-폴링 median** (`lat_history`). idle RTT med 3ms 깨끗, 앵커 30/60/150/400 |
| retry | **버그**: 이 AP는 `tx retries`/`tx failed`를 라디오 전체 카운터로 보고(모든 station 동일값) → station별로 더하니 station 수만큼 뻥튀기 → **수정**(단일 delta). 그래도 idle 18~36% → **라벨 축에서 제외**. rolling(5폴링) + min 표본 50 |

> **주의**: 기존 `metrics_v2.csv` 5574행의 `tx_retries_delta`/`tx_failed_delta`도 이 3× 버그를 갖고 있다(station 수만큼 뻥튀기). 재설계 데이터와 별개로 레거시 취급하는 또 하나의 이유.

**v6 idle 결과**: 77/77 행 label 0, `congestion_score` max 0.17.

### 부하 캘리브레이션 (2026-08-27, 60/60 200초 — 172초에 Opal 크래시, 168초분 확보)

| 축 | idle → 60/60 부하 | 판정 |
|---|---|---|
| occupancy | 17% → med 63% / max 87% | 앵커 OK, label 2~3 잘 뜸 |
| latency | 3ms → med 53ms / p90 186ms / max 291ms | 앵커 OK. 부하 중 ping이 몇 번 실패해 lat=0 행 있음(median으로 대부분 커버) |
| probe jitter | 1.5ms → med 6ms / max 19ms | Y.1541 앵커(20/30/50/100)를 60/60론 못 넘음. 표준 그대로 — 더 센 경합(소패킷·다중 station)에서 오를 것 |
| probe loss | 0% → med 0.27% / max 7.4% | 앵커 OK, label 3 뜸 |

**성과**: label 3 행들이 occupancy 60~73%(포화 아님)에서 latency(140~291ms)·loss(7.4%) 주도로 나옴. occupancy-only 문턱(≥75%)이면 놓쳤을 케이스 → **"occupancy 문턱 대비 LSTM 우위"의 실증 기반**. 부하 라벨 분포 0×8 / 1×1 / 2×12 / 3×9.

### `throughput_score`는 라벨에서 제외

문서(`congestion_label_criteria.md`)에 이미 기록된 대로 label 2 vs 3 변별력이 없다(0.665→0.707). 채널이 빠르게 도는 것 자체는 혼잡의 지표가 아니다. **모델 입력 feature로는 유지**한다.

### `retry_score`는 비율로

현재 `tx_retries_per_s`(초당 절대 개수) → **`tx_retry_ratio`**(재전송 비율)로 변경. 표준 문턱이 전부 비율이다. `iw station dump`가 `tx packets`(성공 전송)를 주므로:

```
tx_retry_ratio = tx_retries_delta / (tx_packets_delta + tx_retries_delta)
```

(첫 폴링·station 재등장 시 델타 0 → 비율 0. `poll_interval_s`는 이제 라벨 계산엔 불필요하지만 진단용으로 유지.)

## 4. 조합 = max

```
congestion_score = max(
    occupancy_score,   # 채널 airtime
    jitter_score,      # 프로브 IPDV
    loss_score,        # 프로브 패킷 손실
    latency_score,     # ping RTT (노트북)
)   # retry는 캘리브레이션에서 제외됨 (§3)

label = 0  if congestion_score < 0.25
        1  if < 0.50
        2  if < 0.75
        3  if ≥ 0.75
```

### 의미

> **label 3 (심각) = 최소 한 개 축이 발표된 심각 문턱을 넘었다.**
> occupancy ≥ 75% (Aruba) **OR** 프로브 jitter ≥ 50ms (Y.1541) **OR** 프로브 loss ≥ 5% (Cisco QoS) **OR** ping 편도 ≥ 150ms = RTT ≥ 300ms (G.114)

심사 방어: "이 라벨 왜 이렇게?" → "각 축을 국제 표준 문턱에 매핑하고, 하나라도 넘으면 심각으로 봤다. 가중치는 없다."

### `max`가 놓치는 것 (알려진 단순화)

여러 축이 "어중간하게 나쁜"(다 0.6대) 경우 — 각각은 심각이 아니라 label 2에 머문다. 실제로는 그런 상태가 한 축만 0.8인 것보다 나쁠 수 있다. 필요하면 나중에:
- 비정규화 p-norm `min(1, (Σ sᵢᵖ)^(1/p))` p≈6 (magic number라 방어 부담)
- 또는 app 축(jitter+loss+latency)만 ITU-T **G.107 E-model(R-factor)**로 조합하고 occupancy·retry는 별도 경고 플래그 (정석, 구현 무거움)

지금은 `max`로 시작한다.

### 실패 = max (2026-08-27, 45/45 런에서 확인)

프로브·ping의 타겟(노트북)이 같은 무선 다운링크를 타기 때문에, 채널이 심하게 포화되면 **프로브 스트림이 완료 불가 / ping 무응답**이 된다. 이때 원래 코드는 두 축을 `None → score 0.0`으로 처리했고, 라벨이 조용히 occupancy 단독으로 되돌아갔다 (재설계 목적 훼손 — occ 60~72%에서 victim이 완전히 죽은 행이 label 2로 떨어짐).

수정: **채널이 실제로 바쁜 상태에서 victim 경로가 완전히 죽으면 해당 축을 1.0으로 본다.**

```
channel_active = throughput ≥ 3 Mbps  OR  occupancy ≥ 40%
ping_hard_fail  = channel_active AND  latency(3-폴링 median) == 0     → latency_score = 1.0
probe_hard_fail = channel_active AND  프로브가 한 번은 됐었음(ever_ok) AND 지금 stale → loss_score = 1.0
```

- **왜 AP 다운이 아니라 채널 포화로 보나**: 이 판정에 도달했다는 건 유선 SSH로 AP 텔레메트리(`iw station/survey`)를 방금 정상 파싱했다는 뜻. AP는 살아있고 무선 채널만 막힌 것.
- **idle 안전장치**: `channel_active` 게이트. 무부하에 ping/프로브가 실패하는 건 셋업 문제(방화벽·로밍)지 혼잡이 아님 → override 안 함. `ever_ok` 게이트로 "프로브 서버 미기동" 케이스도 배제.
- **45/45 런 재처리 결과**: occ<75% label 3이 11 → 22행으로. 전부 `2→3` 전이(0/1→3 없음), 전부 부하 구간(throughput 27~142 Mbps), idle 행 라벨 변화 0. → CSV에서 `probe_ok==0 & loss_score==1.0` / `latency_ms==0 & latency_score==1.0`으로 override 행 식별 가능(별도 컬럼 없음).

## 5. 모델 입력 feature 재정의

### 문제: 라벨이 결정론적 규칙이면 LSTM이 왜 필요한가

`max(표준 문턱)`은 그 자체로 규칙 기반 분류기다. 배포 시점에 규칙 엔진으로 바로 계산하면 되는데 왜 LSTM을 학습하나?

### 답: 프로브가 배포 시점엔 없다

배포된 시스템엔 victim 프로브가 없다(협조하는 싱크 + 300kbps 지속 스트림이 필요). 모델은 **AP-side 텔레메트리만 보고 "지금 victim QoS가 깨지고 있는가"를 예측**해야 한다. 이건 규칙으로 못 한다 — victim 측정값이 없으니까. 거기에 시계열(윈도우 10) 추세 포착 + Early Exit 효율.

### 그래서 모델 입력에서 빼는 것

| feature | 라벨 축? | 모델 입력? | 이유 |
|---|:--:|:--:|---|
| `probe_jitter_ms` | ✅ | ❌ | victim 측정 = "정답". 배포 시 없음 |
| `probe_loss_pct` | ✅ | ❌ | 〃 |
| `jitter_ms` (기존 ping mdev) | — | ❌ **제거** | 프로브 jitter로 대체되고, 라벨 축과 겹쳐 leakage |
| `latency_ms` (ping RTT) | ✅ | ❌ **제거** | 라벨 축. 모델이 정답을 그대로 받는 꼴 (사용자 결정 2026-08-27) |
| `channel_occupancy_percent` | ✅ | ✅ 유지 | 배포 시 AP 텔레메트리로 있음. 모델이 이걸로 shortcut하는 건 "맞는 shortcut"(occ 높으면 대개 혼잡). 중요한 건 occ 중간인데 프로브가 깨진 행 — 거기선 모델이 retry/RSSI/추세로 예측해야 함 |
| `tx_retry_ratio` | ✅ | ✅ 유지 | 〃 배포 시 있음. `(retries+failed)/(retries+failed+packets)` |
| `throughput_mbps` | ❌ | ✅ 유지 | 라벨엔 없지만 부하 지표로 모델엔 유용 |
| `tx_failed` (별도) | — | ❌ | retry에 합침 (결정 2026-08-27) |
| `connected_clients` | ❌ | ❌ | 우리 데이터 2~3으로 변별력 없음, 실배포엔 일반화 안 됨 (결정 2026-08-27) |
| `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm` | ❌ | ✅ 유지 | |

### 확정 모델 입력 (7개) — 구현됨 (`ap_features.py`, authoritative)

각 feature의 정의·출처·스무딩·스케일러는 별도 레퍼런스 `docs/yongsang/model_features.md`(HTML: `model_features.html`).

```
throughput_mbps
channel_occupancy_percent
tx_retry_ratio
rssi_dbm
rssi_delta_db
rssi_moving_avg_dbm
sta_tx_bitrate_mean        ← 2026-08-29 추가 (6→7)
```

제거(9→6): `latency_ms`, `jitter_ms`(정답 leakage), `tx_retries`+`tx_failed` 2개를 `tx_retry_ratio` 하나로 통합. `connected_clients`는 원래 후보였다가 기각(우리 데이터 2~3으로 변별력 없음).

**`sta_tx_bitrate_mean` 승격 (6→7, 2026-08-29)** — 처음엔 뺐다가 다중 시드 재검증에서 되살림: 6-feature 재학습에서 occ 60~72%의 label 2 vs 3이 나머지 6 feature로는 **평균이 완전히 동일**했다(occ 66/65, throughput 66/66, retry 0.30/0.30, rssi -35/-34). 8/28 단일 진단에서는 `min`이 유휴 station 때문에 상수, `mean`이 throughput 추종이라 뺐으나 — **8/29 다중 시드(5개) 검증에서 `sta_tx_bitrate_mean`이 exit-loss 가중치와 무관하게 6-feature보다 Label3 F1 +5~11pt, occ 60~72%에서 Cohen's d=0.52**로 갈라짐이 확인되어 7번째 입력으로 승격. (방향은 실측과 반대 — 부하 테스트라 혼잡 구간에서 오히려 bitrate 오름 — 이지만 변별력은 유효.) `sta_tx_bitrate_min`은 여전히 미사용(CSV 기록만). 상세: `.work-log/current.md` 2026-08-29 2차 체크포인트.

모델의 일: `{occupancy, retry_ratio, throughput, RSSI 3종, sta_tx_bitrate_mean}`(7개)로 `max(occ, probe_jitter, probe_loss, latency)` 라벨을 예측. 못 보는 프로브 축을 채널 상태만으로 추론해야 함 → 진짜 예측 문제. (**RSSI 3종** = `rssi_dbm`(수신 신호 세기, dBm) · `rssi_delta_db`(직전 폴링 대비 변화) · `rssi_moving_avg_dbm`(5폴링 이동평균) — 상세는 `docs/yongsang/model_features.md` §2.)

> **여전히 lean함**: 모델이 볼 게 적음 = 재설계 의도(정답 못 읽고 진짜 예측). RSSI 3종은 서로 상관이라 실질 신호는 ~5개.

## 6. 구현 순서

1. ✅ **`collect_metrics.py`** (2026-08-27 구현, Pi에서 스키마 스모크 통과)
   - `ProbeRunner` 클래스: `iperf3 -u -c <노트북> -p 5203 -b 300k -l 200 -t 2 -J`를 짧게·연속 실행하는 백그라운드 스레드. 각 테스트의 서버 측정 jitter/loss를 캐시. `PROBE_STALE_S`(12초) 넘으면 stale → 축 미사용. 버전 robust하게 지속 스트림 파싱 대신 짧은 테스트 반복.
   - `parse_station_info`에 `tx packets:` 추가, `calculate_station_deltas`가 `packets_delta` 반환. `tx_retry_ratio = (retries+failed) / (retries+failed+packets)`.
   - `anchor_score()` + `ANCHORS` 딕셔너리. `calculate_scores()` = `max(occ, jitter, loss, latency)` (retry는 캘리브레이션에서 제외). `throughput_score`는 계산은 하되 max에서 제외.
   - **실패 = max** (2026-08-27, §4): `channel_active`(throughput≥3 or occ≥40)에서 ping 무응답 → `latency_score=1.0`, 프로브(ever_ok 후) stale → `loss_score=1.0`. `ProbeRunner.get()`이 4번째 값 `ever_ok` 반환.
   - CSV 27컬럼 (`probe_jitter_ms`, `probe_loss_pct`, `probe_ok`, `tx_retx_delta`, `tx_packets_delta`, `tx_retry_ratio` + sub-score 6개). ping jitter는 제거(latency만 남김).
   - `prepare_csv()` 가드가 옛 스키마 CSV에 append 거부 → 재설계는 **새 파일**로 수집.
2. ✅ **`utils/ap_features.py`**: 모델 feature 6개.
3. ✅ **`prepare_ap_metrics_dataset.py`**: `model_excluded_columns` 갱신.
4. ⬜ **노트북**: `iperf3 -s -p 5203` 인스턴스 추가 (프로브 싱크). — 다음 하드웨어 세션
5. ⬜ **`congestion_label_criteria.md`**: 이 설계로 개정 (지금은 상단 배너 + 이 문서 포인터만).
6. ⬜ **retry 앵커 idle 재보정** (§3 관찰) → **재수집(새 파일) → 재변환 → 재학습 → 평가**. — 다음 하드웨어 세션

## 7. 기존 데이터 (`metrics_v2.csv` 5574행) 처리

**완전 relabel 불가.** 옛 데이터에 없는 것:
- `probe_jitter_ms`, `probe_loss_pct` (프로브 자체가 없었음)
- `tx_packets` (retry 비율 계산 불가 — `tx_retries_per_s`만 저장됨)

가능한 것: `max(occupancy_score, ping_jitter_score, ping_latency_score)` 만으로 부분 relabel.

→ **옛 5574행은 "레거시 / 부분축 라벨"로 표기.** 사전학습(pretraining)이나 ablation 비교용으로는 쓰되, 최종 평가·논문 수치는 새 스키마 데이터로. `metrics_v2_legacy.csv`로 아카이브하고 새 파일 시작하는 것도 고려.

## 8. 열린 항목 / 리스크

- **latency 축**: ping 타겟을 폰→노트북으로 바꾼 뒤 idle RTT ~2ms로 깨끗해짐(폰은 전력절약으로 31~295ms 노이즈였음). 45/45 런에서 부하 시 50~250ms(스파이크 814ms)로 잘 반응. 단 노트북이 로밍하면 타겟이 사라지므로 수집 중 노트북을 Opal에 고정 필수(집 공유기 프로필 수동 연결).
- **프로브 생존율**: 45/45 부하 중 `probe_ok` 26/75. 나머지는 "실패=max"로 커버되지만, 프로브가 더 자주 살아남게 하려면 `PROBE_RATE`/`PROBE_LEN`/재시도 간격 튜닝 여지.
- **표준 문턱의 출처** (2026-08-31 원문 대조 완료): 심각 앵커는 표준 지지됨 — jitter 50ms = Y.1541 Class 0/1 IPDV, latency 150/400ms = G.114 티어 경계, occupancy 75% = Aruba. 경고/혼잡 앵커(jitter 20/30, latency 30/60, occupancy 40/55)는 Cisco Enterprise QoS voice 값 또는 그 아래 보간 — 정식 표준 아님. **정정된 것**: RFC 4594는 수치 없음(정성 등급만, Y.1541 위임), G.113 App.I는 손실 문턱 아님(E-model 계수 표) → 두 인용 제거. loss 스케일(0.5~10%)은 Y.1541(0.1%)보다 느슨한 Cisco 실무 기준.
- **occupancy·retry가 라벨 축이자 모델 입력** — leakage 논란 여지. "배포 시 available하니 정당" 논리로 방어하되, occ만으로 label 3 되는 행이 여전히 많으면 occ의 심각 앵커를 올리거나(90%) occ를 라벨에서 빼는 것도 검토.
- **`max`의 단순화** (§4) — 여러 축 어중간한 경우. E-model이 정석이지만 무거움.
- **데이터 손실**: 5574행이 사실상 레거시. label 3(56개)도 새로 쌓아야 함.
- **프로브가 측정을 교란**: 300kbps는 작지만 0은 아님. 프로브 유무로 나눠서 한 번 비교 권장.

## 참고

- `docs/yongsang/congestion_label_criteria.md` — 현재(구) 정의
- `project/results/yongsang/ap_v2_redesign2_threshold_comparison.txt` — "occupancy 문턱 vs 학습 모델" 실측 대조 ("핵심 검증 질문"의 결과)
- `.work-log/current.md` — 2026-08-27 저녁 세션 (문제 발견 경위)
- **앵커 근거** (원문 대조 2026-08-31): ITU-T Y.1541 (jitter IPDV ≤ 50ms, loss IPLR ≤ 0.1% — Class 0/1) · ITU-T G.114 (편도 지연 150 / 400ms) · Cisco Enterprise QoS SRND (voice: 편도 ≤ 150ms, jitter ≤ 30ms, loss ≤ 1%) · Aruba WLAN 설계 가이드 (channel utilization ~50% / 75%) · ITU-T G.107 E-model (§4 대안 조합 방식). RFC 4594·G.113 App.I는 앵커 수치 근거로 부적합(위 §3 참조).

### 소패킷 25/25 부하 캘리브레이션 (2026-08-27, 180초 완주)

폰이 제대로 뿜음(각 20~35Mbps). 55 load rows(occ>35%):

| 축 | 부하 시 | 
|---|---|
| occupancy | med 77% / max 90% (소패킷이 airtime 포화) |
| latency | med 146ms / max 247ms |
| probe jitter | med 10ms / max 11ms — Y.1541 앵커(20ms) 안 넘음. 이 셋업에선 매우 조용한 축 |
| probe loss | med 0.3% / p90 4.3% / max 8.2% |

load labels 2×7 / 3×48, 주도 occ 42 / lat 9 / loss 4.

**두 런 종합**: label 3 셋이 다양해짐 —
- 60/60 → latency·loss 주도, occupancy **60~73%**(포화 아님)
- 소패킷 → occupancy 주도, occ 72~90%

occupancy-only 문턱(≥75%)이면 60/60의 occ 60~73% label 3을 놓친다 → **재설계가 "occupancy 외 혼잡" 라벨 데이터를 실제로 생성함**(핵심 검증 질문의 실증 기반).

**남은 이슈**: (1) jitter 축이 이 셋업에선 거의 안 뜸(표준 유지하되 기여 낮음), (2) 부하 중 ping이 가끔 2~3연속 실패해 `latency_ms=0` 행이 생김(median으로 대부분 커버). (3) 표본이 아직 얇음 — 여러 시나리오로 본격 수집 필요.
