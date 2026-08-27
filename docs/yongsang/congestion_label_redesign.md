# 혼잡 라벨 재설계 — 표준 문턱 + victim 프로브 (2026-08-27 설계, 미착수)

`ap_metrics_v2`의 `congestion_score`/`label` 정의를 근거 기반으로 다시 세운다. 현재 살아있는 정의는 `congestion_label_criteria.md`, 이 문서는 **그걸 대체할 제안**이다.

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

| 축 | 경고 (0.25) | 혼잡 (0.5) | 심각 (0.75) | 완전 (1.0) | 출처 |
|---|---:|---:|---:|---:|---|
| `occupancy_score` (채널 airtime %) | 40% | 55% | 75% | 90% | Cisco/Aruba WLAN 설계 가이드 (>50% 경고, >75% 혼잡), Ekahau 실무 기준 |
| `jitter_score` (프로브 IPDV) | 20ms | 30ms | 50ms | 100ms | ITU-T Y.1541 Class 0/1 (IPDV ≤ 50ms), RFC 4594 (텔레포니 ~30ms), Cisco (voice < 30ms) |
| `loss_score` (프로브 패킷 손실) | 0.5% | 1% | 5% | 10% | Cisco Enterprise QoS (voice loss < 1%, > 5% 사용 불가), ITU-T G.113 App.I (손실 손상), ITU-T Y.1541 |
| `latency_score` (ping RTT) | 150ms | 250ms | 400ms | 700ms | ITU-T G.114 (편도 ≤150ms 양호, >400ms 불가). **파이 홉 때문에 idle RTT가 이미 70~125ms라 앵커를 후하게 잡음** — 수집 토폴로지 바뀌면 재보정 필요 |
| `retry_score` (재전송 비율) | 10% | 15% | 25% | 40% | WLAN 헬스 모니터링 consensus (Cisco / Ekahau / 7signal: retry rate < 10% 정상, > 20% 불량) |

**결정 (2026-08-27)**:
- `latency` — 라벨 축 유지(지연은 QoS 핵심). 단 모델 입력에선 제외.
- `retry` — `tx_failed`를 합침: `retry_ratio = (tx_retries + tx_failed) / (tx_retries + tx_failed + tx_packets)`. 별도 `tx_failed` 축 없음.
- `connected_clients` — 모델 입력에 **추가 안 함**. 우리 데이터에선 2~3으로 변별력 없고 실배포(수십~수백)엔 일반화 안 됨.

**구현 후 관찰 (2026-08-27, 프로브 없이 스키마 스모크)**: idle 상태에서 `tx_retry_ratio` ≈ 8% (retx 18 / packets 207). retry 경고 앵커(10%)에 이미 가까움 → **idle 베이스라인 측정 후 retry 앵커 상향 가능성**(예: 15/20/30/50%). 다음 하드웨어 세션에서 프로브 켜고 idle 30~60초 재보정.

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
    occupancy_score,
    jitter_score,      # 프로브
    loss_score,        # 프로브
    latency_score,     # ping RTT
    retry_score        # 비율
)

label = 0  if congestion_score < 0.25
        1  if < 0.50
        2  if < 0.75
        3  if ≥ 0.75
```

### 의미

> **label 3 (심각) = 최소 한 개 QoS 축이 발표된 심각 문턱을 넘었다.**
> occupancy ≥ 75% (WLAN 가이드) **OR** 프로브 jitter ≥ 50ms (Y.1541) **OR** 프로브 loss ≥ 5% (Cisco) **OR** ping RTT ≥ 300ms (G.114) **OR** retry 비율 ≥ 25% (WLAN 헬스)

심사 방어: "이 라벨 왜 이렇게?" → "각 축을 국제 표준 문턱에 매핑하고, 하나라도 넘으면 심각으로 봤다. 가중치는 없다."

### `max`가 놓치는 것 (알려진 단순화)

여러 축이 "어중간하게 나쁜"(다 0.6대) 경우 — 각각은 심각이 아니라 label 2에 머문다. 실제로는 그런 상태가 한 축만 0.8인 것보다 나쁠 수 있다. 필요하면 나중에:
- 비정규화 p-norm `min(1, (Σ sᵢᵖ)^(1/p))` p≈6 (magic number라 방어 부담)
- 또는 app 축(jitter+loss+latency)만 ITU-T **G.107 E-model(R-factor)**로 조합하고 occupancy·retry는 별도 경고 플래그 (정석, 구현 무거움)

지금은 `max`로 시작한다.

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

### 확정 모델 입력 (6개) — 구현됨 (`ap_features.py`)

```
throughput_mbps
channel_occupancy_percent
tx_retry_ratio
rssi_dbm
rssi_delta_db
rssi_moving_avg_dbm
```

9개 → 6개. 제거: `latency_ms`, `jitter_ms`, `tx_failed` 별도 축, `connected_clients`.

모델의 일: `{occupancy, retry_ratio, throughput, RSSI×3}`로 `max(occ, probe_jitter, probe_loss, latency, retry)` 라벨을 예측. occupancy·retry는 부분 신호로 갖지만, 못 보는 프로브 jitter/loss/latency를 추론해야 함 → 진짜 예측 문제.

> **6개는 lean함**: 모델이 볼 게 적어짐 = 재설계 의도(정답 못 읽고 진짜 예측)지만, recall이 더 낮게 나올 수 있음 — 예상된 트레이드오프. RSSI 3개는 서로 상관이라 실질 신호는 ~4개.

## 6. 구현 순서

1. ✅ **`collect_metrics.py`** (2026-08-27 구현, Pi에서 스키마 스모크 통과)
   - `ProbeRunner` 클래스: `iperf3 -u -c <노트북> -p 5203 -b 300k -l 200 -t 2 -J`를 짧게·연속 실행하는 백그라운드 스레드. 각 테스트의 서버 측정 jitter/loss를 캐시. `PROBE_STALE_S`(12초) 넘으면 stale → 축 미사용. 버전 robust하게 지속 스트림 파싱 대신 짧은 테스트 반복.
   - `parse_station_info`에 `tx packets:` 추가, `calculate_station_deltas`가 `packets_delta` 반환. `tx_retry_ratio = (retries+failed) / (retries+failed+packets)`.
   - `anchor_score()` + `ANCHORS` 딕셔너리. `calculate_scores()` = `max(occ, jitter, loss, latency, retry)`. `throughput_score`는 계산은 하되 max에서 제외.
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

- **latency 축이 약함**: 프로브가 아니라 ping이고, 파이→AP(유선)→폰(무선)→왕복이라 idle 베이스라인이 이미 70~125ms. 셋업별 캘리브레이션 필요할 수 있음. 최악의 경우 latency 축을 라벨에서 빼고 jitter+loss만.
- **표준 문턱의 출처 정밀화**: 위 표의 앵커 값은 "consensus 근처"로 잡은 것. 실제 인용 시 정확한 문서·조항 확인 필요 (Y.1541 표 X, G.114 §Y, Cisco 문서 버전 등).
- **occupancy·retry가 라벨 축이자 모델 입력** — leakage 논란 여지. "배포 시 available하니 정당" 논리로 방어하되, occ만으로 label 3 되는 행이 여전히 많으면 occ의 심각 앵커를 올리거나(90%) occ를 라벨에서 빼는 것도 검토.
- **`max`의 단순화** (§4) — 여러 축 어중간한 경우. E-model이 정석이지만 무거움.
- **데이터 손실**: 5574행이 사실상 레거시. label 3(56개)도 새로 쌓아야 함.
- **프로브가 측정을 교란**: 300kbps는 작지만 0은 아님. 프로브 유무로 나눠서 한 번 비교 권장.

## 참고

- `docs/yongsang/congestion_label_criteria.md` — 현재(구) 정의
- `project/README_AP_V2.md` — "핵심 검증 질문"
- `.work-log/current.md` — 2026-08-27 저녁 세션 (문제 발견 경위)
- ITU-T G.114, G.107, G.113, Y.1541 · RFC 4594 · Cisco Enterprise QoS Design Guide
