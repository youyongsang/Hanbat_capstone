# 모델 입력 feature 기준 (현행 7-feature)

현행 2차 라인(`ap_metrics_v2_redesign2`)의 **모델 입력 7개 feature** 레퍼런스. 각 feature가 무엇이고, 어디서 나오고, 어떻게 스무딩되고, 왜 이건 라벨 축이 아닌데 모델 입력인지 정리한다.

- **정본**: `project/utils/ap_features.py`의 `AP_FEATURE_COLUMNS`. 이 문서와 어긋나면 코드가 맞다.
- **라벨 정의**는 여기가 아니라 `docs/yongsang/congestion_label_redesign.{md,html}` (`max(표준 앵커)` + victim 프로브).
- 옛 `congestion_label_criteria.{md,html}`은 **4개 sub-score 가중합 시절**이라 stale — feature 관련해선 보지 말 것.

> **핵심 원칙 — 라벨 입력 ≠ 모델 입력.** 라벨(`congestion_score`)은 victim 프로브(jitter·loss)와 ping(latency)로 만든다. 모델은 그걸 **못 본다.** 모델의 일은 **AP-side 텔레메트리(채널 상태)만 보고 "지금 victim QoS가 깨지고 있는가"를 예측**하는 것. 그래서 정답을 만드는 축(latency/jitter/loss)은 입력에서 뺀다.

---

## 1. 7개 feature (순서 고정)

| # | feature | 한 줄 | 원천 | 스무딩 | scaler min/max (train) |
|--:|---|---|---|---|---|
| 1 | `throughput_mbps` | 채널 총 전송률 | `iw station dump` 바이트 카운터 델타 | 없음 (raw) | 0.0 / 177.83 |
| 2 | `channel_occupancy_percent` | 채널 airtime 점유율 | `iw survey dump` busy/active 카운터 델타 | **3-폴링 median** | 13.33 / 93.1 |
| 3 | `tx_retry_ratio` | 재전송 비율 | `iw station dump` retries/failed/packets 델타 | **5-폴링 rolling** (denom ≥ 50) | 0.0 / 0.7593 |
| 4 | `rssi_dbm` | 이번 폴링 station 평균 신호세기 | `iw station dump` `signal avg` | 없음 (raw) | −57.0 / 0.0 |
| 5 | `rssi_delta_db` | 직전 폴링 대비 RSSI 변화 | 4번의 폴링 간 차분 | 없음 (1차 차분) | −43.0 / 37.0 |
| 6 | `rssi_moving_avg_dbm` | RSSI 이동평균 | 4번을 5-폴링 평균 | **5-폴링 평균** | −53.6 / −27.93 |
| 7 | `sta_tx_bitrate_mean` | 실제 송신한 station들의 PHY rate 평균 | `iw station dump` `tx bitrate` | 없음 (raw, 활성 station만) | 0.0 / 150.0 |

폴링 주기는 파이 유선 수집 기준 대략 1초. window는 최근 **10 폴링** (`[1, 10, 7]`). (4~6번 **RSSI** = 수신 신호 세기, §2에서 설명.)

---

## 2. feature별 상세

### 1. `throughput_mbps`
- **계산**: `(rx_delta + tx_delta) 바이트 × 8 / 경과초 / 1e6`. 모든 station 합산.
- **의미**: 채널이 얼마나 바쁘게 데이터를 나르는가. 부하 지표.
- **라벨 축 아님**: `throughput_score`(상한 150)는 계산되지만 `max()`에 안 들어간다 — label 2 vs 3 변별력이 없다(stress 구간 평균 0.665 → 0.707). "빠르게 도는 것 자체는 혼잡이 아니다."
- **왜 모델 입력**: 다른 feature(occ, retry)를 해석하는 맥락. occ 60%가 throughput 5Mbps 때문인지 100Mbps 때문인지에 따라 의미가 다르다.

### 2. `channel_occupancy_percent`
- **계산**: `busy_delta / active_delta × 100` (survey 카운터가 정상 증가할 때). 카운터 리셋 시 현재 비율로 폴백. clamp [0, 100].
- **스무딩**: `instantaneous_fallback` 경로가 카운터 리셋 순간 100% 스파이크를 뱉어서 **3-폴링 median**으로 안정화(`occ_history`). CSV·스코어·모델 입력 모두 median 값.
- **라벨 축이면서 모델 입력**: `occupancy_score`가 `max()`의 한 축(앵커 40/55/75/90%). 동시에 모델 입력 — 배포 시 AP 텔레메트리로 available하니 정당하고, "occ 높으면 대개 혼잡"은 **맞는 shortcut**이다. 중요한 건 occ가 중간(60~72%)인데 프로브가 깨진 행 — 거기선 모델이 retry/RSSI/bitrate/추세로 예측해야 한다.

### 3. `tx_retry_ratio`
- **계산**: `(tx_retries + tx_failed) 델타 합 / (그 합 + tx_packets 델타 합)`, 최근 **5-폴링 rolling**. 분모(성공+실패 프레임 수)가 50 미만이면 표본 부족 → 0.0.
- **왜 비율인가**: 절대 개수(`tx_retries_per_s`)는 폴링 주기에 비례해 흔들린다(4초 폴링 = 1초 폴링 × 4). 비율이면 폴링 주기 무관. 9→6 축소 때 `tx_retries` + `tx_failed` 2개를 이 하나로 통합.
- **라벨 축 아님**: 이 2.4GHz AP는 RF가 험해서 **idle에도 retry_ratio med 18% / p90 36%**(자는 폰 백그라운드 버스트)인데 victim 프로브는 완벽. retry는 jitter/loss를 유발하는 *원인*이지 QoS 피해의 독립 증거가 아니다 — 피해가 나면 프로브가 잡는다. → `max()`에서 제외.
- **왜 모델 입력**: "재전송 많다 → victim 곧 깨질 것"이라는 선행 신호를 모델이 추론.

### 4~6. RSSI 3종 (`rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm`)

> **RSSI** (Received Signal Strength Indicator) = AP가 각 station에서 받는 **무선 신호의 세기**. 단위 dBm, 항상 음수이고 **0에 가까울수록 강함**(−30dBm 아주 강함 / −50 좋음 / −70 약함 / −80 이하 불안정). 원천은 `iw station dump`의 `signal avg` 줄, station별 값을 이번 폴링에서 평균낸다.
>
> **왜 혼잡과 관련**: 신호가 약하면 rate control이 낮은 MCS로 떨어져 **같은 데이터를 보내는 데 airtime을 더 쓴다** → 채널 점유가 오르고 다른 station이 굶는다. victim 피해의 **원인 쪽** 신호라서 라벨 축은 아니고 모델 입력으로만 쓴다.
>
> 같은 원천을 3가지로 가공했다 — 서로 상관이 높아 실질 정보는 ~1.5개 축:

| feature | 계산 | 잡는 것 |
|---|---|---|
| `rssi_dbm` | 이번 폴링 station 평균 (raw) | 지금 신호 세기 |
| `rssi_delta_db` | `현재 − 직전 폴링`, 첫 폴링 0 | 급격히 나빠지는 중인지 (station 이동·간섭 유입) — 추세 |
| `rssi_moving_avg_dbm` | 최근 **5-폴링** 평균 (`rssi_history`) | 단일 폴링 노이즈 걷어낸 기저선 |

### 7. `sta_tx_bitrate_mean`
- **계산**: `iw station dump`의 `tx bitrate`(PHY rate)를 **이번 폴링에 실제로 송신한 station**(tx_packets 증가)만 골라 평균. 활성 station 없으면 0.0. 연결만 되고 트래픽 없는 station은 rate가 MCS 0 바닥에 stale하게 물려 있어 제외.
- **왜 추가됐나 (6→7, 2026-08-29)**: occ 60~72% 구간에서 나머지 6개 feature 평균이 label 2 vs 3 사이에 **완전히 동일**해진다(occ 66/65, thr 66/66, retry 0.30/0.30, rssi −35/−34). 여기서 `sta_tx_bitrate_mean`이 두 라벨을 **Cohen's d = 0.52**로 가른다. 5개 랜덤 시드 검증에서 exit-loss 가중치와 무관하게 7-feature가 6-feature보다 Label3 F1 +5~11pt.
- **신호 방향 주의**: 가설은 "혼잡할수록 rate collapse(하락)"였으나 **실측은 반대** — 부하 테스트라 혼잡 구간에서 기기가 데이터를 계속 밀어넣어 오히려 오른다(한가하면 관리 프레임만 잡혀 저속). 방향이 반대일 뿐 변별력은 유효.
- `sta_tx_bitrate_min`은 CSV에 기록만 하고 미사용(min은 위 검증에서 mean만큼 신호가 뚜렷하지 않았음).

---

## 3. 라벨 축 vs 모델 입력

### 3.1 "라벨 축"과 "모델 입력"은 서로 다른 것 (헷갈리기 쉬움)

| | 라벨 축 (`congestion_score`의 재료) | 모델 입력 (7개 feature) |
|---|---|---|
| **정체** | 정답(y)을 **만드는** 값 | 모델이 **보는** 값(X) |
| **뭐가** | `occupancy_score`, `jitter_score`, `loss_score`, `latency_score` → `max()` → 0/1/2/3 | 위 §1의 7개 |
| **모델이 봄?** | ❌ **절대 안 봄** (학습·추론 둘 다) | ✅ |
| **언제 존재?** | 학습/평가 데이터에만 (정답 채점용) | 학습·추론 모두 (동일해야 함) |

- **학습**: `7개 feature` → 모델이 라벨 예측 → 진짜 라벨(라벨 축으로 만든 것)과 비교 → 가중치 수정. 라벨 축 값(jitter_score 등)은 feature 벡터에 **없다**.
- **추론(배포)**: 똑같은 `7개 feature` → 모델이 라벨 예측. 여기엔 정답 라벨도, 라벨 축(victim 프로브·ping)도 아예 없다 — 그게 모델이 맞혀야 하는 것.
- **"라벨 축 = 학습용, 모델 입력 = 추론용"이 아니다.** 7개 feature는 학습·추론에서 **똑같이** 쓰이고, 라벨 축은 어느 쪽에서도 모델에 안 들어간다.

**세 가지 경우:**

| 컬럼 | 라벨 축? | 모델 입력? | 왜 |
|---|:--:|:--:|---|
| `jitter_score`·`loss_score`·`latency_score` (`latency_ms`, victim 프로브) | ✅ | ❌ | 정답 재료 + 배포 시 측정 불가(협조 싱크·프로브 필요) → 주면 **커닝** |
| `channel_occupancy_percent` | ✅ | ✅ | 정답 재료지만 배포 시 AP 텔레메트리로 있음 → 줘도 됨 ("맞는 shortcut") |
| `tx_retry_ratio`·`throughput`·RSSI 3종·`sta_tx_bitrate_mean` | ❌ | ✅ | 정답과 무관, 채널 상태 신호 |

> 모델의 일 = **라벨 축(jitter/loss/latency)을 못 보는 상태에서, 채널 상태 7개만으로 그 라벨을 예측**. latency/jitter를 입력에 넣으면 정답을 그대로 베끼는 꼴이라 뺐다.

### 3.2 컬럼별 판정

**모델 입력 = 아래 7개** (§1과 동일). 이 중 `channel_occupancy_percent`만 라벨 축을 겸한다.

| 모델 입력 (7개) | 라벨 축? | 이유 |
|---|:--:|---|
| `channel_occupancy_percent` | ✅ (`occupancy_score`) | 배포 시 AP 텔레메트리로 available. "occ 높으면 대개 혼잡"은 맞는 shortcut |
| `throughput_mbps` | ❌ (`throughput_score` 계산만, max 제외) | 부하 맥락 |
| `tx_retry_ratio` | ❌ (`retry_score` 계산만, max 제외) | 배포 시 available. "재전송 많다 → victim 곧 깨질 것" 선행 신호 |
| `rssi_dbm` | ❌ | 링크 품질 (victim 피해의 원인 쪽 신호) |
| `rssi_delta_db` | ❌ | 신호 악화 추세 |
| `rssi_moving_avg_dbm` | ❌ | 링크 품질 기저선 |
| `sta_tx_bitrate_mean` | ❌ | occ 60~72%에서 label 2/3 변별 (2026-08-29 추가) |

**라벨 축이지만 모델 입력에서 뺀 것** (정답 leakage — 핵심):

| 제외 컬럼 | 라벨 축 | 왜 모델엔 안 주나 |
|---|:--:|---|
| `latency_ms` (ping RTT) | `latency_score` | 라벨을 만드는 축 → 모델이 정답을 그대로 받는 꼴 |
| `probe_jitter_ms` | `jitter_score` | victim 실측 = "정답". 배포 시 프로브 없음 |
| `probe_loss_pct` | `loss_score` | 〃 |

**애초에 라벨 축도 모델 입력도 아닌 것**: `jitter_ms`(기존 ping mdev — 9-feature 시절 입력이었으나 victim 프로브 jitter로 대체, 라벨 축과 상관 높아 leakage 우려로 제거), `connected_clients`(후보였으나 우리 데이터 2~3으로 변별력 없음 → 기각).

**`collect_metrics.py`가 계산하는 sub-score는 6개** (`occupancy_score`, `jitter_score`, `loss_score`, `latency_score`, `throughput_score`, `retry_score`) — 그중 **4개**(occupancy, jitter, loss, latency)만 `max()`에 들어가 라벨이 된다. 6개 전부 `model_excluded_columns`라 모델 입력엔 안 들어간다.

`dataset_summary.json`의 `model_excluded_columns` (19개):
```
timestamp, scenario, poll_interval_s, channel_occupancy_method,
packet_loss_udp_percent, connected_clients, latency_ms,
probe_jitter_ms, probe_loss_pct, probe_ok, tx_retx_delta, tx_packets_delta,
throughput_score, occupancy_score, jitter_score, loss_score, latency_score,
retry_score, congestion_score
```

---

## 4. 정규화 (scaler)

- 방식: `(x − min) / (max − min)`, `[0, 1]` clip.
- **min/max는 train split에서만** 구해서 `project/data/ap_metrics_v2_redesign2/scaler_params.json`에 저장. val/test·실시간 추론은 **그 파일을 그대로** 써야 한다 (재라벨링만 하고 재변환 안 하면 라벨과 스케일러가 어긋난다).
- windowed 변환: window 10, stride 1, 시나리오별 첫 행 1개 drop. train 1437 / val 308 / test 310 샘플, shape `(N, 10, 7)`.

---

## 5. feature 변천 요약

```
1학기          4개   RPS, occupancy, packet_loss, latency (시뮬레이터 데이터)
초기 ap_metrics_v2   9개   throughput, occupancy, latency_ms, jitter_ms,
                          tx_retries_delta, tx_failed_delta, RSSI 3종
                            │ 라벨 재설계 (2026-08-27)
                            │  −2  latency_ms, jitter_ms  (정답 leakage)
                            │  −1  tx_retries + tx_failed → tx_retry_ratio
redesign        6개   throughput, occupancy, tx_retry_ratio, RSSI 3종
                            │  +1  sta_tx_bitrate_mean  (2026-08-29, occ 60~72% 변별)
redesign2       7개   ← 현행
```

**RSSI 3종** = `rssi_dbm` · `rssi_delta_db` · `rssi_moving_avg_dbm` (문서 전체에서 이 셋을 묶어 부르는 표기, §2 참조).

상세: `project/utils/ap_features.py` 상단 주석, `.work-log/current.md` 2026-08-29 체크포인트.

---

## 참고

- `project/utils/ap_features.py` — 정본 목록
- `docs/yongsang/congestion_label_redesign.{md,html}` — 라벨 정의(`max` 앵커 + victim 프로브)
- `project/scripts/collect_metrics.py` — feature 계산 구현 (`APPoller`, `summarize_stations`, `calculate_channel_occupancy`, `calculate_station_deltas`)
- `project/demo/API.md` §3 — 데모용으로 고정된 feature/추론 계약
- `project/data/ap_metrics_v2_redesign2/dataset_summary.json` — 기계 판독용 (features, excluded, scaler)
