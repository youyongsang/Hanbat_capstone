# 혼잡 라벨 분류 기준

AP 트래픽 혼잡을 4단계로 분류하는 `congestion_score` 계산식과 라벨 경계를 정리한다. 1차(`ap_cleaned_strict`)와 2차(`ap_metrics_v2`)는 가중치가 다르므로 반드시 구분해서 읽는다. 두 라인의 전체 배경은 `CLAUDE.md`의 "데이터 계보" 섹션, 2차 상세는 `project/README_AP_V2.md`를 참고한다.

## 4단계 라벨

| Label | 이름 | congestion_score 범위 |
|---|---|---|
| 0 | 정상 | < 0.25 |
| 1 | 경고 | 0.25 ~ 0.50 |
| 2 | 혼잡 | 0.50 ~ 0.75 |
| 3 | 심각 | ≥ 0.75 |

라벨 경계 자체(0.25 / 0.50 / 0.75)는 1차와 2차가 동일하다. 다른 건 `congestion_score`를 만드는 가중치다.

## congestion_score 계산식

```text
congestion_score = w_t * throughput_score
                  + w_o * occupancy_score
                  + w_r * retry_failed_score
                  + w_j * jitter_score
```

sub-score는 각각 raw 측정값을 상한으로 나눠 0~1로 clamp한 값이다(`project/scripts/collect_metrics.py`의 `calculate_scores()`).

| sub-score | 계산 | 상한 |
|---|---|---:|
| `throughput_score` | `throughput_mbps / THROUGHPUT_MAX_MBPS` | 150 Mbps |
| `occupancy_score` | `channel_occupancy_percent / 100.0` | 100% |
| `retry_failed_score` | `(tx_retries_delta + tx_failed_delta) / RETRY_FAILED_MAX` | 25,000 |
| `jitter_score` | `jitter_ms / JITTER_MAX_MS` | 300 ms |

### 가중치(w_t, w_o, w_r, w_j) — 1차 vs 2차

| | throughput | occupancy | retry/failed | jitter |
|---|---:|---:|---:|---:|
| 1차 (`ap_cleaned_strict`) | 0.35 | 0.35 | 0.20 | 0.10 |
| 2차 (`ap_metrics_v2`) | 0.20 | 0.45 | 0.20 | 0.15 |

## 2차에서 가중치를 바꾼 이유

2026-08-23, `ap_metrics_v2`의 stress_load 구간(535행)에서 label 2(혼잡)와 label 3(심각)의 sub-score 평균을 비교했다.

| sub-score | label 2 평균 | label 3 평균 | 차이 |
|---|---:|---:|---:|
| throughput_score | 0.665 | 0.707 | 0.042 (거의 없음) |
| occupancy_score | 0.449 | 0.898 | 0.449 (가장 큼) |
| retry_failed_score | 0.724 | 0.834 | 0.110 |
| jitter_score | 0.512 | 0.802 | 0.290 |

`throughput_score`는 정상/경고를 가르는 덴 유용했지만 혼잡/심각을 가르는 덴 거의 기여하지 못했다. 반대로 `occupancy_score`와 `jitter_score`는 뚜렷한 차이를 보였다. 그래서 throughput 비중을 낮추고 occupancy·jitter 비중을 높였다. 이 재조정을 이미 모아둔 raw 데이터에 재적용(`project/scripts/relabel_metrics_v2.py`)한 것만으로 — AP를 다시 부하 테스트하지 않고도 — label 3 표본이 21개에서 33개로 늘었다.

## class weight power (2차, 모델 학습 시 클래스 불균형 보정)

라벨 정의와는 별개로, `train_ap_early_exit.py --class-weight-power`(기본값 1.0)가 학습 시 얼마나 label 3을 강하게 밀어붙일지를 결정한다. 이 데이터셋에서는 완만한 트레이드오프가 아니라 절벽형이었다.

| power | 전체 정확도 | Label 0 | Label 1 | Label 2 | Label 3 |
|---|---:|---:|---:|---:|---:|
| 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | 0% |
| 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | 0% |
| 1.0 | 65~66% | 95.5% | 75~77% | 36~42% | 40% |

`power=1.0`(순수 역빈도)에서만 label 3이 잡히기 시작하고, 그 대신 label 2 recall이 하락한다. "심각 미탐지가 혼잡 과잉 경고보다 더 치명적"이라는 판단으로 `power=1.0`을 기본값으로 채택했다.

## 주의

- 1차와 2차는 congestion_score 가중치가 다르므로 label 정의 자체가 다르다. 정확도나 label 분포를 같은 표에서 직접 비교하지 않는다.
- 2차의 label 3 표본은 아직 얇다(test 5개). recall 40%가 통계적으로 안정적이라 보기 어렵다.
- 자세한 재현 명령어, 현재 평가 결과 전체는 `project/README_AP_V2.md`(2차), `project/README_AP_STRICT.md`(1차)를 참고한다.
