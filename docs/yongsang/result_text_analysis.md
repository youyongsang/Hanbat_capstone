# 호중/용상 결과 파일 텍스트 분석

## 분석 기준

이 문서는 `yongsang` 브랜치에서 재학습 및 재평가를 실행한 뒤 생성된 결과 파일을 기준으로 작성했다.

| 구분 | 결과 파일 |
|---|---|
| 호중 종합 비교 결과 | `project/results/hojung/comparison_summary.csv` |
| 호중 종합 비교 텍스트 | `project/results/hojung/comparison_summary.txt` |
| 호중 Baseline LSTM 단독 평가 | `project/results/hojung/baseline_eval_report.txt` |
| 용상 Early Exit 단독 평가 | `project/results/yongsang/early_exit_eval_report.txt` |
| 용상 Stage 2 Early Exit 비교 | `project/results/yongsang/early_exit_stage2_comparison_report.txt` |

## 호중 결과 파일 분석

호중 결과 파일은 4개 제어 방식 전체를 한 표에서 비교하는 용도다. 비교 대상은 현행 임계값 방식, Baseline LSTM, 고정형 Early Exit, 동적 Early Exit이다.

| 알고리즘 방식 | 정확도 | 실측 지연 | Early Exit 비율 (1 / 2 / 3) | 해석 |
|---|---:|---:|---|---|
| Baseline 1 (Threshold) | 100.0% | 0.0024ms | - | 현재 테스트셋 라벨이 채널 점유율 기준과 강하게 맞아 떨어져 가장 높게 측정됨 |
| Baseline 2 (LSTM Full) | 97.3% | 0.3992ms | - | 학습 모델 중 정확도 최고이며 전체 LSTM 계층을 모두 통과함 |
| Baseline 3 (Early Exit Fixed theta) | 96.0% | 0.5626ms | 54.7% / 36.0% / 9.3% | 조기 종료는 발생하지만, 현재 wall-time 기준에서는 Baseline LSTM보다 느림 |
| Baseline 4 (Early Exit Dynamic theta) | 96.7% | 0.6118ms | 62.7% / 35.3% / 2.0% | 고정형보다 정확도와 종료 분포는 개선됐으나 wall-time은 더 큼 |

### 주요 관찰

1. 현행 임계값 방식이 100.0%로 나온 것은 모델 성능이 실제로 가장 우수하다는 의미라기보다, 현재 데이터 라벨 부여 방식이 채널 점유율 기준과 직접 연결되어 있기 때문으로 해석해야 한다.
2. Baseline LSTM은 학습 모델 중 가장 높은 정확도인 97.3%를 보였다.
3. Early Exit 계열은 Baseline LSTM 대비 정확도가 낮았다.
4. 고정형 Early Exit는 96.0%, 동적 Early Exit는 96.7%로 동적 방식이 0.7%p 높았다.
5. 동적 Early Exit는 Exit 1 비율을 54.7%에서 62.7%로 늘리고, Exit 3 비율을 9.3%에서 2.0%로 줄였다.
6. 그러나 `compare_baselines.py`의 실측 wall-time 기준에서는 동적 Early Exit가 고정형보다 0.0492ms 느리고, Baseline LSTM보다도 느리게 측정됐다.

## 용상 결과 파일 분석

용상 결과 파일은 Early Exit 구조 자체를 분석하기 위한 단독 평가 결과다. 이 파일에서는 레이어 깊이에 따른 구조적 지연 모델과 실제 wall-time을 함께 기록한다.

결과 리포트에는 이번 실행에 사용한 threshold 파라미터도 함께 기록했다.

| 파라미터 | 값 |
|---|---:|
| `base_theta_1` | 0.3 |
| `base_theta_2` | 0.6 |
| `high_variance` | 0.15 |
| `mid_variance` | 0.07 |
| `min_threshold` | 0.1 |
| `recent_steps` | 5 |
| `spike_threshold` | 0.2 |

| 방식 | 정확도 | 구조적 평균 지연 | 실측 wall-time | Exit 1 | Exit 2 | Exit 3 |
|---|---:|---:|---:|---:|---:|---:|
| 고정형 Early Exit | 96.0% | 3.280ms | 0.077ms | 54.7% | 36.0% | 9.3% |
| 동적 Early Exit | 96.7% | 2.827ms | 0.079ms | 62.7% | 35.3% | 2.0% |

### 주요 관찰

1. 동적 threshold는 고정 threshold보다 정확도를 0.7%p 높였다.
2. 동적 threshold는 Exit 1 종료율을 높이고 Exit 3 종료율을 크게 낮췄다.
3. 구조적 평균 지연 모델에서는 동적 Early Exit가 3.280ms에서 2.827ms로 0.453ms 감소했다.
4. 하지만 실제 wall-time은 0.077ms에서 0.079ms로 거의 같거나 아주 약간 증가했다.
5. 따라서 동적 Early Exit는 연산 경로를 줄이는 방향으로 동작했지만, 현재 Python 구현에서는 threshold 계산 및 분기 처리 오버헤드가 실제 시간 이득을 상쇄한 것으로 볼 수 있다.

## 두 결과 파일의 차이

호중 결과 파일과 용상 결과 파일은 같은 모델을 보지만 측정 목적이 다르다.

| 구분 | 호중 결과 | 용상 결과 |
|---|---|---|
| 목적 | 4개 알고리즘 전체 비교 | Early Exit 내부 동작 분석 |
| 핵심 지표 | 정확도, 실측 지연, 불필요 전환, Exit 비율 | 정확도, Exit별 정확도, Exit 비율, 구조적 지연 |
| 지연 해석 | `time.perf_counter()` 기반 실제 wall-time | Exit 깊이 기반 구조적 지연 + wall-time |
| 결론 방향 | 현재 구현에서는 Baseline LSTM이 wall-time상 유리 | 동적 threshold가 조기 종료 분포와 구조적 지연을 개선 |

## 종합 해석

이번 실행 결과만 보면 Early Exit는 아직 Baseline LSTM 대비 실측 지연 시간에서 우위를 보이지 못했다. 특히 `compare_baselines.py` 기준에서는 Baseline LSTM이 0.3992ms, 고정형 Early Exit가 0.5626ms, 동적 Early Exit가 0.6118ms로 측정되어 Early Exit 계열이 더 느리다.

다만 Early Exit가 의미 없다는 뜻은 아니다. 용상 단독 평가에서는 동적 Early Exit가 Exit 1 종료율을 높이고 Exit 3 비율을 낮춰 구조적 연산량 감소 가능성을 보여줬다. 즉, 현재 결과는 다음처럼 해석하는 것이 가장 안전하다.

> Early Exit 구조는 더 얕은 계층에서 종료되는 샘플을 늘려 구조적 연산량을 줄일 수 있음을 보였다. 그러나 현재 Python 구현의 실제 wall-time에서는 분기 처리와 동적 threshold 계산 오버헤드로 인해 Baseline LSTM보다 빠른 결과는 얻지 못했다.

## 보고서용 결론 문장

최종 보고서에는 다음과 같이 정리하는 것이 적절하다.

> 동적 Early Exit는 고정형 Early Exit 대비 정확도를 96.0%에서 96.7%로 향상시키고, Exit 3 비율을 9.3%에서 2.0%로 낮춰 더 이른 단계에서 종료되는 샘플을 증가시켰다. 구조적 지연 모델에서도 평균 지연이 3.280ms에서 2.827ms로 감소하였다. 다만 실제 Python wall-time 측정에서는 동적 threshold 계산 및 분기 처리 오버헤드로 인해 Baseline LSTM보다 지연 시간이 크게 낮아지지는 않았다. 따라서 본 결과는 동적 Early Exit의 구조적 효율 가능성을 확인한 단계이며, 실제 지연 시간 개선을 위해서는 threshold 파라미터 튜닝과 구현 최적화가 필요하다.

## 후속 작업

- 동적 threshold 파라미터 튜닝
- threshold 계산 캐싱 또는 벡터화
- batch 단위 Early Exit 처리 최적화
- 더 큰 모델 또는 더 긴 sequence에서 wall-time 재측정
- 임계값 방식이 100.0%로 나온 원인 분석 및 라벨 생성 기준 재검토
