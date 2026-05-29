# 유용상 Stage 3 작업 설명

## 작업 목표

Early Exit LSTM의 고정 threshold 방식과 동적 threshold 방식을 최종 검증하고, 장예나가 제공한 시나리오 분리 데이터 기준으로 상황별 성능 차이를 분석했다.

Stage 3에서는 단순 전체 정확도뿐 아니라 다음 항목을 함께 확인했다.

- 고정 θ와 동적 θ의 정확도 비교
- Exit 1, Exit 2, Exit 3 종료율 비교
- 시나리오 0~3별 정확도 비교
- Standard LSTM Full Inference 대비 정확도와 실측 추론 시간 비교
- 최종 전달용 checkpoint 및 `model_info.json` 저장

---

## 사용 데이터

| 항목 | 값 |
|---|---|
| 데이터 경로 | `project/data/real/test.csv` |
| 시나리오 포함 데이터 | `project/data/real/test_with_scenario.csv` |
| 입력 shape | `(N, 10, 4)` |
| feature | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| label | `0`, `1`, `2`, `3` |
| 전체 테스트 샘플 | 351 samples |

시나리오별 샘플 수는 다음과 같다.

| scenario_id | 시나리오 | 샘플 수 |
|---:|---|---:|
| 0 | `startup_surge` | 104 |
| 1 | `emergency_ramp` | 83 |
| 2 | `lunch_restart` | 84 |
| 3 | `imbalanced_ap_load` | 80 |

---

## 구현 및 산출 파일

| 파일 | 역할 |
|---|---|
| `project/scripts/generate_test_with_scenario.py` | `test.csv`에 `scenario_id` 컬럼을 추가한 `test_with_scenario.csv` 생성 |
| `project/scripts/split_scenario_data.py` | 시나리오별 CSV 및 유용상 분석 템플릿 생성 |
| `project/scripts/fill_scenario_analysis.py` | Early Exit 모델 예측 결과로 시나리오 분석 템플릿 채우기 |
| `project/results/yongsang/scenario_analysis_template.csv` | 예측 결과 입력 전 템플릿 |
| `project/results/yongsang/scenario_analysis_filled.csv` | fixed/dynamic 예측 결과가 채워진 최종 파일 |
| `project/results/yongsang/scenario_analysis_summary.csv` | 시나리오별 정확도와 Exit 종료율 요약 |
| `project/results/scenario_analysis/scenario_0_analysis.csv` | 시나리오 0 분석 결과 |
| `project/results/scenario_analysis/scenario_1_analysis.csv` | 시나리오 1 분석 결과 |
| `project/results/scenario_analysis/scenario_2_analysis.csv` | 시나리오 2 분석 결과 |
| `project/results/scenario_analysis/scenario_3_analysis.csv` | 시나리오 3 분석 결과 |
| `project/checkpoints/early_exit_fixed_final.pth` | 고정 θ 모드 구분용 alias checkpoint |
| `project/checkpoints/early_exit_dynamic_final.pth` | 동적 θ 모드 구분용 alias checkpoint |
| `project/checkpoints/model_info.json` | 최종 파라미터 및 성능 정보 |

---

## Checkpoint 정리

Stage 3에서 새로 저장한 `early_exit_fixed_final.pth`와 `early_exit_dynamic_final.pth`는 새로 학습한 별도 가중치가 아니다.

두 파일은 기존 학습 checkpoint인 `early_exit_lstm_best.pth`와 동일한 모델 가중치를 사용한다. Fixed θ와 Dynamic θ의 차이는 학습 weight가 아니라 추론 시 threshold를 적용하는 방식에서 발생한다.

| 파일 | 실제 의미 |
|---|---|
| `early_exit_lstm_best.pth` | 실제 학습으로 얻은 Early Exit LSTM best checkpoint |
| `early_exit_fixed_final.pth` | 동일 가중치를 고정 θ 모드로 구분하기 위한 alias checkpoint |
| `early_exit_dynamic_final.pth` | 동일 가중치를 동적 θ 모드로 구분하기 위한 alias checkpoint |

따라서 김호중 컴퓨터 환경에서 재실험할 때는 `early_exit_lstm_best.pth` 하나만 사용해도 된다. 이 경우 Fixed θ와 Dynamic θ는 같은 checkpoint를 로드한 뒤 추론 함수에서 `dynamic=False` 또는 `dynamic=True`로 구분하면 된다.

---

## 실행 방법

프로젝트 루트에서 아래 순서로 실행한다.

```bash
python project/scripts/generate_test_with_scenario.py
python project/scripts/split_scenario_data.py
python project/scripts/fill_scenario_analysis.py
python project/scripts/evaluate_early_exit.py
```

Standard LSTM Full Inference와 비교할 때는 아래 명령을 함께 실행한다.

```bash
python project/scripts/evaluate.py
```

---

## 최종 threshold 파라미터

| 파라미터 | 값 |
|---|---:|
| `fixed_theta_1` | 0.3 |
| `fixed_theta_2` | 0.6 |
| `dynamic_base_theta_1` | 0.3 |
| `dynamic_base_theta_2` | 0.6 |
| `dynamic_high_variance` | 0.22 |
| `dynamic_mid_variance` | 0.12 |
| `dynamic_min_threshold` | 0.22 |
| `dynamic_recent_steps` | 5 |
| `dynamic_spike_threshold` | 0.25 |

동적 θ는 최근 `channel_occupancy` 변화량을 기준으로 spike 구간과 안정 구간을 구분한다. 안정 구간에서는 threshold를 높여 더 적극적으로 조기 종료하고, spike 구간에서는 기본 threshold를 유지해 급변 상황에서 과도한 조기 종료를 줄인다.

---

## 전체 성능 결과

| 모델 | 전체 정확도 | 평균 추론 시간 | Exit 1 | Exit 2 | Exit 3 |
|---|---:|---:|---:|---:|---:|
| Standard LSTM Full | 94.9% | 8.000ms | 0.0% | 0.0% | 100.0% |
| Early Exit Fixed θ | 95.7% | 3.897ms | 20.5% | 71.8% | 7.7% |
| Early Exit Dynamic θ | 96.3% | 3.681ms | 25.6% | 69.5% | 4.8% |

동적 θ는 고정 θ 대비 전체 정확도가 `+0.6%p` 향상되었고, 평균 추론 시간은 `0.217ms` 감소했다.

Standard LSTM Full Inference와 비교하면 동적 θ는 정확도 `+1.4%p`, 평균 추론 시간 `-4.319ms`를 기록했다.

---

## 실측 Wall Time 비교

CPU 환경에서 `batch_size=1`로 샘플 단위 실측 추론 시간을 측정했다.

| 모델 | 전체 정확도 | 실측 추론 시간 |
|---|---:|---:|
| Standard LSTM Full | 94.9% | 0.9539ms |
| Early Exit Fixed θ | 95.7% | 0.8226ms |
| Early Exit Dynamic θ | 96.3% | 0.7904ms |

실측 기준에서도 동적 θ가 가장 짧은 추론 시간을 기록했다. Standard LSTM Full 대비 약 `17.1%` 단축되었다.

---

## 시나리오별 정확도

| 시나리오 | Standard LSTM Full | Fixed θ | Dynamic θ |
|---|---:|---:|---:|
| `startup_surge` | 96.2% | 96.2% | 96.2% |
| `emergency_ramp` | 91.6% | 92.8% | 95.2% |
| `lunch_restart` | 98.8% | 96.4% | 96.4% |
| `imbalanced_ap_load` | 92.5% | 97.5% | 97.5% |
| 전체 | 94.9% | 95.7% | 96.3% |

동적 θ는 `emergency_ramp`에서 가장 뚜렷한 개선을 보였다. 갑작스러운 폭증 시나리오에서 Standard LSTM Full 대비 `+3.6%p`, 고정 θ 대비 `+2.4%p` 높은 정확도를 기록했다.

---

## 시나리오별 실측 Wall Time

| 시나리오 | Standard LSTM Full | Fixed θ | Dynamic θ |
|---|---:|---:|---:|
| `startup_surge` | 0.9551ms | 0.8308ms | 0.7968ms |
| `emergency_ramp` | 0.9458ms | 0.8308ms | 0.7794ms |
| `lunch_restart` | 0.9324ms | 0.8180ms | 0.7832ms |
| `imbalanced_ap_load` | 0.9833ms | 0.8083ms | 0.8012ms |
| 전체 | 0.9539ms | 0.8226ms | 0.7904ms |

모든 시나리오에서 동적 θ가 Standard LSTM Full보다 짧은 실측 시간을 보였다. 특히 `emergency_ramp`에서는 정확도와 실측 시간 모두 개선되었다.

---

## 결과 해석

Stage 3 결과에서 동적 θ는 고정 θ보다 Exit 1 종료율을 높이고 Exit 3 종료율을 낮췄다.

```text
Fixed θ   Exit 분포: 20.5% / 71.8% / 7.7%
Dynamic θ Exit 분포: 25.6% / 69.5% / 4.8%
```

즉 동적 θ는 안정적이라고 판단되는 구간에서 더 빠른 Exit를 허용해 계산량을 줄였다. 동시에 급변 구간에서는 threshold를 보수적으로 유지해 `emergency_ramp` 정확도를 높였다.

다만 기존 목표였던 `Exit 1 종료율 60% 이상`은 현재 실제 stepwise 추론 기준으로 달성하지 못했다. 현재 동적 θ의 Exit 1 종료율은 `25.6%`이다. 따라서 이후 단계에서 더 강한 조기 종료를 목표로 한다면 θ 값을 높이는 추가 튜닝이 필요하다. 하지만 현재 결과 기준으로는 정확도와 추론 시간의 균형이 가장 좋은 모델은 동적 θ이다.

---

## 전달 항목

김호중에게 전달할 최종 모델 관련 파일:

```text
project/checkpoints/early_exit_fixed_final.pth
project/checkpoints/early_exit_dynamic_final.pth
project/checkpoints/model_info.json
```

단, `early_exit_fixed_final.pth`와 `early_exit_dynamic_final.pth`는 `early_exit_lstm_best.pth`의 alias checkpoint이므로, 호중 환경에서 반드시 이 두 파일을 사용할 필요는 없다. 실제 재현에는 `early_exit_lstm_best.pth`와 `model_info.json`만 있어도 충분하다.

장예나에게 전달할 시나리오별 분석 파일:

```text
project/results/yongsang/scenario_analysis_filled.csv
project/results/yongsang/scenario_analysis_summary.csv
project/results/scenario_analysis/scenario_0_analysis.csv
project/results/scenario_analysis/scenario_1_analysis.csv
project/results/scenario_analysis/scenario_2_analysis.csv
project/results/scenario_analysis/scenario_3_analysis.csv
```

---

## 체크리스트 반영

`docs/yongsang/guideline_yongsang_stage3.md`에서 실제 완료된 항목은 체크 완료했다.

완료된 항목:

- 2단계 결과 분석 및 튜닝 방향 결정
- 동적 θ 파라미터 튜닝
- 시나리오 0~3 분리 분석
- 고정 θ vs 동적 θ 비교 분석
- `early_exit_fixed_final.pth` 저장
- `early_exit_dynamic_final.pth` 저장
- `model_info.json` 저장

미완료 또는 전달 대기 항목:

- 하이퍼파라미터 튜닝 완료 조건인 Exit 1 종료율 60% 이상
- 김호중에게 최종 모델 및 사용법 전달
- 장예나에게 시나리오별 분석 결과 전달
