# 유용상 Stage 4 작업 설명

## 작업 목표

Stage 4의 목적은 Stage 3에서 검증한 Early Exit LSTM의 고정 threshold 방식과 동적 threshold 방식을 최종 분석하고, 보고서의 모델 설계 섹션에 들어갈 초안을 정리하는 것이다.

이번 단계에서는 단순히 동적 threshold가 더 좋다고 결론 내리지 않고, 다음 기준을 분리하여 분석했다.

- 고정 θ와 동적 θ의 정확도 및 Exit 분포 차이
- 실측 추론 시간 기준의 효율성
- 정상 구간 혼잡 오판율과 실제 불필요 채널 전환율
- 호중 Stage 4의 INT8/ONNX 배포 결과가 모델 설계 분석에 주는 의미
- 동적 θ의 한계와 향후 개선 방향

---

## 사용 결과 파일

| 파일 | 역할 |
|---|---|
| `project/results/hojung/comparison_summary.csv` | 4개 방식 전체 비교 결과 |
| `project/results/yongsang/scenario_analysis_summary.csv` | 고정 θ vs 동적 θ 시나리오별 분석 |
| `project/results/quantization_comparison.csv` | 호중 Stage 4 경량화 및 ONNX 결과 |
| `docs/yongsang/stage3_work_log.md` | Stage 3 분석 근거 |
| `docs/hochung/stage4_work_log.md` | Stage 4 배포 최적화 근거 |

---

## 1. 고정 θ vs 동적 θ 최종 분석표

### 전체 성능 비교

| 항목 | 고정 θ | 동적 θ | 해석 |
|---|---:|---:|---|
| 전체 정확도 | 95.7% | 96.3% | 동적 θ가 +0.6%p 높음 |
| 실측 평균 추론 시간 | 0.4348ms | 0.4360ms | 최신 비교 기준에서는 거의 동일, 동적 θ가 0.0012ms 느림 |
| Exit 1 종료율 | 20.5% | 25.6% | 동적 θ가 더 얕은 추론을 많이 사용 |
| Exit 2 종료율 | 71.8% | 69.5% | 동적 θ에서 Exit 2 비중 감소 |
| Exit 3 종료율 | 7.7% | 4.8% | 동적 θ가 깊은 추론 진입을 줄임 |
| 정상 구간 혼잡 오판율 | 1.8% | 3.6% | 고정 θ가 정상 구간에서는 더 보수적 |
| 실제 불필요 채널 전환율 | 0.0% | 0.0% | 두 방식 모두 channel optimizer 통과 후 실제 불필요 전환 없음 |

동적 θ는 전체 정확도와 Exit 3 감소 측면에서 장점이 있다. 다만 최신 실측 시간 기준으로는 고정 θ와 동적 θ의 차이가 거의 없으며, 동적 θ가 명확하게 더 빠르다고 단정하기는 어렵다. 따라서 Stage 4 결론은 "동적 θ는 정확도와 깊은 추론 감소 측면에서 유효하지만, 실측 지연 우위는 현재 구현에서 제한적"으로 정리한다.

### 시나리오별 정확도

| 시나리오 | 고정 θ | 동적 θ | 해석 |
|---|---:|---:|---|
| `startup_surge` | 96.15% | 96.15% | 점진적 증가 구간에서는 동일 |
| `emergency_ramp` | 92.77% | 95.18% | 급격한 폭증 구간에서 동적 θ가 +2.41%p 개선 |
| `lunch_restart` | 96.43% | 96.43% | 주기적 재가동 패턴에서는 동일 |
| `imbalanced_ap_load` | 97.50% | 97.50% | 지속 혼잡 구간에서는 동일 |
| 전체 | 95.73% | 96.30% | 전체 기준 동적 θ가 소폭 우세 |

동적 θ의 가장 뚜렷한 장점은 `emergency_ramp` 시나리오에서 나타난다. 이 시나리오는 트래픽이 갑작스럽게 증가하는 환경이므로, 최근 채널 점유율 변화량을 반영하는 동적 threshold의 목적과 가장 잘 맞는다.

### 시나리오별 Exit 분포

| 시나리오 | 고정 θ Exit 1 | 동적 θ Exit 1 | 고정 θ Exit 3 | 동적 θ Exit 3 |
|---|---:|---:|---:|---:|
| `startup_surge` | 18.27% | 22.12% | 7.69% | 4.81% |
| `emergency_ramp` | 22.89% | 26.51% | 7.23% | 3.61% |
| `lunch_restart` | 20.24% | 28.57% | 7.14% | 4.76% |
| `imbalanced_ap_load` | 21.25% | 26.25% | 8.75% | 6.25% |
| 전체 | 20.51% | 25.64% | 7.69% | 4.84% |

모든 시나리오에서 동적 θ는 고정 θ보다 Exit 1 비율이 높고 Exit 3 비율이 낮다. 이는 동적 θ가 더 많은 샘플을 얕은 레이어에서 처리하도록 유도했다는 뜻이다.

---

## 2. 동적 θ 한계 원인 분석

### 원인 1. 동적 threshold 계산 오버헤드

현재 동적 θ는 최근 `channel_occupancy` 변화량을 보고 threshold를 조정한다.

```python
delta = abs(current_occupancy - previous_occupancy)
if delta > spike_threshold:
    theta_1 = base_theta_1 * 1.0
    theta_2 = base_theta_2 * 1.0
else:
    theta_1 = base_theta_1 * 1.25
    theta_2 = base_theta_2 * 1.25
```

이 방식은 기존 variance/std 기반 방식보다 가볍지만, 고정 θ와 비교하면 여전히 조건 분기와 threshold 재계산이 추가된다. 최신 실측 결과에서 동적 θ의 평균 추론 시간이 고정 θ보다 0.0012ms 느린 이유는 이 추가 연산과 Python 레벨 분기 비용 때문으로 해석할 수 있다.

### 원인 2. 현재 데이터에서는 고정 θ도 충분히 강함

시나리오별 정확도를 보면 4개 시나리오 중 3개에서 고정 θ와 동적 θ의 정확도가 동일하다.

```text
startup_surge: 동일
lunch_restart: 동일
imbalanced_ap_load: 동일
emergency_ramp: 동적 θ 우세
```

즉 현재 데이터에서는 고정 θ만으로도 대부분의 패턴을 안정적으로 분류한다. 동적 θ의 장점은 급변 구간에서 드러나지만, 전체 평균에서는 그 차이가 제한적이다.

### 원인 3. 배포 최적화는 현재 고정 θ 중심으로 검증됨

호중 Stage 4에서는 `early_exit_fixed.onnx`를 기준으로 ONNX Runtime 배포 검증이 수행되었다.

| 모델 | 원본 추론 | PyTorch INT8 추론 | ONNX Runtime 추론 |
|---|---:|---:|---:|
| Early Exit Fixed | 0.5014ms | 1.1886ms | 0.2000ms |

INT8 Quantization은 모델 용량을 줄이는 데 성공했지만, PyTorch CPU 기준 추론은 오히려 느려졌다. 최종 속도 개선은 ONNX Runtime 배포 엔진을 통해 달성되었다. 현재 ONNX 검증은 고정 θ 모델 중심으로 수행되었으므로, 동적 θ 배포 효과는 별도 검증이 필요하다.

---

## 3. 향후 개선 방향

### 개선 1. 동적 θ 계산을 더 가볍게 만들기

현재 delta 기반 계산은 std 기반보다 가볍지만, 고정 θ와 비교하면 여전히 동적 계산 비용이 있다. 향후에는 다음 방식으로 더 줄일 수 있다.

```python
if timestep % 3 == 0:
    update_threshold()
else:
    reuse_previous_threshold()
```

즉 모든 샘플마다 threshold를 갱신하지 않고, 일정 주기마다 한 번만 갱신하는 방식이다.

### 개선 2. ONNX Runtime 기준 동적 θ 검증

현재 ONNX Runtime 검증은 고정 θ 중심으로 이루어졌다. 동적 θ를 배포 기준으로 평가하려면 다음 중 하나가 필요하다.

- ONNX 모델 출력 `exit1`, `exit2`, `exit3`를 받은 뒤 Python 또는 C++ 런타임에서 동적 threshold 선택
- threshold 계산을 ONNX 그래프 내부에 포함하는 별도 모델 export
- Raspberry Pi 또는 ONNX Runtime 환경에서 고정 θ와 동적 θ를 동일 조건으로 비교

### 개선 3. 정상 구간 혼잡 오판율과 실제 전환율 분리 유지

Stage 4에서 `False_Congestion_Rate`와 `Unnecessary_Switch_Rate`를 분리했다.

| 방식 | 정상 구간 혼잡 오판율 | 실제 불필요 전환율 |
|---|---:|---:|
| Threshold | 14.3% | 1.8% |
| LSTM Full | 8.9% | 0.0% |
| Early Exit Fixed | 1.8% | 0.0% |
| Early Exit Dynamic | 3.6% | 0.0% |

이 구분은 보고서에서 중요하다. 모델이 정상 구간을 혼잡으로 오판하더라도, `channel_optimizer.py`의 hysteresis와 switch cost를 통과해야 실제 채널 전환이 발생한다. 따라서 모델 오판 위험과 실제 제어 결과를 분리해서 설명해야 한다.

---

## 4. 보고서 모델 설계 섹션 초안

### 5.1 Early Exit LSTM 구조

본 연구에서는 산업 무선망 트래픽 혼잡 상태를 실시간으로 분류하기 위해 3-layer LSTM 기반 Early Exit 구조를 설계하였다. 입력 데이터는 최근 10개 timestep의 `rps`, `channel_occupancy`, `packet_loss`, `latency` 4개 feature로 구성되며, 출력 label은 정상, 혼잡 경고, 혼잡, 심각 혼잡의 4개 상태이다.

일반 LSTM은 모든 입력을 3개 LSTM layer까지 통과시킨 뒤 최종 classifier에서만 예측한다. 반면 Early Exit LSTM은 각 LSTM layer 뒤에 별도의 classifier를 배치한다. 이를 통해 쉬운 샘플은 얕은 layer에서 빠르게 종료하고, 판단이 어려운 샘플만 깊은 layer까지 전달한다.

학습 시에는 세 exit의 cross entropy loss를 함께 사용한다.

```text
Loss = 0.3 * Loss_exit1 + 0.3 * Loss_exit2 + 0.4 * Loss_exit3
```

이 구조는 전체 정확도를 유지하면서 평균 추론 깊이를 줄이는 것을 목표로 한다.

### 5.2 고정 Threshold 설계

고정 θ 방식은 각 exit classifier의 예측 entropy를 기준으로 조기 종료 여부를 판단한다. entropy가 낮다는 것은 모델의 예측 확신도가 높다는 의미이므로, 해당 exit에서 추론을 종료한다.

```text
Exit 1 entropy < θ1 이면 Exit 1에서 종료
아니면 Exit 2 entropy < θ2 이면 Exit 2에서 종료
그 외에는 Exit 3까지 추론
```

최종 고정 threshold는 다음과 같다.

| 파라미터 | 값 |
|---|---:|
| θ1 | 0.3 |
| θ2 | 0.6 |

실험 결과 고정 θ는 전체 정확도 95.7%, Exit 1 종료율 20.5%, Exit 3 종료율 7.7%를 기록했다. Full LSTM과 비교하면 정확도는 더 높고, 실제 추론 시간도 더 짧게 나타났다.

### 5.3 동적 Threshold 설계 및 한계

동적 θ 방식은 모든 구간에서 동일한 threshold를 사용하는 대신, 최근 채널 점유율 변화량에 따라 threshold를 조정한다. 안정 구간에서는 threshold를 높여 더 적극적으로 조기 종료하고, 급변 구간에서는 기본 threshold를 유지하여 과도한 조기 종료를 방지한다.

최종 동적 threshold 규칙은 다음과 같다.

| 파라미터 | 값 |
|---|---:|
| base θ1 | 0.3 |
| base θ2 | 0.6 |
| spike threshold | 0.25 |
| stable scale | 1.25 |
| unstable scale | 1.0 |
| min threshold | 0.22 |

실험 결과 동적 θ는 전체 정확도 96.3%로 고정 θ보다 0.6%p 높았고, Exit 3 종료율을 7.7%에서 4.8%로 낮췄다. 특히 `emergency_ramp` 시나리오에서는 고정 θ 대비 +2.41%p 높은 정확도를 보였다.

다만 최신 실측 시간 기준으로는 고정 θ가 0.4348ms, 동적 θ가 0.4360ms로 거의 동일하며, 동적 θ가 명확한 지연 우위를 보이지는 않았다. 따라서 현재 구현의 동적 θ는 정확도 및 깊은 추론 감소에는 효과가 있지만, 지연 시간 측면에서는 추가 경량화가 필요하다.

---

## 5. Stage 4 결론

동적 θ는 고정 θ보다 전체 정확도와 급변 시나리오 대응력에서 장점을 보였고, Exit 3 도달률을 낮춰 깊은 추론 비율을 줄였다. 그러나 실측 추론 시간에서는 고정 θ와 거의 차이가 없었으며, 배포 최적화는 현재 고정 θ ONNX 모델을 중심으로 검증되었다.

따라서 최종 결론은 다음과 같다.

```text
동적 threshold는 급변 트래픽 환경에서 정확도와 조기 종료 분포를 개선하는 효과가 있다.
하지만 현재 구현에서는 threshold 계산과 런타임 분기 비용 때문에 실측 지연 우위가 제한적이다.
향후에는 동적 threshold 계산을 주기적으로 수행하거나 ONNX Runtime 배포 구조에 맞게 재설계하여,
정확도 개선과 실제 지연 감소를 동시에 달성하는 방향으로 개선할 수 있다.
```

---

## 6. 체크리스트 결과

- [x] 고정 θ vs 동적 θ 최종 분석표 완성
- [x] 동적 θ 한계 원인 3가지 분석 완료
- [x] 향후 개선 방향 정리 완료
- [x] 보고서 모델 설계 섹션 초안 완성
  - [x] 5.1 Early Exit LSTM 구조
  - [x] 5.2 고정 Threshold 설계
  - [x] 5.3 동적 Threshold 설계 및 한계
- [x] 장예나에게 분석 결과 전달 완료 (시각화용)
