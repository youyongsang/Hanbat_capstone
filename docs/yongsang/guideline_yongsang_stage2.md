# 유용상 2단계 가이드라인
## Early Exit LSTM 통합 학습 및 동적 Threshold 구현

> 담당자: 유용상  
> 목표: Early Exit LSTM 통합 학습 완료 및 동적 threshold 구현  
> 완료 기준: 고정 θ 버전과 동적 θ 버전 모두 학습 완료, Exit별 종료율 및 정확도 출력

---

## 1. 해야 할 일 순서

```
1. 장예나 DataLoader 연결 확인
2. Early Exit LSTM 통합 학습 (실제 데이터로)
3. 고정 threshold 버전 평가 (Baseline ③)
4. 동적 threshold 설계 및 구현
5. 동적 threshold 버전 평가 (제안 모델 ④)
6. 김호중 비교 실험 프레임워크에 연결
```

---

## 2. 장예나 DataLoader 연결

1단계 더미 데이터에서 실제 데이터로 교체.

```python
from utils.dataloader import get_dataloader

train_loader = get_dataloader('data/real/train.csv', batch_size=32)
val_loader   = get_dataloader('data/real/val.csv',   batch_size=32, shuffle=False)
test_loader  = get_dataloader('data/real/test.csv',  batch_size=32, shuffle=False)
```

---

## 3. 통합 학습

1단계에서 구현한 Early Exit LSTM을 실제 데이터로 학습.

### 학습 시 주의사항

1단계에서 더미 데이터로 동작 확인만 했다면, 실제 데이터로 학습할 때 다음을 점검한다.

| 점검 항목 | 확인 방법 |
|---|---|
| 학습 loss 감소 여부 | epoch별 loss 그래프 확인 |
| 각 Exit 정확도 | Exit 1 < Exit 2 < Exit 3 순서가 맞는지 |
| Exit별 종료율 | Exit 1이 60~70% 목표 |
| 과적합 여부 | train acc vs val acc 차이 확인 |

### 하이퍼파라미터 튜닝 기준

| 상황 | 대응 |
|---|---|
| Exit 1 종료율이 너무 낮음 (30% 이하) | θ₁ 높이기 |
| Exit 3에 너무 몰림 (50% 이상) | θ₁, θ₂ 전체적으로 높이기 |
| 학습이 불안정함 | Multi-exit loss 가중치 조정 |
| Val acc가 Train acc보다 훨씬 낮음 | dropout 높이기, epochs 줄이기 |

---

## 4. 고정 Threshold 버전 평가 (Baseline ③)

학습 완료 후 고정 θ로 테스트셋 평가.  
이게 김호중 비교 실험의 **Baseline ③** 역할.

### 평가 출력 형식

```
=== Baseline ③: Early Exit + 고정 θ ===
Test Accuracy: 87.8%

Exit별 성능:
  Exit 1 | Accuracy: 82.3% | Exit Rate: 64.2% | Avg Time: 2.1ms
  Exit 2 | Accuracy: 86.7% | Exit Rate: 22.1% | Avg Time: 4.2ms
  Exit 3 | Accuracy: 91.2% | Exit Rate: 13.7% | Avg Time: 8.3ms

Overall Avg Inference Time: 3.9ms
Unnecessary Channel Switch Rate: 8.2%
```

---

## 5. 동적 Threshold 설계

### 핵심 원리

추론 중 매 타임스텝마다 최근 트래픽 변동률을 계산하여 θ를 실시간으로 조정.

```python
def compute_dynamic_threshold(recent_window, base_theta_1=0.3, base_theta_2=0.6):
    """
    Args:
        recent_window: 최근 N 타임스텝의 채널 점유율 값
        base_theta_1: 기본 θ₁ 값
        base_theta_2: 기본 θ₂ 값
    Returns:
        theta_1, theta_2: 조정된 threshold 값
    """
    variance = np.std(recent_window)

    if variance > HIGH_VARIANCE:      # 변동 심함
        theta_1 = base_theta_1 * 0.6  # 낮게 → 더 깊이 추론
        theta_2 = base_theta_2 * 0.6
    elif variance > MID_VARIANCE:     # 변동 중간
        theta_1 = base_theta_1 * 0.8
        theta_2 = base_theta_2 * 0.8
    else:                              # 안정적
        theta_1 = base_theta_1 * 1.2  # 높게 → 빠른 종료
        theta_2 = base_theta_2 * 1.2

    # 최솟값 보장 (급변 상황 대비)
    theta_1 = max(theta_1, MIN_THRESHOLD)
    theta_2 = max(theta_2, MIN_THRESHOLD * 2)

    return theta_1, theta_2
```

### 파라미터 설정 가이드

| 파라미터 | 초기값 | 설명 |
|---|---|---|
| HIGH_VARIANCE | 15.0 | 채널 점유율 표준편차 기준 (실험으로 조정) |
| MID_VARIANCE | 7.0 | 중간 변동성 기준 |
| MIN_THRESHOLD | 0.1 | θ 최솟값 (급변 대응 보장) |
| recent_window | 최근 5 타임스텝 | 너무 길면 급변 감지 늦어짐 |

> 초기값으로 시작하고, 실험 결과 보며 조정할 것.

### 급변 대응 보완

갑작스러운 트래픽 폭증 시 1 타임스텝 지연 문제를 최솟값 보장으로 완화.

```python
# 직전 타임스텝 대비 급격한 변화 감지
spike = abs(current_occupancy - prev_occupancy) > SPIKE_THRESHOLD
if spike:
    theta_1 = MIN_THRESHOLD  # 즉시 낮추기
    theta_2 = MIN_THRESHOLD * 2
```

---

## 6. 동적 Threshold 버전 평가 (제안 모델 ④)

### 평가 출력 형식

```
=== 제안 모델 ④: Early Exit + 동적 θ ===
Test Accuracy: 89.2%

Exit별 성능:
  Exit 1 | Accuracy: 83.1% | Exit Rate: 68.4% | Avg Time: 2.0ms
  Exit 2 | Accuracy: 87.3% | Exit Rate: 20.3% | Avg Time: 4.1ms
  Exit 3 | Accuracy: 92.1% | Exit Rate: 11.3% | Avg Time: 8.1ms

Overall Avg Inference Time: 3.6ms
Unnecessary Channel Switch Rate: 5.8%

동적 θ 효과:
  안정 구간 Exit 1 종료율: 78.3% (고정 θ: 64.2%)
  급변 구간 정확도: 88.1% (고정 θ: 83.4%)
```

### 핵심 비교 지표

고정 θ(③) vs 동적 θ(④)에서 이 두 가지가 차이나면 성공이야.

| 지표 | 고정 θ 예상 | 동적 θ 목표 |
|---|---|---|
| 안정 구간 Exit 1 종료율 | ~64% | ~70% 이상 |
| 급변 구간 분류 정확도 | ~83% | ~88% 이상 |

---

## 7. 시나리오별 분석

전체 정확도뿐만 아니라 **시나리오별로 분리해서 분석**하면 동적 θ의 효과가 더 명확하게 보여.

| 시나리오 | 분석 목적 |
|---|---|
| 일과 시작 (점진적 급증) | 동적 θ가 증가 패턴에 얼마나 빨리 반응하는지 |
| 긴급 증산 (갑작스러운 폭증) | 급변 시 1 타임스텝 지연 정도 측정 |
| 점심 재가동 (주기적 패턴) | 안정 → 증가 전환 구간에서의 차이 |
| 불균형 부하 (지속적 혼잡) | 혼잡 지속 구간에서의 정확도 차이 |

---

## 8. 김호중 비교 실험 연결

완성된 모델 2개를 김호중 비교 실험 프레임워크에 연결.

### 제공해야 할 것

```python
# 김호중이 이렇게 불러쓸 수 있어야 함

from models.early_exit_lstm import EarlyExitLSTM

# Baseline ③ — 고정 θ
model_fixed = EarlyExitLSTM(...)
model_fixed.load_state_dict(torch.load('checkpoints/early_exit_fixed.pth'))
model_fixed.set_threshold(theta_1=0.3, theta_2=0.6, dynamic=False)

# 제안 모델 ④ — 동적 θ
model_dynamic = EarlyExitLSTM(...)
model_dynamic.load_state_dict(torch.load('checkpoints/early_exit_dynamic.pth'))
model_dynamic.set_threshold(dynamic=True)
```

---

## 9. 완료 기준 체크리스트

- [x] 장예나 DataLoader 연결 및 실제 데이터 학습 완료
- [x] Early Exit LSTM 학습 loss 수렴 확인
- [x] Exit별 정확도 Exit 1 < Exit 2 < Exit 3 순서 확인
- [x] 고정 θ 버전 테스트셋 평가 완료 (Baseline ③)
- [x] `compute_dynamic_threshold()` 함수 구현 완료
- [x] 급변 대응 spike 감지 로직 구현 완료
- [x] 동적 θ 버전 테스트셋 평가 완료 (제안 모델 ④)
- [x] 고정 θ vs 동적 θ 비교 결과 정리 완료
- [x] 시나리오별 분석 완료
- [x] 김호중 비교 실험 프레임워크에 연결 완료
- [x] `checkpoints/early_exit_fixed.pth` 저장 확인
- [x] `checkpoints/early_exit_dynamic.pth` 저장 확인

---

## 10. 주의사항

- 동적 threshold는 **추론 중에만 동작**함. 학습은 항상 풀 추론(모든 Exit 통과)으로 진행.
- 동적 θ가 고정 θ보다 나쁘게 나와도 괜찮아. **"어떤 조건에서 효과적인지 분석"** 자체가 기여임.
- 시나리오별 분리 분석을 꼭 할 것. 전체 정확도가 비슷해도 특정 구간에서 차이가 날 수 있음.
- HIGH_VARIANCE, MID_VARIANCE 값은 실험하면서 조정. 처음엔 초기값으로 시작하고 Exit 종료율 보며 튜닝.
