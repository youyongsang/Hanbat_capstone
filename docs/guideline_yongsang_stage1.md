# 유용상 1단계 가이드라인
## Early Exit LSTM 설계 및 구현

> 담당자: 유용상  
> 목표: Early Exit LSTM 구조 설계 및 PyTorch 구현, 더미 데이터로 동작 확인  
> 완료 기준: 더미 데이터로 학습이 돌아가고 Exit별 accuracy 및 종료율이 출력되면 됨

---

## 1. 해야 할 일 순서

```
1. 관련 논문 정독 (BranchyNet, ACM Survey)
2. Early Exit LSTM 구조 설계 (손으로 그려보기)
3. early_exit_lstm.py 구현
4. Multi-exit loss 구현
5. 동적 threshold 설계
6. 김호중과 인터페이스 맞추기
7. 더미 데이터로 동작 확인
```

---

## 2. 먼저 읽어야 할 논문

### 필수

| 논문 | 읽어야 할 이유 |
|---|---|
| BranchyNet (Teerapittayanon et al., ICPR 2016) | Early Exit 원조. 구조 설계 방법, Exit point 배치, Entropy 계산 방법이 나옴. |
| Early-Exit DNN: A Comprehensive Survey (ACM, 2024) | 전체 흐름 파악. threshold 동적 최적화가 미해결 과제임을 명시 → 본 연구 핵심 근거. |

### 참고

| 논문 | 내용 |
|---|---|
| Mohammed et al. (2023) arXiv:2308.11100 | 무선 도메인에 Early Exit 적용 사례. 구현 참고용. |
| Verbruggen et al. (2024) arXiv:2405.03222 | Width-wise Early Exit. 구조 아이디어 참고. |

---

## 3. Early Exit LSTM 개념

### 일반 LSTM vs Early Exit LSTM

```
일반 LSTM:
입력 → Layer1 → Layer2 → Layer3 → FC → 출력
       (항상 3레이어 모두 통과)

Early Exit LSTM:
입력 → Layer1 → Entropy Check → (낮으면) Exit1 → 출력
               ↓ (높으면)
             Layer2 → Entropy Check → (낮으면) Exit2 → 출력
                                    ↓ (높으면)
                                  Layer3 → Exit3 → 출력
```

### Entropy란?

모델이 현재 입력에 대해 **얼마나 불확실한지** 나타내는 수치.

```python
import torch.nn.functional as F

def entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
```

- Entropy 낮음 → 확신 높음 → 조기 종료 가능
- Entropy 높음 → 불확실 → 다음 레이어로 진행

### Threshold θ

```python
if entropy < theta_1:
    return exit1_output   # 조기 종료
elif entropy < theta_2:
    return exit2_output   # 중간 종료
else:
    return exit3_output   # 풀 추론
```

---

## 4. 모델 구조 스펙

### 전체 구조

```
입력: (batch, 10, 4)
        ↓
LSTM Layer 1 (hidden_size=128)
        ↓
Exit Classifier 1: FC(128 → 4) + Entropy Check (H < θ₁)
        ↓ (H ≥ θ₁)
LSTM Layer 2 (hidden_size=128)
        ↓
Exit Classifier 2: FC(128 → 4) + Entropy Check (H < θ₂)
        ↓ (H ≥ θ₂)
LSTM Layer 3 (hidden_size=128)
        ↓
Exit Classifier 3: FC(128 → 4)
        ↓
출력: (batch, 4)
```

### 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| hidden_size | 128 | **김호중 베이스라인과 동일하게 맞출 것** |
| num_layers | 3 | |
| dropout | 0.2 | |
| num_classes | 4 | |

> **hidden_size=128은 김호중과 반드시 맞춰야 함.**  
> 공정한 비교를 위해 파라미터 수를 비슷하게 유지해야 하기 때문.

---

## 5. Multi-exit Loss 설계

각 Exit point마다 loss를 계산하고 가중 합산하여 전체 loss를 구한다.

```python
loss = w1 * loss_exit1 + w2 * loss_exit2 + w3 * loss_exit3
```

### 가중치 설정

| Exit | 가중치 | 설정 이유 |
|---|---|---|
| Exit 1 | 0.3 | 얕은 레이어. 정확도 낮아도 됨. |
| Exit 2 | 0.3 | 중간 레이어. |
| Exit 3 | 0.4 | 가장 깊은 레이어. 정확도 최우선. |

> 가중치는 실험하면서 조정 가능. 처음엔 균등(0.33/0.33/0.34)으로 시작해도 됨.

---

## 6. 동적 Threshold 설계

기존 Early Exit은 threshold를 고정값으로 사용한다.  
본 연구는 **최근 트래픽 부하 변동률과 채널 점유율**을 입력으로 θ를 실시간 조정한다.

### 동작 원리

```
트래픽 변동이 잦은 시간대 → θ 낮게 설정 → 더 깊이 추론
안정적인 시간대          → θ 높게 설정 → 빠른 추론 우선
```

### 구현 방향

1단계에서는 우선 **고정 threshold**로 구현하고 동작 확인.  
2단계에서 동적 조정 로직 추가.

```python
# 1단계: 고정값으로 시작
theta_1 = 0.3
theta_2 = 0.6

# 2단계: 동적 조정 (나중에 구현)
def dynamic_threshold(recent_variance, channel_occupancy):
    ...
    return theta_1, theta_2
```

---

## 7. Inference 모드 구현

학습 시에는 모든 Exit을 통과하여 Multi-exit loss를 계산.  
추론 시에는 Entropy 기준으로 조기 종료.

```python
def forward(self, x, inference=False):
    # inference=False: 학습 모드 (모든 Exit 출력)
    # inference=True:  추론 모드 (조기 종료 적용)
    
    out1 = self.exit_classifier1(lstm1_output)
    if inference and entropy(out1) < self.theta_1:
        return out1, exit_point=1
    
    out2 = self.exit_classifier2(lstm2_output)
    if inference and entropy(out2) < self.theta_2:
        return out2, exit_point=2
    
    out3 = self.exit_classifier3(lstm3_output)
    return out3, exit_point=3
```

---

## 8. 파일 구조

```
project/
├── models/
│   └── early_exit_lstm.py    ← 모델 클래스 정의 (여기만 담당)
├── utils/
│   ├── dataloader.py         ← 김호중 것 그대로 사용
│   └── metrics.py            ← 김호중 것 + Exit별 종료율 추가
└── checkpoints/
    └── early_exit_lstm_best.pth
```

---

## 9. 김호중과 맞춰야 할 것

| 항목 | 값 | 확인 방법 |
|---|---|---|
| `hidden_size` | 128 | 김호중한테 직접 확인 |
| 입력 shape | `(batch, 10, 4)` | 동일해야 같은 데이터 사용 가능 |
| 출력 클래스 수 | 4 | 동일해야 같은 레이블 기준 |
| `dataloader.py` | 김호중 것 재사용 | 별도 구현 불필요 |

---

## 10. 출력 예시

### 학습 시

```
Epoch 1/50
  Train Loss: 1.234 | Train Acc: 45.2% | Val Acc: 43.1%
  Exit 1 Rate: 58.3% | Exit 2 Rate: 27.1% | Exit 3 Rate: 14.6%

Epoch 10/50
  Train Loss: 0.842 | Train Acc: 71.4% | Val Acc: 69.8%
  Exit 1 Rate: 64.1% | Exit 2 Rate: 23.5% | Exit 3 Rate: 12.4%
```

### 평가 시

```
Test Accuracy: 88.1%
Exit 1 Accuracy: 82.3% | Exit Rate: 65.2%
Exit 2 Accuracy: 87.1% | Exit Rate: 22.4%
Exit 3 Accuracy: 91.2% | Exit Rate: 12.4%
Average Inference Time: 3.8ms
```

---

## 11. 완료 기준 체크리스트

- [ ] BranchyNet 논문 정독 완료
- [ ] ACM Survey 논문 정독 완료
- [ ] Early Exit LSTM 구조도 손으로 그려보기 완료
- [ ] `models/early_exit_lstm.py` 모델 클래스 구현 완료
- [ ] Entropy 계산 모듈 구현 완료
- [ ] Multi-exit loss 구현 완료
- [ ] Inference 모드 (조기 종료) 구현 완료
- [ ] 고정 threshold로 더미 데이터 학습 돌아가고 accuracy 출력 확인
- [ ] Exit별 종료율 출력 확인
- [ ] 김호중과 hidden_size, 입력 shape 맞추기 완료
- [ ] `checkpoints/early_exit_lstm_best.pth` 저장 확인

---

## 12. 주의사항

- `models/early_exit_lstm.py`에는 **모델 클래스 정의만** 넣을 것. 학습 코드는 `scripts/train.py`에.
- 1단계에서 동적 threshold는 구현하지 않아도 됨. **고정값으로 먼저 동작 확인** 후 2단계에서 추가.
- `hidden_size`는 **김호중과 반드시 맞출 것.** 나중에 비교 실험에서 공정한 비교를 위해.
- Exit별 종료율이 너무 한쪽으로 쏠리면 threshold 값을 조정해야 함.
  - Exit 1 종료율이 너무 낮으면 → θ₁ 높이기
  - Exit 3에 너무 몰리면 → θ₁, θ₂ 전체적으로 높이기
