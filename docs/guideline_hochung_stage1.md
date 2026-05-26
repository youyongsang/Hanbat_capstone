# 김호중 1단계 가이드라인
## 베이스라인 LSTM 구현

> 담당자: 김호중  
> 목표: 일반 LSTM 분류 모델 구현 및 더미 데이터로 동작 확인  
> 완료 기준: 더미 데이터로 학습이 돌아가고 accuracy가 출력되면 됨

---

## 1. 해야 할 일 순서

```
1. 더미 데이터 생성 (generate_data.py)
2. 베이스라인 LSTM 모델 구현 (baseline_lstm.py)
3. 학습 스크립트 구현 (scripts/train.py)
4. 평가 스크립트 구현 (scripts/evaluate.py)
5. 동작 확인
```

---

## 2. 더미 데이터 스펙

장예나 실제 데이터가 나오기 전까지 사용하는 임시 데이터.  
**shape과 레이블은 실제 데이터와 동일하게** 맞춰야 나중에 교체가 쉬움.

### 데이터 구조

```
shape: (샘플 수, 타임스텝, 피처 수)
     = (1000, 10, 4)
```

### 피처 4가지

| 인덱스 | 피처명 | 범위 | 단위 |
|---|---|---|---|
| 0 | 트래픽 양 (RPS) | 0 ~ 1000 | 초당 요청 수 |
| 1 | 채널 점유율 | 0 ~ 100 | % |
| 2 | 패킷 손실률 | 0 ~ 30 | % |
| 3 | 응답 지연 | 0 ~ 500 | ms |

### 레이블 4가지

| 레이블 | 의미 | 채널 점유율 기준 |
|---|---|---|
| 0 | 정상 | 40% 미만 |
| 1 | 혼잡 경고 | 40 ~ 65% |
| 2 | 혼잡 | 65 ~ 85% |
| 3 | 심각 혼잡 | 85% 이상 |

### 샘플 분포

| 레이블 | 샘플 수 | 비율 |
|---|---|---|
| 0 (정상) | 600 | 60% |
| 1 (혼잡 경고) | 200 | 20% |
| 2 (혼잡) | 150 | 15% |
| 3 (심각) | 50 | 5% |
| **합계** | **1000** | **100%** |

### 데이터 분할

| 용도 | 샘플 수 | 비율 |
|---|---|---|
| 학습 (train) | 700 | 70% |
| 검증 (val) | 150 | 15% |
| 테스트 (test) | 150 | 15% |

### ⚠️ 더미 데이터 생성 시 주의사항

단순 랜덤값이 아니라 **레이블에 맞는 범위**에서 생성해야 함.  
그래야 모델이 의미있는 패턴을 학습할 수 있음.

| 레이블 | RPS | 채널 점유율 (%) | 패킷 손실률 (%) | 응답 지연 (ms) |
|---|---|---|---|---|
| 0 (정상) | 0 ~ 400 | 0 ~ 40 | 0 ~ 2 | 0 ~ 50 |
| 1 (혼잡 경고) | 400 ~ 650 | 40 ~ 65 | 2 ~ 8 | 50 ~ 150 |
| 2 (혼잡) | 650 ~ 850 | 65 ~ 85 | 8 ~ 20 | 150 ~ 300 |
| 3 (심각) | 850 ~ 1000 | 85 ~ 100 | 20 ~ 30 | 300 ~ 500 |

---

## 3. 베이스라인 LSTM 모델 스펙

Early Exit 없는 일반 LSTM. 나중에 유용상 Early Exit LSTM과 비교하는 **Baseline ②** 역할.

### 모델 구조

```
입력: (batch, 10, 4)
        ↓
LSTM Layer 1 (hidden_size=128)
        ↓
LSTM Layer 2 (hidden_size=128)
        ↓
LSTM Layer 3 (hidden_size=128)
        ↓
FC Layer (128 → 4)
        ↓
출력: (batch, 4)  ← 4개 클래스 확률
```

### 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| hidden_size | 128 | 유용상 Early Exit 모델과 동일하게 맞출 것 |
| num_layers | 3 | |
| dropout | 0.2 | 과적합 방지 |
| batch_size | 32 | |
| learning_rate | 0.001 | |
| epochs | 50 | |
| optimizer | Adam | |
| loss | CrossEntropyLoss | 4클래스 분류 |

> **hidden_size는 유용상과 반드시 맞춰야 함.**  
> 나중에 비교 실험할 때 공정한 비교가 되어야 하기 때문.

---

## 4. 파일 구조

```
project/
├── data/
│   └── dummy/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   └── baseline_lstm.py      ← 모델 클래스 정의
├── utils/
│   ├── dataloader.py         ← 데이터 로딩
│   └── metrics.py            ← 정확도 측정
├── scripts/
│   ├── generate_data.py      ← 더미 데이터 생성
│   ├── train.py              ← 학습 실행
│   └── evaluate.py           ← 평가 실행
└── checkpoints/              ← 학습된 모델 저장
```

---

## 5. 각 파일 역할

### `scripts/generate_data.py`
더미 데이터를 생성하고 `data/dummy/`에 저장.

출력:
```
data/dummy/train.csv
data/dummy/val.csv
data/dummy/test.csv
```

---

### `models/baseline_lstm.py`
LSTM 모델 클래스 정의만 담당. 학습 코드는 여기 넣지 않음.

```python
class BaselineLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3, num_classes=4):
        ...
    def forward(self, x):
        ...
        return output  # (batch, 4)
```

---

### `utils/dataloader.py`
CSV 파일을 읽어서 PyTorch DataLoader로 반환.

```python
def get_dataloader(path, batch_size=32, shuffle=True):
    ...
    return dataloader
```

---

### `utils/metrics.py`
정확도, 혼동 행렬 등 평가 함수 모음.

```python
def accuracy(y_pred, y_true):
    ...
def confusion_matrix(y_pred, y_true):
    ...
```

---

### `scripts/train.py`
학습 실행 스크립트.

출력 예시:
```
Epoch 1/50 | Train Loss: 1.234 | Train Acc: 45.2% | Val Acc: 43.1%
Epoch 2/50 | Train Loss: 1.102 | Train Acc: 52.3% | Val Acc: 51.8%
...
Best model saved: checkpoints/baseline_lstm_best.pth
```

---

### `scripts/evaluate.py`
테스트셋 평가 스크립트.

출력 예시:
```
Test Accuracy: 87.3%
Label 0 (정상):      Precision: 0.91  Recall: 0.93
Label 1 (혼잡 경고): Precision: 0.84  Recall: 0.82
Label 2 (혼잡):      Precision: 0.86  Recall: 0.85
Label 3 (심각):      Precision: 0.79  Recall: 0.76
```

---

## 6. 유용상과 맞춰야 할 것

나중에 비교 실험 시 공정한 비교를 위해 아래 항목을 유용상과 맞춰야 함.

| 항목 | 맞춰야 하는 이유 |
|---|---|
| `hidden_size` | 두 모델의 파라미터 수를 비슷하게 유지 |
| 입력 shape `(batch, 10, 4)` | 같은 데이터로 학습 |
| 출력 클래스 수 `4` | 같은 레이블 기준 |
| 학습/검증/테스트 분할 | 같은 데이터셋으로 비교 |
| 평가 지표 | accuracy, loss, 추론 시간 동일 기준 |

---

## 7. 실제 데이터 교체 시

장예나 시뮬레이터 완성 후 `data/dummy/` → `data/real/`로 경로만 바꾸면 됨.  
shape과 레이블이 동일하게 맞춰져 있기 때문에 코드 수정 없이 교체 가능.

---

## 8. 완료 기준 체크리스트

- [x] `scripts/generate_data.py` 구현 완료
- [x] `data/dummy/` 경로에 CSV 3개 생성 완료
- [x] `models/baseline_lstm.py` 모델 클래스 구현 완료
- [x] `utils/dataloader.py` 구현 완료
- [x] `utils/metrics.py` 구현 완료
- [x] `scripts/train.py` 구현 완료
- [x] `scripts/evaluate.py` 구현 완료
- [x] 더미 데이터로 학습 돌아가고 accuracy 출력 확인
- [x] `checkpoints/baseline_lstm_best.pth` 저장 확인
- [ ] 유용상에게 `hidden_size`, 입력 shape 확인 완료

---

## 9. 주의사항

- `models/baseline_lstm.py`에는 **모델 클래스 정의만** 넣을 것. 학습 코드는 `scripts/train.py`에.
- 더미 데이터는 나중에 장예나 실제 데이터로 **경로만 바꿔서** 교체할 수 있게 설계할 것.
- hidden_size, 입력 shape은 **유용상과 반드시 사전에 맞출 것.**
- 학습 완료 후 best model은 `checkpoints/`에 저장할 것.
