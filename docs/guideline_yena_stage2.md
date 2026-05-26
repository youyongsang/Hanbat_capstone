# 장예나 2단계 가이드라인
## 데이터 전처리 파이프라인 구축

> 담당자: 장예나  
> 목표: 1단계에서 생성한 시뮬레이터 데이터를 모델 입력 형태로 가공하는 파이프라인 구축  
> 완료 기준: `utils/dataloader.py` 구현 완료, 김호중·유용상이 바로 사용 가능한 상태

---

## 1. 해야 할 일 순서

```
1. 1단계 CSV 데이터 검증 (shape, 레이블 분포 확인)
2. 슬라이딩 윈도우 전처리 구현
3. 정규화 파이프라인 구현
4. DataLoader 구현
5. 김호중·유용상에게 공유
```

---

## 2. 1단계 데이터 검증

전처리 시작 전에 1단계에서 생성한 데이터가 올바른지 확인한다.

### 확인 항목

| 항목 | 기대값 | 확인 방법 |
|---|---|---|
| CSV 컬럼 수 | 6개 (timestamp, rps, channel_occupancy, packet_loss, latency, label) | `df.columns` |
| 레이블 종류 | 0, 1, 2, 3만 존재 | `df['label'].unique()` |
| 결측값 | 없음 | `df.isnull().sum()` |
| 레이블 분포 | 0: 60%, 1: 20%, 2: 15%, 3: 5% 근사 | `df['label'].value_counts()` |
| 피처 범위 | 레이블별 범위 일치 | 레이블별 평균값 확인 |

### 레이블별 피처 범위 검증

```python
for label in [0, 1, 2, 3]:
    subset = df[df['label'] == label]
    print(f"Label {label}:")
    print(f"  RPS: {subset['rps'].min():.1f} ~ {subset['rps'].max():.1f}")
    print(f"  Channel: {subset['channel_occupancy'].min():.1f} ~ {subset['channel_occupancy'].max():.1f}")
```

---

## 3. 슬라이딩 윈도우 전처리

시계열 데이터를 10 타임스텝 단위로 잘라 모델 입력 형태로 만든다.

### 동작 원리

```
원본 시계열 (100 타임스텝):
[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, ...]

윈도우 1: [t1 ~ t10]  → 레이블: t10 기준
윈도우 2: [t2 ~ t11]  → 레이블: t11 기준
윈도우 3: [t3 ~ t12]  → 레이블: t12 기준
...
```

### 파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| window_size | 10 | 타임스텝 수 |
| stride | 1 | 윈도우 이동 간격 |
| label_index | -1 | 마지막 타임스텝 기준 레이블 |

### 출력 shape

```
X: (N, 10, 4)   ← N개 샘플, 10 타임스텝, 4개 피처
y: (N,)         ← N개 레이블
```

---

## 4. 정규화

### Min-Max 정규화

```python
X_norm = (X - X_min) / (X_max - X_min)
```

### 정규화 기준값

| 피처 | X_min | X_max |
|---|---|---|
| RPS | 0 | 1000 |
| 채널 점유율 | 0 | 100 |
| 패킷 손실률 | 0 | 30 |
| 응답 지연 | 0 | 500 |

### ⚠️ 중요 — 기준값 저장

정규화 기준값은 반드시 별도 파일로 저장할 것.  
나중에 실제 추론 시에도 같은 기준값으로 정규화해야 하기 때문.

```python
# scaler_params.json 저장
{
    "rps": {"min": 0, "max": 1000},
    "channel_occupancy": {"min": 0, "max": 100},
    "packet_loss": {"min": 0, "max": 30},
    "latency": {"min": 0, "max": 500}
}
```

---

## 5. DataLoader 구현

### `utils/dataloader.py` 역할

1단계 CSV를 읽어서 PyTorch DataLoader로 반환하는 함수.  
김호중·유용상이 모델 학습 시 그대로 가져다 쓸 수 있어야 함.

### 인터페이스

```python
def get_dataloader(data_path, batch_size=32, shuffle=True, window_size=10):
    """
    Args:
        data_path: CSV 파일 경로 (예: 'data/real/train.csv')
        batch_size: 배치 크기
        shuffle: 데이터 섞기 여부
        window_size: 슬라이딩 윈도우 크기
    Returns:
        DataLoader: (X, y) 배치를 반환하는 DataLoader
                    X shape: (batch, 10, 4)
                    y shape: (batch,)
    """
```

### 사용 예시

```python
# 김호중, 유용상이 이렇게 쓸 수 있어야 함
from utils.dataloader import get_dataloader

train_loader = get_dataloader('data/real/train.csv', batch_size=32, shuffle=True)
val_loader   = get_dataloader('data/real/val.csv',   batch_size=32, shuffle=False)
test_loader  = get_dataloader('data/real/test.csv',  batch_size=32, shuffle=False)

for X, y in train_loader:
    print(X.shape)  # (32, 10, 4)
    print(y.shape)  # (32,)
    break
```

---

## 6. 데이터 저장 구조

```
data/
├── dummy/                  # 1단계 김호중 더미 데이터 (건드리지 않음)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── real/                   # 2단계 완성 데이터
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── scaler_params.json  # 정규화 기준값
```

---

## 7. 완료 기준 체크리스트

- [ ] 1단계 CSV 데이터 검증 완료 (결측값, 레이블 분포, 피처 범위)
- [ ] 슬라이딩 윈도우 전처리 구현 완료
- [ ] Min-Max 정규화 구현 완료
- [ ] `scaler_params.json` 저장 완료
- [ ] `utils/dataloader.py` 구현 완료
- [ ] `get_dataloader()` 함수 인터페이스 확인 완료
- [ ] X shape `(batch, 10, 4)` 출력 확인
- [ ] y shape `(batch,)` 출력 확인
- [ ] 김호중·유용상에게 `data/real/` 경로 및 사용법 공유 완료

---

## 8. 주의사항

- `utils/dataloader.py`는 김호중·유용상이 **그대로 가져다 쓰는 공용 파일**이야. 인터페이스 변경 시 반드시 두 명에게 알릴 것.
- 정규화는 **train 기준값으로 val, test에도 동일하게 적용**할 것. val/test 기준값 따로 계산하면 안 됨.
- 슬라이딩 윈도우 후 데이터 수가 원본보다 줄어드는 건 정상. `(원본 길이 - window_size + 1)`개가 생성됨.
