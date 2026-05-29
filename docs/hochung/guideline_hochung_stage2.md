# 김호중 2단계 가이드라인
## 채널 전환 로직 설계 및 비교 실험 프레임워크 구성

> 담당자: 김호중  
> 목표: 분류 결과 기반 채널 전환 로직 구현 및 4개 방식 비교 실험 프레임워크 구성  
> 완료 기준: 4개 비교 대상이 동일한 조건에서 실행되고 결과가 CSV로 저장되면 됨

---

## 1. 해야 할 일 순서

```
1. 장예나 DataLoader 연결 확인
2. 채널 전환 로직 구현
3. 비교 실험 프레임워크 구성
4. Baseline ① (임계값 방식) 구현
5. Baseline ② (일반 LSTM) 학습 및 평가
6. 실험 결과 CSV 저장
```

---

## 2. 장예나 DataLoader 연결

2단계 시작 전 장예나 `utils/dataloader.py`로 교체 확인.

```python
from utils.dataloader import get_dataloader

# 더미 데이터 → 실제 데이터로 경로만 변경
train_loader = get_dataloader('data/real/train.csv', batch_size=32)
val_loader   = get_dataloader('data/real/val.csv',   batch_size=32, shuffle=False)
test_loader  = get_dataloader('data/real/test.csv',  batch_size=32, shuffle=False)
```

---

## 3. 채널 전환 로직

분류 결과(0~3)를 받아서 채널 전환 명령을 결정하는 규칙 기반 모듈.  
딥러닝 없이 순수 규칙으로 동작.

### 전환 로직

| 분류 결과 | 혼잡 수준 | 채널 전환 전략 |
|---|---|---|
| 0 | 정상 | 채널 유지 |
| 1 | 혼잡 경고 | 인접 채널 모니터링, 전환 준비 |
| 2 | 혼잡 | 덜 혼잡한 채널로 전환 |
| 3 | 심각 혼잡 | 즉시 채널 전환 또는 5GHz 이동 |

### 구현 위치

```
experiments/
└── channel_optimizer.py   ← 채널 전환 로직
```

### 인터페이스

```python
def optimize_channel(predicted_label, current_channel, available_channels):
    """
    Args:
        predicted_label: 분류 결과 (0~3)
        current_channel: 현재 채널 번호
        available_channels: 사용 가능한 채널 목록
    Returns:
        next_channel: 전환할 채널 번호 (변경 없으면 current_channel 반환)
        action: 'keep' / 'monitor' / 'switch' / 'emergency'
    """
```

---

## 4. 비교 실험 프레임워크

4개 방식을 **동일한 조건**에서 비교하는 프레임워크.  
공정한 비교를 위해 같은 테스트 데이터, 같은 평가 지표를 사용해야 함.

### 비교 대상 4개

| 번호 | 방식 | 설명 |
|---|---|---|
| ① | 임계값 방식 (현행) | 채널 점유율 65% 초과 시 무조건 전환 |
| ② | 일반 LSTM | Early Exit 없는 풀 추론 분류 |
| ③ | Early Exit + 고정 θ | threshold 고정값 사용 (유용상 구현) |
| ④ | 제안 모델 | Early Exit + 동적 θ (유용상 구현) |

### 평가 지표

| 지표 | 측정 방법 | 담당 |
|---|---|---|
| 분류 정확도 (%) | 전체 테스트셋 기준 | 김호중 |
| 레이블별 정확도 | 0~3 각각 | 김호중 |
| 평균 추론 시간 (ms) | 100회 평균 | 김호중 |
| Exit별 종료율 (%) | Exit 1/2/3 비율 | 유용상 |
| 불필요 채널 전환 횟수 | 정상 구간에서 전환 발생 수 | 김호중 |

### 실험 결과 저장

```
results/
├── baseline_threshold.csv    # ① 임계값 방식
├── baseline_lstm.csv         # ② 일반 LSTM
├── early_exit_fixed.csv      # ③ Early Exit + 고정 θ
├── early_exit_dynamic.csv    # ④ 제안 모델
└── comparison_summary.csv    # 4개 비교 요약
```

---

## 5. Baseline ① 임계값 방식 구현

딥러닝 없이 채널 점유율 임계값으로만 판단하는 현행 방식.

```python
def threshold_baseline(channel_occupancy):
    """
    현행 임계값 기반 혼잡 감지
    채널 점유율만 보고 레이블 결정
    """
    if channel_occupancy < 40:
        return 0  # 정상
    elif channel_occupancy < 65:
        return 1  # 혼잡 경고
    elif channel_occupancy < 85:
        return 2  # 혼잡
    else:
        return 3  # 심각
```

> 이 방식은 시계열 패턴을 전혀 고려하지 않음.  
> 현재 타임스텝의 채널 점유율 하나만 보고 판단.  
> 이게 LSTM 대비 왜 부족한지 실험으로 증명하는 게 목적.

---

## 6. 비교 실험 실행 스크립트

### `experiments/compare_baselines.py`

4개 방식을 순서대로 실행하고 결과를 저장하는 메인 실험 스크립트.

```python
# 실행 방법
python experiments/compare_baselines.py

# 출력 예시
Running Baseline ① (Threshold)...
  Accuracy: 71.3% | Avg Inference: 0.1ms

Running Baseline ② (LSTM Full)...
  Accuracy: 88.1% | Avg Inference: 8.2ms

Running Baseline ③ (Early Exit Fixed θ)...
  Accuracy: 87.8% | Avg Inference: 3.9ms
  Exit 1: 64.2% | Exit 2: 22.1% | Exit 3: 13.7%

Running Baseline ④ (Early Exit Dynamic θ)...
  Accuracy: 89.2% | Avg Inference: 3.6ms
  Exit 1: 68.4% | Exit 2: 20.3% | Exit 3: 11.3%

Results saved to results/comparison_summary.csv
```

---

## 7. 완료 기준 체크리스트

- [ ] 장예나 DataLoader 연결 및 동작 확인
- [ ] 채널 전환 로직 (`channel_optimizer.py`) 구현 완료
- [ ] Baseline ① 임계값 방식 구현 완료
- [ ] Baseline ② 일반 LSTM 학습 및 평가 완료
- [ ] 비교 실험 프레임워크 (`compare_baselines.py`) 구성 완료
- [ ] 4개 방식 동일 조건 실행 확인
- [ ] 결과 CSV 저장 확인 (`results/` 폴더)
- [ ] 유용상에게 비교 실험 프레임워크 인터페이스 공유 완료

---

## 8. 주의사항

- 4개 방식 비교 시 **반드시 동일한 테스트 데이터**를 사용할 것.
- 추론 시간 측정은 **GPU 아닌 CPU 기준**으로 측정. 실제 엣지 환경 기준이기 때문.
- Baseline ①은 딥러닝 없이 규칙만 쓰는 거라 추론 시간이 거의 0에 가까움. 이게 정상.
- 유용상 모델(③, ④)이 완성되기 전에 ①, ②만 먼저 실행해서 프레임워크 동작 확인해도 됨.
