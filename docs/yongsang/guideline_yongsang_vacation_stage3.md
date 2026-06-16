# 유용상 방학 3단계 가이드라인
## 실측 데이터 기반 재학습 및 동적 Threshold 경량화 실험

> 담당자: 유용상  
> 단계 목표: 실측 데이터로 LSTM/Early Exit를 재학습하고 고정 θ, 기존 동적 θ, 경량 동적 θ를 비교  
> 완료 기준: 실측 데이터 기준 4개 방식 비교와 경량 동적 θ 비교 결과 생성

---

## 1. 방학 3단계 공통 목표

3단계는 세 명 모두 **수집된 데이터를 이용해 본격 실험을 수행하는 단계**다.  
예나는 실시간 입력과 노이즈 데이터를 만들고, 호중은 Pi/ONNX 측정을 수행하며, 용상은 모델 재학습과 threshold 실험을 수행한다.

| 담당 | 3단계 역할 |
|---|---|
| 장예나 | 실시간 입력 루프 검증, 노이즈 추가 시뮬레이터 생성 |
| 김호중 | ONNX/INT8 변환 및 Pi CSV 추론 측정 |
| 유용상 | LSTM/EE 재학습 및 동적 θ 경량화 실험 |

---

## 2. 실측 데이터 재학습

```bash
python project/scripts/train.py
python project/scripts/train_early_exit.py
python project/experiments/compare_baselines.py
```

| 방식 | 목적 |
|---|---|
| 임계값 방식 | 규칙 기반 대조군 |
| 일반 LSTM | 풀 추론 기준 모델 |
| Early Exit 고정 θ | 조기 종료 기준 모델 |
| Early Exit 동적 θ | 동적 threshold 기준 모델 |

---

## 3. 동적 Threshold 경량화

| 방식 | 설명 | 기대 효과 |
|---|---|---|
| max-min 범위 | `std` 대신 `max(window) - min(window)` 사용 | 제곱근 연산 제거 |
| 주기적 업데이트 | 매 시점이 아니라 K 시점마다 θ 갱신 | 계산 횟수 감소 |
| 조합 방식 | max-min + 주기적 업데이트 | 오버헤드 최소화 |

```python
if timestep % 3 == 0:
    variance = max(recent_window) - min(recent_window)
    theta_1, theta_2 = adjust_threshold(variance)
```

---

## 4. 완료 기준 체크리스트

- [ ] 실측 데이터 기준 Baseline LSTM 학습 완료
- [ ] 실측 데이터 기준 Early Exit LSTM 학습 완료
- [ ] 4개 방식 비교 결과 생성
- [ ] max-min 기반 경량 동적 θ 구현
- [ ] 주기적 업데이트 방식 구현
- [ ] 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교

---

## 5. 주의사항

- 체크포인트는 용상 환경에서 생성된 결과로 관리한다.
- 동적 θ가 항상 고정 θ보다 빠르게 나와야 하는 것은 아니다.
- threshold 파라미터를 바꾸면 값과 이유를 문서화한다.
