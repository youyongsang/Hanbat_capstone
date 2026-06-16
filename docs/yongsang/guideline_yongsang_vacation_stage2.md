# 유용상 방학 2단계 가이드라인
## 실측 데이터 기반 LSTM 및 Early Exit LSTM 재학습

> 담당자: 유용상  
> 목표: 실제 WiFi 데이터 기준으로 일반 LSTM과 Early Exit LSTM을 재학습하고 기본 성능 비교  
> 완료 기준: 임계값 방식, 일반 LSTM, EE 고정 θ, EE 동적 θ 비교 결과 생성

---

## 1. 해야 할 일 순서

```
1. 실측 데이터 기준 Baseline LSTM 학습
2. 실측 데이터 기준 Early Exit LSTM 학습
3. 4개 방식 비교 실험 실행
4. 정확도, 추론 시간, Exit 비율 정리
5. 기존 데이터 결과와 차이 분석
```

---

## 2. 실행 흐름

```bash
python project/scripts/train.py
python project/scripts/train_early_exit.py
python project/experiments/compare_baselines.py
```

---

## 3. 비교 대상

| 방식 | 목적 |
|---|---|
| 임계값 방식 | 규칙 기반 대조군 |
| 일반 LSTM | 풀 추론 기준 모델 |
| Early Exit 고정 θ | 조기 종료 기준 모델 |
| Early Exit 동적 θ | 동적 threshold 기준 모델 |

---

## 4. 완료 기준 체크리스트

- [ ] 실측 데이터 기준 Baseline LSTM 학습 완료
- [ ] 실측 데이터 기준 Early Exit LSTM 학습 완료
- [ ] 4개 방식 비교 결과 생성
- [ ] 정확도, 평균 추론 시간, Exit 비율 정리
- [ ] 기존 시뮬레이터/외부 데이터 결과와 차이 분석

---

## 5. 주의사항

- 체크포인트는 용상 환경에서 생성된 결과로 관리한다.
- 호중 Pi 배포용 체크포인트와 혼동하지 않는다.
- 정확도가 낮게 나와도 데이터 분포 변화로 해석하고 그대로 기록한다.
