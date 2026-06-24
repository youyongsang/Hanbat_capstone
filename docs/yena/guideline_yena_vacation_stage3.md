# 장예나 방학 3단계 가이드라인
## SOTA 비교 실험 데이터 관리 및 결과 로깅

> 담당자: 장예나  
> 기간: 방학 5주차  
> 목표: SOTA 비교 실험 결과 데이터 수집 및 정리  
> 완료 기준: SOTA 비교 실험 결과 CSV 수집 완료

---

## 1. 해야 할 일 순서

```
1. 유용상·김호중 SOTA 비교 실험 결과 수집
2. 결과 CSV 정리 및 저장
3. 연산량 절감 비교 데이터 정리
```

---

## 2. 수집할 결과 데이터

### comparison_summary_final.csv

```
model, accuracy, avg_inference_ms, model_size_mb
① 임계값 방식, 42.2, 0.011, -
② 일반 LSTM, 95.4, 0.779, 1.28
③ EE 고정 θ, 96.9, 0.563, 0.33
④ EE 동적 θ, 96.9, 0.582, 0.33
⑤ SOTA 모델, X.X, X.XXX, X.XX   ← 유용상·김호중에게 받기
```

### computation_comparison.csv

```
model, flops, inference_ms, reduction_vs_sota
```

---

## 3. 완료 기준 체크리스트

- [ ] SOTA 비교 실험 결과 CSV 수집 완료
- [ ] `comparison_summary_final.csv` 업데이트 완료
- [ ] `computation_comparison.csv` 정리 완료

---

## 4. 주의사항

- 수치는 유용상·김호중에게 받아서 채울 것. 직접 추측하지 않기.
- 비교 기준이 동일한 테스트 데이터인지 확인할 것.
