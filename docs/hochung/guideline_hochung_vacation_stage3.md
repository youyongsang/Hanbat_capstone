# 김호중 방학 3단계 가이드라인
## SOTA 비교 실험 프레임워크 추가

> 담당자: 김호중  
> 기간: 방학 5주차  
> 목표: 기존 비교 실험 프레임워크에 SOTA 모델 추가  
> 완료 기준: SOTA 비교 실험 완료, 연산량 절감 비교 결과 저장

---

## 1. 해야 할 일 순서

```
1. 유용상이 선정한 SOTA 모델 확인
2. 비교 실험 프레임워크에 SOTA 모델 추가
3. 연산량 절감 중심 비교 실험 수행
4. 결과 CSV 저장
5. 장예나에게 결과 전달
```

---

## 2. 비교 실험 프레임워크 수정

기존 `compare_baselines.py`에 SOTA 모델 추가.

```python
# compare_baselines.py 수정

# 기존 4개 방식
print("Running Baseline ① (Threshold)...")
print("Running Baseline ② (LSTM Full)...")
print("Running Baseline ③ (Early Exit Fixed θ)...")
print("Running Baseline ④ (Early Exit Dynamic θ)...")

# 추가
print("Running SOTA Model...")
sota_result = run_sota_model(test_loader)
print(f"  Accuracy: {sota_result['accuracy']:.1f}%")
print(f"  Avg Inference: {sota_result['inference_ms']:.3f}ms")
```

---

## 3. 연산량 절감 비교

```python
# 연산량 절감률 계산
reduction_vs_sota = (sota_inference - ee_inference) / sota_inference * 100
print(f"SOTA 대비 연산량 절감: {reduction_vs_sota:.1f}%")
```

### 결과 저장

```
project/results/hojung/
├── comparison_summary_final.csv   # 기존 4개 + SOTA
└── computation_comparison.csv     # 연산량 비교
```

`computation_comparison.csv` 컬럼:
```
model, accuracy, avg_inference_ms, model_size_mb, reduction_vs_sota_pct
```

---

## 4. 완료 기준 체크리스트

- [ ] SOTA 모델 비교 실험 프레임워크 추가 완료
- [ ] `comparison_summary_final.csv` 저장 완료
- [ ] `computation_comparison.csv` 저장 완료
- [ ] 장예나에게 결과 전달 완료 (시각화용)
- [ ] 유용상에게 비교 결과 전달 완료

---

## 5. 주의사항

- 동일한 테스트 데이터로 모든 방식 비교할 것
- SOTA 모델은 유용상에게 받아서 구현. 직접 설계하지 않음.
- 연산량은 추론 시간 기준으로 비교 (FLOPs 측정 어려우면 ms 기준으로)
