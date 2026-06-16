# 김호중 방학 3단계 가이드라인
## ONNX/INT8 변환 및 Pi CSV 추론 성능 측정

> 담당자: 김호중  
> 단계 목표: 실측 CSV 기준으로 FP32/INT8 ONNX 및 staged Early Exit Pi 성능 측정  
> 완료 기준: fixed/dynamic, FP32/INT8 조합별 Pi 결과 CSV/TXT/MD 생성

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

## 2. 실행 흐름

```bash
python project/scripts/export_onnx.py --staged
python project/scripts/export_onnx_int8.py
python project/scripts/prepare_pi_bundle.py
```

Pi에서는 fixed/dynamic, FP32/INT8 조합을 각각 실행한다.

```bash
python3 inference_pi.py \
  --mode staged \
  --stage1 early_exit_fixed_stage1.onnx \
  --stage2 early_exit_fixed_stage2.onnx \
  --stage3 early_exit_fixed_stage3.onnx \
  --data test.csv \
  --output pi_fixed_staged_fp32_results.csv \
  --repeats 5
```

---

## 3. 측정 항목

| 항목 | 설명 |
|---|---|
| 정확도 | 실측 CSV 라벨 기준 분류 정확도 |
| 평균 추론 시간 | 반복 측정 평균 |
| p50/p95 지연 | 중앙값과 꼬리 지연 |
| Exit 비율 | Exit1/2/3 도달 비율 |
| FP32 vs INT8 | 모델 크기와 추론 시간 변화 |

---

## 4. 완료 기준 체크리스트

- [ ] fixed FP32 Pi 결과 저장
- [ ] fixed INT8 Pi 결과 저장
- [ ] dynamic FP32 Pi 결과 저장
- [ ] dynamic INT8 Pi 결과 저장
- [ ] 결과 CSV/TXT/MD 분석 파일 생성

---

## 5. 주의사항

- 한 번 측정값보다 `repeats=5` 이상 반복 평균을 기준으로 한다.
- p95 지연을 함께 기록해 꼬리 지연을 확인한다.
- INT8이 항상 더 빠르거나 정확도가 높은 것은 아니므로 결과 그대로 기록한다.
