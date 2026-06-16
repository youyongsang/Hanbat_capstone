# 김호중 방학 3단계 가이드라인
## 실측 CSV 기반 Raspberry Pi 추론 성능 측정

> 담당자: 김호중  
> 목표: 예나가 수집한 실제 WiFi CSV로 Raspberry Pi 추론 성능을 반복 측정  
> 완료 기준: fixed/dynamic, FP32/INT8 조합별 Pi 결과 CSV/TXT/MD 생성

---

## 1. 해야 할 일 순서

```
1. 예나 실측 test.csv를 Pi 배포 폴더에 배치
2. fixed FP32 staged ONNX 추론 실행
3. fixed INT8 staged ONNX 추론 실행
4. dynamic FP32 staged ONNX 추론 실행
5. dynamic INT8 staged ONNX 추론 실행
6. 결과 분석 스크립트 실행
```

---

## 2. 실행 예시

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

```bash
python3 inference_pi.py \
  --mode staged \
  --stage1 early_exit_fixed_stage1_int8.onnx \
  --stage2 early_exit_fixed_stage2_int8.onnx \
  --stage3 early_exit_fixed_stage3_int8.onnx \
  --data test.csv \
  --output pi_fixed_staged_int8_results.csv \
  --repeats 5
```

동적 threshold는 dynamic 모델 파일과 `--dynamic-theta` 옵션을 사용한다.

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
- 예나 데이터 수집 환경과 다른 장소에서 수행했다면, 결과는 절대 성능 비교보다 동작 검증으로 해석한다.
