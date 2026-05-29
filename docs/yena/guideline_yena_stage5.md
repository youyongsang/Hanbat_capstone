# 장예나 5단계 가이드라인
## 실기기 환경 구성 지원 및 최종 결과 정리

> 담당자: 장예나  
> 목표: Raspberry Pi 환경 구성 지원 및 최종 실험 결과 통합 정리  
> 완료 기준: 최종 결과 데이터 통합 완료, 보고서용 자료 완성

---

## 1. 해야 할 일 순서

```
1. Raspberry Pi 환경 구성 지원
2. 실기기 추론 데이터 수집 지원
3. PC vs Pi 성능 비교표 작성
4. 최종 결과 통합 정리
5. 보고서 데이터·실험 섹션 작성
```

---

## 2. Raspberry Pi 환경 구성 지원

### 김호중 Pi 세팅 시 지원할 것

Pi에서 추론할 때 정규화 기준값이 필요해.  
4단계에서 저장한 `scaler_params.json`을 Pi에 전달.

```
전달할 파일:
data/real/scaler_params.json
```

### Pi 추론용 테스트 데이터 준비

Pi에서 추론 테스트할 샘플 데이터 준비.

```python
# 시나리오별 대표 샘플 각 10개씩 추출
# total: 40개 샘플
test_samples = {
    'startup_surge': 10개,
    'emergency_ramp': 10개,
    'lunch_restart': 10개,
    'imbalanced_ap_load': 10개
}
# 저장
data/real/pi_test_samples.csv
```

---

## 3. PC vs Pi 성능 비교표

김호중이 Pi 실측 결과 가져오면 PC 결과와 비교표 작성.

| 항목 | PC (x86) | Raspberry Pi 4 8GB |
|---|---|---|
| Exit 1 추론 시간 | | |
| Exit 2 추론 시간 | | |
| Exit 3 추론 시간 | | |
| 평균 추론 시간 | | |
| 모델 크기 | | |

---

## 4. 최종 결과 통합

### 취합할 데이터

| 출처 | 내용 |
|---|---|
| 김호중 | 4개 방식 비교 결과, 경량화 결과, Pi 실측 결과 |
| 유용상 | 고정 θ vs 동적 θ 분석, 시나리오별 분석 |
| 본인 | 시각화 그래프, 비교표 |

### 최종 저장 구조

```
results/
├── comparison_summary.csv        # 4개 방식 최종 비교
├── quantization_comparison.csv   # 경량화 전후 비교
├── pi_inference_results.csv      # Pi 실측 결과
├── scenario_analysis/            # 시나리오별 분석
├── accuracy_comparison.png       # 그래프
├── accuracy_vs_latency.png       # 그래프
├── exit_rate_comparison.png      # 그래프
└── scenario_accuracy.png         # 그래프
```

---

## 5. 보고서 담당 섹션

### 작성할 내용

**데이터 및 실험 환경 섹션**
- 트래픽 시뮬레이터 설계 설명
- 시나리오 4가지 설명
- 데이터셋 구성 (샘플 수, 분포, 분할)
- 전처리 방법 (슬라이딩 윈도우, 정규화)

**실험 결과 섹션**
- 4개 방식 비교표
- 시나리오별 분석 결과
- PC vs Pi 성능 비교

---

## 6. 완료 기준 체크리스트

- [ ] `scaler_params.json` 김호중에게 전달 완료
- [ ] Pi 테스트 샘플 데이터 준비 완료
- [ ] PC vs Pi 성능 비교표 완성
- [ ] 최종 결과 데이터 통합 완료
- [ ] 보고서 데이터·실험 환경 섹션 초안 완성
- [ ] 보고서 실험 결과 섹션 초안 완성

---

## 7. 주의사항

- Pi 실측 결과가 PC보다 느린 건 당연한 거야. 그대로 기록할 것.
- 보고서 수치는 실험 완료 후 채울 것. 예상값 미리 넣지 않기.
