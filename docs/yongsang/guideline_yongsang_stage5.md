# 유용상 5단계 가이드라인
## 실기기 분류 정확도 검증 및 보고서 모델 설계 섹션 완성

> 담당자: 유용상  
> 목표: Pi 실측 결과 기반 분류 정확도 검증 및 보고서 모델 설계 섹션 최종 완성  
> 완료 기준: 보고서 모델 설계 섹션 완성, 동적 θ 향후 과제 섹션 작성 완료

---

## 1. 해야 할 일 순서

```
1. Pi 실측 결과 분류 정확도 검증
2. PC vs Pi 정확도 차이 분석
3. 보고서 모델 설계 섹션 최종 완성
4. 동적 θ 향후 과제 섹션 작성
```

---

## 2. Pi 실측 분류 정확도 검증

김호중이 Pi에서 추론한 결과(`pi_inference_results.csv`)를 받아서 정확도 검증.

### 확인 항목

| 항목 | 기준 | 확인 방법 |
|---|---|---|
| 분류 정확도 | PC 결과와 동일해야 함 | predicted_label 비교 |
| 정확도 차이 | 2% 이내 | Quantization 영향 확인 |
| 추론 시간 | PC 대비 느린 건 정상 | ms 단위 비교 |

### PC vs Pi 정확도 비교

```python
# PC 결과
pc_results = pd.read_csv('results/early_exit_fixed.csv')

# Pi 결과
pi_results = pd.read_csv('pi_inference_results.csv')

# 정확도 비교
pc_accuracy = (pc_results['predicted'] == pc_results['true_label']).mean()
pi_accuracy = (pi_results['predicted_label'] == pi_results['true_label']).mean()

print(f"PC 정확도: {pc_accuracy:.1%}")
print(f"Pi 정확도: {pi_accuracy:.1%}")
print(f"차이: {abs(pc_accuracy - pi_accuracy):.1%}")
```

---

## 3. 보고서 모델 설계 섹션 최종 완성

4단계에서 작성한 초안을 Pi 실측 결과 반영하여 완성.

### 섹션 구성

**4.1 Early Exit LSTM 구조**

```
- 입력: (batch, 10, 4) 시계열 벡터
- 3레이어 LSTM (hidden_size=128)
- 각 레이어 후 Exit Classifier 배치
- Multi-exit loss로 학습
- 추론 시 Entropy 기반 조기 종료
```

**4.2 고정 Threshold 설계 및 결과**

```
- θ₁, θ₂ 설정값 및 근거
- 실험 결과 수치 (정확도, 추론 시간, Exit 종료율)
- 시나리오별 분석 결과
- 일반 LSTM 대비 Early Exit 효과
```

**4.3 동적 Threshold 설계, 결과 및 한계**

```
- 설계 목적: 변동률 기반 θ 실시간 조정
- 구현 방법: 직전 timestep 대비 channel_occupancy delta 기반 계산
- 실험 결과: 고정 θ 대비 비교
- 한계 분석: 오버헤드 문제
- 향후 개선 방향
```

**4.4 경량화 및 엣지 배포**

```
- INT8 Quantization 적용 결과
- ONNX 변환 및 Pi 배포
- PC vs Pi 추론 시간 비교
- 모델 크기 변화
```

---

## 4. 동적 θ 향후 과제 섹션

보고서 "향후 과제" 섹션에 들어갈 내용.

### 서술 방향

> "본 연구에서 구현한 규칙 기반 동적 threshold는 Exit 1 종료율을 고정 threshold 대비 일관되게 높이는 효과를 보였다. 그러나 변동률 계산 오버헤드로 인해 실측 추론 시간은 고정 threshold 대비 증가하였다.
>
> 이를 개선하기 위한 방향으로는 첫째, K 타임스텝마다 한 번만 계산하는 주기적 업데이트, 둘째, ONNX Runtime 배포 구조에 맞춘 threshold 선택 로직 재설계, 셋째, Multi-Armed Bandit 기반 온라인 학습형 동적 threshold(UAT, 2025) 방식 적용을 제안한다.
>
> 또한 본 연구의 시뮬레이터는 패턴이 명확한 데이터를 생성하여 고정 threshold와의 정확도 차이가 미미하였다. 실제 공장 환경의 노이즈가 반영된 데이터에서는 동적 threshold의 강건성 효과가 더 두드러질 것으로 예상된다."

---

## 5. 완료 기준 체크리스트

- [ ] Pi 실측 분류 정확도 검증 완료
- [ ] PC vs Pi 정확도 차이 분석 완료 (2% 이내 확인)
- [ ] 보고서 4.1 Early Exit LSTM 구조 섹션 완성
- [ ] 보고서 4.2 고정 Threshold 섹션 완성 (수치 포함)
- [ ] 보고서 4.3 동적 Threshold 섹션 완성 (한계 분석 포함)
- [ ] 보고서 4.4 경량화 및 배포 섹션 완성
- [ ] 향후 과제 섹션 작성 완료
- [ ] 장예나에게 보고서 섹션 전달 완료

---

## 6. 주의사항

- Pi 정확도가 PC와 다르면 Quantization 영향일 수 있음. 2% 초과 시 원인 분석 필요.
- 보고서에 동적 θ 결과를 있는 그대로 쓸 것. "한계를 발견하고 개선 방향을 제시한다"는 서술이 완성도 높은 연구임.
- 수치는 실험 완료 후 채울 것. 예상값 미리 넣지 않기.
