# 유용상 방학 가이드라인
## 실측 WiFi 데이터 기반 Early Exit LSTM 및 동적 Threshold 재검증

> 담당자: 유용상  
> 목표: 실제 WiFi 데이터와 노이즈 시뮬레이터 데이터로 Early Exit LSTM 및 경량 동적 threshold를 재검증  
> 완료 기준: 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교표 완성, 실측 데이터 기반 분석 정리

---

## 0. 공통 실험 원칙

용상 브랜치는 모델 구조와 동적 threshold 개선을 담당한다. Raspberry Pi용 ONNX, INT8 모델, Pi 실측 결과는 호중 담당 산출물이므로 용상 브랜치에 억지로 가져오지 않는다.

| 구분 | 기준 |
|---|---|
| 용상 담당 | Early Exit LSTM 구조, 고정/동적 threshold, 경량 동적 θ 개선 |
| 예나 입력 | 실제 WiFi CSV 데이터셋, 라벨 기준, 노이즈 시뮬레이터 데이터 |
| 호중 입력 | Pi 실측 결과, ONNX/INT8 배포 성능 |
| 용상 브랜치 산출물 | 모델 코드, PC 기준 평가 결과, 분석 문서 |
| 제외 대상 | 호중 환경의 체크포인트, ONNX, Pi 결과 파일 |

---

## 1단계. 실측 데이터 수령 및 입력 형식 검증

### 해야 할 일

```
1. 예나가 생성한 data/real_wifi CSV 수령
2. 컬럼 형식 확인
3. 라벨 분포 확인
4. 슬라이딩 윈도우 변환 확인
5. 기존 dataloader와 호환되는지 검증
```

### 확인 기준

| 항목 | 기준 |
|---|---|
| 필수 컬럼 | `timestamp`, `rps`, `channel_occupancy`, `packet_loss`, `latency`, `label` |
| 입력 피처 | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| 라벨 | 0/1/2/3 |
| 윈도우 | 최근 10개 시점 기준 `(10, 4)` |

### 완료 기준

- [ ] `data/real_wifi/train.csv`, `val.csv`, `test.csv` 로딩 확인
- [ ] 라벨별 샘플 수 확인
- [ ] dataloader 출력 shape 확인
- [ ] 기존 외부/시뮬레이터 데이터와 분포 차이 기록

---

## 2단계. Early Exit LSTM 재학습 및 기본 비교

### 실행 흐름

실측 WiFi 데이터 기준으로 일반 LSTM과 Early Exit LSTM을 다시 학습하고 비교한다.

```bash
python project/scripts/train.py
python project/scripts/train_early_exit.py
python project/experiments/compare_baselines.py
```

### 비교 대상

| 방식 | 목적 |
|---|---|
| 임계값 방식 | 규칙 기반 대조군 |
| 일반 LSTM | 풀 추론 기준 모델 |
| Early Exit 고정 θ | 조기 종료 기준 모델 |
| Early Exit 동적 θ | 동적 threshold 기준 모델 |

### 완료 기준

- [ ] 실측 데이터 기준 Baseline LSTM 학습 완료
- [ ] 실측 데이터 기준 Early Exit LSTM 학습 완료
- [ ] 4개 방식 비교 결과 생성
- [ ] 정확도, 평균 추론 시간, Exit 비율 정리

---

## 3단계. 동적 Threshold 경량화 개선

### 기존 문제

```text
std 기반 계산
→ 매 타임스텝 제곱근 연산 발생
→ Exit 비율 개선 이득보다 동적 θ 계산 오버헤드가 커질 수 있음
```

### 개선안

| 방식 | 설명 | 기대 효과 |
|---|---|---|
| max-min 범위 | `std` 대신 `max(window) - min(window)` 사용 | 제곱근 연산 제거 |
| 주기적 업데이트 | 매 시점이 아니라 K 시점마다 θ 갱신 | 계산 횟수 감소 |
| 조합 방식 | max-min + 주기적 업데이트 | 오버헤드 최소화 |

### 구현 예시

```python
if timestep % 3 == 0:
    variance = max(recent_window) - min(recent_window)
    theta_1, theta_2 = adjust_threshold(variance)
```

### 완료 기준

- [ ] max-min 기반 경량 동적 θ 구현
- [ ] 주기적 업데이트 방식 구현
- [ ] 조합 방식 구현
- [ ] 기존 동적 θ와 결과 비교

---

## 4단계. 노이즈 시뮬레이터 및 실측 데이터 재검증

### 비교 데이터

| 데이터 | 목적 |
|---|---|
| 기존 시뮬레이터 | 기준선 비교 |
| 노이즈 추가 시뮬레이터 | 동적 θ 효과가 드러나는지 확인 |
| 실제 WiFi 데이터 | 실환경 입력에서 모델 재검증 |

### 분석 포인트

| 항목 | 확인 내용 |
|---|---|
| 전체 정확도 | 고정 θ와 동적 θ가 정확도를 유지하는지 |
| 추론 시간 | 동적 θ 오버헤드가 줄었는지 |
| Exit1 비율 | 안정 구간에서 조기 종료가 늘어나는지 |
| Exit3 비율 | 깊은 추론 비율이 줄어드는지 |
| 급변 구간 정확도 | 혼잡 변화가 큰 구간에서 동적 θ가 유리한지 |

### 급변 구간 분리 기준

```python
delta = max(recent_occupancy) - min(recent_occupancy)
spike_idx = delta > SPIKE_THRESHOLD
stable_idx = ~spike_idx
```

### 완료 기준

- [ ] 기존 시뮬레이터 결과 정리
- [ ] 노이즈 추가 시뮬레이터 결과 정리
- [ ] 실제 WiFi 데이터 결과 정리
- [ ] 급변/안정 구간 분리 분석
- [ ] 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교표 작성

---

## 5단계. 호중 배포 실험 연동

용상 브랜치에서 Pi 체크포인트나 ONNX 결과를 직접 관리하지 않는다. 대신 검증된 모델 코드와 threshold 설정을 호중에게 전달한다.

### 전달 항목

| 항목 | 설명 |
|---|---|
| 모델 구조 변경사항 | Early Exit LSTM 코드 변경 여부 |
| 최종 threshold 설정 | 고정 θ, 동적 θ 초기값, spike 기준 |
| 실측 데이터 PC 결과 | 정확도, 추론 시간, Exit 비율 |
| 권장 실험 명령어 | 호중이 ONNX/INT8 변환 시 사용할 실행 순서 |

### 완료 기준

- [ ] 최종 모델 코드 커밋
- [ ] threshold 설정값 문서화
- [ ] PC 기준 결과표 작성
- [ ] 호중에게 ONNX 변환 대상 코드 전달
- [ ] 호중 Pi 결과와 용상 PC 결과 비교 분석

---

## 최종 체크리스트

- [ ] 실측 WiFi 데이터 로딩 및 shape 검증
- [ ] 실측 데이터 기준 LSTM/Early Exit 재학습
- [ ] 고정 θ, 기존 동적 θ, 경량 동적 θ 비교
- [ ] 노이즈 시뮬레이터 재실험
- [ ] 급변 구간 분리 분석
- [ ] 최종 threshold 파라미터 문서화
- [ ] 호중 배포 실험용 코드/설정 전달

---

## 주의사항

- 체크포인트와 ONNX는 각 환경에서 다시 생성하는 것을 원칙으로 한다.
- 동적 θ가 항상 고정 θ보다 빠르게 나와야 하는 것은 아니다. 정확도, Exit 분포, 급변 구간 대응력까지 같이 해석한다.
- 실측 데이터 결과와 시뮬레이터 결과는 반드시 분리해서 표기한다.
- Pi 실측 성능은 호중 결과를 기준으로 인용하고, 용상 브랜치에서는 PC 기준 모델 재검증에 집중한다.
