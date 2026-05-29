# 유용상 3단계 가이드라인
## 학습 튜닝 및 최종 성능 검증

> 담당자: 유용상  
> 목표: Early Exit LSTM 하이퍼파라미터 튜닝 및 최종 성능 검증  
> 완료 기준: 고정 θ vs 동적 θ 비교 분석 완료, 시나리오별 분석 결과 정리

---

## 1. 해야 할 일 순서

```
1. 2단계 결과 분석 및 튜닝 방향 결정
2. 하이퍼파라미터 튜닝
3. 동적 threshold 파라미터 튜닝
4. 시나리오별 분석
5. 최종 모델 확정 및 저장
6. 김호중에게 최종 모델 전달
```

---

## 2. 2단계 결과 분석

3단계 시작 전 2단계 결과를 보고 튜닝 방향을 결정한다.

### 점검 항목

| 상황 | 원인 | 대응 |
|---|---|---|
| Exit 1 종료율 낮음 (40% 이하) | θ₁이 너무 낮음 | θ₁ 높이기 |
| Exit 3에 너무 몰림 (40% 이상) | θ₁, θ₂ 전체적으로 낮음 | 둘 다 높이기 |
| 동적 θ가 고정 θ보다 나쁨 | variance 기준값 잘못 설정 | HIGH_VARIANCE, MID_VARIANCE 조정 |
| 급변 구간에서 동적 θ 효과 없음 | 윈도우가 너무 길어서 감지 늦음 | recent_window 크기 줄이기 |
| 전체 정확도 낮음 | 과적합 또는 학습 부족 | dropout 조정, epochs 늘리기 |

---

## 3. 하이퍼파라미터 튜닝

### 튜닝 대상

| 파라미터 | 기본값 | 튜닝 범위 |
|---|---|---|
| hidden_size | 128 | 64, 128 (김호중과 맞출 것) |
| dropout | 0.2 | 0.1 ~ 0.4 |
| learning_rate | 0.001 | 0.0001 ~ 0.01 |
| Multi-exit loss 가중치 | 0.3 / 0.3 / 0.4 | 합이 1이 되도록 조정 |
| θ₁ (고정) | 0.3 | 0.2 ~ 0.5 |
| θ₂ (고정) | 0.6 | 0.5 ~ 0.8 |

### 튜닝 기준

```
좋은 결과의 기준:
1. Exit 1 종료율 60% 이상
2. Exit별 정확도 Exit1 < Exit2 < Exit3
3. 전체 정확도 85% 이상
4. Val acc와 Train acc 차이 5% 이내
```

---

## 4. 동적 Threshold 파라미터 튜닝

### 튜닝 대상

| 파라미터 | 기본값 | 튜닝 범위 | 영향 |
|---|---|---|---|
| HIGH_VARIANCE | 15.0 | 10 ~ 20 | 낮출수록 더 자주 깊은 추론 |
| MID_VARIANCE | 7.0 | 5 ~ 10 | |
| MIN_THRESHOLD | 0.1 | 0.05 ~ 0.2 | 급변 대응 최소 보장 |
| recent_window | 5 | 3 ~ 10 | 짧을수록 급변 빨리 감지 |
| SPIKE_THRESHOLD | 20.0 | 15 ~ 30 | 급변 감지 민감도 |

### 튜닝 기준

```
좋은 결과의 기준:
1. 안정 구간 Exit 1 종료율 > 고정 θ 대비 5% 이상
2. 급변 구간 정확도 > 고정 θ 대비 3% 이상
3. 전체 정확도 고정 θ와 비슷하거나 높음
```

---

## 5. 시나리오별 분석

장예나가 제공한 `test_with_scenario.csv`로 시나리오별 분리 분석.

### 분석할 내용

**시나리오 0 — 일과 시작 (점진적 급증)**
- 트래픽 증가 구간에서 동적 θ가 고정 θ보다 빨리 대응하는지
- Exit 종료율이 증가 구간에서 어떻게 바뀌는지

**시나리오 1 — 긴급 증산 (갑작스러운 폭증)**
- 급변 시 1 타임스텝 지연이 실제로 얼마나 발생하는지
- SPIKE 감지 로직이 동작하는지 확인

**시나리오 2 — 점심 재가동 (주기적 패턴)**
- LSTM이 주기적 패턴을 학습했을 때 임계값 대비 얼마나 좋아지는지
- 동적 θ가 안정 → 증가 전환 구간을 얼마나 잘 처리하는지

**시나리오 3 — 불균형 부하 (지속적 혼잡)**
- 혼잡이 지속되는 구간에서 동적 θ가 계속 낮게 유지되는지
- 오분류가 어느 구간에서 주로 발생하는지

### 저장 형식

```
results/scenario_analysis/
├── scenario_0_analysis.csv
├── scenario_1_analysis.csv
├── scenario_2_analysis.csv
└── scenario_3_analysis.csv
```

각 CSV 컬럼:
```
timestep, true_label, fixed_pred, dynamic_pred, 
fixed_exit_point, dynamic_exit_point,
fixed_theta_1, dynamic_theta_1,
channel_occupancy_variance
```

---

## 6. 최종 모델 확정

### 저장할 파일

```
checkpoints/
├── early_exit_fixed_final.pth      # 고정 θ 최종 모델
└── early_exit_dynamic_final.pth    # 동적 θ 최종 모델
```

### 모델 정보 기록

```
checkpoints/model_info.json
{
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "fixed_theta_1": 0.3,
    "fixed_theta_2": 0.6,
    "dynamic_HIGH_VARIANCE": 15.0,
    "dynamic_MID_VARIANCE": 7.0,
    "dynamic_MIN_THRESHOLD": 0.1,
    "dynamic_recent_window": 5,
    "test_accuracy_fixed": 0.0,
    "test_accuracy_dynamic": 0.0
}
```

---

## 7. 김호중에게 전달할 것

```
checkpoints/early_exit_fixed_final.pth
checkpoints/early_exit_dynamic_final.pth
checkpoints/model_info.json
```

### 사용 방법 전달

```python
from models.early_exit_lstm import EarlyExitLSTM
import json

info = json.load(open('checkpoints/model_info.json'))

# Baseline ③ 고정 θ
model_fixed = EarlyExitLSTM(hidden_size=info['hidden_size'])
fixed_ckpt = torch.load('checkpoints/early_exit_fixed_final.pth', map_location='cpu')
model_fixed.load_state_dict(fixed_ckpt['model_state_dict'])
model_fixed.eval()

# 제안 모델 ④ 동적 θ
model_dynamic = EarlyExitLSTM(hidden_size=info['hidden_size'])
dynamic_ckpt = torch.load('checkpoints/early_exit_dynamic_final.pth', map_location='cpu')
model_dynamic.load_state_dict(dynamic_ckpt['model_state_dict'])
model_dynamic.eval()
```

---

## 8. 완료 기준 체크리스트

- [x] 2단계 결과 분석 및 튜닝 방향 결정 완료
- [ ] 하이퍼파라미터 튜닝 완료 (Exit 1 종료율 60% 이상)
- [x] 동적 θ 파라미터 튜닝 완료
- [x] 시나리오 0~3 분리 분석 완료
- [x] 고정 θ vs 동적 θ 비교 분석 정리 완료
- [x] `early_exit_fixed_final.pth` 저장 완료
- [x] `early_exit_dynamic_final.pth` 저장 완료
- [x] `model_info.json` 저장 완료
- [ ] 김호중에게 최종 모델 및 사용법 전달 완료
- [ ] 장예나에게 시나리오별 분석 결과 전달 완료

### 2026-05-29 실행 결과

| 구분 | Fixed θ | Dynamic θ |
|---|---:|---:|
| 전체 정확도 | 95.7% | 96.3% |
| 평균 추론 시간 | 3.897ms | 3.681ms |
| Exit 1 종료율 | 20.5% | 25.6% |
| Exit 2 종료율 | 71.8% | 69.5% |
| Exit 3 종료율 | 7.7% | 4.8% |

시나리오별 정확도:

| 시나리오 | Fixed θ | Dynamic θ |
|---|---:|---:|
| startup_surge | 96.2% | 96.2% |
| emergency_ramp | 92.8% | 95.2% |
| lunch_restart | 96.4% | 96.4% |
| imbalanced_ap_load | 97.5% | 97.5% |

메모: 현재 실제 stepwise 추론 기준 Exit 1 종료율은 Fixed θ 20.5%, Dynamic θ 25.6%로, 기존 목표였던 60% 이상에는 도달하지 않았다. 대신 Dynamic θ는 전체 정확도 +0.6%p, 평균 추론 시간 -0.217ms, emergency_ramp 정확도 +2.4%p 개선을 보였다.

---

## 9. 주의사항

- 동적 θ가 고정 θ보다 나쁘게 나와도 당황하지 말 것. **왜 그런지 분석하는 게 더 중요함.**
- 튜닝은 val 데이터로. test 데이터는 최종 평가 때만 쓸 것.
- 시나리오별 분석은 전체 정확도보다 **특정 구간에서의 동작 차이**에 집중할 것.
- 모델 확정 후 파라미터 변경하지 말 것. 이후 경량화 단계에서 같은 모델 써야 함.
