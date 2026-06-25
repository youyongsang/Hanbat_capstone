# 유용상 방학 2단계 가이드라인
## 노이즈 데이터 재실험 및 경량 동적 θ 구현

> 담당자: 유용상  
> 기간: 방학 3~4주차  
> 목표: 경량 동적 θ 구현 완료, 노이즈 추가 데이터로 재실험  
> 완료 기준: 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교표 완성

---

## 1. 해야 할 일 순서

```
1. 경량 동적 θ 구현 (방법 3: delta + 주기적 업데이트)
2. 장예나 노이즈 추가 시뮬레이터 데이터로 재실험
3. 호중 AP 원본 CSV를 예나가 가공한 실제 WiFi 데이터로 재실험
4. 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교 분석
```

---

## 2. 경량 동적 θ 구현

```python
import numpy as np

def compute_dynamic_threshold_lightweight(
    recent_window,
    current_theta_1,
    current_theta_2,
    base_theta_1=0.3,
    base_theta_2=0.6,
    timestep=0,
    update_interval=3
):
    """
    경량 동적 threshold 계산
    - 직전 timestep 대비 delta 기반
    - K 타임스텝마다 한 번만 계산
    """
    if timestep % update_interval != 0:
        return current_theta_1, current_theta_2

    # recent_window은 정규화된 channel_occupancy(0~1) 기준
    window = np.asarray(recent_window, dtype=np.float32)
    if len(window) < 2:
        return base_theta_1, base_theta_2

    delta = abs(float(window[-1] - window[-2]))
    SPIKE_THRESHOLD = 0.25
    MIN_THRESHOLD = 0.22

    if delta > SPIKE_THRESHOLD:
        theta_1 = base_theta_1
        theta_2 = base_theta_2
    else:
        theta_1 = base_theta_1 * 1.25
        theta_2 = base_theta_2 * 1.25

    theta_1 = max(theta_1, MIN_THRESHOLD)
    theta_2 = max(theta_2, MIN_THRESHOLD * 2)

    return theta_1, theta_2
```

---

## 3. 재실험 비교 항목

| 항목 | 고정 θ | 기존 동적 θ | 경량 동적 θ |
|---|---|---|---|
| 전체 정확도 | | | |
| 평균 추론 시간 | | | |
| Exit 1 종료율 | | | |
| 급변 구간 정확도 | | | |

### 데이터별 실험

| 데이터 | 비교 목적 |
|---|---|
| 기존 시뮬레이터 | 기존 결과와 동일한지 확인 |
| 노이즈 추가 시뮬레이터 | 동적 θ 효과가 더 잘 나오는지 확인 |
| 실제 WiFi 데이터 | 실제 환경 검증 |

---

## 4. 완료 기준 체크리스트

- [ ] 경량 동적 θ 구현 완료
- [ ] 노이즈 시뮬레이터 데이터 재실험 완료
- [ ] 실제 WiFi 데이터 재실험 완료
- [ ] 고정 θ vs 기존 동적 θ vs 경량 동적 θ 비교표 완성
- [ ] 김호중에게 결과 전달 완료

---

## 5. 주의사항

- 경량 동적 θ가 기존보다 나쁘게 나와도 괜찮아. 결과 그대로 분석하면 돼.
- 급변 구간 분리 분석을 꼭 할 것. 전체 정확도보다 급변 구간 차이가 더 중요함.
