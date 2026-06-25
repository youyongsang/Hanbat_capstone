# 유용상 방학 1단계 가이드라인
## 동적 Threshold 경량화 설계

> 담당자: 유용상  
> 기간: 방학 1~2주차  
> 목표: 동적 threshold 오버헤드 문제 해결을 위한 경량화 방식 설계  
> 완료 기준: 경량 동적 θ 설계 문서 완성, 구현 방향 팀원과 공유

---

## 1. 해야 할 일 순서

```
1. 기존 동적 θ 오버헤드 문제 재분석
2. 경량화 방식 3가지 설계
3. 구현 방향 확정
4. 김호중과 공유
```

---

## 2. 기존 문제 재분석

```
현재 코드 기준 동적 θ:
최근 channel_occupancy 중 마지막 2개 timestep 차이(delta)를 계산
→ delta = abs(occupancy[-1] - occupancy[-2])
→ delta > spike_threshold이면 기본 θ 유지
→ 안정 구간이면 θ₁, θ₂를 1.25배 높여 조기 종료를 더 쉽게 허용
→ 별도의 std/variance/sqrt 계산은 현재 코드에 없음
→ 실측 추론 시간 증가 (0.563ms → 0.582ms)
```

---

## 3. 경량화 방식 3가지

### 방법 1 — 현재 delta 방식 유지

```python
delta = abs(recent_window[-1] - recent_window[-2])
if delta > spike_threshold:
    theta_1, theta_2 = base_theta_1, base_theta_2
else:
    theta_1, theta_2 = base_theta_1 * 1.25, base_theta_2 * 1.25
```

### 방법 2 — 주기적 업데이트

```python
# 매 sample 또는 매 timestep 계산 → K번마다 한 번만
if timestep % 3 == 0:
    theta_1, theta_2 = compute_dynamic_threshold(recent_window)
# threshold 계산 횟수 감소
```

### 방법 3 — delta + 주기적 업데이트 조합 (권장)

```python
if timestep % 3 == 0:
    delta = abs(recent_window[-1] - recent_window[-2])
    theta_1, theta_2 = adjust_threshold(delta)
```

---

## 4. 완료 기준 체크리스트

- [x] 기존 동적 θ 오버헤드 원인 재분석 완료
- [ ] 경량화 방식 3가지 설계 문서 완성
- [ ] 구현 방향 확정 (방법 3 권장)
- [ ] 김호중에게 설계 방향 공유 완료

---

## 5. 주의사항

- 구현은 3~4주차에 진행. 이번 단계는 설계만.
- 방법 3 조합이 권장이지만 실험 결과에 따라 조정 가능.
