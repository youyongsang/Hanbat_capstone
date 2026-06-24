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
기존 동적 θ:
매 타임스텝마다 np.std(recent_window) 계산
→ N개 값의 분산 계산 + 제곱근 연산
→ 오버헤드 > Exit 1 종료율 향상 이득
→ 실측 추론 시간 증가 (0.563ms → 0.582ms)
```

---

## 3. 경량화 방식 3가지

### 방법 1 — max-min 범위 (제일 단순)

```python
# 기존
variance = np.std(recent_window)

# 개선 → 제곱근 연산 없애서 빨라짐
variance = max(recent_window) - min(recent_window)
```

### 방법 2 — 주기적 업데이트

```python
# 매 타임스텝 계산 → K 타임스텝마다 한 번만
if timestep % 3 == 0:
    theta_1, theta_2 = compute_dynamic_threshold(recent_window)
# 계산 횟수 1/3로 줄어듦
```

### 방법 3 — 1+2 조합 (권장)

```python
if timestep % 3 == 0:
    variance = max(recent_window) - min(recent_window)
    theta_1, theta_2 = adjust_threshold(variance)
```

---

## 4. 완료 기준 체크리스트

- [ ] 기존 동적 θ 오버헤드 원인 재분석 완료
- [ ] 경량화 방식 3가지 설계 문서 완성
- [ ] 구현 방향 확정 (방법 3 권장)
- [ ] 김호중에게 설계 방향 공유 완료

---

## 5. 주의사항

- 구현은 3~4주차에 진행. 이번 단계는 설계만.
- 방법 3 조합이 권장이지만 실험 결과에 따라 조정 가능.
