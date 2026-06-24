# 장예나 방학 1단계 가이드라인
## AP 원본 CSV 후처리 기준 설계

> 담당자: 장예나  
> 기간: 방학 1~2주차  
> 목표: 호중이 생성할 AP 원본 CSV를 우리 실험 입력 형식으로 바꾸는 기준 수립  
> 완료 기준: 원본 CSV → 모델 입력 CSV 변환 규칙 확정

---

## 1. 해야 할 일 순서

```
1. 호중이 생성할 AP 원본 CSV 컬럼 확인
2. 우리 실험용 피처 매핑 규칙 정의
3. label/scenario 유지 기준 정리
4. 10-step window 변환 기준 설계
```

---

## 2. 호중 원본 CSV 기준

호중이 AP 장비에서 생성해 전달할 원본 CSV는 아래 형식을 기준으로 한다.

```
timestamp, throughput_mbps, channel_occupancy, packet_loss, latency, label, scenario
```

| 시나리오 | 레이블 | 원본 데이터 기준 |
|---|---|---|
| 정상 | 0 | 낮은 부하 측정 구간 |
| 혼잡 경고 | 1 | 중간 부하 측정 구간 |
| 혼잡 | 2 | 고부하 측정 구간 |
| 심각 | 3 | 대용량 전송 포함 심각 혼잡 구간 |

---

## 3. 피처 매핑 기준

| 원본 컬럼 | 최종 피처 | 처리 |
|---|---|---|
| `throughput_mbps` | `rps` | 0~1000 범위로 스케일링 |
| `channel_occupancy` | `channel_occupancy` | 0~100 범위 확인 후 정규화 |
| `packet_loss` | `packet_loss` | 0~30 범위 확인 후 정규화 |
| `latency` | `latency` | 0~500 범위 확인 후 정규화 |

> RPS는 실측 환경에서 직접 수집하지 않고, 호중이 측정한 `throughput_mbps`를 실험용 `rps` 피처로 매핑한다.

---

## 4. 완료 기준 체크리스트

- [ ] 호중 원본 CSV 컬럼 기준 확인
- [ ] `throughput_mbps` → `rps` 매핑 규칙 확정
- [ ] 최종 windowed CSV 컬럼 정의 완료
- [ ] 김호중에게 원본 CSV 전달 형식 요청 완료

---

## 5. 주의사항

- AP 장비 세팅과 원본 CSV 생성은 김호중 담당.
- 장예나는 원본 CSV를 받아 우리 실험 입력 shape `(N, 10, 4)`로 바꾸는 데 집중.
