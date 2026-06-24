# 방학 중 활동 전체 정리

## 1. 방학 활동 목표

방학 중 작업의 핵심 목표는 기존 시뮬레이터 및 외부 데이터 기반 실험을 실제 GL.iNet AP 환경까지 확장하고, 실제 수집 데이터와 실시간 추론 흐름을 우리 모델 입력 형식에 맞춰 검증하는 것이다.

전체 흐름은 다음과 같다.

```text
GL.iNet AP 실측 데이터 수집
→ AP 원본 CSV 생성
→ 실험용 피처 변환 및 windowed CSV 생성
→ 모델 재실험 및 동적 threshold 경량화 검증
→ SOTA 비교 및 연산량 절감 분석
→ 실시간 추론 파이프라인 구현
→ 보고서 및 발표 자료 완성
```

---

## 2. 팀원별 핵심 역할

| 팀원 | 핵심 역할 | 주요 산출물 |
|---|---|---|
| 김호중 | GL.iNet AP 세팅, AP 원본 CSV 생성, ONNX/Pi 배포, 실시간 추론 파이프라인 구현 | `raw_measurements.csv`, `real_wifi_inference_results.csv`, `onnx_comparison.csv`, `realtime_inference_results.csv` |
| 장예나 | AP 원본 CSV 피처 변환, windowed CSV 생성, scaler 기준 관리, 결과 시각화 | `train.csv`, `val.csv`, `test.csv`, `scaler_params.json`, 최종 그래프 |
| 유용상 | Early Exit LSTM/threshold 분석, 경량 동적 threshold 구현, 실제 WiFi 데이터 재검증, SOTA 비교 | `sota_comparison.csv`, 모델 설계 보고서 섹션 |

---

## 3. 데이터 생성 및 변환 흐름

### 3.1 호중: AP 원본 CSV 생성

호중은 기존에 AP 테스트를 진행한 환경을 기준으로 GL.iNet AP와 Raspberry Pi를 연결하고, 실제 WiFi 트래픽을 발생시켜 원본 CSV를 생성한다.

원본 CSV 기준:

```text
timestamp, throughput_mbps, channel_occupancy, packet_loss, latency, label, scenario
```

이 파일은 모델에 바로 넣는 최종 입력이 아니라, 예나가 후처리할 raw 데이터다.

### 3.2 예나: 실험용 피처 변환

예나는 호중이 만든 원본 CSV를 받아 기존 학습/평가/추론 코드가 요구하는 형식으로 변환한다.

최종 모델 입력 CSV 기준:

```text
sample_id, timestep, rps, channel_occupancy, packet_loss, latency, label, scenario
```

주요 변환:

| 원본 컬럼 | 최종 피처 | 처리 |
|---|---|---|
| `throughput_mbps` | `rps` | 0~1000 범위로 스케일링 |
| `channel_occupancy` | `channel_occupancy` | Min-Max 정규화 |
| `packet_loss` | `packet_loss` | Min-Max 정규화 |
| `latency` | `latency` | Min-Max 정규화 |

예나는 오프라인 CSV 변환과 실시간 추론이 같은 기준을 쓰도록 `scaler_params.json`을 관리한다.

`scaler_params.json`에는 최소한 아래 정보가 포함되어야 한다.

```json
{
  "throughput_mbps": {"min": 0.0, "max": 200.0},
  "rps": {"min": 0.0, "max": 1000.0},
  "channel_occupancy": {"min": 0.0, "max": 100.0},
  "packet_loss": {"min": 0.0, "max": 30.0},
  "latency": {"min": 0.0, "max": 500.0}
}
```

---

## 4. 실시간 추론 원리

기존 방식은 이미 만들어진 `test.csv`에서 `sample_id`별 10개 timestep을 읽어 `(1, 10, 4)` 입력을 만들었다.

실시간 추론은 CSV 파일을 미리 읽는 대신, AP에서 들어오는 raw 지표를 매 시점마다 변환해서 최근 10개 timestep을 버퍼에 쌓는다.

```text
AP raw metric 수집
→ throughput_mbps를 rps로 변환
→ scaler_params.json 기준 정규화
→ 최근 10개 timestep 버퍼 유지
→ (1, 10, 4) 입력 생성
→ ONNX 모델 추론
→ label 0~3 출력
→ 채널 유지/전환 방향 제시
```

실시간 추론에서도 모델 입력 피처 순서는 반드시 아래와 같다.

```text
rps, channel_occupancy, packet_loss, latency
```

즉, 실시간 추론은 CSV 평가와 다른 모델을 쓰는 것이 아니라, **CSV 로딩 부분을 AP 실시간 수집 및 변환 함수로 바꾼 구조**다.

---

## 5. 주차별 활동 계획

| 기간 | 김호중 | 장예나 | 유용상 |
|---|---|---|---|
| 1~2주차 | GL.iNet AP 세팅, Pi 연동, 원본 CSV 수집 스크립트 준비 | AP 원본 CSV 후처리 기준 설계, 피처 매핑/윈도우 기준 정리 | 동적 threshold 오버헤드 분석, 경량화 방식 설계 |
| 3~4주차 | AP 원본 CSV 생성 및 예나에게 전달, ONNX 비교 준비 | 원본 CSV를 `rps` 기반 windowed CSV로 변환, `scaler_params.json` 저장, 노이즈 시뮬레이터 추가 | 경량 동적 threshold 구현, 노이즈/실측 데이터 재실험 |
| 5주차 | SOTA 비교 실험 프레임워크 추가 | SOTA 비교 결과 CSV 수집 및 정리 | Early Exit 관련 SOTA 조사 및 비교 실험 |
| 6주차 | 경량 동적 threshold 비교 실험 | 최종 결과 그래프 생성 | 실제 WiFi 데이터 재검증 및 시뮬레이터 결과 비교 |
| 7주차 | 실시간 추론 파이프라인 구현 | 실시간 추론 변환 기준 검증 및 결과 CSV 정리 | 실시간 추론 정확도 검증 |
| 8주차 | 배포/ONNX/실시간 데모 보고서 및 발표 자료 완성 | 데이터/실험 결과 보고서 및 발표 자료 완성 | 모델 설계/SOTA/threshold 보고서 및 발표 자료 완성 |

---

## 6. 주요 산출물

| 산출물 | 담당 | 설명 |
|---|---|---|
| `project/data/real_wifi/raw_measurements.csv` | 김호중 | AP 장비에서 수집한 원본 실측 CSV |
| `project/data/real_wifi/train.csv` | 장예나 | 모델 학습용 windowed CSV |
| `project/data/real_wifi/val.csv` | 장예나 | 모델 검증용 windowed CSV |
| `project/data/real_wifi/test.csv` | 장예나 | 모델 평가 및 Pi 추론용 windowed CSV |
| `project/data/real_wifi/scaler_params.json` | 장예나 | 오프라인/실시간 공통 피처 변환 기준 |
| `project/results/hojung/real_wifi_inference_results.csv` | 김호중 | 실제 WiFi 데이터 기반 Pi/ONNX 추론 결과 |
| `project/results/hojung/onnx_comparison.csv` | 김호중 | 원본 ONNX와 양자화 ONNX 비교 결과 |
| `project/results/hojung/dynamic_threshold_comparison.csv` | 김호중·유용상 | 고정 θ, 기존 동적 θ, 경량 동적 θ 비교 |
| `project/results/hojung/comparison_summary_final.csv` | 김호중 | 기존 4개 방식과 SOTA 모델 비교 결과 |
| `project/results/hojung/computation_comparison.csv` | 김호중 | SOTA 대비 연산량 또는 추론 시간 절감 비교 |
| `project/results/sota_comparison.csv` | 유용상 | SOTA 모델과 제안 모델 비교 |
| `project/results/hojung/realtime_inference_results.csv` | 김호중 | 실시간 추론 결과 로그 |
| 최종 그래프 | 장예나 | 발표 및 보고서용 시각화 자료 |

---

## 7. 보고서 및 발표 정리 범위

| 팀원 | 보고서/발표 담당 내용 |
|---|---|
| 김호중 | GL.iNet AP 환경, ONNX 변환, INT8 경량화, Pi 배포, 실시간 추론 파이프라인 |
| 장예나 | 데이터 수집 구조, AP 원본 CSV 피처 변환, 데이터셋 구성, 시뮬레이터 vs 실제 데이터 분포, 결과 그래프 |
| 유용상 | Early Exit LSTM 구조, 고정/동적 threshold, 경량 동적 threshold, SOTA 비교, 모델 한계 및 개선 방향 |

---

## 8. 주의사항

- AP 원본 CSV 생성은 김호중 담당이고, 모델 입력용 피처 변환은 예나 담당이다.
- 오프라인 CSV 변환과 실시간 추론은 반드시 같은 `scaler_params.json`을 사용한다.
- 모델 입력 shape은 항상 `(N, 10, 4)` 또는 실시간 단일 추론 기준 `(1, 10, 4)`로 유지한다.
- 피처 순서는 `rps, channel_occupancy, packet_loss, latency`로 고정한다.
- 실측 데이터 결과와 시뮬레이터 결과는 보고서에서 분리해서 표기한다.
- 동적 threshold가 항상 빠르거나 정확도가 높게 나올 필요는 없다. 결과가 불리해도 한계와 개선 방향으로 정리한다.
