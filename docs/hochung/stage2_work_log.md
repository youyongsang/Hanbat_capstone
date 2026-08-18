# 김호중 Stage 2 작업 설명

## 작업 목표

실제 무선 데이터(장예나 DataLoader 기반)를 활용하여 채널 혼잡 분류 결과를 실제 채널 전환 로직과 연동하고, 4가지 무선 제어 알고리즘의 성능 및 추론 지연 시간을 동일 환경(CPU 기준)에서 비교 검증했다.

또한 유용상 팀원의 Early Exit 모델(고정형 / 동적형)과 기존 Baseline LSTM 및 현행 임계값 방식 간의 Ablation 비교 실험 프레임워크를 구축했다.

---

# 사용 데이터

| 항목 | 값 |
|---|---|
| 데이터 경로 | `project/data/real/test.csv` |
| 입력 shape | `(N, 10, 4)` |
| feature | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| label | `0`, `1`, `2`, `3` |
| 테스트 환경 | CPU 강제 실행 (에지 디바이스 환경 가정) |

---

# 비교 대상 알고리즘

| 구분 | 설명 |
|---|---|
| Baseline ① | 현행 임계값 기반 규칙 제어 방식 |
| Baseline ② | Standard Full Inference 3-Layer LSTM |
| Baseline ③ | 고정형 Early Exit 모델 |
| Baseline ④ | 제안 동적 Early Exit 모델 |

---

# 구현 파일

| 파일 | 역할 |
|---|---|
| `project/experiments/channel_optimizer.py` | 분류 결과 기반 채널 제어 전략 인터페이스 |
| `project/experiments/compare_baselines.py` | 4대 알고리즘 종합 비교 실험 엔진 |
| `project/models/baseline_lstm.py` | Full Inference Baseline LSTM |
| `project/models/early_exit_lstm.py` | Early Exit 기반 추론 모델 |
| `project/checkpoints/baseline_lstm_best.pth` | Full LSTM 체크포인트 |
| `project/checkpoints/early_exit_fixed.pth` | 고정형 Early Exit 체크포인트 |
| `project/checkpoints/early_exit_dynamic.pth` | 동적 Early Exit 체크포인트 |
| `project/results/comparison_summary.txt` | 종합 비교 실험 결과 리포트 |

---

# 채널 제어 정책

```text
Label 0 (정상)
  -> 현재 채널 유지 (keep)

Label 1 (혼잡 경고)
  -> 인접 채널 모니터링 및 백업 채널 준비 (monitor)

Label 2 (혼잡)
  -> 덜 혼잡한 채널로 즉시 전환 (switch)

Label 3 (심각 혼잡)
  -> 즉시 채널 전환 + 5GHz 대역 이동 (emergency)
실험 환경

실측 신뢰도를 확보하기 위해 GPU 가속을 사용하지 않고 CPU 환경에서 추론 시간을 정밀 측정했다.

Environment:
- CPU Only
- 동일 테스트셋 사용
- 동일 입력 Shape 사용
- 동일 평가 루프 사용
- ms 단위 추론 시간 측정
실행 방법

프로젝트 루트에서 아래 명령으로 종합 비교 실험을 수행한다.

python project/experiments/compare_baselines.py

실험 결과는 콘솔 출력과 함께 아래 txt 파일에 저장된다.

project/results/comparison_summary.txt
종합 비교 실험 결과
=== 4대 무선 제어 알고리즘 종합 비교 실험 결과 보고서 ===

테스트 데이터 경로:
C:\Users\User\Hanbat_capstone\project\data\real\test.csv

--------------------------------------------------

① 현행 임계값 제어 방식
   정확도: 60.0%
   지연: 0.050ms

② Baseline LSTM 풀 추론 방식
   정확도: 97.3%
   지연: 0.687ms

③ 고정형 Early Exit 방식
   정확도: 93.9%
   지연: 0.504ms
   Exit 분포:
   - Exit1: 58.3%
   - Exit2: 26.7%
   - Exit3: 15.0%

④ 제안 동적 Early Exit 방식
   정확도: 94.8%
   지연: 0.426ms
   Exit 분포:
   - Exit1: 68.0%
   - Exit2: 21.3%
   - Exit3: 10.7%

--------------------------------------------------
```

## 결과 해석

현행 임계값 기반 제어 방식(Baseline ①)은 추론 지연 시간이 매우 짧았지만 시계열 패턴을 고려하지 못하기 때문에 정확도가 60.0% 수준으로 크게 저하되었다.
```
Threshold Baseline:
- Accuracy: 60.0%
- Latency : 0.050ms
```
반면 Standard Full LSTM(Baseline ②)은 가장 높은 정확도를 기록했지만 모든 샘플이 마지막 Layer까지 Full Inference를 수행하므로 상대적으로 높은 연산 지연이 발생했다.
```
Full LSTM:
- Accuracy: 97.3%
- Latency : 0.687ms
```
고정형 Early Exit(Baseline ③)은 일부 샘플을 조기 종료함으로써 연산량을 줄였으며 평균 추론 시간을 단축했다.
```
EE_Fixed Exit Distribution:
- Exit1: 58.3%
- Exit2: 26.7%
- Exit3: 15.0%
```
최종 제안 방식인 동적 Early Exit(Baseline ④)은 실시간 점유율 변동성 및 Spike 기반 동적 임계값 조정을 통해 안정 구간에서 더 적극적으로 조기 종료를 수행했다.
```
EE_Dynamic Exit Distribution:
- Exit1: 68.0%
- Exit2: 21.3%
- Exit3: 10.7%
```
그 결과 정확도는 94.8% 수준을 유지하면서도 Full LSTM 대비 약 38% 수준의 연산량 절감 효과를 달성했다.
```
EE_Dynamic:
- Accuracy: 94.8%
- Latency : 0.426ms
```
이 결과는 제안한 동적 Early Exit 구조가 에지 환경에서 정확도와 연산 효율성 간의 균형을 효과적으로 달성할 수 있음을 보여준다.
