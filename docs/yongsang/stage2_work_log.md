# 유용상 Stage 2 작업 설명

## 작업 목표

예나 Stage 2 DataLoader를 연결해 Early Exit LSTM을 실제 데이터로 다시 학습하고, 고정 threshold 버전과 동적 threshold 버전을 같은 테스트셋에서 비교했다.

## 사용 데이터

| 항목 | 값 |
|---|---|
| 데이터 경로 | `project/data/real/` |
| 입력 shape | `(N, 10, 4)` |
| train | 700 samples |
| val | 150 samples |
| test | 150 samples |
| feature | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| label | `0`, `1`, `2`, `3` |

## 구현 파일

| 파일 | 역할 |
|---|---|
| `project/models/early_exit_lstm.py` | 동적 threshold 계산, spike 감지, `set_threshold()` 인터페이스 추가 |
| `project/scripts/train_early_exit.py` | 실제 데이터 학습 및 fixed/dynamic checkpoint 저장 |
| `project/scripts/evaluate_early_exit.py` | 고정 θ와 동적 θ 비교 평가, label/시나리오별 분석 |
| `project/results/yongsang/early_exit_stage2_comparison_report.txt` | Stage 2 비교 평가 결과 |
| `project/checkpoints/early_exit_fixed.pth` | 고정 θ 비교용 checkpoint |
| `project/checkpoints/early_exit_dynamic.pth` | 동적 θ 비교용 checkpoint |

## 동적 Threshold 설계

동적 threshold는 추론 시 각 sample의 최근 채널 점유율 중 마지막 2개 timestep의 차이(delta)를 보고 조정한다. 데이터는 정규화되어 있으므로 `spike_threshold`도 `0~1` 스케일로 사용했다.

| 파라미터 | 값 | 의미 |
|---|---:|---|
| `base_theta_1` | 0.3 | Exit 1 기본 threshold |
| `base_theta_2` | 0.6 | Exit 2 기본 threshold |
| `min_threshold` | 0.22 | threshold 최솟값 |
| `recent_steps` | 5 | 최근 점유율 확인 범위 |
| `spike_threshold` | 0.25 | 직전 timestep 대비 급변 감지 기준 |

동작 방향은 아래와 같다.

```text
delta > spike_threshold -> 기본 threshold 유지 -> 과도한 조기 종료 방지
delta <= spike_threshold -> threshold 1.25배 상향 -> 빠른 종료 우선
```

## 실행 방법

학습:

```bash
python project/scripts/train_early_exit.py --epochs 50
```

평가:

```bash
python project/scripts/evaluate_early_exit.py --output project/results/yongsang/early_exit_stage2_comparison_report.txt
```

## 학습 결과

```text
Best model saved: C:\Capstone-Design\project\checkpoints\early_exit_lstm_best.pth
Fixed-threshold checkpoint saved: C:\Capstone-Design\project\checkpoints\early_exit_fixed.pth
Dynamic-threshold checkpoint saved: C:\Capstone-Design\project\checkpoints\early_exit_dynamic.pth
Best Val Accuracy: 94.7%
```

학습 loss는 초반 `1.195`에서 후반 `0.185` 수준까지 감소했고, val accuracy는 최고 `94.7%`까지 도달했다.

## 비교 평가 결과

| 항목 | 고정 θ | 동적 θ | 변화 |
|---|---:|---:|---:|
| Test Accuracy | 95.3% | 96.0% | +0.7%p |
| Exit 1 Rate | 68.0% | 72.0% | +4.0%p |
| Exit 2 Rate | 22.0% | 27.3% | +5.3%p |
| Exit 3 Rate | 10.0% | 0.7% | -9.3%p |
| Avg Inference Time | 3.040ms | 2.587ms | -0.453ms |

## 시나리오별 정확도

| 시나리오 | 고정 θ | 동적 θ |
|---|---:|---:|
| `emergency_ramp` | 91.4% | 91.4% |
| `imbalanced_ap_load` | 87.5% | 87.5% |
| `lunch_restart` | 100.0% | 100.0% |
| `startup_surge` | 94.4% | 97.2% |

동적 θ는 `startup_surge`에서 정확도를 높였고, 전체적으로 Exit 3까지 가는 비율을 크게 줄였다.

## 해석

이번 결과에서는 동적 θ가 고정 θ보다 더 빠르고 약간 더 정확했다. 특히 안정적인 구간에서 threshold를 높여 Exit 1 종료율을 늘렸고, 급변 구간에서는 threshold를 낮추는 로직이 적용되도록 설계했다.

다만 Exit별 정확도는 `Exit 1 < Exit 2 < Exit 3` 형태로 나오지 않았다. 현재 데이터에서는 쉬운 sample이 Exit 1에서 먼저 종료되고, 애매한 sample이 뒤쪽 Exit로 밀리기 때문에 Exit 1 정확도가 더 높게 관측된다. 이는 Early Exit 모델에서 자연스럽게 나타날 수 있는 결과이며, Stage 3에서는 exit별 sample 난이도와 confidence calibration을 함께 보는 것이 좋다.

## 김호중 비교 실험 연결

김호중 비교 실험에서는 아래처럼 불러 사용할 수 있다.

```python
from models.early_exit_lstm import EarlyExitLSTM

model_fixed = EarlyExitLSTM()
model_fixed.set_threshold(theta_1=0.3, theta_2=0.6, dynamic=False)

model_dynamic = EarlyExitLSTM()
model_dynamic.set_threshold(theta_1=0.3, theta_2=0.6, dynamic=True)
```

checkpoint 경로는 아래를 사용한다.

```text
project/checkpoints/early_exit_fixed.pth
project/checkpoints/early_exit_dynamic.pth
```
