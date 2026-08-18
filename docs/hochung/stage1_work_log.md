# 김호중 Stage 1 작업 설명

## 작업 목표

기본 LSTM(Baseline LSTM) 백본 모델을 예나 실제 데이터 `project/data/real/` 기준으로 학습하고, 조기 종료가 없는 표준 풀 추론 상태에서 최종 가중치와 성능 지표가 출력되는지 확인했다.

## 사용 데이터

| 항목 | 값 |
|---|---|
| 데이터 경로 | `project/data/real/` |
| 입력 shape | `(N, 10, 4)` |
| feature | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| label | `0`, `1`, `2`, `3` |
| train | 700 samples |
| val | 150 samples |
| test | 150 samples |

## 구현 파일

| 파일 | 역할 |
|---|---|
| `project/models/baseline_lstm.py` | 표준 풀 추론용 기본 3-Layer LSTM 백본 모델 구조 정의 |
| `project/utils/dataloader.py` | CSV를 `(batch, 10, 4)` tensor와 label로 로딩 |
| `project/utils/metrics.py` | 기본 분류 메트릭, Precision/Recall, 시나리오별 통계 산출 유틸 |
| `project/scripts/train_baseline.py` | 단일 CrossEntropyLoss 기반 학습 스크립트 |
| `project/scripts/evaluate_baseline.py` | Ablation Sync용 전체 정확도, Exit별 성능, 시나리오별 평가 스크립트 |
| `project/checkpoints/baseline_lstm_best.pth` | 학습된 best checkpoint |
| `project/results/baseline_lstm_eval_report.txt` | 평가 결과 텍스트 리포트 |

## 모델 구조

```text
입력: (batch, 10, 4)
  -> LSTM Layer 1 (Avg Time: 2.0ms)
  -> LSTM Layer 2 (Avg Time: 4.0ms)
  -> LSTM Layer 3 (Avg Time: 8.0ms) -> 최종 풀 추론 분류 수행
```

학습 및 추론 시에는 별도의 중간 조기 종료 분기 없이 3개의 Layer 전체를 완전히 통과하는 풀 추론 연산을 수행하며, 아래 loss 함수로 단일 역전파를 수행했다.
```
loss = CrossEntropyLoss(logits, labels)
```

# 실행 방법

프로젝트 루트에서 아래 명령으로 학습한다.

```bash
python project/scripts/train_baseline.py --epochs 50
```

평가는 아래 명령으로 실행한다.

```bash
python project/scripts/evaluate_baseline.py
```

평가 스크립트는 콘솔 출력과 함께 기본적으로 아래 txt 파일에 결과를 저장한다.

```text
project/results/baseline_lstm_eval_report.txt
```

---

# 학습 결과

```text
Best model saved:
C:\Users\User\Hanbat_capstone\project\checkpoints\baseline_lstm_best.pth

Best Val Accuracy: 92.0%
```

---

# 평가 결과

```text
Baseline LSTM Stage 1 Evaluation Report (For Ablation Sync)

Data Directory:
C:\Users\User\Hanbat_capstone\project\data\real

Checkpoint:
C:\Users\User\Hanbat_capstone\project\checkpoints\baseline_lstm_best.pth


=== Baseline 2: Standard 3-Layer LSTM ===

Test Accuracy: 86.0%

Exit별 성능:
  Exit 1 | Accuracy: N/A   | Exit Rate: 0.0%   | Avg Time: 2.0ms
  Exit 2 | Accuracy: N/A   | Exit Rate: 0.0%   | Avg Time: 4.0ms
  Exit 3 | Accuracy: 86.0% | Exit Rate: 100.0% | Avg Time: 8.0ms

Overall Avg Inference Time: 8.000ms
Measured Wall Time: 0.131ms


Label별 정확도:
  Label 0 (정상):
    Precision: 0.90
    Recall: 1.00

  Label 1 (혼잡 경고):
    Precision: 0.80
    Recall: 0.40

  Label 2 (혼잡):
    Precision: 0.71
    Recall: 0.87

  Label 3 (심각):
    Precision: 1.00
    Recall: 1.00


시나리오별 정확도:
  emergency_ramp:      91.4%
  imbalanced_ap_load:  75.0%
  lunch_restart:       85.7%
  startup_surge:       86.1%
```

---

# 결과 해석

전체 테스트 정확도는 86.0%로 Stage 1 동작 확인 기준을 만족했다.

Exit별 종료율을 살펴보면 중간 레이어에서 연산을 멈추지 못하므로 다음과 같은 결과가 나타났다.

- Exit 1: 0.0%
- Exit 2: 0.0%
- Exit 3: 100.0%

즉, 모든 샘플이 마지막 레이어까지 Full Inference를 수행한다.

따라서 전체 평균 추론 시간은 3개 층의 누적 오버헤드가 모두 반영된 아래 값으로 측정되었다.

```text
Overall Avg Inference Time: 8.000ms
```

이 결과는 Stage 2 Early Exit 모델과의 비교를 위한 핵심 대조군(Baseline) 데이터로 활용된다.
