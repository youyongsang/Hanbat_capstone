# 유용상 Stage 1 작업 설명

## 작업 목표

Early Exit LSTM을 예나 실제 데이터 `project/data/real/` 기준으로 학습하고, 고정 threshold 추론에서 Exit별 정확도와 종료율이 출력되는지 확인했다.

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
| `project/models/early_exit_lstm.py` | Early Exit LSTM 모델, entropy 계산, multi-exit loss, inference mode |
| `project/utils/dataloader.py` | CSV를 `(batch, 10, 4)` tensor와 label로 로딩 |
| `project/utils/metrics.py` | Exit별 통계와 퍼센트 출력 유틸 |
| `project/scripts/train_early_exit.py` | Multi-exit loss 기반 학습 스크립트 |
| `project/scripts/evaluate_early_exit.py` | Exit별 accuracy, exit rate, 평균 추론 시간 평가 스크립트 |
| `project/checkpoints/early_exit_lstm_best.pth` | 학습된 best checkpoint |
| `project/results/early_exit_eval_report.txt` | 평가 결과 텍스트 리포트 |

## 모델 구조

```text
입력: (batch, 10, 4)
  -> LSTM Layer 1 -> Exit Classifier 1
  -> LSTM Layer 2 -> Exit Classifier 2
  -> LSTM Layer 3 -> Exit Classifier 3
```

학습 시에는 세 Exit의 logits를 모두 계산하고, 아래 가중치로 multi-exit loss를 계산했다.

```text
loss = 0.3 * loss_exit1 + 0.3 * loss_exit2 + 0.4 * loss_exit3
```

추론 시에는 entropy 기준으로 조기 종료한다.

| threshold | 값 |
|---|---:|
| `theta_1` | 0.3 |
| `theta_2` | 0.6 |

## 실행 방법

프로젝트 루트에서 아래 명령으로 학습한다.

```bash
python project/scripts/train_early_exit.py --epochs 50
```

평가는 아래 명령으로 실행한다.

```bash
python project/scripts/evaluate_early_exit.py
```

평가 스크립트는 콘솔 출력과 함께 기본적으로 아래 txt 파일에 결과를 저장한다.

```text
project/results/early_exit_eval_report.txt
```

## 학습 결과

```text
Best model saved: C:\Capstone-Design\project\checkpoints\early_exit_lstm_best.pth
Best Val Accuracy: 94.7%
```

## 평가 결과

```text
Early Exit LSTM Evaluation Report
Data Directory: C:\Capstone-Design\project\data\real
Checkpoint: C:\Capstone-Design\project\checkpoints\early_exit_lstm_best.pth
Test Accuracy: 96.0%
Exit 1 Accuracy: 100.0% | Exit Rate: 50.7%
Exit 2 Accuracy: 93.4% | Exit Rate: 40.7%
Exit 3 Accuracy: 84.6% | Exit Rate: 8.7%
Label 0 (정상) Accuracy: 100.0%
Label 1 (혼잡 경고) Accuracy: 96.7%
Label 2 (혼잡) Accuracy: 78.3%
Label 3 (심각) Accuracy: 100.0%
Average Inference Time: 0.087ms
```

## 결과 해석

전체 테스트 정확도는 `96.0%`로 Stage 1 동작 확인 기준을 만족했다.

Exit별 종료율은 `Exit 1: 50.7%`, `Exit 2: 40.7%`, `Exit 3: 8.7%`로 나타났다. 즉 대부분의 sample이 마지막 LSTM까지 가지 않고 중간 Exit에서 종료되어 Early Exit 구조의 계산 절감 가능성을 확인했다.

## 체크리스트 반영

`docs/yongsang/guideline_yongsang_stage1.md`에서 실행으로 확인 가능한 항목은 체크 완료했다.

김호중과 직접 확인해야 하는 `hidden_size`, 입력 shape 확인 항목은 아직 미체크로 남겼다.
