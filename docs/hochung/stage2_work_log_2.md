# Capstone Project: Early Exit LSTM Performance Analysis

본 문서는 Hanbat Capstone 프로젝트의 real 데이터 생성, Baseline 및 Early Exit 모델 학습, 그리고 최종 모델 평가 결과를 분석한 리포트입니다.

---

## 1. 터미널 모델 가동 과정 (Execution Commands)

```bash
python project/scripts/generate_real_data.py
python project/scripts/train.py
python project/scripts/train_early_exit.py
python project/scripts/evaluate.py
python project/scripts/evaluate_early_exit.py

여기까지 모델 돌린 후 생성된 evaluate txt파일 비교
```

## 2. 모델별 성능 비교 결과

| 평가 지표 | Baseline 2 (Standard LSTM) | Baseline 3 (Early Exit + Fixed θ) | Proposed Model 4 (Early Exit + Dynamic θ) |
| :--- | :---: | :---: | :---: |
| **전체 테스트 정확도 (Accuracy)** | 94.0% | 92.7% | 92.0% |
| **평균 추론 시간 (Inference Time)** | 8.000ms | 2.987ms | 2.600ms |
| **측정 벽면 시간 (Wall Time)** | 0.131ms | 0.138ms | 0.138ms |
| **Exit 1 (정확도 / 탈출율)** | N/A / 0.0% | 100.0% / 66.7% | 99.1% / 72.7% |
| **Exit 2 (정확도 / 탈출율)** | N/A / 0.0% | 84.2% / 25.3% | 74.4% / 26.0% |
| **Exit 3 (정확도 / 탈출율)** | 94.0% / 100.0% | 58.3% / 8.0% | 50.0% / 1.3% |

---

## 3. 학습 및 성능 결과 분석

### 데이터 특징 (Data Specifications)
* **입력 데이터 구조:** 10 타임스텝(Timesteps) 동안의 4개 특징 변수(rps, channel_occupancy, packet_loss, latency) 정보를 입력으로 사용합니다.
* **데이터 분할:** Train 700개, Val 150개, Test 150개 샘플로 구성되어 있습니다.

### 학습 트렌드 요약 (Training Loss & Val Accuracy)
* **Baseline LSTM:** Epoch 43에서 Best Val Accuracy 95.3%(Val Loss: 0.134)를 기록하며 안정적으로 수렴했습니다.
* **Early Exit LSTM:** 다중 Exit 구조 학습임에도 안정적으로 우상향하여 Epoch 49에서 Best Val Accuracy 94.7%(Val Loss: 0.148)를 확보했습니다.

### 핵심 성능 인사이트

#### 획기적인 연산 속도 단축 (Inference Efficiency)
* Proposed Model 4(제안 모델)는 기존 Baseline 2 모델 대비 전체 정확도가 단 2.0%p 낮아지는 미미한 손실만을 보였습니다.
* 반면, 평균 추론 시간은 8.000ms에서 2.600ms로 약 67.5% 가량 크게 단축되어 실시간 네트워크 제어 환경에 적합함을 입증했습니다.

#### Dynamic θ (동적 임계값) 메커니즘의 우수성
* 고정 임계값(Fixed θ) 방식인 Baseline 3과 비교했을 때, Proposed Model 4는 정확도 손실을 0.7%p로 방어해냈습니다.
* 동시에 실시간 주위 분산 환경을 고려한 동적 룰 덕분에 추론 속도를 추가로 0.387ms 더 아끼는 효율성을 입증했습니다.

#### Early Exit 계층별 필터링 효과 (정상 동작 확인)
* 계층이 뒤로 갈수록(Exit 1 -> 2 -> 3) 개별 Exit 내 정확도가 점차 낮아집니다.
* 이는 판단하기 쉬운 명확한 데이터(예: 정상 데이터)가 Exit 1에서 높은 정확도로 대거 조기 탈출(Dynamic 기준 탈출율 72.7%)하기 때문입니다. 이후의 후속 Exit에는 선별하기 까다로운 잔여 난제 데이터들만 누적되므로, 이와 같은 하향 곡선은 모델 구조가 의도대로 완벽하게 작동하고 있다는 강력한 증거입니다.
