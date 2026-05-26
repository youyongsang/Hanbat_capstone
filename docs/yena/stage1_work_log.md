# 장예나 Stage 1 작업 설명

## 작업 목표

공장 운영 패턴을 반영한 트래픽 시계열 데이터를 생성하고, LSTM 입력 형태인 `(N, 10, 4)` 구조로 전처리하여 `project/data/real/` 경로에 학습/검증/테스트 CSV를 저장했다.

## 구현 파일

| 파일 | 역할 |
|---|---|
| `project/simulator/traffic_simulator.py` | 트래픽 시나리오 생성, 레이블 부여, 슬라이딩 윈도우, 정규화, stratified split, CSV 저장 |
| `project/scripts/generate_real_data.py` | 데이터 생성 실행 스크립트 |
| `project/data/real/train.csv` | 학습 데이터 |
| `project/data/real/val.csv` | 검증 데이터 |
| `project/data/real/test.csv` | 테스트 데이터 |
| `project/data/real/scaler_params.json` | Min-Max 정규화 기준값 |
| `project/data/real/dataset_summary.json` | 생성 결과 요약 |

## 구현 순서

1. `docs/yena/guideline_yena_stage1.md`의 체크리스트와 데이터 명세를 확인했다.
2. 네 가지 공장 운영 시나리오를 각각 함수로 분리했다.
3. 채널 점유율 기준으로 레이블을 자동 부여하는 `assign_label()` 함수를 구현했다.
4. 원본 시계열을 10 timestep 슬라이딩 윈도우로 변환했다.
5. 각 윈도우의 마지막 timestep 레이블을 해당 sample의 레이블로 사용했다.
6. 고정 Min-Max 기준으로 4개 feature를 `0~1` 범위로 정규화했다.
7. 전체 label 분포가 `0:600, 1:200, 2:150, 3:50`이 되도록 sample을 선택했다.
8. label 비율을 유지하면서 `train/val/test = 700/150/150`으로 분할했다.
9. CSV 3개, 정규화 기준 JSON, 데이터 요약 JSON을 저장했다.
10. 생성 결과를 검증한 뒤 체크리스트 완료 항목을 갱신했다.

## 시나리오 구성

| 시나리오 | 함수 | 핵심 패턴 |
|---|---|---|
| 일과 시작 | `simulate_startup_surge()` | 대기 상태에서 라인 가동 시작으로 트래픽과 점유율이 급증 |
| 긴급 증산 | `simulate_emergency_ramp()` | 정상 가동 중 예측 없는 폭증으로 심각 혼잡 발생 |
| 점심 재가동 | `simulate_lunch_restart()` | 점심 이후 단계적으로 트래픽이 증가 |
| 불균형 부하 | `simulate_imbalanced_ap_load()` | 특정 AP에 부하가 집중되어 경고, 혼잡, 심각 혼잡으로 진행 |

## 레이블 기준

| label | 의미 | 채널 점유율 |
|---|---|---|
| 0 | 정상 | 40% 미만 |
| 1 | 혼잡 경고 | 40% 이상 65% 미만 |
| 2 | 혼잡 | 65% 이상 85% 미만 |
| 3 | 심각 혼잡 | 85% 이상 |

레이블은 직접 임의 지정하지 않고, 각 timestep의 `channel_occupancy` 값으로 자동 계산한다. 따라서 feature와 label의 일관성이 유지된다.

## CSV 저장 방식

CSV는 모델 입력 shape를 보존하기 위해 sample 단위와 timestep 단위를 함께 저장한다.

```text
sample_id,timestep,rps,channel_occupancy,packet_loss,latency,label,scenario
```

- `sample_id`: 슬라이딩 윈도우 sample 번호
- `timestep`: sample 내부 timestep 번호, `0~9`
- `rps`, `channel_occupancy`, `packet_loss`, `latency`: 정규화된 feature 값
- `label`: 마지막 timestep 기준 혼잡 label
- `scenario`: 해당 sample이 나온 시나리오 이름

## 생성 결과

| split | samples | shape | label 분포 |
|---|---:|---|---|
| train | 700 | `(700, 10, 4)` | `0:420, 1:140, 2:105, 3:35` |
| val | 150 | `(150, 10, 4)` | `0:90, 1:30, 2:22, 3:8` |
| test | 150 | `(150, 10, 4)` | `0:90, 1:30, 2:23, 3:7` |

전체 기준 label 분포는 `0:600, 1:200, 2:150, 3:50`이며, 더미 데이터 명세의 60%, 20%, 15%, 5% 비율과 일치한다.

## 검증 내용

다음 항목을 확인했다.

- `train.csv`, `val.csv`, `test.csv` 생성 완료
- `scaler_params.json` 생성 완료
- 각 sample이 정확히 10 timestep으로 구성됨
- feature 수가 4개로 유지됨
- 정규화된 feature 값이 `0~1` 범위 안에 있음
- train/val/test sample 수가 `700/150/150`으로 분할됨
- label 분포가 stratified split 방식으로 유지됨

## 실행 방법

프로젝트 루트에서 아래 명령을 실행하면 동일한 데이터셋을 다시 생성할 수 있다.

```bash
python project/scripts/generate_real_data.py
```

재현성을 위해 기본 seed는 `42`로 설정되어 있다. 다른 seed를 사용하려면 아래처럼 실행한다.

```bash
python project/scripts/generate_real_data.py --seed 123
```

## 남은 확인 사항

김호중, 유용상에게 아래 내용을 공유하고 확인을 받으면 Stage 1 체크리스트를 완전히 닫을 수 있다.

- 데이터 경로: `project/data/real/`
- 모델 입력 shape: `(N, 10, 4)`
- label 체계: `0, 1, 2, 3`
- CSV column 구조: `sample_id,timestep,rps,channel_occupancy,packet_loss,latency,label,scenario`
