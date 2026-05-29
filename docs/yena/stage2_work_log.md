# 장예나 Stage 2 작업 설명

## 작업 목표

1단계에서 생성한 실제 트래픽 CSV를 김호중, 유용상이 바로 사용할 수 있는 `(batch, 10, 4)` 모델 입력 형태로 로딩하고 검증하는 공용 전처리 파이프라인을 구축했다.

## 사용 데이터

| 항목 | 값 |
|---|---|
| 데이터 경로 | `project/data/real/` |
| 파일 | `train.csv`, `val.csv`, `test.csv` |
| 정규화 기준 | `project/data/real/scaler_params.json` |
| feature | `rps`, `channel_occupancy`, `packet_loss`, `latency` |
| label | `0`, `1`, `2`, `3` |

Stage 1 CSV는 이미 `sample_id`, `timestep` 기준으로 window가 구성되어 있고, feature 값도 `0~1` 범위로 정규화되어 있다.

## 구현 파일

| 파일 | 역할 |
|---|---|
| `project/utils/dataloader.py` | CSV 로딩, 슬라이딩 윈도우, Min-Max 정규화, DataLoader 생성, 데이터 검증 |
| `project/scripts/validate_real_data.py` | train/val/test CSV와 DataLoader 출력 shape 검증 |
| `project/results/yena/yena_stage2_validation_report.txt` | Stage 2 검증 결과 txt 리포트 |

## DataLoader 인터페이스

```python
from utils.dataloader import get_dataloader

train_loader = get_dataloader("project/data/real/train.csv", batch_size=32, shuffle=True)
```

반환되는 batch shape는 아래와 같다.

```text
X: (batch, 10, 4)
y: (batch,)
```

`get_dataloader()`는 `window_size` 인자를 지원한다.

```python
get_dataloader(data_path, batch_size=32, shuffle=True, window_size=10)
```

## 전처리 구성

### 슬라이딩 윈도우

`make_sliding_windows()` 함수로 flat time-series CSV도 `(N, 10, 4)` 형태로 변환할 수 있게 했다. 현재 `project/data/real/` CSV는 이미 windowed format이므로 `sample_id`, `timestep` 기준으로 묶어서 로딩한다.

### Min-Max 정규화

`normalize_features()` 함수는 아래 고정 기준을 사용한다.

| feature | min | max |
|---|---:|---:|
| `rps` | 0 | 1000 |
| `channel_occupancy` | 0 | 100 |
| `packet_loss` | 0 | 30 |
| `latency` | 0 | 500 |

현재 CSV처럼 이미 정규화된 데이터는 그대로 읽고, flat CSV feature가 원 단위 값이면 정규화 후 window로 변환한다.

## 검증 실행 방법

프로젝트 루트에서 아래 명령을 실행한다.

```bash
python project/scripts/validate_real_data.py
```

실행하면 콘솔 출력과 함께 아래 파일에 검증 결과가 저장된다.

```text
project/results/yena/yena_stage2_validation_report.txt
```

## 검증 결과 요약

| split | rows | samples | window shape | label counts | batch X | batch y |
|---|---:|---:|---|---|---|---|
| train | 7000 | 700 | `[700, 10, 4]` | `0:420, 1:140, 2:105, 3:35` | `(32, 10, 4)` | `(32,)` |
| val | 1500 | 150 | `[150, 10, 4]` | `0:90, 1:30, 2:22, 3:8` | `(32, 10, 4)` | `(32,)` |
| test | 1500 | 150 | `[150, 10, 4]` | `0:90, 1:30, 2:23, 3:7` | `(32, 10, 4)` | `(32,)` |

모든 split에서 결측값은 없었고, feature 값은 정규화 범위 `0~1` 안에 있었다.

## 남은 확인 사항

김호중, 유용상에게 아래 내용을 공유하면 Stage 2 체크리스트를 완전히 닫을 수 있다.

- 데이터 경로: `project/data/real/`
- 공용 로더: `project/utils/dataloader.py`
- 사용 함수: `get_dataloader(data_path, batch_size=32, shuffle=True, window_size=10)`
- 출력 shape: `X=(batch, 10, 4)`, `y=(batch,)`
