# 장예나 외부 데이터셋 교체 작업 설명

## 작업 목표

기존 `traffic_simulator.py`로 임의 생성한 데이터를 실제 공개 데이터셋으로 교체하고,  
기존 파이프라인(`dataloader.py`, `train.py`, `evaluate.py` 등)과 완전히 호환되는 형태로 변환하여  
`project/data/real/` 경로에 저장한다.

---

## 선택 데이터셋

| 항목 | 내용 |
|---|---|
| 이름 | 6G Network Slicing QoS Dataset |
| 출처 | Kaggle (ziya07) |
| URL | https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset |
| 원본 행수 | 2,345행 |
| 기간 | 2025-01-01 ~ 2025-04-08 (시간 단위 연속 시계열) |
| 라이선스 | Kaggle 공개 데이터셋 |

### 출처

```
6G Network Slicing QoS Dataset, Kaggle (ziya07).
URL: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
```

---

## 선택 이유


### 1. 피처 직접 대응

다른 후보들은 throughput → rps 변환, 점유율 유도 공식 등 간접 파생이 필요했다.  
이 데이터셋은 프로젝트 피처 4개와 1:1로 바로 대응된다.

| 원본 컬럼 | 프로젝트 피처 | 관계 |
|---|---|---|
| `Traffic Load (bps)` | `rps` | 직접 대응 |
| `Bandwidth Utilization (%)` | `channel_occupancy` | 직접 대응 |
| `Packet Loss Rate (%)` | `packet_loss` | 직접 대응 |
| `Latency (ms)` | `latency` | 직접 대응 |

변환 과정의 논리적 비약이 없어 보고서에서 피처 정의를 그대로 인용할 수 있다.

### 2. 시나리오 컬럼 내장

`Network Slice ID` (1~4) 컬럼이 이미 존재하여 프로젝트의 4가지 공장 시나리오에 직접 매핑했다.  
다른 데이터셋들은 시나리오를 패턴 감지로 추론하거나 임의 순환 배정해야 했다.

| Network Slice ID | 프로젝트 시나리오 |
|---|---|
| 1 | `startup_surge` (점진적 부하 증가) |
| 2 | `emergency_ramp` (고부하 폭증) |
| 3 | `lunch_restart` (저→중 점진 증가) |
| 4 | `imbalanced_ap_load` (지속 고부하) |

### 3. 연속 시계열 구조

시간 단위 연속 측정값(2025-01-01 ~ 2025-04-08)으로, 슬라이딩 윈도우 학습에 필수인  
시계열 연속성이 보장된다. 일부 후보 데이터셋은 집계 통계 테이블이어서 시계열 구조가 없었다.

### 4. 충분한 데이터량

2,345행 → 슬라이딩 윈도우 후 2,336 샘플.  
기존 시뮬레이터 1,000 샘플의 2배 이상으로, 모델 일반화 성능에 유리하다.

### 5. 데이터 품질

결측값 없음, 모든 수치 피처 이미 0~1 정규화 완료, 4개 레이블 클래스 전부 자연 분포.  
별도 클리닝 없이 변환 스크립트만 실행하면 된다.

---

## 구현 파일

| 파일 | 역할 |
|---|---|
| `project/scripts/generate_from_real_dataset.py` | 외부 데이터셋 → 프로젝트 형식 변환 스크립트 |
| `project/data/real/train.csv` | 학습 데이터 (교체됨) |
| `project/data/real/val.csv` | 검증 데이터 (교체됨) |
| `project/data/real/test.csv` | 테스트 데이터 (교체됨) |
| `project/data/real/scaler_params.json` | 정규화 기준값 (기존과 동일) |
| `project/data/real/dataset_summary.json` | 변환 결과 요약 |
| `project/data/real/conversion_report.txt` | 변환 과정 로그 (신규) |
| `project/data/external/.gitignore` | 원본 대용량 파일 git 제외 설정 |

> 기존 `dataloader.py`, `train.py`, `evaluate.py`, `compare_baselines.py`는 **수정 불필요**.  
> CSV 컬럼 구조와 shape이 기존과 완전히 동일하기 때문이다.

---

## 피처 변환 상세

원본 데이터셋의 모든 수치 피처는 이미 `0~1`로 정규화된 상태다.  
변환 스크립트 내부에서 물리적 최댓값을 곱해 단위를 복원한 뒤, 프로젝트 기준으로 재정규화한다.

```
원본 0~1 값  →  × 최댓값 (단위 복원)  →  Min-Max 재정규화  →  최종 0~1
```

이 과정은 항등 변환처럼 보이지만, **레이블 생성 단계에서 channel_occupancy가  
반드시 0~100% 스케일**이어야 하므로 단위 복원 단계가 필수다.

| 피처 | 복원 배율 | 복원 후 범위 |
|---|---|---|
| `rps` | × 1000 | 0 ~ 1000 RPS |
| `channel_occupancy` | × 100 | 0 ~ 100 % |
| `packet_loss` | × 30 | 0 ~ 30 % |
| `latency` | × 500 | 0 ~ 500 ms |

---

## 레이블 생성 기준

기존 프로젝트 정의와 동일하게, `channel_occupancy` 기준으로 자동 부여한다.

| 레이블 | 혼잡 수준 | 채널 점유율 |
|---|---|---|
| 0 | 정상 | 40% 미만 |
| 1 | 혼잡 경고 | 40 ~ 65% |
| 2 | 혼잡 | 65 ~ 85% |
| 3 | 심각 혼잡 | 85% 이상 |

---

## 변환 실행 방법

원본 파일을 `project/data/external/`에 저장한 뒤 실행한다.  
`--input` 옵션 뒤에는 반드시 실제 CSV 파일 경로가 와야 한다.

먼저 파일명을 확인한다.

```powershell
dir project\data\external
```

Windows PowerShell에서는 아래처럼 한 줄로 실행한다.

```powershell
python project\scripts\generate_from_real_dataset.py --dataset kaggle_6g --input project\data\external\6G_network_slicing_qos_dataset_2345.csv --out-dir project\data\real --overwrite-real
```

macOS/Linux 또는 Git Bash에서는 아래처럼 줄바꿈해서 실행할 수 있다.

```bash
python project/scripts/generate_from_real_dataset.py \
    --dataset kaggle_6g \
    --input   project/data/external/6G_network_slicing_qos_dataset_2345.csv \
    --out-dir project/data/real \
    --overwrite-real
```

주의 사항:

- `--input`만 입력하고 파일 경로를 생략하면 `argument --input/-i: expected one argument` 에러가 발생한다.
- 원본 CSV 파일명이 다르면 `--input` 뒤의 파일명도 실제 파일명으로 바꿔야 한다.
- 원본 CSV는 `.gitignore` 대상이므로 GitHub에는 올라가지 않을 수 있다. 다른 컴퓨터에서 처음 실행할 때는 원본 CSV를 직접 `project/data/external/`에 넣어야 한다.

실행하면 `[1/7]` ~ `[7/7]` 순서로 진행 상황이 출력되고,  
`project/data/real/` 에 6개 파일이 자동 생성된다.

---

## 생성 결과

| split | samples | shape | 레이블 분포 |
|---|---:|---|---|
| train | 1,635 | `(1635, 10, 4)` | `0:638, 1:423, 2:330, 3:244` |
| val | 350 | `(350, 10, 4)` | `0:136, 1:91, 2:70, 3:53` |
| test | 351 | `(351, 10, 4)` | `0:137, 1:91, 2:71, 3:52` |
| **합계** | **2,336** | | |

### 시나리오 분포 (train 기준)

| 시나리오 | 샘플 수 |
|---|---:|
| `startup_surge` | 427 |
| `lunch_restart` | 408 |
| `emergency_ramp` | 407 |
| `imbalanced_ap_load` | 393 |

### 기존 시뮬레이터 데이터와 비교

| 항목 | 시뮬레이터 (기존) | 외부 데이터셋 (신규) |
|---|---|---|
| 전체 샘플 수 | 1,000 | **2,336** |
| 레이블 분포 | 60:20:15:5 | 39:26:20:15 |
| 시나리오 출처 | 공장 패턴 수식 | Network Slice ID 직접 매핑 |
| 데이터 출처 | NumPy 임의 생성 | 실제 측정 기반 공개 데이터셋 |

> **레이블 분포 변화 주의:**  
> 레이블 3(심각 혼잡) 비율이 5% → 15%로 늘었다.  
> 실제 데이터의 자연 분포이므로 정상이나, 모델 학습 시  
> `CrossEntropyLoss`에 클래스 가중치(`weight` 인자) 적용을 권장한다.  
> 호중, 용상에게 공유 필요.

---

## 검증 내용

변환 완료 후 아래 항목을 확인했다.

- `train.csv`, `val.csv`, `test.csv` 생성 완료
- `scaler_params.json` 생성 완료 (기존 기준값과 동일)
- 각 샘플이 정확히 10 timestep으로 구성됨
- 피처 수 4개 유지
- 정규화된 피처 값이 `0~1` 범위 안에 있음
- Stratified Split으로 레이블 비율 유지 확인
- 기존 `get_dataloader()` 함수로 정상 로딩 확인 (`X=(32, 10, 4)`, `y=(32,)`)

---

## 공유 사항

- 데이터 경로: `project/data/real/` (변경 없음)
- 모델 입력 shape: `(N, 10, 4)` (변경 없음)
- 레이블 체계: `0, 1, 2, 3` (변경 없음)
- CSV 컬럼: `sample_id, timestep, rps, channel_occupancy, packet_loss, latency, label, scenario` (변경 없음)
- **새로 확인 필요:** 레이블 3 비율 5% → 15% 증가로 인해 학습 시 클래스 가중치 적용 권장

---

## 남은 확인 사항

- [ ] 호중·용상에게 데이터 교체 및 레이블 분포 변화 공유
- [ ] 호중·용상이 새 데이터로 기존 모델 재학습 후 성능 확인
- [ ] 클래스 가중치 적용 여부 팀 협의
- [ ] 보고서 데이터 출처 항목에 위 인용 표기 추가
