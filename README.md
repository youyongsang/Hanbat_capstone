# Early Exit LSTM 기반 산업 무선망 트래픽 혼잡 감지 시스템

> Raspberry Pi 엣지 환경에서 동작하는 산업 무선망 혼잡 감지 및 경량 추론 파이프라인

**컴퓨터공학과 | 장예나 · 유용상 · 김호중**

---

## 프로젝트 개요

본 프로젝트는 산업 무선망에서 발생하는 트래픽 혼잡 상태를 LSTM 기반 시계열 모델로 분류하고, Early Exit 구조와 INT8 Quantization을 적용하여 Raspberry Pi와 같은 엣지 장비에서도 빠르게 추론할 수 있도록 설계한 시스템이다.

입력 데이터는 RPS, 채널 점유율, 패킷 손실률, 지연시간으로 구성되며, 모델은 무선망 상태를 0~3단계 혼잡 수준으로 분류한다. 학습된 Early Exit LSTM은 ONNX 형식으로 변환된 뒤, INT8 ONNX 모델로 경량화되어 Raspberry Pi에서 실측 검증되었다.

---

## 최종 성능 요약

Raspberry Pi 실기기에서 FP32 ONNX와 INT8 ONNX를 동일한 테스트셋 351개 샘플로 비교하였다.

| 모델 | 정확도 | 평균 추론 시간 | p50 | p95 | Exit1 / Exit2 / Exit3 |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | 95.7% | 1.512ms | 1.488ms | 1.580ms | 41.3% / 44.7% / 14.0% |
| INT8 ONNX | 95.7% | 0.919ms | 0.895ms | 1.021ms | 41.3% / 45.0% / 13.7% |

INT8 ONNX 모델은 FP32 ONNX 모델과 동일한 정확도 95.7%를 유지하면서 평균 추론 시간을 약 39.2% 단축하였다.

---

## 핵심 알고리즘

### Early Exit LSTM

Early Exit LSTM은 3개 LSTM 레이어 뒤에 각각 Exit Classifier를 배치한 구조이다. 각 Exit point에서 Shannon Entropy를 계산하고, 불확실성이 threshold 이하이면 해당 레이어에서 추론을 종료한다. 학습 시에는 모든 Exit의 손실을 가중 합산한 Multi-exit Loss를 사용하고, 추론 시에만 조기 종료를 적용한다.

### 고정 및 동적 Threshold

고정 threshold는 모든 입력에 동일한 조기 종료 기준을 적용한다. 동적 threshold는 최근 timestep의 채널 점유율 변화를 이용해 threshold를 조정한다. 동적 threshold는 Exit 3 도달률 감소와 정확도 개선에 효과가 있었지만, 추가 계산 오버헤드로 인해 실측 지연 우위는 제한적이었다.

### ONNX 및 INT8 경량화

학습된 PyTorch Early Exit 모델을 FP32 ONNX로 변환한 뒤, ONNX Runtime 기반 INT8 Quantization을 적용한다. 최종 배포 모델은 다음 두 가지이다.

| 모델 파일 | 설명 |
|---|---|
| `early_exit_fixed.onnx` | FP32 ONNX 기준 모델 |
| `early_exit_fixed_int8.onnx` | INT8 경량화 ONNX 모델 |

---

## 프로젝트 구조

```text
project/
├── checkpoints/
│   ├── baseline_lstm_best.pth
│   ├── early_exit_fixed.pth
│   ├── early_exit_fixed.onnx
│   └── early_exit_fixed_int8.onnx
├── data/
│   ├── external/
│   └── real/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── deploy/
│   └── raspberry_pi/
│       ├── early_exit_fixed.onnx
│       ├── early_exit_fixed_int8.onnx
│       ├── inference_pi.py
│       ├── test.csv
│       └── README.md
├── models/
│   ├── baseline_lstm.py
│   └── early_exit_lstm.py
├── results/
│   ├── hojung/
│   ├── yena/
│   └── yongsang/
├── scripts/
│   ├── generate_from_real_dataset.py
│   ├── train.py
│   ├── train_early_exit.py
│   ├── evaluate.py
│   ├── evaluate_early_exit.py
│   ├── compare_baselines.py
│   ├── export_onnx.py
│   ├── export_onnx_int8.py
│   ├── prepare_pi_bundle.py
│   ├── inference_pi.py
│   └── analyze_pi_results.py
└── utils/
    ├── dataloader.py
    └── logger.py
```

---

## AP 실측 strict 데이터 파이프라인

방학 중 AP 장비에서 수집한 실측 CSV를 기준으로 9개 feature 모델 파이프라인을 별도로 구성했다. 기존 1학기 `project/data/real` 4-feature 실험과 구분하기 위해 AP 실측 데이터는 아래 문서를 기준으로 실행한다.

```text
project/README_AP_STRICT.md
```

해당 문서에는 AP strict CSV 기준 feature 목록, 제외 컬럼, train/val/test 변환, AP용 Baseline/Early Exit 학습, 평가, 4개 모델 비교표 생성 절차가 정리되어 있다.

---

## 설치 방법

레포지토리를 받은 뒤 `hojung` 브랜치를 기준으로 실행한다.

```powershell
git clone https://github.com/youyongsang/Hanbat_capstone.git
cd Hanbat_capstone
git switch hojung
```

Python 가상환경을 생성하고 패키지를 설치한다.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install onnx onnxruntime onnxscript
```

---

## 전체 실행 순서

### 1. 데이터 변환

외부 CSV 데이터셋을 프로젝트 학습 형식으로 변환한다.

```powershell
python project\scripts\generate_from_real_dataset.py --dataset kaggle_6g --input project\data\external\6G_network_slicing_qos_dataset_2345.csv --out-dir project\data\real --overwrite-real
```

생성 결과:

| 파일 | 설명 |
|---|---|
| `project/data/real/train.csv` | 학습 데이터 |
| `project/data/real/val.csv` | 검증 데이터 |
| `project/data/real/test.csv` | 테스트 및 Pi 추론 입력 데이터 |
| `project/data/real/dataset_summary.json` | 데이터 변환 요약 |

### 2. 모델 학습

Baseline LSTM을 학습한다.

```powershell
python project\scripts\train.py
```

Early Exit LSTM을 학습한다.

```powershell
python project\scripts\train_early_exit.py
```

주요 체크포인트:

| 파일 | 설명 |
|---|---|
| `baseline_lstm_best.pth` | Baseline LSTM 모델 |
| `early_exit_lstm_best.pth` | Early Exit LSTM 최적 모델 |
| `early_exit_fixed.pth` | 고정 threshold Early Exit 모델 |
| `early_exit_dynamic.pth` | 동적 threshold Early Exit 모델 |

### 3. 모델 평가 및 비교

```powershell
python project\scripts\evaluate.py
python project\scripts\evaluate_early_exit.py
python project\scripts\compare_baselines.py
```

주요 결과:

| 파일 | 설명 |
|---|---|
| `project/results/hojung/comparison_summary.csv` | 전체 알고리즘 비교 결과 |
| `project/results/hojung/comparison_summary.txt` | 전체 알고리즘 비교 요약 |

### 4. 시나리오 분석

```powershell
python project\scripts\generate_test_with_scenario.py
python project\scripts\fill_scenario_analysis.py
```

주요 결과:

| 파일 | 설명 |
|---|---|
| `project/results/yongsang/scenario_analysis_summary.csv` | 시나리오별 정확도 및 Exit 분포 |
| `project/results/scenario_analysis/` | 개별 시나리오 분석 결과 |

### 5. ONNX 변환 및 INT8 경량화

FP32 ONNX 모델을 생성한다.

```powershell
python project\scripts\export_onnx.py
```

INT8 ONNX 모델을 생성한다.

```powershell
python project\scripts\export_onnx_int8.py
```

Raspberry Pi 배포 번들을 생성한다.

```powershell
python project\scripts\prepare_pi_bundle.py
```

생성 위치:

```text
project/deploy/raspberry_pi/
```

---

## Raspberry Pi 실행 방법

PC에서 생성한 `project/deploy/raspberry_pi/` 폴더를 Raspberry Pi로 복사한 뒤, Pi에서 해당 폴더로 이동한다.

```bash
cd ~/raspberry_pi
```

Pi 실행 환경을 구성한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas
```

FP32 ONNX 모델을 실행한다.

```bash
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_fp32_results.csv --max-samples 351
```

INT8 ONNX 모델을 실행한다.

```bash
python inference_pi.py --model early_exit_fixed_int8.onnx --data test.csv --output pi_int8_results.csv --max-samples 351
```

결과를 확인한다.

```bash
cat pi_fp32_results.txt
cat pi_int8_results.txt
```

---

## Pi 결과 분석

Pi에서 생성한 결과 파일을 PC의 `project/results/hojung/` 폴더로 복사한 뒤 분석한다.

```powershell
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_fp32_results.csv --output-dir project\results\hojung --name pi_fp32_analysis
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_int8_results.csv --output-dir project\results\hojung --name pi_int8_analysis
```

분석 결과:

| 파일 | 설명 |
|---|---|
| `pi_fp32_analysis.txt` | FP32 ONNX 분석 요약 |
| `pi_fp32_analysis.md` | FP32 ONNX 보고서용 분석 |
| `pi_fp32_analysis_by_exit.csv` | FP32 Exit별 분석 |
| `pi_fp32_analysis_by_scenario.csv` | FP32 시나리오별 분석 |
| `pi_int8_analysis.txt` | INT8 ONNX 분석 요약 |
| `pi_int8_analysis.md` | INT8 ONNX 보고서용 분석 |
| `pi_int8_analysis_by_exit.csv` | INT8 Exit별 분석 |
| `pi_int8_analysis_by_scenario.csv` | INT8 시나리오별 분석 |

---

## 주요 실행 파일

| 파일 | 역할 |
|---|---|
| `generate_from_real_dataset.py` | 외부 CSV 데이터셋을 학습용 데이터로 변환 |
| `train.py` | Baseline LSTM 학습 |
| `train_early_exit.py` | Early Exit LSTM 학습 |
| `evaluate.py` | Baseline LSTM 평가 |
| `evaluate_early_exit.py` | Early Exit LSTM 평가 |
| `compare_baselines.py` | 임계값, Baseline LSTM, 고정 Early Exit, 동적 Early Exit 비교 |
| `generate_test_with_scenario.py` | 테스트 데이터에 시나리오 정보 추가 |
| `fill_scenario_analysis.py` | 시나리오별 성능 분석 |
| `export_onnx.py` | FP32 ONNX 모델 생성 |
| `export_onnx_int8.py` | INT8 ONNX 모델 생성 |
| `prepare_pi_bundle.py` | Raspberry Pi 배포 번들 생성 |
| `inference_pi.py` | Raspberry Pi ONNX 추론 및 지연 측정 |
| `analyze_pi_results.py` | Raspberry Pi 결과 분석 |

---

## 팀 역할

| 팀원 | 담당 영역 | 주요 산출물 |
|---|---|---|
| 장예나 | 데이터 및 시나리오 | 외부 데이터 변환, 시나리오 구성, 결과 시각화 |
| 유용상 | 모델 설계 | Early Exit LSTM, 고정/동적 threshold, 시나리오별 분석 |
| 김호중 | 비교 실험 및 배포 | Baseline LSTM, 4개 방식 비교, ONNX/INT8, Raspberry Pi 실측 |

---

## 한계 및 향후 과제

- 실제 무선 AP의 운영 채널을 자동 변경하는 단계는 AP 제어 API, SSH, OpenWrt 등 추가 연동 환경이 필요하다.
- 동적 threshold는 정확도와 Exit 3 감소 측면에서 효과가 있었지만, 계산 오버헤드로 인해 실측 지연 우위는 제한적이었다.
- 향후 실제 공장 무선망 데이터와 AP 제어 인터페이스를 연동하면 자동 채널 전환 시스템으로 확장할 수 있다.
