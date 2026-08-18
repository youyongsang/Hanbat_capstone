# 전체 실험 터미널 명령어 가이드

이 문서는 레포지토리 루트 디렉터리 기준으로 프로젝트를 처음부터 다시 실행할 때 사용하는 터미널 명령어 모음이다.
각 명령어 아래에는 해당 명령어가 누구 담당 작업에서 구현되었는지, 몇 Stage에 해당하는지, 어떤 결과를 만드는지 함께 정리한다.

## 0. 기본 위치 및 환경 확인

```powershell
cd <레포지토리_루트_경로>
```

- 담당: 공통
- Stage: 전체 공통
- 설명: 모든 명령어는 `project`, `docs` 폴더가 보이는 레포지토리 루트에서 실행한다. 예를 들어 본인 컴퓨터에서 클론한 폴더로 이동하면 된다.

```powershell
git status
```

- 담당: 공통
- Stage: 전체 공통
- 설명: 현재 브랜치와 수정 파일 상태를 확인한다.

```powershell
python --version
```

- 담당: 공통
- Stage: 전체 공통
- 설명: Python 실행 환경을 확인한다.

```powershell
python -c "import torch; print(torch.__version__)"
```

- 담당: 공통
- Stage: 전체 공통
- 설명: PyTorch 설치 여부와 버전을 확인한다.

```powershell
pip install -r requirements.txt
```

- 담당: 공통
- Stage: 전체 공통
- 설명: 프로젝트 실행에 필요한 기본 Python 패키지를 설치한다.

```powershell
pip install onnx onnxruntime onnxscript
```

- 담당: 김호중
- Stage: Stage 4
- 설명: INT8 Quantization, ONNX 변환, ONNX Runtime 추론 검증에 필요한 패키지를 설치한다.

## 1. 데이터 생성 및 변환

먼저 원본 CSV 파일이 실제로 존재하는지 확인한다.

```powershell
dir project\data\external
```

- 담당: 장예나
- Stage: Stage 2
- 설명: 외부 원본 CSV가 `project\data\external` 폴더에 있는지 확인한다. `--input` 뒤에는 반드시 여기에서 확인한 CSV 파일명을 포함한 경로를 넣어야 한다.

```powershell
python project\scripts\generate_from_real_dataset.py --dataset kaggle_6g --input project\data\external\6G_network_slicing_qos_dataset_2345.csv --out-dir project\data\real --overwrite-real
```

- 담당: 장예나
- Stage: Stage 2
- 설명: 외부 6G/Kaggle 계열 CSV를 프로젝트 학습 형식인 `project\data\real\train.csv`, `val.csv`, `test.csv`로 변환한다.
- 주의: `--input`만 쓰고 파일 경로를 생략하면 `expected one argument` 에러가 발생한다.
- 참고: 원본 CSV가 `project\data\external`에 없으면 이 명령어는 실행하지 않는다. 이미 `project\data\real`에 변환된 CSV가 있으면 다음 단계부터 진행 가능하다.

```powershell
dir project\data\real
```

- 담당: 장예나
- Stage: Stage 2
- 설명: 변환 결과로 `train.csv`, `val.csv`, `test.csv`, `scaler_params.json`, `dataset_summary.json`, `conversion_report.txt`가 생성되었는지 확인한다.

```powershell
python project\scripts\generate_test_with_scenario.py
```

- 담당: 장예나
- Stage: Stage 3
- 설명: `project\data\real\test.csv`에 시나리오 컬럼을 추가하여 `test_with_scenario.csv`를 생성한다.

```powershell
python project\scripts\split_scenario_data.py
```

- 담당: 장예나
- Stage: Stage 3
- 설명: 시나리오가 붙은 테스트 데이터를 시나리오별 CSV로 분리한다.

## 2. 모델 학습

```powershell
python project\scripts\train.py
```

- 담당: 김호중
- Stage: Stage 2
- 설명: 일반 Baseline LSTM 모델을 학습하고 `project\checkpoints\baseline_lstm_best.pth`를 생성한다.

```powershell
python project\scripts\train_early_exit.py
```

- 담당: 용상
- Stage: Stage 2
- 설명: Early Exit LSTM 모델을 학습하고 Early Exit 관련 체크포인트를 생성한다.

## 3. 단독 평가

```powershell
python project\scripts\evaluate.py
```

- 담당: 김호중
- Stage: Stage 2
- 설명: 일반 Baseline LSTM 모델의 정확도와 추론 시간을 평가한다.

```powershell
python project\scripts\evaluate_early_exit.py
```

- 담당: 용상
- Stage: Stage 2
- 설명: Early Exit LSTM 모델을 단독 평가하고 고정형/동적형 Early Exit 결과를 확인한다.

## 4. 전체 모델 비교

```powershell
python project\experiments\compare_baselines.py
```

- 담당: 김호중
- Stage: Stage 3
- 설명: 임계값 방식, 일반 LSTM, 고정형 Early Exit, 동적 Early Exit를 한 번에 비교한다.
- 주요 결과: `project\results\hojung\comparison_summary.csv`, `project\results\hojung\comparison_summary.txt`

```powershell
python project\scripts\generate_summary.py
```

- 담당: 장예나
- Stage: Stage 3
- 설명: 공통 logger를 사용하여 핵심 비교 결과를 `project\results\yena\comparison_summary.csv` 형식으로 저장한다.

```powershell
python project\scripts\fill_scenario_analysis.py
```

- 담당: 용상
- Stage: Stage 3
- 설명: 시나리오별 정확도와 Early Exit 결과를 채워 `project\results\yongsang\scenario_analysis_summary.csv`를 생성한다.

## 5. Stage 4 배포 및 경량화 검증

```powershell
python project\scripts\evaluate_quantization.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: 원본 모델과 INT8 양자화 모델의 크기, 정확도, 추론 시간을 비교한다.
- 주요 결과: `project\results\quantization_comparison.csv`

```powershell
python project\scripts\export_onnx.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: PyTorch 모델을 ONNX 형식으로 변환한다.

```powershell
python project\scripts\export_onnx_int8.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: 배포 검증을 위해 ONNX/INT8 관련 모델 파일을 생성한다.

```powershell
python project\scripts\check_onnx.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: 생성된 ONNX 파일의 구조와 유효성을 확인한다.

```powershell
python project\scripts\test_onnx_inference.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: ONNX Runtime에서 Early Exit ONNX 모델이 정상 추론되는지 smoke test를 수행한다.

```powershell
python project\scripts\inference_pi.py
```

- 담당: 김호중
- Stage: Stage 4
- 설명: 라즈베리파이 배포 상황을 가정한 ONNX Runtime 추론 지연 시간을 측정한다.

## 6. 결과 시각화

```powershell
python project\scripts\visualize_results.py
```

- 담당: 장예나
- Stage: Stage 4
- 설명: 정확도, 지연 시간, Early Exit 비율, 양자화 결과, 시나리오별 정확도 그래프를 생성한다.
- 선행 조건: `project\experiments\compare_baselines.py`와 `project\scripts\evaluate_quantization.py` 실행 결과가 있어야 한다.
- 입력 파일: `project\results\hojung\comparison_summary.csv`, `project\results\quantization_comparison.csv`
- 주요 결과: `project\results\yena\accuracy_latency_combined.png`, `exit_rate_comparison.png`, `quantization_comparison.png`, `scenario_accuracy.png`

## 7. Stage 5 Raspberry Pi 실측 준비

```powershell
python project\scripts\prepare_pi_bundle.py
```

- 담당: 김호중
- Stage: Stage 5
- 설명: Raspberry Pi로 옮길 ONNX 모델, 테스트 CSV, 실측 스크립트, 실행 README를 `project\deploy\raspberry_pi`에 모은다.

```powershell
python project\scripts\inference_pi.py --max-samples 100 --output project\results\hojung\pi_inference_results_pc_smoke.csv
```

- 담당: 김호중
- Stage: Stage 5
- 설명: Pi 실측 전 PC에서 ONNX Runtime 추론 CSV 저장 로직이 정상 동작하는지 smoke test한다. 이 결과는 PC 테스트값이므로 최종 Pi 결과로 사용하지 않는다.

Raspberry Pi 안에서는 배포 번들 폴더로 이동한 뒤 아래 명령을 실행한다.

```bash
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_inference_results.csv --max-samples 100
```

- 담당: 김호중
- Stage: Stage 5
- 설명: Raspberry Pi 실기기에서 ONNX Early Exit 모델 추론 지연을 측정하고 `pi_inference_results.csv`, `pi_inference_results.txt`를 저장한다.

## 8. 한 번에 실행하는 종합 순서

원본 외부 CSV가 있는 경우에는 아래 순서로 실행한다.

```powershell
dir project\data\external
python project\scripts\generate_from_real_dataset.py --dataset kaggle_6g --input project\data\external\6G_network_slicing_qos_dataset_2345.csv --out-dir project\data\real --overwrite-real
dir project\data\real
python project\scripts\generate_test_with_scenario.py
python project\scripts\split_scenario_data.py
python project\scripts\train.py
python project\scripts\train_early_exit.py
python project\scripts\evaluate.py
python project\scripts\evaluate_early_exit.py
python project\experiments\compare_baselines.py
python project\scripts\generate_summary.py
python project\scripts\fill_scenario_analysis.py
python project\scripts\evaluate_quantization.py
python project\scripts\export_onnx.py
python project\scripts\export_onnx_int8.py
python project\scripts\check_onnx.py
python project\scripts\test_onnx_inference.py
python project\scripts\inference_pi.py
python project\scripts\visualize_results.py
python project\scripts\prepare_pi_bundle.py
```

이미 `project\data\real`에 변환된 데이터가 있는 경우에는 첫 줄을 제외하고 아래부터 실행한다.

```powershell
python project\scripts\generate_test_with_scenario.py
python project\scripts\split_scenario_data.py
python project\scripts\train.py
python project\scripts\train_early_exit.py
python project\scripts\evaluate.py
python project\scripts\evaluate_early_exit.py
python project\experiments\compare_baselines.py
python project\scripts\generate_summary.py
python project\scripts\fill_scenario_analysis.py
python project\scripts\evaluate_quantization.py
python project\scripts\export_onnx.py
python project\scripts\export_onnx_int8.py
python project\scripts\check_onnx.py
python project\scripts\test_onnx_inference.py
python project\scripts\inference_pi.py
python project\scripts\visualize_results.py
python project\scripts\prepare_pi_bundle.py
```

## 9. 주요 결과 파일 확인 명령어

```powershell
type project\results\hojung\comparison_summary.txt
```

- 담당: 김호중
- Stage: Stage 3
- 설명: 네 가지 방식의 종합 비교 결과를 텍스트로 확인한다.

```powershell
type project\results\hojung\comparison_summary.csv
```

- 담당: 김호중
- Stage: Stage 3
- 설명: 종합 비교 결과를 CSV 형태로 확인한다.

```powershell
type project\results\yena\comparison_summary.csv
```

- 담당: 장예나
- Stage: Stage 3
- 설명: logger 기반 요약 결과를 확인한다.

```powershell
type project\results\yongsang\scenario_analysis_summary.csv
```

- 담당: 용상
- Stage: Stage 3
- 설명: 시나리오별 성능 분석 결과를 확인한다.

```powershell
type project\results\quantization_comparison.csv
```

- 담당: 김호중
- Stage: Stage 4
- 설명: 양자화 및 ONNX Runtime 기반 경량화 결과를 확인한다.

```powershell
dir project\results\yena
```

- 담당: 장예나
- Stage: Stage 4
- 설명: 생성된 시각화 그래프 파일 목록을 확인한다.
