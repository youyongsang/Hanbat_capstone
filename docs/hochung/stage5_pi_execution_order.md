# Raspberry Pi Stage 5 실행 순서

## 목적

정상 ONNX 모델과 INT8 경량화 ONNX 모델을 Raspberry Pi에서 각각 실행하여 실기기 추론 시간과 정확도를 비교한다.

이번 단계에서는 모델 학습을 다시 수행하지 않는다. 이미 학습된 `early_exit_fixed.pth` 체크포인트를 기준으로 ONNX 변환과 INT8 변환만 다시 수행한다.

## 실행 전제

| 항목 | 내용 |
|---|---|
| 기준 브랜치 | `yongsang` |
| 학습 재실행 여부 | 불필요 |
| 기준 체크포인트 | `project/checkpoints/early_exit_fixed.pth` |
| FP32 ONNX | `project/checkpoints/early_exit_fixed.onnx` |
| INT8 ONNX | `project/checkpoints/early_exit_fixed_int8.onnx` |
| Pi 입력 데이터 | `project/deploy/raspberry_pi/test.csv` |
| Pi 실행 스크립트 | `project/deploy/raspberry_pi/inference_pi.py` |

## 1. PC에서 최신 코드 받기

레포지토리 루트에서 실행한다.

```powershell
git pull origin yongsang
```

- 역할: 최신 `inference_pi.py`, `export_onnx.py`, `export_onnx_int8.py`, Pi 배포 번들을 가져온다.
- 학습 여부: 모델 학습은 하지 않는다.

## 2. PC에서 정상 FP32 ONNX 생성

```powershell
python project\scripts\export_onnx.py
```

- 역할: `early_exit_fixed.pth`를 기반으로 정상 FP32 ONNX 모델을 생성한다.
- 출력: `project/checkpoints/early_exit_fixed.onnx`
- 주의: 이전의 잘못된 `Hanbat_Capstone_Quantizer` ONNX 파일을 정상 LSTM ONNX로 덮어쓴다.

## 3. PC에서 INT8 ONNX 생성

```powershell
python project\scripts\export_onnx_int8.py
```

- 역할: 정상 FP32 ONNX를 기반으로 ONNX Runtime 정식 INT8 quantization을 수행한다.
- 입력: `project/checkpoints/early_exit_fixed.onnx`
- 출력: `project/checkpoints/early_exit_fixed_int8.onnx`
- 주의: 이 단계는 ONNX 그래프를 직접 조립하지 않고, 정상 LSTM ONNX 구조를 보존한 상태로 양자화한다.

## 4. PC에서 Raspberry Pi 배포 번들 생성

```powershell
python project\scripts\prepare_pi_bundle.py
```

- 역할: Raspberry Pi로 옮길 파일을 `project/deploy/raspberry_pi/` 폴더에 모은다.
- 포함 파일:
  - `early_exit_fixed.onnx`
  - `early_exit_fixed_int8.onnx`
  - `test.csv`
  - `inference_pi.py`
  - `README.md`

## 5. Raspberry Pi로 배포 폴더 이동

PC의 아래 폴더를 Raspberry Pi로 복사한다.

```text
project/deploy/raspberry_pi/
```

Raspberry Pi에서는 복사한 폴더로 이동한다.

```bash
cd ~/raspberry_pi
```

## 6. Raspberry Pi 실행 환경 구성

처음 한 번만 실행한다. 이미 `.venv`가 있고 패키지가 설치되어 있으면 생략할 수 있다.

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip
```

```bash
pip install onnxruntime numpy pandas
```

## 7. Raspberry Pi에서 FP32 ONNX 추론 실행

```bash
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_fp32_results.csv --max-samples 351
```

- 역할: 정상 FP32 ONNX 모델의 Raspberry Pi 실측 추론 결과를 저장한다.
- 출력:
  - `pi_fp32_results.csv`
  - `pi_fp32_results.txt`

## 8. Raspberry Pi에서 INT8 ONNX 추론 실행

```bash
python inference_pi.py --model early_exit_fixed_int8.onnx --data test.csv --output pi_int8_results.csv --max-samples 351
```

- 역할: INT8 경량화 ONNX 모델의 Raspberry Pi 실측 추론 결과를 저장한다.
- 출력:
  - `pi_int8_results.csv`
  - `pi_int8_results.txt`

## 9. Raspberry Pi에서 결과 확인

```bash
cat pi_fp32_results.txt
```

```bash
cat pi_int8_results.txt
```

확인할 주요 항목:

| 항목 | 의미 |
|---|---|
| `sample_count` | 테스트한 샘플 수. 현재 기준 351개가 정상 |
| `avg_inference_ms` | 평균 추론 시간 |
| `p95_inference_ms` | 상위 95% 지연 시간 |
| `exit1_rate` | Exit 1에서 조기 종료된 비율 |
| `exit2_rate` | Exit 2에서 종료된 비율 |
| `exit3_rate` | 마지막 Exit까지 도달한 비율 |

## 10. PC로 결과 파일 가져오기

Raspberry Pi에서 생성된 아래 4개 파일을 PC의 `project/results/hojung/` 폴더에 넣는다.

```text
pi_fp32_results.csv
pi_fp32_results.txt
pi_int8_results.csv
pi_int8_results.txt
```

## 11. PC에서 Pi 결과 분석

PC 레포지토리 루트에서 실행한다.

```powershell
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_fp32_results.csv --output-dir project\results\hojung --name pi_fp32_analysis
```

```powershell
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_int8_results.csv --output-dir project\results\hojung --name pi_int8_analysis
```

- 역할: Pi 결과 CSV를 기반으로 정확도, 평균 지연, Exit별 정확도, 시나리오별 결과를 분석한다.
- 출력 예시:
  - `pi_fp32_analysis.md`
  - `pi_fp32_analysis.txt`
  - `pi_fp32_analysis_overall.csv`
  - `pi_fp32_analysis_by_exit.csv`
  - `pi_fp32_analysis_by_scenario.csv`
  - `pi_int8_analysis.md`
  - `pi_int8_analysis.txt`
  - `pi_int8_analysis_overall.csv`
  - `pi_int8_analysis_by_exit.csv`
  - `pi_int8_analysis_by_scenario.csv`

## 전체 요약 명령어

### PC

```powershell
git pull origin yongsang
python project\scripts\export_onnx.py
python project\scripts\export_onnx_int8.py
python project\scripts\prepare_pi_bundle.py
```

### Raspberry Pi

```bash
cd ~/raspberry_pi
source .venv/bin/activate
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_fp32_results.csv --max-samples 351
python inference_pi.py --model early_exit_fixed_int8.onnx --data test.csv --output pi_int8_results.csv --max-samples 351
cat pi_fp32_results.txt
cat pi_int8_results.txt
```

### PC 분석

```powershell
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_fp32_results.csv --output-dir project\results\hojung --name pi_fp32_analysis
python project\scripts\analyze_pi_results.py --input project\results\hojung\pi_int8_results.csv --output-dir project\results\hojung --name pi_int8_analysis
```

## 주의사항

- `train.py`, `train_early_exit.py`는 이번 Pi 재실측 단계에서 실행하지 않는다.
- `export_onnx_int8.py`는 반드시 `export_onnx.py` 실행 후 실행한다.
- `early_exit_fixed.onnx`와 `early_exit_fixed_int8.onnx`를 둘 다 Pi에 넣어야 FP32와 INT8 비교가 가능하다.
- `--max-samples 351`은 현재 테스트셋 전체 샘플 수 기준이다.
- 기존 `pi_inference_results.csv`처럼 이름이 하나뿐인 결과는 FP32/INT8 구분이 어려우므로 사용하지 않는다.
- 최종 보고서에는 Pi에서 새로 생성한 `pi_fp32_results.*`, `pi_int8_results.*` 기준 결과만 사용한다.
