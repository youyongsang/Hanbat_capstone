# 김호중 Stage 5 작업 설명

## Raspberry Pi 포팅 및 실기기 추론 지연 실측 준비

## 1. 작업 목표

Stage 4에서 생성한 `early_exit_fixed.onnx` 모델을 Raspberry Pi에서 직접 실행할 수 있도록 배포 구조를 정리하고, Pi 실기기에서 추론 지연을 CSV/TXT로 저장하는 실행 스크립트를 준비한다.

이번 단계의 핵심은 단순 ONNX smoke test가 아니라, Pi에서 실제 테스트 샘플을 반복 추론하여 다음 결과 파일을 남기는 것이다.

| 결과 파일 | 역할 |
|---|---|
| `pi_inference_results.csv` | 샘플별 예측 라벨, Exit 지점, confidence, 추론 시간 저장 |
| `pi_inference_results.txt` | 평균/최소/최대/p50/p95 지연 시간 및 Exit 비율 요약 |

---

## 2. 구현 및 수정 파일

| 파일 | 변경 내용 |
|---|---|
| `project/scripts/inference_pi.py` | Pi 실측용 CLI 스크립트로 재작성 |
| `project/scripts/prepare_pi_bundle.py` | Pi로 옮길 모델/데이터/스크립트 번들 생성 |
| `project/deploy/raspberry_pi/` | Pi 복사용 배포 폴더 생성 |

---

## 3. `inference_pi.py` 주요 기능

기존 Stage 4 스크립트는 더미 입력 1개를 넣어 ONNX Runtime 동작만 확인하는 성격이 강했다. Stage 5에서는 실제 테스트 CSV를 읽고 샘플별 지연 시간을 저장하도록 수정했다.

구현 내용:

- `project/data/real/test.csv`를 읽어 `sample_id` 기준으로 10 timestep 입력 생성
- ONNX Runtime `CPUExecutionProvider`로 `early_exit_fixed.onnx` 추론
- confidence threshold 기반 pseudo Early Exit 지점 선택
- 샘플별 결과 CSV 저장
- 평균/최소/최대/p50/p95 지연 시간 TXT 저장
- 레포 루트 실행과 Pi 번들 폴더 실행을 모두 지원
- ONNX Runtime 경고 로그를 줄이도록 `SessionOptions.log_severity_level = 3` 설정

---

## 4. PC Smoke Test 결과

Pi 실측 전 코드 동작 확인을 위해 PC에서 20개 샘플 기준 smoke test를 수행했다.

실행 명령:

```powershell
python project\scripts\inference_pi.py --max-samples 20 --output project\results\hojung\pi_inference_results_pc_smoke.csv
```

확인 결과:

| 항목 | 결과 |
|---|---:|
| sample_count | 20 |
| avg_inference_ms | 0.027100 ms |
| min_inference_ms | 0.017700 ms |
| max_inference_ms | 0.033700 ms |
| p95_inference_ms | 0.032655 ms |
| Exit 1 비율 | 40.00% |
| Exit 2 비율 | 0.00% |
| Exit 3 비율 | 60.00% |

주의: 이 값은 PC smoke test 결과이므로 최종 보고서의 Raspberry Pi 실측값으로 사용하지 않는다. Pi에서 다시 실행한 결과를 최종값으로 기록해야 한다.

---

## 5. Raspberry Pi 배포 번들 생성

실행 명령:

```powershell
python project\scripts\prepare_pi_bundle.py
```

생성 위치:

```text
project/deploy/raspberry_pi/
```

포함 파일:

| 파일 | 역할 |
|---|---|
| `early_exit_fixed.onnx` | Pi에서 실행할 ONNX Early Exit 모델 |
| `test.csv` | Pi 실측에 사용할 테스트 데이터 |
| `inference_pi.py` | Pi 실측 실행 스크립트 |
| `README.md` | Pi 내부 실행 명령어 |

---

## 6. Raspberry Pi에서 실행할 명령어

Pi로 `project/deploy/raspberry_pi/` 폴더를 복사한 뒤, Pi 터미널에서 해당 폴더로 이동한다.

```bash
cd ~/raspberry_pi
```

가상환경 생성:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

필수 패키지 설치:

```bash
pip install --upgrade pip
pip install onnxruntime numpy pandas
```

실측 실행:

```bash
python inference_pi.py --model early_exit_fixed.onnx --data test.csv --output pi_inference_results.csv --max-samples 100
```

결과 확인:

```bash
cat pi_inference_results.txt
head pi_inference_results.csv
```

---

## 7. PC로 가져와야 할 파일

Pi에서 실행 후 아래 2개 파일을 PC 레포지토리의 `project/results/hojung/` 폴더에 복사한다.

```text
pi_inference_results.csv
pi_inference_results.txt
```

이후 PC vs Pi 비교표를 완성한다.

---

## 8. 진행 상태

| 항목 | 상태 |
|---|---|
| Pi 실측용 `inference_pi.py` 구현 | 완료 |
| Pi 배포 번들 생성 스크립트 구현 | 완료 |
| PC smoke test | 완료 |
| Raspberry Pi OS 설치 | 사용자 진행 필요 |
| Pi에 번들 복사 | 사용자 진행 필요 |
| Pi에서 `pi_inference_results.csv` 생성 | 사용자 진행 필요 |
| PC vs Pi 비교표 작성 | Pi 결과 수신 후 진행 |

---

## 9. 다음 단계

Pi에서 `pi_inference_results.csv`, `pi_inference_results.txt`가 생성되면 그 파일을 `project/results/hojung/`에 넣고, Stage 5 최종 비교표를 작성한다.
