# 유용상 컴퓨터 사양 및 실행 환경 정보

## 기준

이 문서는 `yongsang` 브랜치에서 Early Exit LSTM 및 4개 방식 비교 실험을 실행한 환경을 기록한다.

| 항목 | 값 |
|---|---|
| 브랜치 | `yongsang` |
| 기록 시점 커밋 | `10db434` |
| 작업 경로 | `C:\Capstone-Design` |

## 컴퓨터 사양

| 항목 | 값 |
|---|---|
| PC 이름 | `DESKTOP-5A9LEGQ` |
| 메인보드 제조사 | `ASRock` |
| 시스템 모델 | `A620M-HDV/M.2` |
| 시스템 타입 | `x64-based PC` |
| 운영체제 | `Microsoft Windows 11 Pro` |
| OS 버전 | `10.0.26200 Build 26200` |
| BIOS | `American Megatrends International, LLC. 2.02 (2023-11-17)` |
| CPU | `AMD Ryzen 5 7500F 6-Core Processor` |
| CPU 코어 / 스레드 | `6 cores / 12 logical processors` |
| CPU 최대 클럭 | `3701 MHz` |
| RAM | `32,376 MB` |
| GPU | `NVIDIA GeForce RTX 4060 Ti` |
| GPU 드라이버 | `32.0.15.9186` |
| 네트워크 어댑터 | `Realtek PCIe GbE Family Controller` |
| 시간대 | `Asia/Seoul (UTC+09:00)` |

## 실험 실행 장치

| 항목 | 값 |
|---|---|
| PyTorch CUDA 사용 가능 여부 | `False` |
| 실험 실행 장치 | CPU |

> GPU는 장착되어 있지만, 현재 PyTorch 런타임은 CPU 빌드(`torch 2.12.0+cpu`)이므로 실험은 CPU 기준으로 실행했다.

## Python 런타임

| 항목 | 값 |
|---|---|
| Python 실행 파일 | `C:\Users\PC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` |
| Python 버전 | `3.12.13` |
| 빌드 정보 | `MSC v.1944 64 bit (AMD64)` |

> 기본 Anaconda Python에서는 PyTorch DLL 로딩 문제가 있어, 실험은 Codex 번들 Python 런타임에서 수행했다.

## 주요 패키지

| 패키지 | 버전 |
|---|---|
| `torch` | `2.12.0+cpu` |
| `numpy` | `2.3.5` |
| `pandas` | `3.0.1` |
| `psutil` | `6.0.0` |
| `pip` | `26.0.1` |
| `setuptools` | `70.2.0` |

## 실행한 주요 명령

Baseline LSTM 학습:

```bash
python project/scripts/train.py --epochs 50
```

Baseline LSTM 평가:

```bash
python project/scripts/evaluate.py
```

Early Exit LSTM 학습:

```bash
python project/scripts/train_early_exit.py --epochs 50
```

Early Exit 고정 threshold / 동적 threshold 비교:

```bash
python project/scripts/evaluate_early_exit.py --output project/results/yongsang/early_exit_stage2_comparison_report.txt
```

4개 방식 비교:

```bash
python project/experiments/compare_baselines.py
```

## 결과 파일

| 파일 | 설명 |
|---|---|
| `project/results/hojung/baseline_eval_report.txt` | 일반 LSTM 단독 평가 결과 |
| `project/results/yongsang/early_exit_eval_report.txt` | Early Exit 단독 평가 결과 |
| `project/results/yongsang/early_exit_stage2_comparison_report.txt` | Early Exit 고정/동적 threshold 비교 |
| `project/results/hojung/comparison_summary.csv` | 4개 방식 비교 CSV |
| `project/results/hojung/comparison_summary.txt` | 4개 방식 비교 텍스트 리포트 |

## 측정 관련 주의

- 추론 시간은 CPU 기준으로 측정했다.
- `project/results/hojung/comparison_summary.csv`의 `Avg_Inference(ms)`는 실제 wall-clock 측정값이므로 실행 시점의 시스템 부하에 따라 조금 변동될 수 있다.
- `project/results/yongsang/early_exit_stage2_comparison_report.txt`에는 Exit 1/2/3에 대한 시뮬레이션 시간 기준도 함께 기록되어 있다.
- 최종 보고서에는 같은 컴퓨터, 같은 데이터, 같은 checkpoint 기준으로 한 번에 측정한 결과를 사용하는 것이 좋다.

## 수집 제한

초기 수집 시 일반 권한에서는 일부 하드웨어 조회가 제한되었고, 이후 권한 상승 실행으로 CPU, RAM, GPU, OS 정보를 확인했다.
