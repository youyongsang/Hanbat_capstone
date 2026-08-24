# Early Exit LSTM 기반 AP 트래픽 혼잡 감지 시스템 (2학기, ap_metrics_v2)

> GL.iNet Opal(GL-SFT1200) AP 실측 데이터를 이용해 Early Exit LSTM으로 무선망 혼잡 수준을 실시간 분류하는 캡스톤 프로젝트

**컴퓨터공학과 | 장예나 · 유용상 · 김호중**

---

## 브랜치 안내

이 브랜치(`capstoneDesign2`)는 `yongsang` 브랜치에서 분기해 코드를 정리한 브랜치다.

- **제외**: 1학기 4-feature 학습/평가/ONNX 파이프라인 코드(`train.py`, `train_early_exit.py`, `export_onnx*.py` 등)와 1차 실측(`ap_cleaned_strict`, 인터넷 공개 데이터 기반) 전체 파이프라인 — 해당 자료가 필요하면 `yongsang` 브랜치를 참고한다.
- **포함**: 2학기 실측 데이터 라인(`ap_metrics_v2`, 아래 "실행 순서" 기준) + docs 문서 전체(팀원별 가이드라인·work log 포함) + 1학기 Raspberry Pi 실측 결과(`project/results/hojung/`, `project/results/final_figures/`, `project/deploy/raspberry_pi/`, `origin/hojung`에서 가져옴 — staged ONNX 기준 Fixed/Dynamic Early Exit의 Pi 실측 지연 비교).

---

## 프로젝트 개요

본 프로젝트는 AP(무선 공유기) 트래픽 혼잡 상태를 LSTM 기반 시계열 모델로 분류하고, Early Exit 구조를 적용해 엣지 장비에서도 빠르게 추론할 수 있도록 설계한 시스템이다. 혼잡 라벨은 4단계(0 정상 / 1 경고 / 2 혼잡 / 3 심각)이며, `congestion_score`(throughput·occupancy·retry·jitter 가중합)로 산출한다.

입력 feature는 9개다: `throughput_mbps`, `channel_occupancy_percent`, `latency_ms`, `jitter_ms`, `tx_retries_delta`, `tx_failed_delta`, `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm`.

자세한 라벨 기준, 데이터 계보, 재현 명령어는 `project/README_AP_V2.md`와 `CLAUDE.md`를 참고한다.

---

## 프로젝트 구조

```text
project/
├── checkpoints/ap_v2/                 Early Exit LSTM 체크포인트
├── data/ap_metrics_v2/                windowed train/val/test, scaler
├── models/
│   ├── ap_early_exit_lstm.py
│   └── early_exit_lstm.py
├── results/yongsang/
│   ├── ap_v2_eval_report.txt
│   └── ap_v2_mismatched_scaler_diagnostic.txt
├── scripts/
│   ├── collect_metrics.py             AP 라이브 수집(+congestion_score 계산)
│   ├── relabel_metrics_v2.py          가중치 변경 시 raw CSV 재라벨링
│   ├── prepare_ap_metrics_dataset.py  windowed 변환
│   ├── train_ap_early_exit.py
│   ├── evaluate_ap_early_exit.py
│   └── metrics_v2.csv                 누적 raw 실측 데이터
└── utils/
    ├── ap_features.py
    ├── ap_dataloader.py
    └── metrics.py
```

---

## 설치 방법

```powershell
git clone https://github.com/youyongsang/Hanbat_capstone.git
cd Hanbat_capstone
git switch capstoneDesign2
```

torch/pandas/numpy는 별도 conda 환경에 설치해야 한다(anaconda base에서 torch DLL 로딩 실패 이슈 있음).

```powershell
conda create -n capstone python=3.11
conda activate capstone
pip install torch pandas numpy
```

---

## 실행 순서

데이터 변환:

```powershell
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2.csv --out-dir project\data\ap_metrics_v2 --overwrite
```

가중치 변경 시 재라벨링(재수집 불필요, 이후 위 변환 명령을 다시 실행해야 반영됨):

```powershell
python project\scripts\relabel_metrics_v2.py
```

Early Exit LSTM 학습:

```powershell
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint-dir project\checkpoints\ap_v2 --epochs 50 --batch-size 32 --class-weight-power 1.0
```

평가:

```powershell
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_v2 --checkpoint project\checkpoints\ap_v2\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_v2_eval_report.txt
```

---

## 팀 역할

| 팀원 | 담당 영역 |
|---|---|
| 장예나 | 데이터 및 시나리오 |
| 유용상 | 모델 설계, AP 실측 |
| 김호중 | 경량화 및 배포 |

---

## 알려진 한계

- Label 3(심각) 샘플이 아직 얇아 recall이 세션마다 크게 흔들린다.
- ONNX/INT8/Raspberry Pi 배포 파이프라인이 아직 이 라인에는 없다(1차 `raspberry_pi_ap` 번들은 `yongsang` 브랜치에만 존재).
- AP 하드웨어가 특정 조건에서 반복 크래시하는 문제가 있다. 원인 분석은 `docs/yongsang/ap_crash_analysis.md` 참고.
