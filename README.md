# Early Exit LSTM 기반 AP 트래픽 혼잡 감지 (2학기)

> GL.iNet Opal(GL-SFT1200) AP에서 직접 수집한 실측 데이터로, 채널 상태 시계열만 보고 무선망 혼잡도를
> 4단계로 분류하는 Early Exit LSTM. 엣지(Raspberry Pi)에서 sub-ms 추론.

**컴퓨터공학과 | 장예나 · 유용상 · 김호중**  ·  브랜치 `capstoneDesign2`

---

## 지금 상태 (2026-08-30)

| | |
|---|---|
| 데이터 | `ap_metrics_v2_redesign2` — 실측 2115행, **7 feature**, `max(표준 앵커)` + victim 프로브 라벨 |
| 정확도 (test 310창, 5시드 평균) | Baseline 92.0% · SDN 90.4% · **Proposed Early Exit 90.7%** |
| Pi INT8 지연 | Baseline 0.75 · SDN 0.57 · **Proposed 0.54 ms** (전부 &lt;1ms) |
| 라이브 감지 | Pi에서 실제 AP 부하 → 정상/경고/혼잡/심각 실시간 추적 검증 완료 |
| 발표 목표 | 지연 &lt;1ms **달성** · 정확도 95% **미달**(90~92%, 남은 숙제) |

라벨: 0 정상 / 1 경고 / 2 혼잡 / 3 심각.
입력 feature (7): `throughput_mbps`, `channel_occupancy_percent`, `tx_retry_ratio`, `rssi_dbm`, `rssi_delta_db`, `rssi_moving_avg_dbm`, `sta_tx_bitrate_mean` (정본: `project/utils/ap_features.py`).

---

## 어디부터 보나

| 먼저 | |
|---|---|
| **`CLAUDE.md`** | 전체 맥락·데이터 계보·현재 수치 |
| **`.work-log/current.md`** | 세션별 최신 진행 (CLAUDE.md보다 최신) |
| **`docs/README.md`** | **"뭘 알고 싶을 때 어느 문서를 보나" — 질문별 안내** |
| `docs/yongsang/capstone2_vacation_summary.html` | 방학 개발 흐름 한 장 요약 |

"왜 이런 라벨을 정했나" → `docs/yongsang/congestion_label_redesign.{md,html}`.

---

## 구조

```text
project/
├── utils/ap_features.py              7-feature 계약 (정본)
├── models/
│   ├── ap_early_exit_lstm.py         Proposed (entropy θ, fixed/dynamic)
│   ├── ap_baseline_lstm.py           Baseline (EE 없음, 정확도 상한 기준)
│   └── sdn_lstm.py                   SDN 비교 모델 (Kaya et al. ICML 2019)
├── scripts/
│   ├── collect_metrics.py            AP 라이브 수집 (SSH poller + victim 프로브 + 라벨)
│   ├── remeasure_redesign.py         raw feature → sub-score·라벨 재계산
│   ├── prepare_ap_metrics_dataset.py windowed train/val/test 변환
│   ├── train_ap_{early_exit,baseline_lstm,sdn}.py
│   ├── evaluate_ap_{early_exit,baseline_lstm,sdn}.py
│   ├── export_onnx_ap*.py            staged → unified If 노드 → INT8 재조립
│   ├── generate_ap_comparison.py     Baseline/SDN/Proposed 비교표
│   ├── forecast_eval_redesign.py     조기경보 프레이밍
│   └── live_congestion.py            실시간 혼잡 감지 라이브 추론 루프
├── demo/                             데모 웹 대시보드 (README.md 실행 / API.md 구현 스펙)
├── data/ap_metrics_v2_redesign2/     windowed 데이터 + scaler
├── checkpoints/ap_v2_redesign2/      배포 체크포인트 + ONNX
├── deploy/raspberry_pi_ap_v2/        Pi 번들
└── results/yongsang/                 평가·비교·Pi 지연·라이브 실측 리포트
```

---

## 실행

```powershell
# 환경 (anaconda base는 torch DLL 로딩 실패 — 별도 env 필요)
conda create -n capstone python=3.11 && conda activate capstone
pip install torch pandas numpy onnx onnxruntime

# 라벨 재계산 → windowed 변환 → 학습 → 평가
python project\scripts\remeasure_redesign.py
python project\scripts\prepare_ap_metrics_dataset.py --input project\scripts\metrics_v2_pi_redesign2_relabeled.csv --out-dir project\data\ap_metrics_v2_redesign2 --overwrite
python project\scripts\train_ap_early_exit.py --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint-dir project\checkpoints\ap_v2_redesign2 --epochs 50 --batch-size 32 --seed 0
python project\scripts\evaluate_ap_early_exit.py --data-dir project\data\ap_metrics_v2_redesign2 --checkpoint project\checkpoints\ap_v2_redesign2\ap_early_exit_lstm_best.pth --output project\results\yongsang\ap_v2_redesign2_eval_report.txt
```

전체 명령어: `docs/terminal_command_guide.md`.

데모: `python project\demo\demo_server.py` → <http://localhost:8000/>

---

## 팀 역할

| 팀원 | 담당 |
|---|---|
| 장예나 | 데이터 및 시나리오 |
| 유용상 | 모델 설계, AP 실측 |
| 김호중 | 경량화 및 배포 |

---

## 남은 것

- **Pi 정확도 95%** (현재 90~92%) — label 2 경계 노이즈 / label 3 관측성 한계
- 데모 대시보드 팀 구현 (`project/demo/API.md`)
- 밴드 스티어링 — 발표 슬라이드 7의 최종 목표 (혼잡 판단 → 채널 전환 명령 후보 생성)
- AP 반복 크래시: 신호 비대칭 시 저부하에도 크래시 (`docs/yongsang/ap_crash_analysis.md`)
