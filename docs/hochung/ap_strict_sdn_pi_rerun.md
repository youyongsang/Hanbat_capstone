# AP strict SDN Raspberry Pi 재측정 가이드 (호중 전용)

호중이 보내준 `Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx`를 확인했고, Baseline / Proposed Fixed / Proposed Dynamic 3개는 이미 정확하고 최종본으로 써도 된다. **SDN-style 행만** 아래 두 가지를 바꿔서 재측정하면 된다. 파이프라인 자체(export → quantize → Pi 실행 → 분석)는 호중이 이미 쓰던 방식 그대로 쓰면 되고, 새 스크립트로 갈아탈 필요는 없다.

## 무엇이 왜 바뀌어야 하는가

리포트에 적힌 SDN 실행 조건 두 가지가 최신 기준과 다르다.

| 항목 | 호중이 이번에 쓴 값 | 바꿔야 할 값 | 이유 |
|---|---|---|---|
| SDN 백본 | Proposed Early Exit LSTM의 내부 classifier를 재사용 | **독립적으로 새로 학습한 SDN 전용 백본** (`ap_sdn_lstm_best.pth`) | SDN을 "Early Exit 재사용 + confidence 정책"으로만 흉내내면 진짜 SDN baseline이 아니고, Proposed 모델과 사실상 같은 가중치라 공정 비교가 아님 |
| Confidence threshold | 0.50 | **0.85** | 1학기 SDN 논문 baseline과 `ap_sdn_lstm_best.pth` 학습 시 기본값이 0.85. threshold를 0.50으로 낮추면 Exit1에서 98.8%가 종료되는 착시가 생김 |

Baseline / Fixed / Dynamic은 체크포인트도 threshold도 안 바뀌었으니 손댈 필요 없음.

## 호중 파이프라인에서 바꿀 부분 (개념적으로)

정확한 파일명은 호중 로컬에만 있어서 모르지만, 구조상 아래 두 곳만 고치면 된다.

1. **ONNX export 스크립트** (`export_ap_baseline_onnx.py`류의 SDN 버전)에서 PyTorch 체크포인트를 로드하는 부분을
   ```
   기존: (Early Exit checkpoint 또는 임시 SDN 로직)
   변경: project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth
   ```
   로 바꾸고 다시 export + quantize 실행. 이 체크포인트는 이미 `yongsang` 브랜치에 커밋되어 있으니 `git pull` 후 바로 로드 가능.

2. **Pi 추론 스크립트**에서 confidence threshold 상수를
   ```
   기존: 0.50
   변경: 0.85
   ```
   로 바꾸기.

이 두 가지 외에 나머지(FP32/INT8 변환 방식, `test.csv` 82 샘플, 반복 측정 방식, 결과 CSV 포맷)는 호중이 하던 그대로 유지하면 된다.

## 절차

1. `git pull origin yongsang` — `ap_sdn_lstm_best.pth` 받기
2. 본인 ONNX export 스크립트에서 SDN 체크포인트 경로만 `ap_sdn_lstm_best.pth`로 교체 → 다시 export → 다시 quantize (FP32 + INT8, staged 3단계 그대로)
3. 본인 Pi 추론 스크립트에서 confidence threshold `0.50` → `0.85`로 교체
4. Pi에서 SDN FP32 / INT8 다시 측정 (나머지 3개 모델은 이번엔 안 돌려도 됨)
5. 결과를 `project/results/yongsang/`에 CSV/분석 파일로 넣고 커밋 → 최종 비교표(`ap_model_comparison_cleaned_strict.*`)의 SDN 행 교체

## 결과 검증 방법

새 백본은 confidence가 훨씬 보수적으로 나오므로, threshold를 0.85로 올리면 Exit1 비율이 기존 98.8%에서 크게 줄어드는 게 정상이다 (PC 비교표 기준 대략 Exit1 12% / Exit2 31% / Exit3 57% 근방). 만약 재측정 결과도 여전히 Exit1이 90%대로 나온다면 체크포인트를 잘못 로드했을 가능성이 높으니 1번부터 다시 확인한다.

## 참고 — 새로 짜인 참고용 스크립트/번들

호중이 본인 파이프라인 대신 참고하고 싶으면 아래도 `yongsang` 브랜치에 있다. 필수는 아니다.

- `project/scripts/export_onnx_ap_sdn.py`, `project/scripts/export_onnx_int8_ap.py` — SDN 포함 4개 모델 전체 ONNX/INT8 export
- `project/scripts/inference_pi_ap.py`, `project/deploy/raspberry_pi_ap/` — Pi 실행 번들 (이미 export까지 끝난 ONNX 파일 포함)
