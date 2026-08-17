# AP strict SDN Raspberry Pi 재측정 가이드

## 왜 다시 측정해야 하는가

`docs/hochung/Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx`의 "SDN-style Confidence-only EE" 행은 Early Exit 백본을 그대로 재사용한 임시 confidence-only 정책 기준이었다. `yongsang` 브랜치에 AP 9-feature 전용으로 독립적으로 재학습한 SDN 백본(`ap_sdn_lstm_best.pth`, SDN 논문 loss 가중치 0.15/0.30/0.55로 학습)이 새로 올라왔다. 이 백본은 Exit 분포가 완전히 다르므로(기존 98.8/1.2/0.0% → 신규 12.2/30.5/57.3%), **SDN 행만 Pi에서 다시 측정하면 된다.** Baseline / Proposed Fixed / Proposed Dynamic은 모델 자체가 바뀌지 않았으므로 기존 xlsx 값을 그대로 써도 된다.

## 사전 확인

| 항목 | 내용 |
|---|---|
| 기준 브랜치 | `yongsang` |
| 새 체크포인트 | `project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth` |
| ONNX 변환 여부 | **이미 완료됨.** PC에서 다시 export할 필요 없음 |
| FP32 ONNX | `project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage1.onnx` / `_stage2.onnx` / `_stage3.onnx` |
| INT8 ONNX | `project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage1_int8.onnx` / `_stage2_int8.onnx` / `_stage3_int8.onnx` |
| Pi 입력 데이터 | `project/data/ap_metrics_cleaned_strict/test.csv` (82 샘플, 9-feature — 1학기 351 샘플과 다름) |
| Pi 실행 스크립트 | `project/scripts/inference_pi_ap.py` (또는 `project/deploy/raspberry_pi_ap/inference_pi_ap.py`) |

기존 1학기용 `project/deploy/raspberry_pi/`, `inference_pi.py`와는 별개다. 4-feature `test.csv`와 9-feature `test.csv`를 섞으면 안 된다.

## 1. 최신 브랜치 받기

```bash
git checkout yongsang
git pull origin yongsang
```

이 시점에 위 표의 ONNX/INT8 파일이 전부 로컬에 생기는지 확인한다.

## 2. Pi로 옮길 파일 준비

아래 방법 중 편한 쪽으로 하면 된다.

**방법 A — 폴더 통째로 복사 (권장)**

`project/deploy/raspberry_pi_ap/` 폴더를 그대로 Pi로 복사한다. Baseline/Fixed/Dynamic ONNX까지 전부 들어있어서 나중에 4개 모델 전체를 재측정하고 싶어져도 추가 작업이 없다.

**방법 B — SDN 관련 파일만 복사**

Pi에 아래 6개 파일만 옮긴다.

```text
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage1.onnx
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage2.onnx
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage3.onnx
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage1_int8.onnx
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage2_int8.onnx
project/checkpoints/ap_cleaned_strict/ap_sdn_fixed_stage3_int8.onnx
```

여기에 `project/data/ap_metrics_cleaned_strict/test.csv`, `project/scripts/inference_pi_ap.py`, `project/scripts/analyze_pi_results.py`도 같은 폴더에 함께 넣는다.

**방법 C — Pi에서 직접 git pull**

Pi가 인터넷(GitHub) 접속이 가능하면 Pi에서 바로 `git clone`/`git pull`로 받아도 된다. 이 경우 `project/deploy/raspberry_pi_ap/` 폴더로 이동해서 아래 명령을 실행하면 된다.

## 3. Pi 환경 구성 (처음 한 번만)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pandas
```

## 4. SDN FP32 / INT8 측정

방법 A 또는 C로 `raspberry_pi_ap` 폴더 안에서 실행 중이면 아래 그대로 실행한다. 방법 B로 SDN 파일만 옮겼다면 `--stage1/--stage2/--stage3` 뒤 파일명을 옮긴 경로에 맞게 조정한다.

```bash
python inference_pi_ap.py --mode staged-confidence --stage1 ap_sdn_fixed_stage1.onnx --stage2 ap_sdn_fixed_stage2.onnx --stage3 ap_sdn_fixed_stage3.onnx --data test.csv --output pi_ap_sdn_fp32_results.csv --max-samples 82 --repeats 5

python inference_pi_ap.py --mode staged-confidence --stage1 ap_sdn_fixed_stage1_int8.onnx --stage2 ap_sdn_fixed_stage2_int8.onnx --stage3 ap_sdn_fixed_stage3_int8.onnx --data test.csv --output pi_ap_sdn_int8_results.csv --max-samples 82 --repeats 5
```

- `--mode staged-confidence`: SDN 논문 정책대로 각 exit에서 max softmax confidence가 threshold(기본 0.85) 이상이면 그 자리에서 종료. entropy 기준(Proposed Fixed/Dynamic)과는 다른 정책이므로 반드시 이 모드를 써야 한다.
- `--max-samples 82`은 AP strict test.csv 전체 샘플 수 기준이다.

## 5. 결과 분석

```bash
python analyze_pi_results.py --input pi_ap_sdn_fp32_results.csv --output-dir . --name pi_ap_sdn_fp32_analysis
python analyze_pi_results.py --input pi_ap_sdn_int8_results.csv --output-dir . --name pi_ap_sdn_int8_analysis
```

확인할 주요 항목은 `pi_ap_sdn_fp32_analysis.txt` / `pi_ap_sdn_int8_analysis.txt` 안의 `accuracy`, `avg_inference_ms`, exit별 비율이다. Exit3 비율이 기존 xlsx(0.0%)보다 훨씬 높게(대략 exit1 12%/exit2 30%/exit3 57% 근방) 나오는 게 정상이다 — 새 백본이 더 보수적으로 confidence를 주기 때문이다.

## 6. 결과 파일을 저장소로 복사

Pi에서 생성된 아래 4개 파일을 PC의 `project/results/yongsang/` 폴더에 넣는다.

```text
pi_ap_sdn_fp32_results.csv
pi_ap_sdn_fp32_analysis.*
pi_ap_sdn_int8_results.csv
pi_ap_sdn_int8_analysis.*
```

그다음 커밋해서 `yongsang` 브랜치에 올려주면 최종 비교표(`project/results/yongsang/ap_model_comparison_cleaned_strict.*`)에 Pi 실측 컬럼을 채워 넣을 수 있다.

## 참고 — Baseline/Fixed/Dynamic까지 다 같이 재측정하고 싶다면

방법 A로 폴더를 통째로 옮겼다면 추가 파일 준비 없이 바로 가능하다. `project/deploy/raspberry_pi_ap/README.md`에 4개 모델 × FP32/INT8 = 8개 명령이 전부 정리되어 있으니 그대로 따라 하면 된다. 다만 지금 당장 급한 건 SDN 행뿐이므로 필수는 아니다.
