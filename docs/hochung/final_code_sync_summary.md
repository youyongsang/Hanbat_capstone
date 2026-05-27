# 김호중 최종 비교 실험 코드 동기화 정리

## 목적

최종 4개 방식 비교 실험은 김호중 브랜치와 김호중 실행 환경에서 수행한다.  
단, Early Exit LSTM 및 동적 threshold 구현은 유용상 담당 코드가 원본이므로, `yongsang` 브랜치의 최종 구현본을 `hojung` 브랜치에 동기화하여 같은 코드 기준으로 실험할 수 있게 정리했다.

## 동기화 기준

| 구분 | 기준 |
|---|---|
| 최종 실행 브랜치 | `hojung` |
| Early Exit 코드 원본 | `origin/yongsang` |
| 최종 실험 실행 환경 | 김호중 컴퓨터 |
| 학습 체크포인트 | 김호중 환경에서 새로 생성 |
| 실험 결과 파일 | `yongsang` 정리 구조를 반영하되, 최종 값은 김호중 환경에서 새로 생성 |

## 가져온 코드

아래 파일은 `origin/yongsang`의 통합 실험 코드 기준으로 `hojung` 브랜치에 반영했다.

| 파일 | 역할 | 담당 구분 |
|---|---|---|
| `project/experiments/channel_optimizer.py` | 분류 결과 기반 채널 제어 정책 | 김호중 Stage 2 |
| `project/experiments/compare_baselines.py` | 4개 방식 종합 비교 실험 프레임워크 | 김호중 Stage 2/3 + 유용상 모델 연결 |
| `project/scripts/evaluate.py` | 일반 Baseline LSTM 평가 | 김호중 Stage 1/2 |
| `project/scripts/train_early_exit.py` | Early Exit LSTM 학습 | 유용상 Stage 1/2 |
| `project/scripts/evaluate_early_exit.py` | 고정 threshold와 동적 threshold 비교 평가 | 유용상 Stage 2/3 |
| `project/scripts/validate_real_data.py` | 실제 데이터/DataLoader 검증 | 장예나 Stage 2 성격 |
| `project/utils/metrics.py` | Early Exit 평가 보조 지표 | 유용상 Stage 2 |
| `project/models/__init__.py` | 모델 import 편의용 공통 파일 | 공통 |

## 결과 파일 정리

`project/results`는 담당자별 폴더 구조가 맞아야 실행 결과가 섞이지 않으므로 `origin/yongsang`의 정리 구조를 반영했다.

| 경로 | 용도 |
|---|---|
| `project/results/hojung/` | 임계값, 일반 LSTM, 4개 방식 비교 결과 |
| `project/results/yena/` | 데이터 검증 결과 |
| `project/results/yongsang/` | Early Exit 단독 및 고정/동적 threshold 비교 결과 |

기존 루트 결과 파일 `project/results/baseline_eval_report.txt`는 중복을 막기 위해 제거하고, `project/results/hojung/baseline_eval_report.txt` 위치로 정리했다.

현재 포함된 결과 파일은 구조 확인과 이전 실험 참고용이며, 최종 성능표에 사용할 값은 김호중 환경에서 아래 실행 순서에 따라 다시 생성한다.

## 제외한 파일

아래 파일은 일부러 가져오지 않았다.

| 파일/경로 | 제외 이유 |
|---|---|
| `project/checkpoints/*.pth` | 최종 실험은 김호중 컴퓨터에서 새로 학습한 체크포인트로 수행해야 하므로 제외 |

## 최종 실험 실행 원칙

코드는 `yongsang` 브랜치의 Early Exit 구현을 포함한 통합 코드 기준으로 맞춘다.  
하지만 최종 성능표에 사용할 학습과 평가는 모두 김호중 환경에서 새로 실행한다.

즉, 보고서에는 다음과 같이 정리한다.

> Early Exit LSTM 및 동적 threshold 구현 코드는 유용상 브랜치의 최종 구현본을 기준으로 동기화하였고, 최종 4개 방식 비교 실험은 김호중 환경에서 동일 데이터셋과 동일 실행 코드로 수행하였다.

## 권장 실행 순서

프로젝트 루트에서 아래 순서로 실행한다.

```powershell
python project/scripts/train.py
python project/scripts/train_early_exit.py
python project/scripts/evaluate.py
python project/scripts/evaluate_early_exit.py
python project/experiments/compare_baselines.py
```

## 기대 산출물

김호중 환경에서 실행하면 아래 산출물이 새로 생성되어야 한다.

| 산출물 | 설명 |
|---|---|
| `project/checkpoints/baseline_lstm_best.pth` | 일반 LSTM 학습 체크포인트 |
| `project/checkpoints/early_exit_fixed.pth` | 고정 threshold Early Exit 체크포인트 |
| `project/checkpoints/early_exit_dynamic.pth` | 동적 threshold Early Exit 체크포인트 |
| `project/results/hojung/baseline_eval_report.txt` | 일반 LSTM 평가 리포트 |
| `project/results/yongsang/early_exit_stage2_comparison_report.txt` | Early Exit 고정/동적 비교 리포트 |
| `project/results/hojung/comparison_summary.csv` | 4개 방식 비교 CSV |
| `project/results/hojung/comparison_summary.txt` | 4개 방식 비교 텍스트 리포트 |

## 현재 상태

- `docs` 폴더는 `yongsang` 브랜치 기준으로 최신화 완료.
- 실험 실행 코드는 `origin/yongsang` 기준으로 `hojung` 브랜치에 동기화 완료.
- 체크포인트는 동기화하지 않음.
- `project/results`는 담당자별 폴더 구조로 정리 완료. 최종 값은 김호중 환경에서 재생성한다.
- 최종 비교 실험은 김호중 환경에서 재학습 및 재평가하는 방식으로 진행한다.
