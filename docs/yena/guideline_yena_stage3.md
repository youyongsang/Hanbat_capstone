# 장예나 3단계 가이드라인
## 실험 데이터 관리 및 결과 로깅

> 담당자: 장예나  
> 목표: 비교 실험에 필요한 데이터 관리 및 실험 결과 시각화  
> 완료 기준: 실험 결과 CSV 및 그래프가 `results/` 폴더에 저장되면 됨

---

## 1. 해야 할 일 순서

```
1. 시나리오별 테스트 데이터 분리
2. 실험 결과 로깅 구현
3. 결과 시각화 구현
4. 시나리오별 분석 지원
```

---

## 2. 시나리오별 테스트 데이터 분리

비교 실험에서 전체 정확도뿐만 아니라 시나리오별 분석을 하려면  
테스트 데이터에 시나리오 레이블이 있어야 한다.

### 추가할 컬럼

```
timestamp, rps, channel_occupancy, packet_loss, latency, label, scenario
```

| scenario 값 | 의미 |
|---|---|
| 0 | 일과 시작 (점진적 급증) |
| 1 | 긴급 증산 (갑작스러운 폭증) |
| 2 | 점심 재가동 (주기적 패턴) |
| 3 | 불균형 부하 (지속적 혼잡) |

### 저장 경로

```
data/real/
├── train.csv
├── val.csv
├── test.csv               ← 기존
└── test_with_scenario.csv ← 시나리오 컬럼 추가 버전
```

### 생성 방법

`generate_test_with_scenario.py` 스크립트를 실행하면 자동으로 생성된다.

```bash
python project/scripts/generate_test_with_scenario.py
```

스크립트 내부에서 하는 일:
1. `data/real/test.csv` 읽기
2. `scenario` 문자열 컬럼을 숫자 `scenario_id`로 매핑
3. `data/real/test_with_scenario.csv` 로 저장

| scenario 문자열 | scenario_id |
|---|---|
| startup_surge | 0 |
| emergency_ramp | 1 |
| lunch_restart | 2 |
| imbalanced_ap_load | 3 |

---

## 3. 실험 결과 로깅

### `utils/logger.py` 구현

실험 결과를 CSV로 저장하거나 읽어 확인하는 보조 유틸리티 함수.
최종 4개 방식 비교 결과는 김호중 컴퓨터에서 `compare_baselines.py`를 실행해
생성된 `project/results/hojung/comparison_summary.csv`를 기준으로 사용한다.
예나 단계에서는 이 결과를 다시 계산하지 않고, 시각화와 시나리오별 분석에 활용한다.

```python
def save_results(
    model_name: str,
    accuracy: float,
    inference_time: float,
    exit_rates: dict[int, float] | None,
    save_path: str,
    unnecessary_switch_rate: float = 0.0,
) -> None:
    """
    Args:
        model_name: 모델 이름
        accuracy: 정확도 비율값 (0~1)
        inference_time: 평균 추론 시간 (ms)
        exit_rates: Early Exit 종료율 dict. 일반 모델은 None
        save_path: 저장 경로
        unnecessary_switch_rate: 불필요한 채널 전환 비율
    """
```

제공 함수 3가지:

| 함수 | 역할 |
|---|---|
| `save_results(...)` | 한 모델의 결과를 CSV에 append |
| `load_summary(path)` | 저장된 CSV를 행 리스트로 불러오기 |
| `print_summary(path)` | CSV 내용을 터미널에 표 형태로 출력 |

### 저장 형식

```
results/
├── baseline_threshold.csv
├── baseline_lstm.csv
├── early_exit_fixed.csv
├── early_exit_dynamic.csv
└── comparison_summary.csv
```

`comparison_summary.csv` 컬럼:

```
model, accuracy, avg_inference_ms, exit1_rate, exit2_rate, exit3_rate, unnecessary_switch_rate
```

---

## 4. `comparison_summary.csv` 사용 방법

> **목적:** 4개 방식(임계값·Baseline LSTM·EE 고정 θ·EE 동적 θ)의 실험 결과를  
> 하나의 CSV로 받아 시각화와 분석에 사용한다.  
> 이 파일은 **김호중 비교 실험 실행 결과**를 기준으로 한다.

### 관련 파일 위치

```
project/
├── experiments/
│   └── compare_baselines.py              ← 4개 방식 비교 실험 실행
└── results/
    └── hojung/
        └── comparison_summary.csv        ← 최종 비교 결과 원본
```

### 생성 방법

김호중 컴퓨터 기준으로 아래 스크립트를 실행하면 자동 생성된다.

```bash
python project/experiments/compare_baselines.py
```

생성 파일:

```text
project/results/hojung/comparison_summary.csv
project/results/hojung/comparison_summary.txt
```

### 결과 확인

```bash
type project\results\hojung\comparison_summary.csv
```

### 컬럼 설명

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `Method` | str | 비교 방식 이름 |
| `Accuracy(%)` | float | 분류 정확도 (%) |
| `Avg_Inference(ms)` | float | 평균 추론 지연 (ms) |
| `Unnecessary_Switches` | int | 불필요한 채널 전환 횟수 |
| `Acc_Label_0~3(%)` | float | label별 정확도 (%) |
| `Exit1~3(%)` | float | Early Exit 방식의 exit별 종료율 (%) |

---

## 5. 결과 시각화

### 만들어야 할 그래프 3가지

**그래프 1 — 4개 방식 정확도 비교 막대그래프**
```
X축: 4개 방식
Y축: 분류 정확도 (%)
```

**그래프 2 — 정확도 vs 추론 시간 산점도**
```
X축: 평균 추론 시간 (ms)
Y축: 분류 정확도 (%)
각 점: 4개 방식
→ 오른쪽 위가 좋은 것 (빠르고 정확)
→ 왼쪽 위가 이상적 (빠르면서 정확)
```

**그래프 3 — Exit별 종료율 파이차트**
```
Exit 1 / Exit 2 / Exit 3 비율
고정 θ vs 동적 θ 나란히 비교
```

### 저장 경로

```
results/
├── accuracy_comparison.png
├── accuracy_vs_latency.png
└── exit_rate_comparison.png
```

---

## 6. 시나리오별 분석 지원

유용상이 시나리오별 분석을 할 수 있도록  
시나리오 분리 데이터를 제공하고 분석 결과를 취합한다.

### 시나리오별 분석 결과 저장

```
results/
└── scenario_analysis/
    ├── scenario_0_gradual.csv    # 일과 시작
    ├── scenario_1_spike.csv      # 긴급 증산
    ├── scenario_2_periodic.csv   # 점심 재가동
    └── scenario_3_imbalance.csv  # 불균형 부하
```

---

## 7. 완료 기준 체크리스트

- [x] `test_with_scenario.csv` 생성 완료 (시나리오 컬럼 추가)
- [x] `utils/logger.py` 구현 완료
- [ ] `project/results/hojung/comparison_summary.csv` 수신 및 확인
- [ ] 정확도 비교 막대그래프 생성 완료
- [ ] 정확도 vs 추론 시간 산점도 생성 완료
- [ ] Exit 종료율 파이차트 생성 완료
- [ ] 시나리오별 분석 데이터 분리 완료
- [ ] 김호중·유용상에게 시각화 결과 공유 완료

---

## 8. 주의사항

- 시각화는 **실험 결과 나온 후**에 만들 것. 예시 그래프는 만들지 않음.
- `comparison_summary.csv`는 김호중 컴퓨터에서 실행한 `compare_baselines.py` 결과를 기준으로 한다.
- 예나 단계에서는 `comparison_summary.csv`를 새로 계산하지 않고, 읽어서 그래프와 분석 자료를 만든다.
- 그래프는 발표 자료에 바로 쓸 수 있도록 해상도 높게 저장할 것 (`dpi=300`).
