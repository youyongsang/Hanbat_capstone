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

실험 결과를 CSV로 저장하는 유틸리티 함수.

```python
def save_results(results: list[dict], path: str) -> None:
    """
    Args:
        results: 모델별 결과 dict의 리스트
        path:    저장 경로 (예: 'project/results/yena/comparison_summary.csv')

    dict 필수 키:
        model                  - 모델 이름 (str)
        accuracy               - 정확도 (float, 0~1)
        avg_inference_ms       - 평균 추론 시간 (float, ms 단위)
        exit1_rate             - Exit 1 종료 비율 (float)
        exit2_rate             - Exit 2 종료 비율 (float)
        exit3_rate             - Exit 3 종료 비율 (float)
        unnecessary_switch_rate - 불필요한 채널 전환 비율 (float)
    """
```

제공 함수 3가지:

| 함수 | 역할 |
|---|---|
| `save_results(results, path)` | 결과 리스트를 CSV로 저장 |
| `load_summary(path)` | 저장된 CSV를 DataFrame으로 불러오기 |
| `print_summary(df)` | 터미널에 표 형태로 출력 |

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

## 4. `comparison_summary.csv` 생성 방법

> **목적:** 4개 방식(임계값·Baseline LSTM·EE 고정 θ·EE 동적 θ)의 실험 결과를  
> 하나의 CSV로 취합한다.  
> 이 파일은 **김호중·유용상 결과를 받은 후** 최종 합산·저장한다.

### 관련 파일 위치

```
project/
├── utils/
│   └── logger.py                          ← save_results() 함수 위치
├── scripts/
│   └── generate_summary.py               ← 아래 내용 그대로 실행
└── results/
    └── yena/
        └── comparison_summary.csv        ← 최종 저장 위치
```

### 방법 1 — 스크립트 실행

수치를 받은 후 `generate_summary.py`의 `results` 리스트를  
채워서 실행하면 CSV가 만들어진다.

```python
# project/scripts/generate_summary.py

from utils.logger import save_results

results = [
    {
        "model":                   "threshold",
        "accuracy":                0.422,   # 42.2%
        "avg_inference_ms":        0.003,
        "exit1_rate":              1.0,
        "exit2_rate":              0.0,
        "exit3_rate":              0.0,
        "unnecessary_switch_rate": 0.0,
    },
    {
        "model":                   "baseline_lstm",
        "accuracy":                0.949,   # 94.9%
        "avg_inference_ms":        0.482,
        "exit1_rate":              0.0,
        "exit2_rate":              0.0,
        "exit3_rate":              1.0,
        "unnecessary_switch_rate": 0.0,
    },
    {
        "model":                   "early_exit_fixed_theta",
        "accuracy":                0.957,   # 95.7%
        "avg_inference_ms":        0.390,
        "exit1_rate":              0.31,
        "exit2_rate":              0.28,
        "exit3_rate":              0.41,
        "unnecessary_switch_rate": 0.07,
    },
    {
        "model":                   "early_exit_dynamic_theta",
        "accuracy":                0.963,   # 96.3%
        "avg_inference_ms":        0.405,
        "exit1_rate":              0.29,
        "exit2_rate":              0.25,
        "exit3_rate":              0.46,
        "unnecessary_switch_rate": 0.04,
    },
]

save_results(results, "project/results/yena/comparison_summary.csv")
```

```bash
# 터미널에서 실행
python project/scripts/generate_summary.py
```

### 방법 2 — 평가 코드에 직접 연동

모델 평가 스크립트 끝에 아래 코드를 붙이면 평가 직후 자동 저장된다.

```python
from utils.logger import save_results

results = []
results.append({
    "model":                   "early_exit_dynamic_theta",
    "accuracy":                eval_accuracy,        # 평가 함수 반환값
    "avg_inference_ms":        avg_ms,
    "exit1_rate":              exit1_count / total,
    "exit2_rate":              exit2_count / total,
    "exit3_rate":              exit3_count / total,
    "unnecessary_switch_rate": switch_count / total,
})

save_results(results, "project/results/yena/comparison_summary.csv")
```

### 결과 확인

```python
from utils.logger import load_summary, print_summary

df = load_summary("project/results/yena/comparison_summary.csv")
print_summary(df)  # 터미널에 표 형태로 출력
```

### 컬럼 설명

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `model` | str | 모델 식별자 (예: `early_exit_dynamic_theta`) |
| `accuracy` | float | 분류 정확도 (0~1, 예: 0.963 = 96.3%) |
| `avg_inference_ms` | float | 평균 추론 지연 (ms 단위) |
| `exit1_rate` | float | Exit 1에서 종료된 비율 |
| `exit2_rate` | float | Exit 2에서 종료된 비율 |
| `exit3_rate` | float | Exit 3에서 종료된 비율 (exit1+2+3 합계 = 1.0) |
| `unnecessary_switch_rate` | float | 불필요한 채널 전환 발생 비율 |

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
- [] `comparison_summary.csv` 저장 확인
- [ ] 정확도 비교 막대그래프 생성 완료
- [ ] 정확도 vs 추론 시간 산점도 생성 완료
- [ ] Exit 종료율 파이차트 생성 완료
- [ ] 시나리오별 분석 데이터 분리 완료
- [ ] 김호중·유용상에게 시각화 결과 공유 완료

---

## 8. 주의사항

- 시각화는 **실험 결과 나온 후**에 만들 것. 예시 그래프는 만들지 않음.
- `comparison_summary.csv`는 김호중·유용상 결과를 취합하는 파일이야. 혼자 채우지 말고 두 명 결과 받아서 합칠 것.
- 그래프는 발표 자료에 바로 쓸 수 있도록 해상도 높게 저장할 것 (`dpi=300`).
