# 김호중 방학 4단계 가이드라인
## 동적 θ 경량화 비교 및 실시간 파이프라인 구현

> 담당자: 김호중  
> 기간: 방학 6~7주차  
> 목표: 경량 동적 θ 비교 실험 수행, 실시간 추론 파이프라인 구현  
> 완료 기준: dynamic_threshold_comparison.csv 저장, realtime_inference.py 동작 확인, realtime_inference_results.csv 저장

---

## 1. 6주차 — 동적 θ 경량화 비교 실험 수행

유용상 경량 동적 θ 모델 받아서 비교 실험 수행.

| 항목 | 고정 θ | 기존 동적 θ | 경량 동적 θ |
|---|---|---|---|
| 정확도 | | | |
| 추론 시간 | | | |
| Exit 1 종료율 | | | |

결과 저장:
```
project/results/hojung/dynamic_threshold_comparison.csv
```

### 완료 기준
- [ ] 경량 동적 θ 비교 실험 완료
- [ ] 결과 CSV 저장 완료

---

## 2. 7주차 — 실시간 추론 파이프라인 구현

### 핵심 목표

```
실제 AP → raw 지표 수집 → 예나 변환 규칙 적용 → Pi 추론 → 채널 전환 방향 제시
```

실시간 추론에서도 CSV 파일을 읽지 않을 뿐, 피처 변환 기준은 예나가 만든 오프라인 데이터셋과 동일해야 한다.

```
raw metric:
throughput_mbps, channel_occupancy, packet_loss, latency

model feature:
rps, channel_occupancy, packet_loss, latency
```

`throughput_mbps → rps` 매핑 공식과 Min-Max 정규화 기준은 예나가 저장한 `scaler_params.json`을 사용한다.

### 실시간 파이프라인 스크립트

```python
# realtime_inference.py
import onnxruntime as ort
import numpy as np
import pandas as pd
import time
import subprocess
import json
from collections import deque
from pathlib import Path

FEATURE_COLUMNS = ["rps", "channel_occupancy", "packet_loss", "latency"]
RESULT_PATH = Path('project/results/hojung/realtime_inference_results.csv')
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

session = ort.InferenceSession('early_exit_fixed.onnx')
scaler = json.load(open('scaler_params.json', encoding='utf-8'))
window = deque(maxlen=10)

CHANNEL_ACTION = {
    0: "채널 유지 (Keep Current Channel)",
    1: "모니터링 강화 (Enhance Monitoring)",
    2: "채널 전환 (Switch Channel)",
    3: "즉시 전환 / 5GHz 이동 (Emergency Switch)"
}

def scale_throughput_to_rps(throughput_mbps):
    # 예나가 확정한 변환 기준을 적용
    max_throughput = scaler["throughput_mbps"]["max"]
    return min(max(throughput_mbps / max_throughput * 1000.0, 0.0), 1000.0)

def normalize_feature(name, value):
    params = scaler[name]
    return (value - params["min"]) / (params["max"] - params["min"])

def convert_live_metrics(raw):
    feature_values = {
        "rps": scale_throughput_to_rps(raw["throughput_mbps"]),
        "channel_occupancy": raw["channel_occupancy"],
        "packet_loss": raw["packet_loss"],
        "latency": raw["latency"],
    }
    return [
        normalize_feature(name, feature_values[name])
        for name in FEATURE_COLUMNS
    ]

print("실시간 혼잡 감지 시작...")
print("-" * 60)

results = []

while True:
    # 실시간 raw 지표 수집 후, 예나의 피처 변환 규칙 적용
    raw_metrics = collect_wifi_metrics()
    metrics = convert_live_metrics(raw_metrics)
    window.append(metrics)

    if len(window) == 10:
        input_data = np.array(window).reshape(1, 10, 4).astype(np.float32)

        start = time.perf_counter()
        output = session.run(None, {'input': input_data})
        elapsed = (time.perf_counter() - start) * 1000

        label = np.argmax(output[0])
        confidence = np.max(output[0])

        print(f"[{time.strftime('%H:%M:%S')}] "
              f"혼잡 수준: Label {label} | "
              f"신뢰도: {confidence:.2f} | "
              f"추론: {elapsed:.3f}ms | "
              f"→ {CHANNEL_ACTION[label]}")

        results.append({
            "timestamp": time.strftime('%H:%M:%S'),
            "predicted_label": int(label),
            "confidence": float(confidence),
            "inference_ms": elapsed,
            "action": CHANNEL_ACTION[label],
        })

        # 데모 중간에도 결과를 잃지 않도록 주기적으로 저장
        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(RESULT_PATH, index=False)

    time.sleep(0.01)  # 10ms 간격
```

### 완료 기준
- [ ] `realtime_inference.py` 구현 완료
- [ ] `scaler_params.json` 기반 실시간 피처 변환 적용 완료
- [ ] 오프라인 CSV 전처리와 실시간 변환 결과가 같은 기준인지 확인 완료
- [ ] 실시간 파이프라인 동작 확인
- [ ] `project/results/hojung/realtime_inference_results.csv` 저장 완료
- [ ] 추론 시간 실측 완료
- [ ] 유용상에게 정확도 검증 요청

---

## 3. 주의사항

- 실시간 데모는 발표 때 보여줄 수 있도록 안정적으로 동작해야 함. 미리 여러 번 테스트.
- Pi 온도 높으면 성능 저하. 서멀 쓰로틀링 주의.
- 보고서 수치는 실험 완료 후 채울 것.
- 추론 시간은 항상 `time.perf_counter()` 사용.
- 실시간 추론 입력도 반드시 `(1, 10, 4)`이며 피처 순서는 `rps, channel_occupancy, packet_loss, latency`로 고정.
