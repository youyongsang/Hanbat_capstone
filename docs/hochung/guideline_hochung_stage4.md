# 김호중 4단계 가이드라인
## 경량화 적용 및 Raspberry Pi 배포 준비

> 담당자: 김호중  
> 목표: INT8 Quantization 적용 및 ONNX 변환, Raspberry Pi 배포 준비  
> 완료 기준: 경량화 전후 성능 비교 완료, ONNX 변환 파일 생성

---

## 1. 해야 할 일 순서

```
1. INT8 Quantization 적용
2. 경량화 전후 성능 비교
3. ONNX 변환
4. ONNX 추론 동작 확인
5. Raspberry Pi 배포 준비
```

---

## 2. INT8 Quantization

모델 크기와 추론 속도를 줄이기 위해 FP32 → INT8로 변환.

### 적용 대상

| 모델 | 파일 |
|---|---|
| 일반 LSTM | checkpoints/baseline_lstm_best.pth |
| Early Exit 고정 θ | checkpoints/early_exit_fixed_final.pth |

### Post-training Quantization

```python
import torch.quantization

# 모델 로드
model = BaselineLSTM(...)
model.load_state_dict(torch.load('checkpoints/baseline_lstm_best.pth'))
model.eval()

# Quantization 적용
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.LSTM, torch.nn.Linear},
    dtype=torch.qint8
)

# 저장
torch.save(model_quantized.state_dict(), 'checkpoints/baseline_lstm_quantized.pth')
```

---

## 3. 경량화 전후 비교

### 비교 항목

| 항목 | 측정 방법 |
|---|---|
| 모델 크기 (MB) | 파일 크기 측정 |
| 정확도 (%) | 테스트셋 동일 평가 |
| 추론 시간 (ms) | CPU 기준 100회 평균 |

### 목표치

| 항목 | 목표 |
|---|---|
| 모델 크기 | 1MB 이하 |
| 정확도 저하 | 2% 이내 |
| 추론 시간 | 경량화 전 대비 개선 |

### 결과 저장

```
results/
└── quantization_comparison.csv
```

컬럼:
```
model, original_size_mb, quantized_size_mb, 
original_accuracy, quantized_accuracy,
original_inference_ms, quantized_inference_ms
```

---

## 4. ONNX 변환

PyTorch 모델을 ONNX 형식으로 변환하여 Raspberry Pi에서 추론 가능하게 만든다.

### 변환 대상

경량화 후 가장 성능이 좋은 모델. 일반적으로 Early Exit 고정 θ 버전.

```python
import torch
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 10, 4)  # (batch, timestep, feature)

torch.onnx.export(
    model,
    dummy_input,
    'checkpoints/early_exit_fixed.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

### 변환 후 동작 확인

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('checkpoints/early_exit_fixed.onnx')
dummy = np.random.randn(1, 10, 4).astype(np.float32)
result = session.run(None, {'input': dummy})
print(result)  # 정상 출력 확인
```

---

## 5. Raspberry Pi 배포 준비

### 필요한 것

- Raspberry Pi 4 (4GB 권장)
- microSD 64GB
- ONNX Runtime ARM 버전

### 전달할 파일

```
checkpoints/
├── early_exit_fixed.onnx       # ONNX 변환 모델
└── scaler_params.json          # 정규화 기준값 (장예나 제공)
```

### Pi에서 실행할 추론 스크립트

```python
# scripts/inference_pi.py
import onnxruntime as ort
import numpy as np
import json
import time

# 정규화 기준값 로드
scaler = json.load(open('scaler_params.json'))

# 모델 로드
session = ort.InferenceSession('early_exit_fixed.onnx')

# 추론 시간 측정
times = []
for _ in range(100):
    dummy = np.random.randn(1, 10, 4).astype(np.float32)
    start = time.time()
    result = session.run(None, {'input': dummy})
    times.append((time.time() - start) * 1000)

print(f"평균 추론 시간: {np.mean(times):.3f}ms")
print(f"최소: {np.min(times):.3f}ms / 최대: {np.max(times):.3f}ms")
```

---

## 6. 완료 기준 체크리스트

- [ ] INT8 Quantization 적용 완료 (LSTM, Early Exit 모델)
- [ ] 경량화 전후 모델 크기 비교 완료
- [ ] 경량화 전후 정확도 비교 완료 (저하 2% 이내)
- [ ] 경량화 전후 추론 시간 비교 완료
- [ ] `quantization_comparison.csv` 저장 완료
- [ ] ONNX 변환 완료
- [ ] ONNX Runtime 동작 확인 완료
- [ ] `scripts/inference_pi.py` 구현 완료
- [ ] 장예나에게 경량화 결과 전달 완료 (시각화용)

---

## 7. 주의사항

- Quantization은 **Post-training** 방식으로 진행. 재학습 불필요.
- 정확도 저하가 2% 초과하면 유용상에게 알릴 것. 모델 재조정 필요할 수 있음.
- ONNX 변환 시 입력 shape `(1, 10, 4)` 고정. Raspberry Pi에서도 동일하게 사용.
- Raspberry Pi 없어도 PC에서 ONNX Runtime으로 추론 시간 측정 가능. Pi가 없으면 일단 PC에서 먼저 확인.
