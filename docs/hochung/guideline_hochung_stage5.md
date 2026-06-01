# 김호중 5단계 가이드라인
## Raspberry Pi 포팅 및 실기기 추론 지연 실측

> 담당자: 김호중  
> 목표: ONNX 모델을 Raspberry Pi 4에 배포하고 실기기 추론 지연 실측  
> 완료 기준: Pi에서 모델 추론 동작 확인, 추론 지연 측정 결과 CSV 저장

---

## 1. 해야 할 일 순서

```
1. Raspberry Pi OS 설치
2. Pi 환경 세팅 (Python, ONNX Runtime)
3. 모델 및 데이터 Pi로 전송
4. 추론 스크립트 실행
5. Exit별 추론 지연 실측
6. PC vs Pi 성능 비교
```

---

## 2. Raspberry Pi OS 설치

### 필요한 것
- Raspberry Pi 4 Model B 8GB
- microSD 64GB
- Raspberry Pi Imager (무료 소프트웨어)

### 설치 순서

```
1. PC에서 Raspberry Pi Imager 설치
   https://www.raspberrypi.com/software/

2. microSD 카드 삽입

3. Imager 실행
   OS 선택: Raspberry Pi OS (64-bit) 권장
   저장소: microSD 선택
   Write 클릭

4. microSD를 Pi에 삽입 후 전원 연결

5. 초기 설정 (언어, WiFi, 계정)
```

---

## 3. Pi 환경 세팅

### Python 및 필수 패키지 설치

```bash
# Pi 터미널에서 실행

# 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# Python pip 설치
sudo apt install python3-pip -y

# 필수 패키지 설치
pip3 install onnxruntime numpy pandas
```

### 설치 확인

```python
import onnxruntime as ort
import numpy as np
print("ONNX Runtime 버전:", ort.__version__)
print("사용 가능한 프로바이더:", ort.get_available_providers())
```

---

## 4. 모델 및 데이터 Pi로 전송

### PC → Pi 파일 전송 방법

**방법 1 — SCP (SSH 이용)**
```bash
# PC에서 실행
scp checkpoints/early_exit_fixed.onnx pi@[Pi IP주소]:~/capstone/
scp data/real/scaler_params.json pi@[Pi IP주소]:~/capstone/
scp data/real/pi_test_samples.csv pi@[Pi IP주소]:~/capstone/
```

**방법 2 — USB 드라이브**
파일을 USB에 복사 후 Pi에서 마운트하여 복사.

### Pi에서 파일 구조

```
~/capstone/
├── early_exit_fixed.onnx
├── scaler_params.json
└── pi_test_samples.csv
```

---

## 5. 추론 스크립트

### `inference_pi.py`

```python
import onnxruntime as ort
import numpy as np
import pandas as pd
import json
import time

# 정규화 기준값 로드
scaler = json.load(open('scaler_params.json'))

# 모델 로드
session = ort.InferenceSession('early_exit_fixed.onnx')

# 테스트 데이터 로드
test_data = pd.read_csv('pi_test_samples.csv')

# 추론 시간 측정
results = []
for i in range(len(test_data)):
    input_data = test_data.iloc[i:i+1].values.astype(np.float32)
    input_data = input_data.reshape(1, 10, 4)

    start = time.perf_counter()
    output = session.run(None, {'input': input_data})
    elapsed = (time.perf_counter() - start) * 1000  # ms

    results.append({
        'sample_id': i,
        'inference_time_ms': elapsed,
        'predicted_label': np.argmax(output[0])
    })

df = pd.DataFrame(results)
df.to_csv('pi_inference_results.csv', index=False)

print(f"평균 추론 시간: {df['inference_time_ms'].mean():.3f}ms")
print(f"최소: {df['inference_time_ms'].min():.3f}ms")
print(f"최대: {df['inference_time_ms'].max():.3f}ms")
```

---

## 6. 측정할 지표

| 지표 | 측정 방법 |
|---|---|
| 평균 추론 시간 (ms) | 전체 샘플 평균 |
| Exit별 추론 시간 | Exit 1/2/3 각각 측정 |
| 최소/최대 추론 시간 | 편차 확인 |
| 모델 크기 | ONNX 파일 크기 |

---

## 7. PC vs Pi 비교표

실측 후 아래 표 채우기.

| 항목 | PC (x86 CPU) | Raspberry Pi 4 8GB |
|---|---|---|
| Exit 1 추론 시간 | ms | ms |
| Exit 2 추론 시간 | ms | ms |
| Exit 3 추론 시간 | ms | ms |
| 평균 추론 시간 | ms | ms |
| 모델 크기 | MB | MB |

---

## 8. 완료 기준 체크리스트

- [ ] Raspberry Pi OS 설치 완료
- [ ] ONNX Runtime 설치 확인
- [x] 모델 파일 Pi 전송용 번들 준비 완료
- [x] `inference_pi.py` 실측 CSV 저장 구조 구현 완료
- [ ] `pi_inference_results.csv` 저장 완료
- [ ] PC vs Pi 비교표 완성
- [ ] 장예나에게 Pi 실측 결과 전달 완료

> 현재 상태: PC에서 smoke test와 배포 번들 생성은 완료. 실제 Raspberry Pi에서 실행한 `pi_inference_results.csv`는 사용자가 Pi에서 생성한 뒤 레포지토리로 가져와야 한다.

---

## 9. 주의사항

- Pi는 CPU만 사용. GPU 없어서 PC보다 느린 건 당연함.
- 추론 시간은 `time.perf_counter()` 사용. `time.time()`은 정밀도 낮음.
- Pi 온도가 높으면 성능 저하 발생. 서멀 쓰로틀링 주의. 케이스 있으면 통풍 확인.
- microSD 속도가 느리면 모델 로딩이 오래 걸릴 수 있음. 로딩 시간은 측정에서 제외할 것.
