# 김호중 Stage 4 작업 설명

## 작업 로그 (Work Log): 모델 경량화(INT8), ONNX 가속 변환 및 배포 검증

## 1. 4단계 가이드라인 기준 수행 개요
Stage 3에서 검증된 Early Exit 알고리즘 모델을 자원 제약적인 하드웨어 환경(Raspberry Pi 4 - ARM 구조)에 안정적으로 배포하기 위해 양자화 및 가속 엔진 변환을 수행하였으며, 제시된 5가지 순서의 요구사항을 모두 충족 완료하였다.

### 1) 가이드라인 단계별 진행 항목 및 완료 기준
* **① INT8 Quantization 적용**: LSTM 및 Early Exit 구조 모델에 Post-training Quantization 적용 및 `quantization_comparison.csv` 저장 완료.
* **② 경량화 전후 성능 비교**: 모델 용량 경량화(1MB 이하 달성), 정확도 하락폭 2% 이내 방어, CPU 추론 지연 시간(Latency) 개선 검증 완료.
* **③ ONNX 변환**: ARM 및 구버전 환경 안정성을 고려하여 `dynamic_axes` 세팅 및 `opset_version=16` 규격 기반 최종 ONNX 변환 파일 생성 완료.
* **④ ONNX 추론 동작 확인**: ONNX Runtime 가변 배치 정상 추론 테스트 완료 및 실전 스크립트(`project/scripts/inference_pi.py`) 구현 완료.
* **⑤ Raspberry Pi 배포 준비**: 가속 가중치 파일 셋업 완료 및 입력 데이터 정규화(Scaler) 연동 준비 완료.

---

## 2. INT8 Quantization 및 성능 비교 결과

가이드라인 성능 목표치(용량 1MB 이하, 정확도 저하 2% 이내)를 기준으로 실측한 최적화 벤치마크 데이터셋 지표는 다음과 같다. 본 지표는 `project/results/quantization_comparison.csv` 파일로 격리 저장되었다.

### 1) 경량화 전후 최종 비교표 (`quantization_comparison.csv` 실제 데이터 반영)

| 모델명 (model) | 원본 용량 (original_size_mb) | 양자화 용량 (quantized_size_mb) | 원본 정확도 (original_accuracy) | 양자화 정확도 (quantized_accuracy) | 원본 추론 (original_inference_ms) | 양자화 추론 (quantized_inference_ms) | ONNX 가속화 추론 (최종) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_lstm** | 1.2768 MB | **0.3338 MB** | 95.44% | 95.16% | 0.6937 ms | 1.5637 ms | - |
| **early_exit_fixed** | 1.2819 MB | **0.3375 MB** | 95.16% | **95.44%** | 0.5014 ms | 1.1886 ms | **0.3220 ms** |

> 성능 지표 비하인드 분석  
PyTorch 내장 CPU 환경에서 단순 INT8 변환 시 연산 오버헤드로 인해 `quantized_inference_ms`가 일시적으로 증가하는 현상이 발생함.  
이를 보완하기 위해 Stage 4 최종 단계에서 **ONNX 가속 런타임 엔진 변환을 연계 전개**하였으며,  
그 결과 최종 추론 속도를 **0.3220 ms**까지 단축하며 경량화와 고속화를 동시에 달성함.

---

## 3. 터미널 실전 모델 가동 및 변환 명령어 과정

프로젝트 루트 디렉터리 (`C:\Users\User\Hanbat_capstone`) 기준으로,  
모델을 ONNX 가속 파일로 변환하고 검증하는 전체 실행 파이프라인이다.

---
```
requirement

pip install onnxruntime

pip install onnxscript 
```
### 1) 1단계: Raspberry Pi 극한 호환성 ONNX 변환

ARM 기반 구버전 환경에서의 크래시를 방지하기 위해  
`dynamic_axes` 기반 가변 배치 구조를 적용하고,  
`opset_version=16`으로 호환성을 확보하여 변환을 수행한다.
출력 로그 결과: 엣지 배포용 호환성 ONNX 파일 생성 성공: project/checkpoints/early_exit_fixed.onnx

### 2) 2단계: ONNX Runtime 가변 배치 및 추론 동작 확인 (Stage 4)
변환된 .onnx 가속 가중치를 가동하여 가변 배치(테스트 배치 크기 = 2) 주입 시 연산 붕괴가 없는지 무결성을 교차 검증한다.

```
python project/scripts/test_onnx_inference.py
```
출력 로그 결과: exit1, exit2, exit3 결과 차원이 전부 (2, 4) 규격으로 타 터지지 않고 완벽하게 바인딩 됨을 확인.

### 3) 3단계: 라즈베리파이 실전 배포용 런타임 조기 종료 제어 시뮬레이션
모델 내부 제어문 분기 오버헤드를 피하기 위해 "Multi-head Confidence Filtering(의사 조기 종료)" 기법을 탑재하고, 85% 확률 임계값에 따라 최종 예측값을 안전하게 추출 및 저장하는 실전 배포 스크립트를 검증한다.

```
python project/scripts/inference_pi.py

```
출력 로그 결과: 100회 스트리밍 추론 결과 런타임 전체 inference + decision pipeline latency 0.322 ms 달성 및 Exit 1번 초고속 조기 탈출 성공 비율 79회(79%) 통계치 확보 완료.

## 4. 최종 전달 및 배포 준비 파일 상태

 디바이스 이식용 산출물 체크리스트

```
project/checkpoints/early_exit_fixed.onnx
```

상태: 변환 완료 및 깃허브 업로드(git add -f) 완료.

```
project/scripts/inference_pi.py
```

상태: 데이터 타입 안정성(np.float32) 확보 및 확률 임계값 기반 런타임 조기 종료 제어 파이프라인 구현 완료.
