# 김호중 방학 가이드라인
## Raspberry Pi 실시간 추론 및 ONNX/INT8 배포 검증

> 담당자: 김호중  
> 목표: 예나가 구성한 동일 실측 테스트베드에서 Raspberry Pi 기반 실시간 추론과 ONNX/INT8 배포 성능을 검증  
> 완료 기준: Pi 실시간 추론 동작 확인, FP32/INT8 ONNX 비교 결과 정리, 실행 절차 문서화

---

## 0. 공통 실험 원칙

실측 데이터 수집과 실시간 추론은 가능하면 같은 공간에서 이어서 수행한다. 예나가 AP 기반 데이터 수집 환경을 구성하면, 호중은 같은 테스트베드에서 Pi 추론 스크립트와 배포 파이프라인을 검증한다.

| 구분 | 기준 |
|---|---|
| 데이터 수집 환경 | 예나가 구성한 GL.iNet AP + Pi + 단말 테스트베드 |
| 실시간 추론 환경 | 같은 AP/Pi 환경에서 수행 |
| 호중 담당 범위 | ONNX/INT8 변환, Pi 실행 스크립트, 추론 시간 측정, 결과 분석 |
| 체크포인트/ONNX | 각 실행 환경에서 생성 또는 변환 |
| 결과 해석 | Pi 실측 시간은 환경 영향을 받으므로 반복 측정 평균과 오차 기록 |

---

## 1단계. Pi 실행 환경 및 AP 연동 준비

### 해야 할 일

```
1. Raspberry Pi OS 및 Python 환경 확인
2. ONNX Runtime 설치
3. GL.iNet AP와 Pi 유선 연결 확인
4. 예나 수집 CSV 형식 확인
5. Pi에서 추론 스크립트 도움말 확인
```

### 확인 명령 예시

```bash
python3 --version
python3 -c "import onnxruntime as ort; print(ort.__version__)"
python3 inference_pi.py -h
```

### 완료 기준

- [ ] Pi에서 Python 실행 가능
- [ ] `onnxruntime` import 가능
- [ ] `inference_pi.py`에서 `--mode staged`, `--stage1`, `--stage2`, `--stage3`, `--repeats` 옵션 확인
- [ ] 예나 데이터 CSV를 Pi에서 읽을 수 있음

---

## 2단계. 모델 변환 및 Pi 배포 번들 생성

### 실행 흐름

호중 브랜치 또는 통합 브랜치 기준으로 모델 변환과 배포 번들을 생성한다.

```bash
python project/scripts/export_onnx.py --staged
python project/scripts/export_onnx_int8.py
python project/scripts/prepare_pi_bundle.py
```

### 생성 대상

| 파일 종류 | 설명 |
|---|---|
| FP32 ONNX | 원본 정밀도 ONNX 모델 |
| INT8 ONNX | 양자화된 경량 모델 |
| staged ONNX | stage1/2/3 분리형 Early Exit ONNX |
| Pi bundle | Pi에서 바로 실행할 모델, 스크립트, 테스트 데이터 묶음 |

### 주의

- 용상 브랜치에서 생성한 체크포인트를 그대로 가져오기보다, 최종 실험 환경에서 다시 학습/변환한 모델을 사용한다.
- 체크포인트와 ONNX 파일은 결과 재현성을 위해 생성 시점, 데이터셋, threshold 값을 함께 기록한다.

### 완료 기준

- [ ] fixed FP32 staged ONNX 생성
- [ ] fixed INT8 staged ONNX 생성
- [ ] dynamic FP32 staged ONNX 생성
- [ ] dynamic INT8 staged ONNX 생성
- [ ] Pi 배포 번들 생성

---

## 3단계. 실측 CSV 기반 Pi 추론 성능 측정

예나가 수집한 `data/real_wifi/test.csv`를 기준으로 Pi에서 반복 측정한다.

### 실행 예시

```bash
python3 inference_pi.py \
  --mode staged \
  --stage1 early_exit_fixed_stage1.onnx \
  --stage2 early_exit_fixed_stage2.onnx \
  --stage3 early_exit_fixed_stage3.onnx \
  --data test.csv \
  --output pi_fixed_staged_fp32_results.csv \
  --repeats 5
```

```bash
python3 inference_pi.py \
  --mode staged \
  --stage1 early_exit_fixed_stage1_int8.onnx \
  --stage2 early_exit_fixed_stage2_int8.onnx \
  --stage3 early_exit_fixed_stage3_int8.onnx \
  --data test.csv \
  --output pi_fixed_staged_int8_results.csv \
  --repeats 5
```

동적 threshold도 같은 방식으로 `--dynamic-theta` 옵션을 포함해 실행한다.

### 측정 항목

| 항목 | 설명 |
|---|---|
| 정확도 | 실측 CSV 라벨 기준 분류 정확도 |
| 평균 추론 시간 | 반복 측정 평균 |
| p50/p95 지연 | 중앙값과 꼬리 지연 |
| Exit 비율 | Exit1/2/3 도달 비율 |
| FP32 vs INT8 | 모델 크기와 추론 시간 변화 |

### 완료 기준

- [ ] fixed FP32 Pi 결과 저장
- [ ] fixed INT8 Pi 결과 저장
- [ ] dynamic FP32 Pi 결과 저장
- [ ] dynamic INT8 Pi 결과 저장
- [ ] 결과 CSV/TXT/MD 분석 파일 생성

---

## 4단계. 실시간 혼잡 판단 데모 검증

### 목표

예나의 실시간 수집 루프에서 생성되는 최신 10개 시점 데이터를 모델 입력으로 받아 혼잡 수준을 출력한다.

```text
AP 지표 수집
→ 최근 10개 시점 버퍼
→ ONNX staged Early Exit 추론
→ label 0/1/2/3 출력
→ 채널 유지/전환 후보 출력
```

### 출력 예시

```text
[12:03:01] label=0 normal, action=keep, latency=1.31ms
[12:03:02] label=1 warning, action=monitor, latency=1.28ms
[12:03:03] label=2 congested, action=switch_candidate, latency=1.42ms
```

### 완료 기준

- [ ] 실시간 입력 10개 윈도우 수신 확인
- [ ] ONNX 추론 결과가 콘솔에 출력됨
- [ ] 채널 유지/전환 후보가 함께 출력됨
- [ ] 5분 이상 실행 로그 저장

---

## 5단계. 결과 정리 및 공유

### 정리 표

| 비교 | 포함 항목 |
|---|---|
| FP32 vs INT8 | 모델 크기, 정확도, 평균 추론 시간 |
| fixed vs dynamic | 정확도, 지연, Exit 비율 |
| CSV 평가 vs 실시간 데모 | 정량 평가와 동작 검증 분리 |
| PC vs Pi | 같은 데이터 기준 추론 시간 차이 |

### 완료 기준

- [ ] Pi 실측 결과표 작성
- [ ] 결과 그래프 생성
- [ ] 예나에게 실시간 데모 로그 전달
- [ ] 용상에게 fixed/dynamic 비교 결과 전달
- [ ] 실행 명령어와 환경 정보 문서화

---

## 주의사항

- Pi 실측 시간은 CPU 상태, 온도, 백그라운드 프로세스에 영향을 받는다.
- 한 번 측정값보다 `repeats=5` 이상 반복 평균과 p95를 같이 기록한다.
- 예나 데이터 수집 환경과 다른 장소에서 실시간 추론을 하면, 결과는 절대 성능 비교보다 동작 검증으로 해석한다.
- 호중 담당 산출물은 Pi 배포 코드와 실측 결과이며, 데이터 라벨링 기준은 예나 문서를 따른다.
