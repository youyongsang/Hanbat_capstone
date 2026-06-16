# 김호중 방학 1단계 가이드라인
## Raspberry Pi 실행 환경 및 AP 연동 준비

> 담당자: 김호중  
> 목표: 예나가 구성한 실측 테스트베드에서 Pi 추론을 실행할 수 있는 환경 준비  
> 완료 기준: Pi에서 ONNX Runtime과 최신 추론 스크립트가 정상 동작

---

## 1. 공통 실험 원칙

실측 데이터 수집과 실시간 추론은 가능하면 같은 공간에서 이어서 수행한다. 예나가 AP 기반 데이터 수집 환경을 구성하면, 호중은 같은 테스트베드에서 Pi 추론 스크립트와 배포 파이프라인을 검증한다.

| 구분 | 기준 |
|---|---|
| 데이터 수집 환경 | 예나가 구성한 GL.iNet AP + Pi + 단말 테스트베드 |
| 실시간 추론 환경 | 같은 AP/Pi 환경에서 수행 |
| 호중 담당 범위 | ONNX/INT8 변환, Pi 실행 스크립트, 추론 시간 측정, 결과 분석 |
| 결과 해석 | Pi 실측 시간은 반복 측정 평균과 오차 기록 |

---

## 2. 해야 할 일 순서

```
1. Raspberry Pi OS 및 Python 환경 확인
2. ONNX Runtime 설치
3. GL.iNet AP와 Pi 유선 연결 확인
4. 예나 수집 CSV 형식 확인
5. Pi에서 추론 스크립트 도움말 확인
```

---

## 3. 확인 명령 예시

```bash
python3 --version
python3 -c "import onnxruntime as ort; print(ort.__version__)"
python3 inference_pi.py -h
```

---

## 4. 완료 기준 체크리스트

- [ ] Pi에서 Python 실행 가능
- [ ] `onnxruntime` import 가능
- [ ] AP와 Pi 연결 확인
- [ ] 예나 데이터 CSV를 Pi에서 읽을 수 있음
- [ ] `inference_pi.py`에서 `--mode staged`, `--stage1`, `--stage2`, `--stage3`, `--repeats` 옵션 확인

---

## 5. 주의사항

- Pi에 있는 `inference_pi.py`가 오래된 버전이면 staged ONNX 옵션이 인식되지 않는다.
- Pi 실측 시간은 CPU 온도와 백그라운드 프로세스 영향을 받을 수 있다.
