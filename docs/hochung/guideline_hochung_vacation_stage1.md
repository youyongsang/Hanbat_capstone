# 김호중 방학 1단계 가이드라인
## Raspberry Pi 실행 환경 및 배포 준비 기준 확정

> 담당자: 김호중  
> 단계 목표: Pi 추론 실험을 위한 실행 환경과 배포 스크립트 준비  
> 완료 기준: Pi에서 ONNX Runtime과 최신 추론 스크립트 도움말 확인

---

## 1. 방학 1단계 공통 목표

1단계는 세 명 모두 실제 실험에 들어가기 전 준비 단계다.  
이 단계에서는 아직 본 실험 결과를 만들지 않고, 장비·데이터 형식·실험 기준을 맞춘다.

| 담당 | 1단계 역할 |
|---|---|
| 장예나 | 실측 테스트베드와 데이터 수집/라벨링 기준 확정 |
| 김호중 | Pi 실행 환경과 배포 스크립트 실행 준비 |
| 유용상 | 실측 데이터 입력 형식과 모델 재학습 준비 |

---

## 2. 해야 할 일 순서

```
1. Raspberry Pi OS 및 Python 환경 확인
2. ONNX Runtime 설치 가능 여부 확인
3. GL.iNet AP와 Pi 유선 연결 방식 확인
4. 최신 inference_pi.py 옵션 확인
5. Pi 결과 저장 경로 정리
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
- [ ] AP와 Pi 연결 방식 확인
- [ ] `inference_pi.py` 최신 옵션 확인
- [ ] Pi 결과 저장 경로 결정

---

## 5. 주의사항

- Pi에 있는 `inference_pi.py`가 오래된 버전이면 staged ONNX 옵션이 인식되지 않는다.
- 1단계에서는 실제 성능 측정보다 실행 환경 준비가 목적이다.
