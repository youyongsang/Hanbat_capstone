# 김호중 방학 2단계 가이드라인
## ONNX/INT8 변환 및 Pi 배포 번들 생성

> 담당자: 김호중  
> 목표: Early Exit 모델을 FP32 ONNX, INT8 ONNX, staged ONNX 형태로 변환하고 Pi 배포 번들을 생성  
> 완료 기준: fixed/dynamic 각각 FP32/INT8 staged ONNX와 Pi 실행 번들 생성

---

## 1. 해야 할 일 순서

```
1. 최종 학습 체크포인트 확인
2. FP32 ONNX 변환
3. staged ONNX 변환
4. INT8 ONNX 양자화
5. Pi 배포 번들 생성
```

---

## 2. 실행 명령

```bash
python project/scripts/export_onnx.py --staged
python project/scripts/export_onnx_int8.py
python project/scripts/prepare_pi_bundle.py
```

---

## 3. 생성 대상

| 파일 종류 | 설명 |
|---|---|
| FP32 ONNX | 원본 정밀도 ONNX 모델 |
| INT8 ONNX | 양자화된 경량 모델 |
| staged ONNX | stage1/2/3 분리형 Early Exit ONNX |
| Pi bundle | Pi에서 바로 실행할 모델, 스크립트, 테스트 데이터 묶음 |

---

## 4. 완료 기준 체크리스트

- [ ] fixed FP32 staged ONNX 생성
- [ ] fixed INT8 staged ONNX 생성
- [ ] dynamic FP32 staged ONNX 생성
- [ ] dynamic INT8 staged ONNX 생성
- [ ] Pi 배포 번들 생성
- [ ] 생성 파일 목록 기록

---

## 5. 주의사항

- 체크포인트와 ONNX 파일은 각 실험 환경에서 생성하는 것을 원칙으로 한다.
- 생성 시점, 사용 데이터셋, threshold 값을 함께 기록한다.
- 용상 브랜치의 모델 코드가 변경되면 변환 전 호환성을 다시 확인한다.
