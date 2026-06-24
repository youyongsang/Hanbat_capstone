# 김호중 방학 2단계 가이드라인
## AP 원본 CSV 생성 및 실제 데이터 기반 추론 준비

> 담당자: 김호중  
> 기간: 방학 3~4주차  
> 목표: GL.iNet AP 기반 원본 CSV 생성, 예나가 가공한 실측 데이터로 추론 준비, 원본 ONNX 비교 실험  
> 완료 기준: 원본 CSV 공유 완료, 기존 `inference_pi.py` 기준 실측 CSV 추론 동작 확인, 원본 ONNX vs 양자화+ONNX 비교 완료

---

## 1. 해야 할 일 순서

```
1. GL.iNet AP에서 시나리오별 원본 CSV 생성
2. 원본 CSV를 장예나에게 전달
3. 장예나가 가공한 windowed CSV 수신
4. 기존 `inference_pi.py`로 실측 windowed CSV 로딩 확인
5. 원본 모델 ONNX 변환
6. 원본 ONNX vs 양자화+ONNX 비교 실험
```

---

## 2. AP 원본 CSV 생성

호중은 기존에 AP 테스트를 진행한 환경을 기준으로 실제 WiFi 지표를 수집한다.  
이 단계에서 저장하는 파일은 모델 입력용 최종 CSV가 아니라 예나가 후처리할 원본 CSV다.

```
timestamp, throughput_mbps, channel_occupancy, packet_loss, latency, label, scenario
```

### iperf3 트래픽 생성 예시

```bash
# Pi 또는 AP 측 서버
iperf3 -s

# 단말 측 클라이언트
iperf3 -c [SERVER IP] -t 60 -b 10M   # 정상
iperf3 -c [SERVER IP] -t 60 -b 50M   # 혼잡 경고
iperf3 -c [SERVER IP] -t 60 -b 100M  # 혼잡
iperf3 -c [SERVER IP] -t 60 -b 200M  # 심각
```

### 예나에게 전달할 것

```
project/data/real_wifi/raw_measurements.csv
```

전달 시 각 컬럼의 단위, 시나리오별 label 기준, 측정 장비/위치도 함께 적는다.

---

## 3. 실제 데이터 기반 추론

예나가 가공해서 돌려주는 실측 데이터는 기존 코드와 동일하게 아래 컬럼을 가진 windowed CSV여야 한다.

```
sample_id, timestep, rps, channel_occupancy, packet_loss, latency, label, scenario
```

`project/scripts/inference_pi.py`는 `sample_id`별로 정확히 10개 timestep을 묶어서 `(1, 10, 4)` 입력을 만든다.  
따라서 한 행을 바로 reshape하지 말고, 기존 스크립트로 먼저 로딩과 추론을 확인한다.

```bash
python project/scripts/inference_pi.py \
  --mode full \
  --model project/checkpoints/early_exit_fixed.onnx \
  --data project/data/real_wifi/test.csv \
  --output project/results/hojung/real_wifi_inference_results.csv \
  --repeats 5
```

---

## 4. 원본 ONNX 변환

기존에 양자화+ONNX만 했는데, 원본 모델 ONNX도 추가.

기존 변환 스크립트를 사용한다. 이 스크립트는 전체 ONNX 출력명을 `exit1`, `exit2`, `exit3`로 저장하고, staged 옵션을 주면 exit별 분리 ONNX도 함께 만든다.

```bash
python project/scripts/export_onnx.py --staged
```

---

## 5. 원본 ONNX vs 양자화+ONNX 비교

| 방식 | 모델 크기 | 정확도 | 추론 시간 (PC) | 추론 시간 (Pi) |
|---|---|---|---|---|
| 원본 PyTorch | 1.28MB | 96.9% | 0.6ms | - |
| 양자화 PyTorch | 0.33MB | 96.9% | 1.5ms | - |
| 원본 ONNX | ~1.28MB | | | |
| 양자화 + ONNX | ~0.33MB | 96.9% | 0.9ms | - |

결과 저장:
```
project/results/hojung/onnx_comparison.csv
```

---

## 6. 완료 기준 체크리스트

- [ ] AP 기반 시나리오별 원본 CSV 생성 완료
- [ ] `raw_measurements.csv`를 장예나에게 전달 완료
- [ ] 기존 `inference_pi.py`로 실측 CSV 추론 동작 확인
- [ ] 실제 WiFi 데이터 추론 동작 확인
- [ ] `real_wifi_inference_results.csv` 저장 완료
- [ ] 원본 ONNX 변환 완료
- [ ] 원본 ONNX vs 양자화+ONNX 비교 실험 완료
- [ ] `onnx_comparison.csv` 저장 완료
- [ ] 유용상에게 정확도 검증 요청

---

## 7. 주의사항

- AP 장비를 통한 원본 CSV 생성은 호중 담당.
- 원본 CSV를 우리 실험용 피처/윈도우 형식으로 바꾸는 작업은 예나 담당.
- 실측 CSV도 모델 입력 단계에서는 `rps, channel_occupancy, packet_loss, latency` 컬럼명을 유지할 것.
- `throughput_mbps`는 원본 수집 파일에만 두고, 추론용 CSV에는 예나가 스케일링한 `rps`로 넣을 것.
- 추론 시간은 `time.perf_counter()` 사용. `time.time()`은 정밀도 낮음.
- 원본 ONNX 추론 시간이 양자화+ONNX보다 빠르게 나올 수 있음. 결과 그대로 기록할 것.
