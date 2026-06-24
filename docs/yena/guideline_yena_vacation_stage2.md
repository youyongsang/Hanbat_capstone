# 장예나 방학 2단계 가이드라인
## AP 원본 CSV 피처 변환 및 시뮬레이터 노이즈 추가

> 담당자: 장예나  
> 기간: 방학 3~4주차  
> 목표: 호중이 생성한 AP 원본 CSV를 우리 실험 입력 피처로 변환, 시뮬레이터 노이즈 추가 버전 구현  
> 완료 기준: 기존 코드와 호환되는 windowed CSV 데이터셋 생성 완료, 노이즈 시뮬레이터 완성

---

## 1. 해야 할 일 순서

```
1. 김호중 AP 원본 CSV 수령
2. 원본 컬럼/결측/단위 확인
3. 처리량(throughput_mbps)을 rps 컬럼으로 매핑
4. 슬라이딩 윈도우 전처리
5. 정규화 및 scaler_params 저장
6. 시뮬레이터 노이즈 추가 버전 구현
7. 실제 데이터 vs 시뮬레이터 분포 비교
```

---

## 2. AP 원본 CSV 피처 변환

호중이 AP 장비에서 생성해 전달하는 원본 CSV는 아래 형식을 기준으로 한다.

```
timestamp, throughput_mbps, channel_occupancy, packet_loss, latency, label, scenario
```

기존 학습/평가/추론 코드는 `rps, channel_occupancy, packet_loss, latency` 4개 피처를 사용한다.  
따라서 전처리 후 최종 `train.csv`, `val.csv`, `test.csv`는 반드시 아래 windowed 형식으로 저장한다.

```
sample_id, timestep, rps, channel_occupancy, packet_loss, latency, label, scenario
```

`rps`에는 호중이 측정한 실측 처리량을 0~1000 범위로 스케일링한 값을 넣는다. 원본 `throughput_mbps` 컬럼은 별도 원본 파일에 보관하되, 모델 입력 CSV에는 넣지 않는다.

`scaler_params.json`에는 오프라인 CSV 변환뿐 아니라 호중의 실시간 추론 코드가 같은 기준을 쓸 수 있도록 원본 처리량 범위도 함께 저장한다.

```json
{
  "throughput_mbps": {"min": 0.0, "max": 200.0},
  "rps": {"min": 0.0, "max": 1000.0},
  "channel_occupancy": {"min": 0.0, "max": 100.0},
  "packet_loss": {"min": 0.0, "max": 30.0},
  "latency": {"min": 0.0, "max": 500.0}
}
```

실시간 추론에서는 `throughput_mbps`를 위 기준으로 `rps`에 매핑한 뒤, `rps, channel_occupancy, packet_loss, latency` 순서로 정규화한다.

### 저장 경로

```
data/real_wifi/
├── raw_measurements.csv
├── train.csv
├── val.csv
├── test.csv
└── scaler_params.json
```

---

## 3. 시뮬레이터 노이즈 추가

기존 시뮬레이터 데이터가 너무 깔끔해서 동적 threshold 효과가 제한적이었음.  
실제 환경처럼 노이즈를 추가하여 현실성 개선.

```python
# traffic_simulator_noisy.py
channel_occupancy += np.random.normal(0, 5)   # 표준편차 5%
packet_loss += np.random.normal(0, 1)          # 표준편차 1%
latency += np.random.normal(0, 10)             # 표준편차 10ms

# 범위 클리핑
channel_occupancy = np.clip(channel_occupancy, 0, 100)
packet_loss = np.clip(packet_loss, 0, 30)
latency = np.clip(latency, 0, 500)
```

기존 시뮬레이터 버전은 유지하고 노이즈 버전은 별도 파일로 저장할 것.

---

## 4. 데이터 분포 비교 분석

시뮬레이터 데이터와 실제 WiFi 데이터의 피처 분포를 비교하여  
시뮬레이터의 현실성을 정량적으로 확인한다.

```python
# 각 피처별 평균, 표준편차 비교
for feature in ['rps', 'channel_occupancy', 'packet_loss', 'latency']:
    sim_mean = simulator_df[feature].mean()
    real_mean = real_df[feature].mean()
    print(f"{feature}: 시뮬레이터 {sim_mean:.2f} vs 실제 {real_mean:.2f}")
```

---

## 5. 완료 기준 체크리스트

- [ ] 김호중 AP 원본 CSV 수령 완료
- [ ] 원본 컬럼/결측/단위 확인 완료
- [ ] `data/real_wifi/train.csv`, `val.csv`, `test.csv` 생성 완료
- [ ] 최종 CSV 컬럼이 기존 코드의 `sample_id,timestep,rps,channel_occupancy,packet_loss,latency,label` 형식과 호환됨
- [ ] `throughput_mbps` 원본 범위와 모델 피처 범위를 포함한 `scaler_params.json` 저장 완료
- [ ] 노이즈 추가 시뮬레이터 (`traffic_simulator_noisy.py`) 완성
- [ ] 시뮬레이터 vs 실제 데이터 분포 비교 분석 완료
- [ ] 김호중·유용상에게 데이터셋 공유 완료

---

## 6. 주의사항

- AP 원본 CSV 생성은 김호중 담당, 피처 변환과 windowed CSV 생성은 예나 담당
- 실제 데이터도 shape `(N, 10, 4)` 기존과 동일하게 맞출 것
- 모델 입력 CSV에는 `throughput_mbps` 대신 `rps` 컬럼명을 사용할 것
- 정규화는 train 기준값으로 val, test에도 동일하게 적용할 것
- 실시간 추론 코드도 같은 `scaler_params.json`을 사용하므로 키 이름을 임의로 바꾸지 말 것
- 노이즈 추가 시뮬레이터는 기존 버전과 별도 파일로 관리할 것
