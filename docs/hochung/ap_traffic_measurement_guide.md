# AP 기기 트래픽 실측 방법 정리

## 1. 목적

GL.iNet Opal(GL-SFT1200) AP 환경에서 실제 WiFi 트래픽 지표를 수집하고, 2학기 실측 데이터 기반 혼잡 판단 실험에 사용할 원본 CSV를 생성한다.

이번 실측의 목적은 단순히 AP가 동작하는지 확인하는 것이 아니라, 정상/경고/혼잡/심각 혼잡 상태를 구분할 수 있도록 `throughput`, `channel occupancy`, `latency`, `jitter`, `packet loss`, `retry` 등 품질 저하 지표를 함께 확보하는 것이다.

---

## 2. 기존 방식의 문제점

초기 `collect_metrics.py`는 아래처럼 AP IP를 iperf3 대상 서버로 사용했다.

```text
iperf3 -c 192.168.8.1
```

이 방식은 AP 자체에 트래픽을 보내는 구조라서, 실제 무선 클라이언트 간 경쟁이나 AP 중계 구간의 혼잡이 충분히 드러나지 않을 수 있다. 또한 `iw survey dump`의 `channel occupancy`는 AP 처리량이 아니라 무선 칩셋이 감지한 채널 busy time 비율이므로, 주변 환경이 일정하면 40~46% 근처에서 안정적으로 유지될 수 있다.

따라서 2학기 실측에서는 AP를 iperf3 서버로 두지 않고, AP는 무선 중계 장비로 사용해야 한다.

---

## 3. 권장 실험 토폴로지

```text
WiFi Client             GL.iNet Opal AP             Wired Server
노트북/휴대폰/패드  -->  AP 무선 중계  -->  PC 또는 Raspberry Pi
                       192.168.8.1       192.168.8.x
```

역할은 다음과 같이 나눈다.

| 장비 | 연결 | 역할 |
|---|---|---|
| GL.iNet Opal AP | 중앙 AP | WiFi 중계, `iw survey`, `station dump` 수집 |
| Wired Server | AP LAN 포트 연결 | `iperf3 -s` 서버 |
| WiFi Client | AP WiFi 접속 | `iperf3 -c <SERVER_IP>` 트래픽 생성 |
| Raspberry Pi | 선택 | 실시간 추론 또는 서버 역할 |

핵심은 WiFi 클라이언트가 AP를 거쳐 유선 서버로 트래픽을 보내도록 만드는 것이다.

---

## 4. 전체 실행 과정

실험은 아래 순서로 진행한다.

```text
서버 준비 -> WiFi 부하 생성 -> AP 상태 수집 -> CSV 저장 -> 예나 전처리
```

핵심은 `iperf3` 트래픽 생성과 AP 무선 상태 수집이 같은 시간대에 동시에 일어나도록 맞추는 것이다.

1. 장비 연결
   - AP는 중앙 중계 장비로 둔다.
   - 서버는 AP LAN 포트에 연결한다.
   - WiFi 클라이언트는 AP 무선망에 접속한다.

2. 서버 실행
   - LAN에 연결된 PC 또는 Raspberry Pi에서 `iperf3 -s`를 실행한다.
   - 이 장비는 트래픽을 받는 서버 역할만 한다.

3. WiFi 클라이언트 부하 생성
   - WiFi 클라이언트에서 `iperf3 -c <SERVER_IP>`로 단계별 UDP 부하를 발생시킨다.
   - 부하 단계는 `20M`, `50M`, `100M`, `150M~200M + 병렬 스트림` 순서로 올린다.

4. AP 상태 동시 수집
   - AP에 SSH로 접속해 `iw dev wlan0 survey dump`와 `iw dev wlan0 station dump`를 실행한다.
   - `channel_occupancy`, `signal`, `bitrate`, `tx_retries`, `tx_failed`를 수집한다.

5. 결과 병합 및 CSV 저장
   - `iperf3`의 `throughput`, `jitter`, `loss`와 `ping latency`, `iw` 지표를 timestamp 기준으로 합쳐 raw CSV에 저장한다.
   - 수집 직후 `label`은 `-1`로 둔다.

6. 시나리오 반복 측정
   - `normal_idle`, `low_load`, `medium_load`, `high_load`, `stress_load`, `multi_client_load`를 각각 60초 이상 측정한다.
   - 각 시나리오별로 충분한 행 수를 확보한다.

7. 예나 전처리
   - 원본 CSV를 전달하면 예나가 복합 혼잡 점수로 label 0~3을 재부여한다.
   - 이후 10 timestep windowed CSV로 변환해 모델 입력 형식에 맞춘다.

실행 구조 요약:

```text
WiFi Client --WiFi--> GL.iNet Opal AP --LAN--> Wired Server
```

AP에 직접 `iperf3`를 쏘지 않고, AP를 통과하는 무선 구간의 상태를 측정한다.

---

## 5. 측정 시나리오

처음부터 심각 혼잡을 만들려고 하지 말고, 부하를 단계적으로 올리면서 각 단계의 지표가 어떻게 변하는지 확인한다.

| 시나리오 | 목적 | 예시 |
|---|---|---|
| `normal_idle` | 기준 상태 | iperf 없음, ping만 실행 |
| `low_load` | 낮은 부하 | UDP 20 Mbps |
| `medium_load` | 중간 부하 | UDP 50 Mbps |
| `high_load` | 높은 부하 | UDP 100 Mbps |
| `stress_load` | 혼잡 유도 | UDP 150~200 Mbps, 병렬 스트림 |
| `multi_client_load` | 다중 기기 경쟁 | 여러 클라이언트에서 동시 iperf |

2.4GHz만 사용하면 억지 실험처럼 보일 수 있으므로, 기본은 5GHz에서 수집한다. 다만 혼잡/심각 라벨이 부족하면 2.4GHz 또는 5GHz 20MHz 채널폭 제한을 별도 `controlled stress test`로 사용한다.

---

## 6. iperf3 실행 방법

### 6.1 서버 실행

AP의 LAN 포트에 연결된 PC 또는 Raspberry Pi에서 실행한다.

```bash
iperf3 -s
```

### 6.2 클라이언트 부하 생성

WiFi로 AP에 접속한 노트북에서 실행한다.

```bash
iperf3 -c <SERVER_IP> -u -b 20M -t 60 -i 1 --json
iperf3 -c <SERVER_IP> -u -b 50M -t 60 -i 1 --json
iperf3 -c <SERVER_IP> -u -b 100M -t 60 -i 1 --json
iperf3 -c <SERVER_IP> -u -b 150M -t 60 -i 1 -P 4 --json
```

옵션 의미:

| 옵션 | 의미 |
|---|---|
| `-u` | UDP 트래픽 사용 |
| `-b` | 목표 전송률 지정 |
| `-t 60` | 60초 측정 |
| `-i 1` | 1초 단위 결과 출력 |
| `-P 4` | 병렬 스트림 4개 |
| `--json` | 후처리 가능한 JSON 출력 |

TCP 다운로드보다 UDP가 부하를 단계적으로 강제하기 쉽다. 단, 실험망 내부에서만 수행한다.

---

## 7. 수집해야 할 원본 지표

최소 CSV 컬럼은 다음과 같다.

```text
timestamp,
throughput_mbps,
channel_occupancy,
latency_ms,
jitter_ms,
packet_loss_rate,
tx_retries,
tx_failed,
tx_bitrate_mbps,
rx_bitrate_mbps,
rssi_dbm,
connected_clients,
scenario,
label
```

초기에는 `label`을 `-1`로 저장하고, 예나의 전처리 단계에서 복합 혼잡 점수 기준으로 라벨을 재부여한다.

---

## 8. 지표별 수집 방법

### 8.1 처리량, jitter, 손실률

`iperf3 --json` 결과에서 아래 값을 추출한다.

```text
bits_per_second
jitter_ms
lost_packets
packets
lost_percent
```

CSV 매핑:

| iperf3 값 | CSV 컬럼 |
|---|---|
| `bits_per_second / 1e6` | `throughput_mbps` |
| `jitter_ms` | `jitter_ms` |
| `lost_percent` | `packet_loss_rate` |

### 8.2 지연시간

클라이언트에서 서버 IP로 ping을 실행한다.

```bash
ping <SERVER_IP> -c 20
```

가능하면 평균 RTT뿐 아니라 p95 지연도 저장한다.

```text
latency_avg_ms
latency_p95_ms
```

현재 모델 피처를 4개로 유지해야 한다면 우선 `latency_ms`에는 평균 RTT를 넣는다.

### 8.3 채널 점유율

AP에 SSH 접속해서 `iw survey dump`를 사용한다.

```bash
ssh root@192.168.8.1 "iw dev wlan0 survey dump"
```

계산식:

```text
channel_occupancy = channel busy time / channel active time * 100
```

주의:

- 이 값은 AP 처리량이 아니라 무선 채널 busy time 비율이다.
- AP 자신의 송신, 클라이언트 송신, 주변 AP, 관리 프레임, 간섭이 함께 반영될 수 있다.
- 실험 환경이 일정하면 값이 크게 변하지 않을 수 있다.

### 8.4 재전송, 실패, 링크 품질

가능하면 AP에서 station 통계를 수집한다.

```bash
ssh root@192.168.8.1 "iw dev wlan0 station dump"
```

확인할 항목:

```text
signal
tx bitrate
rx bitrate
tx retries
tx failed
rx packets
tx packets
```

CSV 매핑:

| station dump 값 | CSV 컬럼 |
|---|---|
| `signal` | `rssi_dbm` |
| `tx bitrate` | `tx_bitrate_mbps` |
| `rx bitrate` | `rx_bitrate_mbps` |
| `tx retries` | `tx_retries` |
| `tx failed` | `tx_failed` |

`tx_retries`와 `tx_failed`는 누적값일 수 있으므로, 가능하면 직전 측정값과의 차이를 계산해 `tx_retry_rate`, `tx_failed_rate`로 변환한다.

---

## 9. 혼잡 라벨링 방향

1학기에는 `channel_occupancy`를 대표 지표로 사용해 라벨을 정의했다.

```text
0 정상: 40% 미만
1 경고: 40~65%
2 혼잡: 65~85%
3 심각: 85% 이상
```

하지만 실제 AP에서는 `channel_occupancy`가 40~46% 범위에 머무를 수 있으므로, 2학기 실측 데이터는 복합 혼잡 점수로 라벨을 재정의한다.

권장 정의:

```text
채널 혼잡 상태 =
무선 채널 사용률 증가와 함께 지연, jitter, 손실, 재전송, 처리량 저하 등
통신 품질 저하가 나타나는 상태
```

임시 점수 예시:

```text
congestion_score =
  0.25 * normalized_channel_occupancy
+ 0.25 * normalized_latency
+ 0.20 * normalized_jitter
+ 0.20 * normalized_packet_loss
+ 0.10 * normalized_retry_rate
```

라벨 예시:

```text
0 정상: score < 0.25
1 경고: 0.25 <= score < 0.50
2 혼잡: 0.50 <= score < 0.75
3 심각: score >= 0.75
```

최종 threshold는 실제 수집 데이터 분포를 보고 조정한다.

---

## 10. 현재 `collect_metrics.py` 수정 방향

현재 스크립트의 핵심 문제:

```text
AP_IP = "192.168.8.1"
iperf3 -c AP_IP
ping AP_IP
```

수정 방향:

```text
AP_IP = "192.168.8.1"
SERVER_IP = "<LAN에 연결된 iperf3 서버 IP>"
iperf3 -c SERVER_IP
ping SERVER_IP
```

또한 `packet_loss`라는 컬럼명은 현재 `ip -s link`의 RX/TX error 합계에 가까우므로, 실제 손실률은 iperf3의 `lost_percent`에서 가져오는 것이 더 적절하다.

권장 컬럼명:

```text
packet_loss_rate
link_error_count
tx_retries
tx_failed
```

---

## 11. 팀 내 역할 분리

| 담당 | 작업 |
|---|---|
| 호중 | AP 환경 구성, iperf3 서버/클라이언트 측정, 원본 CSV 생성 |
| 예나 | 원본 CSV 전처리, 복합 라벨 재부여, windowed CSV 생성 |
| 용상 | 복합 라벨 기준 검토, Early Exit LSTM 재실험, 동적 threshold 영향 분석 |

---

## 12. 호중 원본 CSV 전달 체크리스트

- [ ] AP 모델명, 펌웨어 버전 기록
- [ ] 2.4GHz/5GHz 여부 기록
- [ ] 채널 번호와 채널폭 기록
- [ ] 서버/클라이언트 연결 구조 기록
- [ ] 시나리오별 측정 시간 60초 이상 확보
- [ ] 각 시나리오별 최소 수십~수백 행 확보
- [ ] `iperf3 --json` 원본 또는 CSV 변환본 저장
- [ ] `iw survey dump` 기반 channel occupancy 저장
- [ ] `iw station dump` 기반 retry/signal/bitrate 저장
- [ ] label은 초기 `-1`로 두고 후처리 단계에서 부여

---

## 13. 발표/보고서 표현

```text
1학기에는 channel occupancy를 대표 지표로 사용해 혼잡 라벨을 단순 정의하고,
Early Exit LSTM 구조와 경량 배포 가능성을 검증하였다.

2학기 실측 AP 환경에서는 channel occupancy 단일 지표의 변동폭이 제한적이므로,
latency, jitter, packet loss, retransmission rate, throughput 변화를 함께 반영한
복합 혼잡 라벨 기준으로 확장한다.
```
