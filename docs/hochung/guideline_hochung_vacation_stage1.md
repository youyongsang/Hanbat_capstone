# 김호중 방학 1단계 가이드라인
## GL.iNet AP 세팅 및 원본 CSV 수집 준비

> 담당자: 김호중  
> 기간: 방학 1~2주차  
> 목표: GL.iNet AP OpenWrt 세팅 완료, Raspberry Pi 연동 구성, 원본 실측 CSV 수집 준비  
> 완료 기준: AP ↔ Pi 연동 동작 확인, 원본 지표 CSV 수집 스크립트 동작 확인

---

## 1. 해야 할 일 순서

```
1. GL.iNet AP OpenWrt 세팅
2. Pi ↔ AP LAN 연결
3. AP 지표 수집 스크립트 구현
4. 시나리오별 트래픽 생성 및 원본 CSV 저장 흐름 확인
5. 장예나에게 원본 CSV 형식 공유
```

---

## 2. GL.iNet AP 세팅

### 기본 설정

```bash
# AP SSH 접속 (기본 IP)
ssh root@192.168.8.1

# 패키지 업데이트
opkg update

# 필수 도구 설치
opkg install iperf3
opkg install tcpdump
opkg install iw
```

### 확인 항목

```bash
# iperf3 동작 확인
iperf3 -s

# WiFi 인터페이스 확인
iw dev

# 채널 점유율 수집 확인
iw dev wlan0 survey dump
```

---

## 3. Pi ↔ AP 연동 구성

### 환경 구성도

```
[단말 3~4대] → WiFi → [GL.iNet AP (192.168.8.1)]
                              ↓ LAN 케이블
                       [Raspberry Pi 4]
                       (지표 수집 + 원본 CSV 생성 + 추론)
```

### Pi에서 AP 지표 수집 및 원본 CSV 생성 스크립트

```python
# collect_metrics.py
import subprocess
import time
import csv

def collect_wifi_metrics():
    # 채널 점유율 수집
    survey = subprocess.run(
        ['iw', 'dev', 'wlan0', 'survey', 'dump'],
        capture_output=True, text=True
    )
    # 패킷 손실률 수집
    link = subprocess.run(
        ['ip', '-s', 'link', 'show', 'wlan0'],
        capture_output=True, text=True
    )
    return parse_metrics(survey.stdout, link.stdout)

def parse_metrics(survey_output, link_output):
    # 파싱 로직 구현
    channel_occupancy = parse_channel_occupancy(survey_output)
    packet_loss = parse_packet_loss(link_output)
    return channel_occupancy, packet_loss
```

### 원본 CSV 저장 형식

호중은 AP 장비와 Pi에서 측정 가능한 값을 그대로 원본 CSV로 저장한다.  
모델 입력용 컬럼명/정규화/윈도우 변환은 예나가 후처리한다.

```
timestamp, throughput_mbps, channel_occupancy, packet_loss, latency, label, scenario
```

---

## 4. 완료 기준 체크리스트

- [ ] GL.iNet AP OpenWrt 세팅 완료
- [ ] SSH 접속 확인
- [ ] iperf3, iw 설치 확인
- [ ] Pi ↔ AP LAN 연결 확인
- [ ] 원본 지표 CSV 수집 스크립트 동작 확인
- [ ] 시나리오별 원본 CSV 샘플 생성 완료
- [ ] 장예나에게 원본 CSV와 컬럼 의미 공유 완료

---

## 5. 주의사항

- AP 기본 IP는 192.168.8.1. 변경된 경우 확인 필요.
- Pi에서 AP로 SSH 접속 가능한지 확인할 것.
- AP 장비 기반 원본 CSV 생성은 호중 담당.
- 모델 입력에 맞춘 피처명 변경, 정규화, 10-step window 변환은 예나 담당.
