# 라이브 혼잡 감지 데모

버튼으로 계단형 부하를 걸고, 모델이 실시간으로 혼잡 레벨을 어떻게 예측하는지 웹에서 본다.

> **팀 구현용 상세 스펙: [`API.md`](API.md)** — API·SSE 스키마·모델 계약·크래시 주의·확장 항목.
> 이 README는 빠른 실행용.

## 실행 (노트북)

```bash
python project/demo/demo_server.py
```

브라우저에서 <http://localhost:8000/> 열기.

## 실행 (라즈베리 파이 — 엣지 추론)

추론·AP폴링·부하대상을 전부 Pi 로. 번들에 필요한 파일이 이미 다 있다
(`project/deploy/raspberry_pi_ap_v2/` : `demo_server.py` `demo_state.py` `demo_inference.py`
`demo_load.py` `demo_api.py` `demo.html` `collect_metrics.py`
`scaler_params.json` `*_int8_v2.onnx` — 2026-09-02부터 `demo_*.py` 5개 전부 있어야 동작).

```bash
# 1) 번들 복사 (한 번)
scp -r project/deploy/raspberry_pi_ap_v2 capstone@<파이IP>:~/demo

# 2) Pi 공개키를 두 폰에 등록 (한 번, 폰 Termux 에서)
#    cat ~/.ssh/id_ed25519.pub  →  각 폰 ~/.ssh/authorized_keys 에 추가

# 3) Pi 에서 실행
ssh capstone@<파이IP>
cd ~/demo
python3 demo_server.py \
    --iperf-target <파이IP> \
    --s21 <user@폰1IP> --s26 <user@폰2IP>
```

브라우저에서 <http://\<파이IP\>:8000/> 열기. 폰들이 Pi 로 iperf3 를 쏘므로
`--iperf-target` 이 로컬이면 `iperf3 -s` 는 자동 기동된다 (Pi 에 `iperf3` 설치 필요).

## 사전 조건

| 항목 | 확인 |
|---|---|
| AP SSH | `ssh root@192.168.8.1` (collect_metrics.py 와 동일 키) — 노트북·Pi 둘 다 |
| 폰 SSH | `ssh <--s21> echo ok` / `ssh <--s26> echo ok` (Termux sshd) |
| 폰 iperf3 대상 | `--iperf-target` — 노트북 실행 `192.168.8.226`, Pi 실행 Pi 자신 IP |
| Python | `onnxruntime`, `numpy` |
| ONNX / scaler | 저장소 경로 또는 스크립트 옆(Pi 번들) 에서 자동 로드 |

`iperf3 -s` 서버(5201/5202)는 `--iperf-target` 이 로컬 IP면 `demo_server.py` 가 자동 기동한다.
전부 인자로 조정 가능 — `demo_server.py --help`.

## 화면

- **우측 상단 실시간 배지** — 안정화 상태 라벨을 항상 눈에 띄게(디자인은 임시, 확정본은 팀에서 별도 작업).
- **연결확인** — 버튼 하나로 Pi(서버 자신)·AP·S21·S26 SSH 연결을 일괄 점검(`GET /check`).
- **모델 출력 (원시)** — 매 폴링 argmax 라벨 + 4클래스 확률 바. **리포트의 정확도 수치는 전부 이것 기준.**
- **안정화 상태** — 원시 예측이 5폴링(`--confirm`) 연속 같을 때만 갱신하는 debounce 라벨(표시/제어용). **모델·학습·평가엔 없는 후처리** — 경계 구간의 폴링 단위 흔들림만 억제.
- **AP 텔레메트리** — 모델 입력 7-feature 실시간값 + early-exit 지점
- **부하 제어 (참가형, 2026-09-02부터 폰별 독립)** — S26/S21 박스 2개, 각각 `10/20/30/40 Mbps` + `정지` 버튼. 버튼을 누르면 **그 폰에만** 10초간 부하가 들어가고 자동 종료(카운트다운 표시), `정지`로 즉시 종료도 가능. 두 폰을 동시에 다른 rate로 걸 수도 있다.
- **패킷 크기 토글 (같은 날 후속 추가)** — 각 박스 상단에 "일반(1400B)/소패킷(200B)" 토글. bitrate만으로는 "그냥 많이 넣으면 그만"처럼 보인다는 피드백에 대응 — 같은 bitrate라도 소패킷은 초당 패킷 수가 많아 채널 점유율·재전송이 더 크게 뜬다(실기기 검증: 10M/1400B는 "정상" 유지, 10M/200B는 "혼잡"까지 감).
- **최근 90초 차트** — 안정화 상태(굵은 선) / 모델 출력 원시(점선) / 채널 점유율(%)
- 폰 신호세기(S21/S26) — 12 dBm 이상 차이나면 비대칭 경고

## 주의

- **AP 크래시**: 신호 비대칭 상태에서 큰 부하가 들어가면 AP가 죽을 수 있다(`docs/yongsang/ap_crash_analysis.{md,html}`). 신호 확인 후 부하, 이상하면 즉시 정지. 폰별 독립 제어 + 10초 자동종료로 예전(두 폰 동시·무제한)보다 위험은 크게 줄었지만 40M을 두 폰에 동시에 걸면 여전히 조심.
- 부하는 `iperf3 -u -t 15`(10초 + 여유분)로 짧게 흐르다 자동 종료된다. 서버의 `threading.Timer`가 10초 뒤 `off` 처리하고, `iperf3` 자체의 `-t`도 짧아 이중 안전장치. 같은 폰에 새 버튼을 누르면 기존 타이머는 취소되고 새로 시작.
- **2026-09-02 API 레벨 실기기 검증 완료**: 이 API 변경(폰별 독립·`/check`)은 Pi+폰(S21/S26)으로 curl 검증됨 — 폰별 독립 제어·10초 자동종료·동시 부하·수동 정지 전부 정상 동작 확인. 브라우저 UI 시각 확인(배지·카운트다운 표시)만 아직 남음. 서버 재시작 시 `pkill -f demo_server.py`를 쓰면 SSH 세션이 self-match로 끊길 수 있으니 `pkill -f '[d]emo_server.py'`(브라켓 트릭) 사용. `API.md` §7 참고.

## 구조

**2026-09-02부터 5개 파일로 분리**(API가 한 파일에 다 있어서 혼잡하다는 피드백 반영):

| 파일 | 역할 |
|---|---|
| `demo_server.py` | 진입점 — 인자 파싱은 `demo_state`에 맡기고 여기선 iperf3 -s 기동·추론 스레드 시작·HTTP 서버 구동만 |
| `demo_state.py` | 설정·공유 상태(`_state`/`_load_state`)·`collect_metrics` dual-import. 나머지 4개가 전부 이걸 가져다 씀 |
| `demo_inference.py` | `scripts/collect_metrics.py` 의 `APPoller` + 파서를 재사용해 `scripts/live_congestion.py` 와 동일하게 feature 계산(victim 프로브 없음) → ONNX 추론 |
| `demo_load.py` | 폰별 부하 제어(SSH iperf3) + 연결확인 |
| `demo_api.py` | **HTTP API** — `Handler`(GET/POST 라우팅)만, 로직은 위 두 모듈에 위임 |

`demo.html` 은 서버가 서빙(동일 출처). 상세 계약은 `API.md` §6.5.
