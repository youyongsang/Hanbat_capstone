# 라이브 혼잡 감지 데모

버튼으로 계단형 부하를 걸고, 모델이 실시간으로 혼잡 레벨을 어떻게 예측하는지 웹에서 본다.

> **팀 구현용 상세 스펙: [`API.md`](API.md)** — API·SSE 스키마·모델 계약·크래시 주의·확장 항목.
> 이 README는 빠른 실행용.

## 실행 (노트북)

```bash
python project/demo/demo_server.py
```

브라우저에서 <http://localhost:8000/> 열기.

**대역 선택 (2026-10-05)**: `--band 5g`(또는 환경변수 `DEMO_BAND=5g`)면 AP 인터페이스 `wlan1`, 5GHz 시연 모델
(`checkpoints/ap_v2_5g_mixed_20261005/selected/`, 80+40MHz 혼합 학습)과 5GHz 전용 스케일러를 쓴다. 기본 `24g`는 기존 그대로
(`wlan0`, 24세션 모델, 캐노니컬 스케일러). Pi 번들은 `5g/` 폴더에서 모델·스케일러를 찾는다. 상단 칩에 `· 5GHz`가 붙고,
전환 후보는 다른 5GHz 채널(36/44/149/157)로 바뀐다. 시연 시 AP 5GHz 폭은 80MHz로 되돌려 둘 것(학습용 심각 데이터만 40MHz).

```bash
python project/demo/demo_server.py --band 5g
```

## MT6000 원인·전환 후보 표시 (2026-10-20)

MT6000(순정 OpenWrt)은 survey에 AP 송신 시간·우리 BSS 수신 시간을 내므로 "왜 혼잡한가"를 가를 수 있다. `--mt6000`이면
인터페이스 `phy0-ap0`·MT6000 배포 모델(`checkpoints/ap_v2_mt6k_20261018/selected/`)을 쓰고, 경고 이상일 때 상태 카드 아래에
**원인 카드**(이웃 와이파이 / 우리 기기들이 보내는 양 / 신호가 약한 기기 + 전파를 쓴 시간 중 우리 몫·이웃 몫 막대 + 조치)를 띄운다.
규칙은 `project/scripts/cause_advisor.py`(오프라인 `score_cause3.py`·`analyze_csw.py`와 m28~m33 전 폴링 100% 일치).

- 전환 알림(3폴링 중 2번 심각)이 났을 때 원인이 **이웃 와이파이**면 `channel_candidates.py --verify 2`를 백그라운드로 돌려
  후보 채널을 실측(약 90초, 연결 유지)하고 "ch1 → 11− 로 옮기면 이웃 몫 70% → 약 25%"처럼 보여 준다. 쿨다운 10분(`--gen-cooldown`).
- 원인이 **우리 부하·약한 기기**면 "다른 채널로 옮겨야 해요" 대신 "채널을 옮겨도 나아지지 않아요"와 그 원인의 조치.
- 실제 전환은 `--apply-switch`일 때만(기본은 명령 후보 표시).
- 원인이 **우리 부하**면 "채널 대신 5GHz로 옮겨야 해요"를 띄우고 `band_steer.py`로 2.4GHz에서 가장 많이 쓰는 기기를 골라
  같은 공유기의 5GHz(`phy1-ap0`)로 옮길 후보를 낸다(10-22). 실제 이동은 `--apply-steer`일 때만:
  802.11v BTM 요청 → 8초 안에 안 옮겨지면 2.4GHz에서 내보내고 120초 재접속 금지(기기에 5GHz SSID가 저장돼 있어야 스스로 붙음).
  BTM을 쓰려면 AP에 `bss_transition`·`ieee80211k`를 켜야 한다(스크립트는 AP 설정을 안 바꿈 — `band_steer.py` 머리말).
  옮기면 안 되는 기기는 `--steer-keep <MAC>`. 단독 실행: `python project/scripts/band_steer.py [--apply]`.
- **화면 버튼(10-22, 실시간 `--mt6000`에서만 보임)**: "더 나은 채널 찾기"(후보 실측, 쿨다운 무시) → 결과가 나오면 같은 버튼이
  "11− 로 채널 옮기기"로 바뀌고 누르면 CSA로 실제 전환. "5GHz로 옮길 기기 찾기" → "기기(xxxx) 5GHz로 옮기기". 버튼은
  `--apply-*` 옵션과 무관하게 사람이 누르면 실행한다(`POST /action {do: find|apply, kind: channel|5g}`). 재생 모드엔 안 보임.
- 노트북 화면(브라우저 높이 820px 이하)에서는 여백·글자를 줄여 그래프까지 한 화면에 들어오게 했다(10-22). Opal에선 카드가 안 뜬다(BSS 수신 시간 없음 → `advice.available=false`).

```bash
python project/demo/demo_server.py --mt6000 --no-iperf-server                      # 실시간 (MT6000 2.4GHz)
python project/demo/demo_server.py --mt6000 --replay project/.tmp/exp_24g/session_1020_m33/metrics_home24g_1020_m33_cswext.csv
python project/scripts/live_congestion.py --mt6000 --raw                           # 터미널판 (같은 규칙)
```

## 2.4GHz·5GHz 비교 화면 (2026-10-06, 시연용)

서버 두 개(2.4GHz :8000, 5GHz :8001)를 같이 띄우고 한 화면(`/dual`)에 좌우로 보여 준다. 폰을 손으로 5GHz SSID로 옮기면
화면이 폰 위치를 자동으로 찾고(4초마다 두 서버 `/check`), 부하 버튼은 폰이 붙은 대역 서버로 간다(60초 뒤 자동 정지).

```bash
bash project/demo/run_dual_demo.sh        # Pi iperf3(5211/5212) 기동 + 서버 두 개 + 브라우저 http://localhost:8000/dual
bash project/demo/run_dual_demo.sh stop   # 서버 두 개 + Pi iperf3 종료
```

- **부하 받는 쪽은 유선 Pi**(기본 `TARGET=192.168.8.109`). 노트북으로 받으면 폰을 5GHz로 옮겨도 AP→노트북 구간이 노트북 대역에
  남는데, Opal 점유율엔 AP 자신의 송신이 안 잡혀(다운링크 맹점) 비교가 섞인다.
- 폰 SSH 이름: 2.4GHz `s21g`/`s26g`, 5GHz `s21`/`s26`(`~/.ssh/config`). 노트북은 어느 대역이든 상관없다(화면만).
- AP 5GHz는 40MHz(2.4GHz HT40과 같은 폭). 80MHz면 폰 2대로 화면이 안 올라간다.
- 화면을 파일로 열거나 주소 끝에 `#sim`(가짜 데이터)·`#auto`(시연 흐름 자동 재생)를 붙이면 서버 없이 연습 모드.
- 10-06 리허설(폰당 30M, 폰 AP 가까이): 2.4GHz 표시 심각 92%·전환 경보 15초 / 5GHz 경고 92%·경보 없음.

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
