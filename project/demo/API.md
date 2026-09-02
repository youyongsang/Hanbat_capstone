# 데모 시스템 명세 — 팀 구현용

> 대상: 데모 대시보드를 만드는 팀원.
> 레퍼런스 구현: `project/demo/demo_server.py` + `demo.html` (동작·검증 완료, 아래 §7).
> 이 문서는 그 레퍼런스의 **계약(API·스키마·모델 입출력)** 을 고정한다. 프로덕션급으로
> 다시 짜더라도 이 계약을 지키면 프론트/모델/부하 파트가 서로 안 깨진다.

관련 문서:
- `docs/yongsang/demo_api_spec.{md,html}` — 원래 구상한 3-API-면(백엔드/부하 에이전트/파이 서버) 큰 그림. **이 문서가 그 중 "지금 실제로 만든 최소 버전"의 확정 스펙**이다.
- `project/scripts/live_congestion.py` — 웹 없이 터미널로 도는 같은 추론 루프.
- `docs/yongsang/onnx_early_exit_redesign.{md,html}` — ONNX 배포 구조.

---

## 1. 무엇을 하는가

1. 실제 AP(GL.iNet Opal)에 지속 SSH로 붙어 `iw` 텔레메트리를 ~2 Hz 폴링.
2. 폴링마다 **7-feature** 계산 → 최근 10개로 window → min-max 정규화 → **ONNX(unified INT8) 추론** → 4클래스 혼잡 라벨 + 확률.
3. 브라우저에 SSE로 실시간 push. 라벨/확률/feature/early-exit 지점을 표시.
4. 버튼으로 **두 폰에 계단형 iperf3 부하**(10/20/30 Mbps)를 걸고 끈다.

핵심 시연 포인트: **occupancy 단일 문턱(75%)으로 못 잡는 occ 60~73% 구간의 "심각"을 학습 모델이 잡는다**, 그리고 **스파이크가 아닌 지속 상태를 본다**(window 10).

---

## 2. 아키텍처

```
                  ┌─────────────── 노트북 (demo_server.py) ───────────────┐
  GL.iNet Opal    │                                                       │
  192.168.8.1 ──SSH(지속)──▶ APPoller ──▶ feature 계산 ──▶ window(10)      │
   (iw 폴링)      │                                          │            │
                  │                                    scaler 정규화       │
  두 폰           │                                          │            │
  s21 / s26 ◀──SSH(1회)── set_load()               ONNX InferenceSession   │
  (iperf3 UDP)    │                                          │            │
                  │                              _state (라벨·확률·feature) │
  노트북 wifi     │                                          │            │
  192.168.8.226 ◀─iperf3 -s (5201/5202, 자동 기동)   SSE broadcast         │
   (부하 대상)    │                                          │            │
                  └──────────────────── HTTP :8000 ──────────┼────────────┘
                                                             │
                                        브라우저 ◀── GET / , /events(SSE) , POST /load
```

- **추론과 부하 제어가 한 프로세스**(레퍼런스는 단순화를 위해). 팀이 나눠도 됨.
- 노트북·Pi 어디서든 실행 (§6). Pi 실행이 "엣지 추론" 서사에 맞음 — 추론·AP폴링이 Pi, 부하 대상도 Pi.
- `demo.html` 은 서버가 서빙 → **동일 출처**라 CORS·CSP 제약 없음. (claude.ai 아티팩트로는 이 백엔드에 못 붙음 — 로컬 파일이어야 함.)

---

## 3. HTTP API

베이스 URL: `http://<서버>:8000`

### `GET /`
데모 HTML 페이지(`demo.html`) 반환.

### `GET /health`
```json
{ "ready": true }
```
`ready` = window(10)가 채워져 추론이 나오는 중인지.

### `GET /events`  — Server-Sent Events

`Content-Type: text/event-stream`. 접속 즉시 현재 상태 1건, 이후 매 폴링(~0.5s)마다 1건.
15초마다 `: ping` 코멘트(keep-alive). 각 이벤트는 `data: <JSON>\n\n`.

**이벤트 JSON — 상태에 따라 두 형태:**

**(a) 준비 안 됨** (`ready:false`) — window 채우는 중이거나 AP 무응답:
```json
{
  "ready": false,
  "msg": "창 채우는 중 4/10",           // 또는 "AP 응답 없음 (재연결 2회)"
  "features": { ... },                  // 있을 수도 (window 채우는 중이면 포함)
  "clients": 3
}
```

**(b) 준비됨** (`ready:true`) — 매 폴링:
```json
{
  "ready": true,
  "ts": "20:38:14",                     // 서버 로컬 시각 HH:MM:SS
  "label": 3,                           // ★ 안정화(debounce) 라벨 0~3 — 표시/제어용
  "label_name": "심각",
  "raw_label": 3,                       // ★ 모델 원시 출력 argmax 0~3 — 정확도 평가 기준
  "raw_name": "심각",
  "probs": [0.01, 0.02, 0.10, 0.87],    // softmax [정상,경고,혼잡,심각], raw_label 기준
  "exit": 2,                            // early-exit 지점 1~3 (unified 그래프 출력)
  "clients": 3,                         // 연결된 station 수
  "features": {                         // 이번 폴링의 7-feature 원시값 (정규화 전)
    "throughput_mbps": 47.0,
    "channel_occupancy_percent": 66.0,
    "tx_retry_ratio": 0.38,
    "rssi_dbm": -35.0,
    "rssi_delta_db": 0.0,
    "rssi_moving_avg_dbm": -35.0,
    "sta_tx_bitrate_mean": 60.0
  },
  "load": {                              // 폰별 현재 부하, 2026-09-02부터 폰별 독립
    "s21": { "rate": "off", "packet_size": 1400 },
    "s26": { "rate": "30M", "packet_size": 200 }  // packet_size 필드는 같은 날 후속 추가
  }
}
```

> **`label` vs `raw_label`**: `raw_label` 이 모델의 실제 출력이다. `label` 은
> "원시 예측이 N폴링(레퍼런스 5) 연속 같을 때만 갱신"하는 **후처리 debounce** —
> 모델·학습·평가에는 없다. 리포트의 모든 정확도 수치(90~92% 등)는 `raw_label` 기준.
> 프론트는 둘 다 표시해야 정직하다(레퍼런스 `demo.html` 은 좌우 2패널 + 우측 상단 배지).

### `POST /load`  — 부하 제어 (2026-09-02, 폰별 독립 + 자동 종료 + 패킷 크기 축)

요청:
```json
{ "phone": "s26", "rate": "20M", "packet_size": 200 }
// phone: "s21"|"s26" (그 외 400)
// rate: "off"|"10M"|"20M"|"30M"|"40M" (그 외 400)
// packet_size: 1400|200, 생략 시 1400 (그 외 400) — 2026-09-02 후속 추가
```

동작: 지정한 폰에만 SSH → 기존 `iperf3` kill → (`off` 아니면) 새 `iperf3` 시작:
```
iperf3 -u -c 192.168.8.226 -p <5201|5202> -b <rate> -l <packet_size> -t 15   # LOAD_DURATION_S(10) + 5 여유
```
서버가 `LOAD_DURATION_S`(기본 10초) 뒤 `threading.Timer`로 자동 `off` 처리한다(같은 폰에 새
`/load`가 오면 기존 타이머는 취소). `iperf3` 자체의 `-t`도 짧게 잡아 타이머가 죽어도 자동 종료되는
이중 안전장치. 다른 폰의 부하엔 영향 없음(박스 2개가 완전히 독립).

**`packet_size`가 왜 있나**: bitrate만 있으면 "그냥 많이 넣으면 그만" 처럼 보인다는 피드백
(2026-09-02) — 같은 bitrate라도 `-l 200`(소패킷)은 `-l 1400`(일반)보다 초당 패킷 수가 훨씬 많아
채널 점유율·재전송이 더 크게 뛴다. 실기기로 확인: **S26=10M/1400B는 계속 "정상" 유지(occ 23~40%),
S26=10M/200B는 occ 70%·retry 38%로 "혼잡"까지 감** — 같은 bitrate, 다른 결과. 16차 데이터
수집(`smallpkt_knee`/`smallpkt_step` 시나리오)에서 이미 검증된 축을 데모에도 노출한 것.

응답:
```json
{ "ok": true, "phone": "s26", "rate": "20M", "packet_size": 200, "duration_s": 10 }
```
실패 시 `{ "ok": false, "error": "..." }` + 400 (`phone`/`rate`/`packet_size` 오류, SSH 실패 등).

### `GET /check`  — 연결확인 (2026-09-02 신규)

Pi(서버 자신)·AP·S21·S26을 한 번에 점검. "연결확인" 버튼용.
```json
{ "pi": true, "ap": true, "s21": true, "s26": false, "all_ok": false }
```
`pi`는 이 요청이 처리됐다는 것 자체로 항상 `true`. `ap`는 최근 AP 폴링이 정상(`ready` 이거나
"AP 응답 없음" 메시지가 아님)인지. `s21`/`s26`은 그 폰에 SSH `echo ok`가 4초 안에 성공했는지.

### `GET /signal`  — 폰 신호세기 (부하 전 확인용)
```json
{ "s21": -28, "s26": -23, "symmetric": true }
```
`symmetric` = 두 폰 신호 차이 ≤ 12 dBm. **비대칭이면 동시 부하 시 낮은 쪽만 굶고 AP가 불안정해질 수 있음** (§5) — 폰별 독립 제어(2026-09-02)로 위험은 크게 줄었지만 신호 확인은 여전히 권장.
폰이 안 붙어 있으면 해당 키 없음, AP 조회 실패면 `{ "error": "..." }`.

---

## 4. 모델 / 추론 계약  (바꾸지 말 것 — 학습과 묶임)

| 항목 | 값 | 근거 |
|---|---|---|
| feature 순서 | `throughput_mbps, channel_occupancy_percent, tx_retry_ratio, rssi_dbm, rssi_delta_db, rssi_moving_avg_dbm, sta_tx_bitrate_mean` | `project/utils/ap_features.py` (정본) |
| window | 최근 **10** 폴링, shape `[1, 10, 7]` float32 | 학습이 window 10 |
| 정규화 | `(x − min)/(max − min)`, `[0,1]` clip | `data/ap_metrics_v2_redesign2/scaler_params.json` — **학습 때 그 파일** 써야 함 |
| ONNX | `checkpoints/ap_v2_redesign2/ap_early_exit_fixed_unified_int8_v2.onnx` | unified If-노드 + INT8. 입력 1개, 출력 2개(logits, exit_point) |
| feature 안정화 (학습 시점 계약) | occ = 3폴링 median · tx_retry_ratio = 5폴링 rolling 비율(denom≥50) · rssi_moving_avg = 5폴링 평균 | `collect_metrics.py` 가 이렇게 계산해서 학습 데이터를 만듦 |
| 라벨 | 0 정상 / 1 경고 / 2 혼잡 / 3 심각 | congestion_score 문턱 0.25/0.50/0.75 |
| debounce (모델 밖) | `raw_label` N폴링 연속 일치 시 `label` 갱신. 레퍼런스 N=5 | 데모 표시용. 평가엔 미포함 |

feature 계산 로직은 레퍼런스가 `collect_metrics.py` 헬퍼(`APPoller`, `parse_ap_cycle`,
`summarize_stations`, `calculate_channel_occupancy`, `calculate_station_deltas`)를
그대로 import 한다 — 재구현하지 말고 재사용할 것. (Pi 번들엔 `collect_metrics.py` 를
복사, `live_congestion.py` 참고.)

---

## 5. 부하 / AP 크래시 주의

- 부하 상한 **30 Mbps/폰** (합계 60). `ALLOWED_RATES` 로 강제.
- **두 폰 신호 비대칭(>12 dBm) → capture effect → AP가 20M에도 크래시**. 실측 사례:
  step 프로파일 크래시(S21 STALE), 대칭 회복 후 60M ~70초 무사. 상세 `docs/yongsang/ap_crash_analysis.{md,html}`.
- 크래시 = br-lan 전체 다운(AP·Pi·노트북 연결 끊김) → AP 물리 재부팅.
- **프론트에 신호 대칭 표시 필수** (`/signal` 폴링, 레퍼런스 5초). 비대칭이면 부하 버튼 경고.
- AP 무응답 감지: `/events` 가 `ready:false, msg:"AP 응답 없음"` 을 push (poller 6초 stale).

---

## 6. 실행 / 사전조건

```bash
# 노트북 (기본값 그대로)
python project/demo/demo_server.py     # → http://localhost:8000/

# 라즈베리 파이 (엣지 추론) — 부하 대상도 Pi 자신으로
python3 demo_server.py \
    --iperf-target 192.168.8.109 \
    --s21 u0_a123@192.168.8.191 --s26 u0_a45@192.168.8.103
```

호스트/포트/경로/폰 대상은 전부 인자 또는 환경변수로 (`--host --port --iperf-target
--s21 --s26 --s21-port --s26-port --model --scaler --confirm --no-iperf-server`,
`DEMO_HOST` `DEMO_PORT` `DEMO_IPERF_TARGET` `DEMO_S21` `DEMO_S26`). 하드코딩 IP 없음.

| 필요 | 확인 |
|---|---|
| AP SSH | `ssh root@192.168.8.1` (`collect_metrics.py` `SSH_CMD` 옵션, ssh-rsa 허용). 노트북·Pi 둘 다 됨 |
| 폰 SSH | `ssh <--s21> echo ok` / `ssh <--s26> echo ok` — Termux sshd. 노트북은 `~/.ssh/config` 별칭(`s21`/`s26`), Pi 는 `--s21 user@ip` 로 직접 지정 + Pi 공개키를 폰 `~/.ssh/authorized_keys` 에 등록 |
| 폰 iperf3 대상 | `--iperf-target`. 노트북 실행이면 노트북 wifi IP(`192.168.8.226`), Pi 실행이면 Pi 자신의 `192.168.8.x` IP(폰→AP→Pi 경로) |
| iperf3 -s | `--iperf-target` 이 로컬 IP면 서버가 두 포트 자동 기동. 아니면 대상 호스트에서 직접 `iperf3 -s -p 5201` / `-p 5202` |
| Python | `onnxruntime`, `numpy`. 노트북=capstone conda 환경, Pi=시스템 python3 (onnxruntime 1.26) |
| ONNX/scaler | 저장소 경로에 없으면 스크립트 옆(Pi 번들)에서 자동 로드 |

**Pi 번들** (`project/deploy/raspberry_pi_ap_v2/`): `demo_server.py` `demo.html`
`collect_metrics.py` `scaler_params.json` `ap_early_exit_fixed_unified_int8_v2.onnx`
전부 포함. `scp -r` 후 위 명령으로 실행. `demo_server.py` 는 저장소/번들 양쪽에서
같은 파일로 도는 dual-import 구조 (`live_congestion.py` 와 동일).

---

## 7. 레퍼런스 구현 상태 (검증됨, 2026-08-30 — 아래는 그 시점의 "두 폰 동시" 구현 기준)

| | 결과 |
|---|---|
| 전 엔드포인트 | `/health` `/signal` `/events`(SSE) `POST /load` 실측 통과 |
| idle | 정상(0), p≈1.00, exit1 |
| 부하 10/20/30M (버튼) | 경고 → 혼잡 → **심각**(occ 63~76%, retry 38~54%), 정지 → 정상 복귀 |
| 지속 심각 | 30M 에서 확정 라벨 심각 **~58초 연속**(CONFIRM=5) |
| 스파이크 저항 | throughput 3174M 스파이크(SSH 타이밍)에도 라벨 안 흔들림 (occ median + window) |
| AP 크래시 | 없음 (30M×2, 신호 대칭 시) |

로그: `project/results/yongsang/ap_v2_redesign2_demo_full_run_20260830.txt`,
`ap_v2_redesign2_live_*_20260830.txt`.

> **2026-09-02 업데이트 (API 레벨 검증 완료)**: "참가형" 요구사항(폰별 독립 부하 박스 2개, `POST /load`가
> `{phone, rate}`로 변경, 10초 자동 종료 + 수동 정지, `GET /check` 연결확인, 우측 상단 실시간
> 배지)에 맞춰 `demo_server.py`/`demo.html`을 재작성했다. 같은 날 후속 세션에서 폰(S21/S26) 온라인
> 복귀 후 Pi(`capstone@192.168.8.109`)에서 기동해 curl로 실기기 검증: `GET /check` 전부 true →
> `{phone:"s26",rate:"10M"}`만 걸어도 S21은 `off` 유지(폰별 독립) → 10초 뒤 자동 `off` 복귀
> (throughput 13.8→0.08Mbps) → S26=20M+S21=10M 동시 걸어도 서로 안 건드림(합산 throughput
> 25~33Mbps, raw_label 정상→경고로 반응) → S26만 수동 `{rate:"off"}`로 즉시 정지해도 S21은 그대로
> 유지. 위 표는 여전히 구 버전(두 폰 동시 무제한 부하) 기준 기록으로 보존한다.
>
> **배포 절차 이슈 하나 발견·수정** (코드 버그 아님): Pi에 SSH로 `pkill -f demo_server.py; ...
> python3 demo_server.py`를 한 명령으로 보내면, `pkill -f`가 그 SSH 명령 자신의 argv 문자열
> (`"demo_server.py"`가 문자 그대로 들어있음)을 매칭해 스스로를 죽여 SSH 세션이 `exit 255`로
> 끊긴다 — 흔한 `pkill -f` self-match 문제. `pkill -f '[d]emo_server.py'` (브라켓 트릭, 자기
> 자신의 argv엔 `[d]emo_server.py`라는 리터럴이 있으므로 매칭 안 되고 실제 실행 중인
> `demo_server.py` 프로세스만 잡음)로 회피. 서버 재시작 스크립트/문서에 반영할 것.
>
> **아직 안 한 것**: 브라우저 UI 시각 확인(우측 상단 배지·카운트다운 표시·연결확인 버튼 클릭
> 인터랙션) — 이건 curl로는 못 잡는 프론트 렌더링/JS 이벤트 바인딩이라 사람이 직접
> `http://192.168.8.109:8000/`을 열어서 확인해야 한다.
>
> **2026-09-02 같은 날 후속: `packet_size` 축 추가, 실기기 검증 완료.** 사용자 피드백("10~40M
> 버튼만 있으면 그냥 많이 넣으면 그만 아니냐") 대응 — `POST /load`에 `packet_size`(1400|200)
> 추가, 각 부하 박스에 "일반(1400B)/소패킷(200B)" 토글 추가. Pi+S26으로 curl 검증:
> **10M/1400B는 계속 "정상"(occ 23~40%), 10M/200B는 "혼잡"까지 감(occ 70%, retry 38%)** — 같은
> bitrate, 다른 결과를 실측으로 확인. `load` 필드 wire format이 `{"s21":"off",...}`(문자열) →
> `{"s21":{"rate":"off","packet_size":1400},...}`(객체)로 바뀌었으니 프론트 예전 캐시 주의.

---

## 8. 팀이 만들 / 확장할 것

레퍼런스는 **동작하는 프로토타입**이다. 프로덕션/발표용으로 팀이 정할 것:

1. **백엔드 프레임워크** — 레퍼런스는 stdlib `http.server`(의존성 0). FastAPI/Flask 로 다시 짜도 되나 §3 계약 유지. 추론 루프는 별 스레드/프로세스로 분리 권장.
2. **설정 외부화** — AP/폰 IP, 포트, `IPERF_TARGET`, `CONFIRM`, 모델 경로를 `config.json`/env 로. (지금 `demo_server.py` 상단 상수.)
3. **대시보드 강화** — 레퍼런스 `demo.html` 은 최소. 팀이: 큰 게이지/애니메이션, feature별 시계열, 오답 하이라이트, 세션 녹화/리플레이, 발표용 풀스크린 모드.
4. **부하 프로파일** — 지금은 고정 rate 버튼. 계단 자동 진행(10→20→30, N초씩), knee, 커스텀 rate 입력.
5. **안전장치** — 신호 비대칭 시 버튼 비활성화, AP 무응답 시 자동 부하 정지, 부하 최대 지속시간 워치독.
6. **Pi 배포** — §6. 관리 트래픽을 무선 채널에서 분리하는 서사.
7. **밴드 스티어링 훅** (`demo_api_spec.{md,html}` §9, 발표 슬라이드7 최종 목표) — "심각 지속 → 채널 전환 명령 후보 생성". 지금 데모엔 없음. 라벨 스트림을 소비해서 별 모듈로.
8. **인증/멀티유저** — 지금 무인증 로컬. 필요하면.

**깨지 말아야 할 것**: §4 모델 계약, §3 API 스키마(특히 `raw_label` 노출), §5 크래시 주의.
