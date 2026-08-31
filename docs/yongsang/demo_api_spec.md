# 혼잡 데모 API 명세서

부하 제어 + 실시간 혼잡도 대시보드 데모를 만들기 위한 API 규격. 이 문서는 원래 구상한 **3-API-면(백엔드 REST+SSE / 부하 에이전트 / 파이 서버) 큰 그림**이다.

> **(2026-08-30)** 그 중 **"지금 실제로 만들어 동작·검증한 최소 버전"의 확정 스펙은 `project/demo/API.md`** 다 (레퍼런스 구현: `project/demo/demo_server.py` + `demo.html`). 팀이 데모를 만든다면 그 문서의 API·모델 계약을 지키고, 이 문서는 확장 방향(밴드 스티어링 §9 등)의 참고로 본다.

> **(2026-08-29 최신화)** 아래 FeatureVector·SubScores·congestion_score 관련 절은 8/27 라벨 재설계 + 8/29 7-feature 승격을 반영해 갱신했다. 원래 이 문서가 전제했던 9-feature 가중합 스키마는 이미 두 번 바뀌었다(9→6→7-feature, 가중합→max/anchor 방식) — 구현 착수 전에 반드시 `project/utils/ap_features.py`·`project/scripts/collect_metrics.py`의 `calculate_scores()`로 최신 계약을 재확인할 것. ONNX export는 이미 끝나 있다(unified INT8 v2, `project/checkpoints/ap_v2_redesign2/`) — 8/27 시점 "신규 구현 부담은 ONNX export 하나"라는 전제는 더 이상 유효하지 않다(그 작업은 이미 끝났고, 남은 건 대시보드/백엔드/에이전트 4개 컴포넌트 자체).

## 1. 시스템 개요

```text
 [브라우저 (노트북)]  ── 대시보드: 부하 제어판 + 실시간 혼잡 게이지
        │  REST + SSE (동일 origin)
 [노트북 백엔드 (FastAPI)]  ── 대시보드 서빙 · 부하 명령 중계 · iperf3 -s 싱크 · 파이 스트림 재방출
        │                                   │
        │ SSH exec 또는 HTTP 에이전트          │ SSE (유선 관리 서브넷)
        ▼                                   ▼
 [폰 191 / S26 (Termux)]              [라즈베리 파이]
   iperf3 -c 부하 생성                  APPoller(AP 유선 SSH) → 윈도우10 → ONNX Early Exit LSTM → SSE 1Hz
```

- **추론은 파이에서** 수행한다(엣지 추론 서사). 노트북은 표시·제어·중계만.
- 모델 ONNX export는 이미 끝나 있다 — `project/checkpoints/ap_v2_redesign2/ap_early_exit_{fixed,dynamic}_unified_int8_v2.onnx`(권장, staged→unified→INT8 재설계 완료본, `docs/yongsang/onnx_early_exit_redesign.md` 참고). 신규 구현 부담은 파이 추론 서버(면 ③)·부하 에이전트(면 ②)·백엔드+대시보드(면 ①) 4개 컴포넌트.
- 3개 API 면(surface): ① 백엔드 REST+SSE(브라우저용) ② 부하 에이전트(폰용) ③ 추론 스트림(파이용).

## 2. 공유 데이터 계약

### 2.1 FeatureVector

`project/utils/ap_features.py`의 `AP_FEATURE_COLUMNS`와 **순서·이름이 정확히 일치**해야 한다(2026-08-29 기준 7-feature — `sta_tx_bitrate_mean`이 가장 최근에 추가됨, `min-max` 스케일러는 `project/data/ap_metrics_v2_redesign2/scaler_params.json` 기준).

```json
{
  "throughput_mbps": 43.2,
  "channel_occupancy_percent": 88.0,
  "tx_retry_ratio": 0.22,
  "rssi_dbm": -58.0,
  "rssi_delta_db": -2.0,
  "rssi_moving_avg_dbm": -57.1,
  "sta_tx_bitrate_mean": 34.5
}
```

`latency_ms`/`jitter_ms`/`probe_loss_pct`는 더 이상 모델 입력이 아니다(라벨 축이자 배포 시점엔 없는 측정이라 8/27 재설계에서 제외 — 모델은 "채널 상태만 보고 victim QoS를 예측"해야 함). 대신 아래 SubScores 계산에 그대로 쓰인다(victim 프로브·ping으로 별도 측정).

### 2.2 Label

| 값 | 이름 | 의미 |
|---:|---|---|
| 0 | 정상 | congestion_score < 0.25 |
| 1 | 경고 | 0.25 ~ 0.50 |
| 2 | 혼잡 | 0.50 ~ 0.75 |
| 3 | 심각 | ≥ 0.75 |

### 2.3 SubScores

**(2026-08-29 최신화)** 8/27 재설계로 가중합 방식(구버전: throughput/occupancy/retry_failed/jitter 가중합)에서 **표준 문턱 max 방식**으로 바뀌었다 — 정본은 `project/scripts/collect_metrics.py`의 `calculate_scores()`.

```json
{ "occupancy": 0.88, "jitter": 0.70, "loss": 0.0, "latency": 0.25 }
```

각 0~1, `anchor_score()`로 4앵커(경고/혼잡/심각/완전)를 piecewise-linear 매핑. `congestion_score = max(occupancy, jitter, loss, latency)` — 가중합이 아니라 **가장 나쁜 축이 곧 congestion_score**(4축 중 하나라도 완전히 깨지면 나머지가 멀쩡해도 심각으로 판정). `retry`·`throughput`은 정보용 sub-score로만 유지되고 `congestion_score`(라벨 축)엔 들어가지 않는다 — 이 2.4GHz AP는 idle에도 retry_ratio가 18~36%라 라벨 변별력이 없기 때문(상세: `docs/yongsang/congestion_label_redesign.{md,html}`).

앵커(표준 문턱, `ANCHORS` in `collect_metrics.py`):

| 축 | 경고 | 혼잡 | 심각 | 완전 | 근거 |
|---|---:|---:|---:|---:|---|
| occupancy(%) | 40 | 55 | 75 | 90 | Cisco/Aruba WLAN 가이드 |
| jitter(ms, 프로브) | 20 | 30 | 50 | 100 | ITU-T Y.1541 / RFC 4594 |
| loss(%, 프로브) | 0.5 | 1.0 | 5.0 | 10.0 | Cisco QoS / ITU-T G.113 |
| latency(ms, 편도) | 30 | 60 | 150 | 400 | ITU-T G.114(ping RTT/2) |

### 2.4 LoadProfile

```json
{
  "mode": "udp_smallpkt",     // "udp_smallpkt" | "udp_default" | "tcp"
  "rate_mbps": 25,            // 대상 폰 1대당
  "packet_bytes": 250,        // udp_* 모드에서만 (iperf3 -l)
  "duration_s": 300,
  "parallel": 1               // iperf3 -P
}
```

`mode`별 iperf3 매핑:

| mode | iperf3 명령 |
|---|---|
| `udp_smallpkt` | `iperf3 -u -c <server> -p <port> -l <packet_bytes> -b <rate_mbps>M -t <duration_s> -P <parallel>` |
| `udp_default` | `iperf3 -u -c <server> -p <port> -b <rate_mbps>M -t <duration_s> -P <parallel>` |
| `tcp` | `iperf3 -c <server> -p <port> -t <duration_s> -P <parallel>` |

### 2.5 Preset

```json
{
  "id": "smallpkt_25_250",
  "label": "소패킷 25M · l=250",
  "targets": ["s21", "s26"],
  "profile": { "mode": "udp_smallpkt", "rate_mbps": 25, "packet_bytes": 250, "duration_s": 300 },
  "safe": true
}
```

### 2.6 안전 상수 (API에 하드코딩)

| 상수 | 값 | 근거 |
|---|---:|---|
| `MAX_RATE_MBPS` | 75 | 폰 1대당. 80/80은 완전 크래시 확정 |
| `MAX_DURATION_S` | 420 | 60/60도 500초 넘기면 완전 크래시 |
| `COOLDOWN_S` | 60 | run 사이 AP 회복 여유 |
| 금지 조합 | 합계 ≥ 160Mbps, 또는 두 폰 모두 ≥ 80 | 다중 station 고부하 |

범위를 벗어난 요청은 `422`로 거부한다. 상세는 `docs/yongsang/ap_crash_analysis.md`.

## 3. 면 ① — 백엔드 REST + SSE (브라우저 ↔ 노트북 백엔드)

Base URL: `http://<노트북>:8000`

| Method | Path | 용도 |
|---|---|---|
| GET | `/` | 대시보드 HTML |
| GET | `/api/health` | 시스템 상태 (폰·파이·AP·iperf3 서버) |
| GET | `/api/devices` | 부하 클라이언트 목록/상태 |
| GET | `/api/presets` | 안전 프리셋 목록 |
| GET | `/api/load` | 현재 실행 중인 부하 |
| POST | `/api/load` | 부하 시작 |
| DELETE | `/api/load` | 부하 정지 (전체 또는 `?target=s21`) |
| GET | `/api/steer` | 밴드 스티어링 상태 (§9, 미착수 시 `501`) |
| POST | `/api/steer` | 스티어링 `mode` 전환 또는 수동 steer (§9) |
| GET | `/api/stream` | **SSE** — 추론 프레임 + 부하 상태 + 스티어링 + 경보 |
| GET | `/api/history?n=120` | 최근 샘플(스파크라인 백필) |

### GET /api/health

```json
{
  "backend": "ok",
  "iperf3_servers": { "5201": true, "5202": true },
  "pi_stream": { "connected": true, "last_sample_age_s": 0.4 },
  "ap": { "reachable": true, "poll_stall": false },
  "devices": [ { "id": "s21", "reachable": true }, { "id": "s26", "reachable": true } ],
  "cooldown_until": null
}
```

### GET /api/devices

```json
[
  { "id": "s21", "name": "191", "ip": "192.168.8.191", "reachable": true,
    "agent": "ssh", "iperf3": "3.16", "wake_lock": true, "running": null },
  { "id": "s26", "name": "S26", "ip": "192.168.8.103", "reachable": true,
    "agent": "ssh", "iperf3": "3.16", "wake_lock": true,
    "running": { "run_id": "r-20260827-140312", "elapsed_s": 42 } }
]
```

### POST /api/load

Request — 프로필 직접 지정:

```json
{
  "targets": ["s21", "s26"],
  "profile": { "mode": "udp_smallpkt", "rate_mbps": 25, "packet_bytes": 250, "duration_s": 300 }
}
```

또는 프리셋 지정:

```json
{ "preset_id": "smallpkt_25_250" }
```

Response `202 Accepted`:

```json
{
  "run_id": "r-20260827-140312",
  "targets": ["s21", "s26"],
  "profile": { "mode": "udp_smallpkt", "rate_mbps": 25, "packet_bytes": 250, "duration_s": 300 },
  "started_at": "2026-08-27T14:03:12+09:00",
  "expires_at": "2026-08-27T14:08:12+09:00"
}
```

에러:

| code | body `error` | 상황 |
|---:|---|---|
| 422 | `unsafe_profile` | `rate_mbps` > 75, `duration_s` > 420, 금지 조합 등 (`detail`에 사유) |
| 409 | `already_running` / `cooldown` | 실행 중이거나 쿨다운 (`retry_after_s`) |
| 503 | `device_unreachable` | 대상 폰 ping/agent 무응답 (`targets`에 실패 목록) |

### DELETE /api/load

`?target=` 없으면 전체 정지. 폰이 unreachable이어도 best-effort로 진행하고 `200`을 반환한다(데모 중 항상 멈출 수 있어야 함).

```json
{ "stopped": ["s21", "s26"], "failed": [], "run_id": "r-20260827-140312" }
```

### GET /api/stream (SSE)

`Content-Type: text/event-stream`. 이벤트 종류:

```text
event: sample                     ← 면 ③의 sample 프레임을 그대로 재방출 (초당 1회)
data: { ...InferenceFrame... }

event: load                       ← 부하 상태 변화
data: { "run_id": "r-...", "state": "running", "targets": ["s21","s26"], "elapsed_s": 42, "remaining_s": 258 }

event: alert                      ← 주의 필요
data: { "kind": "poll_stall", "detail": "AP SSH 폴링 8s 무응답", "since": "2026-08-27T14:05:01+09:00" }

event: steer                      ← 밴드 스티어링 상태 변화 (§9)
data: { ...SteerState... }
```

`alert.kind`: `ap_unreachable` · `poll_stall` · `device_lost` · `cooldown` · `auto_stopped`.

### GET /api/history

```json
{ "samples": [ { "ts": "...", "features": {...}, "congestion_score": 0.62, "label": 2 }, ... ] }
```

최근 `n`개(기본 120, 최대 600). 링버퍼만 유지하며 영구 저장하지 않는다.

## 4. 면 ② — 부하 에이전트 (노트북 백엔드 → 폰)

두 방식 중 하나를 쓴다. **SSH exec가 폰 쪽 코드 0줄로 가장 간단**하고, HTTP 에이전트가 상태 조회·확장에 유리하다.

### 4.A HTTP 에이전트

폰 Termux에서 ~40줄 스크립트 실행. Base URL: `http://<폰>:8787`

| Method | Path | 용도 |
|---|---|---|
| GET | `/health` | 에이전트·환경 상태 |
| GET | `/load` | 현재 run 상태 |
| POST | `/load` | 부하 시작 |
| DELETE | `/load` | 현재 run 정지 |

`GET /health`:

```json
{ "agent": "0.1", "iperf3": "3.16", "wake_lock": true, "battery_exempt": true, "load": null }
```

`POST /load` — body = `LoadProfile` + 싱크 정보:

```json
{
  "server": "192.168.8.226", "port": 5201,
  "mode": "udp_smallpkt", "rate_mbps": 25, "packet_bytes": 250, "duration_s": 300, "parallel": 1
}
```

Response `201`:

```json
{
  "run_id": "s21-140312",
  "cmd": "iperf3 -u -c 192.168.8.226 -p 5201 -l 250 -b 25M -t 300 -P 1",
  "started_at": "2026-08-27T14:03:12+09:00"
}
```

- 에이전트도 `MAX_RATE_MBPS` / `MAX_DURATION_S`를 자체 검증한다(백엔드 우회 방지). 범위 밖이면 `422`.
- `duration_s` 만료 시 프로세스가 스스로 종료되고 `GET /load`는 `null`을 반환한다.
- 화면 꺼짐 대비 스크립트 시작 시 `termux-wake-lock` 실행. 배터리 최적화 예외는 수동 설정.

### 4.B SSH exec (폰 쪽 코드 없음)

전제: Termux `openssh` 설치, `sshd` (포트 8022) 상시, 노트북 공개키가 `~/.ssh/authorized_keys`에 등록, 노트북 `~/.ssh/config`에 `Host s21` / `Host s26` 항목.

| 동작 | 명령 (노트북에서) |
|---|---|
| 시작 | `ssh -o BatchMode=yes s21 'termux-wake-lock; nohup iperf3 -u -c 192.168.8.226 -p 5201 -l 250 -b 25M -t 300 >/dev/null 2>&1 & echo $!'` |
| 정지 | `ssh s21 'pkill -f "iperf3 -u -c"'` |
| 상태 | `ssh s21 'pgrep -af "iperf3 -u -c" \|\| echo idle'` |

백엔드는 반환된 PID를 `run_id`로 매핑해 관리한다.

## 5. 면 ③ — 추론 스트림 (파이 → 노트북 백엔드)

Base URL: `http://<파이>:9000` (유선 관리 서브넷 경유). 브라우저는 이 API에 **직접 접근하지 않는다** — 백엔드가 `/api/stream`으로 중계한다.

| Method | Path | 용도 |
|---|---|---|
| GET | `/stream` | **SSE**, 초당 1회 `sample` 이벤트 |
| GET | `/meta` | 모델·스케일러·설정 메타 |
| GET | `/health` | AP 도달성·폴링 지연·재연결 |

### GET /stream — `sample` 이벤트 (핵심 페이로드, InferenceFrame)

```json
{
  "seq": 12841,
  "ts": "2026-08-27T14:03:22.514+09:00",
  "window_filled": true,
  "features": {
    "throughput_mbps": 43.2, "channel_occupancy_percent": 88.0,
    "tx_retry_ratio": 0.22, "rssi_dbm": -58.0, "rssi_delta_db": -2.0,
    "rssi_moving_avg_dbm": -57.1, "sta_tx_bitrate_mean": 34.5
  },
  "congestion_score": 0.78,
  "sub_scores": { "occupancy": 0.78, "jitter": 0.70, "loss": 0.0, "latency": 0.25 },
  "label": 3,
  "label_name": "심각",
  "early_exit": { "policy": "dynamic", "exit_taken": 1, "confidence": 0.991, "infer_ms": 0.55 },
  "connected_clients": 3,
  "ap_reachable": true,
  "steer": { "mode": "proposed", "band": "2g", "recommend": "5g", "clients_on_5g": [], "last_steer": null }
}
```

- `window_filled: false` — 스트림 시작 후 첫 10샘플(약 10초). 이때 `label`/`congestion_score`는 잠정값이며 대시보드는 "warming up"으로 표시한다.
- `early_exit.exit_taken` ∈ {1, 2, 3}. `policy`는 `fixed` 또는 `dynamic`.
- `infer_ms` — 파이 온보드 추론 wall time(대시보드에서 Early Exit 이득 시연용).
- `steer` — 밴드 스티어링 상태(§9). 스티어링을 안 쓰면 `"mode": "off"`로 고정.

### GET /meta

```json
{
  "model": { "arch": "APEarlyExitLSTM", "onnx": "ap_early_exit_fixed_unified_int8_v2.onnx",
             "checkpoint": "ap_early_exit_lstm_best.pth",
             "trained_on_rows": 2115, "trained_at": "2026-08-29" },
  "scaler": { "file": "scaler_params.json", "sha256": "…" },
  "features": ["throughput_mbps","channel_occupancy_percent","tx_retry_ratio",
               "rssi_dbm","rssi_delta_db","rssi_moving_avg_dbm","sta_tx_bitrate_mean"],
  "window_size": 10,
  "poll_interval_s": 1.0,
  "labels": { "0": "정상", "1": "경고", "2": "혼잡", "3": "심각" },
  "congestion_formula": "max(occupancy, jitter, loss, latency)",
  "congestion_anchors": {
    "occupancy": [40, 55, 75, 90], "jitter": [20, 30, 50, 100],
    "loss": [0.5, 1.0, 5.0, 10.0], "latency": [30, 60, 150, 400]
  },
  "label_thresholds": [0.25, 0.50, 0.75]
}
```

대시보드는 시작 시 `/meta`를 1회 읽어 feature 순서·앵커·문턱을 표시에 쓴다. **`scaler.sha256`이 학습 때와 다르면 결과를 신뢰하지 말 것**(재라벨링/재변환 후 ONNX·scaler 동기 안 맞음 신호 — feature 개수가 바뀔 때마다(9→6→7) 실제로 반복된 문제였다).

### GET /health

```json
{
  "ap_reachable": true,
  "poll_latency_ms": 34,
  "poll_stall": false,
  "reconnects": 0,
  "last_sample_age_s": 0.4,
  "poller_uptime_s": 5123
}
```

`poll_stall: true`(폴링 응답이 임계치 초과, 기본 5s) 또는 `ap_reachable: false`면 백엔드가 `alert` 이벤트를 내보내고 진행 중인 부하 자동 중지를 제안한다.

## 6. 데모 1회 실행 순서

1. **파이 부팅** → `APPoller`가 유선 SSH로 AP 연결 → `/stream` 방송 시작(첫 10초 `window_filled: false`).
2. **노트북 백엔드 기동** → `iperf3 -s -p 5201` / `-p 5202` → `/api/stream`이 파이 `/stream` 구독 후 브라우저에 재방출.
3. **브라우저에서 대시보드 열기** → `/meta` 1회 읽기 → `EventSource("/api/stream")` 연결 → 게이지·스파크라인 갱신 시작.
4. **발표자가 프리셋 버튼 클릭** (예: "소패킷 25M") → `POST /api/load { "preset_id": "smallpkt_25_250" }`.
5. 백엔드가 **안전 범위 검증** → 각 폰에 `POST /load`(또는 SSH exec).
6. 폰들이 `iperf3 -u -c ...` 실행 → 채널 부하 상승.
7. **약 10초 후** 파이 윈도우가 새 상태로 채워지며 `congestion_score`·`label` 상승 → 게이지 `1 → 2 → 3`.
8. **(스티어링 모드일 때, §9)** `label ≥ 2`가 `STEER_HOLD_S` 유지되면 파이가 victim 클라이언트에 `bss_tm_req` 전송 → 5GHz로 이동 → 2.4GHz occupancy·retry 하강 → 게이지 `3 → 1`. 대시보드에 "s26 → 5GHz" 배지 + victim flow 회복 그래프.
9. `duration_s` 만료 또는 **"전체 정지"** → `DELETE /api/load` → 폰 부하 종료 → 게이지 하강. 스티어링은 `COOLDOWN_S` 후 2.4GHz로 복귀(또는 수동).
10. `COOLDOWN_S` 후 다음 프리셋.

## 7. 안전 · 배포 · 비목표

### 안전 (재확인)
- `POST /api/load` · `POST /load` 양쪽에서 안전 상수 검증(백엔드 우회 방지).
- `DELETE /api/load`는 항상 즉시·best-effort. 폰 무응답이어도 `200`.
- 파이 `/health`의 `poll_stall` / `ap_reachable:false` → 백엔드 `alert` + 자동 중지 제안.
- 프리셋은 전부 안전 범위 안. 자유 입력 프로필은 개발 모드에서만 노출 권장.
- 스티어링(§9): `STEER_COOLDOWN_S`·`STEER_MAX_PER_MIN`으로 플랩 방지. deauth 폴백은 클라이언트 연결을 실제로 끊으므로 `bss_tm_req` 실패 시에만. `mode` 전환은 진행 중인 run에도 즉시 반영.

### 배포
- **LAN 전용**, 외부 노출 금지. 인증 없음 또는 `X-Demo-Token` 공유 헤더 1개.
- 포트: 백엔드 `8000`, 파이 `9000`, 폰 에이전트 `8787`, iperf3 `5201`/`5202`, Termux sshd `8022`.
- CORS: 대시보드와 백엔드가 동일 origin이면 불필요. 브라우저→파이 직접 요청은 없음(백엔드 중계).
- 파이는 유선 관리 서브넷에서만 `9000`을 listen. 무선 인터페이스로 노출 금지.

### 비목표
- 다중 동시 세션/발표자, 인증·권한·감사 로그.
- 부하 명령 큐잉(한 번에 한 run).
- 샘플 영구 저장(링버퍼만).
- 모바일 대시보드 레이아웃(노트북 화면 전제).

## 8. HTTP 상태 코드 요약

| code | 의미 | 사용처 |
|---:|---|---|
| 200 | 정상 | GET 전반, `DELETE /api/load`, `POST /api/steer` |
| 201 | 생성됨 | `POST /load` (에이전트) |
| 202 | 수락됨(비동기) | `POST /api/load` |
| 409 | 충돌 | 실행 중 / 쿨다운 / 스티어링 쿨다운 |
| 422 | 검증 실패 | 안전 범위 밖 프로필, 알 수 없는 steer 대상 |
| 501 | 미구현 | 스티어링 미착수 상태 |
| 503 | 하위 노드 도달 불가 | 폰/파이 무응답 |
| 504 | 상류 타임아웃 | AP 폴링 스톨 |

## 9. 밴드 스티어링 (실험 모드 · 미착수)

혼잡 감지 결과를 **행동**으로 연결하는 확장. LSTM이 2.4GHz 혼잡을 판단하면 파이가 AP에 명령해서 victim 클라이언트를 5GHz로 옮긴다. 배경·신규성 프레이밍·실험 설계는 `.work-log/current.md` "향후 시스템 구상 — 혼잡 감지 기반 밴드 스티어링".

**현재 상태**: 2.4GHz 데이터 수집에 집중 중이라 미착수. 이 절은 API 자리만 잡아둔 것. 미착수 동안 모든 steer 엔드포인트는 `501`을 반환하고 `sample.steer.mode`는 `"off"`.

### 9.1 SteerState (공유 계약)

```json
{
  "mode": "off",                 // "off" | "static" | "proposed" | "forced_2g" | "forced_5g"
  "band": "2g",                  // 현재 victim이 붙어있는 밴드
  "recommend": "5g",             // 정책이 권하는 밴드 (mode=off면 항상 현재 밴드)
  "clients_on_5g": ["s26"],
  "last_steer": {
    "ts": "2026-08-27T14:05:40+09:00",
    "target": "s26", "from": "2g", "to": "5g",
    "trigger": "label>=2 for 6s", "method": "bss_tm_req", "accepted": true
  },
  "cooldown_until": "2026-08-27T14:06:40+09:00"
}
```

**`mode`** — 이게 실험의 독립 변수다:
- `off` — 스티어링 안 함 (baseline)
- `static` — `channel_occupancy_percent > STEER_OCC_THRESHOLD`가 `STEER_HOLD_S` 유지되면 전환 (문턱 방식 비교군)
- `proposed` — LSTM `label ≥ STEER_LABEL_THRESHOLD`가 `STEER_HOLD_S` 유지되면 전환 (제안 방식)
- `forced_2g` / `forced_5g` — 수동 고정 (데모용)

### 9.2 파이(면 ③)

| Method | Path | 용도 |
|---|---|---|
| GET | `/steer` | 현재 `SteerState` + 정책 파라미터 |
| POST | `/steer` | `{ "mode": "proposed" }` 또는 `{ "action": "steer", "target": "s26", "to": "5g" }` (수동) |

`/meta`에 정책 파라미터 추가:

```json
"steer": {
  "modes": ["off","static","proposed","forced_2g","forced_5g"],
  "STEER_LABEL_THRESHOLD": 2,
  "STEER_OCC_THRESHOLD": 80,
  "STEER_HOLD_S": 6,
  "STEER_COOLDOWN_S": 60,
  "STEER_MAX_PER_MIN": 3,
  "method": "bss_tm_req",          // "bss_tm_req"(802.11v) | "dawn" | "deauth"(폴백)
  "victim": "s26",                 // 옮길 대상 클라이언트 (MAC/id)
  "bands": { "2g": "GL-SFT1200-a08", "5g": "GL-SFT1200-a08" }   // 같은 SSID 필수
}
```

- 파이가 **모델·정책·액추에이션을 다 한다**(최저 지연, 단일 결정점). 액추에이터는 파이→AP 유선 SSH로 `hostapd_cli bss_tm_req <mac> pref=1 neighbor=<5g_bssid>,...` 실행. 안 먹으면 `dawn` ubus 호출 또는 2.4 deauth 폴백.
- 플랩 방지: `STEER_COOLDOWN_S` + `STEER_MAX_PER_MIN` + `STEER_HOLD_S`(문턱을 잠깐 스친 게 아니라 유지돼야 발동).

### 9.3 백엔드(면 ①)

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/steer` | 파이 `/steer` 중계 |
| POST | `/api/steer` | 파이 `/steer`로 프록시 (`mode` 전환 또는 수동 steer) |

- `/api/stream`에 `steer` 이벤트 추가: `data: { ...SteerState... }` (상태 변화 시).
- 대시보드: `mode` 선택 컨트롤(off/static/proposed/수동) + 현재 밴드 배지 + `last_steer` 타임라인 + victim flow throughput 그래프(전환 전후 회복 시각화).

### 9.4 실험 모드 (캡스톤 측정)

같은 부하 프리셋을 `mode` 3개로 각각 실행:

| mode | 측정 |
|---|---|
| `off` | victim flow throughput/loss/latency 시계열 (회복 안 됨 또는 느림) |
| `static` | occupancy 문턱 전환 — 전환 시각, 오탐 횟수, 회복 속도 |
| `proposed` | LSTM 전환 — **더 일찍 감지하는가? 오탐이 적은가? 회복이 빠른가?** |

이게 "occupancy만 보는 분류기 대비 LSTM 우위"를 downstream 지표로 증명하는 자리다(`README_AP_V2.md` "핵심 검증 질문").

### 9.5 전제 (미착수 이유이기도 함)

- 양 밴드 **같은 SSID**로 방송해야 seamless. 지금은 `-a08`/`-a08-5G` 따로 → 통합 필요.
- Opal 5GHz 채널 40 → 집 공유기와 충돌, 36/149로 이동.
- 이상적으로는 **5GHz 혼잡 데이터**도 필요(5GHz도 나쁠 때가 있으니). 데모 수준이면 "5GHz는 항상 여유" 가정.
- 191 폰의 802.11v 지원 여부 미확인 (S26은 지원).

## 참고

- `.work-log/current.md` — "향후 데모 구상", "향후 시스템 구상 — 혼잡 감지 기반 밴드 스티어링", 세션별 진행
- `docs/yongsang/ap_crash_analysis.md` — 안전 부하 범위·부하 생성 방법 대안의 근거
- `project/README_AP_V2.md` — 모델·feature·congestion_score 정의, "핵심 검증 질문"
- `project/utils/ap_features.py` — FeatureVector 순서의 정본 (7-feature, `sta_tx_bitrate_mean` 2026-08-29~)
- `project/scripts/collect_metrics.py`의 `calculate_scores()`/`ANCHORS` — congestion_score(max/anchor 방식) 정본
- `docs/yongsang/onnx_early_exit_redesign.md` — 배포용 ONNX(unified INT8 v2) 재설계 기록, Pi latency 수치
- `docs/yongsang/congestion_label_redesign.{md,html}` — congestion_score 가중합→max 방식 재설계 배경
