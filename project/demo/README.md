# 라이브 혼잡 감지 데모

버튼으로 계단형 부하를 걸고, 모델이 실시간으로 혼잡 레벨을 어떻게 예측하는지 웹에서 본다.

> **팀 구현용 상세 스펙: [`API.md`](API.md)** — API·SSE 스키마·모델 계약·크래시 주의·확장 항목.
> 이 README는 빠른 실행용.

## 실행 (노트북)

```bash
python project/demo/demo_server.py
```

브라우저에서 <http://localhost:8000/> 열기.

## 사전 조건

| 항목 | 확인 |
|---|---|
| AP SSH | `ssh root@192.168.8.1` (collect_metrics.py 와 동일 키) |
| 폰 SSH | `ssh s21 echo ok` / `ssh s26 echo ok` (Termux sshd, `~/.ssh/config`) |
| 폰 iperf3 대상 | 노트북 wifi IP = `192.168.8.226` (`demo_server.py` `IPERF_TARGET`) |
| Python | `onnxruntime`, `numpy` (capstone 환경) |
| ONNX / scaler | `checkpoints/ap_v2_redesign2/ap_early_exit_fixed_unified_int8_v2.onnx`, `data/ap_metrics_v2_redesign2/scaler_params.json` |

`iperf3 -s` 서버(5201/5202)는 `demo_server.py` 가 자동 기동한다.

## 화면

- **모델 출력 (원시)** — 매 폴링 argmax 라벨 + 4클래스 확률 바. **리포트의 정확도 수치는 전부 이것 기준.**
- **안정화 상태** — 원시 예측이 3폴링 연속 같을 때만 갱신하는 debounce 라벨(표시/제어용). **모델·학습·평가엔 없는 후처리** — 경계 구간의 폴링 단위 흔들림만 억제.
- **AP 텔레메트리** — 모델 입력 7-feature 실시간값 + early-exit 지점
- **부하 제어** — `10 / 20 / 30 Mbps` 버튼(두 폰 동시 iperf3 UDP), `정지`. 30M 상한.
- **최근 90초 차트** — 안정화 상태(굵은 선) / 모델 출력 원시(점선) / 채널 점유율(%)
- 폰 신호세기(S21/S26) — 12 dBm 이상 차이나면 비대칭 경고(낮은 부하에도 AP 크래시 위험)

## 주의

- **AP 크래시**: 신호 비대칭 + 20M 이상이면 AP가 죽을 수 있다(`docs/yongsang/ap_crash_analysis.md`). 신호 확인 후 부하, 이상하면 즉시 정지.
- 부하는 `iperf3 -u -t 3600` 으로 계속 흐르다가 다음 버튼(또는 정지)에서 `pkill iperf3` 로 교체된다. 서버 종료 시 자동 정지.

## 구조

`demo_server.py` 는 `scripts/collect_metrics.py` 의 `APPoller` + 파서를 재사용해
`scripts/live_congestion.py` 와 동일하게 feature 를 계산한다(victim 프로브 없음).
백엔드 SSE + POST `/load` 만 추가한 얇은 래퍼. `demo.html` 은 서버가 서빙(동일 출처).
