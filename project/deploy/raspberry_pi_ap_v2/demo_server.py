"""데모 서버 — 참가형 웹사이트로 폰별 부하를 걸고, 모델의 실시간 혼잡 예측을 웹으로 본다.

노트북에서 실행 (기본값):
    python project/demo/demo_server.py
라즈베리 파이(엣지 추론)에서 실행:
    python3 demo_server.py --iperf-target <파이 IP> --s21 <user@phone> --s26 <user@phone>
브라우저에서 http://<host>:8000/ 열기.

구성 (stdlib http.server + numpy + onnxruntime 만):
  - 백그라운드 스레드: collect_metrics.APPoller 로 AP `iw` 폴링 → 7-feature →
    window 10 → scaler 정규화 → ONNX(unified INT8) 추론 → 라벨/확률.
    (live_congestion.py 와 동일 로직, victim 프로브 없음)
  - GET  /            데모 페이지
  - GET  /events      SSE — 상태를 ~2Hz 로 스트리밍
  - POST /load        {"phone": "s21"|"s26", "rate": "10M"|"20M"|"30M"|"40M"|"off",
                        "packet_size": 1400|200 (생략 시 1400)}
                       해당 폰에만 iperf3 부하. "off" 외 rate는 LOAD_DURATION_S(10초) 뒤 서버가 자동 종료.
                       packet_size=200(소패킷)은 같은 bitrate라도 초당 패킷 수가 많아 occupancy·
                       retry가 더 크게 뜬다 — bitrate 하나만으로는 "그냥 많이 넣으면 그만" 처럼
                       보인다는 피드백(2026-09-02)에 대응해 추가한 두 번째 부하 축.
  - GET  /signal      두 폰의 현재 신호세기 (부하 전 대칭 확인용)
  - GET  /check       Pi(서버 자신)·AP·S21·S26 연결 상태 일괄 확인 ("연결확인" 버튼용)
  - GET  /health

폰별 최대 40M · 10초 자동종료 (2026-09-02 참가형 데모용 — 기존 "두 폰 동시 30M 무제한"보다
훨씬 짧고 한 폰씩이라 안전 여유가 큼. combined 다중분 부하의 크래시 위험은
docs/yongsang/ap_crash_analysis.{md,html} 참고, 단발 10초는 그 영역 밖).
iperf3 -s 서버(포트 2개)는 서버 시작 시 --iperf-target 이 로컬이면 자동 기동.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(HERE))  # Pi 번들: collect_metrics.py 가 옆에 있음

# 저장소에서 실행하면 scripts.collect_metrics, Pi 번들에서 실행하면 collect_metrics.
try:
    from scripts.collect_metrics import (  # noqa: E402
        APPoller, MOVING_AVG_WINDOW, SSH_CMD, calculate_channel_occupancy,
        calculate_station_deltas, parse_ap_cycle, summarize_stations,
    )
except ImportError:
    from collect_metrics import (  # noqa: E402
        APPoller, MOVING_AVG_WINDOW, SSH_CMD, calculate_channel_occupancy,
        calculate_station_deltas, parse_ap_cycle, summarize_stations,
    )

# ---------------------------------------------------------------- static config
FEATURES = (
    "throughput_mbps", "channel_occupancy_percent", "tx_retry_ratio",
    "rssi_dbm", "rssi_delta_db", "rssi_moving_avg_dbm", "sta_tx_bitrate_mean",
)
WINDOW = 12  # matches utils.ap_features.WINDOW_SIZE + the [1,12,7] ONNX (10->12, 2026-09-01).
CONFIRM = 5                              # 라벨 히스테리시스 (표시/제어용 후처리, 모델·평가엔 없음)
LABELS = ["정상", "경고", "혼잡", "심각"]
ALLOWED_RATES = {"10M", "20M", "30M", "40M", "off"}
ALLOWED_PACKET_SIZES = {1400, 200}       # -l 값. 1400=일반(기본), 200=소패킷(같은 bitrate라도
                                          # 초당 패킷 수가 많아 occupancy·retry가 더 크게 뜸 —
                                          # 16차 데이터 수집(smallpkt_*)에서 검증된 축.
DEFAULT_PACKET_SIZE = 1400
LOAD_DURATION_S = 10                     # 부하 버튼 클릭 시 자동 종료까지 걸리는 시간
PHONE_MAC_PREFIX = {"s21": "06:0f", "s26": "ca:79"}  # dhcp.leases 기준

MODEL_NAME = "ap_early_exit_fixed_unified_int8_v2.onnx"


def _resolve(repo_path: Path, bundle_name: str) -> Path:
    """저장소 경로가 있으면 그걸, 없으면 스크립트 옆(Pi 번들)을 쓴다."""
    return repo_path if repo_path.exists() else (HERE / bundle_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DEMO_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "8000")))
    p.add_argument("--iperf-target", default=os.environ.get("DEMO_IPERF_TARGET", "192.168.8.226"),
                   help="폰들이 iperf3 UDP 를 쏠 대상 IP. 노트북 실행 시 노트북 wifi IP, "
                        "파이 실행 시 파이 자신의 192.168.8.x IP.")
    p.add_argument("--s21", default=os.environ.get("DEMO_S21", "s21"),
                   help="S21 폰 SSH 대상 (host 별칭 또는 user@ip). 노트북은 ~/.ssh/config 별칭, "
                        "파이는 user@192.168.8.x 로.")
    p.add_argument("--s26", default=os.environ.get("DEMO_S26", "s26"))
    p.add_argument("--s21-port", type=int, default=5201, help="S21 부하용 iperf3 -s 포트")
    p.add_argument("--s26-port", type=int, default=5202, help="S26 부하용 iperf3 -s 포트")
    p.add_argument("--model", type=Path,
                   default=_resolve(PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2" / MODEL_NAME,
                                    MODEL_NAME))
    p.add_argument("--scaler", type=Path,
                   default=_resolve(PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2" / "scaler_params.json",
                                    "scaler_params.json"))
    p.add_argument("--confirm", type=int, default=CONFIRM)
    p.add_argument("--no-iperf-server", action="store_true",
                   help="iperf3 -s 자동 기동 안 함 (대상 호스트에서 이미 돌고 있을 때)")
    return p.parse_args()


ARGS = parse_args()
PHONES = {
    "s21": {"ssh": ARGS.s21, "port": ARGS.s21_port},
    "s26": {"ssh": ARGS.s26, "port": ARGS.s26_port},
}

# ---------------------------------------------------------------- shared state
_lock = threading.Lock()
_state: dict = {"ready": False, "msg": "시작 중..."}
_clients: list[queue.Queue] = []
_load_state: dict = {
    "s21": {"rate": "off", "packet_size": DEFAULT_PACKET_SIZE},
    "s26": {"rate": "off", "packet_size": DEFAULT_PACKET_SIZE},
}
_load_timers: dict[str, threading.Timer] = {}


def _broadcast(obj: dict) -> None:
    data = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
    with _lock:
        dead = []
        for q in _clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


# ---------------------------------------------------------------- inference loop
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def norm(feats: dict, scaler: dict) -> np.ndarray:
    out = np.empty(len(FEATURES), dtype=np.float32)
    for i, k in enumerate(FEATURES):
        lo, hi = scaler[k]["min"], scaler[k]["max"]
        v = 0.0 if hi == lo else (feats[k] - lo) / (hi - lo)
        out[i] = min(1.0, max(0.0, v))
    return out


def inference_loop() -> None:
    global _state
    scaler = json.loads(ARGS.scaler.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(ARGS.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    poller = APPoller()

    win: deque = deque(maxlen=WINDOW)
    occ_h: deque = deque(maxlen=3)
    retx_h: deque = deque(maxlen=MOVING_AVG_WINDOW)
    pkts_h: deque = deque(maxlen=MOVING_AVG_WINDOW)
    rssi_h: deque = deque(maxlen=MOVING_AVG_WINDOW)
    streak: deque = deque(maxlen=ARGS.confirm)

    prev_st = prev_active = prev_busy = prev_rssi = prev_time = None
    last_id = -1
    confirmed = None
    stale_since = time.time()

    while True:
        cyc = poller.wait_for_new_cycle(last_id)
        if cyc is None:
            if time.time() - stale_since > 6:
                with _lock:
                    _state = {"ready": False, "msg": f"AP 응답 없음 (재연결 {poller.reconnects}회)"}
                _broadcast(_state)
            continue
        stale_since = time.time()
        last_id, text = cyc
        station, survey = parse_ap_cycle(text)
        if station is None:
            continue
        cur_active, cur_busy, *_ = survey
        now = time.time()

        signal_avg, n_clients, _mn, bitrate_mean = summarize_stations(station, prev_st)
        occ_raw, _ = calculate_channel_occupancy(prev_active, prev_busy, cur_active, cur_busy)
        occ_h.append(occ_raw)
        occ = median(occ_h)

        rx_d, tx_d, retr_d, fail_d, pkts_d = calculate_station_deltas(prev_st, station)
        elapsed = (now - prev_time) if prev_time else 0.0
        thr = round((rx_d + tx_d) * 8 / elapsed / 1e6, 2) if elapsed > 0 else 0.0

        retx_h.append(retr_d + fail_d)
        pkts_h.append(pkts_d)
        denom = sum(retx_h) + sum(pkts_h)
        retry = round(sum(retx_h) / denom, 4) if denom >= 50 else 0.0

        rssi = signal_avg
        rssi_d = 0.0 if prev_rssi is None else rssi - prev_rssi
        rssi_h.append(rssi)
        rssi_ma = sum(rssi_h) / len(rssi_h)

        feats = {
            "throughput_mbps": thr, "channel_occupancy_percent": round(occ, 2),
            "tx_retry_ratio": retry, "rssi_dbm": round(rssi, 2),
            "rssi_delta_db": round(rssi_d, 2), "rssi_moving_avg_dbm": round(rssi_ma, 2),
            "sta_tx_bitrate_mean": round(bitrate_mean, 1),
        }
        prev_st, prev_active, prev_busy = station, cur_active, cur_busy
        prev_rssi, prev_time = rssi, now
        win.append(norm(feats, scaler))

        if len(win) < WINDOW:
            with _lock:
                _state = {"ready": False, "msg": f"창 채우는 중 {len(win)}/{WINDOW}",
                          "features": feats, "clients": n_clients}
            _broadcast(_state)
            continue

        x = np.stack(win)[None, :, :].astype(np.float32)
        outs = sess.run(None, {in_name: x})
        probs = softmax(np.asarray(outs[0]).reshape(-1))
        exit_pt = int(np.asarray(outs[1]).reshape(-1)[0]) if len(outs) > 1 else None
        raw = int(probs.argmax())
        streak.append(raw)
        if len(streak) == ARGS.confirm and len(set(streak)) == 1 and streak[0] != confirmed:
            confirmed = streak[0]
        shown = confirmed if confirmed is not None else raw

        with _lock:
            _state = {
                "ready": True, "ts": time.strftime("%H:%M:%S"),
                "label": shown, "label_name": LABELS[shown],
                "raw_label": raw, "raw_name": LABELS[raw],
                "probs": [round(float(p), 3) for p in probs],
                "exit": exit_pt, "clients": n_clients,
                "features": feats, "load": dict(_load_state),
            }
        _broadcast(_state)


# ---------------------------------------------------------------- load control
def _ssh(host: str, cmd: str, timeout: int = 8) -> tuple[int, str]:
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, cmd],
                           capture_output=True, text=True, timeout=timeout + 4)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def _broadcast_load() -> None:
    snapshot = dict(_load_state)
    with _lock:
        if _state.get("ready"):
            _state["load"] = snapshot
    _broadcast({"load": snapshot})


def _stop_phone(name: str) -> None:
    """타이머 콜백 — LOAD_DURATION_S 뒤 자동 호출. 수동 정지도 이 경로를 탄다."""
    ph = PHONES[name]
    _ssh(ph["ssh"], "pkill -f iperf3 2>/dev/null; true")
    _load_state[name]["rate"] = "off"
    _load_timers.pop(name, None)
    _broadcast_load()


def set_phone_load(name: str, rate: str, packet_size: int = DEFAULT_PACKET_SIZE) -> dict:
    if name not in PHONES:
        return {"ok": False, "error": f"알 수 없는 폰: {name} (허용: {sorted(PHONES)})"}
    if rate not in ALLOWED_RATES:
        return {"ok": False, "error": f"허용 rate: {sorted(ALLOWED_RATES)}"}
    if packet_size not in ALLOWED_PACKET_SIZES:
        return {"ok": False, "error": f"허용 packet_size: {sorted(ALLOWED_PACKET_SIZES)}"}

    old_timer = _load_timers.pop(name, None)
    if old_timer:
        old_timer.cancel()

    ph = PHONES[name]
    _ssh(ph["ssh"], "pkill -f iperf3 2>/dev/null; true")

    if rate == "off":
        _load_state[name]["rate"] = "off"
        _broadcast_load()
        return {"ok": True, "phone": name, "rate": "off"}

    # -t 는 LOAD_DURATION_S + 여유분 — 서버 타이머가 실패해도 iperf3 자체가 끝남 (이중 안전장치).
    # -l packet_size: 1400=일반, 200=소패킷 (같은 bitrate라도 초당 패킷 수↑ → occupancy·retry↑).
    cmd = (f"nohup iperf3 -u -c {ARGS.iperf_target} -p {ph['port']} -b {rate} -l {packet_size} "
           f"-t {LOAD_DURATION_S + 5} >/dev/null 2>&1 & echo started")
    rc, out = _ssh(ph["ssh"], cmd)
    if rc != 0:
        return {"ok": False, "error": f"{name} 부하 시작 실패: {out}"}

    _load_state[name] = {"rate": rate, "packet_size": packet_size}
    timer = threading.Timer(LOAD_DURATION_S, _stop_phone, args=(name,))
    timer.daemon = True
    timer.start()
    _load_timers[name] = timer
    _broadcast_load()
    return {"ok": True, "phone": name, "rate": rate, "packet_size": packet_size, "duration_s": LOAD_DURATION_S}


def phone_signals() -> dict:
    try:
        out = subprocess.run(SSH_CMD + ["iw dev wlan0 station dump"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {"error": "AP 조회 실패"}
    per_mac: dict[str, int] = {}
    cur = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Station "):
            cur = line.split()[1].lower()
        elif line.startswith("signal:") and cur:
            try:
                per_mac[cur] = int(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    named: dict = {}
    for name, prefix in PHONE_MAC_PREFIX.items():
        for mac, sig in per_mac.items():
            if mac.startswith(prefix):
                named[name] = sig
    if "s21" in named and "s26" in named:
        named["symmetric"] = abs(named["s21"] - named["s26"]) <= 12
    return named


def check_connectivity() -> dict:
    """"연결확인" 버튼 — Pi(서버 자신)·AP·폰 2대를 한 번에 점검."""
    result: dict = {"pi": True}  # 이 핸들러가 실행 중이라는 것 자체가 서버(파이/노트북)는 살아있다는 뜻
    with _lock:
        msg = str(_state.get("msg", ""))
        ap_ready = bool(_state.get("ready")) or ("AP 응답 없음" not in msg and _state.get("features") is not None)
    result["ap"] = ap_ready
    for name, ph in PHONES.items():
        rc, _ = _ssh(ph["ssh"], "echo ok", timeout=4)
        result[name] = (rc == 0)
    result["all_ok"] = result["pi"] and result["ap"] and result.get("s21", False) and result.get("s26", False)
    return result


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            with _lock:
                self._send(200, json.dumps({"ready": _state.get("ready")}).encode(), "application/json")
        elif self.path == "/signal":
            self._send(200, json.dumps(phone_signals(), ensure_ascii=False).encode("utf-8"), "application/json")
        elif self.path == "/check":
            self._send(200, json.dumps(check_connectivity(), ensure_ascii=False).encode("utf-8"), "application/json")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=50)
            with _lock:
                _clients.append(q)
                snap = dict(_state)
            try:
                self.wfile.write(f"data: {json.dumps(snap, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        chunk = q.get(timeout=15)
                        self.wfile.write(chunk)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _lock:
                    if q in _clients:
                        _clients.remove(q)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/load":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        try:
            packet_size = int(body.get("packet_size", DEFAULT_PACKET_SIZE))
        except (TypeError, ValueError):
            packet_size = -1  # set_phone_load 가 ALLOWED_PACKET_SIZES 로 걸러서 400 반환
        res = set_phone_load(str(body.get("phone", "")), str(body.get("rate", "off")), packet_size)
        self._send(200 if res.get("ok") else 400,
                   json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")


# ---------------------------------------------------------------- main
def _iperf_target_is_local() -> bool:
    t = ARGS.iperf_target
    if t in ("127.0.0.1", "localhost", "0.0.0.0"):
        return True
    try:
        local_ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                                   timeout=4).stdout.split()
        return t in local_ips
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    if not ARGS.model.exists() or not ARGS.scaler.exists():
        sys.exit(f"모델/스케일러 없음: {ARGS.model}  {ARGS.scaler}")

    iperf_procs = []
    spawn = not ARGS.no_iperf_server and _iperf_target_is_local()
    if spawn:
        for ph in PHONES.values():
            iperf_procs.append(subprocess.Popen(
                ["iperf3", "-s", "-p", str(ph["port"])],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        print(f"iperf3 서버 기동: {[p['port'] for p in PHONES.values()]}  (대상 {ARGS.iperf_target})")
    else:
        print(f"iperf3 -s 자동 기동 안 함 — 대상 {ARGS.iperf_target} 에서 직접 "
              f"iperf3 -s -p {ARGS.s21_port} / -p {ARGS.s26_port} 를 띄워 둘 것")

    print(f"모델 : {ARGS.model.name}")
    print(f"폰   : s21={ARGS.s21}  s26={ARGS.s26}")

    threading.Thread(target=inference_loop, daemon=True).start()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"데모 서버: http://{ARGS.host}:{ARGS.port}/   (Ctrl-C 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n정리 중... 부하 정지")
        for name in PHONES:
            set_phone_load(name, "off")
        for p in iperf_procs:
            p.terminate()


if __name__ == "__main__":
    main()
