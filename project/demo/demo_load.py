"""폰별 부하 제어 (SSH → iperf3) + 연결 확인.

`set_phone_load()`가 이 파일의 핵심 — 지정한 폰에만 iperf3 UDP 부하를 걸고, 서버 타이머로
`LOAD_DURATION_S` 뒤 자동 종료한다(같은 폰에 새 요청이 오면 기존 타이머는 취소). `demo_api.py`의
`Handler`가 이 모듈의 함수만 호출하고 SSH/부하 로직은 전혀 모른다.
"""
from __future__ import annotations

import subprocess
import threading

import demo_state
from demo_state import (
    ARGS, PHONES, ALLOWED_RATES, ALLOWED_PACKET_SIZES, DEFAULT_PACKET_SIZE,
    LOAD_DURATION_S, PHONE_MAC_PREFIX, SSH_CMD,
)


def _ssh(host: str, cmd: str, timeout: int = 8) -> tuple[int, str]:
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, cmd],
                           capture_output=True, text=True, timeout=timeout + 4)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def _broadcast_load() -> None:
    snapshot = dict(demo_state._load_state)
    with demo_state._lock:
        if demo_state._state.get("ready"):
            demo_state._state["load"] = snapshot
    demo_state._broadcast({"load": snapshot})


def _stop_phone(name: str) -> None:
    """타이머 콜백 — LOAD_DURATION_S 뒤 자동 호출. 수동 정지도 이 경로를 탄다."""
    ph = PHONES[name]
    _ssh(ph["ssh"], "pkill -f iperf3 2>/dev/null; true")
    demo_state._load_state[name]["rate"] = "off"
    demo_state._load_timers.pop(name, None)
    _broadcast_load()


def set_phone_load(name: str, rate: str, packet_size: int = DEFAULT_PACKET_SIZE) -> dict:
    if name not in PHONES:
        return {"ok": False, "error": f"알 수 없는 폰: {name} (허용: {sorted(PHONES)})"}
    if rate not in ALLOWED_RATES:
        return {"ok": False, "error": f"허용 rate: {sorted(ALLOWED_RATES)}"}
    if packet_size not in ALLOWED_PACKET_SIZES:
        return {"ok": False, "error": f"허용 packet_size: {sorted(ALLOWED_PACKET_SIZES)}"}

    old_timer = demo_state._load_timers.pop(name, None)
    if old_timer:
        old_timer.cancel()

    ph = PHONES[name]
    _ssh(ph["ssh"], "pkill -f iperf3 2>/dev/null; true")

    if rate == "off":
        demo_state._load_state[name]["rate"] = "off"
        _broadcast_load()
        return {"ok": True, "phone": name, "rate": "off"}

    # -t 는 LOAD_DURATION_S + 여유분 — 서버 타이머가 실패해도 iperf3 자체가 끝남 (이중 안전장치).
    # -l packet_size: 1400=일반, 200=소패킷 (같은 bitrate라도 초당 패킷 수↑ → occupancy·retry↑).
    cmd = (f"nohup iperf3 -u -c {ARGS.iperf_target} -p {ph['port']} -b {rate} -l {packet_size} "
           f"-t {LOAD_DURATION_S + 5} >/dev/null 2>&1 & echo started")
    rc, out = _ssh(ph["ssh"], cmd)
    if rc != 0:
        return {"ok": False, "error": f"{name} 부하 시작 실패: {out}"}

    demo_state._load_state[name] = {"rate": rate, "packet_size": packet_size}
    timer = threading.Timer(LOAD_DURATION_S, _stop_phone, args=(name,))
    timer.daemon = True
    timer.start()
    demo_state._load_timers[name] = timer
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
    with demo_state._lock:
        msg = str(demo_state._state.get("msg", ""))
        ap_ready = bool(demo_state._state.get("ready")) or (
            "AP 응답 없음" not in msg and demo_state._state.get("features") is not None
        )
    result["ap"] = ap_ready
    for name, ph in PHONES.items():
        rc, _ = _ssh(ph["ssh"], "echo ok", timeout=4)
        result[name] = (rc == 0)
    result["all_ok"] = result["pi"] and result["ap"] and result.get("s21", False) and result.get("s26", False)
    return result
