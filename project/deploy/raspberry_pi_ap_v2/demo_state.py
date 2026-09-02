"""공유 설정·상태 — demo_inference.py / demo_load.py / demo_api.py / demo_server.py가 공통으로 쓴다.

sys.path 부트스트랩과 collect_metrics dual-import(저장소/Pi 번들 양쪽 대응)를 여기서 한 번만 하고,
나머지 모듈은 이 모듈에서 필요한 이름을 가져다 쓴다. `_state`/`_load_state`/`_load_timers`처럼
언더스코어가 붙은 것들은 모듈 간 공유되는 가변 상태다 — 재할당(rebind)이 필요한 쪽(주로
demo_inference.py의 `_state`)은 `import demo_state`로 모듈 자체를 들고 `demo_state._state = ...`
식으로 접근해야 다른 모듈에서도 새 값이 보인다. `from demo_state import _state`로 이름만
가져오면 그 모듈 안에서의 재할당이 공유되지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from pathlib import Path

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
