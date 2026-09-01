"""실시간 AP 혼잡 감지 — 최소 라이브 추론 루프.

실제 AP(GL.iNet Opal)에서 `iw` 텔레메트리를 지속 폴링 → 7-feature 계산 →
window 10 → min-max 정규화(scaler_params.json) → ONNX(unified INT8) 추론 →
혼잡 라벨을 1초 간격으로 터미널에 출력한다.

"스파이크가 아닌 현 상태"를 잡기 위한 4겹 안정화:
  1) LSTM window 10 (~10~20초 이력) — 1폴링 튐은 창을 거의 안 움직임 (주 메커니즘, 학습과 동일)
  2) occupancy = 최근 3폴링 median   (collect_metrics.py와 동일)
  3) tx_retry_ratio = 최근 5폴링 rolling 비율
  4) rssi_moving_avg = 최근 5폴링 평균
  + 라벨 히스테리시스: 원시 예측이 K회 연속 일치해야 '확정 라벨'이 바뀜 (--confirm)

feature 계산은 collect_metrics.py의 헬퍼를 그대로 재사용해 학습 시점과 동일하게 맞춘다.
victim 프로브(라벨링 전용)는 없음 — 추론엔 불필요.

사용:
  python project/scripts/live_congestion.py
  python project/scripts/live_congestion.py --model <onnx> --confirm 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # bundle dir (collect_metrics.py 옆)

# 저장소에서 실행하면 scripts.collect_metrics, Pi 번들에서 실행하면 collect_metrics.
try:
    from scripts.collect_metrics import (  # noqa: E402
        APPoller, MOVING_AVG_WINDOW, calculate_channel_occupancy,
        calculate_station_deltas, parse_ap_cycle, summarize_stations,
    )
except ImportError:
    from collect_metrics import (  # noqa: E402
        APPoller, MOVING_AVG_WINDOW, calculate_channel_occupancy,
        calculate_station_deltas, parse_ap_cycle, summarize_stations,
    )

# utils/ap_features.py AP_FEATURE_COLUMNS 와 반드시 일치 (번들 자립을 위해 인라인).
AP_FEATURE_COLUMNS = (
    "throughput_mbps",
    "channel_occupancy_percent",
    "tx_retry_ratio",
    "rssi_dbm",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
    "sta_tx_bitrate_mean",
)

WINDOW_SIZE = 12  # matches utils.ap_features.WINDOW_SIZE + the [1,12,7] ONNX (10->12, 2026-09-01).
LABEL_NAMES = {0: "정상", 1: "경고", 2: "혼잡", 3: "심각"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ck = PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2"
    # 저장소면 checkpoints/, 번들이면 스크립트 옆에 onnx·scaler가 있음.
    default_model = ck / "ap_early_exit_fixed_unified_int8_v2.onnx"
    if not default_model.exists():
        default_model = here / "ap_early_exit_fixed_unified_int8_v2.onnx"
    default_scaler = PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2" / "scaler_params.json"
    if not default_scaler.exists():
        default_scaler = here / "scaler_params.json"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=default_model,
                   help="unified ONNX (라벨 + exit_point 2-output). 기본 = 배포 INT8 Fixed θ.")
    p.add_argument("--scaler", type=Path, default=default_scaler)
    p.add_argument("--confirm", type=int, default=3,
                   help="확정 라벨을 바꾸려면 원시 예측이 몇 회 연속 일치해야 하는가 (히스테리시스).")
    p.add_argument("--raw", action="store_true", help="feature 원시값도 매 줄에 출력")
    return p.parse_args()


def normalize(feats: dict[str, float], scaler: dict[str, dict[str, float]]) -> np.ndarray:
    out = np.empty(len(AP_FEATURE_COLUMNS), dtype=np.float32)
    for i, name in enumerate(AP_FEATURE_COLUMNS):
        lo, hi = scaler[name]["min"], scaler[name]["max"]
        v = 0.0 if hi == lo else (feats[name] - lo) / (hi - lo)
        out[i] = min(1.0, max(0.0, v))
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    args = parse_args()
    scaler = json.loads(args.scaler.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    has_exit = len(sess.get_outputs()) > 1

    print(f"모델 : {args.model.name}")
    print(f"feature ({len(AP_FEATURE_COLUMNS)}) : {', '.join(AP_FEATURE_COLUMNS)}")
    print(f"확정 라벨 히스테리시스 : {args.confirm}회 연속")
    print("AP 폴링 시작... (Ctrl-C 종료)\n")

    poller = APPoller()

    window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    occ_hist: deque[float] = deque(maxlen=3)
    retx_hist: deque[int] = deque(maxlen=MOVING_AVG_WINDOW)
    pkts_hist: deque[int] = deque(maxlen=MOVING_AVG_WINDOW)
    rssi_hist: deque[float] = deque(maxlen=MOVING_AVG_WINDOW)

    prev_stations = None
    prev_active = prev_busy = None
    prev_rssi = None
    prev_time = None
    last_cycle_id = -1

    raw_streak: deque[int] = deque(maxlen=args.confirm)
    confirmed = None

    try:
        while True:
            cycle = poller.wait_for_new_cycle(last_cycle_id)
            if cycle is None:
                print(f"  AP 폴링 대기 (재연결 {poller.reconnects}회)...")
                continue
            last_cycle_id, text = cycle
            station, survey = parse_ap_cycle(text)
            if station is None:
                print("  연결된 station 없음")
                continue
            cur_active, cur_busy, *_ = survey
            now = time.time()

            signal_avg, n_clients, _bmin, sta_bitrate_mean = summarize_stations(station, prev_stations)
            occ_raw, _ = calculate_channel_occupancy(prev_active, prev_busy, cur_active, cur_busy)
            occ_hist.append(occ_raw)
            occupancy = median(occ_hist)

            rx_d, tx_d, retr_d, fail_d, pkts_d = calculate_station_deltas(prev_stations, station)
            elapsed = (now - prev_time) if prev_time else 0.0
            throughput = round((rx_d + tx_d) * 8 / elapsed / 1_000_000, 2) if elapsed > 0 else 0.0

            retx_hist.append(retr_d + fail_d)
            pkts_hist.append(pkts_d)
            retx_sum = sum(retx_hist)
            denom = retx_sum + sum(pkts_hist)
            tx_retry_ratio = round(retx_sum / denom, 4) if denom >= 50 else 0.0

            cur_rssi = signal_avg
            rssi_delta = 0.0 if prev_rssi is None else cur_rssi - prev_rssi
            rssi_hist.append(cur_rssi)
            rssi_moving_avg = sum(rssi_hist) / len(rssi_hist)

            feats = {
                "throughput_mbps": throughput,
                "channel_occupancy_percent": round(occupancy, 2),
                "tx_retry_ratio": tx_retry_ratio,
                "rssi_dbm": round(cur_rssi, 2),
                "rssi_delta_db": round(rssi_delta, 2),
                "rssi_moving_avg_dbm": round(rssi_moving_avg, 2),
                "sta_tx_bitrate_mean": round(sta_bitrate_mean, 1),
            }

            prev_stations, prev_active, prev_busy = station, cur_active, cur_busy
            prev_rssi, prev_time = cur_rssi, now

            window.append(normalize(feats, scaler))

            ts = time.strftime("%H:%M:%S")
            raw_str = ""
            if args.raw:
                raw_str = (f"  [occ {feats['channel_occupancy_percent']:.0f}% "
                           f"thr {feats['throughput_mbps']:.0f}M "
                           f"retry {feats['tx_retry_ratio']*100:.0f}% "
                           f"rssi {feats['rssi_dbm']:.0f} "
                           f"bitrate {feats['sta_tx_bitrate_mean']:.0f}]")

            if len(window) < WINDOW_SIZE:
                print(f"{ts}  창 채우는 중 {len(window)}/{WINDOW_SIZE}{raw_str}")
                continue

            x = np.stack(window)[None, :, :].astype(np.float32)
            outs = sess.run(None, {in_name: x})
            logits = np.asarray(outs[0]).reshape(-1)
            exit_pt = int(np.asarray(outs[1]).reshape(-1)[0]) if has_exit else None
            probs = softmax(logits)
            raw_label = int(probs.argmax())

            raw_streak.append(raw_label)
            changed = False
            if len(raw_streak) == args.confirm and len(set(raw_streak)) == 1:
                if raw_streak[0] != confirmed:
                    confirmed = raw_streak[0]
                    changed = True
            shown = confirmed if confirmed is not None else raw_label

            flip = "  <- 확정 변경" if changed else ""
            exit_str = f" exit{exit_pt}" if exit_pt else ""
            p_str = " ".join(f"{p:.2f}" for p in probs)
            print(
                f"{ts}  혼잡: {LABEL_NAMES[shown]}({shown})"
                f"  [원시 {LABEL_NAMES[raw_label]}({raw_label}) p={probs[raw_label]:.2f}]"
                f"  P(정/경/혼/심)=[{p_str}]"
                f"{exit_str} clients={n_clients}{flip}{raw_str}"
            )

    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        poller.stop()


if __name__ == "__main__":
    main()
