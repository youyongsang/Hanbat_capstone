"""AP 텔레메트리 폴링 → 7-feature → window 12 → ONNX(unified INT8) 추론 루프.

live_congestion.py 와 동일 로직 (victim 프로브 없음). `inference_loop()`가 백그라운드 스레드로
돌면서 `demo_state._state`를 갱신하고 SSE 구독자에게 `demo_state._broadcast()`로 뿌린다.
"""
from __future__ import annotations

import json
import time
from collections import deque
from statistics import median

import numpy as np
import onnxruntime as ort

import demo_state
from demo_state import (
    ARGS, WINDOW, FEATURES, LABELS, MOVING_AVG_WINDOW,
    APPoller, calculate_channel_occupancy, calculate_station_deltas,
    parse_ap_cycle, summarize_stations,
)


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
                with demo_state._lock:
                    demo_state._state = {"ready": False, "msg": f"AP 응답 없음 (재연결 {poller.reconnects}회)"}
                demo_state._broadcast(demo_state._state)
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
            with demo_state._lock:
                demo_state._state = {"ready": False, "msg": f"창 채우는 중 {len(win)}/{WINDOW}",
                          "features": feats, "clients": n_clients}
            demo_state._broadcast(demo_state._state)
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

        with demo_state._lock:
            demo_state._state = {
                "ready": True, "ts": time.strftime("%H:%M:%S"),
                "label": shown, "label_name": LABELS[shown],
                "raw_label": raw, "raw_name": LABELS[raw],
                "probs": [round(float(p), 3) for p in probs],
                "exit": exit_pt, "clients": n_clients,
                "features": feats, "load": dict(demo_state._load_state),
            }
        demo_state._broadcast(demo_state._state)
