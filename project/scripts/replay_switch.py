"""수집된 세션 CSV를 라이브와 같은 경로(정규화 → window 12 → ONNX → 표시 히스테리시스 + 전환 판단)로 재생 (2026-10-01).

AP 없이 live_congestion.py의 판단 흐름을 시험·시연하는 용도. CSV에 이미 계산된 7-feature를 그대로 쓰므로
(collect_metrics.py와 같은 값) 라이브와 같은 입력이 들어간다. 정답 라벨(label 컬럼)이 있으면 함께 보여 준다.

사용:
  python project/scripts/replay_switch.py <metrics.csv> [--model ONNX] [--quiet] [--summary-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_congestion import AP_FEATURE_COLUMNS, LABEL_NAMES, WINDOW_SIZE, normalize, parse_args as live_defaults, softmax  # noqa: E402
from switch_advisor import SwitchAdvisor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--confirm", type=int, default=3)
    p.add_argument("--release", type=int, default=30)
    p.add_argument("--persist", type=int, default=60)
    p.add_argument("--summary-only", action="store_true", help="알림·요약만 출력")
    a = p.parse_args()
    sys.argv = sys.argv[:1]
    d = live_defaults()
    model = a.model or d.model
    scaler = json.loads(d.scaler.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rows = list(csv.DictReader(a.csv.open(encoding="utf-8")))
    window: deque = deque(maxlen=WINDOW_SIZE)
    streak: deque = deque(maxlen=a.confirm)
    confirmed = None
    adv = SwitchAdvisor(release=a.release, persist=a.persist)
    n = dl_polls = 0
    alerts = []
    true_sev = []
    for i, r in enumerate(rows):
        try:
            feats = {c: float(r[c]) for c in AP_FEATURE_COLUMNS}
        except (KeyError, ValueError):
            continue
        window.append(normalize(feats, scaler))
        if len(window) < WINDOW_SIZE:
            continue
        logits = np.asarray(sess.run(None, {in_name: np.stack(window)[None].astype(np.float32)})[0]).reshape(-1)
        probs = softmax(logits); raw = int(probs.argmax()); n += 1
        streak.append(raw)
        if len(streak) == a.confirm and len(set(streak)) == 1:
            confirmed = streak[0]
        shown = confirmed if confirmed is not None else raw
        events = adv.update(raw, feats["throughput_mbps"], feats["channel_occupancy_percent"])
        dl = adv.downlink_dominant(); dl_polls += dl
        truth = r.get("label")
        if truth not in (None, ""):
            true_sev.append(int(float(truth)) == 3)
        if not a.summary_only:
            t = f" 정답 {LABEL_NAMES[int(float(truth))]}" if truth not in (None, "") else ""
            print(f"{r.get('timestamp', i)}  표시 {LABEL_NAMES[shown]}  원시 {LABEL_NAMES[raw]}{t}"
                  f"{'  [다운링크 우세 — 판정 불확실]' if dl else ''}")
        for ev in events:
            alerts.append((r.get("timestamp", i), ev))
            print(f"{r.get('timestamp', i)}  ⚠ {ev.message}")
            for c in ev.candidates:
                print(f"           후보: {c}")
    print(f"\n요약: {a.csv.name} — 판정 {n}폴링, 전환 알림 {sum(e.kind == 'switch' for _, e in alerts)}회, "
          f"심각 지속 알림 {sum(e.kind == 'persist' for _, e in alerts)}회, 다운링크 우세 표시 {dl_polls}폴링({dl_polls / max(n, 1) * 100:.0f}%)"
          + (f", 정답 심각 {sum(true_sev)}폴링" if true_sev else ""))


if __name__ == "__main__":
    main()
