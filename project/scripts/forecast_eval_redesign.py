"""Does the Early Exit LSTM *forecast* the congestion label k polls ahead
better than reactive baselines (current occupancy / current label)?

Rebuilds the windowed dataset from metrics_v2_pi_redesign_relabeled.csv with
the target shifted k polls into the future, retrains, and compares:
  - LSTM
  - occupancy-only classifier at t  (occ>=75 -> 3, >=55 -> 2, >=40 -> 1)
  - persistence (predict label at t+k = label at t)
plus the escalation subset (not severe now, severe in k polls).

Poll interval is ~1-2 s, so k=3 is roughly 3-6 s ahead.

    python project/scripts/forecast_eval_redesign.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from models.early_exit_lstm import multi_exit_loss  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS, WINDOW_SIZE  # noqa: E402

CSV = ROOT / "scripts" / "metrics_v2_pi_redesign2_relabeled.csv"
FEAT = list(AP_FEATURE_COLUMNS)
W = WINDOW_SIZE  # single source (utils.ap_features); 10 -> 12 on 2026-09-01
HORIZONS = (0, 3, 5)
torch.manual_seed(0)
np.random.seed(0)

df = pd.read_csv(CSV)


def occ_rule(occ):
    return np.where(occ >= 75, 3, np.where(occ >= 55, 2, np.where(occ >= 40, 1, 0)))


def build(k):
    X, ytar, ynow, occnow = [], [], [], []
    for _, g in df.groupby("scenario", sort=False):
        g = g.reset_index(drop=True)
        occ = g["channel_occupancy_percent"].to_numpy()
        lab = g["label"].to_numpy()
        feats = g[FEAT].to_numpy(np.float32)
        for s in range(0, len(g) - W + 1 - k):
            e = s + W
            X.append(feats[s:e])
            ytar.append(int(lab[e - 1 + k]))
            ynow.append(int(lab[e - 1]))
            occnow.append(float(occ[e - 1]))
    return np.stack(X), np.array(ytar), np.array(ynow), np.array(occnow)


def split_idx(y, seed=0):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(4):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        ntr, nva = int(n * 0.7), int(n * 0.15)
        tr += list(idx[:ntr])
        va += list(idx[ntr:ntr + nva])
        te += list(idx[ntr + nva:])
    return np.array(tr), np.array(va), np.array(te)


def norm(X, tr):
    flat = X[tr].reshape(-1, len(FEAT))
    lo, hi = flat.min(0), flat.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return ((X - lo) / rng).clip(0, 1).astype(np.float32)


def class_weights(y):
    c = np.bincount(y, minlength=4).astype(float)
    c[c == 0] = 1
    return torch.tensor(len(y) / (4 * c), dtype=torch.float32)


def train_eval(k):
    X, ytar, ynow, occnow = build(k)
    tr, va, te = split_idx(ytar)
    Xt = torch.tensor(norm(X, tr))
    Y = torch.tensor(ytar)
    w = class_weights(ytar[tr])
    model = APEarlyExitLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (-1.0, None)
    for _ in range(45):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            multi_exit_loss(model(Xt[b]), Y[b], class_weights=w).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xt[va])[-1].argmax(1).numpy()
        rec = np.mean([(pv[ytar[va] == c] == c).mean()
                       for c in range(4) if (ytar[va] == c).any()])
        if rec > best[0]:
            best = (rec, {kk: vv.clone() for kk, vv in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pl = model(Xt[te])[-1].argmax(1).numpy()

    yt, yn, oc = ytar[te], ynow[te], occnow[te]

    def sev_prf(pred):
        st, sh = yt == 3, pred == 3
        tp = int((st & sh).sum())
        fp = int((~st & sh).sum())
        fn = int((st & ~sh).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return R, P, (2 * P * R / (P + R) if P + R else 0.0)

    print(f"\n===== k={k} polls ahead | test {len(te)} windows, target severe={int((yt == 3).sum())} =====")
    for name, pred in [("LSTM", pl), ("occupancy rule@t", occ_rule(oc)), ("persistence", yn)]:
        R, P, F = sev_prf(pred)
        print(f"  {name:18s} acc {100 * (pred == yt).mean():5.1f}%  severe R {100 * R:5.1f}% P {100 * P:5.1f}% F1 {100 * F:5.1f}%")
    esc = (yn < 3) & (yt == 3)
    if esc.sum():
        print(f"  escalation (not severe now -> severe in k): {int(esc.sum())} windows")
        print(f"     LSTM caught {int((pl[esc] == 3).sum())}/{int(esc.sum())}  |  occ rule 0/{int(esc.sum())} (structural)")


if __name__ == "__main__":
    for h in HORIZONS:
        train_eval(h)
