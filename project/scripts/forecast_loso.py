"""조기경보(k폴링 뒤 라벨) 세션 단위 LOSO 평가 (2026-10-16, MT6000 라인).

질문: 점유율 규칙이 "지금" 라벨을 이미 98%대로 맞히는 MT6000에서, LSTM이 "k폴링 뒤" 라벨을
반응형 규칙보다 먼저 맞히는가? (forecast_eval_redesign.py의 창 단위 무작위 분할 → 세션 단위 LOSO로 교체)

비교 대상 (k폴링 뒤 라벨을 예측):
  - occ rule@t     : 지금 점유율 점수로 라벨 (40/55/75 앵커, 미래도 지금과 같다고 가정)
  - occ trend      : 지금 점유율 + k × 최근 3폴링 기울기로 외삽한 뒤 같은 앵커
  - persistence    : 지금 라벨 그대로 (정답 라벨을 아는 상한형 반응 기준 — 배포 땐 못 씀)
  - LSTM (EE exit3): 7-feature window 12, 학습 세션만으로 min-max 스케일, plain CE, 고정 epoch
지표: 4단계 정확도, 심각 recall/precision/F1, escalation(지금 심각 아님 → k 뒤 심각) recall.

    python project/scripts/forecast_loso.py --pool project/.tmp/exp_24g/pool_mt6k/pool_relabeled.csv \
        --session-regex "_(m\\d+)$" --horizons 0 3 10 --epochs 20 --out project/results/yongsang/forecast_loso_mt6000.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from models.early_exit_lstm import EarlyExitLSTM, multi_exit_loss  # noqa: E402
from utils.ap_features import AP_FEATURE_COLUMNS, WINDOW_SIZE  # noqa: E402

FEAT = list(AP_FEATURE_COLUMNS)
W = WINDOW_SIZE


def occ_label(score):
    return np.select([score < 0.25, score < 0.5, score < 0.75], [0, 1, 2], 3)


def occ_score_from_pct(p):
    # remeasure_redesign의 점유율 앵커(40/55/75/90%)와 같은 piecewise-linear
    xs, ys = [0, 40, 55, 75, 90], [0, 0.25, 0.5, 0.75, 1.0]
    return np.clip(np.interp(p, xs, ys), 0, 1)


def build(df, k, sess_of):
    out = {n: [] for n in ("X", "y", "ynow", "occ", "trend", "sess")}
    for scen, g in df.groupby("scenario", sort=False):
        g = g.reset_index(drop=True)
        feats = g[FEAT].to_numpy(np.float32)
        lab = g["label"].to_numpy()
        occs = g["occupancy_score"].to_numpy()
        occp = g["channel_occupancy_percent"].to_numpy()
        for s in range(0, len(g) - W + 1 - k):
            e = s + W - 1
            out["X"].append(feats[s:e + 1])
            out["y"].append(int(lab[e + k]))
            out["ynow"].append(int(lab[e]))
            out["occ"].append(occs[e])
            slope = (occp[e] - occp[e - 3]) / 3.0
            out["trend"].append(occ_score_from_pct(occp[e] + k * slope))
            out["sess"].append(sess_of(scen))
    return {n: (np.stack(v) if n == "X" else np.array(v)) for n, v in out.items()}


def metrics(y, p, ynow):
    st, sh = y == 3, p == 3
    tp, fp, fn = int((st & sh).sum()), int((~st & sh).sum()), int((st & ~sh).sum())
    P = tp / (tp + fp) if tp + fp else np.nan
    R = tp / (tp + fn) if tp + fn else np.nan
    F = 2 * P * R / (P + R) if (P == P and R == R and P + R) else np.nan
    esc = (ynow < 3) & st
    return dict(acc=(p == y).mean() * 100, sevR=R * 100, sevP=P * 100, sevF1=F * 100,
                esc_n=int(esc.sum()), esc_R=((p[esc] == 3).mean() * 100) if esc.sum() else np.nan)


def train_lstm(Xtr, ytr, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = APEarlyExitLSTM() if len(FEAT) == len(AP_FEATURE_COLUMNS) else EarlyExitLSTM(input_size=len(FEAT), hidden_size=128, num_classes=4, dropout=0.2, theta_1=0.3, theta_2=0.6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Y = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            multi_exit_loss(model(Xt[b]), Y[b]).backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--session-regex", default=r"_(m\d+)$", help="scenario 이름에서 세션 id를 뽑는 정규식(그룹 1)")
    ap.add_argument("--exclude-scenarios", nargs="*", default=[])
    ap.add_argument("--horizons", nargs="+", type=int, default=[0, 3, 10])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", choices=["all", "occ"], default="all", help="occ = 점유율 1개만(10-18 어블레이션)")
    a = ap.parse_args()
    global FEAT
    if a.features == "occ":
        FEAT = ["channel_occupancy_percent"]
    torch.set_num_threads(4)

    df = pd.read_csv(a.pool)
    df = df[~df.scenario.isin(a.exclude_scenarios)].reset_index(drop=True)
    rx = re.compile(a.session_regex)

    def sess_of(scen):
        m = rx.search(scen)
        return m.group(1) if m else "other"

    rows = []
    for k in a.horizons:
        D = build(df, k, sess_of)
        sessions = sorted(set(D["sess"]))
        preds = {n: np.empty(len(D["y"]), dtype=int) for n in ("LSTM",)}
        ck = Path(a.out).with_name(Path(a.out).stem + f"_k{k}_folds.npz")   # 10-16: 폴드별 저장 → 중단돼도 이어서
        done = dict(np.load(ck, allow_pickle=True)) if ck.exists() else {}
        for s in sessions:
            te, tr = D["sess"] == s, D["sess"] != s
            if s in done and len(done[s]) == int(te.sum()):
                preds["LSTM"][te] = done[s]
                print(f"k={k} fold {s}: 저장본 사용", flush=True)
                continue
            flat = D["X"][tr].reshape(-1, len(FEAT))
            lo, hi = flat.min(0), flat.max(0)
            rng = np.where(hi > lo, hi - lo, 1.0)
            Xn = ((D["X"] - lo) / rng).clip(0, 1).astype(np.float32)
            model = train_lstm(Xn[tr], D["y"][tr], a.epochs, a.seed)
            with torch.no_grad():
                preds["LSTM"][te] = model(torch.tensor(Xn[te]))[-1].argmax(1).numpy()
            done[s] = preds["LSTM"][te].copy()
            np.savez(ck, **done)
            del model, Xn
            print(f"k={k} fold {s}: test {int(te.sum())} windows done", flush=True)
        base = {"occ rule@t": occ_label(D["occ"]), "occ trend": occ_label(D["trend"]),
                "persistence": D["ynow"], "LSTM": preds["LSTM"]}
        for name, p in base.items():
            m = metrics(D["y"], p, D["ynow"])
            rows.append(dict(k=k, model=name, windows=len(D["y"]), **{kk: round(v, 1) if isinstance(v, float) else v for kk, v in m.items()}))
            print(f"  k={k:2d} {name:12s} acc {m['acc']:5.1f} | sev R {m['sevR']:5.1f} P {m['sevP']:5.1f} F1 {m['sevF1']:5.1f} | escalation {m['esc_n']} R {m['esc_R']:5.1f}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
    print("saved", a.out)


if __name__ == "__main__":
    main()
