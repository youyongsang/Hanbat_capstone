"""Recompute every derived column of the redesign-schema CSV.

`metrics_v2_pi_redesign.csv` was collected across several sessions while the
scoring was still changing (latency RTT/2 fix, failure=max). The stored
sub-score / congestion_score / label columns are therefore a mix. This script
replays the *current* `calculate_scores` (collect_metrics.py) over the raw
feature columns so every row is labelled by one consistent definition.

It also replays the live "failure = max" gate, which in the collector depends
on `ProbeRunner.ever_ok` — a per-run flag. Each scenario == one collector run,
so we track "has the probe ever been fresh in this scenario" per scenario.

Dud scenarios (phones sent no real load) are dropped: pass --drop.

Usage:
    python project/scripts/remeasure_redesign.py \
        --input  project/scripts/metrics_v2_pi_redesign.csv \
        --output project/scripts/metrics_v2_pi_redesign_relabeled.csv \
        --drop load_45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_metrics import calculate_scores  # noqa: E402

DERIVED_COLUMNS = [
    "throughput_score",
    "occupancy_score",
    "jitter_score",
    "loss_score",
    "latency_score",
    "retry_score",
    "congestion_score",
    "label",
]

LABEL_NAMES = {0: "정상", 1: "경고", 2: "혼잡", 3: "심각"}

SEVERE = 0.75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--drop",
        nargs="*",
        default=[],
        help="scenario names to drop entirely (dud runs)",
    )
    p.add_argument(
        "--persistence-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "ON by default (labelling policy since 2026-09-02). Demote a label-3 "
            "row to label 2 when its severe verdict comes from a non-occupancy "
            "(victim-probe) axis that spiked for fewer than --gate-m of the last "
            "--gate-k polls. Single-poll jitter/loss/ping spikes are noise, not a "
            "sustained QoS collapse; occupancy-driven severe is never gated. "
            "Pass --no-persistence-gate to reproduce pre-gate runs."
        ),
    )
    p.add_argument("--gate-k", type=int, default=3, help="persistence window (polls)")
    p.add_argument("--gate-m", type=int, default=2, help="min severe polls in the window")
    return p.parse_args()


def _score_to_label(score: float) -> int:
    if score < 0.25:
        return 0
    if score < 0.50:
        return 1
    if score < 0.75:
        return 2
    return 3


def apply_persistence_gate(df: pd.DataFrame, k: int, m: int) -> tuple[pd.DataFrame, int]:
    """Post-pass on the per-poll labels: a label-3 row whose 심각 comes from a
    non-occupancy axis (jitter/loss/latency) only counts if that axis was
    severe (>=0.75) in at least ``m`` of the trailing ``k`` polls of the same
    scenario. Otherwise the transient axis is capped just below severe and the
    row is relabelled from ``max(occupancy_score, capped non-occ)``.

    Rationale: label 3 == "victim QoS collapsed". One dropped-packet poll or one
    ping timeout is jitter, not collapse. This aligns the label with the
    temporal signal the LSTM already sees (window 12), instead of a last-poll
    snapshot that a single spike can flip.
    """
    out = df.copy()
    demoted = 0
    nonocc_cols = ["jitter_score", "loss_score", "latency_score"]
    for _, group in out.groupby("scenario", sort=False):
        idx = group.index.to_numpy()
        occ = group["occupancy_score"].to_numpy()
        sev_nonocc = group[nonocc_cols].to_numpy().max(axis=1)
        for pos, row_idx in enumerate(idx):
            if out.at[row_idx, "label"] != 3 or occ[pos] >= SEVERE:
                continue
            lo = max(0, pos - k + 1)
            if int((sev_nonocc[lo : pos + 1] >= SEVERE).sum()) >= m:
                continue
            capped = min(float(sev_nonocc[pos]), SEVERE - 1e-3)
            new_score = round(max(float(occ[pos]), capped), 4)
            out.at[row_idx, "congestion_score"] = new_score
            out.at[row_idx, "label"] = _score_to_label(new_score)
            demoted += 1
    return out, demoted


def remeasure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen_fresh_probe: set[str] = set()
    new_rows: list[tuple] = []

    for row in df.itertuples(index=False):
        scenario = row.scenario
        probe_fresh = int(row.probe_ok) == 1
        if probe_fresh:
            seen_fresh_probe.add(scenario)

        jitter = None if pd.isna(row.probe_jitter_ms) else float(row.probe_jitter_ms)
        loss = None if pd.isna(row.probe_loss_pct) else float(row.probe_loss_pct)

        throughput = float(row.throughput_mbps)
        occupancy = float(row.channel_occupancy_percent)
        latency_ms = float(row.latency_ms)
        retry_ratio_pct = float(row.tx_retry_ratio) * 100.0

        # live "failure = max" gate (collect_metrics.py main loop)
        channel_active = (throughput >= 3.0) or (occupancy >= 40.0)
        ping_hard_fail = channel_active and (latency_ms == 0.0)
        probe_hard_fail = (
            channel_active
            and (scenario in seen_fresh_probe)
            and (not probe_fresh)
        )

        new_rows.append(
            calculate_scores(
                throughput,
                occupancy,
                latency_ms,
                jitter,
                loss,
                retry_ratio_pct,
                ping_hard_fail=ping_hard_fail,
                probe_hard_fail=probe_hard_fail,
            )
        )

    remeasured = pd.DataFrame(new_rows, columns=DERIVED_COLUMNS)
    for col in DERIVED_COLUMNS:
        out[col] = remeasured[col].to_numpy()
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    before = len(df)

    if args.drop:
        df = df[~df["scenario"].isin(args.drop)].reset_index(drop=True)
        print(f"dropped {args.drop}: {before} -> {len(df)} rows")

    old_label = df["label"].copy()
    df = remeasure(df)
    changed = int((old_label.to_numpy() != df["label"].to_numpy()).sum())

    if args.persistence_gate:
        df, demoted = apply_persistence_gate(df, args.gate_k, args.gate_m)
        print(
            f"persistence gate (k={args.gate_k}, m={args.gate_m}): "
            f"demoted {demoted} transient label-3 rows -> 혼잡"
        )

    df.to_csv(args.output, index=False)

    print(f"remeasured {len(df)} rows, {changed} labels changed")
    print(f"written: {args.output}")
    print("\nlabel distribution (all rows):")
    for k in (0, 1, 2, 3):
        print(f"  {k} {LABEL_NAMES[k]}: {int((df['label'] == k).sum())}")
    print("\nby scenario:")
    for scenario, group in df.groupby("scenario", sort=False):
        counts = {k: int((group["label"] == k).sum()) for k in (0, 1, 2, 3)}
        print(f"  {scenario:12s} {len(group):4d}: {counts}")

    load = df[df["channel_occupancy_percent"] > 35]
    l3 = load[load["label"] == 3]
    below = int((l3["channel_occupancy_percent"] < 75).sum())
    print(
        f"\nload rows (occ>35): {len(load)} | label 3: {len(l3)} | "
        f"of those at occ<75%: {below}"
    )


if __name__ == "__main__":
    main()
