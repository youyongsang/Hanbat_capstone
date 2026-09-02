"""Generate AP redesign2 model comparison summary (Baseline / SDN / Proposed Early Exit)."""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ap_baseline_lstm import APBaselineLSTM  # noqa: E402
from models.ap_early_exit_lstm import APEarlyExitLSTM  # noqa: E402
from models.ap_sdn_lstm import APSDNLSTM  # noqa: E402
from utils.ap_dataloader import load_ap_csv_windows  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "ap_metrics_v2_redesign2"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2"
RESULT_DIR = PROJECT_ROOT / "results" / "yongsang"


def extract(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.S)
    return match.group(1) if match else ""


def parse_early_exit_block(block: str) -> dict[str, str]:
    return {
        "acc": extract(block, r"Test Accuracy: ([0-9.]+%)"),
        "l2": extract(block, r"Label 2 \([^)]*\): ([0-9.]+%)"),
        "l3": extract(block, r"Label 3 \([^)]*\): ([0-9.]+%)"),
        "e1": extract(block, r"Exit 1 \| Accuracy: [^|]+\| Exit Rate: ([0-9.]+%)"),
        "e2": extract(block, r"Exit 2 \| Accuracy: [^|]+\| Exit Rate: ([0-9.]+%)"),
        "e3": extract(block, r"Exit 3 \| Accuracy: [^|]+\| Exit Rate: ([0-9.]+%)"),
        "sim": extract(block, r"Overall Avg Inference Time: ([0-9.]+)ms"),
    }


def load_models() -> tuple[APBaselineLSTM, APSDNLSTM, APEarlyExitLSTM]:
    torch.set_num_threads(1)

    base_checkpoint = torch.load(CHECKPOINT_DIR / "ap_baseline_lstm_best.pth", map_location="cpu")
    sdn_checkpoint = torch.load(CHECKPOINT_DIR / "ap_sdn_lstm_best.pth", map_location="cpu")
    ee_checkpoint = torch.load(CHECKPOINT_DIR / "ap_early_exit_lstm_best.pth", map_location="cpu")

    base_model = APBaselineLSTM(hidden_size=int(base_checkpoint.get("hidden_size", 128))).eval()
    base_model.load_state_dict(base_checkpoint["model_state_dict"])

    sdn_model = APSDNLSTM(
        hidden_size=int(sdn_checkpoint.get("hidden_size", 128)),
        confidence_threshold=float(sdn_checkpoint.get("confidence_threshold", 0.85)),
    ).eval()
    sdn_model.load_state_dict(sdn_checkpoint["model_state_dict"])

    ee_model = APEarlyExitLSTM(hidden_size=int(ee_checkpoint.get("hidden_size", 128))).eval()
    ee_model.load_state_dict(ee_checkpoint["model_state_dict"])

    return base_model, sdn_model, ee_model


def benchmark_pc_time(
    base_model: APBaselineLSTM,
    sdn_model: APSDNLSTM,
    ee_model: APEarlyExitLSTM,
) -> tuple[float, float, float, float]:
    x_np, _ = load_ap_csv_windows(DATA_DIR / "test.csv")
    x = torch.from_numpy(x_np)

    warmup = 20
    repeats = 30

    with torch.no_grad():
        for _ in range(warmup):
            base_model(x)
            sdn_model.infer_batch_stepwise(x)
            ee_model.infer_batch_stepwise(x, dynamic=False)
            ee_model.infer_batch_stepwise(x, dynamic=True)

        start = time.perf_counter()
        for _ in range(repeats):
            base_model(x)
        base_ms = (time.perf_counter() - start) * 1000 / (repeats * len(x))

        start = time.perf_counter()
        for _ in range(repeats):
            sdn_model.infer_batch_stepwise(x)
        sdn_ms = (time.perf_counter() - start) * 1000 / (repeats * len(x))

        start = time.perf_counter()
        for _ in range(repeats):
            ee_model.infer_batch_stepwise(x, dynamic=False)
        fixed_ms = (time.perf_counter() - start) * 1000 / (repeats * len(x))

        start = time.perf_counter()
        for _ in range(repeats):
            ee_model.infer_batch_stepwise(x, dynamic=True)
        dynamic_ms = (time.perf_counter() - start) * 1000 / (repeats * len(x))

    return base_ms, sdn_ms, fixed_ms, dynamic_ms


def main() -> None:
    base_report = (RESULT_DIR / "ap_baseline_lstm_redesign2_eval_report.txt").read_text(encoding="utf-8")
    sdn_report = (RESULT_DIR / "ap_sdn_redesign2_eval_report.txt").read_text(encoding="utf-8")
    ee_report = (RESULT_DIR / "ap_v2_redesign2_eval_report.txt").read_text(encoding="utf-8")

    fixed_block = extract(
        ee_report,
        r"(=== AP Early Exit \+ fixed theta ===.*?)(?=\n=== AP Early Exit \+ dynamic theta ===)",
    )
    dynamic_block = extract(ee_report, r"(=== AP Early Exit \+ dynamic theta ===.*)")

    sdn = parse_early_exit_block(sdn_report)
    fixed = parse_early_exit_block(fixed_block)
    dynamic = parse_early_exit_block(dynamic_block)
    base_model, sdn_model, ee_model = load_models()
    base_ms, sdn_ms, fixed_ms, dynamic_ms = benchmark_pc_time(base_model, sdn_model, ee_model)

    rows = [
        {
            "model": "Baseline LSTM (no early exit)",
            "paper_basis": "standard full-depth LSTM, unweighted (class-weight-power=0.0, 2026-08-30 re-sweep) to match Proposed",
            "threshold_policy": "none; always runs all 3 LSTM layers",
            "test_accuracy": extract(base_report, r"Test Accuracy: ([0-9.]+%)"),
            "label2_accuracy": extract(base_report, r"Label 2 \([^)]*\): ([0-9.]+%)"),
            "label3_accuracy": extract(base_report, r"Label 3 \([^)]*\): ([0-9.]+%)"),
            "exit1_rate": "0.0%",
            "exit2_rate": "0.0%",
            "exit3_rate": "100.0%",
            "pc_measured_ms_per_sample": f"{base_ms:.4f}",
            "simulated_layer_time_ms": "8.000",
            "interpretation": "accuracy upper-bound baseline; no early stopping; 5-seed mean acc 92.7%+-1.5 / L3 F1 84.3%+-3.5 (k2m2 gate); deploy seed2 89.9% (weak test draw, val-bal honest pick -- L3 recall 90.5/F1 78.2). Pi INT8 latency 0.864ms.",
        },
        {
            "model": "SDN (Kaya et al. 2019, adapted)",
            "paper_basis": "SDN (Shallow-Deep Networks, ICML 2019) adapted to the shared 3-layer LSTM base: pooling internal classifiers, curriculum-ramped depth-weighted IC loss (max 0.15/0.30, final 1.0), val-calibrated confidence threshold. Backbone/hparams identical to Proposed (controlled).",
            "threshold_policy": f"max softmax confidence >= {sdn_model.confidence_threshold:.2f}",
            "test_accuracy": sdn["acc"],
            "label2_accuracy": sdn["l2"],
            "label3_accuracy": sdn["l3"],
            "exit1_rate": sdn["e1"],
            "exit2_rate": sdn["e2"],
            "exit3_rate": sdn["e3"],
            "pc_measured_ms_per_sample": f"{sdn_ms:.4f}",
            "simulated_layer_time_ms": sdn["sim"],
            "interpretation": "prior-art comparison: 5-seed mean acc 91.9%+-1.0 vs EE Fixed 91.7%+-0.7 / Dynamic 92.3%+-0.7 (k2m2 gate, all three roughly level); label-3 F1 86.4%+-1.3 vs EE 85.9/86.7 (also level). Deploy seed0 (T=0.72): 92.3%, L3 P/R/F1 83/90/86. Pi INT8 latency 0.575ms this round (T=0.72 front-loads exit1 37.5%); per-exit EE is still lighter at every stage.",
        },
        {
            "model": "Proposed Early Exit Fixed theta",
            "paper_basis": "our AP Early Exit LSTM ablation",
            "threshold_policy": "fixed entropy theta",
            "test_accuracy": fixed["acc"],
            "label2_accuracy": fixed["l2"],
            "label3_accuracy": fixed["l3"],
            "exit1_rate": fixed["e1"],
            "exit2_rate": fixed["e2"],
            "exit3_rate": fixed["e3"],
            "pc_measured_ms_per_sample": f"{fixed_ms:.4f}",
            "simulated_layer_time_ms": fixed["sim"],
            "interpretation": "proposed model without dynamic threshold; uniform exit-loss weights (0.3/0.3/0.4); 5-seed mean acc 91.7%+-0.7 / L3 F1 85.9%+-2.5 / L3 recall 85.7%+-3.7 (k2m2 gate, up from k3m2's 75.2). deploy seed2: acc 92.3% / L3 P/R/F1 84/88/86. Pi INT8 latency 0.625ms -- -28% vs Baseline 0.864ms.",
        },
        {
            "model": "Proposed Early Exit Dynamic theta",
            "paper_basis": "our variant on early-exit LSTM",
            "threshold_policy": "dynamic theta adjusted by recent traffic variation",
            "test_accuracy": dynamic["acc"],
            "label2_accuracy": dynamic["l2"],
            "label3_accuracy": dynamic["l3"],
            "exit1_rate": dynamic["e1"],
            "exit2_rate": dynamic["e2"],
            "exit3_rate": dynamic["e3"],
            "pc_measured_ms_per_sample": f"{dynamic_ms:.4f}",
            "simulated_layer_time_ms": dynamic["sim"],
            "interpretation": "proposed method; uniform exit-loss weights (0.3/0.3/0.4); 5-seed mean acc 92.3%+-0.7 / L3 F1 86.7%+-1.8 (project best, k2m2 gate); deploy acc 92.9% / L3 P/R/F1 86/88/87. Pi INT8 latency 0.632ms (p95 1.034ms).",
        },
    ]

    csv_path = RESULT_DIR / "ap_model_comparison_redesign2.csv"
    txt_path = RESULT_DIR / "ap_model_comparison_redesign2.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "AP redesign2 데이터 모델 비교표 (Baseline / SDN / Proposed Early Exit)",
        f"데이터: {DATA_DIR.relative_to(PROJECT_ROOT.parent)} (window 12, 2551행+라벨 지속성 게이트 k=2/m=2, test 365창, label 3 42개)",
        "모든 모델 class-weight-power=0.0, window 12. 2026-09-02(18차): k/m 스윕에서 k2m2가 k3m2(현행이던 배포 설정)보다 L3 recall·F1 우위로 확인되어 채택 → 3모델 5시드 재학습·재배포.",
        "",
        "| 모델 | 근거/역할 | 정확도 | Label 2 | Label 3 | Exit1 | Exit2 | Exit3 | PC 실측(ms/sample) | 구조상 평균(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {paper_basis} | {test_accuracy} | {label2_accuracy} | "
            "{label3_accuracy} | {exit1_rate} | {exit2_rate} | {exit3_rate} | "
            "{pc_measured_ms_per_sample} | {simulated_layer_time_ms} |".format(**row)
        )

    lines.extend(
        [
            "",
            "해석:",
            "- Baseline LSTM은 정확도 상한선 기준이며 항상 3개 LSTM 계층을 모두 수행한다.",
            f"- SDN(Kaya et al., ICML 2019)은 \"기존 조기종료 방법 vs 우리 방법\" 통제 비교를 위한 것. 3층 LSTM base·하이퍼파라미터는 Proposed와 완전 동일하게 고정하고, SDN이 실제로 규정하는 세 축만 논문대로 다르게 함: (1) pooling internal classifier, (2) 커리큘럼 램프 depth-weighted IC loss, (3) val 캘리브레이션된 confidence threshold(이 배포 체크포인트 T={sdn_model.confidence_threshold:.2f}). Proposed는 각각 last-timestep linear head / 균등 loss / entropy threshold(고정·변동).",
            "- 1학기(ap_cleaned_strict)와 달리 이번엔 Baseline/SDN도 Proposed와 동일한 학습 방식(class-weight-power=0.0)으로 학습했다 — 훈련 방식(class-weight-power)은 통제 변수로 고정 — 이제 SDN은 아키텍처(IC·loss·threshold)도 논문대로 다름.",
            "- Proposed Fixed는 우리 Early Exit 구조에서 dynamic threshold만 제거한 entropy-threshold ablation이다.",
            "- Raspberry Pi INT8 실측 (window 12, 2551+k2m2게이트, test 365창, 2026-09-02 9차): Baseline 0.864 / SDN 0.575 / EE Fixed 0.625 / Dynamic 0.632ms — 전부 avg <1ms(목표2). EE Fixed가 Baseline -28%. per-exit는 EE가 SDN보다 전 stage 가벼움(0.327/0.644/0.953 vs 0.332/0.646/0.962) — SDN 평균이 낮은 건 T=0.72가 exit1을 37.5%로 front-load한 것 (k3m2 시절 T=0.70/exit1 56%보다는 덜 공격적).",
            "- **2026-09-02(18차): 게이트 k·m 스윕 → k2=m2 채택** — 17차 배포(k3m2)의 오답을 더 줄일 수 있는지 k/m 조합 6개(nogate·k2m2·k3m2·k3m3·k5m2·k5m3) x 5시드로 스윕. k2m2가 강등 35개(nogate 대비 거의 최소)로 L3 recall·F1 최고(EE Fixed 5시드 87.1/85.3 vs k3m2의 78.1/83.1) — \"최근 2폴링 중 2폴링 다 심각\"이라는 국소적 지속성 요구가 단발 스파이크는 걸러내되 진짜 지속형 label 3은 거의 안 잃는 최적점으로 확인됨. **3모델 5시드 재학습 결과 k3m2 대비 L3 recall/F1 전반 개선** (Baseline 78.1/81.3→87.1/84.3 / SDN 78.6/84.2→87.6/86.4 / EE Fixed 75.2/82.9→85.7/85.9 / Dynamic 78.1/85.2→83.8/86.7), 정확도는 ±1σ 동급 유지.",
            "- 2026-09-01: window 10→12 승격 + 소패킷 부하 +436행(2115→2551) — L3 recall 55.5→69.6%, 전체 정확도는 게이트 전까진 안 움직였음.",
            "- 2026-08-30: class-weight-power 재스윕에서 power=0.0이 정확도·label3 F1 둘 다 최고 → 기본값 1.0→0.0. SDN은 Kaya et al. 논문대로 재구현(pooling IC·램프 loss·캘리브레이션 T).",
            "- 2026-08-29: sta_tx_bitrate_mean을 7번째 입력 feature로 승격(occ 60~72% 구간에서 label2/3을 가르는 유의미한 신호로 다중 시드 검증됨).",
            "- **5시드 test 평균 (2551+k2m2게이트): Baseline 92.7%±1.5 / SDN 91.9%±1.0 / EE Fixed 91.7%±0.7 / EE Dynamic 92.3%±0.7 — ±1σ 동급.** Label3 F1은 Baseline 84.3±3.5 / SDN 86.4±1.3 / EE Fixed 85.9±2.5 / EE Dynamic 86.7±1.8 (k3m2 대비 전반 개선, EE Dynamic이 프로젝트 최고). 배포 단일: Baseline 89.9%(약한 test draw) / SDN 92.3%(T=0.72) / EE Fixed 92.3% / Dynamic 92.9%. Proposed 가치 주장은 정확도가 아니라 속도(Pi INT8 0.625ms, Baseline 대비 -28%) + 트래픽 적응형 임계값 + 간섭 감지 EE 최초 적용.",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(txt_path.read_text(encoding="utf-8"))
    print(f"CSV saved: {csv_path.resolve().relative_to(REPO_ROOT)}")
    print(f"TXT saved: {txt_path.resolve().relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
