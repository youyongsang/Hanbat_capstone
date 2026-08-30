"""Generate AP redesign2 model comparison summary (Baseline / SDN-style / Proposed Early Exit)."""

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
    repeats = 200

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
            "interpretation": "accuracy upper-bound baseline; no early stopping; real Pi INT8 latency 0.746ms (2026-08-30, power=0.0, 5-seed val-best); 5-seed mean acc 92.0%+-0.7%",
        },
        {
            "model": "SDN-style Early Exit (trained)",
            "paper_basis": "SDN/Shallow-Deep Networks (ICML 2019)-style backbone, separately trained with SDN loss weights (0.15/0.30/0.55), unweighted (class-weight-power=0.0) to match Proposed",
            "threshold_policy": f"max softmax confidence >= {sdn_model.confidence_threshold:.2f}",
            "test_accuracy": sdn["acc"],
            "label2_accuracy": sdn["l2"],
            "label3_accuracy": sdn["l3"],
            "exit1_rate": sdn["e1"],
            "exit2_rate": sdn["e2"],
            "exit3_rate": sdn["e3"],
            "pc_measured_ms_per_sample": f"{sdn_ms:.4f}",
            "simulated_layer_time_ms": sdn["sim"],
            "interpretation": "paper-policy baseline trained independently on AP 7-feature redesign2 data; confidence-only exit; real Pi INT8 latency 0.534ms (2026-08-30, power=0.0)",
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
            "interpretation": "proposed model without dynamic threshold; uniform exit-loss weights (0.3/0.3/0.4); real Pi unified INT8 latency 0.540ms (2026-08-30, power=0.0)",
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
            "interpretation": "proposed method; uniform exit-loss weights (0.3/0.3/0.4); real Pi unified INT8 latency 0.555ms (2026-08-30, power=0.0)",
        },
    ]

    csv_path = RESULT_DIR / "ap_model_comparison_redesign2.csv"
    txt_path = RESULT_DIR / "ap_model_comparison_redesign2.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "AP redesign2 데이터 모델 비교표 (Baseline / SDN-style / Proposed Early Exit)",
        f"데이터: {DATA_DIR.relative_to(PROJECT_ROOT.parent)} (test 310창, label 3 31개)",
        "모든 모델 class-weight-power=0.0으로 학습(2026-08-30 재스윕 — 클래스 가중치 없음이 정확도·label3 F1 둘 다 최고). 공정 비교를 위해 세 모델 동일 적용.",
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
            f"- SDN-style Early Exit는 Early Exit 백본을 재사용하지 않고, SDN 논문의 loss 가중치(0.15/0.30/0.55)로 별도 학습한 백본에 고정 confidence threshold {sdn_model.confidence_threshold:.2f}를 적용한 결과이다.",
            "- 1학기(ap_cleaned_strict)와 달리 이번엔 Baseline/SDN도 Proposed와 동일한 학습 방식(class-weight-power=0.0)으로 학습했다 — 아키텍처 차이만 비교하기 위해 훈련 방식은 통제 변수로 고정.",
            "- Proposed Fixed는 우리 Early Exit 구조에서 dynamic threshold만 제거한 entropy-threshold ablation이다.",
            "- PC Python 실측은 조기종료 판단 오버헤드 때문에 Early Exit에 불리하므로 최종 시간 주장은 Raspberry Pi + ONNX 재측정 기준이어야 한다 — 2026-08-30(power=0.0) Pi INT8 실측: Baseline 0.746ms / SDN 0.534ms / Proposed Fixed 0.540ms / Dynamic 0.555ms(전부 목표2 <1ms 달성, EE가 Baseline보다 -27%). power=1.0 대비 EE가 0.64->0.54ms로 빨라짐 — exit3 도달률이 52%->19%로 줄어든 결과. 상세: project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt.",
            "- 2026-08-30: class-weight-power 재스윕(0.0/0.1/0.15/0.2/0.3/0.5/0.7/0.85/1.0, 3시드씩)에서 power=0.0이 정확도·label3 F1 둘 다 최고 → 기본값 1.0→0.0. 세 모델 전부 재학습·5시드 특성화, val balanced acc 최고를 배포(Baseline seed3 / SDN seed0 / EE seed4). 5시드 test 평균: Baseline 92.0%±0.7 / SDN 90.4%±0.7 / EE Fixed 90.7%±0.7.",
            "- 2026-08-29: sta_tx_bitrate_mean을 7번째 입력 feature로 승격(occ 60~72% 구간에서 label2/3을 가르는 유의미한 신호로 다중 시드 검증됨). 같은 세션에 학습 파이프라인에 --seed 옵션 추가.",
            "- Proposed의 exit-loss 가중치는 균등(0.3/0.3/0.4)으로 유지 — SDN 스타일(0.15/0.30/0.55)을 시도했던 이전 승격은 다중 시드 검증에서 유의미한 이득이 확인되지 않아 보류됨(SDNLSTM은 EarlyExitLSTM과 백본이 의도적으로 동일하므로, 같은 가중치를 쓰면 두 모델이 사실상 같은 네트워크가 되어 버려 비교 자체가 무의미해지는 것도 이유).",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(txt_path.read_text(encoding="utf-8"))
    print(f"CSV saved: {csv_path.resolve().relative_to(REPO_ROOT)}")
    print(f"TXT saved: {txt_path.resolve().relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
