"""
그래프 목록:
  1. accuracy_latency_combined.png  — 정확도(막대) + 추론시간(꺾은선) 통합
  2. exit_rate_comparison.png       — Exit 종료율 비교 (③ vs ④)
  3. scenario_accuracy.png          — 시나리오별 정확도 선그래프
  4. quantization_comparison.png    — 경량화(INT8 / ONNX) 전후 비교
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = ROOT_DIR / "project" / "results" / "yena" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────────
COMPARISON_CSV = ROOT_DIR / "project" / "results" / "hojung" / "comparison_summary.csv"
QUANTIZATION_CSV = ROOT_DIR / "project" / "results" / "quantization_comparison.csv"
OUTPUT_DIR = ROOT_DIR / "project" / "results" / "yena"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# ── 컬러 팔레트 ────────────────────────────────────
COLORS = {
    "threshold":  "#E74C3C",
    "lstm_full":  "#3498DB",
    "ee_fixed":   "#2ECC71",
    "ee_dynamic": "#9B59B6",
}
color_list = list(COLORS.values())
METHOD_LABELS = [
    "① Threshold",
    "② LSTM Full",
    "③ EE Fixed θ",
    "④ EE Dynamic θ\n(Proposed)",
]

# ── 데이터 로드 ────────────────────────────────────
comp_df  = pd.read_csv(COMPARISON_CSV);  comp_df.columns  = comp_df.columns.str.strip()
quant_df = pd.read_csv(QUANTIZATION_CSV); quant_df.columns = quant_df.columns.str.strip()

acc     = comp_df["Accuracy(%)"].tolist()
latency = comp_df["Avg_Inference(ms)"].tolist()


# ──────────────────────────────────────────────────
# 그래프 1 — 정확도(막대) + 추론시간(꺾은선) 통합 이중 축
# ──────────────────────────────────────────────────
def plot_accuracy_latency_combined():
    fig, ax1 = plt.subplots(figsize=(11, 7))
    ax2 = ax1.twinx()   # 오른쪽 y축 (추론시간)

    x     = np.arange(len(METHOD_LABELS))
    width = 0.50

    # ── 막대그래프: 정확도 (왼쪽 y축) ──
    bars = ax1.bar(x, acc, width=width, color=color_list,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.88)
    # 제안 모델(④) 테두리 강조
    bars[3].set_edgecolor("#6C3483"); bars[3].set_linewidth(2.5); bars[3].set_alpha(1.0)
    # 막대 위 정확도 수치
    for bar, val in zip(bars, acc):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.6,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    # LSTM Full 기준선
    ax1.axhline(acc[1], color=COLORS["lstm_full"], linestyle="--",
                linewidth=1.2, alpha=0.6, zorder=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(METHOD_LABELS, fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=13, color="#2C3E50")
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis="y", labelcolor="#2C3E50")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines[["top"]].set_visible(False)

    # ── 꺾은선그래프: 추론시간 (오른쪽 y축) ──
    ax2.plot(x, latency, color="#E67E22", linewidth=2.2,
             marker="D", markersize=9, zorder=6,
             markerfacecolor="white", markeredgewidth=2.2,
             markeredgecolor="#E67E22", label="Avg Inference Time")
    # 꺾은선 수치 레이블
    for xi, val in zip(x, latency):
        offset = 0.035 if xi != 0 else -0.06   # ① Threshold는 값이 작아서 위로
        va_dir = "bottom" if xi != 0 else "top"
        ax2.text(xi + 0.12, val + offset,
                 f"{val:.3f} ms", ha="left", va=va_dir,
                 fontsize=10, color="#E67E22", fontweight="bold")
    # ① Threshold(0.011ms)가 너무 아래쪽이면 y축 여유 확보
    ax2.set_ylabel("Avg Inference Time (ms)", fontsize=13, color="#E67E22")
    y2_max = max(latency) * 1.45
    ax2.set_ylim(-0.05, y2_max)
    ax2.tick_params(axis="y", labelcolor="#E67E22")
    ax2.spines[["top"]].set_visible(False)

    # ── 범례 통합 ──
    bar_handles  = [mpatches.Patch(color=c, label=l, alpha=0.88)
                    for c, l in zip(color_list, [
                        "① Threshold", "② LSTM Full",
                        "③ EE Fixed θ", "④ EE Dynamic θ (Proposed)"])]
    line_handle  = plt.Line2D([0], [0], color="#E67E22", linewidth=2,
                               marker="D", markersize=8,
                               markerfacecolor="white", markeredgecolor="#E67E22",
                               label="Avg Inference Time (ms)")
    ax1.legend(handles=bar_handles + [line_handle],
               fontsize=9.5, loc="upper left",
               framealpha=0.9, edgecolor="#CCCCCC")

    ax1.set_title("Accuracy & Inference Time Comparison\n(Bar: Accuracy / Line: Inference Time)",
                  fontsize=14, fontweight="bold", pad=16)

    plt.tight_layout()
    out = OUTPUT_DIR / "accuracy_latency_combined.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 그래프 3 — Exit별 종료율 비교 (고정 θ vs 동적 θ)
# ──────────────────────────────────────────────────
def plot_exit_rate_comparison():
    fixed_row   = comp_df[comp_df["Method"].str.contains("Fixed",   case=False)].iloc[0]
    dynamic_row = comp_df[comp_df["Method"].str.contains("Dynamic", case=False)].iloc[0]
    fixed_exits   = [fixed_row["Exit1(%)"],   fixed_row["Exit2(%)"],   fixed_row["Exit3(%)"]]
    dynamic_exits = [dynamic_row["Exit1(%)"], dynamic_row["Exit2(%)"], dynamic_row["Exit3(%)"]]

    exits       = ["Exit 1\n(Layer 1)", "Exit 2\n(Layer 2)", "Exit 3\n(Layer 3)"]
    x           = np.arange(len(exits))
    exit_colors = ["#1ABC9C", "#F39C12", "#E74C3C"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, exit_vals, title, alpha_mul in zip(
        axes,
        [fixed_exits, dynamic_exits],
        ["③ EE Fixed θ  (Baseline)", "④ EE Dynamic θ  (Proposed)"],
        [0.75, 1.0],
    ):
        bars = ax.bar(x, exit_vals, color=exit_colors, width=0.5,
                      edgecolor="white", linewidth=1.2, alpha=alpha_mul, zorder=3)
        for bar, val in zip(bars, exit_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(exits, fontsize=11)
        ax.set_ylim(0, 90); ax.set_ylabel("Exit Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    for i, (fv, dv) in enumerate(zip(fixed_exits, dynamic_exits)):
        diff = dv - fv
        sign = "+" if diff >= 0 else ""
        axes[1].text(i, dynamic_exits[i] + 5.5, f"({sign}{diff:.1f}%p)",
                     ha="center", fontsize=10, color="#555555")

    fig.suptitle("Exit Rate Distribution\n③ Fixed θ vs ④ Dynamic θ",
                 fontsize=15, fontweight="bold", y=1.02)
    legend_handles = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(exit_colors,
                                      ["Exit 1 (fastest)", "Exit 2", "Exit 3 (deepest)"])]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = OUTPUT_DIR / "exit_rate_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 그래프 4 — 시나리오별 정확도 선그래프
# ──────────────────────────────────────────────────
def plot_scenario_accuracy():
    scenarios = ["Startup\nSurge", "Emergency\nRamp",
                 "Lunch\nRestart", "Imbalanced\nAP Load"]
    # 출처: docs/yongsang/stage3_work_log.md
    scenario_data = {
        "② LSTM Full":    [96.2, 91.6, 94.0, 92.5],
        "③ EE Fixed θ":   [96.2, 92.8, 96.4, 97.5],
        "④ EE Dynamic θ": [96.2, 95.2, 96.4, 97.5],
    }
    x      = np.arange(len(scenarios))
    styles = [
        ("② LSTM Full",    COLORS["lstm_full"],   "o", 2.0),
        ("③ EE Fixed θ",   COLORS["ee_fixed"],    "s", 2.0),
        ("④ EE Dynamic θ", COLORS["ee_dynamic"],  "^", 2.5),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, marker, lw in styles:
        vals = scenario_data[label]
        ax.plot(x, vals, marker=marker, color=color, linewidth=lw,
                markersize=9, label=label, zorder=4)
        for xi, val in zip(x, vals):
            ax.text(xi, val + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=color)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12); ax.set_ylim(85, 103)
    ax.set_title("Scenario-wise Accuracy Comparison\n(② LSTM Full / ③ EE Fixed θ / ④ EE Dynamic θ)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    out = OUTPUT_DIR / "scenario_accuracy.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 그래프 5 — 경량화(INT8 / ONNX) 전후 비교
# ──────────────────────────────────────────────────
def plot_quantization_comparison():
    model_labels = ["Baseline\nLSTM", "Early Exit\nFixed θ"]
    orig_size  = quant_df["original_size_mb"].tolist()
    quant_size = quant_df["quantized_size_mb"].tolist()
    orig_acc   = quant_df["original_accuracy"].tolist()
    quant_acc  = quant_df["quantized_accuracy"].tolist()
    orig_ms    = quant_df["original_inference_ms"].tolist()
    quant_ms   = quant_df["quantized_inference_ms"].tolist()
    onnx_ms    = [None, 0.0235]

    x     = np.arange(len(model_labels))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Model Quantization & ONNX Conversion Results\n(INT8 Post-Training Quantization)",
                 fontsize=14, fontweight="bold", y=1.02)

    # A: 모델 크기
    ax = axes[0]
    b1 = ax.bar(x - width/2, orig_size,  width, label="Original (FP32)",
                color="#3498DB", edgecolor="white", zorder=3)
    b2 = ax.bar(x + width/2, quant_size, width, label="Quantized (INT8)",
                color="#E67E22", edgecolor="white", zorder=3)
    for bar, val in zip(list(b1)+list(b2), orig_size+quant_size):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, (o, q) in enumerate(zip(orig_size, quant_size)):
        ax.text(i, max(o, q)+0.08, f"×{o/q:.1f}", ha="center",
                fontsize=10, color="#C0392B", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Model Size (MB)", fontsize=11)
    ax.set_title("① Model Size", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(orig_size)*1.45)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

    # B: 정확도
    ax = axes[1]
    b1 = ax.bar(x - width/2, orig_acc,  width, label="Original",
                color="#3498DB", edgecolor="white", zorder=3)
    b2 = ax.bar(x + width/2, quant_acc, width, label="Quantized",
                color="#E67E22", edgecolor="white", zorder=3)
    for bar, val in zip(list(b1)+list(b2), orig_acc+quant_acc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, (o, q) in enumerate(zip(orig_acc, quant_acc)):
        diff = q - o
        sign = "+" if diff >= 0 else ""
        ax.text(i, max(o,q)+0.7, f"{sign}{diff:.2f}%p", ha="center",
                fontsize=9, color="#27AE60" if diff>=0 else "#C0392B", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("② Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(min(orig_acc+quant_acc)-2, max(orig_acc+quant_acc)+3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

    # C: 추론 시간
    ax = axes[2]
    w3 = 0.22
    b1 = ax.bar(x - w3,    orig_ms,  w3, label="Original",
                color="#3498DB", edgecolor="white", zorder=3)
    b2 = ax.bar(x,          quant_ms, w3, label="INT8 Quantized",
                color="#E67E22", edgecolor="white", zorder=3)
    onnx_vals = [v if v is not None else 0 for v in onnx_ms]
    b3 = ax.bar(x + w3, onnx_vals, w3, label="ONNX Runtime",
                color="#9B59B6", edgecolor="white", zorder=3)
    ax.text(b3[1].get_x()+b3[1].get_width()/2, b3[1].get_height()+0.01,
            f"{onnx_ms[1]:.4f}ms", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#6C3483")
    for bar, val in zip(list(b1)+list(b2), orig_ms+quant_ms):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Inference Time (ms)", fontsize=11)
    ax.set_title("③ Inference Time", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "quantization_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("캡스톤디자인 Stage 4 - 결과 시각화")
    print("=" * 55)
    print("\n[1/4] 정확도 + 추론시간 통합 그래프...")
    plot_accuracy_latency_combined()
    print("\n[2/4] Exit 종료율 비교...")
    plot_exit_rate_comparison()
    print("\n[3/4] 시나리오별 정확도...")
    plot_scenario_accuracy()
    print("\n[4/4] 경량화 비교...")
    plot_quantization_comparison()
    print(f"\n완료! 저장 위치: {OUTPUT_DIR.resolve()}")
