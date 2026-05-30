
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────────
COMPARISON_CSV   = "comparison_summary.csv"
QUANTIZATION_CSV = "quantization_comparison.csv"
OUTPUT_DIR       = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
# 그래프 1 — 4개 방식 정확도 비교 막대그래프
# ──────────────────────────────────────────────────
def plot_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    x    = np.arange(len(METHOD_LABELS))
    bars = ax.bar(x, acc, color=color_list, width=0.55,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(METHOD_LABELS, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title("Classification Accuracy Comparison\n(4 Methods)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    bars[3].set_edgecolor("#6C3483"); bars[3].set_linewidth(2.5)
    ax.axhline(acc[1], color=COLORS["lstm_full"], linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"LSTM Full baseline ({acc[1]:.1f}%)")
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.show()
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 그래프 2 — 정확도 vs 추론시간 산점도  ★수정★
#   ②③④ 세 점이 0.56~0.78ms에 몰려 있어
#   텍스트를 절대 좌표로 분산 배치 + 화살표 연결
# ──────────────────────────────────────────────────
def plot_accuracy_vs_latency():
    # ★ 레이블 겹침 수정: 절대 좌표 분산 배치 + 화살표 연결
    #   ②③④ 세 점이 x=0.56~0.78ms에 몰려 있어 충분히 벌림
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (x_val, y_val, color) in enumerate(zip(latency, acc, color_list)):
        ax.scatter(x_val, y_val, c=color, s=320 if i==3 else 180,
                   marker="*" if i==3 else "o",
                   zorder=5, edgecolors="white", linewidths=1.5)

    label_cfg = [
        # (label,                           tx,    ty  )
        ("① Threshold",                    -0.04, 37.0),   # 점 아래
        ("② LSTM Full",                     0.87, 97.0),   # 오른쪽 단독
        ("③ EE Fixed θ",                   0.30, 101.5),  # 왼쪽 위
        ("④ EE Dynamic θ\n(Proposed)",     0.30, 88.5 ),  # 왼쪽 아래
    ]
    for i, ((label, tx, ty), x_val, y_val, color) in enumerate(
        zip(label_cfg, latency, acc, color_list)
    ):
        ax.annotate(
            label, xy=(x_val, y_val), xytext=(tx, ty),
            fontsize=10, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.9,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, lw=1.3, alpha=0.95),
            zorder=6,
        )

    # Ideal 화살표와 텍스트 완전 분리 (좌상단 구석)
    ax.annotate("", xy=(0.02, 103.0), xytext=(0.14, 97.0),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.4), zorder=4)
    ax.text(0.15, 97.5, "← Ideal\n   (Fast & Accurate)",
            fontsize=9, color="#888888", style="italic", va="bottom", ha="left")

    ax.set_xlabel("Avg Inference Time (ms)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Inference Latency\n(left-upper = better)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlim(-0.09, max(latency) * 1.38)
    ax.set_ylim(31, 107)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(color_list,
                               ["① Threshold", "② LSTM Full",
                                "③ EE Fixed θ", "④ EE Dynamic θ (Proposed)"])]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "accuracy_vs_latency.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.show()
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
    out = os.path.join(OUTPUT_DIR, "exit_rate_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.show()
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
    out = os.path.join(OUTPUT_DIR, "scenario_accuracy.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.show()
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
    out = os.path.join(OUTPUT_DIR, "quantization_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.show()
    print(f"[저장] {out}")


# ──────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("캡스톤디자인 Stage 4 — 결과 시각화")
    print("=" * 55)
    print("\n[1/5] 정확도 비교 막대그래프...")
    plot_accuracy_comparison()
    print("\n[2/5] 정확도 vs 추론시간 산점도...")
    plot_accuracy_vs_latency()
    print("\n[3/5] Exit 종료율 비교...")
    plot_exit_rate_comparison()
    print("\n[4/5] 시나리오별 정확도...")
    plot_scenario_accuracy()
    print("\n[5/5] 경량화 비교...")
    plot_quantization_comparison()
    print(f"\n완료! 저장 위치: {os.path.abspath(OUTPUT_DIR)}/")
