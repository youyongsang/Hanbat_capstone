"""Create final PC-side result graphs for report and presentation.

Inputs:
- project/results/hojung/comparison_summary.csv
- project/results/quantization_comparison.csv

Outputs:
- project/results/final_figures/pc_accuracy_latency.png
- project/results/final_figures/pc_exit_distribution.png
- project/results/final_figures/pc_quantization_comparison.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


COMPARISON_CSV = PROJECT_ROOT / "results" / "hojung" / "comparison_summary.csv"
QUANTIZATION_CSV = PROJECT_ROOT / "results" / "quantization_comparison.csv"
DPI = 300

METHOD_LABELS = ["Threshold", "LSTM Full", "EE Fixed", "EE Dynamic"]
MODEL_COLORS = ["#9AA4B2", "#1F5A93", "#12A1A6", "#0D7377"]
EXIT_COLORS = ["#3AAFA9", "#F4A261", "#E76F51"]
LATENCY_COLOR = "#D7263D"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {display_path(path)}")


def percent_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def load_comparison() -> pd.DataFrame:
    require_file(COMPARISON_CSV)
    df = pd.read_csv(COMPARISON_CSV)
    df.columns = df.columns.str.strip()
    return df


def load_quantization() -> pd.DataFrame:
    require_file(QUANTIZATION_CSV)
    df = pd.read_csv(QUANTIZATION_CSV)
    df.columns = df.columns.str.strip()
    return df


def plot_pc_accuracy_latency(df: pd.DataFrame) -> Path:
    x = np.arange(len(df))
    accuracy = df["Accuracy(%)"].astype(float).to_numpy()
    latency = df["Avg_Inference(ms)"].astype(float).to_numpy()
    latency_std = df.get("Std_Inference(ms)", pd.Series([0] * len(df))).astype(float).to_numpy()

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, accuracy, width=0.52, color=MODEL_COLORS, edgecolor="white", linewidth=1.2)
    for bar, value in zip(bars, accuracy):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax2.errorbar(
        x,
        latency,
        yerr=latency_std,
        color=LATENCY_COLOR,
        marker="D",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        label="Avg inference time",
    )
    for xi, value in zip(x, latency):
        ax2.text(xi + 0.08, value, f"{value:.4f}ms", ha="left", va="center", fontsize=9, color=LATENCY_COLOR)

    ax1.set_xticks(x)
    ax1.set_xticklabels(METHOD_LABELS, fontsize=10)
    percent_axis(ax1)
    ax2.set_ylabel("Avg inference time (ms)")
    ax2.spines[["top"]].set_visible(False)
    ax2.tick_params(axis="y", colors=LATENCY_COLOR)
    ax2.yaxis.label.set_color(LATENCY_COLOR)

    fig.suptitle("PC Model Accuracy and Inference Time", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "Timing values are averaged over repeated runs.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out = OUTPUT_DIR / "pc_accuracy_latency.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pc_exit_distribution(df: pd.DataFrame) -> Path:
    ee_df = df[df["Method"].str.contains("Early Exit", case=False, na=False)].copy()
    labels = ["EE Fixed", "EE Dynamic"]
    exit_cols = ["Exit1(%)", "Exit2(%)", "Exit3(%)"]
    values = ee_df[exit_cols].fillna(0).astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for idx, (column, color) in enumerate(zip(exit_cols, EXIT_COLORS)):
        bars = ax.bar(x, values[:, idx], bottom=bottom, width=0.5, color=color, edgecolor="white", label=column)
        for bar, value, base in zip(bars, values[:, idx], bottom):
            if value >= 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
        bottom += values[:, idx]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exit rate (%)")
    ax.set_title("PC Early Exit Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(["Exit 1", "Exit 2", "Exit 3"], loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)

    out = OUTPUT_DIR / "pc_exit_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pc_quantization(df: pd.DataFrame) -> Path:
    labels = ["LSTM", "EE Fixed", "EE Dynamic"]
    x = np.arange(len(df))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))

    size_cols = ("original_size_mb", "quantized_size_mb")
    acc_cols = ("original_accuracy", "quantized_accuracy")
    time_cols = ("original_inference_ms", "quantized_inference_ms")
    time_std_cols = ("original_inference_std_ms", "quantized_inference_std_ms")

    panels = [
        (axes[0], size_cols, "Model Size", "MB", None),
        (axes[1], acc_cols, "Accuracy", "%", None),
        (axes[2], time_cols, "Inference Time", "ms", time_std_cols),
    ]

    for ax, cols, title, ylabel, std_cols in panels:
        original = df[cols[0]].astype(float).to_numpy()
        quantized = df[cols[1]].astype(float).to_numpy()
        yerr_original = df[std_cols[0]].astype(float).to_numpy() if std_cols else None
        yerr_quantized = df[std_cols[1]].astype(float).to_numpy() if std_cols else None

        bars1 = ax.bar(
            x - width / 2,
            original,
            width,
            yerr=yerr_original,
            capsize=3 if std_cols else 0,
            color="#1F5A93",
            edgecolor="white",
            label="FP32",
        )
        bars2 = ax.bar(
            x + width / 2,
            quantized,
            width,
            yerr=yerr_quantized,
            capsize=3 if std_cols else 0,
            color="#D95F02",
            edgecolor="white",
            label="INT8",
        )

        label_offset = 0.04 if title == "Accuracy" else max(max(original.max(), quantized.max()) * 0.015, 0.01)
        for bar, value in zip(list(bars1) + list(bars2), list(original) + list(quantized)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{value:.3f}" if ylabel != "%" else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        if title == "Accuracy":
            ymin = max(90.0, min(original.min(), quantized.min()) - 0.8)
            ymax = max(original.max(), quantized.max()) + 0.25
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("Accuracy (%)\n(axis starts near 90%)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    fig.suptitle("PC Quantization Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUTPUT_DIR / "pc_quantization_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison()
    quantization = load_quantization()

    outputs = [
        plot_pc_accuracy_latency(comparison),
        plot_pc_exit_distribution(comparison),
        plot_pc_quantization(quantization),
    ]
    print("PC result figures generated:")
    for output in outputs:
        print(f"- {display_path(output)}")


if __name__ == "__main__":
    main()
