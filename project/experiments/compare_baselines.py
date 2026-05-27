import os
import sys
from pathlib import Path

# =========================================================================
# [🔥 경로 에러 해결] ModuleNotFoundError 방지를 위해 sys.path 추가를 최상단에 배치
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
import numpy as np
import pandas as pd
import torch

# 프로젝트 모듈 임포트 (용상 최신 사양 반영)
from models.baseline_lstm import BaselineLSTM
from models.early_exit_lstm import EarlyExitLSTM
from experiments.channel_optimizer import optimize_channel
from utils.dataloader import get_dataloader, load_csv_windows


def threshold_baseline(channel_occupancy: float) -> int:
    """Baseline ①: 현행 임계값 기반 혼잡 감지"""
    if channel_occupancy < 40:
        return 0
    elif channel_occupancy < 65:
        return 1
    elif channel_occupancy < 85:
        return 2
    else:
        return 3


def compute_label_accuracies(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    label_accs = {}
    for lbl in range(4):
        mask = (y_true == lbl)
        if np.sum(mask) > 0:
            label_accs[lbl] = np.mean(y_pred[mask] == y_true[mask]) * 100
        else:
            label_accs[lbl] = 0.0
    return label_accs


def save_detailed_csv(filename: str, y_true: list[int], y_pred: list[int], latencies: list[float], actions: list[str]) -> None:
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame({
        "sample_idx": range(len(y_true)),
        "true_label": y_true,
        "predicted_label": y_pred,
        "latency_ms": latencies,
        "action": actions
    })
    df.to_csv(output_dir / filename, index=False)


if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"🖥️  Edge Simulation Core Target Device: {device}\n")

    # 데이터 경로설정 (PROJECT_ROOT 기준)
    test_csv_path = PROJECT_ROOT / "data" / "real" / "test.csv"
    if not test_csv_path.exists():
        print(f"❌ Error: Test data not found at {test_csv_path}")
        sys.exit(1)

    # 용상 최신 dataloader 활용 인터페이스
    test_loader = get_dataloader(test_csv_path, batch_size=1, shuffle=False)
    samples_np, labels_np = load_csv_windows(test_csv_path, window_size=10)
    
    AVAILABLE_CHANNELS = [1, 6, 11, 36]
    START_CHANNEL = 1
    summary_results = []

    # ==========================================
    # 🚀 방식 ①: 임계값 방식 (현행)
    # ==========================================
    print("Running Baseline ① (Threshold)...")
    t1_preds, t1_trues, t1_times, t1_actions = [], [], [], []
    unnecessary_switches_t1 = 0
    current_ch = START_CHANNEL
    simulated_time = time.time()

    for idx in range(len(samples_np)):
        norm_occupancy = samples_np[idx, -1, 1]
        real_occupancy = norm_occupancy * 100.0
        
        simulated_metrics = {
            ch: {'rssi': -65 if ch == current_ch else -60, 'noise': -95, 'utilization': norm_occupancy if ch == current_ch else 0.3}
            for ch in AVAILABLE_CHANNELS
        }
        
        start = time.perf_counter()
        pred = threshold_baseline(real_occupancy)
        end = time.perf_counter()
        
        t1_times.append((end - start) * 1000)
        t1_preds.append(pred)
        
        true_lbl = int(labels_np[idx])
        t1_trues.append(true_lbl)
        
        next_ch, act = optimize_channel(pred, current_ch, AVAILABLE_CHANNELS, metrics=simulated_metrics, current_time=simulated_time)
        t1_actions.append(act)
        
        if true_lbl == 0 and act in ["switch", "emergency"]:
            unnecessary_switches_t1 += 1
            
        current_ch = next_ch
        simulated_time += 10.0

    y_true_t1 = np.array(t1_trues)
    y_pred_t1 = np.array(t1_preds)
    t1_acc = np.mean(y_pred_t1 == y_true_t1) * 100
    t1_avg_time = np.mean(t1_times)
    t1_lbl_accs = compute_label_accuracies(y_true_t1, y_pred_t1)

    print(f"  Accuracy: {t1_acc:.1f}% | Avg Inference: {t1_avg_time:.3f}ms")
    save_detailed_csv("baseline_threshold.csv", t1_trues, t1_preds, t1_times, t1_actions)
    
    summary_results.append({
        "Method": "① Baseline Threshold", "Accuracy": f"{t1_acc:.1f}%", "Avg_Inference_ms": f"{t1_avg_time:.3f}",
        "Exit_1_Rate": "0.0%", "Exit_2_Rate": "0.0%", "Exit_3_Rate": "100.0%", "Unnecessary_Switches": unnecessary_switches_t1,
        "Label_0_Acc": f"{t1_lbl_accs[0]:.1f}%", "Label_1_Acc": f"{t1_lbl_accs[1]:.1f}%", "Label_2_Acc": f"{t1_lbl_accs[2]:.1f}%", "Label_3_Acc": f"{t1_lbl_accs[3]:.1f}%"
    })

    # ==========================================
    # 🚀 방식 ②: 일반 LSTM (호중 님 백본)
    # ==========================================
    print("Running Baseline ② (LSTM Full)...")
    model_lstm = BaselineLSTM(hidden_size=128).to(device)
    lstm_path = PROJECT_ROOT / "checkpoints" / "baseline_lstm_best.pth"
    if lstm_path.exists():
        ckpt = torch.load(lstm_path, map_location=device)
        model_lstm.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model_lstm.eval()

    t2_preds, t2_trues, t2_times, t2_actions = [], [], [], []
    unnecessary_switches_t2 = 0
    current_ch = START_CHANNEL
    simulated_time = time.time()
    idx = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            norm_occupancy = samples_np[idx, -1, 1]
            
            simulated_metrics = {
                ch: {'rssi': -65 if ch == current_ch else -60, 'noise': -95, 'utilization': norm_occupancy if ch == current_ch else 0.3}
                for ch in AVAILABLE_CHANNELS
            }
            
            start = time.perf_counter()
            logits = model_lstm(x_batch)
            pred = torch.argmax(logits, dim=1).item()
            end = time.perf_counter()
            
            t2_times.append((end - start) * 1000)
            t2_preds.append(pred)
            
            true_lbl = y_batch.item()
            t2_trues.append(true_lbl)
            
            next_ch, act = optimize_channel(pred, current_ch, AVAILABLE_CHANNELS, metrics=simulated_metrics, current_time=simulated_time)
            t2_actions.append(act)
            if true_lbl == 0 and act in ["switch", "emergency"]:
                unnecessary_switches_t2 += 1
                
            current_ch = next_ch
            simulated_time += 10.0
            idx += 1

    y_true_t2 = np.array(t2_trues)
    y_pred_t2 = np.array(t2_preds)
    t2_acc = np.mean(y_pred_t2 == y_true_t2) * 100
    t2_avg_time = np.mean(t2_times)
    t2_lbl_accs = compute_label_accuracies(y_true_t2, y_pred_t2)

    print(f"  Accuracy: {t2_acc:.1f}% | Avg Inference: {t2_avg_time:.3f}ms")
    save_detailed_csv("baseline_lstm.csv", t2_trues, t2_preds, t2_times, t2_actions)

    summary_results.append({
        "Method": "② Baseline LSTM Full", "Accuracy": f"{t2_acc:.1f}%", "Avg_Inference_ms": f"{t2_avg_time:.3f}",
        "Exit_1_Rate": "0.
