"""Comprehensive evaluation framework matching Yongsang's Stage 2 branch specs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# [피드백 4번 반영] 모든 경로를 PROJECT_ROOT 기준으로 엄밀하게 매핑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# [피드백 3번 반영] 클래스명을 BaselineLSTM으로 싱크 통합
from models.baseline_lstm import BaselineLSTM
from models.early_exit_lstm import EarlyExitLSTM
from experiments.channel_optimizer import optimize_channel
# [피드백 1번 반영] yongsang 최신본 dataloader.py 인터페이스 활용
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

    # [피드백 4번 반영] 데이터 경로 PROJECT_ROOT 기준 설정
    test_csv_path = PROJECT_ROOT / "data" / "real" / "test.csv"
    if not test_csv_path.exists():
        print(f"❌ Error: Test data not found at {test_csv_path}")
        sys.exit(1)

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

    print(f"  Accuracy: {t1_acc:.1f}% | Avg Inference: {t1_avg_time:.1f}ms")
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
    # [피드백 3번 반영] 클래스명 BaselineLSTM 사용
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

    print(f"  Accuracy: {t2_acc:.1f}% | Avg Inference: {t2_avg_time:.1f}ms")
    save_detailed_csv("baseline_lstm.csv", t2_trues, t2_preds, t2_times, t2_actions)

    summary_results.append({
        "Method": "② Baseline LSTM Full", "Accuracy": f"{t2_acc:.1f}%", "Avg_Inference_ms": f"{t2_avg_time:.3f}",
        "Exit_1_Rate": "0.0%", "Exit_2_Rate": "0.0%", "Exit_3_Rate": "100.0%", "Unnecessary_Switches": unnecessary_switches_t2,
        "Label_0_Acc": f"{t2_lbl_accs[0]:.1f}%", "Label_1_Acc": f"{t2_lbl_accs[1]:.1f}%", "Label_2_Acc": f"{t2_lbl_accs[2]:.1f}%", "Label_3_Acc": f"{t2_lbl_accs[3]:.1f}%"
    })

    # ==========================================
    # 🚀 방식 ③: Early Exit + 고정 θ (유용상 구현)
    # ==========================================
    print("Running Baseline ③ (Early Exit Fixed θ)...")
    model_ee = EarlyExitLSTM(hidden_size=128).to(device)
    # [피드백 2번 반영] Early Exit ③ 실행 직전에 정확하게 가중치 로드 조치
    ee_path = PROJECT_ROOT / "checkpoints" / "early_exit_lstm_best.pth"
    if ee_path.exists():
        ckpt = torch.load(ee_path, map_location=device)
        model_ee.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model_ee.eval()

    t3_preds, t3_trues, t3_times, t3_actions = [], [], [], []
    exit_counts_t3 = {1: 0, 2: 0, 3: 0}
    unnecessary_switches_t3 = 0
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
            decisions = model_ee.infer_batch(x_batch, dynamic=False)
            end = time.perf_counter()
            
            t3_times.append((end - start) * 1000)
            pred = decisions[0].logits.argmax(dim=-1).item()
            t3_preds.append(pred)
            
            exit_counts_t3[decisions[0].exit_point] += 1
            true_lbl = y_batch.item()
            t3_trues.append(true_lbl)
            
            next_ch, act = optimize_channel(pred, current_ch, AVAILABLE_CHANNELS, metrics=simulated_metrics, current_time=simulated_time)
            t3_actions.append(act)
            if true_lbl == 0 and act in ["switch", "emergency"]:
                unnecessary_switches_t3 += 1
                
            current_ch = next_ch
            simulated_time += 10.0
            idx += 1

    y_true_t3 = np.array(t3_trues)
    y_pred_t3 = np.array(t3_preds)
    t3_acc = np.mean(y_pred_t3 == y_true_t3) * 100
    t3_avg_time = np.mean(t3_times)
    total_samples = len(t3_trues)
    e1_r = (exit_counts_t3[1] / total_samples) * 100
    e2_r = (exit_counts_t3[2] / total_samples) * 100
    e3_r = (exit_counts_t3[3] / total_samples) * 100
    t3_lbl_accs = compute_label_accuracies(y_true_t3, y_pred_t3)

    print(f"  Accuracy: {t3_acc:.1f}% | Avg Inference: {t3_avg_time:.1f}ms")
    print(f"  Exit 1: {e1_r:.1f}% | Exit 2: {e2_r:.1f}% | Exit 3: {e3_r:.1f}%")
    save_detailed_csv("early_exit_fixed.csv", t3_trues, t3_preds, t3_times, t3_actions)

    summary_results.append({
        "Method": "③ Early Exit Fixed θ", "Accuracy": f"{t3_acc:.1f}%", "Avg_Inference_ms": f"{t3_avg_time:.3f}",
        "Exit_1_Rate": f"{e1_r:.1f}%", "Exit_2_Rate": f"{e2_r:.1f}%", "Exit_3_Rate": f"{e3_r:.1f}%", "Unnecessary_Switches": unnecessary_switches_t3,
        "Label_0_Acc": f"{t3_lbl_accs[0]:.1f}%", "Label_1_Acc": f"{t3_lbl_accs[1]:.1f}%", "Label_2_Acc": f"{t3_lbl_accs[2]:.1f}%", "Label_3_Acc": f"{t3_lbl_accs[3]:.1f}%"
    })

    # ==========================================
    # 🚀 방식 ④: 제안 모델 (Early Exit + 동적 θ)
    # ==========================================
    print("Running Baseline ④ (Early Exit Dynamic θ)...")
    # [피드백 2번 반영] 방식 ④ 구동 전에도 일관되게 에폭 베스트 가중치 초기성 보장 로드
    if ee_path.exists():
        ckpt = torch.load(ee_path, map_location=device)
        model_ee.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model_ee.eval()

    t4_preds, t4_trues, t4_times, t4_actions = [], [], [], []
    exit_counts_t4 = {1: 0, 2: 0, 3: 0}
    unnecessary_switches_t4 = 0
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
            decisions = model_ee.infer_batch(x_batch, dynamic=True)
            end = time.perf_counter()
            
            t4_times.append((end - start) * 1000)
            pred = decisions[0].logits.argmax(dim=-1).item()
            t4_preds.append(pred)
            
            exit_counts_t4[decisions[0].exit_point] += 1
            true_lbl = y_batch.item()
            t4_trues.append(true_lbl)
            
            next_ch, act = optimize_channel(pred, current_ch, AVAILABLE_CHANNELS, metrics=simulated_metrics, current_time=simulated_time)
            t4_actions.append(act)
            if true_lbl == 0 and act in ["switch", "emergency"]:
                unnecessary_switches_t4 += 1
                
            current_ch = next_ch
            simulated_time += 10.0
            idx += 1

    y_true_t4 = np.array(t4_trues)
    y_pred_t4 = np.array(t4_preds)
    t4_acc = np.mean(y_pred_t4 == y_true_t4) * 100
    t4_avg_time = np.mean(t4_times)
    e1_r_t4 = (exit_counts_t4[1] / total_samples) * 100
    e2_r_t4 = (exit_counts_t4[2] / total_samples) * 100
    e3_r_t4 = (exit_counts_t4[3] / total_samples) * 100
    t4_lbl_accs = compute_label_accuracies(y_true_t4, y_pred_t4)

    print(f"  Accuracy: {t4_acc:.1f}% | Avg Inference: {t4_avg_time:.1f}ms")
    print(f"  Exit 1: {e1_r_t4:.1f}% | Exit 2: {e2_r_t4:.1f}% | Exit 3: {e3_r_t4:.1f}%")
    save_detailed_csv("early_exit_dynamic.csv", t4_trues, t4_preds, t4_times, t4_actions)

    summary_results.append({
        "Method": "④ Proposed (EE + Dynamic θ)", "Accuracy": f"{t4_acc:.1f}%", "Avg_Inference_ms": f"{t4_avg_time:.3f}",
        "Exit_1_Rate": f"{e1_r_t4:.1f}%", "Exit_2_Rate": f"{e2_r_t4:.1f}%", "Exit_3_Rate": f"{e3_r_t4:.1f}%", "Unnecessary_Switches": unnecessary_switches_t4,
        "Label_0_Acc": f"{t4_lbl_accs[0]:.1f}%", "Label_1_Acc": f"{t4_lbl_accs[1]:.1f}%", "Label_2_Acc": f"{t4_lbl_accs[2]:.1f}%", "Label_3_Acc": f"{t4_lbl_accs[3]:.1f}%"
    })

    # ==========================================
    # 📊 [피드백 5번 반영] CSV와 TXT 둘 다 저장 구현
    # ==========================================
    summary_df = pd.DataFrame(summary_results)
    
    # 1. CSV 저장
    csv_path = PROJECT_ROOT / "results" / "comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    # 2. TXT 저장 (Report 가독성 포맷 적용)
    txt_path = PROJECT_ROOT / "results" / "comparison_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("      RRM STAGE 2 MULTI-BASELINE BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        for res in summary_results:
            f.write(f"▶ {res['Method']}\n")
            f.write(f"  - Total Accuracy     : {res['Accuracy']}\n")
            f.write(f"  - Avg Inference Time : {res['Avg_Inference_ms']} ms\n")
            f.write(f"  - Exit Rates         : Exit1({res['Exit_1_Rate']}) | Exit2({res['Exit_2_Rate']}) | Exit3({res['Exit_3_Rate']})\n")
            f.write(f"  - Unnecessary Switch : {res['Unnecessary_Switches']} times\n")
            f.write(f"  - Label Accuracies   : L0({res['Label_0_Acc']}) | L1({res['Label_1_Acc']}) | L2({res['Label_2_Acc']}) | L3({res['Label_3_Acc']})\n")
            f.write("-" * 70 + "\n")

    print("\n✅ Results successfully compilation complete!")
    print("  -> Saved to results/comparison_summary.csv")
    print("  -> Saved to results/comparison_summary.txt")
