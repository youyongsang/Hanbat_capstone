import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from utils.dataloader import get_dataloader

# 채널 최적화 상태 제어 모듈 연동
import experiments.channel_optimizer as ch_opt

# [브런치의 실제 파일 연동] 두 팀원의 모델을 순수하게 그대로 로드
from models.baseline_lstm import Standard3LayerLSTM
from models.early_exit_lstm import EarlyExitLSTM

os.makedirs("results", exist_ok=True)

# ----------------------------------------------------
# 5. Baseline ① 임계값 방식 구현 (명세서 기준 100% 일치)
# ----------------------------------------------------
def threshold_baseline(channel_occupancy):
    """
    현행 임계값 기반 혼잡 감지
    채널 점유율만 보고 레이블 결정
    """
    if channel_occupancy < 40:
        return 0  # 정상
    elif channel_occupancy < 65:
        return 1  # 혼잡 경고
    elif channel_occupancy < 85:
        return 2  # 혼잡
    else:
        return 3  # 심각

# ----------------------------------------------------
# 레이블별 정확도 계산 함수 (김호중 담당 지표)
# ----------------------------------------------------
def calculate_label_accuracies(y_true, y_pred):
    accs = {}
    for label in [0, 1, 2, 3]:
        idx = (y_true == label)
        if np.sum(idx) > 0:
            accs[f"Acc_Label_{label}(%)"] = round(accuracy_score(y_true[idx], y_pred[idx]) * 100, 1)
        else:
            accs[f"Acc_Label_{label}(%)"] = 0.0
    return accs

# ----------------------------------------------------
# 6. 비교 실험 실행 스크립트 메인 루프
# ----------------------------------------------------
def main():
    device = torch.device("cpu") # 엣지 환경 기준 CPU 측정 원칙 준수
    try:
        test_loader = get_dataloader("data/real/test.csv", batch_size=1, shuffle=False)
    except Exception as e:
        print(f"[Error] 데이터 로더 연결 실패: {e}")
        return

    # 호중 님 오리지널 baseline_lstm.py 모델 생성 및 가중치 파일 결합
    model_lstm = Standard3LayerLSTM()
    checkpoint_path = "checkpoints/baseline_lstm_best.pth"
    if os.path.exists(checkpoint_path):
        model_lstm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_lstm.eval()

    # CPU Latency 측정을 위한 50회 예열 (Warm-up)
    dummy_input = torch.randn(1, 10, 4)
    for _ in range(50):
        with torch.no_grad(): _ = model_lstm(dummy_input)

    summary_results = []
    methods = [
        {"id": 1, "name": "Baseline ① (Threshold)", "file": "baseline_threshold.csv"},
        {"id": 2, "name": "Baseline ② (LSTM Full)", "file": "baseline_lstm.csv"},
        {"id": 3, "name": "Baseline ③ (Early Exit Fixed θ)", "file": "early_exit_fixed.csv"},
        {"id": 4, "name": "Baseline ④ (Early Exit Dynamic θ)", "file": "early_exit_dynamic.csv"}
    ]

    for m in methods:
        # 명세서 서식 완전 일치 출력
        print(f"Running {m['name']}...")
        
        # 방식 변경 시 Hysteresis 채널 상태 리셋
        ch_opt._rrm_state = {"last_switch_time": 0, "current_channel": None}

        all_preds, all_labels, inference_times = [], [], []
        unnecessary_switches = 0
        current_channel = 1
        exit_counts = {1: 0, 2: 0, 3: 0} # 유용상 담당 지표 카운터

        # 용상 님 early_exit_lstm.py 모델 할당
        model_ee = None
        if m["id"] in [3, 4]:
            model_ee = EarlyExitLSTM(input_size=4, hidden_size=128, num_classes=4)
            model_ee.set_threshold(dynamic=(m["id"] == 4))
            model_ee.eval() 

        for step, (features, targets) in enumerate(test_loader):
            true_label = int(targets[0].item())
            
            start_time = time.perf_counter()
            
            # --- 4대 방식 모델 추론 분기 ---
            if m["id"] == 1:
                # 명세서 기준: 시계열 없이 현재 타임스텝의 채널 점유율 인덱스 추출 (* 100으로 % 스케일링)
                occ_val = features.numpy()[0, -1, 1] * 100
                pred = threshold_baseline(occ_val)
            elif m["id"] == 2:
                with torch.no_grad():
                    pred = torch.argmax(model_lstm(features), dim=1).item()
            elif m["id"] in [3, 4]:
                with torch.no_grad():
                    decisions = model_ee.infer_batch(features, dynamic=(m["id"] == 4))
                    pred = torch.argmax(decisions[0].logits, dim=-1).item()
                    exit_counts[decisions[0].exit_point] += 1
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000) # ms 단위 보정
            
            all_preds.append(pred)
            all_labels.append(true_label)

            # 불필요 채널 전환 횟수 계산 (김호중 담당 지표)
            next_channel, _ = ch_opt.optimize_channel(
                pred, current_channel, [1, 6, 11, 36], 
                {c: {"rssi": -60, "noise": -95, "utilization": 0.2} for c in [1, 6, 11, 36]}, 
                step * 10
            )
            if true_label == 0 and next_channel != current_channel:
                unnecessary_switches += 1
            current_channel = next_channel

        # 최종 지표 연산 및 포맷 매칭
        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        acc = accuracy_score(y_true, y_pred) * 100
        avg_time = np.mean(inference_times)
        
        label_accs = calculate_label_accuracies(y_true, y_pred)
        
        total = max(len(all_preds), 1)
        e1, e2, e3 = (exit_counts[1]/total)*100, (exit_counts[2]/total)*100, (exit_counts[3]/total)*100

        # 명세서 출력 예시와 공백, 형태까지 완전히 일치화
        print(f"  Accuracy: {acc:.1f}% | Avg Inference: {avg_time:.1f}ms")
        if m["id"] in [3, 4]:
            print(f"  Exit 1: {e1:.1f}% | Exit 2: {e2:.1f}% | Exit 3: {e3:.1f}%")

        # 개별 결과 CSV 출력 저장
        pd.DataFrame({
            "True_Label": y_true, 
            "Predicted_Label": y_pred, 
            "Inference_Time_ms": inference_times
        }).to_csv(f"results/{m['file']}", index=False)
        
        # 통합 요약 데이터 빌드
        res_dict = {
            "Method": m["name"], 
            "Accuracy(%)": round(acc, 1), 
            "Avg_Inference(ms)": round(avg_time, 4), 
            "Unnecessary_Switches": unnecessary_switches
        }
        if m["id"] in [3, 4]:
            res_dict.update({"Exit1(%)": round(e1, 1), "Exit2(%)": round(e2, 1), "Exit3(%)": round(e3, 1)})
        res_dict.update(label_accs)
        summary_results.append(res_dict)

    # 최종 comparison_summary.csv 파일 저장
    pd.DataFrame(summary_results).to_csv("results/comparison_summary.csv", index=False)
    print("Results saved to results/comparison_summary.csv")

if __name__ == "__main__":
    main()
