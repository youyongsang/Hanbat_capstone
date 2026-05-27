import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from utils.dataloader import get_dataloader

# Hysteresis 상태 초기화를 위해 기존 모듈 임포트
import experiments.channel_optimizer as ch_opt

# [정석 연동] 호중 님 파일과 용상 님 파일에서 모델 클래스를 각각 다이렉트로 임포트
from models.baseline_lstm import Standard3LayerLSTM  # 호중 님 진짜 LSTM 클래스명에 맞게 확인 필요
from models.early_exit_lstm import EarlyExitLSTM

os.makedirs("results", exist_ok=True)

# ----------------------------------------------------
# 5. Baseline ① 임계값 방식 구현 (명세서 규칙 100% 반영)
# ----------------------------------------------------
def threshold_baseline(channel_occupancy):
    """현행 임계값 기반 혼잡 감지. 시계열 패턴 없이 현재 타임스텝만 보고 판단."""
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
    device = torch.device("cpu") # 주의사항: 엣지 환경 기준 CPU 측정
    try:
        test_loader = get_dataloader("data/real/test.csv", batch_size=1, shuffle=False)
    except Exception as e:
        print(f"[Error] 데이터 로더 연결 실패: {e}")
        return

    # [원본 성능 복원] baseline_lstm.py에서 가져온 순수 뼈대에 호중 님 진짜 가중치 로드
    model_lstm = Standard3LayerLSTM()
    checkpoint_path = "checkpoints/baseline_lstm_best.pth"
    if os.path.exists(checkpoint_path):
        model_lstm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_lstm.eval()

    # CPU Latency 측정을 위한 예열 (Warm-up)
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
        # 명세서 출력 예시 포맷 적용
        print(f"Running {m['name']}...")
        
        # Hysteresis 상태 초기화
        ch_opt._rrm_state = {"last_switch_time": 0, "current_channel": None}

        all_preds, all_labels, inference_times = [], [], []
        unnecessary_switches = 0
        current_channel = 1
        exit_counts = {1: 0, 2: 0, 3: 0} # 유용상 담당 지표용 카운터

        model_ee = None
        if m["id"] in [3, 4]:
            model_ee = EarlyExitLSTM(input_size=4, hidden_size=128, num_classes=4)
            model_ee.set_threshold(dynamic=(m["id"] == 4))
            model_ee.eval() 

        for step, (features, targets) in enumerate(test_loader):
            true_label = int(targets[0].item())
            
            start_time = time.perf_counter()
            
            # --- 4대 방식 순수 모델 추론 구간 ---
            if m["id"] == 1:
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
            inference_times.append((end_time - start_time) * 1000) # ms 단위 변환
            
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

        # 평가지표 연산
        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        acc = accuracy_score(y_true, y_pred) * 100
        avg_time = np.mean(inference_times)
        
        # 김호중 담당: 레이블별 개별 정확도 추출
        label_accs = calculate_label_accuracies(y_true, y_pred)
        
        # 유용상 담당: Exit별 종료율 추출
        total = max(len(all_preds), 1)
        e1, e2, e3 = (exit_counts[1]/total)*100, (exit_counts[2]/total)*100, (exit_counts[3]/total)*100

        # 명세서 출력 서식 완전 일치 (6번 형식)
        print(f"  Accuracy: {acc:.1f}% | Avg Inference: {avg_time:.1f}ms")
        if m["id"] in [3, 4]:
            print(f"  Exit 1: {e1:.1f}% | Exit 2: {e2:.1f}% | Exit 3: {e3:.1f}%")

        # 개별 파일 csv 저장 (명세서 4번 양식)
        pd.DataFrame({
            "True_Label": y_true, 
            "Predicted_Label": y_pred, 
            "Inference_Time_ms": inference_times
        }).to_csv(f"results/{m['file']}", index=False)
        
        # 통합 summary 데이터 축적
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

    # 최종 요약본 csv 저장 (명세서 4번 양식)
    pd.DataFrame(summary_results).to_csv("results/comparison_summary.csv", index=False)
    print("Results saved to results/comparison_summary.csv")

if __name__ == "__main__":
    main()
