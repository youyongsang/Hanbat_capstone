import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.dataloader import get_dataloader

# Hysteresis 상태 초기화를 위해 모듈 임포트 방식 변경
import experiments.channel_optimizer as ch_opt

# [안전 연동] 용상이 코드 내부의 함수 순서 꼬임 현상을 내장 영역에서 동적 해결
try:
    import builtins
    import models.early_exit_lstm as ee_mod
    if hasattr(ee_mod, 'compute_dynamic_threshold'):
        builtins.compute_dynamic_threshold = ee_mod.compute_dynamic_threshold
    if hasattr(ee_mod, 'entropy_from_logits'):
        builtins.entropy_from_logits = ee_mod.entropy_from_logits
        
    from models.early_exit_lstm import EarlyExitLSTM
    HAS_EARLY_EXIT = True
except Exception:
    HAS_EARLY_EXIT = False

os.makedirs('results', exist_ok=True)

# ----------------------------------------------------
# 1. Baseline ①: 임계값 기반 규칙 탐지기 (명세서 기준: 65% 초과 시 무조건 전환 타겟 레이블 반환)
# ----------------------------------------------------
def threshold_baseline(channel_occupancy):
    # 명세서 반영: 채널 점유율 65% 초과 시 무조건 전환 레이블 생성
    if channel_occupancy > 65:
        return 2  # 채널 전환 유도 레이블
    return 0

# ----------------------------------------------------
# 2. Baseline ②: 정석 3-Layer LSTM 모델
# ----------------------------------------------------
class Standard3LayerLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_classes=4):
        super(Standard3LayerLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :])
        return out

# ----------------------------------------------------
# 레이블별 정확도 계산 함수 (김호중 담당 지표)
# ----------------------------------------------------
def calculate_label_accuracies(y_true, y_pred):
    accs = {}
    for label in [0, 1, 2, 3]:
        idx = (y_true == label)
        if np.sum(idx) > 0:
            accs[f'Acc_Label_{label}(%)'] = round(accuracy_score(y_true[idx], y_pred[idx]) * 100, 2)
        else:
            accs[f'Acc_Label_{label}(%)'] = 0.0
    return accs

# ----------------------------------------------------
# 3. 통합 실험 프레임워크 (Scientific Benchmark)
# ----------------------------------------------------
def main():
    print("=== [Framework] 4대 무선 제어 방식 통합 벤치마크 (Fair Comparison) ===")
    
    device = torch.device('cpu')
    try:
        test_loader = get_dataloader('data/real/test.csv', batch_size=1, shuffle=False)
    except Exception as e:
        print(f"[Error] 데이터 로더 연결 실패: {e}")
        return

    # 모델 가중치 로드
    model_lstm = Standard3LayerLSTM()
    checkpoint_path = 'checkpoints/baseline_lstm_best.pth'
    if os.path.exists(checkpoint_path):
        model_lstm.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_lstm.eval()
        print(" -> Baseline ② (LSTM) 모델 가중치 로드 완료.")
    else:
        print(f"[Warning] {checkpoint_path} 가중치 파일 없음. 초기 가중치로 진행합니다.")
        model_lstm.eval()

    # CPU Latency 측정을 위한 Warm-up (예열)
    print(" -> CPU Caches Warm-up 중...")
    dummy_input = torch.randn(1, 10, 4)
    for _ in range(50):
        with torch.no_grad():
            _ = model_lstm(dummy_input)

    summary_results = []
    
    # 명세서 양식과 완벽하게 일치하는 파일명 맵핑
    methods = [
        {'id': 1, 'name': 'Baseline 1 (Threshold)', 'file': 'baseline_threshold.csv'},
        {'id': 2, 'name': 'Baseline 2 (Standard LSTM)', 'file': 'baseline_lstm.csv'},
        {'id': 3, 'name': 'Early Exit Fixed θ', 'file': 'early_exit_fixed.csv'},
        {'id': 4, 'name': 'Early Exit Dynamic θ', 'file': 'early_exit_dynamic.csv'}
    ]

    for method in methods:
        print(f"\n[{method['name']}] 평가 시작...")
        
        # Hysteresis 상태 초기화
        ch_opt._rrm_state = {'last_switch_time': 0, 'current_channel': None}
        
        # 용상 팀원 모델 자체가 수동 로드도 실패했을 때 예외 처리 Skip
        if method['id'] in [3, 4] and not HAS_EARLY_EXIT:
            print(" -> (Skip) 유용상 팀원의 모델 통합 대기 중 (프레임워크 인터페이스 확보 완료)")
            pd.DataFrame(columns=['True_Label', 'Predicted_Label', 'Inference_Time_ms']).to_csv(f"results/{method['file']}", index=False)
            continue

        all_preds, all_labels, inference_times = [], [], []
        unnecessary_switches = 0
        current_channel = 1
        available_channels = [1, 6, 11, 36]

        # Exit별 카운터 변수 초기화 (유용상 담당 지표 추출용)
        exit_counts = {1: 0, 2: 0, 3: 0}

        # 3, 4번용 조기 종료 모델 인스턴스화
        model_ee = None
        if method['id'] in [3, 4] and HAS_EARLY_EXIT:
            model_ee = EarlyExitLSTM(input_size=4, hidden_size=128, num_classes=4)
            model_ee.set_threshold(dynamic=(method['id'] == 4))
            model_ee.eval()

        for step, (features, targets) in enumerate(test_loader):
            features_np = features.numpy()
            true_label = int(targets[0].item())
            
            start_time = time.perf_counter()
            
            # --- 명세서 반영 순수 모델 추론 구간 ---
            if method['id'] == 1:
                occ_val = features_np[0, -1, 1] * 100 
                pred = threshold_baseline(occ_val)
            elif method['id'] == 2:
                with torch.no_grad():
                    output = model_lstm(features)
                    pred = torch.argmax(output, dim=1).item()
            elif method['id'] in [3, 4]:
                with torch.no_grad():
                    # 용상이 순수 코드의 infer_batch 기능 연동
                    decisions = model_ee.infer_batch(features, dynamic=(method['id'] == 4))
                    pred = torch.argmax(decisions[0].logits, dim=-1).item()
                    # 명세서 지표: 유용상 담당 [Exit별 종료율] 추적을 위해 실제 종료 위치 누적
                    exit_counts[decisions[0].exit_point] += 1
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)
            
            all_preds.append(pred)
            all_labels.append(true_label)
            
            # [동적 환경 구현] 채널 상태 모의
            noise_variance = (step % 5) - 2
            dynamic_metrics = {
                1:  {'rssi': -65 + noise_variance, 'noise': -92, 'utilization': 0.7},
                6:  {'rssi': -70 + noise_variance, 'noise': -95, 'utilization': 0.3},
                11: {'rssi': -55, 'noise': -94, 'utilization': 0.2},
                36: {'rssi': -62, 'noise': -98, 'utilization': 0.1}
            }

            # 채널 제어 로직 호출
            next_channel, action = ch_opt.optimize_channel(
                predicted_label=pred,
                current_channel=current_channel,
                available_channels=available_channels,
                metrics=dynamic_metrics,
                current_time=step * 10
            )
            
            # 명세서 지표: 김호중 담당 [불필요 채널 전환 횟수] (정상 구간Label=0에서 제어권이 이탈해 전환한 건수)
            if true_label == 0 and next_channel != current_channel:
                unnecessary_switches += 1
                
            current_channel = next_channel

        # --- 과학적 학술 지표 계산 ---
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        acc = accuracy_score(y_true, y_pred) * 100
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)

        # 김호중 담당 지표: 레이블별(0~3) 각각의 정확도 연산 호출
        label_accs = calculate_label_accuracies(y_true, y_pred)

        # 유용상 담당 지표: Exit별 종료율 (%) 최종 계산
        total_samples = max(len(all_preds), 1)
        exit1_rate = (exit_counts[1] / total_samples) * 100
        exit2_rate = (exit_counts[2] / total_samples) * 100
        exit3_rate = (exit_counts[3] / total_samples) * 100

        # 원래 형태 출력 포맷 유지
        print(f" -> Accuracy: {acc:.2f}% | Avg_Latency: {avg_time:.3f}ms (±{std_time:.3f}ms)")
        print(f" -> Unnecessary Switches: {unnecessary_switches}")
        if method['id'] in [3, 4]:
            print(f" -> Exit Rates: Exit1: {exit1_rate:.1f}% | Exit2: {exit2_rate:.1f}% | Exit3: {exit3_rate:.1f}%")

        # 세부 결과 지정 CSV 파일명으로 저장 완료
        pd.DataFrame({
            'True_Label': y_true, 
            'Predicted_Label': y_pred, 
            'Inference_Time_ms': inference_times
        }).to_csv(f"results/{method['file']}", index=False)

        # 명세서 요약본 통합 저장 양식 생성
        res_dict = {
            'Method': method['name'],
            'Accuracy(%)': round(acc, 2),
            'Avg_Latency(ms)': round(avg_time, 4),
            'Latency_StdDev(ms)': round(std_time, 4),
            'Unnecessary_Switches': unnecessary_switches,
            'Exit1_Rate(%)': round(exit1_rate, 1),
            'Exit2_Rate(%)': round(exit2_rate, 1),
            'Exit3_Rate(%)': round(exit3_rate, 1)
        }
        # 김호중 담당 지표인 레이블별 정확도 데이터프레임 딕셔너리에 병합
        res_dict.update(label_accs)
        summary_results.append(res_dict)

    # 최종 comparison_summary.csv 저장 완료
    pd.DataFrame(summary_results).to_csv('results/comparison_summary.csv', index=False)
    print("\n[성공] 공정 벤치마크 완료 및 results 폴더 CSV 저장 완료!")

if __name__ == "__main__":
    main()
