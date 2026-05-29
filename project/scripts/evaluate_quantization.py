# project/scripts/evaluate_quantization.py
import os
import time
import csv
import statistics
from pathlib import Path
from typing import Tuple, List

import torch
import torch.quantization
from torch import Tensor, nn
import torch.nn.functional as F

# =====================================================================
# 1. 모델 아키텍처 정의
# =====================================================================
class BaselineLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(out2)
        last_timestep_out = self.dropout(out3[:, -1, :])
        return self.fc(last_timestep_out)


class EarlyExitLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_classes: int = 4, dropout: float = 0.2, theta_1: float = 0.3, theta_2: float = 0.6):
        super().__init__()
        self.input_size = input_size
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.exit_classifier1 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier2 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> List[Tensor]:
        out1, _ = self.lstm1(x)
        logits1 = self.exit_classifier1(self.dropout(out1[:, -1, :]))
        out2, _ = self.lstm2(out1)
        logits2 = self.exit_classifier2(self.dropout(out2[:, -1, :]))
        out3, _ = self.lstm3(out2)
        logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))
        return [logits1, logits2, logits3]

    def infer_stepwise(self, x: Tensor) -> Tuple[Tensor, int]:
        out1, _ = self.lstm1(x)
        logits1 = self.exit_classifier1(self.dropout(out1[:, -1, :]))
        
        log_probs1 = F.log_softmax(logits1, dim=-1)
        probs1 = log_probs1.exp()
        entropy1 = -(probs1 * log_probs1).sum(dim=-1)
        if entropy1[0].item() < self.theta_1:
            return logits1, 1

        out2, _ = self.lstm2(out1)
        logits2 = self.exit_classifier2(self.dropout(out2[:, -1, :]))
        log_probs2 = F.log_softmax(logits2, dim=-1)
        probs2 = log_probs2.exp()
        entropy2 = -(probs2 * log_probs2).sum(dim=-1)
        if entropy2[0].item() < self.theta_2:
            return logits2, 2

        out3, _ = self.lstm3(out2)
        logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))
        return logits3, 3

# =====================================================================
# 2. 헤더 컬럼명을 분석하여 수치 데이터를 동적으로 파싱하는 로더
# =====================================================================
def get_real_test_loader():
    scenario_dir = Path('project/results/scenario_analysis')
    if not scenario_dir.exists():
        scenario_dir = Path('results/scenario_analysis')
        if not scenario_dir.exists():
            raise FileNotFoundError("[❌ 크리티컬 에러] 시나리오 분할 데이터 폴더를 찾을 수 없습니다.")

    all_features = []
    all_labels = []
    
    target_suffixes = ['_gradual.csv', '_spike.csv', '_periodic.csv', '_imbalance.csv']
    scenario_files = [p for p in scenario_dir.glob('scenario_*.csv') if any(p.name.endswith(s) for s in target_suffixes)]
    
    if not scenario_files:
        raise FileNotFoundError("[❌ 크리티컬 에러] 유효한 시나리오 데이터 파일(_gradual, _spike 등)이 존재하지 않습니다.")

    print(f"[🔍 로드 시작] 총 {len(scenario_files)}개의 시나리오 데이터 파일을 정밀 스캔합니다.")

    for file_path in scenario_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            
            header_lower = [col.lower() for col in header]
            feature_indices = []
            label_idx = -1
            
            for idx, col in enumerate(header_lower):
                if 'label' in col or 'class' in col or 'slice_type' in col:
                    label_idx = idx
                    break
            if label_idx == -1:
                label_idx = len(header) - 1

            for idx, col in enumerate(header_lower):
                if idx == label_idx:
                    continue
                if any(k in col for k in ['timestamp', 'scenario', 'name', 'type_string']):
                    continue
                feature_indices.append(idx)
            
            if len(feature_indices) > 4:
                feature_indices = feature_indices[-4:]
            elif len(feature_indices) < 4:
                feature_indices = [i for i in range(len(header)) if i != label_idx][:4]

            current_sequence = []
            for row in reader:
                if not row or len(row) <= max(max(feature_indices), label_idx): 
                    continue
                
                try:
                    label = int(float(row[label_idx]))
                    features = [float(row[i]) for i in feature_indices]
                except ValueError:
                    continue
                
                current_sequence.append(features)
                
                if len(current_sequence) == 10:
                    all_features.append(current_sequence)
                    all_labels.append(label)
                    current_sequence = []

    if not all_features:
        raise ValueError("[❌ 에러] 동적 인덱스 필터링 후에도 유효한 수치 데이터를 추출하지 못했습니다.")

    x_test = torch.tensor(all_features, dtype=torch.float32)
    y_test = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"[📊 로드 완료] 동적 매핑 성공! 총 {x_test.size(0)}개의 시나리오 샘플을 완벽하게 텐서화했습니다.")
    dataset = torch.utils.data.TensorDataset(x_test, y_test)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# =====================================================================
# 3. 측정 유틸리티
# =====================================================================
def evaluate_model(model, data_loader, is_early_exit=False):
    model.eval()
    correct = 0
    total = 0
    inference_times = []

    torch.set_num_threads(1)

    warmup_input = torch.randn(1, 10, 4)
    with torch.no_grad():
        for _ in range(30):
            if is_early_exit:
                _ = model.infer_stepwise(warmup_input)
            else:
                _ = model(warmup_input)

    with torch.no_grad():
        for x, y in data_loader:
            start_time = time.perf_counter_ns()
            
            if is_early_exit:
                logits, _ = model.infer_stepwise(x)
            else:
                logits = model(x)
                
            end_time = time.perf_counter_ns()
            inference_times.append((end_time - start_time) / 1_000_000)

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = (correct / total) * 100
    median_inference_ms = statistics.median(inference_times)
    return accuracy, median_inference_ms

# =====================================================================
# 4. 메인 파이프라인 (체크포인트 딕셔너리 언패킹 로직 추가)
# =====================================================================
def main():
    os.makedirs('project/checkpoints', exist_ok=True)
    os.makedirs('project/results', exist_ok=True)
    
    csv_path = 'project/results/quantization_comparison.csv'
    
    try:
        data_loader = get_real_test_loader()
    except Exception as e:
        print(e)
        return

    targets = {
        "baseline_lstm": {
            "class": BaselineLSTM,
            "orig_path": 'project/checkpoints/baseline_lstm_best.pth',
            "quant_path": 'project/checkpoints/baseline_lstm_quantized.pth',
            "is_ee": False
        },
        "early_exit_fixed": {
            "class": EarlyExitLSTM,
            "orig_path": 'project/checkpoints/early_exit_fixed.pth',
            "quant_path": 'project/checkpoints/early_exit_fixed_quantized.pth',
            "is_ee": True
        }
    }

    results_rows = []

    for model_key, cfg in targets.items():
        print(f"\n>> {model_key} 성능 측정 및 경량화 돌입 (Single Thread Mode)")
        
        if not os.path.exists(cfg["orig_path"]):
            print(f"[❌ 에러] 정식 가중치 파일({cfg['orig_path']})이 없습니다. 경로와 파일명을 다시 확인하세요.")
            return

        # FP32 모델 인스턴스 생성
        model = cfg["class"]()
        
        # [🛠️ 크리티컬 패치] 체크포인트 파일 로드 및 딕셔너리 언패킹 안전장치
        checkpoint = torch.load(cfg["orig_path"], map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 딕셔너리 패키지 구조인 경우 내부의 진짜 가중치 텐서만 바인딩
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 순수 state_dict 형태인 경우 예전 방식 유지
            model.load_state_dict(checkpoint)
            
        orig_size = os.path.getsize(cfg["orig_path"]) / (1024 * 1024)
        orig_acc, orig_time = evaluate_model(model, data_loader, is_early_exit=cfg["is_ee"])

        # Dynamic Quantization 적용
        model.eval()
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.LSTM, torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 가이드라인 규격 준수 저장
        torch.save(model_quantized.state_dict(), cfg["quant_path"])
        
        # INT8 모델 평가
        quant_size = os.path.getsize(cfg["quant_path"]) / (1024 * 1024)
        quant_acc, quant_time = evaluate_model(model_quantized, data_loader, is_early_exit=cfg["is_ee"])

        row = {
            "model": model_key,
            "original_size_mb": round(orig_size, 4),
            "quantized_size_mb": round(quant_size, 4),
            "original_accuracy": round(orig_acc, 2),
            "quantized_accuracy": round(quant_acc, 2),
            "original_inference_ms": round(orig_time, 4),
            "quantized_inference_ms": round(quant_time, 4)
        }
        results_rows.append(row)
        
        print(f"   * 용량 압축: {orig_size:.3f}MB -> {quant_size:.3f}MB")
        print(f"   * 추론 지연(Median): {orig_time:.3f}ms -> {quant_time:.3f}ms")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "original_size_mb", "quantized_size_mb", 
            "original_accuracy", "quantized_accuracy",
            "original_inference_ms", "quantized_inference_ms"
        ])
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
            
    print(f"\n[🎉 최종 완료] 가중치 딕셔너리 해제 및 경량화 벤치마크 완전 성공! -> {csv_path}")

if __name__ == "__main__":
    main()
