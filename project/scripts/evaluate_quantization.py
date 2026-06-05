import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 1. 프로젝트 루트 경로 및 모듈 동기화
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloader import get_dataloader
from models.baseline_lstm import BaselineLSTM
from models.early_exit_lstm import EarlyExitLSTM

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMING_REPEATS = 5
WARMUP_SAMPLES = 10


def get_file_size_mb(path: Path) -> float:
    """물리적 파일 크기 측정"""
    return round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else 0.0


def load_model_info() -> dict:
    info_path = CHECKPOINT_DIR / "model_info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_final_threshold(model: EarlyExitLSTM, cfg: dict, model_info: dict) -> tuple[float, float]:
    """Apply the final Stage 3 threshold used by the main comparison."""

    if cfg["dynamic_theta"]:
        theta_1 = float(model_info.get("dynamic_base_theta_1", 0.3))
        theta_2 = float(model_info.get("dynamic_base_theta_2", 0.6))
    else:
        theta_1 = float(model_info.get("fixed_theta_1", 0.3))
        theta_2 = float(model_info.get("fixed_theta_2", 0.6))

    model.set_threshold(theta_1=theta_1, theta_2=theta_2, dynamic=cfg["dynamic_theta"])
    return theta_1, theta_2


def predict_one(model, features, cfg: dict) -> int:
    if cfg["type"] == "lstm":
        return torch.argmax(model(features), dim=1).item()
    decisions = model.infer_batch_stepwise(features, dynamic=cfg["dynamic_theta"])
    return torch.argmax(decisions[0].logits, dim=-1).item()


def evaluate_model(model, test_loader, cfg: dict) -> tuple[list[int], np.ndarray]:
    """Return predictions from the first pass and timing samples from repeated passes."""

    predictions: list[int] = []
    timing_runs: list[float] = []

    with torch.no_grad():
        for idx, (features, _) in enumerate(test_loader):
            if idx >= WARMUP_SAMPLES:
                break
            predict_one(model, features, cfg)

        for run_idx in range(TIMING_REPEATS):
            run_times: list[float] = []
            run_predictions: list[int] = []
            for features, _ in test_loader:
                start_time = time.perf_counter()
                pred = predict_one(model, features, cfg)
                run_times.append((time.perf_counter() - start_time) * 1000)
                run_predictions.append(pred)

            timing_runs.append(float(np.mean(run_times)))
            if run_idx == 0:
                predictions = run_predictions

    return predictions, np.array(timing_runs, dtype=np.float64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    model_info = load_model_info()

    try:
        test_loader = get_dataloader(PROJECT_ROOT / "data" / "real" / "test.csv", batch_size=1, shuffle=False)
    except Exception as e:
        print(f"❌ 데이터 로더 연결 실패: {e}")
        return

    # 4-Method Comparison Report 기반 실험 대상 시나리오 구성
    configs = [
        {
            "model": "baseline_lstm", 
            "type": "lstm", 
            "dynamic_theta": False, 
            "file": "baseline_lstm_best.pth"
        },
        {
            "model": "early_exit_fixed", 
            "type": "ee", 
            "dynamic_theta": False, 
            "file": "early_exit_fixed.pth"
        },
        {
            "model": "early_exit_dynamic", 
            "type": "ee", 
            "dynamic_theta": True, 
            "file": "early_exit_dynamic.pth"
        }
    ]

    quantization_results = []

    for cfg in configs:
        ckpt_path = CHECKPOINT_DIR / cfg["file"]
        if not ckpt_path.exists():
            print(f"⚠️ 파일 누락: {ckpt_path}")
            continue

        # --- [Step 1] 원본 FP32 모델 로드 ---
        if cfg["type"] == "lstm":
            model_fp32 = BaselineLSTM()
        else:
            model_fp32 = EarlyExitLSTM(input_size=4, hidden_size=128, num_classes=4)
        
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model_fp32.load_state_dict(state_dict)

        theta_1 = ""
        theta_2 = ""
        if cfg["type"] == "ee":
            theta_1, theta_2 = apply_final_threshold(model_fp32, cfg, model_info)
        
        model_fp32.eval()
        orig_size_mb = get_file_size_mb(ckpt_path)

        # --- [Step 2] PyTorch 내장 Dynamic Quantization 적용 (INT8 변환) ---
        # FP32와 INT8 모두 같은 데이터, checkpoint, threshold 조건으로 평가합니다.
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {torch.nn.LSTM, torch.nn.Linear},
            dtype=torch.qint8
        )
        model_int8.eval()

        # 양자화 모델 로컬 디스크 저장 및 용량 측정
        quant_ckpt_path = CHECKPOINT_DIR / f"{cfg['model']}_quantized.pth"
        torch.save(model_int8.state_dict(), quant_ckpt_path)
        quant_size_mb = get_file_size_mb(quant_ckpt_path)

        # --- [Step 3] 같은 조건에서 반복 측정 ---
        fp32_preds, fp32_times = evaluate_model(model_fp32, test_loader, cfg)
        int8_preds, int8_times = evaluate_model(model_int8, test_loader, cfg)
        all_labels = [
            int(targets[0].item())
            for _, targets in test_loader
        ]

        # --- [Step 4] 정직한 지표 정산 ---
        y_true = np.array(all_labels)
        orig_acc = round(float((y_true == np.array(fp32_preds)).mean()) * 100, 2)
        quant_acc = round(float((y_true == np.array(int8_preds)).mean()) * 100, 2)
        
        orig_time_ms = round(float(np.mean(fp32_times)), 4)
        quant_time_ms = round(float(np.mean(int8_times)), 4)
        orig_time_std_ms = round(float(np.std(fp32_times)), 4)
        quant_time_std_ms = round(float(np.std(int8_times)), 4)

        quantization_results.append({
            "model": cfg["model"],
            "checkpoint": cfg["file"],
            "dynamic_theta": cfg["dynamic_theta"],
            "theta_1": theta_1,
            "theta_2": theta_2,
            "original_size_mb": orig_size_mb,
            "quantized_size_mb": quant_size_mb,
            "original_accuracy": orig_acc,
            "quantized_accuracy": quant_acc,
            "original_inference_ms": orig_time_ms,
            "quantized_inference_ms": quant_time_ms,
            "original_inference_std_ms": orig_time_std_ms,
            "quantized_inference_std_ms": quant_time_std_ms,
            "timing_repeats": TIMING_REPEATS
        })

    # 가이드라인 요구 포맷 컬럼으로 빌드 후 CSV 추출
    df = pd.DataFrame(quantization_results)
    columns_order = [
        "model", "checkpoint", "dynamic_theta", "theta_1", "theta_2",
        "original_size_mb", "quantized_size_mb", 
        "original_accuracy", "quantized_accuracy", 
        "original_inference_ms", "quantized_inference_ms",
        "original_inference_std_ms", "quantized_inference_std_ms",
        "timing_repeats"
    ]
    df = df[columns_order]
    
    csv_save_path = RESULTS_DIR / "quantization_comparison.csv"
    df.to_csv(csv_save_path, index=False)
    print(f"평가 완료: 최종 theta 기준 실측 완료 및 CSV 저장 성공: {csv_save_path}")


if __name__ == "__main__":
    main()
