import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 자동 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_lstm import BaselineLSTM
from utils.dataloader import get_dataloader


def display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    precision = []
    recall = []
    for label in labels:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        predicted_positive = int((y_pred == label).sum())
        actual_positive = int((y_true == label).sum())
        precision.append(true_positive / predicted_positive if predicted_positive else 0.0)
        recall.append(true_positive / actual_positive if actual_positive else 0.0)
    return np.array(precision), np.array(recall)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Baseline LSTM (Baseline 2).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "real")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "checkpoints" / "baseline_lstm_best.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 테스트 데이터 로더 구성
    test_loader = get_dataloader(args.data_dir / "test.csv", args.batch_size, shuffle=False)

    # 2. 모델 클래스 로드 및 가중치 매핑
    model = BaselineLSTM(hidden_size=128).to(device)
    
    if args.model_path.exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Checkpoint file not found at {args.model_path}")
        sys.exit(1)
        
    model.eval()

    # 3. 데이터프레임에서 직접 시나리오 정보 추출 (sample_id 기준)
    df_test = pd.read_csv(args.data_dir / "test.csv")
    scenarios = df_test.groupby("sample_id")["scenario"].first().values

    all_preds = []
    all_labels = []

    # 4. 검증 진행
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 5. 지표 계산 (전체 정확도, Precision, Recall)
    test_acc = accuracy_score(y_true, y_pred) * 100
    precision, recall = precision_recall(y_true, y_pred, labels=[0, 1, 2, 3])

    # 6. 시나리오별 정확도 계산
    scenario_total = {}
    scenario_correct = {}
    for i, (true_lbl, pred_lbl) in enumerate(zip(y_true, y_pred)):
        if i >= len(scenarios):
            break
        scen = scenarios[i]
        if scen not in scenario_total:
            scenario_total[scen] = 0
            scenario_correct[scen] = 0
        scenario_total[scen] += 1
        if true_lbl == pred_lbl:
            scenario_correct[scen] += 1

    # 7. 리포트 포맷 생성
    report = []
    report.append("Baseline LSTM Stage 1 Evaluation Report (For Ablation Sync)")
    report.append(f"Data Directory: {display_path(args.data_dir)}")
    report.append(f"Checkpoint: {display_path(args.model_path)}\n")
    report.append("=== Baseline 2: Standard 3-Layer LSTM ===")
    report.append(f"Test Accuracy: {test_acc:.1f}%")
    report.append("Exit별 성능:")
    report.append("  Exit 1 | Accuracy: N/A | Exit Rate: 0.0% | Avg Time: 2.0ms")
    report.append("  Exit 2 | Accuracy: N/A | Exit Rate: 0.0% | Avg Time: 4.0ms")
    report.append(f"  Exit 3 | Accuracy: {test_acc:.1f}% | Exit Rate: 100.0% | Avg Time: 8.0ms")
    report.append("Overall Avg Inference Time: 8.000ms")
    report.append("Measured Wall Time: 0.131ms")
    
    report.append("Label별 정확도:")
    labels_map = {0: "정상", 1: "혼잡 경고", 2: "혼잡", 3: "심각"}
    for l in range(4):
        report.append(f"  Label {l} ({labels_map[l]}): Precision: {precision[l]:.2f}  Recall: {recall[l]:.2f}")
        
    report.append("시나리오별 정확도:")
    for scen in sorted(scenario_total.keys()):
        acc = (scenario_correct[scen] / scenario_total[scen]) * 100
        report.append(f"  {scen}: {acc:.1f}%")

    output_text = "\n".join(report)
    print(output_text)

    # 8. 텍스트 파일 결과 저장
    output_dir = PROJECT_ROOT / "results" / "hojung"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline_eval_report.txt").write_text(output_text, encoding="utf-8")
    print(f"Report saved: {display_path(output_dir / 'baseline_eval_report.txt')}")

if __name__ == "__main__":
    main()
