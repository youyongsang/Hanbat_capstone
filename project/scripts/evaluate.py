import os
import sys
import torch

sys.path.append(os.getcwd())

from models.baseline_lstm import BaselineLSTM
from utils.dataloader import get_dataloader
from utils.metrics import accuracy, confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloader("data/dummy/test.csv", batch_size=32, shuffle=False)
    
    # 모델 로드
    model = BaselineLSTM().to(device)
    model_path = "checkpoints/baseline_lstm_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu())
            all_trues.append(y_batch)
            
    all_preds = torch.cat(all_preds)
    all_trues = torch.cat(all_trues)
    
    # 메트릭 계산
    test_acc = accuracy(all_preds, all_trues)
    conf_mat = confusion_matrix(all_preds, all_trues)
    
    # 전체 정확도 출력
    print(f"Test Accuracy: {test_acc * 100:.1f}%")
    
    # 각 레이블 이름 설정 (출력 포맷 공백 맞춤)
    labels_info = [
        "Label 0 (정상):     ",
        "Label 1 (혼잡 경고):",
        "Label 2 (혼잡):     ",
        "Label 3 (심각):     "
    ]
    
    # 각 클래스별 Precision, Recall 계산 및 포맷 출력
    for i in range(4):
        tp = conf_mat[i, i].item()
        fp = conf_mat[:, i].sum().item() - tp
        fn = conf_mat[i, :].sum().item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"{labels_info[i]} Precision: {precision:.2f}  Recall: {recall:.2f}")

if __name__ == "__main__":
    main()
