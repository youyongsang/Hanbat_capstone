import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# project 폴더를 path에 추가하여 utils 호출이 가능하게 설정
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    # 가이드라인 7번에 맞춤: 기본 경로는 dummy로 두되 필요시 real로 변경 가능
    parser.add_argument("--data_dir", type=str, default="data/dummy")
    parser.add_argument("--model_path", type=str, default="checkpoints/baseline_lstm_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.model_path = Path(args.model_path)

    # 1. 데이터 로더 불러오기
    test_loader = get_dataloader(args.data_dir / "test.csv", args.batch_size, shuffle=False)

    # 2. 가이드라인 3번 & 5번 모델 스펙 완전 정석 이식
    import torch.nn as nn
    class BaselineLSTM(nn.Module):
        def __init__(self, input_size=4, hidden_size=128, num_layers=3, num_classes=4):
            super(BaselineLSTM, self).__init__()
            # 3개 층, dropout 0.2 스펙 일치
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            # 가이드라인 5번 명세 준수: 오직 (batch, 4) 클래스 확률만 반환 (Exit 고정값 반환 삭제)
            return self.fc(out[:, -1, :])

    device = torch.device("cpu")
    model = BaselineLSTM()
    
    # [안전 장치] 조원들 파일 포맷이 달라도 다 읽어오도록 언패킹 내장
    if args.model_path.exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"[Error] 모델 가중치 파일이 없습니다: {args.model_path}")
        return

    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    # 3. 추론 시작
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)  # (batch, 4) 순수 추론
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 4. 평가지표 연산 (Precision, Recall 추출)
    test_acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)

    # 5. 가이드라인 5번 출력 예시 포맷과 100% 일치화
    print(f"Test Accuracy: {test_acc:.1f}%")
    labels_map = {0: "정상", 1: "혼잡 경고", 2: "혼잡", 3: "심각"}
    
    for l in range(4):
        print(f"Label {l} ({labels_map[l]}): Precision: {precision[l]:.2f}  Recall: {recall[l]:.2f}")

    # [선택 사항] 통합 결과를 텍스트 파일로 자동 백업
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    report_lines = [f"Test Accuracy: {test_acc:.1f}%"]
    for l in range(4):
        report_lines.append(f"Label {l} ({labels_map[l]}): Precision: {precision[l]:.2f}  Recall: {recall[l]:.2f}")
    
    (output_dir / "baseline_eval_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

if __name__ == "__main__":
    main()
