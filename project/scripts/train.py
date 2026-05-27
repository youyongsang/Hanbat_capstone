import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# 상대 경로 임포트를 위해 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from models.baseline_lstm import BaselineLSTM
from utils.dataloader import get_dataloader
from utils.metrics import accuracy

def main():
    # 하이퍼파라미터 세팅 (가이드라인 기준)
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = get_dataloader("data/dummy/train.csv", batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader("data/dummy/val.csv", batch_size=batch_size, shuffle=False)
    
    model = BaselineLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_preds, train_trues = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_preds.append(outputs.detach().cpu())
            train_trues.append(y_batch.cpu())
            
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy(torch.cat(train_preds), torch.cat(train_trues))
        
        # 검증 단계
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.append(outputs.cpu())
                val_trues.append(y_batch)
                
        val_acc = accuracy(torch.cat(val_preds), torch.cat(val_trues))
        
        # 가이드라인 지정 포맷으로 출력
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
        
        # 최고 성능 모델 갱신 시 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/baseline_lstm_best.pth")
            
    # 최종 완료 메시지
    print("Best model saved: checkpoints/baseline_lstm_best.pth")

if __name__ == "__main__":
    main()
