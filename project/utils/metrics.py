import torch

def accuracy(y_pred, y_true):
    if y_pred.dim() > 1:
        preds = torch.argmax(y_pred, dim=-1)
    else:
        preds = y_pred
    correct = (preds == y_true).sum().item()
    return correct / len(y_true)

def confusion_matrix(y_pred, y_true, num_classes=4):
    if y_pred.dim() > 1:
        preds = torch.argmax(y_pred, dim=-1)
    else:
        preds = y_pred
        
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)
    for t, p in zip(y_true, preds):
        conf_matrix[t.long(), p.long()] += 1
        
    return conf_matrix

def format_percent(value: float) -> str:
    """소수점 첫째짜리 퍼센트 문자열로 변환 (예: 0.8735 -> 87.4%)"""
    return f"{value * 100:.1f}%"
