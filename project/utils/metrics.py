import torch

def accuracy(y_pred, y_true):
    # 로짓(확률)값이면 가장 높은 클래스로 변환
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
