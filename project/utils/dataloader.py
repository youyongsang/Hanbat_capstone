import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(path, batch_size=32, shuffle=True):
    # CSV 읽기 및 정렬 (시계열 순서 보장)
    df = pd.read_csv(path)
    df = df.sort_values(by=["sample_id", "timestep"]).reset_index(drop=True)
    num_samples = df["sample_id"].nunique()
    
    # 4가지 피처 추출
    features = df[["rps", "occupancy", "loss_rate", "latency"]].values
    
    # 더미 데이터 명세서 기준 Min-Max 정규화 적용
    X_MIN = np.array([0.0, 0.0, 0.0, 0.0])
    X_MAX = np.array([1000.0, 100.0, 30.0, 500.0])
    features_norm = (features - X_MIN) / (X_MAX - X_MIN + 1e-8)
    
    # (샘플 수, 10, 4) 형태로 Reshape
    X = features_norm.reshape(num_samples, 10, 4)
    # 레이블 추출 (윈도우당 1개의 레이블)
    y = df.groupby("sample_id")["label"].first().values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
