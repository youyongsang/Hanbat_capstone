import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(path, batch_size=32, shuffle=True):
    # 1. 실제 CSV 데이터 읽기 및 정렬
    df = pd.read_csv(path)
    df = df.sort_values(by=["sample_id", "timestep"]).reset_index(drop=True)
    num_samples = df["sample_id"].nunique()
    
    # 4가지 핵심 피처 추출
    feature_cols = ["rps", "occupancy", "loss_rate", "latency"]
    features = df[feature_cols].values
    
    dir_name = os.path.dirname(path)  # data/real/
    scaler_path = os.path.join(dir_name, "scaler_params.json")
    
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            scaler_params = json.load(f)
        
        # 피처 순서에 맞게 mean과 scale(std 또는 max-min) 추출
        means = np.array([scaler_params[col]["mean"] for col in feature_cols])
        scales = np.array([scaler_params[col]["scale"] for col in feature_cols])
        
        # 정규화 연산 (StandardScaler 또는 RobustScaler 공식 대응)
        features_norm = (features - means) / (scales + 1e-8)
    else:
        # 안전장치: 만약 json 파일이 없으면 기본 Min-Max 스케일링 수행
        X_MIN = np.array([0.0, 0.0, 0.0, 0.0])
        X_MAX = np.array([1000.0, 100.0, 30.0, 500.0])
        features_norm = (features - X_MIN) / (X_MAX - X_MIN + 1e-8)
    
    # 3. 3차원 (샘플 수, 타임스텝=10, 피처 수=4) 형태로 변환
    X = features_norm.reshape(num_samples, 10, 4)
    y = df.groupby("sample_id")["label"].first().values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
