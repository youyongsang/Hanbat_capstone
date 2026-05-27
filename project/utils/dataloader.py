import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(path, batch_size=32, shuffle=True):
    df = pd.read_csv(path).sort_values(by=["sample_id", "timestep"]).reset_index(drop=True)
    num_samples = df["sample_id"].nunique()
    
    # 1. 실제 CSV 파일 컬럼명 반영
    feature_cols = ["rps", "channel_occupancy", "packet_loss", "latency"]
    features = df[feature_cols].values
    
    dir_name = os.path.dirname(path)
    scaler_path = os.path.join(dir_name, "scaler_params.json")
    
    # 2. 예나 팀원의 MinMaxScaler(min, max) 구조 반영
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            s = json.load(f)
        mins = np.array([s[col]["min"] for col in feature_cols])
        maxs = np.array([s[col]["max"] for col in feature_cols])
        features_norm = (features - mins) / (maxs - mins + 1e-8)
    else:
        features_norm = (features - np.array([0., 0., 0., 0.])) / (np.array([1000., 100., 30., 500.]) + 1e-8)
        
    X = features_norm.reshape(num_samples, 10, 4)
    y = df.groupby("sample_id")["label"].first().values
    
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=batch_size, shuffle=shuffle)
