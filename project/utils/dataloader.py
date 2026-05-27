import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(path, batch_size=32, shuffle=True):
    df = pd.read_csv(path)
    df = df.sort_values(by=["sample_id", "timestep"]).reset_index(drop=True)
    num_samples = df["sample_id"].nunique()
    
    feature_cols = ["rps", "occupancy", "loss_rate", "latency"]
    features = df[feature_cols].values
    
    dir_name = os.path.dirname(path)
    scaler_path = os.path.join(dir_name, "scaler_params.json")
    
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            scaler_params = json.load(f)
        means = np.array([scaler_params[col]["mean"] for col in feature_cols])
        scales = np.array([scaler_params[col]["scale"] for col in feature_cols])
        features_norm = (features - means) / (scales + 1e-8)
    else:
        X_MIN = np.array([0.0, 0.0, 0.0, 0.0])
        X_MAX = np.array([1000.0, 100.0, 30.0, 500.0])
        features_norm = (features - X_MIN) / (X_MAX - X_MIN + 1e-8)
    
    X = features_norm.reshape(num_samples, 10, 4)
    y = df.groupby("sample_id")["label"].first().values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
