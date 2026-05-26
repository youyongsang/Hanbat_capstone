"""Baseline LSTM classifier for traffic congestion labels."""

from __future__ import annotations

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_timestep = output[:, -1, :]
        return self.classifier(last_timestep)
