"""Baseline LSTM for traffic congestion classification.

- three single-layer LSTM blocks (matches EarlyExitLSTM backbone exactly)
- one final fully connected layer for classification
- returns standard logits for CrossEntropyLoss
- always runs all three LSTM layers (no early exit)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BaselineLSTM(nn.Module):
    """Standard 3-layer LSTM classifier without Early Exit."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"expected input shape (batch, timesteps, {self.input_size}), "
                f"got {tuple(x.shape)}"
            )

        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(out2)
        last = self.dropout(out3[:, -1, :])
        return self.fc(last)
