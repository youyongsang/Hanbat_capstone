"""Baseline LSTM for traffic congestion classification.

- input shape: (batch, 10, 4)
- three single-layer LSTM blocks (matches EarlyExitLSTM backbone perfectly)
- one final fully connected layer for classification
- returns standard logits for CrossEntropyLoss
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

        # 단일 LSTM 레이어 3개를 선언. (파라미터 수 및 연산 구조 100% 일치)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        
        # Early Exit은 각 단계마다 Classifier가 있지만, 
        # 베이스라인은 마지막에 단 1개의 Classifier만 존재합니다.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the model.
        
        Args:
            x: Input tensor of shape (batch, 10, 4)
            
        Returns:
            logits: Output tensor of shape (batch, 4)
        """
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"expected input shape (batch, timesteps, {self.input_size}), "
                f"got {tuple(x.shape)}"
            )

        # 1. 3개의 LSTM 레이어를 순차적으로 모두 통과 (중간 종료 없음)
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(out2)

        # 2. 마지막 레이어(lstm3)의 마지막 타임스텝(-1) 데이터만 추출하여 Dropout 적용
        last_timestep_out = self.dropout(out3[:, -1, :])
        
        # 3. FC 레이어를 통과하여 4개 클래스에 대한 로짓(Logits) 반환
        logits = self.fc(last_timestep_out)

        return logits
