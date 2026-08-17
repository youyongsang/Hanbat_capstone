"""AP measurement feature variant of the Early Exit LSTM.

This module keeps the original EarlyExitLSTM architecture but changes the
input feature contract from the first-semester 4-feature simulator format to
the summer AP measurement feature set.
"""

from __future__ import annotations

from models.early_exit_lstm import EarlyExitLSTM
from utils.ap_features import AP_FEATURE_COLUMNS


class APEarlyExitLSTM(EarlyExitLSTM):
    """Early Exit LSTM configured for AP 실측 9-feature windows."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
        theta_1: float = 0.3,
        theta_2: float = 0.6,
    ) -> None:
        super().__init__(
            input_size=len(AP_FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            theta_1=theta_1,
            theta_2=theta_2,
        )
