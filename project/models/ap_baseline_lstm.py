"""AP measurement feature variant of the Baseline LSTM (no early exit)."""

from __future__ import annotations

from models.baseline_lstm import BaselineLSTM
from utils.ap_features import AP_FEATURE_COLUMNS


class APBaselineLSTM(BaselineLSTM):
    """Baseline LSTM configured for AP 실측 6-feature windows."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_size=len(AP_FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )
