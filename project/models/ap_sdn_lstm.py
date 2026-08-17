"""AP measurement feature variant of the SDN-style Early Exit LSTM.

This module keeps the original SDNLSTM architecture and confidence-based
early-exit policy, but changes the input feature contract from the
first-semester 4-feature simulator format to the summer AP measurement
feature set (see utils/ap_features.py).
"""

from __future__ import annotations

from models.sdn_lstm import SDNLSTM
from utils.ap_features import AP_FEATURE_COLUMNS


class APSDNLSTM(SDNLSTM):
    """SDN-style Early Exit LSTM configured for AP 실측 9-feature windows."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
        confidence_threshold: float = 0.85,
    ) -> None:
        super().__init__(
            input_size=len(AP_FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            confidence_threshold=confidence_threshold,
        )
