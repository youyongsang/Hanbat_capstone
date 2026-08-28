"""AP measurement feature variant of the SDN-style Early Exit LSTM.

Keeps the original SDNLSTM architecture and confidence-based early-exit
policy, but uses this branch's 6-feature AP measurement contract (see
utils/ap_features.py) instead of the 1st-semester 4/9-feature formats.
"""

from __future__ import annotations

from models.sdn_lstm import SDNLSTM
from utils.ap_features import AP_FEATURE_COLUMNS


class APSDNLSTM(SDNLSTM):
    """SDN-style Early Exit LSTM configured for AP 실측 6-feature windows."""

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
