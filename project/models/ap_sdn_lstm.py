"""AP measurement feature variant of the SDN comparison model.

Keeps SDNLSTM's paper-faithful IC / weighted-loss / confidence-exit design
(see models/sdn_lstm.py), bound to this branch's 7-feature AP measurement
contract (utils/ap_features.py).
"""

from __future__ import annotations

from models.sdn_lstm import SDNLSTM
from utils.ap_features import AP_FEATURE_COLUMNS


class APSDNLSTM(SDNLSTM):
    """SDN comparison model configured for AP 실측 7-feature windows."""

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
