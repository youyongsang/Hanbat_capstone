"""SDN-style Early Exit LSTM for traffic congestion classification.

Re-implemented based on the official SDN (ICML 2019) confidence-based policy.
- Identical 3-layer LSTM backbone for perfect controlled variable experiments.
- Uses maximum softmax probability (Confidence) instead of predictive entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# SDN weights later exits' loss more heavily to prioritize deep-backbone accuracy.
SDN_LOSS_WEIGHTS: Tuple[float, float, float] = (0.15, 0.30, 0.55)


def confidence_from_logits(logits: Tensor) -> Tensor:
    """Return per-sample maximum softmax confidence from raw class logits."""
    probs = F.softmax(logits, dim=-1)
    return torch.max(probs, dim=-1)[0]


def sdn_multi_exit_loss(
    exit_logits: Sequence[Tensor],
    target: Tensor,
    weights: Sequence[float] = SDN_LOSS_WEIGHTS,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Compute weighted cross-entropy mimicking SDN configurations.

    `class_weights` (per-class, e.g. inverse label frequency) is optional and
    defaults to None (unweighted, matching the original SDN paper) so this
    stays a fair "literal SDN policy" baseline; pass it to compare against
    our class-weighted Early Exit model on equal footing instead.
    """
    if len(exit_logits) != len(weights):
        raise ValueError(f"exit_logits and weights must match: {len(exit_logits)} != {len(weights)}")

    total = exit_logits[0].new_tensor(0.0)
    for logits, weight in zip(exit_logits, weights):
        total = total + float(weight) * F.cross_entropy(logits, target, weight=class_weights)
    return total


@dataclass(frozen=True)
class SDNExitDecision:
    """Inference result for SDN-style inference."""
    logits: Tensor
    exit_point: int
    confidence: Tensor


class SDNLSTM(nn.Module):
    """Three-exit LSTM classifier adapted with SDN's confidence thresholding."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
        confidence_threshold: float = 0.85,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.exit_classifier1 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier2 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, inference: bool = False):
        exit_logits = self._all_exit_logits(x)
        if not inference:
            return exit_logits
        return self._infer_batch_consensus(exit_logits)

    def set_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def infer_batch(self, x: Tensor) -> List[SDNExitDecision]:
        exit_logits = self._all_exit_logits(x)
        confidences = [confidence_from_logits(logits) for logits in exit_logits]
        decisions: List[SDNExitDecision] = []

        for sample_idx in range(x.size(0)):
            if confidences[0][sample_idx].item() >= self.confidence_threshold:
                exit_idx = 0
            elif confidences[1][sample_idx].item() >= self.confidence_threshold:
                exit_idx = 1
            else:
                exit_idx = 2

            decisions.append(
                SDNExitDecision(
                    logits=exit_logits[exit_idx][sample_idx],
                    exit_point=exit_idx + 1,
                    confidence=confidences[exit_idx][sample_idx],
                )
            )
        return decisions

    def infer_batch_stepwise(self, x: Tensor) -> List[SDNExitDecision]:
        """SDN step-by-step inference to evaluate real-world edge latency."""
        decisions: List[SDNExitDecision] = []

        for sample_idx in range(x.size(0)):
            sample = x[sample_idx : sample_idx + 1]

            out1, _ = self.lstm1(sample)
            logits1 = self.exit_classifier1(self.dropout(out1[:, -1, :]))
            conf1 = confidence_from_logits(logits1)[0]
            if conf1.item() >= self.confidence_threshold:
                decisions.append(SDNExitDecision(logits1[0], 1, conf1))
                continue

            out2, _ = self.lstm2(out1)
            logits2 = self.exit_classifier2(self.dropout(out2[:, -1, :]))
            conf2 = confidence_from_logits(logits2)[0]
            if conf2.item() >= self.confidence_threshold:
                decisions.append(SDNExitDecision(logits2[0], 2, conf2))
                continue

            out3, _ = self.lstm3(out2)
            logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))
            conf3 = confidence_from_logits(logits3)[0]
            decisions.append(SDNExitDecision(logits3[0], 3, conf3))

        return decisions

    def _all_exit_logits(self, x: Tensor) -> List[Tensor]:
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"expected shape (batch, timesteps, {self.input_size}), got {tuple(x.shape)}")

        out1, _ = self.lstm1(x)
        logits1 = self.exit_classifier1(self.dropout(out1[:, -1, :]))

        out2, _ = self.lstm2(out1)
        logits2 = self.exit_classifier2(self.dropout(out2[:, -1, :]))

        out3, _ = self.lstm3(out2)
        logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))

        return [logits1, logits2, logits3]

    def _infer_batch_consensus(self, exit_logits: Sequence[Tensor]) -> SDNExitDecision:
        conf1 = confidence_from_logits(exit_logits[0])
        if torch.all(conf1 >= self.confidence_threshold):
            return SDNExitDecision(exit_logits[0], 1, conf1)
        conf2 = confidence_from_logits(exit_logits[1])
        if torch.all(conf2 >= self.confidence_threshold):
            return SDNExitDecision(exit_logits[1], 2, conf2)
        conf3 = confidence_from_logits(exit_logits[2])
        return SDNExitDecision(exit_logits[2], 3, conf3)
