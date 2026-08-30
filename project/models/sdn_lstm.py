"""SDN (Shallow-Deep Networks, Kaya et al., ICML 2019) adapted to the AP
congestion Early-Exit LSTM as the prior-art comparison model.

The point of this model is the controlled comparison "existing early-exit
method vs ours": the 3-layer LSTM backbone and every training hyper-parameter
are held IDENTICAL to `EarlyExitLSTM`; only the three axes that SDN actually
specifies differ, each implemented per the paper / official code
(github.com/yigitcankaya/Shallow-Deep-Networks):

  1. Internal classifier = a learnable max/avg-pool mix over the sequence,
     then one linear (the paper's IC mixes spatial max/avg pooling for CNNs;
     here the pooling is over timesteps). Our EE uses last-timestep -> linear.
  2. Curriculum-ramped, depth-increasing IC loss weights:
       cur_i = min(max_coeff_i, 0.01 + epoch * (max_coeff_i / total_epochs))
     with max_coeff = (0.15, 0.30) for the two ICs and the FINAL classifier
     always weight 1.0. Our EE uses fixed uniform-ish (0.3, 0.3, 0.4).
  3. Confidence-based exit (max softmax probability >= T), with T calibrated
     on validation (official code searches for it) rather than hard-coded.
     Our EE uses a predictive-entropy threshold, fixed or traffic-adaptive.

The exit-3 head is the base network's original classifier (last timestep ->
linear, i.e. identical to `BaselineLSTM`) — SDN "adds ICs to an existing
network and keeps its final head".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# max IC loss coefficients (official max_coeffs[:2]); final classifier is 1.0.
SDN_MAX_LOSS_COEFFS: Tuple[float, float] = (0.15, 0.30)


def confidence_from_logits(logits: Tensor) -> Tensor:
    """Per-sample maximum softmax probability (SDN's exit signal)."""
    return torch.max(F.softmax(logits, dim=-1), dim=-1)[0]


def sdn_loss_coeffs(
    epoch: int,
    total_epochs: int,
    max_coeffs: Sequence[float] = SDN_MAX_LOSS_COEFFS,
) -> Tuple[float, ...]:
    """SDN curriculum-ramped IC loss coefficients.

    Official: ``cur_coeffs = min(max_coeffs, 0.01 + epoch*(max_coeffs/epochs))``
    — ICs start near-unweighted and ramp to their depth-scaled ceiling.
    """
    denom = max(1, total_epochs)
    return tuple(
        min(mc, 0.01 + epoch * (mc / denom)) for mc in max_coeffs
    )


def sdn_multi_exit_loss(
    exit_logits: Sequence[Tensor],
    target: Tensor,
    ic_coeffs: Sequence[float],
    class_weights: Tensor | None = None,
) -> Tensor:
    """SDN weighted loss: each internal classifier scaled by its (ramped)
    coefficient, the final classifier always at weight 1.0.
    """
    *ic_logits, final_logits = exit_logits
    if len(ic_logits) != len(ic_coeffs):
        raise ValueError(
            f"expected {len(ic_logits)} IC coeffs, got {len(ic_coeffs)}"
        )
    total = F.cross_entropy(final_logits, target, weight=class_weights)
    for logits, coeff in zip(ic_logits, ic_coeffs):
        total = total + float(coeff) * F.cross_entropy(
            logits, target, weight=class_weights
        )
    return total


class SDNInternalClassifier(nn.Module):
    """SDN internal classifier: learnable max/avg pooling mix over the
    sequence dimension, then one linear. Mirrors the official CNN IC
    (`alpha*maxpool + (1-alpha)*avgpool -> Linear`), pooling over timesteps
    instead of spatial dims.
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, seq: Tensor) -> Tensor:  # seq: (batch, timesteps, hidden)
        max_pooled = seq.max(dim=1).values
        avg_pooled = seq.mean(dim=1)
        mixed = self.alpha * max_pooled + (1.0 - self.alpha) * avg_pooled
        return self.fc(self.dropout(mixed))


@dataclass(frozen=True)
class SDNExitDecision:
    """Inference result for SDN-style inference."""
    logits: Tensor
    exit_point: int
    confidence: Tensor


class SDNLSTM(nn.Module):
    """Early-exit LSTM with SDN's internal classifiers, weighted training and
    confidence-threshold exit. Backbone identical to `EarlyExitLSTM`.
    """

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

        # identical backbone to EarlyExitLSTM (controlled variable)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        # exits 1/2 = SDN internal classifiers (pooling heads)
        self.exit_classifier1 = SDNInternalClassifier(hidden_size, num_classes, dropout)
        self.exit_classifier2 = SDNInternalClassifier(hidden_size, num_classes, dropout)
        # exit 3 = base network's original head (last timestep -> linear)
        self.exit_classifier3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, inference: bool = False):
        exit_logits = self._all_exit_logits(x)
        if not inference:
            return exit_logits
        return self._infer_batch_consensus(exit_logits)

    def set_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

    def _all_exit_logits(self, x: Tensor) -> List[Tensor]:
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"expected shape (batch, timesteps, {self.input_size}), got {tuple(x.shape)}"
            )
        out1, _ = self.lstm1(x)
        logits1 = self.exit_classifier1(out1)

        out2, _ = self.lstm2(out1)
        logits2 = self.exit_classifier2(out2)

        out3, _ = self.lstm3(out2)
        logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))

        return [logits1, logits2, logits3]

    def infer_batch(self, x: Tensor) -> List[SDNExitDecision]:
        exit_logits = self._all_exit_logits(x)
        confidences = [confidence_from_logits(logits) for logits in exit_logits]
        decisions: List[SDNExitDecision] = []
        for i in range(x.size(0)):
            if confidences[0][i].item() >= self.confidence_threshold:
                e = 0
            elif confidences[1][i].item() >= self.confidence_threshold:
                e = 1
            else:
                e = 2
            decisions.append(
                SDNExitDecision(exit_logits[e][i], e + 1, confidences[e][i])
            )
        return decisions

    def infer_batch_stepwise(self, x: Tensor) -> List[SDNExitDecision]:
        """Step-by-step inference (stops computing deeper layers once an exit
        fires) — used for edge-latency evaluation."""
        decisions: List[SDNExitDecision] = []
        for i in range(x.size(0)):
            sample = x[i : i + 1]

            out1, _ = self.lstm1(sample)
            logits1 = self.exit_classifier1(out1)
            conf1 = confidence_from_logits(logits1)[0]
            if conf1.item() >= self.confidence_threshold:
                decisions.append(SDNExitDecision(logits1[0], 1, conf1))
                continue

            out2, _ = self.lstm2(out1)
            logits2 = self.exit_classifier2(out2)
            conf2 = confidence_from_logits(logits2)[0]
            if conf2.item() >= self.confidence_threshold:
                decisions.append(SDNExitDecision(logits2[0], 2, conf2))
                continue

            out3, _ = self.lstm3(out2)
            logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))
            conf3 = confidence_from_logits(logits3)[0]
            decisions.append(SDNExitDecision(logits3[0], 3, conf3))
        return decisions

    def _infer_batch_consensus(self, exit_logits: Sequence[Tensor]) -> SDNExitDecision:
        conf1 = confidence_from_logits(exit_logits[0])
        if torch.all(conf1 >= self.confidence_threshold):
            return SDNExitDecision(exit_logits[0], 1, conf1)
        conf2 = confidence_from_logits(exit_logits[1])
        if torch.all(conf2 >= self.confidence_threshold):
            return SDNExitDecision(exit_logits[1], 2, conf2)
        conf3 = confidence_from_logits(exit_logits[2])
        return SDNExitDecision(exit_logits[2], 3, conf3)


@torch.no_grad()
def calibrate_confidence_threshold(
    exit_logits: Sequence[Tensor],
    targets: Tensor,
    tolerance: float = 0.01,
    grid: Sequence[float] | None = None,
) -> Tuple[float, dict]:
    """Search the confidence threshold on a validation set, per the SDN
    official `early_exit_experiments.py`. Returns the smallest T whose
    early-exit accuracy stays within `tolerance` of the full-network
    (final-classifier-only) accuracy — i.e. cheapest exit budget that keeps
    accuracy. Falls back to the T with best accuracy if none qualifies.
    """
    if grid is None:
        # floor at 0.50: a max-softmax probability below 0.5 (4 classes) is not
        # a meaningful "confident" prediction to exit on.
        grid = [round(0.50 + 0.01 * k, 2) for k in range(50)]  # 0.50 .. 0.99

    confidences = [confidence_from_logits(lg) for lg in exit_logits]
    preds = [lg.argmax(dim=-1) for lg in exit_logits]
    n = targets.size(0)
    full_acc = (preds[-1] == targets).float().mean().item()

    best_t, best_acc, best_cost = grid[0], -1.0, 4.0
    qualifying: List[Tuple[float, float, float]] = []
    for t in grid:
        exit_idx = torch.full((n,), len(exit_logits) - 1, dtype=torch.long)
        chosen = torch.zeros(n, dtype=torch.bool)
        for e in range(len(exit_logits) - 1):
            fire = (confidences[e] >= t) & (~chosen)
            exit_idx[fire] = e
            chosen |= fire
        picked = torch.stack(
            [preds[exit_idx[i]][i] for i in range(n)]
        )
        acc = (picked == targets).float().mean().item()
        cost = (exit_idx.float() + 1).mean().item()
        if acc > best_acc:
            best_acc, best_t, best_cost = acc, t, cost
        if acc >= full_acc - tolerance:
            qualifying.append((t, acc, cost))

    if qualifying:
        # cheapest exit budget among thresholds that keep accuracy
        t, acc, cost = min(qualifying, key=lambda r: (r[2], r[0]))
        return t, {"val_acc": acc, "avg_exit": cost, "full_acc": full_acc, "qualified": True}
    return best_t, {"val_acc": best_acc, "avg_exit": best_cost, "full_acc": full_acc, "qualified": False}
