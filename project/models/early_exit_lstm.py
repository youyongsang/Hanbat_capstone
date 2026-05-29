"""Early Exit LSTM for traffic congestion classification.

The model follows the Stage 1 Yongsang guideline:
- input shape: (batch, 10, 4)
- three single-layer LSTM blocks with one classifier after each block
- training mode returns all exit logits for multi-exit loss
- inference mode returns the first confident exit based on entropy thresholds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_LOSS_WEIGHTS: Tuple[float, float, float] = (0.3, 0.3, 0.4)


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Return per-sample predictive entropy from raw class logits."""

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1)


def multi_exit_loss(
    exit_logits: Sequence[Tensor],
    target: Tensor,
    weights: Sequence[float] = DEFAULT_LOSS_WEIGHTS,
) -> Tensor:
    """Compute weighted cross-entropy across all exit classifiers."""

    if len(exit_logits) != len(weights):
        raise ValueError(
            f"exit_logits and weights must have the same length: "
            f"{len(exit_logits)} != {len(weights)}"
        )

    total = exit_logits[0].new_tensor(0.0)
    for logits, weight in zip(exit_logits, weights):
        total = total + float(weight) * F.cross_entropy(logits, target)
    return total


@dataclass(frozen=True)
class ExitDecision:
    """Inference result for a batch or a single sample."""

    logits: Tensor
    exit_point: int
    entropy: Tensor


class EarlyExitLSTM(nn.Module):
    """Three-exit LSTM classifier for congestion labels 0~3."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
        theta_1: float = 0.3,
        theta_2: float = 0.6,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.dynamic_threshold = False

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.exit_classifier1 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier2 = nn.Linear(hidden_size, num_classes)
        self.exit_classifier3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, inference: bool = False):
        """Run the model.

        Training mode returns a list of three exit logits.
        Inference mode returns an ExitDecision for one sample or batched consensus.
        For per-sample early exits in a batch, use infer_batch().
        """

        exit_logits = self._all_exit_logits(x)
        if not inference:
            return exit_logits
        return self._infer_batch_consensus(exit_logits)

    def set_threshold(
        self,
        theta_1: float | None = None,
        theta_2: float | None = None,
        dynamic: bool | None = None,
    ) -> None:
        """Update fixed thresholds and dynamic-threshold mode."""

        if theta_1 is not None:
            self.theta_1 = theta_1
        if theta_2 is not None:
            self.theta_2 = theta_2
        if dynamic is not None:
            self.dynamic_threshold = dynamic

    def infer_batch(self, x: Tensor, dynamic: bool | None = None) -> List[ExitDecision]:
        """Return an independent early-exit decision for each sample in a batch."""

        exit_logits = self._all_exit_logits(x)
        entropies = [entropy_from_logits(logits) for logits in exit_logits]
        decisions: List[ExitDecision] = []
        use_dynamic = self.dynamic_threshold if dynamic is None else dynamic

        for sample_idx in range(x.size(0)):
            theta_1, theta_2 = self.theta_1, self.theta_2
            if use_dynamic:
                theta_1, theta_2 = compute_dynamic_threshold(
                    x[sample_idx, :, 1],
                    base_theta_1=self.theta_1,
                    base_theta_2=self.theta_2,
                )

            if entropies[0][sample_idx].item() < theta_1:
                exit_idx = 0
            elif entropies[1][sample_idx].item() < theta_2:
                exit_idx = 1
            else:
                exit_idx = 2

            decisions.append(
                ExitDecision(
                    logits=exit_logits[exit_idx][sample_idx],
                    exit_point=exit_idx + 1,
                    entropy=entropies[exit_idx][sample_idx],
                )
            )

        return decisions

    def infer_batch_stepwise(self, x: Tensor, dynamic: bool | None = None) -> List[ExitDecision]:
        """Return early-exit decisions without computing deeper exits after a stop."""

        decisions: List[ExitDecision] = []
        use_dynamic = self.dynamic_threshold if dynamic is None else dynamic

        for sample_idx in range(x.size(0)):
            sample = x[sample_idx : sample_idx + 1]
            theta_1, theta_2 = self.theta_1, self.theta_2
            if use_dynamic:
                theta_1, theta_2 = compute_dynamic_threshold(
                    sample[0, :, 1],
                    base_theta_1=self.theta_1,
                    base_theta_2=self.theta_2,
                )

            out1, _ = self.lstm1(sample)
            logits1 = self.exit_classifier1(self.dropout(out1[:, -1, :]))
            entropy1 = entropy_from_logits(logits1)[0]
            if entropy1.item() < theta_1:
                decisions.append(ExitDecision(logits1[0], 1, entropy1))
                continue

            out2, _ = self.lstm2(out1)
            logits2 = self.exit_classifier2(self.dropout(out2[:, -1, :]))
            entropy2 = entropy_from_logits(logits2)[0]
            if entropy2.item() < theta_2:
                decisions.append(ExitDecision(logits2[0], 2, entropy2))
                continue

            out3, _ = self.lstm3(out2)
            logits3 = self.exit_classifier3(self.dropout(out3[:, -1, :]))
            entropy3 = entropy_from_logits(logits3)[0]
            decisions.append(ExitDecision(logits3[0], 3, entropy3))

        return decisions

    def exit_rate(self, x: Tensor, dynamic: bool | None = None) -> Dict[int, float]:
        """Measure the fraction of samples exiting at each point."""

        decisions = self.infer_batch(x, dynamic=dynamic)
        total = max(len(decisions), 1)
        return {
            exit_point: sum(d.exit_point == exit_point for d in decisions) / total
            for exit_point in (1, 2, 3)
        }

    def _all_exit_logits(self, x: Tensor) -> List[Tensor]:
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"expected input shape (batch, timesteps, {self.input_size}), "
                f"got {tuple(x.shape)}"
            )

        out1, _ = self.lstm1(x)
        last1 = self.dropout(out1[:, -1, :])
        logits1 = self.exit_classifier1(last1)

        out2, _ = self.lstm2(out1)
        last2 = self.dropout(out2[:, -1, :])
        logits2 = self.exit_classifier2(last2)

        out3, _ = self.lstm3(out2)
        last3 = self.dropout(out3[:, -1, :])
        logits3 = self.exit_classifier3(last3)

        return [logits1, logits2, logits3]

    def _infer_batch_consensus(self, exit_logits: Sequence[Tensor]) -> ExitDecision:
        """Return a single decision for simple demos and one-sample inference."""

        entropy1 = entropy_from_logits(exit_logits[0])
        if torch.all(entropy1 < self.theta_1):
            return ExitDecision(exit_logits[0], 1, entropy1)

        entropy2 = entropy_from_logits(exit_logits[1])
        if torch.all(entropy2 < self.theta_2):
            return ExitDecision(exit_logits[1], 2, entropy2)

        entropy3 = entropy_from_logits(exit_logits[2])
        return ExitDecision(exit_logits[2], 3, entropy3)


def exit_accuracy_by_point(
    decisions: Iterable[ExitDecision],
    targets: Tensor,
) -> Dict[int, float]:
    """Compute accuracy grouped by selected exit point."""

    correct_by_exit = {1: 0, 2: 0, 3: 0}
    count_by_exit = {1: 0, 2: 0, 3: 0}

    for idx, decision in enumerate(decisions):
        pred = int(decision.logits.argmax(dim=-1).item())
        target = int(targets[idx].item())
        count_by_exit[decision.exit_point] += 1
        correct_by_exit[decision.exit_point] += int(pred == target)

    return {
        exit_point: (
            correct_by_exit[exit_point] / count_by_exit[exit_point]
            if count_by_exit[exit_point]
            else 0.0
        )
        for exit_point in (1, 2, 3)
    }


def compute_dynamic_threshold(
    recent_window: Tensor,
    base_theta_1: float = 0.3,
    base_theta_2: float = 0.6,
    high_variance: float = 0.22,
    mid_variance: float = 0.12,
    min_threshold: float = 0.22,
    recent_steps: int = 5,
    spike_threshold: float = 0.25,
) -> tuple[float, float]:
    """Adjust thresholds from recent channel occupancy variation.

    The dataset uses normalized channel occupancy, so variance and spike
    thresholds are represented on a 0~1 scale.
    """

    occupancy = recent_window[-recent_steps:].detach().float()
    if occupancy.numel() < 2:
        return base_theta_1, base_theta_2

    delta = torch.abs(occupancy[-1] - occupancy[-2]).item()
    if delta > spike_threshold:
        theta_1 = base_theta_1 * 1.0
        theta_2 = base_theta_2 * 1.0
    else:
        theta_1 = base_theta_1 * 1.25
        theta_2 = base_theta_2 * 1.25

    theta_1 = max(theta_1, min_threshold)
    theta_2 = max(theta_2, min_threshold * 2)
    return theta_1, theta_2
