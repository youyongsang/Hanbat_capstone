"""Metrics for Early Exit model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExitStats:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def add(self, is_correct: bool) -> None:
        self.correct += int(is_correct)
        self.total += 1


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=-1)
    return (y_pred == y_true).float().mean().item()


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"
