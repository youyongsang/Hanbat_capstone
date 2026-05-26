"""Evaluation metrics for traffic classifiers."""

from __future__ import annotations

import numpy as np
import torch


def accuracy(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == y_true).float().mean().item()


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int = 4) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def precision_recall_from_confusion(matrix: np.ndarray) -> list[tuple[float, float]]:
    scores = []
    for label in range(matrix.shape[0]):
        true_positive = matrix[label, label]
        predicted_positive = matrix[:, label].sum()
        actual_positive = matrix[label, :].sum()
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores.append((precision, recall))
    return scores
