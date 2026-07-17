"""Evaluation metrics for four-frame temporal ordering."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .permutation import ALL_PERMUTATIONS, Permutation, answer_to_class_id, validate_permutation


def _validated_pairs(
    y_true: Sequence[Iterable[int]], y_pred: Sequence[Iterable[int]]
) -> tuple[list[Permutation], list[Permutation]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of samples")
    truth = [validate_permutation(answer, name="y_true permutation") for answer in y_true]
    predictions = [validate_permutation(answer, name="y_pred permutation") for answer in y_pred]
    return truth, predictions


def exact_match_accuracy(
    y_true: Sequence[Iterable[int]], y_pred: Sequence[Iterable[int]]
) -> float | None:
    truth, predictions = _validated_pairs(y_true, y_pred)
    if not truth:
        return None
    return sum(actual == predicted for actual, predicted in zip(truth, predictions)) / len(truth)


def pairwise_accuracy(
    y_true: Sequence[Iterable[int]], y_pred: Sequence[Iterable[int]]
) -> float | None:
    """Compare relative order for all six pairs of supplied input frames."""
    truth, predictions = _validated_pairs(y_true, y_pred)
    if not truth:
        return None
    correct = 0
    total = 0
    for actual, predicted in zip(truth, predictions):
        for left, right in combinations(range(4), 2):
            correct += (actual[left] < actual[right]) == (predicted[left] < predicted[right])
            total += 1
    return correct / total


def compute_metrics(
    y_true: Sequence[Iterable[int]], y_pred: Sequence[Iterable[int]]
) -> dict[str, object]:
    """Compute aggregate, subset, class, and fixed-shape confusion metrics."""
    truth, predictions = _validated_pairs(y_true, y_pred)
    exact = [actual == predicted for actual, predicted in zip(truth, predictions)]
    identity = (1, 2, 3, 4)
    identity_matches = [match for match, actual in zip(exact, truth) if actual == identity]
    non_identity_matches = [match for match, actual in zip(exact, truth) if actual != identity]

    confusion = np.zeros((24, 24), dtype=np.int64)
    true_classes = [answer_to_class_id(answer) for answer in truth]
    predicted_classes = [answer_to_class_id(answer) for answer in predictions]
    for true_class, predicted_class in zip(true_classes, predicted_classes):
        confusion[true_class, predicted_class] += 1

    per_class: dict[int, dict[str, int | float | None]] = {}
    for class_id, _ in enumerate(ALL_PERMUTATIONS):
        indices = [index for index, value in enumerate(true_classes) if value == class_id]
        support = len(indices)
        accuracy = None if support == 0 else sum(exact[index] for index in indices) / support
        per_class[class_id] = {"support": support, "accuracy": accuracy}

    return {
        "exact_match_accuracy": None if not exact else sum(exact) / len(exact),
        "pairwise_accuracy": pairwise_accuracy(truth, predictions),
        "identity_accuracy": (
            None if not identity_matches else sum(identity_matches) / len(identity_matches)
        ),
        "non_identity_accuracy": (
            None if not non_identity_matches else sum(non_identity_matches) / len(non_identity_matches)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
