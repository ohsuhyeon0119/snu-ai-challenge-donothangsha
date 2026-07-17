import numpy as np
import pytest

from snu_ordering.metrics import compute_metrics, exact_match_accuracy, pairwise_accuracy
from snu_ordering.permutation import answer_to_class_id


def test_metrics_include_required_aggregate_subset_and_class_results():
    identity = (1, 2, 3, 4)
    swapped = (2, 1, 3, 4)
    y_true = [identity, swapped, swapped]
    y_pred = [identity, identity, swapped]

    result = compute_metrics(y_true, y_pred)

    assert result["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert result["pairwise_accuracy"] == pytest.approx(17 / 18)
    assert result["identity_accuracy"] == 1.0
    assert result["non_identity_accuracy"] == 0.5

    identity_class = answer_to_class_id(identity)
    swapped_class = answer_to_class_id(swapped)
    assert result["per_class"][identity_class] == {"support": 1, "accuracy": 1.0}
    assert result["per_class"][swapped_class] == {"support": 2, "accuracy": 0.5}
    assert len(result["per_class"]) == 24

    confusion = result["confusion_matrix"]
    assert isinstance(confusion, np.ndarray)
    assert confusion.shape == (24, 24)
    assert confusion[identity_class, identity_class] == 1
    assert confusion[swapped_class, identity_class] == 1
    assert confusion[swapped_class, swapped_class] == 1
    assert int(confusion.sum()) == 3


def test_empty_metrics_are_explicit_and_safe():
    result = compute_metrics([], [])
    assert result["exact_match_accuracy"] is None
    assert result["pairwise_accuracy"] is None
    assert result["identity_accuracy"] is None
    assert result["non_identity_accuracy"] is None
    assert all(entry == {"support": 0, "accuracy": None} for entry in result["per_class"].values())
    assert result["confusion_matrix"].shape == (24, 24)
    assert int(result["confusion_matrix"].sum()) == 0


def test_metric_inputs_must_have_equal_lengths():
    with pytest.raises(ValueError, match="same number"):
        exact_match_accuracy([(1, 2, 3, 4)], [])


def test_metric_inputs_must_be_valid_permutations():
    with pytest.raises(ValueError, match="permutation"):
        pairwise_accuracy([(1, 2, 3, 4)], [(1, 1, 3, 4)])
