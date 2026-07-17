import json

import numpy as np
import pytest

from snu_ordering.experiment_log import append_experiment, load_experiments


def record(experiment_id, score):
    return {
        "experiment_id": experiment_id,
        "split_id": None,
        "config": {"seed": 42, "labels": (1, 2, 3, 4)},
        "metrics": {"exact": score, "confusion": np.zeros((2, 2), dtype=np.int64)},
    }


def test_append_preserves_records_and_serializes_numpy_values(tmp_path):
    path = tmp_path / "logs" / "experiments.jsonl"
    append_experiment(path, record("first", np.float64(0.25)))
    append_experiment(path, record("second", 0.5))

    loaded = load_experiments(path)

    assert [entry["experiment_id"] for entry in loaded] == ["first", "second"]
    assert loaded[0]["config"]["labels"] == [1, 2, 3, 4]
    assert loaded[0]["metrics"]["exact"] == 0.25
    assert loaded[0]["metrics"]["confusion"] == [[0, 0], [0, 0]]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_duplicate_experiment_id_is_rejected_without_changing_log(tmp_path):
    path = tmp_path / "experiments.jsonl"
    append_experiment(path, record("same", 0.25))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="duplicate experiment_id"):
        append_experiment(path, record("same", 0.75))

    assert path.read_bytes() == before


@pytest.mark.parametrize("missing_key", ["experiment_id", "split_id", "config", "metrics"])
def test_required_record_fields_are_enforced(tmp_path, missing_key):
    value = record("trial", 0.1)
    value.pop(missing_key)
    with pytest.raises(ValueError, match=missing_key):
        append_experiment(tmp_path / "experiments.jsonl", value)


def test_load_missing_log_returns_empty_list(tmp_path):
    assert load_experiments(tmp_path / "missing.jsonl") == []


def test_written_lines_are_valid_json_objects(tmp_path):
    path = tmp_path / "experiments.jsonl"
    append_experiment(path, record("trial", 1.0))
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
