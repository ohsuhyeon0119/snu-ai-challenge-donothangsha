from pathlib import Path

import pytest

from snu_ordering.config import DataPaths


def create_dataset_layout(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (root / name).write_text("placeholder", encoding="utf-8")


def test_data_paths_use_environment_override_and_observed_layout(tmp_path, monkeypatch):
    data_root = tmp_path / "competition_data"
    create_dataset_layout(data_root)
    monkeypatch.setenv("SNU_DATA_DIR", str(data_root))

    paths = DataPaths.from_environment()

    assert paths.data_root == data_root
    assert paths.train_csv == data_root / "train.csv"
    assert paths.test_csv == data_root / "test.csv"
    assert paths.sample_submission_csv == data_root / "sample_submission.csv"
    assert paths.train_images == data_root / "train"
    assert paths.test_images == data_root / "test"
    paths.validate()


def test_explicit_data_root_takes_precedence_over_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    create_dataset_layout(explicit)
    monkeypatch.setenv("SNU_DATA_DIR", str(tmp_path / "wrong"))
    assert DataPaths.from_environment(explicit).data_root == explicit


def test_missing_required_items_raise_clear_error(tmp_path):
    paths = DataPaths.from_environment(tmp_path)
    with pytest.raises(FileNotFoundError, match="train.csv"):
        paths.validate()


def test_constructing_paths_does_not_create_output_directories(tmp_path):
    output_root = tmp_path / "not-created"
    paths = DataPaths.from_environment(tmp_path, output_root=output_root)
    assert paths.output_root == output_root
    assert not output_root.exists()
