from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from snu_ordering.dataset import (
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    inspect_image_files,
    load_metadata,
    resolve_image_paths,
)


def row(sample_id="sample1", answer="[1, 2, 3, 4]", no_ordering=True):
    return {
        "Id": sample_id,
        "Sentence": "First this happens, then that happens.",
        "Input_1": "one.png",
        "Input_2": "two.png",
        "Input_3": "three.png",
        "Input_4": "four.png",
        "No_ordering": no_ordering,
        "Answer": answer,
    }


def write_csv(path: Path, rows, columns=TRAIN_COLUMNS):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def create_images(root: Path, sample_id="sample1"):
    folder = root / sample_id
    folder.mkdir(parents=True)
    for index, name in enumerate(("one.png", "two.png", "three.png", "four.png"), start=1):
        Image.new("RGB", (2, 2), (index, index, index)).save(folder / name)


def test_load_training_metadata_validates_schema_and_parses_labels(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [row()])

    frame = load_metadata(csv_path, split="train")

    assert tuple(frame.columns[: len(TRAIN_COLUMNS)]) == TRAIN_COLUMNS
    assert frame.loc[0, "answer_tuple"] == (1, 2, 3, 4)
    assert frame.loc[0, "class_id"] == 0


def test_load_test_metadata_accepts_only_test_schema(tmp_path):
    csv_path = tmp_path / "test.csv"
    test_row = {key: value for key, value in row().items() if key in TEST_COLUMNS}
    write_csv(csv_path, [test_row], TEST_COLUMNS)
    frame = load_metadata(csv_path, split="test")
    assert tuple(frame.columns) == TEST_COLUMNS


def test_metadata_accepts_observed_dataset_column_order_and_normalizes_it(tmp_path):
    csv_path = tmp_path / "train.csv"
    observed_order = (
        "Id",
        "Input_1",
        "Input_2",
        "Input_3",
        "Input_4",
        "Sentence",
        "Answer",
        "No_ordering",
    )
    write_csv(csv_path, [row()], observed_order)

    frame = load_metadata(csv_path, split="train")

    assert tuple(frame.columns[: len(TRAIN_COLUMNS)]) == TRAIN_COLUMNS


def test_schema_mismatch_reports_missing_and_unexpected_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"Id": "x", "Unexpected": "value"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_metadata(path, split="train")


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate Id"):
        load_metadata(path, split="train")


def test_malformed_training_label_is_rejected(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path, [row(answer="[1, 1, 3, 4]", no_ordering=False)])
    with pytest.raises(ValueError, match="sample1"):
        load_metadata(path, split="train")


def test_no_ordering_true_requires_identity_answer(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path, [row(answer="[2, 1, 3, 4]", no_ordering=True)])
    with pytest.raises(ValueError, match="No_ordering"):
        load_metadata(path, split="train")


def test_image_paths_preserve_input_1_through_input_4_order(tmp_path):
    paths = resolve_image_paths(pd.Series(row()), tmp_path)
    assert [path.name for path in paths] == ["one.png", "two.png", "three.png", "four.png"]
    assert all(path.parent == tmp_path / "sample1" for path in paths)


def test_image_inspection_detects_missing_and_unreadable_files(tmp_path):
    frame = pd.DataFrame([row()])
    create_images(tmp_path)
    (tmp_path / "sample1" / "two.png").unlink()
    (tmp_path / "sample1" / "three.png").write_text("not an image", encoding="utf-8")

    issues = inspect_image_files(frame, tmp_path)

    assert [(issue.input_column, issue.kind) for issue in issues] == [
        ("Input_2", "missing"),
        ("Input_3", "unreadable"),
    ]


def test_metadata_limit_reads_only_requested_debug_rows(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path, [row("first"), row("second")])
    assert load_metadata(path, split="train", limit=1)["Id"].tolist() == ["first"]
