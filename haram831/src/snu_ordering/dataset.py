"""Metadata loading, schema validation, and bounded image integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .permutation import answer_to_class_id, parse_answer

INPUT_COLUMNS = ("Input_1", "Input_2", "Input_3", "Input_4")
TRAIN_COLUMNS = ("Id", "Sentence", *INPUT_COLUMNS, "No_ordering", "Answer")
TEST_COLUMNS = ("Id", "Sentence", *INPUT_COLUMNS)
IDENTITY_ANSWER = (1, 2, 3, 4)


@dataclass(frozen=True)
class ImageIssue:
    sample_id: str
    input_column: str
    path: Path
    kind: Literal["missing", "unreadable"]
    detail: str


def _validate_schema(frame: pd.DataFrame, expected: tuple[str, ...], csv_path: Path) -> None:
    actual = tuple(frame.columns)
    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise ValueError(f"Invalid schema for {csv_path}: {'; '.join(parts)}")


def _parse_no_ordering(value: object, sample_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"Id {sample_id}: No_ordering must be True or False")


def load_metadata(
    csv_path: str | Path,
    *,
    split: Literal["train", "test"],
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and validate metadata, optionally bounded to a debug subset."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    frame = pd.read_csv(path, nrows=limit, keep_default_na=False)
    expected = TRAIN_COLUMNS if split == "train" else TEST_COLUMNS
    _validate_schema(frame, expected, path)
    frame = frame.loc[:, expected].copy()

    if frame["Id"].eq("").any():
        raise ValueError(f"{path} contains missing Id values")
    duplicate_mask = frame["Id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = frame.loc[duplicate_mask, "Id"].astype(str).unique().tolist()
        raise ValueError(f"{path} contains duplicate Id values: {duplicate_ids}")

    required_text = ("Id", "Sentence", *INPUT_COLUMNS)
    for column in required_text:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path} contains missing values in {column}")

    if split == "train":
        answers = []
        class_ids = []
        normalized_no_ordering = []
        for _, row in frame.iterrows():
            sample_id = str(row["Id"])
            try:
                answer = parse_answer(row["Answer"])
                no_ordering = _parse_no_ordering(row["No_ordering"], sample_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Id {sample_id}: invalid training label: {exc}") from exc
            if no_ordering and answer != IDENTITY_ANSWER:
                raise ValueError(
                    f"Id {sample_id}: No_ordering=True requires Answer [1, 2, 3, 4]"
                )
            answers.append(answer)
            class_ids.append(answer_to_class_id(answer))
            normalized_no_ordering.append(no_ordering)
        frame = frame.copy()
        frame["No_ordering"] = normalized_no_ordering
        frame["answer_tuple"] = answers
        frame["class_id"] = class_ids
    return frame


def resolve_image_paths(row: pd.Series, image_root: str | Path) -> tuple[Path, Path, Path, Path]:
    """Resolve frame paths strictly in Input_1 through Input_4 order."""
    sample_dir = Path(image_root) / str(row["Id"])
    return tuple(sample_dir / str(row[column]) for column in INPUT_COLUMNS)  # type: ignore[return-value]


def inspect_image_files(frame: pd.DataFrame, image_root: str | Path) -> list[ImageIssue]:
    """Check only the supplied rows for missing or unreadable images."""
    issues: list[ImageIssue] = []
    for _, row in frame.iterrows():
        sample_id = str(row["Id"])
        for column, path in zip(INPUT_COLUMNS, resolve_image_paths(row, image_root)):
            if not path.is_file():
                issues.append(ImageIssue(sample_id, column, path, "missing", "file does not exist"))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                issues.append(ImageIssue(sample_id, column, path, "unreadable", str(exc)))
    return issues
