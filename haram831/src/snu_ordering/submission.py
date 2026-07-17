"""Strict construction and validation of competition submission tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .permutation import validate_permutation

SUBMISSION_COLUMNS = ("Id", "Answer")


def _format_answer(answer: Iterable[int]) -> str:
    validated = validate_permutation(answer, name="submission answer")
    return "[" + ", ".join(str(value) for value in validated) + "]"


def build_submission(expected_ids: Sequence[str], answers: Sequence[Iterable[int]]) -> pd.DataFrame:
    """Build a validated in-memory submission without writing any files."""
    if len(expected_ids) != len(answers):
        raise ValueError("expected_ids and answers must contain the same number of samples")
    frame = pd.DataFrame(
        {"Id": [str(value) for value in expected_ids], "Answer": [_format_answer(a) for a in answers]},
        columns=SUBMISSION_COLUMNS,
    )
    validate_submission(frame, expected_ids)
    return frame


def validate_submission(submission: pd.DataFrame, expected_ids: Sequence[str]) -> None:
    """Raise ``ValueError`` unless every submission contract is satisfied."""
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission must have exact columns in order: {list(SUBMISSION_COLUMNS)}")

    expected = [str(value) for value in expected_ids]
    if len(submission) != len(expected):
        raise ValueError(
            f"Submission row count is {len(submission)}, expected {len(expected)}; debug subsets are invalid"
        )

    actual = submission["Id"].astype(str).tolist()
    duplicate_mask = submission["Id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = submission.loc[duplicate_mask, "Id"].astype(str).unique().tolist()
        raise ValueError(f"Submission contains duplicate IDs: {duplicates}")

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        raise ValueError(f"Submission has unexpected IDs: {unexpected}")
    if missing:
        raise ValueError(f"Submission has missing IDs: {missing}")
    if actual != expected:
        raise ValueError("Submission ID order must exactly match test.csv")

    missing_answer = submission["Answer"].isna() | submission["Answer"].astype(str).str.strip().eq("")
    if missing_answer.any():
        ids = submission.loc[missing_answer, "Id"].astype(str).tolist()
        raise ValueError(f"Submission contains missing answers for IDs: {ids}")

    from .permutation import parse_answer

    for sample_id, raw_answer in zip(actual, submission["Answer"]):
        try:
            parse_answer(raw_answer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Submission has invalid Answer for Id {sample_id}: {exc}") from exc
