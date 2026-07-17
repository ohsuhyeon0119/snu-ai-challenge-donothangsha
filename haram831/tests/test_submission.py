import pandas as pd
import pytest

from snu_ordering.submission import build_submission, validate_submission


EXPECTED_IDS = ["test-a", "test-b", "test-c"]
VALID_ANSWERS = [(1, 2, 3, 4), (2, 1, 3, 4), (4, 3, 2, 1)]


def valid_frame():
    return pd.DataFrame(
        {
            "Id": EXPECTED_IDS,
            "Answer": ["[1, 2, 3, 4]", "[2, 1, 3, 4]", "[4, 3, 2, 1]"],
        }
    )


def test_build_submission_uses_exact_schema_order_and_valid_answers():
    frame = build_submission(EXPECTED_IDS, VALID_ANSWERS)
    assert frame.columns.tolist() == ["Id", "Answer"]
    assert frame["Id"].tolist() == EXPECTED_IDS
    assert frame["Answer"].tolist() == [
        "[1, 2, 3, 4]",
        "[2, 1, 3, 4]",
        "[4, 3, 2, 1]",
    ]
    validate_submission(frame, EXPECTED_IDS)


@pytest.mark.parametrize(
    "frame,match",
    [
        (valid_frame()[["Answer", "Id"]], "exact columns"),
        (valid_frame().assign(Extra="x"), "exact columns"),
        (valid_frame().iloc[:2], "row count"),
        (valid_frame().iloc[[1, 0, 2]], "ID order"),
        (valid_frame().assign(Id=["test-a", "test-a", "test-c"]), "duplicate"),
        (valid_frame().assign(Id=["test-a", "test-b", "unexpected"]), "unexpected IDs"),
        (valid_frame().assign(Answer=["[1, 2, 3, 4]", None, "[4, 3, 2, 1]"]), "missing answers"),
        (valid_frame().assign(Answer=["[1, 2, 3, 4]", "[1, 1, 3, 4]", "[4, 3, 2, 1]"]), "invalid Answer"),
    ],
)
def test_submission_validator_rejects_invalid_frames(frame, match):
    with pytest.raises(ValueError, match=match):
        validate_submission(frame, EXPECTED_IDS)


def test_debug_subset_is_rejected_when_full_ids_are_expected():
    with pytest.raises(ValueError, match="row count"):
        validate_submission(valid_frame().head(1), EXPECTED_IDS)


def test_build_submission_rejects_prediction_count_mismatch():
    with pytest.raises(ValueError, match="same number"):
        build_submission(EXPECTED_IDS, VALID_ANSWERS[:1])
