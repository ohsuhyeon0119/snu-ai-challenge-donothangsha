from itertools import permutations

import pytest

from snu_ordering.permutation import (
    ALL_PERMUTATIONS,
    answer_to_chronological_order,
    answer_to_class_id,
    chronological_order_to_answer,
    class_id_to_answer,
    parse_answer,
)


def test_all_permutations_have_canonical_lexicographic_order():
    expected = tuple(permutations((1, 2, 3, 4)))
    assert ALL_PERMUTATIONS == expected
    assert len(ALL_PERMUTATIONS) == 24


@pytest.mark.parametrize("class_id,answer", tuple(enumerate(permutations((1, 2, 3, 4)))))
def test_all_24_answers_round_trip_through_class_ids(class_id, answer):
    assert answer_to_class_id(answer) == class_id
    assert class_id_to_answer(class_id) == answer


@pytest.mark.parametrize("answer", tuple(permutations((1, 2, 3, 4))))
def test_competition_answer_and_chronological_order_are_inverse(answer):
    chronological = answer_to_chronological_order(answer)
    assert chronological_order_to_answer(chronological) == answer
    for temporal_position, input_index in enumerate(chronological, start=1):
        assert answer[input_index - 1] == temporal_position


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[1, 2, 3, 4]", (1, 2, 3, 4)),
        ("[4,3, 2,1]", (4, 3, 2, 1)),
    ],
)
def test_parse_answer_accepts_only_valid_list_strings(raw, expected):
    assert parse_answer(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "(1, 2, 3, 4)",
        "[1, 1, 3, 4]",
        "[0, 2, 3, 4]",
        "[1, 2, 3]",
        "[True, 2, 3, 4]",
        "not a list",
        "",
        None,
    ],
)
def test_parse_answer_rejects_malformed_values(raw):
    with pytest.raises((TypeError, ValueError)):
        parse_answer(raw)


def test_class_id_must_be_in_range():
    with pytest.raises(ValueError):
        class_id_to_answer(-1)
    with pytest.raises(ValueError):
        class_id_to_answer(24)
