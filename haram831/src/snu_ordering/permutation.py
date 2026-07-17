"""Canonical permutation representations for the competition.

A competition answer stores the temporal position of each input frame:
``answer[input_index - 1] == temporal_position``. A chronological image
order stores the 1-based input indices from earliest to latest. These two
representations are inverse permutations.
"""

from __future__ import annotations

import ast
from itertools import permutations
from typing import Iterable

Permutation = tuple[int, int, int, int]
ALL_PERMUTATIONS: tuple[Permutation, ...] = tuple(permutations((1, 2, 3, 4)))
PERMUTATION_TO_CLASS = {answer: class_id for class_id, answer in enumerate(ALL_PERMUTATIONS)}


def validate_permutation(value: Iterable[int], *, name: str = "permutation") -> Permutation:
    """Return a validated permutation tuple or raise a clear error."""
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of four integers") from exc
    if len(result) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    if any(type(item) is not int for item in result):
        raise ValueError(f"{name} values must be integers, not booleans or other types")
    if sorted(result) != [1, 2, 3, 4]:
        raise ValueError(f"{name} must be a permutation of [1, 2, 3, 4]")
    return result  # type: ignore[return-value]


def parse_answer(raw: str) -> Permutation:
    """Strictly parse the list-string representation stored in ``Answer``."""
    if not isinstance(raw, str):
        raise TypeError("Answer must be a string containing a Python-style list")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Answer is not a parseable list: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Answer must use list syntax, for example '[1, 2, 3, 4]'")
    return validate_permutation(parsed, name="Answer")


def answer_to_class_id(answer: Iterable[int]) -> int:
    """Map a competition answer to its canonical lexicographic class ID."""
    return PERMUTATION_TO_CLASS[validate_permutation(answer, name="competition answer")]


def class_id_to_answer(class_id: int) -> Permutation:
    """Map a class ID in ``0..23`` to a competition answer."""
    if type(class_id) is not int or not 0 <= class_id < len(ALL_PERMUTATIONS):
        raise ValueError("class_id must be an integer from 0 through 23")
    return ALL_PERMUTATIONS[class_id]


def answer_to_chronological_order(answer: Iterable[int]) -> Permutation:
    """Invert a competition answer into earliest-to-latest input indices."""
    validated = validate_permutation(answer, name="competition answer")
    return tuple(validated.index(position) + 1 for position in range(1, 5))  # type: ignore[return-value]


def chronological_order_to_answer(image_order: Iterable[int]) -> Permutation:
    """Invert earliest-to-latest input indices into competition format."""
    validated = validate_permutation(image_order, name="chronological image order")
    answer = [0, 0, 0, 0]
    for temporal_position, input_index in enumerate(validated, start=1):
        answer[input_index - 1] = temporal_position
    return tuple(answer)  # type: ignore[return-value]
