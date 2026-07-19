import json

import pandas as pd
import pytest

from snu_ordering.caption_eda import build_caption_eda_report
from snu_ordering.caption_structure import (
    event_count_bucket,
    extract_caption_structure,
)


def test_extracts_typed_relations_and_collapses_composite_markers():
    caption = (
        "A person opens the box, then removes the item while another person watches; "
        "finally, the item is placed on a table."
    )

    structure = extract_caption_structure(caption)

    assert structure.original == caption
    assert structure.segments == (
        "A person opens the box",
        "removes the item",
        "another person watches",
        "the item is placed on a table.",
    )
    assert [boundary.kind for boundary in structure.boundaries] == [
        "NEXT", "OVERLAP", "NEXT"
    ]
    assert structure.event_count == 4
    assert structure.confident_event_count() == 4


def test_bare_commas_are_weak_and_do_not_force_four_confident_events():
    structure = extract_caption_structure("Red, white, blue, and green objects appear.")

    assert structure.event_count == 4
    assert structure.confident_event_count() == 1
    assert {boundary.kind for boundary in structure.boundaries} == {"WEAK"}


def test_caption_without_boundary_stays_as_one_event():
    structure = extract_caption_structure("A cyclist rides down the road.")

    assert structure.segments == ("A cyclist rides down the road.",)
    assert structure.boundaries == ()
    assert event_count_bucket(structure.event_count) == "1"


def test_event_bucket_caps_counts_at_five_plus():
    assert event_count_bucket(4) == "4"
    assert event_count_bucket(5) == "5+"
    assert event_count_bucket(12) == "5+"
    with pytest.raises(ValueError, match="positive"):
        event_count_bucket(0)


def test_eda_report_contains_surface_confident_and_relation_buckets():
    frame = pd.DataFrame(
        [
            {
                "Id": "a",
                "Sentence": "A opens, then B closes while C waits; finally, D leaves.",
                "No_ordering": True,
            },
            {
                "Id": "b",
                "Sentence": "Red, white, blue, and green objects appear.",
                "No_ordering": False,
            },
            {
                "Id": "c",
                "Sentence": "A cyclist rides away.",
                "No_ordering": False,
            },
        ]
    )

    report, rows = build_caption_eda_report(frame)

    assert report["rows"] == 3
    assert report["surface_event_count"]["histogram"] == {"1": 1, "4": 2}
    assert report["confident_event_count"]["histogram"] == {"1": 2, "4": 1}
    assert report["boundary_kinds"]["NEXT"]["sentences"] == 1
    assert report["boundary_kinds"]["OVERLAP"]["boundary_count"] == 1
    assert report["surface_event_count"]["buckets"]["4"]["no_ordering_rate"] == 0.5
    assert json.loads(rows.loc[0, "segments"])[0] == "A opens"


def test_rejects_blank_caption_and_invalid_threshold():
    with pytest.raises(ValueError, match="non-whitespace"):
        extract_caption_structure("  ")
    structure = extract_caption_structure("A happens.")
    with pytest.raises(ValueError, match="between 0 and 1"):
        structure.confident_event_count(1.1)
