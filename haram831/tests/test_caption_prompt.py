import json
from pathlib import Path

import pytest

from snu_ordering.caption_structure import (
    extract_punctuation_candidates,
    render_punctuation_hints,
)
from snu_ordering.candidate1.config import (
    CaptionPromptConfig,
    Candidate1Config,
)
from snu_ordering.candidate1.data import build_messages, render_instruction


def test_a1_splits_only_on_commas_and_semicolons():
    caption = "A happens, then B happens while C waits; finally D leaves."

    assert extract_punctuation_candidates(caption) == (
        "A happens",
        "then B happens while C waits",
        "finally D leaves.",
    )


def test_a1_keeps_original_and_describes_hints_without_punctuation_mechanics():
    caption = "A happens, then B happens; finally C happens."

    rendered = render_punctuation_hints(caption)

    assert f"Original caption:\n{caption}" in rendered
    assert "Approximate event hints:" in rendered
    assert "[Event 1] A happens" in rendered
    assert "[Event 3] finally C happens." in rendered
    assert "may not correspond one-to-one with the four images" in rendered
    assert "Use the original caption when the hints are ambiguous" in rendered
    assert "comma" not in rendered.lower()
    assert "semicolon" not in rendered.lower()


def test_a1_does_not_force_four_events():
    assert extract_punctuation_candidates("One action happens.") == (
        "One action happens.",
    )
    assert len(extract_punctuation_candidates("A, B; C, D; E.")) == 5


def test_raw_instruction_preserves_the_a0_caption_format():
    assert render_instruction("A then B.", "raw").startswith(
        'Caption: "A then B."\n\nThe caption describes events'
    )


def test_build_messages_renders_a1_and_preserves_image_order(tmp_path):
    row = {
        "Id": "sample",
        "Sentence": "A opens, then B closes.",
        "Input_1": "1.jpg",
        "Input_2": "2.jpg",
        "Input_3": "3.jpg",
        "Input_4": "4.jpg",
    }

    content = build_messages(
        row,
        tmp_path,
        min_pixels=100,
        max_pixels=200,
        caption_prompt_mode="punctuation",
    )[0]["content"]
    images = [Path(item["image"]).name for item in content if item["type"] == "image"]
    prompt = "\n".join(item["text"] for item in content if item["type"] == "text")

    assert images == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    assert row["Sentence"] in prompt
    assert "[Event 1] A opens" in prompt
    assert "[Event 2] then B closes." in prompt


def test_caption_prompt_config_round_trip_and_legacy_default(tmp_path):
    path = tmp_path / "run_config.json"
    configured = Candidate1Config(
        caption_prompt=CaptionPromptConfig(mode="punctuation")
    )
    configured.save(path)
    assert Candidate1Config.load(path).caption_prompt.mode == "punctuation"

    legacy = configured.to_dict()
    legacy.pop("caption_prompt")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert Candidate1Config.load(path).caption_prompt.mode == "raw"


def test_a1_config_file_selects_punctuation_mode():
    config_path = Path(__file__).parents[1] / "configs" / "candidate1-a1.json"
    assert Candidate1Config.load(config_path).caption_prompt.mode == "punctuation"


def test_rejects_unknown_caption_prompt_mode():
    with pytest.raises(ValueError, match="Unsupported caption prompt mode"):
        CaptionPromptConfig(mode="relations")
