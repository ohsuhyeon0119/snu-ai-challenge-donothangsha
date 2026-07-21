from argparse import Namespace

import pytest

from snu_ordering.candidate1.memory_ablation import EXPERIMENTS, configured_run, select_row


def test_memory_ablation_defines_the_four_controlled_modes():
    assert set(EXPERIMENTS) == {"A", "B", "C", "D"}
    assert "pairwise loss off" in EXPERIMENTS["A"]
    assert "all hidden states returned" in EXPERIMENTS["B"]
    assert "legacy A2" in EXPERIMENTS["C"]
    assert "optimized A2" in EXPERIMENTS["D"]


def test_memory_ablation_selects_one_stable_row():
    rows = [{"Id": "first"}, {"Id": "second"}]

    assert select_row(rows, "second", 0) == rows[1]
    assert select_row(rows, None, 1) == rows[1]
    with pytest.raises(ValueError, match="does not exist"):
        select_row(rows, "missing", 0)
    with pytest.raises(ValueError, match="sample-index"):
        select_row(rows, None, 2)


def test_memory_ablation_disables_prompt_dropout_without_changing_pixels(tmp_path):
    config_path = tmp_path / "config.json"
    from snu_ordering.candidate1.config import Candidate1Config

    Candidate1Config().save(config_path)
    args = Namespace(
        config=str(config_path),
        base_model="local-model",
        processor="local-processor",
    )

    configured = configured_run(args)

    assert configured.model.base_model_path == "local-model"
    assert configured.model.processor_path == "local-processor"
    assert configured.model.min_pixels == Candidate1Config().model.min_pixels
    assert configured.model.max_pixels == Candidate1Config().model.max_pixels
    assert configured.caption_prompt.boundary_dropout == 0.0
