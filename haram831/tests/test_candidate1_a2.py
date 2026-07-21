import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from snu_ordering.caption_structure import render_relation_hints
from snu_ordering.candidate1.config import CaptionPromptConfig, Candidate1Config
from snu_ordering.candidate1.data import pairwise_labels_for_row, render_instruction
from snu_ordering.candidate1.model import completion_only_training_forward
from snu_ordering.candidate1.pairwise import (
    combine_training_losses,
    compute_pairwise_auxiliary,
    load_pairwise_head,
    save_pairwise_head,
)
from snu_ordering.permutation import ALL_PERMUTATIONS


def test_a2_renders_high_confidence_relation_edges_and_preserves_original():
    caption = "A opens, then B moves before C waits while D watches; E leaves."

    rendered = render_relation_hints(caption)

    assert f"Original caption:\n{caption}" in rendered
    assert "[NEXT] A opens -> B moves" in rendered
    assert "[BEFORE] B moves -> C waits" in rendered
    assert "[OVERLAP] C waits || D watches" in rendered
    assert "[SEQUENCE] D watches -> E leaves." in rendered
    assert "[WEAK]" not in rendered


def test_a2_after_relation_keeps_its_direction_explicit():
    rendered = render_relation_hints("A happens after B happens.")
    assert "[AFTER] A happens occurs after B happens." in rendered


def test_a2_weak_only_caption_falls_back_to_unchanged_original():
    caption = "Red, white, blue, and green objects appear."
    rendered = render_relation_hints(caption)

    assert rendered == f"Original caption:\n{caption}"
    assert "Approximate action relations" not in rendered


def test_boundary_dropout_is_stable_and_changes_across_training_seeds():
    caption = "A happens, then B happens while C waits; finally D leaves."
    first = render_relation_hints(
        caption, boundary_dropout=0.5, dropout_seed="42:0:sample"
    )
    second = render_relation_hints(
        caption, boundary_dropout=0.5, dropout_seed="42:0:sample"
    )
    variants = {
        render_relation_hints(
            caption, boundary_dropout=0.5, dropout_seed=f"42:{epoch}:sample"
        )
        for epoch in range(12)
    }

    assert first == second
    assert len(variants) > 1


def test_a2_instruction_disables_dropout_by_default():
    caption = "A happens, then B happens."
    rendered = render_instruction(
        caption,
        "relations",
        relation_confidence_threshold=0.7,
    )
    assert caption in rendered
    assert "[NEXT]" in rendered


def test_pairwise_labels_cover_all_six_frame_pairs_for_every_permutation():
    observed = {pairwise_labels_for_row({"answer_tuple": answer}) for answer in ALL_PERMUTATIONS}

    assert len(observed) == 24
    assert pairwise_labels_for_row({"answer_tuple": (1, 2, 3, 4)}) == (1.0,) * 6
    assert pairwise_labels_for_row({"answer_tuple": (4, 3, 2, 1)}) == (0.0,) * 6


def test_pairwise_auxiliary_uses_final_prompt_token_and_backpropagates():
    torch = pytest.importorskip("torch")
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    head = torch.nn.Linear(4, 6)
    prompt_lengths = torch.tensor([3, 5])
    labels = torch.tensor(
        [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=torch.float32
    )

    pairwise_loss, logits = compute_pairwise_auxiliary(
        hidden, head, prompt_lengths, labels
    )
    lm_loss = torch.tensor(2.0, requires_grad=True)
    total = combine_training_losses(lm_loss, pairwise_loss, 0.3)
    total.backward()

    assert logits.shape == (2, 6)
    assert total.item() == pytest.approx(2.0 + 0.3 * pairwise_loss.item())
    assert hidden.grad is not None
    assert head.weight.grad is not None
    assert lm_loss.grad.item() == pytest.approx(1.0)


def test_completion_only_forward_matches_full_causal_lm_loss_and_backpropagates():
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    class RecordingHead(torch.nn.Linear):
        def forward(self, values):
            self.last_input_shape = tuple(values.shape)
            return super().forward(values)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(20, 5)
            self.forward_kwargs = None

        def forward(self, input_ids, **kwargs):
            self.forward_kwargs = kwargs
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = RecordingHead(5, 20, bias=False)

    class PeftLikeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = CausalLM()

        def get_base_model(self):
            return self.base

    model = PeftLikeWrapper()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, -100, -100, 9, -100]]
    )

    expected_hidden = model.base.model.embedding(input_ids)
    expected_logits = model.base.lm_head(expected_hidden)
    shifted = functional.pad(labels, (0, 1), value=-100)[..., 1:]
    expected_loss = functional.cross_entropy(
        expected_logits.float().reshape(-1, 20),
        shifted.reshape(-1),
        ignore_index=-100,
    )

    loss, hidden, stats = completion_only_training_forward(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
    )
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss.item())
    assert hidden.shape == (2, 5, 5)
    assert model.base.lm_head.last_input_shape == (4, 5)
    assert stats == {
        "sequence_tokens": 10,
        "supervised_tokens": 4,
        "logit_rows": 4,
        "vocabulary_size": 20,
    }
    assert model.base.model.forward_kwargs["output_hidden_states"] is False
    assert model.base.model.forward_kwargs["use_cache"] is False
    assert "labels" not in model.base.model.forward_kwargs
    assert model.base.model.embedding.weight.grad is not None
    assert model.base.lm_head.weight.grad is not None


def test_pairwise_head_safetensors_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    source = torch.nn.Linear(4, 6)
    target = torch.nn.Linear(4, 6)
    path = tmp_path / "pairwise_head.safetensors"

    save_pairwise_head(source, path)
    load_pairwise_head(target, path)

    assert path.is_file()
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_a2_config_and_legacy_defaults_round_trip(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "candidate1-a2.json"
    a2 = Candidate1Config.load(config_path)

    assert a2.caption_prompt == CaptionPromptConfig(
        mode="relations", relation_confidence_threshold=0.7, boundary_dropout=0.3
    )
    assert a2.training.pairwise_loss_weight == 0.3

    raw = a2.to_dict()
    raw["caption_prompt"] = {"mode": "raw"}
    raw["training"].pop("pairwise_loss_weight")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = Candidate1Config.load(legacy_path)

    assert legacy.caption_prompt.boundary_dropout == 0.0
    assert legacy.training.pairwise_loss_weight == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"relation_confidence_threshold": 1.1}, "relation_confidence_threshold"),
        ({"boundary_dropout": 1.0}, "boundary_dropout"),
    ],
)
def test_a2_rejects_invalid_caption_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CaptionPromptConfig(mode="relations", **kwargs)
