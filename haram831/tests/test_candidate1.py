import inspect
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from snu_ordering.candidate1.artifacts import ArtifactLayout
from snu_ordering.candidate1.config import ARCHITECTURE_VERSION, Candidate1Config
from snu_ordering.candidate1.data import (
    build_messages,
    chronological_order_for_row,
    collate_rows,
    target_text,
)
from snu_ordering.candidate1.inference import parse_args as parse_inference_args
from snu_ordering.candidate1.model import (
    validate_checkpoint_architecture,
    validate_trainable_parameters,
)
from snu_ordering.candidate1.scoring import (
    ALL_ORDERS,
    completion_log_likelihoods,
    order_to_answer,
    parse_generated_order,
)
from snu_ordering.candidate1.train import (
    is_cuda_out_of_memory,
    run_training,
    stratified_split,
    summarize_predictions,
)


def test_candidate1_defaults_lock_causal_lm_v3():
    config = Candidate1Config()

    assert config.architecture_version == "candidate1_causal_lm_v3"
    assert config.lora.rank == 32
    assert config.lora.alpha == 64
    assert config.lora.target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    assert config.training.epochs == 3
    assert config.training.learning_rate == 1e-4
    assert config.training.warmup_ratio == 0.03
    assert config.training.validation_fraction == 0.12
    assert config.training.epoch_validation_size == 160
    assert config.training.scoring_chunk_size == 12


def test_candidate1_config_round_trips_and_rejects_v2(tmp_path):
    path = tmp_path / "run_config.json"
    Candidate1Config().save(path)
    assert Candidate1Config.load(path) == Candidate1Config()

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["architecture_version"] = "candidate1_classifier_v2"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="start a fresh v3 run"):
        Candidate1Config.load(path)


def test_candidate1_config_migrates_legacy_constrained_validation_size(tmp_path):
    path = tmp_path / "run_config.json"
    raw = Candidate1Config().to_dict()
    raw["training"]["constrained_validation_size"] = raw["training"].pop(
        "epoch_validation_size"
    )
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = Candidate1Config.load(path)

    assert loaded.training.epoch_validation_size == 160


def test_greedy_is_the_default_for_model_selection_and_inference(monkeypatch):
    source = inspect.getsource(run_training)
    assert 'epoch_validation_rows, args.image_root, config, mode="generate"' in source
    assert 'validation_rows, args.image_root, config, mode="generate"' in source

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "candidate1-inference",
            "--base-model", "model",
            "--processor", "processor",
            "--adapter", "adapter",
            "--config", "config.json",
            "--input-csv", "test.csv",
            "--image-root", "test-images",
            "--output-submission", "submission.csv",
        ],
    )
    assert parse_inference_args().mode == "generate"


def test_prompt_keeps_sentence_and_four_images_in_input_order(tmp_path):
    row = {
        "Id": "abc",
        "Sentence": "First the door opens, then a person enters.",
        "Input_1": "one.jpg", "Input_2": "two.jpg",
        "Input_3": "three.jpg", "Input_4": "four.jpg",
    }
    content = build_messages(row, tmp_path, min_pixels=100, max_pixels=200)[0]["content"]
    images = [item for item in content if item["type"] == "image"]
    text = "\n".join(item["text"] for item in content if item["type"] == "text")

    assert [Path(item["image"]).name for item in images] == [
        "one.jpg", "two.jpg", "three.jpg", "four.jpg"
    ]
    assert row["Sentence"] in text
    assert "earliest to latest" in text


def test_answer_is_inverted_to_natural_chronological_target():
    row = {"answer_tuple": (2, 4, 3, 1)}
    order = chronological_order_for_row(row)

    assert order == (4, 1, 3, 2)
    assert target_text(order) == "The correct chronological order is [4, 1, 3, 2]."
    assert order_to_answer(order) == (2, 4, 3, 1)


def test_all_24_orders_round_trip_to_submission_answers():
    assert len(ALL_ORDERS) == 24
    assert len({order_to_answer(order) for order in ALL_ORDERS}) == 24


def test_generated_order_parser_rejects_malformed_output():
    assert parse_generated_order("answer [3, 1, 4, 2]") == (3, 1, 4, 2)
    assert parse_generated_order("answer [1, 1, 2, 3]") is None
    assert parse_generated_order("no list") is None


def test_collate_masks_prompt_and_padding(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        padding_side = "left"
        eos_token = "<eos>"

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return "PROMPT"

        def __call__(self, *, text, images, padding, return_tensors):
            lengths = [6 if "correct chronological" in value else 3 for value in text]
            width = max(lengths)
            ids = torch.zeros((len(text), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, length in enumerate(lengths):
                ids[index, :length] = torch.arange(1, length + 1)
                mask[index, :length] = 1
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "image_grid_thw": torch.ones((4 * len(text), 3), dtype=torch.long),
            }

    fake_utils = types.SimpleNamespace(process_vision_info=lambda messages: ([object()] * 4, None))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_utils)
    row = {
        "Id": "abc", "Sentence": "A then B.",
        "Input_1": "1.jpg", "Input_2": "2.jpg",
        "Input_3": "3.jpg", "Input_4": "4.jpg",
        "answer_tuple": (2, 4, 3, 1),
    }
    batch = collate_rows(
        [row], Processor(), tmp_path,
        min_pixels=100, max_pixels=200, include_labels=True,
    )

    assert batch["labels"][0, :3].tolist() == [-100, -100, -100]
    assert batch["labels"][0, 3:].tolist() == [4, 5, 6]


def test_trainability_guard_accepts_only_lora_parameters():
    class Parameter:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad

    class Model:
        def named_parameters(self):
            return iter([
                ("base.weight", Parameter(False)),
                ("base.q_proj.lora_A.default.weight", Parameter(True)),
            ])

    validate_trainable_parameters(Model())


def test_scoring_uses_completion_only_causal_shift():
    source = inspect.getsource(completion_log_likelihoods)
    assert "_completion_logits(model, inputs, prompt_length)" in source
    assert 'inputs["input_ids"][:, prompt_length:]' in source
    assert "shifted_mask" in source


def test_stratified_split_preserves_every_class():
    rows = [
        {"class_id": class_id, "Id": f"{class_id}-{index}"}
        for class_id in range(3)
        for index in range(10)
    ]
    train, validation = stratified_split(rows, 0.2, 42)

    assert Counter(row["class_id"] for row in validation) == Counter({0: 2, 1: 2, 2: 2})
    assert len(train) == 24


def test_cuda_oom_detection_accepts_accelerator_error_message():
    assert is_cuda_out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_out_of_memory(RuntimeError("CUDA device-side assert triggered"))
    assert not is_cuda_out_of_memory(MemoryError("CPU out of memory"))


def test_prediction_summary_detects_single_class_collapse():
    summary = summarize_predictions([0, 1, 2, 3], [0, 0, 0, 0], collapse_threshold=0.5)
    assert summary["exact_match_accuracy"] == 0.25
    assert summary["max_prediction_share"] == 1.0
    assert summary["collapse_detected"] is True


def test_checkpoint_architecture_uses_run_metadata(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"architecture_version": "candidate1_classifier_v2"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="causal-LM v3"):
        validate_checkpoint_architecture(Candidate1Config(), tmp_path)

    metadata.write_text(
        json.dumps({"architecture_version": ARCHITECTURE_VERSION}), encoding="utf-8"
    )
    assert validate_checkpoint_architecture(Candidate1Config(), tmp_path)[
        "architecture_version"
    ] == ARCHITECTURE_VERSION


def test_artifact_layout_has_no_classifier_head(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    layout.create()
    assert layout.adapter_dir.is_dir()
    assert not hasattr(layout, "classifier_head_path")
    assert layout.trainer_state_path == layout.root / "trainer_state.pt"
