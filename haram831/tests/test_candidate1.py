from pathlib import Path

from snu_ordering.candidate1.artifacts import ArtifactLayout
from snu_ordering.candidate1.config import Candidate1Config
from snu_ordering.candidate1.data import build_messages, validate_image_grid_count
from snu_ordering.candidate1.inference import class_ids_to_answers
from snu_ordering.candidate1.model import validate_trainable_parameters


def test_candidate1_defaults_lock_the_reference_design():
    config = Candidate1Config()

    assert config.model.base_model_path == "Qwen/Qwen2-VL-2B-Instruct"
    assert config.model.num_classes == 24
    assert config.quantization.enabled is True
    assert config.quantization.quant_type == "nf4"
    assert config.quantization.double_quant is True
    assert config.lora.rank == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == 0.05
    assert config.lora.target_modules == ("q_proj", "v_proj")
    assert config.training.tiny_subset_size == 16
    assert config.training.tiny_success_accuracy == 0.95


def test_candidate1_config_round_trips_as_json(tmp_path):
    config = Candidate1Config()
    path = tmp_path / "run_config.json"

    config.save(path)
    loaded = Candidate1Config.load(path)

    assert loaded == config


def test_multimodal_prompt_keeps_full_sentence_and_four_images_in_input_order(tmp_path):
    row = {
        "Id": "abc123",
        "Sentence": "First the door opens. Then the person enters.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
    }

    messages = build_messages(row, tmp_path, min_pixels=100, max_pixels=200)
    content = messages[0]["content"]
    images = [item for item in content if item["type"] == "image"]
    text = "\n".join(item["text"] for item in content if item["type"] == "text")

    assert [Path(item["image"]).name for item in images] == [
        "one.jpg",
        "two.jpg",
        "three.jpg",
        "four.jpg",
    ]
    assert all(item["min_pixels"] == 100 and item["max_pixels"] == 200 for item in images)
    assert row["Sentence"] in text
    assert "class_id" in text


def test_class_ids_use_canonical_repository_mapping():
    assert class_ids_to_answers([0, 1, 23]) == [
        (1, 2, 3, 4),
        (1, 2, 4, 3),
        (4, 3, 2, 1),
    ]


def test_multimodal_batch_requires_four_image_grids_per_sample():
    class Grid:
        shape = (8, 3)

    validate_image_grid_count(Grid(), batch_size=2)

    class MissingGrid:
        shape = (7, 3)

    import pytest

    with pytest.raises(ValueError, match="exactly four images"):
        validate_image_grid_count(MissingGrid(), batch_size=2)


def test_trainability_guard_accepts_only_lora_and_classifier_parameters():
    class Parameter:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad

    class Model:
        def named_parameters(self):
            return iter(
                [
                    ("backbone.base.weight", Parameter(False)),
                    ("backbone.q_proj.lora_A.default.weight", Parameter(True)),
                    ("classifier.0.weight", Parameter(True)),
                    ("classifier.2.weight", Parameter(True)),
                ]
            )

    validate_trainable_parameters(Model())


def test_artifact_layout_is_compact_and_inspectable(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    layout.create()

    assert layout.adapter_dir.is_dir()
    assert layout.config_path == layout.root / "run_config.json"
    assert layout.classifier_head_path == layout.root / "classifier_head.pt"
    assert layout.metrics_path == layout.root / "training_metrics.json"
    assert layout.memory_path == layout.root / "memory_report.json"
    assert layout.trainer_state_path == layout.root / "trainer_state.pt"
