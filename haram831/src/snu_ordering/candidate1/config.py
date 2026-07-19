"""Configuration for Candidate Model 1 v3: Qwen2-VL answer generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ARCHITECTURE_VERSION = "candidate1_causal_lm_v3"
LEGACY_ARCHITECTURE_VERSION = "candidate1_classifier_v1"


@dataclass(frozen=True)
class ModelConfig:
    base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    processor_path: str | None = None
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 256 * 28 * 28
    trust_remote_code: bool = False


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "auto"


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.12
    fast_validation_every_steps: int = 400
    fast_validation_size: int = 120
    epoch_validation_size: int = 160
    scoring_chunk_size: int = 12
    generation_max_new_tokens: int = 32
    collapse_threshold: float = 0.5
    success_accuracy: float = 0.25
    log_every_steps: int = 20
    save_every_steps: int = 100
    gradient_checkpointing: bool = True
    tiny_subset_size: int = 16
    tiny_max_steps: int = 400
    tiny_success_accuracy: float = 0.95


@dataclass(frozen=True)
class Candidate1Config:
    architecture_version: str = ARCHITECTURE_VERSION
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                "Candidate1 config architecture is incompatible with causal-LM v3: "
                f"{self.architecture_version!r}"
            )
        if not self.lora.target_modules:
            raise ValueError("At least one LoRA target module is required")
        if self.training.epochs < 1 or self.training.tiny_subset_size < 1:
            raise ValueError("Epoch and subset sizes must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.training.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        for name in (
            "fast_validation_every_steps",
            "fast_validation_size",
            "epoch_validation_size",
            "scoring_chunk_size",
            "generation_max_new_tokens",
        ):
            if getattr(self.training, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.training.collapse_threshold <= 1.0:
            raise ValueError("collapse_threshold must be in (0, 1]")
        if not 0.0 < self.training.success_accuracy <= 1.0:
            raise ValueError("success_accuracy must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Candidate1Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        version = raw.get("architecture_version", LEGACY_ARCHITECTURE_VERSION)
        if version != ARCHITECTURE_VERSION:
            raise ValueError(
                "Candidate1 config architecture is incompatible with causal-LM v3: "
                f"{version!r}; start a fresh v3 run"
            )
        lora = dict(raw.get("lora", {}))
        if "target_modules" in lora:
            lora["target_modules"] = tuple(lora["target_modules"])
        training = dict(raw.get("training", {}))
        legacy_validation_size = training.pop("constrained_validation_size", None)
        if legacy_validation_size is not None and "epoch_validation_size" not in training:
            training["epoch_validation_size"] = legacy_validation_size
        return cls(
            architecture_version=version,
            model=ModelConfig(**raw.get("model", {})),
            quantization=QuantizationConfig(**raw.get("quantization", {})),
            lora=LoraSettings(**lora),
            training=TrainingConfig(**training),
        )
