"""Configuration for Candidate Model 1: Qwen2-VL + QLoRA + 24-class head."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    processor_path: str | None = None
    num_classes: int = 24
    classifier_dropout: float = 0.0
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
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.15
    log_every_steps: int = 10
    save_every_steps: int = 100
    gradient_checkpointing: bool = True
    tiny_subset_size: int = 16
    tiny_max_steps: int = 400
    tiny_success_accuracy: float = 0.95


@dataclass(frozen=True)
class Candidate1Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.model.num_classes != 24:
            raise ValueError("Candidate Model 1 must use the canonical 24 classes")
        if not self.lora.target_modules:
            raise ValueError("At least one LoRA target module is required")
        if self.training.tiny_subset_size < 1:
            raise ValueError("tiny_subset_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Candidate1Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        lora = dict(raw.get("lora", {}))
        if "target_modules" in lora:
            lora["target_modules"] = tuple(lora["target_modules"])
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            quantization=QuantizationConfig(**raw.get("quantization", {})),
            lora=LoraSettings(**lora),
            training=TrainingConfig(**raw.get("training", {})),
        )
