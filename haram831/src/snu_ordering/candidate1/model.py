"""Qwen2-VL causal-LM loading and QLoRA adapter management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ARCHITECTURE_VERSION, Candidate1Config, LEGACY_ARCHITECTURE_VERSION


def _compute_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested not in choices:
        raise ValueError(f"Unsupported compute dtype: {requested}")
    return choices[requested]


def _base_model_kwargs(config: Candidate1Config, *, local_files_only: bool) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    dtype = _compute_dtype(torch, config.quantization.compute_dtype)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.quantization.enabled:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    return kwargs


def load_processor(config: Candidate1Config, *, local_files_only: bool) -> Any:
    from transformers import AutoProcessor

    path = config.model.processor_path or config.model.base_model_path
    processor = AutoProcessor.from_pretrained(
        path,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        local_files_only=local_files_only,
        trust_remote_code=config.model.trust_remote_code,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_base_model(config: Candidate1Config, *, local_files_only: bool) -> Any:
    from transformers import Qwen2VLForConditionalGeneration

    return Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.base_model_path,
        **_base_model_kwargs(config, local_files_only=local_files_only),
    )


def _enable_training_features(model: Any, config: Candidate1Config) -> Any:
    if config.training.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def load_training_model(config: Candidate1Config, *, local_files_only: bool = False) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    base = load_base_model(config, local_files_only=local_files_only)
    if config.quantization.enabled:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return _enable_training_features(get_peft_model(base, lora_config), config)


def validate_trainable_parameters(model: Any) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not any("lora_" in name for name in trainable):
        raise RuntimeError("No trainable LoRA parameters were found")
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable base-model parameters: {unexpected}")


def validate_checkpoint_architecture(
    config: Candidate1Config, checkpoint_root: str | Path
) -> dict[str, Any]:
    root = Path(checkpoint_root)
    metadata_path = root / "metadata.json" if root.is_dir() else root.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_version = metadata.get(
        "architecture_version", LEGACY_ARCHITECTURE_VERSION
    )
    if checkpoint_version != ARCHITECTURE_VERSION or checkpoint_version != config.architecture_version:
        raise ValueError(
            "Checkpoint architecture does not match causal-LM v3: "
            f"checkpoint={checkpoint_version!r}, config={config.architecture_version!r}"
        )
    return metadata


def load_resumed_training_model(
    config: Candidate1Config,
    *,
    adapter_path: str | Path,
    checkpoint_root: str | Path,
    local_files_only: bool = False,
) -> Any:
    from peft import PeftModel

    validate_checkpoint_architecture(config, checkpoint_root)
    base = load_base_model(config, local_files_only=local_files_only)
    model = PeftModel.from_pretrained(
        base, str(adapter_path), is_trainable=True, local_files_only=local_files_only
    )
    return _enable_training_features(model, config)


def load_inference_model(
    config: Candidate1Config,
    *,
    adapter_path: str | Path,
    checkpoint_root: str | Path,
    local_files_only: bool = True,
) -> Any:
    from peft import PeftModel

    validate_checkpoint_architecture(config, checkpoint_root)
    base = load_base_model(config, local_files_only=local_files_only)
    model = PeftModel.from_pretrained(
        base, str(adapter_path), is_trainable=False, local_files_only=local_files_only
    )
    model.config.use_cache = True
    return model
