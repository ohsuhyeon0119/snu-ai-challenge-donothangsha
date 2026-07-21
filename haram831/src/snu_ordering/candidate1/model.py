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


def _causal_lm_base(model: Any) -> Any:
    """Return the adapter-injected causal LM without disabling active LoRA modules."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise ValueError(
            "Candidate1 optimized training requires a causal LM with model and lm_head modules"
        )
    return base


def final_hidden_state(model: Any, model_inputs: dict[str, Any]) -> Any:
    """Run one multimodal backbone pass and return only the final decoder state.

    Calling the Qwen backbone directly avoids both the full-vocabulary logits and
    the tuple of every decoder layer's hidden state. LoRA layers remain active
    because PEFT injects them into the returned base model in place.
    """

    base = _causal_lm_base(model)
    outputs = base.model(
        **model_inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs[0]
    if hidden.ndim != 3:
        raise ValueError("Expected final hidden state with shape [batch, sequence, hidden]")
    return hidden


def completion_only_training_forward(
    model: Any, batch: dict[str, Any]
) -> tuple[Any, Any, dict[str, int]]:
    """Compute causal-LM loss without materializing prompt/image-token logits.

    The collator masks prompt and padding labels with ``-100``. For causal LM
    training, hidden position ``t`` predicts label ``t + 1``; therefore only
    states whose shifted labels are supervised need to pass through ``lm_head``.
    The complete final hidden tensor is returned for A2's prompt-boundary
    pairwise auxiliary loss.
    """

    import torch.nn.functional as functional

    labels = batch.get("labels")
    if labels is None:
        raise ValueError("completion-only training requires labels")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")

    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    hidden = final_hidden_state(model, model_inputs)
    if tuple(hidden.shape[:2]) != tuple(labels.shape):
        raise ValueError("labels must match the final hidden batch and sequence dimensions")

    shifted_labels = labels[:, 1:].contiguous()
    supervised_mask = shifted_labels.ne(-100)
    supervised_tokens = int(supervised_mask.sum().item())
    if supervised_tokens == 0:
        raise ValueError("completion-only training found no supervised target tokens")

    prediction_hidden = hidden[:, :-1, :][supervised_mask]
    targets = shifted_labels[supervised_mask].to(prediction_hidden.device)
    base = _causal_lm_base(model)
    logits = base.lm_head(prediction_hidden)
    loss = functional.cross_entropy(logits.float(), targets)
    stats = {
        "sequence_tokens": int(labels.numel()),
        "supervised_tokens": supervised_tokens,
        "logit_rows": int(logits.shape[0]),
        "vocabulary_size": int(logits.shape[-1]),
    }
    return loss, hidden, stats


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
