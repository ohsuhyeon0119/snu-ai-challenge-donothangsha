"""Qwen2-VL backbone, QLoRA adapters, and the 24-class head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Candidate1Config


def _compute_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if requested not in choices:
        raise ValueError(f"Unsupported compute dtype: {requested}")
    return choices[requested]


def _hidden_size(backbone: Any) -> int:
    config = backbone.config
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(config, "hidden_size", None) or getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine Qwen2-VL language hidden size from model config")
    return int(hidden_size)


def _base_model_kwargs(config: Candidate1Config, *, local_files_only: bool) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    dtype = _compute_dtype(torch, config.quantization.compute_dtype)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
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
    return AutoProcessor.from_pretrained(
        path,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        local_files_only=local_files_only,
        trust_remote_code=config.model.trust_remote_code,
    )


def load_base_backbone(config: Candidate1Config, *, local_files_only: bool) -> Any:
    from transformers import Qwen2VLForConditionalGeneration

    return Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.base_model_path,
        **_base_model_kwargs(config, local_files_only=local_files_only),
    )


def attach_trainable_lora(backbone: Any, config: Candidate1Config) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if config.quantization.enabled:
        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_config)
    if config.training.gradient_checkpointing:
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    return backbone


def validate_trainable_parameters(model: Any) -> None:
    """Ensure Candidate 1 trains LoRA and the head, but no base weights."""
    named = list(model.named_parameters())
    trainable = [name for name, parameter in named if parameter.requires_grad]
    if not any("lora_" in name for name in trainable):
        raise RuntimeError("No trainable LoRA parameters were found")
    classifier = [(name, parameter) for name, parameter in named if name.startswith("classifier.")]
    if not classifier or not all(parameter.requires_grad for _, parameter in classifier):
        raise RuntimeError("Every classifier-head parameter must be trainable")
    unexpected = [
        name
        for name in trainable
        if "lora_" not in name and not name.startswith("classifier.")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable base-model parameters: {unexpected}")


def extract_multimodal_hidden_state(backbone: Any, inputs: dict[str, Any]) -> Any:
    """Run Qwen2-VL multimodal fusion and return its final decoder state.

    This mirrors the image-token insertion and mRoPE setup in Transformers'
    Qwen2-VL conditional-generation forward, but calls the decoder directly.
    It therefore avoids allocating unused vocabulary logits and retaining every
    hidden layer for this classification-only pipeline.
    """
    core = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    inputs_embeds = core.model.embed_tokens(input_ids)

    def insert_visual_tokens(pixel_key: str, grid: Any, token_id: int) -> None:
        nonlocal inputs_embeds
        pixel_values = inputs.get(pixel_key)
        if pixel_values is None:
            return
        pixel_values = pixel_values.type(core.visual.get_dtype())
        visual_embeds = core.visual(pixel_values, grid_thw=grid)
        token_mask = input_ids == token_id
        token_count = int(token_mask.sum().item())
        feature_count = int(visual_embeds.shape[0])
        if token_count != feature_count:
            raise ValueError(
                f"Visual token/feature mismatch for {pixel_key}: "
                f"{token_count} tokens versus {feature_count} features"
            )
        expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        visual_embeds = visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, visual_embeds)

    insert_visual_tokens("pixel_values", image_grid_thw, core.config.image_token_id)
    if "pixel_values_videos" in inputs:
        insert_visual_tokens("pixel_values_videos", video_grid_thw, core.config.video_token_id)

    position_ids, _ = core.get_rope_index(
        input_ids, image_grid_thw, video_grid_thw, attention_mask
    )
    outputs = core.model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    return outputs.last_hidden_state


def create_classifier(backbone: Any, config: Candidate1Config) -> Any:
    import torch
    from torch import nn

    from .pooling import pool_last_non_padding

    class QwenPermutationClassifier(nn.Module):
        """PEFT Qwen2-VL with LayerNorm -> optional Dropout -> Linear(24)."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            hidden_size = _hidden_size(backbone)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(config.model.classifier_dropout),
                nn.Linear(hidden_size, config.model.num_classes),
            )

        @property
        def device(self) -> Any:
            return next(self.classifier.parameters()).device

        def forward(self, labels: Any = None, **inputs: Any) -> Any:
            from transformers.modeling_outputs import SequenceClassifierOutput

            final_hidden_state = extract_multimodal_hidden_state(self.backbone, inputs)
            pooled = pool_last_non_padding(final_hidden_state, inputs["attention_mask"])
            logits = self.classifier(pooled.to(next(self.classifier.parameters()).dtype))
            loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)

    model = QwenPermutationClassifier()
    model.classifier.to(device=getattr(backbone, "device", None) or "cpu")
    return model


def load_training_model(config: Candidate1Config, *, local_files_only: bool = False) -> Any:
    backbone = attach_trainable_lora(
        load_base_backbone(config, local_files_only=local_files_only), config
    )
    return create_classifier(backbone, config)


def load_resumed_training_model(
    config: Candidate1Config,
    *,
    adapter_path: str | Path,
    classifier_head_path: str | Path,
    local_files_only: bool = False,
) -> Any:
    """Reload adapters and head as trainable parameters for optimizer resume."""
    import torch
    from peft import PeftModel

    base = load_base_backbone(config, local_files_only=local_files_only)
    backbone = PeftModel.from_pretrained(
        base, str(adapter_path), is_trainable=True, local_files_only=local_files_only
    )
    if config.training.gradient_checkpointing:
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
    model = create_classifier(backbone, config)
    state = torch.load(classifier_head_path, map_location=model.device, weights_only=True)
    model.classifier.load_state_dict(state)
    return model


def load_inference_model(
    config: Candidate1Config,
    *,
    adapter_path: str | Path,
    classifier_head_path: str | Path,
    local_files_only: bool = True,
) -> Any:
    import torch
    from peft import PeftModel

    base = load_base_backbone(config, local_files_only=local_files_only)
    backbone = PeftModel.from_pretrained(
        base, str(adapter_path), is_trainable=False, local_files_only=local_files_only
    )
    metadata_path = Path(classifier_head_path).with_name("metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "num_classes": config.model.num_classes,
            "hidden_size": _hidden_size(backbone),
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint metadata does not match inference config: {mismatches}")
    model = create_classifier(backbone, config)
    state = torch.load(classifier_head_path, map_location=model.device, weights_only=True)
    model.classifier.load_state_dict(state)
    return model
