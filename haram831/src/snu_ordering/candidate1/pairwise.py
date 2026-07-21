"""Training-only pairwise temporal auxiliary head for Candidate1 A2."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden_size = getattr(getattr(base, "config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine the language-model hidden size")
    return int(hidden_size)


def create_pairwise_head(model: Any) -> Any:
    import torch

    return torch.nn.Linear(model_hidden_size(model), 6).to(model.device)


def compute_pairwise_auxiliary(
    hidden: Any,
    pairwise_head: Any,
    prompt_lengths: Any,
    pairwise_labels: Any,
) -> tuple[Any, Any]:
    """Compute six binary before/after logits from the final prompt token."""

    import torch
    import torch.nn.functional as functional

    if hidden.ndim != 3:
        raise ValueError("Expected hidden states with shape [batch, sequence, hidden]")
    if prompt_lengths.ndim != 1 or len(prompt_lengths) != hidden.shape[0]:
        raise ValueError("prompt_lengths must contain one value per batch row")
    if tuple(pairwise_labels.shape) != (hidden.shape[0], 6):
        raise ValueError("pairwise_labels must have shape [batch, 6]")

    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    token_indices = prompt_lengths.to(hidden.device) - 1
    if bool((token_indices < 0).any().item()):
        raise ValueError("prompt lengths must be positive")
    prompt_hidden = hidden[batch_indices, token_indices]
    logits = pairwise_head(prompt_hidden.float())
    loss = functional.binary_cross_entropy_with_logits(
        logits, pairwise_labels.to(logits.device, dtype=logits.dtype)
    )
    return loss, logits


def combine_training_losses(lm_loss: Any, pairwise_loss: Any, weight: float) -> Any:
    if weight < 0.0:
        raise ValueError("pairwise loss weight must be non-negative")
    return lm_loss + weight * pairwise_loss


def save_pairwise_head(pairwise_head: Any, path: str | Path) -> None:
    from safetensors.torch import save_file

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in pairwise_head.state_dict().items()
    }
    save_file(state, str(output))


def load_pairwise_head(pairwise_head: Any, path: str | Path) -> None:
    from safetensors.torch import load_file

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Pairwise head checkpoint does not exist: {source}")
    pairwise_head.load_state_dict(load_file(str(source), device="cpu"))
