"""Swappable multimodal hidden-state pooling rules."""

from __future__ import annotations

from typing import Any


def pool_last_non_padding(final_hidden_state: Any, attention_mask: Any) -> Any:
    """Pool the final hidden state at each sample's last non-padding token.

    Candidate Model 1 deliberately uses this explicit representation: run the
    complete four-image/full-caption prompt through Qwen2-VL, select the final
    language-model hidden layer, then select the token with the greatest index
    whose attention mask is 1. This supports both left and right padding and is
    isolated here so later candidates can replace the pooling rule.
    """
    if final_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden states [batch, sequence, hidden] and mask [batch, sequence]")
    if final_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden-state and attention-mask batch/sequence dimensions must match")
    if not bool(attention_mask.bool().any(dim=1).all().item()):
        raise ValueError("Every sample must contain at least one non-padding token")

    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_indices = (attention_mask.to(dtype=torch.long) * (positions + 1)).argmax(dim=1)
    last_indices = last_indices.to(final_hidden_state.device)
    batch_indices = torch.arange(final_hidden_state.shape[0], device=final_hidden_state.device)
    return final_hidden_state[batch_indices, last_indices]
