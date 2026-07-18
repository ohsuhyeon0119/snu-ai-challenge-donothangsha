"""Greedy and constrained prediction for chronological image orders."""

from __future__ import annotations

import re
from itertools import permutations
from typing import Any

from ..permutation import Permutation, chronological_order_to_answer
from .data import render_chat_prompt, target_text

ALL_ORDERS: tuple[Permutation, ...] = tuple(permutations((1, 2, 3, 4)))
_ORDER_RE = re.compile(r"\[([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\]")


def parse_generated_order(text: str) -> Permutation | None:
    match = _ORDER_RE.search(text)
    if match is None:
        return None
    order = tuple(int(match.group(index)) for index in range(1, 5))
    return order if sorted(order) == [1, 2, 3, 4] else None  # type: ignore[return-value]


def _completion_logits(model: Any, inputs: Any, prompt_length: int) -> Any:
    """Run multimodal fusion but project only completion positions to vocabulary."""
    visual_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    core = visual_model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs.get("image_grid_thw")
    inputs_embeds = core.get_input_embeddings()(input_ids)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.type(visual_model.visual.get_dtype())
        visual_embeds = visual_model.visual(pixel_values, grid_thw=image_grid_thw)
        token_mask = input_ids == core.config.image_token_id
        if int(token_mask.sum()) != int(visual_embeds.shape[0]):
            raise ValueError("Visual token and feature counts do not match")
        expanded = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded, visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        )
    position_ids, _ = core.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask
    )
    hidden = core(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    return visual_model.lm_head(hidden[:, prompt_length - 1 : -1])


def completion_log_likelihoods(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> list[float]:
    """Score all 24 valid chronological orders using completion tokens only."""
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = render_chat_prompt(processor, messages)
    image_inputs, _ = process_vision_info(messages)
    prompt_batch = processor(text=[prompt], images=image_inputs, return_tensors="pt")
    prompt_length = int(prompt_batch["attention_mask"][0].sum().item())
    suffixes = [target_text(order) + processor.tokenizer.eos_token for order in ALL_ORDERS]
    scores: list[float] = []
    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"
    try:
        for start in range(0, len(suffixes), chunk_size):
            chunk = suffixes[start : start + chunk_size]
            inputs = processor(
                text=[prompt + suffix for suffix in chunk],
                images=image_inputs * len(chunk),
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            shifted_logits = _completion_logits(model, inputs, prompt_length).float()
            shifted_tokens = inputs["input_ids"][:, prompt_length:]
            shifted_mask = inputs["attention_mask"][:, prompt_length:].bool()
            token_logp = torch.log_softmax(shifted_logits, dim=-1).gather(
                -1, shifted_tokens.unsqueeze(-1)
            ).squeeze(-1)
            scores.extend((token_logp * shifted_mask).sum(dim=-1).cpu().tolist())
    finally:
        processor.tokenizer.padding_side = previous_padding_side
    if len(scores) != len(ALL_ORDERS):
        raise RuntimeError(f"Expected 24 candidate scores, got {len(scores)}")
    return scores


def score_order(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> tuple[Permutation, list[float]]:
    scores = completion_log_likelihoods(
        model, processor, messages, chunk_size=chunk_size
    )
    best = max(range(len(scores)), key=scores.__getitem__)
    return ALL_ORDERS[best], scores


def generate_order(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> Permutation | None:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = render_chat_prompt(processor, messages)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    completion = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    return parse_generated_order(completion)


def order_to_answer(order: Permutation) -> Permutation:
    return chronological_order_to_answer(order)
