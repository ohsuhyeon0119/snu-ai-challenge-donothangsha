"""Four-image prompts and completion-only causal-LM batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from ..caption_structure import render_punctuation_hints, render_relation_hints
from ..dataset import INPUT_COLUMNS, resolve_image_paths
from ..permutation import answer_to_chronological_order, validate_permutation

TASK_INSTRUCTION = (
    "The caption describes events in chronological order. The four images above "
    "are shuffled frames from that sequence. Return the image numbers from earliest "
    "to latest as a list such as [3, 1, 4, 2]."
)
COMPLETION_TEMPLATE = "The correct chronological order is {order}."


def render_instruction(
    sentence: str,
    caption_prompt_mode: str,
    *,
    relation_confidence_threshold: float = 0.7,
    boundary_dropout: float = 0.0,
    boundary_dropout_seed: str | int | None = None,
) -> str:
    """Render the unchanged A0, punctuation A1, or relation-aware A2 prompt."""

    if caption_prompt_mode == "raw":
        caption_context = f'Caption: "{sentence}"'
    elif caption_prompt_mode == "punctuation":
        caption_context = render_punctuation_hints(sentence)
    elif caption_prompt_mode == "relations":
        caption_context = render_relation_hints(
            sentence,
            confidence_threshold=relation_confidence_threshold,
            boundary_dropout=boundary_dropout,
            dropout_seed=boundary_dropout_seed,
        )
    else:
        raise ValueError(f"Unsupported caption prompt mode: {caption_prompt_mode}")
    return f"{caption_context}\n\n{TASK_INSTRUCTION}"


def build_messages(
    row: Mapping[str, Any],
    image_root: str | Path,
    *,
    min_pixels: int,
    max_pixels: int,
    caption_prompt_mode: str = "raw",
    relation_confidence_threshold: float = 0.7,
    boundary_dropout: float = 0.0,
    boundary_dropout_seed: str | int | None = None,
) -> list[dict[str, Any]]:
    """Build one multimodal user message, preserving Input_1..Input_4 order."""
    content: list[dict[str, Any]] = []
    sample_dir = Path(image_root) / str(row["Id"])
    for index, column in enumerate(INPUT_COLUMNS, start=1):
        content.append(
            {
                "type": "image",
                "image": str(sample_dir / str(row[column])),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
        content.append({"type": "text", "text": f"Image {index}"})
    content.append(
        {
            "type": "text",
            "text": render_instruction(
                str(row["Sentence"]),
                caption_prompt_mode,
                relation_confidence_threshold=relation_confidence_threshold,
                boundary_dropout=boundary_dropout,
                boundary_dropout_seed=boundary_dropout_seed,
            ),
        }
    )
    return [{"role": "user", "content": content}]


def render_chat_prompt(processor: Any, messages: list[dict[str, Any]]) -> str:
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def chronological_order_for_row(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    if "answer_tuple" not in row:
        raise ValueError("Training row is missing answer_tuple")
    return answer_to_chronological_order(row["answer_tuple"])


PAIRWISE_FRAME_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def pairwise_labels_for_row(row: Mapping[str, Any]) -> tuple[float, ...]:
    """Return six before/after labels in canonical input-frame pair order."""

    if "answer_tuple" not in row:
        raise ValueError("Training row is missing answer_tuple")
    answer = validate_permutation(row["answer_tuple"], name="competition answer")
    return tuple(
        float(answer[left] < answer[right])
        for left, right in PAIRWISE_FRAME_PAIRS
    )


def target_text(order: Sequence[int]) -> str:
    validated = validate_permutation(order, name="chronological image order")
    rendered = "[" + ", ".join(str(value) for value in validated) + "]"
    return COMPLETION_TEMPLATE.format(order=rendered)


def validate_image_grid_count(image_grid_thw: Any, *, batch_size: int) -> None:
    expected = 4 * batch_size
    actual = int(image_grid_thw.shape[0])
    if actual != expected:
        raise ValueError(
            "Candidate Model 1 requires exactly four images per sample; "
            f"processor returned {actual} image grids for batch size {batch_size} "
            f"(expected {expected})"
        )


def collate_rows(
    rows: Sequence[Mapping[str, Any]],
    processor: Any,
    image_root: str | Path,
    *,
    min_pixels: int,
    max_pixels: int,
    include_labels: bool,
    caption_prompt_mode: str = "raw",
    relation_confidence_threshold: float = 0.7,
    boundary_dropout: float = 0.0,
    boundary_dropout_seed: str | int | None = None,
    include_pairwise_labels: bool = False,
) -> dict[str, Any]:
    """Build a right-padded multimodal batch with prompt tokens masked from loss."""
    import torch
    from qwen_vl_utils import process_vision_info

    full_texts: list[str] = []
    prompt_texts: list[str] = []
    images: list[Any] = []
    for row in rows:
        row_dropout_seed = (
            None
            if boundary_dropout_seed is None
            else f"{boundary_dropout_seed}:{row['Id']}"
        )
        messages = build_messages(
            row,
            image_root,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            caption_prompt_mode=caption_prompt_mode,
            relation_confidence_threshold=relation_confidence_threshold,
            boundary_dropout=boundary_dropout,
            boundary_dropout_seed=row_dropout_seed,
        )
        prompt = render_chat_prompt(processor, messages)
        prompt_texts.append(prompt)
        completion = ""
        if include_labels:
            completion = target_text(chronological_order_for_row(row))
            completion += processor.tokenizer.eos_token
        full_texts.append(prompt + completion)
        image_inputs, _ = process_vision_info(messages)
        images.extend(image_inputs)

    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"
    try:
        batch = processor(
            text=full_texts, images=images, padding=True, return_tensors="pt"
        )
        if include_labels:
            prompt_batch = processor(
                text=prompt_texts, images=images, padding=True, return_tensors="pt"
            )
    finally:
        processor.tokenizer.padding_side = previous_padding_side

    validate_image_grid_count(batch["image_grid_thw"], batch_size=len(rows))
    if include_labels:
        labels = batch["input_ids"].clone()
        for index in range(len(rows)):
            prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
            labels[index, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not bool((labels != -100).any(dim=1).all().item()):
            raise ValueError("Every training sample must supervise completion tokens")
        batch["labels"] = labels.to(dtype=torch.long)
    if include_pairwise_labels:
        if include_labels:
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        else:
            prompt_lengths = batch["attention_mask"].sum(dim=1)
        batch["prompt_lengths"] = prompt_lengths.to(dtype=torch.long)
        batch["pairwise_labels"] = torch.tensor(
            [pairwise_labels_for_row(row) for row in rows], dtype=torch.float32
        )
    return batch


def load_rgb_images(row: Mapping[str, Any], image_root: str | Path) -> list[Image.Image]:
    opened: list[Image.Image] = []
    for path in resolve_image_paths(row, image_root):
        with Image.open(path) as image:
            opened.append(image.convert("RGB").copy())
    return opened
