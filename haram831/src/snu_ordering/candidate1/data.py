"""Four-image prompt construction and processor collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from ..dataset import INPUT_COLUMNS, resolve_image_paths

INSTRUCTION = (
    "Caption:\n{sentence}\n\n"
    "The four frames above are Input_1 through Input_4 in shuffled order. "
    "Use the complete caption and all four frames to identify their original temporal positions. "
    "Return [p1, p2, p3, p4], where pi is the original temporal position of Input_i. "
    "For example, [1, 4, 2, 3] means Input_1 is first, Input_2 is fourth, "
    "Input_3 is second, and Input_4 is third. If the frames are already chronological, "
    "return [1, 2, 3, 4]."
)


def build_messages(
    row: Mapping[str, Any],
    image_root: str | Path,
    *,
    min_pixels: int,
    max_pixels: int,
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
        content.append({"type": "text", "text": f"Frame {column} (input position {index})."})
    content.append(
        {"type": "text", "text": INSTRUCTION.format(sentence=str(row["Sentence"]))}
    )
    return [{"role": "user", "content": content}]


def validate_image_grid_count(image_grid_thw: Any, *, batch_size: int) -> None:
    """Reject processor batches that do not contain four images per sample."""
    expected = 4 * batch_size
    actual = int(image_grid_thw.shape[0])
    if actual != expected:
        raise ValueError(
            f"Candidate Model 1 requires exactly four images per sample; "
            f"processor returned {actual} image grids for batch size {batch_size} "
            f"(expected {expected})"
        )


def render_chat_prompt(processor: Any, messages: list[dict[str, Any]]) -> str:
    """Render the state from which Qwen would generate its ordering answer."""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def collate_rows(
    rows: Sequence[Mapping[str, Any]],
    processor: Any,
    image_root: str | Path,
    *,
    min_pixels: int,
    max_pixels: int,
    include_labels: bool,
) -> dict[str, Any]:
    """Convert rows into one padded Qwen2-VL processor batch.

    ``qwen_vl_utils`` is imported lazily so metadata/config commands remain usable
    on CPU-only machines where the training stack is not installed.
    """
    from qwen_vl_utils import process_vision_info

    texts: list[str] = []
    images: list[Any] = []
    for row in rows:
        messages = build_messages(
            row, image_root, min_pixels=min_pixels, max_pixels=max_pixels
        )
        texts.append(render_chat_prompt(processor, messages))
        image_inputs, _ = process_vision_info(messages)
        images.extend(image_inputs)

    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    validate_image_grid_count(batch["image_grid_thw"], batch_size=len(rows))
    if include_labels:
        import torch

        batch["labels"] = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    return batch


def load_rgb_images(row: Mapping[str, Any], image_root: str | Path) -> list[Image.Image]:
    """Load detached RGB copies; useful for integrity/debug tooling."""
    opened: list[Image.Image] = []
    for path in resolve_image_paths(row, image_root):
        with Image.open(path) as image:
            opened.append(image.convert("RGB").copy())
    return opened
