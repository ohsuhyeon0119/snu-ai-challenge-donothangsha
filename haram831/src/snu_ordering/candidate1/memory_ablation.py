"""Measure Candidate1 A2 memory under four controlled joint-loss variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..dataset import load_metadata
from .config import Candidate1Config


EXPERIMENTS = {
    "A": "standard LM forward; hidden states off; pairwise loss off",
    "B": "standard LM forward; all hidden states returned; pairwise loss off",
    "C": "legacy A2; all hidden states returned; pairwise loss on",
    "D": "optimized A2; final hidden only; completion logits only; pairwise loss on",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--processor")
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def select_row(rows: list[dict[str, Any]], sample_id: str | None, sample_index: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Training CSV contains no rows")
    if sample_id is not None:
        matches = [row for row in rows if str(row["Id"]) == str(sample_id)]
        if not matches:
            raise ValueError(f"Sample ID does not exist in training CSV: {sample_id}")
        return matches[0]
    if sample_index < 0 or sample_index >= len(rows):
        raise ValueError(
            f"sample-index must be in [0, {len(rows) - 1}], got {sample_index}"
        )
    return rows[sample_index]


def configured_run(args: argparse.Namespace) -> Candidate1Config:
    config = Candidate1Config.load(args.config)
    return replace(
        config,
        model=replace(
            config.model,
            base_model_path=args.base_model or config.model.base_model_path,
            processor_path=(
                args.processor if args.processor is not None else config.model.processor_path
            ),
        ),
        # Keep all four experiments byte-for-byte comparable at the prompt level.
        caption_prompt=replace(config.caption_prompt, boundary_dropout=0.0),
    )


def main() -> None:
    args = parse_args()

    import torch

    from .data import collate_rows
    from .memory import GpuMemoryReporter
    from .model import (
        completion_only_training_forward,
        load_processor,
        load_training_model,
        validate_trainable_parameters,
    )
    from .pairwise import (
        combine_training_losses,
        compute_pairwise_auxiliary,
        create_pairwise_head,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the memory ablation")

    config = configured_run(args)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    rows = load_metadata(args.train_csv, split="train").to_dict(orient="records")
    row = select_row(rows, args.sample_id, args.sample_index)

    reporter = GpuMemoryReporter()
    reporter.record_values(
        "experiment",
        {
            "mode": args.mode,
            "description": EXPERIMENTS[args.mode],
            "sample_id": str(row["Id"]),
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
            "pairwise_loss_weight": config.training.pairwise_loss_weight,
            "caption_prompt_mode": config.caption_prompt.mode,
            "boundary_dropout": 0.0,
        },
    )

    processor = load_processor(config, local_files_only=args.local_files_only)
    model = load_training_model(config, local_files_only=args.local_files_only)
    validate_trainable_parameters(model)
    model.train()
    pairwise_head = (
        create_pairwise_head(model) if args.mode in {"C", "D"} else None
    )
    if pairwise_head is not None:
        pairwise_head.train()
    reporter.record_current("after_model_load")

    batch = collate_rows(
        [row],
        processor,
        args.image_root,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_labels=True,
        caption_prompt_mode=config.caption_prompt.mode,
        relation_confidence_threshold=config.caption_prompt.relation_confidence_threshold,
        boundary_dropout=0.0,
        include_pairwise_labels=True,
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    reporter.record_batch("batch", batch)
    reporter.record_current("after_batch_load")
    pairwise_labels = batch.pop("pairwise_labels")
    prompt_lengths = batch.pop("prompt_lengths")
    output_path = Path(args.output_json)

    model.zero_grad(set_to_none=True)
    if pairwise_head is not None:
        pairwise_head.zero_grad(set_to_none=True)
    reporter.reset_peak()
    retained: Any = None
    try:
        if args.mode == "D":
            lm_loss, final_hidden, work = completion_only_training_forward(model, batch)
            retained = final_hidden
            reporter.record_values(
                "forward_work",
                {
                    **work,
                    "avoided_logit_rows": work["sequence_tokens"] - work["logit_rows"],
                    "returned_hidden_layers": 0,
                },
            )
        else:
            return_hidden = args.mode in {"B", "C"}
            retained = model(
                **batch,
                output_hidden_states=return_hidden,
                use_cache=False,
                return_dict=True,
            )
            lm_loss = retained.loss
            reporter.record_values(
                "forward_work",
                {
                    "sequence_tokens": int(batch["labels"].numel()),
                    "supervised_tokens": int(batch["labels"].ne(-100).sum().item()),
                    "logit_rows": int(retained.logits.numel() // retained.logits.shape[-1]),
                    "vocabulary_size": int(retained.logits.shape[-1]),
                    "avoided_logit_rows": 0,
                    "returned_hidden_layers": (
                        0 if retained.hidden_states is None else len(retained.hidden_states)
                    ),
                },
            )
            final_hidden = (
                None if retained.hidden_states is None else retained.hidden_states[-1]
            )
        reporter.record_current("after_forward")

        pairwise_loss = None
        total_loss = lm_loss
        if pairwise_head is not None:
            pairwise_loss, _ = compute_pairwise_auxiliary(
                final_hidden, pairwise_head, prompt_lengths, pairwise_labels
            )
            total_loss = combine_training_losses(
                lm_loss, pairwise_loss, config.training.pairwise_loss_weight
            )
            reporter.record_current("after_pairwise_loss")

        reporter.record_values(
            "losses",
            {
                "lm_loss": float(lm_loss.item()),
                "pairwise_loss": (
                    None if pairwise_loss is None else float(pairwise_loss.item())
                ),
                "total_loss": float(total_loss.item()),
            },
        )
        total_loss.backward()
        reporter.record_peak("peak_forward_backward")
        reporter.record_current("after_backward")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        head_trainable = (
            [] if pairwise_head is None else list(pairwise_head.parameters())
        )
        reporter.record_values(
            "gradients",
            {
                "model_trainable_parameters": len(trainable),
                "model_parameters_with_grad": sum(
                    parameter.grad is not None for parameter in trainable
                ),
                "pairwise_trainable_parameters": len(head_trainable),
                "pairwise_parameters_with_grad": sum(
                    parameter.grad is not None for parameter in head_trainable
                ),
            },
        )
        reporter.record_values("result", {"status": "success"})
    except torch.cuda.OutOfMemoryError as error:
        try:
            reporter.record_peak("peak_at_oom")
        except RuntimeError:
            pass
        reporter.record_values(
            "result", {"status": "oom", "error": str(error)}
        )
    finally:
        # Keep the legacy outputs alive through backward for a faithful B/C measurement.
        _ = retained
        reporter.save(output_path)
        reporter.print_report()
        print(f"Wrote experiment {args.mode} to {output_path}")


if __name__ == "__main__":
    main()
