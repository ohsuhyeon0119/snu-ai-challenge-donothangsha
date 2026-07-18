"""Offline-only inference entrypoint for Candidate Model 1."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from ..dataset import load_metadata
from ..permutation import Permutation, class_id_to_answer
from ..submission import build_submission, validate_submission


def class_ids_to_answers(class_ids: Iterable[int]) -> list[Permutation]:
    """Convert predictions only through the repository's canonical mapping."""
    return [class_id_to_answer(int(class_id)) for class_id in class_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local base-model directory")
    parser.add_argument("--processor", required=True, help="Local processor directory")
    parser.add_argument("--adapter", required=True, help="Local LoRA adapter directory")
    parser.add_argument("--classifier-head", required=True, help="classifier_head.pt path")
    parser.add_argument("--config", required=True, help="run_config.json path")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-submission", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Debug only; output is not a full submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from .config import Candidate1Config, ModelConfig
    from .data import collate_rows
    from .memory import GpuMemoryReporter
    from .model import load_inference_model, load_processor

    config = Candidate1Config.load(args.config)
    config = Candidate1Config(
        architecture_version=config.architecture_version,
        model=ModelConfig(
            **{
                **config.model.__dict__,
                "base_model_path": args.base_model,
                "processor_path": args.processor,
            }
        ),
        quantization=config.quantization,
        lora=config.lora,
        training=config.training,
    )
    frame = load_metadata(args.input_csv, split="test", limit=args.limit)
    rows = frame.to_dict(orient="records")
    processor = load_processor(config, local_files_only=True)
    reporter = GpuMemoryReporter()
    model = load_inference_model(
        config,
        adapter_path=args.adapter,
        classifier_head_path=args.classifier_head,
        local_files_only=True,
    )
    reporter.record_current("after_model_load")

    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda values: collate_rows(
            values,
            processor,
            args.image_root,
            min_pixels=config.model.min_pixels,
            max_pixels=config.model.max_pixels,
            include_labels=False,
        ),
    )
    predictions: list[int] = []
    elapsed = 0.0
    model.eval()
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        if batch_index == 0:
            reporter.reset_peak()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**batch).logits
        elapsed += time.perf_counter() - started
        if batch_index == 0:
            reporter.record_peak("peak_inference_step")
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    answers = class_ids_to_answers(predictions)
    submission = build_submission(frame["Id"].astype(str).tolist(), answers)
    validate_submission(submission, frame["Id"].astype(str).tolist())
    output = Path(args.output_submission)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} predictions to {output}")
    print(f"Inference seconds/sample: {elapsed / max(len(submission), 1):.4f}")
    reporter.print_report()
    if args.limit is not None:
        print("WARNING: --limit produced a debug subset, not a full competition submission")


if __name__ == "__main__":
    main()
