"""Offline inference for Candidate 1 v3 causal-LM checkpoints."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from ..dataset import load_metadata
from ..permutation import parse_answer
from ..submission import build_submission, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local base-model directory")
    parser.add_argument("--processor", required=True, help="Local processor directory")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--config", required=True, help="v3 run_config.json path")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-submission", required=True)
    parser.add_argument("--mode", choices=("generate", "score"), default="generate")
    parser.add_argument("--scoring-chunk-size", type=int)
    parser.add_argument("--limit", type=int, default=None, help="Debug only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from .config import Candidate1Config, ModelConfig
    from .data import build_messages
    from .memory import GpuMemoryReporter
    from .model import load_inference_model, load_processor
    from .scoring import generate_order, order_to_answer, score_order

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
        caption_prompt=config.caption_prompt,
    )
    frame = load_metadata(args.input_csv, split="test", limit=args.limit)
    rows = frame.to_dict(orient="records")
    processor = load_processor(config, local_files_only=True)
    checkpoint_root = Path(args.config).parent
    model = load_inference_model(
        config,
        adapter_path=args.adapter,
        checkpoint_root=checkpoint_root,
        local_files_only=True,
    )
    model.eval()
    reporter = GpuMemoryReporter()
    reporter.record_current("after_model_load")

    output = Path(args.output_submission)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial.csv")
    completed: dict[str, tuple[int, int, int, int]] = {}
    if partial.is_file():
        previous = pd.read_csv(partial, keep_default_na=False)
        if tuple(previous.columns) != ("Id", "Answer"):
            raise ValueError(f"Invalid partial submission columns: {partial}")
        for sample_id, raw_answer in zip(previous["Id"], previous["Answer"]):
            completed[str(sample_id)] = parse_answer(str(raw_answer))
        print(f"Resuming {len(completed)} completed rows from {partial}")

    write_header = not partial.is_file() or partial.stat().st_size == 0
    invalid_generations = 0
    processed_now = 0
    started = time.perf_counter()
    with partial.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["Id", "Answer"])
            handle.flush()
        for row in rows:
            sample_id = str(row["Id"])
            if sample_id in completed:
                continue
            messages = build_messages(
                row,
                args.image_root,
                min_pixels=config.model.min_pixels,
                max_pixels=config.model.max_pixels,
                caption_prompt_mode=config.caption_prompt.mode,
                relation_confidence_threshold=(
                    config.caption_prompt.relation_confidence_threshold
                ),
                boundary_dropout=0.0,
            )
            with torch.inference_mode():
                if args.mode == "generate":
                    order = generate_order(
                        model,
                        processor,
                        messages,
                        max_new_tokens=config.training.generation_max_new_tokens,
                    )
                    if order is None:
                        invalid_generations += 1
                        order, _ = score_order(
                            model,
                            processor,
                            messages,
                            chunk_size=args.scoring_chunk_size or config.training.scoring_chunk_size,
                        )
                else:
                    order, _ = score_order(
                        model,
                        processor,
                        messages,
                        chunk_size=args.scoring_chunk_size or config.training.scoring_chunk_size,
                    )
            answer = order_to_answer(order)
            completed[sample_id] = answer
            writer.writerow([sample_id, str(list(answer))])
            handle.flush()
            processed_now += 1
            if processed_now == 1:
                reporter.record_current("first_prediction")
            if len(completed) % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{len(completed)}/{len(rows)} complete; "
                    f"{elapsed / max(processed_now, 1):.2f}s/new row"
                )

    expected_ids = frame["Id"].astype(str).tolist()
    missing = [sample_id for sample_id in expected_ids if sample_id not in completed]
    if missing:
        raise RuntimeError(f"Partial inference is missing {len(missing)} rows")
    answers = [completed[sample_id] for sample_id in expected_ids]
    submission = build_submission(expected_ids, answers)
    validate_submission(submission, expected_ids)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} predictions to {output}")
    print(f"Inference mode={args.mode}; invalid generations={invalid_generations}")
    reporter.print_report()
    if args.limit is not None:
        print("WARNING: --limit produced a debug subset, not a full submission")


if __name__ == "__main__":
    main()
