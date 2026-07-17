"""Training and tiny-subset overfit runner for Candidate Model 1."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..dataset import load_metadata
from ..metrics import exact_match_accuracy, pairwise_accuracy
from ..permutation import class_id_to_answer
from .artifacts import ArtifactLayout, write_json
from .config import Candidate1Config


def deterministic_split(rows: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make one reproducible split without reading or deriving from test data."""
    if not rows:
        raise ValueError("Training CSV contains no rows")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_size = 0 if len(rows) == 1 else max(1, round(len(rows) * validation_fraction))
    validation_indices = set(indices[:validation_size])
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model: Any, loader: Any, reporter: Any | None = None) -> dict[str, float | None]:
    import torch

    predicted_classes: list[int] = []
    true_classes: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = _move_batch(batch, model.device)
            labels = batch.pop("labels")
            if reporter is not None and batch_index == 0:
                reporter.reset_peak()
            logits = model(**batch).logits
            if reporter is not None and batch_index == 0:
                reporter.record_peak("peak_inference_step")
            predicted_classes.extend(logits.argmax(dim=-1).cpu().tolist())
            true_classes.extend(labels.cpu().tolist())
    truth = [class_id_to_answer(int(value)) for value in true_classes]
    predictions = [class_id_to_answer(int(value)) for value in predicted_classes]
    return {
        "exact_match_accuracy": exact_match_accuracy(truth, predictions),
        "pairwise_accuracy": pairwise_accuracy(truth, predictions),
    }


def save_checkpoint(
    model: Any,
    optimizer: Any,
    config: Candidate1Config,
    layout: ArtifactLayout,
    *,
    global_step: int,
    epoch: int,
    metrics: list[dict[str, Any]],
    reporter: Any,
) -> None:
    import torch

    layout.create()
    model.backbone.save_pretrained(layout.adapter_dir, safe_serialization=True)
    torch.save(model.classifier.state_dict(), layout.classifier_head_path)
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        },
        layout.trainer_state_path,
    )
    config.save(layout.config_path)
    write_json(layout.metrics_path, metrics)
    write_json(
        layout.metadata_path,
        {
            "pipeline": "candidate_model_1",
            "base_model": config.model.base_model_path,
            "processor": config.model.processor_path or config.model.base_model_path,
            "num_classes": 24,
            "hidden_size": int(model.classifier[0].normalized_shape[0]),
            "label_mapping": "snu_ordering.permutation canonical lexicographic mapping",
            "pooling": "final hidden state of last non-padding token",
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "peft_version": version("peft"),
            "compute_dtype": config.quantization.compute_dtype,
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
            "global_step": global_step,
            "epoch": epoch,
        },
    )
    reporter.save(layout.memory_path)


def parse_args(default_tiny: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional Candidate1 run_config.json")
    parser.add_argument("--base-model", help="Hub ID or local model path")
    parser.add_argument("--processor", help="Hub ID or local processor path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules", nargs="+")
    parser.add_argument("--compute-dtype", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny-overfit", action="store_true", default=default_tiny)
    parser.add_argument("--tiny-subset-size", type=int)
    parser.add_argument("--tiny-max-steps", type=int)
    parser.add_argument("--tiny-success-accuracy", type=float)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Candidate1Config:
    config = Candidate1Config.load(args.config) if args.config else Candidate1Config()
    model = replace(
        config.model,
        base_model_path=args.base_model or config.model.base_model_path,
        processor_path=args.processor if args.processor is not None else config.model.processor_path,
    )
    quantization = replace(
        config.quantization,
        enabled=False if args.disable_4bit else config.quantization.enabled,
        compute_dtype=args.compute_dtype or config.quantization.compute_dtype,
    )
    lora = replace(
        config.lora,
        rank=args.lora_rank or config.lora.rank,
        alpha=args.lora_alpha or config.lora.alpha,
        dropout=args.lora_dropout if args.lora_dropout is not None else config.lora.dropout,
        target_modules=tuple(args.lora_target_modules) if args.lora_target_modules else config.lora.target_modules,
    )
    training = replace(
        config.training,
        epochs=args.epochs or config.training.epochs,
        max_steps=args.max_steps if args.max_steps is not None else config.training.max_steps,
        batch_size=args.batch_size or config.training.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps or config.training.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate or config.training.learning_rate,
        tiny_subset_size=args.tiny_subset_size or config.training.tiny_subset_size,
        tiny_max_steps=args.tiny_max_steps or config.training.tiny_max_steps,
        tiny_success_accuracy=(
            args.tiny_success_accuracy
            if args.tiny_success_accuracy is not None
            else config.training.tiny_success_accuracy
        ),
    )
    if args.tiny_overfit:
        training = replace(training, gradient_accumulation_steps=1, validation_fraction=0.0)
    return Candidate1Config(model=model, quantization=quantization, lora=lora, training=training)


def main(default_tiny: bool = False) -> None:
    args = parse_args(default_tiny=default_tiny)
    config = config_from_args(args)

    import torch
    from torch.utils.data import DataLoader

    from .data import collate_rows
    from .memory import GpuMemoryReporter
    from .model import (
        load_processor,
        load_resumed_training_model,
        load_training_model,
        validate_trainable_parameters,
    )

    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    all_rows = load_metadata(args.train_csv, split="train").to_dict(orient="records")
    if args.tiny_overfit:
        train_rows = all_rows[: config.training.tiny_subset_size]
        validation_rows = train_rows
        if len(train_rows) < config.training.tiny_subset_size:
            raise ValueError(
                f"Requested {config.training.tiny_subset_size} tiny samples, found {len(train_rows)}"
            )
    else:
        train_rows, validation_rows = deterministic_split(
            all_rows, config.training.validation_fraction, config.training.seed
        )

    layout = ArtifactLayout(args.output_dir)
    layout.create()
    processor = load_processor(config, local_files_only=args.local_files_only)
    reporter = GpuMemoryReporter()
    if args.resume:
        model = load_resumed_training_model(
            config,
            adapter_path=layout.adapter_dir,
            classifier_head_path=layout.classifier_head_path,
            local_files_only=args.local_files_only,
        )
    else:
        model = load_training_model(config, local_files_only=args.local_files_only)
    validate_trainable_parameters(model)
    reporter.record_current("after_model_load")

    collate_train = lambda values: collate_rows(
        values,
        processor,
        args.image_root,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_labels=True,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_rows,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_train,
    )
    validation_loader = DataLoader(
        validation_rows,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_train,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    global_step = 0
    start_epoch = 0
    metrics_history: list[dict[str, Any]] = []
    if args.resume:
        state = torch.load(layout.trainer_state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])

    micro_batches_per_epoch = len(train_loader)
    normal_steps = (
        math.ceil(
            micro_batches_per_epoch / config.training.gradient_accumulation_steps
        )
        * config.training.epochs
    )
    target_steps = (
        config.training.tiny_max_steps
        if args.tiny_overfit
        else (config.training.max_steps or normal_steps)
    )
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    measured_train_step = False
    stop = False
    while global_step < target_steps and not stop:
        model.train()
        accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if not measured_train_step:
                reporter.reset_peak()
            batch = _move_batch(batch, model.device)
            loss = model(**batch).loss / config.training.gradient_accumulation_steps
            loss.backward()
            accumulated += 1
            accumulation_complete = accumulated >= config.training.gradient_accumulation_steps
            end_of_epoch = batch_index == micro_batches_per_epoch
            if not accumulation_complete and not end_of_epoch:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1
            if not measured_train_step:
                reporter.record_peak("peak_training_step")
                measured_train_step = True

            if global_step % config.training.log_every_steps == 0 or global_step == 1:
                print(f"epoch={epoch} step={global_step}/{target_steps} loss={loss.item() * config.training.gradient_accumulation_steps:.6f}")

            should_evaluate = args.tiny_overfit and (
                global_step % 25 == 0 or global_step == target_steps
            )
            if should_evaluate:
                evaluation = evaluate(
                    model, validation_loader, reporter if global_step == 25 else None
                )
                record = {"step": global_step, "split": "tiny_train", **evaluation}
                metrics_history.append(record)
                print(
                    f"tiny step={global_step}: exact_match={evaluation['exact_match_accuracy']:.4f} "
                    f"pairwise={evaluation['pairwise_accuracy']:.4f}"
                )
                if float(evaluation["exact_match_accuracy"] or 0.0) >= config.training.tiny_success_accuracy:
                    print(
                        f"TINY OVERFIT PASS: exact-match >= {config.training.tiny_success_accuracy:.2f} "
                        f"at step {global_step}"
                    )
                    stop = True

            if global_step % config.training.save_every_steps == 0 or stop:
                save_checkpoint(
                    model,
                    optimizer,
                    config,
                    layout,
                    global_step=global_step,
                    epoch=epoch,
                    metrics=metrics_history,
                    reporter=reporter,
                )
            if global_step >= target_steps or stop:
                break
        epoch += 1

    if not args.tiny_overfit:
        evaluation = evaluate(model, validation_loader, reporter)
        metrics_history.append({"step": global_step, "split": "validation", **evaluation})
        print(
            f"validation exact_match={evaluation['exact_match_accuracy']:.4f} "
            f"pairwise={evaluation['pairwise_accuracy']:.4f}"
        )
    elif not stop:
        final_accuracy = float(metrics_history[-1]["exact_match_accuracy"] or 0.0)
        print(
            f"TINY OVERFIT FAIL: exact-match={final_accuracy:.4f} < "
            f"{config.training.tiny_success_accuracy:.2f} after {global_step} steps"
        )

    save_checkpoint(
        model,
        optimizer,
        config,
        layout,
        global_step=global_step,
        epoch=epoch,
        metrics=metrics_history,
        reporter=reporter,
    )
    reporter.print_report()
    print(f"Artifacts saved to {layout.root}")


if __name__ == "__main__":
    main()
