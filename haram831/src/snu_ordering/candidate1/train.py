"""Training and tiny-subset overfit runner for Candidate Model 1."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import traceback
from collections import Counter
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..dataset import load_metadata
from ..metrics import exact_match_accuracy, pairwise_accuracy
from ..permutation import class_id_to_answer
from .artifacts import ArtifactLayout, write_json
from .config import Candidate1Config


class TeeStream:
    """Mirror writes to the original stream and an append-only log file."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


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


def compute_class_weights(
    rows: list[dict[str, Any]], num_classes: int, power: float
) -> list[float]:
    """Return normalized inverse-frequency weights derived from train rows only."""
    if not rows:
        raise ValueError("Cannot compute class weights from an empty training split")
    counts = Counter(int(row["class_id"]) for row in rows)
    invalid = sorted(value for value in counts if not 0 <= value < num_classes)
    if invalid:
        raise ValueError(f"Training rows contain invalid class IDs: {invalid}")
    raw = [
        (len(rows) / (num_classes * counts[class_id])) ** power
        if counts[class_id]
        else 0.0
        for class_id in range(num_classes)
    ]
    positive = [value for value in raw if value > 0.0]
    scale = sum(positive) / len(positive)
    return [value / scale if value > 0.0 else 0.0 for value in raw]


def summarize_predictions(
    true_classes: list[int],
    predicted_classes: list[int],
    *,
    num_classes: int,
    collapse_threshold: float,
) -> dict[str, Any]:
    """Produce JSON-safe accuracy and class-collapse diagnostics."""
    if len(true_classes) != len(predicted_classes):
        raise ValueError("Truth and prediction class lists must have the same length")
    if not true_classes:
        raise ValueError("Cannot summarize an empty evaluation split")
    truth = [class_id_to_answer(int(value)) for value in true_classes]
    predictions = [class_id_to_answer(int(value)) for value in predicted_classes]
    true_counts = Counter(true_classes)
    predicted_counts = Counter(predicted_classes)
    recalls: dict[str, float | None] = {}
    present_recalls: list[float] = []
    for class_id in range(num_classes):
        support = true_counts[class_id]
        recall = None
        if support:
            recall = sum(
                actual == class_id and predicted == class_id
                for actual, predicted in zip(true_classes, predicted_classes)
            ) / support
            present_recalls.append(recall)
        recalls[str(class_id)] = recall
    maximum_share = max(predicted_counts.values()) / len(predicted_classes)
    return {
        "exact_match_accuracy": exact_match_accuracy(truth, predictions),
        "pairwise_accuracy": pairwise_accuracy(truth, predictions),
        "macro_recall": sum(present_recalls) / len(present_recalls),
        "per_class_recall": recalls,
        "predicted_class_histogram": {
            str(class_id): predicted_counts[class_id]
            for class_id in range(num_classes)
        },
        "max_prediction_share": maximum_share,
        "collapse_detected": maximum_share > collapse_threshold,
    }


def evaluate(
    model: Any,
    loader: Any,
    *,
    num_classes: int = 24,
    collapse_threshold: float = 0.5,
    reporter: Any | None = None,
) -> dict[str, Any]:
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
    return summarize_predictions(
        true_classes,
        predicted_classes,
        num_classes=num_classes,
        collapse_threshold=collapse_threshold,
    )


def print_evaluation(name: str, evaluation: dict[str, Any]) -> None:
    print(
        f"{name} exact_match={evaluation['exact_match_accuracy']:.4f} "
        f"pairwise={evaluation['pairwise_accuracy']:.4f} "
        f"macro_recall={evaluation['macro_recall']:.4f} "
        f"max_class_share={evaluation['max_prediction_share']:.4f}"
    )
    print(f"{name} prediction_histogram={evaluation['predicted_class_histogram']}")
    if evaluation["collapse_detected"]:
        print(
            f"WARNING: {name} prediction collapse detected; one class exceeds "
            "the configured share threshold"
        )


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Candidate1Config,
    layout: ArtifactLayout,
    *,
    global_step: int,
    epoch: int,
    metrics: list[dict[str, Any]],
    class_weights: list[float],
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
            "scheduler": scheduler.state_dict(),
        },
        layout.trainer_state_path,
    )
    config.save(layout.config_path)
    write_json(layout.metrics_path, metrics)
    write_json(
        layout.metadata_path,
        {
            "pipeline": "candidate_model_1",
            "architecture_version": config.architecture_version,
            "base_model": config.model.base_model_path,
            "processor": config.model.processor_path or config.model.base_model_path,
            "num_classes": 24,
            "hidden_size": int(model.classifier[0].normalized_shape[0]),
            "label_mapping": "snu_ordering.permutation canonical lexicographic mapping",
            "pooling": "final hidden state at assistant generation position",
            "class_weights": class_weights,
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
    parser.add_argument("--classifier-learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--class-weight-power", type=float)
    parser.add_argument("--fast-validation-size", type=int)
    parser.add_argument("--full-validation-every-epochs", type=int)
    parser.add_argument("--collapse-threshold", type=float)
    parser.add_argument("--success-accuracy", type=float)
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
        classifier_learning_rate=(
            args.classifier_learning_rate or config.training.classifier_learning_rate
        ),
        warmup_ratio=(
            args.warmup_ratio
            if args.warmup_ratio is not None
            else config.training.warmup_ratio
        ),
        class_weight_power=(
            args.class_weight_power
            if args.class_weight_power is not None
            else config.training.class_weight_power
        ),
        fast_validation_size=(
            args.fast_validation_size or config.training.fast_validation_size
        ),
        full_validation_every_epochs=(
            args.full_validation_every_epochs
            or config.training.full_validation_every_epochs
        ),
        collapse_threshold=(
            args.collapse_threshold
            if args.collapse_threshold is not None
            else config.training.collapse_threshold
        ),
        success_accuracy=(
            args.success_accuracy
            if args.success_accuracy is not None
            else config.training.success_accuracy
        ),
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
    return Candidate1Config(
        architecture_version=config.architecture_version,
        model=model,
        quantization=quantization,
        lora=lora,
        training=training,
    )


def run_training(args: argparse.Namespace) -> None:
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
    class_weights = compute_class_weights(
        train_rows, config.model.num_classes, config.training.class_weight_power
    )
    print(f"class_weights={[round(value, 6) for value in class_weights]}")

    layout = ArtifactLayout(args.output_dir)
    best_layout = ArtifactLayout(layout.root / "best")
    layout.create()
    processor = load_processor(config, local_files_only=args.local_files_only)
    reporter = GpuMemoryReporter()
    if args.resume:
        model = load_resumed_training_model(
            config,
            adapter_path=layout.adapter_dir,
            classifier_head_path=layout.classifier_head_path,
            class_weights=class_weights,
            local_files_only=args.local_files_only,
        )
    else:
        model = load_training_model(
            config,
            class_weights=class_weights,
            local_files_only=args.local_files_only,
        )
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
    fast_validation_loader = DataLoader(
        validation_rows[: config.training.fast_validation_size],
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_train,
    )

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

    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    classifier_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("classifier.")
    ]
    trainable = lora_parameters + classifier_parameters
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": config.training.learning_rate,
                "group_name": "lora",
            },
            {
                "params": classifier_parameters,
                "lr": config.training.classifier_learning_rate,
                "group_name": "classifier",
            },
        ],
        weight_decay=config.training.weight_decay,
    )
    from transformers import get_cosine_schedule_with_warmup

    warmup_steps = round(target_steps * config.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=target_steps,
    )
    global_step = 0
    start_epoch = 0
    metrics_history: list[dict[str, Any]] = []
    if args.resume:
        state = torch.load(layout.trainer_state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        if layout.metrics_path.is_file():
            metrics_history = json.loads(layout.metrics_path.read_text(encoding="utf-8"))

    best_score = -1.0
    best_max_prediction_share = 1.0
    if args.resume and best_layout.metadata_path.is_file():
        best_metadata = json.loads(best_layout.metadata_path.read_text(encoding="utf-8"))
        best_score = float(best_metadata.get("best_exact_match_accuracy", -1.0))
        best_max_prediction_share = float(
            best_metadata.get("best_max_prediction_share", 1.0)
        )
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    measured_train_step = False
    stop = False
    running_loss = 0.0
    running_micro_batches = 0
    last_full_validation_step = -1
    while global_step < target_steps and not stop:
        model.train()
        accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if not measured_train_step:
                reporter.reset_peak()
            batch = _move_batch(batch, model.device)
            raw_loss = model(**batch).loss
            running_loss += float(raw_loss.item())
            running_micro_batches += 1
            loss = raw_loss / config.training.gradient_accumulation_steps
            loss.backward()
            accumulated += 1
            accumulation_complete = accumulated >= config.training.gradient_accumulation_steps
            end_of_epoch = batch_index == micro_batches_per_epoch
            if not accumulation_complete and not end_of_epoch:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1
            if not measured_train_step:
                reporter.record_peak("peak_training_step")
                measured_train_step = True

            if global_step % config.training.log_every_steps == 0 or global_step == 1:
                average_loss = running_loss / max(running_micro_batches, 1)
                rates = {
                    group.get("group_name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                }
                print(
                    f"epoch={epoch} step={global_step}/{target_steps} "
                    f"loss={average_loss:.6f} learning_rates={rates}"
                )
                running_loss = 0.0
                running_micro_batches = 0

            should_evaluate = args.tiny_overfit and (
                global_step % 25 == 0 or global_step == target_steps
            )
            if should_evaluate:
                evaluation = evaluate(
                    model,
                    validation_loader,
                    num_classes=config.model.num_classes,
                    collapse_threshold=config.training.collapse_threshold,
                    reporter=reporter if global_step == 25 else None,
                )
                record = {"step": global_step, "split": "tiny_train", **evaluation}
                metrics_history.append(record)
                print_evaluation(f"tiny step={global_step}", evaluation)
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
                    scheduler,
                    config,
                    layout,
                    global_step=global_step,
                    epoch=epoch,
                    metrics=metrics_history,
                    class_weights=class_weights,
                    reporter=reporter,
                )
            if global_step >= target_steps or stop:
                break

        if not args.tiny_overfit:
            completed_epoch = epoch + 1
            fast_evaluation = evaluate(
                model,
                fast_validation_loader,
                num_classes=config.model.num_classes,
                collapse_threshold=config.training.collapse_threshold,
            )
            metrics_history.append(
                {
                    "step": global_step,
                    "epoch": completed_epoch,
                    "split": "validation_fast",
                    **fast_evaluation,
                }
            )
            print_evaluation(f"validation_fast epoch={completed_epoch}", fast_evaluation)

            full_validation_due = (
                completed_epoch % config.training.full_validation_every_epochs == 0
                or global_step >= target_steps
            )
            if full_validation_due:
                full_evaluation = evaluate(
                    model,
                    validation_loader,
                    num_classes=config.model.num_classes,
                    collapse_threshold=config.training.collapse_threshold,
                    reporter=reporter if last_full_validation_step == -1 else None,
                )
                metrics_history.append(
                    {
                        "step": global_step,
                        "epoch": completed_epoch,
                        "split": "validation",
                        **full_evaluation,
                    }
                )
                print_evaluation(
                    f"validation epoch={completed_epoch}", full_evaluation
                )
                last_full_validation_step = global_step
                full_score = float(full_evaluation["exact_match_accuracy"] or 0.0)
                if full_score > best_score:
                    best_score = full_score
                    best_max_prediction_share = float(
                        full_evaluation["max_prediction_share"]
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        config,
                        best_layout,
                        global_step=global_step,
                        epoch=completed_epoch,
                        metrics=metrics_history,
                        class_weights=class_weights,
                        reporter=reporter,
                    )
                    best_metadata = json.loads(
                        best_layout.metadata_path.read_text(encoding="utf-8")
                    )
                    best_metadata["best_exact_match_accuracy"] = best_score
                    best_metadata["best_max_prediction_share"] = (
                        best_max_prediction_share
                    )
                    write_json(best_layout.metadata_path, best_metadata)
                    print(
                        f"Saved new best checkpoint: exact_match={best_score:.4f} "
                        f"to {best_layout.root}"
                    )
        epoch += 1

    if not args.tiny_overfit and last_full_validation_step != global_step:
        evaluation = evaluate(
            model,
            validation_loader,
            num_classes=config.model.num_classes,
            collapse_threshold=config.training.collapse_threshold,
            reporter=reporter,
        )
        metrics_history.append({"step": global_step, "split": "validation", **evaluation})
        print_evaluation("validation final", evaluation)
    elif args.tiny_overfit and not stop:
        final_accuracy = float(metrics_history[-1]["exact_match_accuracy"] or 0.0)
        print(
            f"TINY OVERFIT FAIL: exact-match={final_accuracy:.4f} < "
            f"{config.training.tiny_success_accuracy:.2f} after {global_step} steps"
        )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        config,
        layout,
        global_step=global_step,
        epoch=epoch,
        metrics=metrics_history,
        class_weights=class_weights,
        reporter=reporter,
    )
    if not args.tiny_overfit:
        passed = (
            best_score >= config.training.success_accuracy
            and best_max_prediction_share <= config.training.collapse_threshold
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"VALIDATION {status}: best_exact_match={best_score:.4f} "
            f"required={config.training.success_accuracy:.4f} "
            f"max_class_share={best_max_prediction_share:.4f} "
            f"allowed={config.training.collapse_threshold:.4f}"
        )
    reporter.print_report()
    print(f"Artifacts saved to {layout.root}")


def main(default_tiny: bool = False) -> None:
    args = parse_args(default_tiny=default_tiny)
    layout = ArtifactLayout(args.output_dir)
    layout.create()
    log_path = layout.root / "train.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout = TeeStream(sys.stdout, log_handle)
        stderr = TeeStream(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"=== Candidate1 train start: {Path.cwd()} ===")
            print(f"argv: {' '.join(sys.argv)}")
            try:
                run_training(args)
            except Exception:
                print("UNHANDLED EXCEPTION DURING TRAINING")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
