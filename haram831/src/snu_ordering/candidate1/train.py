"""Train Candidate 1 v3 with completion-only QLoRA supervision."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..dataset import load_metadata
from ..metrics import exact_match_accuracy, pairwise_accuracy
from ..permutation import answer_to_class_id, class_id_to_answer
from .artifacts import ArtifactLayout, write_json
from .config import Candidate1Config


MAX_OOM_SKIPS = 8


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize both PyTorch OOM exception variants without version coupling."""
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "accelerator" in message)


def recover_from_cuda_oom(torch: Any, optimizer: Any) -> None:
    """Discard partial gradients and release cached allocations after a skipped batch."""
    import gc

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def stratified_split(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic Answer-stratified split without extra dependencies."""
    if not rows:
        raise ValueError("Training CSV contains no rows")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["class_id"])].append(row)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for class_id in sorted(groups):
        group = list(groups[class_id])
        random.Random(seed + class_id).shuffle(group)
        validation_size = max(1, round(len(group) * validation_fraction))
        validation_rows.extend(group[:validation_size])
        train_rows.extend(group[validation_size:])
    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(validation_rows)
    return train_rows, validation_rows


def summarize_predictions(
    true_classes: list[int],
    predicted_classes: list[int],
    *,
    collapse_threshold: float,
    invalid_generations: int = 0,
    entropies: list[float] | None = None,
    margins: list[float] | None = None,
) -> dict[str, Any]:
    if len(true_classes) != len(predicted_classes) or not true_classes:
        raise ValueError("Prediction and truth lists must be non-empty and equally sized")
    truth = [class_id_to_answer(value) for value in true_classes]
    predictions = [class_id_to_answer(value) for value in predicted_classes]
    true_counts = Counter(true_classes)
    predicted_counts = Counter(predicted_classes)
    recalls: dict[str, float | None] = {}
    present_recalls: list[float] = []
    for class_id in range(24):
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
            str(class_id): predicted_counts[class_id] for class_id in range(24)
        },
        "max_prediction_share": maximum_share,
        "collapse_detected": maximum_share > collapse_threshold,
        "invalid_generations": invalid_generations,
        "candidate_entropy_mean": (
            sum(entropies) / len(entropies) if entropies else None
        ),
        "candidate_margin_mean": sum(margins) / len(margins) if margins else None,
    }


def _candidate_diagnostics(scores: list[float]) -> tuple[float, float]:
    import torch

    probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    top = torch.topk(probabilities, 2).values
    return entropy, float(top[0] - top[1])


def evaluate_rows(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: str | Path,
    config: Candidate1Config,
    *,
    mode: str,
) -> dict[str, Any]:
    import gc
    import torch

    from .data import build_messages
    from .scoring import generate_order, order_to_answer, score_order

    if mode not in {"generate", "score"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    true_classes: list[int] = []
    predicted_classes: list[int] = []
    entropies: list[float] = []
    margins: list[float] = []
    invalid = 0
    was_training = model.training
    model.eval()
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for row in rows:
            messages = build_messages(
                row,
                image_root,
                min_pixels=config.model.min_pixels,
                max_pixels=config.model.max_pixels,
                caption_prompt_mode=config.caption_prompt.mode,
                relation_confidence_threshold=config.caption_prompt.relation_confidence_threshold,
                boundary_dropout=0.0,
            )
            scores: list[float] | None = None
            if mode == "generate":
                order = generate_order(
                    model,
                    processor,
                    messages,
                    max_new_tokens=config.training.generation_max_new_tokens,
                )
                if order is None:
                    invalid += 1
                    order, scores = score_order(
                        model,
                        processor,
                        messages,
                        chunk_size=config.training.scoring_chunk_size,
                    )
            else:
                order, scores = score_order(
                    model,
                    processor,
                    messages,
                    chunk_size=config.training.scoring_chunk_size,
                )
            if scores is not None:
                entropy, margin = _candidate_diagnostics(scores)
                entropies.append(entropy)
                margins.append(margin)
            predicted_classes.append(answer_to_class_id(order_to_answer(order)))
            true_classes.append(int(row["class_id"]))
    model.config.use_cache = previous_cache
    if was_training:
        model.train()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summarize_predictions(
        true_classes,
        predicted_classes,
        collapse_threshold=config.training.collapse_threshold,
        invalid_generations=invalid,
        entropies=entropies,
        margins=margins,
    )


def evaluate_pairwise_head(
    model: Any,
    pairwise_head: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: str | Path,
    config: Candidate1Config,
) -> dict[str, float]:
    """Evaluate the training-only A2 head without affecting model selection."""

    import gc
    import torch

    from .data import collate_rows
    from .pairwise import compute_pairwise_auxiliary

    if not rows:
        return {"aux_pairwise_accuracy": 0.0, "aux_pairwise_bce": 0.0}
    was_training = model.training
    head_was_training = pairwise_head.training
    model.eval()
    pairwise_head.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for row in rows:
            batch = collate_rows(
                [row],
                processor,
                image_root,
                min_pixels=config.model.min_pixels,
                max_pixels=config.model.max_pixels,
                include_labels=False,
                caption_prompt_mode=config.caption_prompt.mode,
                relation_confidence_threshold=config.caption_prompt.relation_confidence_threshold,
                boundary_dropout=0.0,
                include_pairwise_labels=True,
            )
            batch = {key: value.to(model.device) for key, value in batch.items()}
            labels = batch.pop("pairwise_labels")
            prompt_lengths = batch.pop("prompt_lengths")
            outputs = model(
                **batch, output_hidden_states=True, use_cache=False, return_dict=True
            )
            loss, logits = compute_pairwise_auxiliary(
                outputs, pairwise_head, prompt_lengths, labels
            )
            total_loss += float(loss.item())
            predictions = logits >= 0
            correct += int((predictions == labels.bool()).sum().item())
            total += int(labels.numel())
    if was_training:
        model.train()
    if head_was_training:
        pairwise_head.train()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "aux_pairwise_accuracy": correct / total,
        "aux_pairwise_bce": total_loss / len(rows),
    }


def print_evaluation(name: str, result: dict[str, Any]) -> None:
    print(
        f"{name} exact_match={result['exact_match_accuracy']:.4f} "
        f"pairwise={result['pairwise_accuracy']:.4f} "
        f"macro_recall={result['macro_recall']:.4f} "
        f"max_class_share={result['max_prediction_share']:.4f} "
        f"invalid={result['invalid_generations']} "
        f"entropy={result['candidate_entropy_mean']} margin={result['candidate_margin_mean']}"
    )
    print(f"{name} prediction_histogram={result['predicted_class_histogram']}")
    if "aux_pairwise_accuracy" in result:
        print(
            f"{name} aux_pairwise_accuracy={result['aux_pairwise_accuracy']:.4f} "
            f"aux_pairwise_bce={result['aux_pairwise_bce']:.6f}"
        )
    if result["collapse_detected"]:
        print(f"WARNING: {name} prediction collapse detected")


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Candidate1Config,
    layout: ArtifactLayout,
    *,
    pairwise_head: Any | None = None,
    global_step: int,
    epoch: int,
    next_batch_index: int,
    metrics: list[dict[str, Any]],
    reporter: Any,
) -> None:
    import torch

    layout.create()
    model.save_pretrained(layout.adapter_dir, safe_serialization=True)
    if pairwise_head is not None:
        from .pairwise import save_pairwise_head

        save_pairwise_head(pairwise_head, layout.pairwise_head_path)
    config.save(layout.config_path)
    write_json(layout.metrics_path, metrics)
    write_json(
        layout.metadata_path,
        {
            "pipeline": "candidate_model_1",
            "architecture_version": config.architecture_version,
            "objective": (
                "completion-only causal language modeling"
                if pairwise_head is None
                else "completion-only causal language modeling + pairwise BCE auxiliary"
            ),
            "target_representation": "chronological input-image indices",
            "base_model": config.model.base_model_path,
            "processor": config.model.processor_path or config.model.base_model_path,
            "global_step": global_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "min_pixels": config.model.min_pixels,
            "max_pixels": config.model.max_pixels,
            "caption_prompt_mode": config.caption_prompt.mode,
            "pairwise_loss_weight": config.training.pairwise_loss_weight,
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "rng_state": _capture_rng_state(torch),
        },
        layout.trainer_state_path,
    )
    reporter.save(layout.memory_path)


def parse_args(default_tiny: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--processor")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--fast-validation-every-steps", type=int)
    parser.add_argument("--fast-validation-size", type=int)
    parser.add_argument(
        "--epoch-validation-size",
        "--constrained-validation-size",
        dest="epoch_validation_size",
        type=int,
        help="Greedy-generation samples used for epoch-end model selection",
    )
    parser.add_argument("--scoring-chunk-size", type=int)
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
    parser.add_argument("--relation-confidence-threshold", type=float)
    parser.add_argument("--boundary-dropout", type=float)
    parser.add_argument("--pairwise-loss-weight", type=float)
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
        gradient_accumulation_steps=args.gradient_accumulation_steps or config.training.gradient_accumulation_steps,
        learning_rate=args.learning_rate or config.training.learning_rate,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else config.training.warmup_ratio,
        fast_validation_every_steps=args.fast_validation_every_steps or config.training.fast_validation_every_steps,
        fast_validation_size=args.fast_validation_size or config.training.fast_validation_size,
        epoch_validation_size=args.epoch_validation_size or config.training.epoch_validation_size,
        scoring_chunk_size=args.scoring_chunk_size or config.training.scoring_chunk_size,
        tiny_subset_size=args.tiny_subset_size or config.training.tiny_subset_size,
        tiny_max_steps=args.tiny_max_steps or config.training.tiny_max_steps,
        tiny_success_accuracy=args.tiny_success_accuracy if args.tiny_success_accuracy is not None else config.training.tiny_success_accuracy,
        pairwise_loss_weight=args.pairwise_loss_weight if args.pairwise_loss_weight is not None else config.training.pairwise_loss_weight,
    )
    caption_prompt = replace(
        config.caption_prompt,
        relation_confidence_threshold=(
            args.relation_confidence_threshold
            if args.relation_confidence_threshold is not None
            else config.caption_prompt.relation_confidence_threshold
        ),
        boundary_dropout=(
            args.boundary_dropout
            if args.boundary_dropout is not None
            else config.caption_prompt.boundary_dropout
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
        caption_prompt=caption_prompt,
    )


def run_training(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from .data import collate_rows
    from .memory import GpuMemoryReporter
    from .model import (
        load_processor,
        load_resumed_training_model,
        load_training_model,
        validate_trainable_parameters,
    )
    from .pairwise import (
        combine_training_losses,
        compute_pairwise_auxiliary,
        create_pairwise_head,
        load_pairwise_head,
    )

    config = config_from_args(args)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    all_rows = load_metadata(args.train_csv, split="train").to_dict(orient="records")
    if args.tiny_overfit:
        train_rows = all_rows[: config.training.tiny_subset_size]
        validation_rows = train_rows
    else:
        train_rows, validation_rows = stratified_split(
            all_rows, config.training.validation_fraction, config.training.seed
        )
    fast_rows = validation_rows[: config.training.fast_validation_size]
    epoch_validation_rows = validation_rows[: config.training.epoch_validation_size]

    layout = ArtifactLayout(args.output_dir)
    best_layout = ArtifactLayout(layout.root / "best")
    layout.create()
    processor = load_processor(config, local_files_only=args.local_files_only)
    reporter = GpuMemoryReporter()
    if args.resume:
        model = load_resumed_training_model(
            config,
            adapter_path=layout.adapter_dir,
            checkpoint_root=layout.root,
            local_files_only=args.local_files_only,
        )
    else:
        model = load_training_model(config, local_files_only=args.local_files_only)
    validate_trainable_parameters(model)
    pairwise_head = None
    if config.training.pairwise_loss_weight > 0.0:
        pairwise_head = create_pairwise_head(model)
        if args.resume:
            load_pairwise_head(pairwise_head, layout.pairwise_head_path)
    reporter.record_current("after_model_load")
    batches_per_epoch = math.ceil(len(train_rows) / config.training.batch_size)
    steps_per_epoch = math.ceil(
        batches_per_epoch / config.training.gradient_accumulation_steps
    )
    target_steps = (
        config.training.tiny_max_steps
        if args.tiny_overfit
        else (config.training.max_steps or steps_per_epoch * config.training.epochs)
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if pairwise_head is not None:
        trainable_parameters.extend(pairwise_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(target_steps * config.training.warmup_ratio),
        num_training_steps=target_steps,
    )

    global_step = 0
    start_epoch = 0
    start_batch_index = 0
    metrics: list[dict[str, Any]] = []
    if args.resume:
        state = torch.load(layout.trainer_state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        start_batch_index = int(state["next_batch_index"])
        _restore_rng_state(torch, state["rng_state"])
        if layout.metrics_path.is_file():
            metrics = json.loads(layout.metrics_path.read_text(encoding="utf-8"))

    best_score = -1.0
    best_pairwise = -1.0
    best_share = 1.0
    if best_layout.metadata_path.is_file():
        best_metadata = json.loads(best_layout.metadata_path.read_text(encoding="utf-8"))
        best_score = float(best_metadata.get("best_exact_match_accuracy", -1.0))
        best_pairwise = float(best_metadata.get("best_pairwise_accuracy", -1.0))
        best_share = float(best_metadata.get("best_max_prediction_share", 1.0))

    running_loss = 0.0
    running_lm_loss = 0.0
    running_pairwise_loss = 0.0
    running_batches = 0
    oom_skips = 0
    stop = False
    interrupt_requested = False

    def request_interrupt(signum: int, frame: Any) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True
        print("Interrupt requested; saving after the current optimizer step")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_interrupt)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    epoch = start_epoch
    while epoch < config.training.epochs and global_step < target_steps and not stop:
        epoch_rows = list(train_rows)
        random.Random(config.training.seed + epoch).shuffle(epoch_rows)
        collate = lambda values, current_epoch=epoch: collate_rows(
            values,
            processor,
            args.image_root,
            min_pixels=config.model.min_pixels,
            max_pixels=config.model.max_pixels,
            include_labels=True,
            caption_prompt_mode=config.caption_prompt.mode,
            relation_confidence_threshold=config.caption_prompt.relation_confidence_threshold,
            boundary_dropout=config.caption_prompt.boundary_dropout,
            boundary_dropout_seed=f"{config.training.seed}:{current_epoch}",
            include_pairwise_labels=pairwise_head is not None,
        )
        loader = DataLoader(
            epoch_rows,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        model.train()
        if pairwise_head is not None:
            pairwise_head.train()
        accumulated = 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            if global_step >= target_steps:
                break
            try:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                pairwise_labels = batch.pop("pairwise_labels", None)
                prompt_lengths = batch.pop("prompt_lengths", None)
                if global_step == 0 and accumulated == 0:
                    reporter.reset_peak()
                outputs = model(
                    **batch,
                    output_hidden_states=pairwise_head is not None,
                    return_dict=True,
                )
                lm_loss = outputs.loss
                pairwise_loss = None
                raw_loss = lm_loss
                if pairwise_head is not None:
                    pairwise_loss, _ = compute_pairwise_auxiliary(
                        outputs,
                        pairwise_head,
                        prompt_lengths,
                        pairwise_labels,
                    )
                    raw_loss = combine_training_losses(
                        lm_loss,
                        pairwise_loss,
                        config.training.pairwise_loss_weight,
                    )
                loss_value = float(raw_loss.item())
                lm_loss_value = float(lm_loss.item())
                pairwise_loss_value = (
                    0.0 if pairwise_loss is None else float(pairwise_loss.item())
                )
                (raw_loss / config.training.gradient_accumulation_steps).backward()
            except Exception as error:
                if not is_cuda_out_of_memory(error):
                    raise
                oom_skips += 1
                accumulated = 0
                batch = None
                raw_loss = None
                outputs = None
                lm_loss = None
                pairwise_loss = None
                pairwise_labels = None
                prompt_lengths = None
                recover_from_cuda_oom(torch, optimizer)
                print(
                    f"WARNING: CUDA OOM at epoch={epoch} batch={batch_index}; "
                    f"discarded partial gradient accumulation and skipped batch "
                    f"({oom_skips}/{MAX_OOM_SKIPS})"
                )
                if oom_skips > MAX_OOM_SKIPS:
                    raise RuntimeError(
                        f"CUDA OOM occurred more than {MAX_OOM_SKIPS} times; "
                        "reduce model.max_pixels before resuming"
                    ) from error
                save_checkpoint(
                    model, optimizer, scheduler, config, layout,
                    pairwise_head=pairwise_head,
                    global_step=global_step, epoch=epoch,
                    next_batch_index=batch_index + 1, metrics=metrics, reporter=reporter,
                )
                print(f"Saved OOM recovery checkpoint at step {global_step}")
                continue
            running_loss += loss_value
            running_lm_loss += lm_loss_value
            running_pairwise_loss += pairwise_loss_value
            running_batches += 1
            accumulated += 1
            end_of_epoch = batch_index + 1 == len(loader)
            if accumulated < config.training.gradient_accumulation_steps and not end_of_epoch:
                continue
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.training.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1
            if global_step == 1:
                reporter.record_peak("peak_training_step")

            if global_step == 1 or global_step % config.training.log_every_steps == 0:
                average_loss = running_loss / max(running_batches, 1)
                average_lm_loss = running_lm_loss / max(running_batches, 1)
                average_pairwise_loss = (
                    running_pairwise_loss / max(running_batches, 1)
                )
                elapsed_minutes = (time.perf_counter() - started) / 60
                print(
                    f"epoch={epoch} step={global_step}/{target_steps} "
                    f"loss={average_loss:.6f} lm_loss={average_lm_loss:.6f} "
                    f"pairwise_loss={average_pairwise_loss:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.8g} "
                    f"elapsed_min={elapsed_minutes:.1f}"
                )
                running_loss = 0.0
                running_lm_loss = 0.0
                running_pairwise_loss = 0.0
                running_batches = 0

            next_batch = batch_index + 1
            if interrupt_requested:
                save_checkpoint(
                    model, optimizer, scheduler, config, layout,
                    pairwise_head=pairwise_head,
                    global_step=global_step, epoch=epoch,
                    next_batch_index=next_batch, metrics=metrics, reporter=reporter,
                )
                print(f"Saved interrupted run at step {global_step}")
                stop = True
                break
            if global_step % config.training.save_every_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, config, layout,
                    pairwise_head=pairwise_head,
                    global_step=global_step, epoch=epoch,
                    next_batch_index=next_batch, metrics=metrics, reporter=reporter,
                )

            if args.tiny_overfit and (global_step % 50 == 0 or global_step == target_steps):
                result = evaluate_rows(
                    model, processor, validation_rows, args.image_root, config, mode="generate"
                )
                if pairwise_head is not None:
                    result.update(
                        evaluate_pairwise_head(
                            model,
                            pairwise_head,
                            processor,
                            validation_rows,
                            args.image_root,
                            config,
                        )
                    )
                metrics.append({"step": global_step, "split": "tiny_train", **result})
                print_evaluation(f"tiny step={global_step}", result)
                pairwise_pass = (
                    pairwise_head is None
                    or result["aux_pairwise_accuracy"] >= config.training.tiny_success_accuracy
                )
                if (
                    result["exact_match_accuracy"] >= config.training.tiny_success_accuracy
                    and pairwise_pass
                ):
                    print(f"TINY OVERFIT PASS at step {global_step}")
                    stop = True
            elif (
                not args.tiny_overfit
                and global_step % config.training.fast_validation_every_steps == 0
            ):
                result = evaluate_rows(
                    model, processor, fast_rows, args.image_root, config, mode="generate"
                )
                metrics.append({"step": global_step, "epoch": epoch + 1, "split": "validation_fast", **result})
                print_evaluation(f"validation_fast step={global_step}", result)

            if stop or global_step >= target_steps:
                break

        if interrupt_requested:
            break
        completed_epoch = epoch + 1
        if not args.tiny_overfit and start_batch_index < len(loader):
            result = evaluate_rows(
                model, processor, epoch_validation_rows, args.image_root, config, mode="generate"
            )
            if pairwise_head is not None:
                result.update(
                    evaluate_pairwise_head(
                        model,
                        pairwise_head,
                        processor,
                        epoch_validation_rows,
                        args.image_root,
                        config,
                    )
                )
            metrics.append({"step": global_step, "epoch": completed_epoch, "split": "validation_epoch", **result})
            print_evaluation(f"validation_epoch epoch={completed_epoch}", result)
            if interrupt_requested:
                stop = True
            score = float(result["exact_match_accuracy"])
            pairwise = float(result["pairwise_accuracy"])
            share = float(result["max_prediction_share"])
            improved = (score, pairwise, -share) > (best_score, best_pairwise, -best_share)
            if improved:
                best_score, best_pairwise, best_share = score, pairwise, share
                save_checkpoint(
                    model, optimizer, scheduler, config, best_layout,
                    pairwise_head=pairwise_head,
                    global_step=global_step, epoch=completed_epoch,
                    next_batch_index=0, metrics=metrics, reporter=reporter,
                )
                metadata = json.loads(best_layout.metadata_path.read_text(encoding="utf-8"))
                metadata.update({
                    "best_exact_match_accuracy": best_score,
                    "best_pairwise_accuracy": best_pairwise,
                    "best_max_prediction_share": best_share,
                })
                write_json(best_layout.metadata_path, metadata)
                print(f"Saved new best checkpoint to {best_layout.root}")
            if completed_epoch == 1 and (
                score <= config.training.success_accuracy
                or share >= config.training.collapse_threshold
            ):
                print(
                    "EARLY STOP: epoch 1 did not clear the v3 acceptance gate; "
                    "inspect completion loss and prediction diagnostics before continuing"
                )
                stop = True
        epoch += 1
        start_batch_index = 0
        save_checkpoint(
            model, optimizer, scheduler, config, layout,
            pairwise_head=pairwise_head,
            global_step=global_step, epoch=epoch,
            next_batch_index=0, metrics=metrics, reporter=reporter,
        )

    if args.tiny_overfit and not stop:
        print(f"TINY OVERFIT FAIL after {global_step} steps")
    if not args.tiny_overfit:
        status = "PASS" if (
            best_score > config.training.success_accuracy
            and best_share < config.training.collapse_threshold
        ) else "FAIL"
        print(
            f"VALIDATION {status}: best_exact_match={best_score:.4f} "
            f"max_class_share={best_share:.4f}"
        )
    reporter.print_report()
    signal.signal(signal.SIGINT, previous_sigint)
    print(f"Artifacts saved to {layout.root}")


def main(default_tiny: bool = False) -> None:
    args = parse_args(default_tiny=default_tiny)
    layout = ArtifactLayout(args.output_dir)
    layout.create()
    with (layout.root / "train.log").open("a", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(TeeStream(sys.stdout, log_handle)), contextlib.redirect_stderr(
            TeeStream(sys.stderr, log_handle)
        ):
            print(f"=== Candidate1 v3 train start: {Path.cwd()} ===")
            print(f"argv: {' '.join(sys.argv)}")
            try:
                run_training(args)
            except Exception:
                print("UNHANDLED EXCEPTION DURING TRAINING")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
