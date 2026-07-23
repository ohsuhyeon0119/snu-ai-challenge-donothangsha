import argparse
import ast
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FrameOrderingDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def parse_answer(value):
    return ast.literal_eval(value)


def sigma_for(sample_id, epoch, seed):
    from snu_frame_ordering.orders import SIGMAS

    digest = hashlib.sha1(f"{seed}:{epoch}:{sample_id}".encode("utf-8")).digest()
    return SIGMAS[digest[0] % len(SIGMAS)]


def resolve_train_csv(data_dir, explicit):
    if explicit:
        return Path(explicit)
    clean = Path(os.environ.get("SNU_WORK_DIR", "data_work")) / "train_clean.csv"
    return clean if clean.exists() else Path(data_dir) / "train.csv"


def resolve_val_csv(data_dir, explicit):
    if explicit:
        return Path(explicit)
    clean = Path(os.environ.get("SNU_WORK_DIR", "data_work")) / "clean_val.csv"
    return clean if clean.exists() else None


def evaluate_exact_match(model, processor, rows, split, score_chunk, tta_k):
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer
    from snu_frame_ordering.paligemma_common import predict_order

    model.eval()
    correct = 0
    total = 0
    for row in tqdm(rows, desc="val", leave=False):
        expected = image_order_from_answer(parse_answer(row["Answer"]))
        predicted = predict_order(model, processor, row, split, tta_k=tta_k, chunk=score_chunk)
        correct += int(predicted == expected)
        total += 1
    model.train()
    return correct, total, correct / total if total else 0.0


def save_adapter(model, processor, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def enable_projector_training(model):
    enabled = []
    for name, param in model.named_parameters():
        lname = name.lower()
        if "projector" in lname or "multi_modal" in lname or "multimodal" in lname:
            param.requires_grad = True
            enabled.append(name)
    return enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--train-csv", default=os.environ.get("SNU_TRAIN_CSV", ""))
    parser.add_argument("--val-csv", default=os.environ.get("SNU_VAL_CSV", ""))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SNU_EPOCHS", "1")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("SNU_TRAIN_MAX_STEPS", "0")))
    parser.add_argument("--max-seconds", type=float, default=float(os.environ.get("SNU_TRAIN_SECONDS", "0")))
    parser.add_argument("--micro-batch", type=int, default=int(os.environ.get("SNU_MB", "1")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("SNU_ACCUM", "8")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("SNU_LR", "2e-4")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("SNU_LORA_R", "16")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SNU_SEED", "42")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("SNU_LOG_EVERY", "20")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SNU_SAVE_EVERY", "0")))
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("SNU_EVAL_EVERY", "200")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("SNU_EVAL_LIMIT", "200")))
    parser.add_argument("--eval-tta-k", type=int, default=int(os.environ.get("SNU_EVAL_TTA_K", "1")))
    parser.add_argument("--score-chunk", type=int, default=int(os.environ.get("SNU_SCORE_CHUNK", "4")))
    parser.add_argument("--train-projector", action=argparse.BooleanOptionalAction, default=os.environ.get("SNU_TRAIN_PROJECTOR", "1") == "1")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer, slot_order, target_text
    from snu_frame_ordering.paligemma_common import encode_supervised, load_base_model, load_processor, lora_target_modules, model_device, row_images

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = resolve_train_csv(data_dir, args.train_csv)
    val_csv = resolve_val_csv(data_dir, args.val_csv)
    train_rows = pd.read_csv(train_csv).to_dict("records")
    val_rows = pd.read_csv(val_csv).to_dict("records") if val_csv and Path(val_csv).exists() else []
    if args.eval_limit > 0:
        val_rows = val_rows[:args.eval_limit]
    if not train_rows:
        raise ValueError("No training rows found")

    processor = load_processor()
    model = load_base_model(load_4bit=True)
    model = prepare_model_for_kbit_training(model)

    lora_r = args.lora_r
    config = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(model),
    )
    model = get_peft_model(model, config)
    projector_params = enable_projector_training(model) if args.train_projector else []
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = adapter_dir / "train_log.csv"
    eval_log_path = adapter_dir / "eval_log.csv"
    best_dir = adapter_dir / "best"
    best_acc = -1.0
    best_metrics = None
    start_time = time.monotonic()
    max_steps = args.max_steps if args.max_steps > 0 else 10**12
    global_step = 0
    batch_seen = 0

    print(
        f"train_csv={train_csv} train={len(train_rows)} val={len(val_rows)} "
        f"epochs={args.epochs} mb={args.micro_batch} accum={args.grad_accum} "
        f"lora_r={args.lora_r} projector_trainable={len(projector_params)}"
    )

    with log_path.open("w", encoding="utf-8", newline="") as log_file, eval_log_path.open("w", encoding="utf-8", newline="") as eval_file:
        logger = csv.DictWriter(log_file, fieldnames=["step", "epoch", "row_id", "loss", "elapsed_seconds"])
        eval_logger = csv.DictWriter(eval_file, fieldnames=["step", "epoch", "correct", "total", "accuracy", "elapsed_seconds"])
        logger.writeheader()
        eval_logger.writeheader()
        progress = tqdm(total=None if max_steps == 10**12 else max_steps)

        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(train_rows)
            loader = DataLoader(FrameOrderingDataset(train_rows), batch_size=1, shuffle=False, collate_fn=lambda xs: xs[0])
            optimizer.zero_grad(set_to_none=True)

            for row in loader:
                sigma = sigma_for(row["Id"], epoch, args.seed)
                images = row_images(row, "train", sigma=sigma)
                image_order = image_order_from_answer(parse_answer(row["Answer"]))
                completion = target_text(slot_order(image_order, sigma))
                batch = encode_supervised(processor, images, row["Sentence"], completion).to(model_device(model))
                loss = model(**batch).loss / args.grad_accum
                loss.backward()
                batch_seen += 1

                if batch_seen % args.grad_accum != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.monotonic() - start_time
                loss_value = float((loss.detach() * args.grad_accum).cpu())
                progress.update(1)
                progress.set_postfix(epoch=epoch + 1, loss=f"{loss_value:.4f}", best=f"{best_acc:.4f}")

                if args.log_every > 0 and (global_step == 1 or global_step % args.log_every == 0):
                    logger.writerow({"step": global_step, "epoch": epoch + 1, "row_id": row["Id"], "loss": loss_value, "elapsed_seconds": elapsed})
                    log_file.flush()

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_adapter(model, processor, adapter_dir / f"checkpoint-{global_step}")

                should_eval = val_rows and args.eval_every > 0 and global_step % args.eval_every == 0
                if should_eval:
                    correct, total, accuracy = evaluate_exact_match(model, processor, val_rows, "train", args.score_chunk, args.eval_tta_k)
                    elapsed = time.monotonic() - start_time
                    eval_logger.writerow({"step": global_step, "epoch": epoch + 1, "correct": correct, "total": total, "accuracy": accuracy, "elapsed_seconds": elapsed})
                    eval_file.flush()
                    progress.write(f"val step={global_step} epoch={epoch + 1} accuracy={accuracy:.4f} ({correct}/{total})")
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_metrics = {"step": global_step, "epoch": epoch + 1, "correct": correct, "total": total, "accuracy": accuracy, "train_size": len(train_rows), "val_size": len(val_rows), "seed": args.seed, "elapsed_seconds": elapsed}
                        save_adapter(model, processor, best_dir)
                        (best_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
                    model.train()

                if global_step >= max_steps:
                    break
                if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
                    progress.write(f"time limit reached at step={global_step}")
                    break

            if global_step >= max_steps:
                break
            if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
                break

        progress.close()

    if val_rows:
        correct, total, accuracy = evaluate_exact_match(model, processor, val_rows, "train", args.score_chunk, args.eval_tta_k)
        elapsed = time.monotonic() - start_time
        with eval_log_path.open("a", encoding="utf-8", newline="") as eval_file:
            eval_logger = csv.DictWriter(eval_file, fieldnames=["step", "epoch", "correct", "total", "accuracy", "elapsed_seconds"])
            eval_logger.writerow({"step": global_step, "epoch": args.epochs, "correct": correct, "total": total, "accuracy": accuracy, "elapsed_seconds": elapsed})
        print(f"val final step={global_step} accuracy={accuracy:.4f} ({correct}/{total})")
        if accuracy > best_acc:
            best_acc = accuracy
            best_metrics = {"step": global_step, "epoch": args.epochs, "correct": correct, "total": total, "accuracy": accuracy, "train_size": len(train_rows), "val_size": len(val_rows), "seed": args.seed, "elapsed_seconds": elapsed}
            save_adapter(model, processor, best_dir)
            (best_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    save_adapter(model, processor, adapter_dir)
    if best_metrics:
        (adapter_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        print(f"best_adapter={best_dir} accuracy={best_metrics['accuracy']:.4f} step={best_metrics['step']}")
    print(adapter_dir)


if __name__ == "__main__":
    main()
