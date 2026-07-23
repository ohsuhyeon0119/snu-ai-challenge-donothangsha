import argparse
import ast
import csv
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
    import hashlib

    from snu_frame_ordering.orders import SIGMAS

    digest = hashlib.sha1(f"{seed}:{epoch}:{sample_id}".encode("utf-8")).digest()
    return SIGMAS[digest[0] % len(SIGMAS)]

def evaluate_exact_match(model, processor, rows, score_chunk):
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer
    from snu_frame_ordering.paligemma_common import predict_order

    model.eval()
    correct = 0
    total = 0
    for row in tqdm(rows, desc="val", leave=False):
        expected = image_order_from_answer(parse_answer(row["Answer"]))
        predicted = predict_order(model, processor, row, "train", tta_k=1, chunk=score_chunk)
        correct += int(predicted == expected)
        total += 1
    model.train()
    return correct, total, correct / total if total else 0.0


def save_adapter(model, processor, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def stratified_split(rows, val_size, seed):
    if val_size <= 0:
        return [], list(rows)

    groups = {}
    for row in rows:
        groups.setdefault(str(parse_answer(row["Answer"])), []).append(row)

    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = len(rows)
    val_size = min(val_size, max(total - 1, 0))
    quotas = []
    assigned = 0
    for label, group_rows in groups.items():
        raw = val_size * len(group_rows) / total
        take = int(raw)
        quotas.append([raw - take, label, take])
        assigned += take

    for _, label, _ in sorted(quotas, reverse=True)[: val_size - assigned]:
        for quota in quotas:
            if quota[1] == label:
                quota[2] += 1
                break

    val_rows = []
    train_rows = []
    quota_by_label = {label: take for _, label, take in quotas}
    for label, group_rows in groups.items():
        take = min(quota_by_label[label], max(len(group_rows) - 1, 0))
        val_rows.extend(group_rows[:take])
        train_rows.extend(group_rows[take:])

    rng.shuffle(val_rows)
    rng.shuffle(train_rows)
    return val_rows, train_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--score-chunk", type=int, default=int(os.environ.get("SNU_SCORE_CHUNK", "1")))
    args = parser.parse_args()

    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer, slot_order, target_text
    from snu_frame_ordering.paligemma_common import (
        encode_supervised,
        load_base_model,
        load_processor,
        lora_target_modules,
        model_device,
        row_images,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "train.csv")
    rows = df.to_dict("records")
    val_size = min(max(args.val_size, 0), max(len(rows) - 1, 0))
    val_rows, train_rows = stratified_split(rows, val_size, args.seed)
    if not train_rows:
        raise ValueError("No training rows left after validation split")

    processor = load_processor()
    model = load_base_model(load_4bit=True)
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(model),
    )
    model = get_peft_model(model, config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = adapter_dir / "train_log.csv"
    eval_log_path = adapter_dir / "eval_log.csv"
    best_dir = adapter_dir / "best"
    best_acc = -1.0
    best_metrics = None
    start_time = time.monotonic()
    last_eval_step = 0

    with log_path.open("w", encoding="utf-8", newline="") as log_file, eval_log_path.open(
        "w", encoding="utf-8", newline=""
    ) as eval_file:
        logger = csv.DictWriter(log_file, fieldnames=["step", "epoch", "row_id", "loss", "elapsed_seconds"])
        eval_logger = csv.DictWriter(eval_file, fieldnames=["step", "epoch", "correct", "total", "accuracy", "elapsed_seconds"])
        logger.writeheader()
        eval_logger.writeheader()
        global_step = 0
        epoch = 0
        progress = tqdm(total=args.max_steps)

        while global_step < args.max_steps:
            epoch += 1
            random.shuffle(train_rows)
            loader = DataLoader(FrameOrderingDataset(train_rows), batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

            for row in loader:
                sigma = sigma_for(row["Id"], epoch, args.seed)
                images = row_images(row, "train", sigma=sigma)
                answer = parse_answer(row["Answer"])
                image_order = image_order_from_answer(answer)
                completion = target_text(slot_order(image_order, sigma))
                batch = encode_supervised(processor, images, row["Sentence"], completion).to(model_device(model))
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                elapsed = time.monotonic() - start_time
                loss_value = float(loss.detach().cpu())
                progress.update(1)
                progress.set_postfix(epoch=epoch, loss=f"{loss_value:.4f}", best=f"{best_acc:.4f}")

                if args.log_every > 0 and (global_step == 1 or global_step % args.log_every == 0):
                    logger.writerow(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "row_id": row["Id"],
                            "loss": loss_value,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    log_file.flush()

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_adapter(model, processor, adapter_dir / f"checkpoint-{global_step}")

                should_eval = val_rows and args.eval_every > 0 and global_step % args.eval_every == 0
                should_eval = should_eval or (val_rows and global_step == args.max_steps)
                if should_eval and last_eval_step != global_step:
                    last_eval_step = global_step
                    correct, total, accuracy = evaluate_exact_match(model, processor, val_rows, args.score_chunk)
                    elapsed = time.monotonic() - start_time
                    eval_logger.writerow(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "correct": correct,
                            "total": total,
                            "accuracy": accuracy,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    eval_file.flush()
                    progress.write(f"val step={global_step} epoch={epoch} accuracy={accuracy:.4f} ({correct}/{total})")

                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_metrics = {
                            "step": global_step,
                            "epoch": epoch,
                            "correct": correct,
                            "total": total,
                            "accuracy": accuracy,
                            "val_size": val_size,
                            "train_size": len(train_rows),
                            "seed": args.seed,
                            "elapsed_seconds": elapsed,
                        }
                        save_adapter(model, processor, best_dir)
                        (best_dir / "best_metrics.json").write_text(
                            json.dumps(best_metrics, indent=2), encoding="utf-8"
                        )

                if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
                    progress.write(f"time limit reached at step={global_step}")
                    break

                if global_step >= args.max_steps:
                    break

            if args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds:
                break

        progress.close()

    if val_rows and last_eval_step != global_step:
        correct, total, accuracy = evaluate_exact_match(model, processor, val_rows, args.score_chunk)
        elapsed = time.monotonic() - start_time
        with eval_log_path.open("a", encoding="utf-8", newline="") as eval_file:
            eval_logger = csv.DictWriter(eval_file, fieldnames=["step", "epoch", "correct", "total", "accuracy", "elapsed_seconds"])
            eval_logger.writerow(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                    "elapsed_seconds": elapsed,
                }
            )
        print(f"val step={global_step} epoch={epoch} accuracy={accuracy:.4f} ({correct}/{total})")
        if accuracy > best_acc:
            best_acc = accuracy
            best_metrics = {
                "step": global_step,
                "epoch": epoch,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "val_size": val_size,
                "train_size": len(train_rows),
                "seed": args.seed,
                "elapsed_seconds": elapsed,
            }
            save_adapter(model, processor, best_dir)
            (best_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    save_adapter(model, processor, adapter_dir)
    if best_metrics:
        (adapter_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(adapter_dir)
    if best_metrics:
        print(f"best_adapter={best_dir} accuracy={best_metrics['accuracy']:.4f} step={best_metrics['step']}")


if __name__ == "__main__":
    main()
