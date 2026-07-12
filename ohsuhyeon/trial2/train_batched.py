"""
Trial 2 — batched QLoRA training for a big-VRAM box (A100).

Same recipe/masked-loss as train.py, but uses a padding collate so we can run
micro-batch > 1 (train.py is micro-batch=1 for the 24GB 3090). Keeps effective
batch = SNU_MICRO_BATCH * SNU_ACCUM ~ 8 to stay comparable to the 2B run.

Padding: right-padded; pad positions get label -100 and attention_mask 0, so a
batched step is mathematically identical to summing single-example masked losses.

Env: SNU_MICRO_BATCH (4) SNU_ACCUM (2) SNU_EPOCHS (2) SNU_LR (1e-4)
     SNU_LORA_R (32) SNU_EVAL_EVERY (150 optim steps) SNU_EVAL_N (120)
     SNU_NUM_WORKERS (8) SNU_GRAD_CKPT (1)  plus all of common.py's knobs.
"""
import ast
import json
import os
import random
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

import common as C
from train import LORA_TARGETS, make_split, build_masked_inputs, load_or_init

EVAL_MODE = os.environ.get("SNU_EVAL_MODE", "gen")  # "gen" (fast) or "score"


@torch.no_grad()
def quick_val(model, processor, val_df, n):
    """Fast greedy-generation exact-match monitor. For big models the 24-way
    constrained scoring is too slow to run mid-training, so we generate.

    Frees the training allocator cache before/after so generate's KV-cache
    allocation doesn't thrash against reserved-but-fragmented memory (this was
    causing eval to stall at high memory with the GPU idle)."""
    import gc
    sub = val_df.iloc[:n]
    use_score = EVAL_MODE == "score"
    gc.collect(); torch.cuda.empty_cache()
    if not use_score:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
    model.eval()
    preds, truths = [], []
    for _, row in sub.iterrows():
        msgs = C.build_messages(row, C.TRAIN_IMG_DIR)
        io = C.score_orders(model, processor, msgs) if use_score else C.generate_order(model, processor, msgs)
        preds.append(C.answer_from_image_order(io))
        truths.append(ast.literal_eval(row["Answer"]))
    model.train()
    if not use_score:
        model.config.use_cache = False
        if GRAD_CKPT:
            model.gradient_checkpointing_enable()
    gc.collect(); torch.cuda.empty_cache()
    return C.exact_match(preds, truths)

MICRO = int(os.environ.get("SNU_MICRO_BATCH", "4"))
ACCUM = int(os.environ.get("SNU_ACCUM", "2"))
EPOCHS = int(os.environ.get("SNU_EPOCHS", "2"))
LR = float(os.environ.get("SNU_LR", "1e-4"))
LORA_R = int(os.environ.get("SNU_LORA_R", "32"))
EVAL_EVERY = int(os.environ.get("SNU_EVAL_EVERY", "150"))
EVAL_N = int(os.environ.get("SNU_EVAL_N", "120"))
NUM_WORKERS = int(os.environ.get("SNU_NUM_WORKERS", "8"))
GRAD_CKPT = os.environ.get("SNU_GRAD_CKPT", "1") == "1"
SEED = 42


class OrderDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        inputs, labels = build_masked_inputs(self.processor, row)
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels[0],
            "pixel_values": inputs["pixel_values"],       # [num_patches, dim] (flat)
            "image_grid_thw": inputs["image_grid_thw"],   # [4, 3]
        }


def make_collate(pad_id):
    def collate(batch):
        seqs = [b["input_ids"] for b in batch]
        maxL = max(s.size(0) for s in seqs)

        def pad(key, val, dtype=None):
            out = torch.full((len(batch), maxL), val,
                             dtype=dtype or batch[0][key].dtype)
            for i, b in enumerate(batch):
                out[i, :b[key].size(0)] = b[key]
            return out

        batched = {
            "input_ids": pad("input_ids", pad_id),
            "attention_mask": pad("attention_mask", 0),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
            "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
        }
        labels = pad("labels", -100)
        return batched, labels
    return collate


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    train_df, val_df = make_split()
    processor = C.load_processor()
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad_id = processor.tokenizer.pad_token_id

    model = load_or_init(C.load_base_model())
    if not GRAD_CKPT:
        model.gradient_checkpointing_disable()

    ds = OrderDataset(train_df, processor)
    loader = DataLoader(
        ds, batch_size=MICRO, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=make_collate(pad_id), persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    C.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    prog_path = C.ADAPTER_DIR / "progress.json"
    best_path = C.ADAPTER_DIR / "best"
    prog = {"best_acc": -1.0, "history": []}

    total_optim = (len(loader) // ACCUM) * EPOCHS
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_optim), total_optim)

    optim_step = 0
    micro = 0
    start = time.perf_counter()
    model.train(); optim.zero_grad()
    for epoch in range(EPOCHS):
        for batched, labels in loader:
            micro += 1
            batched = {k: v.to(model.device) for k, v in batched.items()}
            labels = labels.to(model.device)
            loss = model(**batched, labels=labels).loss / ACCUM
            loss.backward()
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                optim_step += 1
                if optim_step % 10 == 0:
                    el = (time.perf_counter() - start) / 60
                    print(f"epoch {epoch} optim_step {optim_step}/{total_optim} "
                          f"loss={loss.item() * ACCUM:.4f} lr={sched.get_last_lr()[0]:.2e} "
                          f"{el:.1f}min", flush=True)
                if optim_step % EVAL_EVERY == 0:
                    acc = quick_val(model, processor, val_df, EVAL_N)
                    prog["history"].append({"optim_step": optim_step, "val_acc": acc})
                    print(f">>> VAL optim_step {optim_step}: exact_match={acc:.4f} "
                          f"(best {max(prog['best_acc'], acc):.4f})", flush=True)
                    if acc > prog["best_acc"]:
                        prog["best_acc"] = acc
                        model.save_pretrained(best_path)
                        print(f">>> new best {acc:.4f} -> {best_path}", flush=True)
                    model.save_pretrained(C.ADAPTER_DIR)
                    prog_path.write_text(json.dumps(prog, indent=2))

    model.save_pretrained(C.ADAPTER_DIR)
    acc = quick_val(model, processor, val_df, min(len(val_df), 400))
    prog["history"].append({"optim_step": optim_step, "val_acc_final400": acc})
    if acc > prog["best_acc"]:
        prog["best_acc"] = acc
        model.save_pretrained(best_path)
    prog_path.write_text(json.dumps(prog, indent=2))
    print(f"DONE. best_acc={prog['best_acc']:.4f} final400={acc:.4f} "
          f"took {(time.perf_counter() - start) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
