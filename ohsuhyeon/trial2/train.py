"""
Trial 2 training — QLoRA fine-tune with the trial1 bugs fixed.

Key deltas vs trial1 (docs/superpowers/specs/2026-07-12-frame-ordering-trial2-design.md):
  1. MASKED LOSS: labels = -100 for the whole prompt + vision tokens; loss is
     computed only on the answer completion tokens. (trial1 masked only padding,
     so the ~10 answer tokens were <1% of the loss and the model collapsed to the
     marginal answer distribution -> 16.77%.)
  2. LoRA on all linear projections (q,k,v,o,gate,up,down), rank 32.
  3. Cosine LR schedule with warmup.
  4. Periodic constrained-scoring validation on a fixed subsample -> keep BEST
     adapter, not just the last.
  5. Lower max_pixels (in common.py) to cut vision tokens -> faster steps.

Env knobs (plus the ones in common.py):
  SNU_EPOCHS (3) SNU_LR (1e-4) SNU_ACCUM (8) SNU_LORA_R (32)
  SNU_EVAL_EVERY (400 optim steps) SNU_EVAL_N (160 val rows) SNU_VAL_FRAC (0.12)
"""
import json
import os
import random
import time

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_cosine_schedule_with_warmup

import common as C

EPOCHS = int(os.environ.get("SNU_EPOCHS", "3"))
LR = float(os.environ.get("SNU_LR", "1e-4"))
ACCUM = int(os.environ.get("SNU_ACCUM", "8"))
LORA_R = int(os.environ.get("SNU_LORA_R", "32"))
EVAL_EVERY = int(os.environ.get("SNU_EVAL_EVERY", "400"))
EVAL_N = int(os.environ.get("SNU_EVAL_N", "120"))
VAL_FRAC = float(os.environ.get("SNU_VAL_FRAC", "0.12"))
SEED = 42
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_split():
    df = pd.read_csv(C.DATA_DIR / "train.csv")
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRAC, random_state=SEED, stratify=df["Answer"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    C.VAL_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(C.VAL_SPLIT_CSV, index=False)
    # fixed deterministic training order so resume-by-skip is exact
    order = list(range(len(train_df)))
    random.Random(SEED).shuffle(order)
    return train_df.iloc[order].reset_index(drop=True), val_df


def build_masked_inputs(processor, row):
    """One example -> (inputs, labels) with the prompt fully masked to -100."""
    messages = C.build_messages(row, C.TRAIN_IMG_DIR)
    from qwen_vl_utils import process_vision_info
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    io = C.image_order_from_answer(__import__("ast").literal_eval(row["Answer"]))
    completion = C.target_text(io) + processor.tokenizer.eos_token
    full = processor(text=[prompt_text + completion], images=image_inputs, return_tensors="pt")
    prompt_only = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
    plen = prompt_only["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[:, :plen] = -100  # mask prompt + all vision tokens; grade only the answer
    return full, labels


@torch.no_grad()
def quick_val(model, processor, val_df, n):
    model.eval()
    sub = val_df.iloc[:n]
    preds, truths = [], []
    for _, row in sub.iterrows():
        io = C.score_orders(model, processor, C.build_messages(row, C.TRAIN_IMG_DIR))
        preds.append(C.answer_from_image_order(io))
        truths.append(__import__("ast").literal_eval(row["Answer"]))
    model.train()
    return C.exact_match(preds, truths)


def load_or_init(model_base):
    if (C.ADAPTER_DIR / "adapter_config.json").exists():
        print(f"resuming adapter from {C.ADAPTER_DIR}", flush=True)
        model = PeftModel.from_pretrained(model_base, C.ADAPTER_DIR, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        )
        model = get_peft_model(model_base, cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    train_df, val_df = make_split()
    processor = C.load_processor()
    model = load_or_init(C.load_base_model())

    C.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    prog_path = C.ADAPTER_DIR / "progress.json"
    best_path = C.ADAPTER_DIR / "best"
    prog = {"micro_step": 0, "best_acc": -1.0, "history": []}
    if prog_path.exists():
        prog = json.loads(prog_path.read_text())

    n = len(train_df)
    total_micro = n * EPOCHS
    total_optim = total_micro // ACCUM
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_optim), total_optim)
    for _ in range(prog["micro_step"] // ACCUM):
        sched.step()

    micro = 0
    optim_step = prog["micro_step"] // ACCUM
    start = time.perf_counter()
    model.train()
    optim.zero_grad()
    for epoch in range(EPOCHS):
        for _, row in train_df.iterrows():
            micro += 1
            if micro <= prog["micro_step"]:
                continue
            inputs, labels = build_masked_inputs(processor, row)
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            loss = model(**inputs, labels=labels).loss / ACCUM
            loss.backward()

            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                optim_step += 1
                if optim_step % 20 == 0:
                    el = (time.perf_counter() - start) / 60
                    print(f"epoch {epoch} optim_step {optim_step}/{total_optim} "
                          f"loss={loss.item() * ACCUM:.4f} lr={sched.get_last_lr()[0]:.2e} "
                          f"{el:.1f}min", flush=True)
                if optim_step % EVAL_EVERY == 0:
                    acc = quick_val(model, processor, val_df, EVAL_N)
                    prog["history"].append({"optim_step": optim_step, "val_acc": acc})
                    print(f">>> VAL optim_step {optim_step}: exact_match={acc:.4f} "
                          f"(best so far {max(prog['best_acc'], acc):.4f})", flush=True)
                    if acc > prog["best_acc"]:
                        prog["best_acc"] = acc
                        model.save_pretrained(best_path)
                        print(f">>> new best {acc:.4f} -> {best_path}", flush=True)
                    prog["micro_step"] = micro
                    model.save_pretrained(C.ADAPTER_DIR)
                    prog_path.write_text(json.dumps(prog, indent=2))

    prog["micro_step"] = micro
    model.save_pretrained(C.ADAPTER_DIR)
    # final full-ish check on a larger subsample, and ensure a best exists
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
