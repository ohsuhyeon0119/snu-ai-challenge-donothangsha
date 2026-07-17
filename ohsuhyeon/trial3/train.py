"""
Trial 3 training — Qwen3-VL-8B bf16 LoRA, masked SFT with presentation-shuffle
augmentation. Built for a single A100 80GB (vast.ai), driven inside tmux.

What it does
  - LoRA (r=SNU_LORA_R, LLM decoder linears only) on the bf16 base. No 4-bit.
  - Loss masked to the answer completion tokens only.
  - Every epoch each sample is re-presented under a deterministic pseudo-random
    sigma (frame presentation order), label remapped accordingly -> the model
    cannot learn input-position priors.
  - JSONL step log + checkpoint every SNU_SAVE_STEPS optim steps + greedy
    val monitor -> best/ tracks the highest val exact-match adapter.
  - Resumable: picks up from the newest step checkpoint (optimizer+scheduler
    state included; dataloader fast-forwards within the epoch).

Env knobs (defaults sized for A100 80GB):
  SNU_TRAIN_CSV=data_work/train_clean.csv   SNU_VAL_CSV=data_work/clean_val.csv
  SNU_CKPT_DIR=ckpts   SNU_MB=8   SNU_ACCUM=2   SNU_EPOCHS=2   SNU_LR=1e-4
  SNU_LORA_R=32   SNU_SAVE_STEPS=150   SNU_MONITOR_N=120   SNU_SEED=42
"""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import (SIGMAS, TRAIN_IMG_DIR, build_messages,
                    image_order_from_answer, load_model, load_processor,
                    lora_target_modules, slot_order, target_text)

WORK_DIR = os.environ.get("SNU_WORK_DIR", "data_work")
TRAIN_CSV = Path(os.environ.get("SNU_TRAIN_CSV", f"{WORK_DIR}/train_clean.csv"))
VAL_CSV = Path(os.environ.get("SNU_VAL_CSV", f"{WORK_DIR}/clean_val.csv"))
CKPT_DIR = Path(os.environ.get("SNU_CKPT_DIR", "ckpts"))
MB = int(os.environ.get("SNU_MB", "8"))
ACCUM = int(os.environ.get("SNU_ACCUM", "2"))
EPOCHS = int(os.environ.get("SNU_EPOCHS", "2"))
LR = float(os.environ.get("SNU_LR", "1e-4"))
LORA_R = int(os.environ.get("SNU_LORA_R", "32"))
SAVE_STEPS = int(os.environ.get("SNU_SAVE_STEPS", "150"))
MONITOR_N = int(os.environ.get("SNU_MONITOR_N", "120"))
SEED = int(os.environ.get("SNU_SEED", "42"))
WARMUP_FRAC = 0.03


SHUFFLE_EPOCH0 = os.environ.get("SNU_SHUFFLE_EPOCH0", "1") == "1"


def sigma_for(sample_id, epoch):
    """Deterministic per-(sample, epoch) presentation order.

    Default: shuffle every epoch (incl. 0) — training then matches the
    inference-time TTA decision rule, which scores under shuffled sigmas.
    Set SNU_SHUFFLE_EPOCH0=0 to keep epoch 0 canonical (identity) instead."""
    if epoch == 0 and not SHUFFLE_EPOCH0:
        return SIGMAS[0]
    h = hashlib.sha1(f"{SEED}:{epoch}:{sample_id}".encode()).digest()
    return SIGMAS[h[0] % len(SIGMAS)]


class Rows(Dataset):
    def __init__(self, df):
        self.rows = df.to_dict("records")
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        sigma = sigma_for(row["Id"], self.epoch)
        answer = json.loads(row["Answer"])
        so = slot_order(image_order_from_answer(answer), sigma)
        messages = build_messages(row, TRAIN_IMG_DIR, sigma=sigma)
        return messages, target_text(so)


def make_collate(processor):
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        from qwen_vl_utils import process_vision_info
        texts, images, tlens = [], [], []
        for messages, target in batch:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            texts.append(prompt + target)
            imgs, _ = process_vision_info(messages)
            images.append(imgs)
            # target is pure text (no vision tokens) -> its token length is
            # stable under the processor's vision expansion
            tlens.append(len(tok(target, add_special_tokens=False).input_ids))
        enc = [processor(text=[t], images=im, return_tensors="pt")
               for t, im in zip(texts, images)]
        L = max(e["input_ids"].shape[1] for e in enc)
        B = len(enc)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for b, (e, tl) in enumerate(zip(enc, tlens)):
            n = e["input_ids"].shape[1]
            input_ids[b, :n] = e["input_ids"][0]
            attn[b, :n] = 1
            labels[b, n - tl:n] = e["input_ids"][0, n - tl:]  # completion only
        pixel_values = torch.cat([e["pixel_values"] for e in enc], dim=0)
        grid = torch.cat([e["image_grid_thw"] for e in enc], dim=0)
        return {"input_ids": input_ids, "attention_mask": attn,
                "labels": labels, "pixel_values": pixel_values,
                "image_grid_thw": grid}

    return collate


@torch.no_grad()
def greedy_monitor(model, processor, val_rows, n):
    """Fast greedy-decode exact-match on n val rows (identity presentation)."""
    import re
    from qwen_vl_utils import process_vision_info
    pat = re.compile(r"\[([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\]")
    model.eval()
    correct = 0
    rows = val_rows[:n]
    for row in rows:
        messages = build_messages(row, TRAIN_IMG_DIR)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        inputs = processor(text=[prompt], images=imgs,
                           return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = pat.search(text)
        pred = [int(m.group(i)) for i in range(1, 5)] if m else [1, 2, 3, 4]
        if sorted(pred) != [1, 2, 3, 4]:
            pred = [1, 2, 3, 4]
        truth = image_order_from_answer(json.loads(row["Answer"]))
        correct += int(pred == truth)
    model.train()
    return correct / max(1, len(rows))


def save_ckpt(model, opt, sched, step, epoch, in_epoch_step, tag=None):
    d = CKPT_DIR / (tag or f"ckpt-{step}")
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)  # LoRA adapter only (~200MB)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "step": step, "epoch": epoch, "in_epoch_step": in_epoch_step},
               d / "trainer_state.pt")
    return d


def latest_ckpt():
    if not CKPT_DIR.exists():
        return None
    cs = sorted(CKPT_DIR.glob("ckpt-*"),
                key=lambda p: int(p.name.split("-")[1]))
    return cs[-1] if cs else None


def prune_ckpts(keep=3):
    cs = sorted(CKPT_DIR.glob("ckpt-*"),
                key=lambda p: int(p.name.split("-")[1]))
    for c in cs[:-keep]:
        shutil.rmtree(c, ignore_errors=True)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)
    val_rows = pd.read_csv(VAL_CSV).to_dict("records")
    print(f"train={len(df)} val={len(val_rows)} mb={MB} accum={ACCUM} "
          f"epochs={EPOCHS} lora_r={LORA_R}")

    processor = load_processor()
    model = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=lora_target_modules(model))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = Rows(df)
    steps_per_epoch = (len(ds) + MB * ACCUM - 1) // (MB * ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)

    start_step = start_epoch = start_in_epoch = 0
    best_acc = -1.0
    ck = latest_ckpt()
    if ck is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(ck / "adapter_model.safetensors")
        set_peft_model_state_dict(model, sd)
        st = torch.load(ck / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_step, start_epoch = st["step"], st["epoch"]
        start_in_epoch = st["in_epoch_step"]
        bf = CKPT_DIR / "best" / "acc.json"
        if bf.exists():
            best_acc = json.loads(bf.read_text())["val_acc"]
        print(f"resumed from {ck} (step {start_step})")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(CKPT_DIR / "train_log.jsonl", "a")
    collate = make_collate(processor)
    model.train()
    step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        ds.epoch = epoch
        gen = torch.Generator().manual_seed(SEED + epoch)
        dl = DataLoader(ds, batch_size=MB, shuffle=True, generator=gen,
                        num_workers=8, collate_fn=collate, drop_last=False)
        it = iter(dl)
        in_epoch = 0
        # fast-forward after resume (same generator -> same order)
        skip_batches = start_in_epoch * ACCUM if epoch == start_epoch else 0
        for _ in range(skip_batches):
            next(it, None)
        in_epoch = start_in_epoch if epoch == start_epoch else 0

        while True:
            opt.zero_grad(set_to_none=True)
            got = 0
            loss_sum = 0.0
            for _ in range(ACCUM):
                batch = next(it, None)
                if batch is None:
                    break
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)
                if not torch.isfinite(out.loss):
                    raise RuntimeError(
                        f"non-finite loss at step {step} - aborting so the "
                        f"box doesn't burn money; resume from last ckpt with "
                        f"a lower SNU_LR")
                (out.loss / ACCUM).backward()
                loss_sum += out.loss.item()
                got += 1
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            in_epoch += 1

            rec = {"step": step, "epoch": epoch, "loss": loss_sum / got,
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 10 == 0:
                print(f"step {step}/{total_steps} epoch {epoch} "
                      f"loss {rec['loss']:.4f} lr {rec['lr']:.2e}")

            if step % SAVE_STEPS == 0:
                d = save_ckpt(model, opt, sched, step, epoch, in_epoch)
                torch.cuda.empty_cache()
                model.config.use_cache = True
                acc = greedy_monitor(model, processor, val_rows, MONITOR_N)
                model.config.use_cache = False
                torch.cuda.empty_cache()
                rec = {"step": step, "val_acc": acc,
                       "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[val] step {step} greedy exact-match {acc:.4f} "
                      f"(best {max(best_acc, acc):.4f})")
                if acc > best_acc:
                    best_acc = acc
                    best = CKPT_DIR / "best"
                    shutil.rmtree(best, ignore_errors=True)
                    shutil.copytree(d, best)
                    (best / "acc.json").write_text(
                        json.dumps({"val_acc": acc, "step": step}))
                prune_ckpts(keep=3)

    save_ckpt(model, opt, sched, step, EPOCHS - 1, 0, tag=f"ckpt-{step}")
    print(f"done. steps={step} best_val={best_acc:.4f} "
          f"wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
