"""
Trial 4 round 2 — unified training from the BASE model: interleaves masked
SFT and listwise ranking losses on every pass through train_full.csv,
instead of stacking them as two sequential fine-tune stages (round 1's
approach, ckpt-400 -> listwise-best). Design doc:
docs/superpowers/specs/2026-07-21-frame-ordering-trial4-design.md

Per row, a deterministic (seeded, resumable) draw at LISTWISE_PROB picks one
of two branches:
  - SFT branch (~70% of rows): masked next-token loss on the single ground-
    truth completion — train.py's Rows/make_collate mechanism, unchanged,
    just applied one row at a time here (MB=1) instead of train.py's
    batched MBxACCUM, so both branches share one row-by-row loop.
  - listwise branch (~30% of rows): common.broadened_candidate_set
    (adjacent + kendall-2/3 + random, 7 candidates) + listwise ranking
    cross-entropy via candidate_logprob_sum — train_listwise.py's
    mechanism, unchanged. NOT restricted to a mined hard-negative subset:
    there is no trained checkpoint yet at the start of this run to mine
    with, so this substitutes random-row sampling for round 1's margin-
    based mining.

No KV-cache sharing anywhere (that's what OOM'd round 1's first listwise
design at ~93GB) — both branches go through the ordinary batched-forward-
with-gradient-checkpointing path.

Env knobs:
  SNU_TRAIN_CSV      default data_work/train_full.csv (8,871 rows)
  SNU_VAL_CSV        default data_work/val_small.csv (250 rows)
  SNU_CKPT_DIR       default ckpts_unified
  SNU_LISTWISE_PROB  probability a row takes the listwise branch (default 0.30)
  SNU_ACCUM          rows accumulated per optimizer step (default 4)
  SNU_EPOCHS         default 1
  SNU_LR             default 1e-4 (fresh-SFT-scale LR — this is a from-
                     scratch run, not a low-LR fine-tune on a trained ckpt)
  SNU_LORA_R         default 32
  SNU_N_ADJACENT/SNU_N_NEAR/SNU_N_RANDOM2  broadened-candidate counts
                     (default 3/2/1 -> 7 candidates on listwise rows)
  SNU_SAVE_STEPS     checkpoint+monitor cadence (default: total_steps//6,
                     i.e. ~6 checks spread across the whole run)
  SNU_MONITOR_N      default 250 (all of val_small)
  SNU_SEED           default 42
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
import torch.nn.functional as F

import train as sft  # reuse make_collate / score_monitor / save_ckpt / prune_ckpts
from common import (SIGMAS, TRAIN_IMG_DIR, broadened_candidate_set,
                    build_messages, candidate_logprob_sum,
                    image_order_from_answer, load_model, load_processor,
                    lora_target_modules, slot_order, target_text)

WORK_DIR = os.environ.get("SNU_WORK_DIR", "data_work")
TRAIN_CSV = Path(os.environ.get("SNU_TRAIN_CSV", f"{WORK_DIR}/train_full.csv"))
VAL_CSV = Path(os.environ.get("SNU_VAL_CSV", f"{WORK_DIR}/val_small.csv"))
CKPT_DIR = Path(os.environ.get("SNU_CKPT_DIR", "ckpts_unified"))
LISTWISE_PROB = float(os.environ.get("SNU_LISTWISE_PROB", "0.30"))
ACCUM = int(os.environ.get("SNU_ACCUM", "4"))
EPOCHS = int(os.environ.get("SNU_EPOCHS", "1"))
LR = float(os.environ.get("SNU_LR", "1e-4"))
LORA_R = int(os.environ.get("SNU_LORA_R", "32"))
N_ADJACENT = int(os.environ.get("SNU_N_ADJACENT", "3"))
N_NEAR = int(os.environ.get("SNU_N_NEAR", "2"))
N_RANDOM2 = int(os.environ.get("SNU_N_RANDOM2", "1"))
MONITOR_N = int(os.environ.get("SNU_MONITOR_N", "250"))
SEED = int(os.environ.get("SNU_SEED", "42"))
WARMUP_FRAC = 0.03

sft.CKPT_DIR = CKPT_DIR  # redirect reused helpers (save_ckpt/latest_ckpt/prune) here


def branch_for(row_id, epoch):
    """Deterministic per-(row,epoch) branch choice + presentation sigma +
    candidate-selection rng, all derived from one seeded hash -> resumable
    (identical draws replay after a restart), same pattern as train.py's
    sigma_for / train_listwise.py's sigma_and_rng."""
    h = hashlib.sha1(f"{SEED}:{epoch}:{row_id}:unified".encode()).digest()
    is_listwise = (h[0] / 255.0) < LISTWISE_PROB
    sigma = SIGMAS[h[1] % len(SIGMAS)]
    rng = random.Random(h)
    return is_listwise, sigma, rng


def sft_micro_loss(model, collate, row, sigma):
    answer = json.loads(row["Answer"])
    so = slot_order(image_order_from_answer(answer), sigma)
    messages = build_messages(row, TRAIN_IMG_DIR, sigma=sigma)
    batch = collate([(messages, target_text(so))])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    return out.loss


def listwise_micro_loss(model, collate, row, sigma, rng):
    truth = image_order_from_answer(json.loads(row["Answer"]))
    cands = broadened_candidate_set(truth, n_adjacent=N_ADJACENT,
                                    n_near=N_NEAR, n_random=N_RANDOM2, rng=rng)
    messages = build_messages(row, TRAIN_IMG_DIR, sigma=sigma)
    items = [(messages, target_text(slot_order(io, sigma))) for io in cands]
    batch = collate(items)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    labels = batch.pop("labels")
    out = model(**batch)
    scores = candidate_logprob_sum(out.logits, labels)
    return F.cross_entropy(scores.unsqueeze(0),
                           torch.tensor([0], device=scores.device))


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)
    val_rows = pd.read_csv(VAL_CSV).to_dict("records")
    rows = df.to_dict("records")
    print(f"unified round2: train={len(rows)} val={len(val_rows)} "
          f"listwise_prob={LISTWISE_PROB} accum={ACCUM} lr={LR} "
          f"epochs={EPOCHS} lora_r={LORA_R}")

    from common import QUANT
    processor = load_processor()
    model = load_model()
    model.config.use_cache = False
    if QUANT:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=lora_target_modules(model))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    collate = sft.make_collate(processor)

    steps_per_epoch = (len(rows) + ACCUM - 1) // ACCUM
    total_steps = steps_per_epoch * EPOCHS
    save_steps = int(os.environ.get("SNU_SAVE_STEPS",
                                    str(max(1, total_steps // 6))))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)

    start_step = start_epoch = start_in_epoch = 0
    best_acc = -1.0
    ck = sft.latest_ckpt()
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
    model.train()
    step = start_step
    t0 = time.time()
    n_sft = n_lw = 0

    for epoch in range(start_epoch, EPOCHS):
        gen = random.Random(SEED + epoch)
        order = list(range(len(rows)))
        gen.shuffle(order)
        pos = start_in_epoch * ACCUM if epoch == start_epoch else 0

        while pos < len(order):
            opt.zero_grad(set_to_none=True)
            loss_sum, got = 0.0, 0
            for _ in range(ACCUM):
                if pos >= len(order):
                    break
                row = rows[order[pos]]
                pos += 1
                is_lw, sigma, rng = branch_for(row["Id"], epoch)
                if is_lw:
                    loss = listwise_micro_loss(model, collate, row, sigma, rng)
                    n_lw += 1
                else:
                    loss = sft_micro_loss(model, collate, row, sigma)
                    n_sft += 1
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {step} (branch="
                        f"{'listwise' if is_lw else 'sft'}) - aborting so "
                        f"the box doesn't burn money; resume with a lower "
                        f"SNU_LR")
                (loss / ACCUM).backward()
                loss_sum += loss.item()
                got += 1
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            in_epoch = pos // ACCUM

            rec = {"step": step, "epoch": epoch, "loss": loss_sum / got,
                   "lr": sched.get_last_lr()[0],
                   "n_sft": n_sft, "n_listwise": n_lw,
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 10 == 0:
                print(f"step {step}/{total_steps} epoch {epoch} "
                      f"loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                      f"(sft={n_sft} lw={n_lw})")

            if step % save_steps == 0 or pos >= len(order):
                d = sft.save_ckpt(model, opt, sched, step, epoch, in_epoch)
                torch.cuda.empty_cache()
                model.config.use_cache = True
                acc = sft.score_monitor(model, processor, val_rows, MONITOR_N)
                model.config.use_cache = False
                torch.cuda.empty_cache()
                rec = {"step": step, "val_acc": acc,
                       "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[val] step {step} score exact-match {acc:.4f} "
                      f"(best {max(best_acc, acc):.4f})")
                if acc > best_acc:
                    best_acc = acc
                    best = CKPT_DIR / "best"
                    shutil.rmtree(best, ignore_errors=True)
                    shutil.copytree(d, best)
                    (best / "acc.json").write_text(
                        json.dumps({"val_acc": acc, "step": step}))
                sft.prune_ckpts(keep=3)

    sft.save_ckpt(model, opt, sched, step, EPOCHS - 1, 0, tag=f"ckpt-{step}")
    print(f"done. steps={step} best_val={best_acc:.4f} "
          f"wall={time.time() - t0:.0f}s sft_rows={n_sft} listwise_rows={n_lw}")


if __name__ == "__main__":
    main()
