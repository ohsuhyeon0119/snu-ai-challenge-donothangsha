"""
Trial 4 stage-3 — VISION-TOWER LoRA on top of the round-2 winner (ckpt-2218,
LB 0.90924). Design doc: docs/superpowers/specs/2026-07-21-*.md (round 3 note).

Round 2 trained LoRA on the LLM decoder only; the vision encoder stayed
FROZEN. But ordering 4 near-identical frames of one scene is a fine-grained
visual-difference task, and ~20.8% of captions have NO temporal-cue words
(pure-visual rows) — no amount of LLM-side tuning can reach those, because
the bottleneck is what the frozen encoder attends to. This stage adds LoRA
to the vision transformer blocks so the encoder can learn to pick up the
subtle inter-frame differences that determine order.

Starts from EXACTLY ckpt-2218's behavior:
  - LoRA is placed on BOTH the LLM decoder (same targets as round 2) AND the
    vision blocks (new).
  - ckpt-2218's adapter weights (LLM-only) are loaded with strict=False, so
    the LLM LoRA continues from the winner and the fresh vision LoRA starts
    at its zero-init (lora_B = 0 -> no-op). Step 0 therefore reproduces
    ckpt-2218; the vision pathway only changes as training proceeds.
  - Both are trained together at a LOW LR (fine-tune regime, default 3e-5),
    so the system co-adapts without drifting far from the 0.909 behavior.

Same unified SFT + listwise branching / objective as train_unified.py
(imported, unchanged). Fresh optimizer + short cosine schedule (this is a
new fine-tune phase, NOT a resume of round 2's schedule). Frequent val_small
monitoring: if val drops below round 2's level and stays there, the vision
bet isn't paying off -> stop and keep ckpt-2218.

Env knobs:
  SNU_INIT_ADAPTER   ckpt-2218 dir to initialize LLM LoRA from
                     (default ckpts_unified/ckpt-2218)
  SNU_TRAIN_CSV      default data_work/train_full.csv (8,871 rows)
  SNU_VAL_CSV        default data_work/val_small.csv (250 rows)
  SNU_CKPT_DIR       default ckpts_vision
  SNU_LISTWISE_PROB  default 0.30      SNU_ACCUM   default 4
  SNU_EPOCHS         default 1 (expect to early-stop well before a full pass)
  SNU_LR             default 3e-5 (low: fine-tune on top of a strong ckpt)
  SNU_LORA_R         default 32
  SNU_SAVE_STEPS     checkpoint+monitor cadence (default total_steps//12,
                     i.e. ~12 checks — denser than round 2 to catch early)
  SNU_MONITOR_N      default 250
  SNU_SEED           default 42
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

import train as sft
import train_unified as tu  # reuse branch_for / sft_micro_loss / listwise_micro_loss
from common import (enable_vision_input_grads, load_model, load_processor,
                    lora_target_modules, vision_lora_target_modules)

WORK_DIR = os.environ.get("SNU_WORK_DIR", "data_work")
INIT_ADAPTER = os.environ.get("SNU_INIT_ADAPTER", "ckpts_unified/ckpt-2218")
TRAIN_CSV = Path(os.environ.get("SNU_TRAIN_CSV", f"{WORK_DIR}/train_full.csv"))
VAL_CSV = Path(os.environ.get("SNU_VAL_CSV", f"{WORK_DIR}/val_small.csv"))
CKPT_DIR = Path(os.environ.get("SNU_CKPT_DIR", "ckpts_vision"))
LISTWISE_PROB = float(os.environ.get("SNU_LISTWISE_PROB", "0.30"))
ACCUM = int(os.environ.get("SNU_ACCUM", "4"))
EPOCHS = int(os.environ.get("SNU_EPOCHS", "1"))
LR = float(os.environ.get("SNU_LR", "3e-5"))
LORA_R = int(os.environ.get("SNU_LORA_R", "32"))
MONITOR_N = int(os.environ.get("SNU_MONITOR_N", "250"))
SEED = int(os.environ.get("SNU_SEED", "42"))
WARMUP_FRAC = 0.03

# keep train_unified's branch machinery pointed at OUR knobs (they read tu.*)
tu.LISTWISE_PROB = LISTWISE_PROB
tu.SEED = SEED
sft.CKPT_DIR = CKPT_DIR


def build_lora_config(model):
    from peft import LoraConfig
    targets = lora_target_modules(model) + vision_lora_target_modules(model)
    n_vis = len(vision_lora_target_modules(model))
    print(f"LoRA targets: {len(targets)} total "
          f"({len(targets) - n_vis} LLM + {n_vis} vision)")
    return LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=targets)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)
    val_rows = pd.read_csv(VAL_CSV).to_dict("records")
    rows = df.to_dict("records")
    print(f"vision stage-3: train={len(rows)} val={len(val_rows)} "
          f"listwise_prob={LISTWISE_PROB} accum={ACCUM} lr={LR} "
          f"epochs={EPOCHS} lora_r={LORA_R} init={INIT_ADAPTER}")

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
    # critical for vision LoRA: let checkpointing track through vision blocks
    nhook = enable_vision_input_grads(model)
    print(f"vision input-grad hook on {nhook} patch_embed module(s)")

    from peft import get_peft_model
    model = get_peft_model(model, build_lora_config(model))
    model.print_trainable_parameters()

    collate = sft.make_collate(processor)

    steps_per_epoch = (len(rows) + ACCUM - 1) // ACCUM
    total_steps = steps_per_epoch * EPOCHS
    save_steps = int(os.environ.get("SNU_SAVE_STEPS",
                                    str(max(1, total_steps // 12))))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    start_step = start_epoch = start_in_epoch = 0
    best_acc = -1.0

    own_ck = sft.latest_ckpt()  # this stage's own checkpoint (interrupted run)
    if own_ck is not None:
        sd = load_file(own_ck / "adapter_model.safetensors")
        set_peft_model_state_dict(model, sd)  # LLM + vision, full state
        st = torch.load(own_ck / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_step, start_epoch = st["step"], st["epoch"]
        start_in_epoch = st["in_epoch_step"]
        bf = CKPT_DIR / "best" / "acc.json"
        if bf.exists():
            best_acc = json.loads(bf.read_text())["val_acc"]
        print(f"resumed vision stage from {own_ck} (step {start_step})")
    else:
        # cold start: init LLM LoRA from ckpt-2218; vision LoRA stays at its
        # zero-init (lora_B = 0 -> no-op), so step 0 reproduces ckpt-2218.
        # Version-robust: filter ckpt-2218's (LLM-only) keys to the ones this
        # model actually has, then load. No dependency on a `strict=` kwarg or
        # on set_peft_model_state_dict's return shape (both vary across peft
        # versions). Vision keys are absent from init_sd -> untouched.
        from peft import get_peft_model_state_dict
        init_sd = load_file(Path(INIT_ADAPTER) / "adapter_model.safetensors")
        valid = set(get_peft_model_state_dict(model).keys())
        filtered = {k: v for k, v in init_sd.items() if k in valid}
        assert filtered, (
            "no ckpt-2218 keys matched the model's LoRA keys — key format "
            "mismatch (check base model / LLM target_modules unchanged)")
        set_peft_model_state_dict(model, filtered)
        print(f"cold start: loaded {len(filtered)}/{len(init_sd)} LLM-LoRA "
              f"tensors from {INIT_ADAPTER}; vision LoRA at zero-init "
              f"(step0 == ckpt-2218)")

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
                is_lw, sigma, rng = tu.branch_for(row["Id"], epoch)
                if is_lw:
                    loss = tu.listwise_micro_loss(model, collate, row, sigma, rng)
                    n_lw += 1
                else:
                    loss = tu.sft_micro_loss(model, collate, row, sigma)
                    n_sft += 1
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {step} (branch="
                        f"{'listwise' if is_lw else 'sft'}) - aborting; "
                        f"resume with a lower SNU_LR")
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
