"""
Trial 4 stage-2 — listwise ranking fine-tune on top of ckpt-400 (trial3's
LB-0.89 SFT adapter). Reuses train.py's monitor/checkpoint machinery.

Why: masked SFT (train.py) only pushes UP the truth completion's probability;
it never explicitly pushes DOWN the near-miss competitors that argmax has to
beat at decision time. Real-rule val error analysis on ckpt-300 showed 36%
of errors were adjacent-swaps (kendall distance 1) — exactly the competitors
this stage targets. This directly optimizes the deployed decision rule:
"does the truth outscore these hard alternatives" instead of "is the truth
likely" in isolation.

Candidates per row: truth + its 3 adjacent-swap perms (hard_candidate_set's
kendall-distance-1 set) + N_RANDOM others = default 6, truth always index 0.
Loss: cross_entropy(scores, index=0) over the candidates' logprob-sum
"logits" — an InfoNCE-style listwise loss.

MECHANICS NOTE (v2 — v1 OOM'd): the first design shared one prompt forward
across candidates via a KV cache and kept it differentiable, avoiding
gradient checkpointing (checkpointing forces use_cache=False, incompatible
with cache reuse). That saves memory under no_grad (inference) but NOT under
autograd — backward needs every candidate's cache-extension tensors kept
alive simultaneously, so 6 candidates OOM'd a 32B model at ~93GB even
without any batching. Fixed by dropping cache-sharing for TRAINING entirely:
each row's candidates are collated into an ordinary micro-batch (reusing
train.py's make_collate, unchanged) and forwarded with gradient checkpointing
ON, exactly like plain masked SFT — proven safe at this scale (trial3: ~48-
65GB peak for mb4-8). Per-candidate summed logprob is read off the batched
logits via common.candidate_logprob_sum() instead of relying on HF's own
(whole-batch-averaged) loss. Costs 6x the vision-tower compute per row
(no longer shared) — accepted for correctness over speed at this candidate
count and hard-set size.

Rows come from mine_hard.py's hard_train.csv (train-only margin mining,
Rule-3.4-compliant). Fresh optimizer + low LR (stage-2 fine-tune, not a
continuation of SFT's schedule) — NOT resumed from ckpt-400's optimizer
state, only its LoRA weights.

ROUND 2 (2026-07-21, LB 0.9075 after round 1): SNU_CANDIDATE_MODE=broadened
(new default) swaps in common.broadened_candidate_set — round 1 only ever
showed the model kendall=1 (adjacent-swap) competitors, but those were only
36% of ckpt-300's error mix; kendall={2,3} were another ~30% and went
completely untouched. Round 2 also re-mines hard negatives from round 1's
OWN best checkpoint (SNU_BASE_ADAPTER=ckpts_listwise/best), not ckpt-400 —
many of ckpt-400's confusions are already fixed, so re-mining targets what's
ACTUALLY still wrong now instead of stale round-1 blind spots.

Env knobs:
  SNU_BASE_ADAPTER   default ckpts/ckpt-400 (weights to fine-tune from)
  SNU_HARD_CSV       default data_work/hard_train.csv (from mine_hard.py)
  SNU_VAL_CSV        default data_work/clean_val.csv
  SNU_CKPT_DIR       default ckpts_listwise
  SNU_ACCUM          rows accumulated per optimizer step (default 4)
  SNU_LISTWISE_EPOCHS default 1 (passes over the hard subset)
  SNU_LR             default 2e-5 (low: fine-tune, not fresh SFT)
  SNU_CANDIDATE_MODE adjacent (round-1 hard_candidate_set) |
                     broadened (round-2 default: + kendall 2-3 negatives)
  SNU_N_RANDOM       random negatives/row, "adjacent" mode (default 2)
  SNU_N_ADJACENT/SNU_N_NEAR/SNU_N_RANDOM2  "broadened" mode counts
                     (default 3/2/1 -> 7 candidates/row)
  SNU_SAVE_STEPS     default 40   SNU_MONITOR_N   default 150
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import train as sft  # reuse make_collate / save_ckpt / score_monitor / prune_ckpts
from common import (SIGMAS, TRAIN_IMG_DIR, broadened_candidate_set,
                    build_messages, candidate_logprob_sum, hard_candidate_set,
                    image_order_from_answer, load_model, load_processor,
                    slot_order, target_text)

WORK_DIR = os.environ.get("SNU_WORK_DIR", "data_work")
BASE_ADAPTER = os.environ.get("SNU_BASE_ADAPTER", "ckpts/ckpt-400")
HARD_CSV = Path(os.environ.get("SNU_HARD_CSV", f"{WORK_DIR}/hard_train.csv"))
VAL_CSV = Path(os.environ.get("SNU_VAL_CSV", f"{WORK_DIR}/clean_val.csv"))
CKPT_DIR = Path(os.environ.get("SNU_CKPT_DIR", "ckpts_listwise"))
ACCUM = int(os.environ.get("SNU_ACCUM", "4"))
LISTWISE_EPOCHS = int(os.environ.get("SNU_LISTWISE_EPOCHS", "1"))
LR = float(os.environ.get("SNU_LR", "2e-5"))
CANDIDATE_MODE = os.environ.get("SNU_CANDIDATE_MODE", "broadened")
N_RANDOM = int(os.environ.get("SNU_N_RANDOM", "2"))
N_ADJACENT = int(os.environ.get("SNU_N_ADJACENT", "3"))
N_NEAR = int(os.environ.get("SNU_N_NEAR", "2"))
N_RANDOM2 = int(os.environ.get("SNU_N_RANDOM2", "1"))
SAVE_STEPS = int(os.environ.get("SNU_SAVE_STEPS", "40"))
MONITOR_N = int(os.environ.get("SNU_MONITOR_N", "150"))
MONITOR_MODE = os.environ.get("SNU_MONITOR_MODE", "score")
SEED = int(os.environ.get("SNU_SEED", "42"))
WARMUP_FRAC = 0.05

sft.CKPT_DIR = CKPT_DIR  # redirect reused helpers (save_ckpt/latest_ckpt/prune) here


def sigma_and_rng(sample_id, epoch):
    """Deterministic per-(sample, epoch) presentation sigma + a seeded RNG
    for random-negative selection — resumable, same pattern as train.py."""
    import hashlib
    h = hashlib.sha1(f"{SEED}:{epoch}:{sample_id}:listwise".encode()).digest()
    sigma = SIGMAS[h[0] % len(SIGMAS)]
    return sigma, random.Random(h)


def listwise_loss(model, processor, collate, row, epoch):
    """One row -> (loss tensor, correct: bool). Builds a micro-batch of the
    row's candidates (truth first) via the ordinary SFT collate, forwards it
    ONCE (checkpointed, batched — no manual KV-cache reuse), and reduces to
    a listwise cross-entropy over per-candidate summed logprobs."""
    truth = image_order_from_answer(json.loads(row["Answer"]))
    sigma, rng = sigma_and_rng(row["Id"], epoch)
    if CANDIDATE_MODE == "broadened":
        cands = broadened_candidate_set(truth, n_adjacent=N_ADJACENT,
                                        n_near=N_NEAR, n_random=N_RANDOM2,
                                        rng=rng)
    else:
        cands = hard_candidate_set(truth, n_random=N_RANDOM, rng=rng)
    messages = build_messages(row, TRAIN_IMG_DIR, sigma=sigma)
    items = [(messages, target_text(slot_order(io, sigma))) for io in cands]
    batch = collate(items)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    labels = batch.pop("labels")
    out = model(**batch)
    scores = candidate_logprob_sum(out.logits, labels)  # [n_cands]
    loss = F.cross_entropy(scores.unsqueeze(0),
                           torch.tensor([0], device=scores.device))
    correct = int(scores.argmax().item() == 0)
    return loss, correct


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(HARD_CSV)
    val_rows = pd.read_csv(VAL_CSV).to_dict("records")
    rows = df.to_dict("records")
    print(f"listwise stage-2: {len(rows)} hard-mined rows, accum={ACCUM}, "
          f"lr={LR}, mode={CANDIDATE_MODE}, "
          f"cands={'adj3+near2+rand1=7' if CANDIDATE_MODE == 'broadened' else f'adj3+rand{N_RANDOM}'}, "
          f"epochs={LISTWISE_EPOCHS}, base={BASE_ADAPTER}")

    from common import QUANT
    processor = load_processor()
    model = load_model()
    model.config.use_cache = False  # required for gradient checkpointing

    if QUANT:
        # QLoRA: casts norms/lm_head, enables grad ckpt + input grads —
        # same call as train.py's SFT stage, proven safe at this scale.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)
    print(f"base adapter loaded (trainable): {BASE_ADAPTER}")
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable > 0, "no trainable params — adapter not in train mode"

    collate = sft.make_collate(processor)

    steps_per_epoch = (len(rows) + ACCUM - 1) // ACCUM
    total_steps = steps_per_epoch * LISTWISE_EPOCHS
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, max(1, int(total_steps * WARMUP_FRAC)), total_steps)

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
        print(f"resumed listwise from {ck} (step {start_step})")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(CKPT_DIR / "train_log.jsonl", "a")
    model.train()
    step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, LISTWISE_EPOCHS):
        gen = random.Random(SEED + epoch)
        order = list(range(len(rows)))
        gen.shuffle(order)
        pos = start_in_epoch * ACCUM if epoch == start_epoch else 0

        while pos < len(order):
            opt.zero_grad(set_to_none=True)
            loss_sum, acc_sum, got = 0.0, 0, 0
            for _ in range(ACCUM):
                if pos >= len(order):
                    break
                row = rows[order[pos]]
                pos += 1
                loss, correct = listwise_loss(model, processor, collate,
                                              row, epoch)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {step} - aborting so the "
                        f"box doesn't burn money; resume with a lower SNU_LR")
                (loss / ACCUM).backward()
                loss_sum += loss.item()
                acc_sum += correct
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
                   "batch_top1_acc": acc_sum / got,
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 5 == 0:
                print(f"step {step}/{total_steps} epoch {epoch} "
                      f"loss {rec['loss']:.4f} top1 {rec['batch_top1_acc']:.2f} "
                      f"lr {rec['lr']:.2e}")

            if step % SAVE_STEPS == 0 or pos >= len(order):
                d = sft.save_ckpt(model, opt, sched, step, epoch, in_epoch)
                torch.cuda.empty_cache()
                model.config.use_cache = True
                monitor = (sft.score_monitor if MONITOR_MODE == "score"
                           else sft.greedy_monitor)
                acc = monitor(model, processor, val_rows, MONITOR_N)
                model.config.use_cache = False
                torch.cuda.empty_cache()
                rec = {"step": step, "val_acc": acc, "mode": MONITOR_MODE,
                       "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[val] step {step} {MONITOR_MODE} exact-match {acc:.4f} "
                      f"(best {max(best_acc, acc):.4f})")
                if acc > best_acc:
                    best_acc = acc
                    best = CKPT_DIR / "best"
                    shutil.rmtree(best, ignore_errors=True)
                    shutil.copytree(d, best)
                    (best / "acc.json").write_text(
                        json.dumps({"val_acc": acc, "step": step}))
                sft.prune_ckpts(keep=3)

    sft.save_ckpt(model, opt, sched, step, LISTWISE_EPOCHS - 1, 0,
                  tag=f"ckpt-{step}")
    print(f"done. steps={step} best_val={best_acc:.4f} "
          f"wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
