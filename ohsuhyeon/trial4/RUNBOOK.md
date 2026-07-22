# Trial 4 Runbook — H100 NVL, ~9h wall / ~$18 @ $2.07/h

Goal: LB 0.89 (trial3 ckpt-400) → 0.92+. Levers: real-rule checkpoint
selection, epoch 2 resume, offline-tuned TTA-k and identity-prior alpha.
Budget cap **$20** — decision gates below cut scope before overrunning.

## 0. Local (done before renting — $0)

- [x] `python test_perms.py` — 10/10 (permutation + prior/aggregation math)
- [x] `python test_offline_pipeline.py` — dump→analysis dry-run, infer parity
- [x] ckpt-400 backup verified (adapter 1.07GB + trainer_state.pt 2.1GB);
      **ckpt-500 weights are LOST** (config only) — 400 is the only resume point
- [x] versions pinned in setup_box.sh (transformers 5.14.1 etc.)

## 1. Provision (~20min, mostly download)

Instance: **1x H100 NVL 94GB** (e.g. offer $2.067/h, 3.38TB/s — avoid H100
PCIE: 1.76TB/s is *below* A100 for our BW-bound scorer). Disk 120GB.

```bash
# on box, inside tmux
bash setup_box.sh                     # deps + 65GB model download
# upload from laptop (rsync/scp): trial4/*.py, data_work/, data/ (images),
#   outputs/trial3_backup/ckpt-400/  ->  ckpts/ckpt-400/   (3.2GB upload)
python smoke_test.py                  # GPU gate: scorer==reference, masked bwd
```

## 2. Re-rank ckpt-400 + score dump (~1.2h, ~$2.5)

```bash
SNU_ADAPTER_DIR=ckpts/ckpt-400 SNU_SPLIT=val SNU_TTA_K=3 SNU_LIMIT=500 \
SNU_DUMP_SCORES=outputs/val400_scores.jsonl \
SNU_OUT=outputs/val400_pred.csv python infer.py
```

First 20 rows print s/row → **this is the H100 speed measurement**.
GATE-A: if s/row > 12 (i.e. no speedup vs A100), plan reverts to
A100 pricing math — drop step 5, keep the rest.

Pull the dump to the laptop, then offline (free):

```bash
python analyze_scores.py outputs/val400_scores.jsonl   # k x alpha grid
```

Decides: TTA k (1/2/3), alpha. Record val acc of chosen config = baseline.

## 3. Epoch 2 resume (~3.5-4h, ~$8)

```bash
SNU_MB=4 SNU_ACCUM=4 SNU_EPOCHS=2 SNU_SAVE_STEPS=150 SNU_MONITOR_N=150 \
SNU_MONITOR_MODE=score python train.py    # resumes from ckpts/ckpt-400
```

- Schedule continuity: original cosine was built for 1016 steps (lr@506 was
  half-peak) — same envs resume it exactly. Steps 401→1016 ≈ 616 steps.
- Monitor = real rule (tta_k=1), 150 rows ≈ ±3.7%p, every 150 steps
  (~10-15min each on NVL). `best/` now tracks the deployed metric.
- GATE-B: if the LAST monitor point of epoch 2 is below the step-400
  baseline by >4%p, epoch 2 overfit → keep ckpt-400, skip to step 5.

## 4. Re-dump best ckpt on val (~1h, ~$2)

Only if best/ != ckpt-400: rerun step 2 with SNU_ADAPTER_DIR=ckpts/best on
the SAME 500 rows + analyze. Pick final (ckpt, k, alpha) by val acc; on a
tie prefer the simpler config (smaller k, alpha=0).

## 5. Final test inference + submit (~1.5-2.5h, ~$4)

```bash
SNU_ADAPTER_DIR=<winner> SNU_SPLIT=test SNU_TTA_K=<k> SNU_PRIOR_ALPHA=<a> \
SNU_DUMP_SCORES=outputs/test_scores.jsonl \
SNU_OUT=outputs/trial4_submission.csv python infer.py
```

Dumping test scores too: if a later alpha/k retune is wanted, tomorrow's
submission needs NO new GPU run. VRAM peak printed live must stay <24GB
(3090 target; trial3 measured 20.5GB — unchanged model path).

## 6. Teardown checklist (before destroying the box!)

- [ ] download: best adapter + trainer_state, val/test score dumps,
      train_log.jsonl, versions.txt, submission CSV
- [ ] verify sizes locally (adapter ~1.07GB, trainer_state ~2.1GB) —
      **this is what killed ckpt-500 last time**
- [ ] destroy instance; submit on Kaggle; log LB result in SUMMARY.md

## Budget ledger (fill in as you go)

| step | est | actual |
|---|---|---|
| provision + smoke | $0.8 | |
| re-rank dump | $2.5 | |
| epoch 2 | $8 | |
| re-dump best | $2 | |
| test inference | $4 | |
| **total** | **~$17.3** | cap $20 |
