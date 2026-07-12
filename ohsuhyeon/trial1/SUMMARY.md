# Trial 1 — LoRA fine-tune of Qwen2-VL-2B (Track B)

**Date:** 2026-07-12
**Goal:** beat the organizer's zero-shot Qwen2-VL-2B-Instruct baseline and Track A's CLIP
classifier on the frame-reordering task, by actually fine-tuning the VLM instead of just
prompting it.

## Result

| Method | val accuracy |
|---|---|
| Random guess (1/24) | ~4.2% |
| Majority-class baseline (always `[1,2,3,4]`) | 15.46% |
| Track A — CLIP embeddings + Transformer classifier (best of 3 seeds) | 15.72% |
| **Track B — this trial: Qwen2-VL-2B + LoRA, fine-tuned** | **16.77%** |

Evaluated on the same held-out 1,145-row validation split used across both tracks
(stratified 12% of `train.csv`, seed 42 — never used in training).

**Honest read:** a real, positive result — fine-tuning did move the needle above both prior
baselines — but the margin (~+1.3 percentage points) is only about 1.2 standard errors at this
sample size (SE ≈ 0.011 for a proportion at n=1,145), so it's better described as "fine-tuning
helped a little" than "the task is solved." See **Known limitations** below for the most likely
reason the gain wasn't bigger.

## What was built

- `lora_finetune.py` — LoRA fine-tuning script (Qwen2-VL-2B-Instruct, 4-bit base, LoRA r=16 on
  `q_proj`/`v_proj`). Paths (`SNU_DATA_DIR`, `SNU_ADAPTER_DIR`, `SNU_VAL_SPLIT_CSV`,
  `SNU_BATCH_SIZE`) are env-var overridable so the same script runs on Colab, Kaggle, or a
  rented GPU box unchanged. Saves an adapter checkpoint + `progress.json` every 500 steps and
  resumes automatically from the last checkpoint if restarted (built for Colab's aggressive
  session-disconnect behavior; not strictly needed on a paid rented box, but harmless there too).
- `lora_eval.py` — evaluates the fine-tuned adapter on the held-out validation split using
  `model.generate()` + regex parsing of the "The correct order is [...]" output, with the exact
  same exact-match metric as Track A's `validation.py`, so the two tracks are directly
  comparable.
- `lora_infer_test.py` — same generation/parsing logic, run over `test.csv` (which has no
  ground-truth `Answer` column) to produce the actual Kaggle submission CSV.

## Environment

Rented a Vast.ai on-demand RTX 3090 (24GB VRAM) instance (~$0.15/hr) — the local dev machine
(Apple M5, no CUDA GPU) can't run this at all. PyTorch + CUDA base image, packages
(`peft`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`, `transformers`, `pandas`,
`scikit-learn`) installed via pip into the image's default venv.

## What was tried and reverted: batching

Multi-example batching (batch_size 4, then 2) was implemented to speed up training past
Colab T4's ~6s/step, but both **OOM'd a 24GB RTX 3090** even with gradient checkpointing
enabled. Root cause: Qwen2's ~152k-token vocabulary makes the cross-entropy logits tensor
scale directly with `batch_size × sequence_length`, and this task's sequences are already long
(4 images/example → many vision tokens). Settled on `batch_size=1`, which uses ~14.2GB peak
and ~0.6-0.8s/step — still a ~10x speedup over Colab's T4 purely from the better GPU, just not
from batching. The `SNU_BATCH_SIZE` knob is kept in the script for a bigger-VRAM GPU later.

## Training run

- 8,390 train rows × 2 epochs = 16,780 steps, batch_size=1, LR=1e-4 (flat, no schedule).
- **3.88h wall-clock** on the rented RTX 3090.
- Loss: started ~17, dropped quickly to ~7 within the first ~200 steps, then **plateaued
  around 6.8-7.0 for essentially the rest of training** (see the loss chart artifact from the
  session this trial was run in). This plateau is a symptom of the limitation below.

## Known limitations (ranked by likely impact on the modest result)

1. **Loss is computed over the full sequence (prompt + images + answer), not masked to just
   the answer tokens.** Standard SFT practice masks the prompt with `label=-100` so the model
   is only "graded" on generating the correct completion. This wasn't done here. Impact is
   probably moderate rather than fatal — the prompt template is largely fixed/repetitive so
   the model likely learns to predict it almost perfectly within the first few dozen steps,
   after which most remaining gradient should come from the varying (`Sentence` + answer)
   tokens — but it dilutes the signal and is the clearest thing to fix first in a rerun.
2. **No LR schedule/warmup**, flat 1e-4 throughout.
3. **No mid-training validation / early stopping** — evaluation only happened once, at the
   very end, so there's no evidence 2 epochs was the right amount (could be under- or
   over-fit).
4. **Single run, no hyperparameter search** — LoRA rank (16), batch composition, and epoch
   count were picked once and not tuned against validation accuracy.

## A pattern worth investigating: answer collapse on the test set

Running inference on the actual `test.csv` (819 rows, submission generated to
`outputs/submission_lora.csv`, not committed — see `.gitignore`) showed a skewed answer
distribution:

- `[1, 2, 3, 4]`: 28.0% of predictions (train's true no-shuffle rate is 15.5% — notably higher)
- `[3, 2, 4, 1]`: 21.3% of predictions, concentrated on a single permutation (train's 23
  non-identity permutations are each ~3.7% of the data — this is ~6x over-represented)

Together these two answers cover ~49% of all test predictions. This looks like the model
falling back to a small set of "safe" answers when uncertain, rather than reasoning per-example
— consistent with the loss plateau and the missing-prompt-mask limitation above. Worth
re-checking after a masked-loss rerun to see if it persists.

## Suggested next steps (trial 2 candidates)

1. Mask the loss to answer-tokens-only (the single highest-expected-value fix).
2. Add LR warmup + cosine/linear decay.
3. Evaluate on the validation split every N steps during training; keep the best checkpoint
   instead of always the last one.
4. Re-check the answer-collapse pattern after (1) — if it persists, investigate further
   (data noise per the competition's own note that train data isn't fully clean, or a decoding
   issue).
5. If time allows: LoRA rank sweep (32, 64), and/or the permutation-expansion augmentation
   Track A used (Task 7 in the main plan), adapted for this VLM's SFT format.

## Files in this folder

- `lora_finetune.py` — training
- `lora_eval.py` — validation-split evaluation
- `lora_infer_test.py` — test-set inference / submission generation
- `SUMMARY.md` — this file
