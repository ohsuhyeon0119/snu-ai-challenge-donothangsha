# Trial 2 — Frame-Ordering: Properly Fine-Tuned VLM + Constrained Scoring

**Date:** 2026-07-12
**Author:** ohsuhyeon (team donothangsha)
**Goal:** Move exact-match validation accuracy from trial1's 16.77% toward 0.7+, using only
rule-compliant modeling (no metadata/leakage shortcuts).

## 1. Problem restatement

Given a caption (`Sentence`) and 4 temporally-shuffled video frames, predict the permutation
that restores the original temporal order. Scored by 24-way **exact-match** accuracy (no partial
credit). ~15.5% of rows are `No_ordering=True` (answer `[1,2,3,4]`).

Data: 9,535 train rows, 819 test rows. Train contains uncleaned samples (near-black frames,
frames unrelated to the caption, cases under-determined by the frames alone) — an inherent noise
ceiling.

## 2. Diagnosis of trial1 (why it stalled at 16.77%)

`lora_finetune.py` masked **only padding** from the loss:

```python
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100   # padding only
```

The loss was therefore computed over the entire sequence (chat template + hundreds of vision
tokens per image + instruction + answer). The answer ("The correct order is [...]", ~10 tokens)
was <1% of the sequence, so its gradient signal was heavily diluted. Symptoms match exactly:
loss plateaued ~6.8, and test predictions collapsed onto two "safe" answers (`[1,2,3,4]` 28%,
`[3,2,4,1]` 21%) — a model that learned the **marginal** answer distribution, not the
**conditional**. This is the primary, near-fatal bug.

## 3. Key insight — the caption encodes the order

Captions narrate events in temporal order ("...*before*...*then*...*followed by*...*ending
with*..."). So the task is closer to **aligning 4 frames to the caption's narrated sequence**
than to inferring motion from pixels. This is why 0.7+ is legitimately reachable.

**Important:** we do NOT split the caption into 4 clauses. The caption often does not split
cleanly into exactly 4 events (3 events for 4 frames, overlapping actions, etc.), so rigid
segmentation is fragile. Instead we feed the **whole caption + all 4 frames** to a VLM and let it
learn the alignment end-to-end. The "caption encodes order" fact explains *why the task is
learnable to high accuracy*; it is not a preprocessing step.

## 4. Approach

Single fine-tuned vision-language model, staged 2B → 7B. No ensembling (rule-compliant: the same
model family fine-tuned once; final submission is one 7B model).

### 4.1 Model
- **Stage A (validation of technique):** `Qwen/Qwen2-VL-2B-Instruct` — fastest way to confirm the
  fixes work vs. the comparable 16.77% baseline.
- **Stage B (final):** `Qwen/Qwen2.5-VL-7B-Instruct` — stronger temporal/alignment reasoning,
  higher ceiling. Both weights public before 2026-05-31 (Qwen2-VL Aug 2024, Qwen2.5-VL Jan 2025).
- Final inference must fit RTX 3090 24GB via 4-bit; 7B 4-bit + LoRA fits comfortably.

### 4.2 Training recipe (deltas vs trial1)
1. **Masked loss (highest-value fix):** labels = `-100` for all prompt + vision tokens; loss on
   completion tokens only. Implemented by tokenizing the prompt (with
   `add_generation_prompt=True`) separately to get its length, then masking `labels[:prompt_len]`.
2. **LoRA expansion:** target all linear projections (`q,k,v,o,gate,up,down`), rank 32, alpha 64,
   dropout 0.05 (trial1 used only `q,v`, rank 16).
3. **LR schedule:** cosine decay with ~3–5% warmup (trial1 was flat 1e-4). LR 1e-4–2e-4.
4. **Memory/throughput:** cap frame resolution via the processor's `max_pixels` (e.g.
   `256*28*28` per image). Coarse temporal ordering does not need high resolution; this sharply
   cuts vision-token count, enabling real batching and faster steps. Use gradient accumulation
   for effective batch size if per-step batch must stay small.
5. **Best-checkpoint selection:** evaluate val exact-match periodically (every N steps) and keep
   the best checkpoint (trial1 evaluated once, at the end — no evidence 2 epochs was right).
6. **Epochs:** 2–3, governed by the val curve rather than a fixed count.

### 4.3 Inference — constrained permutation scoring
Instead of free-form generation + regex parsing (trial1), score each of the 24 candidate
permutations by the model's likelihood of the corresponding completion string and take argmax.
- Eliminates parse failures and invalid outputs (always a valid permutation).
- Typically edges out free generation.
- Allowed as an "inference strategy"; 819 rows × 24 candidates is well within the 24h budget.
- Same scoring function is used for the periodic validation metric during training, so val and
  test are measured identically.

### 4.4 Data cleaning
- Detect near-black / very-low-variance frames (mean/variance thresholds) and near-duplicate
  frames (perceptual hash). Report counts first.
- Clearly-broken rows: consider excluding from **training only** (never from test). Keep
  `No_ordering=True` rows (they teach the already-ordered case). Do not rebalance away the natural
  identity-answer rate — masked loss should let the model learn the conditional rather than
  over-predicting identity.

### 4.5 Validation protocol
Reuse trial1's split: seed-42, 12% stratified on `Answer`, n=1,145, never trained on. Metric:
constrained-scoring exact-match. This makes trial2 numbers directly comparable to 16.77%.

## 5. Convergence gates
- **G1:** Stage-A (2B) val exact-match. Expect a large jump over 16.77% (target ~0.4–0.6). If the
  masked-loss fix alone does not clearly beat baseline, stop and debug the pipeline before scaling.
- **G2:** Stage-B (7B) val exact-match (target ~0.6–0.8).
- **G3:** If G2 < 0.7, escalate: stronger data cleaning, pairwise-ordering reformulation (6 binary
  "does frame X precede frame Y" decisions aggregated into a total order — still one model), or
  longer training / higher LoRA rank.

Honest caveat: no specific number is guaranteed. The recipe targets trial1's exact failure mode,
so expected value is high, and the staged gates drive toward 0.7.

## 6. Execution
- Training runs on a rented Vast.ai RTX 3090 (local dev machine is Apple M5, no CUDA). Claude
  controls the box over SSH (host/port/key provided by the user), uploads code + data, and
  launches training in a `tmux`/`nohup` background session.
- Progress is checkpointed with a `progress.json` (step, latest val accuracy, best checkpoint) and
  a log file, so the user can ask for status at any time and Claude reports the latest.

## 7. Rule compliance
Single model (no ensembling), provided data only (no external data), no generative augmentation,
open-source weights public before 2026-05-31, final model runs on one RTX 3090 24GB within the 24h
inference budget and 80GB size cap, no metadata/leakage signals.

## 8. Deliverables (code, this session)
- `trial2/clean_data.py` — bad-frame / duplicate detection + report; writes cleaned train split.
- `trial2/train.py` — QLoRA fine-tune with masked loss, LR schedule, periodic val, best-ckpt.
- `trial2/eval.py` — constrained-scoring exact-match on the val split.
- `trial2/infer.py` — constrained-scoring predictions over `test.csv` → submission CSV.
- `trial2/README.md` — env, run order, Vast.ai setup, reproducibility notes.
