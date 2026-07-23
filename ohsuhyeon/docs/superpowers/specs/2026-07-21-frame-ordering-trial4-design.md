# Trial 4 — Design Spec (2026-07-21)

## 0. Where this sits

- **trial3** (spec: `2026-07-17-frame-ordering-trial3-design.md`): Qwen3-VL-32B-Instruct,
  nf4 QLoRA, masked SFT + presentation-shuffle augmentation. 1 epoch (508/1016
  steps, stopped early). **Result: Kaggle LB 0.89.**
- **trial4 round 1** (this repo, `ohsuhyeon/trial4/`, already run and submitted):
  listwise stage-2 ranking fine-tune on top of trial3's ckpt-400. **Result: LB
  0.9075** (+1.75pp over trial3). Details in section 1.
- **trial4 round 2 — this spec**: a from-scratch, single unified training run
  that interleaves the SFT and listwise objectives instead of stacking them
  sequentially. This is the **last training attempt** before the competition
  deadline (2026-07-24) — no round 3.

## 1. What round 1 already established (context, not re-derived here)

Round 1's pipeline: A (re-rank ckpt-400 + tune identity-prior α and TTA-k on
a 500-row real-rule val dump) → listwise stage-2 (mine ckpt-400's low-margin
train rows, train a 6-candidate — truth + 3 adjacent-swap + 2 random —
listwise ranking loss on top of ckpt-400) → final test inference.

**Measured numbers that still apply to round 2** (same base model / quant /
scoring mechanism, only the training recipe changes):
- Inference: 24-way constrained KV-cache-shared scoring + TTA over `k`
  presentation sigmas + `alpha * log P_train(identity)` rescoring.
  `k=3, alpha=0.5` was val-tuned and is the current best config (val
  exact-match 0.618 on 500 rows, vs 0.602 untuned).
- H100 NVL (94GB, 3.34TB/s) inference speed: **13.6–13.9s/row** (TTA k=3,
  nf4). VRAM peak **20.5GB** — well inside the 3090 24GB target.
- Listwise-style candidate-batch training step (ACCUM=4 rows, 6-7 candidates
  each, gradient-checkpointed): **34.5s/step** measured on this exact box
  (H100 NVL, instance 45363494, host 170701).
- Plain masked-SFT step speed on **this** box has never been directly
  measured — only extrapolated from trial3's A100 number (38s/step for 16
  samples = 2.375s/sample) with an assumed ~1.4x NVL speedup. Section 5's
  cost table is built on that extrapolation and should be corrected from the
  real first-few-minutes reading once round 2 starts (same discipline used
  for every GPU phase so far this project).

**Error structure that motivates round 2's design** (ckpt-300 real-rule val,
200 rows, from round 1's own analysis — kept here as it's still the best
error breakdown we have): of the wrong rows, **36% were kendall-distance-1**
(one adjacent pair swapped) and **~30% were kendall-distance 2–3** (moderately
scrambled). Round 1's candidate set (adjacent-swap only) targeted just the
first bucket. Separately, **20.8% of train captions have zero temporal-cue
words** (no then/before/while/finally/etc.) — these rows can only be solved
by visual reasoning, not caption parsing, and are a plausible source of the
irreducible error floor.

**What was tried and rejected between round 1 and round 2** (kept for the
record so it isn't re-litigated):
- *Pairwise-marginal re-ranking* (derive "does frame A precede B" from the
  already-computed 24-way distribution, re-rank the 24 candidates by
  consistency with those marginals instead of plain argmax). Tested **free**,
  offline, on the existing `val400_scores.jsonl` dump (500 rows, ckpt-400 +
  k=3/α=0.5): pure re-ranking **0.610 vs 0.618 baseline** (worse); best blend
  (β=10) 0.620, inside noise. **Rejected** — the model's own argmax is
  already well-calibrated; re-deriving "wisdom of the 24 candidates" doesn't
  add signal, it adds noise. Script: `trial4/pairwise_rerank.py`.
- *Multi-turn "describe then decide" reasoning*: never empirically tested
  (would need real GPU generation, not free like the above). Judged lower
  expected value than round 2 because the model has never been trained on
  this interaction pattern (out-of-distribution zero-shot), and there's a
  Rule-3.4-adjacent grey area if we ever wanted to fine-tune on
  self-generated rationales (using the model itself to generate training
  targets brushes up against "생성형 모델을 이용한 데이터 생성/변형은
  허용하지 않음" — untested whether that's compliant). Parked, not pursued.
- *Redo trial3's SFT alone, unchanged, from scratch*: rejected as pure
  redundant cost — it would just reproduce something close to ckpt-400 at
  ~$9–11 with no structural reason to expect a different outcome.
- *Sequential round 2 (re-mine hard negatives from listwise-best, broadened
  kendall-1 + kendall-2/3 + random candidate set, listwise-only fine-tune on
  top of listwise-best)*: technically sound (this is genuinely trial4's own
  proven mechanism, re-targeted) and cheaper (~$20.5, ~10h, no held-out-val
  redesign needed), but the user judged the ceiling too close to round 1's
  own result (diminishing returns on repeating the same patch) and wanted a
  structurally different bet for the single remaining attempt instead.
  **This remains the fallback if round 2 (below) underperforms its own
  in-training monitor relative to round 1's numbers** — see section 6.

## 2. Round 2's core idea

Instead of stacking a second fine-tune stage on an already-fine-tuned
checkpoint, train **one continuous run from the base model**
(`Qwen/Qwen3-VL-32B-Instruct`, no reuse of ckpt-400 or listwise-best
weights) that **interleaves two loss types on every pass through the data**:

- **SFT steps** (the majority): masked next-token loss on the single ground-
  truth completion, exactly trial3's mechanism (presentation-shuffle
  augmentation, LoRA r=32 on the LLM decoder only, vision tower frozen).
- **Listwise steps** (a fixed fraction of steps, target **30%**): the
  candidate-batch ranking loss from round 1 (`train_listwise.py`'s
  mechanism — batched forward with gradient checkpointing, per-candidate
  logprob via `candidate_logprob_sum`, cross-entropy over candidate scores),
  applied to a **randomly sampled** subset of rows each epoch — not a
  separately-mined hard-negative set (there is no trained checkpoint yet at
  the start of the run to mine with; mining is what round 1 already proved,
  this design deliberately does something else). Candidate set per row:
  truth + 3 adjacent-swap (kendall=1) + 2 near (kendall=2–3) + 1 random = 7,
  reusing `common.broadened_candidate_set` (already implemented and
  unit-tested, 18/18 passing).

**Rationale** (stated plainly, not oversold): sequential fine-tuning risks
the second stage's objective fighting the representations the first stage
already froze in place (a documented failure mode in continual/multi-stage
fine-tuning, part of why modern instruction-tuned LLM pipelines mix SFT and
preference/ranking signal rather than always doing them in strict sequence).
SFT ("the truth is plausible") and listwise ("the truth beats these specific
competitors") are not competing objectives here — they're the same
underlying judgment from two angles — so joint training is expected to
*reinforce* rather than *fight*, unlike genuinely unrelated multi-task
setups where joint training can hurt (negative transfer). **This is a
reasoned bet, not a proven one** — round 1's sequential approach has a real
+1.75pp track record; this approach has none. That trade (higher theoretical
ceiling, zero empirical precedent, on the last attempt) was made explicitly
by the user after the pairwise-rerank idea (section 1) was tested and
rejected, and after being shown round 2's more conservative sequential
alternative and judging its expected ceiling too low.

## 3. Data split

No fully-held-out validation set this trial (`clean_val.csv`'s 1000 rows
were previously never trained on) — the user wants that data used, but not
zero eval signal either (round 2's own earlier iteration considered zero-val
and the user backed off after being shown the risk: no way to catch a
training bug before burning the final Kaggle submission). Compromise reached:
shrink val, don't eliminate it.

- `val_small.csv`: 250 rows, sampled from `clean_val.csv` with `random_state=42`.
- `train_full.csv`: `train_clean.csv` (8,121 rows) + the 750 rows dropped
  from `clean_val.csv` = **8,871 rows**.
- Both derived deterministically from the existing cleaned pool (black/dup
  frame filtering already applied upstream by trial3/trial4's
  `clean_data.py` — not redone here). Script to build them: inline `pandas`
  one-off (already run once against the old box before it went down; must be
  re-run against the restarted box since it's a fresh working directory
  state check — verify `data_work/train_full.csv` / `val_small.csv` exist
  before assuming they survived the stop/restart; the earlier attempt
  failed with `Connection refused` before the split was confirmed written).

Val size resolution: **250 rows** gives ±1σ ≈ 3.1 percentage points
(binomial, p≈0.6) — tighter than round 1's in-training monitor (150 rows,
±3.7pp) and enough to catch a badly broken run, while freeing 750 more rows
into training versus round 1/trial3's 1000-row split.

## 4. Training loop mechanics

- **Base weights**: `Qwen/Qwen3-VL-32B-Instruct`, nf4 QLoRA (double-quant,
  bf16 compute, vision tower skipped from quantization — unchanged from
  trial3/round 1). LoRA r=32 on `q/k/v/o/gate/up/down_proj` of the LLM
  decoder only (unchanged).
- **Per-step branching**: for each training example drawn from `train_full`
  (shuffled per epoch, same deterministic-seed pattern as `train.py`'s
  `sigma_for`), draw a per-row deterministic decision (seeded on
  `SEED:epoch:row_id`, not global RNG state, for resumability): with
  probability 0.30 take the **listwise branch** (build
  `broadened_candidate_set`, batch-collate via `train.make_collate`, compute
  `candidate_logprob_sum` off the batched logits, cross-entropy loss vs
  index 0); otherwise take the **SFT branch** (single-target masked loss,
  exactly `train.py`'s `Rows.__getitem__` + `make_collate` path).
  Gradient-accumulate across whatever mix of branch types lands within one
  optimizer step's `ACCUM` window — the two branches produce differently-
  shaped batches (1 target vs 7 candidates) but both reduce to a single
  scalar loss per micro-step, so accumulation is agnostic to which branch
  produced it.
- **Gradient checkpointing**: ON throughout (both branches use the ordinary
  batched-forward-with-checkpointing path — this is what round 1's post-OOM
  v2 redesign already validated; there is no cache-sharing trick in this
  design at all, so the checkpointing-vs-use_cache conflict that caused
  round 1's v1 OOM does not apply here).
- **Optimizer/schedule**: fresh AdamW, single cosine schedule with warmup
  over the full run's total steps (not two separate schedules) — since this
  is genuinely one continuous run, not two stages.
- **Checkpointing + monitor**: reuse `train.py`'s `save_ckpt` /
  `score_monitor` (real-rule, `tta_k=1`, on `val_small`) /
  `prune_ckpts(keep=3)`. Monitor every ~15% of total steps (5–6 checkpoints
  across the run, ~250-row real-rule check each ≈ 18 min) — sized down from
  round 1's every-40-steps cadence because there are more total steps here
  and monitor cost scales with checkpoint *count*, not just size.
- **No separate hard-negative mining phase** — this is the deliberate
  structural difference from the "round 2 sequential" fallback (section 1).
  The listwise branch's candidates come from `broadened_candidate_set`
  applied to whatever row was already drawn, not from a pre-scored
  low-margin subset.

## 5. Cost / time estimate (uncertain — confirm empirically at kickoff)

| Phase | Basis | Estimate |
|---|---|---|
| Unified training, 8,871 rows × 1 epoch, 30% listwise-branch mix | 70%×~1.7s/row (SFT, **extrapolated**, not measured on this box) + 30%×8.6s/row (listwise, **measured** round 1) | ~10.6h |
| + monitor overhead (5–6 × 250-row real-rule checks) | ~18min each | ~1.7h |
| Final test inference, 819 rows, TTA k=3, α=0.5 (carried over, not re-tuned) | **measured**, 13.6–13.9s/row | ~3.1h |
| **Total** | | **~15.4h, ~$32** at $2.067/h |

**Known unknowns**: the SFT-branch per-row time is the only figure in this
table that isn't directly measured on this exact box — everything else
(listwise step cost, inference cost, VRAM) is empirical. Get a real number
from the first ~50 SFT-only-branch steps before trusting the total; if the
real ratio makes the 70/30 mix meaningfully more expensive than ~$32, the
lever to pull is the listwise-branch probability (0.30 → lower), not epoch
count or data size (per this project's standing "don't cut quality/resolution
for time budget" preference — adjust the *cheap-to-vary* knob, not scope).

## 6. Decision rule at the end (no separate re-comparison run)

`val_small`'s real-rule monitor readings taken *during* training are the
only comparison signal — there is no plan to re-run a large val dump against
the final checkpoint the way round 1's `analyze_scores.py` did (that would
cost another ~$3–4 that this budget doesn't have slack for). Concretely:

1. Take the run's `best/` checkpoint (highest `val_small` score seen during
   training).
2. If `best/`'s final monitor reading is **not clearly better** than round
   1's listwise-best (which scored 0.620 on a *150-row* k=1/α=0 real-rule
   check — not the same val set, so this is a directional sanity check, not
   a rigorous comparison) — treat that as a signal the joint-training bet
   didn't pay off, and **fall back to the "sequential round 2" plan from
   section 1** (re-mine from listwise-best, broadened candidates, ~$20.5,
   ~10h) rather than submitting a plausibly-worse checkpoint on the last
   attempt.
3. If it's clearly better, proceed to final test inference with
   `k=3, alpha=0.5` (carried over unchanged — no budget for re-tuning this
   trial) and produce `trial4_round2_submission.csv`.
4. Either way, verify against `sample_submission.csv` (row count, row order,
   valid-permutation check on every `Answer`) before treating the CSV as
   submit-ready — same checklist round 1 already used successfully.

## 7. Infra / operational notes

- Box: vast.ai instance 45363494 (host 170701, `87.116.91.146:10234`), 1x
  H100 NVL 93.6GB, 3342.2GB/s, the same box round 1 ran on. Was stopped
  (not destroyed) after round 1 — disk shows 82GB used, meaning the 65GB
  model snapshot + data + code should have survived; **verify this before
  re-downloading anything**.
- Round 1's `ckpts_listwise/best/` checkpoint was **never fully backed up
  locally** (connection dropped mid-transfer, deferred by user request) —
  confirm it's still on the box's disk once reachable; it's the fallback
  path's starting point (section 6, step 2) and currently exists nowhere
  else.
- Rule compliance unchanged from round 1: train-only data (Rule 3.4), no
  ensembling (single adapter, single set of weights), open-source base
  model released before 2026-05-31, LoRA/quantization explicitly permitted,
  TTA explicitly permitted as inference-time augmentation.
