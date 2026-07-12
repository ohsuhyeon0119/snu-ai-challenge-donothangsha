# Trial 2 — Summary (2026-07-12)

Fine-tune a VLM to reorder 4 shuffled video frames into their original temporal order,
given the caption. Metric: 24-way exact-match. Goal: move well past trial1's 16.77%.

Design spec: `../docs/superpowers/specs/2026-07-12-frame-ordering-trial2-design.md`.

## Root-cause of trial1's stall (16.77%)
trial1's LoRA loss masked **only padding**, so the whole prompt + hundreds of vision tokens were
included in the loss and the ~10 answer tokens were <1% of it. The model learned the **marginal**
answer distribution, not the conditional — hence the collapse onto two "safe" answers
(`[1,2,3,4]` 28%, `[3,2,4,1]` 21% = ~49% of test predictions) and the loss plateau at ~6.8.

## Key reframing
Captions narrate events in temporal order ("...before...then...ending with..."). So the task is
**aligning the 4 frames to the caption's narrated sequence**, not inferring motion from pixels.
We feed the **whole caption + all 4 frames** and let the model learn the alignment end-to-end —
we do **not** split the caption into 4 clauses (captions don't split cleanly into exactly 4).

## What trial2 changes
1. **Masked loss** — supervise only the answer completion tokens (verified: 17 supervised
   tokens/example). This is the single biggest fix.
2. LoRA on **all** linear projs (`q,k,v,o,gate,up,down`), rank 32 (trial1: `q,v` r=16).
3. Cosine LR schedule with warmup.
4. Periodic validation → keep the **best** adapter (trial1 evaluated once, at the end).
5. Lower `max_pixels` to cut vision tokens / speed steps.
6. **Constrained scoring** at inference — rank all 24 permutations by model likelihood, argmax →
   always a valid permutation, no parse failures. (`infer.py` also has a fast greedy mode.)

## Code (`ohsuhyeon/trial2/`)
| file | purpose |
|---|---|
| `common.py` | model/processor load, prompt, label↔answer mapping, `score_orders`, `generate_order` |
| `train.py` | micro-batch=1 QLoRA (for the 24GB 3090), masked loss, best-ckpt, resumable |
| `train_batched.py` | padding-collate batched QLoRA for a big-VRAM box (A100), greedy-gen monitor |
| `eval.py` | full held-out val exact-match (constrained scoring) |
| `infer.py` | test.csv → submission CSV; `SNU_INFER_MODE=gen|score`; incremental write + resume |
| `clean_data.py` | report near-black / duplicate frames |
| `smoke_test.py` | one masked forward/backward + a few scorings |

## Results (greedy-gen val, seed-42 12% split, n≈1145)
| model | val exact-match |
|---|---|
| majority-class floor | 0.155 |
| trial1 (Qwen2-VL-2B, buggy loss) | 0.168 |
| **trial2 — Qwen2-VL-2B (3090)** | 0.258 → 0.375 → **0.40** over ~1 epoch |
| **trial2 — Qwen2.5-VL-7B (A100)** | 0.40 @ step150 → **0.4167 @ step300** (~0.28 epoch), stopped early |

The 7B reached the 2B's best (0.40) at ~0.14 epoch vs the 2B's ~1.1 epoch — much higher ceiling
and faster learning. Trajectory was still rising when stopped.

## Infrastructure
- **Training on rented GPUs** (local dev machine is Apple M5, no CUDA):
  - 2B: Vast.ai RTX 3090 24GB (micro-batch=1 + grad accum).
  - 7B: Vast.ai A100 80GB, 128 CPU (batched micro-batch=8). Data moved local→box via
    tar-over-ssh; model pre-downloaded in parallel. Controlled over SSH, run in `tmux`.
- One eval-time gotcha fixed: greedy-gen validation stalled at high memory (GPU idle) because the
  training allocator cache fragmented KV-cache allocation → added `torch.cuda.empty_cache()`
  around eval in `train_batched.py`.

## Submission
`infer.py` (greedy mode) over `test.csv` with the 7B best adapter →
`ohsuhyeon/outputs/trial2_submission_7b.csv` (gitignored). Validated: 819 rows, `Id,Answer`,
0 missing/dup/invalid, order matches `sample_submission.csv`. Answer distribution is healthy
(identity `[1,2,3,4]` 20.1%, spread across permutations — **no collapse**, unlike trial1's 49%).

7B adapter + logs + val split backed up to `ohsuhyeon/outputs/trial2_7b_backup/` (gitignored).

## Next levers toward 0.8
1. **Data cleaning** — drop uncleaned/ambiguous train samples (black frames, order not
   determinable) that add noise to training and the val metric.
2. **Full / more epochs** — trajectory was still climbing when stopped.
3. **Constrained-scoring inference** — a few points over greedy (slower ~16s/row; needs a
   vision-encode-once optimization to be fast).
4. **STaR-style self-CoT** — model generates reasoning, keep only traces that reach the correct
   (known) answer, fine-tune on those. Rules allow CoT at inference; confirm with organizers re:
   self-generated training traces before relying on it.
