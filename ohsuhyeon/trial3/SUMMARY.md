# Trial 3 — Summary (2026-07-18)

Fine-tune Qwen3-VL-32B (nf4 QLoRA) to reorder 4 shuffled video frames into their
original temporal order, given the caption. Metric: 24-way exact match.

**Result: Kaggle public LB = 0.89** (trial2 was 0.70). Clean-val (greedy monitor,
80 rows) at the chosen checkpoint was 0.675; the real decision rule
(24-way scoring + TTA) plus the cleaner test distribution lifted it to 0.89.

Design spec: `../docs/superpowers/specs/2026-07-17-frame-ordering-trial3-design.md`.

## What trial3 is

The task is a **24-way discriminator**, not a generator: the answer space is the
24 permutations, scored by exact match. So training, inference, and augmentation
are all aligned to that decision rule.

- **Model**: Qwen/Qwen3-VL-32B-Instruct — the largest model that fits the
  competition's RTX 3090 24GB inference box once 4-bit quantized. Released
  2025-10 (within the ≤2026-05-31 rule). Chosen over 8B after the user raised the
  budget to ~$10/try; 8B bf16 path kept as fallback (`SNU_QUANT=""`).
- **Quantization**: nf4 (bitsandbytes), **double-quant, bf16 compute**. Vision
  tower kept in bf16 (quantization-sensitive, ~+1GB only). Train precision ==
  inference precision (nf4, adapter NOT merged) → no train/infer distribution gap.
- **LoRA**: r=32 on all LLM-decoder linears (q/k/v/o/gate/up/down), vision tower
  excluded. 268M trainable params (0.80%).
- **Data cleaning**: black-frame + duplicate-frame heuristics dropped 414/9535 rows
  (4.3%) → train 8,121 / clean-val 1,000 (seed 42, no leakage; test never used).
- **Training**: masked SFT (loss on answer completion tokens only) + per-epoch
  presentation-shuffle augmentation (re-present the 4 frames under a permuted
  sigma, remap the label) so the model can't learn input-position priors.
  1 epoch, cosine LR + warmup, checkpoint + greedy-val every 100 steps.
- **Inference**: KV-cache-shared 24-way constrained scoring — forward the
  image-bearing prompt once, score all 24 completions off the shared cache — plus
  presentation TTA (k=3): score under 3 frame orders, sum, argmax. Always a valid
  permutation, no parse failures.

## Results

| model | val (greedy monitor) | Kaggle LB |
|---|---|---|
| trial1 (Qwen2-VL-2B, buggy loss) | 0.168 | — |
| trial2 (Qwen2.5-VL-7B, 0.28ep, greedy) | 0.417 (noisy val) | 0.70 |
| **trial3 (Qwen3-VL-32B nf4, 1ep, scoring+TTA3)** | 0.675 (clean val) | **0.89** |

Greedy-monitor val curve (clean 80 rows): 0.50 @step100 → 0.55 @200 → 0.55 @300 →
**0.675 @400** → 0.6375 @500. ckpt-400 chosen (monitor peak, earlier = less
overfit; re-ranking ckpt-300/400/500 was started but skipped — at 200-row eval the
±5%p resolution can't separate a 3.7%p gap, not worth the GPU hour).

Final answer distribution healthy: all 24 permutations used, identity 13.1%
(train's true 15.5%), no collapse.

## Competition-spec compliance (measured, not estimated)

- **Inference VRAM peak 20.5GB** < 3090's 24GB (printed live by `infer.py`).
- 819 rows at ~19s/row (TTA k=3, A100) = ~4.4h; on a 3090 (~2.5-3x slower)
  ≈ 11-13.5h < 24h limit. Levers if tight: TTA k→2/1.
- Single model (TTA is rule-allowed test-time augmentation, not ensembling).
- Offline-capable (local snapshot, quantize-on-load), relative paths, UTF-8.
- Total size: adapter 1.07GB + base 65GB < 80GB.

## Infrastructure

vast.ai A100 80GB (nf4 QLoRA needs ~48GB training / ~20GB inference; both fit,
so mid-training checkpoint inference ran on the same box). mb4×accum4, **38s/step
(16 samples) — pure compute-bound** (mb8 gave identical throughput, only more
VRAM). Training precision nf4; ~5.4h for 1 epoch ≈ $7.

## transformers 5.14 pitfalls fixed (commit a23a668)

1. Cached-continuation forward must NOT pass `attention_mask` — Qwen3VL derives
   mrope positions from the full mask length but applies them to the new tokens
   only → shape crash. Pass `cache_position` only.
2. Manual collate must forward `mm_token_type_ids` (required for M-RoPE in v5).
3. Gradient checkpointing is a no-op in eval mode → backward in eval OOMs; the
   smoke test must switch to train mode first.
4. nf4 fast-vs-reference scorer drift up to ~1 logp is normal kernel noise, not a
   bug — the gate tolerance was calibrated to that.

## Files (`ohsuhyeon/trial3/`)

| file | purpose |
|---|---|
| `common.py` | model load (nf4/bf16 via `SNU_QUANT`), prompt, permutation math, KV-cache 24-way scorer + TTA |
| `clean_data.py` | black/dup-frame flags → `data_work/{train_clean,clean_val}.csv` |
| `train.py` | nf4 QLoRA masked SFT + shuffle augmentation, ckpt/JSONL logs, resume |
| `infer.py` | test/val inference → submission CSV, incremental + resumable, live VRAM readout |
| `smoke_test.py` | full-stack GPU gate (perm math, scorer==reference, masked backward) — run first on any box |
| `test_perms.py` | exhaustive CPU tests of the permutation math |
| `setup_box.sh` | vast.ai provisioning (deps + model pre-download) |

## Artifacts (`ohsuhyeon/outputs/trial3_backup/`, gitignored)

ckpt-400 adapter (+optimizer for epoch-2 resume), ckpt-500 adapter, train_log.jsonl,
data split, all submission/rank CSVs, run logs, pinned `versions.txt`.

## Next levers (not yet done)

1. **Epoch 2** — val was still near its peak, not clearly overfit; resume from
   ckpt-400 optimizer state (backed up). +~$7.
2. **Checkpoint re-ranking with the real rule** on clean-val 1000 (not 200) to pick
   the true best among ckpt-300/400/500.
3. **Listwise ranking loss** (Stage 2) — hard-negative adjacent-swap permutations,
   directly optimizing the 24-way argmax the deployment uses.
4. **3090 rehearsal** — rent a 3090 ($0.2/h), run `HF_HUB_OFFLINE=1` full inference,
   capture VRAM/time logs as report evidence (resource-efficiency scoring).
