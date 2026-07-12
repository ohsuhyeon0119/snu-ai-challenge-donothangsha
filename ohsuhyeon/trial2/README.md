# Trial 2 — Frame-Ordering: fixed masked-loss QLoRA + constrained scoring

Design: `../docs/superpowers/specs/2026-07-12-frame-ordering-trial2-design.md`.

## What this fixes vs trial1 (16.77%)
trial1 masked only padding from the loss, so the ~10 answer tokens were <1% of the
sequence loss and the model collapsed to the marginal answer distribution. Trial2:
1. **Masked loss** — supervise only the answer completion tokens (verified: ~17
   supervised tokens per example in `smoke_test.py`).
2. LoRA on all linear projections (`q,k,v,o,gate,up,down`), rank 32.
3. Cosine LR schedule with warmup.
4. Periodic constrained-scoring validation → keep the **best** adapter.
5. Lower `max_pixels` → fewer vision tokens → faster steps.
6. **Constrained scoring** at inference: rank all 24 permutations by model
   likelihood, take argmax → always a valid permutation, no parse failures.

## Files
| file | purpose |
|---|---|
| `common.py` | model/processor load, prompt, label↔answer mapping, `score_orders` |
| `train.py` | QLoRA fine-tune, masked loss, best-ckpt selection, resumable |
| `eval.py` | full held-out val exact-match with the best adapter |
| `infer.py` | test.csv → submission CSV (constrained scoring) |
| `clean_data.py` | report near-black / duplicate frames (run first, decide later) |
| `smoke_test.py` | one forward/backward + a few scorings, no checkpoints |

## Run order (on the GPU box)
```bash
export HF_HOME=/workspace/.hf_home
# --- Stage A: 2B (technique validation) ---
export SNU_MODEL="Qwen/Qwen2-VL-2B-Instruct"
export SNU_ADAPTER_DIR=/workspace/trial2_adapter_2b
export SNU_VAL_SPLIT_CSV=/workspace/trial2_val_split.csv
/venv/main/bin/python smoke_test.py          # sanity
/venv/main/bin/python train.py               # ~5-7h, writes best/ adapter
/venv/main/bin/python eval.py                # full val exact-match (best adapter)
/venv/main/bin/python infer.py               # -> /workspace/trial2_submission.csv

# --- Stage B: 7B (final) — needs more instance disk (fp16 ~16GB) ---
export SNU_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export SNU_ADAPTER_DIR=/workspace/trial2_adapter_7b
# same three commands
```

Launch training detached so it survives disconnects:
```bash
tmux new-session -d -s train2b "bash run_2b.sh > /workspace/trial2_2b.log 2>&1"
tmux attach -t train2b     # watch;  Ctrl-b d to detach
tail -f /workspace/trial2_2b.log
cat /workspace/trial2_adapter_2b/progress.json   # best_acc + val history
```

## Env knobs
`SNU_MODEL SNU_DATA_DIR SNU_ADAPTER_DIR SNU_VAL_SPLIT_CSV SNU_MAX_PIXELS SNU_MIN_PIXELS`
`SNU_EPOCHS SNU_LR SNU_ACCUM SNU_LORA_R SNU_EVAL_EVERY SNU_EVAL_N SNU_SCORE_CHUNK`
`SNU_ADAPTER_SUBDIR` (eval/infer: `best` default, or empty for last)

## Rule compliance
Single model (no ensembling), provided data only, no generative augmentation,
Qwen2-VL / Qwen2.5-VL weights public before 2026-05-31, final inference on one
RTX 3090 24GB within the 24h budget, no metadata/leakage signals.
