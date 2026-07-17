# Trial 3 — Qwen3-VL-8B 24-way discriminator

Design: [../docs/superpowers/specs/2026-07-17-frame-ordering-trial3-design.md](../docs/superpowers/specs/2026-07-17-frame-ordering-trial3-design.md)

bf16 end-to-end (no quantization), LoRA SFT masked to answer tokens,
presentation-shuffle augmentation, constrained 24-way scoring (+TTA) at
inference. Fits the competition target: RTX 3090 24GB, 819 rows well inside
24h.

## Files

| file | purpose |
|---|---|
| `common.py` | model load (bf16), prompt, permutation math, KV-cache 24-way scorer |
| `test_perms.py` | exhaustive CPU tests of the permutation math |
| `clean_data.py` | flags black/duplicate-frame rows → `data_work/{train_clean,clean_val}.csv` |
| `train.py` | bf16 LoRA masked SFT + shuffle augmentation, ckpt/JSONL logs, resume |
| `infer.py` | test/val inference → submission CSV, incremental + resumable |
| `smoke_test.py` | 3-min full-stack check — run first on every fresh GPU box |
| `setup_box.sh` | vast.ai box provisioning (deps + model pre-download) |

## Run order (fresh A100 box)

```bash
bash setup_box.sh
# upload data + this dir, e.g.:
#   tar -C .. -czf - snuaichallenge_data trial3 | ssh BOX 'tar -xzf - -C /workspace'
export SNU_DATA_DIR=/workspace/snuaichallenge_data
python clean_data.py          # or upload data_work/ made locally
python smoke_test.py          # MUST pass before training
tmux new -s train
python train.py               # ~2 epochs, ckpt every 150 steps → ckpts/
```

Mid-training submission from any checkpoint (separate 3090 box is fine):

```bash
scp -r A100:/workspace/trial3/ckpts/best ./ckpt
SNU_ADAPTER_DIR=ckpt SNU_TTA_K=3 python infer.py   # → outputs/trial3_submission.csv
```

Validation readout: `SNU_SPLIT=val SNU_ADAPTER_DIR=ckpts/best python infer.py`

## Env knobs

`SNU_MODEL` (default Qwen/Qwen3-VL-8B-Instruct) · `SNU_DATA_DIR` ·
`SNU_MAX_PIXELS` (default 512·28·28 ≥ native, i.e. no downscaling) ·
train: `SNU_MB` 8 / `SNU_ACCUM` 2 / `SNU_EPOCHS` 2 / `SNU_LR` 1e-4 /
`SNU_LORA_R` 32 / `SNU_SAVE_STEPS` 150 · infer: `SNU_ADAPTER_DIR` /
`SNU_TTA_K` / `SNU_SPLIT` / `SNU_LIMIT`

## Competition-spec notes

- Inference precision bf16; weights ~18GB → fits 3090 24GB without
  quantization. Verify with the rehearsal run (`SNU_LIMIT=20 python infer.py`
  prints VRAM peak + s/row).
- All paths relative / env-var overridable; UTF-8; no internet needed at
  runtime once the model snapshot is cached (`HF_HUB_OFFLINE=1` to prove it).
