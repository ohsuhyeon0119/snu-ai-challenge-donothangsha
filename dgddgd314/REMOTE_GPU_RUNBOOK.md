# Remote GPU Runbook

Use this on the rented GPU box. Keep local work limited to data checks, submission validation, and code edits.

## Setup

```bash
cd /workspace/project/dgddgd314
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SNU_DATA_DIR=/workspace/project/dgddgd314/data/snuaichallenge_data
export SNU_MODEL_REPO=google/paligemma2-10b-pt-224
export SNU_MODEL_DIR=/workspace/models/paligemma2-10b-pt-224
export SNU_MODEL=/workspace/models/paligemma2-10b-pt-224
export SNU_AUTO_DOWNLOAD_MODEL=1
export SNU_ADAPTER_DIR=/workspace/project/dgddgd314/outputs/paligemma_lora
export SNU_WORK_DIR=/workspace/project/dgddgd314/data_work
export SNU_TTA_K=3
export SNU_SCORE_CHUNK=4
```

The overnight runner downloads the model to `SNU_MODEL_DIR` if `SNU_MODEL/config.json` is missing. If the Google model is gated, login before loading or set `HF_TOKEN`:

```bash
huggingface-cli login
# or: export HF_TOKEN=...
```

## First Run

```bash
python scripts/clean_data.py --data-dir "$SNU_DATA_DIR" --out-dir "$SNU_WORK_DIR"
python scripts/test_perms.py
python scripts/paligemma_smoke.py --data-dir "$SNU_DATA_DIR" --no-load-model
python scripts/paligemma_train_skeleton.py --max-steps 20 --eval-every 0
python scripts/paligemma_infer.py --limit 20 --tta-k 1 --out /workspace/outputs/paligemma_probe.csv
python scripts/validate_submission.py --data-dir "$SNU_DATA_DIR" --submission /workspace/outputs/paligemma_probe.csv
```

## Overnight

```bash
bash scripts/run_paligemma_overnight.sh
```

## Notes

- Final compatibility target is one RTX 3090 24GB.
- Main path uses true multi-image inputs, not contact sheets.
- Start with 224 resolution; test 448 only after the 224 pipeline is stable.
- Do not use external data.
- Do not push generated data, outputs, checkpoints, or model weights.
