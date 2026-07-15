# Remote GPU Runbook

Use this tomorrow on the rented GPU box. Keep local work limited to data checks,
submission validation, and code edits.

## Setup

```bash
cd /workspace/project/dgddgd314
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SNU_DATA_DIR=/workspace/data/snuaichallenge_data
export SNU_MODEL=google/paligemma-3b-pt-448
export SNU_ADAPTER_DIR=/workspace/outputs/paligemma_lora
export SNU_CONTACT_SHEET_SIZE=448
```

If the Google model is gated, login before loading:

```bash
huggingface-cli login
```

## First Run

```bash
python scripts/paligemma_smoke.py
python scripts/audit_data.py --data-dir "$SNU_DATA_DIR" --out /workspace/outputs/data_audit.csv
python scripts/paligemma_train_skeleton.py --max-steps 100
python scripts/paligemma_infer.py --limit 20 --out /workspace/outputs/paligemma_probe.csv
python scripts/validate_submission.py --data-dir "$SNU_DATA_DIR" --submission /workspace/outputs/paligemma_probe.csv
```

## Notes

- Final compatibility target is one RTX 3090 24GB.
- Keep image processing code and settings recorded.
- Do not use external data.
- Do not push generated data, outputs, checkpoints, or model weights.

