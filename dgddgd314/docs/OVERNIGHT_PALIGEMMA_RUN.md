# Overnight PaliGemma Run

This is the unattended 9-hour plan: about 7 hours for training, then full test inference and output packaging.

## Assumptions

- Repo is at `/workspace/project/dgddgd314`.
- Dataset is unpacked at `/workspace/project/dgddgd314/data/snuaichallenge_data`.
- Local model is unpacked at `/workspace/models/paligemma-3b-pt-448`.
- The active Python environment has CUDA PyTorch installed.

## Server Commands

```bash
cd /workspace/project/dgddgd314
git pull origin dgddgd314/gemma
chmod +x scripts/run_paligemma_overnight.sh
```

Run inside tmux:

```bash
tmux new -s overnight
```

Inside tmux:

```bash
cd /workspace/project/dgddgd314
export PYTHON_BIN=python
export SNU_TRAIN_SECONDS=25200
export SNU_VAL_SIZE=50
export SNU_EVAL_EVERY=5000
export SNU_SCORE_CHUNK=1
export SNU_ADAPTER_DIR=/workspace/project/dgddgd314/outputs/paligemma_lora_overnight
export SNU_SUBMISSION_OUT=outputs/paligemma_submission_overnight.csv
export SNU_PACKAGE_OUT=/workspace/paligemma_overnight_outputs.tar.gz
bash scripts/run_paligemma_overnight.sh
```

Detach:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t overnight
```

## Morning Checks

```bash
cd /workspace/project/dgddgd314
cat outputs/paligemma_lora_overnight/eval_log.csv
cat outputs/paligemma_lora_overnight/best_metrics.json
ls -lh outputs/paligemma_submission_overnight.csv
ls -lh /workspace/paligemma_overnight_outputs.tar.gz
```

Download from local PowerShell:

```powershell
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P <port> root@<host>:/workspace/paligemma_overnight_outputs.tar.gz .\paligemma_overnight_outputs.tar.gz
```

## Error Cases

If `python` is not the CUDA PyTorch Python, use:

```bash
export PYTHON_BIN=/venv/main/bin/python
```

If inference OOMs, keep:

```bash
export SNU_SCORE_CHUNK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If no best adapter exists, the runner automatically uses the final adapter folder.
