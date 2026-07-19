# Linux CUDA 12.4 training setup

This environment targets the competition deployment machine: Linux, one NVIDIA
RTX 3090 with 24 GB VRAM, and an NVIDIA driver compatible with CUDA 12.4.

The pinned CUDA packages follow the official PyTorch CUDA 12.4 pairing:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
```

Use CPython 3.11. Python 3.10 or 3.12 should also work, but do not use Python
3.14 with this torch/vision pair.

## 1. Check the server

```bash
nvidia-smi
python3.11 --version
```

`nvidia-smi` must show the RTX 3090 and a driver supporting CUDA 12.4 or newer.
The full CUDA toolkit does not need to be installed because the PyTorch wheel
contains its CUDA runtime; the NVIDIA driver is still required.

## 2. Create the environment

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r haram831/requirements-linux-cu124.txt
```

Do not copy a Windows `.venv` or a virtual environment from another server.

## 3. Verify CUDA and runtime dependencies

```bash
python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("runtime CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

python -m pip check
python -m bitsandbytes
```

Expected core values:

```text
torch: 2.6.0+cu124
torchvision: 0.21.0+cu124
runtime CUDA: 12.4
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

The optional `triton not found; flop counting will not work` warning can be
ignored by Candidate1 unless it is followed by a real exception. Candidate1
does not call `torch.compile` or require Triton directly.

## 4. Verify the local model and processor

Replace the model path if the checkpoint is stored elsewhere:

```bash
export PYTHONPATH="$PWD/haram831/src"

python - <<'PY'
from transformers import AutoProcessor

AutoProcessor.from_pretrained(
    "/workspace/models/Qwen2-VL-2B-Instruct",
    local_files_only=True,
)
print("processor OK")
PY
```

The competition data and model weights are not included in `git clone`; copy
them to the server separately.

## 5. Run tests and the bounded A1 smoke test

```bash
cd haram831
export PYTHONPATH="$PWD/src"
python -m pytest -q --basetemp .pytest-tmp/linux-cu124
```

Example A1 tiny overfit command:

```bash
python -m snu_ordering.candidate1.tiny_overfit \
  --config configs/candidate1-a1.json \
  --train-csv /workspace/data/train.csv \
  --image-root /workspace/data/train \
  --output-dir runs/candidate1-a1-tiny \
  --base-model /workspace/models/Qwen2-VL-2B-Instruct \
  --processor /workspace/models/Qwen2-VL-2B-Instruct \
  --local-files-only
```

Do not start the 1,200-step screening run until the tiny run completes a
forward pass, backward pass, checkpoint save/reload, and CUDA memory report.
