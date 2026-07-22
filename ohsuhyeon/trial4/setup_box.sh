#!/usr/bin/env bash
# Provision a fresh vast.ai box (pytorch/cuda base image) for trial4.
# Versions PINNED to the exact set trial3's LB-0.89 run used (versions.txt),
# so the transformers-5.14 workarounds in common.py stay valid.
# Usage:  bash setup_box.sh            # installs deps + pre-downloads model
set -euo pipefail

pip install -q "transformers==5.14.1" "peft==0.19.1" "accelerate==1.14.0" \
    "qwen-vl-utils==0.0.14" bitsandbytes pandas pillow numpy safetensors

python - <<'EOF'
from huggingface_hub import snapshot_download
import os
m = os.environ.get("SNU_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
print("downloading", m)
snapshot_download(m)
print("done")
EOF

pip freeze | grep -Ei "transformers|peft|torch==|qwen-vl|accelerate|bitsandbytes" \
    > versions.txt || true
echo "--- setup complete; pinned versions in versions.txt"
echo "next: upload data + ckpt-400, then:  python smoke_test.py"
