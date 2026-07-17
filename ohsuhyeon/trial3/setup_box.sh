#!/usr/bin/env bash
# Provision a fresh vast.ai box (pytorch/cuda base image) for trial3.
# Usage:  bash setup_box.sh            # installs deps + pre-downloads model
set -euo pipefail

pip install -q -U "transformers>=4.57.0" "peft>=0.13" accelerate \
    "qwen-vl-utils>=0.0.14" bitsandbytes pandas pillow numpy safetensors

python - <<'EOF'
from huggingface_hub import snapshot_download
import os
m = os.environ.get("SNU_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
print("downloading", m)
snapshot_download(m)
print("done")
EOF

pip freeze | grep -Ei "transformers|peft|torch==|qwen-vl|accelerate" \
    > versions.txt || true
echo "--- setup complete; pinned versions in versions.txt"
echo "next: upload data, then:  python smoke_test.py"
