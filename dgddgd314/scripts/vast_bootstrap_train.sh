#!/usr/bin/env bash
set -euo pipefail

cd "${SNU_REPO_DIR:-/workspace/project/dgddgd314}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${SNU_VENV_DIR:-.venv}"

export SNU_DATA_DIR="${SNU_DATA_DIR:-/workspace/project/dgddgd314/data/snuaichallenge_data}"
export SNU_MODEL_REPO="${SNU_MODEL_REPO:-google/paligemma2-10b-pt-224}"
export SNU_MODEL_DIR="${SNU_MODEL_DIR:-/workspace/models/${SNU_MODEL_REPO##*/}}"
export SNU_MODEL="${SNU_MODEL:-${SNU_MODEL_DIR}}"
export SNU_AUTO_DOWNLOAD_MODEL="${SNU_AUTO_DOWNLOAD_MODEL:-1}"
export SNU_ADAPTER_DIR="${SNU_ADAPTER_DIR:-/workspace/project/dgddgd314/outputs/paligemma_lora_probe}"
export SNU_WORK_DIR="${SNU_WORK_DIR:-/workspace/project/dgddgd314/data_work}"
export SNU_SCORE_CHUNK="${SNU_SCORE_CHUNK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== gpu =="
nvidia-smi || true

if [ "${SNU_CREATE_VENV:-1}" = "1" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN=python
fi

echo "== python =="
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", exc)
PY

echo "== dependencies =="
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install pandas pillow tqdm transformers peft accelerate bitsandbytes huggingface_hub

echo "== unpack data/model archives if present =="
mkdir -p data /workspace/models outputs "${SNU_WORK_DIR}"
if [ -f /workspace/snu_data.tar ] && [ ! -f "${SNU_DATA_DIR}/train.csv" ]; then
  tar -xf /workspace/snu_data.tar
fi
if [ -f "/workspace/${SNU_MODEL_REPO##*/}.tar" ] && [ ! -f "${SNU_MODEL}/config.json" ]; then
  tar -xf "/workspace/${SNU_MODEL_REPO##*/}.tar" -C /workspace/models
fi

echo "== data =="
test -f "${SNU_DATA_DIR}/train.csv"
test -f "${SNU_DATA_DIR}/test.csv"
ls -lh "${SNU_DATA_DIR}" | head

echo "== model =="
if [ "${SNU_AUTO_DOWNLOAD_MODEL}" = "1" ] && [ ! -f "${SNU_MODEL}/config.json" ]; then
  "${PYTHON_BIN}" scripts/download_hf_model.py \
    --repo-id "${SNU_MODEL_REPO}" \
    --out-dir "${SNU_MODEL_DIR}" \
    --cache-dir "${SNU_HF_CACHE_DIR:-/workspace/models/.hf_cache}" \
    --no-archive \
    --skip-if-exists
  export SNU_MODEL="${SNU_MODEL_DIR}"
fi
echo "${SNU_MODEL}"
test -f "${SNU_MODEL}/config.json"

echo "== smoke and probe train =="
"${PYTHON_BIN}" scripts/clean_data.py --data-dir "${SNU_DATA_DIR}" --out-dir "${SNU_WORK_DIR}" --val-size "${SNU_VAL_SIZE:-1000}"
"${PYTHON_BIN}" scripts/test_perms.py
"${PYTHON_BIN}" scripts/paligemma_smoke.py --data-dir "${SNU_DATA_DIR}" --no-load-model
"${PYTHON_BIN}" scripts/paligemma_train_skeleton.py --max-steps "${SNU_PROBE_STEPS:-10}" --eval-every 0

echo "== probe inference =="
"${PYTHON_BIN}" scripts/paligemma_infer.py --limit 20 --tta-k 1 --score-chunk "${SNU_SCORE_CHUNK}" --out outputs/paligemma_probe.csv
"${PYTHON_BIN}" scripts/validate_submission.py --data-dir "${SNU_DATA_DIR}" --submission outputs/paligemma_probe.csv

echo "Probe succeeded. Start the long run with: bash scripts/run_paligemma_overnight.sh"
