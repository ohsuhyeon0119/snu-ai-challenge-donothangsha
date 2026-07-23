#!/usr/bin/env bash
set -euo pipefail

cd "${SNU_REPO_DIR:-/workspace/project/dgddgd314}"

export SNU_DATA_DIR="${SNU_DATA_DIR:-/workspace/project/dgddgd314/data/snuaichallenge_data}"
export SNU_MODEL_REPO="${SNU_MODEL_REPO:-google/paligemma2-10b-pt-224}"
export SNU_MODEL_DIR="${SNU_MODEL_DIR:-/workspace/models/${SNU_MODEL_REPO##*/}}"
export SNU_MODEL="${SNU_MODEL:-${SNU_MODEL_DIR}}"
export SNU_AUTO_DOWNLOAD_MODEL="${SNU_AUTO_DOWNLOAD_MODEL:-1}"
export SNU_ADAPTER_DIR="${SNU_ADAPTER_DIR:-/workspace/project/dgddgd314/outputs/paligemma_lora_overnight}"
export SNU_WORK_DIR="${SNU_WORK_DIR:-/workspace/project/dgddgd314/data_work}"
export SNU_SCORE_CHUNK="${SNU_SCORE_CHUNK:-4}"
export SNU_TTA_K="${SNU_TTA_K:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SECONDS="${SNU_TRAIN_SECONDS:-25200}"
MAX_STEPS="${SNU_TRAIN_MAX_STEPS:-0}"
EPOCHS="${SNU_EPOCHS:-1}"
EVAL_LIMIT="${SNU_EVAL_LIMIT:-200}"
EVAL_EVERY="${SNU_EVAL_EVERY:-200}"
LOG_EVERY="${SNU_LOG_EVERY:-20}"
SAVE_EVERY="${SNU_SAVE_EVERY:-0}"
SUBMISSION_OUT="${SNU_SUBMISSION_OUT:-outputs/paligemma_submission_overnight.csv}"
PACKAGE_OUT="${SNU_PACKAGE_OUT:-/workspace/paligemma_overnight_outputs.tar.gz}"

mkdir -p outputs "${SNU_WORK_DIR}" "$(dirname "${SNU_MODEL_DIR}")"

if [ "${SNU_AUTO_DOWNLOAD_MODEL}" = "1" ]; then
  if [ -f "${SNU_MODEL}/config.json" ]; then
    echo "== model =="
    echo "using existing model: ${SNU_MODEL}"
  else
    echo "== model download =="
    echo "model not found at: ${SNU_MODEL}"
    export SNU_MODEL="${SNU_MODEL_DIR}"
    mkdir -p "$(dirname "${SNU_MODEL}")"
    if [ -f "${SNU_MODEL}/config.json" ]; then
      echo "using existing model: ${SNU_MODEL}"
    else
      echo "downloading repo: ${SNU_MODEL_REPO}"
      echo "download target: ${SNU_MODEL}"
      "${PYTHON_BIN}" scripts/download_hf_model.py \
        --repo-id "${SNU_MODEL_REPO}" \
        --out-dir "${SNU_MODEL}" \
        --cache-dir "${SNU_HF_CACHE_DIR:-/workspace/models/.hf_cache}" \
        --no-archive \
        --skip-if-exists
    fi
  fi
else
  echo "== model =="
  echo "auto-download disabled; SNU_MODEL=${SNU_MODEL}"
fi

echo "== clean data =="
"${PYTHON_BIN}" scripts/clean_data.py --data-dir "${SNU_DATA_DIR}" --out-dir "${SNU_WORK_DIR}" --val-size "${SNU_VAL_SIZE:-1000}"

echo "== smoke =="
"${PYTHON_BIN}" scripts/test_perms.py
"${PYTHON_BIN}" scripts/paligemma_smoke.py --data-dir "${SNU_DATA_DIR}" --no-load-model

echo "== train =="
echo "model: ${SNU_MODEL}"
echo "adapter: ${SNU_ADAPTER_DIR}"
echo "train_seconds: ${TRAIN_SECONDS}"
echo "eval_every: ${EVAL_EVERY}"
echo "eval_limit: ${EVAL_LIMIT}"

"${PYTHON_BIN}" scripts/paligemma_train_skeleton.py \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --max-seconds "${TRAIN_SECONDS}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-limit "${EVAL_LIMIT}" \
  --eval-tta-k 1 \
  --score-chunk "${SNU_SCORE_CHUNK}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}"

ADAPTER_FOR_INFER="${SNU_ADAPTER_DIR}/best"
if [ ! -d "${ADAPTER_FOR_INFER}" ]; then
  ADAPTER_FOR_INFER="${SNU_ADAPTER_DIR}"
fi

export SNU_ADAPTER_DIR="${ADAPTER_FOR_INFER}"

echo "== infer =="
echo "adapter_for_infer: ${SNU_ADAPTER_DIR}"
echo "submission: ${SUBMISSION_OUT}"
echo "tta_k: ${SNU_TTA_K}"

"${PYTHON_BIN}" scripts/paligemma_infer.py --out "${SUBMISSION_OUT}" --tta-k "${SNU_TTA_K}" --score-chunk "${SNU_SCORE_CHUNK}"
"${PYTHON_BIN}" scripts/validate_submission.py --data-dir "${SNU_DATA_DIR}" --submission "${SUBMISSION_OUT}"

"${PYTHON_BIN}" -c "import pandas as pd; p='${SUBMISSION_OUT}'; df=pd.read_csv(p); print('rows:', len(df)); print('unique answers:', df['Answer'].nunique()); print(df['Answer'].value_counts().head(24)); print(df.head(10).to_string(index=False))" | tee outputs/overnight_submission_stats.txt

tar -czf "${PACKAGE_OUT}" "${SNU_ADAPTER_DIR}" "${SUBMISSION_OUT}" "${SUBMISSION_OUT%.csv}.metrics.jsonl" outputs/overnight_submission_stats.txt
ls -lh "${PACKAGE_OUT}"
