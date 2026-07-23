#!/usr/bin/env bash
set -euo pipefail

cd "${SNU_REPO_DIR:-/workspace/project/dgddgd314}"

export SNU_DATA_DIR="${SNU_DATA_DIR:-/workspace/project/dgddgd314/data/snuaichallenge_data}"
export SNU_MODEL="${SNU_MODEL:-/workspace/models/paligemma2-10b-pt-224}"
export SNU_ADAPTER_DIR="${SNU_ADAPTER_DIR:-/workspace/project/dgddgd314/outputs/paligemma_lora_overnight}"
export SNU_CONTACT_SHEET_SIZE="${SNU_CONTACT_SHEET_SIZE:-448}"
export SNU_SCORE_CHUNK="${SNU_SCORE_CHUNK:-1}"
export SNU_TTA_K="${SNU_TTA_K:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SECONDS="${SNU_TRAIN_SECONDS:-25200}"
MAX_STEPS="${SNU_TRAIN_MAX_STEPS:-1000000}"
VAL_SIZE="${SNU_VAL_SIZE:-50}"
EVAL_EVERY="${SNU_EVAL_EVERY:-5000}"
LOG_EVERY="${SNU_LOG_EVERY:-100}"
SAVE_EVERY="${SNU_SAVE_EVERY:-0}"
SUBMISSION_OUT="${SNU_SUBMISSION_OUT:-outputs/paligemma_submission_overnight.csv}"
PACKAGE_OUT="${SNU_PACKAGE_OUT:-/workspace/paligemma_overnight_outputs.tar.gz}"

mkdir -p outputs

echo "== train =="
echo "adapter: ${SNU_ADAPTER_DIR}"
echo "train_seconds: ${TRAIN_SECONDS}"
echo "val_size: ${VAL_SIZE}"
echo "eval_every: ${EVAL_EVERY}"

"${PYTHON_BIN}" scripts/paligemma_train_skeleton.py \
  --max-steps "${MAX_STEPS}" \
  --max-seconds "${TRAIN_SECONDS}" \
  --val-size "${VAL_SIZE}" \
  --eval-every "${EVAL_EVERY}" \
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

"${PYTHON_BIN}" scripts/paligemma_infer.py --out "${SUBMISSION_OUT}" --tta-k "${SNU_TTA_K}" --score-chunk "${SNU_SCORE_CHUNK}"
"${PYTHON_BIN}" scripts/validate_submission.py --data-dir "${SNU_DATA_DIR}" --submission "${SUBMISSION_OUT}"

"${PYTHON_BIN}" -c "import pandas as pd; p='${SUBMISSION_OUT}'; df=pd.read_csv(p); print('rows:', len(df)); print('unique answers:', df['Answer'].nunique()); print(df['Answer'].value_counts().head(24)); print(df.head(10).to_string(index=False))" | tee outputs/overnight_submission_stats.txt

tar -czf "${PACKAGE_OUT}" "${SNU_ADAPTER_DIR}" "${SUBMISSION_OUT}" outputs/overnight_submission_stats.txt
ls -lh "${PACKAGE_OUT}"
