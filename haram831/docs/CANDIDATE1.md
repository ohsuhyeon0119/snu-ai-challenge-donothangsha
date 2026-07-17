# Candidate Model 1: Qwen2-VL-2B QLoRA classifier

This is the repository's first end-to-end reference pipeline. It sends the full
`Sentence` and all four frames to `Qwen/Qwen2-VL-2B-Instruct`, pools the final
language hidden state at the last non-padding token, and predicts one of the
canonical 24 permutation classes with `LayerNorm -> Linear(24)`. It does not
parse generated text. The wrapper performs Qwen2-VL's visual-token insertion
and mRoPE setup, then calls the language decoder directly; this avoids creating
unused full-vocabulary logits. Only LoRA adapters and the classifier head are
trained.

The defaults in `configs/candidate1.json` use 4-bit NF4 double quantization,
auto BF16/FP16 compute dtype, LoRA rank 16/alpha 32/dropout 0.05, and
`q_proj`/`v_proj` targets. Four-image resolution is conservatively bounded.

## Environment

Use Python 3.11 on a CUDA machine. Install the dedicated dependencies:

```bash
python -m pip install -r requirements-candidate1.txt
export PYTHONPATH=src
```

The pipeline requires Transformers 4.49 or newer for the Qwen2-VL mRoPE API
used by the classification-only hidden-state path.

Download the base model/processor before entering an offline environment. No
competition test data is read by training. Use only the provided `train.csv`
and `train/` image tree.

## Required entrypoints

Tiny-subset overfit (defaults: 16 rows, at most 400 optimizer steps, pass at
training exact-match >= 0.95):

```bash
PYTHONPATH=src python -m snu_ordering.candidate1.tiny_overfit \
  --train-csv data/train.csv \
  --image-root data/train \
  --output-dir runs/candidate1-tiny \
  --base-model /models/Qwen2-VL-2B-Instruct \
  --processor /models/Qwen2-VL-2B-Instruct \
  --local-files-only
```

Normal training:

```bash
PYTHONPATH=src python -m snu_ordering.candidate1.train \
  --config configs/candidate1.json \
  --train-csv data/train.csv \
  --image-root data/train \
  --output-dir runs/candidate1 \
  --base-model /models/Qwen2-VL-2B-Instruct \
  --processor /models/Qwen2-VL-2B-Instruct \
  --local-files-only
```

Add `--resume` to reload the adapter, classifier head, optimizer, epoch, and
global step from the same output directory. CLI options expose quantization,
compute dtype, LoRA parameters, batch/accumulation, learning rate, and step
limits. Omit `--local-files-only` only when intentionally downloading a public
pretrained checkpoint during setup.

Offline inference (all model-related arguments are local paths):

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
python -m snu_ordering.candidate1.inference \
  --base-model /models/Qwen2-VL-2B-Instruct \
  --processor /models/Qwen2-VL-2B-Instruct \
  --adapter runs/candidate1/adapter \
  --classifier-head runs/candidate1/classifier_head.pt \
  --config runs/candidate1/run_config.json \
  --input-csv data/test.csv \
  --image-root data/test \
  --output-submission outputs/submission.csv
```

Inference always passes `local_files_only=True`, converts `class_id` through
`snu_ordering.permutation.class_id_to_answer`, then builds and validates the CSV
with `snu_ordering.submission`.

## Artifact layout

```text
runs/candidate1/
├── adapter/                 # PEFT LoRA adapter
├── classifier_head.pt       # LayerNorm/Linear state dict
├── run_config.json          # complete reload configuration
├── training_metrics.json    # exact-match and pairwise metrics
├── memory_report.json       # load/train-step/inference-step CUDA memory
├── metadata.json            # mapping and pooling rules
└── trainer_state.pt         # optimizer/global-step/epoch resume state
```

CUDA memory peaks are reset explicitly before the measured training and
inference blocks. CPU-only runs record `cuda_available: false`; 4-bit
bitsandbytes execution is intended for CUDA. Use `--disable-4bit` only for a
CPU/full-precision smoke test with sufficient RAM. `--limit` on inference is a
debug aid and intentionally warns that its output is not a full submission.
