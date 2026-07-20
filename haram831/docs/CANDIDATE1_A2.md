# Candidate1 A2: relational caption hints and pairwise auxiliary loss

A2 preserves the complete caption and appends only relation edges extracted at
confidence `0.7` or higher. Bare-comma `WEAK` boundaries are excluded. During
training, each eligible edge is dropped with probability `0.3` using a stable
seed derived from the training seed, epoch, sample ID, and boundary index.
Validation and inference never apply boundary dropout.

The six canonical input-frame pairs are supervised by a training-only linear
head on the final prompt-token hidden state. Training uses:

```text
total_loss = causal_lm_loss + 0.3 * pairwise_bce
```

Greedy generation and best-checkpoint selection remain unchanged. The pairwise
head is saved for reproducible resume and diagnostics, but final inference does
not load or use it.

## Tiny overfit

```bash
export PYTHONPATH=src
python -m snu_ordering.candidate1.tiny_overfit \
  --config configs/candidate1-a2.json \
  --train-csv /home/ubuntu3/Project/data/train.csv \
  --image-root /home/ubuntu3/Project/data/train \
  --output-dir runs/candidate1-a2-tiny \
  --base-model /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --processor /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --local-files-only
```

## Full training

Start from the base model in a new output directory. Do not resume an A0 or A1
adapter.

```bash
export PYTHONPATH=src
python -m snu_ordering.candidate1.train \
  --config configs/candidate1-a2.json \
  --train-csv /home/ubuntu3/Project/data/train.csv \
  --image-root /home/ubuntu3/Project/data/train \
  --output-dir runs/candidate1-a2-full \
  --base-model /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --processor /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --local-files-only
```

## Inference

```bash
export PYTHONPATH=src
python -m snu_ordering.candidate1.inference \
  --base-model /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --processor /home/ubuntu3/Project/models/Qwen2-VL-2B-Instruct \
  --adapter runs/candidate1-a2-full/best/adapter \
  --config runs/candidate1-a2-full/best/run_config.json \
  --input-csv /home/ubuntu3/Project/data/test.csv \
  --image-root /home/ubuntu3/Project/data/test \
  --output-submission outputs/candidate1-a2-test.csv
```
