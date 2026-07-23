"""
Hard-negative mining for listwise stage-2 (see train_listwise.py).

Scores ckpt-400 on a sample of train_clean.csv under identity presentation
(sigma[0], k=1 — mining only needs RELATIVE margins, not the final decision
config) and ranks rows by margin = score(truth) - score(runner-up). Small or
negative margin = the SFT model is genuinely torn on this row, exactly the
rows where a ranking loss over hard alternatives has signal to give.

Train-only (no test data touched) — rule-compliant model-design use of the
training set (Rule 3.4 only forbids using EVALUATION data this way).

Env knobs:
  SNU_ADAPTER_DIR   default ckpts/ckpt-400
  SNU_TRAIN_CSV     default data_work/train_clean.csv
  SNU_SAMPLE_N      rows to score (default 2500, random subset, seed 42)
  SNU_KEEP_N        hardest rows to keep (default 1200)
  SNU_KEEP_RANDOM   additional random (non-hardest) rows for diversity (default 300)
  SNU_OUT           default data_work/hard_train.csv

Usage:  python mine_hard.py
"""
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch

from common import (ALL_ORDERS, SIGMAS, TRAIN_IMG_DIR,
                    image_order_from_answer, load_model, load_processor,
                    score_row)

ADAPTER_DIR = os.environ.get("SNU_ADAPTER_DIR", "ckpts/ckpt-400")
TRAIN_CSV = Path(os.environ.get("SNU_TRAIN_CSV", "data_work/train_clean.csv"))
SAMPLE_N = int(os.environ.get("SNU_SAMPLE_N", "2500"))
KEEP_N = int(os.environ.get("SNU_KEEP_N", "1200"))
KEEP_RANDOM = int(os.environ.get("SNU_KEEP_RANDOM", "300"))
OUT = Path(os.environ.get("SNU_OUT", "data_work/hard_train.csv"))
RAW = OUT.with_suffix(".raw.csv")  # incremental margin scores, for resume only
SEED = 42


def main():
    df = pd.read_csv(TRAIN_CSV)
    rng = random.Random(SEED)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    sample = df.iloc[idx[:SAMPLE_N]].to_dict("records")
    print(f"mining margins on {len(sample)} train rows "
          f"(of {len(df)}, seed {SEED})")

    processor = load_processor()
    model = load_model()
    if ADAPTER_DIR:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        print(f"adapter loaded: {ADAPTER_DIR}")
    model.eval()

    done = {}
    if RAW.exists() and os.environ.get("SNU_RESUME", "1") == "1":
        prev = pd.read_csv(RAW)
        done = dict(zip(prev.Id, prev.margin))
        print(f"resuming: {len(done)} rows already scored")

    margins, t0 = dict(done), time.time()
    todo = [r for r in sample if r["Id"] not in done]
    for i, row in enumerate(todo):
        truth = image_order_from_answer(json.loads(row["Answer"]))
        ti = ALL_ORDERS.index(truth)
        with torch.no_grad():
            mat = score_row(model, processor, row, TRAIN_IMG_DIR, SIGMAS[:1])
        scores = mat[0]
        runner_up = max(s for j, s in enumerate(scores) if j != ti)
        margins[row["Id"]] = scores[ti] - runner_up
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            per = el / (i + 1)
            print(f"{i + 1}/{len(todo)}  {per:.2f}s/row  "
                  f"ETA {(len(todo) - i - 1) * per / 60:.1f}min  "
                  f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        if (i + 1) % 50 == 0:
            pd.DataFrame({"Id": list(margins), "margin": list(margins.values())}
                         ).to_csv(RAW, index=False)

    mdf = pd.DataFrame({"Id": list(margins), "margin": list(margins.values())})
    mdf.to_csv(RAW, index=False)

    mdf_sorted = mdf.sort_values("margin")
    hard = mdf_sorted.head(KEEP_N)
    rest = mdf_sorted.iloc[KEEP_N:]
    extra = rest.sample(n=min(KEEP_RANDOM, len(rest)), random_state=SEED) \
        if len(rest) else rest
    keep = pd.concat([hard, extra]).drop_duplicates("Id")

    kept_df = df[df.Id.isin(set(keep.Id))].merge(
        keep[["Id", "margin"]], on="Id", how="left")
    kept_df.to_csv(OUT, index=False)
    neg_frac = (mdf.margin < 0).mean()
    print(f"\nmargin stats: mean={mdf.margin.mean():.3f} "
          f"median={mdf.margin.median():.3f} "
          f"negative(wrong-argmax)={neg_frac:.1%}")
    print(f"kept {len(kept_df)} rows -> {OUT} "
          f"({KEEP_N} hardest + {len(keep) - KEEP_N} random)")


if __name__ == "__main__":
    main()
