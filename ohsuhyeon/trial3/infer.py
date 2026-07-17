"""
Trial 3 inference — constrained 24-way scoring (+ optional presentation TTA)
over test.csv (submission) or clean_val.csv (accuracy readout).

Runs on the competition target (RTX 3090 24GB) in bf16, no quantization.
Incremental CSV writing + resume: safe to interrupt and rerun.

Env knobs:
  SNU_ADAPTER_DIR  path to a LoRA checkpoint dir (absent -> zero-shot base)
  SNU_SPLIT        test (default) | val
  SNU_TTA_K        presentation orders to average (default 1; 3 recommended)
  SNU_OUT          output csv (default outputs/trial3_submission.csv)
  SNU_LIMIT        optional row cap (smoke/dry runs)

Usage:
  SNU_ADAPTER_DIR=ckpts/best python infer.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from common import (DATA_DIR, TEST_IMG_DIR, TRAIN_IMG_DIR,
                    answer_from_image_order, exact_match,
                    image_order_from_answer, load_model, load_processor,
                    predict_order)

ADAPTER_DIR = os.environ.get("SNU_ADAPTER_DIR", "")
SPLIT = os.environ.get("SNU_SPLIT", "test")
TTA_K = int(os.environ.get("SNU_TTA_K", "1"))
_default_out = ("outputs/trial3_submission.csv" if SPLIT == "test"
                else f"outputs/trial3_{SPLIT}_pred.csv")
OUT = Path(os.environ.get("SNU_OUT", _default_out))
LIMIT = int(os.environ.get("SNU_LIMIT", "0"))


def main():
    if SPLIT == "test":
        df = pd.read_csv(DATA_DIR / "test.csv")
        img_dir = TEST_IMG_DIR
    else:
        df = pd.read_csv(Path(os.environ.get(
            "SNU_VAL_CSV", "data_work/clean_val.csv")))
        img_dir = TRAIN_IMG_DIR
    if LIMIT:
        df = df.head(LIMIT)

    processor = load_processor()
    model = load_model()
    if ADAPTER_DIR:
        from peft import PeftModel
        from common import QUANT
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        if not QUANT:
            model = model.merge_and_unload()  # bake LoRA in (bf16 only)
        print(f"adapter loaded: {ADAPTER_DIR} (merged={not QUANT})")
    model.eval()

    done = {}
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = dict(zip(prev.Id, prev.Answer))
        print(f"resuming: {len(done)} rows already done")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    preds, t0 = {}, time.time()
    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        if row["Id"] in done:
            preds[row["Id"]] = done[row["Id"]]
            continue
        io = predict_order(model, processor, row, img_dir, tta_k=TTA_K)
        preds[row["Id"]] = str(answer_from_image_order(io))
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            per = el / max(1, i + 1 - len(done))
            print(f"{i + 1}/{len(rows)}  {per:.2f}s/row  "
                  f"ETA {(len(rows) - i - 1) * per / 60:.1f}min  "
                  f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        if (i + 1) % 25 == 0:  # incremental write
            pd.DataFrame({"Id": list(preds), "Answer": list(preds.values())}
                         ).to_csv(OUT, index=False)

    sub = pd.DataFrame({"Id": list(preds), "Answer": list(preds.values())})
    if SPLIT == "test":  # align to sample_submission row order
        ss = pd.read_csv(DATA_DIR / "sample_submission.csv")
        sub = ss[["Id"]].merge(sub, on="Id", how="left")
        assert sub.Answer.notna().all(), "missing predictions!"
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(sub)} rows, tta_k={TTA_K})")

    if SPLIT != "test":
        truths = [json.loads(r["Answer"]) for r in rows]
        got = [json.loads(preds[r["Id"]]) for r in rows]
        print(f"val exact-match: {exact_match(got, truths):.4f} (n={len(rows)})")


if __name__ == "__main__":
    main()
