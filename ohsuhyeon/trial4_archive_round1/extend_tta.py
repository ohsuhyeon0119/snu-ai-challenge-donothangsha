"""
Extend an existing score dump with additional TTA sigmas, reusing what's
already been computed.

Round 2's test dump has 3 sigmas/row (k=3). Round 1's val sweep showed
accuracy rising monotonically with k and not yet saturating (k=1 0.606 ->
k=2 0.612 -> k=3 0.618 at alpha=0.5), so k=4/5 plausibly add a little more.
Recomputing from scratch at k=5 would redo the 3 sigmas we already paid for;
this script computes ONLY the missing sigmas per row and writes a merged
dump that infer.py can then consume with SNU_TTA_K=5 for free.

Resumable: the output dump is appended row-by-row and re-read on restart,
so an interrupted run picks up where it left off (same pattern as infer.py).

Env knobs:
  SNU_ADAPTER_DIR   LoRA checkpoint (default ckpts_unified/ckpt-2218)
  SNU_IN_DUMP       existing dump (default outputs/test_scores_round2.jsonl)
  SNU_OUT_DUMP      merged dump (default outputs/test_scores_round2_k5.jsonl)
  SNU_TARGET_K      total sigmas per row (default 5)
  SNU_SPLIT         test (default) | val
  SNU_LIMIT         optional row cap

Usage:
  SNU_ADAPTER_DIR=ckpts_unified/ckpt-2218 python extend_tta.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from common import (DATA_DIR, SIGMAS, TEST_IMG_DIR, TRAIN_IMG_DIR, load_model,
                    load_processor, score_row)

ADAPTER_DIR = os.environ.get("SNU_ADAPTER_DIR", "ckpts_unified/ckpt-2218")
IN_DUMP = Path(os.environ.get("SNU_IN_DUMP", "outputs/test_scores_round2.jsonl"))
OUT_DUMP = Path(os.environ.get("SNU_OUT_DUMP",
                               "outputs/test_scores_round2_k5.jsonl"))
TARGET_K = int(os.environ.get("SNU_TARGET_K", "5"))
SPLIT = os.environ.get("SNU_SPLIT", "test")
LIMIT = int(os.environ.get("SNU_LIMIT", "0"))


def load_dump(path):
    mats = {}
    if not Path(path).exists():
        return mats
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from an interrupted run
            mats[rec["Id"]] = rec["scores"]
    return mats


def main():
    assert TARGET_K <= len(SIGMAS), \
        f"TARGET_K={TARGET_K} exceeds the {len(SIGMAS)} defined SIGMAS"

    if SPLIT == "test":
        df = pd.read_csv(DATA_DIR / "test.csv")
        img_dir = TEST_IMG_DIR
    else:
        df = pd.read_csv(Path(os.environ.get(
            "SNU_VAL_CSV", "data_work/val_small.csv")))
        img_dir = TRAIN_IMG_DIR
    if LIMIT:
        df = df.head(LIMIT)
    rows = df.to_dict("records")

    base = load_dump(IN_DUMP)
    done = load_dump(OUT_DUMP)  # resume: rows already extended
    print(f"input dump: {len(base)} rows | already extended: {len(done)} "
          f"| target k={TARGET_K}")

    todo = [r for r in rows
            if len(done.get(r["Id"], [])) < TARGET_K]
    print(f"rows needing work: {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    processor = load_processor()
    model = load_model()
    if ADAPTER_DIR:
        from peft import PeftModel
        from common import QUANT
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        if not QUANT:
            model = model.merge_and_unload()
        print(f"adapter loaded: {ADAPTER_DIR}")
    model.eval()

    OUT_DUMP.parent.mkdir(parents=True, exist_ok=True)
    out = open(OUT_DUMP, "a")
    t0, n = time.time(), 0

    for i, row in enumerate(rows):
        rid = row["Id"]
        if len(done.get(rid, [])) >= TARGET_K:
            continue
        have = base.get(rid, [])[:TARGET_K]  # reuse what round 2 already paid for
        need = SIGMAS[len(have):TARGET_K]
        extra = score_row(model, processor, row, img_dir, need) if need else []
        merged = have + extra
        assert len(merged) == TARGET_K, \
            f"{rid}: got {len(merged)} sigmas, expected {TARGET_K}"
        out.write(json.dumps({"Id": rid, "scores": merged}) + "\n")
        out.flush()
        n += 1
        if n % 10 == 0 or i + 1 == len(rows):
            per = (time.time() - t0) / n
            left = sum(1 for r in rows[i + 1:]
                       if len(done.get(r["Id"], [])) < TARGET_K)
            print(f"{i + 1}/{len(rows)}  {per:.2f}s/row  "
                  f"ETA {left * per / 60:.1f}min  "
                  f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    out.close()
    final = load_dump(OUT_DUMP)
    short = [i for i, m in final.items() if len(m) < TARGET_K]
    print(f"\nwrote {OUT_DUMP}: {len(final)} rows, "
          f"{len(short)} short of k={TARGET_K}")
    assert not short, f"incomplete rows: {short[:5]}"


if __name__ == "__main__":
    main()
