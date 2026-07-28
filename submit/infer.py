"""
Trial 4 inference — constrained 24-way scoring (+ presentation TTA + optional
train-derived identity prior) over test.csv (submission) or clean_val.csv.

Runs on the competition target (RTX 3090 24GB) with nf4 quantization.
Incremental CSV writing + resume: safe to interrupt and rerun.

Trial4 additions over trial3:
  - SNU_DUMP_SCORES=path.jsonl dumps each row's per-sigma 24-candidate score
    matrix (canonical space). With a dump present, TTA-k comparison and
    identity-prior alpha sweeps run OFFLINE via analyze_scores.py — no GPU.
  - When the dump exists, resume reconstructs predictions from dumped scores
    (so re-running with a different SNU_PRIOR_ALPHA/SNU_TTA_K <= dumped k
    costs nothing for already-dumped rows).
  - SNU_PRIOR_ALPHA adds alpha * log P_train(perm) at decision time
    (train-side statistic; tune on clean-val only).

Env knobs:
  SNU_ADAPTER_DIR  path to a LoRA checkpoint dir (absent -> zero-shot base)
  SNU_SPLIT        test (default) | val
  SNU_TTA_K        presentation orders to average (default 1; 3 recommended)
  SNU_PRIOR_ALPHA  identity-prior weight (default 0 = off)
  SNU_DUMP_SCORES  optional JSONL path for raw score dumps
  SNU_OUT          output csv (default outputs/trial4_submission.csv)
  SNU_LIMIT        optional row cap (smoke/dry runs)

Usage:
  SNU_ADAPTER_DIR=ckpts/best SNU_SPLIT=val SNU_TTA_K=3 \
      SNU_DUMP_SCORES=outputs/val_scores.jsonl python infer.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from common import (ALL_ORDERS, DATA_DIR, SIGMAS, TEST_IMG_DIR, TRAIN_IMG_DIR,
                    aggregate_scores, answer_from_image_order, exact_match,
                    load_model, load_processor, score_row)

ADAPTER_DIR = os.environ.get("SNU_ADAPTER_DIR", "")
SPLIT = os.environ.get("SNU_SPLIT", "test")
TTA_K = int(os.environ.get("SNU_TTA_K", "1"))
PRIOR_ALPHA = float(os.environ.get("SNU_PRIOR_ALPHA", "0"))
DUMP = os.environ.get("SNU_DUMP_SCORES", "")
_default_out = ("outputs/trial4_submission.csv" if SPLIT == "test"
                else f"outputs/trial4_{SPLIT}_pred.csv")
OUT = Path(os.environ.get("SNU_OUT", _default_out))
LIMIT = int(os.environ.get("SNU_LIMIT", "0"))


def predict_from_mat(mat):
    totals = aggregate_scores(mat[:TTA_K], prior_alpha=PRIOR_ALPHA)
    best = max(range(len(totals)), key=lambda i: totals[i])
    return ALL_ORDERS[best]


def load_dump(path):
    mats = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line from an interrupted run
                mats[rec["Id"]] = rec["scores"]
    return mats


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

    # resume sources: score dump (preferred — reusable under new alpha/k);
    # without a dump, the output csv as in trial3
    mats = load_dump(DUMP) if DUMP else {}
    # only reuse a dumped row if it covers the requested TTA_K sigmas;
    # shallower rows are recomputed and re-appended (loader keeps last record)
    mats = {k: v for k, v in mats.items() if len(v) >= TTA_K}
    done_csv = {}
    if not DUMP and OUT.exists():
        prev = pd.read_csv(OUT)
        done_csv = dict(zip(prev.Id, prev.Answer))
    if mats or done_csv:
        print(f"resuming: {len(mats) or len(done_csv)} rows already done")

    rows = df.to_dict("records")
    todo = [r for r in rows if r["Id"] not in mats and r["Id"] not in done_csv]

    model = processor = None
    if todo:
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump_f = None
    if DUMP:
        Path(DUMP).parent.mkdir(parents=True, exist_ok=True)
        dump_f = open(DUMP, "a")

    preds, t0, new = {}, time.time(), 0
    for i, row in enumerate(rows):
        rid = row["Id"]
        if rid in mats:
            preds[rid] = str(answer_from_image_order(predict_from_mat(mats[rid])))
            continue
        if rid in done_csv:
            preds[rid] = done_csv[rid]
            continue
        mat = score_row(model, processor, row, img_dir, SIGMAS[:max(1, TTA_K)])
        if dump_f:
            dump_f.write(json.dumps({"Id": rid, "scores": mat}) + "\n")
            dump_f.flush()
        preds[rid] = str(answer_from_image_order(predict_from_mat(mat)))
        new += 1
        if new % 10 == 0 or i + 1 == len(rows):
            per = (time.time() - t0) / max(1, new)
            print(f"{i + 1}/{len(rows)}  {per:.2f}s/row  "
                  f"ETA {(len(rows) - i - 1) * per / 60:.1f}min  "
                  f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        if new % 25 == 0:  # incremental write
            pd.DataFrame({"Id": list(preds), "Answer": list(preds.values())}
                         ).to_csv(OUT, index=False)

    if dump_f:
        dump_f.close()
    sub = pd.DataFrame({"Id": list(preds), "Answer": list(preds.values())})
    if SPLIT == "test":  # align to sample_submission row order
        ss = pd.read_csv(DATA_DIR / "sample_submission.csv")
        sub = ss[["Id"]].merge(sub, on="Id", how="left")
        assert sub.Answer.notna().all(), "missing predictions!"
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(sub)} rows, tta_k={TTA_K}, alpha={PRIOR_ALPHA})")

    if SPLIT != "test":
        truths = [json.loads(r["Answer"]) for r in rows]
        got = [json.loads(preds[r["Id"]]) for r in rows]
        print(f"val exact-match: {exact_match(got, truths):.4f} (n={len(rows)})")


if __name__ == "__main__":
    main()
