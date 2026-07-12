"""
Trial 2 evaluation — constrained-scoring exact-match on the FULL held-out val
split (the same seed-42 12% split train.py wrote), using the BEST adapter.

This is the number to compare against trial1's 16.77% and the 15.46% floor.
Set SNU_ADAPTER_SUBDIR=best (default) to eval the best checkpoint, or ""/"last"
for the final one.
"""
import ast
import os
import time

import pandas as pd
from peft import PeftModel

import common as C

SUBDIR = os.environ.get("SNU_ADAPTER_SUBDIR", "best")
LIMIT = int(os.environ.get("SNU_EVAL_LIMIT", "0"))  # 0 = all


def main():
    processor = C.load_processor()
    model = C.load_base_model()
    adapter = C.ADAPTER_DIR / SUBDIR if SUBDIR else C.ADAPTER_DIR
    print(f"loading adapter from {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True

    val_df = pd.read_csv(C.VAL_SPLIT_CSV)
    if LIMIT:
        val_df = val_df.iloc[:LIMIT]
    preds, truths = [], []
    start = time.perf_counter()
    for i, (_, row) in enumerate(val_df.iterrows()):
        io = C.score_orders(model, processor, C.build_messages(row, C.TRAIN_IMG_DIR))
        preds.append(C.answer_from_image_order(io))
        truths.append(ast.literal_eval(row["Answer"]))
        if (i + 1) % 50 == 0:
            acc = C.exact_match(preds, truths)
            print(f"  {i + 1}/{len(val_df)} running exact_match={acc:.4f} "
                  f"({(time.perf_counter() - start):.0f}s)", flush=True)
    acc = C.exact_match(preds, truths)
    print(f"FULL VAL exact_match={acc:.4f} on n={len(val_df)} "
          f"(trial1=0.1677, floor=0.1546)", flush=True)


if __name__ == "__main__":
    main()
