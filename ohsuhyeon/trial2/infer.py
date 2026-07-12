"""
Trial 2 inference — constrained-scoring predictions over test.csv -> submission.

Output is always a valid permutation (no parse fallbacks). Writes Id,Answer.
"""
import csv
import os
import time
from pathlib import Path

import pandas as pd
from peft import PeftModel

import common as C

SUBDIR = os.environ.get("SNU_ADAPTER_SUBDIR", "best")
OUT = Path(os.environ.get("SNU_SUBMISSION_OUT", "/workspace/trial2_submission.csv"))
# "score" = 24-way constrained scoring (slow ~16s/row, most accurate);
# "gen"   = greedy generation (fast ~0.5-1s/row, slightly lower).
INFER_MODE = os.environ.get("SNU_INFER_MODE", "score")


def main():
    processor = C.load_processor()
    model = C.load_base_model()
    adapter = C.ADAPTER_DIR / SUBDIR if SUBDIR else C.ADAPTER_DIR
    print(f"loading adapter from {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True

    test_df = pd.read_csv(C.DATA_DIR / "test.csv")

    # Resume: skip Ids already written, so a disconnect/credit-out never loses
    # work — restart the same command and it continues.
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev["Id"].astype(str))
        print(f"resuming: {len(done)} rows already done in {OUT}", flush=True)

    start = time.perf_counter()
    n_done = len(done)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["Id", "Answer"]); f.flush()
        for i, (_, row) in enumerate(test_df.iterrows()):
            if str(row["Id"]) in done:
                continue
            msgs = C.build_messages(row, C.TEST_IMG_DIR)
            io = (C.generate_order(model, processor, msgs) if INFER_MODE == "gen"
                  else C.score_orders(model, processor, msgs))
            w.writerow([row["Id"], str(C.answer_from_image_order(io))])
            f.flush()  # persist every row
            n_done += 1
            if n_done % 25 == 0:
                el = time.perf_counter() - start
                print(f"  {n_done}/{len(test_df)} ({el:.0f}s, "
                      f"{el / max(1, n_done - len(done)):.2f}s/row)", flush=True)
    print(f"DONE {n_done}/{len(test_df)} rows -> {OUT} "
          f"({(time.perf_counter() - start) / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
