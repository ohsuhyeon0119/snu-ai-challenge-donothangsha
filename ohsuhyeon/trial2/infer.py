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


def main():
    processor = C.load_processor()
    model = C.load_base_model()
    adapter = C.ADAPTER_DIR / SUBDIR if SUBDIR else C.ADAPTER_DIR
    print(f"loading adapter from {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True

    test_df = pd.read_csv(C.DATA_DIR / "test.csv")
    rows = []
    start = time.perf_counter()
    for i, (_, row) in enumerate(test_df.iterrows()):
        io = C.score_orders(model, processor, C.build_messages(row, C.TEST_IMG_DIR))
        rows.append((row["Id"], C.answer_from_image_order(io)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_df)} ({(time.perf_counter() - start):.0f}s)", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Answer"])
        for id_, ans in rows:
            w.writerow([id_, str(list(ans))])
    print(f"wrote {len(rows)} rows -> {OUT} "
          f"({(time.perf_counter() - start) / 60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
