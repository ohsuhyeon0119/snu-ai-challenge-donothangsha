"""
Train-set cleaning + clean-val construction (CPU only, runs locally).

The organizers state train contains uncleaned rows (black frames, orders not
determinable from frames). Those rows carry noisy labels; we flag and drop
them from BOTH the training set and the validation split, so the local val
metric tracks the (cleaner) leaderboard.  Uses TRAIN data only (rule 3.4).

Flags (thresholds picked by eyeballing flagged examples, see SUMMARY):
  black : any frame with mean<12 and std<8 on a 16x16 grayscale thumb
  dup   : any frame pair with mean abs diff < 3 on the same thumbs
          (frames indistinguishable -> order between them undecidable)

Outputs (in trial3/data_work/):
  train_flags.csv  per-row flags for the report
  train_clean.csv  unflagged rows minus val  -> training
  clean_val.csv    1000 unflagged rows, seed 42 -> validation

Run:  python clean_data.py     (SNU_DATA_DIR to point at the dataset)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import DATA_DIR, TRAIN_IMG_DIR

OUT_DIR = Path(os.environ.get("SNU_WORK_DIR", "data_work"))
VAL_N = int(os.environ.get("SNU_VAL_N", "1000"))
SEED = 42

BLACK_MEAN, BLACK_STD, DUP_DIFF = 12.0, 8.0, 3.0


def thumb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32)


def flag_row(row):
    thumbs = [thumb(TRAIN_IMG_DIR / row.Id / getattr(row, f"Input_{i}"))
              for i in range(1, 5)]
    black = any(t.mean() < BLACK_MEAN and t.std() < BLACK_STD for t in thumbs)
    dup = any(np.abs(thumbs[a] - thumbs[b]).mean() < DUP_DIFF
              for a in range(4) for b in range(a + 1, 4))
    return black, dup


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA_DIR / "train.csv")
    print(f"train rows: {len(tr)}")

    flags = []
    for i, row in enumerate(tr.itertuples(index=False)):
        black, dup = flag_row(row)
        flags.append({"Id": row.Id, "black": black, "dup": dup})
        if (i + 1) % 1000 == 0:
            print(f"  scanned {i + 1}/{len(tr)}")
    fl = pd.DataFrame(flags)
    fl["noisy"] = fl.black | fl.dup
    fl.to_csv(OUT_DIR / "train_flags.csv", index=False)

    n_black, n_dup, n_noisy = fl.black.sum(), fl.dup.sum(), fl.noisy.sum()
    print(f"flagged: black={n_black} dup={n_dup} noisy(any)={n_noisy} "
          f"({n_noisy / len(fl):.1%})")

    clean = tr[~tr.Id.isin(fl[fl.noisy].Id)].reset_index(drop=True)
    val = clean.sample(n=min(VAL_N, len(clean) // 5), random_state=SEED)
    train = clean.drop(val.index)
    val.to_csv(OUT_DIR / "clean_val.csv", index=False)
    train.to_csv(OUT_DIR / "train_clean.csv", index=False)
    print(f"train_clean={len(train)}  clean_val={len(val)}")
    print(f"val No_ordering rate: {val.No_ordering.mean():.3f} "
          f"(train_clean: {train.No_ordering.mean():.3f})")


if __name__ == "__main__":
    main()
