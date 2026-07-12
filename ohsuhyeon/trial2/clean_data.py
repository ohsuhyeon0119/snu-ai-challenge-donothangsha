"""
Trial 2 data-cleaning REPORT (run first, decide later).

Flags likely-bad train frames using only pixel content (no metadata):
  - near-black / near-constant frames (mean brightness or std below threshold)
  - duplicate frames within a sample (identical average-hash)

Writes a per-sample report CSV. By default it only REPORTS; pass
SNU_WRITE_EXCLUDE=1 to also write an exclude-id list (samples with >=1 flagged
frame) that train.py could later drop from TRAINING ONLY (never from test).
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image

import common as C

REPORT_OUT = Path(os.environ.get("SNU_CLEAN_REPORT", "/workspace/trial2_clean_report.csv"))
EXCLUDE_OUT = Path(os.environ.get("SNU_EXCLUDE_OUT", "/workspace/trial2_exclude_ids.txt"))
WRITE_EXCLUDE = os.environ.get("SNU_WRITE_EXCLUDE", "0") == "1"
MEAN_MIN = float(os.environ.get("SNU_MEAN_MIN", "10"))
STD_MIN = float(os.environ.get("SNU_STD_MIN", "5"))


def ahash(img):
    g = img.convert("L").resize((8, 8), Image.BILINEAR)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= (1 << i)
    return bits, (avg, _std(px, avg))


def _std(px, avg):
    return (sum((p - avg) ** 2 for p in px) / len(px)) ** 0.5


def main():
    df = pd.read_csv(C.DATA_DIR / "train.csv")
    rows, excl = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        hashes, flags = [], []
        for k in range(1, 5):
            f = C.TRAIN_IMG_DIR / r["Id"] / r[f"Input_{k}"]
            try:
                h, (mean, std) = ahash(Image.open(f))
            except Exception as e:
                flags.append(f"Input_{k}:openfail"); continue
            hashes.append(h)
            if mean < MEAN_MIN or std < STD_MIN:
                flags.append(f"Input_{k}:blank(mean={mean:.0f},std={std:.0f})")
        dups = len(hashes) - len(set(hashes))
        if dups:
            flags.append(f"dup_frames={dups}")
        if flags:
            excl.append(r["Id"])
            rows.append({"Id": r["Id"], "flags": "; ".join(flags)})
        if (i + 1) % 1000 == 0:
            print(f"  scanned {i + 1}/{len(df)}, flagged {len(rows)}", flush=True)

    pd.DataFrame(rows).to_csv(REPORT_OUT, index=False)
    print(f"flagged {len(rows)}/{len(df)} samples -> {REPORT_OUT}", flush=True)
    if WRITE_EXCLUDE:
        EXCLUDE_OUT.write_text("\n".join(excl))
        print(f"wrote {len(excl)} exclude ids -> {EXCLUDE_OUT}", flush=True)


if __name__ == "__main__":
    main()
