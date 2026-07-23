import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


BLACK_MEAN = 12.0
BLACK_STD = 8.0
DUP_DIFF = 3.0


def thumb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize((16, 16)), dtype=np.float32)


def flag_row(row, image_root):
    thumbs = [
        thumb(image_root / row.Id / getattr(row, f"Input_{idx}"))
        for idx in range(1, 5)
    ]
    black = any(t.mean() < BLACK_MEAN and t.std() < BLACK_STD for t in thumbs)
    dup = any(
        np.abs(thumbs[a] - thumbs[b]).mean() < DUP_DIFF
        for a in range(4)
        for b in range(a + 1, 4)
    )
    return black, dup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/snuaichallenge_data",
        help="Directory containing train.csv and train image folder",
    )
    parser.add_argument("--out-dir", default="data_work")
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    image_root = data_dir / "train"
    flags = []
    print(f"train rows: {len(train)}")

    for idx, row in enumerate(train.itertuples(index=False), start=1):
        black, dup = flag_row(row, image_root)
        flags.append({"Id": row.Id, "black": black, "dup": dup})
        if idx % 1000 == 0:
            print(f"  scanned {idx}/{len(train)}")

    flag_df = pd.DataFrame(flags)
    flag_df["noisy"] = flag_df["black"] | flag_df["dup"]
    flag_df.to_csv(out_dir / "train_flags.csv", index=False)

    clean = train[~train.Id.isin(flag_df[flag_df.noisy].Id)].reset_index(drop=True)
    val_n = min(max(args.val_size, 0), max(len(clean) // 5, 0))
    val = clean.sample(n=val_n, random_state=args.seed) if val_n else clean.head(0)
    clean_train = clean.drop(val.index)
    clean_train.to_csv(out_dir / "train_clean.csv", index=False)
    val.to_csv(out_dir / "clean_val.csv", index=False)

    noisy = int(flag_df.noisy.sum())
    print(
        f"flagged: black={int(flag_df.black.sum())} dup={int(flag_df.dup.sum())} "
        f"noisy(any)={noisy} ({noisy / max(1, len(flag_df)):.1%})"
    )
    print(f"train_clean={len(clean_train)} clean_val={len(val)}")


if __name__ == "__main__":
    main()
