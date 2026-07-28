"""
Build the round-2 train/val split from the cleaned data, deterministically.

train_unified.py trains on train_full.csv (8,871 rows) and monitors on
val_small.csv (250 rows). Both are derived here from the cleaned pool with a
fixed seed, so the split is reproducible on any box:

    python make_split.py

Inputs  (already black/dup-frame cleaned upstream):
    data_work/train_clean.csv   8,121 rows
    data_work/clean_val.csv     1,000 rows  (held out in round 1)
Outputs:
    data_work/val_small.csv      250 rows  (random_state=42 from clean_val)
    data_work/train_full.csv   8,871 rows  (train_clean + the other 750)
"""
import pandas as pd

tc = pd.read_csv("data_work/train_clean.csv")
cv = pd.read_csv("data_work/clean_val.csv")

val_small = cv.sample(n=250, random_state=42)
train_full = pd.concat([tc, cv.drop(val_small.index)], ignore_index=True)

val_small.to_csv("data_work/val_small.csv", index=False)
train_full.to_csv("data_work/train_full.csv", index=False)

assert set(val_small.Id).isdisjoint(set(train_full.Id))
assert len(val_small) + len(train_full) == len(tc) + len(cv)
print(f"val_small={len(val_small)}  train_full={len(train_full)}  (no id overlap)")
