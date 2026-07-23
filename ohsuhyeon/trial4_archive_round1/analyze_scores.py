"""
Offline analysis of a score dump produced by infer.py (SNU_DUMP_SCORES).

Pure stdlib — runs on a laptop, no GPU / torch / pandas. Uses the exact same
aggregate_scores() as inference, so a tuned alpha transfers verbatim.

Reports, on the val split:
  1. exact-match accuracy for every (tta_k, prior_alpha) on a grid
  2. the best config + its accuracy, with binomial +-1 sigma
  3. error structure at the best config: Kendall-distance histogram,
     identity misses / identity overcalls
  4. TTA decision-flip rate (how often k>1 changes the k=1 argmax)
     and score-margin quartiles (top1 - top2) for correct vs wrong rows

Usage:
  python analyze_scores.py outputs/val_scores.jsonl [data_work/clean_val.csv]
"""
import csv
import json
import math
import sys
from collections import Counter

from common import ALL_ORDERS, aggregate_scores, image_order_from_answer

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def load_dump(path):
    mats = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mats[rec["Id"]] = rec["scores"]
    return mats


def load_val(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["Id"]: r for r in csv.DictReader(f)}


def kendall(a, b):
    """pairwise-inversion distance between two image_orders (0..6)."""
    pa = {v: i for i, v in enumerate(a)}
    pb = {v: i for i, v in enumerate(b)}
    return sum(1 for i in range(1, 5) for j in range(i + 1, 5)
               if (pa[i] - pa[j]) * (pb[i] - pb[j]) < 0)


def predict(mat, k, alpha):
    totals = aggregate_scores(mat[:k], prior_alpha=alpha)
    return ALL_ORDERS[max(range(24), key=lambda i: totals[i])], totals


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/val_scores.jsonl"
    val_path = sys.argv[2] if len(sys.argv) > 2 else "data_work/clean_val.csv"
    mats = load_dump(dump_path)
    val = load_val(val_path)
    ids = [i for i in mats if i in val]
    if not ids:
        sys.exit("no overlap between dump and val csv")
    kmax = min(len(mats[i]) for i in ids)
    n = len(ids)
    truths = {i: image_order_from_answer(json.loads(val[i]["Answer"]))
              for i in ids}
    print(f"rows={n}  kmax={kmax}  (+-1sigma at p=0.7 ~ "
          f"{math.sqrt(0.7 * 0.3 / n):.3f})")

    # 1) grid
    best = (-1.0, 1, 0.0)
    print("\nacc by (k, alpha):")
    print("k\\a  " + "  ".join(f"{a:>5}" for a in ALPHAS))
    for k in range(1, kmax + 1):
        row = []
        for a in ALPHAS:
            acc = sum(predict(mats[i], k, a)[0] == truths[i]
                      for i in ids) / n
            row.append(acc)
            if acc > best[0]:
                best = (acc, k, a)
        print(f"k={k}  " + "  ".join(f"{x:.3f}" for x in row))
    acc, bk, ba = best
    print(f"\nbest: k={bk} alpha={ba} acc={acc:.4f}")

    # 3) error structure at best config
    kd = Counter()
    id_miss = id_over = 0
    margins_ok, margins_bad = [], []
    ident = [1, 2, 3, 4]
    for i in ids:
        pred, totals = predict(mats[i], bk, ba)
        st = sorted(totals, reverse=True)
        margin = st[0] - st[1]
        if pred == truths[i]:
            margins_ok.append(margin)
        else:
            margins_bad.append(margin)
            kd[kendall(pred, truths[i])] += 1
            if truths[i] == ident:
                id_miss += 1
            if pred == ident:
                id_over += 1
    print(f"errors={len(margins_bad)}  kendall hist={dict(sorted(kd.items()))}")
    print(f"identity: missed={id_miss}  overcalled={id_over}")

    def q(v):
        if not v:
            return "n/a"
        v = sorted(v)
        return " ".join(f"{v[int(len(v) * p)]:.2f}" for p in (0.25, 0.5, 0.75))

    print(f"margin q25/50/75  correct: {q(margins_ok)}  wrong: {q(margins_bad)}")

    # 4) TTA flip rate vs k=1
    if kmax > 1:
        for k in range(2, kmax + 1):
            flips = sum(predict(mats[i], k, ba)[0] != predict(mats[i], 1, ba)[0]
                        for i in ids)
            print(f"decision flips k=1 -> k={k}: {flips}/{n} ({flips / n:.1%})")


if __name__ == "__main__":
    main()
