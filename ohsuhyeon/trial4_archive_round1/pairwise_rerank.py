"""
FREE (no GPU) test of pairwise-marginal re-ranking against the existing
argmax decision rule, using the score dump we already have on disk.

Idea: argmax only trusts the single top-scoring candidate. Error analysis
showed wrong rows have a much narrower top1-vs-top2 margin (median 0.42)
than correct rows (median 3.46) — i.e. mistakes cluster exactly where the
top pick is barely ahead. Pairwise-marginal re-ranking instead asks, for
each of the 6 frame pairs (i,j), "summed over ALL 24 candidates, how much
does the model believe i comes before j" — then picks whichever of the 24
candidates is most CONSISTENT with all 6 pairwise beliefs, rather than just
trusting whichever single candidate happens to be in front.

Pure stdlib, runs on a laptop against outputs/val400_scores.jsonl (500
val rows, ckpt-400, k=3 sigmas) + data_work/clean_val.csv (ground truth).

Usage:  python pairwise_rerank.py
"""
import csv
import json
import math

from common import ALL_ORDERS, aggregate_scores, image_order_from_answer

DUMP = "outputs/val400_scores.jsonl"
VAL_CSV = "data_work/clean_val.csv"
K, ALPHA = 3, 0.5  # already-tuned best config from analyze_scores.py


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


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def pairwise_marginals(probs):
    """probs: 24-length, aligned to ALL_ORDERS. -> {(i,j): P(i before j)} i<j."""
    marg = {}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            p = 0.0
            for idx, order in enumerate(ALL_ORDERS):
                if order.index(i) < order.index(j):
                    p += probs[idx]
            marg[(i, j)] = p
    return marg


def consistency_logscore(order, marg):
    s = 0.0
    for i in range(1, 5):
        for j in range(i + 1, 5):
            p = marg[(i, j)] if order.index(i) < order.index(j) else 1 - marg[(i, j)]
            s += math.log(max(p, 1e-9))
    return s


def main():
    mats = load_dump(DUMP)
    val = load_val(VAL_CSV)
    ids = [i for i in mats if i in val and len(mats[i]) >= K]
    n = len(ids)
    truths = {i: image_order_from_answer(json.loads(val[i]["Answer"])) for i in ids}
    print(f"rows={n}  config k={K} alpha={ALPHA}")

    correct_argmax = correct_pure = 0
    correct_blend = {b: 0 for b in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)}
    flips_helped = flips_hurt = flips_neutral = 0

    for i in ids:
        totals = aggregate_scores(mats[i][:K], prior_alpha=ALPHA)
        base_idx = max(range(24), key=lambda k: totals[k])
        base_pred = ALL_ORDERS[base_idx]
        base_ok = base_pred == truths[i]
        correct_argmax += base_ok

        probs = softmax(totals)
        marg = pairwise_marginals(probs)
        cons = [consistency_logscore(o, marg) for o in ALL_ORDERS]

        pure_idx = max(range(24), key=lambda k: cons[k])
        pure_pred = ALL_ORDERS[pure_idx]
        correct_pure += pure_pred == truths[i]

        for b in correct_blend:
            blend = [totals[k] + b * cons[k] for k in range(24)]
            b_idx = max(range(24), key=lambda k: blend[k])
            b_pred = ALL_ORDERS[b_idx]
            correct_blend[b] += b_pred == truths[i]
            if b == 1.0:  # track flips at a representative blend weight
                if b_pred != base_pred:
                    b_ok = b_pred == truths[i]
                    if b_ok and not base_ok:
                        flips_helped += 1
                    elif not b_ok and base_ok:
                        flips_hurt += 1
                    else:
                        flips_neutral += 1

    print(f"\nbaseline argmax                 acc = {correct_argmax / n:.4f}")
    print(f"pure pairwise-consistency argmax acc = {correct_pure / n:.4f}")
    print("\nblend (totals + beta*consistency):")
    for b, c in correct_blend.items():
        print(f"  beta={b:<5} acc = {c / n:.4f}")
    print(f"\nat beta=1.0, decision flips vs baseline: "
          f"helped={flips_helped} hurt={flips_hurt} neutral(still wrong/right)={flips_neutral}")


if __name__ == "__main__":
    main()
