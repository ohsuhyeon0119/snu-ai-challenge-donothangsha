"""End-to-end CPU dry-run of the dump -> offline-analysis pipeline.

Builds a synthetic score dump from a fake scorer with a KNOWN pathology
(identity systematically scored 0.1 below the runner-up — the trial3 failure
mode), plus a matching val csv, then checks that analyze_scores:
  - reproduces the alpha=0 accuracy exactly (identity rows all missed)
  - finds an alpha > 0 that recovers the identity rows
  - agrees with infer.py's predict_from_mat on every row/config

Run locally:  python test_offline_pipeline.py
"""
import csv
import json
import random
import tempfile
from pathlib import Path

import analyze_scores as an
from common import ALL_ORDERS, IDENTITY_IDX, aggregate_scores, \
    answer_from_image_order

random.seed(0)
N = 200
IDENT = [1, 2, 3, 4]


def fake_mat(truth, k=3):
    """Per-sigma canonical score rows. Truth scored highest EXCEPT identity,
    which lands 0.1 below one specific wrong candidate."""
    ti = ALL_ORDERS.index(truth)
    mat = []
    for s in range(k):
        row = [-8.0 + 0.01 * ((s * 7 + i) % 5) for i in range(24)]
        if ti == IDENTITY_IDX:
            row[ti] = -4.0            # identity truth: close second
            row[(ti + 5) % 24] = -3.9  # a wrong perm wins by 0.1
        else:
            row[ti] = -3.0            # non-identity truth: clear winner
        mat.append(row)
    return mat


def main():
    tmp = Path(tempfile.mkdtemp())
    truths = []
    for i in range(N):
        truths.append(IDENT if i < 30 else list(random.sample(range(1, 5), 4)))
        # avoid accidental identities in the "shuffled" part (train property)
        if i >= 30 and truths[-1] == IDENT:
            truths[-1] = [2, 1, 3, 4]

    dump, val = tmp / "scores.jsonl", tmp / "val.csv"
    with open(dump, "w") as f, open(val, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["Id", "Answer"])
        for i, t in enumerate(truths):
            rid = f"row{i:04d}"
            f.write(json.dumps({"Id": rid, "scores": fake_mat(t)}) + "\n")
            w.writerow([rid, str(answer_from_image_order(t))])

    mats = an.load_dump(dump)
    vals = an.load_val(val)
    ids = list(mats)
    assert len(ids) == N

    # alpha=0: every identity row must be missed -> acc = (N-30)/N
    acc0 = sum(an.predict(mats[i], 3, 0.0)[0] == truths[int(i[3:])]
               for i in ids) / N
    assert abs(acc0 - (N - 30) / N) < 1e-9, acc0

    # the 0.1 gap needs alpha*(log p_id - log p_other) > 0.1 -> alpha >= ~0.06;
    # every grid alpha >= 0.25 must recover all identity rows
    acc1 = sum(an.predict(mats[i], 3, 0.5)[0] == truths[int(i[3:])]
               for i in ids) / N
    assert abs(acc1 - 1.0) < 1e-9, acc1

    # too-large alpha must start OVERCALLING identity on non-identity rows
    # (their truth margin is 1.0 over the identity baseline row values)
    big = 3.5 / (an.aggregate_scores.__globals__["log_prior_vector"]()[
        IDENTITY_IDX] - an.aggregate_scores.__globals__["log_prior_vector"]()[1])
    acc_big = sum(an.predict(mats[i], 3, abs(big) * 4)[0] == truths[int(i[3:])]
                  for i in ids) / N
    assert acc_big < 1.0, "huge alpha should hurt"

    # infer.predict_from_mat parity (same math path, different module)
    import importlib
    import sys
    import types
    # stub torch/pandas so infer.py imports on a laptop
    for name in ("torch", "pandas"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    import os
    os.environ["SNU_TTA_K"] = "3"
    os.environ["SNU_PRIOR_ALPHA"] = "0.5"
    infer = importlib.import_module("infer")
    for i in ids:
        p_an = an.predict(mats[i], 3, 0.5)[0]
        p_inf = infer.predict_from_mat(mats[i])
        assert p_an == p_inf, i

    # kendall sanity
    assert an.kendall([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert an.kendall([1, 2, 3, 4], [2, 1, 3, 4]) == 1
    assert an.kendall([1, 2, 3, 4], [4, 3, 2, 1]) == 6

    print("ok  synthetic acc@a=0 =", round(acc0, 3), " acc@a=0.5 = 1.0")
    print("ok  infer.predict_from_mat parity on", N, "rows")
    print("ok  kendall sanity")
    print("\noffline pipeline dry-run passed")


if __name__ == "__main__":
    main()
