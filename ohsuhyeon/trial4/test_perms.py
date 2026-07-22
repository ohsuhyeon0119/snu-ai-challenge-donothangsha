"""Exhaustive unit tests for the permutation math in common.py.

Pure CPU / no model. Run locally before anything touches a GPU:
    python test_perms.py
"""
import math
from itertools import permutations

from common import (ALL_ORDERS, IDENTITY_IDX, SIGMAS, TRAIN_P_IDENTITY,
                    adjacent_transpositions, aggregate_scores,
                    answer_from_image_order, broadened_candidate_set,
                    hard_candidate_set, image_order_from_answer,
                    image_order_from_slot_order, inv_perm, kendall_distance,
                    log_prior_vector, slot_order, target_text)


def test_roundtrip_answer_image_order():
    for a in permutations([1, 2, 3, 4]):
        a = list(a)
        io = image_order_from_answer(a)
        assert answer_from_image_order(io) == a, (a, io)


def test_sigma_inverse():
    for s in permutations([1, 2, 3, 4]):
        inv = inv_perm(list(s))
        for j, v in enumerate(s, start=1):
            assert inv[v - 1] == j


def test_slot_order_matches_direct_recompute():
    """slot_order(io, sigma) must equal what you get by physically permuting
    the inputs with sigma and recomputing the order from the permuted answer."""
    for a in permutations([1, 2, 3, 4]):
        a = list(a)
        io = image_order_from_answer(a)
        for sigma in permutations([1, 2, 3, 4]):
            sigma = list(sigma)
            # Physically permuted row: slot j displays Input_{sigma[j]},
            # so the slot's temporal position is a[sigma[j]-1].
            answer_pres = [a[sigma[j] - 1] for j in range(4)]
            io_pres = image_order_from_answer(answer_pres)
            assert slot_order(io, sigma) == io_pres, (a, sigma)
            # and the inverse mapping returns to canonical space
            assert image_order_from_slot_order(io_pres, sigma) == io


def test_all_orders_cover_and_unique():
    assert len(ALL_ORDERS) == 24
    assert len({tuple(o) for o in ALL_ORDERS}) == 24
    assert all(sorted(o) == [1, 2, 3, 4] for o in ALL_ORDERS)


def test_sigmas_valid():
    assert SIGMAS[0] == [1, 2, 3, 4], "first sigma must be identity"
    assert all(sorted(s) == [1, 2, 3, 4] for s in SIGMAS)
    assert len({tuple(s) for s in SIGMAS}) == len(SIGMAS)


def test_target_text_equal_char_layout():
    texts = [target_text(o) for o in ALL_ORDERS]
    assert len({len(t) for t in texts}) == 1, "completions must be equal length"
    assert len({t.replace("1", "N").replace("2", "N").replace("3", "N")
                .replace("4", "N") for t in texts}) == 1, \
        "completions must differ only in the digits"


def test_log_prior_vector():
    lp = log_prior_vector()
    assert len(lp) == 24
    assert abs(sum(math.exp(x) for x in lp) - 1.0) < 1e-9, "prior must sum to 1"
    assert lp[IDENTITY_IDX] == max(lp), "identity gets the largest prior"
    assert ALL_ORDERS[IDENTITY_IDX] == [1, 2, 3, 4]
    others = [x for i, x in enumerate(lp) if i != IDENTITY_IDX]
    assert len(set(others)) == 1, "non-identity perms share one prior"
    assert math.exp(lp[IDENTITY_IDX]) - TRAIN_P_IDENTITY < 1e-12


def test_aggregate_scores_mean_and_prior():
    # two sigma rows -> mean; alpha=0 leaves argmax == argmax of the sum
    mat = [[float(i) for i in range(24)], [float(23 - i) for i in range(24)]]
    tot = aggregate_scores(mat)
    assert all(abs(t - 11.5) < 1e-12 for t in tot), "mean of the two rows"
    # alpha tips a tie toward identity, and only alpha*logprior is added
    tot_a = aggregate_scores(mat, prior_alpha=1.0)
    lp = log_prior_vector()
    assert all(abs(ta - (t + p)) < 1e-12 for ta, t, p in zip(tot_a, tot, lp))
    assert max(range(24), key=lambda i: tot_a[i]) == IDENTITY_IDX


def test_aggregate_alpha_invariant_to_k():
    """The prior term must not scale with tta_k (mean aggregation)."""
    row = [float(i % 7) for i in range(24)]
    t1 = aggregate_scores([row], prior_alpha=0.8)
    t3 = aggregate_scores([row, row, row], prior_alpha=0.8)
    assert all(abs(a - b) < 1e-12 for a, b in zip(t1, t3))


def test_score_row_canonical_alignment():
    """Column i of a score_row matrix must mean canonical ALL_ORDERS[i] under
    EVERY sigma: the completion built for (io, sigma) must map back to io."""
    for sigma in SIGMAS:
        for i, io in enumerate(ALL_ORDERS):
            so = slot_order(io, sigma)
            assert image_order_from_slot_order(so, sigma) == io, (sigma, io)
            # distinct completions per column under this sigma
        comps = [target_text(slot_order(io, sigma)) for io in ALL_ORDERS]
        assert len(set(comps)) == 24


def test_kendall_distance_basic():
    assert kendall_distance([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert kendall_distance([1, 2, 3, 4], [2, 1, 3, 4]) == 1
    assert kendall_distance([1, 2, 3, 4], [4, 3, 2, 1]) == 6
    assert kendall_distance([1, 2, 3, 4], [2, 1, 4, 3]) == 2


def test_kendall_distance_symmetric_and_matches_brute_force():
    for a in permutations([1, 2, 3, 4]):
        for b in permutations([1, 2, 3, 4]):
            a, b = list(a), list(b)
            d = kendall_distance(a, b)
            assert d == kendall_distance(b, a)
            # brute-force: count pairs out of order between the two rankings
            pa = {v: i for i, v in enumerate(a)}
            pb = {v: i for i, v in enumerate(b)}
            brute = sum(1 for x in range(1, 5) for y in range(1, 5)
                        if x < y and (pa[x] < pa[y]) != (pb[x] < pb[y]))
            assert d == brute


def test_adjacent_transpositions():
    for order in ALL_ORDERS:
        adj = adjacent_transpositions(order)
        assert len(adj) == 3
        assert len({tuple(o) for o in adj}) == 3, "must be distinct"
        for o in adj:
            assert sorted(o) == [1, 2, 3, 4]
            assert o != order
            assert kendall_distance(order, o) == 1, \
                "an adjacent swap must be exactly kendall-distance 1"


def test_hard_candidate_set():
    import random
    rng = random.Random(0)
    for truth in ALL_ORDERS:
        cands = hard_candidate_set(truth, n_random=2, rng=rng)
        assert cands[0] == list(truth), "truth must be first (target index 0)"
        assert len(cands) == 6, "3 adjacent + 2 random + truth, none colliding"
        assert len({tuple(c) for c in cands}) == 6, "no duplicate candidates"
        for c in cands[1:4]:
            assert kendall_distance(truth, c) == 1
        for c in cands:
            assert sorted(c) == [1, 2, 3, 4]


def test_hard_candidate_set_no_random_collision_with_adjacent():
    """Regression: the random pool must exclude truth AND its 3 adjacent
    swaps, even under an adversarial rng that would otherwise pick them."""
    import random
    truth = [1, 2, 3, 4]
    adj = {tuple(o) for o in adjacent_transpositions(truth)}
    for seed in range(30):
        cands = hard_candidate_set(truth, n_random=2, rng=random.Random(seed))
        tail = {tuple(c) for c in cands[4:]}
        assert tail.isdisjoint(adj | {tuple(truth)})


def test_kendall_distance_mahonian_counts():
    """Sanity-check the Mahonian numbers for n=4 that broadened_candidate_set
    relies on: exactly {1,3,5,6,5,3,1} perms at kendall distance {0..6}."""
    truth = [1, 2, 3, 4]
    from collections import Counter
    c = Counter(kendall_distance(truth, o) for o in ALL_ORDERS)
    assert c == {0: 1, 1: 3, 2: 5, 3: 6, 4: 5, 5: 3, 6: 1}, c


def test_broadened_candidate_set():
    import random
    rng = random.Random(0)
    for truth in ALL_ORDERS:
        cands = broadened_candidate_set(truth, n_adjacent=3, n_near=2,
                                        n_random=1, rng=rng)
        assert cands[0] == list(truth), "truth must be first (target index 0)"
        assert len(cands) == 7
        assert len({tuple(c) for c in cands}) == 7, "no duplicate candidates"
        for c in cands:
            assert sorted(c) == [1, 2, 3, 4]
        for c in cands[1:4]:
            assert kendall_distance(truth, c) == 1, "adjacent slice wrong"
        for c in cands[4:6]:
            assert kendall_distance(truth, c) in (2, 3), "near slice wrong"


def test_broadened_candidate_set_no_cross_tier_collision():
    """Regression: adjacent/near/random tiers must never overlap, across
    many seeds (the dedup-by-`seen` logic is the thing under test)."""
    import random
    for truth in ALL_ORDERS[:6]:  # a few representative truths
        for seed in range(20):
            cands = broadened_candidate_set(truth, n_adjacent=3, n_near=2,
                                            n_random=1,
                                            rng=random.Random(seed))
            assert len({tuple(c) for c in cands}) == len(cands)


if __name__ == "__main__":
    import sys
    mod = sys.modules["__main__"]
    fns = [v for k, v in vars(mod).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} permutation tests passed")
